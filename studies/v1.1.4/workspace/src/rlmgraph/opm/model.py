from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from torch import Tensor, nn

from .generation import Operation, derive_uint64


class ModelKind(StrEnum):
    OPM_SHARED = "OPM_SHARED"
    PROC_UNTIED = "PROC_UNTIED"
    PROC_CLONE = "PROC_CLONE"
    DOMAIN_GENERALIST = "DOMAIN_GENERALIST"


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    d_model: int = 192
    d_entity: int = 64
    d_domain: int = 16
    d_hidden_primitive: int = 384
    max_length: int = 12
    primitive_count: int = 8
    max_steps: int = 2
    dropout: float = 0.1
    heads: int = 4
    encoder_layers: int = 2
    encoder_ff: int = 768


@dataclass
class OPMBatch:
    fact_tokens: Tensor  # [B,F,T]
    fact_token_mask: Tensor  # [B,F,T], True for real tokens
    query_tokens: Tensor  # [B,T]
    query_token_mask: Tensor  # [B,T]
    domain_ids: Tensor  # [B]
    argument_entity_ids: Tensor  # [B,2]
    fact_endpoint_ids: Tensor  # [B,F,2]
    evidence_indices: Tensor  # [B,2], -1 for PAD
    operation_ids: Tensor  # [B,2]
    step_mask: Tensor  # [B,2]
    labels: Tensor | None = None


class SurfaceEncoder(nn.Module):
    def __init__(self, config: ModelConfig, token_embedding: nn.Embedding) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.position = nn.Embedding(config.max_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.encoder_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers)
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-5)

    def forward(self, token_ids: Tensor, mask: Tensor) -> Tensor:
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask.bool())
        return self.final_norm(hidden)[:, 0]


class PrimitiveTransition(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float, extra: int = 0) -> None:
        super().__init__()
        input_width = d_model * 2 + extra
        self.input_norm = nn.LayerNorm(input_width, eps=1e-5)
        self.hidden = nn.Linear(input_width, d_hidden)
        self.gate = nn.Linear(input_width, d_model)
        self.output = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.state_norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, state: Tensor, evidence: Tensor, extra: Tensor | None = None) -> Tensor:
        parts = [state, evidence]
        if extra is not None:
            parts.append(extra)
        inputs = self.input_norm(torch.cat(parts, dim=-1))
        hidden = self.dropout(torch.nn.functional.gelu(self.hidden(inputs)))
        gate = torch.sigmoid(self.gate(inputs))
        return self.state_norm(state + gate * self.output(hidden))


class OPMModel(nn.Module):
    def __init__(self, config: ModelConfig, kind: ModelKind, model_seed: int) -> None:
        super().__init__()
        self.config = config
        self.kind = kind
        self.model_seed = model_seed
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.d_model, padding_idx=0)
        self.entity_embedding = nn.Embedding(49, config.d_entity, padding_idx=48)
        self.domain_embedding = nn.Embedding(3, config.d_domain)
        self.fact_encoder = SurfaceEncoder(config, self.token_embedding)
        self.query_encoder = SurfaceEncoder(config, self.token_embedding)
        self.endpoint_projection = nn.Linear(config.d_entity * 2, config.d_model)
        self.evidence_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.domain_projection = nn.Linear(config.d_domain, config.d_model)
        self.argument_projection = nn.Linear(config.d_entity * 2, config.d_model)
        self.state_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.decoder_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.decoder = nn.Linear(config.d_model, 2)
        self.operation_to_primitive = self._operation_permutation(model_seed)
        if kind == ModelKind.OPM_SHARED:
            self.primitives = nn.ModuleList(
                PrimitiveTransition(config.d_model, config.d_hidden_primitive, config.dropout)
                for _ in range(config.primitive_count)
            )
        elif kind in (ModelKind.PROC_UNTIED, ModelKind.PROC_CLONE):
            self.primitives = nn.ModuleList(
                PrimitiveTransition(config.d_model, config.d_hidden_primitive, config.dropout)
                for _ in range(3 * config.primitive_count)
            )
        elif kind == ModelKind.DOMAIN_GENERALIST:
            self.operation_embedding = nn.Embedding(4, 32)
            self.generalists = nn.ModuleList(
                PrimitiveTransition(config.d_model, config.d_hidden_primitive, config.dropout, extra=32)
                for _ in range(3)
            )
        else:  # pragma: no cover
            raise ValueError(kind)
        self.apply(self._initialize)
        if kind == ModelKind.PROC_CLONE:
            self._clone_domain_primitives()

    @staticmethod
    def _operation_permutation(seed: int) -> tuple[int, ...]:
        rng = np.random.Generator(
            np.random.PCG64DXSM(derive_uint64("opm-v1", "module-permutation", seed))
        )
        active = [int(value) for value in rng.permutation(8)[:4]]
        # Specification maps sorted names [CHAIN,LIFT,LOOKUP,REVERSE].
        by_name = dict(zip(sorted(Operation, key=lambda item: item.name), active, strict=True))
        return tuple(by_name[operation] for operation in Operation)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _clone_domain_primitives(self) -> None:
        for primitive_index in range(self.config.primitive_count):
            source = self.primitives[primitive_index].state_dict()
            for domain in (1, 2):
                self.primitives[domain * self.config.primitive_count + primitive_index].load_state_dict(
                    source
                )

    def _encode(self, batch: OPMBatch) -> tuple[Tensor, Tensor]:
        batch_size, fact_count, token_count = batch.fact_tokens.shape
        facts = self.fact_encoder(
            batch.fact_tokens.reshape(batch_size * fact_count, token_count),
            batch.fact_token_mask.reshape(batch_size * fact_count, token_count),
        ).reshape(batch_size, fact_count, self.config.d_model)
        endpoints = self.entity_embedding(batch.fact_endpoint_ids).flatten(-2)
        facts = self.evidence_norm(facts + self.endpoint_projection(endpoints))
        query = self.query_encoder(batch.query_tokens, batch.query_token_mask)
        arguments = self.entity_embedding(batch.argument_entity_ids).flatten(-2)
        state = self.state_norm(
            self.query_projection(query)
            + self.domain_projection(self.domain_embedding(batch.domain_ids))
            + self.argument_projection(arguments)
        )
        return facts, state

    def _transition(self, state: Tensor, evidence: Tensor, operation: Tensor, domain: Tensor) -> Tensor:
        output = torch.empty_like(state)
        if self.kind == ModelKind.DOMAIN_GENERALIST:
            op_embedding = self.operation_embedding(operation)
            for domain_id in range(3):
                selected = domain == domain_id
                if selected.any():
                    output[selected] = self.generalists[domain_id](
                        state[selected], evidence[selected], op_embedding[selected]
                    )
            return output
        primitive_ids = torch.tensor(
            self.operation_to_primitive, dtype=torch.long, device=operation.device
        )[operation]
        if self.kind in (ModelKind.PROC_UNTIED, ModelKind.PROC_CLONE):
            primitive_ids = primitive_ids + domain * self.config.primitive_count
        for primitive_id in primitive_ids.unique(sorted=True):
            selected = primitive_ids == primitive_id
            output[selected] = self.primitives[int(primitive_id)](state[selected], evidence[selected])
        return output

    def forward(self, batch: OPMBatch, *, return_states: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        facts, state = self._encode(batch)
        states = [state]
        batch_indices = torch.arange(state.shape[0], device=state.device)
        for step in range(self.config.max_steps):
            active = batch.step_mask[:, step].bool()
            if active.any():
                indices = batch.evidence_indices[:, step].clamp_min(0)
                evidence = facts[batch_indices, indices]
                transitioned = self._transition(
                    state[active],
                    evidence[active],
                    batch.operation_ids[active, step],
                    batch.domain_ids[active],
                )
                state = state.clone()
                state[active] = transitioned
            states.append(state)
        logits = self.decoder(self.decoder_norm(state))
        return (logits, states) if return_states else logits

    @torch.no_grad()
    def encode_selected_evidence(self, batch: OPMBatch) -> Tensor:
        """Return frozen selected evidence representations as [B,2,d_model]."""
        self.eval()
        facts, _ = self._encode(batch)
        batch_indices = torch.arange(facts.shape[0], device=facts.device)
        outputs: list[Tensor] = []
        for step in range(self.config.max_steps):
            indices = batch.evidence_indices[:, step].clamp_min(0)
            selected = facts[batch_indices, indices]
            selected = selected * batch.step_mask[:, step].unsqueeze(-1)
            outputs.append(selected)
        return torch.stack(outputs, dim=1)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
