from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import OPMBatch, OPMModel
from .protocol import CURRENT_PROTOCOL


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    max_steps: int = 50_000


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    final_loss: float


def train_validation(
    model: OPMModel,
    batches: list[OPMBatch],
    config: TrainingConfig,
    *,
    validation_step_limit: int,
) -> TrainingResult:
    """Run only an implementation-validation smoke loop, never a primary protocol."""
    CURRENT_PROTOCOL.require_validation()
    if validation_step_limit <= 0 or validation_step_limit > 100:
        raise ValueError("implementation validation is limited to 1..100 optimizer steps")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    model.train()
    final_loss = float("nan")
    for step in range(validation_step_limit):
        batch = batches[step % len(batches)]
        if batch.labels is None:
            raise ValueError("training batch has no labels")
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        final_loss = float(loss.detach())
    return TrainingResult(validation_step_limit, final_loss)
