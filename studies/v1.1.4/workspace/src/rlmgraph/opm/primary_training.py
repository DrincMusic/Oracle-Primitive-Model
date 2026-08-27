from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .artifacts import canonical_json_bytes, load_checkpoint, run_id, save_checkpoint
from .data import Vocabulary, batch_to
from .generation import Operation
from .model import ModelConfig, ModelKind, OPMBatch, OPMModel
from .protocol import CURRENT_PROTOCOL
from .rendering import Domain
from .sampler import CanonicalSampler, Stratum


@dataclass(frozen=True)
class CanonicalTrainingConfig:
    learning_rate: float
    dropout: float
    max_steps: int = 50_000
    batch_size: int = 256
    warmup_steps: int = 2_000
    minimum_learning_rate: float = 3e-5
    validation_every_steps: int = 500
    checkpoint_every_steps: int = 500
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0

    def validate(self, *, canonical: bool = True) -> None:
        if (
            self.max_steps <= 0
            or self.batch_size <= 0
            or self.warmup_steps <= 0
            or self.warmup_steps >= self.max_steps
            or self.validation_every_steps <= 0
            or self.checkpoint_every_steps != self.validation_every_steps
            or self.max_steps % self.validation_every_steps != 0
        ):
            raise ValueError("training schedule cannot guarantee a selectable checkpoint")
        if canonical and (
            self.max_steps != 50_000
            or self.batch_size != 256
            or self.warmup_steps != 2_000
            or self.validation_every_steps != 500
            or self.checkpoint_every_steps != 500
        ):
            raise ValueError("canonical training cadence does not match TRN-001")
        if self.learning_rate not in (1e-4, 3e-4, 6e-4) or self.dropout not in (0.0, 0.1):
            raise ValueError("training configuration is outside the frozen tuning grid")


@dataclass(frozen=True)
class TrainingSummary:
    run_id: str
    model_kind: str
    model_seed: int
    completed_steps: int
    final_loss: float
    selected_step: int
    selected_macro_validation_accuracy: float
    selected_checkpoint_sha256: str
    train_sha256: str
    validation_sha256: str
    primary_training: bool


class TensorizedRows:
    def __init__(self, path: Path) -> None:
        vocabulary = Vocabulary.build()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows.sort(key=lambda row: row["example_id"])
        self.example_ids = tuple(str(row["example_id"]) for row in rows)
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("dataset contains duplicate example IDs")
        self.index_by_id = {value: index for index, value in enumerate(self.example_ids)}
        fact_tokens = []
        fact_masks = []
        query_tokens = []
        query_masks = []
        for row in rows:
            encoded_facts = [vocabulary.encode(text, "[FACT]") for text in row["rendered_facts"]]
            fact_tokens.append([item[0] for item in encoded_facts])
            fact_masks.append([item[1] for item in encoded_facts])
            query = vocabulary.encode(row["rendered_query"], "[QUERY]")
            query_tokens.append(query[0])
            query_masks.append(query[1])
        self.tensors = {
            "fact_tokens": torch.tensor(fact_tokens, dtype=torch.long),
            "fact_token_mask": torch.tensor(fact_masks, dtype=torch.bool),
            "query_tokens": torch.tensor(query_tokens, dtype=torch.long),
            "query_token_mask": torch.tensor(query_masks, dtype=torch.bool),
            "domain_ids": torch.tensor([int(Domain[row["domain"]]) for row in rows]),
            "argument_entity_ids": torch.tensor([row["argument_entity_ids"] for row in rows]),
            "fact_endpoint_ids": torch.tensor([row["fact_endpoint_ids"] for row in rows]),
            "evidence_indices": torch.tensor([row["evidence_indices"] for row in rows]),
            "operation_ids": torch.tensor([row["operation_ids"] for row in rows]),
            "step_mask": torch.tensor([row["step_mask"] for row in rows], dtype=torch.bool),
            "labels": torch.tensor([row["label"] for row in rows]),
        }
        self.pools: dict[Stratum, tuple[str, ...]] = {}
        grouped: dict[Stratum, list[str]] = {}
        for row in rows:
            stratum = (
                Domain[row["domain"]],
                Operation(int(row["operation_ids"][0])),
                int(row["label"]),
            )
            grouped.setdefault(stratum, []).append(str(row["example_id"]))
        self.pools = {key: tuple(values) for key, values in grouped.items()}

    def batch_for_ids(self, example_ids: list[str], device: torch.device) -> OPMBatch:
        indices = torch.tensor([self.index_by_id[value] for value in example_ids])
        batch = OPMBatch(**{name: tensor.index_select(0, indices) for name, tensor in self.tensors.items()})
        return batch_to(batch, device)

    def batches(self, batch_size: int, device: torch.device):
        for start in range(0, len(self.example_ids), batch_size):
            yield self.batch_for_ids(list(self.example_ids[start : start + batch_size]), device)


def learning_rate_at_step(config: CanonicalTrainingConfig, completed_step: int) -> float:
    if completed_step < 1 or completed_step > config.max_steps:
        raise ValueError("completed step outside training schedule")
    if completed_step <= config.warmup_steps:
        return config.learning_rate * completed_step / config.warmup_steps
    progress = (completed_step - config.warmup_steps) / (
        config.max_steps - config.warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


@torch.no_grad()
def macro_validation_accuracy(
    model: OPMModel, validation: TensorizedRows, device: torch.device
) -> float:
    model.eval()
    cell_correct: dict[tuple[int, int], int] = {}
    cell_count: dict[tuple[int, int], int] = {}
    for batch in validation.batches(256, device):
        predictions = model(batch).argmax(dim=-1)
        for domain, operation, prediction, label in zip(
            batch.domain_ids.tolist(),
            batch.operation_ids[:, 0].tolist(),
            predictions.tolist(),
            batch.labels.tolist(),
            strict=True,
        ):
            key = (domain, operation)
            cell_correct[key] = cell_correct.get(key, 0) + int(prediction == label)
            cell_count[key] = cell_count.get(key, 0) + 1
    return sum(cell_correct[key] / cell_count[key] for key in cell_count) / len(cell_count)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_canonical_run(
    *,
    kind: ModelKind,
    model_seed: int,
    config: CanonicalTrainingConfig,
    train_path: Path,
    validation_path: Path,
    output_root: Path,
    code_revision: str,
    device: str = "cuda:0",
    canonical: bool = True,
    resume_checkpoint: Path | None = None,
    stop_after_steps: int | None = None,
) -> TrainingSummary:
    CURRENT_PROTOCOL.require_primary()
    config.validate(canonical=canonical)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA primary device is unavailable")
    train_hash = _sha256(train_path)
    validation_hash = _sha256(validation_path)
    configuration = {"model_kind": kind.value, **asdict(config)}
    identifier = run_id(
        configuration=configuration,
        specification_version="1.0.0",
        dataset_fingerprints={"train": train_hash, "validation": validation_hash},
        model_seed=model_seed,
        code_revision=code_revision,
    )
    run_directory = output_root / identifier
    manifest_payload = {
        "run_id": identifier,
        "lifecycle": "PRIMARY_RUNS",
        "primary_run": canonical,
        "model_seed": model_seed,
        "configuration": configuration,
        "dataset_fingerprints": {"train": train_hash, "validation": validation_hash},
        "code_revision": code_revision,
        "sealed_labels_accessed": False,
        "aggregate_test_evaluation": False,
    }
    if resume_checkpoint is None:
        run_directory.mkdir(parents=True, exist_ok=False)
        (run_directory / "run-manifest.json").write_bytes(canonical_json_bytes(manifest_payload))
    else:
        if not run_directory.is_dir():
            raise FileNotFoundError("resume run directory is missing")
        existing_manifest = json.loads(
            (run_directory / "run-manifest.json").read_text(encoding="utf-8")
        )
        if existing_manifest != manifest_payload:
            raise ValueError("resume run manifest does not match requested run")
    train = TensorizedRows(train_path)
    validation = TensorizedRows(validation_path)
    sampler = CanonicalSampler(train.pools, model_seed)
    optimizer_factory = lambda parameters: torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    start_step = 0
    best_accuracy = -1.0
    best_step = 0
    if resume_checkpoint is None:
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = OPMModel(
            ModelConfig(len(Vocabulary.build().tokens), dropout=config.dropout), kind, model_seed
        ).to(torch_device)
        optimizer = optimizer_factory(model.parameters())
    else:
        model, optimizer, start_step, metadata = load_checkpoint(
            resume_checkpoint, optimizer_factory=optimizer_factory, device=torch_device
        )
        if metadata.get("run_id") != identifier or not metadata.get("primary_run") == canonical:
            raise ValueError("checkpoint identity does not match requested run")
        sampler.load_state_dict(dict(metadata["sampler_state"]))
        if sampler.next_step != start_step:
            raise ValueError("checkpoint and sampler steps differ")
        best_accuracy = float(metadata["best_accuracy"])
        best_step = int(metadata["best_step"])
    events_path = run_directory / "events.jsonl"
    final_loss = float("nan")
    terminal_step = config.max_steps if stop_after_steps is None else stop_after_steps
    if terminal_step <= start_step or terminal_step > config.max_steps:
        raise ValueError("controlled stop must be after the resumed step and within the run")
    if terminal_step % config.checkpoint_every_steps != 0:
        raise ValueError("controlled stop must coincide with a checkpoint boundary")
    for step_index in range(start_step, terminal_step):
        model.train()
        selection = sampler.next_batch()
        batch = train.batch_for_ids([item.example_id for item in selection], torch_device)
        completed_step = step_index + 1
        learning_rate = learning_rate_at_step(config, completed_step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(batch), batch.labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        event: dict[str, object] = {
            "step": completed_step,
            "loss": final_loss,
            "learning_rate": learning_rate,
        }
        if completed_step % config.validation_every_steps == 0:
            accuracy = macro_validation_accuracy(model, validation, torch_device)
            event["macro_validation_accuracy"] = accuracy
            if accuracy > best_accuracy + 0.001:
                best_accuracy = accuracy
                best_step = completed_step
            checkpoint_path = run_directory / "checkpoints" / f"step-{completed_step:05d}.pt"
            digest = save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=completed_step,
                metadata={
                    "sampler_state": sampler.state_dict(),
                    "run_id": identifier,
                    "validation_accuracy": accuracy,
                    "primary_run": canonical,
                    "best_accuracy": best_accuracy,
                    "best_step": best_step,
                },
            )
            event["checkpoint_sha256"] = digest
        with events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    selected_checkpoint = run_directory / "checkpoints" / f"step-{best_step:05d}.pt"
    best_digest = _sha256(selected_checkpoint)
    summary = TrainingSummary(
        run_id=identifier,
        model_kind=kind.value,
        model_seed=model_seed,
        completed_steps=terminal_step,
        final_loss=final_loss,
        selected_step=best_step,
        selected_macro_validation_accuracy=best_accuracy,
        selected_checkpoint_sha256=best_digest,
        train_sha256=train_hash,
        validation_sha256=validation_hash,
        primary_training=canonical,
    )
    summary_name = "summary.json" if terminal_step == config.max_steps else "interim-summary.json"
    (run_directory / summary_name).write_bytes(canonical_json_bytes(asdict(summary)))
    return summary
