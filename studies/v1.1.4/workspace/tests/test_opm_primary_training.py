import json
from pathlib import Path

import pytest
import torch

from rlmgraph.opm.artifacts import load_checkpoint
from rlmgraph.opm.model import ModelKind
from rlmgraph.opm.primary_training import (
    CanonicalTrainingConfig,
    learning_rate_at_step,
    train_canonical_run,
)
from rlmgraph.opm.splits import (
    SplitName,
    SplitValidationConfig,
    allocate_validation_split,
    record_dict,
)


def _write_split(path: Path, split: SplitName) -> None:
    records = allocate_validation_split(SplitValidationConfig(split, examples_per_cell=1))
    payload = "".join(
        json.dumps(record_dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _optimizer(parameters):
    return torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.98), eps=1e-8)


def test_learning_rate_schedule_endpoints() -> None:
    config = CanonicalTrainingConfig(
        learning_rate=1e-4, dropout=0.0, max_steps=4, warmup_steps=2
    )
    assert learning_rate_at_step(config, 1) == pytest.approx(5e-5)
    assert learning_rate_at_step(config, 2) == pytest.approx(1e-4)
    assert learning_rate_at_step(config, 4) == pytest.approx(3e-5)


def test_interrupted_resume_matches_uninterrupted(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_split(train_path, SplitName.TRAIN)
    _write_split(validation_path, SplitName.VALIDATION)
    config = CanonicalTrainingConfig(
        learning_rate=1e-4,
        dropout=0.0,
        max_steps=2,
        batch_size=256,
        warmup_steps=1,
        validation_every_steps=1,
        checkpoint_every_steps=1,
    )
    common = {
        "kind": ModelKind.OPM_SHARED,
        "model_seed": 1101,
        "config": config,
        "train_path": train_path,
        "validation_path": validation_path,
        "code_revision": "resume-test",
        "device": "cpu",
        "canonical": False,
    }
    uninterrupted = train_canonical_run(output_root=tmp_path / "whole", **common)
    interim = train_canonical_run(
        output_root=tmp_path / "resumed", stop_after_steps=1, **common
    )
    resumed_directory = tmp_path / "resumed" / interim.run_id
    resumed = train_canonical_run(
        output_root=tmp_path / "resumed",
        resume_checkpoint=resumed_directory / "checkpoints" / "step-00001.pt",
        **common,
    )
    whole_checkpoint = (
        tmp_path / "whole" / uninterrupted.run_id / "checkpoints" / "step-00002.pt"
    )
    resumed_checkpoint = resumed_directory / "checkpoints" / "step-00002.pt"
    whole_model, _, whole_step, _ = load_checkpoint(
        whole_checkpoint, optimizer_factory=_optimizer
    )
    resumed_model, _, resumed_step, _ = load_checkpoint(
        resumed_checkpoint, optimizer_factory=_optimizer
    )
    assert whole_step == resumed_step == 2
    assert all(
        torch.equal(whole_model.state_dict()[name], value)
        for name, value in resumed_model.state_dict().items()
    )
    assert uninterrupted.final_loss == resumed.final_loss
    assert uninterrupted.selected_step == resumed.selected_step
    assert (
        uninterrupted.selected_macro_validation_accuracy
        == resumed.selected_macro_validation_accuracy
    )
