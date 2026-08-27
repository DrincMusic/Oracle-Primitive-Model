import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rlmgraph.opm.artifacts import run_id
from rlmgraph.opm.model import ModelKind
from rlmgraph.opm.primary_training import CanonicalTrainingConfig, TrainingSummary
from scripts.opm_primary_executor import (
    CODE_REVISION,
    DECLARED_MODEL_SEEDS,
    FROZEN_DROPOUT,
    FROZEN_LEARNING_RATE,
    execute_primary_run,
    preflight_primary_run,
    validate_frozen_selection,
)

PILOT_ACCURACIES = {
    (0.0001, 0.0): (0.9975, 0.9938, 0.9943, 0.9913),
    (0.0001, 0.1): (0.9924, 0.9915, 0.9896, 0.9867),
    (0.0003, 0.0): (0.9980, 0.9972, 0.9972, 0.9974),
    (0.0003, 0.1): (0.9963, 0.9825, 0.9942, 0.9913),
    (0.0006, 0.0): (0.9993, 0.9988, 0.9987, 0.9972),
    (0.0006, 0.1): (0.9995, 0.9986, 0.9993, 0.9988),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_selection_ledger(tmp_path: Path) -> Path:
    rows = []
    means = []
    condition_order = list(ModelKind)
    for condition_index, kind in enumerate(condition_order):
        for learning_rate, dropout in PILOT_ACCURACIES:
            run_identifier = (
                f"{kind.value.lower()}-{learning_rate:.4f}-{dropout:.1f}".replace(".", "-")
            )
            rows.append(
                {
                    "condition": kind.value,
                    "learning_rate": learning_rate,
                    "dropout": dropout,
                    "state": "COMPLETED",
                    "run_id": run_identifier,
                }
            )
            run_directory = tmp_path / "pilots" / run_identifier
            run_directory.mkdir(parents=True, exist_ok=True)
            summary = {
                "run_id": run_identifier,
                "model_kind": kind.value,
                "model_seed": 1101,
                "completed_steps": 50_000,
                "primary_training": True,
                "selected_macro_validation_accuracy": PILOT_ACCURACIES[
                    (learning_rate, dropout)
                ][condition_index],
            }
            (run_directory / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

    computed_means = {
        pair: sum(accuracies) / len(accuracies)
        for pair, accuracies in PILOT_ACCURACIES.items()
    }
    maximum = max(computed_means.values())
    tie_set = {pair for pair, mean in computed_means.items() if maximum - mean <= 0.001}
    for pair, mean in computed_means.items():
        means.append(
            {
                "learning_rate": pair[0],
                "dropout": pair[1],
                "mean": mean,
                "gap_from_M": maximum - mean,
                "in_tie_set": pair in tie_set,
            }
        )
    selected = min(tie_set)
    ledger = {
        "version": "1.1.4",
        "runs": rows,
        "selection": {
            "state": "FROZEN",
            "means": means,
            "maximum_mean": maximum,
            "selected_learning_rate": selected[0],
            "selected_dropout": selected[1],
            "primary_model_seeds": list(DECLARED_MODEL_SEEDS),
            "pilot_runs_enter_claim_statistics": False,
        },
    }
    path = tmp_path / "pilot-matrix.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_bytes(b"canonical-train-fixture\n")
    validation_path.write_bytes(b"canonical-validation-fixture\n")
    return train_path, validation_path


def _write_primary_matrix(
    tmp_path: Path, ledger: Path, train_path: Path, validation_path: Path
) -> Path:
    config = CanonicalTrainingConfig(
        learning_rate=FROZEN_LEARNING_RATE, dropout=FROZEN_DROPOUT
    )
    rows = []
    for kind in ModelKind:
        configuration = {"model_kind": kind.value, **asdict(config)}
        for seed in DECLARED_MODEL_SEEDS:
            identifier = run_id(
                configuration=configuration,
                specification_version="1.0.0",
                dataset_fingerprints={
                    "train": _sha256(train_path),
                    "validation": _sha256(validation_path),
                },
                model_seed=seed,
                code_revision=CODE_REVISION,
            )
            rows.append(
                {
                    "condition": kind.value,
                    "model_seed": seed,
                    "state": "PENDING",
                    "expected_run_id": identifier,
                    "run_id": None,
                }
            )
    payload = {
        "version": "1.1.4",
        "state": "READY",
        "artifact_root": "primary",
        "selection_ledger": ledger.name,
        "selection_ledger_sha256": _sha256(ledger),
        "code_revision": CODE_REVISION,
        "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path),
        "configuration": {
            "learning_rate": FROZEN_LEARNING_RATE,
            "dropout": FROZEN_DROPOUT,
        },
        "declared_model_seeds": list(DECLARED_MODEL_SEEDS),
        "pilot_runs_enter_claim_statistics": False,
        "runs": rows,
    }
    path = tmp_path / "primary-matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preflight(tmp_path: Path):
    ledger = _write_selection_ledger(tmp_path)
    train_path, validation_path = _inputs(tmp_path)
    _write_primary_matrix(tmp_path, ledger, train_path, validation_path)
    result = preflight_primary_run(
        kind=ModelKind.PROC_CLONE,
        model_seed=2202,
        train_path=train_path,
        validation_path=validation_path,
        selection_ledger=ledger,
        expected_train_sha256=_sha256(train_path),
        expected_validation_sha256=_sha256(validation_path),
    )
    return result, ledger, train_path, validation_path


def test_frozen_selection_is_recomputed_from_all_pilot_summaries(tmp_path: Path) -> None:
    ledger = _write_selection_ledger(tmp_path)
    selection = validate_frozen_selection(ledger)
    assert selection.learning_rate == FROZEN_LEARNING_RATE
    assert selection.dropout == FROZEN_DROPOUT
    assert selection.tie_set == ((0.0006, 0.0), (0.0006, 0.1))
    assert selection.pilot_matrix_sha256 == _sha256(ledger)


def test_frozen_selection_rejects_incomplete_or_tampered_ledger(tmp_path: Path) -> None:
    ledger = _write_selection_ledger(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["runs"][0]["state"] = "RUNNING"
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        validate_frozen_selection(ledger)

    ledger = _write_selection_ledger(tmp_path)
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["selection"]["selected_dropout"] = 0.1
    ledger.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="approved tie rule"):
        validate_frozen_selection(ledger)


def test_preflight_accepts_only_declared_seed_and_derives_primary_identity(
    tmp_path: Path,
) -> None:
    result, ledger, _, _ = _preflight(tmp_path)
    assert result.state == "PASS"
    assert result.run_role == "PRIMARY_RESULT_REPLICATE"
    assert result.model_kind == "PROC_CLONE"
    assert result.model_seed == 2202
    assert result.configuration["learning_rate"] == FROZEN_LEARNING_RATE
    assert result.configuration["dropout"] == FROZEN_DROPOUT
    assert Path(result.run_directory).parent == ledger.parent / "primary"
    assert not Path(result.run_directory).exists()
    assert result.code_revision == CODE_REVISION

    with pytest.raises(ValueError, match="not declared"):
        preflight_primary_run(
            kind=ModelKind.OPM_SHARED,
            model_seed=9999,
            train_path=Path("missing-train"),
            validation_path=Path("missing-validation"),
            selection_ledger=Path("missing-ledger"),
        )


def test_executor_dispatches_frozen_canonical_configuration(tmp_path: Path) -> None:
    ledger = _write_selection_ledger(tmp_path)
    train_path, validation_path = _inputs(tmp_path)
    primary_matrix = _write_primary_matrix(tmp_path, ledger, train_path, validation_path)
    calls = []

    def trainer(**kwargs):
        calls.append(kwargs)
        return TrainingSummary(
            run_id="stub-run",
            model_kind=kwargs["kind"].value,
            model_seed=kwargs["model_seed"],
            completed_steps=50_000,
            final_loss=0.0,
            selected_step=50_000,
            selected_macro_validation_accuracy=1.0,
            selected_checkpoint_sha256="a" * 64,
            train_sha256=_sha256(train_path),
            validation_sha256=_sha256(validation_path),
            primary_training=True,
        )

    preflight, summary = execute_primary_run(
        kind=ModelKind.DOMAIN_GENERALIST,
        model_seed=5505,
        train_path=train_path,
        validation_path=validation_path,
        selection_ledger=ledger,
        device="cpu",
        trainer=trainer,
        expected_train_sha256=_sha256(train_path),
        expected_validation_sha256=_sha256(validation_path),
    )
    assert preflight.state == "PASS"
    assert summary.run_id == "stub-run"
    assert len(calls) == 1
    assert calls[0]["canonical"] is True
    assert calls[0]["model_seed"] == 5505
    assert calls[0]["config"].learning_rate == FROZEN_LEARNING_RATE
    assert calls[0]["config"].dropout == FROZEN_DROPOUT
    assert calls[0]["output_root"] == ledger.parent / "primary"
    assert calls[0]["code_revision"] == CODE_REVISION
    matrix = json.loads(primary_matrix.read_text(encoding="utf-8"))
    completed = next(
        row
        for row in matrix["runs"]
        if row["condition"] == "DOMAIN_GENERALIST" and row["model_seed"] == 5505
    )
    assert completed["state"] == "COMPLETED"
    assert completed["run_id"] == preflight.run_id


def test_executor_records_a_prestart_failure_without_silent_restart(tmp_path: Path) -> None:
    ledger = _write_selection_ledger(tmp_path)
    train_path, validation_path = _inputs(tmp_path)
    primary_matrix = _write_primary_matrix(tmp_path, ledger, train_path, validation_path)

    def failing_trainer(**_kwargs):
        raise RuntimeError("controlled prestart failure")

    with pytest.raises(RuntimeError, match="controlled prestart failure"):
        execute_primary_run(
            kind=ModelKind.OPM_SHARED,
            model_seed=3303,
            train_path=train_path,
            validation_path=validation_path,
            selection_ledger=ledger,
            device="cpu",
            trainer=failing_trainer,
            expected_train_sha256=_sha256(train_path),
            expected_validation_sha256=_sha256(validation_path),
        )
    matrix = json.loads(primary_matrix.read_text(encoding="utf-8"))
    failed = next(
        row
        for row in matrix["runs"]
        if row["condition"] == "OPM_SHARED" and row["model_seed"] == 3303
    )
    assert matrix["state"] == "ATTENTION_REQUIRED"
    assert failed["state"] == "FAILED_PRESTART"
    assert failed["run_id"] == failed["expected_run_id"]
    with pytest.raises(ValueError, match="requires reconciliation"):
        preflight_primary_run(
            kind=ModelKind.OPM_SHARED,
            model_seed=4404,
            train_path=train_path,
            validation_path=validation_path,
            selection_ledger=ledger,
            expected_train_sha256=_sha256(train_path),
            expected_validation_sha256=_sha256(validation_path),
        )


def test_resume_preflight_requires_exact_event_and_checkpoint_boundary(
    tmp_path: Path,
) -> None:
    first, ledger, train_path, validation_path = _preflight(tmp_path)
    run_directory = Path(first.run_directory)
    checkpoint_root = run_directory / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    checkpoint = checkpoint_root / "step-00500.pt"
    checkpoint.write_bytes(b"durable-checkpoint")
    checkpoint_digest = _sha256(checkpoint)
    manifest = {
        "run_id": first.run_id,
        "lifecycle": "PRIMARY_RUNS",
        "primary_run": True,
        "model_seed": first.model_seed,
        "configuration": first.configuration,
        "dataset_fingerprints": {
            "train": first.train_sha256,
            "validation": first.validation_sha256,
        },
        "code_revision": CODE_REVISION,
        "sealed_labels_accessed": False,
        "aggregate_test_evaluation": False,
    }
    (run_directory / "run-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    events = [json.dumps({"step": step}) for step in range(1, 500)]
    events.append(json.dumps({"step": 500, "checkpoint_sha256": checkpoint_digest}))
    (run_directory / "events.jsonl").write_text(
        "\n".join(events) + "\n", encoding="utf-8"
    )
    matrix_path = ledger.parent / "primary-matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in matrix["runs"]
        if item["condition"] == "PROC_CLONE" and item["model_seed"] == 2202
    )
    row["state"] = "RUNNING"
    row["run_id"] = first.run_id
    matrix["state"] = "RUNNING"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

    resumed = preflight_primary_run(
        kind=ModelKind.PROC_CLONE,
        model_seed=2202,
        train_path=train_path,
        validation_path=validation_path,
        selection_ledger=ledger,
        resume_checkpoint=checkpoint,
        expected_train_sha256=_sha256(train_path),
        expected_validation_sha256=_sha256(validation_path),
    )
    assert resumed.run_id == first.run_id
    assert resumed.resume_step == 500

    with (run_directory / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 501}) + "\n")
    with pytest.raises(ValueError, match="end exactly"):
        preflight_primary_run(
            kind=ModelKind.PROC_CLONE,
            model_seed=2202,
            train_path=train_path,
            validation_path=validation_path,
            selection_ledger=ledger,
            resume_checkpoint=checkpoint,
            expected_train_sha256=_sha256(train_path),
            expected_validation_sha256=_sha256(validation_path),
        )
