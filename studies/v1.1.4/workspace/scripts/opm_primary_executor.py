from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from rlmgraph.opm.artifacts import run_id
from rlmgraph.opm.model import ModelKind
from rlmgraph.opm.primary_training import (
    CanonicalTrainingConfig,
    TrainingSummary,
    train_canonical_run,
)
from rlmgraph.opm.protocol import CURRENT_PROTOCOL

APPROVED_OPM_SOURCE_SHA256 = (
    "c885ef2eb39feaa5a1ab2116b5bc387fff4bf6713b4b1292befaf084787d6366"
)
CODE_REVISION = f"opm-source-sha256:{APPROVED_OPM_SOURCE_SHA256}"
CANONICAL_TRAIN_SHA256 = (
    "4f2c07bfc0400c992a936fa7b64f3fabefcd2f8b29064bf610a7c83c283deafa"
)
CANONICAL_VALIDATION_SHA256 = (
    "1dee006f0db36bf77925f6f375a8d776f4049059c2994d65ee1f1e9eed58236f"
)
DECLARED_MODEL_SEEDS = (1101, 2202, 3303, 4404, 5505)
PILOT_SEED = 1101
FROZEN_LEARNING_RATE = 0.0006
FROZEN_DROPOUT = 0.0
TIE_THRESHOLD = 0.001
GRID = tuple(
    (learning_rate, dropout)
    for learning_rate in (0.0001, 0.0003, 0.0006)
    for dropout in (0.0, 0.1)
)


@dataclass(frozen=True)
class FrozenSelection:
    learning_rate: float
    dropout: float
    maximum_mean: float
    tie_set: tuple[tuple[float, float], ...]
    pilot_matrix_sha256: str


@dataclass(frozen=True)
class PrimaryPreflight:
    state: str
    run_role: str
    run_id: str
    model_kind: str
    model_seed: int
    configuration: dict[str, object]
    run_directory: str
    selection_ledger: str
    selection_ledger_sha256: str
    primary_matrix: str
    primary_matrix_sha256: str
    primary_row_state: str
    code_revision: str
    train_sha256: str
    validation_sha256: str
    resume_checkpoint: str | None
    resume_step: int | None
    sealed_labels_accessed: bool
    aggregate_test_evaluation_performed: bool
    claim_decisions_performed: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object: {path}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-15)


def validate_frozen_selection(ledger_path: Path) -> FrozenSelection:
    ledger = _read_json(ledger_path, "pilot matrix ledger")
    if ledger.get("version") != "1.1.4":
        raise ValueError("pilot matrix ledger version is not v1.1.4")

    run_rows = ledger.get("runs")
    if not isinstance(run_rows, list) or len(run_rows) != 24:
        raise ValueError("pilot matrix ledger must contain exactly 24 rows")
    expected_rows = {
        (kind.value, learning_rate, dropout)
        for kind in ModelKind
        for learning_rate, dropout in GRID
    }
    observed_rows: set[tuple[str, float, float]] = set()
    observed_run_ids: set[str] = set()
    accuracies: dict[tuple[float, float], list[float]] = {pair: [] for pair in GRID}
    pilot_root = ledger_path.parent / "pilots"

    for index, raw_row in enumerate(run_rows):
        if not isinstance(raw_row, dict):
            raise TypeError(f"pilot matrix row {index} is not an object")
        condition = str(raw_row.get("condition"))
        learning_rate = _number(raw_row.get("learning_rate"), "pilot learning rate")
        dropout = _number(raw_row.get("dropout"), "pilot dropout")
        row_key = (condition, learning_rate, dropout)
        if row_key in observed_rows:
            raise ValueError(f"pilot matrix contains a duplicate row: {row_key!r}")
        observed_rows.add(row_key)
        if raw_row.get("state") != "COMPLETED":
            raise ValueError(f"pilot matrix row is not completed: {row_key!r}")
        run_identifier = raw_row.get("run_id")
        if not isinstance(run_identifier, str) or not run_identifier:
            raise ValueError(f"pilot matrix row has no run ID: {row_key!r}")
        if run_identifier in observed_run_ids:
            raise ValueError(f"pilot matrix contains a duplicate run ID: {run_identifier}")
        observed_run_ids.add(run_identifier)

        summary = _read_json(
            pilot_root / run_identifier / "summary.json", f"pilot summary {run_identifier}"
        )
        expected_summary = {
            "run_id": run_identifier,
            "model_kind": condition,
            "model_seed": PILOT_SEED,
            "completed_steps": 50_000,
            "primary_training": True,
        }
        observed_summary = {name: summary.get(name) for name in expected_summary}
        if observed_summary != expected_summary:
            raise ValueError(
                f"pilot summary identity does not match matrix row: {run_identifier}"
            )
        accuracy = _number(
            summary.get("selected_macro_validation_accuracy"),
            "selected-checkpoint macro-validation accuracy",
        )
        if accuracy < 0.0 or accuracy > 1.0:
            raise ValueError(f"pilot summary accuracy is outside [0, 1]: {run_identifier}")
        accuracies[(learning_rate, dropout)].append(accuracy)

    if observed_rows != expected_rows:
        missing = sorted(expected_rows - observed_rows)
        unexpected = sorted(observed_rows - expected_rows)
        raise ValueError(f"pilot matrix grid mismatch; missing={missing!r}, unexpected={unexpected!r}")

    means = {pair: sum(values) / 4 for pair, values in accuracies.items()}
    maximum_mean = max(means.values())
    tie_set = tuple(
        sorted(pair for pair, mean in means.items() if maximum_mean - mean <= TIE_THRESHOLD)
    )
    selected_pair = tie_set[0]

    selection = ledger.get("selection")
    if not isinstance(selection, dict) or selection.get("state") != "FROZEN":
        raise ValueError("pilot selection is not frozen")
    if selection.get("primary_model_seeds") != list(DECLARED_MODEL_SEEDS):
        raise ValueError("pilot selection does not declare the five frozen model seeds")
    if selection.get("pilot_runs_enter_claim_statistics") is not False:
        raise ValueError("pilot selection must exclude pilot runs from claim statistics")
    recorded_maximum = _number(selection.get("maximum_mean"), "recorded maximum mean")
    if not _same_number(recorded_maximum, maximum_mean):
        raise ValueError("recorded maximum pilot mean does not match pilot summaries")

    mean_rows = selection.get("means")
    if not isinstance(mean_rows, list) or len(mean_rows) != len(GRID):
        raise ValueError("pilot selection must record all six means")
    recorded_pairs: set[tuple[float, float]] = set()
    for raw_mean in mean_rows:
        if not isinstance(raw_mean, dict):
            raise TypeError("pilot selection mean row is not an object")
        pair = (
            _number(raw_mean.get("learning_rate"), "selection learning rate"),
            _number(raw_mean.get("dropout"), "selection dropout"),
        )
        if pair not in means or pair in recorded_pairs:
            raise ValueError(f"pilot selection contains an invalid mean row: {pair!r}")
        recorded_pairs.add(pair)
        recorded_mean = _number(raw_mean.get("mean"), "recorded pilot mean")
        recorded_gap = _number(raw_mean.get("gap_from_M"), "recorded gap from M")
        if not _same_number(recorded_mean, means[pair]):
            raise ValueError(f"recorded pilot mean does not match summaries: {pair!r}")
        if not _same_number(recorded_gap, maximum_mean - means[pair]):
            raise ValueError(f"recorded pilot gap does not match summaries: {pair!r}")
        if raw_mean.get("in_tie_set") is not (pair in tie_set):
            raise ValueError(f"recorded pilot tie membership is incorrect: {pair!r}")
    if recorded_pairs != set(GRID):
        raise ValueError("pilot selection mean rows do not cover the frozen grid")

    recorded_pair = (
        _number(selection.get("selected_learning_rate"), "selected learning rate"),
        _number(selection.get("selected_dropout"), "selected dropout"),
    )
    if recorded_pair != selected_pair:
        raise ValueError("recorded frozen pair does not follow the approved tie rule")
    if recorded_pair != (FROZEN_LEARNING_RATE, FROZEN_DROPOUT):
        raise ValueError("recorded frozen pair does not match the completed v1.1.4 selection")

    return FrozenSelection(
        learning_rate=selected_pair[0],
        dropout=selected_pair[1],
        maximum_mean=maximum_mean,
        tie_set=tie_set,
        pilot_matrix_sha256=sha256_file(ledger_path),
    )


def _expected_run_identity(
    *,
    kind: ModelKind,
    model_seed: int,
    configuration: dict[str, object],
    train_digest: str,
    validation_digest: str,
) -> str:
    run_configuration = {**configuration, "model_kind": kind.value}
    return run_id(
        configuration=run_configuration,
        specification_version="1.0.0",
        dataset_fingerprints={"train": train_digest, "validation": validation_digest},
        model_seed=model_seed,
        code_revision=CODE_REVISION,
    )


def _validate_primary_matrix(
    *,
    matrix_path: Path,
    selection_ledger: Path,
    selection: FrozenSelection,
    kind: ModelKind,
    model_seed: int,
    configuration: dict[str, object],
    train_digest: str,
    validation_digest: str,
    resume_requested: bool,
) -> tuple[dict[str, object], int, str, str]:
    matrix = _read_json(matrix_path, "primary matrix ledger")
    expected_top_level = {
        "version": "1.1.4",
        "artifact_root": "primary",
        "selection_ledger": selection_ledger.name,
        "selection_ledger_sha256": selection.pilot_matrix_sha256,
        "code_revision": CODE_REVISION,
        "train_sha256": train_digest,
        "validation_sha256": validation_digest,
        "declared_model_seeds": list(DECLARED_MODEL_SEEDS),
        "pilot_runs_enter_claim_statistics": False,
    }
    observed_top_level = {name: matrix.get(name) for name in expected_top_level}
    if observed_top_level != expected_top_level:
        raise ValueError("primary matrix provenance does not match the frozen execution inputs")
    matrix_state = matrix.get("state")
    if matrix_state not in {"READY", "RUNNING", "ATTENTION_REQUIRED", "COMPLETED"}:
        raise ValueError("primary matrix has an invalid lifecycle state")
    recorded_configuration = matrix.get("configuration")
    if recorded_configuration != {
        "learning_rate": selection.learning_rate,
        "dropout": selection.dropout,
    }:
        raise ValueError("primary matrix does not use the frozen hyperparameter pair")

    rows = matrix.get("runs")
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError("primary matrix must contain exactly 20 rows")
    expected_rows = {(member.value, seed) for member in ModelKind for seed in DECLARED_MODEL_SEEDS}
    observed_rows: set[tuple[str, int]] = set()
    requested_index = -1
    requested_state = ""
    requested_identifier = ""
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise TypeError(f"primary matrix row {index} is not an object")
        condition = str(raw_row.get("condition"))
        seed_value = raw_row.get("model_seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError(f"primary matrix row {index} has an invalid model seed")
        row_key = (condition, seed_value)
        if row_key in observed_rows:
            raise ValueError(f"primary matrix contains a duplicate row: {row_key!r}")
        observed_rows.add(row_key)
        try:
            row_kind = ModelKind(condition)
        except ValueError as error:
            raise ValueError(f"primary matrix has an invalid condition: {condition}") from error
        expected_identifier = _expected_run_identity(
            kind=row_kind,
            model_seed=seed_value,
            configuration=configuration,
            train_digest=train_digest,
            validation_digest=validation_digest,
        )
        if raw_row.get("expected_run_id") != expected_identifier:
            raise ValueError(f"primary matrix run identity is incorrect: {row_key!r}")
        state = raw_row.get("state")
        if state not in {"PENDING", "RUNNING", "INTERRUPTED", "FAILED_PRESTART", "COMPLETED"}:
            raise ValueError(f"primary matrix row has an invalid state: {row_key!r}")
        recorded_run_id = raw_row.get("run_id")
        if state == "PENDING" and recorded_run_id is not None:
            raise ValueError(f"pending primary row already records a run ID: {row_key!r}")
        if state != "PENDING" and recorded_run_id != expected_identifier:
            raise ValueError(f"started primary row has the wrong run ID: {row_key!r}")
        if row_key == (kind.value, model_seed):
            requested_index = index
            requested_state = str(state)
            requested_identifier = expected_identifier
    if observed_rows != expected_rows:
        raise ValueError("primary matrix does not cover all four conditions and five seeds")
    if requested_index < 0:
        raise ValueError("requested condition and seed are absent from the primary matrix")
    running_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("state") == "RUNNING"
    ]
    if resume_requested:
        if requested_state not in {"RUNNING", "INTERRUPTED"}:
            raise ValueError("primary recovery requires a RUNNING or INTERRUPTED ledger row")
        if any(
            row.get("condition") != kind.value or row.get("model_seed") != model_seed
            for row in running_rows
        ):
            raise ValueError("another primary row is already running")
    elif requested_state != "PENDING":
        raise ValueError("new primary execution requires a PENDING ledger row")
    elif running_rows or matrix_state == "ATTENTION_REQUIRED":
        raise ValueError("primary matrix requires reconciliation before a new row can start")
    return matrix, requested_index, requested_state, requested_identifier


def _atomic_json(path: Path, payload: dict[str, object]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return sha256_file(path)


def _transition_primary_row(
    preflight: PrimaryPreflight,
    *,
    expected_matrix_sha256: str,
    state: str,
    summary: TrainingSummary | None = None,
) -> str:
    matrix_path = Path(preflight.primary_matrix)
    if sha256_file(matrix_path) != expected_matrix_sha256:
        raise RuntimeError("primary matrix changed after preflight; refusing an unsafe transition")
    matrix = _read_json(matrix_path, "primary matrix ledger")
    rows = matrix.get("runs")
    if not isinstance(rows, list):
        raise TypeError("primary matrix runs must be a list")
    matching_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("condition") == preflight.model_kind
        and row.get("model_seed") == preflight.model_seed
    ]
    if len(matching_rows) != 1:
        raise RuntimeError("primary matrix no longer has exactly one requested row")
    row = matching_rows[0]
    if state == "RUNNING" and row.get("state") != preflight.primary_row_state:
        raise RuntimeError("primary row state changed after preflight")
    row["state"] = state
    row["run_id"] = preflight.run_id
    if summary is not None:
        row["completed_steps"] = summary.completed_steps
        row["selected_step"] = summary.selected_step
        row["selected_macro_validation_accuracy"] = summary.selected_macro_validation_accuracy
        row["selected_checkpoint_sha256"] = summary.selected_checkpoint_sha256
    states = {item.get("state") for item in rows if isinstance(item, dict)}
    if states == {"COMPLETED"}:
        matrix["state"] = "COMPLETED"
    elif "INTERRUPTED" in states or "FAILED_PRESTART" in states:
        matrix["state"] = "ATTENTION_REQUIRED"
    else:
        matrix["state"] = "RUNNING"
    return _atomic_json(matrix_path, matrix)


def _validate_resume_boundary(
    resume_checkpoint: Path,
    run_directory: Path,
    expected_manifest: dict[str, object],
) -> int:
    expected_checkpoint_root = (run_directory / "checkpoints").resolve()
    if resume_checkpoint.resolve().parent != expected_checkpoint_root:
        raise ValueError("resume checkpoint is not inside the requested content-addressed run")
    match = re.fullmatch(r"step-(\d{5})\.pt", resume_checkpoint.name)
    if not resume_checkpoint.is_file() or match is None:
        raise ValueError("resume checkpoint must be a complete five-digit checkpoint file")
    resume_step = int(match.group(1))
    if resume_step <= 0 or resume_step >= 50_000 or resume_step % 500:
        raise ValueError("resume checkpoint step is outside the canonical recovery boundary")
    if (run_directory / "summary.json").exists():
        raise ValueError("completed primary runs cannot be resumed")

    manifest = _read_json(run_directory / "run-manifest.json", "primary run manifest")
    if manifest != expected_manifest:
        raise ValueError("primary run manifest does not match the requested run")

    events_path = run_directory / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"primary event log is missing: {events_path}")
    event_count = 0
    checkpoint_digest: str | None = None
    with events_path.open("r", encoding="utf-8") as handle:
        for event_count, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"primary event log is invalid at line {event_count}") from error
            if not isinstance(event, dict) or event.get("step") != event_count:
                raise ValueError(f"primary event sequence is not contiguous at line {event_count}")
            if event_count == resume_step:
                value = event.get("checkpoint_sha256")
                checkpoint_digest = value if isinstance(value, str) else None
    if event_count != resume_step:
        raise ValueError(
            "event log must end exactly at the checkpoint selected for canonical recovery"
        )
    if checkpoint_digest != sha256_file(resume_checkpoint):
        raise ValueError("resume checkpoint hash does not match its event record")
    return resume_step


def preflight_primary_run(
    *,
    kind: ModelKind,
    model_seed: int,
    train_path: Path,
    validation_path: Path,
    selection_ledger: Path,
    resume_checkpoint: Path | None = None,
    expected_train_sha256: str = CANONICAL_TRAIN_SHA256,
    expected_validation_sha256: str = CANONICAL_VALIDATION_SHA256,
) -> PrimaryPreflight:
    CURRENT_PROTOCOL.require_primary()
    if model_seed not in DECLARED_MODEL_SEEDS:
        raise ValueError(f"model seed is not declared by DAT-003: {model_seed}")
    selection = validate_frozen_selection(selection_ledger)
    train_digest = sha256_file(train_path)
    validation_digest = sha256_file(validation_path)
    if train_digest != expected_train_sha256:
        raise ValueError("training input fingerprint does not match the canonical artifact")
    if validation_digest != expected_validation_sha256:
        raise ValueError("validation input fingerprint does not match the canonical artifact")

    config = CanonicalTrainingConfig(
        learning_rate=selection.learning_rate, dropout=selection.dropout
    )
    config.validate(canonical=True)
    configuration = {"model_kind": kind.value, **asdict(config)}
    identifier = _expected_run_identity(
        kind=kind,
        model_seed=model_seed,
        configuration=configuration,
        train_digest=train_digest,
        validation_digest=validation_digest,
    )
    primary_matrix = selection_ledger.parent / "primary-matrix.json"
    _, _, primary_row_state, matrix_identifier = _validate_primary_matrix(
        matrix_path=primary_matrix,
        selection_ledger=selection_ledger,
        selection=selection,
        kind=kind,
        model_seed=model_seed,
        configuration=configuration,
        train_digest=train_digest,
        validation_digest=validation_digest,
        resume_requested=resume_checkpoint is not None,
    )
    if matrix_identifier != identifier:
        raise ValueError("requested run identity does not match the primary matrix")
    output_root = primary_matrix.parent / "primary"
    run_directory = output_root / identifier
    expected_manifest = {
        "run_id": identifier,
        "lifecycle": "PRIMARY_RUNS",
        "primary_run": True,
        "model_seed": model_seed,
        "configuration": configuration,
        "dataset_fingerprints": {"train": train_digest, "validation": validation_digest},
        "code_revision": CODE_REVISION,
        "sealed_labels_accessed": False,
        "aggregate_test_evaluation": False,
    }
    resume_step = None
    if resume_checkpoint is None:
        if run_directory.exists():
            raise FileExistsError(
                "content-addressed primary run already exists; refuse duplicate or implicit restart"
            )
    else:
        resume_step = _validate_resume_boundary(
            resume_checkpoint, run_directory, expected_manifest
        )

    return PrimaryPreflight(
        state="PASS",
        run_role="PRIMARY_RESULT_REPLICATE",
        run_id=identifier,
        model_kind=kind.value,
        model_seed=model_seed,
        configuration=configuration,
        run_directory=str(run_directory),
        selection_ledger=str(selection_ledger),
        selection_ledger_sha256=selection.pilot_matrix_sha256,
        primary_matrix=str(primary_matrix),
        primary_matrix_sha256=sha256_file(primary_matrix),
        primary_row_state=primary_row_state,
        code_revision=CODE_REVISION,
        train_sha256=train_digest,
        validation_sha256=validation_digest,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None,
        resume_step=resume_step,
        sealed_labels_accessed=False,
        aggregate_test_evaluation_performed=False,
        claim_decisions_performed=False,
    )


def execute_primary_run(
    *,
    kind: ModelKind,
    model_seed: int,
    train_path: Path,
    validation_path: Path,
    selection_ledger: Path,
    resume_checkpoint: Path | None = None,
    device: str = "cuda:0",
    trainer: Callable[..., TrainingSummary] = train_canonical_run,
    expected_train_sha256: str = CANONICAL_TRAIN_SHA256,
    expected_validation_sha256: str = CANONICAL_VALIDATION_SHA256,
) -> tuple[PrimaryPreflight, TrainingSummary]:
    preflight = preflight_primary_run(
        kind=kind,
        model_seed=model_seed,
        train_path=train_path,
        validation_path=validation_path,
        selection_ledger=selection_ledger,
        resume_checkpoint=resume_checkpoint,
        expected_train_sha256=expected_train_sha256,
        expected_validation_sha256=expected_validation_sha256,
    )
    running_matrix_sha256 = _transition_primary_row(
        preflight,
        expected_matrix_sha256=preflight.primary_matrix_sha256,
        state="RUNNING",
    )
    try:
        summary = trainer(
            kind=kind,
            model_seed=model_seed,
            config=CanonicalTrainingConfig(
                learning_rate=FROZEN_LEARNING_RATE, dropout=FROZEN_DROPOUT
            ),
            train_path=train_path,
            validation_path=validation_path,
            output_root=selection_ledger.parent / "primary",
            code_revision=CODE_REVISION,
            device=device,
            canonical=True,
            resume_checkpoint=resume_checkpoint,
        )
    except BaseException:
        run_directory = Path(preflight.run_directory)
        has_checkpoint = any((run_directory / "checkpoints").glob("step-*.pt"))
        failure_state = "INTERRUPTED" if has_checkpoint else "FAILED_PRESTART"
        _transition_primary_row(
            preflight,
            expected_matrix_sha256=running_matrix_sha256,
            state=failure_state,
        )
        raise
    _transition_primary_row(
        preflight,
        expected_matrix_sha256=running_matrix_sha256,
        state="COMPLETED",
        summary=summary,
    )
    return preflight, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute one declared-seed OPM v1.1.4 primary-result replicate."
    )
    parser.add_argument("model_kind", choices=[kind.value for kind in ModelKind])
    parser.add_argument("model_seed", type=int, choices=DECLARED_MODEL_SEEDS)
    parser.add_argument("train_path", type=Path)
    parser.add_argument("validation_path", type=Path)
    parser.add_argument("selection_ledger", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "kind": ModelKind(args.model_kind),
        "model_seed": args.model_seed,
        "train_path": args.train_path,
        "validation_path": args.validation_path,
        "selection_ledger": args.selection_ledger,
        "resume_checkpoint": args.resume_checkpoint,
    }
    if args.preflight_only:
        preflight = preflight_primary_run(**common)
        print(json.dumps(asdict(preflight), indent=2, sort_keys=True))
        return
    preflight, summary = execute_primary_run(**common, device=args.device)
    print(
        json.dumps(
            {"preflight": asdict(preflight), "summary": asdict(summary)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
