from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np
import torch

from rlmgraph.opm.artifacts import canonical_json_bytes
from rlmgraph.opm.model import ModelKind, OPMBatch, OPMModel

PROTOCOL_VERSION = "v1.1.4"
DECLARED_CONDITIONS = tuple(kind.value for kind in ModelKind)
DECLARED_SEEDS = (1101, 2202, 3303, 4404, 5505)
TEST_SPLITS = (
    "test-recombination",
    "test-interpolation",
    "test-renderer",
    "test-structural",
)
TRANSITION_SCHEMA = "opm-post-primary-transition-v1"
CHECKPOINT_MANIFEST_SCHEMA = "opm-selected-checkpoint-manifest-v1"
EVALUATION_SPEC_SCHEMA = "opm-stage1-evaluation-spec-v1"
PROBE_SPEC_SCHEMA = "opm-stage1-probe-spec-v1"
INTERVENTION_SPEC_SCHEMA = "opm-stage1-intervention-spec-v1"
RESOURCE_MANIFEST_SCHEMA = "opm-stage1-resource-manifest-v1"
JOB_SPEC_SCHEMA = "opm-stage1-job-spec-v1"
PREDICTION_SCHEMA = "opm-stage1-label-blind-prediction-v1"
INTERVENTION_SCHEMA = "opm-stage1-intervention-prediction-v1"
PROBE_SCHEMA = "opm-stage1-neural-probe-v1"
ARTIFACT_MANIFEST_SCHEMA = "opm-stage1-artifact-manifest-v1"
RECONCILIATION_SCHEMA = "opm-stage1-reconciliation-v1"
FREEZE_SCHEMA = "opm-stage1-freeze-v1"
FREEZE_FILENAME = "OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json"

AUTHORIZED_OPERATIONS = (
    "canonical_probe_generation",
    "canonical_probe_fitting",
    "label_blind_prediction_generation",
    "causal_intervention_prediction_generation",
    "artifact_reconciliation",
    "artifact_freezing",
)
PROHIBITED_OPERATIONS = (
    "base_model_training",
    "checkpoint_reselection",
    "sealed_label_access",
    "sealed_metric_computation",
    "bootstrap_effect_estimation",
    "claim_decision",
)

FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "correct",
        "correctness",
        "accuracy",
        "loss",
        "paired_effect",
        "pass",
        "passed",
        "claim",
        "h1",
    }
)
FORBIDDEN_OUTPUT_FIELDS = FORBIDDEN_TARGET_FIELDS | frozenset(
    {"aggregate", "bootstrap", "confidence_interval", "p_value"}
)
SEALED_PATH_MARKERS = frozenset(
    {"sealed-labels", "sealed_labels", "sealed-targets", "sealed_targets"}
)


class Stage1Error(RuntimeError):
    """Base class for fail-closed Stage 1 errors."""


class AuthorizationError(Stage1Error, PermissionError):
    pass


class IntegrityError(Stage1Error, ValueError):
    pass


class ReconciliationError(Stage1Error, ValueError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise IntegrityError(f"missing {description}: {path}") from error
    except json.JSONDecodeError as error:
        raise IntegrityError(f"invalid {description}: {path}") from error
    if not isinstance(payload, dict):
        raise IntegrityError(f"{description} must be a JSON object: {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, immutable: bool = True) -> str:
    """Validate bytes, fsync a temporary sibling, and atomically install without overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists():
        existing = sha256_file(path)
        if immutable or existing != digest:
            raise FileExistsError(f"refusing to overwrite immutable Stage 1 artifact: {path}")
        return existing
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temporary) != digest:
            raise IntegrityError(f"temporary artifact failed content verification: {path}")
        if path.exists():
            raise FileExistsError(f"artifact appeared during atomic write: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def atomic_write_json(path: Path, payload: object, *, immutable: bool = True) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(payload), immutable=immutable)


def relative_artifact_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    root = workspace.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise IntegrityError(f"artifact is outside the workspace: {resolved}") from error


def resolve_artifact_path(value: object, workspace: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{description} path is missing")
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as error:
        raise IntegrityError(f"{description} escapes the workspace: {resolved}") from error
    reject_sealed_path(resolved)
    return resolved


def reject_sealed_path(path: Path) -> None:
    markers = {part.casefold() for part in path.parts}
    if markers & SEALED_PATH_MARKERS:
        raise AuthorizationError(f"sealed-label resource is prohibited in Stage 1: {path}")


def validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{description} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise IntegrityError(f"{description} is not a SHA-256 digest") from error
    return value.lower()


def git_commit(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_manifest(paths: Sequence[Path], workspace: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted((item.resolve() for item in paths), key=str):
        if not path.is_file():
            raise IntegrityError(f"executor source is missing: {path}")
        records.append(
            {
                "path": relative_artifact_path(path, workspace),
                "sha256": sha256_file(path),
            }
        )
    return records


class Capability(StrEnum):
    CHECKPOINT_READ = "checkpoint_read"
    BASE_MODEL_FORWARD = "base_model_forward"
    CANONICAL_ACTIVATION_CAPTURE = "canonical_activation_capture"
    CANONICAL_PROBE_FIT = "canonical_probe_fit"
    LABEL_BLIND_PREDICTION = "label_blind_prediction"
    INTERVENTION_FORWARD = "intervention_forward"
    ARTIFACT_RECONCILIATION = "artifact_reconciliation"
    ARTIFACT_FREEZING = "artifact_freezing"
    BASE_MODEL_PARAMETER_UPDATE = "base_model_parameter_update"
    CHECKPOINT_WRITE = "checkpoint_write"
    CHECKPOINT_SELECTION = "checkpoint_selection"
    SEALED_LABEL_READ = "sealed_label_read"
    AGGREGATE_TEST_SCORING = "aggregate_test_scoring"
    CLAIM_THRESHOLD_APPLICATION = "claim_threshold_application"


_GRANTS: dict[str, tuple[Capability, ...]] = {
    "canonical_probe_generation": (
        Capability.CHECKPOINT_READ,
        Capability.BASE_MODEL_FORWARD,
        Capability.CANONICAL_ACTIVATION_CAPTURE,
    ),
    "canonical_probe_fitting": (Capability.CANONICAL_PROBE_FIT,),
    "label_blind_prediction_generation": (
        Capability.CHECKPOINT_READ,
        Capability.BASE_MODEL_FORWARD,
        Capability.LABEL_BLIND_PREDICTION,
    ),
    "causal_intervention_prediction_generation": (
        Capability.CHECKPOINT_READ,
        Capability.BASE_MODEL_FORWARD,
        Capability.INTERVENTION_FORWARD,
    ),
    "artifact_reconciliation": (Capability.ARTIFACT_RECONCILIATION,),
    "artifact_freezing": (Capability.ARTIFACT_FREEZING,),
}

_ALWAYS_DENIED = frozenset(
    {
        Capability.BASE_MODEL_PARAMETER_UPDATE,
        Capability.CHECKPOINT_WRITE,
        Capability.CHECKPOINT_SELECTION,
        Capability.SEALED_LABEL_READ,
        Capability.AGGREGATE_TEST_SCORING,
        Capability.CLAIM_THRESHOLD_APPLICATION,
    }
)


class CapabilityGate:
    """Deny-all gate populated only by a verified transition record."""

    def __init__(self) -> None:
        self._enabled: set[Capability] = set()
        self._transition_sha256: str | None = None

    @classmethod
    def from_transition(
        cls, transition: Mapping[str, Any], transition_sha256: str
    ) -> CapabilityGate:
        if transition.get("schema_version") != TRANSITION_SCHEMA:
            raise AuthorizationError("transition schema is not authorized")
        authorized = transition.get("authorized_operations")
        prohibited = transition.get("prohibited_operations")
        if authorized != list(AUTHORIZED_OPERATIONS):
            raise AuthorizationError("transition grants an unexpected Stage 1 operation set")
        if prohibited != list(PROHIBITED_OPERATIONS):
            raise AuthorizationError("transition does not preserve the complete prohibition set")
        gate = cls()
        for operation in authorized:
            gate._enabled.update(_GRANTS[str(operation)])
        gate._enabled.difference_update(_ALWAYS_DENIED)
        gate._transition_sha256 = validate_sha256(transition_sha256, "transition content hash")
        return gate

    def require(self, capability: Capability) -> None:
        if capability in _ALWAYS_DENIED or capability not in self._enabled:
            raise AuthorizationError(f"Stage 1 capability denied: {capability.value}")

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(item.value for item in self._enabled))


@dataclass(frozen=True)
class ResourcePolicy:
    manifest_sha256: str
    readable_roots: tuple[Path, ...]
    writable_root: Path

    @classmethod
    def load(cls, manifest_path: Path, workspace: Path) -> ResourcePolicy:
        reject_sealed_path(manifest_path)
        manifest = read_json(manifest_path, "Stage 1 resource manifest")
        if manifest.get("schema_version") != RESOURCE_MANIFEST_SCHEMA:
            raise AuthorizationError("resource manifest schema is not authorized")
        if manifest.get("sealed_labels_mounted") is not False:
            raise AuthorizationError("resource manifest must attest sealed labels are unmounted")
        prohibited = manifest.get("prohibited_resources")
        if prohibited != sorted(SEALED_PATH_MARKERS):
            raise AuthorizationError("resource manifest lacks the frozen sealed-resource deny list")
        roots_raw = manifest.get("readable_roots")
        if not isinstance(roots_raw, list) or not roots_raw:
            raise AuthorizationError("resource manifest has no readable roots")
        roots: list[Path] = []
        for value in roots_raw:
            root = resolve_artifact_path(value, workspace, "readable resource root")
            if not root.exists():
                raise AuthorizationError(f"declared readable resource is missing: {root}")
            roots.append(root)
        writable = resolve_artifact_path(
            manifest.get("writable_root"), workspace, "writable resource root"
        )
        return cls(sha256_file(manifest_path), tuple(roots), writable)

    def require_read(self, path: Path) -> Path:
        reject_sealed_path(path)
        resolved = path.resolve()
        for root in self.readable_roots:
            if resolved == root or root.is_dir() and root in resolved.parents:
                return resolved
        raise AuthorizationError(f"path is not in the Stage 1 read allowlist: {resolved}")

    def require_write(self, path: Path) -> Path:
        reject_sealed_path(path)
        resolved = path.resolve()
        if resolved == self.writable_root or self.writable_root in resolved.parents:
            return resolved
        raise AuthorizationError(f"path is not in the Stage 1 write allowlist: {resolved}")


def _event_at_step(path: Path, selected_step: int) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    event_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for event_count, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError(
                    f"invalid primary event at line {event_count}: {path}"
                ) from error
            if not isinstance(event, dict) or event.get("step") != event_count:
                raise IntegrityError(f"noncontiguous primary event sequence: {path}")
            if event_count == selected_step:
                selected = event
    if event_count != 50_000:
        raise IntegrityError(f"primary event log is incomplete: {path}")
    if selected is None:
        raise IntegrityError(f"selected step is absent from primary events: {path}")
    return selected


def build_selected_checkpoint_manifest(
    *, workspace: Path, primary_matrix_path: Path, dataset_version: str = "v1.1.3"
) -> dict[str, Any]:
    matrix = read_json(primary_matrix_path, "completed primary matrix")
    if matrix.get("state") != "COMPLETED":
        raise IntegrityError("primary matrix is not complete")
    rows = matrix.get("runs")
    if not isinstance(rows, list) or len(rows) != 20:
        raise IntegrityError("primary matrix must contain exactly 20 rows")
    expected = {(condition, seed) for condition in DECLARED_CONDITIONS for seed in DECLARED_SEEDS}
    identities: set[tuple[str, int]] = set()
    entries: list[dict[str, Any]] = []
    primary_root = primary_matrix_path.parent / str(matrix.get("artifact_root", "primary"))
    for row in rows:
        if not isinstance(row, dict):
            raise IntegrityError("primary matrix row is not an object")
        condition = str(row.get("condition"))
        seed = int(row.get("model_seed", -1))
        identity = (condition, seed)
        if identity in identities:
            raise IntegrityError(f"duplicate primary identity: {condition}/{seed}")
        identities.add(identity)
        if identity not in expected:
            raise IntegrityError(f"undeclared primary identity: {condition}/{seed}")
        if row.get("state") != "COMPLETED" or row.get("completed_steps") != 50_000:
            raise IntegrityError(f"primary row is not complete: {condition}/{seed}")
        run_id = str(row.get("run_id"))
        if run_id != row.get("expected_run_id"):
            raise IntegrityError(f"primary run identity mismatch: {condition}/{seed}")
        run_root = primary_root / run_id
        run_manifest_path = run_root / "run-manifest.json"
        summary_path = run_root / "summary.json"
        events_path = run_root / "events.jsonl"
        run_manifest = read_json(run_manifest_path, "primary run manifest")
        summary = read_json(summary_path, "primary training summary")
        selected_step = int(row.get("selected_step", -1))
        checkpoint_path = run_root / "checkpoints" / f"step-{selected_step:05d}.pt"
        checkpoint_sha256 = sha256_file(checkpoint_path)
        recorded_checkpoint_sha256 = validate_sha256(
            row.get("selected_checkpoint_sha256"), "matrix checkpoint hash"
        )
        if checkpoint_sha256 != recorded_checkpoint_sha256:
            raise IntegrityError(f"selected checkpoint byte hash mismatch: {condition}/{seed}")
        selected_event = _event_at_step(events_path, selected_step)
        validation_result = float(row.get("selected_macro_validation_accuracy"))
        required_matches = {
            "manifest run": run_manifest.get("run_id") == run_id,
            "manifest seed": run_manifest.get("model_seed") == seed,
            "manifest condition": (
                isinstance(run_manifest.get("configuration"), dict)
                and run_manifest["configuration"].get("model_kind") == condition
            ),
            "summary run": summary.get("run_id") == run_id,
            "summary condition": summary.get("model_kind") == condition,
            "summary seed": summary.get("model_seed") == seed,
            "summary step": summary.get("selected_step") == selected_step,
            "summary checkpoint": summary.get("selected_checkpoint_sha256") == checkpoint_sha256,
            "summary validation": math.isclose(
                float(summary.get("selected_macro_validation_accuracy")),
                validation_result,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "event checkpoint": selected_event.get("checkpoint_sha256") == checkpoint_sha256,
            "event validation": math.isclose(
                float(selected_event.get("macro_validation_accuracy")),
                validation_result,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
        }
        failures = [name for name, passes in required_matches.items() if not passes]
        if failures:
            raise IntegrityError(
                f"checkpoint reconciliation failed for {condition}/{seed}: {', '.join(failures)}"
            )
        configuration = run_manifest.get("configuration")
        dataset_fingerprints = run_manifest.get("dataset_fingerprints")
        if not isinstance(configuration, dict) or not isinstance(dataset_fingerprints, dict):
            raise IntegrityError(f"run manifest bindings are invalid: {condition}/{seed}")
        reconciliation = {
            "matrix_row_sha256": canonical_sha256(row),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "summary_sha256": sha256_file(summary_path),
            "events_sha256": sha256_file(events_path),
            "selected_event_sha256": canonical_sha256(selected_event),
            "checkpoint_sha256": checkpoint_sha256,
        }
        entries.append(
            {
                "condition": condition,
                "training_seed": seed,
                "run_id": run_id,
                "selected_step": selected_step,
                "selected_validation_result": validation_result,
                "checkpoint_path": relative_artifact_path(checkpoint_path, workspace),
                "checkpoint_artifact_id": f"{run_id}:step-{selected_step:05d}",
                "checkpoint_sha256": checkpoint_sha256,
                "training_configuration_sha256": canonical_sha256(configuration),
                "dataset_version": dataset_version,
                "dataset_fingerprints": dataset_fingerprints,
                "training_code_revision": run_manifest.get("code_revision"),
                "repository_commit": git_commit(workspace),
                "reconciliation": reconciliation,
                "reconciliation_record_sha256": canonical_sha256(reconciliation),
            }
        )
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise IntegrityError(f"primary matrix identity mismatch; missing={missing}, extra={extra}")
    entries.sort(
        key=lambda item: (DECLARED_CONDITIONS.index(item["condition"]), item["training_seed"])
    )
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "primary_matrix_path": relative_artifact_path(primary_matrix_path, workspace),
        "primary_matrix_sha256": sha256_file(primary_matrix_path),
        "checkpoint_count": len(entries),
        "conditions": list(DECLARED_CONDITIONS),
        "training_seeds": list(DECLARED_SEEDS),
        "entries": entries,
    }


def verify_selected_checkpoint_manifest(
    manifest: Mapping[str, Any], *, workspace: Path, expected_primary_matrix_sha256: str
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise IntegrityError("selected checkpoint manifest schema mismatch")
    if manifest.get("primary_matrix_sha256") != expected_primary_matrix_sha256:
        raise IntegrityError("selected checkpoint manifest is bound to the wrong primary matrix")
    entries = manifest.get("entries")
    if (
        not isinstance(entries, list)
        or manifest.get("checkpoint_count") != 20
        or len(entries) != 20
    ):
        raise IntegrityError("selected checkpoint manifest must contain exactly 20 entries")
    expected = {(condition, seed) for condition in DECLARED_CONDITIONS for seed in DECLARED_SEEDS}
    identities: set[tuple[str, int]] = set()
    verified: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            raise IntegrityError("checkpoint entry is not an object")
        identity = (str(raw.get("condition")), int(raw.get("training_seed", -1)))
        if identity in identities:
            raise IntegrityError(f"duplicate checkpoint identity: {identity}")
        identities.add(identity)
        if identity not in expected:
            raise IntegrityError(f"unapproved checkpoint identity: {identity}")
        selected_step = int(raw.get("selected_step", -1))
        expected_artifact_id = f"{raw.get('run_id')}:step-{selected_step:05d}"
        if raw.get("checkpoint_artifact_id") != expected_artifact_id:
            raise IntegrityError(f"wrong selected step or artifact id: {identity}")
        checkpoint = resolve_artifact_path(raw.get("checkpoint_path"), workspace, "checkpoint")
        expected_hash = validate_sha256(raw.get("checkpoint_sha256"), "checkpoint hash")
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
            raise IntegrityError(f"altered or missing checkpoint bytes: {identity}")
        reconciliation = raw.get("reconciliation")
        if not isinstance(reconciliation, dict) or canonical_sha256(reconciliation) != raw.get(
            "reconciliation_record_sha256"
        ):
            raise IntegrityError(f"checkpoint reconciliation record mismatch: {identity}")
        if reconciliation.get("checkpoint_sha256") != expected_hash:
            raise IntegrityError(f"unapproved checkpoint reconciliation: {identity}")
        verified.append(dict(raw))
    if identities != expected:
        raise IntegrityError(f"missing checkpoint identities: {sorted(expected - identities)}")
    return verified


def build_evaluation_spec(
    *, workspace: Path, input_directory: Path, batch_size: int = 256
) -> dict[str, Any]:
    splits = []
    for split in TEST_SPLITS:
        input_path = input_directory / f"{split}.v1.1.3.canonical.jsonl"
        manifest_path = input_directory / f"{split}.v1.1.3.canonical.manifest.json"
        manifest = read_json(manifest_path, f"{split} input manifest")
        if manifest.get("split") != split or manifest.get("labels_separated") is not True:
            raise IntegrityError(f"{split} is not a label-separated canonical input")
        digest = sha256_file(input_path)
        if digest != manifest.get("sha256"):
            raise IntegrityError(f"{split} input hash does not match its manifest")
        splits.append(
            {
                "split_name": split,
                "input_path": relative_artifact_path(input_path, workspace),
                "input_sha256": digest,
                "input_manifest_path": relative_artifact_path(manifest_path, workspace),
                "input_manifest_sha256": sha256_file(manifest_path),
                "row_count": int(manifest.get("row_count", -1)),
            }
        )
    return {
        "schema_version": EVALUATION_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "prediction_schema_version": PREDICTION_SCHEMA,
        "precision": "float32",
        "batch_size": batch_size,
        "deterministic_algorithms": True,
        "symmetric_condition_logic": True,
        "splits": splits,
    }


def build_probe_spec(
    *, workspace: Path, input_directory: Path, batch_size: int = 256
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for split in ("train", "validation"):
        path = input_directory / f"{split}.v1.1.3.canonical.jsonl"
        manifest_path = input_directory / f"{split}.v1.1.3.canonical.manifest.json"
        manifest = read_json(manifest_path, f"probe {split} manifest")
        if manifest.get("labels_separated") is not False:
            raise IntegrityError(f"probe {split} must contain authorized non-test targets")
        if sha256_file(path) != manifest.get("sha256"):
            raise IntegrityError(f"probe {split} hash mismatch")
        inputs[split] = {
            "path": relative_artifact_path(path, workspace),
            "sha256": sha256_file(path),
            "manifest_path": relative_artifact_path(manifest_path, workspace),
            "manifest_sha256": sha256_file(manifest_path),
            "row_count": int(manifest.get("row_count", -1)),
        }
    return {
        "schema_version": PROBE_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema_version": PROBE_SCHEMA,
        "scope": "each condition x each seed x evidence step {1,2}",
        "inputs": inputs,
        "selection": "two-step tasks only; query representation excluded",
        "network": {"widths": [192, 128, 2], "activation": "GELU"},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "batch_size": batch_size,
            "epochs": 100,
            "early_stopping": False,
        },
        "standardization": "training-feature mean/std; zero std replaced by one",
        "seed_namespace": "derive_uint64(opm-v1,neural-probe,condition,seed,step)",
        "base_model_parameters_frozen": True,
        "captured_activations_detached": True,
        "probe_optimizer_parameter_scope": "probe-only",
        "claim_threshold_application": False,
    }


def build_intervention_spec() -> dict[str, Any]:
    return {
        "schema_version": INTERVENTION_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_schema_version": INTERVENTION_SCHEMA,
        "families": [
            {
                "name": "ablation",
                "coverage": "each active primitive/component plus one unused sentinel, all inputs",
                "operation": "replace selected component delta with zero",
            },
            {
                "name": "replacement",
                "coverage": "each ordered distinct active component pair, all inputs",
                "operation": "replace source component with target component at fixed inputs",
            },
            {
                "name": "interchange",
                "coverage": "deterministic renderer-paired two-step examples",
                "operation": (
                    "swap normalized execution states before the last transition while retaining "
                    "destination evidence"
                ),
            },
            {
                "name": "adapter-only",
                "coverage": "all inputs",
                "operation": "zero every primitive/generalist delta",
            },
            {
                "name": "surface-reversal",
                "coverage": "test-renderer variant 2 rows",
                "operation": "apply frozen cyclic relation-token permutation to rendered surface only",
            },
        ],
        "surface_relation_permutation": {
            "source": "src/rlmgraph/opm/rendering.py::V2_ALIASES",
            "policy": "consume the frozen renderer-variant-2 rows; never regenerate test inputs",
        },
        "interchange_pairing": {
            "group_key": ["world_id", "query", "operation_ids", "step_mask"],
            "renderer_variant_pairs": [[0, 1], [1, 0]],
            "requires_two_steps": True,
        },
        "metrics_computed": False,
        "claim_threshold_application": False,
    }


def build_resource_manifest(
    *, workspace: Path, readable_roots: Sequence[Path], writable_root: Path
) -> dict[str, Any]:
    for path in [*readable_roots, writable_root]:
        reject_sealed_path(path.resolve())
    return {
        "schema_version": RESOURCE_MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "readable_roots": [relative_artifact_path(path, workspace) for path in readable_roots],
        "writable_root": relative_artifact_path(writable_root, workspace),
        "sealed_labels_mounted": False,
        "prohibited_resources": sorted(SEALED_PATH_MARKERS),
        "enforcement": "executor path allowlist; sealed evaluator is not imported or initialized",
    }


def build_transition_record(
    *,
    workspace: Path,
    primary_matrix_path: Path,
    checkpoint_manifest_path: Path,
    handoff_path: Path,
    evaluation_spec_path: Path,
    probe_spec_path: Path,
    intervention_spec_path: Path,
    executor_sources: Sequence[Path],
) -> dict[str, Any]:
    checkpoint_manifest = read_json(checkpoint_manifest_path, "selected checkpoint manifest")
    matrix_digest = sha256_file(primary_matrix_path)
    verify_selected_checkpoint_manifest(
        checkpoint_manifest,
        workspace=workspace,
        expected_primary_matrix_sha256=matrix_digest,
    )
    commit = git_commit(workspace)
    sources = source_manifest(executor_sources, workspace)
    return {
        "schema_version": TRANSITION_SCHEMA,
        "from_state": "PRIMARY_TRAINING_COMPLETE",
        "to_state": "POST_TRAINING_STAGE_1_AUTHORIZED",
        "primary_matrix_sha256": matrix_digest,
        "selected_checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "selected_checkpoint_count": 20,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_commit": commit,
        "handoff_path": relative_artifact_path(handoff_path, workspace),
        "handoff_sha256": sha256_file(handoff_path),
        "executor_source_commit": commit,
        "executor_sources": sources,
        "executor_source_aggregate_sha256": canonical_sha256(sources),
        "dirty_tree_policy": (
            "ALLOW_UNRELATED_DIRTY_PATHS; REQUIRE_EXACT_HASHES_FOR_ALL_STAGE1_SOURCES_AND_SPECS"
        ),
        "evaluation_spec_sha256": sha256_file(evaluation_spec_path),
        "probe_spec_sha256": sha256_file(probe_spec_path),
        "intervention_spec_sha256": sha256_file(intervention_spec_path),
        "authorized_operations": list(AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "sealed_label_access_authorized": False,
        "aggregate_test_evaluation_authorized": False,
        "claim_decisions_authorized": False,
    }


def _field_names(value: object) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            names.add(str(key).casefold().replace("-", "_"))
            names.update(_field_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_field_names(child))
    return names


@dataclass(frozen=True)
class InputInventory:
    split_name: str
    path: str
    sha256: str
    manifest_sha256: str
    row_count: int
    example_ids: tuple[str, ...]


def validate_label_blind_input(
    *,
    split_spec: Mapping[str, Any],
    workspace: Path,
    resources: ResourcePolicy,
) -> InputInventory:
    split_name = str(split_spec.get("split_name"))
    if split_name not in TEST_SPLITS:
        raise IntegrityError(f"unknown label-blind split: {split_name}")
    path = resolve_artifact_path(split_spec.get("input_path"), workspace, "input")
    manifest_path = resolve_artifact_path(
        split_spec.get("input_manifest_path"), workspace, "input manifest"
    )
    resources.require_read(path)
    resources.require_read(manifest_path)
    expected_hash = validate_sha256(split_spec.get("input_sha256"), "input hash")
    expected_manifest_hash = validate_sha256(
        split_spec.get("input_manifest_sha256"), "input manifest hash"
    )
    if sha256_file(path) != expected_hash or sha256_file(manifest_path) != expected_manifest_hash:
        raise IntegrityError(f"label-blind input or manifest hash mismatch: {split_name}")
    manifest = read_json(manifest_path, f"{split_name} manifest")
    if manifest.get("labels_separated") is not True or manifest.get("sha256") != expected_hash:
        raise IntegrityError(f"input is not the frozen label-separated artifact: {split_name}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError(f"invalid input JSON at {split_name}:{line_number}") from error
            if not isinstance(row, dict):
                raise IntegrityError(f"input row is not an object at {split_name}:{line_number}")
            forbidden = _field_names(row) & FORBIDDEN_TARGET_FIELDS
            if forbidden:
                raise AuthorizationError(
                    f"target/scoring field present in {split_name}:{line_number}: {sorted(forbidden)}"
                )
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise IntegrityError(f"missing example_id at {split_name}:{line_number}")
            ids.append(example_id)
    if len(ids) != int(split_spec.get("row_count", -1)) or len(ids) != int(
        manifest.get("row_count", -1)
    ):
        raise IntegrityError(f"input row-count mismatch: {split_name}")
    if len(set(ids)) != len(ids):
        raise IntegrityError(f"duplicate input example IDs: {split_name}")
    return InputInventory(
        split_name=split_name,
        path=relative_artifact_path(path, workspace),
        sha256=expected_hash,
        manifest_sha256=expected_manifest_hash,
        row_count=len(ids),
        example_ids=tuple(ids),
    )


@dataclass(frozen=True)
class Stage1Preflight:
    state: str
    job_id: str
    resume: bool
    transition_sha256: str
    checkpoint_manifest_sha256: str
    evaluation_spec_sha256: str
    probe_spec_sha256: str
    intervention_spec_sha256: str
    resource_manifest_sha256: str
    primary_matrix_sha256: str
    checkpoint_count: int
    input_row_count: int
    output_root: str
    enabled_capabilities: tuple[str, ...]
    sealed_labels_accessible: bool
    aggregate_test_evaluation_performed: bool
    claim_decisions_performed: bool


def preflight_stage1(
    *,
    workspace: Path,
    transition_path: Path,
    expected_transition_sha256: str,
    checkpoint_manifest_path: Path,
    evaluation_spec_path: Path,
    probe_spec_path: Path,
    intervention_spec_path: Path,
    resource_manifest_path: Path,
    output_root: Path,
) -> tuple[Stage1Preflight, CapabilityGate, list[dict[str, Any]], list[InputInventory]]:
    workspace = workspace.resolve()
    transition_path = transition_path.resolve()
    checkpoint_manifest_path = checkpoint_manifest_path.resolve()
    evaluation_spec_path = evaluation_spec_path.resolve()
    probe_spec_path = probe_spec_path.resolve()
    intervention_spec_path = intervention_spec_path.resolve()
    resource_manifest_path = resource_manifest_path.resolve()
    output_root = output_root.resolve()
    for supplied in (
        transition_path,
        checkpoint_manifest_path,
        evaluation_spec_path,
        probe_spec_path,
        intervention_spec_path,
        resource_manifest_path,
        output_root,
    ):
        reject_sealed_path(supplied.resolve())
    expected_transition_sha256 = validate_sha256(
        expected_transition_sha256, "expected transition hash"
    )
    if not transition_path.is_file() or sha256_file(transition_path) != expected_transition_sha256:
        raise AuthorizationError("missing or invalid post-primary transition hash")
    transition = read_json(transition_path, "post-primary transition")
    if (
        transition.get("from_state") != "PRIMARY_TRAINING_COMPLETE"
        or transition.get("to_state") != "POST_TRAINING_STAGE_1_AUTHORIZED"
    ):
        raise AuthorizationError("post-primary lifecycle transition is not authorized")
    if transition.get("protocol_version") != PROTOCOL_VERSION:
        raise AuthorizationError("post-primary transition protocol version mismatch")
    if transition.get("protocol_commit") != git_commit(workspace) or transition.get(
        "executor_source_commit"
    ) != git_commit(workspace):
        raise AuthorizationError("repository commit differs from the authorized transition")
    if transition.get("sealed_label_access_authorized") is not False:
        raise AuthorizationError("Stage 1 transition cannot authorize sealed labels")
    if transition.get("aggregate_test_evaluation_authorized") is not False:
        raise AuthorizationError("Stage 1 transition cannot authorize aggregate evaluation")
    if transition.get("claim_decisions_authorized") is not False:
        raise AuthorizationError("Stage 1 transition cannot authorize claim decisions")
    sources = transition.get("executor_sources")
    if not isinstance(sources, list) or canonical_sha256(sources) != transition.get(
        "executor_source_aggregate_sha256"
    ):
        raise IntegrityError("executor source manifest is invalid")
    for source in sources:
        if not isinstance(source, dict):
            raise IntegrityError("executor source entry is invalid")
        path = resolve_artifact_path(source.get("path"), workspace, "executor source")
        if sha256_file(path) != source.get("sha256"):
            raise IntegrityError(f"executor source changed after authorization: {path}")
    bound_specs = (
        (checkpoint_manifest_path, "selected_checkpoint_manifest_sha256"),
        (evaluation_spec_path, "evaluation_spec_sha256"),
        (probe_spec_path, "probe_spec_sha256"),
        (intervention_spec_path, "intervention_spec_sha256"),
    )
    for path, field in bound_specs:
        if not path.is_file() or sha256_file(path) != transition.get(field):
            raise IntegrityError(f"transition-bound artifact hash mismatch: {field}")
    checkpoint_manifest = read_json(checkpoint_manifest_path, "selected checkpoint manifest")
    primary_matrix_sha256 = validate_sha256(
        transition.get("primary_matrix_sha256"), "primary matrix hash"
    )
    checkpoints = verify_selected_checkpoint_manifest(
        checkpoint_manifest,
        workspace=workspace,
        expected_primary_matrix_sha256=primary_matrix_sha256,
    )
    if transition.get("selected_checkpoint_count") != len(checkpoints):
        raise IntegrityError("transition checkpoint count mismatch")
    resources = ResourcePolicy.load(resource_manifest_path, workspace)
    resources.require_write(output_root)
    for checkpoint in checkpoints:
        path = resolve_artifact_path(checkpoint["checkpoint_path"], workspace, "checkpoint")
        resources.require_read(path)
    evaluation_spec = read_json(evaluation_spec_path, "Stage 1 evaluation spec")
    probe_spec = read_json(probe_spec_path, "Stage 1 probe spec")
    intervention_spec = read_json(intervention_spec_path, "Stage 1 intervention spec")
    if evaluation_spec.get("schema_version") != EVALUATION_SPEC_SCHEMA:
        raise IntegrityError("evaluation spec schema mismatch")
    if probe_spec.get("schema_version") != PROBE_SPEC_SCHEMA:
        raise IntegrityError("probe spec schema mismatch")
    if intervention_spec.get("schema_version") != INTERVENTION_SPEC_SCHEMA:
        raise IntegrityError("intervention spec schema mismatch")
    inventories = [
        validate_label_blind_input(split_spec=item, workspace=workspace, resources=resources)
        for item in evaluation_spec.get("splits", [])
    ]
    if tuple(item.split_name for item in inventories) != TEST_SPLITS:
        raise IntegrityError("evaluation spec split order/coverage mismatch")
    for split in ("train", "validation"):
        probe_input = probe_spec.get("inputs", {}).get(split, {})
        path = resolve_artifact_path(probe_input.get("path"), workspace, f"probe {split}")
        manifest_path = resolve_artifact_path(
            probe_input.get("manifest_path"), workspace, f"probe {split} manifest"
        )
        resources.require_read(path)
        resources.require_read(manifest_path)
        if sha256_file(path) != probe_input.get("sha256") or sha256_file(
            manifest_path
        ) != probe_input.get("manifest_sha256"):
            raise IntegrityError(f"probe {split} input hash mismatch")
    gate = CapabilityGate.from_transition(transition, expected_transition_sha256)
    job_material = {
        "schema_version": JOB_SPEC_SCHEMA,
        "transition_sha256": expected_transition_sha256,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "evaluation_spec_sha256": sha256_file(evaluation_spec_path),
        "probe_spec_sha256": sha256_file(probe_spec_path),
        "intervention_spec_sha256": sha256_file(intervention_spec_path),
        "resource_manifest_sha256": resources.manifest_sha256,
        "executor_source_aggregate_sha256": transition["executor_source_aggregate_sha256"],
        "repository_commit": git_commit(workspace),
        "artifact_schemas": [PREDICTION_SCHEMA, INTERVENTION_SCHEMA, PROBE_SCHEMA],
    }
    job_id = canonical_sha256(job_material)[:24]
    job_spec = {**job_material, "job_id": job_id}
    job_spec_path = output_root / "job-spec.json"
    freeze_path = output_root / FREEZE_FILENAME
    resume = output_root.exists()
    if resume:
        if freeze_path.exists():
            raise AuthorizationError("frozen Stage 1 bundle is immutable and cannot be resumed")
        existing = read_json(job_spec_path, "resume job specification")
        if existing != job_spec:
            raise IntegrityError("resume job specification hash does not match exactly")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(job_spec_path, job_spec)
    preflight = Stage1Preflight(
        state="PASS",
        job_id=job_id,
        resume=resume,
        transition_sha256=expected_transition_sha256,
        checkpoint_manifest_sha256=sha256_file(checkpoint_manifest_path),
        evaluation_spec_sha256=sha256_file(evaluation_spec_path),
        probe_spec_sha256=sha256_file(probe_spec_path),
        intervention_spec_sha256=sha256_file(intervention_spec_path),
        resource_manifest_sha256=resources.manifest_sha256,
        primary_matrix_sha256=primary_matrix_sha256,
        checkpoint_count=len(checkpoints),
        input_row_count=sum(item.row_count for item in inventories),
        output_root=str(output_root),
        enabled_capabilities=gate.enabled,
        sealed_labels_accessible=False,
        aggregate_test_evaluation_performed=False,
        claim_decisions_performed=False,
    )
    preflight_path = output_root / "preflight.json"
    if preflight_path.exists():
        existing = read_json(preflight_path, "resume preflight")
        comparable = asdict(preflight)
        comparable["resume"] = existing.get("resume")
        if canonical_sha256(existing) != canonical_sha256(comparable):
            raise IntegrityError("resume preflight differs from the existing invocation")
    else:
        atomic_write_json(preflight_path, asdict(preflight))
    return preflight, gate, checkpoints, inventories


def tensor_state_sha256(model: OPMModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_bytes(list(detached.shape)))
        digest.update(detached.numpy().tobytes(order="C"))
    return digest.hexdigest()


def freeze_base_model(model: OPMModel) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise IntegrityError("base model parameter freeze failed")


def assert_base_model_unchanged(model: OPMModel, expected_sha256: str) -> None:
    if tensor_state_sha256(model) != expected_sha256:
        raise AuthorizationError("base model changed during Stage 1 execution")


def assert_probe_optimizer_scope(
    optimizer: torch.optim.Optimizer, probe: torch.nn.Module, base_model: OPMModel
) -> None:
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group.get("params", [])
    }
    probe_ids = {id(parameter) for parameter in probe.parameters()}
    base_ids = {id(parameter) for parameter in base_model.parameters()}
    if optimizer_ids != probe_ids or optimizer_ids & base_ids:
        raise AuthorizationError("probe optimizer is not restricted to probe parameters")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_prediction_row(row: Mapping[str, Any], *, intervention: bool = False) -> None:
    expected_schema = INTERVENTION_SCHEMA if intervention else PREDICTION_SCHEMA
    if row.get("artifact_schema_version") != expected_schema:
        raise ReconciliationError("prediction artifact schema mismatch")
    forbidden = _field_names(dict(row)) & FORBIDDEN_OUTPUT_FIELDS
    if forbidden:
        raise ReconciliationError(f"forbidden target/scoring fields in output: {sorted(forbidden)}")
    if (
        row.get("condition") not in DECLARED_CONDITIONS
        or row.get("training_seed") not in DECLARED_SEEDS
    ):
        raise ReconciliationError("prediction identity is not declared")
    prediction = row.get("prediction")
    logits = row.get("logits_or_scores")
    if prediction not in (0, 1) or not isinstance(logits, list) or len(logits) != 2:
        raise ReconciliationError("prediction/logit shape is invalid")
    if not all(_finite_number(value) for value in logits):
        raise ReconciliationError("prediction contains NaN or infinite output")
    if intervention:
        for field in (
            "intervention_family",
            "intervention_location",
            "intervention_specification_sha256",
        ):
            if not isinstance(row.get(field), str) or not row.get(field):
                raise ReconciliationError(f"intervention provenance is missing: {field}")


def validate_probe_row(row: Mapping[str, Any]) -> None:
    if row.get("artifact_schema_version") != PROBE_SCHEMA:
        raise ReconciliationError("probe artifact schema mismatch")
    forbidden = _field_names(dict(row)) & frozenset(
        {"pass", "passed", "claim", "h1", "mechanism_decision", "aggregate"}
    )
    if forbidden:
        raise ReconciliationError(f"forbidden decision fields in probe output: {sorted(forbidden)}")
    if (
        row.get("condition") not in DECLARED_CONDITIONS
        or row.get("training_seed") not in DECLARED_SEEDS
    ):
        raise ReconciliationError("probe identity is not declared")
    if row.get("evidence_step") not in (1, 2):
        raise ReconciliationError("probe evidence step is invalid")
    for field in (
        "validation_accuracy",
        "p_value",
        "wilson_low",
        "wilson_high",
    ):
        if not _finite_number(row.get(field)):
            raise ReconciliationError(f"probe contains a nonfinite result: {field}")


class AtomicJsonlArtifact:
    """Streaming JSONL writer with row validation and atomic immutable publication."""

    def __init__(
        self,
        path: Path,
        *,
        artifact_kind: str,
        expected_rows: int,
        expected_example_ids: Iterable[str] | None = None,
    ) -> None:
        if path.exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
            raise FileExistsError(f"refusing to replace an existing Stage 1 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        self.path = path
        self.temporary = Path(temporary_name)
        self.handle: BinaryIO = os.fdopen(descriptor, "wb")
        self.artifact_kind = artifact_kind
        self.expected_rows = expected_rows
        self.expected_example_ids = (
            set(expected_example_ids) if expected_example_ids is not None else None
        )
        self.seen_ids: set[str] = set()
        self.rows = 0
        self.digest = hashlib.sha256()
        self.closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("artifact writer is closed")
        if self.artifact_kind == "probe":
            validate_probe_row(row)
        else:
            validate_prediction_row(row, intervention=self.artifact_kind == "intervention")
        example_id = row.get("example_id")
        if self.artifact_kind == "probe":
            identity = str(row.get("probe_id", ""))
        else:
            if not isinstance(example_id, str) or not example_id:
                raise ReconciliationError("prediction example_id is missing")
            identity = str(row.get("intervention_record_id", example_id))
        if not identity:
            raise ReconciliationError("artifact identity is missing")
        if identity in self.seen_ids:
            raise ReconciliationError(f"duplicate prediction identity: {identity}")
        self.seen_ids.add(identity)
        self.rows += 1
        encoded = canonical_json_bytes(dict(row))
        self.handle.write(encoded)
        self.digest.update(encoded)

    def commit(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("artifact writer is closed")
        self.closed = True
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        if self.rows != self.expected_rows:
            self.temporary.unlink(missing_ok=True)
            raise ReconciliationError(
                f"artifact row count mismatch: expected {self.expected_rows}, got {self.rows}"
            )
        if self.expected_example_ids is not None and self.seen_ids != self.expected_example_ids:
            self.temporary.unlink(missing_ok=True)
            missing = len(self.expected_example_ids - self.seen_ids)
            unknown = len(self.seen_ids - self.expected_example_ids)
            raise ReconciliationError(
                f"artifact example coverage mismatch: missing={missing}, unknown={unknown}"
            )
        digest = self.digest.hexdigest()
        if sha256_file(self.temporary) != digest:
            self.temporary.unlink(missing_ok=True)
            raise IntegrityError("temporary JSONL artifact hash mismatch")
        if self.path.exists():
            self.temporary.unlink(missing_ok=True)
            raise FileExistsError(f"artifact appeared during atomic publication: {self.path}")
        os.replace(self.temporary, self.path)
        _fsync_directory(self.path.parent)
        manifest = {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA,
            "artifact_kind": self.artifact_kind,
            "artifact_path": self.path.name,
            "artifact_sha256": digest,
            "row_count": self.rows,
            "identity_set_sha256": canonical_sha256(sorted(self.seen_ids)),
            **dict(metadata),
        }
        manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        atomic_write_json(manifest_path, manifest)
        return manifest

    def abort(self) -> None:
        if not self.closed:
            self.closed = True
            self.handle.close()
        self.temporary.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        if exc_type is not None:
            self.abort()


def verify_jsonl_artifact(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    expected_ids: set[str] | None = None,
) -> None:
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ReconciliationError(f"invalid artifact manifest schema: {path}")
    if manifest.get("artifact_path") != path.name or sha256_file(path) != manifest.get(
        "artifact_sha256"
    ):
        raise ReconciliationError(f"artifact content hash mismatch: {path}")
    artifact_kind = manifest.get("artifact_kind")
    intervention = artifact_kind == "intervention"
    seen: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for rows, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReconciliationError(f"invalid JSONL at {path}:{rows}") from error
            if not isinstance(row, dict):
                raise ReconciliationError(f"nonobject row at {path}:{rows}")
            if artifact_kind == "probe":
                validate_probe_row(row)
                identity = str(row.get("probe_id", ""))
            else:
                validate_prediction_row(row, intervention=intervention)
                identity = str(row.get("intervention_record_id", row.get("example_id", "")))
            if not identity or identity in seen:
                raise ReconciliationError(f"missing/duplicate output identity at {path}:{rows}")
            seen.add(identity)
    if rows != manifest.get("row_count"):
        raise ReconciliationError(f"artifact manifest row count mismatch: {path}")
    if canonical_sha256(sorted(seen)) != manifest.get("identity_set_sha256"):
        raise ReconciliationError(f"artifact identity-set hash mismatch: {path}")
    if expected_ids is not None and seen != expected_ids:
        raise ReconciliationError(f"artifact exact coverage mismatch: {path}")


def merkle_root(records: Sequence[Mapping[str, Any]]) -> str:
    leaves = [bytes.fromhex(canonical_sha256(dict(record))) for record in records]
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return leaves[0].hex()


def environment_record(device: str) -> dict[str, Any]:
    return {
        "schema_version": "opm-stage1-environment-v1",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "sealed_labels_accessible": False,
        "aggregate_evaluator_imported": False,
    }


def reconcile_and_freeze(
    *,
    output_root: Path,
    transition_sha256: str,
    checkpoints: Sequence[Mapping[str, Any]],
    expected_artifacts: Sequence[Mapping[str, Any]],
    environment_path: Path,
    execution_log_path: Path,
    gate: CapabilityGate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate.require(Capability.ARTIFACT_RECONCILIATION)
    freeze_path = output_root / FREEZE_FILENAME
    if freeze_path.exists():
        raise AuthorizationError("frozen Stage 1 bundle cannot be modified")
    identities = {(str(entry["condition"]), int(entry["training_seed"])) for entry in checkpoints}
    expected_identities = {
        (condition, seed) for condition in DECLARED_CONDITIONS for seed in DECLARED_SEEDS
    }
    if identities != expected_identities:
        raise ReconciliationError("reconciliation checkpoint coverage is not exactly 4 x 5")
    artifact_records: list[dict[str, Any]] = []
    for expected in expected_artifacts:
        relative = expected.get("path")
        if not isinstance(relative, str):
            raise ReconciliationError("expected artifact path is missing")
        path = (output_root / relative).resolve()
        if output_root.resolve() not in path.parents:
            raise ReconciliationError(f"expected artifact escapes output root: {path}")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = read_json(manifest_path, "artifact sidecar")
        verify_jsonl_artifact(
            path,
            manifest,
            expected_ids=set(expected["expected_ids"]) if "expected_ids" in expected else None,
        )
        for field in (
            "artifact_kind",
            "row_count",
            "split_name",
            "condition",
            "training_seed",
            "checkpoint_sha256",
            "model_pre_sha256",
            "model_post_sha256",
            "identity_set_sha256",
            "intervention_specification_sha256",
            "evaluation_spec_sha256",
            "probe_spec_sha256",
        ):
            if field in expected and manifest.get(field) != expected[field]:
                raise ReconciliationError(f"artifact binding mismatch for {field}: {path}")
        if manifest.get("model_pre_sha256") != manifest.get("model_post_sha256"):
            raise ReconciliationError(f"base model changed during artifact generation: {path}")
        artifact_records.extend(
            [
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "row_count": manifest.get("row_count"),
                    "artifact_kind": manifest.get("artifact_kind"),
                },
                {
                    "path": manifest_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                    "artifact_kind": "artifact-manifest",
                },
            ]
        )
    for required in (
        environment_path,
        execution_log_path,
        output_root / "job-spec.json",
        output_root / "preflight.json",
    ):
        if not required.is_file():
            raise ReconciliationError(f"required execution artifact is missing: {required}")
        artifact_records.append(
            {
                "path": required.relative_to(output_root).as_posix(),
                "sha256": sha256_file(required),
                "bytes": required.stat().st_size,
                "artifact_kind": "execution-metadata",
            }
        )
    artifact_records.sort(key=lambda record: str(record["path"]))
    reconciliation = {
        "schema_version": RECONCILIATION_SCHEMA,
        "state": "PASS",
        "transition_sha256": transition_sha256,
        "checkpoint_count": len(checkpoints),
        "checkpoint_identities": sorted([list(identity) for identity in identities]),
        "expected_artifact_count": len(expected_artifacts),
        "published_file_count": len(artifact_records),
        "artifact_records": artifact_records,
        "artifact_merkle_root": merkle_root(artifact_records),
        "schemas_verified": True,
        "exact_coverage_verified": True,
        "finite_outputs_verified": True,
        "model_hashes_verified": True,
        "sealed_label_fields_found": 0,
        "aggregate_scoring_artifacts_found": 0,
        "failed_jobs_represented_as_complete": 0,
    }
    reconciliation_path = output_root / "reconciliation.json"
    atomic_write_json(reconciliation_path, reconciliation)
    gate.require(Capability.ARTIFACT_FREEZING)
    reconciliation_record = {
        "path": reconciliation_path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(reconciliation_path),
        "bytes": reconciliation_path.stat().st_size,
        "artifact_kind": "reconciliation",
    }
    frozen_records = sorted(
        [*artifact_records, reconciliation_record], key=lambda item: item["path"]
    )
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "state": "ARTIFACTS_RECONCILED_AND_FROZEN",
        "transition_record_sha256": transition_sha256,
        "selected_checkpoint_count": len(checkpoints),
        "selected_checkpoint_hashes": [
            {
                "condition": entry["condition"],
                "training_seed": entry["training_seed"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
            }
            for entry in checkpoints
        ],
        "artifact_count": len(frozen_records),
        "artifacts": frozen_records,
        "merkle_root": merkle_root(frozen_records),
        "sealed_labels_inaccessible": True,
        "sealed_label_access_count": 0,
        "aggregate_test_evaluation_performed": False,
        "bootstrap_effect_estimation_performed": False,
        "claim_decisions_performed": False,
        "next_state_requires_separate_authorization": "SEALED_LABEL_AGGREGATE_EVALUATION",
    }
    atomic_write_json(freeze_path, freeze)
    return reconciliation, freeze


def refuse_aggregate_evaluation(gate: CapabilityGate) -> None:
    gate.require(Capability.AGGREGATE_TEST_SCORING)


def refuse_claim_decision(gate: CapabilityGate) -> None:
    gate.require(Capability.CLAIM_THRESHOLD_APPLICATION)


def batch_to_device(batch: OPMBatch, device: torch.device | str) -> OPMBatch:
    return OPMBatch(
        **{
            field: getattr(batch, field).to(device) if getattr(batch, field) is not None else None
            for field in OPMBatch.__dataclass_fields__
        }
    )
