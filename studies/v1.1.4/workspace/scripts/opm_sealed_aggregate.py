from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Self

import numpy as np

PROTOCOL_VERSION = "v1.1.4"
AUTHORIZATION_SCHEMA = "opm-sealed-aggregate-authorization-v1"
AGGREGATE_SPEC_SCHEMA = "opm-sealed-aggregate-spec-v1"
PREFLIGHT_SCHEMA = "opm-sealed-aggregate-preflight-v1"
BASELINE_METRICS_SCHEMA = "opm-sealed-baseline-metrics-v1"
PRIMARY_EFFECTS_SCHEMA = "opm-sealed-primary-effects-v1"
INTERVENTION_METRICS_SCHEMA = "opm-sealed-intervention-metrics-v1"
PROBE_METRICS_SCHEMA = "opm-sealed-probe-metrics-v1"
EXECUTION_SUMMARY_SCHEMA = "opm-sealed-aggregate-execution-summary-v1"
FREEZE_SCHEMA = "opm-sealed-aggregate-freeze-v1"
FREEZE_FILENAME = "OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json"

STAGE1_FREEZE_SCHEMA = "opm-stage1-freeze-v1"
STAGE1_STATE = "ARTIFACTS_RECONCILED_AND_FROZEN"
AUTHORIZED_STATE = "SEALED_LABEL_AGGREGATE_EVALUATION_AUTHORIZED"
COMPLETE_STATE = "SEALED_LABEL_AGGREGATE_EVALUATION_COMPLETE_AND_FROZEN"

CONDITIONS = ("DOMAIN_GENERALIST", "OPM_SHARED", "PROC_CLONE", "PROC_UNTIED")
SEEDS = (1101, 2202, 3303, 4404, 5505)
SPLITS = (
    "test-recombination",
    "test-interpolation",
    "test-renderer",
    "test-structural",
)
OPERATION_NAMES = ("LOOKUP", "REVERSE", "CHAIN", "LIFT")

AUTHORIZED_OPERATIONS = (
    "frozen_stage1_artifact_read",
    "bound_label_blind_metadata_read",
    "sealed_target_read",
    "exact_key_join",
    "descriptive_aggregate_computation",
    "paired_two_level_bootstrap",
    "aggregate_package_freezing",
)
PROHIBITED_OPERATIONS = (
    "checkpoint_read",
    "model_load",
    "model_forward",
    "prediction_generation",
    "intervention_generation",
    "training",
    "checkpoint_selection",
    "claim_threshold_application",
    "claim_decision",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    "rlmgraph.opm.model",
    "rlmgraph.opm.primary_training",
)


class AggregateError(RuntimeError):
    pass


class AuthorizationError(AggregateError, PermissionError):
    pass


class IntegrityError(AggregateError, ValueError):
    pass


class EvaluationError(AggregateError, ValueError):
    pass


class Capability(StrEnum):
    FROZEN_ARTIFACT_READ = "frozen_stage1_artifact_read"
    METADATA_READ = "bound_label_blind_metadata_read"
    SEALED_TARGET_READ = "sealed_target_read"
    EXACT_JOIN = "exact_key_join"
    AGGREGATE_COMPUTE = "descriptive_aggregate_computation"
    BOOTSTRAP = "paired_two_level_bootstrap"
    PACKAGE_FREEZE = "aggregate_package_freezing"
    CHECKPOINT_READ = "checkpoint_read"
    MODEL_LOAD = "model_load"
    PREDICTION_GENERATION = "prediction_generation"
    CLAIM_THRESHOLD_APPLICATION = "claim_threshold_application"
    CLAIM_DECISION = "claim_decision"


DENIED_CAPABILITIES = frozenset(
    {
        Capability.CHECKPOINT_READ,
        Capability.MODEL_LOAD,
        Capability.PREDICTION_GENERATION,
        Capability.CLAIM_THRESHOLD_APPLICATION,
        Capability.CLAIM_DECISION,
    }
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise IntegrityError(f"missing {description}: {path}") from error
    except json.JSONDecodeError as error:
        raise IntegrityError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"{description} must be a JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists():
        if sha256_file(path) == digest:
            return digest
        raise FileExistsError(f"refusing to overwrite immutable aggregate artifact: {path}")
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
            raise IntegrityError(f"temporary artifact failed verification: {path}")
        if path.exists():
            raise FileExistsError(f"artifact appeared during publication: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def atomic_write_json(path: Path, value: object) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def relative_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise IntegrityError(f"path is outside workspace: {path}") from error


def resolve_workspace_path(value: object, workspace: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{description} path is missing")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as error:
        raise IntegrityError(f"{description} escapes workspace: {resolved}") from error
    return resolved


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
    return [
        {"path": relative_path(path, workspace), "sha256": sha256_file(path)}
        for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix())
    ]


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


class CapabilityGate:
    def __init__(self, enabled: Iterable[Capability] = ()) -> None:
        self._enabled = frozenset(enabled) - DENIED_CAPABILITIES

    @classmethod
    def from_authorization(cls, authorization: Mapping[str, Any]) -> Self:
        if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
            raise AuthorizationError("aggregate authorization schema is not approved")
        if authorization.get("authorized_operations") != list(AUTHORIZED_OPERATIONS):
            raise AuthorizationError("aggregate authorization operation set changed")
        if authorization.get("prohibited_operations") != list(PROHIBITED_OPERATIONS):
            raise AuthorizationError("aggregate prohibition set is incomplete")
        if authorization.get("sealed_label_access_authorized") is not True:
            raise AuthorizationError("sealed-target access is not authorized")
        if authorization.get("aggregate_test_evaluation_authorized") is not True:
            raise AuthorizationError("aggregate evaluation is not authorized")
        if authorization.get("claim_decisions_authorized") is not False:
            raise AuthorizationError("claim decisions must remain unauthorized")
        return cls(Capability(operation) for operation in AUTHORIZED_OPERATIONS)

    def require(self, capability: Capability) -> None:
        if capability in DENIED_CAPABILITIES or capability not in self._enabled:
            raise AuthorizationError(f"aggregate evaluator capability denied: {capability.value}")

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(capability.value for capability in self._enabled))


@dataclass(frozen=True)
class ResourcePolicy:
    workspace: Path
    frozen_root: Path
    metadata_paths: frozenset[Path]
    target_paths: frozenset[Path]
    output_root: Path
    gate: CapabilityGate

    def require_read(self, path: Path, capability: Capability) -> Path:
        self.gate.require(capability)
        resolved = path.resolve()
        if capability == Capability.FROZEN_ARTIFACT_READ:
            if resolved != self.frozen_root and self.frozen_root not in resolved.parents:
                raise AuthorizationError(f"not a frozen Stage 1 resource: {resolved}")
        elif capability == Capability.METADATA_READ:
            if resolved not in self.metadata_paths:
                raise AuthorizationError(f"not a bound metadata resource: {resolved}")
        elif capability == Capability.SEALED_TARGET_READ:
            if resolved not in self.target_paths:
                raise AuthorizationError(f"not an authorized sealed target: {resolved}")
        else:
            raise AuthorizationError(f"capability cannot authorize a file read: {capability.value}")
        return resolved

    def require_write(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.output_root and self.output_root not in resolved.parents:
            raise AuthorizationError(f"path is outside aggregate output root: {resolved}")
        return resolved


def assert_evaluator_source_is_aggregate_only(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise IntegrityError(f"invalid evaluator source: {path}") from error
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        if any(
            name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES
        ):
            raise AuthorizationError(
                f"aggregate evaluator imports prohibited model runtime: {name}"
            )


def build_locked_aggregate_spec(
    *,
    workspace: Path,
    stage1_freeze_path: Path,
    expected_stage1_freeze_sha256: str,
    stage1_evaluation_spec_path: Path,
) -> dict[str, Any]:
    freeze = read_json(stage1_freeze_path, "Stage 1 freeze root")
    if sha256_file(stage1_freeze_path) != validate_sha256(
        expected_stage1_freeze_sha256, "Stage 1 freeze hash"
    ):
        raise IntegrityError("Stage 1 freeze root hash mismatch")
    if freeze.get("schema_version") != STAGE1_FREEZE_SCHEMA or freeze.get("state") != STAGE1_STATE:
        raise IntegrityError("Stage 1 freeze is not at the required lifecycle state")
    if freeze.get("aggregate_test_evaluation_performed") is not False:
        raise IntegrityError("Stage 1 freeze already records aggregate evaluation")
    evaluation = read_json(stage1_evaluation_spec_path, "Stage 1 evaluation spec")
    split_specs = evaluation.get("splits")
    if not isinstance(split_specs, list) or len(split_specs) != 4:
        raise IntegrityError("Stage 1 evaluation spec does not bind four splits")
    targets: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for raw in split_specs:
        if not isinstance(raw, dict) or raw.get("split_name") not in SPLITS:
            raise IntegrityError("invalid Stage 1 split binding")
        manifest_path = resolve_workspace_path(
            raw.get("input_manifest_path"), workspace, "manifest"
        )
        manifest = read_json(manifest_path, "canonical split manifest")
        split = str(raw["split_name"])
        if manifest.get("split") != split or manifest.get("labels_separated") is not True:
            raise IntegrityError(f"canonical manifest is not label-separated: {split}")
        inputs.append(
            {
                "split_name": split,
                "path": str(raw["input_path"]),
                "sha256": validate_sha256(raw.get("input_sha256"), f"{split} input hash"),
                "manifest_path": str(raw["input_manifest_path"]),
                "manifest_sha256": validate_sha256(
                    raw.get("input_manifest_sha256"), f"{split} manifest hash"
                ),
                "row_count": int(raw["row_count"]),
            }
        )
        targets.append(
            {
                "split_name": split,
                "path": (
                    "evidence/implementation_validation/generated/v1.1.3-canonical-data/"
                    f"sealed-labels/{split}.v1.1.3.labels.jsonl"
                ),
                "sha256": validate_sha256(manifest.get("labels_sha256"), f"{split} target hash"),
                "row_count": int(manifest["row_count"]),
            }
        )
    inputs.sort(key=lambda item: item["split_name"])
    targets.sort(key=lambda item: item["split_name"])
    return {
        "schema_version": AGGREGATE_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "stage1_freeze": {
            "path": relative_path(stage1_freeze_path, workspace),
            "sha256": expected_stage1_freeze_sha256,
            "merkle_root": freeze.get("merkle_root"),
        },
        "label_blind_inputs": inputs,
        "sealed_targets": targets,
        "baseline_statistics": [
            "split_accuracy_by_condition_and_seed",
            "split_accuracy_pooled_across_seeds",
            "recombination_macro_accuracy_across_withheld_cells",
            "interpolation_accuracy",
            "one_step_and_two_step_accuracy",
            "renderer_accuracy",
            "structural_accuracy",
            "per_domain_accuracy",
            "per_operation_accuracy",
            "expected_calibration_error_10_fixed_width_bins",
        ],
        "primary_effect_statistics": {
            "effects": ["delta_generalist", "delta_untied"],
            "pairing": "matched model seed and exact external example key",
            "population": "all five declared model seeds",
            "bootstrap": {
                "method": "resample model seeds, then world IDs within each selected seed",
                "replicates": 10000,
                "rng": "numpy.PCG64DXSM",
                "seed": 99117,
                "interval": "percentile_95",
            },
        },
        "intervention_statistics": [
            "ablation_accuracy_change_by_component_domain_operation",
            "sentinel_max_absolute_logit_change",
            "replacement_operation_by_replacement_accuracy",
            "interchange_swapped_accuracy_and_unswapped_drop",
            "adapter_only_accuracy",
            "surface_reversal_accuracy_drop",
        ],
        "probe_statistics": "copy all 40 frozen validation-scale probe results and means",
        "calibration_definition": (
            "ECE=sum_b(n_b/N)*abs(accuracy_b-mean_max_softmax_confidence_b), "
            "using bins [0,.1),...,[.9,1.0]"
        ),
        "output_policy": {
            "aggregate_only": True,
            "copy_sealed_labels": False,
            "emit_per_example_correctness": False,
            "claim_threshold_application": False,
            "claim_decision": False,
            "next_state_requires_separate_authorization": "CLAIM_DECISION",
        },
    }


def create_authorization(
    *,
    workspace: Path,
    authorization_directory: Path,
    stage1_freeze_path: Path,
    expected_stage1_freeze_sha256: str,
    stage1_evaluation_spec_path: Path,
    evaluator_sources: Sequence[Path],
    user_request: str,
) -> tuple[Path, str, Path, str]:
    for source in evaluator_sources:
        assert_evaluator_source_is_aggregate_only(source)
    spec = build_locked_aggregate_spec(
        workspace=workspace,
        stage1_freeze_path=stage1_freeze_path,
        expected_stage1_freeze_sha256=expected_stage1_freeze_sha256,
        stage1_evaluation_spec_path=stage1_evaluation_spec_path,
    )
    spec_path = authorization_directory / "OPM_V1_1_4_SEALED_AGGREGATE_SPEC.json"
    spec_sha256 = atomic_write_json(spec_path, spec)
    sources = source_manifest(evaluator_sources, workspace)
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "from_state": STAGE1_STATE,
        "to_state": AUTHORIZED_STATE,
        "authorized_at_utc": utc_now(),
        "authorization_basis": {
            "actor": "workspace_owner_via_codex",
            "request": user_request,
            "scope": "start separately gated aggregate-only sealed evaluation",
        },
        "stage1_freeze_path": relative_path(stage1_freeze_path, workspace),
        "stage1_freeze_sha256": expected_stage1_freeze_sha256,
        "stage1_merkle_root": spec["stage1_freeze"]["merkle_root"],
        "aggregate_spec_path": relative_path(spec_path, workspace),
        "aggregate_spec_sha256": spec_sha256,
        "evaluator_source_commit": git_commit(workspace),
        "evaluator_sources": sources,
        "evaluator_source_aggregate_sha256": canonical_sha256(sources),
        "sealed_target_bindings": spec["sealed_targets"],
        "authorized_operations": list(AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "sealed_label_access_authorized": True,
        "aggregate_test_evaluation_authorized": True,
        "claim_decisions_authorized": False,
        "separate_claim_authorization_required": True,
    }
    authorization_path = authorization_directory / "OPM_V1_1_4_SEALED_AGGREGATE_AUTHORIZATION.json"
    authorization_sha256 = atomic_write_json(authorization_path, authorization)
    return authorization_path, authorization_sha256, spec_path, spec_sha256


@dataclass(frozen=True)
class PreflightResult:
    schema_version: str
    state: str
    authorization_sha256: str
    stage1_freeze_sha256: str
    stage1_merkle_root: str
    aggregate_spec_sha256: str
    frozen_file_count: int
    prediction_file_count: int
    intervention_file_count: int
    probe_file_count: int
    sealed_target_files_opened: int
    checkpoint_accesses: int
    model_loads: int
    prediction_generations: int
    claim_threshold_applications: int
    claim_decisions: int
    enabled_capabilities: tuple[str, ...]


def _validate_frozen_inventory(
    *, workspace: Path, freeze_path: Path, freeze: Mapping[str, Any]
) -> tuple[Path, dict[str, dict[str, Any]]]:
    stage_root = freeze_path.parent.resolve()
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 765:
        raise IntegrityError("Stage 1 freeze must bind exactly 765 files")
    if freeze.get("artifact_count") != 765:
        raise IntegrityError("Stage 1 freeze artifact count changed")
    records: dict[str, dict[str, Any]] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    for raw in artifacts:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise IntegrityError("invalid Stage 1 frozen artifact record")
        relative = str(raw["path"])
        if relative in records:
            raise IntegrityError(f"duplicate Stage 1 frozen path: {relative}")
        path = (stage_root / PurePosixPath(relative)).resolve()
        if stage_root not in path.parents:
            raise IntegrityError(f"Stage 1 frozen path escapes root: {relative}")
        if path.suffix.casefold() in {".pt", ".pth", ".ckpt", ".safetensors"}:
            raise AuthorizationError(f"checkpoint-like artifact is not readable: {relative}")
        expected = validate_sha256(raw.get("sha256"), f"frozen artifact {relative}")
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(f"Stage 1 frozen artifact hash mismatch: {relative}")
        records[relative] = dict(raw)
        counts[str(raw.get("artifact_kind"))] += 1
    if counts["prediction"] != 80 or counts["intervention"] != 280 or counts["probe"] != 20:
        raise IntegrityError(f"unexpected Stage 1 result inventory: {dict(counts)}")
    return stage_root, records


def preflight_aggregate_evaluation(
    *,
    workspace: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    output_root: Path,
) -> tuple[PreflightResult, CapabilityGate, ResourcePolicy, dict[str, Any], dict[str, Any]]:
    expected = validate_sha256(expected_authorization_sha256, "authorization hash")
    if not authorization_path.is_file() or sha256_file(authorization_path) != expected:
        raise AuthorizationError("missing or invalid aggregate authorization record")
    authorization = read_json(authorization_path, "aggregate authorization")
    gate = CapabilityGate.from_authorization(authorization)
    if (
        authorization.get("from_state") != STAGE1_STATE
        or authorization.get("to_state") != AUTHORIZED_STATE
    ):
        raise AuthorizationError("aggregate authorization lifecycle boundary changed")
    source_records = authorization.get("evaluator_sources")
    if not isinstance(source_records, list) or len(source_records) < 2:
        raise AuthorizationError("aggregate evaluator sources are not fully bound")
    verified_sources: list[dict[str, str]] = []
    for raw in source_records:
        if not isinstance(raw, dict):
            raise AuthorizationError("invalid evaluator source binding")
        source = resolve_workspace_path(raw.get("path"), workspace, "evaluator source")
        digest = validate_sha256(raw.get("sha256"), "evaluator source hash")
        if not source.is_file() or sha256_file(source) != digest:
            raise AuthorizationError(f"evaluator source hash mismatch: {source}")
        assert_evaluator_source_is_aggregate_only(source)
        verified_sources.append({"path": str(raw["path"]), "sha256": digest})
    if canonical_sha256(verified_sources) != authorization.get("evaluator_source_aggregate_sha256"):
        raise AuthorizationError("evaluator source aggregate binding changed")
    spec_path = resolve_workspace_path(authorization.get("aggregate_spec_path"), workspace, "spec")
    spec_sha256 = validate_sha256(authorization.get("aggregate_spec_sha256"), "spec hash")
    if not spec_path.is_file() or sha256_file(spec_path) != spec_sha256:
        raise AuthorizationError("locked aggregate specification hash mismatch")
    spec = read_json(spec_path, "locked aggregate specification")
    if spec.get("schema_version") != AGGREGATE_SPEC_SCHEMA:
        raise AuthorizationError("locked aggregate specification schema changed")
    policy = spec.get("output_policy")
    if not isinstance(policy, dict) or policy != {
        "aggregate_only": True,
        "claim_decision": False,
        "claim_threshold_application": False,
        "copy_sealed_labels": False,
        "emit_per_example_correctness": False,
        "next_state_requires_separate_authorization": "CLAIM_DECISION",
    }:
        raise AuthorizationError("aggregate-only output policy changed")
    stage1_freeze_path = resolve_workspace_path(
        authorization.get("stage1_freeze_path"), workspace, "Stage 1 freeze"
    )
    stage1_freeze_sha256 = validate_sha256(
        authorization.get("stage1_freeze_sha256"), "Stage 1 freeze hash"
    )
    if not stage1_freeze_path.is_file() or sha256_file(stage1_freeze_path) != stage1_freeze_sha256:
        raise IntegrityError("Stage 1 freeze root hash mismatch")
    freeze = read_json(stage1_freeze_path, "Stage 1 freeze root")
    if freeze.get("schema_version") != STAGE1_FREEZE_SCHEMA or freeze.get("state") != STAGE1_STATE:
        raise IntegrityError("Stage 1 lifecycle state is not frozen")
    if freeze.get("merkle_root") != authorization.get("stage1_merkle_root"):
        raise IntegrityError("Stage 1 Merkle root binding changed")
    if any(
        freeze.get(field) is not False
        for field in (
            "aggregate_test_evaluation_performed",
            "bootstrap_effect_estimation_performed",
            "claim_decisions_performed",
        )
    ):
        raise IntegrityError("Stage 1 freeze records a prohibited later-stage operation")
    if freeze.get("sealed_label_access_count") != 0:
        raise IntegrityError("Stage 1 freeze does not attest zero sealed-label access")
    stage_root, records = _validate_frozen_inventory(
        workspace=workspace, freeze_path=stage1_freeze_path, freeze=freeze
    )
    metadata_paths: set[Path] = set()
    target_paths: set[Path] = set()
    inputs = spec.get("label_blind_inputs")
    targets = spec.get("sealed_targets")
    if (
        not isinstance(inputs, list)
        or not isinstance(targets, list)
        or len(inputs) != 4
        or len(targets) != 4
    ):
        raise IntegrityError("aggregate spec does not bind four inputs and four targets")
    for raw in inputs:
        if not isinstance(raw, dict) or raw.get("split_name") not in SPLITS:
            raise IntegrityError("invalid metadata input binding")
        for path_field, hash_field in (("path", "sha256"), ("manifest_path", "manifest_sha256")):
            path = resolve_workspace_path(raw.get(path_field), workspace, "metadata input")
            expected_hash = validate_sha256(raw.get(hash_field), "metadata input hash")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise IntegrityError(f"metadata input hash mismatch: {path}")
            metadata_paths.add(path.resolve())
    auth_targets = authorization.get("sealed_target_bindings")
    if targets != auth_targets:
        raise AuthorizationError("sealed target bindings changed after authorization")
    for raw in targets:
        if not isinstance(raw, dict) or raw.get("split_name") not in SPLITS:
            raise IntegrityError("invalid sealed target binding")
        validate_sha256(raw.get("sha256"), "sealed target hash")
        target = resolve_workspace_path(raw.get("path"), workspace, "sealed target")
        if "sealed-labels" not in {part.casefold() for part in target.parts}:
            raise AuthorizationError("sealed target path lacks required sealed-label marker")
        # Deliberately do not stat, open, or hash target paths during preflight.
        target_paths.add(target.resolve())
    output = output_root.resolve()
    if output.exists():
        raise AuthorizationError(f"aggregate output root already exists: {output}")
    resource_policy = ResourcePolicy(
        workspace=workspace.resolve(),
        frozen_root=stage_root,
        metadata_paths=frozenset(metadata_paths),
        target_paths=frozenset(target_paths),
        output_root=output,
        gate=gate,
    )
    preflight = PreflightResult(
        schema_version=PREFLIGHT_SCHEMA,
        state="PASS",
        authorization_sha256=expected,
        stage1_freeze_sha256=stage1_freeze_sha256,
        stage1_merkle_root=str(freeze["merkle_root"]),
        aggregate_spec_sha256=spec_sha256,
        frozen_file_count=len(records),
        prediction_file_count=80,
        intervention_file_count=280,
        probe_file_count=20,
        sealed_target_files_opened=0,
        checkpoint_accesses=0,
        model_loads=0,
        prediction_generations=0,
        claim_threshold_applications=0,
        claim_decisions=0,
        enabled_capabilities=gate.enabled,
    )
    return preflight, gate, resource_policy, spec, freeze


def _iter_jsonl_bytes(payload: bytes, description: str) -> Iterable[dict[str, Any]]:
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise IntegrityError(f"invalid JSONL at {description}:{line_number}") from error
        if not isinstance(value, dict):
            raise IntegrityError(f"non-object JSONL row at {description}:{line_number}")
        yield value


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise IntegrityError(f"non-object JSONL row at {path}:{line_number}")
            yield value


@dataclass(frozen=True)
class ExampleMetadata:
    split: str
    world_id: int
    domain: str
    renderer_variant: int
    operation: str
    step_count: int


def load_metadata(
    *, workspace: Path, spec: Mapping[str, Any], resources: ResourcePolicy
) -> tuple[dict[str, dict[str, ExampleMetadata]], dict[str, tuple[str, ExampleMetadata]]]:
    by_split: dict[str, dict[str, ExampleMetadata]] = {}
    global_ids: dict[str, tuple[str, ExampleMetadata]] = {}
    for raw in spec["label_blind_inputs"]:
        split = str(raw["split_name"])
        path = resolve_workspace_path(raw["path"], workspace, "metadata input")
        resources.require_read(path, Capability.METADATA_READ)
        rows: dict[str, ExampleMetadata] = {}
        for row in iter_jsonl(path):
            example_id = row.get("example_id")
            operations = row.get("operation_ids")
            step_mask = row.get("step_mask")
            if (
                not isinstance(example_id, str)
                or not isinstance(operations, list)
                or not operations
                or not isinstance(step_mask, list)
            ):
                raise IntegrityError(f"invalid metadata row in {split}")
            operation_index = int(operations[0])
            if operation_index not in range(len(OPERATION_NAMES)):
                raise IntegrityError(f"invalid operation in {split}: {operation_index}")
            metadata = ExampleMetadata(
                split=split,
                world_id=int(row["world_id"]),
                domain=str(row["domain"]),
                renderer_variant=int(row["renderer_variant"]),
                operation=OPERATION_NAMES[operation_index],
                step_count=sum(value is True for value in step_mask),
            )
            if example_id in rows or example_id in global_ids:
                raise IntegrityError(f"duplicate canonical example key: {example_id}")
            rows[example_id] = metadata
            global_ids[example_id] = (split, metadata)
        if len(rows) != int(raw["row_count"]):
            raise IntegrityError(f"metadata row count mismatch: {split}")
        by_split[split] = rows
    if set(by_split) != set(SPLITS):
        raise IntegrityError("metadata split coverage mismatch")
    return by_split, global_ids


def _append_audit(path: Path, entry: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(entry)))
        handle.flush()
        os.fsync(handle.fileno())


def load_sealed_targets(
    *,
    workspace: Path,
    spec: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, ExampleMetadata]],
    resources: ResourcePolicy,
    audit_path: Path,
) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    for raw in spec["sealed_targets"]:
        split = str(raw["split_name"])
        target = resolve_workspace_path(raw["path"], workspace, "sealed target")
        resources.require_read(target, Capability.SEALED_TARGET_READ)
        try:
            payload = target.read_bytes()
        except OSError as error:
            _append_audit(
                audit_path,
                {
                    "timestamp_utc": utc_now(),
                    "decision": "DENIED",
                    "operation": "sealed_target_read",
                    "split_name": split,
                    "reason": str(error),
                },
            )
            raise IntegrityError(f"authorized sealed target is unreadable: {target}") from error
        digest = sha256_bytes(payload)
        expected = validate_sha256(raw["sha256"], f"{split} sealed target")
        _append_audit(
            audit_path,
            {
                "timestamp_utc": utc_now(),
                "decision": "ALLOWED",
                "operation": "sealed_target_read",
                "split_name": split,
                "path": relative_path(target, workspace),
                "expected_sha256": expected,
                "observed_sha256": digest,
                "bytes": len(payload),
            },
        )
        if digest != expected:
            raise IntegrityError(f"sealed target hash mismatch: {split}")
        by_id: dict[str, int] = {}
        for row in _iter_jsonl_bytes(payload, str(target)):
            if set(row) != {"example_id", "label"}:
                raise IntegrityError(f"sealed target schema mismatch: {split}")
            example_id = row["example_id"]
            label = row["label"]
            if (
                not isinstance(example_id, str)
                or example_id in by_id
                or not isinstance(label, int)
                or isinstance(label, bool)
                or label not in (0, 1)
            ):
                raise IntegrityError(f"invalid sealed target row: {split}")
            by_id[example_id] = label
        if set(by_id) != set(metadata[split]) or len(by_id) != int(raw["row_count"]):
            raise IntegrityError(f"sealed target keys do not exactly match inputs: {split}")
        labels[split] = by_id
    return labels


@dataclass
class AccuracyCounter:
    correct: int = 0
    count: int = 0
    baseline_correct: int = 0
    confidence_sum: float = 0.0

    def add(
        self,
        correct: int,
        *,
        baseline_correct: int | None = None,
        confidence: float | None = None,
    ) -> None:
        self.correct += int(correct)
        self.count += 1
        if baseline_correct is not None:
            self.baseline_correct += int(baseline_correct)
        if confidence is not None:
            self.confidence_sum += float(confidence)


def _accuracy_record(
    key_names: Sequence[str], key: Sequence[object], value: AccuracyCounter
) -> dict[str, Any]:
    if value.count <= 0:
        raise EvaluationError("cannot serialize empty accuracy counter")
    row = dict(zip(key_names, key, strict=True))
    row.update(
        {"correct": value.correct, "count": value.count, "accuracy": value.correct / value.count}
    )
    return row


def _intervention_record(
    key_names: Sequence[str], key: Sequence[object], value: AccuracyCounter
) -> dict[str, Any]:
    row = _accuracy_record(key_names, key, value)
    baseline_accuracy = value.baseline_correct / value.count
    row.update(
        {
            "baseline_correct": value.baseline_correct,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_change": row["accuracy"] - baseline_accuracy,
        }
    )
    return row


def _softmax_confidence(scores: object, prediction: int) -> float:
    if not isinstance(scores, list) or len(scores) != 2:
        raise EvaluationError("prediction logits must contain two values")
    left, right = float(scores[0]), float(scores[1])
    if not math.isfinite(left) or not math.isfinite(right):
        raise EvaluationError("prediction logits are not finite")
    maximum = max(left, right)
    denominator = math.exp(left - maximum) + math.exp(right - maximum)
    confidence = math.exp((right if prediction == 1 else left) - maximum) / denominator
    return confidence


def _sorted_records(
    counters: Mapping[tuple[object, ...], AccuracyCounter],
    key_names: Sequence[str],
    *,
    intervention: bool = False,
) -> list[dict[str, Any]]:
    function = _intervention_record if intervention else _accuracy_record
    return [
        function(key_names, key, counters[key])
        for key in sorted(counters, key=lambda item: tuple(map(str, item)))
    ]


def _frozen_data_records(freeze: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    records = [dict(item) for item in freeze["artifacts"] if item.get("artifact_kind") == kind]
    return sorted(records, key=lambda item: str(item["path"]))


def _world_macro_accuracy(values: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[1:] != (3, 2):
        raise EvaluationError("world scores require world x three-cell x {correct,count} arrays")
    totals = np.sum(values, axis=0)
    if not np.all(totals[:, 1] > 0):
        raise EvaluationError("world resample omitted an entire withheld cell")
    result = float(np.mean(totals[:, 0] / totals[:, 1]))
    if not math.isfinite(result):
        raise EvaluationError("world resample produced a nonfinite macro accuracy")
    return result


def paired_two_level_bootstrap(
    world_scores: Mapping[int, Mapping[str, np.ndarray]],
    *,
    replicates: int = 10_000,
    seed: int = 99117,
) -> dict[str, Any]:
    if set(world_scores) != set(SEEDS):
        raise EvaluationError("bootstrap requires all five declared model seeds")
    for model_seed in SEEDS:
        conditions = world_scores[model_seed]
        if set(conditions) != {"OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED"}:
            raise EvaluationError(f"bootstrap condition coverage mismatch: {model_seed}")
        shapes = {np.asarray(value).shape for value in conditions.values()}
        if len(shapes) != 1:
            raise EvaluationError(f"bootstrap world alignment mismatch: {model_seed}")
        shape = next(iter(shapes))
        if len(shape) != 3 or shape[1:] != (3, 2) or shape[0] == 0:
            raise EvaluationError(
                f"bootstrap requires world x three-cell x count arrays: {model_seed}"
            )
        masks = {
            condition: np.asarray(value)[:, :, 1] > 0 for condition, value in conditions.items()
        }
        if any(not np.array_equal(mask, masks["OPM_SHARED"]) for mask in masks.values()):
            raise EvaluationError(f"bootstrap cell-presence alignment mismatch: {model_seed}")
        if not np.all(np.any(masks["OPM_SHARED"], axis=0)):
            raise EvaluationError(f"bootstrap seed lacks a withheld cell: {model_seed}")

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    generalist_samples = np.empty(replicates, dtype=np.float64)
    untied_samples = np.empty(replicates, dtype=np.float64)
    seed_values = np.asarray(SEEDS, dtype=np.int64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seed_values, size=len(SEEDS), replace=True)
        shared_values: list[float] = []
        generalist_values: list[float] = []
        untied_values: list[float] = []
        for sampled_seed in sampled_seeds:
            data = world_scores[int(sampled_seed)]
            world_count = data["OPM_SHARED"].shape[0]
            indices = rng.integers(0, world_count, size=world_count)
            shared_values.append(_world_macro_accuracy(data["OPM_SHARED"][indices]))
            generalist_values.append(_world_macro_accuracy(data["DOMAIN_GENERALIST"][indices]))
            untied_values.append(_world_macro_accuracy(data["PROC_UNTIED"][indices]))
        generalist_samples[replicate] = np.mean(shared_values) - np.mean(generalist_values)
        untied_samples[replicate] = np.mean(shared_values) - np.mean(untied_values)
    points = {
        condition: float(
            np.mean(
                [_world_macro_accuracy(world_scores[model_seed][condition]) for model_seed in SEEDS]
            )
        )
        for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED")
    }
    return {
        "delta_generalist": points["OPM_SHARED"] - points["DOMAIN_GENERALIST"],
        "delta_untied": points["OPM_SHARED"] - points["PROC_UNTIED"],
        "delta_generalist_percentile_95_interval": np.quantile(
            generalist_samples, [0.025, 0.975]
        ).tolist(),
        "delta_untied_percentile_95_interval": np.quantile(untied_samples, [0.025, 0.975]).tolist(),
        "replicates": replicates,
        "rng": "numpy.PCG64DXSM",
        "seed": seed,
    }


def compute_aggregates(
    *,
    stage_root: Path,
    freeze: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, ExampleMetadata]],
    global_metadata: Mapping[str, tuple[str, ExampleMetadata]],
    labels: Mapping[str, Mapping[str, int]],
    gate: CapabilityGate,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate.require(Capability.EXACT_JOIN)
    gate.require(Capability.AGGREGATE_COMPUTE)
    prediction_records = _frozen_data_records(freeze, "prediction")
    intervention_records = _frozen_data_records(freeze, "intervention")
    probe_records = _frozen_data_records(freeze, "probe")
    prediction_index: dict[tuple[str, int, str], Path] = {}
    for record in prediction_records:
        parts = PurePosixPath(record["path"]).parts
        if len(parts) != 4 or parts[0] != "predictions" or not parts[2].startswith("seed-"):
            raise EvaluationError(f"invalid frozen prediction path: {record['path']}")
        key = (parts[1], int(parts[2].removeprefix("seed-")), Path(parts[3]).stem)
        if key in prediction_index:
            raise EvaluationError(f"duplicate frozen prediction artifact: {key}")
        prediction_index[key] = stage_root / PurePosixPath(record["path"])
    expected_prediction_keys = {
        (condition, model_seed, split)
        for condition in CONDITIONS
        for model_seed in SEEDS
        for split in SPLITS
    }
    if set(prediction_index) != expected_prediction_keys:
        raise EvaluationError("frozen prediction identity coverage is not exactly 4 x 5 x 4")

    baseline_seed: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(AccuracyCounter)
    baseline_pooled: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(AccuracyCounter)
    dimensions: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(AccuracyCounter)
    ece_bins: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(AccuracyCounter)
    recombination_world: dict[int, dict[str, defaultdict[int, list[list[int]]]]] = {
        seed: {
            condition: defaultdict(lambda: [[0, 0], [0, 0], [0, 0]])
            for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED")
        }
        for seed in SEEDS
    }
    recombination_cells = {("PROGRAM", "CHAIN"): 0, ("SCENE", "LIFT"): 1, ("SET", "REVERSE"): 2}
    baseline_cache: dict[tuple[str, int], dict[str, tuple[int, tuple[float, float]]]] = {}

    for condition in CONDITIONS:
        for model_seed in SEEDS:
            all_predictions: dict[str, tuple[int, tuple[float, float]]] = {}
            for split in SPLITS:
                path = prediction_index[(condition, model_seed, split)]
                seen: set[str] = set()
                for row in iter_jsonl(path):
                    if (
                        row.get("condition") != condition
                        or int(row.get("training_seed", -1)) != model_seed
                    ):
                        raise EvaluationError(f"prediction binding mismatch: {path}")
                    example_id = row.get("example_id")
                    prediction = row.get("prediction")
                    if (
                        not isinstance(example_id, str)
                        or example_id in seen
                        or example_id not in metadata[split]
                        or prediction not in (0, 1)
                        or isinstance(prediction, bool)
                    ):
                        raise EvaluationError(f"invalid prediction row: {path}")
                    scores_raw = row.get("logits_or_scores")
                    confidence = _softmax_confidence(scores_raw, int(prediction))
                    scores = (float(scores_raw[0]), float(scores_raw[1]))
                    label = labels[split][example_id]
                    correct = int(prediction == label)
                    meta = metadata[split][example_id]
                    baseline_seed[(condition, model_seed, split)].add(correct)
                    baseline_pooled[(condition, split)].add(correct)
                    for dimension, value in (
                        ("domain", meta.domain),
                        ("operation", meta.operation),
                        ("step_count", meta.step_count),
                        ("renderer_variant", meta.renderer_variant),
                    ):
                        dimensions[(condition, split, dimension, value)].add(correct)
                    bin_index = min(int(confidence * 10), 9)
                    ece_bins[(condition, split, bin_index)].add(correct, confidence=confidence)
                    if (
                        split == "test-recombination"
                        and condition in recombination_world[model_seed]
                    ):
                        cell = recombination_cells.get((meta.domain, meta.operation))
                        if cell is None:
                            raise EvaluationError("recombination example is outside withheld cells")
                        bucket = recombination_world[model_seed][condition][meta.world_id][cell]
                        bucket[0] += correct
                        bucket[1] += 1
                    seen.add(example_id)
                    if example_id in all_predictions:
                        raise EvaluationError(
                            f"example key appears in multiple prediction splits: {example_id}"
                        )
                    all_predictions[example_id] = (int(prediction), scores)
                if seen != set(metadata[split]):
                    raise EvaluationError(f"prediction keys do not exactly match {split}: {path}")
            baseline_cache[(condition, model_seed)] = all_predictions

    ece_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for split in SPLITS:
            total = sum(ece_bins[(condition, split, index)].count for index in range(10))
            ece = 0.0
            bins: list[dict[str, Any]] = []
            for index in range(10):
                counter = ece_bins[(condition, split, index)]
                accuracy = counter.correct / counter.count if counter.count else None
                mean_confidence = counter.confidence_sum / counter.count if counter.count else None
                if counter.count:
                    ece += counter.count / total * abs(float(accuracy) - float(mean_confidence))
                bins.append(
                    {
                        "bin_index": index,
                        "lower": index / 10,
                        "upper": (index + 1) / 10,
                        "count": counter.count,
                        "accuracy": accuracy,
                        "mean_confidence": mean_confidence,
                    }
                )
            ece_records.append(
                {
                    "condition": condition,
                    "split_name": split,
                    "count": total,
                    "ece": ece,
                    "bins": bins,
                }
            )

    per_seed_recombination: list[dict[str, Any]] = []
    world_arrays: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    for model_seed in SEEDS:
        expected_worlds: tuple[int, ...] | None = None
        for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED"):
            worlds = recombination_world[model_seed][condition]
            ordered_worlds = tuple(sorted(worlds))
            if expected_worlds is None:
                expected_worlds = ordered_worlds
            elif ordered_worlds != expected_worlds:
                raise EvaluationError(f"recombination world pairing mismatch: {model_seed}")
            values = np.zeros((len(ordered_worlds), 3, 2), dtype=np.float64)
            for world_index, world_id in enumerate(ordered_worlds):
                cells = worlds[world_id]
                if not any(count > 0 for _correct, count in cells):
                    raise EvaluationError(f"recombination world has no withheld cell: {world_id}")
                values[world_index] = cells
            world_arrays[model_seed][condition] = values
            per_seed_recombination.append(
                {
                    "condition": condition,
                    "training_seed": model_seed,
                    "world_count": len(ordered_worlds),
                    "macro_accuracy": _world_macro_accuracy(values),
                }
            )
    gate.require(Capability.BOOTSTRAP)
    bootstrap = paired_two_level_bootstrap(world_arrays, replicates=10_000, seed=99117)
    per_seed_effects: list[dict[str, Any]] = []
    for model_seed in SEEDS:
        values = {
            condition: _world_macro_accuracy(world_arrays[model_seed][condition])
            for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED")
        }
        per_seed_effects.append(
            {
                "training_seed": model_seed,
                "opm_shared_macro_accuracy": values["OPM_SHARED"],
                "domain_generalist_macro_accuracy": values["DOMAIN_GENERALIST"],
                "proc_untied_macro_accuracy": values["PROC_UNTIED"],
                "delta_generalist": values["OPM_SHARED"] - values["DOMAIN_GENERALIST"],
                "delta_untied": values["OPM_SHARED"] - values["PROC_UNTIED"],
            }
        )

    intervention_summary: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(
        AccuracyCounter
    )
    intervention_detail: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(
        AccuracyCounter
    )
    intervention_seed: defaultdict[tuple[object, ...], AccuracyCounter] = defaultdict(
        AccuracyCounter
    )
    sentinel_max: defaultdict[tuple[object, ...], float] = defaultdict(float)
    intervention_rows_read = 0
    for record in intervention_records:
        parts = PurePosixPath(record["path"]).parts
        if len(parts) != 5 or parts[0] != "interventions" or not parts[2].startswith("seed-"):
            raise EvaluationError(f"invalid frozen intervention path: {record['path']}")
        condition = parts[1]
        model_seed = int(parts[2].removeprefix("seed-"))
        family = parts[3]
        artifact_split = Path(parts[4]).stem
        path = stage_root / PurePosixPath(record["path"])
        row_count = 0
        for row in iter_jsonl(path):
            row_count += 1
            if (
                row.get("condition") != condition
                or int(row.get("training_seed", -1)) != model_seed
                or row.get("intervention_family") != family
            ):
                raise EvaluationError(f"intervention binding mismatch: {path}")
            example_id = row.get("target_example", row.get("example_id"))
            if not isinstance(example_id, str) or example_id not in global_metadata:
                raise EvaluationError(f"intervention target is not a canonical example: {path}")
            target_split, meta = global_metadata[example_id]
            if artifact_split in SPLITS and target_split != artifact_split:
                raise EvaluationError(f"intervention target split mismatch: {path}")
            prediction = row.get("prediction")
            if prediction not in (0, 1) or isinstance(prediction, bool):
                raise EvaluationError(f"invalid intervention prediction: {path}")
            baseline = baseline_cache[(condition, model_seed)].get(example_id)
            if baseline is None:
                raise EvaluationError(f"intervention lacks unswapped baseline: {example_id}")
            label = labels[target_split][example_id]
            correct = int(prediction == label)
            baseline_correct = int(baseline[0] == label)
            variant = str(row.get("intervention_variant"))
            component = str(row.get("component_or_primitive_identity"))
            summary_key = (condition, artifact_split, family, variant)
            detail_key = (
                condition,
                artifact_split,
                family,
                variant,
                component,
                meta.domain,
                meta.operation,
            )
            seed_key = (condition, model_seed, artifact_split, family, variant)
            intervention_summary[summary_key].add(correct, baseline_correct=baseline_correct)
            intervention_detail[detail_key].add(correct, baseline_correct=baseline_correct)
            intervention_seed[seed_key].add(correct, baseline_correct=baseline_correct)
            if family == "ablation" and variant == "sentinel":
                scores = row.get("logits_or_scores")
                if not isinstance(scores, list) or len(scores) != 2:
                    raise EvaluationError(f"sentinel logits are invalid: {path}")
                maximum = max(abs(float(scores[index]) - baseline[1][index]) for index in (0, 1))
                key = (condition, model_seed, artifact_split)
                sentinel_max[key] = max(sentinel_max[key], maximum)
        if row_count != int(record["row_count"]):
            raise EvaluationError(f"intervention row count mismatch: {path}")
        intervention_rows_read += row_count

    probe_rows: list[dict[str, Any]] = []
    probe_means: defaultdict[tuple[object, ...], list[float]] = defaultdict(list)
    for record in probe_records:
        path = stage_root / PurePosixPath(record["path"])
        row_count = 0
        for row in iter_jsonl(path):
            row_count += 1
            selected = {
                "probe_id": row.get("probe_id"),
                "condition": row.get("condition"),
                "training_seed": row.get("training_seed"),
                "evidence_step": row.get("evidence_step"),
                "validation_accuracy": row.get("validation_accuracy"),
                "validation_count": row.get("validation_count"),
                "p_value": row.get("p_value"),
                "wilson_low": row.get("wilson_low"),
                "wilson_high": row.get("wilson_high"),
            }
            if selected["condition"] not in CONDITIONS or selected["evidence_step"] not in (1, 2):
                raise EvaluationError(f"invalid probe result: {path}")
            value = float(selected["validation_accuracy"])
            if not math.isfinite(value):
                raise EvaluationError(f"nonfinite probe result: {path}")
            probe_rows.append(selected)
            probe_means[(selected["condition"], selected["evidence_step"])].append(value)
        if row_count != int(record["row_count"]):
            raise EvaluationError(f"probe row count mismatch: {path}")

    baseline_metrics = {
        "schema_version": BASELINE_METRICS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "thresholds_applied": False,
        "per_seed_split_accuracy": _sorted_records(
            baseline_seed, ("condition", "training_seed", "split_name")
        ),
        "pooled_split_accuracy": _sorted_records(baseline_pooled, ("condition", "split_name")),
        "dimension_accuracy": _sorted_records(
            dimensions, ("condition", "split_name", "dimension", "value")
        ),
        "calibration": ece_records,
        "prediction_rows_joined": sum(counter.count for counter in baseline_seed.values()),
    }
    primary_effects = {
        "schema_version": PRIMARY_EFFECTS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "thresholds_applied": False,
        "claim_decision_performed": False,
        "per_seed_recombination_macro_accuracy": per_seed_recombination,
        "per_seed_effects": per_seed_effects,
        "paired_two_level_bootstrap": bootstrap,
    }
    intervention_metrics = {
        "schema_version": INTERVENTION_METRICS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "thresholds_applied": False,
        "summary": _sorted_records(
            intervention_summary,
            ("condition", "artifact_split", "family", "variant"),
            intervention=True,
        ),
        "by_seed": _sorted_records(
            intervention_seed,
            ("condition", "training_seed", "artifact_split", "family", "variant"),
            intervention=True,
        ),
        "by_component_domain_operation": _sorted_records(
            intervention_detail,
            (
                "condition",
                "artifact_split",
                "family",
                "variant",
                "component",
                "target_domain",
                "target_operation",
            ),
            intervention=True,
        ),
        "sentinel_max_absolute_logit_change": [
            {
                "condition": key[0],
                "training_seed": key[1],
                "artifact_split": key[2],
                "max_absolute_logit_change": sentinel_max[key],
            }
            for key in sorted(sentinel_max, key=lambda item: tuple(map(str, item)))
        ],
        "intervention_rows_joined": intervention_rows_read,
    }
    probe_metrics = {
        "schema_version": PROBE_METRICS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "thresholds_applied": False,
        "results": sorted(
            probe_rows,
            key=lambda row: (
                str(row["condition"]),
                int(row["training_seed"]),
                int(row["evidence_step"]),
            ),
        ),
        "means": [
            {
                "condition": key[0],
                "evidence_step": key[1],
                "seed_count": len(probe_means[key]),
                "mean_validation_accuracy": float(np.mean(probe_means[key])),
            }
            for key in sorted(probe_means, key=lambda item: tuple(map(str, item)))
        ],
    }
    return baseline_metrics, primary_effects, intervention_metrics, probe_metrics


def freeze_aggregate_package(
    *,
    output_root: Path,
    authorization_sha256: str,
    spec_sha256: str,
    result_paths: Sequence[Path],
    sealed_target_bindings: Sequence[Mapping[str, Any]],
    gate: CapabilityGate,
) -> tuple[Path, str, dict[str, Any]]:
    gate.require(Capability.PACKAGE_FREEZE)
    records: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in result_paths), key=lambda item: item.as_posix()):
        if output_root.resolve() not in path.parents or not path.is_file():
            raise IntegrityError(f"aggregate result is missing or outside output root: {path}")
        records.append(
            {
                "path": path.relative_to(output_root.resolve()).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "state": COMPLETE_STATE,
        "authorization_sha256": authorization_sha256,
        "aggregate_spec_sha256": spec_sha256,
        "artifact_count": len(records),
        "artifacts": records,
        "merkle_root": merkle_root(records),
        "sealed_target_bindings": [dict(item) for item in sealed_target_bindings],
        "sealed_target_access_count": 4,
        "checkpoint_access_count": 0,
        "model_load_count": 0,
        "prediction_generation_count": 0,
        "claim_threshold_application_count": 0,
        "claim_decision_count": 0,
        "aggregate_only": True,
        "next_state_requires_separate_authorization": "CLAIM_DECISION",
    }
    path = output_root / FREEZE_FILENAME
    digest = atomic_write_json(path, freeze)
    return path, digest, freeze


def environment_record() -> dict[str, Any]:
    return {
        "schema_version": "opm-sealed-aggregate-environment-v1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "model_runtime_imported": False,
        "checkpoint_access_count": 0,
        "prediction_generation_count": 0,
        "claim_threshold_application_count": 0,
        "claim_decision_count": 0,
    }


class AtomicJsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to replace aggregate log: {path}")
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        self.temporary = Path(name)
        self.handle: BinaryIO = os.fdopen(descriptor, "wb")
        self.closed = False

    def write(self, row: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("aggregate log is closed")
        self.handle.write(canonical_json_bytes(dict(row)))

    def commit(self) -> str:
        if self.closed:
            raise RuntimeError("aggregate log is closed")
        self.closed = True
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.path.exists():
            self.temporary.unlink(missing_ok=True)
            raise FileExistsError(f"aggregate log appeared during publication: {self.path}")
        os.replace(self.temporary, self.path)
        _fsync_directory(self.path.parent)
        return sha256_file(self.path)

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
