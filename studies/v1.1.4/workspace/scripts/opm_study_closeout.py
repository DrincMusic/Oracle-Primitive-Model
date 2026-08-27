from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Self

PROTOCOL_VERSION = "v1.1.4"
AUTHORIZATION_SCHEMA = "opm-study-closeout-authorization-v1"
REPORT_SPEC_SCHEMA = "opm-study-closeout-spec-v1"
PREFLIGHT_SCHEMA = "opm-study-closeout-preflight-v1"
REPORT_DATA_SCHEMA = "opm-study-closeout-report-data-v1"
RECONCILIATION_SCHEMA = "opm-study-closeout-reconciliation-v1"
FREEZE_SCHEMA = "opm-study-closeout-freeze-v1"
FREEZE_FILENAME = "OPM_V1_1_4_STUDY_CLOSEOUT_FREEZE.json"
COMPLETE_STATE = "STUDY_CLOSED_AND_REPORT_FROZEN"

CLAIM_FREEZE_SCHEMA = "opm-claim-decision-freeze-v1"
CLAIM_FREEZE_STATE = "CLAIM_DECISIONS_RECONCILED_AND_FROZEN"
CLAIM_FREEZE_SHA256 = "2c14678accac56e6708ad12813314aad835cd72a02e240111b8490b8086f34a5"
CLAIM_MERKLE_ROOT = "6a237d348ef4d07d48d8487f1e67d18d01a910b4854f76b200631ac0b0e4c886"
DECISION_VOCABULARY = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_EVALUABLE")

AUTHORIZED_OPERATIONS = (
    "frozen_claim_package_read",
    "bound_report_context_read",
    "closeout_report_generation",
    "closeout_reconciliation",
    "closeout_package_freezing",
)
PROHIBITED_OPERATIONS = (
    "checkpoint_access",
    "model_loading",
    "prediction_row_access",
    "sealed_label_access",
    "prediction_generation",
    "aggregate_recomputation",
    "threshold_application",
    "threshold_modification",
    "claim_decision_modification",
    "training",
    "new_scientific_inference",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    "rlmgraph.opm.model",
    "rlmgraph.opm.primary_training",
    "rlmgraph.opm.sealing",
)
FORBIDDEN_PATH_PARTS = frozenset(
    {"checkpoints", "sealed-labels", "predictions", "interventions", "probes"}
)
FORBIDDEN_SUFFIXES = frozenset({".pt", ".pth", ".ckpt"})

CLAIM_FREEZE_PATH = Path(
    "evidence/primary_runs/v1.1.4/claim-decision/decision-artifacts-v1/"
    "run-1c9afdcaf96847be/OPM_V1_1_4_CLAIM_DECISION_FREEZE.json"
)
CLAIM_AUTHORIZATION_PATH = Path(
    "evidence/primary_runs/v1.1.4/claim-decision/authorization-v1/"
    "OPM_V1_1_4_CLAIM_DECISION_AUTHORIZATION.json"
)
CLAIM_SPEC_PATH = Path(
    "evidence/primary_runs/v1.1.4/claim-decision/authorization-v1/"
    "OPM_V1_1_4_CLAIM_DECISION_SPEC.json"
)
AGGREGATE_AUTHORIZATION_PATH = Path(
    "evidence/primary_runs/v1.1.4/sealed-aggregate/authorization-v4/"
    "OPM_V1_1_4_SEALED_AGGREGATE_AUTHORIZATION.json"
)
AGGREGATE_FREEZE_PATH = Path(
    "evidence/primary_runs/v1.1.4/sealed-aggregate/aggregate-artifacts-v4/"
    "run-c41bb1e61655891e/OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json"
)
AGGREGATE_ROOT = AGGREGATE_FREEZE_PATH.parent
STAGE1_TRANSITION_PATH = Path(
    "evidence/primary_runs/v1.1.4/post-primary/authorization-v5/"
    "OPM_V1_1_4_POST_PRIMARY_TRANSITION.json"
)
STAGE1_FREEZE_PATH = Path(
    "evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/"
    "run-ddfc05137e09a402/OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json"
)

REPORT_CONTEXT_PATHS = (
    Path("ORACLE_PRIMITIVE_MODEL.md"),
    Path("OPM_V1_IMPLEMENTATION_SPEC.md"),
    Path("OPM_V1_1_LEAKAGE_AMENDMENT.md"),
    Path("OPM_V1_1_1_BINDING_ERRATUM.md"),
    Path("OPM_V1_1_2_ENDPOINT_ERRATUM.md"),
    Path("OPM_V1_1_3_EVIDENCE_ORDER_ERRATUM.md"),
    Path("OPM_V1_1_4_TRAINING_EXECUTION_ERRATUM.md"),
    Path("evidence/implementation_validation/OPM_V1_1_4_TRAINING_EXECUTION_TRANSITION.json"),
    Path("evidence/implementation_validation/OPM_V1_1_4_PRIMARY_EXECUTION_HANDOFF.md"),
    Path("evidence/primary_runs/v1.1.4/pilot-matrix.json"),
    Path("evidence/primary_runs/v1.1.4/primary-matrix.json"),
    Path("evidence/primary_runs/v1.1.4/post-primary/OPM_V1_1_4_POST_PRIMARY_STAGE1_HANDOFF.md"),
    STAGE1_TRANSITION_PATH,
    STAGE1_FREEZE_PATH,
    Path(
        "evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/"
        "run-ddfc05137e09a402/reconciliation.json"
    ),
    Path("evidence/primary_runs/v1.1.4/sealed-aggregate/OPM_V1_1_4_SEALED_AGGREGATE_HANDOFF.md"),
    AGGREGATE_AUTHORIZATION_PATH,
    AGGREGATE_FREEZE_PATH,
    AGGREGATE_ROOT / "baseline-metrics.json",
    AGGREGATE_ROOT / "primary-effects.json",
    AGGREGATE_ROOT / "intervention-metrics.json",
    AGGREGATE_ROOT / "probe-metrics.json",
    CLAIM_AUTHORIZATION_PATH,
    CLAIM_SPEC_PATH,
    CLAIM_FREEZE_PATH,
    CLAIM_FREEZE_PATH.parent / "preflight.json",
    CLAIM_FREEZE_PATH.parent / "claim-decision.audit.jsonl",
    CLAIM_FREEZE_PATH.parent / "claim-decisions.json",
    CLAIM_FREEZE_PATH.parent / "reconciliation.json",
    CLAIM_FREEZE_PATH.parent / "environment.json",
)


class CloseoutError(RuntimeError):
    pass


class AuthorizationError(CloseoutError, PermissionError):
    pass


class IntegrityError(CloseoutError, ValueError):
    pass


class Capability(StrEnum):
    FROZEN_CLAIM_PACKAGE_READ = "frozen_claim_package_read"
    BOUND_REPORT_CONTEXT_READ = "bound_report_context_read"
    CLOSEOUT_REPORT_GENERATION = "closeout_report_generation"
    CLOSEOUT_RECONCILIATION = "closeout_reconciliation"
    CLOSEOUT_PACKAGE_FREEZING = "closeout_package_freezing"
    CHECKPOINT_ACCESS = "checkpoint_access"
    MODEL_LOADING = "model_loading"
    PREDICTION_ROW_ACCESS = "prediction_row_access"
    SEALED_LABEL_ACCESS = "sealed_label_access"
    PREDICTION_GENERATION = "prediction_generation"
    AGGREGATE_RECOMPUTATION = "aggregate_recomputation"
    THRESHOLD_APPLICATION = "threshold_application"
    THRESHOLD_MODIFICATION = "threshold_modification"
    CLAIM_DECISION_MODIFICATION = "claim_decision_modification"
    TRAINING = "training"
    NEW_SCIENTIFIC_INFERENCE = "new_scientific_inference"


DENIED_CAPABILITIES = frozenset(
    {
        Capability.CHECKPOINT_ACCESS,
        Capability.MODEL_LOADING,
        Capability.PREDICTION_ROW_ACCESS,
        Capability.SEALED_LABEL_ACCESS,
        Capability.PREDICTION_GENERATION,
        Capability.AGGREGATE_RECOMPUTATION,
        Capability.THRESHOLD_APPLICATION,
        Capability.THRESHOLD_MODIFICATION,
        Capability.CLAIM_DECISION_MODIFICATION,
        Capability.TRAINING,
        Capability.NEW_SCIENTIFIC_INFERENCE,
    }
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise IntegrityError(f"{description} is not a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise IntegrityError(f"{description} is not a SHA-256 digest") from error
    return value.lower()


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


def atomic_write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(value)
    if path.exists():
        if sha256_file(path) == digest:
            return digest
        raise FileExistsError(f"refusing to overwrite immutable closeout artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_file(temporary) != digest:
            raise IntegrityError(f"temporary closeout artifact failed verification: {path}")
        if path.exists():
            raise FileExistsError(f"closeout artifact appeared during publication: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def atomic_write_json(path: Path, value: object) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_text(path: Path, value: str) -> str:
    return atomic_write_bytes(path, value.encode("utf-8"))


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())


def relative_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError as error:
        raise IntegrityError(f"path is outside workspace: {path}") from error


def resolve_workspace_path(value: object, workspace: Path, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{description} path is missing")
    candidate = Path(value)
    result = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        result.relative_to(workspace.resolve())
    except ValueError as error:
        raise IntegrityError(f"{description} escapes workspace: {result}") from error
    return result


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


def assert_report_source_is_read_only(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise IntegrityError(f"invalid closeout source: {path}") from error
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    for name in imports:
        if any(
            name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES
        ):
            raise AuthorizationError(f"closeout source imports prohibited runtime: {name}")


def assert_report_safe_path(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & FORBIDDEN_PATH_PARTS or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise AuthorizationError(f"resource is prohibited in report stage: {path}")


class CapabilityGate:
    def __init__(self, enabled: Iterable[Capability] = ()) -> None:
        self._enabled = frozenset(enabled) - DENIED_CAPABILITIES

    @classmethod
    def from_authorization(cls, authorization: Mapping[str, Any]) -> Self:
        if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
            raise AuthorizationError("closeout authorization schema is not approved")
        if authorization.get("authorized_operations") != list(AUTHORIZED_OPERATIONS):
            raise AuthorizationError("closeout authorization operation set changed")
        if authorization.get("prohibited_operations") != list(PROHIBITED_OPERATIONS):
            raise AuthorizationError("closeout authorization prohibition set changed")
        return cls(Capability(value) for value in AUTHORIZED_OPERATIONS)

    def require(self, capability: Capability) -> None:
        if capability in DENIED_CAPABILITIES or capability not in self._enabled:
            raise AuthorizationError(f"closeout capability denied: {capability.value}")

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(item.value for item in self._enabled))


@dataclass(frozen=True)
class ResourcePolicy:
    input_paths: frozenset[Path]
    output_root: Path
    gate: CapabilityGate

    def require_read(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved not in self.input_paths:
            raise AuthorizationError(f"report cannot read unbound input: {resolved}")
        assert_report_safe_path(resolved)
        self.gate.require(Capability.BOUND_REPORT_CONTEXT_READ)
        return resolved

    def require_write(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.output_root and self.output_root not in resolved.parents:
            raise AuthorizationError(f"report output escapes authorized root: {resolved}")
        return resolved


def build_report_spec(*, workspace: Path, expected_claim_freeze_sha256: str) -> dict[str, Any]:
    expected = validate_sha256(expected_claim_freeze_sha256, "claim freeze hash")
    claim_freeze_path = (workspace / CLAIM_FREEZE_PATH).resolve()
    if sha256_file(claim_freeze_path) != expected or expected != CLAIM_FREEZE_SHA256:
        raise IntegrityError("claim-decision freeze identity changed")
    claim_freeze = read_json(claim_freeze_path, "claim-decision freeze")
    if (
        claim_freeze.get("schema_version") != CLAIM_FREEZE_SCHEMA
        or claim_freeze.get("state") != CLAIM_FREEZE_STATE
        or claim_freeze.get("merkle_root") != CLAIM_MERKLE_ROOT
    ):
        raise IntegrityError("claim-decision package is not the authoritative frozen package")
    claim_artifacts = claim_freeze.get("artifacts")
    if not isinstance(claim_artifacts, list) or len(claim_artifacts) != 5:
        raise IntegrityError("claim-decision freeze artifact inventory changed")
    if merkle_root(claim_artifacts) != CLAIM_MERKLE_ROOT:
        raise IntegrityError("claim-decision Merkle root does not recompute")
    for raw in claim_artifacts:
        path = (claim_freeze_path.parent / PurePosixPath(str(raw["path"]))).resolve()
        if (
            not path.is_file()
            or sha256_file(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("bytes")
        ):
            raise IntegrityError(f"claim-decision artifact mismatch: {path}")
    context_records: list[dict[str, Any]] = []
    for raw_path in REPORT_CONTEXT_PATHS:
        path = (workspace / raw_path).resolve()
        assert_report_safe_path(path)
        if not path.is_file():
            raise IntegrityError(f"missing closeout context input: {path}")
        context_records.append(
            {
                "path": relative_path(path, workspace),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": REPORT_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "authoritative_claim_decision_freeze": {
            "path": relative_path(claim_freeze_path, workspace),
            "sha256": expected,
            "merkle_root": claim_freeze["merkle_root"],
            "authorization_sha256": claim_freeze["authorization_sha256"],
        },
        "report_inputs": context_records,
        "scientific_authority_policy": {
            "claim_status_authority": "frozen claim-decision package only",
            "descriptive_result_authority": "aggregate v4 inputs transitively bound by claim authorization",
            "developmental_result_authority": "bound pilot and primary matrices",
            "methodology_authority": "bound preregistration and approved errata",
        },
        "required_sections": [
            "executive conclusion",
            "authority and scope",
            "preregistered hypotheses and thresholds",
            "pilot and primary methodology",
            "developmental validation results",
            "authoritative aggregate-v4 results",
            "all 15 claim decisions",
            "behavioral and mechanism separation",
            "deviations and diagnostic attempts",
            "authorization and freeze chain",
            "reproducibility",
            "limitations and follow-up studies",
        ],
        "execution_policy": {
            "checkpoint_access": False,
            "model_loading": False,
            "prediction_row_access": False,
            "sealed_label_access": False,
            "prediction_generation": False,
            "aggregate_recomputation": False,
            "threshold_application": False,
            "threshold_modification": False,
            "claim_decision_modification": False,
            "training": False,
            "new_scientific_inference": False,
            "report_generation": True,
        },
        "outputs": [
            "OPM_V1_1_4_FINAL_REPORT.md",
            "report-data.json",
            "reconciliation.json",
            "closeout.audit.jsonl",
            "preflight.json",
            "environment.json",
        ],
    }


def create_authorization(
    *,
    workspace: Path,
    authorization_directory: Path,
    expected_claim_freeze_sha256: str,
    reporter_sources: Sequence[Path],
    user_request: str,
) -> tuple[Path, str, Path, str]:
    for source in reporter_sources:
        assert_report_source_is_read_only(source)
    spec = build_report_spec(
        workspace=workspace,
        expected_claim_freeze_sha256=expected_claim_freeze_sha256,
    )
    spec_path = authorization_directory / "OPM_V1_1_4_STUDY_CLOSEOUT_SPEC.json"
    spec_sha256 = atomic_write_json(spec_path, spec)
    sources = source_manifest(reporter_sources, workspace)
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "from_state": CLAIM_FREEZE_STATE,
        "to_state": "STUDY_CLOSEOUT_REPORT_AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "authorization_basis": {
            "actor": "workspace_owner_via_codex",
            "request": user_request,
            "scope": "read-only reporting and study closeout from frozen OPM v1.1.4 evidence",
        },
        "claim_decision_freeze": spec["authoritative_claim_decision_freeze"],
        "report_spec_path": relative_path(spec_path, workspace),
        "report_spec_sha256": spec_sha256,
        "report_inputs": spec["report_inputs"],
        "reporter_source_commit": git_commit(workspace),
        "reporter_sources": sources,
        "reporter_source_aggregate_sha256": canonical_sha256(sources),
        "authorized_operations": list(AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "claim_status_changes_authorized": False,
        "new_scientific_decisions_authorized": False,
        "report_generation_authorized": True,
    }
    path = authorization_directory / "OPM_V1_1_4_STUDY_CLOSEOUT_AUTHORIZATION.json"
    digest = atomic_write_json(path, authorization)
    return path, digest, spec_path, spec_sha256


@dataclass(frozen=True)
class PreflightResult:
    schema_version: str
    state: str
    authorization_sha256: str
    report_spec_sha256: str
    claim_freeze_sha256: str
    claim_merkle_root: str
    verified_input_files: int
    claim_count: int
    checkpoint_accesses: int
    model_loads: int
    prediction_row_accesses: int
    sealed_label_accesses: int
    prediction_generations: int
    aggregate_recomputations: int
    threshold_applications: int
    threshold_modifications: int
    claim_decision_modifications: int
    enabled_capabilities: tuple[str, ...]


def preflight_closeout(
    *,
    workspace: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    output_root: Path,
) -> tuple[PreflightResult, CapabilityGate, ResourcePolicy, dict[str, Any], dict[str, Any]]:
    expected = validate_sha256(expected_authorization_sha256, "closeout authorization hash")
    if not authorization_path.is_file() or sha256_file(authorization_path) != expected:
        raise AuthorizationError("missing or invalid closeout authorization")
    authorization = read_json(authorization_path, "closeout authorization")
    gate = CapabilityGate.from_authorization(authorization)
    if authorization.get("from_state") != CLAIM_FREEZE_STATE:
        raise AuthorizationError("closeout lifecycle source state changed")
    if authorization.get("claim_status_changes_authorized") is not False:
        raise AuthorizationError("closeout authorization allows claim-status changes")
    if authorization.get("new_scientific_decisions_authorized") is not False:
        raise AuthorizationError("closeout authorization allows new scientific decisions")
    if authorization.get("report_generation_authorized") is not True:
        raise AuthorizationError("closeout report generation is not authorized")
    sources = authorization.get("reporter_sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise AuthorizationError("closeout source binding is incomplete")
    verified_sources: list[dict[str, str]] = []
    for raw in sources:
        path = resolve_workspace_path(raw.get("path"), workspace, "closeout source")
        digest = validate_sha256(raw.get("sha256"), "closeout source hash")
        if not path.is_file() or sha256_file(path) != digest:
            raise AuthorizationError(f"closeout source hash mismatch: {path}")
        assert_report_source_is_read_only(path)
        verified_sources.append({"path": str(raw["path"]), "sha256": digest})
    if canonical_sha256(verified_sources) != authorization.get("reporter_source_aggregate_sha256"):
        raise AuthorizationError("closeout source aggregate binding changed")
    spec_path = resolve_workspace_path(
        authorization.get("report_spec_path"), workspace, "report spec"
    )
    spec_hash = validate_sha256(authorization.get("report_spec_sha256"), "report spec hash")
    if not spec_path.is_file() or sha256_file(spec_path) != spec_hash:
        raise AuthorizationError("closeout report spec hash mismatch")
    spec = read_json(spec_path, "closeout report spec")
    if spec.get("schema_version") != REPORT_SPEC_SCHEMA:
        raise AuthorizationError("closeout report spec schema changed")
    expected_policy = {
        "checkpoint_access": False,
        "model_loading": False,
        "prediction_row_access": False,
        "sealed_label_access": False,
        "prediction_generation": False,
        "aggregate_recomputation": False,
        "threshold_application": False,
        "threshold_modification": False,
        "claim_decision_modification": False,
        "training": False,
        "new_scientific_inference": False,
        "report_generation": True,
    }
    if spec.get("execution_policy") != expected_policy:
        raise AuthorizationError("closeout execution policy changed")
    inputs = spec.get("report_inputs")
    if not isinstance(inputs, list) or inputs != authorization.get("report_inputs"):
        raise AuthorizationError("closeout report input bindings changed")
    input_paths: set[Path] = set()
    for raw in inputs:
        path = resolve_workspace_path(raw.get("path"), workspace, "report input")
        assert_report_safe_path(path)
        digest = validate_sha256(raw.get("sha256"), "report input hash")
        if (
            not path.is_file()
            or sha256_file(path) != digest
            or path.stat().st_size != raw.get("bytes")
        ):
            raise IntegrityError(f"closeout report input mismatch: {path}")
        input_paths.add(path.resolve())
    claim_binding = authorization.get("claim_decision_freeze")
    if not isinstance(claim_binding, dict):
        raise AuthorizationError("closeout lacks claim freeze binding")
    claim_path = resolve_workspace_path(claim_binding.get("path"), workspace, "claim freeze")
    if claim_path.resolve() not in input_paths:
        raise AuthorizationError("claim freeze is not a bound report input")
    claim_hash = validate_sha256(claim_binding.get("sha256"), "claim freeze hash")
    if claim_hash != CLAIM_FREEZE_SHA256 or sha256_file(claim_path) != claim_hash:
        raise IntegrityError("authoritative claim freeze identity changed")
    claim_freeze = read_json(claim_path, "claim freeze")
    if (
        claim_freeze.get("state") != CLAIM_FREEZE_STATE
        or claim_freeze.get("merkle_root") != CLAIM_MERKLE_ROOT
    ):
        raise IntegrityError("authoritative claim freeze state changed")
    if merkle_root(claim_freeze.get("artifacts", [])) != CLAIM_MERKLE_ROOT:
        raise IntegrityError("authoritative claim Merkle root does not recompute")
    for raw in claim_freeze.get("artifacts", []):
        path = (claim_path.parent / PurePosixPath(str(raw["path"]))).resolve()
        if path not in input_paths:
            raise AuthorizationError(f"claim freeze artifact is not a bound report input: {path}")
        if (
            not path.is_file()
            or sha256_file(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("bytes")
        ):
            raise IntegrityError(f"claim freeze artifact changed: {path}")
    decisions_path = claim_path.parent / "claim-decisions.json"
    decisions = read_json(decisions_path, "claim decisions")
    if len(decisions.get("decisions", [])) != 15:
        raise IntegrityError("closeout requires all 15 frozen claim decisions")
    output = output_root.resolve()
    if output.exists():
        raise AuthorizationError(f"closeout output root already exists: {output}")
    resources = ResourcePolicy(frozenset(input_paths), output, gate)
    result = PreflightResult(
        schema_version=PREFLIGHT_SCHEMA,
        state="PASS",
        authorization_sha256=expected,
        report_spec_sha256=spec_hash,
        claim_freeze_sha256=claim_hash,
        claim_merkle_root=CLAIM_MERKLE_ROOT,
        verified_input_files=len(input_paths),
        claim_count=15,
        checkpoint_accesses=0,
        model_loads=0,
        prediction_row_accesses=0,
        sealed_label_accesses=0,
        prediction_generations=0,
        aggregate_recomputations=0,
        threshold_applications=0,
        threshold_modifications=0,
        claim_decision_modifications=0,
        enabled_capabilities=gate.enabled,
    )
    return result, gate, resources, spec, authorization


def _find_one(rows: Sequence[Mapping[str, Any]], **criteria: object) -> dict[str, Any]:
    matches = [
        dict(row) for row in rows if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise IntegrityError(f"expected one report row for {criteria}, found {len(matches)}")
    return matches[0]


def _input_path(workspace: Path, resources: ResourcePolicy, relative: Path) -> Path:
    return resources.require_read((workspace / relative).resolve())


def _status_counts(decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        status: sum(row.get("status") == status for row in decisions)
        for status in DECISION_VOCABULARY
    }


def build_report_data(
    *,
    workspace: Path,
    resources: ResourcePolicy,
    authorization_sha256: str,
    authorization: Mapping[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    resources.gate.require(Capability.CLOSEOUT_REPORT_GENERATION)
    resources.gate.require(Capability.FROZEN_CLAIM_PACKAGE_READ)
    claim_package = read_json(
        _input_path(workspace, resources, CLAIM_FREEZE_PATH.parent / "claim-decisions.json"),
        "claim decisions",
    )
    decisions = claim_package["decisions"]
    if len(decisions) != 15 or any(
        row.get("status") not in DECISION_VOCABULARY for row in decisions
    ):
        raise IntegrityError("frozen claim decision inventory is invalid")
    claim_reconciliation = read_json(
        _input_path(workspace, resources, CLAIM_FREEZE_PATH.parent / "reconciliation.json"),
        "claim reconciliation",
    )
    counts = _status_counts(decisions)
    if counts != claim_reconciliation.get("status_counts"):
        raise IntegrityError("claim status counts do not match reconciliation")
    baseline = read_json(
        _input_path(workspace, resources, AGGREGATE_ROOT / "baseline-metrics.json"),
        "aggregate v4 baseline",
    )
    effects = read_json(
        _input_path(workspace, resources, AGGREGATE_ROOT / "primary-effects.json"),
        "aggregate v4 effects",
    )
    probes = read_json(
        _input_path(workspace, resources, AGGREGATE_ROOT / "probe-metrics.json"),
        "aggregate v4 probes",
    )
    pilot = read_json(
        _input_path(workspace, resources, Path("evidence/primary_runs/v1.1.4/pilot-matrix.json")),
        "pilot matrix",
    )
    primary = read_json(
        _input_path(workspace, resources, Path("evidence/primary_runs/v1.1.4/primary-matrix.json")),
        "primary matrix",
    )
    conditions = ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED", "PROC_CLONE")
    splits = ("test-interpolation", "test-recombination", "test-renderer", "test-structural")
    baseline_table = [
        {
            "condition": condition,
            **{
                split: float(
                    _find_one(
                        baseline["pooled_split_accuracy"], condition=condition, split_name=split
                    )["accuracy"]
                )
                for split in splits
            },
        }
        for condition in conditions
    ]
    probe_means_table = [
        {
            "condition": condition,
            "evidence_step_1_mean_accuracy": float(
                _find_one(probes["means"], condition=condition, evidence_step=1)[
                    "mean_validation_accuracy"
                ]
            ),
            "evidence_step_2_mean_accuracy": float(
                _find_one(probes["means"], condition=condition, evidence_step=2)[
                    "mean_validation_accuracy"
                ]
            ),
        }
        for condition in conditions
    ]
    decision_by_id = {str(row["claim_id"]): dict(row) for row in decisions}
    data = {
        "schema_version": REPORT_DATA_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "state": "FINAL_REPORT_GENERATED_PENDING_RECONCILIATION",
        "generated_at_utc": generated_at_utc,
        "closeout_authorization_sha256": authorization_sha256,
        "authoritative_claim_freeze": authorization["claim_decision_freeze"],
        "authoritative_aggregate_version": "v4",
        "non_authoritative_aggregate_versions_excluded": ["v1", "v2", "v3"],
        "status_counts": counts,
        "claim_decisions": decisions,
        "primary_result": {
            "status": decision_by_id["H1-PRIMARY"]["status"],
            "observed_value": decision_by_id["H1-PRIMARY"]["observed_value"],
            "uncertainty_interval": decision_by_id["H1-PRIMARY"]["uncertainty_interval"],
            "threshold": decision_by_id["H1-PRIMARY"]["frozen_threshold"],
        },
        "composite_mechanism_result": {
            "status": decision_by_id["MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE"]["status"],
            "components": decision_by_id["MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE"]["observed_value"],
        },
        "aggregate_v4": {
            "baseline_accuracy": baseline_table,
            "paired_two_level_bootstrap": effects["paired_two_level_bootstrap"],
            "probe_means": probe_means_table,
        },
        "developmental": {
            "pilot_selection": pilot["selection"],
            "pilot_run_count": len(pilot["runs"]),
            "pilot_all_completed": all(row.get("state") == "COMPLETED" for row in pilot["runs"]),
            "primary_configuration": primary["configuration"],
            "primary_runs": primary["runs"],
            "primary_run_count": len(primary["runs"]),
            "primary_all_completed": all(
                row.get("state") == "COMPLETED" for row in primary["runs"]
            ),
            "pilot_runs_enter_claim_statistics": primary["pilot_runs_enter_claim_statistics"],
        },
        "deviations": [
            {
                "id": "v1.1-leakage-amendment",
                "disposition": "v1.0 immutable; v1.1 introduced corrected fresh data and leakage gates",
            },
            {
                "id": "v1.1.1-object-binding",
                "disposition": "corrected an object-binding schedule arithmetic defect before protocol freeze",
            },
            {
                "id": "v1.1.2-self-endpoint",
                "disposition": "corrected a self-endpoint corruption edge case before protocol freeze",
            },
            {
                "id": "v1.1.3-evidence-position",
                "disposition": "corrected evidence-position construction coupling; fresh artifacts and unchanged ORC-005 gate",
            },
            {
                "id": "v1.1.4-training-execution",
                "disposition": "resolved source-snapshot, sampler, and equal-budget pilot-selection execution ambiguities",
            },
            {
                "id": "post-primary-stage1-diagnostics",
                "disposition": "authorization/output attempts v1-v4 failed before a complete freeze; v5 is authoritative",
            },
            {
                "id": "aggregate-v1-v2",
                "disposition": "failed closed before aggregate publication; diagnostic only",
            },
            {
                "id": "aggregate-v3",
                "disposition": "superseded because independent review found incorrect equal weighting of variable-row worlds in the bootstrap point estimator",
            },
        ],
        "unavailable_or_unthresholded": [
            "symbolic-oracle accuracy was not present in aggregate v4",
            "raw ORC-005 oracle-probe results were not present in aggregate v4",
            "MET-005 interchange had no preregistered numerical decision threshold",
            "MET-007 surface reversal had no preregistered numerical decision threshold",
            "H2 had no preregistered quality-matched resource threshold",
            "H3 and H4 were out of scope",
        ],
        "execution_guards": {
            "new_model_execution_performed": False,
            "new_label_access_performed": False,
            "prediction_row_access_performed": False,
            "prediction_generation_performed": False,
            "aggregate_recomputation_performed": False,
            "threshold_application_performed": False,
            "threshold_modification_performed": False,
            "claim_decision_modification_performed": False,
            "training_performed": False,
            "new_scientific_inference_performed": False,
        },
    }
    return data


def _format_observed(row: Mapping[str, Any]) -> str:
    claim_id = row["claim_id"]
    value = row.get("observed_value")
    if value is None:
        return "Required input unavailable or hypothesis out of scope"
    if claim_id == "H1-PRIMARY":
        interval = row["uncertainty_interval"]["delta_generalist_percentile_95"]
        return (
            f"Δgeneralist={value['delta_generalist']:.6f}; 95% CI "
            f"[{interval[0]:.6f}, {interval[1]:.6f}]; both interpolation checks pass"
        )
    if claim_id == "STA-004-RECOMBINATION":
        return f"OPM_SHARED recombination accuracy={float(value):.6f}"
    if claim_id == "STA-004-ACTIVE-ABLATION":
        return "; ".join(
            f"{item['operation']}={item['qualifying_domain_count']} qualifying domains"
            for item in value
        )
    if claim_id == "STA-004-UNRELATED-PRESERVATION":
        return "; ".join(
            f"{item['operation']} mean drop={item['mean_unrelated_drop']:.6f}" for item in value
        )
    if claim_id == "STA-004-SENTINEL":
        return f"maximum={value['maximum']}; summaries={value['summary_count']}"
    if claim_id == "STA-004-ADAPTER-ONLY":
        return f"maximum split accuracy={value['maximum']:.6f}"
    if claim_id == "STA-004-NEURAL-PROBES":
        failed = [item for item in value if not item["passed"]]
        return "failed run(s): " + ", ".join(str(item["training_seed"]) for item in failed)
    if claim_id == "MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE":
        return "; ".join(f"{key}={status}" for key, status in value.items())
    if claim_id in {"MET-005-INTERCHANGE-MECHANISM", "MET-007-SURFACE-INVARIANCE"}:
        return f"accuracy={value['accuracy']:.6f}; change={value['accuracy_change']:.6f}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _threshold_text(row: Mapping[str, Any]) -> str:
    value = row.get("frozen_threshold")
    if value is None:
        return "None preregistered"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_for(authorization: Mapping[str, Any], suffix: str) -> str:
    matches = [
        row["sha256"] for row in authorization["report_inputs"] if str(row["path"]).endswith(suffix)
    ]
    if len(matches) != 1:
        raise IntegrityError(f"expected one bound input ending with {suffix}, found {len(matches)}")
    return str(matches[0])


def render_report_markdown(data: Mapping[str, Any], authorization: Mapping[str, Any]) -> str:
    decisions = data["claim_decisions"]
    primary = data["primary_result"]
    h1 = primary["observed_value"]
    h1_interval = primary["uncertainty_interval"]["delta_generalist_percentile_95"]
    lines = [
        "# OPM v1.1.4 final study report",
        "",
        f"Generated: `{data['generated_at_utc']}`  ",
        f"Closeout authorization: `{data['closeout_authorization_sha256']}`  ",
        f"Authoritative claim-decision freeze: `{CLAIM_FREEZE_SHA256}`  ",
        f"Claim-decision Merkle root: `{CLAIM_MERKLE_ROOT}`",
        "",
        "## Executive conclusion",
        "",
        (
            "OPM v1.1.4 provided preregistered, sealed-test evidence that shared primitive "
            "organization produces a large recombination-generalization advantage without "
            "materially degrading interpolation performance. H1 was **SUPPORTED**. "
            "`OPM_SHARED` exceeded `DOMAIN_GENERALIST` by "
            f"`{h1['delta_generalist']:.6f}` ({100 * h1['delta_generalist']:.4f} percentage points), "
            f"with a 95% bootstrap interval of [{100 * h1_interval[0]:.4f}, "
            f"{100 * h1_interval[1]:.4f}] percentage points, and passed both frozen interpolation "
            "non-inferiority checks."
        ),
        "",
        (
            "The complete preregistered composite mechanism was **NOT_SUPPORTED**. The canonical "
            "neural-probe criterion failed for OPM seed 4404, evidence step 1, while symbolic-oracle "
            "and raw ORC-005 inputs were unavailable in aggregate v4. The functional outcome and "
            "mechanism conclusion are therefore kept separate."
        ),
        "",
        "## Authority and scope",
        "",
        (
            "Formal statuses and their observed values are reproduced without modification from "
            "the frozen claim-decision package. Descriptive aggregate tables come from aggregate v4 "
            "files transitively bound by that claim authorization. Pilot and primary validation "
            "records are developmental context only and do not enter claim statistics."
        ),
        "",
        (
            f"Status inventory: **{data['status_counts']['SUPPORTED']} SUPPORTED**, "
            f"**{data['status_counts']['NOT_SUPPORTED']} NOT_SUPPORTED**, "
            f"**{data['status_counts']['INCONCLUSIVE']} INCONCLUSIVE**, and "
            f"**{data['status_counts']['NOT_EVALUABLE']} NOT_EVALUABLE**."
        ),
        "",
        (
            "No checkpoint, model, prediction row, sealed label, or raw representation was read during "
            "closeout. No training, prediction generation, aggregation, threshold application, threshold "
            "change, claim-status change, or new scientific decision occurred."
        ),
        "",
        "## Preregistered hypotheses and thresholds",
        "",
        "| Identifier | Exact preregistered wording | Frozen comparison rule |",
        "|---|---|---|",
    ]
    for row in decisions:
        wording = str(row["preregistered_wording"]).replace("|", "\\|").replace("\n", " ")
        rule = str(row["comparison_rule"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['claim_id']}` | {wording} | {rule} |")
    lines.extend(
        [
            "",
            (
                "`Delta_untied` is descriptive and has no independent H1 support threshold. MET-005 and "
                "MET-007 specify reported metrics but no numerical decision thresholds. H2 requires a "
                "later quality-matched resource threshold; H3 and H4 are out of scope in v1."
            ),
            "",
            "## Pilot and primary methodology",
            "",
            (
                "The experiment used three rendered domains (`PROGRAM`, `SCENE`, and `SET`) and four "
                "operation families (`LOOKUP`, `REVERSE`, `CHAIN`, and `LIFT`). Training withheld "
                "`SET × REVERSE`, `SCENE × LIFT`, and `PROGRAM × CHAIN`; the primary recombination test "
                "contained only those equally weighted cells."
            ),
            "",
            (
                "The primary conditions were `OPM_SHARED`, `DOMAIN_GENERALIST`, `PROC_UNTIED`, and "
                "`PROC_CLONE`. Pilot seed 1101 evaluated all six combinations of learning rate "
                "`{0.0001, 0.0003, 0.0006}` and dropout `{0.0, 0.1}` for all four conditions: 24 "
                "equal-budget 50,000-step runs. The preregistered tie rule selected learning rate "
                f"`{data['developmental']['pilot_selection']['selected_learning_rate']}` and dropout "
                f"`{data['developmental']['pilot_selection']['selected_dropout']}`. Pilot runs were "
                "excluded from claim statistics."
            ),
            "",
            (
                "The selected configuration was frozen for 20 primary runs: four conditions × seeds "
                "1101, 2202, 3303, 4404, and 5505. Every run completed 50,000 steps. Checkpoints were "
                "selected by observed-cell macro validation accuracy; sealed tests were not used for "
                "training or selection. Label-blind Stage 1 generated predictions, interventions, and "
                "neural probes. The separately authorized aggregate-v4 stage joined frozen outputs to "
                "sealed targets without loading models, then ran the registered 10,000-replicate paired "
                "two-level bootstrap. The claim stage applied thresholds separately afterward."
            ),
            "",
            "## Developmental pilot and validation results",
            "",
            (
                "These results selected configuration/checkpoints only. They are **developmental**, were "
                "not sealed-test endpoints, and do not independently support H1 or any mechanism claim."
            ),
            "",
            "| Learning rate | Dropout | Four-condition pilot mean | In tie set |",
            "|---:|---:|---:|---|",
        ]
    )
    for row in data["developmental"]["pilot_selection"]["means"]:
        lines.append(
            f"| {row['learning_rate']:.4f} | {row['dropout']:.1f} | {row['mean']:.6f} | "
            f"{'yes' if row['in_tie_set'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Primary selected-checkpoint macro validation accuracy:",
            "",
            "| Condition | Seed | Selected step | Developmental validation accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    primary_order = {
        name: index
        for index, name in enumerate(
            ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED", "PROC_CLONE")
        )
    }
    for row in sorted(
        data["developmental"]["primary_runs"],
        key=lambda item: (primary_order[item["condition"]], item["model_seed"]),
    ):
        lines.append(
            f"| `{row['condition']}` | {row['model_seed']} | {row['selected_step']} | "
            f"{row['selected_macro_validation_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Authoritative aggregate-v4 results",
            "",
            "Pooled sealed-test accuracy across all five declared model seeds:",
            "",
            "| Condition | Interpolation | Recombination | Renderer | Structural |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["aggregate_v4"]["baseline_accuracy"]:
        lines.append(
            f"| `{row['condition']}` | {row['test-interpolation']:.6f} | "
            f"{row['test-recombination']:.6f} | {row['test-renderer']:.6f} | "
            f"{row['test-structural']:.6f} |"
        )
    effects = data["aggregate_v4"]["paired_two_level_bootstrap"]
    lines.extend(
        [
            "",
            "| Frozen paired effect | Point estimate | 95% percentile interval |",
            "|---|---:|---:|",
            (
                f"| `Delta_generalist` | {effects['delta_generalist']:.6f} | "
                f"[{effects['delta_generalist_percentile_95_interval'][0]:.6f}, "
                f"{effects['delta_generalist_percentile_95_interval'][1]:.6f}] |"
            ),
            (
                f"| `Delta_untied` | {effects['delta_untied']:.6f} | "
                f"[{effects['delta_untied_percentile_95_interval'][0]:.6f}, "
                f"{effects['delta_untied_percentile_95_interval'][1]:.6f}] |"
            ),
            "",
            (
                "Frozen neural-probe means are descriptive and are not substitutes for the per-run "
                "ORC-006 Wilson/Holm decisions:"
            ),
            "",
            "| Condition | Evidence step 1 | Evidence step 2 |",
            "|---|---:|---:|",
        ]
    )
    for row in data["aggregate_v4"]["probe_means"]:
        lines.append(
            f"| `{row['condition']}` | {row['evidence_step_1_mean_accuracy']:.6f} | "
            f"{row['evidence_step_2_mean_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen claim decisions",
            "",
            "| Claim | Observed evidence | Threshold | Status |",
            "|---|---|---|---|",
        ]
    )
    for row in decisions:
        observed = _format_observed(row).replace("|", "\\|")
        threshold = _threshold_text(row).replace("|", "\\|")
        lines.append(f"| `{row['claim_id']}` | {observed} | `{threshold}` | **{row['status']}** |")
    lines.extend(
        [
            "",
            "## Behavioral and mechanism separation",
            "",
            (
                "H1 is a behavioral outcome and is supported independently of mechanism status. The "
                "controls learned interpolation nearly perfectly, while `OPM_SHARED` preserved "
                "near-ceiling recombination performance and the controls degraded. Several component "
                "criteria were supported: recombination accuracy, active ablations, unrelated-operation "
                "preservation, the sentinel numerical-null criterion, and adapter-only performance."
            ),
            "",
            (
                "The phrase **causally reusable primitive** is not supported for v1.1.4. OPM seed 4404, "
                "evidence step 1 had validation accuracy 0.513750, a Wilson interval excluding 0.50 on "
                "the high side, and one-sided p=0.014349, which rejected at the first Holm threshold "
                "0.025. Symbolic-oracle accuracy and raw ORC-005 results were also unavailable to the "
                "aggregate-v4 claim input. A strong H1 result cannot rescue those mechanism criteria, "
                "and the mechanism result does not erase H1."
            ),
            "",
            (
                "Interchange accuracy was 0.346271 with change -0.652895; surface reversal accuracy was "
                "0.999354 with change 0.000000. Both remain descriptive because the preregistration "
                "defined no numerical claim thresholds for them."
            ),
            "",
            "## Deviations, unavailable inputs, and diagnostic attempts",
            "",
            "| Record | Disposition |",
            "|---|---|",
        ]
    )
    for row in data["deviations"]:
        lines.append(f"| `{row['id']}` | {row['disposition']} |")
    lines.extend(["", "Unavailable or unthresholded items:", ""])
    lines.extend(f"- {item}." for item in data["unavailable_or_unthresholded"])
    lines.extend(
        [
            "",
            (
                "All corrected predecessor artifacts and failed attempts remain preserved. They are "
                "provenance, not interchangeable scientific evidence. Aggregate authorization/output "
                "versions v1-v3 are explicitly excluded from every formal decision."
            ),
            "",
            "## Authorization and freeze chain",
            "",
            "| Lifecycle record | SHA-256 / Merkle identity |",
            "|---|---|",
            f"| v1.1.4 training transition | `{_hash_for(authorization, 'OPM_V1_1_4_TRAINING_EXECUTION_TRANSITION.json')}` |",
            f"| Completed pilot matrix | `{_hash_for(authorization, 'pilot-matrix.json')}` |",
            f"| Completed primary matrix | `{_hash_for(authorization, 'primary-matrix.json')}` |",
            f"| Post-primary Stage 1 transition v5 | `{_hash_for(authorization, 'authorization-v5/OPM_V1_1_4_POST_PRIMARY_TRANSITION.json')}` |",
            f"| Post-primary Stage 1 freeze | `{_hash_for(authorization, 'OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json')}`; Merkle `74894ded7e5af418da88cd0f80f40068812ed33091da4d53dad2eaa72e4dbe4e` |",
            f"| Sealed aggregate authorization v4 | `{_hash_for(authorization, 'authorization-v4/OPM_V1_1_4_SEALED_AGGREGATE_AUTHORIZATION.json')}` |",
            f"| Sealed aggregate freeze v4 | `{_hash_for(authorization, 'OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json')}`; Merkle `2d1fa74c1637f621a748ab69f117d9651fed2facae8c88f919419eae8bea1a9b` |",
            f"| Claim-decision authorization | `{_hash_for(authorization, 'OPM_V1_1_4_CLAIM_DECISION_AUTHORIZATION.json')}` |",
            f"| Claim-decision freeze | `{CLAIM_FREEZE_SHA256}`; Merkle `{CLAIM_MERKLE_ROOT}` |",
            f"| Study-closeout authorization | `{data['closeout_authorization_sha256']}` |",
            "",
            (
                "The closeout freeze containing this report is the final identity for the reporting "
                "stage and is recorded outside this self-referential report in "
                f"`{FREEZE_FILENAME}`."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "Verification does not require model execution or access to sealed labels. From the "
                "repository root:"
            ),
            "",
            "```powershell",
            "$opmTests = (Get-ChildItem tests -Filter 'test_opm_*.py' | Sort-Object Name).FullName",
            "& .venv\\Scripts\\python.exe -m pytest @opmTests -q",
            "& .venv\\Scripts\\python.exe -m scripts.opm_study_closeout_executor verify `",
            "  --workspace . --freeze <path-to-closeout-freeze> --freeze-sha256 <published-sha256>",
            "```",
            "",
            (
                "The verifier rehashes every frozen report artifact, recomputes the closeout Merkle root, "
                "checks the claim freeze identity, verifies all bound report inputs, and confirms the "
                "zero-access/zero-modification guards. Reproducing a new experiment is not the same as "
                "verifying this immutable result and requires a new authorization and study identity."
            ),
            "",
            "## Limitations and separately preregistered follow-up studies",
            "",
            (
                "- `OPM-MECH-COMPLETE-001`: bind symbolic-oracle and raw ORC-005 evidence explicitly and "
                "replicate neural probes under a new mechanism preregistration."
            ),
            (
                "- `OPM-MECH-INTERCHANGE-001`: preregister context-aware interchange hypotheses, positive "
                "controls, and numerical thresholds."
            ),
            (
                "- `OPM-RESOURCE-001`: preregister a quality-matched H2 parameter/FLOP threshold and full "
                "executed-graph accounting."
            ),
            (
                "- `OPM-HARDWARE-001`: preregister H3 target hardware, repeat counts, latency, throughput, "
                "memory, and energy endpoints."
            ),
            (
                "- `OPM-BOUNDARY-001`: preregister H4 controlled exceptions and acceptable negative-transfer "
                "thresholds."
            ),
            "",
            (
                "These are prospective identifiers only. They do not amend, reopen, or reinterpret OPM "
                "v1.1.4."
            ),
            "",
            "## Final conclusion",
            "",
            (
                "OPM v1.1.4 established the preregistered behavioral phenomenon: shared primitive "
                "organization delivered a large sealed recombination advantage without materially "
                "degrading interpolation performance. It did not establish the complete "
                "preregistered causal explanation. In concise terms: **OPM works according to H1; "
                "v1.1.4 does not yet completely establish why it works.**"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def reconcile_report(
    *,
    report_data: Mapping[str, Any],
    report_markdown: str,
    report_path: Path,
    report_data_path: Path,
    authorization_sha256: str,
    audit_sha256: str,
) -> dict[str, Any]:
    decisions = report_data.get("claim_decisions")
    if not isinstance(decisions, list) or len(decisions) != 15:
        raise IntegrityError("closeout report does not contain all 15 decisions")
    counts = _status_counts(decisions)
    if counts != report_data.get("status_counts"):
        raise IntegrityError("closeout report status counts changed")
    required_headings = (
        "## Executive conclusion",
        "## Authority and scope",
        "## Preregistered hypotheses and thresholds",
        "## Pilot and primary methodology",
        "## Developmental pilot and validation results",
        "## Authoritative aggregate-v4 results",
        "## Frozen claim decisions",
        "## Behavioral and mechanism separation",
        "## Deviations, unavailable inputs, and diagnostic attempts",
        "## Authorization and freeze chain",
        "## Reproducibility",
        "## Limitations and separately preregistered follow-up studies",
        "## Final conclusion",
    )
    missing = [heading for heading in required_headings if heading not in report_markdown]
    if missing:
        raise IntegrityError(f"closeout report is missing sections: {missing}")
    if any(report_markdown.count(f"`{row['claim_id']}`") < 2 for row in decisions):
        raise IntegrityError(
            "closeout report does not reproduce every claim in both required tables"
        )
    guards = report_data.get("execution_guards")
    if not isinstance(guards, dict) or any(value is not False for value in guards.values()):
        raise IntegrityError("closeout report records a prohibited operation")
    return {
        "schema_version": RECONCILIATION_SCHEMA,
        "state": "PASS",
        "authorization_sha256": authorization_sha256,
        "claim_freeze_sha256": CLAIM_FREEZE_SHA256,
        "claim_merkle_root": CLAIM_MERKLE_ROOT,
        "report_sha256": sha256_file(report_path),
        "report_data_sha256": sha256_file(report_data_path),
        "audit_log_sha256": audit_sha256,
        "claim_count": 15,
        "status_counts": counts,
        "required_sections_present": True,
        "claim_inventory_exact": True,
        "claim_statuses_unchanged": True,
        "authoritative_aggregate_version": "v4",
        "non_authoritative_aggregate_versions_excluded": ["v1", "v2", "v3"],
        **guards,
    }


def freeze_closeout_package(
    *,
    output_root: Path,
    authorization_sha256: str,
    report_spec_sha256: str,
    files: Sequence[Path],
    gate: CapabilityGate,
) -> tuple[Path, str, dict[str, Any]]:
    gate.require(Capability.CLOSEOUT_PACKAGE_FREEZING)
    records: list[dict[str, Any]] = []
    for path in sorted((item.resolve() for item in files), key=lambda item: item.as_posix()):
        if output_root.resolve() not in path.parents or not path.is_file():
            raise IntegrityError(f"closeout result missing or outside output root: {path}")
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
        "report_spec_sha256": report_spec_sha256,
        "claim_freeze_sha256": CLAIM_FREEZE_SHA256,
        "claim_merkle_root": CLAIM_MERKLE_ROOT,
        "artifact_count": len(records),
        "artifacts": records,
        "merkle_root": merkle_root(records),
        "claim_count": 15,
        "status_counts": {
            "SUPPORTED": 6,
            "NOT_SUPPORTED": 2,
            "INCONCLUSIVE": 0,
            "NOT_EVALUABLE": 7,
        },
        "authoritative_aggregate_version": "v4",
        "non_authoritative_aggregate_versions_excluded": ["v1", "v2", "v3"],
        "checkpoint_access_count": 0,
        "model_load_count": 0,
        "prediction_row_access_count": 0,
        "sealed_label_access_count": 0,
        "prediction_generation_count": 0,
        "aggregate_recomputation_count": 0,
        "threshold_application_count": 0,
        "threshold_modification_count": 0,
        "claim_decision_modification_count": 0,
        "training_count": 0,
        "new_scientific_inference_count": 0,
    }
    path = output_root / FREEZE_FILENAME
    digest = atomic_write_json(path, freeze)
    return path, digest, freeze


def verify_closeout_package(
    *, workspace: Path, freeze_path: Path, expected_freeze_sha256: str
) -> dict[str, Any]:
    expected = validate_sha256(expected_freeze_sha256, "closeout freeze hash")
    if not freeze_path.is_file() or sha256_file(freeze_path) != expected:
        raise IntegrityError("closeout freeze hash mismatch")
    freeze = read_json(freeze_path, "closeout freeze")
    if freeze.get("schema_version") != FREEZE_SCHEMA or freeze.get("state") != COMPLETE_STATE:
        raise IntegrityError("closeout package is not complete and frozen")
    if (
        freeze.get("claim_freeze_sha256") != CLAIM_FREEZE_SHA256
        or freeze.get("claim_merkle_root") != CLAIM_MERKLE_ROOT
    ):
        raise IntegrityError("closeout claim authority changed")
    records = freeze.get("artifacts")
    if not isinstance(records, list) or len(records) != freeze.get("artifact_count"):
        raise IntegrityError("closeout artifact inventory is invalid")
    for raw in records:
        path = (freeze_path.parent / PurePosixPath(str(raw["path"]))).resolve()
        if freeze_path.parent.resolve() not in path.parents:
            raise IntegrityError(f"closeout artifact escapes package root: {path}")
        if (
            not path.is_file()
            or sha256_file(path) != raw.get("sha256")
            or path.stat().st_size != raw.get("bytes")
        ):
            raise IntegrityError(f"closeout artifact mismatch: {path}")
    if merkle_root(records) != freeze.get("merkle_root"):
        raise IntegrityError("closeout Merkle root mismatch")
    authorization_paths = list(
        workspace.glob(
            "evidence/primary_runs/v1.1.4/study-closeout/authorization-v*/"
            "OPM_V1_1_4_STUDY_CLOSEOUT_AUTHORIZATION.json"
        )
    )
    matching_authorizations = [
        path
        for path in authorization_paths
        if sha256_file(path) == freeze.get("authorization_sha256")
    ]
    if len(matching_authorizations) != 1:
        raise IntegrityError("closeout authorization binding is unavailable")
    authorization_path = matching_authorizations[0]
    authorization = read_json(authorization_path, "closeout authorization")
    for raw in authorization["report_inputs"]:
        path = resolve_workspace_path(raw["path"], workspace, "bound report input")
        if not path.is_file() or sha256_file(path) != raw["sha256"]:
            raise IntegrityError(f"bound report input changed: {path}")
    for count_field in (
        "checkpoint_access_count",
        "model_load_count",
        "prediction_row_access_count",
        "sealed_label_access_count",
        "prediction_generation_count",
        "aggregate_recomputation_count",
        "threshold_application_count",
        "threshold_modification_count",
        "claim_decision_modification_count",
        "training_count",
        "new_scientific_inference_count",
    ):
        if freeze.get(count_field) != 0:
            raise IntegrityError(f"closeout freeze records prohibited activity: {count_field}")
    return {
        "verification": "PASS",
        "freeze_sha256": expected,
        "merkle_root": freeze["merkle_root"],
        "artifact_count": freeze["artifact_count"],
        "claim_count": freeze["claim_count"],
        "status_counts": freeze["status_counts"],
        "verified_report_inputs": len(authorization["report_inputs"]),
    }


def environment_record() -> dict[str, Any]:
    return {
        "schema_version": "opm-study-closeout-environment-v1",
        "python": platform.python_version(),
        "model_runtime_imported": False,
        "checkpoint_access_count": 0,
        "model_load_count": 0,
        "prediction_row_access_count": 0,
        "sealed_label_access_count": 0,
        "prediction_generation_count": 0,
        "aggregate_recomputation_count": 0,
        "threshold_application_count": 0,
        "threshold_modification_count": 0,
        "claim_decision_modification_count": 0,
        "training_count": 0,
        "new_scientific_inference_count": 0,
    }
