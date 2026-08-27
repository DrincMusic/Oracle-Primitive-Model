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
AUTHORIZATION_SCHEMA = "opm-claim-decision-authorization-v1"
DECISION_SPEC_SCHEMA = "opm-claim-decision-spec-v1"
PREFLIGHT_SCHEMA = "opm-claim-decision-preflight-v1"
DECISION_RECORD_SCHEMA = "opm-claim-decision-record-v1"
RECONCILIATION_SCHEMA = "opm-claim-decision-reconciliation-v1"
FREEZE_SCHEMA = "opm-claim-decision-freeze-v1"
FREEZE_FILENAME = "OPM_V1_1_4_CLAIM_DECISION_FREEZE.json"

AGGREGATE_FREEZE_SCHEMA = "opm-sealed-aggregate-freeze-v1"
AGGREGATE_STATE = "SEALED_LABEL_AGGREGATE_EVALUATION_COMPLETE_AND_FROZEN"
AUTHORIZED_STATE = "CLAIM_DECISION_AUTHORIZED"
COMPLETE_STATE = "CLAIM_DECISIONS_RECONCILED_AND_FROZEN"

DECISION_VOCABULARY = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_EVALUABLE")
AGGREGATE_FILENAMES = (
    "baseline-metrics.json",
    "primary-effects.json",
    "intervention-metrics.json",
    "probe-metrics.json",
)
AUTHORIZED_OPERATIONS = (
    "frozen_aggregate_read",
    "claim_threshold_application",
    "decision_record_generation",
    "decision_reconciliation",
    "decision_package_freezing",
)
PROHIBITED_OPERATIONS = (
    "checkpoint_access",
    "model_loading",
    "prediction_row_access",
    "sealed_label_access",
    "prediction_generation",
    "aggregate_recomputation",
    "threshold_modification",
    "training",
    "scientific_narrative_generation",
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


class ClaimDecisionError(RuntimeError):
    pass


class AuthorizationError(ClaimDecisionError, PermissionError):
    pass


class IntegrityError(ClaimDecisionError, ValueError):
    pass


class DecisionError(ClaimDecisionError, ValueError):
    pass


class Capability(StrEnum):
    FROZEN_AGGREGATE_READ = "frozen_aggregate_read"
    CLAIM_THRESHOLD_APPLICATION = "claim_threshold_application"
    DECISION_RECORD_GENERATION = "decision_record_generation"
    DECISION_RECONCILIATION = "decision_reconciliation"
    DECISION_PACKAGE_FREEZING = "decision_package_freezing"
    CHECKPOINT_ACCESS = "checkpoint_access"
    MODEL_LOADING = "model_loading"
    PREDICTION_ROW_ACCESS = "prediction_row_access"
    SEALED_LABEL_ACCESS = "sealed_label_access"
    PREDICTION_GENERATION = "prediction_generation"
    AGGREGATE_RECOMPUTATION = "aggregate_recomputation"
    THRESHOLD_MODIFICATION = "threshold_modification"
    TRAINING = "training"
    SCIENTIFIC_NARRATIVE_GENERATION = "scientific_narrative_generation"


DENIED_CAPABILITIES = frozenset(
    {
        Capability.CHECKPOINT_ACCESS,
        Capability.MODEL_LOADING,
        Capability.PREDICTION_ROW_ACCESS,
        Capability.SEALED_LABEL_ACCESS,
        Capability.PREDICTION_GENERATION,
        Capability.AGGREGATE_RECOMPUTATION,
        Capability.THRESHOLD_MODIFICATION,
        Capability.TRAINING,
        Capability.SCIENTIFIC_NARRATIVE_GENERATION,
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


def atomic_write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(payload)
    if path.exists():
        if sha256_file(path) == digest:
            return digest
        raise FileExistsError(f"refusing to overwrite immutable claim artifact: {path}")
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
            raise IntegrityError(f"temporary claim artifact failed verification: {path}")
        if path.exists():
            raise FileExistsError(f"claim artifact appeared during publication: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def atomic_write_json(path: Path, value: object) -> str:
    return atomic_write_bytes(path, canonical_json_bytes(value))


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
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as error:
        raise IntegrityError(f"{description} escapes workspace: {resolved}") from error
    return resolved


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


def assert_claim_source_is_read_only(path: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise IntegrityError(f"invalid claim source: {path}") from error
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
            raise AuthorizationError(f"claim executor imports prohibited runtime: {name}")


class CapabilityGate:
    def __init__(self, enabled: Iterable[Capability] = ()) -> None:
        self._enabled = frozenset(enabled) - DENIED_CAPABILITIES

    @classmethod
    def from_authorization(cls, authorization: Mapping[str, Any]) -> Self:
        if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
            raise AuthorizationError("claim authorization schema is not approved")
        if authorization.get("authorized_operations") != list(AUTHORIZED_OPERATIONS):
            raise AuthorizationError("claim authorization operation set changed")
        if authorization.get("prohibited_operations") != list(PROHIBITED_OPERATIONS):
            raise AuthorizationError("claim authorization prohibition set is incomplete")
        return cls(Capability(operation) for operation in AUTHORIZED_OPERATIONS)

    def require(self, capability: Capability) -> None:
        if capability in DENIED_CAPABILITIES or capability not in self._enabled:
            raise AuthorizationError(f"claim capability denied: {capability.value}")

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(capability.value for capability in self._enabled))


@dataclass(frozen=True)
class ResourcePolicy:
    aggregate_paths: frozenset[Path]
    output_root: Path
    gate: CapabilityGate

    def require_aggregate_read(self, path: Path) -> Path:
        self.gate.require(Capability.FROZEN_AGGREGATE_READ)
        resolved = path.resolve()
        if resolved not in self.aggregate_paths:
            raise AuthorizationError(
                f"claim executor cannot read non-aggregate resource: {resolved}"
            )
        return resolved

    def require_write(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved != self.output_root and self.output_root not in resolved.parents:
            raise AuthorizationError(f"claim output escapes authorized root: {resolved}")
        return resolved


def build_decision_spec(
    *,
    workspace: Path,
    aggregate_freeze_path: Path,
    aggregate_freeze_sha256: str,
    threshold_sources: Sequence[Path],
) -> dict[str, Any]:
    freeze = read_json(aggregate_freeze_path, "aggregate v4 freeze")
    if sha256_file(aggregate_freeze_path) != validate_sha256(
        aggregate_freeze_sha256, "aggregate freeze hash"
    ):
        raise IntegrityError("aggregate v4 freeze hash mismatch")
    if (
        freeze.get("schema_version") != AGGREGATE_FREEZE_SCHEMA
        or freeze.get("state") != AGGREGATE_STATE
    ):
        raise IntegrityError("aggregate package is not complete and frozen")
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, list):
        raise IntegrityError("aggregate freeze lacks artifact bindings")
    artifact_by_name = {
        str(item.get("path")): dict(item) for item in artifacts if isinstance(item, dict)
    }
    if any(name not in artifact_by_name for name in AGGREGATE_FILENAMES):
        raise IntegrityError("aggregate freeze does not bind every decision input")
    inputs = [
        {
            "path": relative_path(aggregate_freeze_path.parent / name, workspace),
            "sha256": validate_sha256(artifact_by_name[name].get("sha256"), f"{name} hash"),
            "bytes": int(artifact_by_name[name]["bytes"]),
        }
        for name in AGGREGATE_FILENAMES
    ]
    threshold_records = [
        {"path": relative_path(path, workspace), "sha256": sha256_file(path)}
        for path in threshold_sources
    ]
    return {
        "schema_version": DECISION_SPEC_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "authoritative_aggregate_version": "v4",
        "non_authoritative_versions_excluded": ["v1", "v2", "v3"],
        "aggregate_freeze": {
            "path": relative_path(aggregate_freeze_path, workspace),
            "sha256": aggregate_freeze_sha256,
            "merkle_root": freeze.get("merkle_root"),
            "authorization_sha256": freeze.get("authorization_sha256"),
        },
        "aggregate_inputs": inputs,
        "threshold_sources": threshold_records,
        "decision_vocabulary": list(DECISION_VOCABULARY),
        "rules": [
            {
                "claim_id": "H1-PRIMARY",
                "claim_type": "preregistered_hypothesis",
                "wording": (
                    "Reusable primitives improve performance on valid, withheld domain-operation "
                    "combinations or unseen procedural compositions relative to suitable "
                    "non-sharing and generic baselines."
                ),
                "rule": (
                    "SUPPORTED iff lower95(Delta_generalist)>0.02 and OPM interpolation is no "
                    "more than 0.01 below DOMAIN_GENERALIST and PROC_UNTIED; NOT_SUPPORTED iff "
                    "upper95<=0.02; otherwise INCONCLUSIVE"
                ),
            },
            {
                "claim_id": "STA-004-SYMBOLIC-ORACLE",
                "claim_type": "mechanism_criterion",
                "wording": "symbolic oracle accuracy equals `1.000`;",
                "rule": "observed accuracy == 1.000",
            },
            {
                "claim_id": "STA-004-RECOMBINATION",
                "claim_type": "mechanism_criterion",
                "wording": "shared-model recombination accuracy exceeds `0.80`;",
                "rule": "OPM_SHARED pooled recombination accuracy > 0.80",
            },
            {
                "claim_id": "STA-004-ACTIVE-ABLATION",
                "claim_type": "mechanism_criterion",
                "wording": (
                    "ablating the corresponding active primitive reduces its selected-operation "
                    "accuracy by at least `0.20` in at least two trained domains;"
                ),
                "rule": (
                    "for every active operation, count trained interpolation domains with "
                    "baseline_accuracy-intervention_accuracy >= 0.20; require count >= 2"
                ),
            },
            {
                "claim_id": "STA-004-UNRELATED-PRESERVATION",
                "claim_type": "mechanism_criterion",
                "wording": (
                    "average unrelated-operation accuracy drop under that ablation is below `0.05`;"
                ),
                "rule": (
                    "for every active operation, arithmetic mean of baseline_accuracy-"
                    "intervention_accuracy over unrelated interpolation domain-operation "
                    "aggregate rows < 0.05"
                ),
            },
            {
                "claim_id": "STA-004-SENTINEL",
                "claim_type": "mechanism_criterion",
                "wording": "sentinel ablation changes all logits by less than `1e-7`;",
                "rule": "maximum of all frozen sentinel max-absolute-logit-change values < 1e-7",
            },
            {
                "claim_id": "STA-004-ADAPTER-ONLY",
                "claim_type": "mechanism_criterion",
                "wording": "adapter-only accuracy is at most `0.60`;",
                "rule": "maximum OPM_SHARED adapter-only split accuracy <= 0.60",
            },
            {
                "claim_id": "STA-004-RAW-ORACLE-PROBES",
                "claim_type": "mechanism_criterion",
                "wording": "raw oracle leakage probes pass ORC-005;",
                "rule": "requires all three ORC-005 results and Holm family decision",
            },
            {
                "claim_id": "STA-004-NEURAL-PROBES",
                "claim_type": "mechanism_criterion",
                "wording": (
                    "and neural evidence-vector probes pass ORC-006 for every run used in the "
                    "mechanism claim."
                ),
                "rule": (
                    "for each OPM_SHARED seed, both accuracies <=0.55, both Wilson intervals contain "
                    "0.50, and two-test Holm at alpha=0.05 rejects neither one-sided null"
                ),
            },
            {
                "claim_id": "MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE",
                "claim_type": "composite_mechanism_claim",
                "wording": "Evidence of reusable computation additionally requires all of:",
                "rule": "SUPPORTED only if every STA-004 criterion is SUPPORTED",
            },
            {
                "claim_id": "MET-005-INTERCHANGE-MECHANISM",
                "claim_type": "descriptive_metric_without_threshold",
                "wording": (
                    "Pair examples from the same latent world, query, label, and procedure but "
                    "different renderers. At the boundary before the last primitive call, swap "
                    "normalized execution states while retaining the destination evidence vector. "
                    "Report swapped accuracy and the drop from unswapped paired accuracy."
                ),
                "rule": "no preregistered numerical decision threshold; NOT_EVALUABLE",
            },
            {
                "claim_id": "MET-007-SURFACE-INVARIANCE",
                "claim_type": "descriptive_metric_without_threshold",
                "wording": (
                    "Use renderer variant 2 with relation symbols reassigned through a test-only "
                    "permutation. Oracle bindings remain correct. Report the accuracy drop relative "
                    "to the ordinary renderer test."
                ),
                "rule": "no preregistered numerical decision threshold; NOT_EVALUABLE",
            },
            {
                "claim_id": "H2-THEORETICAL-RESOURCE-ADVANTAGE",
                "claim_type": "deferred_hypothesis",
                "wording": (
                    "At matched quality, the sparse primitive architecture uses meaningfully fewer "
                    "active parameters, active FLOPs, or stored duplicate parameters than an "
                    "appropriate dense or non-sharing alternative."
                ),
                "rule": "STA-005 has no preregistered quality-matched resource threshold; NOT_EVALUABLE",
            },
            {
                "claim_id": "H3-REAL-HARDWARE-ADVANTAGE",
                "claim_type": "out_of_scope_hypothesis",
                "wording": (
                    "The theoretical savings produce measurable improvements in wall-clock latency, "
                    "throughput, training time, memory, or energy on specified target hardware."
                ),
                "rule": "STA-006 declares H3 out of scope; NOT_EVALUABLE",
            },
            {
                "claim_id": "H4-CORRECT-BOUNDARY-OF-REUSE",
                "claim_type": "out_of_scope_hypothesis",
                "wording": (
                    "The architecture transfers rules that are genuinely shared without "
                    "catastrophically transferring rules that are domain-specific."
                ),
                "rule": "MET-008 and STA-007 declare H4 out of scope; NOT_EVALUABLE",
            },
        ],
        "execution_policy": {
            "new_model_execution": False,
            "new_label_access": False,
            "prediction_row_access": False,
            "aggregate_recomputation": False,
            "aggregate_values_modified": False,
            "thresholds_modified": False,
            "scientific_narrative": False,
        },
    }


def create_authorization(
    *,
    workspace: Path,
    authorization_directory: Path,
    aggregate_freeze_path: Path,
    aggregate_freeze_sha256: str,
    threshold_sources: Sequence[Path],
    evaluator_sources: Sequence[Path],
    user_request: str,
) -> tuple[Path, str, Path, str]:
    for source in evaluator_sources:
        assert_claim_source_is_read_only(source)
    spec = build_decision_spec(
        workspace=workspace,
        aggregate_freeze_path=aggregate_freeze_path,
        aggregate_freeze_sha256=aggregate_freeze_sha256,
        threshold_sources=threshold_sources,
    )
    spec_path = authorization_directory / "OPM_V1_1_4_CLAIM_DECISION_SPEC.json"
    spec_sha256 = atomic_write_json(spec_path, spec)
    sources = source_manifest(evaluator_sources, workspace)
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "from_state": AGGREGATE_STATE,
        "to_state": AUTHORIZED_STATE,
        "authorized_at_utc": utc_now(),
        "authorization_basis": {
            "actor": "workspace_owner_via_codex",
            "request": user_request,
            "scope": "read-only claim decision from authoritative aggregate v4",
        },
        "aggregate_authorization_sha256": spec["aggregate_freeze"]["authorization_sha256"],
        "aggregate_freeze_path": spec["aggregate_freeze"]["path"],
        "aggregate_freeze_sha256": aggregate_freeze_sha256,
        "aggregate_merkle_root": spec["aggregate_freeze"]["merkle_root"],
        "aggregate_inputs": spec["aggregate_inputs"],
        "threshold_sources": spec["threshold_sources"],
        "decision_spec_path": relative_path(spec_path, workspace),
        "decision_spec_sha256": spec_sha256,
        "decision_vocabulary": list(DECISION_VOCABULARY),
        "evaluator_source_commit": git_commit(workspace),
        "evaluator_sources": sources,
        "evaluator_source_aggregate_sha256": canonical_sha256(sources),
        "authorized_operations": list(AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(PROHIBITED_OPERATIONS),
        "new_model_execution_authorized": False,
        "new_label_access_authorized": False,
        "aggregate_recomputation_authorized": False,
        "threshold_modification_authorized": False,
        "claim_threshold_application_authorized": True,
        "decision_record_generation_authorized": True,
    }
    path = authorization_directory / "OPM_V1_1_4_CLAIM_DECISION_AUTHORIZATION.json"
    digest = atomic_write_json(path, authorization)
    return path, digest, spec_path, spec_sha256


@dataclass(frozen=True)
class PreflightResult:
    schema_version: str
    state: str
    authorization_sha256: str
    aggregate_freeze_sha256: str
    aggregate_merkle_root: str
    decision_spec_sha256: str
    verified_aggregate_files: int
    decision_input_files: int
    checkpoint_accesses: int
    model_loads: int
    prediction_row_accesses: int
    sealed_label_accesses: int
    prediction_generations: int
    aggregate_recomputations: int
    threshold_modifications: int
    enabled_capabilities: tuple[str, ...]


def preflight_claim_decision(
    *,
    workspace: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    output_root: Path,
) -> tuple[PreflightResult, CapabilityGate, ResourcePolicy, dict[str, Any], dict[str, Any]]:
    expected = validate_sha256(expected_authorization_sha256, "claim authorization hash")
    if not authorization_path.is_file() or sha256_file(authorization_path) != expected:
        raise AuthorizationError("missing or invalid claim authorization")
    authorization = read_json(authorization_path, "claim authorization")
    gate = CapabilityGate.from_authorization(authorization)
    if (
        authorization.get("from_state") != AGGREGATE_STATE
        or authorization.get("to_state") != AUTHORIZED_STATE
    ):
        raise AuthorizationError("claim authorization lifecycle boundary changed")
    for field in (
        "new_model_execution_authorized",
        "new_label_access_authorized",
        "aggregate_recomputation_authorized",
        "threshold_modification_authorized",
    ):
        if authorization.get(field) is not False:
            raise AuthorizationError(f"claim authorization improperly enables {field}")
    if authorization.get("claim_threshold_application_authorized") is not True:
        raise AuthorizationError("claim threshold application is not authorized")
    sources = authorization.get("evaluator_sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise AuthorizationError("claim evaluator source binding is incomplete")
    verified_sources: list[dict[str, str]] = []
    for raw in sources:
        if not isinstance(raw, dict):
            raise AuthorizationError("invalid claim source binding")
        path = resolve_workspace_path(raw.get("path"), workspace, "claim source")
        digest = validate_sha256(raw.get("sha256"), "claim source hash")
        if not path.is_file() or sha256_file(path) != digest:
            raise AuthorizationError(f"claim source hash mismatch: {path}")
        assert_claim_source_is_read_only(path)
        verified_sources.append({"path": str(raw["path"]), "sha256": digest})
    if canonical_sha256(verified_sources) != authorization.get("evaluator_source_aggregate_sha256"):
        raise AuthorizationError("claim source aggregate binding changed")
    spec_path = resolve_workspace_path(
        authorization.get("decision_spec_path"), workspace, "decision spec"
    )
    spec_sha256 = validate_sha256(authorization.get("decision_spec_sha256"), "decision spec hash")
    if not spec_path.is_file() or sha256_file(spec_path) != spec_sha256:
        raise AuthorizationError("claim decision spec hash mismatch")
    spec = read_json(spec_path, "claim decision spec")
    if spec.get("schema_version") != DECISION_SPEC_SCHEMA:
        raise AuthorizationError("claim decision spec schema changed")
    if spec.get("decision_vocabulary") != list(DECISION_VOCABULARY):
        raise AuthorizationError("claim decision vocabulary changed")
    if spec.get("authoritative_aggregate_version") != "v4" or spec.get(
        "non_authoritative_versions_excluded"
    ) != ["v1", "v2", "v3"]:
        raise AuthorizationError("claim decision aggregate-version binding changed")
    expected_policy = {
        "new_model_execution": False,
        "new_label_access": False,
        "prediction_row_access": False,
        "aggregate_recomputation": False,
        "aggregate_values_modified": False,
        "thresholds_modified": False,
        "scientific_narrative": False,
    }
    if spec.get("execution_policy") != expected_policy:
        raise AuthorizationError("claim execution policy changed")
    for raw in spec.get("threshold_sources", []):
        path = resolve_workspace_path(raw.get("path"), workspace, "threshold source")
        digest = validate_sha256(raw.get("sha256"), "threshold source hash")
        if not path.is_file() or sha256_file(path) != digest:
            raise IntegrityError(f"preregistered threshold source hash mismatch: {path}")
    freeze_path = resolve_workspace_path(
        authorization.get("aggregate_freeze_path"), workspace, "freeze"
    )
    freeze_sha256 = validate_sha256(authorization.get("aggregate_freeze_sha256"), "freeze hash")
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze_sha256:
        raise IntegrityError("authoritative aggregate freeze hash mismatch")
    freeze = read_json(freeze_path, "authoritative aggregate freeze")
    if (
        freeze.get("schema_version") != AGGREGATE_FREEZE_SCHEMA
        or freeze.get("state") != AGGREGATE_STATE
    ):
        raise IntegrityError("authoritative aggregate package is not frozen")
    if freeze.get("merkle_root") != authorization.get("aggregate_merkle_root"):
        raise IntegrityError("authoritative aggregate Merkle binding changed")
    if any(
        freeze.get(field) != 0
        for field in (
            "checkpoint_access_count",
            "model_load_count",
            "prediction_generation_count",
            "claim_threshold_application_count",
            "claim_decision_count",
        )
    ):
        raise IntegrityError("aggregate v4 freeze records a prohibited operation")
    artifact_records = freeze.get("artifacts")
    if not isinstance(artifact_records, list) or len(artifact_records) != 8:
        raise IntegrityError("aggregate freeze must bind exactly eight result files")
    artifact_by_path: dict[str, dict[str, Any]] = {}
    for raw in artifact_records:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise IntegrityError("invalid aggregate artifact record")
        name = str(raw["path"])
        path = (freeze_path.parent / PurePosixPath(name)).resolve()
        if freeze_path.parent.resolve() not in path.parents:
            raise IntegrityError(f"aggregate artifact escapes freeze root: {name}")
        digest = validate_sha256(raw.get("sha256"), f"aggregate artifact {name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise IntegrityError(f"aggregate artifact hash mismatch: {name}")
        artifact_by_path[name] = dict(raw)
    aggregate_paths: set[Path] = set()
    inputs = spec.get("aggregate_inputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 4
        or inputs != authorization.get("aggregate_inputs")
    ):
        raise AuthorizationError("claim aggregate input bindings changed")
    for raw in inputs:
        path = resolve_workspace_path(raw.get("path"), workspace, "claim aggregate input")
        name = path.name
        if name not in AGGREGATE_FILENAMES or name not in artifact_by_path:
            raise AuthorizationError(f"claim input is not one of four frozen aggregates: {path}")
        digest = validate_sha256(raw.get("sha256"), "claim aggregate input hash")
        if digest != artifact_by_path[name]["sha256"] or sha256_file(path) != digest:
            raise IntegrityError(f"claim aggregate input hash mismatch: {path}")
        aggregate_paths.add(path.resolve())
    output = output_root.resolve()
    if output.exists():
        raise AuthorizationError(f"claim output root already exists: {output}")
    resources = ResourcePolicy(frozenset(aggregate_paths), output, gate)
    result = PreflightResult(
        schema_version=PREFLIGHT_SCHEMA,
        state="PASS",
        authorization_sha256=expected,
        aggregate_freeze_sha256=freeze_sha256,
        aggregate_merkle_root=str(freeze["merkle_root"]),
        decision_spec_sha256=spec_sha256,
        verified_aggregate_files=8,
        decision_input_files=4,
        checkpoint_accesses=0,
        model_loads=0,
        prediction_row_accesses=0,
        sealed_label_accesses=0,
        prediction_generations=0,
        aggregate_recomputations=0,
        threshold_modifications=0,
        enabled_capabilities=gate.enabled,
    )
    return result, gate, resources, spec, authorization


def _find_one(rows: Sequence[Mapping[str, Any]], **criteria: object) -> dict[str, Any]:
    matches = [
        dict(row) for row in rows if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise DecisionError(f"expected one aggregate row for {criteria}, found {len(matches)}")
    return matches[0]


def _decision(
    rule: Mapping[str, Any],
    status: str,
    *,
    metric_name: str,
    observed: object,
    uncertainty_interval: object,
    threshold: object,
    comparison_rule: str,
    rationale: Mapping[str, Any],
) -> dict[str, Any]:
    if status not in DECISION_VOCABULARY:
        raise DecisionError(f"invalid claim status: {status}")
    return {
        "claim_id": rule["claim_id"],
        "claim_type": rule["claim_type"],
        "preregistered_wording": rule["wording"],
        "metric_name": metric_name,
        "observed_value": observed,
        "uncertainty_interval": uncertainty_interval,
        "frozen_threshold": threshold,
        "comparison_rule": comparison_rule,
        "status": status,
        "machine_readable_rationale": dict(rationale),
    }


def _holm_two_probe_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise DecisionError("ORC-006 requires exactly two evidence-step probes per run")
    ordered = sorted(rows, key=lambda row: float(row["p_value"]))
    rejections: list[dict[str, Any]] = []
    stopped = False
    for rank, row in enumerate(ordered):
        threshold = 0.05 / (2 - rank)
        reject = not stopped and float(row["p_value"]) <= threshold
        rejections.append(
            {
                "evidence_step": int(row["evidence_step"]),
                "p_value": float(row["p_value"]),
                "holm_threshold": threshold,
                "rejected": reject,
            }
        )
        if not reject:
            stopped = True
    descriptive = [
        {
            "evidence_step": int(row["evidence_step"]),
            "accuracy": float(row["validation_accuracy"]),
            "accuracy_at_most_0_55": float(row["validation_accuracy"]) <= 0.55,
            "wilson_contains_0_50": float(row["wilson_low"]) <= 0.5 <= float(row["wilson_high"]),
        }
        for row in rows
    ]
    passed = all(not item["rejected"] for item in rejections) and all(
        item["accuracy_at_most_0_55"] and item["wilson_contains_0_50"] for item in descriptive
    )
    return {"passed": passed, "holm": rejections, "descriptive": descriptive}


def evaluate_claims(
    *,
    baseline: Mapping[str, Any],
    effects: Mapping[str, Any],
    interventions: Mapping[str, Any],
    probes: Mapping[str, Any],
    spec: Mapping[str, Any],
    gate: CapabilityGate,
) -> list[dict[str, Any]]:
    gate.require(Capability.CLAIM_THRESHOLD_APPLICATION)
    if baseline.get("schema_version") != "opm-sealed-baseline-metrics-v1":
        raise IntegrityError("baseline aggregate schema mismatch")
    if effects.get("schema_version") != "opm-sealed-primary-effects-v1":
        raise IntegrityError("primary-effect aggregate schema mismatch")
    if interventions.get("schema_version") != "opm-sealed-intervention-metrics-v1":
        raise IntegrityError("intervention aggregate schema mismatch")
    if probes.get("schema_version") != "opm-sealed-probe-metrics-v1":
        raise IntegrityError("probe aggregate schema mismatch")
    if any(
        value.get("thresholds_applied") is not False
        for value in (baseline, effects, interventions, probes)
    ):
        raise IntegrityError("an aggregate input already records threshold application")
    rules = {str(rule["claim_id"]): dict(rule) for rule in spec["rules"]}
    decisions: list[dict[str, Any]] = []

    bootstrap = effects["paired_two_level_bootstrap"]
    interval = [float(value) for value in bootstrap["delta_generalist_percentile_95_interval"]]
    interpolation = {
        condition: _find_one(
            baseline["pooled_split_accuracy"],
            condition=condition,
            split_name="test-interpolation",
        )["accuracy"]
        for condition in ("OPM_SHARED", "DOMAIN_GENERALIST", "PROC_UNTIED")
    }
    noninferiority = {
        control: float(interpolation["OPM_SHARED"]) >= float(interpolation[control]) - 0.01
        for control in ("DOMAIN_GENERALIST", "PROC_UNTIED")
    }
    if interval[1] <= 0.02:
        h1_status = "NOT_SUPPORTED"
    elif interval[0] > 0.02 and all(noninferiority.values()):
        h1_status = "SUPPORTED"
    else:
        h1_status = "INCONCLUSIVE"
    decisions.append(
        _decision(
            rules["H1-PRIMARY"],
            h1_status,
            metric_name="delta_generalist_and_interpolation_noninferiority",
            observed={
                "delta_generalist": float(bootstrap["delta_generalist"]),
                "opm_interpolation_accuracy": float(interpolation["OPM_SHARED"]),
                "domain_generalist_interpolation_accuracy": float(
                    interpolation["DOMAIN_GENERALIST"]
                ),
                "proc_untied_interpolation_accuracy": float(interpolation["PROC_UNTIED"]),
                "noninferiority_checks": noninferiority,
                "delta_untied_descriptive": float(bootstrap["delta_untied"]),
            },
            uncertainty_interval={"delta_generalist_percentile_95": interval},
            threshold={"delta_generalist_lower_bound_gt": 0.02, "interpolation_margin": 0.01},
            comparison_rule=rules["H1-PRIMARY"]["rule"],
            rationale={
                "lower_bound_condition": interval[0] > 0.02,
                "upper_bound_not_supported_condition": interval[1] <= 0.02,
                "interpolation_noninferiority": noninferiority,
            },
        )
    )

    decisions.append(
        _decision(
            rules["STA-004-SYMBOLIC-ORACLE"],
            "NOT_EVALUABLE",
            metric_name="symbolic_oracle_accuracy",
            observed=None,
            uncertainty_interval=None,
            threshold={"equals": 1.0},
            comparison_rule=rules["STA-004-SYMBOLIC-ORACLE"]["rule"],
            rationale={"required_input_present": False, "reason": "not in aggregate v4 inputs"},
        )
    )
    recombination = _find_one(
        baseline["pooled_split_accuracy"],
        condition="OPM_SHARED",
        split_name="test-recombination",
    )
    recombination_status = (
        "SUPPORTED" if float(recombination["accuracy"]) > 0.80 else "NOT_SUPPORTED"
    )
    decisions.append(
        _decision(
            rules["STA-004-RECOMBINATION"],
            recombination_status,
            metric_name="opm_shared_recombination_accuracy",
            observed=float(recombination["accuracy"]),
            uncertainty_interval=None,
            threshold={"operator": ">", "value": 0.8},
            comparison_rule=rules["STA-004-RECOMBINATION"]["rule"],
            rationale={"comparison_result": recombination_status == "SUPPORTED"},
        )
    )

    details = [
        row
        for row in interventions["by_component_domain_operation"]
        if row.get("condition") == "OPM_SHARED"
        and row.get("family") == "ablation"
        and row.get("artifact_split") == "test-interpolation"
        and row.get("variant") != "sentinel"
    ]
    operation_results: list[dict[str, Any]] = []
    unrelated_results: list[dict[str, Any]] = []
    for operation in ("LOOKUP", "REVERSE", "CHAIN", "LIFT"):
        variant = f"operation:{operation}"
        rows = [row for row in details if row.get("variant") == variant]
        selected = [row for row in rows if row.get("target_operation") == operation]
        unrelated = [row for row in rows if row.get("target_operation") != operation]
        selected_drops = [
            {
                "domain": row["target_domain"],
                "drop": -float(row["accuracy_change"]),
                "meets_0_20": -float(row["accuracy_change"]) >= 0.20,
            }
            for row in selected
        ]
        qualifying = sum(item["meets_0_20"] for item in selected_drops)
        operation_results.append(
            {
                "operation": operation,
                "trained_domain_drops": selected_drops,
                "qualifying_domain_count": qualifying,
                "required_domain_count": 2,
                "passed": qualifying >= 2,
            }
        )
        if not unrelated:
            raise DecisionError(f"missing unrelated-operation aggregates for {operation}")
        mean_drop = sum(-float(row["accuracy_change"]) for row in unrelated) / len(unrelated)
        unrelated_results.append(
            {
                "operation": operation,
                "aggregate_row_count": len(unrelated),
                "mean_unrelated_drop": mean_drop,
                "passed": mean_drop < 0.05,
            }
        )
    ablation_status = (
        "SUPPORTED" if all(item["passed"] for item in operation_results) else "NOT_SUPPORTED"
    )
    decisions.append(
        _decision(
            rules["STA-004-ACTIVE-ABLATION"],
            ablation_status,
            metric_name="selected_operation_accuracy_drop_by_trained_domain",
            observed=operation_results,
            uncertainty_interval=None,
            threshold={"minimum_drop": 0.2, "minimum_trained_domains": 2},
            comparison_rule=rules["STA-004-ACTIVE-ABLATION"]["rule"],
            rationale={"all_active_operations_passed": ablation_status == "SUPPORTED"},
        )
    )
    unrelated_status = (
        "SUPPORTED" if all(item["passed"] for item in unrelated_results) else "NOT_SUPPORTED"
    )
    decisions.append(
        _decision(
            rules["STA-004-UNRELATED-PRESERVATION"],
            unrelated_status,
            metric_name="mean_unrelated_operation_accuracy_drop",
            observed=unrelated_results,
            uncertainty_interval=None,
            threshold={"operator": "<", "value": 0.05},
            comparison_rule=rules["STA-004-UNRELATED-PRESERVATION"]["rule"],
            rationale={"all_active_operations_passed": unrelated_status == "SUPPORTED"},
        )
    )

    sentinel_values = [
        float(row["max_absolute_logit_change"])
        for row in interventions["sentinel_max_absolute_logit_change"]
    ]
    if len(sentinel_values) != 80:
        raise DecisionError("sentinel criterion requires exactly 80 frozen summaries")
    sentinel_max = max(sentinel_values)
    sentinel_status = "SUPPORTED" if sentinel_max < 1e-7 else "NOT_SUPPORTED"
    decisions.append(
        _decision(
            rules["STA-004-SENTINEL"],
            sentinel_status,
            metric_name="sentinel_max_absolute_logit_change",
            observed={"maximum": sentinel_max, "summary_count": len(sentinel_values)},
            uncertainty_interval=None,
            threshold={"operator": "<", "value": 1e-7},
            comparison_rule=rules["STA-004-SENTINEL"]["rule"],
            rationale={
                "comparison_result": sentinel_status == "SUPPORTED",
                "interpretation_scope": "numerical-null criterion only; not standalone causal irrelevance",
                "active_ablation_positive_control_status": ablation_status,
            },
        )
    )

    adapter_rows = [
        row
        for row in interventions["summary"]
        if row.get("condition") == "OPM_SHARED" and row.get("family") == "adapter-only"
    ]
    if len(adapter_rows) != 4:
        raise DecisionError("adapter-only criterion requires four split summaries")
    adapter_observed = {str(row["artifact_split"]): float(row["accuracy"]) for row in adapter_rows}
    adapter_max = max(adapter_observed.values())
    adapter_status = "SUPPORTED" if adapter_max <= 0.60 else "NOT_SUPPORTED"
    decisions.append(
        _decision(
            rules["STA-004-ADAPTER-ONLY"],
            adapter_status,
            metric_name="maximum_opm_shared_adapter_only_accuracy",
            observed={"by_split": adapter_observed, "maximum": adapter_max},
            uncertainty_interval=None,
            threshold={"operator": "<=", "value": 0.6},
            comparison_rule=rules["STA-004-ADAPTER-ONLY"]["rule"],
            rationale={"all_split_accuracies_at_most_threshold": adapter_status == "SUPPORTED"},
        )
    )

    decisions.append(
        _decision(
            rules["STA-004-RAW-ORACLE-PROBES"],
            "NOT_EVALUABLE",
            metric_name="orc_005_raw_oracle_probe_family",
            observed=None,
            uncertainty_interval=None,
            threshold={
                "accuracy_at_most": 0.525,
                "wilson_contains": 0.5,
                "holm_familywise_alpha": 0.05,
            },
            comparison_rule=rules["STA-004-RAW-ORACLE-PROBES"]["rule"],
            rationale={"required_input_present": False, "reason": "not in aggregate v4 inputs"},
        )
    )

    probe_rows = [row for row in probes["results"] if row.get("condition") == "OPM_SHARED"]
    run_results: list[dict[str, Any]] = []
    for seed in (1101, 2202, 3303, 4404, 5505):
        rows = [row for row in probe_rows if int(row["training_seed"]) == seed]
        result = _holm_two_probe_status(rows)
        run_results.append({"condition": "OPM_SHARED", "training_seed": seed, **result})
    neural_status = "SUPPORTED" if all(item["passed"] for item in run_results) else "NOT_SUPPORTED"
    decisions.append(
        _decision(
            rules["STA-004-NEURAL-PROBES"],
            neural_status,
            metric_name="orc_006_opm_shared_run_families",
            observed=run_results,
            uncertainty_interval=None,
            threshold={
                "accuracy_at_most": 0.55,
                "wilson_contains": 0.5,
                "holm_family_size": 2,
                "familywise_alpha": 0.05,
            },
            comparison_rule=rules["STA-004-NEURAL-PROBES"]["rule"],
            rationale={
                "all_opm_shared_runs_passed": neural_status == "SUPPORTED",
                "failed_runs": [
                    item["training_seed"] for item in run_results if not item["passed"]
                ],
            },
        )
    )

    mechanism_component_ids = (
        "STA-004-SYMBOLIC-ORACLE",
        "STA-004-RECOMBINATION",
        "STA-004-ACTIVE-ABLATION",
        "STA-004-UNRELATED-PRESERVATION",
        "STA-004-SENTINEL",
        "STA-004-ADAPTER-ONLY",
        "STA-004-RAW-ORACLE-PROBES",
        "STA-004-NEURAL-PROBES",
    )
    component_statuses = {
        claim_id: next(row["status"] for row in decisions if row["claim_id"] == claim_id)
        for claim_id in mechanism_component_ids
    }
    if all(status == "SUPPORTED" for status in component_statuses.values()):
        mechanism_status = "SUPPORTED"
    elif any(status == "NOT_SUPPORTED" for status in component_statuses.values()):
        mechanism_status = "NOT_SUPPORTED"
    else:
        mechanism_status = "NOT_EVALUABLE"
    decisions.append(
        _decision(
            rules["MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE"],
            mechanism_status,
            metric_name="sta_004_all_of",
            observed=component_statuses,
            uncertainty_interval=None,
            threshold={"operator": "all", "required_status": "SUPPORTED"},
            comparison_rule=rules["MECHANISM-CAUSALLY-REUSABLE-PRIMITIVE"]["rule"],
            rationale={
                "all_components_supported": mechanism_status == "SUPPORTED",
                "not_supported_components": [
                    claim_id
                    for claim_id, status in component_statuses.items()
                    if status == "NOT_SUPPORTED"
                ],
                "not_evaluable_components": [
                    claim_id
                    for claim_id, status in component_statuses.items()
                    if status == "NOT_EVALUABLE"
                ],
            },
        )
    )

    interchange = _find_one(
        interventions["summary"],
        condition="OPM_SHARED",
        artifact_split="renderer-pairs",
        family="interchange",
        variant="cross-domain-renderer-state-swap",
    )
    decisions.append(
        _decision(
            rules["MET-005-INTERCHANGE-MECHANISM"],
            "NOT_EVALUABLE",
            metric_name="cross_domain_interchange_accuracy_and_drop",
            observed={
                "accuracy": float(interchange["accuracy"]),
                "baseline_accuracy": float(interchange["baseline_accuracy"]),
                "accuracy_change": float(interchange["accuracy_change"]),
            },
            uncertainty_interval=None,
            threshold=None,
            comparison_rule=rules["MET-005-INTERCHANGE-MECHANISM"]["rule"],
            rationale={"preregistered_decision_threshold_present": False},
        )
    )
    surface = _find_one(
        interventions["summary"],
        condition="OPM_SHARED",
        artifact_split="test-renderer",
        family="surface-reversal",
        variant="renderer-v2",
    )
    decisions.append(
        _decision(
            rules["MET-007-SURFACE-INVARIANCE"],
            "NOT_EVALUABLE",
            metric_name="surface_reversal_accuracy_drop",
            observed={
                "accuracy": float(surface["accuracy"]),
                "baseline_accuracy": float(surface["baseline_accuracy"]),
                "accuracy_change": float(surface["accuracy_change"]),
            },
            uncertainty_interval=None,
            threshold=None,
            comparison_rule=rules["MET-007-SURFACE-INVARIANCE"]["rule"],
            rationale={"preregistered_decision_threshold_present": False},
        )
    )
    for claim_id, metric, reason in (
        (
            "H2-THEORETICAL-RESOURCE-ADVANTAGE",
            "quality_matched_resource_advantage",
            "STA-005 requires a later preregistered quality-matched threshold",
        ),
        (
            "H3-REAL-HARDWARE-ADVANTAGE",
            "hardware_performance_advantage",
            "STA-006 declares H3 out of scope",
        ),
        (
            "H4-CORRECT-BOUNDARY-OF-REUSE",
            "bounded_sharing_outcome",
            "MET-008 and STA-007 declare H4 out of scope",
        ),
    ):
        decisions.append(
            _decision(
                rules[claim_id],
                "NOT_EVALUABLE",
                metric_name=metric,
                observed=None,
                uncertainty_interval=None,
                threshold=None,
                comparison_rule=rules[claim_id]["rule"],
                rationale={"reason": reason},
            )
        )
    if {row["claim_id"] for row in decisions} != set(rules):
        raise DecisionError("decision coverage does not exactly match the locked claim rule set")
    return sorted(decisions, key=lambda row: str(row["claim_id"]))


def finalize_decision_records(
    *,
    decisions: Sequence[Mapping[str, Any]],
    authorization_sha256: str,
    authorization: Mapping[str, Any],
    audit_sha256: str,
    execution_timestamp: str,
) -> list[dict[str, Any]]:
    input_hashes = {
        Path(str(row["path"])).name: row["sha256"] for row in authorization["aggregate_inputs"]
    }
    records = []
    for decision in decisions:
        records.append(
            {
                **dict(decision),
                "authoritative_input_hashes": input_hashes,
                "aggregate_freeze_sha256": authorization["aggregate_freeze_sha256"],
                "authorization_identifier": authorization_sha256,
                "code_commit": authorization["evaluator_source_commit"],
                "execution_timestamp_utc": execution_timestamp,
                "audit_log_sha256": audit_sha256,
            }
        )
    return records


def reconcile_decisions(
    *,
    decision_package: Mapping[str, Any],
    spec: Mapping[str, Any],
    authorization_sha256: str,
    audit_sha256: str,
    decision_path: Path,
) -> dict[str, Any]:
    decisions = decision_package.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(spec["rules"]):
        raise IntegrityError("claim decision count does not match locked spec")
    expected_ids = {str(row["claim_id"]) for row in spec["rules"]}
    observed_ids = {str(row.get("claim_id")) for row in decisions}
    if observed_ids != expected_ids or len(observed_ids) != len(decisions):
        raise IntegrityError("claim decision identity coverage mismatch")
    if any(row.get("status") not in DECISION_VOCABULARY for row in decisions):
        raise IntegrityError("claim decision uses an unauthorized status")
    if any(row.get("authorization_identifier") != authorization_sha256 for row in decisions):
        raise IntegrityError("claim decision authorization binding mismatch")
    if any(row.get("audit_log_sha256") != audit_sha256 for row in decisions):
        raise IntegrityError("claim decision audit binding mismatch")
    status_counts = {
        status: sum(row["status"] == status for row in decisions) for status in DECISION_VOCABULARY
    }
    return {
        "schema_version": RECONCILIATION_SCHEMA,
        "state": "PASS",
        "authorization_sha256": authorization_sha256,
        "decision_record_sha256": sha256_file(decision_path),
        "audit_log_sha256": audit_sha256,
        "claim_count": len(decisions),
        "claim_ids_exact": True,
        "decision_vocabulary_exact": True,
        "authoritative_aggregate_version": "v4",
        "non_authoritative_versions_excluded": ["v1", "v2", "v3"],
        "status_counts": status_counts,
        "new_model_execution_performed": False,
        "new_label_access_performed": False,
        "prediction_row_access_performed": False,
        "aggregate_recomputation_performed": False,
        "aggregate_values_modified": False,
        "thresholds_modified": False,
        "scientific_narrative_generated": False,
    }


def freeze_decision_package(
    *,
    output_root: Path,
    authorization_sha256: str,
    decision_spec_sha256: str,
    files: Sequence[Path],
    gate: CapabilityGate,
) -> tuple[Path, str, dict[str, Any]]:
    gate.require(Capability.DECISION_PACKAGE_FREEZING)
    records = []
    for path in sorted((item.resolve() for item in files), key=lambda item: item.as_posix()):
        if output_root.resolve() not in path.parents or not path.is_file():
            raise IntegrityError(f"claim result missing or outside output root: {path}")
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
        "decision_spec_sha256": decision_spec_sha256,
        "artifact_count": len(records),
        "artifacts": records,
        "merkle_root": merkle_root(records),
        "authoritative_aggregate_version": "v4",
        "non_authoritative_versions_excluded": ["v1", "v2", "v3"],
        "new_model_execution_count": 0,
        "new_label_access_count": 0,
        "prediction_row_access_count": 0,
        "prediction_generation_count": 0,
        "aggregate_recomputation_count": 0,
        "threshold_modification_count": 0,
        "scientific_narrative_generation_count": 0,
    }
    path = output_root / FREEZE_FILENAME
    digest = atomic_write_json(path, freeze)
    return path, digest, freeze


def environment_record() -> dict[str, Any]:
    return {
        "schema_version": "opm-claim-decision-environment-v1",
        "python": platform.python_version(),
        "model_runtime_imported": False,
        "new_model_execution_count": 0,
        "new_label_access_count": 0,
        "prediction_row_access_count": 0,
        "prediction_generation_count": 0,
        "aggregate_recomputation_count": 0,
        "threshold_modification_count": 0,
        "scientific_narrative_generation_count": 0,
    }
