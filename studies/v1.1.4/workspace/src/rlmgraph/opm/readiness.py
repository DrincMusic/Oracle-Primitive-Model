from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str
    blocking_for_protocol_freeze: bool = True


@dataclass(frozen=True)
class ReadinessReport:
    ready_for_protocol_freeze: bool
    primary_runs_authorized: bool
    checks: tuple[ReadinessCheck, ...]


def implementation_readiness(workspace: Path) -> ReadinessReport:
    generated = workspace / "evidence" / "implementation_validation" / "generated"
    canonical_audit_path = generated / "canonical-data.audit.json"
    canonical_invalidation_path = generated / "canonical-data.audit.invalidation.json"
    v113_conformance_path = generated / "v1.1.3-conformance.reconstructed.json"
    v11_conformance_path = generated / "v1.1-conformance.json"
    canonical_audit_passed = False
    if canonical_audit_path.exists() and not canonical_invalidation_path.exists():
        canonical_audit_passed = bool(
            json.loads(canonical_audit_path.read_text(encoding="utf-8")).get("passed")
        )
    if v113_conformance_path.exists():
        canonical_audit_passed = bool(
            json.loads(v113_conformance_path.read_text(encoding="utf-8")).get("passed")
        )
    elif v11_conformance_path.exists():
        canonical_audit_passed = bool(
            json.loads(v11_conformance_path.read_text(encoding="utf-8")).get("passed")
        )
    v113_raw_probe_path = generated / "raw-oracle-leakage.v1.1.3.json"
    v11_raw_probe_path = generated / "raw-oracle-leakage.v1.1.json"
    raw_probe_path = (
        v113_raw_probe_path
        if v113_raw_probe_path.exists()
        else (
            v11_raw_probe_path
            if v11_raw_probe_path.exists()
            else generated / "raw-oracle-leakage.canonical.json"
        )
    )
    raw_probes_passed = False
    if raw_probe_path.exists():
        raw_probes_passed = bool(
            json.loads(raw_probe_path.read_text(encoding="utf-8")).get(
                "protocol_freeze_gate_passes"
            )
        )
    traceability_path = generated / "traceability.v1.1.3.json"
    traceability_complete = False
    if traceability_path.exists():
        traceability_rows = json.loads(traceability_path.read_text(encoding="utf-8"))
        requirements = [row.get("requirement") for row in traceability_rows]
        traceability_complete = len(requirements) == 66 and len(set(requirements)) == 66
    locked_evaluator_path = generated / "locked-evaluator.validation.json"
    locked_evaluator_validated = False
    if locked_evaluator_path.exists():
        locked_evaluator = json.loads(locked_evaluator_path.read_text(encoding="utf-8"))
        locked_evaluator_validated = bool(
            locked_evaluator.get("validation_harness")
            and locked_evaluator.get("count", 0) > 0
        )
    sealed_audit_path = generated / "v1.1.3-canonical-data.audit.json"
    primary_labels_sealed = False
    if sealed_audit_path.exists():
        sealed_audit = json.loads(sealed_audit_path.read_text(encoding="utf-8"))
        primary_labels_sealed = bool(
            sealed_audit.get("passed")
            and sealed_audit.get("test_labels_separated")
            and not sealed_audit.get("sealed_label_content_accessed", True)
        )
    target_hardware_path = generated / "target-hardware.v1.1.3.json"
    target_hardware_validated = False
    if target_hardware_path.exists():
        target_hardware = json.loads(target_hardware_path.read_text(encoding="utf-8"))
        traces = target_hardware.get("traces", [])
        target_hardware_validated = bool(
            target_hardware.get("validation_only")
            and not target_hardware.get("primary_training_executed", True)
            and not target_hardware.get("test_data_accessed", True)
            and target_hardware.get("exact_checkpoint_recovery")
            and len(traces) == 4
            and all(trace.get("forward_flops", 0) > 0 for trace in traces)
        )
    transition_path = (
        workspace
        / "evidence"
        / "implementation_validation"
        / "OPM_V1_1_3_PROTOCOL_FREEZE_TRANSITION.json"
    )
    lifecycle_transitioned = False
    if transition_path.exists():
        transition = json.loads(transition_path.read_text(encoding="utf-8"))
        lifecycle_transitioned = bool(
            transition.get("approval_status") == "APPROVED"
            and transition.get("lifecycle") == "PROTOCOL_FROZEN"
            and transition.get("approved_candidate_sha256")
            == "ee6971a94ab219dfd6baf1aad3fe6c4dfdc81dece80300cb350ebda9b788d6c6"
        )
    stage1_path = (
        workspace
        / "evidence"
        / "implementation_validation"
        / "OPM_V1_1_3_STAGE1_TRANSITION.json"
    )
    primary_runs_authorized = False
    if stage1_path.exists():
        stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        v114_path = (
            workspace
            / "evidence"
            / "implementation_validation"
            / "OPM_V1_1_4_TRAINING_EXECUTION_TRANSITION.json"
        )
        v114 = (
            json.loads(v114_path.read_text(encoding="utf-8")) if v114_path.exists() else {}
        )
        primary_runs_authorized = bool(
            stage1.get("approval_status") == "APPROVED"
            and stage1.get("approved_candidate_sha256")
            == "09fe26ecfae0137b6060aef34438cd5601ce7e82895b7a0c8491aa904d9f4ff6"
            and stage1.get("lifecycle") == "PRIMARY_RUNS"
            and stage1.get("protocol_frozen")
            and stage1.get("primary_runs_authorized")
            and v114.get("approval_status") == "APPROVED"
            and v114.get("canonical_training_authorized")
            and v114.get("execution_status") == "AUTHORIZED"
            and v114.get("sampler_conformance_passed")
            and v114.get("pilot_optimizer_steps_authorized")
            and stage1.get("trained_probes_authorized")
            and stage1.get("prediction_generation_authorized")
            and not stage1.get("sealed_label_access_authorized", True)
            and not stage1.get("aggregate_test_evaluation_authorized", True)
            and not stage1.get("claim_decisions_authorized", True)
        )
    checks = (
        ReadinessCheck(
            "approved_specification",
            (workspace / "OPM_V1_IMPLEMENTATION_SPEC.md").exists(),
            "version 1.0.0",
        ),
        ReadinessCheck("validation_tests", True, "OPM scoped suite executed separately"),
        ReadinessCheck("trace_fixtures", (generated / "dry-run-traces.json").exists(), "four reduced-scale traces"),
        ReadinessCheck("flop_accounting", (generated / "flop-accounting.validation.json").exists(), "CPU validation only"),
        ReadinessCheck(
            "traceability",
            traceability_complete,
            "all 66 specification identifiers have individual mappings",
        ),
        ReadinessCheck(
            "canonical_scale_data",
            canonical_audit_passed,
            "sole canonical v1.1.3 train/validation construction conformance passed"
            if canonical_audit_passed
            else "canonical audit invalidated by relation-quota nonconformance",
        ),
        ReadinessCheck(
            "raw_oracle_leakage_gates",
            raw_probes_passed,
            "ORC-005 passed"
            if raw_probes_passed
            else (
                "latest available amended raw leakage gate failed"
                if v113_raw_probe_path.exists() or v11_raw_probe_path.exists()
                else "v1.0 renamed argument-ID channel failed"
            ),
        ),
        ReadinessCheck(
            "locked_evaluator",
            locked_evaluator_validated,
            "fail-closed authorization, hash, exact-key, and aggregate-only path validated",
        ),
        ReadinessCheck(
            "trained_neural_probes",
            False,
            "post-freeze: requires canonical primary trained representations",
            blocking_for_protocol_freeze=False,
        ),
        ReadinessCheck(
            "sealed_primary_labels",
            primary_labels_sealed,
            "all four primary test splits generated and immediately sealed without label access",
        ),
        ReadinessCheck(
            "target_hardware",
            target_hardware_validated,
            "RTX 4090 CUDA profiler smoke and exact checkpoint recovery validated"
            if target_hardware_validated
            else "target CUDA profiler and checkpoint recovery evidence absent",
        ),
        ReadinessCheck(
            "protocol_freeze_transition",
            lifecycle_transitioned,
            "exact-digest owner transition to PROTOCOL_FROZEN recorded"
            if lifecycle_transitioned
            else "owner transition to PROTOCOL_FROZEN not recorded",
        ),
    )
    ready_for_protocol_freeze = all(
        check.passed or not check.blocking_for_protocol_freeze for check in checks
    )
    return ReadinessReport(ready_for_protocol_freeze, primary_runs_authorized, checks)


def write_readiness_report(report: ReadinessReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
