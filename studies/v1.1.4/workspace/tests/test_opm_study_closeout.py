from pathlib import Path

import pytest

from scripts import opm_study_closeout as closeout

WORKSPACE = Path(__file__).resolve().parents[1]


def _authorization(report_inputs: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": closeout.AUTHORIZATION_SCHEMA,
        "authorized_operations": list(closeout.AUTHORIZED_OPERATIONS),
        "prohibited_operations": list(closeout.PROHIBITED_OPERATIONS),
        "report_inputs": report_inputs or [],
        "claim_decision_freeze": {
            "path": closeout.CLAIM_FREEZE_PATH.as_posix(),
            "sha256": closeout.CLAIM_FREEZE_SHA256,
            "merkle_root": closeout.CLAIM_MERKLE_ROOT,
        },
    }


def test_capability_gate_allows_reporting_and_denies_scientific_mutation() -> None:
    gate = closeout.CapabilityGate.from_authorization(_authorization())
    gate.require(closeout.Capability.CLOSEOUT_REPORT_GENERATION)
    for capability in (
        closeout.Capability.CHECKPOINT_ACCESS,
        closeout.Capability.MODEL_LOADING,
        closeout.Capability.PREDICTION_ROW_ACCESS,
        closeout.Capability.SEALED_LABEL_ACCESS,
        closeout.Capability.PREDICTION_GENERATION,
        closeout.Capability.AGGREGATE_RECOMPUTATION,
        closeout.Capability.THRESHOLD_APPLICATION,
        closeout.Capability.THRESHOLD_MODIFICATION,
        closeout.Capability.CLAIM_DECISION_MODIFICATION,
        closeout.Capability.TRAINING,
        closeout.Capability.NEW_SCIENTIFIC_INFERENCE,
    ):
        with pytest.raises(closeout.AuthorizationError, match=capability.value):
            gate.require(capability)


def test_source_and_path_audits_fail_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.py"
    allowed.write_text("import json\n", encoding="utf-8")
    closeout.assert_report_source_is_read_only(allowed)
    denied = tmp_path / "denied.py"
    denied.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(closeout.AuthorizationError, match="prohibited runtime"):
        closeout.assert_report_source_is_read_only(denied)
    for path in (
        tmp_path / "checkpoints" / "step.pt",
        tmp_path / "sealed-labels" / "test.jsonl",
        tmp_path / "predictions" / "rows.jsonl",
    ):
        with pytest.raises(closeout.AuthorizationError, match="prohibited"):
            closeout.assert_report_safe_path(path)


def test_locked_report_spec_binds_authoritative_claim_freeze_and_safe_inputs() -> None:
    spec = closeout.build_report_spec(
        workspace=WORKSPACE,
        expected_claim_freeze_sha256=closeout.CLAIM_FREEZE_SHA256,
    )
    assert spec["authoritative_claim_decision_freeze"] == {
        "path": closeout.CLAIM_FREEZE_PATH.as_posix(),
        "sha256": closeout.CLAIM_FREEZE_SHA256,
        "merkle_root": closeout.CLAIM_MERKLE_ROOT,
        "authorization_sha256": "1c9afdcaf96847be7a57e3f75fcd32f22537bc25ab2fe76e15ce8f01670685c9",
    }
    assert len(spec["report_inputs"]) == len(closeout.REPORT_CONTEXT_PATHS)
    for binding in spec["report_inputs"]:
        closeout.assert_report_safe_path(WORKSPACE / binding["path"])
    assert all(
        value is False
        for key, value in spec["execution_policy"].items()
        if key != "report_generation"
    )
    assert spec["execution_policy"]["report_generation"] is True


def test_report_reproduces_all_frozen_decisions_and_required_sections(tmp_path: Path) -> None:
    spec = closeout.build_report_spec(
        workspace=WORKSPACE,
        expected_claim_freeze_sha256=closeout.CLAIM_FREEZE_SHA256,
    )
    authorization = _authorization(spec["report_inputs"])
    gate = closeout.CapabilityGate.from_authorization(authorization)
    resources = closeout.ResourcePolicy(
        input_paths=frozenset((WORKSPACE / row["path"]).resolve() for row in spec["report_inputs"]),
        output_root=tmp_path.resolve(),
        gate=gate,
    )
    data = closeout.build_report_data(
        workspace=WORKSPACE,
        resources=resources,
        authorization_sha256="a" * 64,
        authorization=authorization,
        generated_at_utc="2026-08-27T00:00:00+00:00",
    )
    assert data["status_counts"] == {
        "SUPPORTED": 6,
        "NOT_SUPPORTED": 2,
        "INCONCLUSIVE": 0,
        "NOT_EVALUABLE": 7,
    }
    assert data["primary_result"]["status"] == "SUPPORTED"
    assert data["composite_mechanism_result"]["status"] == "NOT_SUPPORTED"
    assert len(data["claim_decisions"]) == 15
    report = closeout.render_report_markdown(data, authorization)
    report_path = tmp_path / "report.md"
    data_path = tmp_path / "report-data.json"
    report_path.write_text(report, encoding="utf-8")
    closeout.atomic_write_json(data_path, data)
    reconciliation = closeout.reconcile_report(
        report_data=data,
        report_markdown=report,
        report_path=report_path,
        report_data_path=data_path,
        authorization_sha256="a" * 64,
        audit_sha256="b" * 64,
    )
    assert reconciliation["state"] == "PASS"
    assert reconciliation["claim_inventory_exact"] is True
    assert reconciliation["claim_statuses_unchanged"] is True
    assert "## Developmental pilot and validation results" in report
    assert "## Authoritative aggregate-v4 results" in report
    assert "## Frozen claim decisions" in report
