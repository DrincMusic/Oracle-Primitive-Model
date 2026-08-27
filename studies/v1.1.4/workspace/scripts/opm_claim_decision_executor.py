from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import opm_claim_decision as claims


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def authorize(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    path, digest, spec_path, spec_digest = claims.create_authorization(
        workspace=workspace,
        authorization_directory=Path(args.authorization_directory).resolve(),
        aggregate_freeze_path=Path(args.aggregate_freeze).resolve(),
        aggregate_freeze_sha256=args.aggregate_freeze_sha256,
        threshold_sources=[
            workspace / "OPM_V1_IMPLEMENTATION_SPEC.md",
            workspace / "ORACLE_PRIMITIVE_MODEL.md",
        ],
        evaluator_sources=[
            workspace / "scripts/opm_claim_decision.py",
            workspace / "scripts/opm_claim_decision_executor.py",
        ],
        user_request=args.user_request,
    )
    _emit(
        "claim_authorization_recorded",
        authorization_path=str(path),
        authorization_sha256=digest,
        decision_spec_path=str(spec_path),
        decision_spec_sha256=spec_digest,
        sealed_label_access_count=0,
        prediction_row_access_count=0,
    )
    return 0


def preflight(args: argparse.Namespace) -> int:
    result, _gate, _resources, _spec, _authorization = claims.preflight_claim_decision(
        workspace=Path(args.workspace).resolve(),
        authorization_path=Path(args.authorization).resolve(),
        expected_authorization_sha256=args.authorization_sha256,
        output_root=Path(args.output_root).resolve(),
    )
    _emit("claim_preflight_passed", **asdict(result))
    return 0


def execute(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output_root = Path(args.output_root).resolve()
    preflight_result, gate, resources, spec, authorization = claims.preflight_claim_decision(
        workspace=workspace,
        authorization_path=Path(args.authorization).resolve(),
        expected_authorization_sha256=args.authorization_sha256,
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    preflight_path = output_root / "preflight.json"
    claims.atomic_write_json(preflight_path, asdict(preflight_result))
    audit_path = output_root / "claim-decision.audit.jsonl"
    claims.append_jsonl(
        audit_path,
        {
            "timestamp_utc": claims.utc_now(),
            "event": "preflight_passed",
            "authorization_sha256": args.authorization_sha256,
            "sealed_label_access_count": 0,
            "prediction_row_access_count": 0,
        },
    )
    aggregates: dict[str, dict[str, Any]] = {}
    for binding in spec["aggregate_inputs"]:
        path = claims.resolve_workspace_path(binding["path"], workspace, "claim input")
        resources.require_aggregate_read(path)
        aggregates[path.name] = claims.read_json(path, path.name)
        claims.append_jsonl(
            audit_path,
            {
                "timestamp_utc": claims.utc_now(),
                "event": "frozen_aggregate_read",
                "path": binding["path"],
                "sha256": binding["sha256"],
            },
        )
    decisions = claims.evaluate_claims(
        baseline=aggregates["baseline-metrics.json"],
        effects=aggregates["primary-effects.json"],
        interventions=aggregates["intervention-metrics.json"],
        probes=aggregates["probe-metrics.json"],
        spec=spec,
        gate=gate,
    )
    claims.append_jsonl(
        audit_path,
        {
            "timestamp_utc": claims.utc_now(),
            "event": "locked_claim_rules_applied",
            "claim_count": len(decisions),
            "decision_vocabulary": list(claims.DECISION_VOCABULARY),
            "aggregate_recomputation_count": 0,
            "threshold_modification_count": 0,
        },
    )
    audit_sha256 = claims.sha256_file(audit_path)
    execution_timestamp = claims.utc_now()
    records = claims.finalize_decision_records(
        decisions=decisions,
        authorization_sha256=args.authorization_sha256,
        authorization=authorization,
        audit_sha256=audit_sha256,
        execution_timestamp=execution_timestamp,
    )
    decision_package = {
        "schema_version": claims.DECISION_RECORD_SCHEMA,
        "protocol_version": claims.PROTOCOL_VERSION,
        "state": "CLAIM_DECISIONS_GENERATED_PENDING_RECONCILIATION",
        "authoritative_aggregate_version": "v4",
        "non_authoritative_versions_excluded": ["v1", "v2", "v3"],
        "authorization_sha256": args.authorization_sha256,
        "decision_spec_sha256": authorization["decision_spec_sha256"],
        "decision_vocabulary": list(claims.DECISION_VOCABULARY),
        "decisions": records,
        "new_model_execution_performed": False,
        "new_label_access_performed": False,
        "prediction_row_access_performed": False,
        "aggregate_recomputation_performed": False,
        "aggregate_values_modified": False,
        "thresholds_modified": False,
        "scientific_narrative_generated": False,
    }
    decision_path = output_root / "claim-decisions.json"
    claims.atomic_write_json(decision_path, decision_package)
    gate.require(claims.Capability.DECISION_RECONCILIATION)
    reconciliation = claims.reconcile_decisions(
        decision_package=decision_package,
        spec=spec,
        authorization_sha256=args.authorization_sha256,
        audit_sha256=audit_sha256,
        decision_path=decision_path,
    )
    reconciliation_path = output_root / "reconciliation.json"
    claims.atomic_write_json(reconciliation_path, reconciliation)
    environment_path = output_root / "environment.json"
    claims.atomic_write_json(environment_path, claims.environment_record())
    freeze_path, freeze_sha256, freeze = claims.freeze_decision_package(
        output_root=output_root,
        authorization_sha256=args.authorization_sha256,
        decision_spec_sha256=authorization["decision_spec_sha256"],
        files=(preflight_path, audit_path, decision_path, reconciliation_path, environment_path),
        gate=gate,
    )
    _emit(
        "claim_package_frozen",
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_sha256,
        merkle_root=freeze["merkle_root"],
        claim_count=len(records),
        status_counts=reconciliation["status_counts"],
        new_model_execution_count=0,
        new_label_access_count=0,
        prediction_row_access_count=0,
        aggregate_recomputation_count=0,
        threshold_modification_count=0,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="OPM v1.1.4 read-only claim decision executor")
    subparsers = result.add_subparsers(dest="command", required=True)
    authorization = subparsers.add_parser("authorize")
    authorization.add_argument("--workspace", required=True)
    authorization.add_argument("--authorization-directory", required=True)
    authorization.add_argument("--aggregate-freeze", required=True)
    authorization.add_argument("--aggregate-freeze-sha256", required=True)
    authorization.add_argument("--user-request", required=True)
    authorization.set_defaults(function=authorize)
    check = subparsers.add_parser("preflight")
    check.add_argument("--workspace", required=True)
    check.add_argument("--authorization", required=True)
    check.add_argument("--authorization-sha256", required=True)
    check.add_argument("--output-root", required=True)
    check.set_defaults(function=preflight)
    run = subparsers.add_parser("execute")
    run.add_argument("--workspace", required=True)
    run.add_argument("--authorization", required=True)
    run.add_argument("--authorization-sha256", required=True)
    run.add_argument("--output-root", required=True)
    run.set_defaults(function=execute)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
