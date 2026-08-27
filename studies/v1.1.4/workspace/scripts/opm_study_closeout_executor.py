from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import opm_study_closeout as closeout


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def authorize(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    path, digest, spec_path, spec_digest = closeout.create_authorization(
        workspace=workspace,
        authorization_directory=Path(args.authorization_directory).resolve(),
        expected_claim_freeze_sha256=args.claim_freeze_sha256,
        reporter_sources=[
            workspace / "scripts/opm_study_closeout.py",
            workspace / "scripts/opm_study_closeout_executor.py",
        ],
        user_request=args.user_request,
    )
    _emit(
        "study_closeout_authorized",
        authorization_path=str(path),
        authorization_sha256=digest,
        report_spec_path=str(spec_path),
        report_spec_sha256=spec_digest,
        checkpoint_access_count=0,
        sealed_label_access_count=0,
        prediction_row_access_count=0,
    )
    return 0


def preflight(args: argparse.Namespace) -> int:
    result, _gate, _resources, _spec, _authorization = closeout.preflight_closeout(
        workspace=Path(args.workspace).resolve(),
        authorization_path=Path(args.authorization).resolve(),
        expected_authorization_sha256=args.authorization_sha256,
        output_root=Path(args.output_root).resolve(),
    )
    _emit("study_closeout_preflight_passed", **asdict(result))
    return 0


def execute(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output_root = Path(args.output_root).resolve()
    preflight_result, gate, resources, _spec, authorization = closeout.preflight_closeout(
        workspace=workspace,
        authorization_path=Path(args.authorization).resolve(),
        expected_authorization_sha256=args.authorization_sha256,
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    preflight_path = resources.require_write(output_root / "preflight.json")
    closeout.atomic_write_json(preflight_path, asdict(preflight_result))
    audit_path = resources.require_write(output_root / "closeout.audit.jsonl")
    closeout.append_jsonl(
        audit_path,
        {
            "timestamp_utc": closeout.utc_now(),
            "event": "preflight_passed",
            "authorization_sha256": args.authorization_sha256,
            "verified_input_files": preflight_result.verified_input_files,
            "checkpoint_access_count": 0,
            "sealed_label_access_count": 0,
            "prediction_row_access_count": 0,
        },
    )
    for binding in authorization["report_inputs"]:
        path = closeout.resolve_workspace_path(binding["path"], workspace, "report input")
        resources.require_read(path)
        closeout.append_jsonl(
            audit_path,
            {
                "timestamp_utc": closeout.utc_now(),
                "event": "bound_report_input_verified",
                "path": binding["path"],
                "sha256": binding["sha256"],
            },
        )
    generated_at = closeout.utc_now()
    report_data = closeout.build_report_data(
        workspace=workspace,
        resources=resources,
        authorization_sha256=args.authorization_sha256,
        authorization=authorization,
        generated_at_utc=generated_at,
    )
    report_data_path = resources.require_write(output_root / "report-data.json")
    closeout.atomic_write_json(report_data_path, report_data)
    report_markdown = closeout.render_report_markdown(report_data, authorization)
    report_path = resources.require_write(output_root / "OPM_V1_1_4_FINAL_REPORT.md")
    closeout.atomic_write_text(report_path, report_markdown)
    closeout.append_jsonl(
        audit_path,
        {
            "timestamp_utc": closeout.utc_now(),
            "event": "closeout_report_generated",
            "claim_count": 15,
            "status_counts": report_data["status_counts"],
            "claim_decision_modification_count": 0,
            "aggregate_recomputation_count": 0,
            "new_scientific_inference_count": 0,
        },
    )
    audit_sha256 = closeout.sha256_file(audit_path)
    gate.require(closeout.Capability.CLOSEOUT_RECONCILIATION)
    reconciliation = closeout.reconcile_report(
        report_data=report_data,
        report_markdown=report_markdown,
        report_path=report_path,
        report_data_path=report_data_path,
        authorization_sha256=args.authorization_sha256,
        audit_sha256=audit_sha256,
    )
    reconciliation_path = resources.require_write(output_root / "reconciliation.json")
    closeout.atomic_write_json(reconciliation_path, reconciliation)
    environment_path = resources.require_write(output_root / "environment.json")
    closeout.atomic_write_json(environment_path, closeout.environment_record())
    freeze_path, freeze_sha256, freeze = closeout.freeze_closeout_package(
        output_root=output_root,
        authorization_sha256=args.authorization_sha256,
        report_spec_sha256=authorization["report_spec_sha256"],
        files=(
            preflight_path,
            audit_path,
            report_data_path,
            report_path,
            reconciliation_path,
            environment_path,
        ),
        gate=gate,
    )
    _emit(
        "study_closeout_package_frozen",
        report_path=str(report_path),
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_sha256,
        merkle_root=freeze["merkle_root"],
        artifact_count=freeze["artifact_count"],
        claim_count=freeze["claim_count"],
        status_counts=freeze["status_counts"],
        checkpoint_access_count=0,
        sealed_label_access_count=0,
        prediction_row_access_count=0,
        aggregate_recomputation_count=0,
        claim_decision_modification_count=0,
    )
    return 0


def verify(args: argparse.Namespace) -> int:
    result = closeout.verify_closeout_package(
        workspace=Path(args.workspace).resolve(),
        freeze_path=Path(args.freeze).resolve(),
        expected_freeze_sha256=args.freeze_sha256,
    )
    _emit("study_closeout_verification_passed", **result)
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="OPM v1.1.4 read-only study closeout executor")
    subparsers = result.add_subparsers(dest="command", required=True)
    authorization = subparsers.add_parser("authorize")
    authorization.add_argument("--workspace", required=True)
    authorization.add_argument("--authorization-directory", required=True)
    authorization.add_argument("--claim-freeze-sha256", required=True)
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
    verification = subparsers.add_parser("verify")
    verification.add_argument("--workspace", required=True)
    verification.add_argument("--freeze", required=True)
    verification.add_argument("--freeze-sha256", required=True)
    verification.set_defaults(function=verify)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
