from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts import opm_sealed_aggregate as aggregate


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def authorize(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    authorization_path, authorization_sha256, spec_path, spec_sha256 = (
        aggregate.create_authorization(
            workspace=workspace,
            authorization_directory=Path(args.authorization_directory).resolve(),
            stage1_freeze_path=Path(args.stage1_freeze).resolve(),
            expected_stage1_freeze_sha256=args.stage1_freeze_sha256,
            stage1_evaluation_spec_path=Path(args.stage1_evaluation_spec).resolve(),
            evaluator_sources=[
                workspace / "scripts/opm_sealed_aggregate.py",
                workspace / "scripts/opm_sealed_aggregate_executor.py",
            ],
            user_request=args.user_request,
        )
    )
    _emit(
        "authorization_recorded",
        authorization_path=str(authorization_path),
        authorization_sha256=authorization_sha256,
        aggregate_spec_path=str(spec_path),
        aggregate_spec_sha256=spec_sha256,
        sealed_target_files_opened=0,
    )
    return 0


def preflight(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    result, _gate, _resources, _spec, _freeze = aggregate.preflight_aggregate_evaluation(
        workspace=workspace,
        authorization_path=Path(args.authorization).resolve(),
        expected_authorization_sha256=args.authorization_sha256,
        output_root=Path(args.output_root).resolve(),
    )
    _emit("preflight_passed", **asdict(result))
    return 0


def execute(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    authorization_path = Path(args.authorization).resolve()
    output_root = Path(args.output_root).resolve()
    preflight_result, gate, resources, spec, freeze = aggregate.preflight_aggregate_evaluation(
        workspace=workspace,
        authorization_path=authorization_path,
        expected_authorization_sha256=args.authorization_sha256,
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    preflight_path = output_root / "preflight.json"
    aggregate.atomic_write_json(preflight_path, asdict(preflight_result))
    _emit("preflight_recorded", path=str(preflight_path), sealed_target_files_opened=0)
    metadata, global_metadata = aggregate.load_metadata(
        workspace=workspace, spec=spec, resources=resources
    )
    _emit("metadata_verified", row_count=sum(len(rows) for rows in metadata.values()))
    audit_path = output_root / "sealed-target-access.audit.jsonl"
    labels = aggregate.load_sealed_targets(
        workspace=workspace,
        spec=spec,
        metadata=metadata,
        resources=resources,
        audit_path=audit_path,
    )
    _emit(
        "sealed_targets_verified",
        file_count=4,
        row_count=sum(len(rows) for rows in labels.values()),
    )
    stage_root = aggregate.resolve_workspace_path(
        spec["stage1_freeze"]["path"], workspace, "Stage 1 freeze"
    ).parent
    baseline, effects, interventions, probes = aggregate.compute_aggregates(
        stage_root=stage_root,
        freeze=freeze,
        metadata=metadata,
        global_metadata=global_metadata,
        labels=labels,
        gate=gate,
    )
    result_payloads = {
        "baseline-metrics.json": baseline,
        "primary-effects.json": effects,
        "intervention-metrics.json": interventions,
        "probe-metrics.json": probes,
        "environment.json": aggregate.environment_record(),
    }
    result_paths: list[Path] = [preflight_path, audit_path]
    for filename, payload in result_payloads.items():
        path = output_root / filename
        resources.require_write(path)
        aggregate.atomic_write_json(path, payload)
        result_paths.append(path)
    summary = {
        "schema_version": aggregate.EXECUTION_SUMMARY_SCHEMA,
        "protocol_version": aggregate.PROTOCOL_VERSION,
        "state": "AGGREGATES_COMPUTED_PENDING_FREEZE",
        "authorization_sha256": args.authorization_sha256,
        "aggregate_spec_sha256": aggregate.sha256_file(
            aggregate.resolve_workspace_path(
                json.loads(authorization_path.read_text(encoding="utf-8"))["aggregate_spec_path"],
                workspace,
                "aggregate spec",
            )
        ),
        "prediction_rows_joined": baseline["prediction_rows_joined"],
        "intervention_rows_joined": interventions["intervention_rows_joined"],
        "sealed_target_rows_joined": sum(len(rows) for rows in labels.values()),
        "bootstrap_replicates": effects["paired_two_level_bootstrap"]["replicates"],
        "sealed_target_access_count": 4,
        "checkpoint_access_count": 0,
        "model_load_count": 0,
        "prediction_generation_count": 0,
        "claim_threshold_application_count": 0,
        "claim_decision_count": 0,
    }
    summary_path = output_root / "execution-summary.json"
    aggregate.atomic_write_json(summary_path, summary)
    result_paths.append(summary_path)
    freeze_path, freeze_sha256, aggregate_freeze = aggregate.freeze_aggregate_package(
        output_root=output_root,
        authorization_sha256=args.authorization_sha256,
        spec_sha256=summary["aggregate_spec_sha256"],
        result_paths=result_paths,
        sealed_target_bindings=spec["sealed_targets"],
        gate=gate,
    )
    _emit(
        "aggregate_package_frozen",
        freeze_path=str(freeze_path),
        freeze_sha256=freeze_sha256,
        merkle_root=aggregate_freeze["merkle_root"],
        artifact_count=aggregate_freeze["artifact_count"],
        checkpoint_access_count=0,
        model_load_count=0,
        prediction_generation_count=0,
        claim_threshold_application_count=0,
        claim_decision_count=0,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="OPM v1.1.4 aggregate-only sealed evaluator")
    subparsers = result.add_subparsers(dest="command", required=True)
    authorization = subparsers.add_parser("authorize")
    authorization.add_argument("--workspace", required=True)
    authorization.add_argument("--authorization-directory", required=True)
    authorization.add_argument("--stage1-freeze", required=True)
    authorization.add_argument("--stage1-freeze-sha256", required=True)
    authorization.add_argument("--stage1-evaluation-spec", required=True)
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
