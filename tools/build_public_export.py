from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_VERSION = "v1.1.4"
SNAPSHOT_ROOT = PurePosixPath("studies/v1.1.4/workspace")
CLOSEOUT_FREEZE = PurePosixPath(
    "evidence/primary_runs/v1.1.4/study-closeout/report-artifacts-v1/"
    "run-4a87ff153338c18e/OPM_V1_1_4_STUDY_CLOSEOUT_FREEZE.json"
)
CLOSEOUT_FREEZE_SHA256 = (
    "f9ee482d973fca5522940bf35a168333f6b12864e614d185b29201a709ed165f"
)
CLOSEOUT_AUTHORIZATION = PurePosixPath(
    "evidence/primary_runs/v1.1.4/study-closeout/authorization-v1/"
    "OPM_V1_1_4_STUDY_CLOSEOUT_AUTHORIZATION.json"
)
STAGE1_FREEZE = PurePosixPath(
    "evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts-v5/"
    "run-ddfc05137e09a402/OPM_V1_1_4_POST_PRIMARY_STAGE1_FREEZE.json"
)
AGGREGATE_FREEZE = PurePosixPath(
    "evidence/primary_runs/v1.1.4/sealed-aggregate/aggregate-artifacts-v4/"
    "run-c41bb1e61655891e/OPM_V1_1_4_SEALED_AGGREGATE_FREEZE.json"
)
CLAIM_FREEZE = PurePosixPath(
    "evidence/primary_runs/v1.1.4/claim-decision/decision-artifacts-v1/"
    "run-1c9afdcaf96847be/OPM_V1_1_4_CLAIM_DECISION_FREEZE.json"
)
SELECTED_CHECKPOINTS = PurePosixPath(
    "evidence/primary_runs/v1.1.4/post-primary/authorization-v5/"
    "OPM_V1_1_4_SELECTED_CHECKPOINTS.json"
)
STAGE1_RESOURCES = PurePosixPath(
    "evidence/primary_runs/v1.1.4/post-primary/authorization-v5/"
    "OPM_V1_1_4_STAGE1_RESOURCE_MANIFEST.json"
)
AGGREGATE_AUTHORIZATION = PurePosixPath(
    "evidence/primary_runs/v1.1.4/sealed-aggregate/authorization-v4/"
    "OPM_V1_1_4_SEALED_AGGREGATE_AUTHORIZATION.json"
)
APPROVED_OPM_SOURCE_SHA256 = (
    "c885ef2eb39feaa5a1ab2116b5bc387fff4bf6713b4b1292befaf084787d6366"
)


class ExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ExportError(f"JSON root is not an object: {path}")
    return value


def as_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    result = PurePosixPath(str(value).replace("\\", "/"))
    if result.is_absolute() or not result.parts or ".." in result.parts:
        raise ExportError(f"unsafe workspace-relative path: {value}")
    return result


def native_path(root: Path, relative: PurePosixPath) -> Path:
    result = (root / Path(*relative.parts)).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ExportError(f"path escapes root: {relative}") from error
    return result


class ExportBuilder:
    def __init__(self, source: Path, destination: Path) -> None:
        self.source = source.resolve()
        self.destination = destination.resolve()
        self.records: dict[str, dict[str, Any]] = {}
        self.external_records: dict[str, dict[str, Any]] = {}

    def add_file(
        self,
        relative: str | PurePosixPath,
        classification: str,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
    ) -> None:
        source_relative = as_relative_path(relative)
        source_path = native_path(self.source, source_relative)
        if not source_path.is_file():
            raise ExportError(f"required export source is missing: {source_relative}")
        actual_bytes = source_path.stat().st_size
        if expected_bytes is not None and actual_bytes != int(expected_bytes):
            raise ExportError(
                f"byte count mismatch for {source_relative}: "
                f"expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha256 = sha256_file(source_path)
        if (
            expected_sha256 is not None
            and actual_sha256 != str(expected_sha256).lower()
        ):
            raise ExportError(
                f"SHA-256 mismatch for {source_relative}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        public_relative = SNAPSHOT_ROOT / source_relative
        key = public_relative.as_posix()
        prior = self.records.get(key)
        if prior is not None:
            if (
                prior["sha256"] != actual_sha256
                or prior["source_path"] != source_relative.as_posix()
            ):
                raise ExportError(
                    f"conflicting duplicate export path: {public_relative}"
                )
            prior["classifications"].add(classification)
            return

        destination_path = native_path(self.destination, public_relative)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            if (
                not destination_path.is_file()
                or sha256_file(destination_path) != actual_sha256
            ):
                raise ExportError(
                    f"refusing to overwrite changed destination: {public_relative}"
                )
        else:
            shutil.copyfile(source_path, destination_path)
        if sha256_file(destination_path) != actual_sha256:
            raise ExportError(f"copied file failed rehash: {public_relative}")

        self.records[key] = {
            "source_path": source_relative.as_posix(),
            "public_path": public_relative.as_posix(),
            "sha256": actual_sha256,
            "bytes": actual_bytes,
            "classifications": {classification},
        }

    def add_external(
        self,
        relative: str | PurePosixPath,
        classification: str,
        *,
        expected_sha256: str,
        expected_bytes: int | None = None,
        reason: str,
    ) -> None:
        source_relative = as_relative_path(relative)
        key = source_relative.as_posix()
        source_path = native_path(self.source, source_relative)
        observed_bytes = source_path.stat().st_size if source_path.is_file() else None
        if (
            expected_bytes is not None
            and observed_bytes is not None
            and observed_bytes != expected_bytes
        ):
            raise ExportError(
                f"external evidence byte count mismatch for {source_relative}: "
                f"expected {expected_bytes}, got {observed_bytes}"
            )
        record = {
            "source_path": key,
            "sha256": str(expected_sha256).lower(),
            "bytes": int(expected_bytes)
            if expected_bytes is not None
            else observed_bytes,
            "classification": classification,
            "repository_status": "EXCLUDED_FROM_GIT_HISTORY",
            "archive_status": "PENDING_EXTERNAL_ARCHIVE",
            "reason": reason,
        }
        prior = self.external_records.get(key)
        if prior is not None and (
            prior["sha256"] != record["sha256"] or prior["bytes"] != record["bytes"]
        ):
            raise ExportError(
                f"conflicting external evidence identity: {source_relative}"
            )
        self.external_records[key] = record

    def add_directory_files(
        self,
        relative: str | PurePosixPath,
        classification: str,
        *,
        pattern: str = "*",
        recursive: bool = False,
    ) -> None:
        directory_relative = as_relative_path(relative)
        directory = native_path(self.source, directory_relative)
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        for path in sorted(iterator, key=lambda item: item.as_posix()):
            if path.is_file() and "__pycache__" not in path.parts:
                self.add_file(path.relative_to(self.source).as_posix(), classification)

    def add_freeze_artifacts(
        self,
        freeze_relative: PurePosixPath,
        classification: str,
    ) -> None:
        self.add_file(freeze_relative, classification)
        freeze_path = native_path(self.source, freeze_relative)
        freeze = read_json(freeze_path)
        artifacts = freeze.get("artifacts")
        if not isinstance(artifacts, list):
            raise ExportError(f"freeze has no artifact inventory: {freeze_relative}")
        for raw in artifacts:
            if not isinstance(raw, dict):
                raise ExportError(f"invalid freeze artifact record: {freeze_relative}")
            artifact_relative = freeze_relative.parent / as_relative_path(
                str(raw.get("path", ""))
            )
            self.add_file(
                artifact_relative,
                classification,
                expected_sha256=str(raw.get("sha256", "")),
                expected_bytes=int(raw["bytes"]) if "bytes" in raw else None,
            )

    def collect_closeout_chain(self) -> None:
        self.add_file(
            CLOSEOUT_FREEZE,
            "frozen-study-closeout",
            expected_sha256=CLOSEOUT_FREEZE_SHA256,
        )
        freeze = read_json(native_path(self.source, CLOSEOUT_FREEZE))
        self.add_file(
            CLOSEOUT_AUTHORIZATION,
            "frozen-study-authorization",
            expected_sha256=str(freeze.get("authorization_sha256", "")),
        )
        authorization = read_json(native_path(self.source, CLOSEOUT_AUTHORIZATION))
        spec_relative = as_relative_path(str(authorization.get("report_spec_path", "")))
        self.add_file(
            spec_relative,
            "frozen-study-authorization",
            expected_sha256=str(authorization.get("report_spec_sha256", "")),
        )
        for raw in authorization.get("report_inputs", []):
            self.add_file(
                str(raw["path"]),
                "frozen-report-input",
                expected_sha256=str(raw["sha256"]),
                expected_bytes=int(raw["bytes"]),
            )
        for raw in authorization.get("reporter_sources", []):
            self.add_file(
                str(raw["path"]),
                "frozen-closeout-source",
                expected_sha256=str(raw["sha256"]),
            )
        for raw in freeze.get("artifacts", []):
            self.add_file(
                CLOSEOUT_FREEZE.parent / as_relative_path(str(raw["path"])),
                "frozen-closeout-artifact",
                expected_sha256=str(raw["sha256"]),
                expected_bytes=int(raw["bytes"]),
            )

    def collect_code_tests_and_approvals(self) -> None:
        self.add_directory_files(
            "src/rlmgraph/opm", "frozen-study-source", pattern="*.py"
        )
        if native_path(self.source, PurePosixPath("scripts/__init__.py")).is_file():
            self.add_file("scripts/__init__.py", "historical-python-namespace")
        self.add_directory_files("scripts", "frozen-study-source", pattern="opm*.py")
        self.add_directory_files("tests", "frozen-study-test", pattern="test_opm*.py")
        self.add_directory_files(
            "evidence/specification",
            "protocol-approval-provenance",
            pattern="*.md",
        )

        supporting = (
            "evidence/implementation_validation/OPM_V1_LEAKAGE_DISPOSITION.md",
            "evidence/implementation_validation/OPM_V1_STATUS_PROTOCOL_FREEZE_APPROVAL_CANDIDATE.md",
            "evidence/implementation_validation/OPM_V1_STATUS_STAGE1_AUTHORIZATION_CANDIDATE.md",
            "evidence/implementation_validation/OPM_V1_1_3_PROTOCOL_FREEZE_TRANSITION.json",
            "evidence/implementation_validation/OPM_V1_1_3_STAGE1_AUTHORIZATION.json",
            "evidence/implementation_validation/OPM_V1_1_3_STAGE1_PREFLIGHT.json",
            "evidence/implementation_validation/OPM_V1_1_3_STAGE1_TRANSITION.json",
        )
        for relative in supporting:
            self.add_file(relative, "lifecycle-provenance")

        for directory, classification in (
            (
                "evidence/primary_runs/v1.1.4/post-primary/authorization-v5",
                "authoritative-stage1-authorization",
            ),
            (
                "evidence/primary_runs/v1.1.4/sealed-aggregate/authorization-v4",
                "authoritative-aggregate-authorization",
            ),
            (
                "evidence/primary_runs/v1.1.4/claim-decision/authorization-v1",
                "authoritative-claim-authorization",
            ),
            (
                "evidence/primary_runs/v1.1.4/study-closeout/authorization-v1",
                "authoritative-closeout-authorization",
            ),
        ):
            self.add_directory_files(directory, classification, pattern="*")

        generated = (
            "evidence/implementation_validation/generated/v1.1.3-canonical-data.audit.json",
            "evidence/implementation_validation/generated/v1.1.3-conformance.json",
            "evidence/implementation_validation/generated/sampler-conformance.v1.1.4.json",
            "evidence/implementation_validation/generated/target-hardware.v1.1.3.json",
        )
        for relative in generated:
            self.add_file(relative, "implementation-validation")
        self.add_directory_files(
            "evidence/implementation_validation/generated/v1.1.3-canonical-data",
            "canonical-data-manifest",
            pattern="*.manifest.json",
        )

    def collect_training_metadata(self) -> None:
        for matrix_name in ("pilot-matrix.json", "primary-matrix.json"):
            matrix_relative = (
                PurePosixPath("evidence/primary_runs/v1.1.4") / matrix_name
            )
            self.add_file(matrix_relative, "training-matrix")
            matrix = read_json(native_path(self.source, matrix_relative))
            artifact_root = str(
                matrix.get("artifact_root")
                or ("pilots" if matrix_name == "pilot-matrix.json" else "")
            )
            if artifact_root not in {"pilots", "primary"}:
                raise ExportError(
                    f"unexpected training artifact root in {matrix_relative}"
                )
            for run in matrix.get("runs", []):
                run_id = str(run.get("run_id", ""))
                run_root = (
                    PurePosixPath("evidence/primary_runs/v1.1.4")
                    / artifact_root
                    / run_id
                )
                for filename in ("run-manifest.json", "summary.json"):
                    relative = run_root / filename
                    if native_path(self.source, relative).is_file():
                        self.add_file(relative, "training-run-metadata")
                events_relative = run_root / "events.jsonl"
                events_path = native_path(self.source, events_relative)
                if events_path.is_file():
                    self.add_external(
                        events_relative,
                        "training-event-stream",
                        expected_sha256=sha256_file(events_path),
                        expected_bytes=events_path.stat().st_size,
                        reason="Full training event stream is reserved for the external archive.",
                    )

        selected = read_json(native_path(self.source, SELECTED_CHECKPOINTS))
        for raw in selected.get("entries", []):
            checkpoint_relative = as_relative_path(str(raw["checkpoint_path"]))
            self.add_external(
                checkpoint_relative,
                "selected-training-checkpoint",
                expected_sha256=str(raw["checkpoint_sha256"]),
                expected_bytes=None,
                reason="Selected model checkpoint is reserved for the external archival release.",
            )
            run_root = checkpoint_relative.parent.parent
            reconciliation = raw.get("reconciliation", {})
            expected = {
                "run-manifest.json": reconciliation.get("run_manifest_sha256"),
                "summary.json": reconciliation.get("summary_sha256"),
            }
            for filename, digest in expected.items():
                if digest:
                    self.add_file(
                        run_root / filename,
                        "selected-training-run-metadata",
                        expected_sha256=str(digest),
                    )
            events_relative = run_root / "events.jsonl"
            events_path = native_path(self.source, events_relative)
            self.add_external(
                events_relative,
                "selected-training-event-stream",
                expected_sha256=str(reconciliation.get("events_sha256", "")),
                expected_bytes=events_path.stat().st_size,
                reason="Full selected-run event stream is reserved for the external archive.",
            )

    def collect_stage1_and_aggregate_evidence(self) -> None:
        stage1_path = native_path(self.source, STAGE1_FREEZE)
        stage1 = read_json(stage1_path)
        self.add_file(STAGE1_FREEZE, "authoritative-stage1-freeze")
        for raw in stage1.get("artifacts", []):
            artifact_relative = STAGE1_FREEZE.parent / as_relative_path(
                str(raw["path"])
            )
            artifact_path = as_relative_path(str(raw["path"]))
            if len(artifact_path.parts) == 1 or artifact_path.name.endswith(
                ".manifest.json"
            ):
                self.add_file(
                    artifact_relative,
                    "stage1-metadata-or-sidecar",
                    expected_sha256=str(raw["sha256"]),
                    expected_bytes=int(raw["bytes"]),
                )
            else:
                self.add_external(
                    artifact_relative,
                    f"stage1-{raw.get('artifact_kind', 'artifact')}",
                    expected_sha256=str(raw["sha256"]),
                    expected_bytes=int(raw["bytes"]),
                    reason="Large row-level Stage-1 evidence is reserved for the external archive.",
                )

        for freeze_relative, classification in (
            (AGGREGATE_FREEZE, "authoritative-aggregate-package"),
            (CLAIM_FREEZE, "authoritative-claim-package"),
        ):
            self.add_freeze_artifacts(freeze_relative, classification)

        aggregate_authorization = read_json(
            native_path(self.source, AGGREGATE_AUTHORIZATION)
        )
        for raw in aggregate_authorization.get("sealed_target_bindings", []):
            self.add_file(
                str(raw["path"]),
                "sealed-target-evidence",
                expected_sha256=str(raw["sha256"]),
            )

    def collect_external_canonical_data(self) -> None:
        resources = read_json(native_path(self.source, STAGE1_RESOURCES))
        for raw_path in resources.get("readable_roots", []):
            relative = as_relative_path(str(raw_path))
            if relative.suffix != ".jsonl" or "sealed-labels" in relative.parts:
                continue
            manifest_relative = PurePosixPath(str(relative) + ".manifest.json")
            manifest_path = native_path(self.source, manifest_relative)
            if not manifest_path.is_file():
                continue
            manifest = read_json(manifest_path)
            self.add_external(
                relative,
                "canonical-dataset",
                expected_sha256=str(manifest["sha256"]),
                expected_bytes=native_path(self.source, relative).stat().st_size,
                reason="Generated canonical dataset is represented in Git by its bound manifest.",
            )

    def write_manifests(self) -> None:
        records = []
        for key in sorted(self.records):
            record = dict(self.records[key])
            record["classifications"] = sorted(record["classifications"])
            records.append(record)
        export_manifest = {
            "schema_version": "opm-public-export-manifest-v1",
            "protocol_version": PROTOCOL_VERSION,
            "snapshot_root": SNAPSHOT_ROOT.as_posix(),
            "source_workspace": "private RLMGraph development origin (path intentionally omitted)",
            "declared_opm_source_tree_sha256": APPROVED_OPM_SOURCE_SHA256,
            "closeout_freeze_path": (SNAPSHOT_ROOT / CLOSEOUT_FREEZE).as_posix(),
            "closeout_freeze_sha256": CLOSEOUT_FREEZE_SHA256,
            "file_count": len(records),
            "total_bytes": sum(int(record["bytes"]) for record in records),
            "files": records,
        }
        external = [self.external_records[key] for key in sorted(self.external_records)]
        external_manifest = {
            "schema_version": "opm-external-evidence-manifest-v1",
            "protocol_version": PROTOCOL_VERSION,
            "archive_status": "PENDING_EXTERNAL_ARCHIVE",
            "artifact_count": len(external),
            "known_total_bytes": sum(
                int(record["bytes"])
                for record in external
                if record["bytes"] is not None
            ),
            "artifacts": external,
        }
        previous_path = self.destination / "OPM_PUBLIC_EXPORT_MANIFEST.json"
        if previous_path.is_file():
            previous = read_json(previous_path)
            if previous.get("schema_version") != "opm-public-export-manifest-v1":
                raise ExportError(
                    f"refusing to replace unknown prior manifest: {previous_path}"
                )
            for raw in previous.get("files", []):
                public_path = as_relative_path(str(raw.get("public_path", "")))
                if public_path.as_posix() in self.records:
                    continue
                obsolete = native_path(self.destination, public_path)
                if not obsolete.is_file() or sha256_file(obsolete) != str(
                    raw.get("sha256", "")
                ):
                    raise ExportError(
                        f"refusing to prune changed generated file: {public_path}"
                    )
                obsolete.unlink()
        for filename, value in (
            ("OPM_PUBLIC_EXPORT_MANIFEST.json", export_manifest),
            ("OPM_EXTERNAL_EVIDENCE_MANIFEST.json", external_manifest),
        ):
            path = self.destination / filename
            encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            path.write_bytes(encoded)

    def build(self) -> None:
        self.collect_closeout_chain()
        self.collect_code_tests_and_approvals()
        self.collect_training_metadata()
        self.collect_stage1_and_aggregate_evidence()
        self.collect_external_canonical_data()
        self.write_manifests()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the byte-preserving OPM v1.1.4 public export"
    )
    result.add_argument(
        "--source", required=True, type=Path, help="Private RLMGraph workspace"
    )
    result.add_argument(
        "--destination", required=True, type=Path, help="Public staging repository"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not (source / ".git").is_dir():
        raise ExportError(f"source is not the expected local Git workspace: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    ExportBuilder(source, destination).build()
    print(
        json.dumps(
            {"event": "opm_public_export_built", "destination": str(destination)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
