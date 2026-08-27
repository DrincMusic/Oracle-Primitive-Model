from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_FREEZE_SHA256 = (
    "f9ee482d973fca5522940bf35a168333f6b12864e614d185b29201a709ed165f"
)
MAX_GIT_OBJECT_BYTES = 100_000_000


class VerificationError(RuntimeError):
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
        raise VerificationError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"JSON root is not an object: {path}")
    return value


def repository_path(repository: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise VerificationError(f"unsafe repository-relative path: {value}")
    result = (repository / Path(*relative.parts)).resolve()
    try:
        result.relative_to(repository.resolve())
    except ValueError as error:
        raise VerificationError(f"path escapes repository: {value}") from error
    return result


def is_generated_cache(path: Path) -> bool:
    return (
        "__pycache__" in path.parts
        or ".pytest_cache" in path.parts
        or path.suffix in {".pyc", ".pyo"}
    )


def verify_manifest(repository: Path) -> tuple[dict[str, Any], int]:
    manifest = read_json(repository / "OPM_PUBLIC_EXPORT_MANIFEST.json")
    if manifest.get("schema_version") != "opm-public-export-manifest-v1":
        raise VerificationError("unsupported public export manifest schema")
    if manifest.get("closeout_freeze_sha256") != EXPECTED_FREEZE_SHA256:
        raise VerificationError(
            "public export manifest names the wrong closeout freeze"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise VerificationError("public export manifest has no file inventory")

    expected_paths: set[str] = set()
    total_bytes = 0
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise VerificationError(
                "public export manifest contains an invalid file record"
            )
        public_path = str(raw.get("public_path", ""))
        if public_path in expected_paths:
            raise VerificationError(f"duplicate public path in manifest: {public_path}")
        expected_paths.add(public_path)
        path = repository_path(repository, public_path)
        if not path.is_file():
            raise VerificationError(f"manifest file is missing: {public_path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != int(raw.get("bytes", -1)):
            raise VerificationError(f"byte count mismatch: {public_path}")
        if sha256_file(path) != str(raw.get("sha256", "")):
            raise VerificationError(f"SHA-256 mismatch: {public_path}")
        total_bytes += actual_bytes

    snapshot_root = repository_path(repository, str(manifest.get("snapshot_root", "")))
    actual_paths = {
        path.relative_to(repository).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file() and not is_generated_cache(path)
    }
    missing = expected_paths - actual_paths
    unmanifested = actual_paths - expected_paths
    if missing:
        raise VerificationError(f"snapshot is missing {len(missing)} manifest files")
    if unmanifested:
        preview = ", ".join(sorted(unmanifested)[:5])
        raise VerificationError(f"snapshot contains unmanifested files: {preview}")
    if total_bytes != int(manifest.get("total_bytes", -1)):
        raise VerificationError("public export aggregate byte count changed")
    if len(expected_paths) != int(manifest.get("file_count", -1)):
        raise VerificationError("public export file count changed")
    return manifest, total_bytes


def verify_external_manifest(repository: Path) -> dict[str, Any]:
    manifest = read_json(repository / "OPM_EXTERNAL_EVIDENCE_MANIFEST.json")
    if manifest.get("schema_version") != "opm-external-evidence-manifest-v1":
        raise VerificationError("unsupported external evidence manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != int(
        manifest.get("artifact_count", -1)
    ):
        raise VerificationError("external evidence inventory count changed")
    paths = [
        str(item.get("source_path", "")) for item in artifacts if isinstance(item, dict)
    ]
    if len(paths) != len(set(paths)):
        raise VerificationError("external evidence inventory contains duplicate paths")
    return manifest


def verify_repository_size_boundary(repository: Path) -> None:
    for path in repository.rglob("*"):
        if not path.is_file() or ".git" in path.parts or is_generated_cache(path):
            continue
        if path.stat().st_size > MAX_GIT_OBJECT_BYTES:
            raise VerificationError(
                f"file exceeds the normal GitHub object boundary: {path.relative_to(repository)}"
            )


def verify_closeout(repository: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    workspace = repository_path(repository, str(manifest["snapshot_root"]))
    freeze = repository_path(repository, str(manifest["closeout_freeze_path"]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.opm_study_closeout_executor",
            "verify",
            "--workspace",
            str(workspace),
            "--freeze",
            str(freeze),
            "--freeze-sha256",
            EXPECTED_FREEZE_SHA256,
        ],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VerificationError(
            "closeout verifier failed:\n"
            + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise VerificationError("closeout verifier returned invalid output") from error
    if payload.get("verification") != "PASS":
        raise VerificationError("closeout verifier did not return PASS")
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify the OPM public export and closed study"
    )
    result.add_argument("--repository", type=Path, default=Path.cwd())
    return result


def main() -> int:
    args = parser().parse_args()
    repository = args.repository.resolve()
    manifest, total_bytes = verify_manifest(repository)
    external = verify_external_manifest(repository)
    verify_repository_size_boundary(repository)
    closeout = verify_closeout(repository, manifest)
    print(
        json.dumps(
            {
                "verification": "PASS",
                "export_file_count": manifest["file_count"],
                "export_total_bytes": total_bytes,
                "external_artifact_count": external["artifact_count"],
                "closeout_freeze_sha256": closeout["freeze_sha256"],
                "closeout_merkle_root": closeout["merkle_root"],
                "claim_count": closeout["claim_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
