from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY / "studies" / "v1.1.4" / "workspace"
EXPERIMENT = REPOSITORY / "OPM_EXPERIMENT.json"
REQUIREMENTS = REPOSITORY / "requirements-study.txt"
CPU_TORCH_REQUIREMENTS = REPOSITORY / "requirements-torch-cpu.txt"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
VERIFIER = REPOSITORY / "tools" / "verify_public_export.py"
MINIMUM_PYTHON = (3, 11)
RECOMMENDED_PYTHON = (3, 12)


def display_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def run(command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"+ {display_command(command)}", flush=True)
    result = subprocess.run(list(command), cwd=cwd, env=env, check=False)
    return result.returncode


def venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def longest_repository_path() -> tuple[str, int]:
    longest = REPOSITORY
    longest_length = len(str(longest))
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for root, directory_names, file_names in os.walk(REPOSITORY):
        directory_names[:] = [name for name in directory_names if name not in excluded]
        for name in [*directory_names, *file_names]:
            path = Path(root) / name
            length = len(str(path.resolve()))
            if length > longest_length:
                longest = path
                longest_length = length
    return longest.relative_to(REPOSITORY).as_posix(), longest_length


def doctor_payload() -> tuple[dict[str, Any], bool]:
    required_paths = [
        EXPERIMENT,
        REQUIREMENTS,
        CPU_TORCH_REQUIREMENTS,
        VERIFIER,
        WORKSPACE / "src",
        WORKSPACE / "tests",
    ]
    missing = [
        path.relative_to(REPOSITORY).as_posix()
        for path in required_paths
        if not path.exists()
    ]
    contract: dict[str, Any] = {}
    contract_error: str | None = None
    try:
        contract = load_json(EXPERIMENT)
        if contract.get("schema_version") != "opm-agent-entrypoint-v1":
            contract_error = "unsupported OPM_EXPERIMENT.json schema"
    except (OSError, TypeError, json.JSONDecodeError) as error:
        contract_error = str(error)

    python_supported = sys.version_info[:2] >= MINIMUM_PYTHON
    longest_path, longest_path_length = longest_repository_path()
    disk = shutil.disk_usage(REPOSITORY)
    ready = not missing and contract_error is None and python_supported
    payload = {
        "status": "PASS" if ready else "FAIL",
        "repository": str(REPOSITORY),
        "python": {
            "executable": sys.executable,
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "minimum": ".".join(str(value) for value in MINIMUM_PYTHON),
            "recommended": ".".join(str(value) for value in RECOMMENDED_PYTHON),
            "supported": python_supported,
        },
        "missing_required_paths": missing,
        "contract_error": contract_error,
        "longest_repository_path": longest_path,
        "longest_absolute_path_characters": longest_path_length,
        "free_disk_bytes": disk.free,
        "profiles": {
            "closed_result_verification": "READY" if ready else "BLOCKED",
            "source_test_environment": "READY_TO_INSTALL" if ready else "BLOCKED",
            "full_computational_reproduction": "BLOCKED_PENDING_REPRODUCTION_BUNDLE",
        },
        "study_state": contract.get("study_state"),
    }
    return payload, ready


def command_doctor(args: argparse.Namespace) -> int:
    payload, ready = doctor_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"OPM agent doctor: {payload['status']}")
        print(f"Repository: {payload['repository']}")
        print(
            "Python: "
            f"{payload['python']['version']} "
            f"(minimum {payload['python']['minimum']}; recommended {payload['python']['recommended']})"
        )
        print(
            f"Closed-result verification: {payload['profiles']['closed_result_verification']}"
        )
        print(
            f"Source-test environment: {payload['profiles']['source_test_environment']}"
        )
        print(
            f"Full reproduction: {payload['profiles']['full_computational_reproduction']}"
        )
        if payload["missing_required_paths"]:
            print("Missing: " + ", ".join(payload["missing_required_paths"]))
        if payload["contract_error"]:
            print("Contract error: " + str(payload["contract_error"]))
        if os.name == "nt" and int(payload["longest_absolute_path_characters"]) >= 240:
            print(
                "Windows path warning: use a short clone destination and "
                "git -c core.longpaths=true clone."
            )
    return 0 if ready else 1


def command_verify(_args: argparse.Namespace) -> int:
    return run(
        [sys.executable, str(VERIFIER), "--repository", str(REPOSITORY)],
        cwd=REPOSITORY,
    )


def run_tests(python: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE / "src"), str(WORKSPACE)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return run(
        [str(python), "-m", "pytest", "-p", "no:cacheprovider", "tests", "-q"],
        cwd=WORKSPACE,
        env=env,
    )


def command_test(_args: argparse.Namespace) -> int:
    return run_tests(Path(sys.executable))


def command_setup(args: argparse.Namespace) -> int:
    venv_root = args.venv
    if not venv_root.is_absolute():
        venv_root = REPOSITORY / venv_root
    python = venv_python(venv_root)
    commands: list[tuple[list[str], Path]] = []
    if not python.is_file():
        commands.append(([sys.executable, "-m", "venv", str(venv_root)], REPOSITORY))
    if args.torch_profile == "cpu" and sys.platform in {"linux", "win32"}:
        commands.append(
            (
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--index-url",
                    CPU_TORCH_INDEX,
                    "-r",
                    str(CPU_TORCH_REQUIREMENTS),
                ],
                REPOSITORY,
            )
        )
    commands.append(
        ([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], REPOSITORY)
    )

    if args.dry_run:
        print("Setup plan (no changes made):")
        for command, _cwd in commands:
            print(f"+ {display_command(command)}")
        print(
            f"+ {display_command([str(python), str(Path(__file__).resolve()), 'verify'])}"
        )
        print(
            f"+ {display_command([str(python), str(Path(__file__).resolve()), 'test'])}"
        )
        return 0

    for command, cwd in commands:
        code = run(command, cwd=cwd)
        if code != 0:
            return code
    code = run([str(python), str(Path(__file__).resolve()), "verify"], cwd=REPOSITORY)
    if code != 0:
        return code
    return run([str(python), str(Path(__file__).resolve()), "test"], cwd=REPOSITORY)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Agent-friendly setup and verification entry point for OPM v1.1.4"
    )
    subparsers = result.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="Inspect prerequisites without changing anything"
    )
    doctor.add_argument(
        "--json", action="store_true", help="Emit machine-readable output"
    )
    doctor.set_defaults(handler=command_doctor)

    verify = subparsers.add_parser("verify", help="Verify the byte-bound closed study")
    verify.set_defaults(handler=command_verify)

    test = subparsers.add_parser("test", help="Run the standalone OPM source tests")
    test.set_defaults(handler=command_test)

    setup = subparsers.add_parser(
        "setup",
        help="Create a virtual environment, install dependencies, verify, and test",
    )
    setup.add_argument(
        "--venv",
        type=Path,
        default=Path(".venv"),
        help="Virtual-environment path (default: .venv)",
    )
    setup.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the setup commands without changing anything",
    )
    setup.add_argument(
        "--torch-profile",
        choices=("cpu", "default"),
        default="cpu",
        help="Install a compact CPU wheel or use the platform's default PyTorch wheel",
    )
    setup.set_defaults(handler=command_setup)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.handler(args))
    except OSError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
