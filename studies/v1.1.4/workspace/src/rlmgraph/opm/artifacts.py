from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import ModelConfig, ModelKind, OPMModel


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode(
        "utf-8"
    )


def run_id(
    *,
    configuration: dict[str, object],
    specification_version: str,
    dataset_fingerprints: dict[str, str],
    model_seed: int,
    code_revision: str,
) -> str:
    payload = {
        "configuration": configuration,
        "specification_version": specification_version,
        "dataset_fingerprints": dataset_fingerprints,
        "model_seed": model_seed,
        "code_revision": code_revision,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:24]


@dataclass(frozen=True)
class EnvironmentManifest:
    python: str
    pytorch: str
    numpy: str
    cuda_available: bool
    deterministic_algorithms: bool
    device: str


def environment_manifest(device: str = "cpu") -> EnvironmentManifest:
    return EnvironmentManifest(
        python=platform.python_version(),
        pytorch=torch.__version__,
        numpy=np.__version__,
        cuda_available=torch.cuda.is_available(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        device=device,
    )


def write_run_manifest(
    output_directory: Path,
    *,
    configuration: dict[str, object],
    dataset_fingerprints: dict[str, str],
    model_seed: int,
    code_revision: str,
) -> tuple[str, Path]:
    identifier = run_id(
        configuration=configuration,
        specification_version="1.0.0",
        dataset_fingerprints=dataset_fingerprints,
        model_seed=model_seed,
        code_revision=code_revision,
    )
    directory = output_directory / identifier
    directory.mkdir(parents=True, exist_ok=False)
    payload = {
        "run_id": identifier,
        "specification_version": "1.0.0",
        "charter_version": "1.2.0",
        "lifecycle": "IMPLEMENTATION_VALIDATION",
        "primary_run": False,
        "configuration": configuration,
        "dataset_fingerprints": dataset_fingerprints,
        "model_seed": model_seed,
        "code_revision": code_revision,
        "environment": asdict(environment_manifest()),
    }
    path = directory / "run-manifest.json"
    path.write_bytes(canonical_json_bytes(payload))
    return identifier, path


def save_checkpoint(
    path: Path,
    *,
    model: OPMModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, object],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "specification_version": "1.0.0",
        "model_kind": model.kind.value,
        "model_seed": model.model_seed,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_random_state": np.random.get_state(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(
    path: Path, *, optimizer_factory: Any, device: str | torch.device = "cpu"
) -> tuple[OPMModel, torch.optim.Optimizer, int, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = OPMModel(config, ModelKind(payload["model_kind"]), int(payload["model_seed"])).to(
        device
    )
    model.load_state_dict(payload["model_state"])
    optimizer = optimizer_factory(model.parameters())
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_rng_states = payload.get("cuda_rng_states")
    if cuda_rng_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])
    np.random.set_state(payload["numpy_random_state"])
    return model, optimizer, int(payload["step"]), dict(payload["metadata"])
