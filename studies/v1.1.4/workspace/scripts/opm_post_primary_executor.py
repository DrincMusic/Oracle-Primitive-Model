from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.stats import binomtest
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlmgraph.opm.data import Vocabulary
from rlmgraph.opm.generation import Operation, derive_uint64
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMBatch, OPMModel
from rlmgraph.opm.probes import wilson_interval
from rlmgraph.opm.rendering import Domain
from scripts.opm_post_primary import (
    FREEZE_FILENAME,
    INTERVENTION_SCHEMA,
    PREDICTION_SCHEMA,
    PROBE_SCHEMA,
    AtomicJsonlArtifact,
    AuthorizationError,
    Capability,
    CapabilityGate,
    InputInventory,
    IntegrityError,
    ReconciliationError,
    ResourcePolicy,
    assert_base_model_unchanged,
    atomic_write_json,
    batch_to_device,
    build_evaluation_spec,
    build_intervention_spec,
    build_probe_spec,
    build_resource_manifest,
    build_selected_checkpoint_manifest,
    build_transition_record,
    canonical_sha256,
    environment_record,
    freeze_base_model,
    git_commit,
    preflight_stage1,
    read_json,
    reconcile_and_freeze,
    refuse_aggregate_evaluation,
    refuse_claim_decision,
    sha256_file,
    tensor_state_sha256,
    verify_jsonl_artifact,
)

DEFAULT_PRIMARY_MATRIX = Path("evidence/primary_runs/v1.1.4/primary-matrix.json")
DEFAULT_HANDOFF = Path("evidence/implementation_validation/OPM_V1_1_4_PRIMARY_EXECUTION_HANDOFF.md")
DEFAULT_INPUT_DIRECTORY = Path("evidence/implementation_validation/generated/v1.1.3-canonical-data")
DEFAULT_AUTHORIZATION_DIRECTORY = Path("evidence/primary_runs/v1.1.4/post-primary/authorization")
DEFAULT_STAGE1_ROOT = Path("evidence/primary_runs/v1.1.4/post-primary/stage1-artifacts")


def _publish_or_verify(path: Path, payload: object) -> str:
    expected = canonical_sha256(payload)
    if path.exists():
        if sha256_file(path) != expected:
            raise IntegrityError(f"immutable authorization artifact differs: {path}")
        return expected
    return atomic_write_json(path, payload)


def _authorization_paths(directory: Path) -> dict[str, Path]:
    return {
        "checkpoints": directory / "OPM_V1_1_4_SELECTED_CHECKPOINTS.json",
        "evaluation": directory / "OPM_V1_1_4_STAGE1_EVALUATION_SPEC.json",
        "probe": directory / "OPM_V1_1_4_STAGE1_PROBE_SPEC.json",
        "intervention": directory / "OPM_V1_1_4_STAGE1_INTERVENTION_SPEC.json",
        "resources": directory / "OPM_V1_1_4_STAGE1_RESOURCE_MANIFEST.json",
        "transition": directory / "OPM_V1_1_4_POST_PRIMARY_TRANSITION.json",
    }


def record_transition(
    *,
    workspace: Path,
    primary_matrix_path: Path,
    handoff_path: Path,
    input_directory: Path,
    authorization_directory: Path,
    stage1_root: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    primary_matrix_path = (workspace / primary_matrix_path).resolve()
    handoff_path = (workspace / handoff_path).resolve()
    input_directory = (workspace / input_directory).resolve()
    authorization_directory = (workspace / authorization_directory).resolve()
    stage1_root = (workspace / stage1_root).resolve()
    paths = _authorization_paths(authorization_directory)
    authorization_directory.mkdir(parents=True, exist_ok=True)

    checkpoints = build_selected_checkpoint_manifest(
        workspace=workspace, primary_matrix_path=primary_matrix_path
    )
    _publish_or_verify(paths["checkpoints"], checkpoints)
    evaluation = build_evaluation_spec(workspace=workspace, input_directory=input_directory)
    _publish_or_verify(paths["evaluation"], evaluation)
    probe = build_probe_spec(workspace=workspace, input_directory=input_directory)
    _publish_or_verify(paths["probe"], probe)
    intervention = build_intervention_spec()
    _publish_or_verify(paths["intervention"], intervention)

    readable_files: list[Path] = [
        authorization_directory,
        primary_matrix_path,
        handoff_path,
    ]
    readable_files.extend(
        (workspace / entry["checkpoint_path"]).resolve() for entry in checkpoints["entries"]
    )
    for item in evaluation["splits"]:
        readable_files.extend(
            [
                (workspace / item["input_path"]).resolve(),
                (workspace / item["input_manifest_path"]).resolve(),
            ]
        )
    for item in probe["inputs"].values():
        readable_files.extend(
            [
                (workspace / item["path"]).resolve(),
                (workspace / item["manifest_path"]).resolve(),
            ]
        )
    resources = build_resource_manifest(
        workspace=workspace,
        readable_roots=readable_files,
        writable_root=stage1_root,
    )
    _publish_or_verify(paths["resources"], resources)
    transition = build_transition_record(
        workspace=workspace,
        primary_matrix_path=primary_matrix_path,
        checkpoint_manifest_path=paths["checkpoints"],
        handoff_path=handoff_path,
        evaluation_spec_path=paths["evaluation"],
        probe_spec_path=paths["probe"],
        intervention_spec_path=paths["intervention"],
        executor_sources=[
            workspace / "scripts/opm_post_primary.py",
            workspace / "scripts/opm_post_primary_executor.py",
        ],
    )
    transition_sha256 = _publish_or_verify(paths["transition"], transition)
    return {
        "state": "POST_PRIMARY_TRANSITION_RECORDED",
        "transition_path": str(paths["transition"]),
        "transition_sha256": transition_sha256,
        "checkpoint_count": checkpoints["checkpoint_count"],
        "primary_matrix_sha256": checkpoints["primary_matrix_sha256"],
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
    }


class ExecutionLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: str, **fields: object) -> None:
        row = {
            "event": event,
            "recorded_at": datetime.now(UTC).isoformat(),
            **fields,
        }
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise IntegrityError(f"nonobject JSONL row at {path}:{line_number}")
            yield row


def _batch_from_rows(rows: Sequence[Mapping[str, Any]], vocabulary: Vocabulary) -> OPMBatch:
    fact_tokens: list[list[list[int]]] = []
    fact_masks: list[list[list[bool]]] = []
    query_tokens: list[list[int]] = []
    query_masks: list[list[bool]] = []
    for row in rows:
        encoded_facts = [vocabulary.encode(str(text), "[FACT]") for text in row["rendered_facts"]]
        fact_tokens.append([item[0] for item in encoded_facts])
        fact_masks.append([item[1] for item in encoded_facts])
        query = vocabulary.encode(str(row["rendered_query"]), "[QUERY]")
        query_tokens.append(query[0])
        query_masks.append(query[1])
    labels = None
    if all("label" in row for row in rows):
        labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    return OPMBatch(
        fact_tokens=torch.tensor(fact_tokens, dtype=torch.long),
        fact_token_mask=torch.tensor(fact_masks, dtype=torch.bool),
        query_tokens=torch.tensor(query_tokens, dtype=torch.long),
        query_token_mask=torch.tensor(query_masks, dtype=torch.bool),
        domain_ids=torch.tensor([int(Domain[str(row["domain"])]) for row in rows]),
        argument_entity_ids=torch.tensor([row["argument_entity_ids"] for row in rows]),
        fact_endpoint_ids=torch.tensor([row["fact_endpoint_ids"] for row in rows]),
        evidence_indices=torch.tensor([row["evidence_indices"] for row in rows]),
        operation_ids=torch.tensor([row["operation_ids"] for row in rows]),
        step_mask=torch.tensor([row["step_mask"] for row in rows], dtype=torch.bool),
        labels=labels,
    )


def _iter_batches(
    path: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    *,
    two_step_only: bool = False,
    renderer_variant: int | None = None,
) -> Iterator[tuple[list[dict[str, Any]], OPMBatch]]:
    pending: list[dict[str, Any]] = []
    for row in _read_rows(path):
        if two_step_only and list(row.get("step_mask", [])) != [True, True]:
            continue
        if renderer_variant is not None and row.get("renderer_variant") != renderer_variant:
            continue
        pending.append(row)
        if len(pending) == batch_size:
            yield pending, _batch_from_rows(pending, vocabulary)
            pending = []
    if pending:
        yield pending, _batch_from_rows(pending, vocabulary)


def _load_model(
    checkpoint: Mapping[str, Any],
    *,
    workspace: Path,
    resources: ResourcePolicy,
    gate: CapabilityGate,
    device: torch.device,
) -> OPMModel:
    gate.require(Capability.CHECKPOINT_READ)
    path = resources.require_read((workspace / str(checkpoint["checkpoint_path"])).resolve())
    if sha256_file(path) != checkpoint["checkpoint_sha256"]:
        raise IntegrityError("checkpoint changed immediately before model load")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("model_kind") != checkpoint["condition"]
        or int(payload.get("model_seed", -1)) != checkpoint["training_seed"]
        or int(payload.get("step", -1)) != checkpoint["selected_step"]
    ):
        raise IntegrityError("loaded checkpoint identity or selected step is unapproved")
    model = OPMModel(
        ModelConfig(**payload["model_config"]),
        ModelKind(payload["model_kind"]),
        int(payload["model_seed"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_base_model(model)
    return model


def _base_prediction_fields(
    *,
    checkpoint: Mapping[str, Any],
    split_name: str,
    evaluation_spec_sha256: str,
    input_inventory: InputInventory,
    executor_commit: str,
    executor_source_sha256: str,
) -> dict[str, Any]:
    return {
        "condition": checkpoint["condition"],
        "training_seed": checkpoint["training_seed"],
        "selected_step": checkpoint["selected_step"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "split_name": split_name,
        "executor_commit": executor_commit,
        "executor_source_sha256": executor_source_sha256,
        "input_sha256": input_inventory.sha256,
        "input_manifest_sha256": input_inventory.manifest_sha256,
        "evaluation_spec_sha256": evaluation_spec_sha256,
        "rng_seed": derive_uint64(
            "opm-v1.1.4",
            "stage1-forward",
            checkpoint["condition"],
            checkpoint["training_seed"],
            split_name,
        ),
    }


def _artifact_is_complete(path: Path, expected: Mapping[str, Any]) -> bool:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.exists() and not manifest_path.exists():
        return False
    if not path.is_file() or not manifest_path.is_file():
        raise ReconciliationError(f"partial artifact publication detected: {path}")
    manifest = read_json(manifest_path, "existing artifact manifest")
    verify_jsonl_artifact(path, manifest)
    for field in (
        "artifact_kind",
        "row_count",
        "identity_set_sha256",
        "condition",
        "training_seed",
        "checkpoint_sha256",
        "split_name",
        "evaluation_spec_sha256",
        "probe_spec_sha256",
        "intervention_specification_sha256",
    ):
        if field in expected and manifest.get(field) != expected[field]:
            raise ReconciliationError(f"resume artifact binding mismatch for {field}: {path}")
    if manifest.get("model_pre_sha256") != manifest.get("model_post_sha256"):
        raise ReconciliationError(f"resume artifact records a changed base model: {path}")
    return True


def _prediction_path(output_root: Path, checkpoint: Mapping[str, Any], split: str) -> Path:
    return (
        output_root
        / "predictions"
        / str(checkpoint["condition"])
        / f"seed-{checkpoint['training_seed']}"
        / f"{split}.jsonl"
    )


def _intervention_path(
    output_root: Path, checkpoint: Mapping[str, Any], family: str, split: str
) -> Path:
    return (
        output_root
        / "interventions"
        / str(checkpoint["condition"])
        / f"seed-{checkpoint['training_seed']}"
        / family
        / f"{split}.jsonl"
    )


def _probe_path(output_root: Path, checkpoint: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "probes"
        / str(checkpoint["condition"])
        / f"seed-{checkpoint['training_seed']}"
        / "neural-evidence-vector.jsonl"
    )


def _identity_sha256(values: Iterable[str]) -> str:
    return canonical_sha256(sorted(values))


def _ablation_variants(condition: str) -> tuple[str, ...]:
    if condition == ModelKind.DOMAIN_GENERALIST.value:
        return (*(f"domain:{domain.name}" for domain in Domain), "sentinel")
    return (*(f"operation:{operation.name}" for operation in Operation), "sentinel")


def _replacement_variants(condition: str) -> tuple[tuple[str, str], ...]:
    if condition == ModelKind.DOMAIN_GENERALIST.value:
        labels = tuple(f"domain:{domain.name}" for domain in Domain)
    else:
        labels = tuple(f"operation:{operation.name}" for operation in Operation)
    return tuple((source, target) for source in labels for target in labels if source != target)


def _expected_artifacts(
    *,
    output_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    inventories: Sequence[InputInventory],
    interchange_pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    evaluation_spec_sha256: str,
    probe_spec_sha256: str,
    intervention_spec_sha256: str,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        binding = {
            "condition": checkpoint["condition"],
            "training_seed": checkpoint["training_seed"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        }
        probe_ids = [
            f"{checkpoint['condition']}:{checkpoint['training_seed']}:evidence-step-{step}"
            for step in (1, 2)
        ]
        expected.append(
            {
                "path": _probe_path(output_root, checkpoint).relative_to(output_root).as_posix(),
                "artifact_kind": "probe",
                "row_count": 2,
                "identity_set_sha256": _identity_sha256(probe_ids),
                "probe_spec_sha256": probe_spec_sha256,
                **binding,
            }
        )
        for inventory in inventories:
            prediction_path = _prediction_path(output_root, checkpoint, inventory.split_name)
            expected.append(
                {
                    "path": prediction_path.relative_to(output_root).as_posix(),
                    "artifact_kind": "prediction",
                    "row_count": inventory.row_count,
                    "identity_set_sha256": _identity_sha256(inventory.example_ids),
                    "split_name": inventory.split_name,
                    "evaluation_spec_sha256": evaluation_spec_sha256,
                    **binding,
                }
            )
            for family, variants in (
                ("ablation", _ablation_variants(str(checkpoint["condition"]))),
                (
                    "replacement",
                    tuple(
                        f"{source}->{target}"
                        for source, target in _replacement_variants(str(checkpoint["condition"]))
                    ),
                ),
                ("adapter-only", ("all-components-zero-delta",)),
            ):
                identities = (
                    f"{family}:{variant}:{example_id}"
                    for example_id in inventory.example_ids
                    for variant in variants
                )
                expected.append(
                    {
                        "path": _intervention_path(
                            output_root, checkpoint, family, inventory.split_name
                        )
                        .relative_to(output_root)
                        .as_posix(),
                        "artifact_kind": "intervention",
                        "row_count": inventory.row_count * len(variants),
                        "identity_set_sha256": _identity_sha256(identities),
                        "split_name": inventory.split_name,
                        "intervention_specification_sha256": intervention_spec_sha256,
                        **binding,
                    }
                )
        renderer = next(item for item in inventories if item.split_name == "test-renderer")
        surface_ids = (
            f"surface-reversal:renderer-v2:{example_id}" for example_id in renderer.example_ids
        )
        expected.append(
            {
                "path": _intervention_path(
                    output_root, checkpoint, "surface-reversal", "test-renderer"
                )
                .relative_to(output_root)
                .as_posix(),
                "artifact_kind": "intervention",
                "row_count": renderer.row_count,
                "identity_set_sha256": _identity_sha256(surface_ids),
                "split_name": "test-renderer",
                "intervention_specification_sha256": intervention_spec_sha256,
                **binding,
            }
        )
        interchange_ids = (
            f"interchange:{source['example_id']}->{target['example_id']}"
            for source, target in interchange_pairs
        )
        expected.append(
            {
                "path": _intervention_path(output_root, checkpoint, "interchange", "renderer-pairs")
                .relative_to(output_root)
                .as_posix(),
                "artifact_kind": "intervention",
                "row_count": len(interchange_pairs),
                "identity_set_sha256": _identity_sha256(interchange_ids),
                "split_name": "renderer-pairs",
                "intervention_specification_sha256": intervention_spec_sha256,
                **binding,
            }
        )
    return expected


def _build_interchange_pairs(
    inventories: Sequence[InputInventory], workspace: Path
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for inventory in inventories:
        path = workspace / inventory.path
        for row in _read_rows(path):
            if list(row.get("step_mask", [])) != [True, True]:
                continue
            key = (
                row["world_id"],
                tuple(row["query"]),
                tuple(row["operation_ids"]),
                tuple(row["step_mask"]),
            )
            row = dict(row)
            row["_split_name"] = inventory.split_name
            row["_input_sha256"] = inventory.sha256
            row["_input_manifest_sha256"] = inventory.manifest_sha256
            grouped[key].append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: str(row["example_id"]))
        for source in ordered:
            for target in ordered:
                if source["example_id"] == target["example_id"]:
                    continue
                if source["domain"] == target["domain"]:
                    continue
                if source["renderer_variant"] == target["renderer_variant"]:
                    continue
                pairs.append((source, target))
    pairs.sort(key=lambda pair: (pair[0]["example_id"], pair[1]["example_id"]))
    if not pairs:
        raise IntegrityError("frozen inputs yield no protocol-defined interchange pairs")
    return pairs


def _group_pairs_by_fact_shape(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[tuple[int, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    grouped: dict[tuple[int, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for source, target in pairs:
        shape = (len(source["rendered_facts"]), len(target["rendered_facts"]))
        grouped[shape].append((source, target))
    return dict(grouped)


def _model_modules(model: OPMModel, label: str) -> list[nn.Module]:
    if model.kind == ModelKind.DOMAIN_GENERALIST:
        if label == "sentinel":
            return []
        domain = Domain[label.split(":", 1)[1]]
        return [model.generalists[int(domain)]]
    if label == "sentinel":
        active = set(model.operation_to_primitive)
        sentinel = next(
            index for index in range(model.config.primitive_count) if index not in active
        )
        domains = (0,) if model.kind == ModelKind.OPM_SHARED else range(3)
        return [
            model.primitives[domain * model.config.primitive_count + sentinel] for domain in domains
        ]
    operation = Operation[label.split(":", 1)[1]]
    primitive = model.operation_to_primitive[int(operation)]
    domains = (0,) if model.kind == ModelKind.OPM_SHARED else range(3)
    return [
        model.primitives[domain * model.config.primitive_count + primitive] for domain in domains
    ]


@contextmanager
def _zero_delta(model: OPMModel, labels: Sequence[str]) -> Iterator[None]:
    handles = []

    def zero(_module, _inputs, output):
        return torch.zeros_like(output)

    for label in labels:
        for module in _model_modules(model, label):
            handles.append(module.output.register_forward_hook(zero))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _replace_component(model: OPMModel, source: str, target: str) -> Iterator[None]:
    sources = _model_modules(model, source)
    targets = _model_modules(model, target)
    if len(sources) != len(targets):
        raise IntegrityError("replacement component cardinality mismatch")
    handles = []
    for source_module, target_module in zip(sources, targets, strict=True):

        def replace(_module, inputs, _output, replacement=target_module):
            return replacement(*inputs)

        handles.append(source_module.register_forward_hook(replace))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _all_component_labels(model: OPMModel) -> tuple[str, ...]:
    if model.kind == ModelKind.DOMAIN_GENERALIST:
        return tuple(f"domain:{domain.name}" for domain in Domain)
    return tuple(f"operation:{operation.name}" for operation in Operation)


@torch.inference_mode()
def _interchange_logits(model: OPMModel, source: OPMBatch, target: OPMBatch) -> torch.Tensor:
    source_facts, source_state = model._encode(source)
    target_facts, target_state = model._encode(target)
    batch_indices = torch.arange(source_state.shape[0], device=source_state.device)
    source_evidence = source_facts[batch_indices, source.evidence_indices[:, 0]]
    target_evidence = target_facts[batch_indices, target.evidence_indices[:, 0]]
    source_state = model._transition(
        source_state, source_evidence, source.operation_ids[:, 0], source.domain_ids
    )
    target_state = model._transition(
        target_state, target_evidence, target.operation_ids[:, 0], target.domain_ids
    )
    del target_state
    destination_evidence = target_facts[batch_indices, target.evidence_indices[:, 1]]
    swapped = model._transition(
        source_state, destination_evidence, target.operation_ids[:, 1], target.domain_ids
    )
    return model.decoder(model.decoder_norm(swapped))


def _prediction_rows(
    rows: Sequence[Mapping[str, Any]], logits: torch.Tensor, common: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    values = logits.detach().float().cpu().tolist()
    for row, scores in zip(rows, values, strict=True):
        yield {
            "artifact_schema_version": PREDICTION_SCHEMA,
            **dict(common),
            "example_id": row["example_id"],
            "prediction": int(scores[1] > scores[0]),
            "logits_or_scores": [float(scores[0]), float(scores[1])],
        }


def _intervention_rows(
    rows: Sequence[Mapping[str, Any]],
    logits: torch.Tensor,
    common: Mapping[str, Any],
    *,
    family: str,
    variant: str,
    location: str,
    component: str,
    replacement_source: str | None,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    values = logits.detach().float().cpu().tolist()
    for index, (row, scores) in enumerate(zip(rows, values, strict=True)):
        source = source_rows[index] if source_rows is not None else None
        identity = (
            f"interchange:{source['example_id']}->{row['example_id']}"
            if source is not None
            else f"{family}:{variant}:{row['example_id']}"
        )
        yield {
            "artifact_schema_version": INTERVENTION_SCHEMA,
            **dict(common),
            "example_id": row["example_id"],
            "prediction": int(scores[1] > scores[0]),
            "logits_or_scores": [float(scores[0]), float(scores[1])],
            "intervention_record_id": identity,
            "source_checkpoint": common["checkpoint_sha256"],
            "target_checkpoint": common["checkpoint_sha256"],
            "source_example": source["example_id"] if source is not None else row["example_id"],
            "target_example": row["example_id"],
            "intervention_family": family,
            "intervention_variant": variant,
            "intervention_location": location,
            "component_or_primitive_identity": component,
            "replacement_source": replacement_source,
            "intervention_specification_sha256": common["intervention_specification_sha256"],
        }


def _collect_probe_features(
    *,
    model: OPMModel,
    path: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    gate: CapabilityGate,
) -> tuple[np.ndarray, np.ndarray]:
    gate.require(Capability.CANONICAL_ACTIVATION_CAPTURE)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _rows, batch in _iter_batches(path, batch_size, vocabulary, two_step_only=True):
        if batch.labels is None:
            raise AuthorizationError("canonical probe input lacks authorized probe targets")
        batch = batch_to_device(batch, device)
        encoded = model.encode_selected_evidence(batch).detach().float().cpu().numpy()
        features.append(encoded)
        labels.append(batch.labels.detach().cpu().numpy())
    if not features:
        raise IntegrityError("canonical probe input has no two-step examples")
    return np.concatenate(features), np.concatenate(labels)


def _fit_probe(
    *,
    model: OPMModel,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    condition: str,
    training_seed: int,
    evidence_step: int,
    device: torch.device,
    gate: CapabilityGate,
) -> dict[str, Any]:
    gate.require(Capability.CANONICAL_PROBE_FIT)
    seed = derive_uint64("opm-v1", "neural-probe", condition, training_seed, evidence_step)
    torch.manual_seed(seed % (2**63 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**63 - 1))
    selected_train = train_features[:, evidence_step - 1, :]
    selected_validation = validation_features[:, evidence_step - 1, :]
    mean = selected_train.mean(axis=0, keepdims=True)
    scale = selected_train.std(axis=0, keepdims=True)
    scale[scale == 0] = 1
    train_x = torch.tensor((selected_train - mean) / scale, dtype=torch.float32, device=device)
    validation_x = torch.tensor(
        (selected_validation - mean) / scale, dtype=torch.float32, device=device
    )
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    probe = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 2)).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in probe.parameters()} or optimizer_ids & {
        id(parameter) for parameter in model.parameters()
    }:
        raise AuthorizationError("probe optimizer escaped the probe-only parameter scope")
    generator = torch.Generator(device=device).manual_seed(seed % (2**63 - 1))
    for _epoch in range(100):
        order = torch.randperm(len(train_x), generator=generator, device=device)
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(probe(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        predictions = probe(validation_x).argmax(dim=-1).cpu().numpy()
    correct = int(np.sum(predictions == validation_labels))
    count = len(validation_labels)
    low, high = wilson_interval(correct, count)
    weights = {
        name: tensor.detach().float().cpu().tolist()
        for name, tensor in sorted(probe.state_dict().items())
    }
    return {
        "probe_seed": seed,
        "train_count": len(train_labels),
        "validation_count": count,
        "correct_count": correct,
        "validation_accuracy": correct / count,
        "p_value": float(binomtest(correct, count, p=0.5, alternative="greater").pvalue),
        "wilson_low": low,
        "wilson_high": high,
        "standardization_mean": mean.reshape(-1).tolist(),
        "standardization_scale": scale.reshape(-1).tolist(),
        "probe_state": weights,
    }


def _run_probe_artifact(
    *,
    path: Path,
    expected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    model: OPMModel,
    model_hash: str,
    probe_spec: Mapping[str, Any],
    probe_spec_sha256: str,
    workspace: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    gate: CapabilityGate,
) -> None:
    if _artifact_is_complete(path, expected):
        return
    train_path = workspace / probe_spec["inputs"]["train"]["path"]
    validation_path = workspace / probe_spec["inputs"]["validation"]["path"]
    train_features, train_labels = _collect_probe_features(
        model=model,
        path=train_path,
        batch_size=batch_size,
        vocabulary=vocabulary,
        device=device,
        gate=gate,
    )
    validation_features, validation_labels = _collect_probe_features(
        model=model,
        path=validation_path,
        batch_size=batch_size,
        vocabulary=vocabulary,
        device=device,
        gate=gate,
    )
    with AtomicJsonlArtifact(path, artifact_kind="probe", expected_rows=2) as writer:
        for evidence_step in (1, 2):
            result = _fit_probe(
                model=model,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                condition=str(checkpoint["condition"]),
                training_seed=int(checkpoint["training_seed"]),
                evidence_step=evidence_step,
                device=device,
                gate=gate,
            )
            assert_base_model_unchanged(model, model_hash)
            writer.write(
                {
                    "artifact_schema_version": PROBE_SCHEMA,
                    "probe_id": (
                        f"{checkpoint['condition']}:{checkpoint['training_seed']}:"
                        f"evidence-step-{evidence_step}"
                    ),
                    "condition": checkpoint["condition"],
                    "training_seed": checkpoint["training_seed"],
                    "selected_step": checkpoint["selected_step"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "evidence_step": evidence_step,
                    "probe_spec_sha256": probe_spec_sha256,
                    "train_input_sha256": probe_spec["inputs"]["train"]["sha256"],
                    "validation_input_sha256": probe_spec["inputs"]["validation"]["sha256"],
                    "architecture": [192, 128, 2],
                    "activation": "GELU",
                    "optimizer": "AdamW",
                    "learning_rate": 0.001,
                    "batch_size": 256,
                    "epochs": 100,
                    "early_stopping": False,
                    "base_model_parameters_frozen": True,
                    "captured_activations_detached": True,
                    **result,
                }
            )
        writer.commit(
            {
                "condition": checkpoint["condition"],
                "training_seed": checkpoint["training_seed"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_step": checkpoint["selected_step"],
                "probe_spec_sha256": probe_spec_sha256,
                "model_pre_sha256": model_hash,
                "model_post_sha256": tensor_state_sha256(model),
            }
        )


def _run_prediction_artifact(
    *,
    path: Path,
    expected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    inventory: InputInventory,
    model: OPMModel,
    model_hash: str,
    workspace: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    common: Mapping[str, Any],
    gate: CapabilityGate,
) -> None:
    if _artifact_is_complete(path, expected):
        return
    gate.require(Capability.LABEL_BLIND_PREDICTION)
    with AtomicJsonlArtifact(
        path,
        artifact_kind="prediction",
        expected_rows=inventory.row_count,
        expected_example_ids=inventory.example_ids,
    ) as writer:
        for rows, batch in _iter_batches(workspace / inventory.path, batch_size, vocabulary):
            batch = batch_to_device(batch, device)
            with torch.inference_mode():
                logits = model(batch)
            for output in _prediction_rows(rows, logits, common):
                writer.write(output)
        writer.commit(
            {
                "condition": checkpoint["condition"],
                "training_seed": checkpoint["training_seed"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_step": checkpoint["selected_step"],
                "split_name": inventory.split_name,
                "evaluation_spec_sha256": common["evaluation_spec_sha256"],
                "model_pre_sha256": model_hash,
                "model_post_sha256": tensor_state_sha256(model),
            }
        )


def _run_standard_intervention(
    *,
    path: Path,
    expected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    inventory: InputInventory,
    model: OPMModel,
    model_hash: str,
    workspace: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    common: Mapping[str, Any],
    family: str,
    gate: CapabilityGate,
) -> None:
    if _artifact_is_complete(path, expected):
        return
    gate.require(Capability.INTERVENTION_FORWARD)
    if family == "ablation":
        variants: tuple[object, ...] = _ablation_variants(str(checkpoint["condition"]))
    elif family == "replacement":
        variants = _replacement_variants(str(checkpoint["condition"]))
    elif family == "adapter-only":
        variants = ("all-components-zero-delta",)
    else:
        raise ValueError(family)
    with AtomicJsonlArtifact(
        path, artifact_kind="intervention", expected_rows=int(expected["row_count"])
    ) as writer:
        for rows, batch in _iter_batches(workspace / inventory.path, batch_size, vocabulary):
            batch = batch_to_device(batch, device)
            for variant in variants:
                if family == "ablation":
                    label = str(variant)
                    context = _zero_delta(model, [label])
                    variant_name = label
                    component = label
                    replacement = "zero-delta"
                elif family == "replacement":
                    source, target = variant
                    context = _replace_component(model, source, target)
                    variant_name = f"{source}->{target}"
                    component = source
                    replacement = target
                else:
                    context = _zero_delta(model, _all_component_labels(model))
                    variant_name = "all-components-zero-delta"
                    component = "all-active-components"
                    replacement = "zero-delta"
                with context, torch.inference_mode():
                    logits = model(batch)
                for output in _intervention_rows(
                    rows,
                    logits,
                    common,
                    family=family,
                    variant=variant_name,
                    location="primitive-transition-delta",
                    component=component,
                    replacement_source=replacement,
                ):
                    writer.write(output)
        writer.commit(
            {
                "condition": checkpoint["condition"],
                "training_seed": checkpoint["training_seed"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_step": checkpoint["selected_step"],
                "split_name": inventory.split_name,
                "intervention_family": family,
                "intervention_specification_sha256": common["intervention_specification_sha256"],
                "model_pre_sha256": model_hash,
                "model_post_sha256": tensor_state_sha256(model),
            }
        )


def _run_surface_reversal(
    *,
    path: Path,
    expected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    inventory: InputInventory,
    model: OPMModel,
    model_hash: str,
    workspace: Path,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    common: Mapping[str, Any],
    gate: CapabilityGate,
) -> None:
    if _artifact_is_complete(path, expected):
        return
    gate.require(Capability.INTERVENTION_FORWARD)
    with AtomicJsonlArtifact(
        path, artifact_kind="intervention", expected_rows=int(expected["row_count"])
    ) as writer:
        for rows, batch in _iter_batches(
            workspace / inventory.path,
            batch_size,
            vocabulary,
            renderer_variant=2,
        ):
            batch = batch_to_device(batch, device)
            with torch.inference_mode():
                logits = model(batch)
            for output in _intervention_rows(
                rows,
                logits,
                common,
                family="surface-reversal",
                variant="renderer-v2",
                location="frozen-input-surface-rendering",
                component="V2_ALIASES",
                replacement_source="pre-generated-test-only-permutation",
            ):
                writer.write(output)
        writer.commit(
            {
                "condition": checkpoint["condition"],
                "training_seed": checkpoint["training_seed"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_step": checkpoint["selected_step"],
                "split_name": "test-renderer",
                "intervention_family": "surface-reversal",
                "intervention_specification_sha256": common["intervention_specification_sha256"],
                "model_pre_sha256": model_hash,
                "model_post_sha256": tensor_state_sha256(model),
            }
        )


def _run_interchange(
    *,
    path: Path,
    expected: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    model: OPMModel,
    model_hash: str,
    batch_size: int,
    vocabulary: Vocabulary,
    device: torch.device,
    common: Mapping[str, Any],
    gate: CapabilityGate,
) -> None:
    if _artifact_is_complete(path, expected):
        return
    gate.require(Capability.INTERVENTION_FORWARD)
    with AtomicJsonlArtifact(
        path, artifact_kind="intervention", expected_rows=len(pairs)
    ) as writer:
        shape_groups = _group_pairs_by_fact_shape(pairs)
        for shape in sorted(shape_groups):
            shaped_pairs = shape_groups[shape]
            for start in range(0, len(shaped_pairs), batch_size):
                selected = shaped_pairs[start : start + batch_size]
                source_rows = [dict(pair[0]) for pair in selected]
                target_rows = [dict(pair[1]) for pair in selected]
                source_batch = batch_to_device(_batch_from_rows(source_rows, vocabulary), device)
                target_batch = batch_to_device(_batch_from_rows(target_rows, vocabulary), device)
                logits = _interchange_logits(model, source_batch, target_batch)
                batch_common = {
                    **common,
                    "split_name": "renderer-pairs",
                    "input_sha256": canonical_sha256(
                        sorted({str(row["_input_sha256"]) for row in [*source_rows, *target_rows]})
                    ),
                    "input_manifest_sha256": canonical_sha256(
                        sorted(
                            {
                                str(row["_input_manifest_sha256"])
                                for row in [*source_rows, *target_rows]
                            }
                        )
                    ),
                }
                for output in _intervention_rows(
                    target_rows,
                    logits,
                    batch_common,
                    family="interchange",
                    variant="cross-domain-renderer-state-swap",
                    location="before-last-primitive-call",
                    component="normalized-execution-state",
                    replacement_source="source-example-state-after-step-1",
                    source_rows=source_rows,
                ):
                    writer.write(output)
        writer.commit(
            {
                "condition": checkpoint["condition"],
                "training_seed": checkpoint["training_seed"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "selected_step": checkpoint["selected_step"],
                "split_name": "renderer-pairs",
                "intervention_family": "interchange",
                "intervention_specification_sha256": common["intervention_specification_sha256"],
                "model_pre_sha256": model_hash,
                "model_post_sha256": tensor_state_sha256(model),
            }
        )


def execute_stage1(
    *,
    workspace: Path,
    transition_path: Path,
    transition_sha256: str,
    checkpoint_manifest_path: Path,
    evaluation_spec_path: Path,
    probe_spec_path: Path,
    intervention_spec_path: Path,
    resource_manifest_path: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    paths = [
        transition_path,
        checkpoint_manifest_path,
        evaluation_spec_path,
        probe_spec_path,
        intervention_spec_path,
        resource_manifest_path,
        output_root,
    ]
    resolved = [path if path.is_absolute() else workspace / path for path in paths]
    (
        transition_path,
        checkpoint_manifest_path,
        evaluation_spec_path,
        probe_spec_path,
        intervention_spec_path,
        resource_manifest_path,
        output_root,
    ) = (path.resolve() for path in resolved)
    preflight, gate, checkpoints, inventories = preflight_stage1(
        workspace=workspace,
        transition_path=transition_path,
        expected_transition_sha256=transition_sha256,
        checkpoint_manifest_path=checkpoint_manifest_path,
        evaluation_spec_path=evaluation_spec_path,
        probe_spec_path=probe_spec_path,
        intervention_spec_path=intervention_spec_path,
        resource_manifest_path=resource_manifest_path,
        output_root=output_root,
    )
    resources = ResourcePolicy.load(resource_manifest_path, workspace)
    resources.require_write(output_root)
    evaluation_spec = read_json(evaluation_spec_path, "evaluation spec")
    probe_spec = read_json(probe_spec_path, "probe spec")
    read_json(intervention_spec_path, "intervention spec")
    evaluation_spec_sha256 = sha256_file(evaluation_spec_path)
    probe_spec_sha256 = sha256_file(probe_spec_path)
    intervention_spec_sha256 = sha256_file(intervention_spec_path)
    transition = read_json(transition_path, "transition")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise IntegrityError("requested CUDA device is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    vocabulary = Vocabulary.build()
    log = ExecutionLog(output_root / "execution.log.jsonl")
    log.write("STAGE1_PREFLIGHT_PASS", **asdict(preflight))
    environment_path = output_root / "environment.json"
    environment = environment_record(device_name)
    if environment_path.exists():
        if read_json(environment_path, "environment") != environment:
            raise IntegrityError("resume environment differs from the frozen job environment")
    else:
        atomic_write_json(environment_path, environment)
    interchange_pairs = _build_interchange_pairs(inventories, workspace)
    expected = _expected_artifacts(
        output_root=output_root,
        checkpoints=checkpoints,
        inventories=inventories,
        interchange_pairs=interchange_pairs,
        evaluation_spec_sha256=evaluation_spec_sha256,
        probe_spec_sha256=probe_spec_sha256,
        intervention_spec_sha256=intervention_spec_sha256,
    )
    plan = {
        "schema_version": "opm-stage1-artifact-plan-v1",
        "job_id": preflight.job_id,
        "checkpoint_count": len(checkpoints),
        "artifact_count": len(expected),
        "interchange_pair_count": len(interchange_pairs),
        "artifacts": expected,
    }
    plan_path = output_root / "artifact-plan.json"
    if plan_path.exists():
        if read_json(plan_path, "artifact plan") != plan:
            raise IntegrityError("resume artifact plan differs from the frozen job")
    else:
        atomic_write_json(plan_path, plan)
    expected_by_path = {str(item["path"]): item for item in expected}
    executor_commit = git_commit(workspace)
    executor_source_sha256 = str(transition["executor_source_aggregate_sha256"])
    batch_size = int(evaluation_spec["batch_size"])

    for checkpoint in checkpoints:
        log.write(
            "CHECKPOINT_START",
            condition=checkpoint["condition"],
            training_seed=checkpoint["training_seed"],
            checkpoint_sha256=checkpoint["checkpoint_sha256"],
        )
        model = _load_model(
            checkpoint,
            workspace=workspace,
            resources=resources,
            gate=gate,
            device=device,
        )
        model_hash = tensor_state_sha256(model)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AuthorizationError("base model is trainable after checkpoint load")
        probe_path = _probe_path(output_root, checkpoint)
        _run_probe_artifact(
            path=probe_path,
            expected=expected_by_path[probe_path.relative_to(output_root).as_posix()],
            checkpoint=checkpoint,
            model=model,
            model_hash=model_hash,
            probe_spec=probe_spec,
            probe_spec_sha256=probe_spec_sha256,
            workspace=workspace,
            batch_size=batch_size,
            vocabulary=vocabulary,
            device=device,
            gate=gate,
        )
        for inventory in inventories:
            common = _base_prediction_fields(
                checkpoint=checkpoint,
                split_name=inventory.split_name,
                evaluation_spec_sha256=evaluation_spec_sha256,
                input_inventory=inventory,
                executor_commit=executor_commit,
                executor_source_sha256=executor_source_sha256,
            )
            prediction_path = _prediction_path(output_root, checkpoint, inventory.split_name)
            _run_prediction_artifact(
                path=prediction_path,
                expected=expected_by_path[prediction_path.relative_to(output_root).as_posix()],
                checkpoint=checkpoint,
                inventory=inventory,
                model=model,
                model_hash=model_hash,
                workspace=workspace,
                batch_size=batch_size,
                vocabulary=vocabulary,
                device=device,
                common=common,
                gate=gate,
            )
            intervention_common = {
                **common,
                "intervention_specification_sha256": intervention_spec_sha256,
            }
            for family in ("ablation", "replacement", "adapter-only"):
                path = _intervention_path(output_root, checkpoint, family, inventory.split_name)
                _run_standard_intervention(
                    path=path,
                    expected=expected_by_path[path.relative_to(output_root).as_posix()],
                    checkpoint=checkpoint,
                    inventory=inventory,
                    model=model,
                    model_hash=model_hash,
                    workspace=workspace,
                    batch_size=batch_size,
                    vocabulary=vocabulary,
                    device=device,
                    common=intervention_common,
                    family=family,
                    gate=gate,
                )
        renderer = next(item for item in inventories if item.split_name == "test-renderer")
        renderer_common = {
            **_base_prediction_fields(
                checkpoint=checkpoint,
                split_name="test-renderer",
                evaluation_spec_sha256=evaluation_spec_sha256,
                input_inventory=renderer,
                executor_commit=executor_commit,
                executor_source_sha256=executor_source_sha256,
            ),
            "intervention_specification_sha256": intervention_spec_sha256,
        }
        surface_path = _intervention_path(
            output_root, checkpoint, "surface-reversal", "test-renderer"
        )
        _run_surface_reversal(
            path=surface_path,
            expected=expected_by_path[surface_path.relative_to(output_root).as_posix()],
            checkpoint=checkpoint,
            inventory=renderer,
            model=model,
            model_hash=model_hash,
            workspace=workspace,
            batch_size=batch_size,
            vocabulary=vocabulary,
            device=device,
            common=renderer_common,
            gate=gate,
        )
        interchange_path = _intervention_path(
            output_root, checkpoint, "interchange", "renderer-pairs"
        )
        _run_interchange(
            path=interchange_path,
            expected=expected_by_path[interchange_path.relative_to(output_root).as_posix()],
            checkpoint=checkpoint,
            pairs=interchange_pairs,
            model=model,
            model_hash=model_hash,
            batch_size=batch_size,
            vocabulary=vocabulary,
            device=device,
            common=renderer_common,
            gate=gate,
        )
        assert_base_model_unchanged(model, model_hash)
        checkpoint_path = workspace / str(checkpoint["checkpoint_path"])
        if sha256_file(checkpoint_path) != checkpoint["checkpoint_sha256"]:
            raise IntegrityError("checkpoint bytes changed during Stage 1 execution")
        log.write(
            "CHECKPOINT_COMPLETE",
            condition=checkpoint["condition"],
            training_seed=checkpoint["training_seed"],
            model_pre_sha256=model_hash,
            model_post_sha256=tensor_state_sha256(model),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log.write("GENERATION_COMPLETE", artifact_count=len(expected))
    reconciliation, freeze = reconcile_and_freeze(
        output_root=output_root,
        transition_sha256=transition_sha256,
        checkpoints=checkpoints,
        expected_artifacts=expected,
        environment_path=environment_path,
        execution_log_path=log.path,
        gate=gate,
    )
    return {
        "state": freeze["state"],
        "job_id": preflight.job_id,
        "checkpoint_count": len(checkpoints),
        "artifact_count": freeze["artifact_count"],
        "merkle_root": freeze["merkle_root"],
        "freeze_path": str(output_root / FREEZE_FILENAME),
        "freeze_sha256": sha256_file(output_root / FREEZE_FILENAME),
        "reconciliation_state": reconciliation["state"],
        "sealed_label_access_count": 0,
        "aggregate_test_evaluation_performed": False,
        "claim_decisions_performed": False,
    }


def _add_common_authorization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--authorization-directory", type=Path, default=DEFAULT_AUTHORIZATION_DIRECTORY
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed OPM v1.1.4 post-primary Stage 1 executor."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record-transition")
    _add_common_authorization_arguments(record)
    record.add_argument("--primary-matrix", type=Path, default=DEFAULT_PRIMARY_MATRIX)
    record.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    record.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    record.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)

    for name in ("preflight", "execute", "deny-aggregate", "deny-claim"):
        command = commands.add_parser(name)
        _add_common_authorization_arguments(command)
        command.add_argument("--transition-sha256", required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "execute":
            command.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _bound_paths(args: argparse.Namespace) -> dict[str, Path]:
    authorization = args.authorization_directory
    paths = _authorization_paths(authorization)
    return {
        "transition_path": paths["transition"],
        "checkpoint_manifest_path": paths["checkpoints"],
        "evaluation_spec_path": paths["evaluation"],
        "probe_spec_path": paths["probe"],
        "intervention_spec_path": paths["intervention"],
        "resource_manifest_path": paths["resources"],
    }


def main() -> None:
    args = parse_args()
    if args.command == "record-transition":
        result = record_transition(
            workspace=args.workspace,
            primary_matrix_path=args.primary_matrix,
            handoff_path=args.handoff,
            input_directory=args.input_directory,
            authorization_directory=args.authorization_directory,
            stage1_root=args.stage1_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    common = {
        "workspace": args.workspace,
        "expected_transition_sha256": args.transition_sha256,
        "output_root": args.output_root,
        **_bound_paths(args),
    }
    if args.command == "preflight":
        preflight, gate, _checkpoints, _inventories = preflight_stage1(**common)
        print(json.dumps({"preflight": asdict(preflight), "capabilities": gate.enabled}, indent=2))
        return
    if args.command == "execute":
        common["transition_sha256"] = common.pop("expected_transition_sha256")
        result = execute_stage1(**common, device_name=args.device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    transition = read_json(common["transition_path"], "transition")
    if sha256_file(common["transition_path"]) != args.transition_sha256:
        raise AuthorizationError("invalid transition hash")
    gate = CapabilityGate.from_transition(transition, args.transition_sha256)
    if args.command == "deny-aggregate":
        refuse_aggregate_evaluation(gate)
    else:
        refuse_claim_decision(gate)


if __name__ == "__main__":
    main()
