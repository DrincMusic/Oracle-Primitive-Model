from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .data import MaterializedExample, materialize
from .generation import (
    Corruption,
    Example,
    Operation,
    derive_uint64,
    enumerate_negative_candidates,
    enumerate_positive_templates,
    generate_world,
)
from .protocol import CURRENT_PROTOCOL
from .rendering import Domain


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST_INTERPOLATION = "test-interpolation"
    TEST_RECOMBINATION = "test-recombination"
    TEST_RENDERER = "test-renderer"
    TEST_STRUCTURAL = "test-structural"


HOLDOUTS: dict[Domain, Operation] = {
    Domain.SET: Operation.REVERSE,
    Domain.SCENE: Operation.LIFT,
    Domain.PROGRAM: Operation.CHAIN,
}


@dataclass(frozen=True)
class SplitValidationConfig:
    split: SplitName
    examples_per_cell: int = 2
    maximum_world_indices: int = 2_000
    seed_prefix: str = "opm-v1"

    def __post_init__(self) -> None:
        if self.examples_per_cell <= 0 or self.examples_per_cell > 100:
            raise ValueError("validation examples_per_cell must be in 1..100")


CANONICAL_EXAMPLES_PER_CELL: dict[SplitName, int] = {
    SplitName.TRAIN: 8_000,
    SplitName.VALIDATION: 800,
    SplitName.TEST_INTERPOLATION: 800,
    SplitName.TEST_RECOMBINATION: 3_000,
    SplitName.TEST_RENDERER: 400,
    SplitName.TEST_STRUCTURAL: 400,
}

CANONICAL_MAXIMUM_WORLD_INDICES: dict[SplitName, int] = {
    SplitName.TRAIN: 100_000,
    SplitName.VALIDATION: 20_000,
    SplitName.TEST_INTERPOLATION: 20_000,
    SplitName.TEST_RECOMBINATION: 20_000,
    SplitName.TEST_RENDERER: 20_000,
    SplitName.TEST_STRUCTURAL: 20_000,
}


@dataclass(frozen=True)
class CanonicalSplitConfig:
    split: SplitName

    @property
    def examples_per_cell(self) -> int:
        return CANONICAL_EXAMPLES_PER_CELL[self.split]

    @property
    def maximum_world_indices(self) -> int:
        return CANONICAL_MAXIMUM_WORLD_INDICES[self.split]

    @property
    def seed_prefix(self) -> str:
        return "opm-v1"

    @property
    def binding_version(self) -> str:
        return "1.0"


@dataclass(frozen=True)
class AmendedCanonicalSplitConfig:
    split: SplitName

    @property
    def examples_per_cell(self) -> int:
        return CANONICAL_EXAMPLES_PER_CELL[self.split]

    @property
    def maximum_world_indices(self) -> int:
        return CANONICAL_MAXIMUM_WORLD_INDICES[self.split]

    @property
    def seed_prefix(self) -> str:
        return "opm-v1.1"

    @property
    def binding_version(self) -> str:
        return "1.1.1"

    @property
    def evidence_order_version(self) -> str:
        return "1.0"


@dataclass(frozen=True)
class EvidenceOrderCanonicalSplitConfig:
    split: SplitName

    @property
    def examples_per_cell(self) -> int:
        return CANONICAL_EXAMPLES_PER_CELL[self.split]

    @property
    def maximum_world_indices(self) -> int:
        return CANONICAL_MAXIMUM_WORLD_INDICES[self.split]

    @property
    def seed_prefix(self) -> str:
        return "opm-v1.1.3"

    @property
    def binding_version(self) -> str:
        return "1.1.1"

    @property
    def evidence_order_version(self) -> str:
        return "1.1.3"


@dataclass(frozen=True)
class DiagnosticSplitConfig:
    """Canonical-sized allocation in a noncanonical namespace for diagnosis only."""

    split: SplitName
    seed_prefix: str

    def __post_init__(self) -> None:
        if self.split != SplitName.VALIDATION:
            raise ValueError("diagnostic replicas are restricted to validation distributions")
        if not self.seed_prefix.startswith("opm-v1-diagnostic/"):
            raise ValueError("diagnostic namespace must be explicit")

    @property
    def examples_per_cell(self) -> int:
        return CANONICAL_EXAMPLES_PER_CELL[self.split]

    @property
    def maximum_world_indices(self) -> int:
        return CANONICAL_MAXIMUM_WORLD_INDICES[self.split]

    @property
    def binding_version(self) -> str:
        return "1.0"


@dataclass(frozen=True)
class SplitManifest:
    split: str
    specification_version: str
    validation_only: bool
    row_count: int
    sha256: str
    cell_counts: dict[str, int]
    seed_namespace: str
    generator_revision: str = "opm-v1-spec-1.0.0"
    renderer_revision: str = "opm-v1-spec-1.0.0"
    holdout_matrix: dict[str, str] | None = None
    labels_separated: bool = False
    labels_sha256: str | None = None
    amendment_version: str | None = None
    binding_version: str = "1.0"
    evidence_order_version: str = "1.0"


def allowed_operations(split: SplitName, domain: Domain) -> tuple[Operation, ...]:
    if split == SplitName.TEST_RECOMBINATION:
        return (HOLDOUTS[domain],)
    if split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST_INTERPOLATION):
        return tuple(operation for operation in Operation if operation != HOLDOUTS[domain])
    return tuple(Operation)


def renderer_variant(split: SplitName, example: Example, domain: Domain) -> int:
    if split == SplitName.TEST_RENDERER:
        return 2
    bit = derive_uint64("renderer", split.value, example.world_id, domain.name, example.canonical_key()) & 1
    return int(bit)


def _cell(domain: Domain, operation: Operation, label: int) -> tuple[Domain, Operation, int]:
    return domain, operation, label


def _candidate_rank(split: SplitName, domain: Domain, example: Example) -> str:
    material = f"{split.value}/{domain.name}/{example.canonical_key()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _collect_candidates(
    config: SplitValidationConfig
    | CanonicalSplitConfig
    | AmendedCanonicalSplitConfig
    | EvidenceOrderCanonicalSplitConfig
    | DiagnosticSplitConfig,
) -> dict[tuple[Domain, Operation, int], list[tuple[object, Example]]]:
    buckets: dict[tuple[Domain, Operation, int], list[tuple[object, Example]]] = defaultdict(list)
    required_cells = {
        _cell(domain, operation, label)
        for domain in Domain
        for operation in allowed_operations(config.split, domain)
        for label in (0, 1)
    }
    seen: dict[tuple[Domain, Operation, int], set[tuple[object, ...]]] = defaultdict(set)
    for world_index in range(config.maximum_world_indices):
        seed = derive_uint64(config.seed_prefix, config.split.value, "world", world_index)
        if config.split == SplitName.TEST_STRUCTURAL:
            size_seed = derive_uint64(seed, "structural-size")
            n_objects = 12 + size_seed % 5
            n_containers = 6 + (size_seed // 5) % 3
            world = generate_world(seed, int(n_objects), int(n_containers))
        else:
            world = generate_world(seed)
        positives = enumerate_positive_templates(world)
        candidates = list(positives)
        candidates.extend(
            negative
            for positive in positives
            for negative in enumerate_negative_candidates(world, positive)
        )
        for domain in Domain:
            allowed = set(allowed_operations(config.split, domain))
            for example in candidates:
                operation = example.operations[0]
                if operation not in allowed:
                    continue
                cell = _cell(domain, operation, example.label)
                key = example.canonical_key()
                if key not in seen[cell]:
                    seen[cell].add(key)
                    buckets[cell].append((world, example))
        if world_index % 100 == 99 and all(
            len(buckets[cell]) >= config.examples_per_cell for cell in required_cells
        ):
            try:
                for domain, operation, label in required_cells:
                    items = buckets[_cell(domain, operation, label)]
                    if label == 0:
                        _choose_negative_examples(
                            items, config.examples_per_cell, config.split, domain, operation
                        )
                    else:
                        _choose_positive_examples(
                            items, config.examples_per_cell, config.split, domain, operation
                        )
            except RuntimeError:
                continue
            break
    missing = [cell for cell in sorted(required_cells) if not buckets[cell]]
    if missing:
        raise RuntimeError(f"unable to populate split cells: {missing}")
    return buckets


def _choose_negative_examples(
    items: list[tuple[object, Example]],
    count: int,
    split: SplitName,
    domain: Domain,
    operation: Operation,
) -> list[tuple[object, Example]]:
    by_relation: dict[str, list[tuple[object, Example]]] = defaultdict(list)
    for item in items:
        by_relation[item[1].query.relation.name].append(item)
    relations = _relation_order(operation)
    relation_base, relation_remainder = divmod(count, len(relations))
    chosen: list[tuple[object, Example]] = []
    for relation_index, relation in enumerate(relations):
        relation_quota = relation_base + (1 if relation_index < relation_remainder else 0)
        relation_items = by_relation[relation]
        by_corruption: dict[Corruption, list[tuple[object, Example]]] = defaultdict(list)
        for item in relation_items:
            by_corruption[item[1].corruption].append(item)
        corruption_types = _valid_corruptions(operation, relation)
        base, remainder = divmod(relation_quota, len(corruption_types))
        for index, corruption in enumerate(corruption_types):
            quota = base + (1 if index < remainder else 0)
            ranked = sorted(
                by_corruption[corruption],
                key=lambda item: _candidate_rank(split, domain, item[1]),
            )
            if len(ranked) < quota:
                raise RuntimeError(
                    f"corruption quota unavailable: {domain.name}/{operation.name}/"
                    f"{relation}/{corruption.name}"
                )
            chosen.extend(ranked[:quota])
    return chosen


def _relation_order(operation: Operation) -> tuple[str, ...]:
    if operation == Operation.LOOKUP:
        return ("BEFORE", "LINK", "SAME", "WITHIN")
    if operation in (Operation.REVERSE, Operation.CHAIN):
        return ("BEFORE", "SAME") if operation == Operation.CHAIN else ("LINK", "SAME")
    return ("WITHIN",)


def _valid_corruptions(operation: Operation, relation: str) -> tuple[Corruption, ...]:
    if operation in (Operation.LOOKUP, Operation.REVERSE):
        corruptions = [Corruption.DIRECT_MISMATCH, Corruption.ENDPOINT]
        if operation == Operation.LOOKUP and relation == "BEFORE":
            corruptions.append(Corruption.ORDER)
        return tuple(sorted(corruptions))
    corruptions = [Corruption.ENDPOINT, Corruption.MIDDLE]
    if operation == Operation.CHAIN and relation == "BEFORE":
        corruptions.append(Corruption.ORDER)
    return tuple(sorted(corruptions))


def _choose_positive_examples(
    items: list[tuple[object, Example]],
    count: int,
    split: SplitName,
    domain: Domain,
    operation: Operation,
) -> list[tuple[object, Example]]:
    relation_order = _relation_order(operation)
    by_relation: dict[str, list[tuple[object, Example]]] = defaultdict(list)
    for item in items:
        by_relation[item[1].query.relation.name].append(item)
    base, remainder = divmod(count, len(relation_order))
    chosen: list[tuple[object, Example]] = []
    for index, relation in enumerate(relation_order):
        quota = base + (1 if index < remainder else 0)
        ranked = sorted(
            by_relation[relation], key=lambda item: _candidate_rank(split, domain, item[1])
        )
        if len(ranked) < quota:
            raise RuntimeError(f"relation quota unavailable: {domain.name}/{operation.name}/{relation}")
        chosen.extend(ranked[:quota])
    return chosen


def allocate_split(
    config: SplitValidationConfig
    | CanonicalSplitConfig
    | AmendedCanonicalSplitConfig
    | EvidenceOrderCanonicalSplitConfig
    | DiagnosticSplitConfig,
) -> list[MaterializedExample]:
    """Allocate a deterministic validation or approved canonical split."""
    if isinstance(
        config,
        (CanonicalSplitConfig, AmendedCanonicalSplitConfig, EvidenceOrderCanonicalSplitConfig),
    ):
        CURRENT_PROTOCOL.require_dataset_construction()
    else:
        CURRENT_PROTOCOL.require_validation()
    buckets = _collect_candidates(config)
    records: list[MaterializedExample] = []
    for cell in sorted(buckets):
        domain, operation, label = cell
        items = buckets[cell]
        if label == 0:
            selected = _choose_negative_examples(
                items, config.examples_per_cell, config.split, domain, operation
            )
        else:
            selected = _choose_positive_examples(
                items, config.examples_per_cell, config.split, domain, operation
            )
        binding_indices: dict[tuple[object, ...], int] = {}
        if getattr(config, "binding_version", "1.0") == "1.1.1":
            by_relation: dict[str, list[tuple[object, Example]]] = defaultdict(list)
            for item in selected:
                by_relation[item[1].query.relation.name].append(item)
            for relation_items in by_relation.values():
                ranked = sorted(
                    relation_items,
                    key=lambda item: _candidate_rank(config.split, domain, item[1]),
                )
                for index, (_, example) in enumerate(ranked):
                    binding_indices[example.canonical_key()] = index
        for world, example in selected:
            records.append(
                materialize(
                    world,
                    example,
                    domain,
                    renderer_variant(config.split, example, domain),
                    fact_count=12 if config.split == SplitName.TEST_STRUCTURAL else 8,
                    binding_index=binding_indices.get(example.canonical_key()),
                    evidence_order_index=(
                        binding_indices.get(example.canonical_key())
                        if getattr(config, "evidence_order_version", "1.0") == "1.1.3"
                        else None
                    ),
                )
            )
    return sorted(records, key=lambda record: record.example_id)


def allocate_validation_split(config: SplitValidationConfig) -> list[MaterializedExample]:
    """Allocate a reduced, explicitly validation-only split."""
    return allocate_split(config)


def canonical_expected_rows(split: SplitName) -> int:
    cells = sum(len(allowed_operations(split, domain)) * 2 for domain in Domain)
    return cells * CANONICAL_EXAMPLES_PER_CELL[split]


def validate_split_collection(splits: dict[SplitName, list[MaterializedExample]]) -> None:
    """Reject cross-split world/latent duplication and split-policy contamination."""
    world_owner: dict[int, SplitName] = {}
    latent_owner: dict[tuple[object, ...], SplitName] = {}
    for split, records in splits.items():
        for record in records:
            previous_world = world_owner.setdefault(record.world_id, split)
            if previous_world != split:
                raise ValueError(f"world_id contamination: {record.world_id}")
            latent = (
                tuple(record.facts),
                record.query,
                tuple(record.operation_ids),
            )
            previous_latent = latent_owner.setdefault(latent, split)
            if previous_latent != split:
                raise ValueError(f"latent example contamination: {record.example_id}")
            operation = Operation(record.operation_ids[0])
            if split == SplitName.TRAIN:
                if record.renderer_variant == 2:
                    raise ValueError("renderer variant 2 is prohibited in training")
                if operation == HOLDOUTS[record.domain]:
                    raise ValueError("recombination holdout present in training")
            if operation not in allowed_operations(split, record.domain):
                raise ValueError(f"undeclared operation in {split.value}: {operation.name}")


def record_dict(record: MaterializedExample) -> dict[str, object]:
    return {
        "example_id": record.example_id,
        "world_id": record.world_id,
        "domain": record.domain.name,
        "renderer_variant": record.renderer_variant,
        "facts": [
            [fact.relation.name, fact.arg1, fact.arg2] for fact in record.facts
        ],
        "rendered_facts": list(record.rendered_facts),
        "query": [record.query.relation.name, record.query.arg1, record.query.arg2],
        "rendered_query": record.rendered_query,
        "label": record.label,
        "argument_entity_ids": list(record.argument_entity_ids),
        "fact_endpoint_ids": [list(pair) for pair in record.fact_endpoint_ids],
        "evidence_indices": list(record.evidence_indices),
        "operation_ids": list(record.operation_ids),
        "step_mask": list(record.step_mask),
    }


def _manifest(
    config: SplitValidationConfig
    | CanonicalSplitConfig
    | AmendedCanonicalSplitConfig
    | EvidenceOrderCanonicalSplitConfig
    | DiagnosticSplitConfig,
    records: list[MaterializedExample],
    payload: bytes,
    *,
    validation_only: bool,
    labels_separated: bool = False,
    labels_sha256: str | None = None,
) -> SplitManifest:
    cell_counts: dict[str, int] = defaultdict(int)
    for record in records:
        operation = Operation(record.operation_ids[0]).name
        cell_counts[f"{record.domain.name}/{operation}/{record.label}"] += 1
    return SplitManifest(
        split=config.split.value,
        specification_version="1.0.0",
        validation_only=validation_only,
        row_count=len(records),
        sha256=hashlib.sha256(payload).hexdigest(),
        cell_counts=dict(sorted(cell_counts.items())),
        seed_namespace=f"{config.seed_prefix}/{config.split.value}/world/<index>",
        holdout_matrix={domain.name: operation.name for domain, operation in HOLDOUTS.items()},
        labels_separated=labels_separated,
        labels_sha256=labels_sha256,
        amendment_version=(
            (
                "1.1.3"
                if getattr(config, "evidence_order_version", "1.0") == "1.1.3"
                else "1.1.2"
            )
            if getattr(config, "binding_version", "1.0") == "1.1.1"
            else None
        ),
        binding_version=getattr(config, "binding_version", "1.0"),
        evidence_order_version=getattr(config, "evidence_order_version", "1.0"),
    )


def write_validation_split(
    config: SplitValidationConfig, output_directory: Path
) -> tuple[Path, Path, SplitManifest]:
    records = allocate_validation_split(config)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / f"{config.split.value}.validation.jsonl"
    lines = [json.dumps(record_dict(record), sort_keys=True, separators=(",", ":")) for record in records]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    data_path.write_bytes(payload)
    manifest = _manifest(config, records, payload, validation_only=True)
    manifest_path = output_directory / f"{config.split.value}.validation.manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", "utf-8")
    return data_path, manifest_path, manifest


def write_canonical_split(
    split: SplitName, output_directory: Path
) -> tuple[Path, Path | None, Path, SplitManifest]:
    """Write an approved-scale dataset; test labels are always kept out of input JSONL."""
    CURRENT_PROTOCOL.require_dataset_construction()
    config = CanonicalSplitConfig(split)
    records = allocate_split(config)
    if len(records) != canonical_expected_rows(split):
        raise RuntimeError(f"canonical row-count mismatch for {split.value}")
    output_directory.mkdir(parents=True, exist_ok=True)
    test_split = split not in (SplitName.TRAIN, SplitName.VALIDATION)
    input_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for record in records:
        row = record_dict(record)
        label = int(row.pop("label"))
        if not test_split:
            row["label"] = label
        else:
            label_rows.append({"example_id": record.example_id, "label": label})
        input_rows.append(row)
    payload = (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in input_rows)
        + "\n"
    ).encode("utf-8")
    data_path = output_directory / f"{split.value}.canonical.jsonl"
    data_path.write_bytes(payload)
    label_path: Path | None = None
    label_digest: str | None = None
    if test_split:
        label_payload = (
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in label_rows
            )
            + "\n"
        ).encode("utf-8")
        label_path = output_directory / "sealed-labels" / f"{split.value}.labels.jsonl"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_bytes(label_payload)
        label_digest = hashlib.sha256(label_payload).hexdigest()
    manifest = _manifest(
        config,
        records,
        payload,
        validation_only=False,
        labels_separated=test_split,
        labels_sha256=label_digest,
    )
    manifest_path = output_directory / f"{split.value}.canonical.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, label_path, manifest_path, manifest


def write_amended_canonical_split(
    split: SplitName, output_directory: Path
) -> tuple[Path, Path, SplitManifest]:
    """Write a fresh v1.1.1 train or validation artifact for implementation validation."""
    CURRENT_PROTOCOL.require_dataset_construction()
    if split not in (SplitName.TRAIN, SplitName.VALIDATION):
        raise PermissionError("v1.1.1 test-split generation remains blocked")
    config = AmendedCanonicalSplitConfig(split)
    records = allocate_split(config)
    if len(records) != canonical_expected_rows(split):
        raise RuntimeError(f"amended canonical row-count mismatch for {split.value}")
    rows = [record_dict(record) for record in records]
    payload = (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    ).encode("utf-8")
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / f"{split.value}.v1.1.canonical.jsonl"
    data_path.write_bytes(payload)
    manifest = _manifest(config, records, payload, validation_only=False)
    manifest_path = output_directory / f"{split.value}.v1.1.canonical.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, manifest_path, manifest


def write_v113_canonical_split(
    split: SplitName, output_directory: Path
) -> tuple[Path, Path, SplitManifest]:
    """Write a fresh v1.1.3 train or validation artifact for implementation validation."""
    CURRENT_PROTOCOL.require_dataset_construction()
    if split not in (SplitName.TRAIN, SplitName.VALIDATION):
        raise PermissionError("v1.1.3 test-split generation remains blocked")
    config = EvidenceOrderCanonicalSplitConfig(split)
    records = allocate_split(config)
    if len(records) != canonical_expected_rows(split):
        raise RuntimeError(f"v1.1.3 canonical row-count mismatch for {split.value}")
    rows = [record_dict(record) for record in records]
    payload = (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n"
    ).encode("utf-8")
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / f"{split.value}.v1.1.3.canonical.jsonl"
    data_path.write_bytes(payload)
    manifest = _manifest(config, records, payload, validation_only=False)
    manifest_path = output_directory / f"{split.value}.v1.1.3.canonical.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, manifest_path, manifest


def write_v113_test_split(
    split: SplitName, output_directory: Path
) -> tuple[Path, Path, Path, SplitManifest]:
    """Generate and immediately seal one owner-authorized v1.1.3 primary test split."""
    CURRENT_PROTOCOL.require_dataset_construction()
    if split in (SplitName.TRAIN, SplitName.VALIDATION):
        raise PermissionError("v1.1.3 sealed writer accepts primary test splits only")
    config = EvidenceOrderCanonicalSplitConfig(split)
    records = allocate_split(config)
    if len(records) != canonical_expected_rows(split):
        raise RuntimeError(f"v1.1.3 canonical row-count mismatch for {split.value}")

    input_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for record in records:
        row = record_dict(record)
        label = int(row.pop("label"))
        input_rows.append(row)
        label_rows.append({"example_id": record.example_id, "label": label})
    input_payload = (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in input_rows)
        + "\n"
    ).encode("utf-8")
    label_payload = (
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in label_rows)
        + "\n"
    ).encode("utf-8")

    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / f"{split.value}.v1.1.3.canonical.jsonl"
    label_path = output_directory / "sealed-labels" / f"{split.value}.v1.1.3.labels.jsonl"
    manifest_path = output_directory / f"{split.value}.v1.1.3.canonical.manifest.json"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(input_payload)
    label_path.write_bytes(label_payload)
    label_digest = hashlib.sha256(label_payload).hexdigest()
    manifest = _manifest(
        config,
        records,
        input_payload,
        validation_only=False,
        labels_separated=True,
        labels_sha256=label_digest,
    )
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    creation_record = {
        "action": "GENERATED_AND_SEALED",
        "split": split.value,
        "row_count": len(records),
        "input_sha256": manifest.sha256,
        "label_sha256": label_digest,
        "label_content_accessed_after_sealing": False,
        "training_executed": False,
        "evaluation_executed": False,
    }
    with (output_directory / "sealed-labels" / "seal-creation.audit.jsonl").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(creation_record, sort_keys=True, separators=(",", ":")) + "\n")
    return data_path, label_path, manifest_path, manifest
