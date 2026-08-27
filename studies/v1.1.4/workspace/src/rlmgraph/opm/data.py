from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .algebra import Fact, Query, Relation, World, entailed
from .generation import Example, Operation, canonical_json
from .model import OPMBatch
from .rendering import (
    PROGRAM_ALIASES,
    SCENE_ALIASES,
    SET_ALIASES,
    V2_ALIASES,
    Domain,
    fact_order_key,
    lex,
    render,
)

SPECIAL_TOKENS = ("[PAD]", "[FACT]", "[QUERY]", "[UNK]")
PUNCTUATION = tuple(sorted(("(", ")", "[", "]", ",", ";", ":", "?", "@", "<<", "<t", "==", "--")))


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls) -> Vocabulary:
        aliases = set(SET_ALIASES.values()) | set(SCENE_ALIASES.values()) | set(
            PROGRAM_ALIASES.values()
        )
        for mapping in V2_ALIASES.values():
            aliases.update(mapping.values())
        reserved = aliases | {"ask", "assert", "WITHIN"}
        entities = (
            [f"e{i:02d}" for i in range(32)]
            + [f"o{i:02d}" for i in range(32)]
            + [f"v{i:02d}" for i in range(32)]
            + [f"s{i:02d}" for i in range(16)]
            + [f"r{i:02d}" for i in range(16)]
            + [f"c{i:02d}" for i in range(16)]
        )
        ordered = list(SPECIAL_TOKENS)
        ordered.extend(token for token in PUNCTUATION if token not in ordered)
        ordered.extend(token for token in sorted(reserved) if token not in ordered)
        ordered.extend(token for token in entities if token not in ordered)
        return cls(tuple(ordered))

    def encode(self, text: str, prefix: str, max_length: int = 12) -> tuple[list[int], list[bool]]:
        mapping = self.token_to_id
        tokens = (prefix,) + lex(text)
        if len(tokens) > max_length:
            raise ValueError(f"sequence exceeds {max_length} tokens: {tokens}")
        try:
            ids = [mapping[token] for token in tokens]
        except KeyError as exc:
            raise ValueError(f"token missing from frozen vocabulary: {exc.args[0]}") from exc
        mask = [True] * len(ids)
        padding = max_length - len(ids)
        return ids + [0] * padding, mask + [False] * padding

    def as_json(self) -> str:
        return json.dumps({token: index for index, token in enumerate(self.tokens)}, indent=2)


@dataclass(frozen=True)
class MaterializedExample:
    example_id: str
    world_id: int
    domain: Domain
    renderer_variant: int
    facts: tuple[Fact, ...]
    fact_endpoint_ids: tuple[tuple[int, int], ...]
    rendered_facts: tuple[str, ...]
    query: Query
    rendered_query: str
    label: int
    argument_entity_ids: tuple[int, int]
    evidence_indices: tuple[int, int]
    operation_ids: tuple[int, int]
    step_mask: tuple[bool, bool]


def example_id(example: Example, domain: Domain, renderer_variant: int) -> str:
    value = {
        "world_id": example.world_id,
        "domain": domain.name,
        "renderer_variant": renderer_variant,
        "query": [example.query.relation.name, example.query.arg1, example.query.arg2],
        "operations": [operation.name for operation in example.operations],
        "evidence": [
            [fact.relation.name, fact.arg1, fact.arg2] for fact in example.evidence
        ],
        "corruption": example.corruption.name,
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:32]


def _distractors(world: World, example: Example, count: int, identifier: str) -> list[Fact]:
    evidence = set(example.evidence)
    candidates = [fact for fact in world.facts if fact not in evidence]
    shared = [
        fact
        for fact in candidates
        if example.query.arg1 in (fact.arg1, fact.arg2)
        or example.query.arg2 in (fact.arg1, fact.arg2)
    ]
    rank = lambda fact: hashlib.sha256(
        f"distractor/{identifier}/{fact.relation.name}/{fact.arg1}/{fact.arg2}".encode()
    ).hexdigest()
    selected: list[Fact] = []

    def safe(candidate: Fact) -> bool:
        if example.label == 1:
            return True
        projected = World(
            world.world_id,
            world.entities,
            tuple(selected + list(example.evidence) + [candidate]),
        )
        return not entailed(projected, example.query)

    for candidate in sorted(shared, key=rank):
        if len(selected) >= min(2, count):
            break
        if safe(candidate):
            selected.append(candidate)
    remaining = [fact for fact in candidates if fact not in selected]
    relation_order = (
        Relation.DIRECT_IN,
        Relation.NESTED_IN,
        Relation.BEFORE,
        Relation.SAME,
        Relation.LINK,
    )
    while len(selected) < count:
        added = False
        for relation in relation_order:
            choices = sorted((fact for fact in remaining if fact.relation == relation), key=rank)
            if not choices:
                continue
            candidate = choices[0]
            if not safe(candidate):
                remaining.remove(candidate)
                continue
            selected.append(candidate)
            remaining.remove(candidate)
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    if len(selected) != count:
        raise ValueError(f"insufficient safe distractors for {identifier}: {len(selected)}/{count}")
    return selected


def materialize(
    world: World,
    example: Example,
    domain: Domain,
    renderer_variant: int,
    fact_count: int = 8,
    binding_index: int | None = None,
    evidence_order_index: int | None = None,
) -> MaterializedExample:
    identifier = example_id(example, domain, renderer_variant)
    distractors = _distractors(world, example, fact_count - len(example.evidence), identifier)
    occurrences: dict[Fact, int] = {}
    combined: list[tuple[Fact, int]] = []
    for fact in list(example.evidence) + distractors:
        occurrence = occurrences.get(fact, 0)
        occurrences[fact] = occurrence + 1
        combined.append((fact, occurrence))
    if evidence_order_index is None:
        combined.sort(key=lambda item: fact_order_key(identifier, item[0], item[1]))
        facts = tuple(item[0] for item in combined)
    else:
        facts = _scheduled_fact_order(
            example, distractors, identifier, fact_count, evidence_order_index
        )
    if evidence_order_index is None:
        evidence_indices = [facts.index(fact) for fact in example.evidence]
        if len(evidence_indices) == 1:
            evidence_indices.append(-1)
    else:
        # The schedule identifies evidence occurrences, not merely equal Fact values.
        # Value lookup collapses repeated evidence facts onto their first occurrence.
        evidence_indices = list(
            evidence_positions(evidence_order_index, fact_count, len(example.evidence))
        )
    operation_ids = [int(operation) for operation in example.operations]
    if len(operation_ids) == 1:
        operation_ids.append(int(Operation.LOOKUP))
    rendered = render(world, facts, example.query, domain, renderer_variant)
    object_count = sum(entity.type.value == 0 for entity in world.entities)
    binding = (
        _counterbalanced_binding(world, example, identifier, binding_index)
        if binding_index is not None
        else {
            entity.id: entity.id if entity.id < object_count else 32 + entity.id - object_count
            for entity in world.entities
        }
    )

    return MaterializedExample(
        example_id=identifier,
        world_id=world.world_id,
        domain=domain,
        renderer_variant=renderer_variant,
        facts=facts,
        fact_endpoint_ids=tuple(
            (binding[fact.arg1], binding[fact.arg2]) for fact in facts
        ),
        rendered_facts=rendered.facts,
        query=example.query,
        rendered_query=rendered.query,
        label=example.label,
        argument_entity_ids=(
            binding[example.query.arg1],
            binding[example.query.arg2],
        ),
        evidence_indices=tuple(evidence_indices),
        operation_ids=tuple(operation_ids),
        step_mask=(True, len(example.operations) == 2),
    )


def evidence_positions(index: int, fact_count: int, steps: int) -> tuple[int, int]:
    """Return the approved v1.1.3 evidence slots for a stratum-local rank."""
    if index < 0 or fact_count < 2 or steps not in (1, 2):
        raise ValueError("invalid evidence-position schedule input")
    first = index % fact_count
    if steps == 1:
        return first, -1
    second = (first + 1 + ((index // fact_count) % (fact_count - 1))) % fact_count
    if first == second:
        raise AssertionError("evidence-position schedule collided")
    return first, second


def _scheduled_fact_order(
    example: Example,
    distractors: list[Fact],
    identifier: str,
    fact_count: int,
    index: int,
) -> tuple[Fact, ...]:
    targets = evidence_positions(index, fact_count, len(example.evidence))
    slots: list[Fact | None] = [None] * fact_count
    for step, fact in enumerate(example.evidence):
        slots[targets[step]] = fact
    occurrences: dict[Fact, int] = {}
    ranked_distractors: list[tuple[str, Fact]] = []
    for fact in distractors:
        occurrence = occurrences.get(fact, 0)
        occurrences[fact] = occurrence + 1
        ranked_distractors.append((fact_order_key(identifier, fact, occurrence), fact))
    remaining = iter(fact for _, fact in sorted(ranked_distractors))
    for slot, value in enumerate(slots):
        if value is None:
            slots[slot] = next(remaining)
    if any(value is None for value in slots):
        raise AssertionError("scheduled fact order left an empty slot")
    return tuple(value for value in slots if value is not None)


def _counterbalanced_binding(
    world: World, example: Example, identifier: str, binding_index: int
) -> dict[int, int]:
    """Build the approved v1.1.1 type-preserving, example-scoped entity bijection."""
    if binding_index < 0:
        raise ValueError("binding index must be nonnegative")
    types = world.entity_types
    first, second = example.query.arg1, example.query.arg2
    first_type, second_type = types[first], types[second]
    if first_type != second_type:
        targets = {
            first: binding_index % 32 if first_type.value == 0 else 32 + binding_index % 16,
            second: binding_index % 32 if second_type.value == 0 else 32 + binding_index % 16,
        }
    else:
        if first_type.value != 0:
            raise ValueError("v1.1.1 has no container/container query")
        first_target = binding_index % 32
        second_target = (
            first_target + 1 + ((binding_index // 32) % 31)
        ) % 32
        targets = {first: first_target, second: second_target}
    mapping = dict(targets)
    for entity_type, inventory in ((0, range(32)), (1, range(32, 48))):
        entities = sorted(entity.id for entity in world.entities if entity.type.value == entity_type)
        unused = set(inventory) - {target for entity, target in mapping.items() if types[entity].value == entity_type}
        for entity_id in entities:
            if entity_id in mapping:
                continue
            target = min(
                unused,
                key=lambda candidate: hashlib.sha256(
                    f"opm-v1.1/binding/{identifier}/{entity_type}/{entity_id}/{candidate}".encode()
                ).hexdigest(),
            )
            mapping[entity_id] = target
            unused.remove(target)
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError("binding is not injective")
    for entity_id, target in mapping.items():
        if types[entity_id].value == 0 and not 0 <= target < 32:
            raise AssertionError("object binding escaped object inventory")
        if types[entity_id].value == 1 and not 32 <= target < 48:
            raise AssertionError("container binding escaped container inventory")
    if mapping[first] == mapping[second]:
        raise AssertionError(
            f"query entities collided: example={identifier} index={binding_index} "
            f"entities=({first},{second}) target={mapping[first]} relation={example.query.relation.name}"
        )
    return mapping


def collate(examples: Sequence[MaterializedExample], vocabulary: Vocabulary) -> OPMBatch:
    fact_ids: list[list[list[int]]] = []
    fact_masks: list[list[list[bool]]] = []
    query_ids: list[list[int]] = []
    query_masks: list[list[bool]] = []
    endpoints: list[list[list[int]]] = []
    for example in examples:
        encoded_facts = [vocabulary.encode(text, "[FACT]") for text in example.rendered_facts]
        fact_ids.append([item[0] for item in encoded_facts])
        fact_masks.append([item[1] for item in encoded_facts])
        query_encoded = vocabulary.encode(example.rendered_query, "[QUERY]")
        query_ids.append(query_encoded[0])
        query_masks.append(query_encoded[1])
        endpoints.append([list(pair) for pair in example.fact_endpoint_ids])
    return OPMBatch(
        fact_tokens=torch.tensor(fact_ids, dtype=torch.long),
        fact_token_mask=torch.tensor(fact_masks, dtype=torch.bool),
        query_tokens=torch.tensor(query_ids, dtype=torch.long),
        query_token_mask=torch.tensor(query_masks, dtype=torch.bool),
        domain_ids=torch.tensor([int(item.domain) for item in examples], dtype=torch.long),
        argument_entity_ids=torch.tensor(
            [item.argument_entity_ids for item in examples], dtype=torch.long
        ),
        fact_endpoint_ids=torch.tensor(endpoints, dtype=torch.long),
        evidence_indices=torch.tensor([item.evidence_indices for item in examples], dtype=torch.long),
        operation_ids=torch.tensor([item.operation_ids for item in examples], dtype=torch.long),
        step_mask=torch.tensor([item.step_mask for item in examples], dtype=torch.bool),
        labels=torch.tensor([item.label for item in examples], dtype=torch.long),
    )


def batch_to(batch: OPMBatch, device: torch.device | str) -> OPMBatch:
    values = {
        field.name: (
            getattr(batch, field.name).to(device)
            if isinstance(getattr(batch, field.name), Tensor)
            else getattr(batch, field.name)
        )
        for field in fields(batch)
    }
    return OPMBatch(**values)
