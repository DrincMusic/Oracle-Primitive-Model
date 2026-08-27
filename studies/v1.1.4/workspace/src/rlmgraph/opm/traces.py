from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .algebra import Relation
from .data import MaterializedExample, materialize
from .generation import (
    Corruption,
    Operation,
    enumerate_negative_candidates,
    enumerate_positive_templates,
    generate_world,
)
from .rendering import Domain


@dataclass(frozen=True)
class TraceFixture:
    trace_id: str
    record: MaterializedExample


def build_trace_fixtures() -> tuple[TraceFixture, ...]:
    world = generate_world(1101, n_objects=10, n_containers=5)
    positives = enumerate_positive_templates(world)
    reverse = next(
        item
        for item in positives
        if item.operations == (Operation.REVERSE,) and item.query.relation == Relation.LINK
    )
    chain = next(
        item
        for item in positives
        if item.operations == (Operation.CHAIN, Operation.CHAIN)
        and item.query.relation == Relation.BEFORE
    )
    negatives = [
        candidate
        for positive in positives
        for candidate in enumerate_negative_candidates(world, positive)
    ]
    lookup_negative = next(
        item
        for item in negatives
        if item.operations == (Operation.LOOKUP,)
        and item.query.relation == Relation.BEFORE
    )
    lift_negative = next(
        item
        for item in negatives
        if item.operations == (Operation.LIFT, Operation.LIFT)
        and item.corruption == Corruption.MIDDLE
    )
    return (
        TraceFixture("TRACE-001", materialize(world, reverse, Domain.SET, 0)),
        TraceFixture("TRACE-002", materialize(world, lookup_negative, Domain.SCENE, 0)),
        TraceFixture("TRACE-003", materialize(world, chain, Domain.PROGRAM, 0)),
        TraceFixture("TRACE-004", materialize(world, lift_negative, Domain.SET, 0)),
    )


def trace_dict(fixture: TraceFixture) -> dict[str, object]:
    record = fixture.record
    return {
        "trace_id": fixture.trace_id,
        "example_id": record.example_id,
        "world_id": record.world_id,
        "domain": record.domain.name,
        "facts": [[fact.relation.name, fact.arg1, fact.arg2] for fact in record.facts],
        "rendered_facts": list(record.rendered_facts),
        "query": [record.query.relation.name, record.query.arg1, record.query.arg2],
        "rendered_query": record.rendered_query,
        "label": record.label,
        "fact_endpoint_ids": [list(pair) for pair in record.fact_endpoint_ids],
        "argument_entity_ids": list(record.argument_entity_ids),
        "evidence_indices": list(record.evidence_indices),
        "operation_ids": list(record.operation_ids),
        "step_mask": list(record.step_mask),
    }


def write_trace_fixtures(path: Path) -> tuple[str, int]:
    fixtures = [trace_dict(item) for item in build_trace_fixtures()]
    payload = (json.dumps(fixtures, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(fixtures)
