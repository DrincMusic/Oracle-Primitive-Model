import hashlib
import json
from collections import Counter

import pytest

from rlmgraph.opm.algebra import Relation
from rlmgraph.opm.data import _counterbalanced_binding, evidence_positions, materialize
from rlmgraph.opm.generation import (
    Operation,
    enumerate_negative_candidates,
    enumerate_positive_templates,
    generate_world,
)
from rlmgraph.opm.rendering import Domain
from rlmgraph.opm.splits import (
    CANONICAL_EXAMPLES_PER_CELL,
    HOLDOUTS,
    DiagnosticSplitConfig,
    SplitName,
    SplitValidationConfig,
    allocate_split,
    allocate_validation_split,
    canonical_expected_rows,
    validate_split_collection,
    write_v113_test_split,
    write_validation_split,
)
from rlmgraph.opm.statistics import PairedPrediction, paired_two_level_bootstrap
from rlmgraph.opm.traces import build_trace_fixtures, write_trace_fixtures


def test_recombination_contains_only_heldout_cells() -> None:
    records = allocate_validation_split(
        SplitValidationConfig(SplitName.TEST_RECOMBINATION, examples_per_cell=1)
    )
    assert len(records) == 6
    for record in records:
        assert Operation(record.operation_ids[0]) == HOLDOUTS[record.domain]


def test_interpolation_excludes_heldout_cells() -> None:
    records = allocate_validation_split(
        SplitValidationConfig(SplitName.TEST_INTERPOLATION, examples_per_cell=1)
    )
    assert len(records) == 18
    for record in records:
        assert Operation(record.operation_ids[0]) != HOLDOUTS[record.domain]


def test_validation_manifest_is_reproducible(tmp_path) -> None:
    config = SplitValidationConfig(SplitName.VALIDATION, examples_per_cell=1)
    first_data, _, first = write_validation_split(config, tmp_path / "first")
    second_data, _, second = write_validation_split(config, tmp_path / "second")
    assert first.sha256 == second.sha256
    assert first_data.read_bytes() == second_data.read_bytes()
    assert hashlib.sha256(first_data.read_bytes()).hexdigest() == first.sha256


def test_canonical_row_plan_matches_approved_counts() -> None:
    assert CANONICAL_EXAMPLES_PER_CELL == {
        SplitName.TRAIN: 8_000,
        SplitName.VALIDATION: 800,
        SplitName.TEST_INTERPOLATION: 800,
        SplitName.TEST_RECOMBINATION: 3_000,
        SplitName.TEST_RENDERER: 400,
        SplitName.TEST_STRUCTURAL: 400,
    }
    assert {split: canonical_expected_rows(split) for split in SplitName} == {
        SplitName.TRAIN: 144_000,
        SplitName.VALIDATION: 14_400,
        SplitName.TEST_INTERPOLATION: 14_400,
        SplitName.TEST_RECOMBINATION: 18_000,
        SplitName.TEST_RENDERER: 9_600,
        SplitName.TEST_STRUCTURAL: 9_600,
    }


def test_positive_relation_subquotas_are_exact() -> None:
    records = allocate_validation_split(
        SplitValidationConfig(SplitName.TEST_RENDERER, examples_per_cell=8)
    )
    for domain in {record.domain for record in records}:
        lookup = [
            record
            for record in records
            if record.domain == domain
            and record.label == 1
            and record.operation_ids[0] == Operation.LOOKUP
        ]
        chain = [
            record
            for record in records
            if record.domain == domain
            and record.label == 1
            and record.operation_ids[0] == Operation.CHAIN
        ]
        assert Counter(record.query.relation for record in lookup) == {
            Relation.BEFORE: 2,
            Relation.LINK: 2,
            Relation.SAME: 2,
            Relation.WITHIN: 2,
        }
        assert Counter(record.query.relation for record in chain) == {
            Relation.BEFORE: 4,
            Relation.SAME: 4,
        }
        lookup_negative = [
            record
            for record in records
            if record.domain == domain
            and record.label == 0
            and record.operation_ids[0] == Operation.LOOKUP
        ]
        chain_negative = [
            record
            for record in records
            if record.domain == domain
            and record.label == 0
            and record.operation_ids[0] == Operation.CHAIN
        ]
        assert Counter(record.query.relation for record in lookup_negative) == {
            Relation.BEFORE: 2,
            Relation.LINK: 2,
            Relation.SAME: 2,
            Relation.WITHIN: 2,
        }
        assert Counter(record.query.relation for record in chain_negative) == {
            Relation.BEFORE: 4,
            Relation.SAME: 4,
        }


def test_cross_split_contamination_is_rejected() -> None:
    records = allocate_validation_split(
        SplitValidationConfig(SplitName.TRAIN, examples_per_cell=1)
    )
    validate_split_collection({SplitName.TRAIN: records})
    with pytest.raises(ValueError, match="world_id contamination"):
        validate_split_collection(
            {SplitName.TRAIN: records, SplitName.VALIDATION: records}
        )


def test_test_split_records_keep_labels_out_of_model_inputs() -> None:
    from rlmgraph.opm.splits import CanonicalSplitConfig, _manifest, record_dict

    records = allocate_validation_split(
        SplitValidationConfig(SplitName.TEST_RECOMBINATION, examples_per_cell=1)
    )
    rows = []
    labels = []
    for record in records:
        row = record_dict(record)
        labels.append(row.pop("label"))
        rows.append(row)
    assert all("label" not in row for row in rows)
    assert labels == [record.label for record in records]
    payload = b"placeholder\n"
    manifest = _manifest(
        CanonicalSplitConfig(SplitName.TEST_RECOMBINATION),
        records,
        payload,
        validation_only=False,
        labels_separated=True,
        labels_sha256="abc",
    )
    assert manifest.labels_separated is True


def test_diagnostic_namespace_is_explicit_and_disjoint() -> None:
    with pytest.raises(ValueError, match="namespace must be explicit"):
        DiagnosticSplitConfig(SplitName.VALIDATION, "opm-v1")
    canonical_like = allocate_validation_split(
        SplitValidationConfig(SplitName.VALIDATION, examples_per_cell=1)
    )
    diagnostic = allocate_split(
        DiagnosticSplitConfig(
            SplitName.VALIDATION, "opm-v1-diagnostic/test-replica"
        )
    )
    assert len(diagnostic) == canonical_expected_rows(SplitName.VALIDATION)
    assert {record.world_id for record in canonical_like}.isdisjoint(
        record.world_id for record in diagnostic
    )


def test_v111_binding_schedule_is_collision_free_and_deterministic() -> None:
    world = generate_world(1101, n_objects=10, n_containers=5)
    example = next(
        item
        for item in enumerate_positive_templates(world)
        if item.query.relation in (Relation.BEFORE, Relation.SAME, Relation.LINK)
    )
    for index in range(8_000):
        first = _counterbalanced_binding(world, example, "fixture", index)
        second = _counterbalanced_binding(world, example, "fixture", index)
        assert first == second
        assert first[example.query.arg1] != first[example.query.arg2]
        assert all(0 <= value < 48 for value in first.values())


def test_v113_evidence_schedule_is_collision_free_and_places_facts() -> None:
    for index in range(8_000):
        one = evidence_positions(index, 8, 1)
        two = evidence_positions(index, 8, 2)
        assert one == evidence_positions(index, 8, 1)
        assert 0 <= one[0] < 8 and one[1] == -1
        assert 0 <= two[0] < 8 and 0 <= two[1] < 8 and two[0] != two[1]
    world = generate_world(2202, n_objects=10, n_containers=5)
    example = next(item for item in enumerate_positive_templates(world) if len(item.evidence) == 2)
    record = materialize(
        world,
        example,
        Domain.SET,
        0,
        binding_index=992,
        evidence_order_index=992,
    )
    expected = evidence_positions(992, 8, 2)
    assert record.evidence_indices == expected
    assert tuple(record.facts[index] for index in expected) == example.evidence


def test_v113_evidence_schedule_preserves_distinct_repeated_occurrences() -> None:
    duplicate = None
    selected_world = None
    for seed in range(100):
        world = generate_world(seed, n_objects=10, n_containers=5)
        for positive in enumerate_positive_templates(world):
            for negative in enumerate_negative_candidates(world, positive):
                if len(negative.evidence) == 2 and negative.evidence[0] == negative.evidence[1]:
                    duplicate = negative
                    selected_world = world
                    break
            if duplicate is not None:
                break
        if duplicate is not None:
            break
    assert duplicate is not None and selected_world is not None

    expected = evidence_positions(57, 8, 2)
    record = materialize(
        selected_world,
        duplicate,
        Domain.SET,
        0,
        binding_index=57,
        evidence_order_index=57,
    )
    assert expected[0] != expected[1]
    assert record.evidence_indices == expected
    assert tuple(record.facts[index] for index in expected) == duplicate.evidence


def test_v113_test_writer_immediately_separates_labels(tmp_path, monkeypatch) -> None:
    import rlmgraph.opm.splits as splits_module
    from rlmgraph.opm.protocol import ProtocolState

    monkeypatch.setattr(splits_module, "CURRENT_PROTOCOL", ProtocolState())
    monkeypatch.setitem(CANONICAL_EXAMPLES_PER_CELL, SplitName.TEST_RECOMBINATION, 1)
    data_path, label_path, manifest_path, manifest = write_v113_test_split(
        SplitName.TEST_RECOMBINATION, tmp_path
    )
    assert '"label"' not in data_path.read_text(encoding="utf-8")
    assert '"label"' in label_path.read_text(encoding="utf-8")
    assert manifest.labels_separated is True
    assert manifest.labels_sha256 is not None
    assert manifest_path.exists()
    audit = json.loads((tmp_path / "sealed-labels" / "seal-creation.audit.jsonl").read_text())
    assert audit["label_content_accessed_after_sealing"] is False
    assert audit["evaluation_executed"] is False


def test_trace_fixtures_cover_four_required_cases(tmp_path) -> None:
    fixtures = build_trace_fixtures()
    assert [item.trace_id for item in fixtures] == [
        "TRACE-001",
        "TRACE-002",
        "TRACE-003",
        "TRACE-004",
    ]
    digest, count = write_trace_fixtures(tmp_path / "traces.json")
    assert count == 4
    assert len(digest) == 64


def test_two_level_bootstrap_is_deterministic() -> None:
    rows = [
        PairedPrediction(seed, f"example-{index}", 1, index % 2, 0)
        for seed in (1101, 2202)
        for index in range(10)
    ]
    first = paired_two_level_bootstrap(rows, replicates=200, seed=5)
    second = paired_two_level_bootstrap(rows, replicates=200, seed=5)
    assert first == second
    assert first.delta_untied == 1.0
