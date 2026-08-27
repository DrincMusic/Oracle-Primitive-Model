import json

import numpy as np

from rlmgraph.opm.accounting import active_flop_ratio, trace_forward
from rlmgraph.opm.claims import ClaimStatus, decide_h1, write_claim_ledger
from rlmgraph.opm.data import Vocabulary, collate, materialize
from rlmgraph.opm.generation import enumerate_positive_templates, generate_world
from rlmgraph.opm.model import ModelConfig, ModelKind, OPMModel
from rlmgraph.opm.probes import neural_evidence_probe, neural_probe_pair_passes
from rlmgraph.opm.rendering import Domain
from rlmgraph.opm.statistics import BootstrapResult
from rlmgraph.opm.traceability import ENTRIES, write_traceability


def _batch():
    world = generate_world(9191, n_objects=8, n_containers=4)
    examples = [
        example for example in enumerate_positive_templates(world) if len(example.operations) == 2
    ][:2]
    vocabulary = Vocabulary.build()
    records = [materialize(world, example, Domain.SET, 0) for example in examples]
    return collate(records, vocabulary), vocabulary


def test_evidence_extraction_and_flop_trace() -> None:
    batch, vocabulary = _batch()
    model = OPMModel(ModelConfig(len(vocabulary.tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101)
    evidence = model.encode_selected_evidence(batch)
    assert evidence.shape == (2, 2, 192)
    report = trace_forward(model, batch, repeats=1)
    assert report.forward_flops > 0
    assert report.median_latency_ms > 0
    assert active_flop_ratio(report, report) == 1.0


def test_neural_probe_contract_at_validation_scale() -> None:
    rng = np.random.default_rng(44)
    train_x = rng.normal(size=(40, 192))
    train_y = rng.integers(0, 2, size=40)
    valid_x = rng.normal(size=(40, 192))
    valid_y = rng.integers(0, 2, size=40)
    results = [
        neural_evidence_probe(
            train_x,
            train_y,
            valid_x,
            valid_y,
            model_condition="OPM_SHARED",
            model_seed=1101,
            step=step,
            epochs=1,
            threshold=1.0,
        )
        for step in (1, 2)
    ]
    assert all(result.model_seed == 1101 for result in results)
    assert isinstance(neural_probe_pair_passes(results), bool)


def test_claim_decision_and_ledger(tmp_path) -> None:
    bootstrap = BootstrapResult(0.05, 0.10, (0.03, 0.07), (0.08, 0.12), 10_000)
    claim = decide_h1(
        bootstrap,
        shared_interpolation_accuracy=0.91,
        generalist_interpolation_accuracy=0.90,
        untied_interpolation_accuracy=0.90,
    )
    assert claim.status == ClaimStatus.SUPPORTED
    path = tmp_path / "claims.json"
    write_claim_ledger([claim], path)
    assert json.loads(path.read_text())[0]["claim_id"] == "H1-PRIMARY"


def test_traceability_is_serializable(tmp_path) -> None:
    path = tmp_path / "traceability.json"
    write_traceability(path)
    payload = json.loads(path.read_text())
    assert len(payload) == len(ENTRIES)
    requirements = [entry["requirement"] for entry in payload]
    assert len(requirements) == 66
    assert len(set(requirements)) == 66
    assert requirements[0] == "ALG-001"
    assert requirements[-1] == "SYS-004"
