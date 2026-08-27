from __future__ import annotations

import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import typer

from .accounting import accounting_matrix, trace_forward
from .algebra import entailed
from .amendment_validation import validate_amended_data, write_amendment_report
from .artifacts import load_checkpoint, save_checkpoint, write_run_manifest
from .claims import ClaimStatus, decide_h1, write_claim_ledger
from .data import Vocabulary, batch_to, collate, materialize
from .dataset_audit import (
    audit_canonical_directory,
    audit_v113_inputs_without_labels,
    write_dataset_audit,
)
from .evidence_diagnostics import diagnose_evidence_leakage, write_evidence_diagnostics
from .generation import enumerate_negative_candidates, enumerate_positive_templates, generate_world
from .leakage_diagnostics import (
    diagnose_argument_leakage,
    diagnose_replica_namespaces,
    write_leakage_diagnostics,
    write_replica_report,
)
from .model import ModelConfig, ModelKind, OPMModel
from .primary_training import CanonicalTrainingConfig, train_canonical_run
from .probes import neural_evidence_probe, neural_probe_pair_passes
from .protocol import CURRENT_PROTOCOL
from .raw_leakage import run_raw_leakage_gates, write_raw_leakage_report
from .readiness import implementation_readiness, write_readiness_report
from .rendering import Domain
from .sealing import audit_label_access, evaluate_sealed_predictions, seal_validation_labels
from .splits import (
    SplitName,
    SplitValidationConfig,
    allocate_validation_split,
    write_amended_canonical_split,
    write_canonical_split,
    write_v113_canonical_split,
    write_v113_test_split,
    write_validation_split,
)
from .statistics import BootstrapResult
from .traceability import write_traceability
from .traces import write_trace_fixtures
from .training import TrainingConfig, train_validation

app = typer.Typer(help="OPM v1 implementation-validation commands. Primary runs are unavailable.")


@app.command("primary-pilot-run")
def primary_pilot_run(
    model_kind: ModelKind,
    learning_rate: float,
    dropout: float,
    train_path: Path,
    validation_path: Path,
    output_directory: Path,
    code_revision: str,
    resume_checkpoint: Path | None = None,
    device: str = "cuda:0",
) -> None:
    """Execute one frozen-grid, full-duration pilot run using pilot seed 1101."""
    summary = train_canonical_run(
        kind=model_kind,
        model_seed=1101,
        config=CanonicalTrainingConfig(learning_rate=learning_rate, dropout=dropout),
        train_path=train_path,
        validation_path=validation_path,
        output_root=output_directory,
        code_revision=code_revision,
        device=device,
        canonical=True,
        resume_checkpoint=resume_checkpoint,
    )
    typer.echo(json.dumps(asdict(summary), indent=2))


@app.command("environment")
def environment() -> None:
    """Print the implementation-validation environment manifest."""
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "protocol": CURRENT_PROTOCOL.__dict__,
    }
    typer.echo(json.dumps(payload, indent=2, default=str))


def _fixture_records():
    world = generate_world(1101, n_objects=10, n_containers=5)
    positives = enumerate_positive_templates(world)
    positive_one = next(example for example in positives if len(example.operations) == 1)
    positive_two = next(example for example in positives if len(example.operations) == 2)
    negative = next(
        candidate
        for source in positives
        for candidate in enumerate_negative_candidates(world, source)
        if len(candidate.operations) == 2
    )
    return world, [
        materialize(world, positive_one, Domain.SET, 0),
        materialize(world, positive_two, Domain.SCENE, 1),
        materialize(world, negative, Domain.PROGRAM, 2),
    ]


@app.command("validate-fixtures")
def validate_fixtures(output: Path | None = None) -> None:
    """Validate deterministic semantics and optionally write a fixture report."""
    CURRENT_PROTOCOL.require_validation()
    world, records = _fixture_records()
    payload = {
        "world_id": world.world_id,
        "fact_count": len(world.facts),
        "records": [
            {
                "example_id": record.example_id,
                "domain": record.domain.name,
                "variant": record.renderer_variant,
                "label": record.label,
                "evidence_indices": record.evidence_indices,
                "rendered_query": record.rendered_query,
            }
            for record in records
        ],
        "positive_entailments_valid": all(
            entailed(world, record.query) for record in records if record.label == 1
        ),
    }
    rendered = json.dumps(payload, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    typer.echo(rendered)


@app.command("model-smoke")
def model_smoke(steps: int = typer.Option(2, min=1, max=100)) -> None:
    """Run bounded implementation-validation training for every approved control."""
    CURRENT_PROTOCOL.require_validation()
    _, records = _fixture_records()
    vocabulary = Vocabulary.build()
    batch = collate(records, vocabulary)
    config = ModelConfig(len(vocabulary.tokens), dropout=0.0)
    payload: dict[str, object] = {"steps": steps, "models": {}}
    for kind in ModelKind:
        torch.manual_seed(1101)
        model = OPMModel(config, kind, model_seed=1101)
        result = train_validation(
            model, [batch], TrainingConfig(), validation_step_limit=steps
        )
        payload["models"][kind.value] = {
            "final_loss": result.final_loss,
            "accounting": accounting_matrix(config)[list(ModelKind).index(kind)],
        }
    typer.echo(json.dumps(payload, indent=2))


@app.command("validate-target-hardware")
def validate_target_hardware(output: Path, checkpoint: Path) -> None:
    """Run bounded CUDA profiler and checkpoint/recovery validation without primary data."""
    CURRENT_PROTOCOL.require_validation()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA target hardware is unavailable")
    device = torch.device("cuda:0")
    _, records = _fixture_records()
    vocabulary = Vocabulary.build()
    batch = batch_to(collate(records, vocabulary), device)
    config = ModelConfig(len(vocabulary.tokens), dropout=0.0)
    traces: list[dict[str, object]] = []
    for kind in ModelKind:
        torch.manual_seed(1101)
        torch.cuda.manual_seed_all(1101)
        model = OPMModel(config, kind, model_seed=1101).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        trace = trace_forward(model, batch, repeats=5)
        traces.append(
            {
                **asdict(trace),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
        )

    torch.manual_seed(1101)
    torch.cuda.manual_seed_all(1101)
    model = OPMModel(config, ModelKind.OPM_SHARED, model_seed=1101).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def step(active_model: OPMModel, active_optimizer: torch.optim.Optimizer) -> None:
        active_optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(active_model(batch), batch.labels)
        loss.backward()
        active_optimizer.step()
        torch.cuda.synchronize()

    step(model, optimizer)
    checkpoint_digest = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=1,
        metadata={"validation_only": True, "primary_training": False, "device": str(device)},
    )
    step(model, optimizer)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resumed, resumed_optimizer, resumed_step, metadata = load_checkpoint(
        checkpoint,
        optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, lr=3e-4),
        device=device,
    )
    step(resumed, resumed_optimizer)
    exact_recovery = all(
        torch.equal(expected[name], value) for name, value in resumed.state_dict().items()
    )
    payload = {
        "validation_only": True,
        "primary_training_executed": False,
        "test_data_accessed": False,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "traces": traces,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": resumed_step,
        "checkpoint_metadata": metadata,
        "exact_checkpoint_recovery": exact_recovery,
    }
    if not exact_recovery:
        raise RuntimeError("target-hardware checkpoint recovery was not exact")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2))


@app.command("generate-validation-split")
def generate_validation_split(
    split: SplitName,
    output_directory: Path,
    examples_per_cell: int = typer.Option(2, min=1, max=100),
) -> None:
    """Generate a reduced, explicitly non-primary split and immutable manifest."""
    CURRENT_PROTOCOL.require_validation()
    data_path, manifest_path, manifest = write_validation_split(
        SplitValidationConfig(split, examples_per_cell), output_directory
    )
    typer.echo(
        json.dumps(
            {
                "data": str(data_path),
                "manifest": str(manifest_path),
                "rows": manifest.row_count,
                "sha256": manifest.sha256,
                "validation_only": True,
            },
            indent=2,
        )
    )


@app.command("generate-canonical-split")
def generate_canonical_split(split: SplitName, output_directory: Path) -> None:
    """Generate one approved-scale split; test labels are written separately."""
    CURRENT_PROTOCOL.require_validation()
    data_path, label_path, manifest_path, manifest = write_canonical_split(
        split, output_directory
    )
    typer.echo(
        json.dumps(
            {
                "data": str(data_path),
                "labels": str(label_path) if label_path else None,
                "manifest": str(manifest_path),
                "rows": manifest.row_count,
                "sha256": manifest.sha256,
                "labels_separated": manifest.labels_separated,
                "primary_training_executed": False,
            },
            indent=2,
        )
    )


@app.command("generate-v11-split")
def generate_v11_split(split: SplitName, output_directory: Path) -> None:
    """Generate an approved v1.1.1 train or validation artifact only."""
    data_path, manifest_path, manifest = write_amended_canonical_split(
        split, output_directory
    )
    typer.echo(
        json.dumps(
            {
                "data": str(data_path),
                "manifest": str(manifest_path),
                "rows": manifest.row_count,
                "sha256": manifest.sha256,
                "amendment_version": manifest.amendment_version,
                "primary_training_executed": False,
            },
            indent=2,
        )
    )


@app.command("generate-v113-split")
def generate_v113_split(split: SplitName, output_directory: Path) -> None:
    """Generate an approved v1.1.3 train or validation artifact only."""
    data_path, manifest_path, manifest = write_v113_canonical_split(
        split, output_directory
    )
    typer.echo(
        json.dumps(
            {
                "data": str(data_path),
                "manifest": str(manifest_path),
                "rows": manifest.row_count,
                "sha256": manifest.sha256,
                "amendment_version": manifest.amendment_version,
                "evidence_order_version": manifest.evidence_order_version,
                "primary_training_executed": False,
            },
            indent=2,
        )
    )


@app.command("generate-v113-test-split")
def generate_v113_test_split(split: SplitName, output_directory: Path) -> None:
    """Generate and immediately seal an owner-authorized v1.1.3 primary test split."""
    data_path, label_path, manifest_path, manifest = write_v113_test_split(
        split, output_directory
    )
    typer.echo(
        json.dumps(
            {
                "data": str(data_path),
                "sealed_labels": str(label_path),
                "manifest": str(manifest_path),
                "rows": manifest.row_count,
                "input_sha256": manifest.sha256,
                "label_sha256": manifest.labels_sha256,
                "labels_separated": manifest.labels_separated,
                "label_content_accessed_after_sealing": False,
                "primary_training_executed": False,
                "evaluation_executed": False,
            },
            indent=2,
        )
    )


@app.command("validate-v113-data")
def validate_v113_data(directory: Path, output: Path) -> None:
    """Validate v1.1.3 construction and endpoint diagnostics."""
    report = validate_amended_data(
        directory / "train.v1.1.3.canonical.jsonl",
        directory / "validation.v1.1.3.canonical.jsonl",
    )
    write_amendment_report(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("audit-v113-sealed-data")
def audit_v113_sealed_data(directory: Path, output: Path) -> None:
    """Audit all v1.1.3 inputs and seal metadata without opening primary labels."""
    report = audit_v113_inputs_without_labels(directory)
    write_dataset_audit(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("raw-leakage-gates-v113")
def raw_leakage_gates_v113(directory: Path, output: Path) -> None:
    """Run unchanged ORC-005 once on fresh v1.1.3 train and validation."""
    report = run_raw_leakage_gates(
        directory / "train.v1.1.3.canonical.jsonl",
        directory / "validation.v1.1.3.canonical.jsonl",
        argument_source="serialized_binding",
    )
    write_raw_leakage_report(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("validate-v11-data")
def validate_v11_data(directory: Path, output: Path) -> None:
    """Validate v1.1.1 relation, binding, and endpoint diagnostic requirements."""
    report = validate_amended_data(
        directory / "train.v1.1.canonical.jsonl",
        directory / "validation.v1.1.canonical.jsonl",
    )
    write_amendment_report(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("raw-leakage-gates-v11")
def raw_leakage_gates_v11(directory: Path, output: Path) -> None:
    """Run unchanged ORC-005 using serialized v1.1.1 counterbalanced bindings."""
    report = run_raw_leakage_gates(
        directory / "train.v1.1.canonical.jsonl",
        directory / "validation.v1.1.canonical.jsonl",
        argument_source="serialized_binding",
    )
    write_raw_leakage_report(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("diagnose-evidence-leakage-v11")
def diagnose_evidence_leakage_v11(directory: Path, output: Path) -> None:
    """Diagnose the frozen v1.1 evidence-position failure without test access."""
    report = diagnose_evidence_leakage(
        directory / "train.v1.1.canonical.jsonl",
        directory / "validation.v1.1.canonical.jsonl",
    )
    write_evidence_diagnostics(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("audit-canonical-data")
def audit_canonical_data(directory: Path, output: Path) -> None:
    """Verify hashes, counts, separation, and cross-split contamination policies."""
    CURRENT_PROTOCOL.require_validation()
    audit = audit_canonical_directory(directory)
    write_dataset_audit(audit, output)
    typer.echo(json.dumps(asdict(audit), indent=2))


@app.command("raw-leakage-gates")
def raw_leakage_gates(canonical_directory: Path, output: Path) -> None:
    """Run the three pre-freeze raw oracle-channel leakage probes."""
    CURRENT_PROTOCOL.require_validation()
    report = run_raw_leakage_gates(
        canonical_directory / "train.canonical.jsonl",
        canonical_directory / "validation.canonical.jsonl",
    )
    write_raw_leakage_report(report, output)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "train_rows": report.train_rows,
                "validation_rows": report.validation_rows,
                "protocol_freeze_gate_passes": report.protocol_freeze_gate_passes,
                "results": [asdict(result) for result in report.results],
            },
            indent=2,
        )
    )


@app.command("diagnose-argument-leakage")
def diagnose_leakage(canonical_directory: Path, output: Path) -> None:
    """Diagnose the failed argument channel without reading any test labels."""
    CURRENT_PROTOCOL.require_validation()
    report = diagnose_argument_leakage(
        canonical_directory / "train.canonical.jsonl",
        canonical_directory / "validation.canonical.jsonl",
    )
    write_leakage_diagnostics(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("diagnose-leakage-replicas")
def diagnose_leakage_replicas(
    canonical_directory: Path,
    output: Path,
    replicas: int = typer.Option(5, min=1, max=10),
) -> None:
    """Run noncanonical replica namespaces without altering the canonical gate."""
    CURRENT_PROTOCOL.require_validation()
    report = diagnose_replica_namespaces(
        canonical_directory / "train.canonical.jsonl",
        canonical_directory / "validation.canonical.jsonl",
        replica_count=replicas,
    )
    write_replica_report(report, output)
    typer.echo(json.dumps(asdict(report), indent=2))


@app.command("generate-traces")
def generate_traces(output: Path) -> None:
    """Generate the four deterministic implementation-validation trace fixtures."""
    CURRENT_PROTOCOL.require_validation()
    digest, count = write_trace_fixtures(output)
    typer.echo(json.dumps({"output": str(output), "count": count, "sha256": digest}, indent=2))


@app.command("account-validation")
def account_validation(output: Path) -> None:
    """Trace reduced-scale CPU forward FLOPs for every approved neural condition."""
    CURRENT_PROTOCOL.require_validation()
    _, records = _fixture_records()
    vocabulary = Vocabulary.build()
    batch = collate(records, vocabulary)
    config = ModelConfig(len(vocabulary.tokens), dropout=0.0)
    reports = []
    for kind in ModelKind:
        torch.manual_seed(1101)
        model = OPMModel(config, kind, model_seed=1101)
        traced = trace_forward(model, batch, repeats=1)
        reports.append({**traced.__dict__, "parameters": accounting_matrix(config)[list(ModelKind).index(kind)]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output), "models": len(reports)}, indent=2))


@app.command("write-traceability")
def traceability(output: Path) -> None:
    """Write the requirement-to-implementation validation matrix."""
    CURRENT_PROTOCOL.require_validation()
    write_traceability(output)
    typer.echo(json.dumps({"output": str(output)}, indent=2))


@app.command("claim-ledger-smoke")
def claim_ledger_smoke(output: Path) -> None:
    """Validate claim-ledger serialization with synthetic, non-scientific inputs."""
    CURRENT_PROTOCOL.require_validation()
    synthetic = BootstrapResult(0.05, 0.10, (0.03, 0.07), (0.08, 0.12), 10_000)
    claim = decide_h1(
        synthetic,
        shared_interpolation_accuracy=0.91,
        generalist_interpolation_accuracy=0.90,
        untied_interpolation_accuracy=0.90,
    )
    claim = replace(
        claim,
        status=ClaimStatus.INCONCLUSIVE,
        deviations=("SYNTHETIC_VALIDATION_ONLY_DO_NOT_INTERPRET",),
    )
    write_claim_ledger([claim], output)
    typer.echo(json.dumps({"output": str(output), "synthetic_validation_only": True}, indent=2))


@app.command("neural-probe-smoke")
def neural_probe_smoke(output: Path) -> None:
    """Exercise per-step neural probes on untrained validation representations."""
    CURRENT_PROTOCOL.require_validation()
    train_records = allocate_validation_split(
        SplitValidationConfig(SplitName.TRAIN, examples_per_cell=2)
    )
    validation_records = allocate_validation_split(
        SplitValidationConfig(SplitName.VALIDATION, examples_per_cell=2)
    )
    vocabulary = Vocabulary.build()
    train_batch = collate(train_records, vocabulary)
    validation_batch = collate(validation_records, vocabulary)
    torch.manual_seed(1101)
    model = OPMModel(ModelConfig(len(vocabulary.tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101)
    train_features = model.encode_selected_evidence(train_batch).numpy()
    validation_features = model.encode_selected_evidence(validation_batch).numpy()
    results = [
        neural_evidence_probe(
            train_features[:, step - 1],
            train_batch.labels.numpy(),
            validation_features[:, step - 1],
            validation_batch.labels.numpy(),
            model_condition=ModelKind.OPM_SHARED.value,
            model_seed=1101,
            step=step,
            epochs=1,
            threshold=1.0,
        )
        for step in (1, 2)
    ]
    payload = {
        "synthetic_validation_only": True,
        "untrained_model": True,
        "pair_passes_smoke_threshold": neural_probe_pair_passes(results),
        "results": [result.__dict__ for result in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output), "results": len(results)}, indent=2))


@app.command("trained-neural-probe-validation")
def trained_neural_probe_validation(output: Path) -> None:
    """Run the specified 100-epoch probes after bounded validation training."""
    CURRENT_PROTOCOL.require_validation()
    train_records = allocate_validation_split(
        SplitValidationConfig(SplitName.TRAIN, examples_per_cell=2)
    )
    validation_records = allocate_validation_split(
        SplitValidationConfig(SplitName.VALIDATION, examples_per_cell=2)
    )
    vocabulary = Vocabulary.build()
    train_batch = collate(train_records, vocabulary)
    validation_batch = collate(validation_records, vocabulary)
    torch.manual_seed(1101)
    model = OPMModel(
        ModelConfig(len(vocabulary.tokens), dropout=0.0), ModelKind.OPM_SHARED, 1101
    )
    training = train_validation(model, [train_batch], TrainingConfig(), validation_step_limit=5)
    model.eval()
    with torch.no_grad():
        train_features = model.encode_selected_evidence(train_batch).numpy()
        validation_features = model.encode_selected_evidence(validation_batch).numpy()
    results = [
        neural_evidence_probe(
            train_features[:, step - 1],
            train_batch.labels.numpy(),
            validation_features[:, step - 1],
            validation_batch.labels.numpy(),
            model_condition=ModelKind.OPM_SHARED.value,
            model_seed=1101,
            step=step,
            epochs=100,
        )
        for step in (1, 2)
    ]
    payload = {
        "validation_only": True,
        "canonical_mechanism_claim": False,
        "training_steps": training.steps,
        "probe_epochs": 100,
        "pair_passes": neural_probe_pair_passes(results),
        "results": [result.__dict__ for result in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps({"output": str(output), "results": len(results)}, indent=2))


@app.command("write-run-manifest")
def run_manifest(output_directory: Path, code_revision: str = "working-tree") -> None:
    """Create an immutable implementation-validation run identity and manifest."""
    CURRENT_PROTOCOL.require_validation()
    identifier, path = write_run_manifest(
        output_directory,
        configuration={"purpose": "implementation_validation", "primary_run": False},
        dataset_fingerprints={"dataset": "not-generated-canonical"},
        model_seed=1101,
        code_revision=code_revision,
    )
    typer.echo(json.dumps({"run_id": identifier, "manifest": str(path)}, indent=2))


@app.command("seal-validation-labels")
def seal_labels(output_directory: Path) -> None:
    """Separate reduced validation inputs and labels and record authorized access."""
    CURRENT_PROTOCOL.require_validation()
    records = allocate_validation_split(
        SplitValidationConfig(SplitName.VALIDATION, examples_per_cell=2)
    )
    manifest = seal_validation_labels(records, output_directory, "validation")
    audit_label_access(
        output_directory / "label-access.audit.jsonl",
        actor="opm-v1-cli",
        purpose="implementation-validation seal verification",
        label_path=Path(manifest.label_path),
    )
    typer.echo(json.dumps(manifest.__dict__, indent=2))


@app.command("validate-locked-evaluator")
def validate_locked_evaluator(
    manifest: Path, predictions: Path, audit: Path, output: Path
) -> None:
    """Exercise the locked evaluator using a validation-only seal."""
    CURRENT_PROTOCOL.require_validation()
    result = evaluate_sealed_predictions(
        manifest,
        predictions,
        audit,
        protocol=CURRENT_PROTOCOL,
        actor="opm-v1-cli",
        purpose="locked-evaluator implementation validation",
        validation_harness=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("locked-evaluator-fixture")
def locked_evaluator_fixture(
    manifest: Path, predictions: Path, audit: Path, output: Path
) -> None:
    """Create label-blind constant predictions and validate the evaluator boundary."""
    CURRENT_PROTOCOL.require_validation()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not bool(manifest_payload.get("validation_only")):
        raise PermissionError("fixture command accepts validation-only seals")
    input_path = Path(str(manifest_payload["input_path"]))
    input_rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    prediction_payload = "".join(
        json.dumps(
            {"example_id": row["example_id"], "prediction": 0},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in input_rows
    )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(prediction_payload, encoding="utf-8", newline="\n")
    result = evaluate_sealed_predictions(
        manifest,
        predictions,
        audit,
        protocol=CURRENT_PROTOCOL,
        actor="opm-v1-cli",
        purpose="locked-evaluator label-blind fixture",
        validation_harness=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(asdict(result), indent=2))


@app.command("readiness-audit")
def readiness_audit(workspace: Path, output: Path) -> None:
    """Write the fail-closed protocol-freeze readiness report."""
    report = implementation_readiness(workspace)
    write_readiness_report(report, output)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "ready_for_protocol_freeze": report.ready_for_protocol_freeze,
                "primary_runs_authorized": report.primary_runs_authorized,
            },
            indent=2,
        )
    )


@app.command("primary-run")
def primary_run() -> None:
    """Explicitly demonstrate that primary experiments remain gated."""
    CURRENT_PROTOCOL.require_primary()
