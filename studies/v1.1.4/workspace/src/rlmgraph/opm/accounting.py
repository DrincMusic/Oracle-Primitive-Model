from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import torch

from .model import ModelConfig, ModelKind, OPMBatch, OPMModel, parameter_counts


@dataclass(frozen=True)
class AccountingReport:
    model_kind: str
    total_parameters: int
    trainable_parameters: int
    active_primitive_parameters_one_step: int
    active_primitive_parameters_two_step: int
    selected_modules_per_step: int


@dataclass(frozen=True)
class TraceReport:
    model_kind: str
    forward_flops: int
    median_latency_ms: float
    batch_size: int
    profiler: str


def _transition_parameters(model: OPMModel) -> int:
    if model.kind == ModelKind.DOMAIN_GENERALIST:
        transition = model.generalists[0]
    else:
        transition = model.primitives[0]
    return sum(parameter.numel() for parameter in transition.parameters())


def account(model: OPMModel) -> AccountingReport:
    counts = parameter_counts(model)
    active = _transition_parameters(model)
    return AccountingReport(
        model_kind=model.kind.value,
        total_parameters=counts["total"],
        trainable_parameters=counts["trainable"],
        active_primitive_parameters_one_step=active,
        active_primitive_parameters_two_step=2 * active,
        selected_modules_per_step=1,
    )


def accounting_matrix(config: ModelConfig, seed: int = 1101) -> list[dict[str, object]]:
    return [asdict(account(OPMModel(config, kind, seed))) for kind in ModelKind]


def trace_forward(model: OPMModel, batch: OPMBatch, repeats: int = 3) -> TraceReport:
    """Trace forward FLOPs and synchronized latency during implementation validation."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    model.eval()
    device_type = next(model.parameters()).device.type
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            model(batch)
        torch.cuda.synchronize()
    with torch.no_grad(), torch.profiler.profile(
        activities=activities, record_shapes=True, with_flops=True
    ) as profile:
        model(batch)
        if device_type == "cuda":
            torch.cuda.synchronize()
    forward_flops = int(sum(event.flops for event in profile.key_averages()))
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            started = perf_counter()
            model(batch)
            if device_type == "cuda":
                torch.cuda.synchronize()
            latencies.append((perf_counter() - started) * 1_000)
    latencies.sort()
    return TraceReport(
        model_kind=model.kind.value,
        forward_flops=forward_flops,
        median_latency_ms=latencies[len(latencies) // 2],
        batch_size=batch.fact_tokens.shape[0],
        profiler=f"torch.profiler.{device_type}.with_flops",
    )


def active_flop_ratio(left: TraceReport, right: TraceReport) -> float:
    if right.forward_flops <= 0:
        raise ValueError("reference FLOP count must be positive")
    return left.forward_flops / right.forward_flops
