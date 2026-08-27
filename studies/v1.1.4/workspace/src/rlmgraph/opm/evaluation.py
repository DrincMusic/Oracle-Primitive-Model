from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import OPMBatch, OPMModel


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    count: int
    predictions: tuple[int, ...]


@torch.no_grad()
def evaluate(model: OPMModel, batches: list[OPMBatch]) -> EvaluationResult:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    for batch in batches:
        if batch.labels is None:
            raise ValueError("evaluation batch has no labels")
        predictions.extend(model(batch).argmax(dim=-1).cpu().tolist())
        labels.extend(batch.labels.cpu().tolist())
    correct = sum(left == right for left, right in zip(predictions, labels, strict=True))
    return EvaluationResult(correct / len(labels), len(labels), tuple(predictions))


@torch.no_grad()
def ablation_logits(model: OPMModel, batch: OPMBatch, primitive_index: int) -> Tensor:
    if not hasattr(model, "primitives"):
        raise ValueError("primitive-index ablation applies only to primitive models")
    primitive = model.primitives[primitive_index]
    handles = []

    def zero_delta(_module, _inputs, output):
        return torch.zeros_like(output)

    handles.append(primitive.output.register_forward_hook(zero_delta))
    try:
        return model(batch)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def adapter_only_logits(model: OPMModel, batch: OPMBatch) -> Tensor:
    modules = list(getattr(model, "primitives", ())) + list(getattr(model, "generalists", ()))
    handles = [module.output.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out)) for module in modules]
    try:
        return model(batch)
    finally:
        for handle in handles:
            handle.remove()
