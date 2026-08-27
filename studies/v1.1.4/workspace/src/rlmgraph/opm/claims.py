from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from .statistics import BootstrapResult


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    hypothesis: str
    required_comparison: str
    primary_metric: str
    threshold: str
    result: float
    interval: tuple[float, float]
    status: ClaimStatus
    deviations: tuple[str, ...] = ()


def decide_h1(
    bootstrap: BootstrapResult,
    *,
    shared_interpolation_accuracy: float,
    generalist_interpolation_accuracy: float,
    untied_interpolation_accuracy: float,
) -> ClaimRecord:
    lower, upper = bootstrap.generalist_interval
    noninferior = (
        shared_interpolation_accuracy >= generalist_interpolation_accuracy - 0.01
        and shared_interpolation_accuracy >= untied_interpolation_accuracy - 0.01
    )
    if lower > 0.02 and noninferior:
        status = ClaimStatus.SUPPORTED
    elif upper <= 0.02:
        status = ClaimStatus.NOT_SUPPORTED
    else:
        status = ClaimStatus.INCONCLUSIVE
    return ClaimRecord(
        claim_id="H1-PRIMARY",
        hypothesis="H1",
        required_comparison="OPM_SHARED vs DOMAIN_GENERALIST",
        primary_metric="Delta_generalist",
        threshold="lower_95_ci>0.02 and interpolation_noninferiority<=0.01",
        result=bootstrap.delta_generalist,
        interval=bootstrap.generalist_interval,
        status=status,
    )


def write_claim_ledger(records: list[ClaimRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, default=str) + "\n",
        encoding="utf-8",
    )
