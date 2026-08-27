from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Lifecycle(StrEnum):
    SPEC_APPROVED = "SPEC_APPROVED"
    IMPLEMENTATION_VALIDATION = "IMPLEMENTATION_VALIDATION"
    PILOT_ONLY = "PILOT_ONLY"
    PROTOCOL_FROZEN = "PROTOCOL_FROZEN"
    PRIMARY_RUNS = "PRIMARY_RUNS"


@dataclass(frozen=True)
class ProtocolState:
    specification_version: str = "1.0.0"
    charter_version: str = "1.2.0"
    lifecycle: Lifecycle = Lifecycle.IMPLEMENTATION_VALIDATION
    protocol_frozen: bool = False
    primary_runs_authorized: bool = False
    sealed_label_access_authorized: bool = False
    aggregate_test_evaluation_authorized: bool = False
    trained_probes_authorized: bool = False
    prediction_generation_authorized: bool = False
    claim_decisions_authorized: bool = False
    canonical_training_authorized: bool = False
    execution_status: str = "BLOCKED_PENDING_V1_1_4_APPROVAL"
    sampler_conformance_passed: bool = False

    def require_validation(self) -> None:
        if self.lifecycle not in (
            Lifecycle.IMPLEMENTATION_VALIDATION,
            Lifecycle.PILOT_ONLY,
            Lifecycle.PROTOCOL_FROZEN,
            Lifecycle.PRIMARY_RUNS,
        ):
            raise RuntimeError(f"operation requires implementation validation, got {self.lifecycle}")

    def require_dataset_construction(self) -> None:
        if self.lifecycle not in (Lifecycle.IMPLEMENTATION_VALIDATION, Lifecycle.PILOT_ONLY):
            raise PermissionError(
                f"canonical dataset construction is prohibited after freeze: {self.lifecycle}"
            )

    def require_primary(self) -> None:
        if (
            self.lifecycle != Lifecycle.PRIMARY_RUNS
            or not self.protocol_frozen
            or not self.primary_runs_authorized
            or not self.canonical_training_authorized
            or not self.sampler_conformance_passed
        ):
            raise PermissionError(
                "primary experiments are prohibited until PROTOCOL_FROZEN and explicit authorization"
            )

    def require_trained_probes(self) -> None:
        self.require_primary()
        if not self.trained_probes_authorized:
            raise PermissionError("canonical trained probes require separate owner authorization")

    def require_prediction_generation(self) -> None:
        self.require_primary()
        if not self.prediction_generation_authorized:
            raise PermissionError("canonical prediction generation requires owner authorization")

    def require_locked_evaluation(self) -> None:
        if not self.protocol_frozen or self.lifecycle not in (
            Lifecycle.PROTOCOL_FROZEN,
            Lifecycle.PRIMARY_RUNS,
        ):
            raise PermissionError(
                "sealed test labels are unavailable until the protocol is frozen"
            )
        if not (
            self.sealed_label_access_authorized
            and self.aggregate_test_evaluation_authorized
        ):
            raise PermissionError(
                "locked evaluation requires separate sealed-label-access and "
                "aggregate-test-evaluation authorization"
            )


CURRENT_PROTOCOL = ProtocolState(
    lifecycle=Lifecycle.PRIMARY_RUNS,
    protocol_frozen=True,
    primary_runs_authorized=True,
    sealed_label_access_authorized=False,
    aggregate_test_evaluation_authorized=False,
    trained_probes_authorized=True,
    prediction_generation_authorized=True,
    claim_decisions_authorized=False,
    canonical_training_authorized=True,
    execution_status="AUTHORIZED",
    sampler_conformance_passed=True,
)
