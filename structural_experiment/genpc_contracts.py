"""Experiment-layer controls specific to the frozen GenPC reconciliation R."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .experiment1_contracts import ContractError, Experiment1ControlPlane, fingerprint


@dataclass(frozen=True)
class GenPCReconciliationSpec:
    """Immutable description of the downstream GenPC fusion operator."""

    operator_id: str = "genpc.remove_close_points_fps_statistical_filter"
    operator_revision: str = "upstream-reg-xyz-bac9e7b59f4f"
    distance_threshold: float = 0.0001
    union_semantics: str = "O_then_retained_G"
    fps_requested_count: int = 20000
    noise_std_ratio: float = 2.5
    downstream_implementation_fingerprint: str = "0" * 64
    serialization_policy: str = "float64_arrays; original_index_order; npz-sidecar-boundary"

    def __post_init__(self) -> None:
        if not self.operator_id or not self.operator_revision:
            raise ContractError("GenPC reconciliation operator identity is required")
        if self.distance_threshold <= 0 or self.fps_requested_count <= 0 or self.noise_std_ratio <= 0:
            raise ContractError("GenPC reconciliation numeric controls must be positive")
        if self.union_semantics != "O_then_retained_G":
            raise ContractError("the frozen GenPC union semantics must be O followed by retained G")
        if len(self.downstream_implementation_fingerprint) != 64:
            raise ContractError("downstream_implementation_fingerprint must be a SHA-256")

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenPCExperiment1Control:
    """Existing Experiment 1 control plane plus the explicit GenPC R control."""

    base: Experiment1ControlPlane
    reconciliation: GenPCReconciliationSpec

    def frozen_fingerprints(self) -> dict[str, str]:
        frozen = dict(self.base.frozen_fingerprints())
        frozen["reconciliation"] = self.reconciliation.fingerprint
        return frozen

    def manifest(self, association: Any) -> dict[str, Any]:
        return {
            "schema": "experiment1-genpc-control-plane/v1",
            "frozen": self.frozen_fingerprints(),
            "association": association.to_dict(),
            "association_fingerprint": association.fingerprint,
            "evaluation": self.base.evaluation.to_dict(),
            "reconciliation": self.reconciliation.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": {
                "observed_partition": self.base.observed_partition.to_dict(),
                "generated_partition": self.base.generated_partition.to_dict(),
                "reference_partition": self.base.reference_partition.to_dict(),
                "preprocessing_fingerprint": self.base.preprocessing_fingerprint,
                "sampling_fingerprint": self.base.sampling_fingerprint,
                "renderer_fingerprint": self.base.renderer_fingerprint,
                "evaluation": self.base.evaluation.to_dict(),
                "reference_model_fingerprint": self.base.reference_model_fingerprint,
                "seed": self.base.seed,
            },
            "reconciliation": self.reconciliation.to_dict(),
            "frozen_fingerprints": self.frozen_fingerprints(),
        }
