"""Deterministic controlled-run execution, manifests, and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import fingerprint, write_json
from .contracts import Association, FrozenScene, ReconciledResult, ReconciliationConfiguration
from .evaluation import MetricResult, evaluate
from .reconcile import Reconciler


MANIFEST_SCHEMA = "structural-experiment/run-manifest/v1"


def controlled_fingerprints(scene: FrozenScene, configuration: ReconciliationConfiguration) -> dict[str, str]:
    """Everything here must stay fixed when A is varied."""
    return {
        "observation": scene.observation.fingerprint,
        "generated_hypothesis": scene.hypothesis.fingerprint,
        "observation_preprocessing": fingerprint(scene.observation.preprocessing),
        "observation_sampling": fingerprint(scene.observation.sampling),
        "reconciliation_configuration": configuration.fingerprint,
        "evaluation_reference": scene.evaluation_reference.fingerprint,
        "evaluation_regions": fingerprint([region.to_dict() for region in scene.evaluation_reference.regions]),
        "model_provenance": scene.model_provenance.fingerprint,
    }


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    scene: FrozenScene
    association: Association
    configuration: ReconciliationConfiguration
    result: ReconciledResult
    metrics: Mapping[str, Sequence[MetricResult]]
    warnings: tuple[str, ...]

    @property
    def controls(self) -> dict[str, str]:
        return controlled_fingerprints(self.scene, self.configuration)

    def manifest(self, artifact_locations: Mapping[str, str] | None = None) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA, "run_id": self.run_id, "scene_id": self.scene.scene_id,
            "scene_fingerprint": self.scene.fingerprint, "controlled_fingerprints": self.controls,
            "observation_fingerprint": self.scene.observation.fingerprint, "generated_hypothesis_fingerprint": self.scene.hypothesis.fingerprint,
            "association_id": self.association.association_id, "association_fingerprint": self.association.fingerprint,
            "reconciliation_configuration": self.configuration.to_dict(),
            "evaluation_configuration": {"reference_id": self.scene.evaluation_reference.reference_id,
                "regions": [region.to_dict() for region in self.scene.evaluation_reference.regions]},
            "model_provenance": self.scene.model_provenance.to_dict(), "result_fingerprint": self.result.fingerprint,
            "metric_results": {region: [metric.to_dict() for metric in values] for region, values in self.metrics.items()},
            "artifact_locations": dict(artifact_locations or {}), "warnings": list(self.warnings),
        }


class ExperimentHarness:
    def __init__(self, reconciler: Reconciler) -> None:
        self.reconciler = reconciler

    def run(self, scene: FrozenScene, association: Association, configuration: ReconciliationConfiguration) -> CompletedRun:
        result = self.reconciler.reconcile(scene.observation, scene.hypothesis, association, configuration)
        metric_results, warnings = evaluate(result, scene.evaluation_reference)
        run_id = fingerprint({"scene": scene.fingerprint, "association": association.fingerprint,
                              "reconciliation_configuration": configuration.fingerprint})[:20]
        return CompletedRun(run_id, scene, association, configuration, result, metric_results, tuple(warnings))

    def persist(self, completed: CompletedRun, destination: str | Path) -> Path:
        root = Path(destination) / completed.run_id
        root.mkdir(parents=True, exist_ok=True)
        locations = {"manifest": "manifest.json", "review": "review.json"}
        write_json(root / "manifest.json", completed.manifest(locations))
        write_json(root / "review.json", {
            "schema": "structural-experiment/review/v1", "run_id": completed.run_id,
            "observation": completed.scene.observation.to_dict(), "generated_hypothesis": completed.scene.hypothesis.to_dict(),
            "association": completed.association.to_dict(), "reconciled_result": completed.result.to_dict(),
            "evaluation_regions": [region.to_dict() for region in completed.scene.evaluation_reference.regions],
            "surface_id_ownership": {"observation": list(completed.scene.observation.geometry.surface_ids),
                "generated_hypothesis": list(completed.scene.hypothesis.geometry.surface_ids),
                "evaluation_reference": list(completed.scene.evaluation_reference.geometry.surface_ids)},
        })
        return root


@dataclass(frozen=True)
class ControlInvariantReport:
    preserved: bool
    association_fingerprints: tuple[str, ...]
    controls: Mapping[str, str]
    differences: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {"preserved": self.preserved, "association_fingerprints": list(self.association_fingerprints),
                "controls": dict(self.controls), "differences": {key: list(value) for key, value in self.differences.items()}}


def verify_only_association_changes(runs: Sequence[CompletedRun]) -> ControlInvariantReport:
    if len(runs) < 2:
        raise ValueError("At least two runs are needed to test the control invariant.")
    baseline = runs[0].controls
    differences: dict[str, tuple[str, ...]] = {}
    for key, expected in baseline.items():
        actual = tuple(run.controls[key] for run in runs)
        if any(value != expected for value in actual[1:]):
            differences[key] = actual
    associations = tuple(run.association.fingerprint for run in runs)
    if len(set(associations)) == 1:
        differences["association"] = associations
    return ControlInvariantReport(not differences, associations, baseline, differences)
