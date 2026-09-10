"""A deliberately minimal reconciliation boundary for infrastructure tests."""

from __future__ import annotations

from typing import Protocol

from .contracts import Association, GeneratedHypothesis, Geometry, Observation, ReconciledResult, ReconciliationConfiguration


class Reconciler(Protocol):
    def reconcile(
        self, observation: Observation, hypothesis: GeneratedHypothesis, association: Association,
        configuration: ReconciliationConfiguration,
    ) -> ReconciledResult: ...


class ReferenceOverlayReconciler:
    """Preserves O and G verbatim while recording the injected association.

    This is not Structural Attachment and does not infer, edit, merge, or
    optimize an association.  It exists only to exercise the experimental API.
    """

    operator_id = "reference-overlay-v1"

    def reconcile(
        self, observation: Observation, hypothesis: GeneratedHypothesis, association: Association,
        configuration: ReconciliationConfiguration,
    ) -> ReconciledResult:
        if configuration.operator_id != self.operator_id:
            raise ValueError(f"{self.operator_id} cannot execute {configuration.operator_id!r}.")
        observation_ids = set(observation.geometry.surface_ids)
        hypothesis_ids = set(hypothesis.geometry.surface_ids)
        for entry in association.entries:
            if entry.observation_surface_id and entry.observation_surface_id not in observation_ids:
                raise ValueError(f"Association references unknown O surface {entry.observation_surface_id!r}.")
            if entry.generated_surface_id and entry.generated_surface_id not in hypothesis_ids:
                raise ValueError(f"Association references unknown G surface {entry.generated_surface_id!r}.")
        geometry = Geometry(
            geometry_id=f"overlay:{observation.geometry.geometry_id}:{hypothesis.geometry.geometry_id}",
            surfaces=observation.geometry.surfaces + hypothesis.geometry.surfaces,
            units=observation.geometry.units,
            coordinate_system=observation.geometry.coordinate_system,
            normal_orientation=observation.geometry.normal_orientation,
            metadata={"operator": self.operator_id, "input_geometry_preserved": True},
        )
        if hypothesis.geometry.units != observation.geometry.units:
            raise ValueError("Reference overlay refuses to silently convert units.")
        return ReconciledResult(
            result_id=f"overlay:{observation.observation_id}:{hypothesis.hypothesis_id}:{association.association_id}",
            geometry=geometry, association=association, reconciliation_configuration=configuration,
            notes={"unmatched_policy": configuration.unmatched_policy, "association_inferred": False,
                   "geometry_edited": False, "purpose": "infrastructure_smoke_operator"},
        )
