"""Post-hoc validation accounting for the unmodified Worklog #12 detector."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np

from .genpc_interaction_audit import TAU


DETECTOR_SOURCE = Path(__file__).with_name("genpc_interaction_audit.py")
FROZEN_DETECTOR_SHA256 = "d3b3049e3b31eed56292d1ab861c3c0dc5d551a1170b929d2b779742b83e3c02"


def frozen_detector_definition() -> dict[str, object]:
    """Return the Part B implementation identity and its immutable semantics."""
    digest = sha256(DETECTOR_SOURCE.read_bytes()).hexdigest()
    if digest != FROZEN_DETECTOR_SHA256:
        raise RuntimeError(
            "Worklog #12 detector source changed; validation must not proceed "
            f"(expected {FROZEN_DETECTOR_SHA256}, found {digest})"
        )
    return {
        "module": "structural_experiment.genpc_interaction_audit.native_interaction_diagnostic",
        "source_sha256": digest,
        "tau": TAU,
        "native_interaction": "nearest observed distance < tau",
        "generated_connectivity": "interacting G cKDTree.query_pairs(tau)",
        "observed_local_neighborhood": "nearest observed point only",
        "multiplicity": "unique tau-connected component IDs per nearest observed point",
        "candidate_criterion": "strongest event component_count > 1",
        "ranking": "max(component_count, interacting_generated_count, -observed_index)",
    }


def _region_for_index(index: int, regions: Mapping[str, np.ndarray]) -> str:
    for region_id, indices in regions.items():
        if np.any(indices == index):
            return region_id
    return "unlabelled"


def provenance_accounting(interaction: Mapping[str, object], regions: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Compare frozen-detector output to known identities after detection.

    No region information is accepted by or influences the detector itself.
    """
    component_rows = []
    for component in interaction["components"]:  # type: ignore[index]
        members = [int(index) for index in component["generated_indices"]]
        membership: dict[str, int] = {}
        for index in members:
            region_id = _region_for_index(index, regions)
            membership[region_id] = membership.get(region_id, 0) + 1
        component_rows.append({
            "component_id": component["component_id"],
            "generated_count": component["generated_count"],
            "known_surface_membership": membership,
            "known_structural_surface_count": len(membership),
            "contains_correct_and_distinct": bool(membership.get("G_correct") and membership.get("G_distinct")),
        })
    supports: dict[int, set[str]] = {}
    for generated_index, observed_index in zip(
        interaction["interacting_generated_indices"], interaction["nearest_observed_indices"]  # type: ignore[index]
    ):
        supports.setdefault(int(observed_index), set()).add(_region_for_index(int(generated_index), regions))
    both = sorted(
        observed_index
        for observed_index, labels in supports.items()
        if {"G_correct", "G_distinct"}.issubset(labels)
    )
    collapsed = [row["component_id"] for row in component_rows if row["contains_correct_and_distinct"]]
    return {
        "components": component_rows,
        "local_observed_neighborhoods_receiving_both_correct_and_distinct": len(both),
        "local_observed_indices_receiving_both_correct_and_distinct": both,
        "component_collapse_across_known_structural_surfaces": bool(collapsed),
        "collapsed_component_ids": collapsed,
    }
