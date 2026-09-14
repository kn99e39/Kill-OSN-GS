"""Full GenPC-compatible negative controls for removal-harm measurement."""

from __future__ import annotations

import numpy as np

from .controlled_fixtures import (
    CONTINUATION_HALF_X_RATIO,
    GRID_STEP_RATIO,
    HALF_Y_RATIO,
    OBSERVED_HALF_X_RATIO,
    OVERLAP_X_RATIO,
    PARALLEL_OFFSET_RATIO,
    TAU,
    ControlledFixture,
    _axis,
    _grid,
    _partition,
)
from .experiment1_contracts import AssociationDeclaration


def _fixture(
    fixture_id: str,
    observed: np.ndarray,
    correct: np.ndarray,
    distinct: np.ndarray,
    gt: np.ndarray,
    *,
    rule: str,
) -> ControlledFixture:
    generated = np.concatenate((correct, distinct), axis=0)
    correct_indices = np.arange(len(correct), dtype=np.int64)
    distinct_indices = np.arange(len(correct), len(generated), dtype=np.int64)
    observed_id = f"{fixture_id}:O_surface"
    correct_id = f"{fixture_id}:G_correct"
    observed_partition = _partition(f"{fixture_id}:O", observed, ((observed_id, np.arange(len(observed))),))
    generated_regions = [(correct_id, correct_indices)]
    region_map = {"G_correct": correct_indices}
    links = ((observed_id, correct_id),)
    if len(distinct):
        generated_regions.append((f"{fixture_id}:G_distinct", distinct_indices))
        region_map["G_distinct"] = distinct_indices
        links += ((observed_id, f"{fixture_id}:G_distinct"),)
    generated_partition = _partition(f"{fixture_id}:G", generated, tuple(generated_regions))
    protocol = {
        "tau": TAU,
        "grid_step_ratio": GRID_STEP_RATIO,
        "parallel_offset_ratio": PARALLEL_OFFSET_RATIO,
        "rule": rule,
    }
    return ControlledFixture(
        fixture_id=fixture_id,
        observed_points=observed,
        observed_colors=np.tile(np.asarray([[0.20, 0.45, 0.80]]), (len(observed), 1)),
        generated_points=generated,
        generated_colors=np.concatenate((
            np.tile(np.asarray([[0.20, 0.75, 0.35]]), (len(correct), 1)),
            np.tile(np.asarray([[0.95, 0.45, 0.15]]), (len(distinct), 1)),
        )),
        gt_points=gt,
        observed_partition=observed_partition,
        generated_partition=generated_partition,
        generated_regions={f"{fixture_id}:{key}": value for key, value in region_map.items()},
        associations={
            "A_empty": AssociationDeclaration("A_empty", ()),
            "A_native": AssociationDeclaration("A_native", links),
        },
        protocol=protocol,
    )


def single_continuation_fixture() -> ControlledFixture:
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    observed = _grid(_axis(-OBSERVED_HALF_X_RATIO, 0), y, plane="xy", fixed=0.0)
    # Keep the generated continuation entirely inside O's already observed
    # surface.  Native removal may discard this duplicate, but cannot remove
    # non-redundant GT coverage at the frontier.
    correct = _grid(_axis(-OVERLAP_X_RATIO, -2), y, plane="xy", fixed=0.0)
    gt_top = _grid(_axis(-OBSERVED_HALF_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    return _fixture(
        "single_continuation",
        observed,
        correct,
        np.empty((0, 3), dtype=np.float64),
        gt_top,
        rule="fixed Part-A density; one horizontal continuation; no distinct surface",
    )


def noncompeting_distinct_surface_fixture() -> ControlledFixture:
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    observed = _grid(_axis(-OBSERVED_HALF_X_RATIO, 0), y, plane="xy", fixed=0.0)
    correct = _grid(_axis(-OVERLAP_X_RATIO, -2), y, plane="xy", fixed=0.0)
    distinct = _grid(
        _axis(8, CONTINUATION_HALF_X_RATIO),
        y,
        plane="xy",
        fixed=-PARALLEL_OFFSET_RATIO * TAU,
    )
    gt_top = _grid(_axis(-OBSERVED_HALF_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    return _fixture(
        "noncompeting_distinct_surface",
        observed,
        correct,
        distinct,
        np.concatenate((gt_top, distinct), axis=0),
        rule="fixed Part-A density; distinct underside starts at x=8*tau, outside O interaction opportunity",
    )


def removal_harm_negative_controls() -> tuple[ControlledFixture, ControlledFixture]:
    return single_continuation_fixture(), noncompeting_distinct_surface_fixture()
