"""Ambiguity-free controls for validating the frozen Part B detector.

These constructors deliberately contain no detector logic.  They reuse the
Part A point density and all geometry is expressed as a fixed multiple of
``tau``; their known identities are only for post-detection accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controlled_fixtures import (
    CONTINUATION_HALF_X_RATIO,
    GRID_STEP_RATIO,
    HALF_Y_RATIO,
    OBSERVED_HALF_X_RATIO,
    OVERLAP_X_RATIO,
    PARALLEL_OFFSET_RATIO,
    TAU,
    _axis,
    _grid,
)


@dataclass(frozen=True)
class DetectorControlFixture:
    fixture_id: str
    known_ambiguity: bool
    observed_points: np.ndarray
    generated_points: np.ndarray
    generated_regions: dict[str, np.ndarray]
    protocol: dict[str, float | int | str]


def _base() -> tuple[np.ndarray, np.ndarray]:
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    observed = _grid(_axis(-OBSERVED_HALF_X_RATIO, 0), y, plane="xy", fixed=0.0)
    correct = _grid(_axis(-OVERLAP_X_RATIO, CONTINUATION_HALF_X_RATIO), y, plane="xy", fixed=0.0)
    return observed, correct


def _fixture(fixture_id: str, generated: np.ndarray, regions: dict[str, np.ndarray], *, rule: str) -> DetectorControlFixture:
    return DetectorControlFixture(
        fixture_id=fixture_id,
        known_ambiguity=False,
        observed_points=_base()[0],
        generated_points=generated,
        generated_regions=regions,
        protocol={
            "tau": TAU,
            "grid_step_ratio": GRID_STEP_RATIO,
            "parallel_offset_ratio": PARALLEL_OFFSET_RATIO,
            "rule": rule,
        },
    )


def single_continuation_fixture() -> DetectorControlFixture:
    """One continuation only; no structurally distinct competitor exists."""
    _, correct = _base()
    return _fixture(
        "single_continuation",
        correct,
        {"G_correct": np.arange(len(correct), dtype=np.int64)},
        rule="fixed Part-A density; one horizontal continuation; no distinct surface",
    )


def noncompeting_distinct_surface_fixture() -> DetectorControlFixture:
    """A legitimate underside exists, but begins 8*tau beyond O's frontier."""
    _, correct = _base()
    y = _axis(-HALF_Y_RATIO, HALF_Y_RATIO)
    distinct = _grid(
        _axis(8, CONTINUATION_HALF_X_RATIO),
        y,
        plane="xy",
        fixed=-PARALLEL_OFFSET_RATIO * TAU,
    )
    generated = np.concatenate((correct, distinct), axis=0)
    return _fixture(
        "noncompeting_distinct_surface",
        generated,
        {
            "G_correct": np.arange(len(correct), dtype=np.int64),
            "G_distinct": np.arange(len(correct), len(generated), dtype=np.int64),
        },
        rule="fixed Part-A density; distinct underside starts at x=8*tau, outside O interaction opportunity",
    )


def detector_negative_controls() -> tuple[DetectorControlFixture, DetectorControlFixture]:
    return single_continuation_fixture(), noncompeting_distinct_surface_fixture()
