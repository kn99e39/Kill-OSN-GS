"""Deterministic smoke fixtures, not a scientific benchmark."""

from __future__ import annotations

from .contracts import (Association, AssociationEntry, EvaluationReference, EvaluationRegion, FrozenScene,
                        GeneratedHypothesis, Geometry, ModelProvenance, Observation, Surface)


def continuation_fixture() -> FrozenScene:
    """A two-surface continuation fixture with fixture-owned oracle regions."""
    observed = Surface("obs-visible", "observed", ((0, 0, 0), (0.5, 0, 0), (1, 0, 0)), normals=((0, 1, 0),) * 3,
                       tags=("full", "observed", "interface"))
    completion = Surface("gen-continuation", "hidden", ((1, 0, 0), (1.5, 0, 0), (2, 0, 0)), normals=((0, 1, 0),) * 3,
                         tags=("full", "hidden", "interface", "structural"))
    observation = Observation("smoke-continuation-O", Geometry("smoke-continuation-O-geometry", (observed,)),
                              preprocessing={"fixture": "continuation-v1", "coordinate_transform": "identity"},
                              sampling={"method": "explicit_fixture_points", "count": 3})
    hypothesis = GeneratedHypothesis("smoke-continuation-G", Geometry("smoke-continuation-G-geometry", (completion,)),
                                     generation={"fixture": "continuation-v1", "materialized": True, "seed": 7})
    oracle = Association("oracle-continuation", (AssociationEntry("associated", "obs-visible", "gen-continuation"),), "synthetic_oracle")
    reference = EvaluationReference("smoke-continuation-reference", Geometry("smoke-continuation-reference-geometry", (observed, completion)),
        (EvaluationRegion("full_geometry", ("obs-visible", "gen-continuation"), "Fixture-defined complete geometry."),
         EvaluationRegion("hidden_region", ("gen-continuation",), "Fixture-defined hidden continuation."),
         EvaluationRegion("interface_neighborhood", ("obs-visible", "gen-continuation"), "Fixture-defined join neighborhood."),
         EvaluationRegion("structural_region", ("gen-continuation",), "Fixture tag: structural continuation.")), oracle)
    provenance = ModelProvenance("synthetic-smoke-fixture", "local://fixtures", "continuation-v1", "continuation", 7,
        coordinate_transform={"name": "identity", "source": "fixture"}, extraction={"stage_semantics": "fixture generated hypothesis"},
        instrumentation={"enabled": False})
    return FrozenScene("smoke-continuation", observation, hypothesis, reference, provenance)


def unmatched_fixture() -> FrozenScene:
    observed = Surface("obs-anchor", "observed", ((0, 0, 0), (0.5, 0, 0)), normals=((0, 1, 0),) * 2, tags=("full", "observed"))
    detached = Surface("gen-detached", "hidden", ((4, 0, 0), (4.5, 0, 0)), normals=((0, 1, 0),) * 2, tags=("full", "hidden", "structural"))
    observation = Observation("smoke-unmatched-O", Geometry("smoke-unmatched-O-geometry", (observed,)),
        preprocessing={"fixture": "unmatched-v1", "coordinate_transform": "identity"}, sampling={"method": "explicit_fixture_points", "count": 2})
    hypothesis = GeneratedHypothesis("smoke-unmatched-G", Geometry("smoke-unmatched-G-geometry", (detached,)),
        generation={"fixture": "unmatched-v1", "materialized": True, "seed": 11})
    oracle = Association("oracle-unmatched", (AssociationEntry("unmatched_generated", None, "gen-detached"),), "synthetic_oracle")
    reference = EvaluationReference("smoke-unmatched-reference", Geometry("smoke-unmatched-reference-geometry", (observed, detached)),
        (EvaluationRegion("full_geometry", ("obs-anchor", "gen-detached"), "Fixture-defined complete geometry."),
         EvaluationRegion("hidden_region", ("gen-detached",), "Fixture-defined detached region."),
         EvaluationRegion("structural_region", ("gen-detached",), "Fixture tag: detached structure.")), oracle)
    provenance = ModelProvenance("synthetic-smoke-fixture", "local://fixtures", "unmatched-v1", "unmatched", 11,
        coordinate_transform={"name": "identity", "source": "fixture"}, extraction={"stage_semantics": "fixture generated hypothesis"},
        instrumentation={"enabled": False})
    return FrozenScene("smoke-unmatched", observation, hypothesis, reference, provenance)
