"""Controlled experimental substrate for frozen 3D completion artifacts."""

from .artifacts import LaSCompArtifactExporter, read_frozen_scene, write_frozen_scene
from .fixtures import continuation_fixture, unmatched_fixture
from .reconcile import ReferenceOverlayReconciler
from .runner import ExperimentHarness, verify_only_association_changes

__all__ = ["ExperimentHarness", "LaSCompArtifactExporter", "ReferenceOverlayReconciler", "continuation_fixture",
           "read_frozen_scene", "unmatched_fixture", "verify_only_association_changes", "write_frozen_scene"]
