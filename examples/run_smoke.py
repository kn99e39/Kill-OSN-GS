"""Run two synthetic A conditions solely to verify experimental isolation."""

from __future__ import annotations

import argparse

from structural_experiment.canonical import write_json
from structural_experiment.contracts import Association, AssociationEntry, ReconciliationConfiguration
from structural_experiment.fixtures import continuation_fixture
from structural_experiment.reconcile import ReferenceOverlayReconciler
from structural_experiment.runner import ExperimentHarness, verify_only_association_changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoke", help="Directory for deterministic JSON review artifacts.")
    args = parser.parse_args()
    scene = continuation_fixture()
    associated = Association("smoke-associated", (AssociationEntry("associated", "obs-visible", "gen-continuation"),), "manual")
    unmatched = Association("smoke-unmatched", (
        AssociationEntry("unmatched_observation", "obs-visible", None),
        AssociationEntry("unmatched_generated", None, "gen-continuation"),
    ), "manual")
    harness = ExperimentHarness(ReferenceOverlayReconciler())
    configuration = ReconciliationConfiguration()
    first, second = (harness.run(scene, association, configuration) for association in (associated, unmatched))
    first_dir, second_dir = harness.persist(first, args.output), harness.persist(second, args.output)
    report = verify_only_association_changes((first, second))
    write_json(f"{args.output}/control_invariant.json", report.to_dict())
    if not report.preserved:
        raise SystemExit("Control invariant failed.")
    print(f"Control invariant preserved. Review artifacts: {first_dir}, {second_dir}")


if __name__ == "__main__":
    main()
