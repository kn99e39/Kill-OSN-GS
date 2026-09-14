"""Validate the frozen Worklog #12 detector on fixed controlled identities."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RUN_ROOT = ROOT / "runs" / "detector_validation_20260914"


def _positive_fixture_rows():
    from structural_experiment.controlled_fixtures import controlled_fixtures

    for fixture in controlled_fixtures():
        yield fixture.fixture_id, True, fixture.observed_points, fixture.generated_points, {
            "G_correct": fixture.generated_regions[f"{fixture.fixture_id}:G_correct"],
            "G_distinct": fixture.generated_regions[f"{fixture.fixture_id}:G_distinct"],
        }, fixture.protocol


def _negative_fixture_rows():
    from structural_experiment.detector_control_fixtures import detector_negative_controls

    for fixture in detector_negative_controls():
        yield fixture.fixture_id, fixture.known_ambiguity, fixture.observed_points, fixture.generated_points, fixture.generated_regions, fixture.protocol


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    from structural_experiment.detector_validation import frozen_detector_definition, provenance_accounting
    from structural_experiment.genpc_interaction_audit import native_interaction_diagnostic

    if RUN_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite validation evidence: {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    fixtures = {}
    for fixture_id, known_ambiguity, observed, generated, regions, protocol in (*_positive_fixture_rows(), *_negative_fixture_rows()):
        # The detector receives only O/G. Known identities are consulted below.
        interaction = native_interaction_diagnostic(observed, generated)
        provenance = provenance_accounting(interaction, regions)
        detector_state = "INTERACTION_AMBIGUITY_CANDIDATE" if interaction["multiple_component_opportunity"] else "NO_INTERACTION_AMBIGUITY"
        fixtures[fixture_id] = {
            "known_ambiguity": known_ambiguity,
            "detector_state": detector_state,
            "classification_pass": bool(known_ambiguity == interaction["multiple_component_opportunity"]),
            "protocol": protocol,
            "counts": {
                "observed": len(observed),
                "generated": len(generated),
                "native_interacting_generated": len(interaction["interacting_generated_indices"]),
                "detector_components": len(interaction["components"]),
            },
            "strongest_event": interaction["strongest_event"],
            "provenance_after_detection": provenance,
        }
    positives_pass = all(fixtures[name]["classification_pass"] for name in ("parallel_sheet", "junction"))
    negatives_pass = all(fixtures[name]["classification_pass"] for name in ("single_continuation", "noncompeting_distinct_surface"))
    if not positives_pass:
        decision = "REAL_AUDIT_DIAGNOSTIC_FALSE_NEGATIVE"
        attribution = "tau-connectivity merges known structurally distinct nearby surfaces into one interacting-G component"
    elif not negatives_pass:
        decision = "REAL_AUDIT_DIAGNOSTIC_FALSE_POSITIVE"
        attribution = "the fixed multiplicity criterion marks an ambiguity-free control as a candidate"
    else:
        decision = "REAL_AUDIT_DIAGNOSTIC_VALIDATED"
        attribution = "all fixed known-positive and known-negative fixture classifications passed"
    report = {
        "run_id": "detector_validation_20260914",
        "frozen_detector": frozen_detector_definition(),
        "fixtures": fixtures,
        "validity_matrix": [
            {"fixture": name, "known_ambiguity": data["known_ambiguity"], "detector_state": data["detector_state"], "pass": data["classification_pass"]}
            for name, data in fixtures.items()
        ],
        "measurement_decision": decision,
        "failure_attribution": attribution,
        "worklog_12_real_relevance_null": (
            "NOT ADMISSIBLE AS ARCHITECTURE-KILL EVIDENCE"
            if decision != "REAL_AUDIT_DIAGNOSTIC_VALIDATED"
            else "METHODOLOGICALLY ADMISSIBLE FOR THE FIVE SUCCESSFUL CASES BUT AUDIT COVERAGE REMAINS INCOMPLETE"
        ),
        "hard_stop_applied": decision != "REAL_AUDIT_DIAGNOSTIC_VALIDATED",
    }
    (RUN_ROOT / "detector_validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"measurement_decision": decision, "validity_matrix": report["validity_matrix"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
