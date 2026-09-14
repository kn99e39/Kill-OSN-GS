# Worklog #13 — Validation of the Worklog #12 real-audit detector

Date: 2026-09-14 (Asia/Seoul)

## Scope and preservation

Worklog #12 is preserved unchanged, including its historical controlled result
and architecture decision.  This worklog tests whether the exact measurement
instrument used by its Part B real audit could detect the known structural
ambiguity used by Part A.  No real sample was added or rerun, and the Part B
detector, `tau`, component construction, local-neighborhood rule, ranking, and
candidate criterion were not edited.

The new negative-control constructors use the Part A grid density and fixed
tau-derived geometry.  The detector consumes only pre-reconciliation O/G
geometry, exactly as it did in Part B; known fixture identities are supplied
only after detection for provenance accounting.  Because the positive-control
failure below invokes the prescribed hard stop, no new frozen-R causal run,
matched structural pair, or real audit work was performed in this batch.

## Frozen detector

The exact implementation is
`structural_experiment.genpc_interaction_audit.native_interaction_diagnostic`.
Its source SHA-256 is:

```text
d3b3049e3b31eed56292d1ab861c3c0dc5d551a1170b929d2b779742b83e3c02
```

Its frozen definition is:

- native interaction: nearest observed O distance `< tau`, with `tau = 0.01`;
- generated components: `cKDTree.query_pairs(tau)` over interacting G;
- local O neighborhood: each G point's nearest O point only;
- multiplicity: unique generated-component IDs at that O neighborhood;
- candidate: strongest event has `component_count > 1`;
- ranking: `max(component_count, interacting_generated_count, -observed_index)`.

## Controlled replay

The unmodified detector was executed once each on the two existing Part A
fixtures and two frozen negative controls.  Semantic identities were not passed
to it.  They were joined to component membership only after its output was
complete.

| Fixture | Known ambiguity | Interacting G | Detector components | O neighborhoods with correct + distinct support | Candidate state | Pass |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `parallel_sheet` | yes | 13,317 | 1 | 6,369 | `NO_INTERACTION_AMBIGUITY` | no |
| `junction` | yes | 8,299 | 1 | 193 | `NO_INTERACTION_AMBIGUITY` | no |
| `single_continuation` | no | 6,948 | 1 | 0 | `NO_INTERACTION_AMBIGUITY` | yes |
| `noncompeting_distinct_surface` | no | 6,948 | 1 | 0 | `NO_INTERACTION_AMBIGUITY` | yes |

The detector component accounting confirms the positive-control failure is not
an absence of competing support:

| Fixture | Component | `G_correct` members | `G_distinct` members | Known structural surfaces | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| `parallel_sheet` | 0 | 6,948 | 6,369 | 2 | `COMPONENT_COLLAPSE_ACROSS_KNOWN_STRUCTURAL_SURFACES` |
| `junction` | 0 | 6,948 | 1,351 | 2 | `COMPONENT_COLLAPSE_ACROSS_KNOWN_STRUCTURAL_SURFACES` |

In both known-positive cases, the surfaces are within the detector's exact
tau-connectivity scale and it merges them into one component.  Its
component-multiplicity condition can therefore never become true for that
merged local opportunity, despite both provenance classes supporting the same
nearest-O neighborhoods.

The two negative controls were not spuriously marked as candidates.  In
`single_continuation` there is only the continuation.  In
`noncompeting_distinct_surface`, the fixed-density distinct underside begins at
`x = 8 * tau`, outside the observed-surface interaction opportunity.

## Measurement decision and hard stop

The positive-control contract failed for both known-positive fixtures.  The
failure is attributable to tau-connectivity merging structurally distinct
nearby generated surfaces, rather than a real absence of competing structural
identity.

```text
MEASUREMENT DECISION:
    REAL_AUDIT_DIAGNOSTIC_FALSE_NEGATIVE

WORKLOG #12 REAL-RELEVANCE NULL:
    NOT ADMISSIBLE AS ARCHITECTURE-KILL EVIDENCE
```

Accordingly, this batch stops here.  It does not tune or replace the detector,
does not execute matched-pair or ambiguity-free causal follow-ons, and does
not extend or rerun the frozen real audit set.

## Evidence and regression

Machine-readable validation evidence is local runtime output at:
`reference_models/genpc/runs/detector_validation_20260914/detector_validation_report.json`.
It records the frozen detector fingerprint, raw classification matrix, and
post-detection provenance accounting.

Focused tests passed before the controlled replay:

```text
Ran 6 tests in 0.062s
OK
```

The full regression is run once at batch closure.
