# Worklog #12 — Decomposed GenPC causality and real-relevance audit

Date: 2026-09-14 (Asia/Seoul)

## Preserved prior result

Worklog #11 is unchanged. Its 07136 result remains `INCONCLUSIVE`: no real
causal intervention was reached and no structural oracle was asserted.

This worklog separates the two questions that #11 could not jointly answer:

1. Can only A causally change the result when structural identity is known?
2. Do bounded real GenPC completions show local native O-G interaction
   multiplicity where that capability could be relevant?

R remains the same frozen GenPC operator: threshold `0.0001` under squared
Open3D distance semantics (`tau=0.01`), O-then-retained-G union, 20,000-point
FPS, and 20-neighbor statistical filtering at `std_ratio=2.5`.

## Part A — controlled causality

Two known-identity fixtures were constructed separately from
`AGatedGenPCFusion`; all dimensions are fixed multiples of tau with no sweep.

- `parallel_sheet`: a top-surface continuation and a legitimate nearby
  parallel underside at `0.5 * tau` separation.
- `junction`: a top-surface continuation and a legitimate near vertical sheet
  whose strip passes at `0.5 * tau` from the observed surface.

Each fixture froze `A_native`, `A_wrong`, and `A_oracle` before execution.
Only association eligibility differed. Actual frozen R was run in every case;
pre-FPS provenance is the primary evidence and post-FPS output is secondary.

```text
parallel_sheet
                G_correct removed / retained     G_distinct removed / retained
A_native        6,948 / 24,125                   6,369 / 0
A_wrong          0 / 31,073                      6,369 / 0
A_oracle        6,948 / 24,125                       0 / 6,369

junction
                G_correct removed / retained     G_distinct removed / retained
A_native        6,948 / 24,125                   1,351 / 35,898
A_wrong          0 / 31,073                      1,351 / 35,898
A_oracle        6,948 / 24,125                       0 / 37,249
```

The pre-FPS union changes are exactly attributable to the declared pair:
parallel-sheet native/oracle/wrong counts are `49,022 / 55,391 / 55,970`; the
junction counts are `84,920 / 86,271 / 91,868`. In both known-identity cases,
oracle A retains the structurally distinct valid surface while reconciling the
top duplicate; native/wrong removes the distinct geometry or leaves the
duplicate continuation as declared.

Controlled causal verdict: `CONTROLLED_CAUSAL_SIGNAL_PRESENT`.

Evidence: `reference_models/genpc/runs/controlled_causality_20260914/`.

## Part B — frozen bounded real audit

The pre-output audit list contains all 13 author-bundled input/GT pairs,
sorted lexicographically. Because there are more than ten, the frozen audit
set is the first ten:

```text
01184 01373 05117 05452 06127 06145 06188 06830 07089 07136
```

`07136` reused its existing frozen substrate without generation or
registration. Every other selected ID was attempted once, in order, at seed
42 and the same method-valid configuration. No failed sample was rerun.

| Sample | Materialisation / classification | Native interacting G | G components | O interaction density |
| --- | --- | ---: | ---: | ---: |
| 01184 | one-time failure: uncached online scheduler resolution | — | — | — |
| 01373 | one-time failure: upstream category mapping absent | — | — | — |
| 05117 | one-time failure: shared HF module-cache permission | — | — | — |
| 05452 | one-time failure: upstream compatibility stdout encoding | — | — | — |
| 06127 | NO_INTERACTION_AMBIGUITY | 8,168 | 36 | 0.06737 |
| 06145 | NO_INTERACTION_AMBIGUITY | 41,173 | 4 | 0.24859 |
| 06188 | NO_INTERACTION_AMBIGUITY | 22,875 | 4 | 0.08403 |
| 06830 | NO_INTERACTION_AMBIGUITY | 10,250 | 12 | 0.05645 |
| 07089 | one-time failure: upstream category mapping absent | — | — | — |
| 07136 | NO_INTERACTION_AMBIGUITY | 10,872 | 3 | 0.05423 |

For each successful frozen case, the audit records the native removed-G set,
nearest O provenance, squared distances, tau-connected interacting-G
components, O support, and G-to-GT distances as an explicitly separate
diagnostic. No GT distance was used as continuation identity.

No successful case had more than one tau-disconnected interacting G component
at the same local O neighborhood. Therefore no
`INTERACTION_AMBIGUITY_CANDIDATE` or defensible oracle relation was found.
Part C was not entered: no real region annotation or real A condition was
defined.

The raw reports, arrays, and fixed review views are persisted under
`reference_models/genpc/runs/real_ambiguity_audit_20260914/interaction_diagnostics/`.

## Regression

The relevant full unittest regression passed:

```text
Ran 31 tests in 0.185s
OK
```

## Architecture decision

Controlled A-only causality is real under the unchanged GenPC R. However, the
bounded non-cherry-picked real audit did not observe a multi-candidate native
interaction opportunity in any successfully frozen completion. The one-time
pipeline failures are retained as provenance and are not replaced by a
favorable sample.

ARCHITECTURE DECISION:
    DO_NOT_PROCEED_REAL_RELEVANCE_NOT_OBSERVED

