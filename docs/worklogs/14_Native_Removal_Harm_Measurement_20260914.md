# Worklog #14 — Direct native-removal harm measurement

Date: 2026-09-14 (Asia/Seoul)

## Scope and preservation

Worklogs #11, #12, and #13 remain unchanged.  Worklog #13's
`REAL_AUDIT_DIAGNOSTIC_FALSE_NEGATIVE` result is preserved as historical
evidence; its invalid structural-multiplicity detector was not repaired or
used here.  The frozen real O/G/GT substrates were reused without generation
or registration.

This batch measures only the direct effect of frozen native GenPC close-point
reconciliation.  It does not infer whether any affected geometry has the
wrong structural attachment.

## Measurement definition

For fixed O, G, GT, and the existing frozen GenPC R:

- `A_empty` has no eligible O/G region pair and therefore produces
  `U_all = O union G`;
- `A_native` uses the existing native all-pairs close-point path and produces
  `U_native = O union G_retained_native`;
- the primary comparison is pre-FPS only: for every GT point `q`,
  `delta(q) = d_native(q) - d_all(q)`.

The existing `AGatedGenPCFusion` and native Open3D close-point backend were
used.  FPS and statistical filtering were replaced by an identity boundary in
the measurement runner and were not executed.  The frozen reconciliation
fingerprint was `6fb8de83ed486081e724ea2c3fa1faa7e75d8f4d9517b6cf62effa152f03940f`.

Numerical policy was fixed before reading outcomes:

```text
tolerance = 64 * finfo(float64).eps
            * max(1.0, maximum absolute coordinate over O, G, and GT)
meaningful harm: delta > tolerance
meaningful negative: delta < -tolerance
```

No scientific threshold was introduced.  Complete `U_all`, `U_native`,
removal masks, `d_all`, `d_native`, `delta`, nearest indices, and provenance
are persisted in each fixture/sample NPZ sidecar.

## Measurement validation on controls

The existing positive fixtures and new ambiguity-free negative controls were
run once with the actual frozen GenPC close-point R and no geometry sweep.

| Fixture | Known state | Removed G | delta sum | delta mean | delta median | delta max | Positive delta count | GT median 1-NN | Mean delta / NN | Max delta / NN | Mean U_all→GT | Mean U_native→GT | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `parallel_sheet` | positive | 13,317 | 33.775 | 0.000603448275862069 | 0 | 0.005 | 6,948 | 0.0025 | 0.24137931034482768 | 2.000000000000001 | 0 | 0 | yes |
| `junction` | positive | 8,299 | 5.790000000000001 | 0.00006666666666666668 | 0 | 0.005 | 1,737 | 0.0025 | 0.026666666666666686 | 2.000000000000001 | 0 | 0 | yes |
| `single_continuation` | negative | 4,825 | 0 | 0 | 0 | 0 | 0 | 0.0025 | 0 | 0 | 0 | 0 | yes |
| `noncompeting_distinct_surface` | negative | 4,825 | 0 | 0 | 0 | 0 | 0 | 0.0025 | 0 | 0 | 0 | 0 | yes |

The negative controls remove only generated points already redundantly covered
by O.  The distinct surface in `noncompeting_distinct_surface` starts at
`x = 8 * tau`, outside the native O interaction opportunity.  Neither control
has a meaningful negative delta.

The positive controls show non-zero GT coverage harm under the direct
measurement.  Their positive-delta quantiles are:

```text
parallel_sheet: p50=0.005, p90=0.005, p95=0.005, p99=0.005, p100=0.005
junction:       p50=0.0025, p90=0.005, p95=0.005, p99=0.005, p100=0.005
```

## Frozen successful real set

The five successful Worklog #12 cases were evaluated in frozen lexicographic
order.  No structural labels were assigned.

| Sample | Real result | O | G | GT | Removed G | delta sum | delta mean | delta median | delta max | Positive delta count | GT median 1-NN | Mean U_all→GT | Mean U_native→GT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `06127` | `NATIVE_REMOVAL_HARM_PRESENT` | 42,658 | 163,840 | 33,155 | 8,168 | 1.6917180848836697 | 0.000051024523748564915 | 0 | 0.006478687661853458 | 1,044 | 0.003275439098157348 | 0.015577918629984481 | 1.9779600437383036 | 0.08906939341737287 | 0.0924091042842926 |
| `06145` | `NATIVE_REMOVAL_HARM_PRESENT` | 55,597 | 163,840 | 27,139 | 41,173 | 4.451945557423844 | 0.00016404235813492922 | 0 | 0.008544179672006167 | 3,321 | 0.002871139707145703 | 0.05713492719516922 | 2.975884332880556 | 0.028213013911453977 | 0.033802858675161584 |
| `06188` | `NATIVE_REMOVAL_HARM_PRESENT` | 86,336 | 163,840 | 75,995 | 22,875 | 4.979407403376074 | 0.00006552282917792058 | 0 | 0.007616774652203442 | 2,869 | 0.0018034692844667653 | 0.03633154705891973 | 4.223401373012852 | 0.02689745594932558 | 0.028989236778313273 |
| `06830` | `NO_NATIVE_REMOVAL_HARM` | 46,253 | 163,840 | 165,546 | 10,250 | 0 | 0 | 0 | 0 | 0 | 0.0030438461122663848 | 0 | 0 | 2.6883792744699333 | 2.69633500622755 |
| `07136` | `NATIVE_REMOVAL_HARM_PRESENT` | 88,916 | 163,840 | 77,108 | 10,872 | 2.030015242875289 | 0.00002632690826709792 | 0 | 0.007345641441347153 | 1,367 | 0.0016722122547773851 | 0.01574375991724969 | 4.392768573703012 | 0.04690508434001795 | 0.04876780670736584 |


The separate precision distributions (mean removed-G→GT / mean retained-G→GT)
are `06127: 0.007976732271904385 / 0.11619624351153347`,
`06145: 0.004011034775671417 / 0.04776159375103645`,
`06188: 0.006112149972943841 / 0.044389341772529414`,
`06830: 2.533267342602879 / 2.7305494785080433`, and
`07136: 0.005462595643956861 / 0.07601275958875506`.  The full p95/max
distributions and positive-delta quantiles are retained in the machine report.

The positive-delta quantiles, removed-G→GT mean, retained-G→GT mean, and all
per-harmed-GT provenance rows are in the machine report.  The precision
tradeoff is kept separate: for the four harmed samples, mean U_native→GT is
higher than mean U_all→GT; `06830` has no GT coverage harm despite its
different U-level precision means.

## First harmed review selection

The first harmed sample in the frozen lexicographic order is `06127`.  A fixed
review export was created around the five highest-delta GT loci, ordered by
`(-delta, gt_index)`, with local radius `10 * tau = 0.1`.  It shows O, all G
partitioned only as native removed/retained, GT, harmed GT points, nearest
removed-G provenance, and nearest-O provenance.  No `correct`, `wrong`,
continuation, or distinct-surface label is applied.

Local artifacts:

- `reference_models/genpc/runs/native_removal_harm_20260914_v2/pre_fps_measurements/06127_native_removal_harm_review.png`
- `reference_models/genpc/runs/native_removal_harm_20260914_v2/pre_fps_measurements/06127_strongest_harm.csv`
- `reference_models/genpc/runs/native_removal_harm_20260914_v2/native_removal_harm_report.json`

The first five GT indices are `3356, 2032, 29094, 1289, 6071`.

## Final states

```text
STRUCTURAL INTERPRETATION:
    NOT YET ADJUDICATED

MEASUREMENT DECISION:
    REMOVAL_HARM_MEASUREMENT_VALIDATED

REAL EVIDENCE:
    REAL_RECONCILIATION_HARM_PRESENT
```

This batch stops at direct removal harm.  It does not claim that the removed
geometry is structurally wrong and does not proceed to solver design.

## Testing

Focused tests passed before the v2 controlled/real execution:

```text
Ran 7 tests in 0.016s
OK
```

The relevant full regression is run once at batch closure.
