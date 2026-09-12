# Worklog #11 — Experiment 1 on frozen GenPC 07136

Date: 2026-09-13 (Asia/Seoul)

## Scope and preservation

Experiment 1 was executed from the already frozen GenPC registration/fusion
boundary. No generation, registration, upstream edit, or prior-run edit was
performed.

The historical state remains:

```text
EXACT PUBLISHED BASELINE: BLOCKED
GENPC METHOD-VALID SUBSTRATE: READY
```

Worklogs #9 and #10, the frozen substrate manifest, and the canonical frozen
fused result were preserved unchanged.

## Exact frozen inputs

The new exact adapter binds the lossless NPZ sidecars directly. It validates
the file SHA-256, exact `points`/`colors` key layout, float64 `N x 3` shape,
point count, content fingerprints, and read-only original index order. It does
not normalize, align, reorder, resample, quantize, or segment.

```text
O: reference_models/genpc/runs/frozen_g_instrumented_07136_v6/pre_fusion/O_aligned_exact.npz
   file SHA-256: 31567e843d57cf20e11848bcb6dd5fb644444e5f70cd005547c9a9697fdae3ba
   points: 88,916
   points content SHA-256: eab06cb84fa1c92e8aee568b4799847db0b980aab38586fa17f5c9e87629cd9f
   colors content SHA-256: 97ad69aff58e6330ebc9a0eb4d3e2ccf654d5bbc7a4112c16cf24b993382c279

G: reference_models/genpc/runs/frozen_g_instrumented_07136_v6/pre_fusion/G_aligned_exact.npz
   file SHA-256: 68c44193013ceee48fff1febe5650d79b97d5eb73d415a96f0eb4a579d4d5adc
   points: 163,840
   points content SHA-256: 8fdbdc5ce11eec40112dc7737a5a696f016b36397a269e7cda6ea73b45a9ce82
   colors content SHA-256: 77955d91aee237495cd68fae7a8201efc8aa0fb9778a59cec6e8b350ee03f80b

GT: reference_models/genpc/upstream/data/GT/07136.ply
    file SHA-256: 94c1d6e37ed3a282233d29882c45879b804902cf51a050a8859ed0d9f57ded72
    points: 77,108
```

## Frozen reconciliation R

`GenPCReconciliationSpec` wraps the historical
`Experiment1ControlPlane`; the existing partition, association, evaluation,
and manifest contracts remain in use. R is frozen as:

```text
operator: genpc.remove_close_points_fps_statistical_filter
revision: upstream-reg-xyz@bac9e7b59f4fea0eaaa936f24ef60f342f349829
distance_threshold: 0.0001 (Open3D squared-distance semantics)
union: O followed by retained G
FPS requested count: 20,000
statistical noise removal: std_ratio=2.5, nb_neighbors=20
downstream implementation fingerprint: 56db856f7cb22373afd52038240ec920e2796347ea41d258603ed49627030789
R fingerprint: 38e55a4e897295bbbeddce415b7b7ee5e7da74502d571664f4d5f91a82a317d4
```

The downstream fingerprint records the preserved upstream `reg_xyz.py` and
`dataUtils.py` hashes, upstream commit, Open3D `0.19.0`, fpsample `1.0.2`, and
the new experiment operator revision.

## A-only invariant and synthetic contract tests

The shared `verify_only_association_changed` invariant now accepts the small
GenPC control wrapper. A focused negative test proves that changing R's
distance threshold is rejected; R's threshold, FPS count, noise setting, and
operator identity all participate in its content fingerprint.

The full relevant unittest regression passed:

```text
Ran 28 tests in 0.114s
OK
```

This includes exact-NPZ identity, R fingerprint/invariant rejection,
many-to-many A, unmatched-G preservation, native all-pairs semantics, the
existing Experiment 1 contracts, fixed evaluation, and real-data substrate
tests.

## A_native equivalence

The all-points O/G partitions and `A_native` were used only for the required
native equivalence proof. The saved NumPy state immediately before native FPS
was restored. The A-gated operator reproduced the frozen downstream sequence
without touching pristine GenPC code.

```text
O:                         88,916
G:                        163,840
removed G:                 10,872
retained G:              152,968
pre-FPS union:            241,884
FPS selected:              20,000
final:                      19,763
```

Canonical frozen fused result SHA-256:
`2f0e8400ffe3d91e86ff7bf36460ecc7d9340443e2945ae056893fc2535e5541`.

The final coordinate array matched the canonical fused PLY exactly, in order.
The float64 sidecar colors matched the canonical PLY exactly after the same
Open3D uint8 color serialization. Therefore `A_native` exact equivalence
passed.

## 07136 qualification

The qualification audit used only O_exact, G_exact, and GT. Neutral fixed
projection and 3D exports, plus G-to-GT nearest-distance views, are persisted
under the ignored runtime directory:

```text
reference_models/genpc/runs/experiment1_07136_20260913/review_exports/
```

Scene diagnostics include:

```text
O -> GT mean nearest distance: 0.001896535481035986
G -> GT mean nearest distance: 0.07133123259651968
G -> GT median nearest distance: 0.05800524785204369
G points within 1 cm of GT: 12.7337646484375%
G points within 2 cm of GT: 24.2962646484375%
```

G therefore contains spatially mixed near-GT and far-GT geometry. That is
scene-level support, not a defensible externally frozen semantic identity for
`G_correct`, and it does not establish a meaningful plausible `G_wrong`.
Creating those labels from proximity, or choosing them after observing
condition metrics, would manufacture the ambiguity prohibited by the
protocol.

Qualification is consequently `INCONCLUSIVE`. No real region annotation,
diagnostic ROI, `A_wrong`, or `A_oracle` was defined or executed. The only A
condition persisted is the equivalence-only `A_native`.

## Persisted evidence

The reproducible entrypoint is:

```text
reference_models/genpc/run_experiment1_genpc_07136.py
```

The run report and native artifact bundle are under:

```text
reference_models/genpc/runs/experiment1_07136_20260913/
```

The manifest was verified before native execution. Its fingerprint is:
`de62e6fbf20918900269805ca40404f63728aad5636f03662749544b157047f8`.

Primary failure attribution: `AMBIGUOUS`.

Architecture judgment: `INCONCLUSIVE`.

EXACT PUBLISHED BASELINE: BLOCKED
GENPC METHOD-VALID SUBSTRATE: READY

EXPERIMENT 1 RESULT: INCONCLUSIVE
