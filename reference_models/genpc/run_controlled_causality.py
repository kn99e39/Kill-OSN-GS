"""Part A: controlled A-only causality under the frozen GenPC R."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
UPSTREAM = ROOT / "upstream"
RUN_ROOT = ROOT / "runs" / "controlled_causality_20260914"
UPSTREAM_COMMIT = "bac9e7b59f4fea0eaaa936f24ef60f342f349829"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r_spec():
    from structural_experiment.experiment1_contracts import fingerprint
    from structural_experiment.genpc_contracts import GenPCReconciliationSpec

    downstream = fingerprint({
        "upstream_commit": UPSTREAM_COMMIT,
        "reg_xyz_sha256": sha256(UPSTREAM / "reg_xyz.py"),
        "data_utils_sha256": sha256(UPSTREAM / "utils" / "dataUtils.py"),
        "operator": "structural_experiment.genpc_fusion-v1",
    })
    return GenPCReconciliationSpec(
        operator_id="genpc.remove_close_points_fps_statistical_filter",
        operator_revision=f"upstream-reg-xyz@{UPSTREAM_COMMIT}",
        distance_threshold=0.0001,
        union_semantics="O_then_retained_G",
        fps_requested_count=20000,
        noise_std_ratio=2.5,
        downstream_implementation_fingerprint=downstream,
        serialization_policy="float64_arrays; original_index_order; controlled-fixture-v1",
    )


def distances(points: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    from scipy.spatial import cKDTree

    forward = cKDTree(gt).query(points, k=1)[0]
    reverse = cKDTree(points).query(gt, k=1)[0]
    return {
        "result_count": int(len(points)),
        "gt_count": int(len(gt)),
        "result_to_gt_mean": float(forward.mean()),
        "gt_to_result_mean": float(reverse.mean()),
        "symmetric_mean": float((forward.mean() + reverse.mean()) / 2),
        "result_to_gt_p95": float(np.quantile(forward, 0.95)),
        "gt_to_result_p95": float(np.quantile(reverse, 0.95)),
    }


def main() -> int:
    if RUN_ROOT.exists():
        raise RuntimeError(f"refusing to reuse controlled-causality run directory: {RUN_ROOT}")
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(UPSTREAM))
    from scipy.spatial import cKDTree
    from structural_experiment.controlled_fixtures import controlled_fixtures
    from structural_experiment.genpc_fusion import AGatedGenPCFusion, native_genpc_backend

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    spec = r_spec()
    all_results = {"run_id": "controlled_causality_20260914", "reconciliation": {"spec": spec.to_dict(), "fingerprint": spec.fingerprint}, "fixtures": {}}
    rng_state = np.random.RandomState(42).get_state()
    for fixture in controlled_fixtures():
        operator = AGatedGenPCFusion(
            observed_points=fixture.observed_points,
            observed_colors=fixture.observed_colors,
            generated_points=fixture.generated_points,
            generated_colors=fixture.generated_colors,
            observed_partition=fixture.observed_partition,
            generated_partition=fixture.generated_partition,
            reconciliation=spec,
            backend=native_genpc_backend(),
        )
        interaction = cKDTree(fixture.observed_points).query(fixture.generated_points, k=1)[0] < 0.01
        fixture_result = {"protocol": fixture.protocol, "counts": {"O": len(fixture.observed_points), "G": len(fixture.generated_points), "GT": len(fixture.gt_points)}, "conditions": {}}
        for condition, association in fixture.associations.items():
            result = operator.run(association, numpy_rng_state=rng_state)
            region_counts = {}
            for region_id, indices in fixture.generated_regions.items():
                removed = result.removed_g_mask[indices]
                region_counts[region_id] = {
                    "total": int(len(indices)),
                    "native_interaction_zone": int(interaction[indices].sum()),
                    "removed": int(removed.sum()),
                    "retained": int((~removed).sum()),
                    "interaction_zone_removed": int((removed & interaction[indices]).sum()),
                    "interaction_zone_retained": int(((~removed) & interaction[indices]).sum()),
                }
            pre_metrics = distances(result.pre_fps_union_points, fixture.gt_points)
            final_metrics = distances(result.points, fixture.gt_points)
            attribution = {}
            for item in result.removal_attribution:
                key = f"{item['observed_region_id']}->{item['generated_region_id']}"
                attribution[key] = attribution.get(key, 0) + 1
            fixture_result["conditions"][condition] = {
                "association": association.to_dict(),
                "association_fingerprint": association.fingerprint,
                "point_accounting": result.accounting(observed_count=len(fixture.observed_points), generated_count=len(fixture.generated_points), spec=spec),
                "generated_region_survival": region_counts,
                "removal_attribution_by_pair": attribution,
                "pre_fps_local_metrics": pre_metrics,
                "post_fps_local_metrics": final_metrics,
            }
            np.savez_compressed(
                RUN_ROOT / f"{fixture.fixture_id}_{condition}_provenance.npz",
                removed_g_mask=result.removed_g_mask,
                retained_g_indices=result.retained_g_indices,
                pre_fps_union_points=result.pre_fps_union_points,
                fps_selected_union_indices=result.fps_selected_union_indices,
                final_selected_union_indices=result.final_selected_union_indices,
                final_points=result.points,
            )
        native = fixture_result["conditions"]["A_native"]["generated_region_survival"]
        wrong = fixture_result["conditions"]["A_wrong"]["generated_region_survival"]
        oracle = fixture_result["conditions"]["A_oracle"]["generated_region_survival"]
        correct_id, distinct_id = fixture.generated_regions.keys()
        fixture_result["causal_contract_pass"] = bool(
            native[distinct_id]["removed"] > 0
            and wrong[distinct_id]["removed"] > 0
            and wrong[correct_id]["interaction_zone_retained"] > 0
            and oracle[distinct_id]["retained"] == oracle[distinct_id]["total"]
            and oracle[correct_id]["interaction_zone_removed"] > 0
        )
        all_results["fixtures"][fixture.fixture_id] = fixture_result
    all_results["controlled_causal_verdict"] = "CONTROLLED_CAUSAL_SIGNAL_PRESENT" if all(
        entry["causal_contract_pass"] for entry in all_results["fixtures"].values()
    ) else "CONTROLLED_CAUSAL_SIGNAL_ABSENT"
    (RUN_ROOT / "controlled_causality_result.json").write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": all_results["controlled_causal_verdict"], "fixtures": {key: value["causal_contract_pass"] for key, value in all_results["fixtures"].items()}}, indent=2))
    return 0 if all_results["controlled_causal_verdict"] == "CONTROLLED_CAUSAL_SIGNAL_PRESENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
