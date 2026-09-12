"""Execute the controlled Experiment 1 boundary audit for frozen GenPC 07136.

This script starts at the preserved O/G NPZ sidecars.  It never invokes a
generator or registration stage.  A_native is used only for the mandatory
native-equivalence proof.  The real structural conditions are intentionally
not created unless the preconditioned qualification is defensible.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
UPSTREAM = ROOT / "upstream"
RUN_ROOT = ROOT / "runs" / "experiment1_07136_20260913"
EQUIVALENCE_ROOT = RUN_ROOT / "native_equivalence"
REVIEW_ROOT = RUN_ROOT / "review_exports"
O_PATH = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "pre_fusion" / "O_aligned_exact.npz"
G_PATH = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "pre_fusion" / "G_aligned_exact.npz"
RNG_PATH = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "pre_fusion" / "numpy_rng_before_fusion.pkl"
CANONICAL_FUSED = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "07136" / "07136_fused.ply"
GT_PATH = UPSTREAM / "data" / "GT" / "07136.ply"
O_SHA256 = "31567e843d57cf20e11848bcb6dd5fb644444e5f70cd005547c9a9697fdae3ba"
G_SHA256 = "68c44193013ceee48fff1febe5650d79b97d5eb73d415a96f0eb4a579d4d5adc"
GT_SHA256 = "94c1d6e37ed3a282233d29882c45879b804902cf51a050a8859ed0d9f57ded72"
CANONICAL_FUSED_SHA256 = "2f0e8400ffe3d91e86ff7bf36460ecc7d9340443e2945ae056893fc2535e5541"
UPSTREAM_COMMIT = "bac9e7b59f4fea0eaaa936f24ef60f342f349829"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(UPSTREAM))


def one_region_partition(exact: Any, artifact_id: str, partition_id: str, source: str):
    from structural_experiment.experiment1_contracts import RegionHandle, RegionPartition

    return RegionPartition(
        partition_id=partition_id,
        raw_artifact=exact.raw_artifact_lineage(artifact_id=artifact_id),
        source=source,
        regions=(RegionHandle(f"{artifact_id}:all", tuple(range(exact.point_count)), label="all exact points"),),
        creation_protocol_id="genpc-exact-boundary-equivalence-v1",
        notes="All-points partition used only for A_native equivalence; no semantic annotation.",
    )


def gt_partition(gt_points: np.ndarray):
    from structural_experiment.experiment1_contracts import RawArtifactLineage, RegionHandle, RegionPartition

    lineage = RawArtifactLineage(
        artifact_id="GT-07136",
        original_file_sha256=GT_SHA256,
        representation="point_cloud",
        vertex_count=len(gt_points),
        face_count=0,
        source_locator=str(GT_PATH),
        source_revision=UPSTREAM_COMMIT,
    )
    return RegionPartition(
        partition_id="GT-07136-whole",
        raw_artifact=lineage,
        source="benchmark_gt",
        regions=(RegionHandle("GT-07136:all", tuple(range(len(gt_points))), label="all GT points"),),
        creation_protocol_id="genpc-exact-boundary-equivalence-v1",
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_control(o: Any, g: Any, gt_points: np.ndarray):
    from structural_experiment.experiment1_contracts import EvaluationSelector, Experiment1ControlPlane, FixedEvaluationDefinition, fingerprint
    from structural_experiment.genpc_contracts import GenPCExperiment1Control, GenPCReconciliationSpec

    operator_files = {
        "reg_xyz.py": sha256(UPSTREAM / "reg_xyz.py"),
        "dataUtils.py": sha256(UPSTREAM / "utils" / "dataUtils.py"),
    }
    downstream_fingerprint = fingerprint(
        {
            "upstream_commit": UPSTREAM_COMMIT,
            "operator_files": operator_files,
            "open3d_version": package_version("open3d"),
            "fpsample_version": package_version("fpsample"),
            "experimental_operator_revision": "structural_experiment.genpc_fusion-v1",
        }
    )
    reconciliation = GenPCReconciliationSpec(
        operator_id="genpc.remove_close_points_fps_statistical_filter",
        operator_revision=f"upstream-reg-xyz@{UPSTREAM_COMMIT}",
        distance_threshold=0.0001,
        union_semantics="O_then_retained_G",
        fps_requested_count=20000,
        noise_std_ratio=2.5,
        downstream_implementation_fingerprint=downstream_fingerprint,
        serialization_policy="float64_arrays; original_index_order; npz-sidecar-boundary",
    )
    observed_partition = one_region_partition(o, "O-07136", "O-07136-equivalence", "precomputed")
    generated_partition = one_region_partition(g, "G-07136", "G-07136-equivalence", "precomputed")
    reference_partition = gt_partition(gt_points)
    preprocessing = fingerprint(
        {
            "boundary": "frozen_registration_fusion_boundary",
            "O_sidecar_sha256": O_SHA256,
            "G_sidecar_sha256": G_SHA256,
            "no_additional_preprocessing": True,
        }
    )
    sampling = fingerprint(
        {
            "fps_requested_count": 20000,
            "numpy_rng_state_path": str(RNG_PATH),
            "serialization_policy": reconciliation.serialization_policy,
        }
    )
    evaluation = FixedEvaluationDefinition(
        evaluation_id="experiment1-07136-fixed-whole-scene",
        metric_id="symmetric_mean_nearest_distance",
        result=EvaluationSelector("result-whole", "GenPC-fused-07136", "whole_artifact"),
        reference=EvaluationSelector("GT-whole", "GT-07136", "whole_artifact"),
        protocol_revision="fixed-evaluation-v1; native-equivalence-only",
        preprocessing_fingerprint=preprocessing,
        sampling_fingerprint=sampling,
    )
    base = Experiment1ControlPlane(
        observed_partition=observed_partition,
        generated_partition=generated_partition,
        reference_partition=reference_partition,
        preprocessing_fingerprint=preprocessing,
        sampling_fingerprint=sampling,
        renderer_fingerprint=fingerprint({"renderer": "not run; frozen fused PLY is comparison target"}),
        evaluation=evaluation,
        reference_model_fingerprint=fingerprint({"GT_sha256": GT_SHA256, "upstream_commit": UPSTREAM_COMMIT}),
        seed=42,
    )
    return GenPCExperiment1Control(base=base, reconciliation=reconciliation), {
        "upstream_commit": UPSTREAM_COMMIT,
        "operator_files": operator_files,
        "open3d_version": package_version("open3d"),
        "fpsample_version": package_version("fpsample"),
    }


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def serialize_result(result: Any, *, control: Any, expected_points: np.ndarray, expected_colors: np.ndarray) -> dict[str, Any]:
    quantized_colors = np.clip(np.rint(result.colors * 255.0), 0, 255) / 255.0
    return {
        "point_accounting": result.accounting(
            observed_count=len(control.observed_points),
            generated_count=len(control.generated_points),
            spec=control.reconciliation,
        ),
        "native_equivalence": {
            "canonical_fused_sha256": sha256(CANONICAL_FUSED),
            "expected_canonical_fused_sha256": CANONICAL_FUSED_SHA256,
            "canonical_file_identity_pass": sha256(CANONICAL_FUSED) == CANONICAL_FUSED_SHA256,
            "final_point_array_exact_identity": bool(np.array_equal(result.points, expected_points)),
            "final_point_array_shape": list(result.points.shape),
            "final_ply_color_serialization_exact_identity": bool(np.array_equal(quantized_colors, expected_colors)),
            "color_comparison_policy": "canonical PLY stores Open3D colors as uint8; compare the exact equivalent quantization of frozen float64 colors",
            "exact_native_equivalence_pass": bool(
                np.array_equal(result.points, expected_points)
                and np.array_equal(quantized_colors, expected_colors)
            ),
        },
        "removal_attribution": list(result.removal_attribution),
    }


def make_review_exports(o: Any, g: Any, gt: np.ndarray, generated_distance: np.ndarray) -> dict[str, str]:
    """Write neutral, fixed-view qualification exports; no region labels are invented."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7136)

    def sample(points: np.ndarray, count: int = 20000) -> np.ndarray:
        if len(points) <= count:
            return points
        return points[rng.choice(len(points), size=count, replace=False)]

    O = sample(o.points)
    G = sample(g.points)
    GT = sample(gt)
    exports: dict[str, str] = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    projections = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"), (2, 1, "z", "y")]
    for axis, (left, right, left_name, right_name) in zip(axes.flat, projections):
        axis.scatter(O[:, left], O[:, right], s=0.3, alpha=0.35, c="#2878b5", label="O_exact")
        axis.scatter(G[:, left], G[:, right], s=0.3, alpha=0.16, c="#e78f2f", label="G_exact")
        axis.scatter(GT[:, left], GT[:, right], s=0.3, alpha=0.20, c="#4ba35b", label="GT")
        axis.set_xlabel(left_name)
        axis.set_ylabel(right_name)
        axis.set_title(f"{left_name} / {right_name}")
        axis.set_aspect("equal", adjustable="box")
    axes.flat[0].legend(markerscale=8)
    fig.suptitle("07136 frozen GenPC qualification review: neutral fixed projections")
    fig.tight_layout()
    overview = REVIEW_ROOT / "qualification_neutral_projections.png"
    fig.savefig(overview, dpi=180)
    plt.close(fig)
    exports["neutral_projections"] = str(overview)

    fig = plt.figure(figsize=(16, 12))
    views = [(25, -55), (25, 35), (60, -55), (5, 90)]
    for index, (elev, azim) in enumerate(views, start=1):
        axis = fig.add_subplot(2, 2, index, projection="3d")
        axis.scatter(O[:, 0], O[:, 1], O[:, 2], s=0.18, alpha=0.22, c="#2878b5", label="O_exact")
        axis.scatter(G[:, 0], G[:, 1], G[:, 2], s=0.18, alpha=0.10, c="#e78f2f", label="G_exact")
        axis.scatter(GT[:, 0], GT[:, 1], GT[:, 2], s=0.18, alpha=0.12, c="#4ba35b", label="GT")
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(f"elev{elev} az{azim}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    fig.suptitle("07136 frozen GenPC qualification review: neutral fixed 3D views")
    fig.tight_layout()
    views_path = REVIEW_ROOT / "qualification_neutral_3d_views.png"
    fig.savefig(views_path, dpi=180)
    plt.close(fig)
    exports["neutral_3d_views"] = str(views_path)

    distances = generated_distance
    distance_sample = sample(np.column_stack((g.points, distances)))
    fig = plt.figure(figsize=(16, 12))
    for index, (elev, azim) in enumerate(views, start=1):
        axis = fig.add_subplot(2, 2, index, projection="3d")
        axis.scatter(O[:, 0], O[:, 1], O[:, 2], s=0.16, alpha=0.12, c="black", label="O_exact")
        plot = axis.scatter(
            distance_sample[:, 0], distance_sample[:, 1], distance_sample[:, 2],
            s=0.18, alpha=0.20, c=distance_sample[:, 3], cmap="turbo", vmin=0.0, vmax=float(np.quantile(distances, 0.98)),
        )
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(f"elev{elev} az{azim}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    fig.colorbar(plot, ax=fig.axes, shrink=0.55, label="G -> GT nearest distance")
    fig.suptitle("07136 frozen GenPC qualification review: G support distance to GT")
    fig.tight_layout()
    distance_path = REVIEW_ROOT / "qualification_G_to_GT_distance.png"
    fig.savefig(distance_path, dpi=180)
    plt.close(fig)
    exports["G_to_GT_distance"] = str(distance_path)
    return exports


def main() -> int:
    if RUN_ROOT.exists():
        raise RuntimeError(f"refusing to reuse existing Experiment 1 run directory: {RUN_ROOT}")
    configure_imports()
    from scipy.spatial import cKDTree
    import open3d as o3d
    from structural_experiment.experiment1_artifacts import verify_experiment1_manifest, write_experiment1_manifest
    from structural_experiment.experiment1_contracts import AssociationDeclaration
    from structural_experiment.genpc_exact import load_exact_npz
    from structural_experiment.genpc_fusion import AGatedGenPCFusion, native_genpc_backend

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    o = load_exact_npz(O_PATH, expected_sha256=O_SHA256, expected_point_count=88916)
    g = load_exact_npz(G_PATH, expected_sha256=G_SHA256, expected_point_count=163840)
    if sha256(GT_PATH) != GT_SHA256:
        raise RuntimeError("GT SHA-256 does not match the frozen Worklog #10 binding")
    if sha256(CANONICAL_FUSED) != CANONICAL_FUSED_SHA256:
        raise RuntimeError("canonical frozen fused result SHA-256 does not match the declared binding")
    gt_cloud = o3d.io.read_point_cloud(str(GT_PATH))
    gt = np.asarray(gt_cloud.points, dtype=np.float64).copy()
    if len(gt) == 0:
        raise RuntimeError("GT point cloud is empty")

    control, downstream = build_control(o, g, gt)
    association = AssociationDeclaration("A_native", (("O-07136:all", "G-07136:all"),), rationale="all-permissive native equivalence only")
    manifest_path = EQUIVALENCE_ROOT / "manifest_A_native.json"
    manifest = write_experiment1_manifest(
        manifest_path,
        controls=control,
        association=association,
        artifact_locations={"O": str(O_PATH), "G": str(G_PATH), "GT": str(GT_PATH), "canonical_fused": str(CANONICAL_FUSED)},
    )
    verified_manifest = verify_experiment1_manifest(manifest_path, controls=control)

    with RNG_PATH.open("rb") as handle:
        numpy_rng_state = pickle.load(handle)
    fusion = AGatedGenPCFusion(
        observed_points=o.points,
        observed_colors=o.colors,
        generated_points=g.points,
        generated_colors=g.colors,
        observed_partition=control.base.observed_partition,
        generated_partition=control.base.generated_partition,
        reconciliation=control.reconciliation,
        backend=native_genpc_backend(),
    )
    native_result = fusion.run(association, numpy_rng_state=numpy_rng_state)
    canonical = o3d.io.read_point_cloud(str(CANONICAL_FUSED))
    expected_points = np.asarray(canonical.points, dtype=np.float64)
    expected_colors = np.asarray(canonical.colors, dtype=np.float64)
    equivalence = serialize_result(native_result, control=fusion, expected_points=expected_points, expected_colors=expected_colors)
    if not equivalence["native_equivalence"]["exact_native_equivalence_pass"]:
        raise RuntimeError("A_native did not reproduce the frozen GenPC fusion exactly; qualification is blocked")

    write_npz(
        EQUIVALENCE_ROOT / "A_native_exact_artifacts.npz",
        removed_g_mask=native_result.removed_g_mask,
        retained_g_indices=native_result.retained_g_indices,
        pre_fps_union_points=native_result.pre_fps_union_points,
        pre_fps_union_colors=native_result.pre_fps_union_colors,
        fps_selected_union_indices=native_result.fps_selected_union_indices,
        final_selected_union_indices=native_result.final_selected_union_indices,
        final_points=native_result.points,
        final_colors=native_result.colors,
    )
    (EQUIVALENCE_ROOT / "A_native_pre_fps_provenance.json").write_text(
        json.dumps(list(native_result.pre_fps_union_provenance), indent=2) + "\n", encoding="utf-8"
    )

    distances = cKDTree(gt).query(g.points, k=1)[0]
    O_to_GT = cKDTree(gt).query(o.points, k=1)[0]
    review_exports = make_review_exports(o, g, gt, distances)
    qualification = {
        "status": "completed",
        "sample": "07136",
        "qualification": "INCONCLUSIVE",
        "primary_failure_attribution": "AMBIGUOUS",
        "decision_basis": "The frozen geometry shows scene-level GT support inside G, but no defensible externally frozen semantic G_correct/G_wrong identities can be established from O, G, and GT alone without manufacturing a distractor or inferring regions after outcomes.",
        "correct_candidate_exists": "not_defensibly_established",
        "plausible_competing_association_exists": "not_defensibly_established",
        "scene_diagnostics": {
            "O_to_GT_mean": float(O_to_GT.mean()),
            "O_to_GT_median": float(np.median(O_to_GT)),
            "G_to_GT_mean": float(distances.mean()),
            "G_to_GT_median": float(np.median(distances)),
            "G_to_GT_quantiles": {str(q): float(np.quantile(distances, q)) for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]},
            "G_support_fraction_below_0.01": float(np.mean(distances < 0.01)),
            "G_support_fraction_below_0.02": float(np.mean(distances < 0.02)),
            "G_support_fraction_below_0.05": float(np.mean(distances < 0.05)),
            "interpretation": "G has a spatially mixed near-GT and far-GT population; nearest-distance support alone does not identify a structural continuation or a meaningful wrong continuation.",
        },
        "review_exports": review_exports,
        "conditions_run": [],
    }
    (REVIEW_ROOT / "qualification_07136.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "run_id": "experiment1_07136_20260913",
        "historical_state_preserved": {
            "exact_published_baseline": "BLOCKED",
            "genpc_method_valid_substrate": "READY",
            "worklog_9_and_10_modified": False,
            "upstream_modified": False,
            "generation_or_registration_rerun": False,
        },
        "exact_inputs": {"O": o.to_dict(), "G": g.to_dict(), "GT": {"path": str(GT_PATH), "sha256": GT_SHA256, "point_count": len(gt)}},
        "reconciliation": {"spec": control.reconciliation.to_dict(), "fingerprint": control.reconciliation.fingerprint, "downstream": downstream},
        "control_manifest": {"path": str(manifest_path), "manifest_fingerprint": manifest["manifest_fingerprint"], "verified": verified_manifest["control_plane"]["frozen_fingerprints"] == control.frozen_fingerprints()},
        "synthetic_and_contract_tests": "passed_before_real_execution",
        "a_native_equivalence": equivalence,
        "qualification": qualification,
        "primary_failure_attribution": "AMBIGUOUS",
        "architecture_judgment": "INCONCLUSIVE",
        "experiment1_result": "INCONCLUSIVE",
        "run_policy": "A_native was used only for the mandatory all-permissive equivalence proof; A_wrong/A_oracle were not defined or executed because real-sample qualification was inconclusive.",
    }
    (RUN_ROOT / "experiment1_07136_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
