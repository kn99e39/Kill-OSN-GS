"""Measure pre-FPS GT coverage change caused by frozen native GenPC removal."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RUN_ROOT = ROOT / "runs" / "native_removal_harm_20260914_v2"
REAL_RUN_ROOT = ROOT / "runs" / "real_ambiguity_audit_20260914"


def _pre_fps_backend(native_backend):
    from structural_experiment.genpc_fusion import GenPCFusionBackend

    def all_indices(points, _requested_count):
        return np.arange(len(points), dtype=np.int64)

    def identity_filter(points, colors, _std_ratio):
        return np.asarray(points).copy(), np.asarray(colors).copy(), np.arange(len(points), dtype=np.int64)

    return GenPCFusionBackend(
        close_point_filter=native_backend.close_point_filter,
        fps_sampler=all_indices,
        noise_filter=identity_filter,
    )


def _spec():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "upstream"))
    from run_controlled_causality import r_spec

    return r_spec()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_fixture(fixture, spec, native_backend, output_dir: Path) -> dict:
    from structural_experiment.experiment1_contracts import AssociationDeclaration
    from structural_experiment.genpc_fusion import AGatedGenPCFusion
    from structural_experiment.removal_harm_measurement import array_identity, coverage_harm

    operator = AGatedGenPCFusion(
        observed_points=fixture.observed_points,
        observed_colors=fixture.observed_colors,
        generated_points=fixture.generated_points,
        generated_colors=fixture.generated_colors,
        observed_partition=fixture.observed_partition,
        generated_partition=fixture.generated_partition,
        reconciliation=spec,
        backend=_pre_fps_backend(native_backend),
    )
    results = {}
    for condition in ("A_empty", "A_native"):
        association = (
            AssociationDeclaration("A_empty", ())
            if condition == "A_empty"
            else fixture.associations["A_native"]
        )
        result = operator.run(association)
        if condition == "A_empty" and not np.array_equal(result.retained_g_indices, np.arange(len(fixture.generated_points))):
            raise AssertionError(f"A_empty removed generated points in {fixture.fixture_id}")
        results[condition] = result
    empty = results["A_empty"]
    native = results["A_native"]
    expected_all = np.concatenate((fixture.observed_points, fixture.generated_points), axis=0)
    if not np.array_equal(empty.pre_fps_union_points, expected_all):
        raise AssertionError(f"A_empty pre-FPS union identity mismatch in {fixture.fixture_id}")
    harm = coverage_harm(
        fixture.observed_points,
        fixture.generated_points,
        native.removed_g_mask,
        fixture.gt_points,
    )
    if not np.array_equal(harm["U_all"], empty.pre_fps_union_points):
        raise AssertionError(f"U_all does not match A_empty in {fixture.fixture_id}")
    if not np.array_equal(harm["U_native"], native.pre_fps_union_points):
        raise AssertionError(f"U_native does not match A_native in {fixture.fixture_id}")
    native_provenance = [{"kind": kind, "index": int(index)} for kind, index in native.pre_fps_union_provenance]
    empty_provenance = [{"kind": kind, "index": int(index)} for kind, index in empty.pre_fps_union_provenance]
    if empty_provenance != harm["all_provenance"] or native_provenance != harm["native_provenance"]:
        raise AssertionError(f"pre-FPS provenance mismatch in {fixture.fixture_id}")
    all_identity = array_identity(empty.pre_fps_union_points, empty_provenance)
    native_identity = array_identity(native.pre_fps_union_points, native_provenance)
    arrays_path = output_dir / f"{fixture.fixture_id}_pre_fps_measurement.npz"
    np.savez_compressed(
        arrays_path,
        U_all=empty.pre_fps_union_points,
        U_native=native.pre_fps_union_points,
        removed_g_mask=native.removed_g_mask,
        retained_g_indices=native.retained_g_indices,
        d_all=harm["d_all"],
        d_native=harm["d_native"],
        delta=harm["delta"],
        nearest_all_indices=harm["nearest_all_indices"],
        nearest_native_indices=harm["nearest_native_indices"],
    )
    summary = {
        "fixture": fixture.fixture_id,
        "known_state": "known_positive" if fixture.fixture_id in {"parallel_sheet", "junction"} else "known_negative",
        "protocol": fixture.protocol,
        "A_empty": {
            "removed_G_point_count": 0,
            "pre_fps_union_identity": all_identity,
        },
        "A_native": {
            "removed_G_point_count": int(len(native.removed_g_indices)),
            "removed_G_indices_sha256": _sha256_bytes(np.ascontiguousarray(native.removed_g_indices).tobytes()),
            "pre_fps_union_identity": native_identity,
        },
        "harm": {key: value for key, value in harm.items() if key not in {"U_all", "U_native", "all_provenance", "native_provenance", "d_all", "d_native", "delta", "nearest_all_indices", "nearest_native_indices"}},
        "measurement_pass": bool(
            (fixture.fixture_id in {"parallel_sheet", "junction"} and harm["summary"]["strictly_positive_delta_count"] > 0)
            or (fixture.fixture_id in {"single_continuation", "noncompeting_distinct_surface"} and harm["summary"]["strictly_positive_delta_count"] == 0 and harm["summary"]["negative_delta_beyond_tolerance_count"] == 0)
        ),
        "arrays": str(arrays_path),
    }
    return summary, harm, native


def _real_paths(sample: str) -> tuple[Path, Path, Path] | None:
    if sample == "07136":
        base = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "pre_fusion"
    else:
        sample_result = REAL_RUN_ROOT / sample / "result.json"
        if not sample_result.exists() or json.loads(sample_result.read_text(encoding="utf-8")).get("status") != "completed":
            return None
        base = REAL_RUN_ROOT / sample / "pre_fusion"
    gt = ROOT / "upstream" / "data" / "GT" / f"{sample}.ply"
    if not all(path.exists() for path in (base / "O_aligned_exact.npz", base / "G_aligned_exact.npz", gt)):
        return None
    return base / "O_aligned_exact.npz", base / "G_aligned_exact.npz", gt


def _load_gt(path: Path) -> np.ndarray:
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def _run_real(sample: str, spec, native_backend, output_dir: Path) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from structural_experiment.experiment1_contracts import AssociationDeclaration
    from structural_experiment.genpc_fusion import AGatedGenPCFusion
    from structural_experiment.removal_harm_measurement import array_identity, coverage_harm

    paths = _real_paths(sample)
    if paths is None:
        return {"sample": sample, "result": "MATERIALIZATION_NOT_AVAILABLE"}, {}, np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    o_path, g_path, gt_path = paths
    observed = np.load(o_path)["points"]
    generated = np.load(g_path)["points"]
    gt = _load_gt(gt_path)
    # The successful real substrate has no real region annotation in Part B.
    # Reconstruct the exact all-pairs native A over whole O and whole G.
    from structural_experiment.experiment1_contracts import RawArtifactLineage, RegionHandle, RegionPartition

    def partition(artifact_id, count):
        lineage = RawArtifactLineage(artifact_id, _sha256_bytes(np.asarray(count, dtype=np.int64).tobytes()), "point_cloud", count, 0)
        return RegionPartition(
            f"{artifact_id}-partition",
            lineage,
            "precomputed",
            (RegionHandle(f"{artifact_id}:all", tuple(range(count))),),
            "frozen-real-pre-fps-whole-cloud-v1",
        )

    observed_partition = partition(f"{sample}:O", len(observed))
    generated_partition = partition(f"{sample}:G", len(generated))
    colors_o = np.zeros_like(observed)
    colors_g = np.zeros_like(generated)
    operator = AGatedGenPCFusion(
        observed_points=observed,
        observed_colors=colors_o,
        generated_points=generated,
        generated_colors=colors_g,
        observed_partition=observed_partition,
        generated_partition=generated_partition,
        reconciliation=spec,
        backend=_pre_fps_backend(native_backend),
    )
    a_empty = AssociationDeclaration("A_empty", ())
    a_native = AssociationDeclaration("A_native", ((f"{sample}:O:all", f"{sample}:G:all"),))
    empty = operator.run(a_empty)
    native = operator.run(a_native)
    harm = coverage_harm(observed, generated, native.removed_g_mask, gt)
    if not np.array_equal(empty.pre_fps_union_points, harm["U_all"]) or not np.array_equal(native.pre_fps_union_points, harm["U_native"]):
        raise AssertionError(f"real pre-FPS union mismatch for {sample}")
    arrays_path = output_dir / f"{sample}_pre_fps_measurement.npz"
    np.savez_compressed(
        arrays_path,
        U_all=harm["U_all"],
        U_native=harm["U_native"],
        removed_g_mask=native.removed_g_mask,
        retained_g_indices=native.retained_g_indices,
        d_all=harm["d_all"],
        d_native=harm["d_native"],
        delta=harm["delta"],
        nearest_all_indices=harm["nearest_all_indices"],
        nearest_native_indices=harm["nearest_native_indices"],
    )
    summary = {
        "sample": sample,
        "result": "NATIVE_REMOVAL_HARM_PRESENT" if harm["summary"]["strictly_positive_delta_count"] else "NO_NATIVE_REMOVAL_HARM",
        "counts": {"O": len(observed), "G": len(generated), "GT": len(gt)},
        "A_empty_pre_fps_union_identity": array_identity(harm["U_all"], harm["all_provenance"]),
        "A_native_pre_fps_union_identity": array_identity(harm["U_native"], harm["native_provenance"]),
        "removed_G_point_count": int(len(native.removed_g_indices)),
        "harm": {key: value for key, value in harm.items() if key not in {"U_all", "U_native", "all_provenance", "native_provenance", "d_all", "d_native", "delta", "nearest_all_indices", "nearest_native_indices"}},
        "arrays": str(arrays_path),
    }
    return summary, harm, observed, generated, gt, native.removed_g_mask


def _export_review(sample: str, observed: np.ndarray, generated: np.ndarray, gt: np.ndarray, removed_mask: np.ndarray, harm: dict, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = harm["harmed_gt_rows"]
    rows = sorted(rows, key=lambda row: (-row["delta"], row["gt_index"]))
    top_rows = rows[:5]
    csv_path = output_dir / f"{sample}_strongest_harm.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("rank", "gt_index", "delta", "d_all", "d_native", "removed_G_index", "removed_G_to_GT_distance", "removed_G_to_nearest_O_distance"))
        writer.writeheader()
        for rank, row in enumerate(top_rows, 1):
            writer.writerow({"rank": rank, **{key: row[key] for key in writer.fieldnames if key != "rank"}})
    centers = gt[np.asarray([row["gt_index"] for row in top_rows], dtype=np.int64)] if top_rows else np.empty((0, 3))
    center = centers[0] if len(centers) else np.mean(gt, axis=0)
    radius = 10.0 * 0.01
    local = np.linalg.norm(np.concatenate((observed, generated, gt), axis=0) - center, axis=1) <= radius
    o_local = local[:len(observed)]
    g_local = local[len(observed):len(observed) + len(generated)]
    gt_local = local[len(observed) + len(generated):]
    removed_local = g_local & removed_mask
    retained_local = g_local & ~removed_mask
    harmed_indices = np.asarray([row["gt_index"] for row in top_rows], dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    views = ((0, 1, "XY"), (0, 2, "XZ"))
    for axis, (a, b, title) in zip(axes.flat[:2], views):
        axis.scatter(observed[:, a], observed[:, b], s=0.2, c="#2878b5", alpha=0.2, label="O")
        axis.scatter(generated[~removed_mask, a], generated[~removed_mask, b], s=0.2, c="#e78f2f", alpha=0.15, label="G retained")
        axis.scatter(generated[removed_mask, a], generated[removed_mask, b], s=0.2, c="#d62728", alpha=0.2, label="G removed")
        axis.scatter(gt[:, a], gt[:, b], s=0.2, c="#4ba35b", alpha=0.12, label="GT")
        axis.scatter(gt[harmed_indices, a], gt[harmed_indices, b], s=8, c="#c51b8a", label="harmed GT")
        axis.set_title(f"{sample} full {title}")
    for axis, (a, b, title) in zip(axes.flat[2:], views):
        axis.scatter(observed[o_local, a], observed[o_local, b], s=4, c="#2878b5", alpha=0.4, label="O")
        axis.scatter(generated[retained_local, a], generated[retained_local, b], s=4, c="#e78f2f", alpha=0.5, label="G retained")
        axis.scatter(generated[removed_local, a], generated[removed_local, b], s=7, c="#d62728", alpha=0.65, label="G removed")
        axis.scatter(gt[gt_local, a], gt[gt_local, b], s=4, c="#4ba35b", alpha=0.5, label="GT")
        axis.scatter(gt[harmed_indices, a], gt[harmed_indices, b], s=25, c="#c51b8a", label="harmed GT")
        axis.set_xlim(center[a] - radius, center[a] + radius); axis.set_ylim(center[b] - radius, center[b] + radius)
        axis.set_title(f"{sample} fixed local {title}; radius=10*tau")
    axes[0, 0].legend(markerscale=5)
    for axis in axes.flat:
        axis.set_aspect("equal", adjustable="box"); axis.set_xlabel("coordinate"); axis.set_ylabel("coordinate")
    fig.tight_layout()
    png_path = output_dir / f"{sample}_native_removal_harm_review.png"
    fig.savefig(png_path, dpi=160); plt.close(fig)
    return {"png": str(png_path), "table": str(csv_path), "top_gt_indices": [int(row["gt_index"]) for row in top_rows], "local_radius": radius, "local_radius_formula": "10 * tau"}


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    from structural_experiment.controlled_fixtures import controlled_fixtures
    from structural_experiment.removal_harm_fixtures import removal_harm_negative_controls
    from structural_experiment.genpc_fusion import native_genpc_backend

    if RUN_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite removal-harm evidence: {RUN_ROOT}")
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    output_dir = RUN_ROOT / "pre_fps_measurements"
    output_dir.mkdir()
    spec = _spec()
    native_backend = native_genpc_backend()
    controls = list(controlled_fixtures()) + list(removal_harm_negative_controls())
    controlled = {}
    for fixture in controls:
        summary, _, _ = _run_fixture(fixture, spec, native_backend, output_dir)
        controlled[fixture.fixture_id] = summary
    real = {}
    successful = ("06127", "06145", "06188", "06830", "07136")
    for sample in successful:
        summary, harm, observed, generated, gt, removed_mask = _run_real(sample, spec, native_backend, output_dir)
        if summary["result"] == "MATERIALIZATION_NOT_AVAILABLE":
            real[sample] = summary
            continue
        real[sample] = summary
    harmed_samples = [sample for sample in successful if real[sample].get("result") == "NATIVE_REMOVAL_HARM_PRESENT"]
    review = None
    if harmed_samples:
        first = harmed_samples[0]
        _, harm, observed, generated, gt, removed_mask = _run_real(first, spec, native_backend, output_dir)
        review = {"selected_first_harmed_sample": first, **_export_review(first, observed, generated, gt, removed_mask, harm, output_dir)}
    controls_valid = all(item["measurement_pass"] for item in controlled.values())
    measurement_decision = "REMOVAL_HARM_MEASUREMENT_VALIDATED" if controls_valid else "REMOVAL_HARM_MEASUREMENT_INVALID"
    real_state = "REAL_RECONCILIATION_HARM_PRESENT" if harmed_samples else "REAL_RECONCILIATION_HARM_NOT_OBSERVED_IN_FROZEN_SUCCESSFUL_SET"
    report = {
        "run_id": "native_removal_harm_20260914_v2",
        "measurement": {
            "primary_stage": "pre-FPS only",
            "FPS_executed": False,
            "statistical_filter_executed": False,
            "comparison": "A_empty: O union G versus A_native: O union G_retained_native",
            "reconciliation_spec": spec.to_dict(),
            "reconciliation_fingerprint": spec.fingerprint,
            "numerical_policy": "64 * float64 machine epsilon * max(1.0, maximum absolute scene coordinate); no scientific threshold",
        },
        "controlled": controlled,
        "real_successful_frozen_set": real,
        "real_harmed_samples_in_lexicographic_order": harmed_samples,
        "review": review,
        "measurement_decision": measurement_decision,
        "real_evidence": real_state,
        "structural_interpretation": "NOT YET ADJUDICATED" if harmed_samples else "not entered; no native removal harm observed",
    }
    report_path = RUN_ROOT / "native_removal_harm_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"measurement_decision": measurement_decision, "real_evidence": real_state, "harmed_samples": harmed_samples}, indent=2))
    return 0 if measurement_decision == "REMOVAL_HARM_MEASUREMENT_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
