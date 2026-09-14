"""Part B: O-G interaction audit over the frozen bounded real sample set."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
UPSTREAM = ROOT / "upstream"
AUDIT_SET = ROOT / "REAL_AMBIGUITY_AUDIT_SET_20260914.json"
RUN_ROOT = ROOT / "runs" / "real_ambiguity_audit_20260914"
OUT = RUN_ROOT / "interaction_diagnostics"


def source_paths(sample: str) -> tuple[Path, Path] | None:
    if sample == "07136":
        base = ROOT / "runs" / "frozen_g_instrumented_07136_v6" / "pre_fusion"
        return base / "O_aligned_exact.npz", base / "G_aligned_exact.npz"
    result = RUN_ROOT / sample / "result.json"
    if not result.exists() or json.loads(result.read_text(encoding="utf-8")).get("status") != "completed":
        return None
    base = RUN_ROOT / sample / "pre_fusion"
    return base / "O_aligned_exact.npz", base / "G_aligned_exact.npz"


def draw(sample: str, O: np.ndarray, G: np.ndarray, GT: np.ndarray, interaction: dict, gt_distance: np.ndarray) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(sample))
    def choose(values: np.ndarray, limit: int = 20000) -> np.ndarray:
        return values if len(values) <= limit else values[rng.choice(len(values), limit, replace=False)]
    interaction_idx = np.asarray(interaction["interacting_generated_indices"], dtype=np.int64)
    components = np.asarray(interaction["component_id_by_generated"], dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    o, g, gt = choose(O), choose(G), choose(GT)
    axes[0, 0].scatter(o[:, 0], o[:, 1], s=.2, c="#2878b5", alpha=.35, label="O")
    axes[0, 0].scatter(g[:, 0], g[:, 1], s=.2, c="#e78f2f", alpha=.15, label="G")
    axes[0, 0].scatter(gt[:, 0], gt[:, 1], s=.2, c="#4ba35b", alpha=.15, label="GT")
    axes[0, 0].legend(markerscale=8); axes[0, 0].set_title("O / G / GT fixed XY")
    axes[0, 1].scatter(o[:, 0], o[:, 2], s=.2, c="#2878b5", alpha=.25)
    axes[0, 1].scatter(g[:, 0], g[:, 2], s=.2, c="#e78f2f", alpha=.12)
    axes[0, 1].scatter(gt[:, 0], gt[:, 2], s=.2, c="#4ba35b", alpha=.12); axes[0, 1].set_title("O / G / GT fixed XZ")
    if len(interaction_idx):
        plot_indices = interaction_idx if len(interaction_idx) <= 20000 else interaction_idx[rng.choice(len(interaction_idx), 20000, replace=False)]
        points = G[plot_indices]
        axes[1, 0].scatter(points[:, 0], points[:, 1], s=.5, c=components[plot_indices], cmap="tab20", alpha=.7)
    axes[1, 0].set_title("native interacting G component IDs")
    sample_idx = rng.choice(len(G), min(len(G), 20000), replace=False)
    scatter = axes[1, 1].scatter(G[sample_idx, 0], G[sample_idx, 1], s=.3, c=gt_distance[sample_idx], cmap="turbo", alpha=.6)
    fig.colorbar(scatter, ax=axes[1, 1], label="G → GT distance (diagnostic only)"); axes[1, 1].set_title("G → GT distance")
    for axis in axes.flat:
        axis.set_aspect("equal", adjustable="box"); axis.set_xlabel("x"); axis.set_ylabel("y/z")
    fig.suptitle(f"{sample}: frozen O-G native interaction audit")
    fig.tight_layout()
    path = OUT / f"{sample}_interaction_review.png"; fig.savefig(path, dpi=160); plt.close(fig)
    return str(path)


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    from scipy.spatial import cKDTree
    import open3d as o3d
    from structural_experiment.genpc_interaction_audit import native_interaction_diagnostic

    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite diagnostics: {OUT}")
    OUT.mkdir(parents=True)
    audit = json.loads(AUDIT_SET.read_text(encoding="utf-8"))
    report = {"audit_set": audit["selected_audit_sample_ids"], "samples": {}}
    for sample in audit["selected_audit_sample_ids"]:
        paths = source_paths(sample)
        if paths is None:
            result = json.loads((RUN_ROOT / sample / "result.json").read_text(encoding="utf-8"))
            report["samples"][sample] = {"classification": "MATERIALIZATION_FAILED", "error": result.get("error")}
            continue
        o_path, g_path = paths
        O = np.load(o_path)["points"]
        G = np.load(g_path)["points"]
        GT = np.asarray(o3d.io.read_point_cloud(str(UPSTREAM / "data" / "GT" / f"{sample}.ply")).points)
        interaction = native_interaction_diagnostic(O, G)
        gt_distance = cKDTree(GT).query(G, k=1)[0]
        classification = "INTERACTION_AMBIGUITY_CANDIDATE" if interaction["multiple_component_opportunity"] else "NO_INTERACTION_AMBIGUITY"
        if classification == "INTERACTION_AMBIGUITY_CANDIDATE":
            classification = "ORACLE_RELATION_UNRESOLVED"
        interaction_npz = OUT / f"{sample}_interaction_arrays.npz"
        np.savez_compressed(interaction_npz, interacting_generated_indices=np.asarray(interaction["interacting_generated_indices"]), nearest_observed_indices=np.asarray(interaction["nearest_observed_indices"]), squared_distances=np.asarray(interaction["squared_distances"]), component_id_by_generated=np.asarray(interaction["component_id_by_generated"]), gt_distances=gt_distance)
        report["samples"][sample] = {"classification": classification, "interaction_count": len(interaction["interacting_generated_indices"]), "component_count": len(interaction["components"]), "interaction_density_over_observed": interaction["interaction_density_over_observed"], "strongest_event": interaction["strongest_event"], "review_export": draw(sample, O, G, GT, interaction, gt_distance), "arrays": str(interaction_npz), "gt_distance_note": "separate diagnostic; not structural identity"}
        (OUT / f"{sample}_interaction_raw.json").write_text(json.dumps(interaction, indent=2) + "\n", encoding="utf-8")
    (OUT / "real_interaction_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value["classification"] for key, value in report["samples"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
