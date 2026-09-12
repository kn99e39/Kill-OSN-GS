"""Capture the corrected pre-fusion boundary from the completed 07136 replay.

The previous full replay already completed RMBG and InstantMesh.  This helper
copies those exact replay artifacts and invokes only the published registration
function once more, with the corrected frame-walking hook, so the transform
lineage is complete without another generative pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from run_instrumented_equivalence_07136 import (
    BASELINE_OUTPUT,
    BASELINE_ROOT,
    FLAG,
    FusionBoundaryHook,
    INSTANTMESH,
    ROOT,
    STAGE2_FILES,
    UPSTREAM,
    compare_files,
    compare_pointclouds,
    configure_cuda_runtime,
    cuda_memory,
    load_cfg,
    pcd_meta,
    set_seed,
    sha256,
)


SOURCE_ROOT = ROOT / "runs" / "instrumented_equivalence_07136_offline"
SOURCE_OUTPUT = SOURCE_ROOT / FLAG
RUN_ROOT = ROOT / "runs" / "instrumented_boundary_07136"
OUTPUT = RUN_ROOT / FLAG
RESULT_PATH = RUN_ROOT / "boundary_capture_result.json"


def main() -> int:
    configure_cuda_runtime()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {
        "run_id": "instrumented_boundary_07136_20260912",
        "status": "started",
        "equivalence_pass": False,
        "source_replay_root": str(SOURCE_ROOT),
        "baseline_root": str(BASELINE_ROOT),
        "official_upstream_commit": "bac9e7b59f4fea0eaaa936f24ef60f342f349829",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        if RUN_ROOT.exists():
            raise RuntimeError(f"refusing to reuse boundary directory: {RUN_ROOT}")
        if not SOURCE_OUTPUT.exists():
            raise FileNotFoundError(f"completed replay output is missing: {SOURCE_OUTPUT}")
        for name in STAGE2_FILES:
            if not (SOURCE_OUTPUT / name).exists():
                raise FileNotFoundError(f"completed replay artifact is missing: {SOURCE_OUTPUT / name}")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        OUTPUT.mkdir()
        copied = {}
        for name in STAGE2_FILES:
            source = SOURCE_OUTPUT / name
            target = OUTPUT / name
            shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError(f"byte-copy changed replay artifact: {name}")
            copied[name] = {"source": str(source), "path": str(target), "sha256": sha256(target)}
        result["copied_replay_stage2"] = copied

        set_seed(42)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(UPSTREAM))
        sys.path.insert(0, str(INSTANTMESH))
        os.chdir(UPSTREAM)
        cfg = load_cfg(RUN_ROOT)
        import reg_xyz

        hook = FusionBoundaryHook(RUN_ROOT / "pre_fusion")
        original_remove_close_points = reg_xyz.remove_close_points

        def hooked_remove_close_points(source_pcd, target_pcd, *args, **kwargs):
            return hook(original_remove_close_points, source_pcd, target_pcd, *args, **kwargs)

        reg_xyz.remove_close_points = hooked_remove_close_points
        torch.cuda.reset_peak_memory_stats()
        os.chdir(RUN_ROOT)
        reg_xyz.reg(cfg, FLAG, cd_inv_weight=0.5, diff_init=True, reg_fine_xyz=True)
        result["registration_and_fusion_memory"] = cuda_memory()
        if not hook.called:
            raise RuntimeError("corrected fusion boundary hook was not called")

        fused_path = OUTPUT / f"{FLAG}_fused.ply"
        final_transform_path = RUN_ROOT / "final_transform.npy"
        if not fused_path.exists() or not final_transform_path.exists():
            raise RuntimeError("boundary registration did not produce required outputs")

        os.chdir(UPSTREAM)
        from run_method_valid_baseline_07136 import compute_author_metrics

        baseline_result = json.loads((BASELINE_ROOT / "baseline_result.json").read_text(encoding="utf-8"))
        baseline_metrics = baseline_result["metrics"]
        replay_metrics = compute_author_metrics(
            fused_path, UPSTREAM / "data" / "GT" / f"{FLAG}.ply"
        )
        metric_tolerance = 1e-9
        metrics = {
            "baseline": baseline_metrics,
            "replay": replay_metrics,
            "tolerance": metric_tolerance,
            "cd_abs_diff": abs(float(baseline_metrics["cd"]) - float(replay_metrics["cd"])),
            "emd_abs_diff": abs(float(baseline_metrics["emd"]) - float(replay_metrics["emd"])),
        }
        metrics["strict_pass"] = bool(
            metrics["cd_abs_diff"] <= metric_tolerance
            and metrics["emd_abs_diff"] <= metric_tolerance
        )
        file_comparison = compare_files(BASELINE_OUTPUT, OUTPUT, STAGE2_FILES)
        baseline_transform = BASELINE_ROOT / "final_transform.npy"
        file_comparison["final_transform.npy"] = {
            "baseline_path": str(baseline_transform),
            "replay_path": str(final_transform_path),
            "baseline_sha256": sha256(baseline_transform),
            "replay_sha256": sha256(final_transform_path),
            "exact_match": sha256(baseline_transform) == sha256(final_transform_path),
        }
        fused_comparison = compare_pointclouds(
            BASELINE_OUTPUT / f"{FLAG}_fused.ply", fused_path
        )
        result["outputs"] = {
            "stage2_files": copied,
            "fused": pcd_meta(fused_path),
            "final_transform": {
                "path": str(final_transform_path),
                "sha256": sha256(final_transform_path),
                "matrix": np.load(final_transform_path).tolist(),
            },
            "pre_fusion": hook.capture,
        }
        result["comparisons"] = {
            "file_hashes": file_comparison,
            "fused_pointcloud": fused_comparison,
            "metrics": metrics,
        }
        result["equivalence_pass"] = bool(
            all(item["exact_match"] for item in file_comparison.values())
            and fused_comparison["strict_geometry_pass"]
            and metrics["strict_pass"]
        )
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" and result["equivalence_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
