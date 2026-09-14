"""One-time frozen materialisation for the lexicographically locked real audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
UPSTREAM = ROOT / "upstream"
INSTANTMESH = ROOT / "dependencies" / "InstantMesh"
AUDIT_SET = ROOT / "REAL_AMBIGUITY_AUDIT_SET_20260914.json"
RUN_ROOT = ROOT / "runs" / "real_ambiguity_audit_20260914"
UPSTREAM_COMMIT = "bac9e7b59f4fea0eaaa936f24ef60f342f349829"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pcd_exact(pcd: Any, path: Path) -> dict[str, Any]:
    points = np.asarray(pcd.points).copy()
    colors = np.asarray(pcd.colors).copy()
    if points.dtype != np.float64 or colors.shape != points.shape:
        raise RuntimeError("boundary point cloud did not expose float64 points/colors")
    np.savez_compressed(path, points=points, colors=colors)
    return {"path": str(path), "sha256": sha256(path), "point_count": int(len(points))}


class BoundaryCapture:
    """Read-only hook at the exact native remove_close_points boundary."""

    def __init__(self, root: Path):
        self.root = root
        self.capture: dict[str, Any] | None = None

    def __call__(self, original, source_pcd, target_pcd, *args, **kwargs):
        if self.capture is not None:
            raise RuntimeError("fusion boundary invoked more than once")
        self.root.mkdir(parents=True, exist_ok=False)
        state_path = self.root / "numpy_rng_before_fusion.pkl"
        with state_path.open("wb") as handle:
            pickle.dump(np.random.get_state(), handle, protocol=4)
        self.capture = {
            "boundary": "immediately before upstream reg_xyz.remove_close_points",
            "distance_threshold": float(kwargs.get("distance_threshold", args[0] if args else 0.0001)),
            "O_aligned_exact": pcd_exact(source_pcd, self.root / "O_aligned_exact.npz"),
            "G_aligned_exact": pcd_exact(target_pcd, self.root / "G_aligned_exact.npz"),
            "numpy_rng_before_fusion": {"path": str(state_path), "sha256": sha256(state_path)},
        }
        (self.root / "boundary_manifest.json").write_text(json.dumps(self.capture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return original(source_pcd, target_pcd, *args, **kwargs)


def audit() -> dict[str, Any]:
    value = json.loads(AUDIT_SET.read_text(encoding="utf-8"))
    if value["selected_audit_sample_ids"] != sorted(value["selected_audit_sample_ids"]):
        raise RuntimeError("frozen audit list is not lexicographically sorted")
    return value


def expected_sample(argument: str | None) -> str:
    value = audit()
    pending = [sample for sample in value["new_single_run_sample_ids"] if not (RUN_ROOT / sample / "result.json").exists()]
    if not pending:
        raise RuntimeError("all new audit samples already have one frozen result")
    if argument is not None and argument != pending[0]:
        raise RuntimeError(f"next locked sample is {pending[0]}, not {argument}; do not reorder the audit")
    return pending[0]


def input_record(sample: str) -> dict[str, str]:
    for item in audit()["all_paired_samples"]:
        if item["sample"] == sample:
            return item
    raise RuntimeError(f"sample {sample} is outside the frozen audit list")


def run(sample: str) -> dict[str, Any]:
    if (RUN_ROOT / sample).exists():
        raise RuntimeError(f"refusing to reuse sample run directory: {RUN_ROOT / sample}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(UPSTREAM))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    from run_method_valid_baseline_07136 import configure_cuda_runtime, install_rmbg_revision_compat, load_cfg, set_seed

    configure_cuda_runtime()
    # The verified method-valid environment is fully cached.  Keep it
    # offline so diffusers resolves the frozen scheduler cache rather than
    # turning a one-time audit run into a network retry loop.
    os.environ["HF_HUB_OFFLINE"] = "1"
    record = input_record(sample)
    input_path = UPSTREAM / "data" / f"{sample}.ply"
    gt_path = UPSTREAM / "data" / "GT" / f"{sample}.ply"
    if sha256(input_path) != record["input_sha256"] or sha256(gt_path) != record["gt_sha256"]:
        raise RuntimeError("frozen audit input hashes do not match local files")
    run_dir = RUN_ROOT / sample
    output = run_dir / sample
    run_dir.mkdir(parents=True, exist_ok=False)
    output.mkdir()
    os.environ["HF_MODULES_CACHE"] = str(run_dir / "hf_modules_cache")
    result: dict[str, Any] = {"sample": sample, "status": "started", "seed": 42, "input": record, "official_upstream_commit": UPSTREAM_COMMIT, "stages": {}}
    try:
        set_seed(42)
        cfg = load_cfg(run_dir)
        device = torch.device(cfg.device)
        os.chdir(UPSTREAM)
        from instantmesh_api_compat import install_diffusers_compat, install_mesh_util_compat
        from utils.dataUtils import load_xyz
        from diffusers.utils import load_image
        from DepthPrompting import DepthPrompting

        xyz_np, rgb_np = load_xyz(str(input_path))
        xyz = torch.from_numpy(xyz_np).to(device)
        rgb = torch.from_numpy(rgb_np).to(device)
        torch.cuda.reset_peak_memory_stats()
        depth_prompting = DepthPrompting(cfg)
        depth_prompting.getDepth(xyz=xyz, flag=sample, rgb=rgb)
        depth = load_image(str(output / "depth.png"))
        set_seed(42)
        image = depth_prompting.depth2Image.generate(depth, sample, size=cfg.generate_res, controlnet_conditioning_scale=0.99, num_inference_steps=30, save_path=str(output / "img.png"))
        if image is None or not (output / "img.png").exists():
            raise RuntimeError("author ControlNet path did not produce img.png")
        result["stages"]["depth_prompting_and_controlnet"] = "completed"
        del depth_prompting
        torch.cuda.empty_cache()

        os.chdir(INSTANTMESH)
        sys.path.insert(0, str(INSTANTMESH))
        result["compatibility_shim"] = install_mesh_util_compat()
        result["diffusers_compatibility_shim"] = install_diffusers_compat()
        result["rmbg_revision_compatibility_shim"] = install_rmbg_revision_compat()
        from ScaleAdapter import ScaleAdapter
        import reg_xyz

        adapter = ScaleAdapter(cfg)
        os.chdir(run_dir)
        adapter.scaleAdapter(xyz, sample)
        result["stages"]["scale_adapter"] = "completed"
        capture = BoundaryCapture(run_dir / "pre_fusion")
        original = reg_xyz.remove_close_points
        reg_xyz.remove_close_points = lambda source, target, *args, **kwargs: capture(original, source, target, *args, **kwargs)
        try:
            adapter.scaleReg(sample)
        finally:
            reg_xyz.remove_close_points = original
        if capture.capture is None:
            raise RuntimeError("registration did not reach the frozen fusion boundary")
        result["stages"]["registration_and_fusion"] = "completed"
        result["pre_fusion"] = capture.capture
        fused = output / f"{sample}_fused.ply"
        if not fused.exists():
            raise RuntimeError("registration completed without a fused result")
        result["fused"] = {"path": str(fused), "sha256": sha256(fused)}
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample")
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "frozen_audit_set.json").write_text(AUDIT_SET.read_text(encoding="utf-8"), encoding="utf-8")
    sample = expected_sample(args.sample)
    result = run(sample)
    print(json.dumps({"sample": sample, "status": result["status"], "error": result.get("error")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
