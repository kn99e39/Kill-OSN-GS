"""Replay the frozen 07136 downstream path with a read-only fusion boundary hook.

The preserved depth/ControlNet output is copied byte-for-byte from the completed
baseline.  This replay therefore starts at RMBG and runs the published
RMBG -> InstantMesh -> registration/fusion path in a separate output directory.
The only instrumentation is a hook immediately before reg_xyz.remove_close_points
that writes O_aligned, G_aligned, and the transform lineage.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml
from munch import Munch


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
INSTANTMESH = ROOT / "dependencies" / "InstantMesh"
MANIFEST_PATH = ROOT / "METHOD_VALID_BASELINE_07136_20260912.json"
BASELINE_ROOT = ROOT / "runs" / "baseline_07136"
BASELINE_OUTPUT = BASELINE_ROOT / "07136"
RUN_ROOT = ROOT / "runs" / "instrumented_equivalence_07136_offline"
OUTPUT = RUN_ROOT / "07136"
RESULT_PATH = RUN_ROOT / "instrumented_equivalence_result.json"
FLAG = "07136"
CUDA_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
STAGE1_FILES = [
    "img.png",
    "depth.png",
    "raw_depth.png",
    "mask.png",
    "point_uv.npy",
    "camera.pth",
    "viewpoint.npy",
]
STAGE2_FILES = [
    "img_sam.png",
    "07136_zero123++.png",
    "07136_instantmesh.glb",
    "color_point.ply",
    "07136_fused.ply",
]


def configure_cuda_runtime() -> None:
    os.environ["CUDA_HOME"] = str(CUDA_HOME)
    os.environ["CUDA_PATH"] = str(CUDA_HOME)
    os.environ["CUDA_PATH_V12_8"] = str(CUDA_HOME)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["HF_MODULES_CACHE"] = str(RUN_ROOT / "hf_modules_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PATH"] = os.pathsep.join(
        [str(CUDA_HOME / "bin"), str(CUDA_HOME / "lib" / "x64"), os.environ["PATH"]]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from pytorch_lightning import seed_everything

        seed_everything(seed, workers=True)
    except Exception:
        pass


def cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    torch.cuda.synchronize()
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def load_cfg(output_path: Path) -> Munch:
    config = yaml.safe_load(
        (UPSTREAM / "configs" / "config.yaml").read_text(encoding="utf-8")
    )
    config.update(
        {
            "output_path": str(output_path),
            "view_num": 2048,
            "cam_res": 256,
            "distance": 1.6,
            "fovy": 49.1,
            "point_size": 1,
            "mask_pixel_rate": 3,
            "res": 256,
            "generate_res": 1024,
            "inpainter": "cv2",
            "control_model": "controlnet",
            "generative_model": "instantmesh",
            "rembg_model": "RMBG",
            "dataset": "redwood",
            "device": "cuda",
        }
    )
    return Munch.fromDict(config)


def pcd_meta(path: Path) -> dict[str, Any]:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "point_count": int(len(points)),
        "has_colors": bool(cloud.has_colors()),
        "xyz_min": points.min(axis=0).tolist() if len(points) else [],
        "xyz_max": points.max(axis=0).tolist() if len(points) else [],
        "color_min": colors.min(axis=0).tolist() if len(colors) else [],
        "color_max": colors.max(axis=0).tolist() if len(colors) else [],
    }


def install_instantmesh_local_weights_compat() -> str:
    """Resolve the already cached official InstantMesh weights without a HEAD request."""
    import huggingface_hub

    repo_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--TencentARC--InstantMesh"
    ref_path = repo_cache / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"InstantMesh cache ref is missing: {ref_path}")
    snapshot = repo_cache / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
    required = [snapshot / "diffusion_pytorch_model.bin", snapshot / "instant_mesh_base.ckpt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("cached InstantMesh weights are missing: " + ", ".join(missing))
    original = huggingface_hub.hf_hub_download

    def local_hf_hub_download(repo_id, filename, *args, **kwargs):
        if repo_id == "TencentARC/InstantMesh":
            local_path = snapshot / filename
            if local_path.exists():
                return str(local_path)
        return original(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = local_hf_hub_download
    return f"local_cached_snapshot:{snapshot}"


def validate_inputs(manifest: dict[str, Any]) -> None:
    if manifest["run_selection_locked_before_execution"] is not True:
        raise RuntimeError("run selection is not locked")
    if manifest["experiment1_run"] is not False:
        raise RuntimeError("Experiment 1 must not run in this replay")
    if manifest["sample"] != FLAG or manifest["seed"] != 42:
        raise RuntimeError("manifest sample/seed mismatch")
    if json.loads((BASELINE_ROOT / "baseline_result.json").read_text(encoding="utf-8"))["status"] != "completed":
        raise RuntimeError("completed baseline result is required")
    input_path = UPSTREAM / "data" / f"{FLAG}.ply"
    gt_path = UPSTREAM / "data" / "GT" / f"{FLAG}.ply"
    if sha256(input_path) != manifest["input"]["sha256"]:
        raise RuntimeError("input hash does not match the locked manifest")
    if sha256(gt_path) != manifest["ground_truth"]["sha256"]:
        raise RuntimeError("ground-truth hash does not match the locked manifest")
    required_baseline = [BASELINE_OUTPUT / name for name in STAGE1_FILES]
    required_baseline += [BASELINE_OUTPUT / name for name in STAGE2_FILES]
    required_baseline += [BASELINE_ROOT / "final_transform.npy"]
    missing = [str(path) for path in required_baseline if not path.exists()]
    if missing:
        raise FileNotFoundError("completed baseline artifacts are missing: " + ", ".join(missing))


def copy_preserved_stage1() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in STAGE1_FILES:
        source = BASELINE_OUTPUT / name
        target = OUTPUT / name
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"byte-copy changed preserved artifact: {name}")
        copied[name] = {"source": str(source), "path": str(target), "sha256": target_hash}
    return copied


class FusionBoundaryHook:
    """Capture reg() locals immediately before the original point filter call."""

    def __init__(self, pre_fusion_dir: Path):
        self.pre_fusion_dir = pre_fusion_dir
        self.called = False
        self.capture: dict[str, Any] = {}

    @staticmethod
    def _matrix(value: Any) -> list[list[float]] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (4, 4):
            return None
        return array.tolist()

    def __call__(self, original, source_pcd, target_pcd, *args, **kwargs):
        if self.called:
            raise RuntimeError("remove_close_points hook was called more than once")
        self.called = True
        import open3d as o3d

        self.pre_fusion_dir.mkdir(parents=True, exist_ok=False)
        source_path = self.pre_fusion_dir / "O_aligned.ply"
        target_path = self.pre_fusion_dir / "G_aligned.ply"
        if not o3d.io.write_point_cloud(str(source_path), source_pcd):
            raise RuntimeError("failed to write O_aligned.ply")
        if not o3d.io.write_point_cloud(str(target_path), target_pcd):
            raise RuntimeError("failed to write G_aligned.ply")

        frame = inspect.currentframe()
        caller_locals: dict[str, Any] = {}
        try:
            caller = frame.f_back if frame is not None else None
            for _ in range(4):
                if caller is None:
                    break
                if caller.f_code.co_name == "reg":
                    caller_locals = caller.f_locals
                    break
                caller = caller.f_back
            matrices = {
                "diff_transform": self._matrix(caller_locals.get("diff_transform")),
                "inverse_diff_transform_at_boundary": self._matrix(caller_locals.get("inv")),
                "coarse_transformation": self._matrix(caller_locals.get("coarse_transformation")),
                "best_scales_transformation": self._matrix(caller_locals.get("best_scales_transformation")),
                "best_transformation_xyz": self._matrix(caller_locals.get("best_transformation_xyz")),
            }
        finally:
            del frame

        if matrices["diff_transform"] is not None:
            matrices["inverse_diff_transform"] = np.linalg.inv(
                np.asarray(matrices["diff_transform"], dtype=np.float64)
            ).tolist()
        if matrices["coarse_transformation"] is not None:
            matrices["inverse_coarse_transformation"] = np.linalg.inv(
                np.asarray(matrices["coarse_transformation"], dtype=np.float64)
            ).tolist()
        if matrices["best_scales_transformation"] is not None:
            matrices["inverse_best_scales_transformation"] = np.linalg.inv(
                np.asarray(matrices["best_scales_transformation"], dtype=np.float64)
            ).tolist()
        if matrices["best_transformation_xyz"] is not None:
            matrices["inverse_best_transformation_xyz"] = np.linalg.inv(
                np.asarray(matrices["best_transformation_xyz"], dtype=np.float64)
            ).tolist()

        lineage = {
            "boundary": "immediately before official reg_xyz.remove_close_points(source_pcd, target_pcd, distance_threshold=0.0001)",
            "hook_behavior": "read-only capture; original remove_close_points called with unchanged objects and arguments",
            "source_label": "O_aligned",
            "target_label": "G_aligned",
            "source_semantics": [
                "upstream color_point.ply point cloud",
                "remove_noise_from_point_cloud",
                "diff_transform applied",
                "coarse_transformation applied for ICP search",
                "inverse coarse transformation applied",
                "inverse diff transformation applied",
                "O_aligned captured",
            ],
            "target_semantics": [
                "InstantMesh GLB converted by glb2point",
                "normalize_numpy(range=0.5)",
                "InstantMesh x-axis +90 degree rotation",
                "InstantMesh y-axis +90 degree rotation",
                "inverse best_scales_transformation applied",
                "inverse best_transformation_xyz applied",
                "inverse coarse transformation applied",
                "inverse diff transformation applied",
                "G_aligned captured",
            ],
            "matrices": matrices,
            "distance_threshold": float(kwargs.get("distance_threshold", args[0] if args else 0.0001)),
            "source_pcd": pcd_meta(source_path),
            "target_pcd": pcd_meta(target_path),
        }
        (self.pre_fusion_dir / "transform_lineage.json").write_text(
            json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.capture = lineage
        return original(source_pcd, target_pcd, *args, **kwargs)


def compare_pointclouds(baseline_path: Path, replay_path: Path) -> dict[str, Any]:
    import open3d as o3d

    baseline = o3d.io.read_point_cloud(str(baseline_path))
    replay = o3d.io.read_point_cloud(str(replay_path))
    baseline_xyz = np.asarray(baseline.points, dtype=np.float64)
    replay_xyz = np.asarray(replay.points, dtype=np.float64)
    baseline_rgb = np.asarray(baseline.colors, dtype=np.float64)
    replay_rgb = np.asarray(replay.colors, dtype=np.float64)
    result: dict[str, Any] = {
        "baseline": pcd_meta(baseline_path),
        "replay": pcd_meta(replay_path),
        "same_shape": bool(baseline_xyz.shape == replay_xyz.shape and baseline_rgb.shape == replay_rgb.shape),
        "geometry_tolerance": 1e-8,
        "color_tolerance": 1e-12,
    }
    if not result["same_shape"] or not len(baseline_xyz):
        result.update({"max_abs_xyz_diff": None, "max_abs_color_diff": None, "strict_geometry_pass": False})
        return result
    baseline_order = np.lexsort((baseline_xyz[:, 2], baseline_xyz[:, 1], baseline_xyz[:, 0]))
    replay_order = np.lexsort((replay_xyz[:, 2], replay_xyz[:, 1], replay_xyz[:, 0]))
    xyz_diff = np.abs(baseline_xyz[baseline_order] - replay_xyz[replay_order])
    color_diff = np.abs(baseline_rgb[baseline_order] - replay_rgb[replay_order])
    max_xyz = float(xyz_diff.max())
    max_color = float(color_diff.max()) if color_diff.size else 0.0
    result.update(
        {
            "max_abs_xyz_diff": max_xyz,
            "max_abs_color_diff": max_color,
            "strict_geometry_pass": bool(max_xyz <= result["geometry_tolerance"] and max_color <= result["color_tolerance"]),
        }
    )
    return result


def compare_files(baseline_dir: Path, replay_dir: Path, names: list[str]) -> dict[str, Any]:
    comparison = {}
    for name in names:
        baseline = baseline_dir / name
        replay = replay_dir / name
        baseline_hash = sha256(baseline)
        replay_hash = sha256(replay)
        comparison[name] = {
            "baseline_path": str(baseline),
            "replay_path": str(replay),
            "baseline_sha256": baseline_hash,
            "replay_sha256": replay_hash,
            "exact_match": baseline_hash == replay_hash,
        }
    return comparison


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    configure_cuda_runtime()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    result: dict[str, Any] = {
        "run_id": "instrumented_equivalence_07136_20260912",
        "status": "started",
        "equivalence_pass": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(MANIFEST_PATH),
        "official_upstream_commit": "bac9e7b59f4fea0eaaa936f24ef60f342f349829",
        "baseline_root": str(BASELINE_ROOT),
        "replay_root": str(RUN_ROOT),
        "rmbg_revision": RMBG_REVISION,
        "stages": {},
    }
    try:
        validate_inputs(manifest)
        if RUN_ROOT.exists():
            if any(RUN_ROOT.iterdir()):
                raise RuntimeError(f"refusing to reuse non-empty replay directory: {RUN_ROOT}")
            raise RuntimeError(f"refusing to reuse existing replay directory: {RUN_ROOT}")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        result["preserved_stage1"] = copy_preserved_stage1()

        set_seed(manifest["seed"])
        cfg = load_cfg(RUN_ROOT)
        device = torch.device(cfg.device)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(UPSTREAM))
        sys.path.insert(0, str(INSTANTMESH))

        os.chdir(UPSTREAM)
        from instantmesh_api_compat import install_diffusers_compat, install_mesh_util_compat
        from utils.dataUtils import load_xyz

        xyz_np, _rgb_np = load_xyz(str(UPSTREAM / "data" / f"{FLAG}.ply"))
        xyz = torch.from_numpy(xyz_np).to(device)

        os.chdir(INSTANTMESH)
        result["compatibility_shim"] = install_mesh_util_compat()
        result["diffusers_compatibility_shim"] = install_diffusers_compat()
        result["instantmesh_local_weights_compatibility_shim"] = install_instantmesh_local_weights_compat()
        from run_method_valid_baseline_07136 import install_rmbg_revision_compat

        result["rmbg_revision_compatibility_shim"] = install_rmbg_revision_compat()
        from ScaleAdapter import ScaleAdapter
        import reg_xyz

        hook = FusionBoundaryHook(RUN_ROOT / "pre_fusion")
        original_remove_close_points = reg_xyz.remove_close_points

        def hooked_remove_close_points(source_pcd, target_pcd, *args, **kwargs):
            return hook(original_remove_close_points, source_pcd, target_pcd, *args, **kwargs)

        reg_xyz.remove_close_points = hooked_remove_close_points

        torch.cuda.reset_peak_memory_stats()
        scale_adapter = ScaleAdapter(cfg)
        os.chdir(RUN_ROOT)
        scale_adapter.scaleAdapter(xyz, FLAG)
        result["stages"]["rmbg_and_instantmesh"] = cuda_memory()
        result["stages"]["rmbg_and_instantmesh"]["status"] = "completed"

        torch.cuda.reset_peak_memory_stats()
        scale_adapter.scaleReg(FLAG)
        result["stages"]["registration_and_fusion"] = cuda_memory()
        result["stages"]["registration_and_fusion"]["status"] = "completed"
        if not hook.called:
            raise RuntimeError("fusion boundary hook was not called")

        fused_path = OUTPUT / f"{FLAG}_fused.ply"
        final_transform_path = RUN_ROOT / "final_transform.npy"
        if not fused_path.exists() or not final_transform_path.exists():
            raise RuntimeError("replay completed without fused point cloud or final transform")

        os.chdir(UPSTREAM)
        from run_method_valid_baseline_07136 import compute_author_metrics

        replay_metrics = compute_author_metrics(fused_path, UPSTREAM / "data" / "GT" / f"{FLAG}.ply")
        baseline_result = json.loads((BASELINE_ROOT / "baseline_result.json").read_text(encoding="utf-8"))
        baseline_metrics = baseline_result["metrics"]
        metric_tolerance = 1e-9
        metric_comparison = {
            "baseline": baseline_metrics,
            "replay": replay_metrics,
            "tolerance": metric_tolerance,
            "cd_abs_diff": abs(float(baseline_metrics["cd"]) - float(replay_metrics["cd"])),
            "emd_abs_diff": abs(float(baseline_metrics["emd"]) - float(replay_metrics["emd"])),
        }
        metric_comparison["strict_pass"] = bool(
            metric_comparison["cd_abs_diff"] <= metric_tolerance
            and metric_comparison["emd_abs_diff"] <= metric_tolerance
        )

        file_comparison = compare_files(BASELINE_OUTPUT, OUTPUT, STAGE2_FILES)
        file_comparison["final_transform.npy"] = {
            "baseline_path": str(BASELINE_ROOT / "final_transform.npy"),
            "replay_path": str(final_transform_path),
            "baseline_sha256": sha256(BASELINE_ROOT / "final_transform.npy"),
            "replay_sha256": sha256(final_transform_path),
            "exact_match": sha256(BASELINE_ROOT / "final_transform.npy") == sha256(final_transform_path),
        }
        fused_comparison = compare_pointclouds(
            BASELINE_OUTPUT / f"{FLAG}_fused.ply", fused_path
        )
        result["outputs"] = {
            "stage2_files": {
                name: {"path": str(OUTPUT / name), "sha256": sha256(OUTPUT / name)}
                for name in STAGE2_FILES
            },
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
            "metrics": metric_comparison,
        }
        result["equivalence_pass"] = bool(
            all(item["exact_match"] for item in file_comparison.values())
            and fused_comparison["strict_geometry_pass"]
            and metric_comparison["strict_pass"]
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
