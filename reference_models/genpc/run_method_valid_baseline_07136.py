"""Run exactly one predeclared, method-valid GenPC baseline for sample 07136.

This runner stays outside the pristine upstream checkout. It preserves the
published GenPC modules and fixes only the orchestration defect that otherwise
leaves the downstream ScaleAdapter without img.png.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
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
RUN_ROOT = ROOT / "runs" / "baseline_07136"
FLAG = "07136"
CUDA_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
RMBG_MODEL_DIR = ROOT / "model_cache" / "RMBG-2.0" / RMBG_REVISION


def configure_cuda_runtime() -> None:
    os.environ["CUDA_HOME"] = str(CUDA_HOME)
    os.environ["CUDA_PATH"] = str(CUDA_HOME)
    os.environ["CUDA_PATH_V12_8"] = str(CUDA_HOME)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["PATH"] = os.pathsep.join(
        [str(CUDA_HOME / "bin"), str(CUDA_HOME / "lib" / "x64"), os.environ["PATH"]]
    )


def install_rmbg_revision_compat() -> str:
    """Keep the author RMBG call on the already verified exact local revision."""
    from transformers import AutoModelForImageSegmentation

    if not RMBG_MODEL_DIR.exists():
        raise FileNotFoundError(f"verified RMBG model directory is missing: {RMBG_MODEL_DIR}")
    original_from_pretrained = AutoModelForImageSegmentation.from_pretrained

    def exact_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        if pretrained_model_name_or_path == "briaai/RMBG-2.0":
            pretrained_model_name_or_path = str(RMBG_MODEL_DIR)
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    AutoModelForImageSegmentation.from_pretrained = exact_from_pretrained
    return f"local_exact_revision:{RMBG_REVISION}"


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


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["run_selection_locked_before_execution"] is not True:
        raise RuntimeError("run selection is not locked")
    if manifest["experiment1_run"] is not False:
        raise RuntimeError("Experiment 1 must not run in this baseline")
    if manifest["sample"] != FLAG or manifest["seed"] != 42:
        raise RuntimeError("manifest sample/seed mismatch")
    input_path = UPSTREAM / "data" / f"{FLAG}.ply"
    gt_path = UPSTREAM / "data" / "GT" / f"{FLAG}.ply"
    if sha256(input_path) != manifest["input"]["sha256"]:
        raise RuntimeError("input hash does not match the locked manifest")
    if sha256(gt_path) != manifest["ground_truth"]["sha256"]:
        raise RuntimeError("ground-truth hash does not match the locked manifest")


def compute_author_metrics(pred_path: Path, gt_path: Path) -> dict[str, float]:
    import open3d as o3d
    from fpsample import fps_sampling
    from utils.loss_util import Completionloss

    gt = o3d.io.read_point_cloud(str(gt_path))
    pred = o3d.io.read_point_cloud(str(pred_path))
    gt_points = np.asarray(gt.points).astype(np.float32)
    pred_points = np.asarray(pred.points).astype(np.float32)
    gt_indices = fps_sampling(gt_points, 16384)
    pred_indices = fps_sampling(pred_points, 16384)
    gt_tensor = torch.from_numpy(gt_points[gt_indices]).unsqueeze(0).cuda()
    pred_tensor = torch.from_numpy(pred_points[pred_indices]).unsqueeze(0).cuda()
    completion_cd = Completionloss(loss_func="cd_l1")
    completion_emd = Completionloss(loss_func="emd")
    cd = float(completion_cd.get_loss(gen=pred_tensor, gt=gt_tensor).item())
    emd = float(completion_emd.get_loss(gen=pred_tensor, gt=gt_tensor).item())
    return {"cd": cd, "emd": emd}


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    configure_cuda_runtime()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    output_dir = RUN_ROOT / FLAG
    result_path = RUN_ROOT / "baseline_result.json"
    resume_depth_stage = False
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
        if not result_path.exists() or not output_dir.exists():
            raise RuntimeError(f"refusing to reuse non-empty run directory: {RUN_ROOT}")
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        prior_stage = prior.get("stages", {}).get("depth_prompting_and_controlnet", {})
        resume_depth_stage = (
            prior.get("status") == "failed"
            and prior_stage.get("status") == "completed"
            and (output_dir / "img.png").exists()
            and not (output_dir / f"{FLAG}_fused.ply").exists()
        )
        if not resume_depth_stage:
            raise RuntimeError(f"refusing to reuse non-empty run directory: {RUN_ROOT}")
        result = prior
        result["resumed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result["resume_reason"] = "depth/controlnet completed; prior failure was before ScaleAdapter import"
    else:
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir()
        result = {
            "run_id": manifest["run_id"],
            "status": "started",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest": str(MANIFEST_PATH),
            "official_upstream_commit": "bac9e7b59f4fea0eaaa936f24ef60f342f349829",
            "input_sha256": sha256(UPSTREAM / "data" / f"{FLAG}.ply"),
            "ground_truth_sha256": sha256(UPSTREAM / "data" / "GT" / f"{FLAG}.ply"),
            "stages": {},
        }

    try:
        set_seed(manifest["seed"])
        cfg = load_cfg(RUN_ROOT)
        device = torch.device(cfg.device)
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(UPSTREAM))

        os.chdir(UPSTREAM)
        from instantmesh_api_compat import (
            install_diffusers_compat,
            install_mesh_util_compat,
        )
        from utils.dataUtils import load_xyz

        xyz_np, rgb_np = load_xyz(str(UPSTREAM / "data" / f"{FLAG}.ply"))
        xyz = torch.from_numpy(xyz_np).to(device)
        rgb = torch.from_numpy(rgb_np).to(device)
        if not resume_depth_stage:
            from diffusers.utils import load_image
            from DepthPrompting import DepthPrompting

            torch.cuda.reset_peak_memory_stats()
            dp = DepthPrompting(cfg)
            dp.getDepth(xyz=xyz, flag=FLAG, rgb=rgb)
            depth = load_image(str(output_dir / "depth.png"))
            set_seed(manifest["seed"])
            image = dp.depth2Image.generate(
                depth,
                FLAG,
                size=cfg.generate_res,
                controlnet_conditioning_scale=0.99,
                num_inference_steps=30,
                save_path=str(output_dir / "img.png"),
            )
            if image is None or not (output_dir / "img.png").exists():
                raise RuntimeError("author ControlNet path did not produce img.png")
            result["stages"]["depth_prompting_and_controlnet"] = cuda_memory()
            result["stages"]["depth_prompting_and_controlnet"]["status"] = "completed"
            del dp
            torch.cuda.empty_cache()

        os.chdir(INSTANTMESH)
        sys.path.insert(0, str(INSTANTMESH))
        result["compatibility_shim"] = install_mesh_util_compat()
        result["diffusers_compatibility_shim"] = install_diffusers_compat()
        result["rmbg_revision_compatibility_shim"] = install_rmbg_revision_compat()
        from ScaleAdapter import ScaleAdapter

        torch.cuda.reset_peak_memory_stats()
        scale_adapter = ScaleAdapter(cfg)
        os.chdir(RUN_ROOT)
        scale_adapter.scaleAdapter(xyz, FLAG)
        result["stages"]["scale_adapter"] = cuda_memory()
        result["stages"]["scale_adapter"]["status"] = "completed"

        torch.cuda.reset_peak_memory_stats()
        scale_adapter.scaleReg(FLAG)
        result["stages"]["registration_and_fusion"] = cuda_memory()
        result["stages"]["registration_and_fusion"]["status"] = "completed"

        fused_path = output_dir / f"{FLAG}_fused.ply"
        if not fused_path.exists():
            raise RuntimeError("registration completed without fused point cloud")
        result["outputs"] = {
            "fused_path": str(fused_path),
            "fused_sha256": sha256(fused_path),
            "final_transform_path": str(RUN_ROOT / "final_transform.npy"),
        }

        os.chdir(UPSTREAM)
        try:
            result["metrics"] = compute_author_metrics(
                fused_path, UPSTREAM / "data" / "GT" / f"{FLAG}.ply"
            )
            result["metrics_status"] = "author_metric_reproduced"
        except Exception as exc:  # metrics do not trigger a second inference
            result["metrics_status"] = "unavailable_after_single_run"
            result["metrics_error"] = f"{type(exc).__name__}: {exc}"

        result.pop("error", None)
        result["status"] = "completed"
    except Exception as exc:  # preserve the one-run failure evidence
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
