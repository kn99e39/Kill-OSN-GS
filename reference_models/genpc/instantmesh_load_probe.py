"""Load-only probe for the GenPC InstantMesh integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
INSTANTMESH = ROOT / "dependencies" / "InstantMesh"


def main() -> None:
    result: dict[str, object] = {
        "device": "cuda",
        "source": "GenPC tools/instantmesh.py + TencentARC/InstantMesh src",
        "models": {
            "multiview_pipeline": "sudo-ai/zero123plus-v1.2",
            "custom_unet": "TencentARC/InstantMesh:diffusion_pytorch_model.bin",
            "reconstruction": "TencentARC/InstantMesh:instant_mesh_base.ckpt",
            "encoder": "facebook/dino-vitb16",
        },
        "load_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        os.chdir(INSTANTMESH)
        sys.path.insert(0, str(INSTANTMESH))
        sys.path.insert(0, str(UPSTREAM))
        torch.cuda.reset_peak_memory_stats()
        from instantmesh_api_compat import install_diffusers_compat, install_mesh_util_compat

        result["compatibility_shim"] = install_mesh_util_compat()
        result["diffusers_compatibility_shim"] = install_diffusers_compat()
        import tools.instantmesh  # noqa: F401 - import performs the source load gate

        result["status"] = "loaded"
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except Exception as exc:  # noqa: BLE001 - load gate records the exact failure
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
