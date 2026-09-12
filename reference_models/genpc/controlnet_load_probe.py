"""Load-only probe for the GenPC author-provided ControlNet path."""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream"))


def main() -> None:
    result: dict[str, object] = {
        "device": "cuda",
        "models": {
            "base": "stabilityai/stable-diffusion-xl-base-1.0",
            "controlnet": "xinsir/controlnet-depth-sdxl-1.0",
            "vae": "madebyollin/sdxl-vae-fp16-fix",
        },
        "load_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        from tools.controlnet_depth import ControlNet_Depth

        model = ControlNet_Depth("cuda")
        result["status"] = "loaded"
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - load gate records the exact failure
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if torch.cuda.is_available():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
