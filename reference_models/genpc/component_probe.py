"""Read-only dependency and source-path probe for the GenPC reference setup."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
DEPENDENCIES = ROOT / "dependencies"


def import_status(name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - probe records the exact failure
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}


def main() -> None:
    sys.path.insert(0, str(UPSTREAM))
    result = {
        "imports": {
            name: import_status(name)
            for name in (
                "diffusers",
                "transformers",
                "open3d",
                "nvdiffrast",
                "kaolin",
                "tools.controlnet_depth",
            )
        },
        "source_paths": {
            "genpc_ddnm_module": (UPSTREAM / "models" / "DDNM" / "ddnm_inpainting.py").exists(),
            "genpc_instantmesh_src": (UPSTREAM / "src").exists(),
            "external_ddnm_repo": (DEPENDENCIES / "DDNM").exists(),
            "external_instantmesh_repo": (DEPENDENCIES / "InstantMesh").exists(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
