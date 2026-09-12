"""External compatibility shims for the released GenPC/InstantMesh adapters.

The pinned GenPC adapter imports ``save_ply`` even though its active export
path uses the released InstantMesh ``save_glb`` function.  The pinned official
InstantMesh checkout does not expose ``save_ply``.  This shim adds the
historical, signature-compatible helper without changing either checkout.
"""

from __future__ import annotations

import numpy as np
import trimesh


def install_mesh_util_compat() -> str:
    import src.utils.mesh_util as mesh_util

    if hasattr(mesh_util, "save_ply"):
        return "already_present"

    def save_ply(pointnp_px3, colornp_px3, fpath):
        pointnp_px3 = np.asarray(pointnp_px3)
        colornp_px3 = np.asarray(colornp_px3)
        point_cloud = trimesh.PointCloud(vertices=pointnp_px3, colors=colornp_px3)
        point_cloud.export(fpath)

    mesh_util.save_ply = save_ply
    return "injected_external_compat"


def install_diffusers_compat() -> str:
    """Allow the released custom Zero123++ pipeline on current diffusers."""
    import diffusers

    original = diffusers.DiffusionPipeline.from_pretrained
    if getattr(original, "_genpc_trust_remote_code", False):
        return "already_present"

    def from_pretrained(*args, **kwargs):
        if kwargs.get("custom_pipeline") == "zero123plus":
            kwargs.setdefault("trust_remote_code", True)
        return original(*args, **kwargs)

    from_pretrained._genpc_trust_remote_code = True
    diffusers.DiffusionPipeline.from_pretrained = from_pretrained
    return "injected_custom_pipeline_trust_remote_code"
