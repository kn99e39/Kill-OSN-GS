"""Minimal GPU smoke test for the PyTorch3D APIs used by GenPC."""

import json

import torch
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds


def main() -> None:
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    points = torch.tensor(
        [
            [-0.45, -0.45, 0.0],
            [0.45, -0.45, 0.0],
            [0.0, 0.45, 0.0],
            [0.0, 0.0, 0.35],
        ],
        dtype=torch.float32,
        device=device,
    )
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    eye = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=device)
    rotation, translation = look_at_view_transform(eye=eye, device=device)
    cameras = PerspectiveCameras(
        focal_length=(4.0,),
        R=rotation,
        T=translation,
        image_size=((32, 32),),
        device=device,
    )
    raster_settings = PointsRasterizationSettings(
        image_size=(32, 32),
        radius=torch.full((points.shape[0],), 0.05, device=device),
    )
    renderer = PulsarPointsRenderer(
        PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    ).to(device)
    image = renderer(
        Pointclouds(points=points[None], features=colors[None]),
        gamma=(1e-2,),
        zfar=(5.0,),
        znear=(1e-4,),
        radius_world=True,
        bg_col=torch.zeros(3, dtype=torch.float32, device=device),
    )
    torch.cuda.synchronize()
    result = {
        "status": "pass",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "render_shape": list(image.shape),
        "finite": bool(torch.isfinite(image).all().item()),
        "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
