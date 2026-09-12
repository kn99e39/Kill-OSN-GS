import json

import torch
import nvdiffrast.torch as dr


def main():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    pos = torch.tensor(
        [[[-1.0, -1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0]]],
        device=device,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    ctx = dr.RasterizeCudaContext(device=device)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[8, 8])
    torch.cuda.synchronize(device)
    result = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "nvdiffrast_context": type(ctx).__name__,
        "raster_shape": list(rast.shape),
        "covered_pixels": int((rast[..., 3] > 0).sum().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
