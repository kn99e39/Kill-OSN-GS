import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
from loss_functions import chamfer_3DDist, emdModule


def main():
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    p1 = torch.linspace(-1.0, 1.0, 256, device=device).reshape(1, 256, 1).repeat(1, 1, 3)
    p2 = p1 + 0.01
    chamfer = chamfer_3DDist()
    d1, d2, _, _ = chamfer(p1, p2)
    emd = emdModule()
    emd_dist, _ = emd(p1, p2, eps=0.005, iters=10)
    torch.cuda.synchronize(device)
    print(json.dumps({
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "chamfer_shape": list(d1.shape),
        "chamfer_mean": float(d1.mean().item()),
        "emd_shape": list(emd_dist.shape),
        "emd_mean": float(emd_dist.mean().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }, indent=2))


if __name__ == "__main__":
    main()
