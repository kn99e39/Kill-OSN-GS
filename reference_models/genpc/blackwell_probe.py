"""Read-only runtime probe for the isolated GenPC Blackwell environment."""

import json
import platform
import sys

import torch


def main() -> None:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        result["devices"] = devices
        result["arch_list"] = torch.cuda.get_arch_list()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
