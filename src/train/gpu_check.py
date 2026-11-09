"""GPU check shipped by overbrainer, run before the trainer's first command.

Usage: python gpu_check.py RESULT MIN_GPUS

Checks that the trainer's PyTorch sees at least MIN_GPUS CUDA or ROCm devices
(any MPS device on macOS), writes the result as JSON to RESULT, prints it on one
stderr line, and exits 0 when the check passes. Otherwise it exits with the
reason's code: 3 when torch cannot be imported, 4 when it sees no device, 5 when
it sees fewer than MIN_GPUS.
"""

import json
import sys

IMPORT_FAILED = 3
NO_DEVICE = 4
TOO_FEW = 5


def _check(min_gpus):
    """The result of the check as a dict, its "code" 0 when it passes."""
    try:
        import torch
    except Exception as error:  # any failure to import means no usable torch
        return {"code": IMPORT_FAILED, "error": type(error).__name__}
    result = {"torch": getattr(torch, "__version__", None), "required": min_gpus}
    version = getattr(torch, "version", None)
    result["cuda"] = getattr(version, "cuda", None)
    result["hip"] = getattr(version, "hip", None)
    if sys.platform == "darwin":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(mps is not None and mps.is_available())
        result["backend"] = "mps"
        result["devices"] = ["mps"] if available else []
        result["code"] = 0 if available else NO_DEVICE
        return result
    result["backend"] = "cuda"
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    result["devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
    if count == 0:
        result["code"] = NO_DEVICE
    elif count < min_gpus:
        result["code"] = TOO_FEW
    else:
        result["code"] = 0
    return result


def main():
    """Runs the check, writes its result, and exits with its code."""
    path, min_gpus = sys.argv[1], int(sys.argv[2])
    result = _check(min_gpus)
    line = json.dumps(result, sort_keys=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(line + "\n")
    print("overbrainer gpu check: " + line, file=sys.stderr)
    sys.exit(result["code"])


if __name__ == "__main__":
    main()
