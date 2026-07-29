import os

import torch


def get_device():
    """
    Return the requested PyTorch device.

    ``SBFL_DEVICE`` may be set to ``auto`` (the default), ``cuda``, ``mps``,
    or ``cpu``. Explicit requests fail early when the requested accelerator is
    unavailable; this prevents a full artefact run from silently falling back
    to an impractically slow CPU execution.
    """
    requested = os.environ.get("SBFL_DEVICE", "auto").strip().lower()
    supported = {"auto", "cuda", "mps", "cpu"}

    if requested not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported SBFL_DEVICE={requested!r}; choose {choices}")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SBFL_DEVICE=cuda was requested, but CUDA is not available. "
                "Run the CUDA image with GPU access (for Docker: --gpus all)."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "SBFL_DEVICE=mps was requested, but MPS is not available. "
                "MPS is available to native macOS PyTorch, not Linux containers."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# TODO Possibly deprecated
# def check_cache_size(limit_gb=10):
#     """
#     Check the size of the cache directory. Raise an exception if the cache size exceeds specified limit (GB).
#     """
#     try:
#         cache_info = scan_cache_dir()
#         cache_size_gb = cache_info.size_on_disk / (1024**3)
#         print(f"Current cache size: {cache_size_gb:.2f} GB")

#         if cache_size_gb > limit_gb:
#             raise Exception(f"Cache size is too large: {cache_size_gb} GB")
#     except Exception as e:
#         print(f"Error checking cache size: {e}")


# TODO Possibly deprecated
# def check_hardware_acceleration():
#     """
#     Check if hardware acceleration is available (and used).
#     """
#     import torch

#     if torch.cuda.is_available():
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name()}")

#     device = torch.device("cuda" if torch.cuda.is_available() else
#                           "mps" if torch.backends.mps.is_available() else
#                           "cpu")
#     print(f"PyTorch is using: {device}")
