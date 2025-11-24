import torch


def __get_best_device():
    """
    Returns the best available device in the following order of preference:
    1. MPS (Apple Silicon GPU)
    2. CUDA (NVIDIA GPU)
    3. CPU (default fallback)
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = __get_best_device()


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
