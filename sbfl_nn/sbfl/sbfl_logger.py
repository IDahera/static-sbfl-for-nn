import contextlib
from tqdm import tqdm

@contextlib.contextmanager
def computation_progress_step(pbar: tqdm, model_file: str):
    pbar.set_description(f"Evaluating {model_file}")
    try:
        yield
    finally:
        pbar.update(1)
