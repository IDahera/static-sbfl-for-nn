import contextlib
import logging
import time
from tqdm import tqdm


@contextlib.contextmanager
def log_model_training(model_idx: int,
                       num_layers: int,
                       layer_sizes: list[int]):
    logging.info(
        f"Created model {model_idx} architecture with {num_layers} layers ({sum(layer_sizes)} neurons)")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logging.info(f"Completed loading models in {elapsed:.2f} seconds")


@contextlib.contextmanager
def log_loading_all_models(directory: str):
    logging.info(f"Load models from directory {directory}")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logging.info(f"Completed loading models in {elapsed:.2f} seconds")


@contextlib.contextmanager
def log_loading_dataset(dataset_name: str):
    logging.info(f"Load dataset {dataset_name}")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logging.info(f"{dataset_name} dataset loaded in {elapsed:.2f} seconds")


@contextlib.contextmanager
def computation_progress_step(pbar: tqdm, model_file: str):
    pbar.set_description(f"Evaluating {model_file}")
    try:
        yield
    finally:
        pbar.update(1)
