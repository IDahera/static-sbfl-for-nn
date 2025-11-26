"""
Suspiciousness Values Transformation

This script transforms suspiciousness values from multiple models into a
cohesive dataset for neural network fault localization. It processes
layer-by-layer suspiciousness values from multiple neural network models
and compiles them into a single dataset for training suspiciousness
prediction models.

The workflow is:
1. Load suspiciousness values computed by SBFL techniques (Ochiai/Tarantula)
2. Process each layer and neuron, extracting features and suspiciousness
3. Create a unified dataset that combines data from all models
4. Save the dataset for later use in training prediction models
"""

import pickle
import torch
import os
import click

from tqdm import tqdm

from sbfl_nn.data.dataset_generation import (
    SusPredictionDataset,
    DatasetEntries,
)
from sbfl_nn.utils.files import get_pkl_files, natural_sort_key
from sbfl_nn.utils.gpu import DEVICE

from scripts.utils import (
    get_logger,
    log_section,
    log_subsection,
    log_duration,
    log_completion,
    find_subdirs,
    Timer,
)

logger = get_logger(__name__)


def load_sus_values(sus_file):
    """Load suspiciousness values for a given file."""
    with open(sus_file, "rb") as f:
        sus_values = pickle.load(f)

    return sus_values


def load_model(model_file):
    """Load model corresponding to the given suspiciousness file."""

    with open(model_file, "rb") as f:
        model = torch.load(f, map_location=DEVICE, weights_only=False)
    return model


def create_suspiciousness_dataset(sus_files):
    """Process all suspiciousness files and create a dataset.

    This function is the main processing pipeline that:
    1. Initializes empty tensors to hold all data
    2. Iterates through each suspiciousness file
    3. Loads the corresponding model and suspiciousness values
    4. Processes each layer in the model
    5. Creates a final dataset from the accumulated tensors

    Parameters:
    -----------
    sus_files : list
        List of suspiciousness value files to process
    dir_model : str
        Directory containing the model files

    Returns:
    --------
    SusPredictionDataset
        Dataset containing features and suspiciousness values for all neurons
    """
    # Initialize empty stacks - accumulate data from all models and layers
    stacks = DatasetEntries()

    # Process each file
    pbar = tqdm(sus_files, desc="Processing files")

    for sus_file in pbar:
        pbar.set_description(f"Processing {sus_file}")

        # Load model and suspiciousness values
        sus_values = load_sus_values(sus_file)
        model_file = sus_file.replace(
            "outputs/suspiciousness", "outputs/models"
        ).replace("_sus.pkl", ".pth")
        model = load_model(model_file)

        # Process each layer
        for layer_name in sus_values.keys():
            layer_idx = torch.tensor(
                list(sus_values.keys()).index(layer_name)
            ).to(DEVICE)

            # Process layer data and update stacks
            stacks.process_layer_suspiciousness(
                model, sus_values, layer_name, layer_idx
            )

    # Create and return the dataset
    return SusPredictionDataset(stacks)


@click.command()
@click.option(
    "--sus-dir",
    default="outputs/suspiciousness",
    help="Base directory containing suspiciousness value files.",
)
@click.option(
    "--output-dir",
    default="outputs/datasets",
    help="Base directory to save datasets.",
)
def main(sus_dir: str, output_dir: str):
    """Transform suspiciousness values into datasets."""

    # Find all config subdirectories in the sus directory
    config_dirs = find_subdirs(sus_dir)
    if not config_dirs:
        logger.error(f"No config subdirectories found in {sus_dir}")
        return

    logger.info(f"Found {len(config_dirs)} config(s): {config_dirs}")

    # Process each config directory
    for config_name in config_dirs:
        config_sus_path = os.path.join(sus_dir, config_name)
        log_section(logger, f"Processing config: {config_name}")

        with Timer(logger, f"Config '{config_name}' completed"):
            # Find dataset subdirectories (mnist, fmnist, etc.)
            dataset_dirs = find_subdirs(config_sus_path, required=False)

            if not dataset_dirs:
                logger.warning(f"No dataset subdirs in {config_sus_path}")
                continue

            # Process each dataset directory
            for dataset_name in dataset_dirs:
                dataset_sus_path = os.path.join(config_sus_path, dataset_name)
                log_subsection(logger, f"Processing dataset: {dataset_name}")

                # Get list of suspiciousness files
                sus_files = get_pkl_files(dataset_sus_path)
                sus_files.sort(key=natural_sort_key)

                if not sus_files:
                    logger.warning(f"No .pkl files in {dataset_sus_path}")
                    continue

                logger.info(f"Found {len(sus_files)} .pkl files")

                # Create dataset from suspiciousness values
                dataset = create_suspiciousness_dataset(sus_files)

                # Create output directory for this config and dataset
                dataset_output_dir = os.path.join(
                    output_dir, config_name, dataset_name
                )
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Define output paths
                file_output_torch = os.path.join(
                    dataset_output_dir, f"{dataset_name}_dataset_torch.pth"
                )
                file_output_numpy = os.path.join(
                    dataset_output_dir, f"{dataset_name}_dataset_numpy.pkl"
                )

                # Save PyTorch dataset
                torch.save(dataset, file_output_torch)
                logger.info(f"Saved PyTorch dataset: {file_output_torch}")

                # Save NumPy dataset
                features, targets = dataset.get_numpy()
                numpy_object = {"features": features, "targets": targets}
                with open(file_output_numpy, "wb") as f:
                    pickle.dump(numpy_object, f)
                logger.info(f"Saved NumPy dataset: {file_output_numpy}")

    log_completion(logger, "Dataset transformation completed")


if __name__ == "__main__":
    main()
