"""
Model Evaluation Script

This script loads previously trained models from a specified directory,
evaluates them on the MNIST test dataset, and saves the accuracy results
to a CSV file.
"""

import torch
import click
import os.path

from tqdm import tqdm

from sbfl_nn.models.architectures import (
    load_dataset,
    DenseModel,
    DenseModelHelper,
    MLDataset,
)
from sbfl_nn.models.training_logger import (
    log_loading_all_models,
    log_loading_dataset,
    computation_progress_step,
)
from sbfl_nn.utils.files import natural_sort_key, get_pth_files
from sbfl_nn.utils.gpu import DEVICE

from scripts.utils import (
    get_logger,
    log_section,
    log_subsection,
    log_completion,
    find_subdirs,
    Timer,
)

logger = get_logger(__name__)


def get_evaluation_progress_bar(num_models):
    """Initialize a progress bar for model evaluation."""
    return tqdm(total=num_models, desc="Evaluating models", unit="model")


def create_csv_file(csv_file):
    """Create a CSV file with headers if it doesn't exist."""
    with open(csv_file, "w") as f:
        f.write("Model File,Number of Layers,Accuracy\n")


def expand_csv_file(
    model_name: str, acc: float, num_layers: int, csv_file: str
):
    """Expand the CSV file with new data."""
    with open(csv_file, "a") as f:
        f.write(f"{model_name},{num_layers},{acc:.2f}\n")


@click.command()
@click.option(
    "--models-dir",
    default="outputs/models",
    help="Base directory containing model config subdirectories.",
)
def main(models_dir: str):

    # Find all config subdirectories in the models directory
    config_dirs = find_subdirs(models_dir)
    if not config_dirs:
        logger.error(f"No config subdirectories found in {models_dir}")
        return

    logger.info(f"Found {len(config_dirs)} config(s): {config_dirs}")

    # Process each config directory
    for config_name in config_dirs:
        config_path = os.path.join(models_dir, config_name)
        log_section(logger, f"Processing config: {config_name}")

        with Timer(logger, f"Config '{config_name}' evaluation completed"):
            # Find dataset subdirectories (mnist, fmnist, etc.)
            dataset_dirs = find_subdirs(config_path, required=False)

            if not dataset_dirs:
                logger.warning(f"No dataset subdirectories in {config_path}")
                continue

            # Process each dataset directory
            for dataset_name in dataset_dirs:
                dataset_path = os.path.join(config_path, dataset_name)
                log_subsection(logger, f"Processing dataset: {dataset_name}")

                with log_loading_all_models(dataset_path):
                    model_files = get_pth_files(dataset_path)
                    model_files.sort(key=natural_sort_key)

                if not model_files:
                    logger.warning(f"No model files found in {dataset_path}")
                    continue

                # Determine which dataset to use based on directory name
                if dataset_name.lower() == "mnist":
                    ml_dataset = MLDataset.MNIST
                elif dataset_name.lower() == "fmnist":
                    ml_dataset = MLDataset.FMNIST
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue

                with log_loading_dataset(ml_dataset):
                    _, test_loader = load_dataset(ml_dataset)

                # CSV file path for this dataset
                csv_file = os.path.join(dataset_path, "model_accuracies.csv")

                # Check if the file exists and create with header if it doesn't
                if not os.path.exists(csv_file):
                    create_csv_file(csv_file)

                pbar = get_evaluation_progress_bar(len(model_files))
                for model_file in model_files:
                    with computation_progress_step(pbar, model_file):

                        # Load the model
                        model: DenseModel = torch.load(
                            model_file, map_location=DEVICE, weights_only=False
                        )

                        # Evaluate the model using custom evaluate method
                        acc = DenseModelHelper.evaluate(model, test_loader)

                        # Count the number of layers in the model
                        # Subtract 1 to exclude the model itself
                        num_layers = len(list(model.named_modules())) - 1

                        expand_csv_file(
                            model_file, acc, num_layers, csv_file=csv_file
                        )

                        # Clean up to avoid memory issues
                        del model

    log_completion(logger, "Model evaluation completed for all configs")


if __name__ == "__main__":
    main()
