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

from helpers_models.models import (load_dataset,
                                   DenseModel,
                                   DenseModelHelper,
                                   MLDataset)
from helpers_models.logging import (log_loading_all_models,
                                    log_loading_dataset,
                                    computation_progress_step)
from helpers_general.file_handling import (natural_sort_key,
                                           get_pth_files)
from helpers_general.gpu import DEVICE


def get_evaluation_progress_bar(num_models):
    """Initialize a progress bar for model evaluation."""
    return tqdm(total=num_models, desc="Evaluating models", unit="model")


def create_csv_file(csv_file):
    """Create a CSV file with headers if it doesn't exist."""
    with open(csv_file, "w") as f:
        f.write("Model File,Number of Layers,Accuracy\n")


def expand_csv_file(model_name: str,
                    acc: float,
                    num_layers: int,
                    csv_file: str):
    """Expand the CSV file with new data."""
    with open(csv_file, "a") as f:
        f.write(f"{model_name},{num_layers},{acc:.2f}\n")


@click.command()
@click.option('--dir',
              default="models_image_classification/mnist",
              help='Directory to load models from.')
@click.option('--csv',
              default="models_image_classification/mnist/model_accuracies.csv",
              help='CSV file to save accuracies.')
def main(dir: str, csv: str):

    with log_loading_all_models(dir):
        model_files = get_pth_files(dir)
        model_files.sort(key=natural_sort_key)

    with log_loading_dataset(MLDataset.MNIST):
        _, test_loader = load_dataset(MLDataset.MNIST)

    # Check if the file exists and create with header if it doesn't
    if not os.path.exists(csv):
        create_csv_file(csv)

    pbar = get_evaluation_progress_bar(len(model_files))
    for model_file in model_files:
        with computation_progress_step(pbar, model_file):

            # Load the model
            model: DenseModel = torch.load(model_file,
                                           map_location=DEVICE,
                                           weights_only=False)

            # Evaluate the model using custom evaluate method
            acc = DenseModelHelper.evaluate(model,
                                            test_loader)

            # Count the number of layers in the model
            # Subtract 1 to exclude the model itself
            num_layers = len(list(model.named_modules())) - 1

            expand_csv_file(model_file,
                            acc,
                            num_layers,
                            csv_file=csv)

            # Clean up to avoid memory issues
            del model


if __name__ == "__main__":
    main()
