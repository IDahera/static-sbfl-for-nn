"""
Suspiciousness Computation Script

Computes suspiciousness values for neural network models using SBFL metrics.
"""

import os
import torch
import click
import time
import csv

from tqdm import tqdm
from torch.utils.data import Subset

from sbfl_nn.models.architectures import DenseModel, MLDataset, load_dataset
from sbfl_nn.sbfl.metrics import (
    get_activations_with_hooks,
    compute_hit_spectrum,
    get_ochiai,
    get_tarantula,
    save_sus_values,
    unsqueeze_tensors,
)
from sbfl_nn.sbfl.sbfl_logger import computation_progress_step
from sbfl_nn.utils.files import get_pth_files, natural_sort_key
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


def initialize_timing_csv(model_dir):
    """Initialize CSV file for timing data with headers."""
    csv_path = os.path.join(model_dir, "sus_computation_times.csv")

    # Create CSV with headers if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Model_Name",
                    "Model_File",
                    "Samples_Used",
                    "Threshold",
                    "Total_Computation_Time_Seconds",
                    "Num_Layers_Processed",
                    "Avg_Time_Per_Layer_Seconds",
                    "Timestamp",
                ]
            )

    return csv_path


def append_timing_data(
    csv_path,
    model_name,
    model_file,
    samples,
    threshold,
    computation_time,
    num_layers,
):
    """Append timing data for a model to the CSV file."""
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        avg_time_per_layer = (
            computation_time / num_layers if num_layers > 0 else 0.0
        )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow(
            [
                model_name,
                os.path.basename(model_file),
                samples,
                threshold,
                f"{computation_time:.4f}",
                num_layers,
                f"{avg_time_per_layer:.4f}",
                timestamp,
            ]
        )


def collect_all_activations(model, test_loader, target_class=0):
    """
    Collect activations for all layers of the model across the test dataset
    """
    device = next(model.parameters()).device
    all_activations = {}
    all_targets = {}

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # Create target tensor for binary classification
            # (is target_class or not)
            target_binary = target == target_class

            # Get activations from all layers for this batch
            batch_activations, outputs = get_activations_with_hooks(
                model, data
            )

            # Store the activations and targets
            for layer_name, activation in batch_activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                    all_targets[layer_name] = []
                all_activations[layer_name].append(activation)
                all_targets[layer_name].append(target_binary)

    # Concatenate all batches for each layer
    for layer_name in all_activations:
        all_activations[layer_name] = torch.cat(
            all_activations[layer_name], dim=0
        )
        all_targets[layer_name] = torch.cat(all_targets[layer_name], dim=0)

    return all_activations, all_targets


def analyze_all_layers(
    model: DenseModel,
    test_loader: torch.utils.data.DataLoader,
    target_class: int = 0,
    threshold: float = 0.0,
):
    """
    Analyze all layers and compute hit spectrum and suspiciousness.
    """
    # Collect activations for all layers
    all_activations, all_targets = collect_all_activations(
        model, test_loader, target_class
    )

    results = {}

    # Process each layer
    for layer_name, activations in all_activations.items():
        # Prepare tensors
        act_reshaped, targets_reshaped = unsqueeze_tensors(
            activations, all_targets[layer_name]
        )

        # Compute hit spectrum
        hs = compute_hit_spectrum(act_reshaped, targets_reshaped, threshold)

        # Compute suspiciousness metrics
        ochiai = get_ochiai(hs)
        tarantula = get_tarantula(hs)

        # Reshape metrics to the original layer shape
        ochiai_shaped = ochiai.reshape(hs.layer_shape)
        tarantula_shaped = tarantula.reshape(hs.layer_shape)

        # Store results
        results[layer_name] = {
            "hit_spectrum": hs,
            "ochiai": ochiai_shaped,
            "tarantula": tarantula_shaped,
        }

    return results


@click.command()
@click.option(
    "--models-dir",
    default="outputs/models",
    help="Base directory containing model config subdirectories.",
)
@click.option(
    "--output-dir",
    default="outputs/suspiciousness",
    help="Base directory to save suspiciousness values.",
)
@click.option(
    "--samples", default=1000, help="Number of samples to use for analysis."
)
@click.option(
    "--threshold",
    default=0.0,
    help="Activation threshold for hit spectrum computation.",
)
def main(models_dir: str, output_dir: str, samples: int, threshold: float):

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

        with Timer(logger, f"Config '{config_name}' completed"):
            # Create output directory for this config
            config_output_dir = os.path.join(output_dir, config_name)
            os.makedirs(config_output_dir, exist_ok=True)

            # Find dataset subdirectories (mnist, fmnist, etc.)
            dataset_dirs = find_subdirs(config_path, required=False)

            if not dataset_dirs:
                logger.warning(f"No dataset subdirectories in {config_path}")
                continue

            # Process each dataset directory
            for dataset_name in dataset_dirs:
                dataset_path = os.path.join(config_path, dataset_name)
                log_subsection(logger, f"Processing dataset: {dataset_name}")

                # Get list of model files
                model_files = get_pth_files(dataset_path)
                model_files.sort(key=natural_sort_key)

                if not model_files:
                    logger.warning(f"No model files in {dataset_path}")
                    continue

                logger.info(f"Found {len(model_files)} .pth files")

                # Create output subdirectory for dataset
                dataset_output_dir = os.path.join(
                    config_output_dir, dataset_name
                )
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Initialize timing CSV file for this dataset
                timing_csv_path = initialize_timing_csv(dataset_output_dir)
                logger.info(f"Timing CSV: {timing_csv_path}")

                # Determine which dataset to use
                if dataset_name.lower() == "mnist":
                    ml_dataset = MLDataset.MNIST
                elif dataset_name.lower() == "fmnist":
                    ml_dataset = MLDataset.FMNIST
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue

                pbar = tqdm(
                    total=len(model_files),
                    desc="Processing models",
                    unit="model",
                )
                for model_file in model_files:

                    with computation_progress_step(pbar, model_file):
                        model_name = os.path.basename(model_file).replace(
                            ".pth", ""
                        )

                        # Start timing for this model
                        start_time = time.time()

                        # Load model
                        model = torch.load(
                            model_file, map_location=DEVICE, weights_only=False
                        )
                        model.eval()

                        # Get test loader
                        _, test_loader = load_dataset(
                            ml_dataset, batch_size=64
                        )

                        # Create subset if needed
                        if len(test_loader.dataset) > samples:
                            sub_test_dataset = Subset(
                                test_loader.dataset, range(samples)
                            )
                            sub_test_loader = torch.utils.data.DataLoader(
                                sub_test_dataset, batch_size=64
                            )
                        else:
                            sub_test_loader = test_loader

                        # Analyze all layers (main computation)
                        sus_results = analyze_all_layers(
                            model=model,
                            test_loader=sub_test_loader,
                            target_class=0,
                            threshold=threshold,
                        )

                        # End timing for this model
                        computation_time = time.time() - start_time
                        num_layers = len(sus_results)

                        # Save results in dataset output directory
                        save_sus_values(
                            os.path.join(
                                dataset_output_dir, f"{model_name}_sus.pkl"
                            ),
                            sus_results,
                        )

                        # Log timing data to CSV
                        append_timing_data(
                            timing_csv_path,
                            model_name,
                            model_file,
                            samples,
                            threshold,
                            computation_time,
                            num_layers,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        del model, sus_results

    log_completion(logger, f"Suspiciousness computation done → {output_dir}")


if __name__ == "__main__":
    main()
