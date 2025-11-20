"""
Suspiciousness Values Transformation

This script transforms suspiciousness values from multiple models into a cohesive dataset
for neural network fault localization. It processes layer-by-layer suspiciousness values
from multiple neural network models and compiles them into a single dataset for training
suspiciousness prediction models.

The workflow is:
1. Load suspiciousness values computed by SBFL techniques (Ochiai and Tarantula)
2. Process each layer and neuron, extracting features and suspiciousness scores
3. Create a unified dataset that combines data from all models
4. Save the dataset for later use in training suspiciousness prediction models
"""
import pickle
import torch
import os
import time
from tqdm import tqdm

from helpers_preprocess.dataset_generation import (
    SusPredictionDataset, DatasetEntries)
from helpers_general.file_handling import get_pkl_files, natural_sort_key
from helpers_general.gpu import DEVICE

LOGGING = False                 # Enable/disable verbose logging


def load_sus_values(sus_file):
    """Load suspiciousness values for a given file."""
    with open(sus_file, 'rb') as f:
        sus_values = pickle.load(f)

    return sus_values


def load_model(model_file):
    """Load model corresponding to the given suspiciousness file."""

    with open(model_file, 'rb') as f:
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
        model_file = sus_file.replace("output_sus_values", "output_models").replace("_sus.pkl", ".pth")
        model = load_model(model_file)

        # Process each layer
        for layer_name in sus_values.keys():
            layer_idx = torch.tensor(
                list(sus_values.keys()).index(layer_name)).to(DEVICE)

            if LOGGING:
                layer_info = (
                    f"Layer name: {layer_name}, index: "
                    f"{layer_idx.item() if layer_idx.numel() > 0 else 0}")
                print(layer_info)

            # Process layer data and update stacks
            stacks.process_layer_suspiciousness(
                model,
                sus_values,
                layer_name,
                layer_idx)

    # Create and return the dataset
    return SusPredictionDataset(stacks)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--sus-dir', default="output_sus_values",
                  help='Base directory containing suspiciousness value files.')
    @click.option('--output-dir', default="output_sus_datasets",
                  help='Base directory to save datasets.')
    def main(sus_dir: str, output_dir: str):
        """Transform suspiciousness values into datasets."""

        # Find all config subdirectories in the sus directory
        if not os.path.exists(sus_dir):
            print(f"Error: Suspiciousness directory {sus_dir} does not exist")
            return

        # Get all subdirectories (each representing a config)
        config_dirs = [
            d for d in os.listdir(sus_dir)
            if os.path.isdir(os.path.join(sus_dir, d))
        ]

        if not config_dirs:
            print(f"No config subdirectories found in {sus_dir}")
            return

        config_dirs.sort()
        print(f"Found {len(config_dirs)} config(s): {config_dirs}")

        # Process each config directory
        for config_name in config_dirs:
            config_sus_path = os.path.join(sus_dir, config_name)
            print(f"\n{'='*60}")
            print(f"Processing config: {config_name}")
            print(f"{'='*60}\n")
            
            # Start timing for entire config
            config_start_time = time.time()

            # Find dataset subdirectories (mnist, fmnist, etc.)
            dataset_dirs = [
                d for d in os.listdir(config_sus_path)
                if os.path.isdir(os.path.join(config_sus_path, d))
            ]

            if not dataset_dirs:
                print(f"No dataset subdirectories found in {config_sus_path}")
                continue

            # Process each dataset directory
            for dataset_name in dataset_dirs:
                dataset_sus_path = os.path.join(config_sus_path, dataset_name)

                print(f"\n{'-'*40}")
                print(f"Processing dataset: {dataset_name}")
                print(f"{'-'*40}\n")

                # Get list of suspiciousness files
                sus_files = get_pkl_files(dataset_sus_path)
                sus_files.sort(key=natural_sort_key)

                if not sus_files:
                    print(f"No .pkl files found in {dataset_sus_path}")
                    continue

                print(f"Found {len(sus_files)} .pkl files in "
                      f"{dataset_sus_path}")

                # Create dataset from suspiciousness values
                dataset = create_suspiciousness_dataset(sus_files)

                # Create output directory for this config and dataset
                dataset_output_dir = os.path.join(
                    output_dir, config_name, dataset_name)
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Define output paths
                file_output_torch = os.path.join(
                    dataset_output_dir,
                    f"{dataset_name}_dataset_torch.pth")
                file_output_numpy = os.path.join(
                    dataset_output_dir,
                    f"{dataset_name}_dataset_numpy.pkl")

                # Save PyTorch dataset
                torch.save(dataset, file_output_torch)
                print(f"Saved PyTorch dataset to: {file_output_torch}")

                # Save NumPy dataset
                features, targets = dataset.get_numpy()
                numpy_object = {
                    'features': features,
                    'targets': targets
                }
                with open(file_output_numpy, 'wb') as f:
                    pickle.dump(numpy_object, f)
                print(f"Saved NumPy dataset to: {file_output_numpy}")
            
            # Log total time for this config
            config_duration = time.time() - config_start_time
            config_duration_minutes = config_duration / 60
            config_duration_hours = config_duration / 3600
            
            print(f"\n{'='*60}")
            print(f"Configuration '{config_name}' transformation completed")
            print(f"Total transformation time for all datasets in this config:")
            print(f"  {config_duration:.2f} seconds")
            print(f"  {config_duration_minutes:.2f} minutes")
            print(f"  {config_duration_hours:.2f} hours")
            print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print("Dataset transformation completed for all configs and datasets")
        print(f"{'='*60}")

    main()
