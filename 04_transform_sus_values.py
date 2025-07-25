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
from tqdm import tqdm

from helpers_preprocess.dataset_generation import (
    SusPredictionDataset, DatasetEntries)
from helpers_general.file_handling import get_pkl_files, natural_sort_key
from helpers_general.gpu import DEVICE

# Configuration constants
DIR_SUS = "models_image_classification/mnist"        # Directory containing suspiciousness value files
DIR_MODEL = "models_image_classification/mnist"      # Directory containing model files
# output directory for the dataset

FILE_OUTPUT_TORCH = "models_image_classification/mnist/dataset/mnist_dataset_torch.pth"  # Output dataset path
FILE_OUTPUT_NUMPY = "models_image_classification/mnist/dataset/mnist_dataset_numpy.pkl"  # Output dataset path
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


def load_model_and_sus_values(sus_file):
    """Load model and suspiciousness values for a given file."""
    sus_values = load_sus_values(sus_file)
    model_file = sus_file.replace(
        DIR_SUS, DIR_MODEL).replace("_sus.pkl", ".pth")
    model = load_model(model_file)
    return model, sus_values


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

    Returns:
    --------
    SusPredictionDataset
        Dataset containing features and suspiciousness values for all neurons
    """
    # Initialize empty stacks - these will accumulate data from all models and layers
    stacks = DatasetEntries()

    # Process each file
    pbar = tqdm(sus_files, desc="Processing files")

    for sus_file in pbar:
        pbar.set_description(f"Processing {sus_file}")

        # Load model and suspiciousness values
        sus_values = load_sus_values(sus_file)
        model_file = sus_file.replace("_sus.pkl", ".pth")
        model = load_model(model_file)

        # Process each layer
        for layer_name in sus_values.keys():
            layer_idx = torch.tensor(
                list(sus_values.keys()).index(layer_name)).to(DEVICE)

            if LOGGING:
                print(
                    f"Layer name: {layer_name}, index: {layer_idx.item() if layer_idx.numel() > 0 else 0}")

            model_file = sus_file.replace("_sus.pkl", ".pth")

            # Process layer data and update stacks
            stacks.process_layer_suspiciousness(
                model,
                sus_values,
                layer_name,
                layer_idx)

    # Create and return the dataset
    return SusPredictionDataset(stacks)


if __name__ == "__main__":
    # Get list of model files
    sus_files = get_pkl_files(DIR_SUS)
    sus_files.sort(key=natural_sort_key)

    print(f"Found {len(sus_files)} .pkl files in {DIR_SUS}")

    # Create dataset from suspiciousness values
    dataset = create_suspiciousness_dataset(sus_files)
    torch.save(dataset, FILE_OUTPUT_TORCH)

    features, targets = dataset.get_numpy()
    numpy_object = {
        'features': features,
        'targets': targets
    }
    with open(FILE_OUTPUT_NUMPY, 'wb') as f:
        pickle.dump(numpy_object, f)
