"""
Sequential Model Generator

This script generates multiple neural network models with random architectures,
trains them on the MNIST or Fashion MNIST dataset, and saves the trained
models to disk.
"""
import random
import time
import os
import glob
import json
import torch.optim as optim

from helpers_general.logging import logging
from helpers_general.gpu import DEVICE
from helpers_models.models import (MLDataset, DenseModel,
                                   DenseModelHelper, load_dataset)
from helpers_models.randomization import generate_rand_dense_seq
import click


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Error loading config: {e}")
        return None


def save_config(config_path: str, config: dict):
    """Save configuration to JSON file."""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        logging.info(f"Configuration saved to {config_path}")
    except IOError as e:
        logging.error(f"Error saving config: {e}")


def configs_match(old_config: dict, new_config: dict) -> bool:
    """Check if relevant configuration parameters match."""
    if old_config is None:
        return False

    relevant_keys = ['min_layers', 'max_layers', 'min_neurons',
                     'max_neurons', 'learning_rate', 'epochs']

    for key in relevant_keys:
        if old_config.get(key) != new_config.get(key):
            return False

    return True


def prompt_user_for_deletion(output_dir: str) -> bool:
    """Ask user whether to delete existing models due to config change."""
    print("\n" + "="*60)
    print("⚠️  CONFIGURATION CHANGE DETECTED!")
    print("="*60)
    print("\nThe training configuration has changed.")
    print(f"Existing models in '{output_dir}' may be inconsistent.")
    print("\nOptions:")
    print("  - Delete existing models and start fresh")
    print("  - Keep existing models and continue from where you left off")
    print("\n" + "="*60)

    response = input("\nDelete existing models? [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def check_configuration(dataset_name: str,
                        dataset_output_dir: str,
                        existing_config: dict,
                        current_config: dict) -> None:
    """Check config changes and handle model deletion if needed."""
    if not existing_config or configs_match(existing_config, current_config):
        return

    logging.warning(
        f"Configuration mismatch detected for {dataset_name}!")
    logging.info("Old config:")
    for key, value in existing_config.items():
        if key in current_config:
            old_val = value
            new_val = current_config[key]
            if old_val != new_val:
                logging.info(f"  {key}: {old_val} -> {new_val}")

    # Ask user what to do
    if prompt_user_for_deletion(dataset_output_dir):
        logging.info(
            f"Deleting models in {dataset_output_dir}...")
        # Delete model files
        model_files = glob.glob(
            os.path.join(dataset_output_dir, "model_*.pth"))
        for model_file in model_files:
            os.remove(model_file)
            logging.info(f"Deleted: {model_file}")
        logging.info("All models deleted. Starting fresh.")
    else:
        logging.info(
            "Keeping existing models. "
            "Continuing with current configuration.")


def find_highest_model_index(output_dir: str) -> int:
    """Find the highest existing model index in the output directory."""
    if not os.path.exists(output_dir):
        logging.info(
            f"Output directory {output_dir} does not exist, "
            f"starting from index 0")
        return -1

    # Look for model files with pattern model_*.pth
    model_files = glob.glob(os.path.join(output_dir, "model_*.pth"))

    if not model_files:
        logging.info(
            f"No existing models found in {output_dir}, starting from index 0")
        return -1

    # Extract model indices from filenames
    model_indices = []
    for model_file in model_files:
        filename = os.path.basename(model_file)
        # Extract number from filename like "model_123.pth"
        try:
            index_str = filename.replace("model_", "").replace(".pth", "")
            model_indices.append(int(index_str))
        except ValueError:
            # Skip files that don't match the expected pattern
            continue

    if not model_indices:
        logging.info(
            f"No valid model files found in {output_dir}, "
            f"starting from index 0")
        return -1

    highest_index = max(model_indices)
    logging.info(f"Found {len(model_indices)} existing models in {output_dir}")
    logging.info(f"Highest existing model index: {highest_index}")
    return highest_index


def generate_and_train_models(dataset: MLDataset,
                              num_models: int,
                              epochs: int,
                              learning_rate: float,
                              layer_bounds: tuple[int, int],
                              neuron_bounds: tuple[int, int],
                              output_dir: str):
    """Generate random models, train them and save to disk."""
    # Find the highest existing model index
    highest_existing_index = find_highest_model_index(output_dir)
    start_index = highest_existing_index + 1
    end_index = start_index + num_models

    dataset_name = dataset.name
    logging.info(f"Training on dataset: {dataset_name}")
    logging.info(
        f"Starting model generation from index {start_index} "
        f"to {end_index - 1}")
    logging.info(f"Will generate {num_models} new models")

    # Load dataset once before the loop
    train_loader, test_loader = load_dataset(dataset)

    # Main generation loop
    for model_idx in range(start_index, end_index):

        # Generate random architecture
        num_layers = random.randint(*layer_bounds)
        layer_sizes = generate_rand_dense_seq(num_layers,
                                              neuron_bounds)

        # Create model
        model = DenseModel(input_size=784,
                           output_size=10,
                           layer_sizes=layer_sizes).to(DEVICE)
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate)

        # Train the model
        start_time = time.time()

        DenseModelHelper.train(
            model,
            model_name=f"model_{model_idx}",
            epochs=epochs,
            loader=train_loader,
            optimizer=optimizer)

        # Log training time
        training_time = time.time() - start_time
        logging.info(f'Training completed in {training_time:.2f} seconds')

        # Evaluate model
        accuracy = DenseModelHelper.evaluate(model,
                                             test_loader)
        logging.info(f"Model {model_idx}: Accuracy = {accuracy:.2f}%")

        # Save the model
        DenseModelHelper.save_model(model,
                                    model_idx,
                                    num_layers,
                                    output_dir)

        # Clean up to avoid memory issues
        del model


@click.command()
@click.option('--num-models',
              default=1000,
              help='Number of models to generate')
@click.option('--epochs',
              default=1,
              help='Number of training epochs')
@click.option('--min-layers',
              default=5,
              help='Minimum number of layers')
@click.option('--max-layers',
              default=25,
              help='Maximum number of layers')
@click.option('--min-neurons',
              default=10,
              help='Minimum neurons per layer')
@click.option('--max-neurons',
              default=50,
              help='Maximum neurons per layer')
@click.option('--learning-rate',
              default=0.001,
              help='Learning rate for training')
@click.option('--output-dir',
              default='models/',
              help='Base directory to save models')
def main(num_models,
         epochs,
         min_layers,
         max_layers,
         min_neurons,
         max_neurons,
         learning_rate,
         output_dir):
    """Generate and train models with random architectures."""

    logging.info("Starting sequential model generation")
    
    # Create current configuration
    current_config = {
        'min_layers': min_layers,
        'max_layers': max_layers,
        'min_neurons': min_neurons,
        'max_neurons': max_neurons,
        'learning_rate': learning_rate,
        'epochs': epochs
    }
    
    # Train on both MNIST and Fashion MNIST
    datasets_to_train = [
        (MLDataset.MNIST, 'mnist'),
        (MLDataset.FMNIST, 'fmnist')
    ]
    
    for dataset, dataset_name in datasets_to_train:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {dataset_name.upper()} dataset")
        logging.info(f"{'='*60}\n")
        
        # Create dataset-specific output directory
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        config_path = os.path.join(dataset_output_dir, 'config.json')
        
        # Ensure output directory exists
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Load existing config
        existing_config = load_config(config_path)

        # Check if configuration has changed and handle it
        check_configuration(dataset_name, dataset_output_dir,
                            existing_config, current_config)

        # Save current configuration
        save_config(config_path, current_config)
        
        # Generate and train models
        generate_and_train_models(
            dataset=dataset,
            num_models=num_models,
            epochs=epochs,
            learning_rate=learning_rate,
            layer_bounds=(min_layers, max_layers),
            neuron_bounds=(min_neurons, max_neurons),
            output_dir=dataset_output_dir)
    
    logging.info("\nSequential model generation completed for all datasets")


if __name__ == "__main__":
    main()
