"""
Sequential Model Generator

This script generates multiple neural network models with random architectures,
trains them on the MNIST or Fashion MNIST dataset, and saves the trained
models to disk.
"""

import random
import os
import glob
import json
import yaml
import time
import csv
import torch
import torch.optim as optim
import click

from sbfl_nn.utils.gpu import DEVICE
from sbfl_nn.models.architectures import (
    MLDataset,
    DenseModel,
    DenseModelHelper,
    load_dataset,
)
from sbfl_nn.models.randomization import generate_rand_dense_seq

from scripts.utils import (
    get_logger,
    log_section,
    log_subsection,
    log_duration,
    log_completion,
    Timer,
)

logger = get_logger(__name__)


def append_total_timing(output_base_dir, config_name, dataset_name,
                        num_models, total_time):
    """Append total timing data for model generation to CSV in outputs folder."""
    csv_path = os.path.join(output_base_dir, "model_generation_times.csv")
    
    # Create CSV with headers if it doesn't exist
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "Config",
                "Dataset",
                "Num_Models",
                "Total_Time_Seconds",
                "Avg_Time_Per_Model_Seconds",
                "Timestamp",
            ])
        
        avg_time = total_time / num_models if num_models > 0 else 0.0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        writer.writerow([
            config_name,
            dataset_name,
            num_models,
            f"{total_time:.2f}",
            f"{avg_time:.4f}",
            timestamp,
        ])
    
    logger.info(f"Timing logged to {csv_path}")


def load_config(config_path: str) -> dict:
    """Load configuration from JSON or YAML file."""
    if not os.path.exists(config_path):
        return None

    try:
        # Check file extension to determine format
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError, yaml.YAMLError) as e:
        logger.error(f"Error loading config: {e}")
        return None


def save_config(config_path: str, config: dict):
    """Save configuration to JSON file."""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Configuration saved to {config_path}")
    except IOError as e:
        logger.error(f"Error saving config: {e}")


def configs_match(old_config: dict, new_config: dict) -> bool:
    """Check if relevant configuration parameters match."""
    if old_config is None:
        return False

    relevant_keys = [
        "min_layers",
        "max_layers",
        "min_neurons",
        "max_neurons",
        "learning_rate",
        "epochs",
        "random_seed",
    ]

    for key in relevant_keys:
        if old_config.get(key) != new_config.get(key):
            return False

    return True


def prompt_user_for_deletion(output_dir: str) -> bool:
    """Ask user whether to delete existing models due to config change."""
    print("\n" + "=" * 60)
    print("⚠️  CONFIGURATION CHANGE DETECTED!")
    print("=" * 60)
    print("\nThe training configuration has changed.")
    print(f"Existing models in '{output_dir}' may be inconsistent.")
    print("\nOptions:")
    print("  - Delete existing models and start fresh")
    print("  - Keep existing models and continue from where you left off")
    print("\n" + "=" * 60)

    response = input("\nDelete existing models? [y/N]: ").strip().lower()
    return response in ["y", "yes"]


def check_configuration(
    dataset_name: str,
    dataset_output_dir: str,
    existing_config: dict,
    current_config: dict,
) -> None:
    """Check config changes and handle model deletion if needed."""
    if not existing_config or configs_match(existing_config, current_config):
        return

    logger.warning(f"Configuration mismatch detected for {dataset_name}!")
    logger.info("Old config:")
    for key, value in existing_config.items():
        if key in current_config:
            old_val = value
            new_val = current_config[key]
            if old_val != new_val:
                logger.info(f"  {key}: {old_val} -> {new_val}")

    # Ask user what to do
    if prompt_user_for_deletion(dataset_output_dir):
        logger.info(f"Deleting models in {dataset_output_dir}...")
        # Delete model files
        model_files = glob.glob(
            os.path.join(dataset_output_dir, "model_*.pth")
        )
        for model_file in model_files:
            os.remove(model_file)
            logger.info(f"Deleted: {model_file}")
        logger.info("All models deleted. Starting fresh.")
    else:
        logger.info(
            "Keeping existing models. "
            "Continuing with current configuration."
        )


def find_highest_model_index(output_dir: str) -> int:
    """Find the highest existing model index in the output directory."""
    if not os.path.exists(output_dir):
        logger.info(
            f"Output directory {output_dir} does not exist, "
            f"starting from index 0"
        )
        return -1

    # Look for model files with pattern model_*.pth
    model_files = glob.glob(os.path.join(output_dir, "model_*.pth"))

    if not model_files:
        logger.info(f"No existing models in {output_dir}, starting from 0")
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
        logger.info(f"No valid model files in {output_dir}, starting from 0")
        return -1

    highest_index = max(model_indices)
    logger.info(f"Found {len(model_indices)} existing models in {output_dir}")
    logger.info(f"Highest existing model index: {highest_index}")
    return highest_index


def generate_and_train_models(
    dataset: MLDataset,
    num_models: int,
    epochs: int,
    learning_rate: float,
    layer_bounds: tuple[int, int],
    neuron_bounds: tuple[int, int],
    output_dir: str,
):
    """Generate models until ``num_models`` exist in ``output_dir``."""
    highest_existing_index = find_highest_model_index(output_dir)
    existing_model_files = glob.glob(os.path.join(output_dir, "model_*.pth"))
    models_to_generate = max(0, num_models - len(existing_model_files))
    start_index = highest_existing_index + 1
    end_index = start_index + models_to_generate

    dataset_name = dataset.name
    logger.info(f"Training on dataset: {dataset_name}")
    logger.info(
        f"Target models: {num_models}; existing: {len(existing_model_files)}; "
        f"remaining: {models_to_generate}"
    )

    if models_to_generate == 0:
        logger.info("Target model count already reached; nothing to generate")
        return 0

    logger.info(f"Model generation: index {start_index} to {end_index - 1}")

    # Load dataset once before the loop
    train_loader, test_loader = load_dataset(dataset)

    # Main generation loop
    for model_idx in range(start_index, end_index):

        # Generate random architecture
        num_layers = random.randint(*layer_bounds)
        layer_sizes = generate_rand_dense_seq(num_layers, neuron_bounds)

        # Create model
        model = DenseModel(
            input_size=784, output_size=10, layer_sizes=layer_sizes
        ).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        with Timer(logger, f"Model {model_idx} training"):
            DenseModelHelper.train(
                model,
                model_name=f"model_{model_idx}",
                epochs=epochs,
                loader=train_loader,
                optimizer=optimizer,
            )

        # Evaluate model
        accuracy = DenseModelHelper.evaluate(model, test_loader)
        logger.info(f"Model {model_idx}: Accuracy = {accuracy:.2f}%")

        # Save the model
        DenseModelHelper.save_model(model, model_idx, num_layers, output_dir)

        # Clean up to avoid memory issues
        del model

    return models_to_generate


@click.command()
@click.option(
    "--config-dir",
    default="configs/model_gen",
    show_default=True,
    help="Directory containing config YAML files (used when --config is absent).",
)
@click.option(
    "--config",
    "config_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Specific YAML config to run. May be passed more than once.",
)
@click.option(
    "--dataset",
    "dataset_names",
    multiple=True,
    type=click.Choice(["mnist", "fmnist"], case_sensitive=False),
    help="Dataset to generate. May be repeated; defaults to both datasets.",
)
@click.option(
    "--output-dir",
    default="outputs/models",
    show_default=True,
    help="Base directory to save models",
)
def main(config_dir, config_paths, dataset_names, output_dir):
    """Generate and train models based on configuration files."""

    logger.info("Starting sequential model generation from config files")

    if config_paths:
        config_files = list(config_paths)
    else:
        # Find all YAML config files in the config directory
        config_files = glob.glob(
            os.path.join(config_dir, "*.yaml")
        ) + glob.glob(os.path.join(config_dir, "*.yml"))

    if not config_files:
        logger.error("No config files selected")
        return

    config_files.sort()
    config_names = [os.path.basename(f) for f in config_files]
    logger.info(f"Found {len(config_files)} config file(s): {config_names}")

    # Process each config file
    for config_file in config_files:
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        log_section(logger, f"Processing configuration: {config_name}")

        with Timer(logger, f"Config '{config_name}' completed"):
            # Load configuration from YAML file
            config = load_config(config_file)
            if config is None:
                logger.error(f"Failed to load config from {config_file}")
                continue

            # Extract parameters from config
            num_models = config.get("num_models", 1000)
            epochs = config.get("epochs", 1)
            min_layers = config.get("min_layers", 5)
            max_layers = config.get("max_layers", 25)
            min_neurons = config.get("min_neurons", 10)
            max_neurons = config.get("max_neurons", 50)
            learning_rate = config.get("learning_rate", 0.001)
            random_seed = config.get("random_seed", 42)

            logger.info(
                f"Config: models={num_models}, epochs={epochs}, "
                f"layers=[{min_layers},{max_layers}], "
                f"neurons=[{min_neurons},{max_neurons}], lr={learning_rate}, "
                f"seed={random_seed}"
            )

            # Create current configuration for comparison
            current_config = {
                "min_layers": min_layers,
                "max_layers": max_layers,
                "min_neurons": min_neurons,
                "max_neurons": max_neurons,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "random_seed": random_seed,
            }

            available_datasets = {
                "mnist": MLDataset.MNIST,
                "fmnist": MLDataset.FMNIST,
            }
            selected_dataset_names = (
                [name.lower() for name in dataset_names]
                if dataset_names
                else ["mnist", "fmnist"]
            )
            datasets_to_train = [
                (available_datasets[name], name)
                for name in selected_dataset_names
            ]

            for dataset_index, (dataset, dataset_name) in enumerate(
                datasets_to_train
            ):
                log_subsection(logger, f"Processing {dataset_name.upper()}")

                # Use a stable, distinct seed per dataset. This controls Python's
                # architecture randomization and PyTorch's initialization and
                # DataLoader shuffling.
                dataset_seed = random_seed + dataset_index
                random.seed(dataset_seed)
                torch.manual_seed(dataset_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(dataset_seed)

                # Create config-specific output directory structure
                config_output_dir = os.path.join(output_dir, config_name)
                dataset_output_dir = os.path.join(
                    config_output_dir, dataset_name
                )
                config_path = os.path.join(dataset_output_dir, "config.json")

                # Ensure output directory exists
                os.makedirs(dataset_output_dir, exist_ok=True)

                # Load existing config
                existing_config = load_config(config_path)

                # Check if configuration has changed and handle it
                check_configuration(
                    dataset_name,
                    dataset_output_dir,
                    existing_config,
                    current_config,
                )

                # Save current configuration
                save_config(config_path, current_config)

                # Generate and train models with timing
                start_time = time.time()
                generated_models = generate_and_train_models(
                    dataset=dataset,
                    num_models=num_models,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    layer_bounds=(min_layers, max_layers),
                    neuron_bounds=(min_neurons, max_neurons),
                    output_dir=dataset_output_dir,
                )
                total_time = time.time() - start_time
                
                # Log total timing to outputs folder (base dir)
                output_base = os.path.dirname(output_dir.rstrip('/'))
                if not output_base:
                    output_base = "outputs"
                append_total_timing(
                    output_base, config_name, dataset_name,
                    generated_models, total_time
                )

    log_completion(logger, "Model generation completed for all configs")


if __name__ == "__main__":
    main()
