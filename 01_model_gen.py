"""
Sequential Model Generator

This script generates multiple neural network models with random architectures,
trains them on the MNIST dataset, and saves the trained models to disk.
"""
import random
import time
import os
import glob
import torch.optim as optim

from helpers_general.logging import logging
from helpers_general.gpu import DEVICE
from helpers_models.models import MLDataset, DenseModel, DenseModelHelper, load_dataset
from helpers_models.randomization import generate_rand_dense_seq
import click


def find_highest_model_index(output_dir: str) -> int:
    """Find the highest existing model index in the output directory."""
    if not os.path.exists(output_dir):
        logging.info(f"Output directory {output_dir} does not exist, starting from index 0")
        return -1
    
    # Look for model files with pattern model_*.pth
    model_files = glob.glob(os.path.join(output_dir, "model_*.pth"))
    
    if not model_files:
        logging.info(f"No existing models found in {output_dir}, starting from index 0")
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
        logging.info(f"No valid model files found in {output_dir}, starting from index 0")
        return -1
    
    highest_index = max(model_indices)
    logging.info(f"Found {len(model_indices)} existing models in {output_dir}")
    logging.info(f"Highest existing model index: {highest_index}")
    return highest_index


def generate_and_train_models(num_models: int,
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
    
    logging.info(f"Starting model generation from index {start_index} to {end_index - 1}")
    logging.info(f"Will generate {num_models} new models")
    
    # Load dataset once before the loop
    train_loader, test_loader = load_dataset(MLDataset.MNIST)

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
              default='models_image_classification/mnist', 
              help='Directory to save models')
def main(num_models,
         epochs,
         min_layers,
         max_layers,
         min_neurons,
         max_neurons,
         learning_rate,
         output_dir):
    """Generate and train multiple neural network models with random architectures."""

    logging.info("Starting sequential model generation")
    generate_and_train_models(num_models=num_models,
                              epochs=epochs,
                              learning_rate=learning_rate,
                              layer_bounds=(min_layers, max_layers),
                              neuron_bounds=(min_neurons, max_neurons),
                              output_dir=output_dir)
    logging.info("Sequential model generation completed")


if __name__ == "__main__":
    main()
