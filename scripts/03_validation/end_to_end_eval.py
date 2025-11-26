"""
End-to-End Validation

This script loads a selected model and creates a stack of metadata for
validation purposes. Based on the suspiciousness values transformation
pipeline, but focused on metadata extraction without creating full dataset.
"""

import pickle
import torch
import click
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from sbfl_nn.data.dataset_generation import (
    DatasetEntries,
    SusPredictionDataset,
)
from sbfl_nn.utils.gpu import DEVICE

from scripts.utils import get_logger

logger = get_logger(__name__)


def load_sus_values(sus_file):
    """Load suspiciousness values for a given file."""
    with open(sus_file, "rb") as f:
        sus_values = pickle.load(f)
    return sus_values


def load_model(model_file):
    """Load model corresponding to the given model file."""
    with open(model_file, "rb") as f:
        model = torch.load(f, map_location=DEVICE, weights_only=False)
    return model


def create_metadata_stack(model_file, sus_file):
    """Create metadata stack for a model and its suspiciousness values.

    Processes a single model and its corresponding suspiciousness values
    to create a metadata stack containing layer information and features.

    Parameters:
    -----------
    model_file : str
        Path to the model file (.pth)
    sus_file : str
        Path to the suspiciousness values file (.pkl)

    Returns:
    --------
    tuple[DatasetEntries, float]
        Stack with metadata/features for all layers, and creation time
    """
    # Start timing stack creation
    start_time = time.time()

    # Initialize empty stacks for metadata
    stacks = DatasetEntries()

    # Load model and suspiciousness values
    model = load_model(model_file)
    sus_values = load_sus_values(sus_file)

    logger.info(f"Processing model: {model_file}")
    logger.info(f"Suspiciousness file: {sus_file}")
    logger.info(f"Found {len(sus_values)} layers")

    # Process each layer
    for layer_name in tqdm(sus_values.keys(), desc="Processing layers"):
        layer_idx = torch.tensor(list(sus_values.keys()).index(layer_name)).to(
            DEVICE
        )

        # Process layer data and update stacks
        stacks.process_layer_suspiciousness(
            model, sus_values, layer_name, layer_idx
        )

    # Calculate stack creation time
    stack_time = time.time() - start_time

    return stacks, stack_time


def load_trained_model(model_path):
    """Load a pre-trained model for predictions."""
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded trained model: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def make_predictions(model, dataset, metric_name="Tarantula"):
    """Make predictions using the trained model on the dataset.

    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    dataset : SusPredictionDataset
        Dataset to make predictions on
    metric_name : str
        Name of the suspiciousness metric (for target selection)

    Returns:
    --------
    dict : Dictionary containing predictions and analysis
    """
    start_time = time.time()

    # Get features and targets from dataset
    features, targets = dataset.get_numpy()

    # Select target column based on metric name (Ochiai=4, Tarantula=5)
    target_idx = 4 if metric_name.lower() == "ochiai" else 5
    continuous_targets = targets[:, target_idx]

    # Create binary targets using same threshold logic as training (top 20%)
    threshold = np.percentile(continuous_targets, 80)
    binary_targets = (continuous_targets >= threshold).astype(int)

    # Make predictions
    predictions = model.predict(features)
    prediction_probabilities = (
        model.predict_proba(features)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    prediction_time = time.time() - start_time

    # Calculate metrics
    correct_predictions = np.sum(predictions == binary_targets)
    accuracy = correct_predictions / len(predictions) * 100

    # Analyze suspiciousness values
    suspicious_mask = predictions == 1
    not_suspicious_mask = predictions == 0

    analysis = {
        "total_samples": len(features),
        "predicted_suspicious": np.sum(suspicious_mask),
        "predicted_not_suspicious": np.sum(not_suspicious_mask),
        "accuracy": accuracy,
        "threshold_used": threshold,
        "prediction_time": prediction_time,
        "predictions": predictions,
        "probabilities": prediction_probabilities,
        "continuous_targets": continuous_targets,
        "binary_targets": binary_targets,
        "ground_truth_binary": binary_targets,  # Add ground truth for comparison
    }

    if np.sum(suspicious_mask) > 0:
        analysis["mean_suspiciousness_predicted_suspicious"] = np.mean(
            continuous_targets[suspicious_mask]
        )

    if np.sum(not_suspicious_mask) > 0:
        analysis["mean_suspiciousness_predicted_not_suspicious"] = np.mean(
            continuous_targets[not_suspicious_mask]
        )

    return analysis


def create_architecture_visualization(
    model,
    dataset,
    prediction_results,
    save_path="model_suspiciousness_heatmap.png",
):
    """Create a visualization of the model architecture with suspicious neurons highlighted.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model
    dataset : SusPredictionDataset
        Dataset containing neuron information
    prediction_results : dict
        Results from make_predictions function
    save_path : str
        Path to save the visualization
    """
    logger.info("Creating architecture visualization...")

    # Get model layers and their sizes
    layer_sizes = []
    layer_names = []

    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(module.weight.shape) >= 2:
            if "Linear" in str(type(module)) or "Conv" in str(type(module)):
                if "Linear" in str(type(module)):
                    layer_sizes.append(
                        module.weight.shape[0]
                    )  # Output features
                    layer_names.append(
                        f"{name}\n({module.weight.shape[0]} neurons)"
                    )
                elif "Conv" in str(type(module)):
                    # For conv layers, use output channels
                    layer_sizes.append(module.weight.shape[0])
                    layer_names.append(
                        f"{name}\n({module.weight.shape[0]} channels)"
                    )

    if not layer_sizes:
        logger.warning("No suitable layers found for visualization")
        return

    # Get predictions and ground truth for visualization
    predictions = prediction_results["predictions"]
    ground_truth = prediction_results["ground_truth_binary"]

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(24, len(layer_sizes) * 4), 8)
    )

    # Calculate positions for layers
    layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
    max_neurons = max(layer_sizes)

    # Color map: red for suspicious (1), green for not suspicious (0)
    colors = ["green", "red"]

    # Create both visualizations
    for ax, data, title_suffix in [
        (ax1, predictions, "Model Predictions"),
        (ax2, ground_truth, "Ground Truth"),
    ]:

        # Track neuron index across all layers
        neuron_idx = 0

        for i, (layer_size, layer_pos) in enumerate(
            zip(layer_sizes, layer_positions)
        ):
            # Calculate neuron positions for this layer
            if layer_size <= max_neurons:
                neuron_positions = np.linspace(0.1, 0.9, layer_size)
            else:
                # If too many neurons, sample some for display
                display_neurons = min(
                    layer_size, 50
                )  # Limit display to 50 neurons max
                neuron_positions = np.linspace(0.1, 0.9, display_neurons)
                sample_indices = np.linspace(
                    0, layer_size - 1, display_neurons, dtype=int
                )

            # Draw neurons
            for j, neuron_pos in enumerate(neuron_positions):
                if layer_size > 50:
                    actual_neuron_idx = neuron_idx + sample_indices[j]
                else:
                    actual_neuron_idx = neuron_idx + j

                # Get prediction for this neuron (if available)
                if actual_neuron_idx < len(data):
                    color = colors[int(data[actual_neuron_idx])]
                    alpha = 0.8
                else:
                    color = "gray"
                    alpha = 0.3

                # Draw neuron as circle
                circle = patches.Circle(
                    (layer_pos, neuron_pos),
                    0.01,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(circle)

            # Add layer labels
            ax.text(
                layer_pos,
                0.05,
                layer_names[i],
                ha="center",
                va="center",
                fontsize=8,
                rotation=45,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                ),
            )

            # Update neuron index
            neuron_idx += layer_size

        # Connect layers with lines (simplified representation)
        for i in range(len(layer_positions) - 1):
            ax.plot(
                [layer_positions[i], layer_positions[i + 1]],
                [0.5, 0.5],
                "k--",
                alpha=0.3,
                linewidth=1,
            )

        # Set plot properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add subplot titles
        if title_suffix == "Model Predictions":
            predicted_suspicious = prediction_results["predicted_suspicious"]
            total_count = prediction_results["total_samples"]
            accuracy = prediction_results["accuracy"]
            ax.set_title(
                f"{title_suffix}\nRed: Suspicious ({predicted_suspicious}), Green: Not Suspicious ({total_count - predicted_suspicious})\nAccuracy: {accuracy:.2f}%",
                fontsize=10,
                pad=10,
            )
        else:
            ground_truth_suspicious = np.sum(ground_truth)
            total_count = len(ground_truth)
            ax.set_title(
                f"{title_suffix}\nRed: Suspicious ({ground_truth_suspicious}), Green: Not Suspicious ({total_count - ground_truth_suspicious})",
                fontsize=10,
                pad=10,
            )

    # Add main title
    plt.suptitle(
        "Neural Network Architecture - Tarantula Suspiciousness Visualization\nModel Predictions vs Ground Truth Comparison",
        fontsize=14,
        y=0.95,
    )

    # Add legend
    legend_elements = [
        patches.Patch(color="red", alpha=0.8, label="Suspicious Neurons"),
        patches.Patch(
            color="green", alpha=0.8, label="Not Suspicious Neurons"
        ),
        patches.Patch(color="gray", alpha=0.3, label="No Data Available"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Architecture visualization saved to: {save_path}")

    # Also create layer-wise comparison heatmaps
    create_layer_comparison_heatmap(
        layer_sizes,
        layer_names,
        predictions,
        ground_truth,
        save_path.replace(".png", "_layer_comparison.png"),
    )

    plt.close()


def create_layer_comparison_heatmap(
    layer_sizes, layer_names, predictions, ground_truth, save_path
):
    """Create a layer-wise comparison heatmap showing suspiciousness distribution for both predictions and ground truth."""

    # Calculate suspiciousness ratio per layer for both predictions and ground truth
    prediction_ratios = []
    ground_truth_ratios = []
    neuron_idx = 0

    for layer_size in layer_sizes:
        # Predictions
        layer_predictions = predictions[neuron_idx : neuron_idx + layer_size]
        if len(layer_predictions) > 0:
            prediction_ratio = np.mean(layer_predictions)
            prediction_ratios.append(prediction_ratio)
        else:
            prediction_ratios.append(0)

        # Ground truth
        layer_ground_truth = ground_truth[neuron_idx : neuron_idx + layer_size]
        if len(layer_ground_truth) > 0:
            ground_truth_ratio = np.mean(layer_ground_truth)
            ground_truth_ratios.append(ground_truth_ratio)
        else:
            ground_truth_ratios.append(0)

        neuron_idx += layer_size

    # Create comparison heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Create heatmap data
    prediction_data = np.array(prediction_ratios).reshape(1, -1)
    ground_truth_data = np.array(ground_truth_ratios).reshape(1, -1)

    # Plot predictions
    im1 = ax1.imshow(
        prediction_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(
        "Model Predictions - Suspiciousness Ratio by Layer",
        fontsize=12,
        pad=10,
    )

    # Add text annotations for predictions
    for i, ratio in enumerate(prediction_ratios):
        ax1.text(
            i,
            0,
            f"{ratio:.2%}",
            ha="center",
            va="center",
            color="white" if ratio > 0.5 else "black",
            fontweight="bold",
        )

    # Plot ground truth
    im2 = ax2.imshow(
        ground_truth_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1
    )
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(
        [name.split("\n")[0] for name in layer_names], rotation=45, ha="right"
    )
    ax2.set_yticks([])
    ax2.set_title(
        "Ground Truth - Suspiciousness Ratio by Layer", fontsize=12, pad=10
    )

    # Add text annotations for ground truth
    for i, ratio in enumerate(ground_truth_ratios):
        ax2.text(
            i,
            0,
            f"{ratio:.2%}",
            ha="center",
            va="center",
            color="white" if ratio > 0.5 else "black",
            fontweight="bold",
        )

    # Add colorbar
    cbar = plt.colorbar(im2, ax=[ax1, ax2], location="right", shrink=0.8)
    cbar.set_label("Suspiciousness Ratio", rotation=270, labelpad=20)

    plt.suptitle(
        "Layer-wise Suspiciousness Distribution Comparison\nModel Predictions vs Ground Truth",
        fontsize=14,
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Layer comparison heatmap saved to: {save_path}")
    plt.close()


def create_layer_heatmap(layer_sizes, layer_names, predictions, save_path):
    """Create a layer-wise heatmap showing suspiciousness distribution."""

    # Calculate suspiciousness ratio per layer
    layer_ratios = []
    neuron_idx = 0

    for layer_size in layer_sizes:
        layer_predictions = predictions[neuron_idx : neuron_idx + layer_size]
        if len(layer_predictions) > 0:
            suspicious_ratio = np.mean(layer_predictions)
            layer_ratios.append(suspicious_ratio)
        else:
            layer_ratios.append(0)
        neuron_idx += layer_size

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Create heatmap data
    heatmap_data = np.array(layer_ratios).reshape(1, -1)

    im = ax.imshow(
        heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1
    )

    # Set ticks and labels
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(
        [name.split("\n")[0] for name in layer_names], rotation=45, ha="right"
    )
    ax.set_yticks([])

    # Add text annotations
    for i, ratio in enumerate(layer_ratios):
        ax.text(
            i,
            0,
            f"{ratio:.2%}",
            ha="center",
            va="center",
            color="white" if ratio > 0.5 else "black",
            fontweight="bold",
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Suspiciousness Ratio", rotation=270, labelpad=20)

    plt.title("Layer-wise Suspiciousness Distribution", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Layer heatmap saved to: {save_path}")
    plt.close()


@click.command()
@click.option(
    "--model-path",
    "-m",
    default="outputs/models/config1/mnist/model_0.pth",
    type=click.Path(exists=True),
    help="Path to the neural network model file (.pth)",
)
@click.option(
    "--sus-path",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Path to suspiciousness values file (.pkl). Defaults to model_path with _sus.pkl suffix",
)
@click.option(
    "--predictor-model",
    "-p",
    default="outputs/results/config1/mnist/classifier_model_xgboost.pkl",
    type=click.Path(exists=True),
    help="Path to trained predictor model (.pkl)",
)
@click.option(
    "--metric",
    "-t",
    default="Tarantula",
    type=click.Choice(["Ochiai", "Tarantula"], case_sensitive=False),
    help="Suspiciousness metric to predict [default: Tarantula]",
)
@click.option(
    "--visualize",
    "--viz",
    is_flag=True,
    default=False,
    help="Create visualization of suspicious neurons",
)
@click.option(
    "--viz-output",
    default="model_suspiciousness_visualization.png",
    help="Output path for visualization [default: model_suspiciousness_visualization.png]",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging (debug level)",
)
def main(
    model_path, sus_path, predictor_model, metric, visualize, viz_output, verbose
):
    """Main function to load a selected model and create metadata stack."""

    # Infer suspiciousness path from model path if not provided
    if sus_path is None:
        # Models in outputs/models/, suspiciousness in outputs/suspiciousness/
        sus_path = model_path.replace(".pth", "_sus.pkl").replace(
            "outputs/models/", "outputs/suspiciousness/"
        )

    logger.info("E2E Validation - Metadata Stack Creation")
    logger.info("=" * 50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Suspiciousness: {sus_path}")
    if visualize:
        logger.info(f"Viz output: {viz_output}")

    try:
        # Create metadata stack for the selected model
        logger.info("Creating metadata stack...")
        metadata_stack, stack_time = create_metadata_stack(
            model_path, sus_path
        )
        logger.info(f"Stack created in {stack_time:.2f}s")

        # Convert stack to SusPredictionDataset
        logger.info("Transforming to SusPredictionDataset...")
        dataset_start_time = time.time()

        sus_dataset = SusPredictionDataset(metadata_stack)

        dataset_time = time.time() - dataset_start_time
        logger.info(f"Transformation done in {dataset_time:.2f}s")

        # Log final dataset information
        logger.info("SusPredictionDataset created!")
        logger.info(f"Dataset length: {len(sus_dataset)}")

        # Inspect stack contents
        if hasattr(metadata_stack, "layer_indices"):
            if metadata_stack.layer_indices is not None:
                n_layers = len(metadata_stack.layer_indices)
                logger.info(f"Layer indices count: {n_layers}")

        # Log timing summary
        total_time = stack_time + dataset_time
        logger.info("=" * 50)
        logger.info("TIMING SUMMARY:")
        logger.info(f"  Stack creation:   {stack_time:.2f}s")
        logger.info(f"  Transformation:   {dataset_time:.2f}s")
        logger.info(f"  Total:            {total_time:.2f}s")

        # Load trained predictor model and make predictions
        logger.info("=" * 50)
        logger.info("PREDICTION PHASE")
        logger.info("=" * 50)

        trained_model = load_trained_model(predictor_model)
        if trained_model is not None:
            logger.info(f"Predicting {metric} suspiciousness...")

            pred_results = make_predictions(trained_model, sus_dataset, metric)

            # Log prediction results
            pred_time = pred_results["prediction_time"]
            logger.info(f"Prediction done in {pred_time:.2f}s")
            logger.info(f"Samples: {pred_results['total_samples']}")
            logger.info(f"Suspicious: {pred_results['predicted_suspicious']}")
            not_sus = pred_results["predicted_not_suspicious"]
            logger.info(f"Not suspicious: {not_sus}")
            logger.info(f"Accuracy: {pred_results['accuracy']:.2f}%")
            logger.info(f"Threshold: {pred_results['threshold_used']:.6f}")

            key_sus = "mean_suspiciousness_predicted_suspicious"
            if key_sus in pred_results:
                val = pred_results[key_sus]
                logger.info(f"Mean sus (predicted sus): {val:.6f}")

            key_not = "mean_suspiciousness_predicted_not_suspicious"
            if key_not in pred_results:
                val = pred_results[key_not]
                logger.info(f"Mean sus (predicted not): {val:.6f}")

            # Create visualization if requested
            if visualize:
                logger.info("=" * 50)
                logger.info("VISUALIZATION PHASE")
                logger.info("=" * 50)

                original_model = load_model(model_path)
                create_architecture_visualization(
                    original_model, sus_dataset, pred_results, viz_output
                )

            # Final timing
            total_time += pred_time
            logger.info("=" * 50)
            logger.info("FINAL TIMING:")
            logger.info(f"  Stack:       {stack_time:.2f}s")
            logger.info(f"  Transform:   {dataset_time:.2f}s")
            logger.info(f"  Prediction:  {pred_time:.2f}s")
            logger.info(f"  Total:       {total_time:.2f}s")

            return sus_dataset, pred_results
        else:
            logger.warning("Skipping prediction - model load failed")
            return sus_dataset, None

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Check model and suspiciousness paths exist.")
        return None
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return None


if __name__ == "__main__":
    main()
