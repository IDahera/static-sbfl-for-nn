"""
Classifier Training Script

Trains classification models on suspiciousness prediction datasets.
"""

import click
import pandas as pd
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

from sbfl_nn.prediction.classifiers import MODELS

from scripts.utils import (
    get_logger,
    log_section,
    log_subsection,
    log_completion,
    find_subdirs,
    select_subdirs,
)

logger = get_logger(__name__)

# Metric Explanations
METRIC_EXPLANATIONS = """
=== CLASSIFICATION METRICS GUIDE ===
F1 Score:    Balance between precision and recall (higher is better)
Precision:   Of predicted suspicious neurons, how many were correct
Recall:      Of truly suspicious neurons, how many did we find
Accuracy:    Overall correctness of predictions

Suspiciousness Scores:
True Positive Mean:  Avg suspiciousness of correctly identified neurons
False Positive Mean: Avg suspiciousness of incorrectly flagged neurons
Prediction Gap:      Difference between TP and FP means (higher=better)
====================================
"""


def evaluate_classifier(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100,
    }


def train_and_evaluate(models, X_train, X_val, y_train, y_val, metric_names):
    """Train all models and return their scores for each target variable"""

    logger.info(METRIC_EXPLANATIONS)

    results = {}
    interpretation_models = {}

    for target_idx, metric_name in enumerate(metric_names):
        log_section(logger, f"Training models for {metric_name}")

        # Get continuous labels for this target
        y_train_continuous = y_train[:, target_idx]
        y_val_continuous = y_val[:, target_idx]

        # Convert to binary classification using threshold (top 20%)
        threshold = np.percentile(y_train_continuous, 80)
        y_train_binary = (y_train_continuous >= threshold).astype(int)
        y_val_binary = (y_val_continuous >= threshold).astype(int)

        logger.info(f"Threshold: {threshold:.6f}")
        logger.info(
            f"Train: {np.sum(y_train_binary)}/{len(y_train_binary)} "
            f"| Val: {np.sum(y_val_binary)}/{len(y_val_binary)} suspicious"
        )

        results[metric_name] = {}

        for name, model in models.items():
            logger.info(f"[{name}]")

            # Time the training
            train_start = time.time()
            model.fit(X_train, y_train_binary)
            train_time = time.time() - train_start

            if metric_name == "Tarantula":
                # RQ3 interprets the Tarantula classifiers. These are the
                # models persisted after both RQ1/RQ2 targets are evaluated.
                interpretation_models[name] = model

            # Make predictions
            val_start = time.time()
            y_pred = model.predict(X_val)
            val_time = time.time() - val_start

            # Calculate standard classification metrics
            metrics = evaluate_classifier(y_val_binary, y_pred)

            # Add timing information
            metrics["train_time"] = train_time
            metrics["validation_time"] = val_time

            # Calculate mean suspiciousness values for predicted classes
            suspicious_mask = y_pred == 1
            not_suspicious_mask = y_pred == 0

            if np.sum(suspicious_mask) > 0:
                true_positive_mean = np.mean(y_val_continuous[suspicious_mask])
                metrics["true_positive_mean"] = round(true_positive_mean, 3)
            else:
                metrics["true_positive_mean"] = None

            if np.sum(not_suspicious_mask) > 0:
                false_positive_mean = np.mean(
                    y_val_continuous[not_suspicious_mask]
                )
                metrics["false_positive_mean"] = round(false_positive_mean, 3)
            else:
                metrics["false_positive_mean"] = None

            # Calculate prediction gap
            tp_mean = metrics["true_positive_mean"]
            fp_mean = metrics["false_positive_mean"]
            if tp_mean is not None and fp_mean is not None:
                metrics["prediction_gap"] = round(tp_mean - fp_mean, 3)
            else:
                metrics["prediction_gap"] = None

            # Store all metrics
            results[metric_name][name] = metrics

            # Log results in clean format
            logger.info(
                f"  F1: {metrics['f1_score']:.1f}% | "
                f"P: {metrics['precision']:.1f}% | "
                f"R: {metrics['recall']:.1f}% | "
                f"Acc: {metrics['accuracy']:.1f}%"
            )

            if metrics["true_positive_mean"] is not None:
                logger.info(f"  TP Mean: {metrics['true_positive_mean']:.6f}")
            if metrics["false_positive_mean"] is not None:
                logger.info(f"  FP Mean: {metrics['false_positive_mean']:.6f}")
            if metrics["prediction_gap"] is not None:
                logger.info(f"  Gap: {metrics['prediction_gap']:.6f}")

            logger.info(f"  Train: {train_time:.3f}s | Val: {val_time:.3f}s")

    return results, interpretation_models


def save_models(models_dict, save_dir="saved_models"):
    """Save all trained models with classifier_model prefix"""
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}

    for name, model in models_dict.items():
        # Save model with classifier_model prefix
        filename = f"classifier_model_{name.replace(' ', '_').lower()}.pkl"
        filepath = os.path.join(save_dir, filename)
        joblib.dump(model, filepath)
        saved_paths[name] = filepath
        logger.info(f"Saved {name} to {filepath}")

    return saved_paths


def save_results_to_csv(results, save_dir="models_sus_prediction"):
    """Save combined results to a single CSV file"""
    os.makedirs(save_dir, exist_ok=True)

    # Get all unique models
    all_models = set()
    for target_name, target_results in results.items():
        all_models.update(target_results.keys())
    all_models = sorted(all_models)

    # Define the order of metrics we want to display
    metric_order = [
        "F1_Score",
        "Precision",
        "Recall",
        "Accuracy",
        "True_Positive_Mean",
        "False_Positive_Mean",
        "Prediction_Gap",
        "Train_Time_s",
        "Validation_Time_s",
    ]

    # Create combined CSV with both Ochiai and Tarantula
    combined_data = []

    for metric_key in metric_order:
        row = {"Metric": metric_key}

        # Add columns for each target and model combination
        for target_name in sorted(results.keys()):
            target_results = results[target_name]

            for model_name in all_models:
                column_name = f"{target_name}_{model_name}"

                if model_name in target_results:
                    metrics = target_results[model_name]

                    # Map internal metric names to display names
                    value = None
                    if metric_key == "F1_Score":
                        value = round(metrics.get("f1_score", 0), 2)
                    elif metric_key == "Precision":
                        value = round(metrics.get("precision", 0), 2)
                    elif metric_key == "Recall":
                        value = round(metrics.get("recall", 0), 2)
                    elif metric_key == "Accuracy":
                        value = round(metrics.get("accuracy", 0), 2)
                    elif metric_key == "True_Positive_Mean":
                        value = round(
                            metrics.get("true_positive_mean", 0) or 0, 6
                        )
                    elif metric_key == "False_Positive_Mean":
                        value = round(
                            metrics.get("false_positive_mean", 0) or 0, 6
                        )
                    elif metric_key == "Prediction_Gap":
                        value = round(metrics.get("prediction_gap", 0) or 0, 6)
                    elif metric_key == "Train_Time_s":
                        value = round(metrics.get("train_time", 0), 3)
                    elif metric_key == "Validation_Time_s":
                        value = round(metrics.get("validation_time", 0), 3)

                    row[column_name] = value
                else:
                    row[column_name] = "N/A"

        combined_data.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(combined_data)

    csv_filename = os.path.join(save_dir, "classification_results.csv")
    df.to_csv(csv_filename, index=False)

    logger.info(f"Results saved to: {csv_filename}")

    return csv_filename


@click.command()
@click.option(
    "--dataset-dir",
    default="outputs/datasets",
    help="Base directory containing dataset files",
)
@click.option(
    "--output-dir",
    default="outputs/results",
    help="Base directory to save results",
)
@click.option(
    "--train-split",
    default=0.8,
    type=float,
    help="Fraction of data for training (default: 0.8)",
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    help="Random state for reproducibility",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="Maximum number of neuron rows to use (not neural-network models).",
)
@click.option(
    "--config",
    "config_names",
    multiple=True,
    help="Process only this configuration; repeat for multiple configurations.",
)
def main(
    dataset_dir,
    output_dir,
    train_split,
    random_state,
    max_samples,
    config_names,
):
    """Train classification models on suspiciousness prediction dataset."""

    # Find all config subdirectories
    config_dirs = select_subdirs(dataset_dir, config_names)
    if not config_dirs:
        logger.error(f"No config subdirectories found in {dataset_dir}")
        return

    logger.info(f"Found {len(config_dirs)} config(s): {config_dirs}")

    # Process each config directory
    for config_name in config_dirs:
        config_dataset_path = os.path.join(dataset_dir, config_name)
        log_section(logger, f"Processing config: {config_name}")

        # Find dataset subdirectories (mnist, fmnist, etc.)
        dataset_subdirs = find_subdirs(config_dataset_path, required=False)

        if not dataset_subdirs:
            logger.warning(f"No dataset subdirs in {config_dataset_path}")
            continue

        # Process each dataset
        for dataset_name in dataset_subdirs:
            dataset_path = os.path.join(config_dataset_path, dataset_name)
            log_subsection(logger, f"Processing dataset: {dataset_name}")

            # Look for numpy dataset file
            data_file = os.path.join(
                dataset_path, f"{dataset_name}_dataset_numpy.pkl"
            )

            if not os.path.exists(data_file):
                logger.warning(f"Dataset file not found: {data_file}")
                continue

            # Load data
            logger.info(f"Loading data from {data_file}")
            data = pd.read_pickle(data_file)
            X = data["features"]
            y = data["targets"]

            # Check for and handle NaN values
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values, replacing")
                X = np.nan_to_num(X, nan=0.0)

            # Each row represents one neuron; model boundaries are not retained
            # by the transformed NumPy dataset.
            if max_samples is not None and max_samples > 0:
                logger.info(f"Limiting to first {max_samples} neuron rows")
                X = X[:max_samples]
                y = y[:max_samples]

            logger.info(f"Total samples: {len(X)}")

            # Split the data into training and validation
            val_size = 1.0 - train_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=random_state
            )

            logger.info(
                f"Train: {len(X_train)} ({train_split*100:.0f}%) | "
                f"Val: {len(X_val)} ({val_size*100:.0f}%)"
            )

            # Use Ochiai (column 4) and Tarantula (column 5) as targets
            target_names = ["Ochiai", "Tarantula"]
            target_indices = [4, 5]

            # Extract only the suspiciousness columns
            y_train_suspicious = y_train[:, target_indices]
            y_val_suspicious = y_val[:, target_indices]

            # Train and evaluate models
            results, trained_models = train_and_evaluate(
                MODELS,
                X_train,
                X_val,
                y_train_suspicious,
                y_val_suspicious,
                target_names,
            )

            # Create output directory for this config and dataset
            config_output_dir = os.path.join(
                output_dir, config_name, dataset_name
            )
            os.makedirs(config_output_dir, exist_ok=True)

            logger.info(f"Saving models to {config_output_dir}...")
            save_models(trained_models, save_dir=config_output_dir)

            # Save results to CSV
            logger.info("Saving results...")
            save_results_to_csv(results, save_dir=config_output_dir)

            # Print summary
            log_section(logger, f"SUMMARY ({config_name}/{dataset_name})")

            # Find best model for each target
            for target_name in target_names:
                best_model = max(
                    results[target_name].keys(),
                    key=lambda m: results[target_name][m]["f1_score"],
                )
                best_metrics = results[target_name][best_model]
                logger.info(f"{target_name}: Best={best_model}")
                logger.info(
                    f"  F1: {best_metrics['f1_score']:.1f}% | "
                    f"P: {best_metrics['precision']:.1f}% | "
                    f"R: {best_metrics['recall']:.1f}%"
                )

                if best_metrics.get("true_positive_mean") is not None:
                    tp = best_metrics["true_positive_mean"]
                    logger.info(f"  TP Mean: {tp:.6f}")
                if best_metrics.get("false_positive_mean") is not None:
                    fp = best_metrics["false_positive_mean"]
                    logger.info(f"  FP Mean: {fp:.6f}")
                if best_metrics.get("prediction_gap") is not None:
                    logger.info(f"  Gap: {best_metrics['prediction_gap']:.6f}")

    log_completion(logger, "Classification training completed")


if __name__ == "__main__":
    main()
