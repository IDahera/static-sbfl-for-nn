import click
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

from helpers_prediction.models_regression import IMPROVED_MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def train_and_evaluate(models,
                       X_train,
                       X_test,
                       y_train,
                       y_test):
    """
    Train all models and return their scores for each target variable
    """

    results = {}
    n_targets = y_train.shape[1] if len(y_train.shape) > 1 else 1
    target_names = [f'Target_{i}' for i in range(n_targets)]
    
    # If we know the target names, we can customize them
    if n_targets == 2:
        target_names = ['Ochiai', 'Tarantula']
    elif n_targets == 6:
        # Customize based on your actual 6 targets
        target_names = ['AS', 'AF', 'NS', 'NF', 'OCHIAI', 'TARANTULA']

    logging.info(f"Training models for {n_targets} target variables: {target_names}")

    for name, model in models.items():
        logging.info(f"\n--- Training {name} ---")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate scores for each target
        model_results = {}
        
        if n_targets == 1:
            # Single target
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[target_names[0]] = {
                'R²': r2,
                'MSE': mse,
                'MAE': mae
            }
            logging.info(f"  {target_names[0]}: R² = {r2:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")
            
        else:
            # Multiple targets
            for i, target_name in enumerate(target_names):
                r2 = r2_score(y_test[:, i], y_pred[:, i])
                mse = mean_squared_error(y_test[:, i], y_pred[:, i])
                mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
                
                model_results[target_name] = {
                    'R²': r2,
                    'MSE': mse,
                    'MAE': mae
                }
                logging.info(f"  {target_name}: R² = {r2:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")
        
        results[name] = model_results

    return results

FILE_OUTPUT_NUMPY = "models/mnist/dataset/mnist_dataset_numpy.pkl"

@click.command()
@click.option('--data-path', default="models/mnist/dataset/mnist_dataset_numpy.pkl",
              help='Path to the numpy dataset file')
@click.option('--test-size', default=0.2, type=float,
              help='Test set size as a fraction (default: 0.2)')
@click.option('--random-state', default=42, type=int,
              help='Random state for reproducibility (default: 42)')
@click.option('--verbose', is_flag=True, default=False,
              help='Enable verbose logging')
def main(data_path, test_size, random_state, verbose):
    """Train regression models on suspiciousness prediction dataset."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Loading numpy data from {data_path}")
    data = pd.read_pickle(data_path)
    X = data['features']
    y = data['targets']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logging.info(f"Feature shape: {X_train.shape}, Target shape: {y_train.shape}")

    # Train and evaluate models
    results = train_and_evaluate(IMPROVED_MODELS, X_train, X_test, y_train, y_test)
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY - Best R² Scores by Target")
    logging.info("="*80)
    
    # Find best model for each target
    if len(y_train.shape) > 1:
        n_targets = y_train.shape[1]
        target_names = [f'Target_{i}' for i in range(n_targets)]
        
        for target_name in target_names:
            best_model = max(results.keys(), key=lambda model: results[model][target_name]['R²'])
            best_score = results[best_model][target_name]['R²']
            logging.info(f"{target_name:12}: {best_model:20} (R² = {best_score:.4f})")
    else:
        target_name = 'Target_0'
        best_model = max(results.keys(), key=lambda model: results[model][target_name]['R²'])
        best_score = results[best_model][target_name]['R²']
        logging.info(f"{target_name:12}: {best_model:20} (R² = {best_score:.4f})")


if __name__ == "__main__":
    main()
