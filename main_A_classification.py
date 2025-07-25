import click
import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import joblib

from helpers_prediction.models_classification import MODELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Define classifiers
CLASSIFIERS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

def evaluate_classifier(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn = tp = fp = fn = 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }

def train_and_evaluate(models, X_train, X_test, y_train, y_test, metric_names):
    """Train all models and return their scores for each target variable"""
    
    # DEBUG: Check data structure
    logging.info("DEBUG - Data structure analysis:")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    logging.info(f"y_train dtype: {y_train.dtype}")
    logging.info(f"y_test dtype: {y_test.dtype}")
    
    # Check the actual values in each column
    for i in range(y_train.shape[1] if len(y_train.shape) > 1 else 1):
        col_train = y_train[:, i] if len(y_train.shape) > 1 else y_train
        col_test = y_test[:, i] if len(y_test.shape) > 1 else y_test
        logging.info(f"Column {i} (should be {metric_names[i] if i < len(metric_names) else 'unknown'}):")
        logging.info(f"  Train - min: {np.min(col_train):.6f}, max: {np.max(col_train):.6f}, mean: {np.mean(col_train):.6f}")
        logging.info(f"  Test - min: {np.min(col_test):.6f}, max: {np.max(col_test):.6f}, mean: {np.mean(col_test):.6f}")
        
    results = {}
    
    for target_idx, metric_name in enumerate(metric_names):
        logging.info(f"\n{'='*60}")
        logging.info(f"Training models for {metric_name}")
        logging.info(f"{'='*60}")
        
        print(f"target_idx: {target_idx}, metric_name: {metric_name}")

        # Get continuous labels for this target (original suspiciousness scores)
        y_train_continuous = y_train[:, target_idx]
        y_test_continuous = y_test[:, target_idx]
        
        # Convert to binary classification using threshold (top 20%)
        threshold = np.percentile(y_train_continuous, 80)
        y_train_binary = (y_train_continuous >= threshold).astype(int)
        y_test_binary = (y_test_continuous >= threshold).astype(int)
        
        logging.info(f"Threshold: {threshold:.6f}")
        logging.info(f"Class distribution - Train: {np.sum(y_train_binary)}/{len(y_train_binary)} positive")
        logging.info(f"Class distribution - Test: {np.sum(y_test_binary)}/{len(y_test_binary)} positive")
        
        results[metric_name] = {}
        trained_models = {}
        
        for name, model in models.items():
            logging.info(f"\n--- {name} ---")
            
            # Time the training
            import time
            train_start = time.time()
            model.fit(X_train, y_train_binary)
            train_end = time.time()
            train_time = train_end - train_start
            
            trained_models[name] = model
            
            # Time the testing
            test_start = time.time()
            y_pred = model.predict(X_test)
            test_end = time.time()
            test_time = test_end - test_start
            
            # Calculate standard classification metrics
            metrics = evaluate_classifier(y_test_binary, y_pred)
            
            # Add timing and sample information
            metrics['train_time'] = train_time
            metrics['test_time'] = test_time
            metrics['n_train_samples'] = len(X_train)
            metrics['n_test_samples'] = len(X_test)
            
            # Calculate mean suspiciousness values for predicted classes
            # Entries predicted as suspicious (y_pred == 1)
            suspicious_mask = y_pred == 1
            # Entries predicted as not suspicious (y_pred == 0)
            not_suspicious_mask = y_pred == 0
            
            if np.sum(suspicious_mask) > 0:
                mean_suspicious = np.mean(y_test_continuous[suspicious_mask])
                metrics['mean_suspiciousness_predicted_suspicious'] = mean_suspicious
                logging.info(f"  Mean suspiciousness of predicted suspicious entries: {mean_suspicious:.6f}")
            else:
                metrics['mean_suspiciousness_predicted_suspicious'] = None
                logging.info(f"  No entries predicted as suspicious")
            
            if np.sum(not_suspicious_mask) > 0:
                mean_not_suspicious = np.mean(y_test_continuous[not_suspicious_mask])
                metrics['mean_suspiciousness_predicted_not_suspicious'] = mean_not_suspicious
                logging.info(f"  Mean suspiciousness of predicted not suspicious entries: {mean_not_suspicious:.6f}")
            else:
                metrics['mean_suspiciousness_predicted_not_suspicious'] = None
                logging.info(f"  No entries predicted as not suspicious")
            
            # Additional useful statistics
            if np.sum(suspicious_mask) > 0 and np.sum(not_suspicious_mask) > 0:
                difference = mean_suspicious - mean_not_suspicious
                metrics['suspiciousness_difference'] = difference
                logging.info(f"  Difference in mean suspiciousness: {difference:.6f}")
            
            # Store all metrics
            results[metric_name][name] = metrics
            
            # Log standard results
            logging.info(f"  F1: {metrics['f1_score']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%")
            logging.info(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, TN: {metrics['TN']}, FN: {metrics['FN']}")
            logging.info(f"  Train time: {train_time:.4f}s, Test time: {test_time:.4f}s")
    
    return results, trained_models

def save_models(models_dict, save_dir='saved_models'):
    """Save all trained models"""
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}
    
    for name, model in models_dict.items():
        # Save model
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        filepath = os.path.join(save_dir, filename)
        joblib.dump(model, filepath)
        saved_paths[name] = filepath
        logging.info(f"Saved {name} to {filepath}")
    
    return saved_paths


def save_results_to_csv(results, save_dir='models_sus_prediction'):
    """Save detailed results to CSV file with transposed format (models as columns)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all unique models and metrics
    all_models = set()
    all_metrics = set()
    
    for target_name, target_results in results.items():
        for model_name, metrics in target_results.items():
            all_models.add(model_name)
            all_metrics.update(metrics.keys())
    
    all_models = sorted(all_models)
    
    # Define the order of metrics we want to display
    metric_order = [
        'F1_Score', 'Precision', 'Recall', 'Accuracy',
        'TP', 'FP', 'TN', 'FN',
        'Train_Time_s', 'Test_Time_s', 
        'N_Train_Samples', 'N_Test_Samples',
        'Mean_Suspiciousness_Predicted_Suspicious',
        'Mean_Suspiciousness_Predicted_Not_Suspicious',
        'Suspiciousness_Difference'
    ]
    
    # Create separate CSV for each target metric
    for target_name in results.keys():
        target_results = results[target_name]
        
        # Prepare transposed data - metrics as rows, models as columns
        transposed_data = []
        
        for metric_key in metric_order:
            row = {'Metric': metric_key}
            
            for model_name in all_models:
                if model_name in target_results:
                    metrics = target_results[model_name]
                    
                    # Map internal metric names to display names
                    value = None
                    if metric_key == 'F1_Score':
                        value = round(metrics.get('f1_score', 0), 4)
                    elif metric_key == 'Precision':
                        value = round(metrics.get('precision', 0), 4)
                    elif metric_key == 'Recall':
                        value = round(metrics.get('recall', 0), 4)
                    elif metric_key == 'Accuracy':
                        value = round(metrics.get('accuracy', 0), 4)
                    elif metric_key == 'TP':
                        value = metrics.get('TP', 0)
                    elif metric_key == 'FP':
                        value = metrics.get('FP', 0)
                    elif metric_key == 'TN':
                        value = metrics.get('TN', 0)
                    elif metric_key == 'FN':
                        value = metrics.get('FN', 0)
                    elif metric_key == 'Train_Time_s':
                        value = round(metrics.get('train_time', 0), 4)
                    elif metric_key == 'Test_Time_s':
                        value = round(metrics.get('test_time', 0), 4)
                    elif metric_key == 'N_Train_Samples':
                        value = metrics.get('n_train_samples', 0)
                    elif metric_key == 'N_Test_Samples':
                        value = metrics.get('n_test_samples', 0)
                    elif metric_key == 'Mean_Suspiciousness_Predicted_Suspicious':
                        value = round(metrics.get('mean_suspiciousness_predicted_suspicious', 0) or 0, 6)
                    elif metric_key == 'Mean_Suspiciousness_Predicted_Not_Suspicious':
                        value = round(metrics.get('mean_suspiciousness_predicted_not_suspicious', 0) or 0, 6)
                    elif metric_key == 'Suspiciousness_Difference':
                        value = round(metrics.get('suspiciousness_difference', 0) or 0, 6)
                    
                    row[model_name] = value
                else:
                    row[model_name] = 'N/A'
            
            transposed_data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(transposed_data)
        
        csv_filename = os.path.join(save_dir, f'classification_results_{target_name.lower()}.csv')
        df.to_csv(csv_filename, index=False)
        
        logging.info(f"Results for {target_name} saved to CSV: {csv_filename}")
    
    # Create a combined summary CSV with best models per target (transposed format)
    summary_data = []
    
    # Add metadata row
    metadata_row = {'Metric': 'Target_Type'}
    for target_name in results.keys():
        best_model = max(results[target_name].keys(), 
                        key=lambda model: results[target_name][model]['f1_score'])
        metadata_row[target_name] = f'Best: {best_model}'
    summary_data.append(metadata_row)
    
    # Add metric rows for best models
    for metric_key in metric_order:
        row = {'Metric': metric_key}
        
        for target_name in results.keys():
            best_model = max(results[target_name].keys(), 
                            key=lambda model: results[target_name][model]['f1_score'])
            best_metrics = results[target_name][best_model]
            
            # Map internal metric names to display names (same logic as above)
            value = None
            if metric_key == 'F1_Score':
                value = round(best_metrics.get('f1_score', 0), 4)
            elif metric_key == 'Precision':
                value = round(best_metrics.get('precision', 0), 4)
            elif metric_key == 'Recall':
                value = round(best_metrics.get('recall', 0), 4)
            elif metric_key == 'Accuracy':
                value = round(best_metrics.get('accuracy', 0), 4)
            elif metric_key == 'TP':
                value = best_metrics.get('TP', 0)
            elif metric_key == 'FP':
                value = best_metrics.get('FP', 0)
            elif metric_key == 'TN':
                value = best_metrics.get('TN', 0)
            elif metric_key == 'FN':
                value = best_metrics.get('FN', 0)
            elif metric_key == 'Train_Time_s':
                value = round(best_metrics.get('train_time', 0), 4)
            elif metric_key == 'Test_Time_s':
                value = round(best_metrics.get('test_time', 0), 4)
            elif metric_key == 'N_Train_Samples':
                value = best_metrics.get('n_train_samples', 0)
            elif metric_key == 'N_Test_Samples':
                value = best_metrics.get('n_test_samples', 0)
            elif metric_key == 'Mean_Suspiciousness_Predicted_Suspicious':
                value = round(best_metrics.get('mean_suspiciousness_predicted_suspicious', 0) or 0, 6)
            elif metric_key == 'Mean_Suspiciousness_Predicted_Not_Suspicious':
                value = round(best_metrics.get('mean_suspiciousness_predicted_not_suspicious', 0) or 0, 6)
            elif metric_key == 'Suspiciousness_Difference':
                value = round(best_metrics.get('suspiciousness_difference', 0) or 0, 6)
            
            row[target_name] = value
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_filename = os.path.join(save_dir, 'classification_summary_transposed.csv')
    summary_df.to_csv(summary_csv_filename, index=False)
    
    logging.info(f"Summary saved to CSV: {summary_csv_filename}")
    
    return summary_csv_filename


@click.command()
@click.option('--data-path', default="models_image_classification/mnist/dataset/mnist_dataset_numpy.pkl",
              help='Path to the numpy dataset file')
@click.option('--test-size', default=0.2, type=float,
              help='Test set size as a fraction (default: 0.2)')
@click.option('--random-state', default=42, type=int,
              help='Random state for reproducibility (default: 42)')
@click.option('--threshold-percentile', default=80, type=int,
              help='Percentile threshold for binary classification (default: 80)')
@click.option('--max-models', default=100, type=int,
              help='Maximum number of models to use for training and testing (default: None, use all models)')
def main(data_path, test_size, random_state, threshold_percentile, max_models):
    """Train classification models on suspiciousness prediction dataset."""
    
    # Load data
    logging.info(f"Loading numpy data from {data_path}")
    data = pd.read_pickle(data_path)
    X = data['features']
    y = data['targets']
    
    # Limit the number of models if specified
    if max_models is not None and max_models > 0:
        logging.info(f"Limiting dataset to first {max_models} models")
        X = X[:max_models]
        y = y[:max_models]
        logging.info(f"Dataset size after limiting: {len(X)} samples")
    else:
        logging.info(f"Using full dataset: {len(X)} samples")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logging.info(f"Feature shape: {X_train.shape}, Target shape: {y_train.shape}")
    
    # DEBUG: Check data structure
    logging.info("DEBUG - Data structure analysis:")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"y_test shape: {y_test.shape}")
    
    # Check the actual values in each column
    for i in range(y_train.shape[1]):
        col_train = y_train[:, i]
        col_test = y_test[:, i]
        logging.info(f"Column {i}:")
        logging.info(f"  Train - min: {np.min(col_train):.6f}, max: {np.max(col_train):.6f}, mean: {np.mean(col_train):.6f}")
        logging.info(f"  Test - min: {np.min(col_test):.6f}, max: {np.max(col_test):.6f}, mean: {np.mean(col_test):.6f}")
    
    # FIXED: Use correct target indices for Ochiai (column 4) and Tarantula (column 5)
    target_names = ['Ochiai', 'Tarantula']
    target_indices = [4, 5]  # Correct indices based on the debug output
    
    # Extract only the suspiciousness columns
    y_train_suspicious = y_train[:, target_indices]
    y_test_suspicious = y_test[:, target_indices]
    
    logging.info(f"Using suspiciousness targets with shape: {y_train_suspicious.shape}")
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate(MODELS, X_train, X_test, 
                               y_train_suspicious, y_test_suspicious, target_names)
    
    logging.info("Saving trained models...")
    save_models(trained_models, save_dir='models_sus_prediction')
    logging.info("Models saved successfully.")
    
    # Save results to CSV
    logging.info("Saving results to CSV...")
    summary_csv_file = save_results_to_csv(results, save_dir='models_sus_prediction')
    logging.info("CSV files saved successfully.")
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("SUMMARY - Best F1 Scores by Target")
    logging.info("="*80)
    
    # Find best model for each target
    for target_name in target_names:
        best_model = max(results[target_name].keys(), 
                        key=lambda model: results[target_name][model]['f1_score'])
        best_metrics = results[target_name][best_model]
        logging.info(f"\n{target_name}:")
        logging.info(f"  Best Model: {best_model}")
        logging.info(f"  F1: {best_metrics['f1_score']:.2f}%, "
                    f"Precision: {best_metrics['precision']:.2f}%, "
                    f"Recall: {best_metrics['recall']:.2f}%")
        
        # Add suspiciousness analysis to summary
        if best_metrics.get('mean_suspiciousness_predicted_suspicious') is not None:
            logging.info(f"  Mean suspiciousness (predicted suspicious): {best_metrics['mean_suspiciousness_predicted_suspicious']:.6f}")
        if best_metrics.get('mean_suspiciousness_predicted_not_suspicious') is not None:
            logging.info(f"  Mean suspiciousness (predicted not suspicious): {best_metrics['mean_suspiciousness_predicted_not_suspicious']:.6f}")
        if best_metrics.get('suspiciousness_difference') is not None:
            logging.info(f"  Difference: {best_metrics['suspiciousness_difference']:.6f}")

if __name__ == "__main__":
    main()