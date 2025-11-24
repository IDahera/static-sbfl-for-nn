import joblib
import click
import os
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def get_feature_names():
    """Define feature names based on dataset structure"""
    return [
        'layer_indices',
        'total_layers',
        'neuron_indices',
        'total_neurons',
        'layer_types',
        'weight_l1_norms',
        'weight_l2_norms',
        'weight_mean',
        'weight_std',
        'bias_l1_norms',
        'bias_l2_norms',
        'bias_mean',
        'bias_std',
        'predecessor_layer_sizes',
        'successor_layer_sizes',
        'normalized_layer_position',
        'normalized_neuron_position',
        'model_original_sizes'
    ]


def load_models(model_path):
    """Load all classifier models from directory"""
    models = {}
    
    for file in os.listdir(model_path):
        if file.startswith('classifier_model_') and file.endswith('.pkl'):
            model_name = file.replace('classifier_model_', '').replace('.pkl', '')
            model_name = model_name.replace('_', ' ').title()
            
            filepath = os.path.join(model_path, file)
            try:
                model = joblib.load(filepath)
                models[model_name] = model
                logging.info(f"Loaded: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")
    
    return models


def extract_feature_rankings(models, feature_names):
    """Extract feature importance rankings for each model"""
    rankings = {}
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            # Tree-based models have built-in feature importance
            importances = model.feature_importances_
            
            # Create ranking (1 = most important)
            sorted_indices = np.argsort(importances)[::-1]
            rankings[model_name] = {
                feature_names[idx]: rank + 1
                for rank, idx in enumerate(sorted_indices)
            }
        else:
            # Models without built-in importance get N/A
            rankings[model_name] = {
                feature: 'N/A' for feature in feature_names
            }
    
    return rankings


def create_ranking_table(rankings, feature_names, output_path):
    """Create consolidated ranking table and save to CSV"""
    
    # Create DataFrame with features as rows, models as columns
    data = []
    
    for feature in feature_names:
        row = {'Property': feature}
        
        # Collect ranks for mean calculation
        numeric_ranks = []
        
        for model_name in sorted(rankings.keys()):
            rank_value = rankings[model_name].get(feature, 'N/A')
            row[model_name] = rank_value
            
            # Add to numeric ranks if not N/A
            if rank_value != 'N/A':
                numeric_ranks.append(rank_value)
        
        # Calculate mean rank (only from numeric values)
        if numeric_ranks:
            row['Mean_Rank'] = round(np.mean(numeric_ranks), 2)
        else:
            row['Mean_Rank'] = 'N/A'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by mean rank (ascending = better/lower rank)
    df_sorted = df.copy()
    # Move N/A to bottom by replacing with large number for sorting
    df_sorted['_sort_key'] = df_sorted['Mean_Rank'].apply(
        lambda x: float('inf') if x == 'N/A' else x
    )
    df_sorted = df_sorted.sort_values('_sort_key')
    df_sorted = df_sorted.drop('_sort_key', axis=1)
    
    # Save to CSV
    csv_path = os.path.join(output_path, 'feature_rankings.csv')
    df_sorted.to_csv(csv_path, index=False)
    
    logging.info(f"\nFeature Rankings saved to: {csv_path}")
    
    # Display summary
    logging.info("\n" + "="*70)
    logging.info("FEATURE RANKING SUMMARY (sorted by mean rank)")
    logging.info("="*70)
    logging.info(df_sorted.to_string(index=False))
    
    return df_sorted


@click.command()
@click.option('--results-dir',
              default="outputs/results/",
              help='Directory containing config subdirectories with models')
def main(results_dir):
    """Feature importance interpretation for classification models"""
    
    if not os.path.exists(results_dir):
        logging.error(f"Error: Directory {results_dir} does not exist")
        return
    
    # Find all config subdirectories
    config_dirs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    
    if not config_dirs:
        logging.error(f"No config subdirectories found in {results_dir}")
        return
    
    config_dirs.sort()
    logging.info(f"Found {len(config_dirs)} config(s): {config_dirs}")
    
    # Process each config directory
    for config_name in config_dirs:
        config_results_path = os.path.join(results_dir, config_name)
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing config: {config_name}")
        logging.info(f"{'='*70}\n")
        
        # Find dataset subdirectories (mnist, fmnist, etc.)
        dataset_subdirs = [
            d for d in os.listdir(config_results_path)
            if os.path.isdir(os.path.join(config_results_path, d))
        ]
        
        if not dataset_subdirs:
            logging.warning(
                f"No dataset subdirectories in {config_results_path}")
            continue
        
        # Process each dataset
        for dataset_name in dataset_subdirs:
            model_path = os.path.join(config_results_path, dataset_name)
            logging.info(f"\n{'-'*50}")
            logging.info(f"Processing dataset: {dataset_name}")
            logging.info(f"{'-'*50}\n")
            
            # Load models
            models = load_models(model_path)
            
            if not models:
                logging.error(
                    f"No models loaded from {model_path}")
                continue
            
            logging.info(f"Successfully loaded {len(models)} models")
            
            # Get feature names
            feature_names = get_feature_names()
            
            # Extract rankings
            rankings = extract_feature_rankings(models, feature_names)
            
            # Create and save ranking table
            create_ranking_table(rankings, feature_names, model_path)
    
    logging.info(f"\n{'='*70}")
    logging.info("Interpretation completed for all configs and datasets")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
