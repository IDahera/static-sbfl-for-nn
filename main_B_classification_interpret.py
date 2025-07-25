import joblib
import click
import os
import glob
import numpy as np
import pandas as pd
import logging
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def get_feature_names(n_features=None):
    """Define feature names based on the actual dataset structure from dataset_generation.py"""
    # Complete feature names matching the order in dataset_generation.py get_numpy() method
    # Updated to 18 features total (removed model_accuracies)
    base_names = [
        'layer_indices',          # 0
        'total_layers',           # 1
        'neuron_indices',         # 2
        'total_neurons',          # 3
        'layer_types',            # 4
        'weight_l1_norms',        # 5
        'weight_l2_norms',        # 6
        'weight_mean',            # 7
        'weight_std',             # 8
        'bias_l1_norms',          # 9
        'bias_l2_norms',          # 10
        'bias_mean',              # 11
        'bias_std',               # 12
        'predecessor_layer_sizes', # 13
        'successor_layer_sizes',  # 14
        'normalized_layer_position', # 15
        'normalized_neuron_position', # 16
        'model_original_sizes'    # 17
    ]
    
    # If we need more features than we have names, generate generic ones
    if n_features and n_features > len(base_names):
        for i in range(len(base_names), n_features):
            base_names.append(f'feature_{i}')
    
    # If we need fewer features, truncate
    if n_features and n_features < len(base_names):
        base_names = base_names[:n_features]
    
    return base_names


def analyze_feature_importance(model, model_name, model_path, X_test=None, y_test=None):
    """Analyze feature importance for a given model and save to CSV"""
    logging.info(f"\n{'='*60}")
    logging.info(f"FEATURE IMPORTANCE ANALYSIS: {model_name}")
    logging.info(f"{'='*60}")
    
    # Determine number of features from model
    n_features = None
    if hasattr(model, 'feature_importances_'):
        n_features = len(model.feature_importances_)
    elif hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    elif X_test is not None:
        n_features = X_test.shape[1]
    
    feature_names = get_feature_names(n_features)
    
    # Initialize combined results DataFrame
    combined_results = pd.DataFrame({'Feature': feature_names})
    
    # 1. Tree-based feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        logging.info("\n📊 Tree-based Feature Importance:")
        importance_scores = model.feature_importances_
        
        # Ensure we have the right number of feature names
        if len(feature_names) != len(importance_scores):
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
            combined_results = pd.DataFrame({'Feature': feature_names})
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)
        
        logging.info(importance_df.to_string(index=False, float_format='%.4f'))
        
        # Add to combined results
        combined_results['Tree_Based_Importance'] = importance_scores
        
        # Visualize top features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
        plt.title(f'Top 10 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # 2. Permutation importance (if test data available)
    if X_test is not None and y_test is not None:
        logging.info("\n🔄 Permutation Importance Analysis:")
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, 
                n_repeats=10, 
                random_state=42,
                scoring='accuracy'
            )
            
            # Ensure feature names match permutation importance length
            perm_feature_names = feature_names[:len(perm_importance.importances_mean)]
            if len(perm_feature_names) != len(perm_importance.importances_mean):
                perm_feature_names = [f'feature_{i}' for i in range(len(perm_importance.importances_mean))]
            
            perm_df = pd.DataFrame({
                'Feature': perm_feature_names,
                'Mean_Importance': perm_importance.importances_mean,
                'Std_Importance': perm_importance.importances_std
            }).sort_values('Mean_Importance', ascending=False)
            
            logging.info(perm_df.to_string(index=False, float_format='%.4f'))
            
            # Add permutation importance to combined results
            if len(perm_importance.importances_mean) == len(combined_results):
                combined_results['Permutation_Importance'] = perm_importance.importances_mean
                combined_results['Permutation_Std'] = perm_importance.importances_std
            
        except Exception as e:
            logging.warning(f"Could not compute permutation importance: {e}")
    
    # 3. Model-specific interpretability
    logging.info(f"\n🔍 Model-specific Analysis:")
    
    if isinstance(model, DecisionTreeClassifier):
        logging.info(f"  - Tree depth: {model.get_depth()}")
        logging.info(f"  - Number of leaves: {model.get_n_leaves()}")
        
    elif isinstance(model, RandomForestClassifier):
        logging.info(f"  - Number of estimators: {model.n_estimators}")
        logging.info(f"  - Average tree depth: {np.mean([tree.get_depth() for tree in model.estimators_]):.2f}")
        
    elif isinstance(model, GradientBoostingClassifier):
        logging.info(f"  - Number of estimators: {model.n_estimators}")
        logging.info(f"  - Learning rate: {model.learning_rate}")
        
    elif hasattr(model, 'get_params'):
        # For XGBoost and other models
        params = model.get_params()
        important_params = ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight']
        for param in important_params:
            if param in params:
                logging.info(f"  - {param}: {params[param]}")
        
    # 4. Feature correlation analysis (if applicable)
    if hasattr(model, 'feature_importances_'):
        logging.info(f"\n📈 Top 5 Most Important Features:")
        top_features = importance_df.head(5)
        for idx, row in top_features.iterrows():
            logging.info(f"  {row['Feature']:25}: {row['Importance']:.4f}")
    
    # 5. Save combined results to CSV
    csv_filename = os.path.join(model_path, f"impact_analysis_{model_name}_feature_importance.csv")
    
    # Sort by the most comprehensive importance measure available
    if 'Tree_Based_Importance' in combined_results.columns:
        combined_results = combined_results.sort_values('Tree_Based_Importance', ascending=False)
    elif 'Permutation_Importance' in combined_results.columns:
        combined_results = combined_results.sort_values('Permutation_Importance', ascending=False)
    
    # Round numerical columns for cleaner output
    numeric_columns = combined_results.select_dtypes(include=[np.number]).columns
    combined_results[numeric_columns] = combined_results[numeric_columns].round(6)
    
    combined_results.to_csv(csv_filename, index=False)
    logging.info(f"\n💾 Feature importance saved to: {csv_filename}")
    
    return combined_results


def evaluate_model_parameters(models, model_path, X_test=None, y_test=None):
    """Evaluate parameter impact for all loaded models and save results"""
    logging.info(f"\n🎯 COMPREHENSIVE MODEL PARAMETER ANALYSIS")
    logging.info(f"{'='*80}")
    
    if X_test is None or y_test is None:
        logging.warning("⚠️  Warning: No test data provided. Limited analysis available.")
        logging.warning("   Only intrinsic model properties will be analyzed.")
    
    all_results = {}
    
    for name, model in models.items():
        try:
            results = analyze_feature_importance(model, name, model_path, X_test, y_test)
            all_results[name] = results
            
        except Exception as e:
            logging.error(f"❌ Error analyzing {name}: {e}")
    
    # Create consolidated feature importance CSV (features as rows, models as columns)
    if all_results:
        logging.info(f"\n📊 Creating consolidated feature importance analysis...")
        create_consolidated_feature_importance_csv(all_results, model_path)
    
    # Summary comparison
    logging.info(f"\n📋 SUMMARY COMPARISON")
    logging.info(f"{'='*50}")
    
    for name, model in models.items():
        logging.info(f"\n{name}:")
        logging.info(f"  Model type: {type(model).__name__}")
        
        if hasattr(model, 'feature_importances_'):
            try:
                top_feature_idx = np.argmax(model.feature_importances_)
                # Get appropriate feature names based on model's feature count
                model_feature_names = get_feature_names(len(model.feature_importances_))
                top_feature_name = model_feature_names[top_feature_idx]
                top_importance = model.feature_importances_[top_feature_idx]
                logging.info(f"  Most important feature: {top_feature_name} ({top_importance:.4f})")
            except Exception as e:
                logging.warning(f"  Could not determine top feature: {e}")
        
        if hasattr(model, 'score') and X_test is not None and y_test is not None:
            try:
                accuracy = model.score(X_test, y_test)
                logging.info(f"  Test accuracy: {accuracy:.4f}")
            except Exception as e:
                logging.warning(f"  Test accuracy: Could not compute - {e}")
    
    return all_results


def create_consolidated_feature_importance_csv(all_results, model_path):
    """Create a single CSV file with features as rows and models as columns"""
    
    # Get all unique features across all models
    all_features = set()
    for model_name, results_df in all_results.items():
        all_features.update(results_df['Feature'].tolist())
    
    # Sort features to ensure consistent ordering
    all_features = sorted(all_features)
    
    # Get all model names
    model_names = sorted(all_results.keys())
    
    # Create consolidated DataFrame
    consolidated_data = []
    
    for feature in all_features:
        row = {'Feature': feature}
        
        for model_name in model_names:
            results_df = all_results[model_name]
            
            # Find the feature importance for this model
            feature_row = results_df[results_df['Feature'] == feature]
            
            if not feature_row.empty:
                # Use Tree_Based_Importance if available, otherwise Permutation_Importance
                if 'Tree_Based_Importance' in feature_row.columns:
                    importance = feature_row['Tree_Based_Importance'].iloc[0]
                elif 'Permutation_Importance' in feature_row.columns:
                    importance = feature_row['Permutation_Importance'].iloc[0]
                else:
                    importance = 0.0
            else:
                importance = 0.0
            
            # Clean model name for column header
            clean_model_name = model_name.replace('_model', '').replace('_', ' ').title()
            row[clean_model_name] = round(importance, 6)
        
        consolidated_data.append(row)
    
    # Create DataFrame
    consolidated_df = pd.DataFrame(consolidated_data)
    
    # Sort by the sum of importance across all models (most important features first)
    model_columns = [col for col in consolidated_df.columns if col != 'Feature']
    consolidated_df['Total_Importance'] = consolidated_df[model_columns].sum(axis=1)
    consolidated_df = consolidated_df.sort_values('Total_Importance', ascending=False)
    consolidated_df = consolidated_df.drop('Total_Importance', axis=1)
    
    # Save to CSV
    consolidated_csv_filename = os.path.join(model_path, 'consolidated_feature_importance.csv')
    consolidated_df.to_csv(consolidated_csv_filename, index=False)
    
    logging.info(f"📋 Consolidated feature importance saved to: {consolidated_csv_filename}")
    
    # Also create a summary showing top 5 features for each model
    summary_data = []
    
    for model_name in model_names:
        results_df = all_results[model_name]
        
        # Get top 5 features for this model
        if 'Tree_Based_Importance' in results_df.columns:
            top_features = results_df.nlargest(5, 'Tree_Based_Importance')
            importance_col = 'Tree_Based_Importance'
        elif 'Permutation_Importance' in results_df.columns:
            top_features = results_df.nlargest(5, 'Permutation_Importance')
            importance_col = 'Permutation_Importance'
        else:
            continue
        
        clean_model_name = model_name.replace('_model', '').replace('_', ' ').title()
        
        for idx, (_, row) in enumerate(top_features.iterrows()):
            summary_data.append({
                'Model': clean_model_name,
                'Rank': idx + 1,
                'Feature': row['Feature'],
                'Importance': round(row[importance_col], 6)
            })
    
    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_filename = os.path.join(model_path, 'top_features_by_model.csv')
        summary_df.to_csv(summary_csv_filename, index=False)
        
        logging.info(f"📋 Top features by model saved to: {summary_csv_filename}")
    
    return consolidated_csv_filename


def load_test_data():
    """Load test data for evaluation (if available)"""
    # Try to load test data from common locations
    test_data_paths = [
        "models/mnist/dataset/mnist_dataset_numpy.pkl",
        "dataset/test_data.pkl",
        "data/test_features.pkl"
    ]
    
    for path in test_data_paths:
        if os.path.exists(path):
            try:
                logging.info(f"📁 Loading test data from {path}")
                data = joblib.load(path) if path.endswith('.pkl') else pd.read_pickle(path)
                
                if isinstance(data, dict) and 'features' in data and 'targets' in data:
                    X = data['features']
                    y = data['targets']
                    
                    # Handle multi-target case - use first target for classification
                    if len(y.shape) > 1 and y.shape[1] > 1:
                        logging.info(f"  Multi-target detected, using first target for analysis")
                        y = y[:, 0]
                    
                    logging.info(f"  ✓ Loaded {X.shape[0]} samples with {X.shape[1]} features")
                    return X, y
                    
            except Exception as e:
                logging.error(f"  ❌ Failed to load {path}: {e}")
                continue
    
    logging.warning("⚠️  No test data found. Analysis will be limited to model intrinsics.")
    return None, None


def load_models(model_directory):
    """Load all pkl models from the specified directory"""
    loaded_models = {}
    
    # Find all pkl files in the directory
    pkl_pattern = os.path.join(model_directory, "*.pkl")
    pkl_files = glob.glob(pkl_pattern)
    
    if not pkl_files:
        logging.error(f"No .pkl files found in {model_directory}")
        return loaded_models
    
    logging.info(f"Found {len(pkl_files)} .pkl files in {model_directory}")
    
    for filepath in pkl_files:
        # Extract model name from filename (without extension)
        model_name = os.path.splitext(os.path.basename(filepath))[0]
        
        try:
            loaded_models[model_name] = joblib.load(filepath)
            logging.info(f"✓ Loaded {model_name} from {filepath}")
        except Exception as e:
            logging.error(f"✗ Failed to load {model_name} from {filepath}: {e}")
    
    return loaded_models


@click.command()
@click.option('--model-path',
              default="models_sus_prediction/",
              help='Directory path containing model pkl files')
@click.option('--test-data-path',
              default=None,
              help='Path to test data for evaluation (optional)')
@click.option('--save-plots',
              is_flag=True,
              default=True,
              help='Save feature importance plots')
def main(model_path, test_data_path, save_plots):
    """Main function for classification interpretation and parameter analysis"""
    
    # Ensure the path exists
    if not os.path.exists(model_path):
        logging.error(f"Error: Directory {model_path} does not exist")
        return
    
    # Load models from the specified directory
    models = load_models(model_path)
    
    if not models:
        logging.error("No models were loaded successfully.")
        return
    
    logging.info(f"\nSuccessfully loaded {len(models)} models:")
    
    # Display loaded models
    for name, model in models.items():
        logging.info(f"  - {name}: {type(model).__name__}")
    
    # Load test data for evaluation
    if test_data_path:
        if os.path.exists(test_data_path):
            try:
                data = joblib.load(test_data_path)
                X_test, y_test = data['features'], data['targets']
                if len(y_test.shape) > 1:
                    y_test = y_test[:, 0]  # Use first target
                logging.info(f"\n✓ Loaded test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
            except Exception as e:
                logging.error(f"❌ Failed to load test data: {e}")
                X_test, y_test = load_test_data()
        else:
            logging.error(f"❌ Test data path {test_data_path} does not exist")
            X_test, y_test = load_test_data()
    else:
        X_test, y_test = load_test_data()
    
    # Conduct comprehensive parameter impact analysis
    results = evaluate_model_parameters(models, model_path, X_test, y_test)
    
    if save_plots:
        logging.info(f"\n💾 Feature importance plots saved to current directory")
        logging.info(f"   Files: feature_importance_*.png")
    
    logging.info(f"\n📄 CSV files created in {model_path}:")
    logging.info(f"   - Individual model files: impact_analysis_[model_name]_feature_importance.csv")
    logging.info(f"   - Consolidated analysis: consolidated_feature_importance.csv")
    logging.info(f"   - Top features summary: top_features_by_model.csv")
    

if __name__ == '__main__':
    main()