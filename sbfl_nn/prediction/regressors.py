# Popular Regression Models Collection
# Import all necessary libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb

# 1. Linear Regression - Simple baseline model
linear_reg = LinearRegression()

# 2. Random Forest Regressor - Robust ensemble method
random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=100
)

# 3. XGBoost Regressor - High-performance gradient boosting
xgb_reg = xgb.XGBRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1
)

# 4. Ridge Regression - L2 regularized linear model
ridge_reg = Ridge(
    alpha=1.0,
    random_state=42
)

# 5. Lasso Regression - L1 regularized linear model (feature selection)
lasso_reg = Lasso(
    alpha=1.0,
    random_state=42
)

# 7. Gradient Boosting Regressor - Sequential ensemble method
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# 8. Multi-layer Perceptron Regressor - Neural network
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)

# List of all models for easy iteration
REGULAR_MODELS = {
    'Linear Regression': linear_reg,
    'Random Forest': random_forest,
    'XGBoost': xgb_reg,
    'Ridge': ridge_reg,
    'Lasso': lasso_reg,
    # 'Gradient Boosting': gb_reg,
    'MLP': mlp_reg
}


# 1. Linear Regression - baseline
improved_linear_reg = LinearRegression()

# 2. Random Forest with better hyperparameters
improved_random_forest = RandomForestRegressor(
    n_estimators=500,           # More trees
    max_depth=None,               # Limit depth to prevent overfitting
    min_samples_split=15,       # Require more samples to split
    min_samples_leaf=8,         # Require more samples in leaf nodes
    max_features=0.8,        # Use sqrt of features for each split
    n_jobs=-1                   # Use all cores
)

# 3. XGBoost with improved parameters
improved_xgb_reg = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,         # Lower learning rate
    max_depth=15,                # Moderate depth
    subsample=0.8,              # Sample 80% of data
    colsample_bytree=0.9,       # Sample 80% of features
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=0.8,             # L2 regularization
    random_state=42,
    n_jobs=-1
)

fast_learning_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.3,          # Very aggressive learning
    max_depth=12,
    subsample=0.95,
    colsample_bytree=0.95,
    reg_alpha=0,
    reg_lambda=0.001,           # Minimal regularization
    min_child_weight=0.1,
    gamma=0,
    random_state=42,
    n_jobs=-1
)

# 4. Ridge with cross-validation for alpha
improved_ridge_reg = Ridge(
    alpha=10.0,                 # Higher regularization
    random_state=42
)

# 5. Lasso with cross-validation for alpha
improved_lasso_reg = Lasso(
    alpha=0.1,                  # Lower alpha for less aggressive feature selection
    max_iter=500,              # More iterations
    random_state=42
)

# 6. ElasticNet - combines L1 and L2 regularization
improved_elastic_net = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,               # Balance between L1 and L2
    max_iter=500,
    random_state=42
)

# 7. Gradient Boosting with better parameters
improved_gb_reg = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=32,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# 8. Extra Trees (similar to Random Forest but more randomized)
improved_extra_trees = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',        # Use sqrt of features for each split
    random_state=42,
    n_jobs=-1
)

aggressive_extra_trees = ExtraTreesRegressor(
    n_estimators=500,           # Increase from 300
    max_depth=None,             # Let trees grow fully
    min_samples_split=2,        # Most aggressive splitting
    min_samples_leaf=1,         # Smallest possible leaves
    max_features='log2',        # Good for Extra Trees
    bootstrap=False,            # Keep Extra Trees behavior
    random_state=42,
    n_jobs=-1
)

balanced_extra_trees = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=50,               # Deep but limited
    min_samples_split=5,        # Moderate splitting
    min_samples_leaf=2,         # Small but not tiny leaves
    max_features='sqrt',        # Conservative feature selection
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

# 10. Support Vector Regression
improved_svr_reg = SVR(
    kernel='rbf',
    C=100,
    epsilon=0.1,
    gamma='scale'
)

conservative_xgb = xgb.XGBRegressor(
    n_estimators=100,           # Much fewer trees
    learning_rate=0.01,         # Very slow learning
    max_depth=3,                # Shallow trees
    subsample=0.6,              # Strong regularization
    colsample_bytree=0.6,
    reg_alpha=1.0,              # Add back regularization
    reg_lambda=10.0,            # Strong L2 regularization
    min_child_weight=10,        # Large minimum samples
    random_state=42
)

simple_extra_trees = ExtraTreesRegressor(
    n_estimators=50,            # Few trees
    max_depth=5,                # Very shallow
    min_samples_split=20,       # Conservative splitting
    min_samples_leaf=10,        # Large leaves
    max_features=0.3,           # Use few features
    random_state=42
)

# Updated models dictionary
IMPROVED_MODELS = {
    # 'Linear Regression': improved_linear_reg,
    # 'Random Forest': improved_random_forest,
    'Conservative XGBoost': conservative_xgb,
    # 'Ridge': improved_ridge_reg,
    # 'Lasso': improved_lasso_reg,
    # 'ElasticNet': improved_elastic_net,
    # 'Gradient Boosting': improved_gb_reg,
    'Simple Extra Trees': simple_extra_trees,
    # 'MLP': improved_mlp_reg,
    # 'SVR': improved_svr_reg
}
