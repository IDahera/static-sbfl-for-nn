from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import xgboost as xgb
import lightgbm as lgb


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible ensemble classifier"""

    def __init__(self, models, weights=None, threshold=0.5):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        self.threshold = threshold
        self.fitted_models = None

    def fit(self, X, y):
        """Train all models in the ensemble"""
        self.fitted_models = []
        for model in self.models:
            # Create a copy and fit
            fitted_model = model.__class__(**model.get_params())
            fitted_model.fit(X, y)
            self.fitted_models.append(fitted_model)
        return self

    def predict_proba(self, X):
        """Get ensemble probability predictions"""
        if self.fitted_models is None:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get predictions from each model
        predictions = []
        for model in self.fitted_models:
            # Probability of positive class
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)

        # Return in sklearn format
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

    def predict(self, X):
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'models': self.models,
            'weights': self.weights,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


RANDOM_STATE = 42

# ============================================================================
# TREE-BASED MODELS
# ============================================================================

# Single Decision Trees
decision_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)

# Forest-based Models
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

extra_trees = ExtraTreesClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ============================================================================
# GRADIENT BOOSTING MODELS
# ============================================================================

# Scikit-learn Gradient Boosting
gradient_boosting = GradientBoostingClassifier(
    random_state=RANDOM_STATE
)

# XGBoost Models
xgboost_regular = xgb.XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='logloss'
)

precision_xgboost = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=3.0,  # Favor positive class (suspicious neurons)
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss'
)

# LightGBM Models
lightgbm_regular = lgb.LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1
)

precision_lightgbm = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    class_weight={0: 1, 1: 3},  # Favor positive class (suspicious neurons)
    random_state=RANDOM_STATE,
    verbose=-1
)

# ============================================================================
# LINEAR MODELS
# ============================================================================

logistic_regression = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=1000
)

# ============================================================================
# PROBABILISTIC MODELS
# ============================================================================

naive_bayes = GaussianNB()

# ============================================================================
# INSTANCE-BASED MODELS
# ============================================================================

knn = KNeighborsClassifier()

# ============================================================================
# MODEL REGISTRY - GROUPED BY TYPE
# ============================================================================

TREE_MODELS = {
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'Extra Trees': extra_trees,
}

BOOSTING_MODELS = {
    'Gradient Boosting': gradient_boosting,
    'XGBoost': xgboost_regular,
    'Precision XGBoost': precision_xgboost,
    'LightGBM': lightgbm_regular,
    'Precision LightGBM': precision_lightgbm,
}

ENSEMBLE_MODELS = {
    # Ensemble models removed
}

LINEAR_MODELS = {
    'Logistic Regression': logistic_regression,
}

PROBABILISTIC_MODELS = {
    'Naive Bayes': naive_bayes,
}

INSTANCE_MODELS = {
    'KNN': knn,
}

# Combined registry for backward compatibility
MODELS = {
    **TREE_MODELS,
    **BOOSTING_MODELS,
    **LINEAR_MODELS,
    **PROBABILISTIC_MODELS,
    **INSTANCE_MODELS,
}
