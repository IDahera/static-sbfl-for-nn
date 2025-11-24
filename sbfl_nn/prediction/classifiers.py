from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


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

# ============================================================================
# GRADIENT BOOSTING MODELS
# ============================================================================

# Scikit-learn Gradient Boosting
gradient_boosting = GradientBoostingClassifier(
    random_state=RANDOM_STATE
)

# ============================================================================
# MODEL REGISTRY - GROUPED BY TYPE
# ============================================================================

TREE_MODELS = {
    'Decision Tree': decision_tree,
    'Random Forest': random_forest
}

BOOSTING_MODELS = {
    'Gradient Boosting': gradient_boosting,
}

# Combined registry for backward compatibility
MODELS = {
    **TREE_MODELS,
    **BOOSTING_MODELS,
}
