"""
Model ensemble for combining multiple trading models.
"""
from typing import List, Any
import numpy as np

class ModelEnsemble:
    """
    Ensemble of multiple trading models (stacking/blending).
    """
    def __init__(self, models: List[Any]):
        self.models = models

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0) 