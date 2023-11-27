"""
Created on Wed Nov 22 11:49:28 2023, @author: nicolas.

This module implements optimizers for scikit-learn ensemble models.
"""
import numpy as np
from sklearn import ensemble
from .base import BaseIndependentLearnerOptimizer
from .parameter_grid import ParameterGrid


class RandomForestClassifierOptimizer(BaseIndependentLearnerOptimizer):
    """An optimizer for the RandomForestClassifier class."""

    __slots__ = [
        'class_weight', 'oob_score', 'class_weight'
    ]

    def __init__(
            self, bootstrap=True, oob_score=False, class_weight="balanced",
            seed=None, optimizer_threads=-1, estimator_threads=1
    ):
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        super().__init__(
            ensemble.RandomForestClassifier, estimator_threads,
            optimizer_threads, seed
        )

    def get_default_search_grid(self, trial, X, y=None):
        """
        Return the default ParameterGrid for the current estimator.

        Each optimizer implements its default ParameterGrid.

        Parameters
        ----------
        trial: optuna.trial.Trial
            An optuna trial used to optimize the estimator.

        X: 2D array-like, default=None
            Dataset.

        y: array-like, default=None
            Target values.

        Returns
        -------
        search_grid: dict
            Parameters of the estimator.
        """
        n_features = len(X)
        n_samples = len(X[0])
        params_dict = {
            # Parameters to optimize
            "n_estimators": {"type": "int", "grid": [1, 10000], "log": True},
            "criterion": {
                "type": "categorical", "grid": ["gini", "entropy", "log_loss"]
            },
            "max_depth": {"type": "int", "grid": [1, 50], "log": True},
            "min_samples_split": {
                "type": "int", "grid": [2, int(np.sqrt(n_samples))],
                "log": True
            },
            "min_samples_leaf": {
                "type": "int", "grid": [1, int(np.sqrt(n_samples))],
                "log": True
            },
            "min_weight_fraction_leaf": {
                "type": "float", "grid": [1e-8, 0.5], "log": True
            },
            "max_features": {
                "type": "int", "grid": [2, n_features], "log": n_features > 50
            },
            "max_leaf_nodes": {"type": "int", "grid": [2, 50], "log": True},
            "max_samples": {"type": "float", "grid": [1e-8, 1], "log": True},

            # User-defined parameters
            "bootstrap": {"type": "fixed", "grid": self.bootstrap},
            "oob_score": {"type": "fixed", "grid": self.oob_score},
            "random_state": {"type": "fixed", "grid": self.seed},
            "class_weight": {"type": "fixed", "grid": self.class_weight}
        }
        return ParameterGrid.from_dict(params_dict)
