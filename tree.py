"""
Created on Sun Nov 19 15:35:09 2023, @author: nicolas.

This module implements an auto optimizer for tree models.
"""
from sklearn import tree
import numpy as np
from .parameter_grid import ParameterGrid
from .base import BaseTreeOptimizer


class DecisionTreeClassifierOptimizer(BaseTreeOptimizer):
    """
    Class dedicated to optimize a DecisionTreeClassifier object.

    Parameters
    ----------
    fit_intercept: bool, default=True
        Whether to fit an intercept or not.

    class_weight: dict, {class_label: weight} or 'balanced', default='balanced'
        Weights associated with classes. If not given, all classes are
        supposed to have weight one.

    optimizer_threads: int, default=-1
        Number of cpus the optimizer will use. If -1, then uses all available
        threads.

    early_stopping: bool, default=True
        Wether to use early stopping or not.

    n_iter_no_change: int, default=10
        Number of iterations without performance improvement to wait before
        applying early stopping.

    seed: int or None, default=None
        Random seed.

    direction: {"minimize", "maximize"}, default="maximize"
        Direction of the optimization task.

    Attributes
    ----------
    estimator:
        The base estimator to optimize. Equals to DecisionTreeClassifier.

    optimized_estimator: An optimized instance of the base estimator.
    """

    __slots__ = ['class_weight']

    def __init__(
            self, *, class_weight="balanced", seed=None, estimator_threads=1,
            optimizer_threads=-1
    ):
        self.class_weight = class_weight
        super().__init__(
            tree.DecisionTreeClassifier, estimator_threads, optimizer_threads,
            seed
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
            "criterion": {
                "type": "categorical", "grid": ["gini", "entropy", "log_loss"]
            },
            "splitter": {
                "type": "categorical", "grid": ["best", "random"]
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

            # User-defined parameters
            "random_state": {"type": "fixed", "grid": self.seed}
        }
        return ParameterGrid.from_dict(params_dict)
