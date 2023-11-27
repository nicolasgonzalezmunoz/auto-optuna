"""
Created on Sat Oct  7 16:18:07 2023, @author: nicolas.

This module implements an auto optimizer for linear models.
"""
from .base import BaseOptimizer
from .parameter_grid import ParameterGrid
from sklearn import linear_model


class SGDClassifierOptimizer(BaseOptimizer):
    """
    Class dedicated to optimize a SGDClassifier object.

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

    output_proba: bool, default=True
        If true, the optimizer only uses loss functions that output
        probabilities. This ensures that the predict_proba method works with
        the classifier.

    seed: int or None, default=None
        Random seed.

    Attributes
    ----------
    study: optuna.study.Study
        A study instance with information about the last optimization task.

    direction: {"minimize", "maximize"}
        Direction of the last optimization task.

    optimized_estimator: sklearn.linear_model.SGDClassifier
        An optimized instance of the base estimator.
    """

    __slots__ = [
        'fit_intercept', 'class_weight', 'early_stopping', 'n_iter_no_change',
        'output_proba'
    ]

    def __init__(
            self, *, fit_intercept=True, class_weight="balanced",
            early_stopping=True, n_iter_no_change=10, seed=None,
            output_proba=True, estimator_threads=1, optimizer_threads=1
    ):
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.output_proba = output_proba
        super().__init__(
            linear_model.SGDClassifier, estimator_threads, optimizer_threads,
            seed
        )

    def get_default_search_grid(self, trial, X=None, y=None):
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
        if self.output_proba:
            losses = ["log_loss", "modified_huber"]
        else:
            losses = [
                "hinge", "log_loss", "modified_huber", "squared_hinge",
                "perceptron"
            ]
        params_dict = {
            # Parameters to optimize
            "loss": {"type": "categorical", "grid": losses},
            "penalty": {
                "type": "categorical", "grid": [
                    "l2", "l1", "elasticnet", None
                ]
            },
            "alpha": {"type": "float", "grid": [1e-8, 100], "log": True},
            "l1_ratio": {"type": "float", "grid": [1e-8, 1], "log": True},
            "epsilon": {"type": "float", "grid": [1e-4, 10], "log": True},

            # User-defined parameters
            "random_state": {"type": "fixed", "grid": self.seed},
            "early_stopping": {"type": "fixed", "grid": self.early_stopping},
            "n_iter_no_change": {
                "type": "fixed", "grid": self.n_iter_no_change
            },
            "fit_intercept": {"type": "fixed", "grid": self.fit_intercept},

            # Fixed params
            "learning_rate": {"type": "fixed", "grid": "optimal"}
        }
        return ParameterGrid.from_dict(params_dict)
