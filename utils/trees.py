"""
Created on Sat Nov 25 17:28:16 2023, @author: nicolas.

Implements utility functions for decision trees.
"""
import numpy as np
from .metrics import cross_validation_score


def prune_tree(
        tree_class, tree_params, X_train, y_train, scorer, *, direction=None,
        eval_set=None, cv=5, seed=None, cv_mode="normalize"
):
    """
    Maximize decision tree performance by pruning it.

    tree_params is modified in-place to append the best cpp_alpha value, and
    a fitted model is returned.

    Parameters
    ----------
    tree_class: class
        A scikit-learn-compatible decision tree model.

    tree_params: dict
        Parameters to pass to the tree class.

    X_train, y_train: array-like
        Training dataset.

    scorer: callable with signature (estimator, X, y)
        The function to use to score the model.

    direction: {"minimize", "maximize"}, default=None
        How to decide which score is the best. If None, then the function will
        try to find the best optimization direction based on the provided
        scorer.

    eval_set: list of tuple, default=None
        Evaluation set in the format [(X_eval, y_eval)].

    cv: int or callable, default=5
        How to split the dataset in case cross-validation must be performed.
        If eval_set is provided, this argument is ignored.

    seed: int, default=None
        Random seed.

    cv_mode: {'mean', 'normalize'}, default='normalize'
        How to calculate the cross-validation score. If eval_set is provided,
        this argument is ignored.

    Returns
    -------
    tree: decision tree model
        A fitted decision tree instance with the best cpp_alpha.
    """
    cpp_alphas = tree_class(tree_params).cost_complexity_pruning_path(
        X_train, y_train
    ).ccp_alphas

    if direction is None:
        if scorer._sign > 0:
            direction = "maximize"
        else:
            direction = "minimize"

    scores = np.zeros(shape=cpp_alphas.shape)
    if eval_set is None:
        for i, cpp_alpha in enumerate(cpp_alphas):
            params_copy = tree_params.copy()
            params_copy["cpp_alpha"] = cpp_alpha
            tree = tree_class(params_copy)
            scores[i] = cross_validation_score(
                tree, X_train, y_train, scorer, cv=cv, seed=seed, mode=cv_mode
            )
    else:
        for i, cpp_alpha in enumerate(cpp_alphas):
            params_copy = tree_params.copy()
            params_copy["cpp_alpha"] = cpp_alpha
            tree = tree_class(params_copy)
            tree.fit(X_train, y_train)
            scores[i] = scorer(tree, eval_set[0][0], eval_set[0][1])
    if direction == "maximize":
        best_index = np.argmax(scores)
    else:
        best_index = np.argmin(scores)
    cpp_alpha = cpp_alphas[best_index]
    tree_params["cpp_alpha"] = cpp_alpha
    tree = tree_class(tree_params)
    tree.fit(X_train, y_train)
    return tree
