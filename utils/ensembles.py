#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:55:34 2023, @author: nicolas.

Utilities specific for ensemble models.
"""
import copy
import numpy as np
import scipy.stats as ss
from sklearn.base import is_classifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from .utils.metrics import make_scorer


def get_prediction_matrix(learners, X):
    """
    Get the predictions from X of each learner in learners.

    Parameters
    ----------
    learners: list
        Learners to get the predictions from.

    X: 2D array-like of shape (n_samples, n_features)
        Dataset to make the predictions on.

    Returns
    -------
    prediction_matrix: 2D ndarray of shape (n_samples, n_learners)
        Matrix with the predictions of each learner.
    """
    is_clf = is_classifier(learners[0])
    n_estimators = len(learners)
    n_samples = len(X)
    prediction_matrix = np.empty((n_samples, n_estimators))
    if is_clf:
        for i, learner in enumerate(learners):
            prediction_matrix[:, i] = learner.predict_proba(X)
    else:
        for i, learner in enumerate(learners):
            prediction_matrix[:, i] = learner.predict(X)
            return prediction_matrix


def get_scores_array(learners, X, y, scorer):
    """
    Get the scores of each learner over the dataset (X, y).

    Parameters
    ----------
    learners: list
        Learners to score.

    X: 2D array-like of shape (n_samples, n_features)
        Dataset to make the predictions on.

    y: array-like
        Targets to compare with predicted values.

    scorer: callable with signature (estimator, X, y)
        Metric used to score the learners.

    Returns
    -------
    scores_array: 1D ndarray of shape (n_learners)
        Array with the scores of each learner.
    """
    is_clf = is_classifier(learners[0])
    n_estimators = len(learners)
    scores_array = np.empty(shape=(n_estimators, ))
    if is_clf:
        y_enc = LabelEncoder().fit_transform(y)
        for i, learner in enumerate(learners):
            scores_array[i] = scorer(learner, X, y_enc)
    else:
        for i, learner in enumerate(learners):
            scores_array[i] = scorer(learner, X, y)
    return scores_array


def get_predictions_and_scores(learners, X, y, scorer):
    """
    Get the predictions and scores of each learner in learners.

    Parameters
    ----------
    learners: list
        Learners to score.

    X: 2D array-like of shape (n_samples, n_features)
        Dataset to make the predictions on.

    y: array-like
        Targets to compare with predicted values.

    scorer: callable with signature (estimator, X, y)
        Metric used to score the learners.

    Returns
    -------
    prediction_matrix: 2D ndarray of shape (n_samples, n_learners)
        Matrix with the predictions of each learner.

    scores_array: 1D ndarray of shape (n_learners)
        Array with the scores of each learner.
    """
    n_estimators = len(learners)
    n_samples = len(X)
    is_clf = is_classifier(learners[0])

    prediction_matrix = np.empty(shape=(n_samples, n_estimators))
    scores_array = np.empty(shape=(n_estimators, ))

    if is_clf:
        y_enc = LabelEncoder().fit_transform(y)
        for i, learner in enumerate(learners):
            scores_array[i] = scorer(learner, X, y_enc)
            prediction_matrix[:, i] = learner.predict_proba(X)
    else:
        for i, learner in enumerate(learners):
            scores_array[i] = scorer(learner, X, y)
            prediction_matrix[:, i] = learner.predict(X)
    return prediction_matrix, scores_array


def uncorrelate_estimators(
        estimator, X, y, score_fn, *, confidence_level=0.99, test="spearman",
        inplace=False, scorer_kwargs=None, test_kwargs=None
):
    """
    Generate a new ensemble estimator where its estimators are uncorrelated.

    This function uses the criterion and confidence level provided by the user
    to perform a hypothesis test on the correlation of the estimators. If two
    or more estimators are correlated, the function selects the estimator with
    the best score to keep it, then drop the other correlated estimators.

    If the estimator is a classifier, then the function uses the predict_proba
    method to generate predictions.

    Note: This function is designed to be used only with ensemble methods that
    train each estimator independently from each other, like random forest.

    Parameters
    ----------
    estimator: An scikit-learn ensemble estimator
        An ensemble estimator that trains each estimator independently from
        each other.

    X: 2D array-like
        The values to be used for predictions.

    y: 1D array-like
        Target.

    score_fn: callable
        Function to use for scoring performance.

    confidence_level: float, default=0.99
         Confidence level of the  hypothesis test.

    test: {"pearson", "spearman", "kendalltau", "weightedtau"},
    default="spearman"
        The hypothesis test to be applied to the estimators, which are the ones
        implemented in the scipy.stats module.

    inplace: bool, default=False
        Whether to modify the estimator in-place or not.

    scorer_kwargs: dict, default=None
        Additional parameters to be passed to the scorer.

    test_kwargs: dict, default=None
        Additional parameters to be passed to the correlation test.

    Returns
    -------
    new_estimator: sklearn ensemble model, or None
        If inplace=False, returns a new ensemble model built from the original,
        user-provided ensemble, else returns None and the ensemble is modified
        inplace.
    """
    if test == "pearson":
        corr_test = ss.pearsonr
    elif test == "spearman":
        corr_test = ss.spearmanr
    elif test == "kendalltau":
        corr_test = ss.kendalltau
    else:
        corr_test = ss.weightedtau

    scorer = make_scorer(estimator, score_fn, **scorer_kwargs)
    greater_is_better = scorer._sign > 0

    learners = estimator.estimators_.copy()
    n_estimators = len(learners)
    if n_estimators == 1:
        return estimator

    prediction_matrix, scores_array = get_predictions_and_scores(
        learners, X, y, scorer
    )

    i = 0
    while i < len(learners) - 1:
        drop_i = False
        x1 = prediction_matrix[:, i]
        score1 = scores_array[i]
        j = i + 1
        while j < len(learners):
            x2 = prediction_matrix[:, j]
            score2 = scores_array[j]
            are_correlated = corr_test(
                x1, x2, **test_kwargs
            ).pvalue < (1 - confidence_level)
            if are_correlated:
                drop_i = (greater_is_better and score1 < score2)
                drop_i = drop_i or (not greater_is_better and score1 > score2)
                if drop_i:
                    learners.pop(i)
                    prediction_matrix = np.delete(prediction_matrix, i, axis=1)
                    scores_array = np.delete(scores_array, i)
                    break
                else:
                    learners.pop(j)
                    prediction_matrix = np.delete(prediction_matrix, j, axis=1)
                    scores_array = np.delete(scores_array, j)
            else:
                j += 1
        if not drop_i:
            i += 1
    if inplace:
        estimator.estimators_ = learners
        return None
    new_estimator = copy.deepcopy(estimator)
    new_estimator.estimators_ = learners
    return new_estimator


def select_important_estimators(
    estimator, X, y, selector=Lasso(random_state=42), *, threshold=None,
    norm_order=1, max_estimators=None, inplace=False
):
    """
    Select a set of learners using a SelectFromModel object.

    The SelectFromModel uses a selector model to sort the importance of each
    individual learner on inference time. Learners hows importance is below
    a given threshold are dropped from the ensemble.

    Note: This function assumes that estimations made by each learner don't use
    estimations of other learners as input.

    Parameters
    ----------
    estimator: A sklearn ensemble model
        sckit-learn ensemble model.

    X: 2D array-like
        The values to be used for predictions.

    y: 1D array-like
        Target.

    selector: sklearn model
        An estimator used to select the learners from the ensemble.

    threshold: str or float, default=None
        The threshold value to use for feature selection.

    norm_order: non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        threshold in the case where the coef_ attribute of the estimator is
        of dimension 2.

    max_estimators: int, default=None
        Maximum number of learners to keep in the ensemble. None means no
        limit.

    inplace: bool, default=False
        Whether to modify the ensemble inplace or not.

    Returns
    -------
    new_estimator: sklearn ensemble model, or None
        If inplace=False, returns a new ensemble model built from the original,
        user-provided ensemble, else returns None and the ensemble is modified
        inplace.
    """
    learners = estimator.estimators_.copy()
    n_estimators = len(learners)
    if n_estimators == 1:
        return estimator
    prediction_matrix = get_prediction_matrix(learners, X)
    drop_estimator = SelectFromModel(
        selector, threshold=threshold, norm_order=norm_order,
        max_features=max_estimators
    ).fit(prediction_matrix, y).get_support()

    if np.sum(drop_estimator) < n_estimators:
        learners = np.delete(learners, drop_estimator).tolist()
        if inplace:
            estimator.estimators_ = learners
            return None
        new_estimator = copy.deepcopy(estimator)
        new_estimator.estimators_ = learners
        return new_estimator
    return estimator


def sort_estimators(
    estimator, X, y, score_fn, *, inplace=False, scorer_kwargs=None
):
    """
    Sort learners from an ensemble by performance.

    Note: This function assumes that estimations made by each learner don't use
    estimations of other learners as input.

    Parameters
    ----------
    estimator: sklearn ensemble model
        Ensemble model

    X: 2D array-like
        The values to be used for predictions.

    y: 1D array-like
        Target.

    score_fn: callable
        Function to use for scoring performance.

    inplace: bool, default=False
        Whether to modify the ensemble inplace or not.

    scorer_kwargs: dict
        Additional parameters to be passed to the scorer.

    Returns
    -------
    new_estimator: sklearn ensemble model, or None
        If inplace=False, returns a new ensemble model built from the original,
        user-provided ensemble, else returns None and the ensemble is modified
        inplace.
    """
    scorer = make_scorer(estimator, score_fn, **scorer_kwargs)
    greater_is_better = scorer._sign > 0
    learners = estimator.estimators_.copy()
    n_estimators = len(learners)
    if n_estimators == 1:
        return estimator
    score_array = get_scores_array(learners, X, y, scorer)
    if greater_is_better:
        sorted_idxs = np.argsort(score_array)[::-1]
    else:
        sorted_idxs = np.argsort(score_array)
    if inplace:
        estimator.estimators_ = np.array(learners)[sorted_idxs].tolist()
        return None
    sorted_estimator = copy.deepcopy(estimator)
    sorted_estimator.estimators_ = np.array(learners)[sorted_idxs].tolist()
    return sorted_estimator


def select_best_sorted_chain(
    estimator, X, y, score_fn, inplace=False, scorer_kwargs=None
):
    """
    Select the best ensemble sorted chain.

    The function sorts the learners of the ensemble by performance, then
    selects the ordered subset that retains the best general performance.

    Note: This function assumes that estimations made by each learner don't use
    estimations of other learners as input.

    Parameters
    ----------
    estimator: sklearn ensemble model
        Ensemble model

    X: 2D array-like
        The values to be used for predictions.

    y: 1D array-like
        Target.

    score_fn: callableselect_important_estimators
        Function to use for scoring performance.

    inplace: bool, default=False
        Whether to modify the ensemble inplace or not.

    scorer_kwargs: dict
        Additional parameters to be passed to the scorer.

    Returns
    -------
    new_estimator: sklearn ensemble model, or None
        If inplace=False, returns a new ensemble model built from the original,
        user-provided ensemble, else returns None and the ensemble is modified
        inplace.
    """
    sorted_estimator = sort_estimators(
        estimator, X, y, score_fn, inplace=inplace
    )
    scorer = make_scorer(estimator, score_fn, **scorer_kwargs)
    estimators = sorted_estimator.estimators_.copy()
    n_estimators = len(estimators)
    if n_estimators == 1:
        return estimator

    cumscores = np.empty(shape=(n_estimators, ))
    for i in np.arange(1, n_estimators + 1):
        small_estimator = copy.deepcopy(sorted_estimator)
        small_estimator.estimators_ = estimators[:i]
        cumscores[i - 1] = scorer(small_estimator, X, y)

    best_n_estimators = np.argmax(cumscores[:, 1]) + 1
    if inplace:
        sorted_estimator.estimators_ = estimators[:best_n_estimators]
        return None
    best_estimator = copy.deepcopy(sorted_estimator)
    best_estimator.estimators_ = estimators[:best_n_estimators]
    return best_estimator


def select_best_ensemble(
    estimator, X, y, score_fn, *, uncorrelate=True,
    drop_noncritical=True, confidence_level=0.99, test="pearson",
    selector=Lasso(random_state=42), threshold=None, norm_order=1,
    max_estimators=None, inplace=False, test_kwargs=None, scorer_kwargs=None
):
    """
    Select the best ensemble from the methods specified by the user.

    The algorithm applies sequentially the functions uncorrelate_estimators,
    select_important_estimators and select_best_sorted_chain. However, the user
    can control whether uncorrelate_estimators and/or
    select_important_estimators are applied by changing the values of
    uncorrelate and/or drop_noncritical.

    Parameters
    ----------
    estimator: An scikit-learn ensemble estimator
        An ensemble estimator that trains each estimator independently from
        each other.

    X: 2D array-like
        The values to be used for predictions.

    y: 1D array-like
        Target.

    score_fn: callable
        Function to use for scoring performance.

    greater_is_better: bool, default=True
        Whether greater scores mean better performance.

    confidence_level: float, default=0.99
         Confidence level of the correlation hypothesis test.

    test: {"pearson", "spearman", "kendalltau", "weightedtau"},
    default="pearson"
        The hypothesis test to be applied to the ensemble to check for
        correlation, which are the ones implemented in the scipy.stats module.

    selector: sklearn model
        An estimator used to select the learners from the ensemble.

    threshold: str or float, default=None
        The threshold value to use for feature selection.

    norm_order: non-zero int, inf, -inf, default=1
        Order of the norm used to filter the vectors of coefficients below
        threshold in the case where the coef_ attribute of the estimator is
        of dimension 2.

    max_estimators: int, default=None
        Maximum number of learners to keep in the ensemble. None means no
        limit.

    inplace: bool, default=False
        Whether to modify the estimator in-place or not.

    test_kwargs: dict, default=None
        Additional parameters to be passed to the correlation test.

    scorer_kwargs: dict, default=None
        Additional parameters to be passed to the scorer.

    Returns
    -------
    new_estimator: sklearn ensemble model, or None
        If inplace=False, returns a new ensemble model built from the original,
        user-provided ensemble, else returns None and the ensemble is modified
        inplace.
    """
    if not inplace:
        new_estimator = copy.deepcopy(estimator)
        select_best_ensemble(
            new_estimator, X, y, score_fn, uncorrelate=uncorrelate,
            drop_noncritical=drop_noncritical,
            confidence_level=confidence_level, test=test,
            selector=selector, threshold=threshold, norm_order=norm_order,
            max_estimators=max_estimators, inplace=True,
            test_kwargs=test_kwargs, scorer_kwargs=scorer_kwargs
        )
        return new_estimator
    else:
        scorer = make_scorer(estimator, score_fn, **scorer_kwargs)
        if uncorrelate:
            uncorrelate_estimators(
                estimator, X, y, scorer, confidence_level=confidence_level,
                test=test, inplace=inplace, test_kwargs=test_kwargs
            )
        if drop_noncritical:
            select_important_estimators(
                estimator, X, y, selector=selector, threshold=threshold,
                norm_order=norm_order, max_estimators=max_estimators,
                inplace=inplace
            )
        select_best_sorted_chain(estimator, X, y, scorer, inplace=inplace)
        return None
