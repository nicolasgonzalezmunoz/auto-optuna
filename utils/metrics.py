"""
Created on Sat Nov 18 19:12:51 2023, @author: nicolas.

This module provides some utilitary functions to define metrics and general
cross-validation strategies.
"""
import copy
import numpy as np
import pandas as pd
from inspect import signature
from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced
from .defaults import default_cv_splitter, default_score_fn


def make_scorer(estimator, score_fn=None, **scorer_kwargs):
    """
    Build a scorer with signature (estimator, X, y).

    If score_fn is None, then it is set with the defaults.

    Parameters
    ----------
    estimator:
        An estimator

    score_fn: str or callable, default=None
        Score function to be converted to a scorer.

    scorer_kwargs: dict
        Additional arguments to be passed to the scorer.
    """
    score_fn = default_score_fn(estimator, score_fn)
    scorer = metrics.get_scorer(score_fn)
    scorer = metrics.make_scorer(scorer, **scorer_kwargs)


def cross_validation_score(
        estimator, X, y, *, score_fn=None, cv=5, seed=None, mode="normalize",
        **scorer_kwargs
):
    """
    Produce a cross-validation score using the user's settings.

    The function returns an array of cv scores if mode='raw', and the mean of
    the said array if mode='mean'. If mode='normalize', the mean cv score is
    normalized by its standard deviation to account for training stability.

    Parameters
    ----------
    estimator: scikit-learn-compatible estimator
        A scikit-learn-compatible model

    X: 2D array-like
        Design matrix.

    y: 1D array-like
        Target.

    score_fn: str or callable, default=None
        A str or callable that can be converted to a sklearn scorer.

    cv: int or sklearn cross-validation iterator, default=5
        Indicates how X and y will be splitted for cross-validation. If an int,
        StratifiedShuffleSplitter will be used classifiers and ShuffleSplitter
        in other case, with cv number of splits.

    seed: int, default=None
        Random seed

    mode: {'raw', 'mean', 'normalize'}, default='normalize'
        How the cv scores will be outputed. If mode='raw', an array is returned
        with the scores of each iteration, and the mean of the said array if
        mode='mean'. If mode='normalize', the mean cv score is normalized by
        its standard deviation to account for training stability.

    scorer_kwargs: dict
        Additional arguments to be passed to the scorer.
    """
    estimator_copy = copy.deepcopy(estimator)
    scorer = make_scorer(estimator_copy, score_fn, **scorer_kwargs)
    greater_is_better = scorer._sign > 0
    has_eval_set = 'eval_set' in str(signature(estimator_copy.fit))
    splitter = default_cv_splitter(estimator_copy, cv, seed)
    cv_scores = []
    for train_idx, valid_idx in splitter.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx, :]
        y_valid = y.iloc[valid_idx]
        if has_eval_set:
            estimator_copy.fit(
                X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False
            )
        else:
            estimator_copy.fit(X_train, y_train)
        cv_scores.append(
            scorer(estimator_copy, X_valid, y_valid)
        )
    cv_scores = np.array(cv_scores)
    if mode == "raw":
        return cv_scores
    mean_score = cv_scores.mean()
    if mode == "mean":
        return mean_score
    cv_std = cv_scores.std()
    mult = mean_score < 0 and greater_is_better
    mult = mult or (mean_score >= 0 and not greater_is_better)
    if mult:
        score = mean_score * cv_std
    else:
        score = mean_score / cv_std
    return score


def classification_report(
        self, y_true, y_pred, *, labels=None, target_names=None,
        sample_weight=None, digits=2, alpha=0.1, output_mode="dataframe",
        zero_division='warn', imbalanced=False
):
    """
    Make a report for the estimator with several classification metrics.

    If imbalanced is False, uses the classification_report function from
    scikit-learn to generate the report, else, uses the
    classification_report_imbalanced from imbalanced-learn.

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred: 2d array-like
        Estimated targets as returned by a classifier.

    labels: array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.

    target_names: list of str of shape (n_labels,), default=None
        Optional display names matching the labels (same order).

    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.

    digits: int, default=2
        Number of digits for formatting output floating point values. When
        output_dict is True, this will be ignored and the returned values
        will not be rounded.

    alpha: float, default=0.1
        Weighting factor.

    output_mode: {"text", "dict", "dataframe"}, default="dataframe"
        How the metrics are reported.

    zero_division“warn” or {0, 1}, default=”warn”
        Sets the value to return when there is a zero division. If set to
        “warn”, this acts as 0, but warnings are also raised.

    imbalanced: bool, default=False
        If True, return specialized metrics for imbalanced data.

    Returns
    -------
    report: string / dict
        Text summary of the precision, recall and F1 score for each class.
        If imbalanced=True, specificity, geometric mean, and index
        balanced accuracy are also returned. Dictionary returned if
        output_dict is True.
    """
    if output_mode in {"dict", "dataframe"}:
        output_dict = True
    if imbalanced:
        report = classification_report_imbalanced(
            y_true, y_pred, labels=labels, target_names=target_names,
            sample_weight=sample_weight, digits=digits, alpha=alpha,
            output_dict=output_dict, zero_division=zero_division
        )
    else:
        report = metrics.classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            sample_weight=sample_weight, digits=digits,
            output_dict=output_dict, zero_division=zero_division
        )
    if output_mode == "dataframe":
        report = pd.DataFrame.from_dict(report, orient='index')
    return report


def _regression_metrics_dict(y_true, y_pred, digits=2):
    """
    Generate a dict with regression metrics.

    The metrics generated are R2, max error, mean absolute error, mean squared
    error, mean absolute percentage error and median absolute error.

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred: 2d array-like
        Estimated targets as returned by a classifier.

    digits: int, default=2
        Number of digits for formatting output floating point values.
    """
    metrics_dict = {
        'R2': np.round(metrics.r2_score(y_true, y_pred), digits),
        'ME': np.round(metrics.max_error(y_true, y_pred), digits),
        'MAE': np.round(metrics.mean_absolute_error(y_true, y_pred), digits),
        'MSE': np.round(metrics.mean_squared_error(y_true, y_pred), digits),
        'MAPE': np.round(
            metrics.mean_absolute_percentage_error(y_true, y_pred), digits
        ),
        'MeAE': np.round(metrics.median_absolute_error(y_true, y_pred), digits)
    }
    return metrics_dict


def regression_report(y_true, y_pred, *, bins='doane', digits=2):
    """
    Generate a report with regression metrics in a histogram-like way.

    The target and predictions are binned to generate histogram-like data. The
    metrics are then computed over each bin, as well as at the macro level.

    The metrics generated are R2, max error, mean absolute error, mean squared
    error, mean absolute percentage error and median absolute error.

    Parameters
    ----------
    y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred: 2d array-like
        Estimated targets as returned by a classifier.

    bins: int or sequence of scalars or str, default='doane'
        Same description as in numpy.histogram_bin_edges.

    digits: int, default=2
        Number of digits for formatting output floating point values.

    Returns
    -------
    report: pandas.DataFrame
        A datafrane with the reported metrics.
    """
    report = dict()
    report["macro"] = _regression_metrics_dict(y_true, y_pred, digits)
    bin_edges = np.histogram_bin_edges(y_true, bins=bins)
    for i, bin_low in enumerate(bin_edges[:-1]):
        bin_up = bin_edges[i + 1]
        bin_tag = f"[{np.round(bin_low, digits)}, {np.round(bin_up, digits)})"
        y_true_bin = y_true.loc[(y_true >= bin_low) & (y_true < bin_up)]
        y_pred_bin = y_pred.loc[(y_true >= bin_low) & (y_true < bin_up)]
        report[bin_tag] = _regression_metrics_dict(
            y_true_bin, y_pred_bin, digits
        )
    report = pd.DataFrame.from_dict(report, orient='index')
    return report
