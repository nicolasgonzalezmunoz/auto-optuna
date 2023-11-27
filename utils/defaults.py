"""
Created on Sat Nov 25 17:29:24 2023, @author: nicolas.

This module implements functions to setup some default parameters.
"""
import optuna
from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def default_score_fn(estimator, score_fn=None):
    """
    Get a default score function.

    If score_fn is not None, then it is returned as is. Otherwise 'accuracy'
    is returned if estimator is a classifier and 'r2' in other case.

    estimator:
        An estimator.

    score_fn: str or callable, default=None
        If score_fn is None, then it is set with the defaults.
    """
    if score_fn is not None:
        return score_fn
    is_clf = is_classifier(estimator)
    if score_fn is None:
        if is_clf:
            score_fn = 'accuracy'
        else:
            score_fn = 'r2'
    return score_fn


def default_optuna_settings(
        n_trials, sampler=None, prune=False, pruner=None, seed=None
):
    """
    Get default settings for optuna study given the user inputs.

    Parameters
    ----------
    n_trials: int
        Number of trials for the study object.

    sampler: optuna.samplers.BaseSampler, default=None
        An optuna sampler. If None, then sampler is set to the defaults.

    prune: bool, default=False
        Whether to activate pruneing in the study or not. If a pruner is
        provided, this parameter is ignored.

    pruner: optuna.pruners.BasePruner
        An optuna pruner. If None, then pruner is set to the defaults.

    seed: int, default=None
        Sampler's seed.

    Returns
    -------
    sampler: optuna.samplers.BaseSampler
        If a sampler was provided, then the same sampler is returned. Else,
        sampler is set as TPESampler if n_trials<1000 or RandomSampler
        otherwise.

    pruner: optuna.pruners.BasePruner
        If a pruner was provided, then the same pruner is returned. Else,
        if prune is False, then a NopPruner object is returned. Otherwise,
        pruner is set as HyperbandPruner if n_trials<1000 or MedianPruner
        otherwise.
    """
    if sampler is None:
        if n_trials < 1000:
            sampler = optuna.samplers.TPESampler(
                seed=seed, multivariate=True
            )
        else:
            sampler = optuna.samplers.RandomSampler(seed=seed)
    if pruner is None:
        if prune:
            if n_trials < 1000:
                pruner = optuna.pruners.HyperbandPruner()
            else:
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        else:
            pruner = optuna.pruners.NopPruner()
    return sampler, pruner


def default_cv_splitter(estimator, cv, seed=None):
    """
    Return a default splitter.

    If cv is a callable, the function returns it as is. If estimator is a
    classifier, the function returns a StratifiedShuffleSplit object,
    otherwise, returns a ShuffleSplit instance. In both cases, the splitter is
    set with n_splits=cv, test_size=1/cv and random_state=seed.
    """
    if not isinstance(cv, float):
        return cv

    cv = int(cv)
    is_clf = is_classifier(estimator)
    if is_clf:
        splitter = StratifiedShuffleSplit(
            n_splits=cv, test_size=1/cv, random_state=seed
        )
    else:
        splitter = ShuffleSplit(
            n_splits=cv, test_size=1/cv, random_state=seed
        )
    return splitter
