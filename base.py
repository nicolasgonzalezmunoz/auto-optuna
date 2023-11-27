"""
Created on Sat Oct 7 11:39:40 2023, @author: nicolas.

This file implements base classes for model optimization.
"""
import inspect
import optuna
from sklearn.linear_model import Lasso
from .utils.defaults import default_optuna_settings, cross_validation_score
from .utils.trials import best_params_as_dict
from .utils.metrics import make_scorer
from .utils.trees import prune_tree
from .utils import ensembles


class BaseOptimizer:
    """
    Base class for all model optimizers.

    Each model optimizer optimizes a given type of model from the sklearn
    library and the most common ML frameworks, like CatBoost and XGBoost.

    Parameters
    ----------
    estimator:
        A machine-learning model to be optimized.

    seed: int, default=None
        Random seed.

    estimator_threads: int, default=1
        Number of CPUs that the estimator will use.

    optimizer_threads: int, default=1
        Number of CPUs to use on the optimization task. if -1, then uses all
        the available CPUs.

    Attributes
    ----------
    study: optuna.study.Study
        A study object containing information about the last optimization task.

    direction: {"minimize", "maximize"}
        The direction of the last optimization task, which is determined using
        the score function used on the optimization task.

    optimized_estimator: sklearn.BaseEstimator
        An optimized instance of an sklearn estimator.
    """

    __slots__ = [
        '_estimator', 'estimator_threads', 'seed', '_optimizer_threads',
        '_study', '_direction', '_optimized_estimator'
    ]

    def __init__(
            self, estimator, estimator_threads=1, optimizer_threads=1,
            seed=None
    ):
        self._estimator = estimator
        self.estimator_threads = estimator_threads
        self._optimizer_threads = optimizer_threads
        self.seed = seed

    @property
    def estimator(self):
        """Get the estimator class of this object."""
        return self._estimator

    @property
    def optimizer_threads(self):
        """Get number of threads where the optimizer runs."""
        return self.optimizer_threads

    @property
    def study(self):
        """Get the study of the last optimization task."""
        return self._study

    @property
    def direction(self):
        """Get the direction of the last optimization task."""
        return self._direction

    @property
    def optimized_estimator(self):
        """Get an optimized DecisionTreeClassifier instance."""
        return self._optimized_estimator

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
        pass

    def get_optimal_params(self, alpha=0.01):
        """
        Get a dict with the optimal parameters for a given estimator.

        Each optimizer class implements its own method.
        """
        # Get optimized parameters
        params = best_params_as_dict(self.study, self.direction, alpha)

        # Get user-defined parameters
        for member in inspect.getmembers(self):
            if not member[0].startswith('_'):
                if not inspect.ismethod(member[1]):
                    if member[0] != 'seed':
                        params[member[0]] = member[1]

        if hasattr(self.estimator(), "random_state"):
            params["random_state"] = self.seed
        elif hasattr(self.estimator(), "seed"):
            params["random_seed"] = self.seed

        if hasattr(self.estimator(), "n_jobs"):
            params["n_jobs"] = self.estimator_threads
        elif hasattr(self.estimator(), "thread_count"):
            params["thread_count"] = self.estimator_threads
        return params

    def make_objective(
            self, X, y, scorer, *, eval_set=None, param_grid=None,
            cv=5, cv_mode='normalize', **kwargs
    ):
        """
        Create a default objective to be optimized by optuna.

        This is the objective that most optimizers use. However, some
        exceptions like RandomForestClassifierOptimizer can implement its own
        objective function.

        Parameters
        ----------
        estimator: A Machine-learning model
            The estimator is set internally by each class.

        X: 2D array-like
            Training dataset.

        y: array-like
            Training target.

        scorer: callable
            A callable with the signature (estimator, X, y).

        eval_set: list of tuple, default=None
            Evaluation set in the format [(X_val, y_val)]

        param_grid: ParameterGrid
            An instance to build the parameter dictionary.

        cv: int or sklearn splitter, default=5
            Determine how the dataset is splitted on cross-validation.

        kwargs: dict
            Additional arguments to be passed to the function.
        """
        def objective(trial):
            if param_grid is None:
                params = self._get_default_search_grid(trial, X, y)
            else:
                params = param_grid.get_params_dict(trial, X, y)

            model = self.estimator(**params)
            if eval_set is None:
                score = cross_validation_score(
                    model, X, y, scorer, cv=cv, seed=self.seed,
                    mode=cv_mode
                )
            else:
                model.fit(X, y)
                score = scorer(model, eval_set[0][0], eval_set[0][1])
            return score
        return objective

    def _run_trials(
            self, X, y, eval_set=None, param_grid=None, *, cv_mode='normalize',
            score_fn=None, cv=5, n_trials=100, timeout=300, sampler=None,
            pruner=None, study_name=None, load_if_exists=False, prune=False,
            pipeline=None, scorer_kwargs=None, **kwargs
    ):
        """
        Run optimization trials.

        Model params
        ------------
        estimator: class
            The estimator to be optimized.

        X, y: array-like
            Training set.

        eval_set: list of tuple, default=None
            Validation set, in the format [(X_val, y_val)].

        Cross-validation-only params
        ----------------------------
        score_fn: str or callable, default=None
            score function to use for validation. Can be any
            sklearn-compatible object. If score_fn=None, then score_fn is set
            as 'accuracy' if the estimator to optimize is a classifier, and as
            'r2' if isn't.

        cv: int or scikit-learn splitter, defaut=5
            Cross-validation splitter. Can be a integer or any
            sklearn-compatible splitter.

        cv_mode: {"mean", "normalize"}, default="normalize"
            How to compute the cross-validation score. If cv_mode="mean", then
            the mean of all the cv scores is computed, otherwise the mean is
            multiplied by a factor involving the cv's standard deviation to
            account for model stability.

        Optimizer params
        ----------------
        n_trials: int, default=100
            Number of trials to run on the optimization task.

        timeout: float, default=300
            Time (in seconds) after which the optimization task is
            interrupted.

        sampler: optuna.samplers.BaseSampler or None, default=None
            Hyperparameter sampling strategy. Can be either of the
            ones available in Optuna. If None, sets a RandomSampler.

        pruner: optuna.pruners.BasePruner or None, default=None
            A pruner object that decides early stopping of unpromising
            trials.

        study_name: str or None, default=None
            Name of the study.

        load_if_exists: bool, default=False
            Flag to control the behavior to handle a conflict
            of study names. In the case where a study named study_name
            already exists in the storage, a DuplicatedStudyError is
            raised if load_if_exists is set to False. Otherwise, the
            creation of the study is skipped, and the existing one is
            returned.

        Additional params
        ----------------
        pipeline: A scikit-learn object that implements fit and transform
        methods, or None. Default=None
            A transformer or sequence of transformers to be applied to the
            dataset.

        scorer_kwargs: dict or None
            Additional parameters to pass to the score function.

        kwargs: dict
            Additional parameters to be used to build the objective function.

        Returns
        -------
        None
        """
        # Setting the logging level WARNING, the INFO logs are suppressed.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scorer = make_scorer(self.estimator, score_fn, **scorer_kwargs)

        if scorer._sign > 0:
            self._direction = "maximize"
        else:
            self._direction = "minimize"

        if pipeline is not None:
            copy_X = pipeline.fit_transform(X, y)
            if eval_set is not None:
                X_val = pipeline.transform(eval_set[0][0], eval_set[0][1])
        else:
            copy_X = X
            if eval_set is not None:
                X_val = eval_set[0][0]
                y_val = eval_set[0][1]

        objective = self.make_objective(
            self.estimator, copy_X, y, scorer, eval_set=[(X_val, y_val)],
            param_grid=param_grid, cv=5, **kwargs
        )

        # Set optimizer parameters
        sampler, pruner = default_optuna_settings(
            n_trials, sampler=sampler, prune=prune, pruner=pruner,
            seed=self.seed
        )

        study = optuna.create_study(
            direction=self.direction, sampler=sampler, study_name=study_name,
            pruner=pruner, load_if_exists=load_if_exists
        )
        study.optimize(
            objective, n_trials=n_trials, timeout=timeout,
            n_jobs=self.optimier_threads
        )
        self._study = study

    def optimize(
            self, X, y, eval_set=None, param_grid=None, *,
            score_fn=None, cv=5, n_trials=100, timeout=300, sampler=None,
            pruner=None, study_name=None, load_if_exists=False, prune=False,
            pipeline=None, alpha=0.01, scorer_kwargs=None, **kwargs
    ):
        """
        Set a model's hyperparameters using the trial with best score.

        Since two models could have similar scores but different training speed
        and cross-validation stability, is recommended to use the optimize and
        setup_model_from_trial methods instead.

        Model params
        ------------
        estimator: machine-learning model
            The estimator to optimize

        X, y: array-like
            Training set.

        eval_set: list of tuple
            Validation set in the format [(X_val, y_val)].

        Validation params
        ----------------------------
        score_fn: str or callable, default=None
            score function to use for validation. Can be any
            sklearn-compatible object. If score_fn=None, then score_fn is set
            as 'accuracy' if the estimator to optimize is a classifier, and as
            'r2' if isn't.

        cv: int or scikit-learn splitter, defaut=5
            Cross-validation splitter. Can be a integer or any
            sklearn-compatible splitter.

        Optimizer params
        ----------------
        n_trials: int, default=100
            Number of trials to run on the optimization task.

        timeout: float, default=300
            Time (in seconds) after which the optimization task is
            interrupted.

        sampler: optuna.samplers.BaseSampler or None, default=None
            Hyperparameter sampling strategy. Can be either of the
            ones available in Optuna. If None, sets a RandomSampler.

        pruner: optuna.pruners.BasePruner or None, default=None
            A pruner object that decides early stopping of unpromising
            trials.

        study_name: str or None, default=None
            Name of the study.

        load_if_exists: bool, default=False
            Flag to control the behavior to handle a conflict
            of study names. In the case where a study named study_name
            already exists in the storage, a DuplicatedStudyError is
            raised if load_if_exists is set to False. Otherwise, the
            creation of the study is skipped, and the existing one is
            returned.

        Additional params
        ----------------
        alpha: float, default=0.01
            Performance loss threshold.

        pipeline: A scikit-learn object that implements fit and transform
        methods, or None. Default=None
            A transformer or sequence of transformers to be applied to the
            dataset.

        scorer_kwargs: dict or None
            Additional parameters to pass to the score function.

        Return
        ------
        None
        """
        self._run_trials(
            X, y, eval_set=eval_set, score_fn=score_fn, cv=cv,
            n_trials=n_trials, timeout=timeout, sampler=sampler, pruner=pruner,
            study_name=study_name, load_if_exists=load_if_exists, prune=prune,
            pipeline=pipeline, scorer_kwargs=scorer_kwargs, **kwargs
        )

        params = self.get_optimal_params(alpha)
        self._optimized_model = self.estimator(**params)


class BaseTreeOptimizer(BaseOptimizer):
    """
    Modifies the base optimizer to include prunning for decision trees.

    Parameters
    ----------
    estimator:
        A machine-learning model to be optimized.

    seed: int, default=None
        Random seed.

    estimator_threads: int, default=1
        Number of CPUs that the estimator will use.

    optimizer_threads: int, default=1
        Number of CPUs to use on the optimization task. if -1, then uses all
        the available CPUs.

    Attributes
    ----------
    study: optuna.study.Study
        A study object containing information about the last optimization task.

    direction: {"minimize", "maximize"}
        The direction of the last optimization task, which is determined using
        the score function used on the optimization task.

    optimized_estimator: sklearn.BaseEstimator
        An optimized instance of an sklearn estimator.
    """

    def make_objective(
            self, X, y, scorer, *, eval_set=None, param_grid=None,
            cv=5, cv_mode='normalize', post_pruning=False, **kwargs
    ):
        """
        Create a default objective to be optimized by optuna.

        General objective for decision trees. It's almost the same as the one
        on BaseOptimizer, but adds post-pruning to improve performance before
        scoring the estimator.

        Parameters
        ----------
        estimator: A Machine-learning model
            The estimator is set internally by each class.

        X: 2D array-like
            Training dataset.

        y: array-like
            Training target.

        scorer: callable
            A callable with the signature (estimator, X, y).

        eval_set: list of tuple, default=None
            Evaluation set in the format [(X_val, y_val)]

        param_grid: ParameterGrid
            An instance to build the parameter dictionary.

        cv: int or sklearn splitter, default=5
            Determine how the dataset is splitted on cross-validation.

        kwargs: dict
            Additional arguments to be passed to the function.
        """
        def objective(trial):
            if param_grid is None:
                params = self._get_default_search_grid(trial, X, y)
            else:
                params = param_grid.get_params_dict(trial, X, y)

            if post_pruning:
                model = prune_tree(
                    self.estimator, params, X, y, scorer,
                    direction=self.direction, eval_set=eval_set, cv=cv,
                    seed=self.seed, cv_mode=cv_mode
                )
            else:
                model = self.estimator(**params)
            if eval_set is None:
                score = cross_validation_score(
                    model, X, y, scorer, cv=cv, seed=self.seed,
                    mode=cv_mode
                )
            else:
                model.fit(X, y)
                score = scorer(model, eval_set[0][0], eval_set[0][1])
            return score
        return objective


class BaseIndependentLearnerOptimizer(BaseOptimizer):
    """
    Base optimizer for ensembles that train independent learners.

    An ensemble is considered to be conformed of independent learners if all
    the learners are trained independently from each other and all of them
    have the same weight. That's the case, for example, of Random Forest
    algorithms.

    Parameters
    ----------
    estimator:
        A machine-learning model to be optimized.

    seed: int, default=None
        Random seed.

    estimator_threads: int, default=1
        Number of CPUs that the estimator will use.

    optimizer_threads: int, default=1
        Number of CPUs to use on the optimization task. if -1, then uses all
        the available CPUs.

    Attributes
    ----------
    study: optuna.study.Study
        A study object containing information about the last optimization task.

    direction: {"minimize", "maximize"}
        The direction of the last optimization task, which is determined using
        the score function used on the optimization task.

    optimized_estimator: sklearn.BaseEstimator
        An optimized instance of an sklearn estimator.
    """

    def make_objective(
            self, X, y, scorer, *, eval_set=None, param_grid=None,
            cv=5, cv_mode='normalize', uncorrelate=False,
            select_best_learners=False, select_best_chain=False,
            confidence_level=0.99, test="spearman", selector=Lasso(),
            threshold=None, norm_order=1, test_kwargs, **kwargs
    ):
        """
        Create a default objective to be optimized by optuna.

        General objective for some ensemble models. It's similar to the one
        on BaseOptimizer, but applies some statistical techniques to improve
        the model before scoring it.

        Statistical tools implemented on this method include:
            - Dropping correlated learners.
            - Selecting the most important learners using a model-based
            approach.
            - Sorting learners by performance and selecting the best learner
            chain.

        Note: The statistical techniques are applied only if eval_set is not
        None, since cross-validation fits the model on each fold, undoing the
        changes made when computing the scores.

        Parameters
        ----------
        estimator: A Machine-learning model
            The estimator is set internally by each class.

        X: 2D array-like
            Training dataset.

        y: array-like
            Training target.

        scorer: callable
            A callable with the signature (estimator, X, y).

        eval_set: list of tuple, default=None
            Evaluation set in the format [(X_val, y_val)]

        param_grid: ParameterGrid
            An instance to build the parameter dictionary.

        cv: int or sklearn splitter, default=5
            Determine how the dataset is splitted on cross-validation.

        cv_mode: {"mean", "normalize"}, default="normalize"
            How to compute cv score. If "normalize", then normalize the mean
            score multiplying by a factor to account for model stability.

        uncorrelate: bool, default=False
            Decide whether to drop correlated learners or not.

        select_best_learners: bool, default=False
            Whether to use model-based approach to select the most important
            learners or not make the selection at all.

        select_best_chain: bool, default=False
            If True, sort the learners by performance and choose the best
            chain.

        confidence_level: float, default=0.99
             Confidence level of the correlation hypothesis test.

        test: {"pearson", "spearman", "kendalltau", "weightedtau"},
        default="spearman"
            The hypothesis test to be applied to the estimators, which are the
            ones implemented in the scipy.stats module.

        threshold: str or float, default=None
            The threshold value to use for feature selection.

        norm_order: non-zero int, inf, -inf, default=1
            Order of the norm used to filter the vectors of coefficients below
            threshold in the case where the coef_ attribute of the estimator is
            of dimension 2.

        test_kwargs: dict, default=None
            Additional parameters to be passed to the correlation test.

        kwargs: dict
            Additional arguments to be passed to the function.
        """
        def objective(trial):
            if param_grid is None:
                params = self._get_default_search_grid(trial, X, y)
            else:
                params = param_grid.get_params_dict(trial, X, y)

            model = self.estimator(**params)
            if eval_set is None:
                score = cross_validation_score(
                    model, X, y, scorer, cv=cv, seed=self.seed,
                    mode=cv_mode
                )
            else:
                model.fit(X, y)
                X_eval = eval_set[0][0]
                y_eval = eval_set[0][1]
                if uncorrelate:
                    ensembles.uncorrelate_estimators(
                        model, X_eval, y_eval, scorer,
                        confidence_level=confidence_level, test=test,
                        test_kwargs=test_kwargs, inplace=True
                    )
                if select_best_learners:
                    ensembles.select_important_estimators(
                        model, X_eval, y_eval, selector=selector,
                        threshold=threshold, norm_order=norm_order,
                        inplace=True
                    )
                if select_best_chain:
                    ensembles.select_best_sorted_chain(
                        model, X_eval, y_eval, scorer, inplace=True
                    )
                score = scorer(model, X_eval, y_eval)
            return score
        return objective
