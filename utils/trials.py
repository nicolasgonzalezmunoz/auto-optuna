"""
Created on Sat Nov 25 17:49:04 2023, @author: nicolas.

Implements some utilitary functions related to optuna trials.
"""


def study_to_frame(study, direction):
    """
    Generate a Pandas DataFrame from a study object.

    The DataFrame contains information about each optimization trial,
    sorted by its corresponding score, according to the optimization
    direction.

    Parameters
    ----------
    study: optuna.study.Study
        The study to convert to a dataframe.

    direction: {"minimize", "maximize"}
        Direction of the optimization task.

    Returns
    -------
    study_results: pandas.DataFrame
        DataFrame containing information about the trials.
    """
    study_results = study.trials_dataframe()
    study_results.sort_values(
        'value', ascending=direction == 'minimize', inplace=True
    )
    return study_results


def get_best_trial(study, direction, *, alpha=0.01):
    """
    Get the best trial from the study.

    The selection criterion accepts a performance loss with respect of the
    best trial of at most alpha%.

    Parameters
    ----------
    study: optuna.study.Study
        The study to convert to a dataframe.

    direction: {"minimize", "maximize"}
        Direction of the optimization task.

    alpha: float, default=0.01
        performance loss threshold.

    Returns
    -------
    best_trial: pandas.Series
        A series with information on the selected best trial.
    """
    study_results = study_to_frame()
    if direction == "minimize":
        best_score = study_results.value.min()
        if best_score >= 0:
            trial_numbers = study_results.value <= (1 + alpha)*best_score
        else:
            trial_numbers = study_results.value <= (1 - alpha)*best_score
    else:
        best_score = study_results.value.max()
        if best_score >= 0:
            trial_numbers = study_results.value >= (1 - alpha)*best_score
        else:
            trial_numbers = study_results.value >= (1 + alpha)*best_score

    best_results = study_results.loc[trial_numbers, :]
    fastest_trial = best_results.duration == best_results.duration.min()
    best_trial = best_results.loc[fastest_trial, :]
    if len(best_trial.shape) != 1:
        best_trial = best_trial.iloc[0]
    return best_trial


def best_params_as_dict(study, direction, *, alpha=0.01):
    """
    Return a dictionary in the format {param_name: best_value}.

    study: optuna.study.Study
        The study to convert to a dataframe.

    direction: {"minimize", "maximize"}
        Direction of the optimization task.

    alpha: float, default=0.01
        performance loss threshold.

    Returns
    -------
    params: dict
        A dictionary with the parameters.
    """
    best_trial = get_best_trial(study, direction, alpha=alpha)
    params = {}
    for column in best_trial.index:
        if 'params_' in column:
            params[column[7:]] = best_trial.loc[column]
    return params
