"""
Created on Fri Nov 24 21:03:23 2023, @author: nicolas.

Implements the ParamneterGrid class to configure Optuna search spaces.
"""
import numpy as np
import warnings


class ParameterGrid:
    """
    A class to build a parameter grid search for optuna.

    This class contain information about parameter names, its space search type
    and the values the parameter can take.

    Parameters
    ----------
    params_dict: dict of dicts, default=None
        Dictionary on a JSON style, following the format {<param_name>: dict},
        where the inner dict has the following fields:
            - type: {"categorical", "float", "int", "fixed"}
                The type of the search space.
            - grid: Any
                The posible values or range of values the parameter can take.
                If type!="fixed", a list with at least two entries is
                expected. If type="fixed", a single value is expected. If type
                is "int" or "float", the minimum and maximum of grid are taken
                as the parameter range.
            - step: int or float, default=None
                Steps on the search grid. If type='int', the step will be
                converted to int.
            - log: bool
                Whether to sample values in a log scale or not. This field is
                optional, but when missing and type is "int" or "float", log
                is set to False as default. Ignored when type is "categorical"
                or "fixed".
    """

    __slots__ = ['params_dict']

    def __init__(
            self, params_dict=None
    ):
        self._validate_params_dict(params_dict)
        self.params_dict = params_dict

    def _validate_param(
            self, param_name, param_type, param_grid, step=None,
            is_log_scale=False
    ):
        """
        Validate the format of a single parameter.

        Parameters
        ----------
        param_name: str
            Name of the parameter.

        param_type: {"categorical", "float", "int", "fixed"}
            Type of the parameter's search space.

        param_grid: Any
            Posible values the parameter can take.

        step: int or float, default=None
            Steps on the search grid.

        is_log_scale: bool, default=False
            If the search is performed in the log scale.
        """
        types = ["categorical", "float", "int", "fixed"]
        if not isinstance(param_name, str):
            msg = (
                "param_name expected to be of type str, got "
                f"{type(param_name)} instead."
            )
            raise ValueError(msg)
        if param_type not in types:
            msg = (
                f"Invalid type for parameter {param_name}. "
                "Type should be one of 'categorical', 'float', 'int' "
                f"or 'fixed'. Got {type(param_type)} instead."
            )
            raise ValueError(msg)
        if step is not None:
            if param_type == "int" and (not isinstance(step, float)):
                msg = (
                    "Invalid value for 'step' field. Expected value of "
                    "type float or None. Got {type(step)} instead."
                )
                raise ValueError(msg)
        if is_log_scale is not None:
            if not isinstance(is_log_scale, bool):
                msg = (
                    "Invalid value for 'log' field. Expected value of "
                    "type bool or None. Got {type(is_log_scale)} instead."
                )
                raise ValueError(msg)
            if step is not None:
                if step != 1 and is_log_scale:
                    msg = (
                        "The step != 1 and log arguments cannot be used at the"
                        " same time. To set the log argument to True, set the"
                        " step argument to 1."
                    )
                    raise ValueError(msg)

    def _validate_params_dict(self, params_dict):
        """
        Validate the format a parameter dict.

        Parameters
        ----------
        params_dict: dict
            Dictionary to validate.
        """
        if not isinstance(params_dict, dict):
            msg = (
                "'params_dict' should be of type dict. Got "
                f"{type(params_dict)} instead."
            )
            TypeError(msg)
        types = ["categorical", "float", "int", "fixed"]
        for param_name, param_info in params_dict.items():
            if not isinstance(param_name, str):
                msg = (
                    "param_name expected to be of type str, got "
                    f"{type(param_name)} instead."
                )
                raise ValueError(msg)
            if param_info.get("type") is None:
                msg = (
                    f"'type' field missing for parameter '{param_name}'."
                )
                raise ValueError(msg)
            if param_info["type"] not in types:
                msg = (
                    f"Invalid type for parameter {param_name}. "
                    "Type should be one of 'categorical', 'float', 'int' "
                    f"or 'fixed'. Got {type(param_info['type'])} instead."
                )
                raise ValueError(msg)
            if param_info.get("step") is not None:
                if not isinstance(param_info["step"], (float, None)):
                    msg = (
                        "Invalid value for 'step' field. Expected value of "
                        "type float or None. Got {type(param_info['step'])} "
                        "instead."
                    )
                    raise ValueError(msg)
            if param_info.get("log") is not None:
                if not isinstance(param_info["log"], (bool, None)):
                    msg = (
                        "Invalid value for 'log' field. Expected value of "
                        "type bool or None. Got {type(param_info['log'])} "
                        "instead."
                    )
                    raise ValueError(msg)
                if param_info.get("step") is not None:
                    if param_info["step"] != 1 and param_info["log"]:
                        msg = (
                            "The step != 1 and log arguments cannot be used at"
                            " the same time. To set the log argument to True,"
                            " set the step argument to 1 on parameter "
                            f"{param_name}."
                        )
                        raise ValueError(msg)

    def update_param(
            self, param_name, param_type, param_grid, step=None,
            is_log_scale=False
    ):
        """
        Update a single parameter to the ParameterGrid object.

        The update is made by either adding a new parameter or modifying an
        existing one.

        Parameters
        ----------
        param_name: str
            Name of the parameter to update.

        param_type: {"int", "float", "categorical", "fixed"}
            Type of the parameter to update.

        step: int or float, default=None
            Steps on the search grid.

        param_grid: list, int, float or str
            Posible values of the parameter.

        is_log_scale: list of bool and/or None
            If the sampling on the search space should be in log scale or not.
        """
        self._validate_param(
            param_name, param_type, param_grid, step, is_log_scale
        )
        if self.params_dict is None:
            self.params_dict = {}
        if param_type in ["float", "int"]:
            if step is None:
                param_dict = {
                    "type": param_type, "grid": param_grid, "log": is_log_scale
                }
            else:
                if param_type == "int":
                    step = int(step)
                param_dict = {
                    "type": param_type, "grid": param_grid, "step": step,
                    "log": is_log_scale
                }
        else:
            param_dict = {
                "type": param_type, "grid": param_grid
            }
        self.params_dict[param_name] = param_dict

    def update_params(
            self, param_names, param_types, param_grids, steps=None,
            is_log_scale=None
    ):
        """
        Update a several parameters to the ParameterGrid object.

        The update is made by either adding new parameters and/or modifying
        existing ones.

        Parameters
        ----------
        param_names: list of str
            Name of the parameter to update.

        param_types: list of {"int", "float", "categorical", "fixed"}
            Type of the parameter to update.

        param_grids: list of list, int, float and/or str
            Posible values of the parameter.

        is_log_scale: list of bool and/or None
            If the sampling on the search space should be in log scale or not.
        """
        n_params = len(param_names)
        for i in np.arange(n_params):
            if is_log_scale is None:
                log_scale = False
            else:
                log_scale = is_log_scale[i]
            if steps is None:
                step = None
            else:
                step = steps[i]
            self.update_param(
                param_names[i], param_types[i], param_grids[i], step, log_scale
            )

    def update_from_dict(self, params_dict):
        """
        Update parameters on the ParameterGrid instance using a dictionary.

        params_dict: dict
            A dictionary with the format
            {<name>: {'type': <str>, 'grid': <values>, 'log': <bool>}}.
        """
        self._validate_params_dict(params_dict)
        for param_name, param_info in params_dict.items():
            if param_info.get("step") is None:
                step = None
            if param_info.get("log") is None:
                param_info["log"] = False
            self.update_param(
                param_name, param_info["type"],
                param_info["grid"], step, param_info["log"]
            )

    def remove_param(self, param_name):
        """
        Remove parameter from the ParameterGrid instance.

        Parameters
        ----------
        param_name: str
            Name of the parameter to remove.
        """
        if param_name not in self.params_dict.keys():
            warnings.warn(f"Parameter {param_name} not found.")
        else:
            self.params_dict.pop(param_name)

    def remove_params(self, param_names):
        """
        Remove parameters from the ParameterGrid instance.

        Parameters
        ----------
        param_names: list of str
        Names of the parameters to remove.
        """
        for param_name in param_names:
            self.remove_param(param_name)

    def get_params_dict(self, trial, X=None, y=None):
        """
        Generate a dict of parameters to be passed to an estimator.

        Parameters
        ----------
        trial: optuna.trial.Trial
            An optuna trial passed to an objective function.

        X: 2D array-like, default=None
            Dataset.

        y: array-like, default=None
            Target values.
        """
        params_dict = {}
        for param_name, param_info in self.params_dict.items():
            param_type = param_info["type"]
            param_grid = param_info["grid"]
            if param_info.get("step") is None:
                if param_type == "int":
                    step = 1
                else:
                    step = None
            else:
                step = param_info["step"]
                if param_type == "int":
                    step = int(step)
            if param_info.get("log") is None:
                log_scale = False
            else:
                log_scale = param_info["log"]
            if param_type == 'categorical':
                params_dict[param_name] = trial.suggest_categorical(
                    param_name, param_grid
                )
            elif param_type == 'float':
                params_dict[param_name] = trial.suggest_float(
                    param_name, np.min(param_grid), np.max(param_grid), step,
                    log=log_scale
                )
            elif param_type == 'int':
                params_dict[param_name] = trial.suggest_int(
                    param_name, int(np.min(param_grid)),
                    int(np.max(param_grid)), step, log=log_scale
                )
            else:
                params_dict[param_name] = param_grid
