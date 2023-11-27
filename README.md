# auto-optuna
A set of functions and classes to automate optimization processes related to ML models using [optuna](https://optuna.readthedocs.io/en/stable/).

**Disclamer**: This is not an installable package (at least for now, I lack the knowledge to do so), so if you want to use it for one of your projects, you will need to clone this project and include it as a dependency of yours.

# Purpose
This idea was born after I realized that, when tuning the hyperparameters of certain models using optuna, I was always tuning the same ones with the same values range, so I decided to build some tools to automate the process and stopping copying and pasting the same lines of codes over and over again. If you think the same, feel free to copy and paste this code to your project, but please, if you do so, cite the author of this project.

# How it works
As the name says, this project is built in top of optuna, which is probably the most used optimization tool in the field of Data Science. It also uses [numpy](https://numpy.org/) to make some computations, as well as some [scikit-learn](https://scikit-learn.org/stable/)'s features. This project is built to be general and flexible, so you have full control of what you're doing and you can even build your own optimizer class if needed.

It first implements a ParameterGrid class that can be used to set up an optuna search space inside an objective. This class takes a JSON-like dictionary with information about a set of parameters, in the format {<param_name>: {'type': <param_type>, 'grid': <param_range>, 'step': <grid_steps>, 'log', <bool>}}. It then can take an optuna trial to generate a params dictionary that can be used to initialize a model.

Next we have the BaseOptimizer class, which is the base class that implements the basic functionality for all the optimizers, then each optimizer differs only in the default settings and, in some cases, the objective function. Therefore, you only have to set the 'set_default_parameter_grid' and 'make_objective' methods to adapt each class to any model you want. You can also feed to the optimizers a custom ParameterGrid instance to control which parameters to turn and how.
