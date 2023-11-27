"""
Created on Wed Nov 22 11:45:21 2023, @author: nicolas.

This module implements some exceptions for other modules.
"""


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting."""
