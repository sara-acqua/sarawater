import numpy as np


def compute_consecutive_lengths(array: np.ndarray) -> list:
    """Compute lengths of consecutive True values in array.

    Parameters
    ----------
    array : np.ndarray
        Boolean array to analyze

    Returns
    -------
    list
        List of consecutive True value lengths
    """
    lengths = []
    current_length = 0

    for value in array:
        if value:
            current_length += 1
        else:
            if current_length > 0:
                lengths.append(current_length)
                current_length = 0

    if current_length > 0:
        lengths.append(current_length)

    return lengths


def _validate_positive_numeric(value, param_name):
    """Validate that a value is a positive finite number.

    Parameters
    ----------
    value : any
        The value to validate.
    param_name : str
        Name of the parameter for error messages.

    Raises
    ------
    ValueError
        If value is not a positive finite number.
    """
    if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{param_name} must be a positive finite number, got {value}")
