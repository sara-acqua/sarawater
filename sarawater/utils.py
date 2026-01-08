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
