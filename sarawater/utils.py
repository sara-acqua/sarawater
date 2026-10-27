from datetime import datetime
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def _compute_date_mask(
    dates: list[datetime],
    start_date: str | datetime | None = None,
    end_date: str | datetime | None = None,
) -> np.ndarray:
    """Build a boolean mask selecting dates within an inclusive range.

    Either bound may be omitted. If neither is given, all dates are selected.

    Parameters
    ----------
    dates : list of datetime
        Dates to filter.
    start_date : str or datetime, optional
        Inclusive lower bound ('YYYY-MM-DD' string or datetime).
    end_date : str or datetime, optional
        Inclusive upper bound ('YYYY-MM-DD' string or datetime).

    Returns
    -------
    np.ndarray
        Boolean array with the same length as ``dates``.
    """
    mask = np.ones(len(dates), dtype=bool)
    if start_date is None and end_date is None:
        return mask

    dates_index = pd.DatetimeIndex(dates)
    if start_date is not None:
        mask &= np.asarray(dates_index >= pd.to_datetime(start_date))
    if end_date is not None:
        mask &= np.asarray(dates_index <= pd.to_datetime(end_date))
    return mask


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


def _validate_weights(
    weights: Optional[Sequence[float]], n_weights: int, param_name: str = "weights"
) -> tuple[float, ...]:
    """Validate a set of weights and return them as a tuple of floats.

    Parameters
    ----------
    weights : sequence of float or None
        Weights to validate. If None, equal weights ``1 / n_weights`` are returned.
    n_weights : int
        Expected number of weights.
    param_name : str, optional
        Name of the parameter for error messages. Default is ``"weights"``.

    Returns
    -------
    tuple of float
        Validated weights.

    Raises
    ------
    ValueError
        If the weights are not numeric, their number differs from ``n_weights``,
        they are not finite and non-negative, or they do not sum to 1.
    """
    if weights is None:
        return (1.0 / n_weights,) * n_weights
    try:
        w = tuple(float(x) for x in weights)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{param_name} must be a sequence of numbers.") from exc
    if len(w) != n_weights:
        raise ValueError(f"{param_name} must contain exactly {n_weights} values.")
    if any(not np.isfinite(x) or x < 0 for x in w):
        raise ValueError(f"{param_name} must be finite and non-negative.")
    if not np.isclose(sum(w), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{param_name} must sum to 1 (got {sum(w)}).")
    return w


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
