from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sarawater.utils import compute_consecutive_lengths


@dataclass(frozen=True)
class UCUTCurve:
    """
    Uniform continuous under-threshold (UCUT) curve for one habitat threshold.

    All arrays have the same length: one entry per integer duration between
    the longest under-threshold event and 1 day (durations that never
    occurred are included).

    Attributes
    ----------
    durations : np.ndarray
        Continuous under-threshold durations [days], in descending order.
    cum_days : np.ndarray
        Total number of days spent in under-threshold events lasting at least the corresponding duration [days].
    cum_freq : np.ndarray
        ``cum_days`` divided by the total number of days of the series [-].
    """

    durations: np.ndarray
    cum_days: np.ndarray
    cum_freq: np.ndarray


@dataclass
class HabitatIndicesResult:
    """Container for species habitat outputs computed from natural and altered flows."""

    Q_threshold_ref: float
    H_threshold_ref: float
    H_ref: np.ndarray
    ucut_ref: UCUTCurve
    H_alt: np.ndarray
    ucut_alt: UCUTCurve
    ITH: float
    ISH: float
    IH: float
    HSD: float


def resample_HQ_curve(HQ: np.ndarray, n_resample: int = 13) -> np.ndarray:
    """
    Resample a habitat-discharge curve on evenly spaced discharges.

    Parameters
    ----------
    HQ : numpy.ndarray, shape (m, 2)
        Habitat-discharge table (Q, H), sorted by increasing Q.
    n_resample : int, optional
        Number of points of the resampled curve. Default is 13.

    Returns
    -------
    numpy.ndarray, shape (n_resample, 2)
        Resampled habitat-discharge table (Q, H), spanning the same discharge range as ``HQ``.
    """
    if n_resample < 2:
        raise ValueError("n_resample must be at least 2")
    HQ_curve = np.zeros((n_resample, 2))
    HQ_curve[:, 0] = np.linspace(HQ[0, 0], HQ[-1, 0], n_resample)
    HQ_curve[:, 1] = np.interp(HQ_curve[:, 0], HQ[:, 0], HQ[:, 1])
    return HQ_curve


def compute_habitat_series(HQ_curve: np.ndarray, Q_series: np.ndarray) -> np.ndarray:
    """
    Compute the habitat time series of a discharge time series.

    Parameters
    ----------
    HQ_curve : numpy.ndarray, shape (m, 2)
        Habitat-discharge table (Q, H), sorted by increasing Q.
    Q_series : np.ndarray, shape (n,)
        Discharge time series.

    Returns
    -------
    np.ndarray, shape (n,)
        Habitat time series, rounded to 3 decimals. It is NaN where the discharge
        exceeds the maximum discharge of the HQ curve.
    """
    Q_series = np.asarray(Q_series)
    H_series = np.full(Q_series.shape, np.nan, dtype=np.float64)
    # Discharges above the maximum of the HQ curve are not considered for habitat calculation
    mask = Q_series <= HQ_curve[-1, 0]
    H_series[mask] = np.interp(Q_series[mask], HQ_curve[:, 0], HQ_curve[:, 1])
    return np.round(H_series, 3)


def compute_habitat_threshold(HQ_curve: np.ndarray, Q_threshold: float) -> float:
    """
    Compute the habitat threshold corresponding to a threshold discharge.

    Parameters
    ----------
    HQ_curve : numpy.ndarray, shape (m, 2)
        Habitat-discharge table (Q, H), sorted by increasing Q.
    Q_threshold : float
        Threshold discharge (e.g., 3rd percentile of the natural flow).

    Returns
    -------
    float
        Habitat at ``Q_threshold`` rounded up to the next integer, or 0 if
        ``Q_threshold`` exceeds the maximum discharge of the HQ curve.
    """
    if Q_threshold > HQ_curve[-1, 0]:
        return 0.0
    return float(np.ceil(np.interp(Q_threshold, HQ_curve[:, 0], HQ_curve[:, 1])))


def compute_ucut(H_series: np.ndarray, H_threshold: float) -> UCUTCurve:
    """
    Compute the UCUT curve of a habitat time series.

    Parameters
    ----------
    H_series : np.ndarray, shape (n,)
        Habitat time series. NaN values are never counted as under-threshold,
        but are included in the total number of days.
    H_threshold : float
        Habitat threshold. A day is under threshold if ``H < H_threshold``.

    Returns
    -------
    UCUTCurve
        UCUT curve (empty if there are no under-threshold events).
    """
    H_series = np.asarray(H_series)
    # is_under_threshold is True if H < H_threshold, False if H >= H_threshold or H is NaN
    is_under_threshold = H_series < H_threshold
    event_durations = np.array(
        compute_consecutive_lengths(is_under_threshold)
    )  # durations of the continuous under-threshold periods
    if event_durations.size == 0:
        return UCUTCurve(
            durations=np.array([], dtype=np.int64),
            cum_days=np.array([], dtype=float),
            cum_freq=np.array([], dtype=float),
        )

    # y-axis of the UCUT curve: every integer duration from the longest event down to 1 day
    UCUT_durations = np.arange(event_durations.max(), 0, -1, dtype=np.int64)

    # Total days spent in events of each exact duration d = 1..max (zero if no such event),
    # e.g. durations [11, 7, 5, 4, 3, 3, 2, 1] -> days_per_duration[d - 1] = [1, 2, 6, 4, 5, 0, 7, 0, 0, 0, 11]
    days_per_duration = np.bincount(event_durations)[1:] * np.arange(
        1, event_durations.max() + 1
    )

    # Days in events lasting at least d days, ordered from the longest duration down to 1
    UCUT_cum_days = np.cumsum(days_per_duration[::-1]).astype(float)

    return UCUTCurve(
        durations=UCUT_durations,
        cum_days=UCUT_cum_days,
        cum_freq=UCUT_cum_days / H_series.size,
    )


def compute_IH(
    ucut_ref: UCUTCurve,
    ucut_alt: UCUTCurve,
    H_ref: np.ndarray,
    H_alt: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Calculate HSD, ISH, ITH, IH according to the MATLAB function logic.

    Parameters
    ----------
    ucut_ref : UCUTCurve
        UCUT curve in reference conditions.
    ucut_alt : UCUTCurve
        UCUT curve in altered conditions.
    H_ref : array-like
        Habitat time series in reference conditions.
    H_alt : array-like
        Habitat time series in altered conditions.

    Returns
    -------
    ITH : float
    ISH : float
    IH : float
    HSD : float
    """
    cum_days_ref = np.asarray(ucut_ref.cum_days)
    cum_days_alt = np.asarray(ucut_alt.cum_days)
    max_duration_ref = np.max(ucut_ref.durations)
    H_ref = np.asarray(H_ref)
    H_alt = np.asarray(H_alt)

    l_ref = len(cum_days_ref)
    l_alt = len(cum_days_alt)

    # Calculate HSD (Habitat Stress Days)
    if l_alt == 1:
        HSD = np.nan
    elif l_alt < l_ref:
        HSD = (
            np.nansum(
                np.abs(cum_days_alt - cum_days_ref[-l_alt:]) / cum_days_ref[-l_alt:]
            )
            / max_duration_ref
        )
    elif l_alt >= l_ref:
        HSD = (
            np.nansum(np.abs(cum_days_alt[-l_ref:] - cum_days_ref) / cum_days_ref)
            / max_duration_ref
        )

    # ITH Index
    ITH = np.exp(-0.38 * HSD)

    # ISH Index
    H_avg_ref = np.nanmean(H_ref)
    H_avg_alt = np.nanmean(H_alt)
    ISH_cond = np.abs(H_avg_ref - H_avg_alt) / H_avg_ref

    if ISH_cond <= 1:
        ISH = 1 - ISH_cond
    else:
        ISH = 0

    # IH Index
    if np.isnan(ITH):
        IH = np.nan
    else:
        IH = min(ISH, ITH)

    return ITH, ISH, IH, HSD


def compute_habitat_indices(
    Qnat, Qalt, HQ, HQ_curve_resampling=False, n_resample=13
) -> HabitatIndicesResult:
    """
    Calculate Q_threshold, UCUT, habitat time series and indices IH, ISH, ITH, HSD for natural and altered series.

    Parameters
    ----------
    Qnat : array-like
        Natural discharge time series.
    Qalt : array-like
        Altered discharge time series.
    HQ : array-like
        Habitat-discharge table (Q, H).
    HQ_curve_resampling : bool, optional
        Whether to resample the HQ curve for habitat calculation. Default is False.
    n_resample : int, optional
        Number of points to resample the HQ curve if HQ_curve_resampling is True. Default is 13.

    Returns
    -------
    HabitatIndicesResult
        Dataclass containing the reference thresholds, habitat time series, UCUT curves, and IH indices.
    """
    Qnat = np.asarray(Qnat)
    Qalt = np.asarray(Qalt)
    HQ = np.asarray(HQ)

    # Threshold discharge: 3rd percentile of the natural discharge (Q97 exceedance)
    Q_threshold = float(np.percentile(Qnat, 3))

    HQ_curve = resample_HQ_curve(HQ, n_resample) if HQ_curve_resampling else HQ

    H_threshold = compute_habitat_threshold(HQ_curve, Q_threshold)
    H_ref = compute_habitat_series(HQ_curve, Qnat)
    H_alt = compute_habitat_series(HQ_curve, Qalt)
    ucut_ref = compute_ucut(H_ref, H_threshold)
    ucut_alt = compute_ucut(H_alt, H_threshold)

    ITH, ISH, IH, HSD = compute_IH(ucut_ref, ucut_alt, H_ref, H_alt)

    return HabitatIndicesResult(
        Q_threshold_ref=Q_threshold,
        H_threshold_ref=H_threshold,
        H_ref=H_ref,
        ucut_ref=ucut_ref,
        H_alt=H_alt,
        ucut_alt=ucut_alt,
        ITH=ITH,
        ISH=ISH,
        IH=IH,
        HSD=HSD,
    )
