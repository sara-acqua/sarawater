from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Any, Literal

from sarawater.utils import compute_consecutive_lengths


@dataclass
class HabitatIndicesResult:
    """Container for species habitat outputs computed from natural and altered flows."""

    Q97_ref: float
    H97_ref: float
    UCUT_cum_ref: np.ndarray
    UCUT_events_ref: np.ndarray
    H_ref: np.ndarray
    UCUT_cum_alt: np.ndarray
    UCUT_events_alt: np.ndarray
    H_alt: np.ndarray
    ITH: float
    ISH: float
    IH: float
    HSD: float


def compute_h_ucut(
    HQ,
    date,
    Q,
    Q97,
    H97_ref: float | None = None,
    mode: Literal["reference", "altered"] | None = None,
    HQ_curve_resampling: bool = False,
    n_resample: int = 13,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Compute habitat time series and UCUT curve for a discharge time series and habitat-discharge curve.

    Parameters
    ----------
    HQ : array-like, shape (m, 2)
        Habitat-discharge table (Q, H).
    date : array-like
        Time series of dates (same length as Q).
    Q : array-like
        Discharge time series.
    Q97 : float
        Threshold discharge value (e.g., 3rd percentile).
    H97_ref : float, optional
        Habitat threshold to use (only for mode ``'altered'``).
    mode : {'reference', 'altered'}
        Type of calculation.
    HQ_curve_resampling : bool, optional
        Whether to resample the HQ curve for habitat calculation. Default is False.
    n_resample : int, optional
        Number of points to resample the HQ curve if HQ_curve_resampling is True. Default is 13.

    Returns
    -------
    UCUT_cumsum : np.ndarray
        Cumulative frequency of under-threshold events.
    UCUT_events : np.ndarray
        Durations of under-threshold events.
    H : np.ndarray
        Habitat time series.
    UCUT_cumpes : np.ndarray
        Cumulative frequency of under-threshold events, normalized.
    H97 : float
        Habitat threshold value used in the calculation.
    """
    Qstart = HQ[0, 0]
    Qend = HQ[-1, 0]

    H = np.full(Q.shape, np.nan, dtype=np.float64)
    mask = (
        Q < Qend
    )  # Flow discharge values higher than the maximum flow in the HQ curve are not considered for habitat calculation

    if HQ_curve_resampling:
        HQ_resampled = np.zeros((n_resample, 2))
        HQ_resampled[:, 0] = np.linspace(Qstart, Qend, n_resample)
        HQ_resampled[:, 1] = np.interp(HQ_resampled[:, 0], HQ[:, 0], HQ[:, 1])
        HQ_interp = np.copy(HQ_resampled)
    else:
        HQ_interp = np.copy(HQ)

    H[mask] = np.interp(Q[mask], HQ_interp[:, 0], HQ_interp[:, 1])
    H = np.round(H, 3)
    # Calculate H97 threshold
    if mode == "reference":
        if Q97 > Qend:
            H97 = 0
        else:
            H97 = np.interp(Q97, HQ_interp[:, 0], HQ_interp[:, 1])
            H97 = np.ceil(H97)
    elif mode == "altered":
        if H97_ref is None:
            raise ValueError("H97_ref must be provided when mode='altered'")
        H97 = H97_ref
        H97 = np.ceil(H97)
    else:
        raise ValueError("mode must be 'reference' or 'altered'")

    # H_UT (Under Threshold) takes value True if H<H97, value False if H>=H97 or if H is NaN
    H_UT = H < H97
    UT_days = compute_consecutive_lengths(H_UT)

    UT_days = np.array(UT_days)
    if UT_days.size == 0:
        # No under-threshold events
        return (
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
            H,
            np.array([], dtype=float),
            H97,
        )

    # sort the array in descending order
    UT_days_sorted = np.sort(UT_days)[::-1]

    # create an array that starts from UT_days_sorted[0] and ends with UT_days_sorted[-1] with a step of 1
    UCUT_events = np.arange(UT_days_sorted[0], 0, -1, dtype=np.int64)

    # create an array that contains the number of durations of each event and an array that contains the number of counts of each event
    durations, counts = np.unique(UT_days_sorted, return_counts=True)
    durations = durations[::-1]
    counts = counts[::-1]
    # e.g., durations = [11,  7,  5,  4,  3,  2,  1], counts = [1, 1, 1, 1, 2, 1, 1]

    # UT_days_sum = array that contains the sum of durations multiplied by counts
    UT_days_sum = durations * counts
    # e.g., UT_days_sum = [11, 7, 5, 4, 6, 2, 1]

    # Create an array of zeros with length equal to the max value in UT_days_sum
    out1 = np.zeros(UCUT_events[0])
    # e.g., out1 = [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    # Place each value at its corresponding index (arr[i] at index i)
    for i, v in enumerate(durations):
        out1[v - 1] = UT_days_sum[i]

    UCUT_cumsum = np.cumsum(out1[::-1])

    # e.g., UCUT_cumsum = [11., 11., 11., 11., 18., 18., 23., 27., 33., 35., 36.]
    days_tot = len(date)
    # Normalized version on total number of days
    UCUT_cumpes = UCUT_cumsum / days_tot

    return UCUT_cumsum, UCUT_events, H, UCUT_cumpes, H97


def compute_IH(
    UCUT_cum_ref, UCUT_cum_alt, H_ref, H_alt, UCUT_events_ref
) -> tuple[float, float, float, float]:
    """
    Calculate HSD, ISH, ITH, IH according to the MATLAB function logic.

    Parameters
    ----------
    UCUT_cum_ref : array-like
        Cumulative UCUT curve in reference conditions.
    UCUT_cum_alt : array-like
        Cumulative UCUT curve in altered conditions.
    H_ref : array-like
        Habitat time series in reference conditions.
    H_alt : array-like
        Habitat time series in altered conditions.
    UCUT_events_ref : array-like
        Under-threshold events in reference conditions.

    Returns
    -------
    ITH : float
    ISH : float
    IH : float
    HSD : float
    """
    UCUT_cum_ref = np.asarray(UCUT_cum_ref)
    UCUT_cum_alt = np.asarray(UCUT_cum_alt)
    H_ref = np.asarray(H_ref)
    H_alt = np.asarray(H_alt)
    UCUT_events_ref = np.asarray(UCUT_events_ref)

    l_ref = len(UCUT_cum_ref)
    l_alt = len(UCUT_cum_alt)

    # Calculate HSD (Habitat Stress Days)
    if l_alt == 1:
        HSD = np.nan
    elif l_alt < l_ref:
        HSD = np.nansum(
            np.abs(UCUT_cum_alt - UCUT_cum_ref[-l_alt:]) / UCUT_cum_ref[-l_alt:]
        ) / np.max(UCUT_events_ref)
    elif l_alt >= l_ref:
        HSD = np.nansum(
            np.abs(UCUT_cum_alt[-l_ref:] - UCUT_cum_ref) / UCUT_cum_ref
        ) / np.max(UCUT_events_ref)

    # ISH Index
    H_avg_ref = np.nanmean(H_ref)
    H_avg_alt = np.nanmean(H_alt)
    ISH_cond = np.abs(H_avg_ref - H_avg_alt) / H_avg_ref

    if ISH_cond <= 1:
        ISH = 1 - ISH_cond
    else:
        ISH = 0

    # ITH Index
    ITH = np.exp(-0.38 * HSD)

    # IH Index
    if np.isnan(ITH):
        IH = np.nan
    else:
        IH = min(ISH, ITH)

    return ITH, ISH, IH, HSD


def compute_habitat_indices(
    Qnat, Qalt, HQ, date, HQ_curve_resampling=False, n_resample=13
) -> HabitatIndicesResult:
    """
    Calculate Q97, UCUT, habitat time series and indices IH, ISH, ITH, HSD for natural and altered series.

    Parameters
    ----------
    Qnat : array-like
        Natural discharge time series.
    Qalt : array-like
        Altered discharge time series.
    HQ : array-like
        Habitat-discharge table (Q, H).
    date : array-like
        Time series of dates (same length as Qnat and Qalt).
    HQ_curve_resampling : bool, optional
        Whether to resample the HQ curve for habitat calculation. Default is False.
    n_resample : int, optional
        Number of points to resample the HQ curve if HQ_curve_resampling is True. Default is 13.

    Returns
    -------
    HabitatIndicesResult
        Dataclass containing Q97, UCUT outputs, habitat time series, and IH indices.
    """
    Qnat = np.asarray(Qnat)
    Qalt = np.asarray(Qalt)
    HQ = np.asarray(HQ)
    date = np.asarray(date)

    # Calculate Q97 (e.g., 3rd percentile of natural discharge)
    Q97 = np.percentile(Qnat, 3)

    # Calculate UCUT and habitat time series for the natural series (reference)
    UCUT_cum_ref, UCUT_events_ref, H_ref, UCUT_cum_pes_ref, H97_ref = compute_h_ucut(
        HQ,
        date,
        Qnat,
        Q97,
        mode="reference",
        HQ_curve_resampling=HQ_curve_resampling,
        n_resample=n_resample,
    )

    # Calculate UCUT and habitat time series for the altered series (altered)
    UCUT_cum_alt, UCUT_events_alt, H_alt, UCUT_cum_pes_alt, H97_alt = compute_h_ucut(
        HQ,
        date,
        Qalt,
        Q97,
        H97_ref=H97_ref,
        mode="altered",
        HQ_curve_resampling=HQ_curve_resampling,
        n_resample=n_resample,
    )

    # Calculate IH, ISH, ITH, HSD indices
    ITH, ISH, IH, HSD = compute_IH(
        UCUT_cum_ref, UCUT_cum_alt, H_ref, H_alt, UCUT_events_ref
    )

    return HabitatIndicesResult(
        Q97_ref=Q97,
        H97_ref=H97_ref,
        UCUT_cum_ref=UCUT_cum_ref,
        UCUT_events_ref=UCUT_events_ref,
        H_ref=H_ref,
        UCUT_cum_alt=UCUT_cum_alt,
        UCUT_events_alt=UCUT_events_alt,
        H_alt=H_alt,
        ITH=ITH,
        ISH=ISH,
        IH=IH,
        HSD=HSD,
    )
