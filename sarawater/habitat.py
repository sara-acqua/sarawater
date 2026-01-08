import numpy as np
from typing import Tuple

from sarawater.utils import compute_consecutive_lengths


def compute_h_ucut(
    HQ, date, Q, Q97, H97_ref=None, mode=None, n=100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
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
        Habitat threshold to use (only for mode='altered').
    mode : str, 'reference' or 'altered'
        Type of calculation.
    n : int, optional
        Number of points for habitat curve interpolation (default: 100).

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
    extra : dict
        Dictionary with Q97, H97, Qfit, splineHQ.
    """
    HQ = np.asarray(HQ)
    Q = np.asarray(Q)

    Qstart = HQ[0, 0]
    Qend = HQ[-1, 0]

    Qfit = np.linspace(Qstart, Qend, n + 1)
    # pp = CubicSpline(HQ[:, 0], HQ[:, 1])
    # splineHQ = pp(Qfit)

    H = np.full_like(Q, np.nan, dtype=float)
    mask = Q < Qend
    H[mask] = np.interp(Q[mask], HQ[:, 0], HQ[:, 1])
    H = np.round(H, 3)

    # Calculate H97 threshold
    if mode == "reference":
        if Q97 > Qfit[-1]:
            H97 = 0
        else:
            # H97 = pp(Q97)
            H97 = np.interp(Q97, HQ[:, 0], HQ[:, 1])
            H97 = np.ceil(H97)
    elif mode == "altered":
        H97 = H97_ref
        H97 = np.ceil(H97)
    else:
        raise ValueError("mode must be 'reference' or 'altered'")

    # H_under_threshold takes value True if H<H97, value False if H>=H97 or if H is NaN
    H_UT = H < H97
    UT_days = compute_consecutive_lengths(H_UT)

    UT_days = np.array(UT_days)
    if UT_days.size == 0:
        # No under-threshold events
        return (
            np.array([]),
            np.array([]),
            H,
            {"Q97": Q97, "H97": H97, "Qfit": Qfit, "splineHQ": None},
        )

    # sort the array in descending order
    UT_days_sorted = np.sort(UT_days)[::-1]

    # create an array that starts from UT_days_sorted[0] and ends with UT_days_sorted[-1] with a step of 1
    UCUT_events = np.arange(UT_days_sorted[0], 0, -1)

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

    extra = {"Q97": Q97, "H97": H97, "Qfit": Qfit, "splineHQ": None}
    return UCUT_cumsum, UCUT_events, H, UCUT_cumpes, extra


def compute_IH(
    UCUT_cum_ref, UCUT_cum_alt, H_ref, H_alt, UCUT_events_ref
) -> Tuple[float, float, float, float]:
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
    # HSD is calculated as
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


def compute_habitat_indices(Qnat, Qalt, HQ, date) -> dict:
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

    Returns
    -------
    dict with keys: Q97, UCUT_cum_ref, UCUT_events_ref, H_ref, UCUT_cum_alt, UCUT_events_alt, H_alt, ITH, ISH, IH, HSD
    """
    Qnat = np.asarray(Qnat)
    Qalt = np.asarray(Qalt)
    HQ = np.asarray(HQ)
    date = np.asarray(date)

    # Calculate Q97 (e.g., 3rd percentile of natural discharge)
    Q97 = np.percentile(Qnat, 3)

    # Calculate UCUT and habitat time series for the natural series (reference)
    UCUT_cum_ref, UCUT_events_ref, H_ref, UCUT_cum_pes_ref, extra_ref = compute_h_ucut(
        HQ, date, Qnat, Q97, mode="reference"
    )

    # Calculate UCUT and habitat time series for the altered series (altered)
    UCUT_cum_alt, UCUT_events_alt, H_alt, UCUT_cum_pes_alt, extra_alt = compute_h_ucut(
        HQ, date, Qalt, Q97, H97_ref=extra_ref["H97"], mode="altered"
    )

    # Calculate IH, ISH, ITH, HSD indices
    ITH, ISH, IH, HSD = compute_IH(
        UCUT_cum_ref, UCUT_cum_alt, H_ref, H_alt, UCUT_events_ref
    )

    return {
        "Q97_ref": Q97,
        "H97_ref": extra_ref["H97"],
        "UCUT_cum_ref": UCUT_cum_ref,
        "UCUT_events_ref": UCUT_events_ref,
        "H_ref": H_ref,
        "UCUT_cum_alt": UCUT_cum_alt,
        "UCUT_events_alt": UCUT_events_alt,
        "H_alt": H_alt,
        "ITH": ITH,
        "ISH": ISH,
        "IH": IH,
        "HSD": HSD,
    }
