from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sarawater.hydraulics import shear_stress, steady_flow_solver
from sarawater.utils import _validate_weights, compute_consecutive_lengths

# Default sediment transport settings used throughout the package.
DEFAULT_TRANSPORT_FORMULA = "mpm"
DEFAULT_MPM_THETA_C = 0.047

# Standard phi scale grain size classes
PHI_RANGE = np.arange(-9.5, 7.5 + 1, 1)
DMI = 2 ** (-PHI_RANGE) / 1000.0  # Grain diameters in meters

# Decay coefficient of the exponential normalisation of the ITS index. The habitat index ITH uses 0.38.
TPD_DECAY_COEFFICIENT = 0.38


def shields_parameter(tau_b, rho_w, rho_s, g, D):
    """
    Dimensionless Shields parameter.
    """
    return tau_b / ((rho_s - rho_w) * g * D)


def wilcock_crowe_2003(theta_i, Fi, g=9.81, Delta=1.65):
    """
    Computes the dimensionless sediment transport rate for each grain size class using the Wilcock & Crowe (2003) model. This model estimates fractional bedload transport rates
    in gravel-bed rivers, accounting for the effects of grain size distribution and hiding/exposure effects.

    Parameters
    ----------
    theta_i : array-like
        Dimensionless Shields parameter for each grain size class (unitless).
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81 m/s^2.
    Delta : float, optional
        Ratio of sediment density to water density minus 1. Default is 1.65.

    Returns
    -------
    qsi : ndarray
        Array of sediment transport rates for each grain size class (m^3/s).
        Each element corresponds to the transport rate for the respective phi class.
    """
    cumsum = np.cumsum(Fi)
    D50 = 2 ** (-np.interp(0.5, cumsum, PHI_RANGE)) / 1000
    Fr_s = np.sum((PHI_RANGE > -1) * Fi)
    b = 0.67 / (1 + np.exp(1.5 - DMI / D50))

    theta_r50 = 0.021 + 0.015 * np.exp(-20 * Fr_s)
    F_tau = (DMI / D50) ** b
    phi_ri = theta_i / (theta_r50 * F_tau)

    W_i = np.where(
        phi_ri >= 1.35,
        14 * np.maximum(1 - 0.894 / np.sqrt(phi_ri), 0) ** 4.5,
        0.002 * phi_ri**7.5,
    )
    qsi = (g * Delta * DMI**3) ** 0.5 * theta_i**1.5 * W_i * Fi
    return qsi


def meyer_peter_mueller(
    theta_i,
    Fi,
    theta_c=DEFAULT_MPM_THETA_C,
    g=9.81,
    Delta=1.65,
):
    """
    Compute the unit transport capacity for a given grain size distribution using the classic Meyer-Peter & Müller (1948) formula.

    This implementation provides per-phi-class transport rates using a simple availability
    weighting (Fi). For each grain size class i with diameter d_i, the per-width unit
    transport rate is:

        qb_i = 8 * max(theta_i - theta_c, 0)^{3/2} * sqrt(g * Delta * d_i^3)

    where theta_i is the dimensionless Shields parameter for each grain size class. The volumetric class transport is then qs_i = qb_i * Fi_i.

    Parameters
    ----------
    theta_i : array-like
        Dimensionless Shields parameter for each grain size class (unitless).
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
    theta_c : float, default=0.047
        Critical Shields parameter for initiation of motion.
    g : float, default=9.81
        Gravitational acceleration (m/s^2).
    Delta : float, default=1.65
        Ratio of sediment density to water density minus 1.

    Returns
    -------
    np.ndarray
        Array of volumetric sediment transport per phi-class (m^3/s) if Fi is provided.
    """
    phi_i = np.maximum(theta_i - theta_c, 0.0)
    qb_i = 8.0 * (phi_i**1.5) * np.sqrt(g * Delta * DMI**3)
    qsi = qb_i * Fi  # availability-weighted fractional transport
    return qsi


def transport_capacity(
    theta_i,
    Fi,
    transport_formula=DEFAULT_TRANSPORT_FORMULA,
    mpm_theta_c=DEFAULT_MPM_THETA_C,
    g=9.81,
    Delta=1.65,
):
    """
    Compute sediment transport capacity using the specified formula.

    Parameters
    ----------
    theta_i : array-like
        Dimensionless Shields parameter for each grain size class (unitless).
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
    transport_formula : {"wilcock_crowe", "mpm"}, optional
        Transport formula to use. Default "mpm" (Meyer-Peter & Müller, 1948).
        If "wilcock_crowe", uses Wilcock & Crowe (2003).
    mpm_theta_c : float, optional
        Critical Shields parameter for MPM. Used only if transport_formula is "mpm". Default 0.047.
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81 m/s^2.
    Delta : float, optional
        Ratio of sediment density to water density minus 1. Default is 1.65.

    Returns
    -------
    np.ndarray
        Sediment transport rate for each grain size class (m^3/s).

    Raises
    ------
    ValueError
        If transport_formula is not recognized.
    """
    if transport_formula == "wilcock_crowe":
        return wilcock_crowe_2003(theta_i, Fi, g=g, Delta=Delta)
    elif transport_formula == "mpm":
        return meyer_peter_mueller(theta_i, Fi, theta_c=mpm_theta_c, g=g, Delta=Delta)
    else:
        raise ValueError(
            f"Unknown transport formula '{transport_formula}'. Supported: 'wilcock_crowe', 'mpm'."
        )


def integrate_transport_across_section(
    qs: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray, h: float
) -> np.ndarray:
    """
    Integrate unit transport rates across cross-section.

    Parameters
    ----------
    qs : ndarray, shape (number of grain classes, N)
        Unit transport rate at each point for each grain class (m²/s)
    y_coords : ndarray, shape (N,)
        Horizontal coordinates
    z_coords : ndarray, shape (N,)
        Bed elevations
    h : float
        Water surface elevation

    Returns
    -------
    Qsi : ndarray, shape (number of grain classes,)
        Total transport per grain class (m³/s)
    """
    N = y_coords.size
    n_classes = qs.shape[0]
    depth = np.zeros_like(y_coords)
    wet = z_coords < h
    depth[wet] = h - z_coords[wet]

    Qsi = np.zeros(n_classes)

    for i in range(N - 1):
        depth_L, depth_R = depth[i], depth[i + 1]
        dy = y_coords[i + 1] - y_coords[i]
        dz = z_coords[i + 1] - z_coords[i]

        # Handle partial submersion
        if depth_L == 0 and depth_R == 0:
            dy = 0
        elif depth_L == 0 and depth_R > 0:
            dy = dy * (depth_R / (-dz))
        elif depth_L > 0 and depth_R == 0:
            dy = dy * (depth_L / dz)

        # Trapezoidal rule for each grain class
        for j in range(n_classes):
            qs_L = qs[j, i]
            qs_R = qs[j, i + 1]
            Qsi[j] += 0.5 * (qs_L + qs_R) * dy

    return Qsi


def compute_sediment_load(
    Qseries,
    dates,
    y_coords,
    z_coords,
    slope,
    ks,
    Fi,
    transport_formula=DEFAULT_TRANSPORT_FORMULA,
    rho_w=1000,
    rho_s=2650,
    g=9.81,
    mpm_theta_c=DEFAULT_MPM_THETA_C,
    to_csv=None,
):
    """
    Compute sediment load per size class and total for a given flow discharge time series, using the selected transport formula. Returns a DataFrame with a time series of sediment transport rates (both total and for each phi class), along with flow depth and other hydraulic parameters.

    Parameters
    ----------
    Qseries : ndarray
        Flow discharge time series.
    dates : list[datetime]
        List of datetime objects corresponding to each discharge value in Qseries.
        Must have the same length as Qseries.
    y_coords : ndarray
        Horizontal coordinates along cross-section (m).
    z_coords : ndarray
        Bed elevation at each point (m).
    slope : float
        Reach slope (m/m).
    ks : float
        Strickler coefficient (m^(1/3)/s).
    Fi : ndarray
        Fraction of sediment in each phi class.
    transport_formula : {"wilcock_crowe", "mpm"}, optional
        Sediment transport formula. Default "mpm" (Meyer-Peter & Müller, 1948)
        with availability weighting. If "wilcock_crowe", uses Wilcock & Crowe (2003).
    mpm_theta_c : float, optional
        Critical Shields parameter for MPM. Default 0.047.
    rho_w : float, optional
        Water density (kg/m^3). Default is 1000 kg/m^3.
    rho_s : float, optional
        Sediment density (kg/m^3). Default is 2650 kg/m^3.
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81 m/s^2.
    to_csv : str, optional
        File path to save the results as CSV.

    Returns
    -------
    pd.DataFrame
        Sediment load per phi class and total (Qs_total) with 'Datetime' column as first column.

    Raises
    ------
    ValueError
        If dates has different length than Qseries.
    """
    # Validate dates parameter
    if len(dates) != len(Qseries):
        raise ValueError(
            f"Length of dates ({len(dates)}) must match length of Qseries ({len(Qseries)})"
        )

    Delta = (rho_s - rho_w) / rho_w

    results = []
    for i, Q in enumerate(Qseries):
        h, Omega, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)
        depth = h - z_coords
        theta_i = np.zeros((len(DMI), len(depth)))
        qs = np.zeros_like(theta_i)
        for stripe_idx, stripe_depth in enumerate(depth):
            tau_b = shear_stress(rho_w, g, stripe_depth, slope)
            theta_i[:, stripe_idx] = shields_parameter(
                tau_b, rho_w, rho_s, g, DMI
            )  # Compute Shields parameter for each grain size class for the current stripe

            qs[:, stripe_idx] = transport_capacity(
                theta_i[:, stripe_idx],
                Fi,
                transport_formula=transport_formula,
                Delta=Delta,
                mpm_theta_c=mpm_theta_c,
            )  # Compute transport rate for each grain size class for the current stripe

        # Integrate transport across the cross-section for each grain size class
        Qs = integrate_transport_across_section(qs, y_coords, z_coords, h)

        row = {
            "Datetime": dates[i],
            "Q": Q,
            "h": h,
            "Omega": Omega,
            "U": U,
        }
        row.update({f"Qs_phi_{phi}": Qs[j] for j, phi in enumerate(PHI_RANGE)})
        row["Qs_total"] = Qs.sum()
        results.append(row)

    df = pd.DataFrame(results)

    if to_csv is not None:
        df.to_csv(to_csv, index=False)

    return df


def compute_annual_sediment_volume(df, to_csv=None):
    """
    Compute annual sediment volume (m³/year) per phi class and total
    from the sediment transport rates computed with ``compute_sediment_load``.

    The function multiplies the transport rate time series (m³/s) by the time step
    between observations to obtain volumes (m³), then aggregates these volumes
    on an annual basis. To obtain mass, multiply the result by the sediment density.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series output from
        ``compute_sediment_load``.
        Required columns are:

        - ``Datetime`` with datetime-like objects sorted in ascending order.
        - ``Qs_phi_<phi>`` columns with per-class transport rates (m³/s).
        - ``Qs_total`` with total transport rate (m³/s).

    to_csv : str, optional
        Path to save the resulting annual sediment volume or mass table as a CSV file.
        If None (default), no file is saved.

    Returns
    -------
    pandas.DataFrame or dict
        Annual sediment volumes in m³/year, in DataFrame or dictionary form.

    Notes
    -----
        - The time step is inferred from the first two records in the
            ``Datetime`` column and assumed constant over the series.
        - The function does not resample irregular time steps automatically;
            ensure the input time series has a uniform time interval.
    """
    phi_cols = [c for c in df.columns if c.startswith("Qs_phi_")]
    total_col = "Qs_total"

    dt_seconds = (df["Datetime"].iloc[1] - df["Datetime"].iloc[0]).total_seconds()

    vol_df = df.copy()
    for col in phi_cols + [total_col]:
        vol_df[col] = vol_df[col] * dt_seconds

    vol_df["Year"] = vol_df["Datetime"].dt.year
    annual = vol_df.groupby("Year")[phi_cols + [total_col]].sum()
    annual = annual.rename(
        columns=lambda col: col[3:] if col.startswith("Qs_") else col
    )  # Remove the "Qs_" prefix from column names for clarity

    if to_csv is not None:
        annual.to_csv(to_csv)
    return annual


@dataclass
class ACTResult:
    """Above-threshold Continuous Transport (ACT) curve of a transport capacity series.

    The ACT curve is the sediment counterpart of the habitat UCUT curve, but it
    considers the continuous events where the transport capacity is strictly above
    a threshold. Both arrays are empty if there are no above-threshold events.

    Attributes
    ----------
    cum : np.ndarray
        Cumulative number of time steps belonging to events lasting at least
        as long as the corresponding entry of ``events``.
    events : np.ndarray
        Event durations (time steps), from the longest event down to 1.
    """

    cum: np.ndarray
    cum_norm: np.ndarray
    events: np.ndarray


@dataclass
class SedimentIndicesResult:
    """Container for the sediment-regime alteration indices of a scenario.

    Attributes
    ----------
    Qs_threshold_ref : float
        Transport capacity threshold (m³/s) computed as a percentile of the
        reference transport capacity time series.
    ACT_ref, ACT_alt : ACTResult
        ACT curves in reference and altered conditions.
    TPD : float
        Transport Persistence Deviation, the distance between the reference
        and altered ACT curves (analogous to the Habitat Stress Days).
    ITS : float
        Index of Temporal alteration of Sediment transport, in [0, 1] (1 = unaltered).
    ISV : float
        Sediment Volume Index, in [0, 1] (1 = unaltered).
    ISR : float
        Sediment Regime Index, weighted sum of ITS and ISV.
    """

    Qs_threshold_ref: float
    ACT_ref: ACTResult
    ACT_alt: ACTResult
    TPD: float
    ITS: float
    ISV: float
    ISR: float


def compute_act_curve(Qs: np.ndarray, threshold: float) -> ACTResult:
    """Compute the Above-threshold Continuous Transport (ACT) curve.

    Parameters
    ----------
    Qs : np.ndarray
        Total transport capacity time series (m³/s).
    threshold : float
        Transport capacity threshold (m³/s). Events are consecutive time steps with ``Qs > threshold``.

    Returns
    -------
    ACTResult
        Cumulative durations and event durations of the curve.
    """
    lengths = np.asarray(compute_consecutive_lengths(Qs > threshold), dtype=np.int64)
    if lengths.size == 0:
        return ACTResult(
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
        )

    longest = int(lengths.max())
    counts = np.bincount(lengths, minlength=longest + 1)[
        1:
    ]  # counts the number of events of each length from 1 to longest. The first element of the array corresponds to the number of events with length 1, the second to length 2, and so on.
    steps_per_duration = counts * np.arange(
        1, longest + 1
    )  # multiply the number of events of each length by the length itself to get the total number of time steps for each duration.
    cum = np.cumsum(steps_per_duration[::-1]).astype(float)
    return ACTResult(
        cum=cum,
        cum_norm=cum / Qs.size,
        events=np.arange(longest, 0, -1, dtype=np.int64),
    )


def compute_transport_persistence_deviation(
    ACT_ref: ACTResult, ACT_alt: ACTResult
) -> float:
    """Compute the Transport Persistence Deviation (TPD).

    TPD is the analogue of the Habitat Stress Days: the relative distance
    between the altered and reference ACT curves, aligned on the shortest
    events, summed and normalised by the longest reference event. Before computing the distance, the altered curve is extended with zeros to match the length of the reference curve if it has fewer events.

    Parameters
    ----------
    ACT_ref : ACTResult
        Reference ACT curve (must be non-empty).
    ACT_alt : ACTResult
        Altered ACT curve. If empty (no above-threshold transport at all), the
        deviation is infinite and the corresponding index is 0.

    Returns
    -------
    float
        Transport Persistence Deviation.

    Raises
    ------
    ValueError
        If the reference curve is empty.
    """
    if ACT_ref.cum.size == 0:
        raise ValueError(
            "The reference scenario has no events above the transport capacity threshold; "
            "lower the threshold percentile."
        )
    if ACT_alt.cum.size == 0:
        return float("inf")

    if ACT_alt.cum.size < ACT_ref.cum.size:
        missing_events = np.arange(
            ACT_ref.events.max(), ACT_alt.events.max(), -1, dtype=np.int64
        )
        ACT_alt.events = np.concatenate((missing_events, ACT_alt.events))
        ACT_alt.cum = np.concatenate(
            (np.zeros(missing_events.size, dtype=float), ACT_alt.cum)
        )
        ACT_alt.cum_norm = np.concatenate(
            (np.zeros(missing_events.size, dtype=float), ACT_alt.cum_norm)
        )

    n = min(ACT_ref.cum.size, ACT_alt.cum.size)
    ref = ACT_ref.cum[-n:]
    alt = ACT_alt.cum[-n:]
    return float(np.nansum(np.abs(alt - ref) / ref) / np.max(ACT_ref.events))


def compute_sediment_indices(
    Qs_ref: np.ndarray,
    Qs_alt: np.ndarray,
    threshold_percentile: float = 3.0,
    weights: tuple[float, float] | None = None,
) -> SedimentIndicesResult:
    """Compute the indices describing the alteration of the sediment transport regime.

    Two complementary sub-indices, both in [0, 1] with 1 meaning no alteration,
    are computed and combined in the Sediment Regime Index (ISR):

    - **ITS** (Index of Temporal alteration of Sediment transport): frequency and duration of the events
      with transport capacity above a threshold, using ACT curves (the
      above-threshold counterpart of the habitat UCUT curves). The threshold is
      a percentile of the reference transport capacity. The distance between
      the reference and altered curves is the Transport Persistence Deviation
      (TPD), normalised as ``ITS = exp(-0.38 * TPD)`` like the habitat ITH.
    - **ISV** (Sediment Volume Index): mean annual transported volume,
      ``ISV = 1 - |V_ref - V_alt| / V_ref`` (clipped at 0).

    ``ISR = w_ITS * ITS + w_ISV * ISV``.

    The threshold of ITS is derived from the reference series only and applied
    unchanged to the altered one. Reference and altered series must share the
    same dates and time step.

    Parameters
    ----------
    Qs_ref : np.ndarray
        Reference (natural) total transport capacity (m³/s).
    Qs_alt : np.ndarray
        Altered total transport capacity (m³/s).
    threshold_percentile : float, optional
        Percentile in [0, 100) of ``Qs_ref`` used as threshold of ITS. Zero-values in Qs_ref are excluded. Default 3.
        Events are defined by ``Qs > threshold``, so a percentile whose value is
        zero (reference inactive for a larger share of time) selects every
        time step with transport.
    weights : tuple[float, float], optional
        Non-negative weights of (ITS, ISV), summing to 1. If None (default),
        equal weights are used.

    Returns
    -------
    SedimentIndicesResult
        Sub-indices, combined index and the thresholds adopted.

    Raises
    ------
    ValueError
        If inputs have inconsistent lengths, parameters are out of range, or the
        reference has no above-threshold events or a null transport volume.
    """
    if Qs_ref.shape != Qs_alt.shape:
        raise ValueError("Reference and altered series must have the same length.")
    if not 0 <= threshold_percentile < 100:
        raise ValueError("threshold_percentile must be in the range [0, 100).")
    w_ITS, w_ISV = _validate_weights(weights, 2)

    Qs_mean_ref = float(np.mean(Qs_ref))
    if Qs_mean_ref <= 0:
        raise ValueError("The reference sediment volume is zero.")

    # ITS index: persistence of above-threshold transport
    Qs_threshold = float(np.percentile(Qs_ref[Qs_ref > 0], threshold_percentile))
    ACT_ref = compute_act_curve(Qs_ref, Qs_threshold)
    ACT_alt = compute_act_curve(Qs_alt, Qs_threshold)

    TPD = compute_transport_persistence_deviation(ACT_ref, ACT_alt)
    ITS = float(np.exp(-TPD_DECAY_COEFFICIENT * TPD))

    # ISV index: mean annual volume (proportional to the mean transport capacity
    # because both series share the same time step)
    ISV = max(0.0, 1.0 - abs(Qs_mean_ref - float(np.mean(Qs_alt))) / Qs_mean_ref)

    return SedimentIndicesResult(
        Qs_threshold_ref=Qs_threshold,
        ACT_ref=ACT_ref,
        ACT_alt=ACT_alt,
        TPD=TPD,
        ITS=ITS,
        ISV=ISV,
        ISR=float(w_ITS * ITS + w_ISV * ISV),
    )
