import numpy as np
import pandas as pd

from sarawater.hydraulics import shear_stress, steady_flow_solver

# Standard phi scale grain size classes
PHI_RANGE = np.arange(-9.5, 7.5 + 1, 1)
DMI = 2 ** (-PHI_RANGE) / 1000.0  # Grain diameters in meters


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
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
    theta_i : array-like
        Dimensionless Shields parameter for each grain size class (unitless).
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81 m/s^2.
    Delta : float, optional
        Ratio of sediment density to water density minus 1. Default is 1.65.

    Returns
    -------
    qsi : ndarray
        Array of sediment transport rates for each grain size class (m^3/s).
        Each element corresponds to the transport rate for the respective phi class.

    D50 : float
        Median grain size (meters).
    D84 : float
        84th percentile grain size (meters).

    Returns
    -------
    Qsi : ndarray
        Array of sediment transport rates for each grain size class (m^3/s).
        Each element corresponds to the transport rate for the respective phi class.

    Method
    ------
    The Wilcock & Crowe 2003 model estimates fractional bedload transport rates
    in gravel-bed rivers, accounting for the effects of grain size distribution
    and hiding/exposure effects.
    """
    sed_range = np.arange(-9.5, 7.5 + 1, 1)
    dmi = 2 ** (-sed_range) / 1000
    rho_w = 1000
    rho_s = 2650
    g = 9.81
    R = rho_s / rho_w - 1

    Fr_s = np.sum((sed_range > -1) * Fi)
    b = 0.67 / (1 + np.exp(1.5 - dmi / D50))
    tau = (
        (rho_w * g * h * slope)
        * ((2.5) ** 2 * (h / D84) ** (5 / 3))
        / ((6.5) ** 2 + (2.5) ** 2 * (h / D84) ** (5 / 3))
    )
    tau_r50 = (0.021 + 0.015 * np.exp(-20 * Fr_s)) * (rho_w * R * g * D50)
    tau_ri = tau_r50 * (dmi / D50) ** b
    phi_ri = tau / tau_ri

    W_i = np.where(
        phi_ri >= 1.35,
        14 * np.maximum(1 - 0.894 / np.sqrt(phi_ri), 0) ** 4.5,
        0.002 * phi_ri**7.5,
    )
    Qsi = B * W_i * Fi * (tau / rho_w) ** (3 / 2) / (R * g)
    Qsi[np.isnan(Qsi)] = 0
    return Qsi


def meyer_peter_mueller(
    Fi,
    slope,
    B,
    h,
    D50=None,
    theta_c=0.047,
    rho_w=1000.0,
    rho_s=2650.0,
    g=9.81,
):
    """
    Compute fractional sediment transport using the classic Meyer-Peter & Müller (1948) formula.

    This implementation provides per-phi-class transport rates using a simple availability
    weighting (Fi). For each grain size class i with diameter d_i, the per-width unit
    transport rate is:

        qb_i = 8 * max(theta_i - theta_c, 0)^{3/2} * sqrt(g * (s-1) * d_i^3)

    where theta_i = tau_b / ((rho_s - rho_w) * g * d_i), tau_b = rho_w * g * h * slope,
    and s = rho_s / rho_w. The volumetric class transport is then Qs_i = qb_i * B * Fi_i.

    Parameters
    ----------
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
        If None, falls back to total transport using D50 (requires D50 not None).
    slope : float
        Channel bed slope (m/m).
    B : float
        Channel width (m).
    h : float
        Flow depth (m).
    D50 : float, optional
        Median grain size (m). Used only if Fi is None to compute total transport.
    theta_c : float, default=0.047
        Critical Shields parameter for initiation of motion.
    rho_w : float, default=1000.0
        Water density (kg/m^3).
    rho_s : float, default=2650.0
        Sediment density (kg/m^3).
    g : float, default=9.81
        Gravitational acceleration (m/s^2).

    Returns
    -------
    np.ndarray
        Array of volumetric sediment transport per phi-class (m^3/s) if Fi is provided.
        If Fi is None, returns a length-1 array with the total transport using D50.
    """
    # Handle degenerate flow
    if h <= 0:
        if Fi is None:
            return np.array([0.0])
        return np.zeros_like(np.asarray(Fi, dtype=float))

    tau_b = shear_stress(rho_w, g, h, slope)
    s_minus_1 = rho_s / rho_w - 1.0

    if Fi is None:
        if D50 is None or D50 <= 0:
            return np.array([0.0])
        theta = shields_parameter(tau_b, rho_w, rho_s, g, D50)
        phi = max(theta - theta_c, 0.0)
        qb = 8.0 * (phi**1.5) * np.sqrt(g * s_minus_1 * D50**3)
        Qs = qb * B
        return np.array([Qs])

    # Fractional computation over phi classes
    sed_range = np.arange(-9.5, 7.5 + 1, 1)
    dmi = 2 ** (-sed_range) / 1000.0

    Fi = np.asarray(Fi, dtype=float)

    # Validate Fi length matches phi class range
    # if Fi.shape[0] != dmi.shape[0]:
    #     raise ValueError(
    #         f"Fi length ({Fi.shape[0]}) does not match expected phi-class range "
    #         f"[-9.5, 7.5] with {dmi.shape[0]} classes. "
    #         f"Ensure grain size distribution from Reach.add_cross_section_info() "
    #         f"produces correct number of phi classes."
    #     )

    theta_i = tau_b / ((rho_s - rho_w) * g * dmi)
    phi_i = np.maximum(theta_i - theta_c, 0.0)
    qb_i = 8.0 * (phi_i**1.5) * np.sqrt(g * s_minus_1 * dmi**3)
    Qsi = qb_i * B * Fi  # availability-weighted fractional transport

    # Replace NaNs with zeros for numerical safety
    Qsi[~np.isfinite(Qsi)] = 0.0
    return Qsi


def compute_sediment_load(
    Qrel,
    dates,
    B,
    slope,
    Fi,
    to_csv=None,
    method: str = "wilcock_crowe",
    mpm_theta_c: float = 0.047,
    subdivisions=None,
    manning_n=0.03,
):
    """
    Compute sediment load per size class and total using the selected transport formula.

    Supports both simple rectangular channel and Engelund-Gauss subdivision approaches.

    Parameters
    ----------
    Qrel : ndarray
        Scenario discharge time series.
    dates : list
        List of datetime objects corresponding to flow rates
    B : float
        Channel width (m). Used only if subdivisions is None.
    slope : float
        Reach slope (m/m).
    Fi : ndarray
        Fraction of sediment in each phi class.
    to_csv : str, optional
        File path to save the results as CSV.
    method : {"wilcock_crowe", "mpm"}, optional
        Sediment transport formula. Default "wilcock_crowe" (Wilcock & Crowe, 2003).
        If "mpm", uses Meyer-Peter & Müller (1948) formula with availability weighting.
    mpm_theta_c : float, optional
        Critical Shields parameter for MPM. Default 0.047.
    subdivisions : DataFrame, optional
        Engelund-Gauss subdivision data. If provided, uses subdivision-based calculation.
        Must have columns: 'width', 'height', 'area'
    manning_n : float, optional
        Manning's roughness coefficient for conveyance calculation. Default 0.03.

    Returns
    -------
    pd.DataFrame
        Sediment load per phi class and total (qS).
    """
    # Use Engelund-Gauss method if subdivisions are provided
    if subdivisions is not None and len(subdivisions) > 0:
        return compute_sediment_load_engelund_gauss(
            subdivisions=subdivisions,
            Qrel=Qrel,
            dates=dates,
            slope=slope,
            Fi=Fi,
            manning_n=manning_n,
            to_csv=to_csv,
            method=method,
            mpm_theta_c=mpm_theta_c,
        )

    # Otherwise, use simple rectangular channel approach
    sed_range = np.arange(-9.5, 7.5 + 1, 1)
    if dates is None:
        dates = np.arange(len(Qrel))

    # Fi is assumed to be fractions corresponding to sed_range
    cumsum = np.cumsum(Fi)
    D50 = 2 ** (-np.interp(0.5, cumsum, sed_range)) / 1000
    D84 = 2 ** (-np.interp(0.84, cumsum, sed_range)) / 1000

    results = []
    for i, Q in enumerate(Qrel):
        h, Omega, v = steady_flow_solver(B, slope, Q, D84)

        if method == "wilcock_crowe":
            qS = wilcock_crowe_2003(Fi, slope, B, h, D50, D84)
        elif method == "mpm":
            qS = meyer_peter_mueller(Fi, slope, B, h, D50, theta_c=mpm_theta_c)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Supported: 'wilcock_crowe', 'mpm'."
            )
        row = {
            "Datetime": dates[i],
            "Q": Q,
            "h": h,
            "Omega": Omega,
            "v": v,
        }
        row.update({f"qS_phi_{phi}": qS[j] for j, phi in enumerate(sed_range)})
        row["qS_total"] = np.sum(qS)
        results.append(row)

    df = pd.DataFrame(results)

    if to_csv is not None:
        df.to_csv(to_csv, index=False)

    return df


def compute_annual_sediment_volume(
    df, to_csv=None, as_dict=False, to_ton=False, rho_s=2650
):
    """
    Compute annual sediment volume (m³/year) or mass (ton/year) per phi class and total
    from the sediment transport rates computed with `compute_sediment_load`.

    The function multiplies the transport rate time series (m³/s) by the time step
    between observations to obtain volumes (m³), then aggregates these volumes
    on an annual basis. Optionally, it can also convert volumes to mass using a
    specified sediment density.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series output from `compute_sediment_load`.
        It must include:
            - A 'Datetime' column with datetime-like objects (sorted in ascending order).
            - Columns named 'qS_phi_<phi>' representing sediment transport rate
            (m³/s) for each phi size class.
            - A 'qS_total' column representing total sediment transport rate (m³/s).

    to_csv : str, optional
        Path to save the resulting annual sediment volume or mass table as a CSV file.
        If None (default), no file is saved.

    as_dict : bool, default=False
        If True, the function returns the results as a nested Python dictionary in the form:
        {
            year_1: {'qS_phi_-9.5': value, ..., 'qS_total': total_value},
            year_2: {...},
            ...
        }
        If False (default), the function returns a pandas.DataFrame.

    to_ton : bool, default=False
        If True, the output values are converted from m³/year to ton/year using:
        mass (ton) = volume (m³) × rho_s (kg/m³) / 1000.

    rho_s : float, default=2650
        Sediment density in kg/m³. Default corresponds to quartz density.
        Only used if `to_ton=True`.

    Returns
    -------
    pandas.DataFrame or dict
        - If `as_dict=False` (default): a DataFrame indexed by year with one column
        for each phi class and one for total sediment flux. Units depend on `to_ton`:
            * m³/year if to_ton=False
            * ton/year if to_ton=True

        - If `as_dict=True`: a nested dictionary structured as:
            {year: {phi_class_name: value, ..., 'qS_total': total_value}}
        where each value is either in m³/year or ton/year depending on `to_ton`.

    Notes
    -----
    - The time step is inferred from the first two records in the 'Datetime' column.
    It is assumed to be constant throughout the series.
    - The function does not resample irregular time steps automatically — ensure
    your input time series has a uniform time interval.
    - The conversion to tons uses:
        1 m³ × 2650 kg/m³ ÷ 1000 kg/ton = 2.65 ton
    """
    phi_cols = [c for c in df.columns if c.startswith("qS_phi_")]
    total_col = "qS_total"

    dt_seconds = (df["Datetime"].iloc[1] - df["Datetime"].iloc[0]).total_seconds()

    vol_df = df.copy()
    for col in phi_cols + [total_col]:
        vol_df[col] = vol_df[col] * dt_seconds

    vol_df["Year"] = vol_df["Datetime"].dt.year
    annual = vol_df.groupby("Year")[phi_cols + [total_col]].sum()

    if to_ton:
        annual = annual * (rho_s / 1000.0)  # convert m³ to ton

    if to_csv is not None:
        annual.to_csv(to_csv)

    if as_dict:
        return annual.to_dict(orient="index")

    return annual
