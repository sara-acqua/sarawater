import numpy as np
import pandas as pd


def shear_stress(rho_w, g, h, slope):
    """
    Bed shear stress (N/m^2) for wide rectangular channel approx:
        tau = rho_w * g * h * slope
    where h is flow depth (m).
    """
    return rho_w * g * h * slope


def shields_parameter(tau_b, rho_w, rho_s, g, D):
    """
    Dimensionless Shields parameter:
        theta = tau_b / ((rho_s - rho_w) * g * D)
    D: representative grain diameter (m) (typically D50)
    """
    denom = (rho_s - rho_w) * g * D
    # avoid division by zero
    if denom <= 0:
        raise ValueError(
            "Zero or negative denominator in shields_parameter calculation."
        )
    return tau_b / denom


def steady_flow_solver(B, slope, Q, D84, g=9.81, tol=1e-6, max_iter=1000):
    """
    Solves for steady flow depth, cross-section area, and velocity in a rectangular channel using an iterative Newton-Raphson method. A logarithmic formula is used to estimate the Chézy friction coefficient based on flow depth and sediment size.

    Parameters
    ----------
    B : float
        Channel width (meters).
    slope : float
        Channel bed slope (dimensionless, m/m).
    Q : float
        Water discharge (cubic meters per second, m^3/s).
    D84 : float
        Characteristic grain size (meters).
    g : float, optional
        Gravitational acceleration (meters per second squared, m/s^2). Default is 9.81.
    tol : float, optional
        Convergence tolerance for Newton-Raphson iterations (meters). Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    h : float
        Computed flow depth (meters).
    Omega : float
        Cross-sectional area (square meters, m^2).
    v : float
        Mean flow velocity (meters per second, m/s).
    """
    # Handle zero or negligible discharge explicitly
    if Q <= 0 or slope <= 0:
        return 0.0, 0.0, 0.0

    # Initial guess for flow depth
    h = max(0.1 * D84, 1e-4)  # ensure physically meaningful start

    for _ in range(max_iter):
        # Effective depth (avoid division by zero)
        h_eff = max(h, 1e-6)

        # Hydraulic geometry
        Omega = B * h_eff
        Rh = Omega / (B + 2 * h_eff)
        ratio = max(h_eff / D84, 1e-6)  # avoid zero or negative ratios

        # Empirical friction coefficient (stabilized)
        denom = (6.5) ** 2 + (2.5) ** 2 * ratio ** (5 / 3)
        C = (6.5 * 2.5 * ratio) / np.sqrt(denom)

        # Main flow function
        f = g * Rh * slope - (Q / (Omega * C)) ** 2

        # Finite difference derivative
        dh = 0.01 * h_eff
        h_d = h_eff + dh
        ratio_d = max(h_d / D84, 1e-6)
        denom_d = (6.5) ** 2 + (2.5) ** 2 * ratio_d ** (5 / 3)
        C_d = (6.5 * 2.5 * ratio_d) / np.sqrt(denom_d)
        C_d = np.clip(C_d, 1e-3, 100)

        Omega_d = B * h_d
        Rh_d = Omega_d / (B + 2 * h_d)
        f_d = g * Rh_d * slope - (Q / (Omega_d * C_d)) ** 2

        df = (f_d - f) / dh if dh > 0 else 1e-6
        if not np.isfinite(df) or abs(df) < 1e-12:
            break

        h_new = h - f / df

        # Convergence and physical limits
        if abs(h_new - h) < tol:
            h = h_new
            break

        h = h_new

    # Final calculations
    h = float(max(h, 0.0))
    Omega = B * h
    v = Q / Omega if Omega > 0 else 0.0

    # Ensure finite numeric results
    if not np.isfinite(h):
        h = 0.0
    if not np.isfinite(v):
        v = 0.0

    return h, Omega, v


def wilcock_crowe_2003(Fi, slope, B, h, D50, D84):
    """
    Computes the dimensionless sediment transport rate for each grain size class
    using the Wilcock & Crowe (2003) model.

    Parameters
    ----------
    Fi : array-like
        Fractional abundance of each grain size class in the bed surface (unitless, sum to 1).
    slope : float
        Channel bed slope (m/m).
    B : float
        Channel width (meters).
    h : float
        Flow depth (meters).
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
    if h <= 0 or slope <= 0 or B <= 0:
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
    if Fi.shape[0] != dmi.shape[0]:
        raise ValueError("Fi length does not match phi-class range [-9.5, 7.5].")

    theta_i = tau_b / ((rho_s - rho_w) * g * dmi)
    phi_i = np.maximum(theta_i - theta_c, 0.0)
    qb_i = 8.0 * (phi_i**1.5) * np.sqrt(g * s_minus_1 * dmi**3)
    Qsi = qb_i * B * Fi  # availability-weighted fractional transport

    # Replace NaNs with zeros for numerical safety
    Qsi[~np.isfinite(Qsi)] = 0.0
    return Qsi


def compute_sediment_load_mpm_total(
    QS,
    dates: list,
    B,
    slope,
    D50,
    D84=None,
    to_csv=None,
    theta_c=0.047,
    rho_w=1000.0,
    rho_s=2650.0,
    g=9.81,
):
    """
    Compute total sediment load using the Meyer-Peter & Müller (1948) formula
    with a representative grain size (typically D50), without phi-class breakdown.

    This function is a simplified alternative to `compute_sediment_load(..., method='mpm')`
    when you only need total transport and don't have a full grain size distribution.

    The classic MPM formula computes unit-width bedload transport as:
        qb = 8 * max(theta - theta_c, 0)^{3/2} * sqrt(g * (s-1) * D^3)
    where theta = tau_b / ((rho_s - rho_w) * g * D), tau_b = rho_w * g * h * slope,
    s = rho_s / rho_w, and D is the characteristic grain size (e.g., D50).

    Total volumetric transport is Qs = qb * B.

    Parameters
    ----------
    QS : array-like
        Discharge time series (m³/s).
    dates : list
        List of datetime objects corresponding to flow rates.
    B : float
        Channel width (m).
    slope : float
        Channel bed slope (m/m).
    D50 : float
        Representative grain size diameter (m), typically the median (D50).
    D84 : float, optional
        84th percentile grain size (m). Used by steady_flow_solver for roughness.
        If None, defaults to D50.
    to_csv : str, optional
        Path to save results as CSV. If None, does not save.
    theta_c : float, default=0.047
        Critical Shields parameter for initiation of motion.
    rho_w : float, default=1000.0
        Water density (kg/m³).
    rho_s : float, default=2650.0
        Sediment density (kg/m³).
    g : float, default=9.81
        Gravitational acceleration (m/s²).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Datetime: time step identifier
        - Q: discharge (m³/s)
        - h: flow depth (m)
        - Omega: cross-sectional area (m²)
        - v: velocity (m/s)
        - qS_total: total sediment transport (m³/s)

    Notes
    -----
    Unlike the phi-class version, this function returns only the total transport.
    Use `compute_sediment_load(..., method='mpm')` if you need fractional transport.
    """
    if dates is None:
        dates = np.arange(len(QS))

    if D84 is None:
        D84 = D50  # fallback for flow solver roughness

    s_minus_1 = rho_s / rho_w - 1.0

    results = []
    for i, Q in enumerate(QS):
        h, Omega, v = steady_flow_solver(B, slope, Q, D84)

        # Compute bed shear stress and Shields parameter
        tau_b = shear_stress(rho_w, g, h, slope)
        theta = shields_parameter(tau_b, rho_w, rho_s, g, D50)

        # Meyer-Peter & Müller formula
        phi = max(theta - theta_c, 0.0)
        qb = 8.0 * (phi**1.5) * np.sqrt(g * s_minus_1 * D50**3)
        qS_total = qb * B

        row = {
            "Datetime": dates[i],
            "Q": Q,
            "h": h,
            "Omega": Omega,
            "v": v,
            "qS_total": qS_total,
        }
        results.append(row)

    df = pd.DataFrame(results)

    if to_csv is not None:
        df.to_csv(to_csv, index=False)

    return df


def compute_sediment_load(
    QS,
    dates,
    B,
    slope,
    Fi,
    to_csv=None,
    method: str = "wilcock_crowe",
    mpm_theta_c: float = 0.047,
):
    """
    Compute sediment load per size class and total using the selected transport formula.

    Parameters
    ----------
    QS : ndarray
        Scenario discharge time series.
    B : float
        Channel width (m).
    slope : float
        Reach slope (m/m).
    Fi : ndarray
        Fraction of sediment in each phi class.
    dates : list
        List of datetime objects corresponding to flow rates
    to_csv : str, optional
        File path to save the results as CSV.

    method : {"wilcock_crowe", "mpm"}, optional
        Sediment transport formula. Default "wilcock_crowe" (Wilcock & Crowe, 2003).
        If "mpm", uses Meyer-Peter & Müller (1948) formula with availability weighting.
    mpm_theta_c : float, optional
        Critical Shields parameter for MPM. Default 0.047.

    Returns
    -------
    pd.DataFrame
        Sediment load per phi class and total (qS).
    """
    sed_range = np.arange(-9.5, 7.5 + 1, 1)
    if dates is None:
        dates = np.arange(len(QS))

    # Fi is assumed to be fractions corresponding to sed_range
    cumsum = np.cumsum(Fi)
    D50 = 2 ** (-np.interp(0.5, cumsum, sed_range)) / 1000
    D84 = 2 ** (-np.interp(0.84, cumsum, sed_range)) / 1000

    results = []
    for i, Q in enumerate(QS):
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
