import numpy as np
import pandas as pd
from scipy.optimize import fsolve


def distribute_discharge_by_conveyance(subdivisions, Q_total, manning_n=0.03):
    """
    Distribute total discharge among Engelund-Gauss strips based on conveyance.

    Uses Manning's equation to compute conveyance for each trapezoidal strip:
    K_i = (1/n) * A_i * R_i^(2/3)

    where R_i is the hydraulic radius (A_i / P_i) for strip i.
    Each strip is treated as a trapezoid with depths y_left and y_right.

    Parameters
    ----------
    subdivisions : DataFrame
        DataFrame with columns: 'width', 'area', 'y_left', 'y_right' for each strip
    Q_total : float
        Total discharge (m³/s)
    manning_n : float, optional
        Manning's roughness coefficient (default: 0.03)

    Returns
    -------
    ndarray
        Discharge per strip (m³/s)
    """
    if Q_total <= 0:
        return np.zeros(len(subdivisions))

    conveyances = []
    for strip in subdivisions.itertuples(index=False):
        strip_area = strip.area
        strip_width = strip.width
        y_left = strip.y_left
        y_right = strip.y_right

        if A > 0:
            # Wetted perimeter for trapezoidal strip
            # P = strip_width + left_side + right_side
            # For vertical strips: left and right sides are y_left and y_right
            P = strip_width + y_left + y_right
            hydraulic_radius = A / P  # Hydraulic radius

            # Conveyance: (1/n) * A * R^(2/3)
            conveyance = (1 / manning_n) * A * (R ** (2 / 3))
            conveyances.append(conveyance)
        else:
            conveyances.append(0.0)

    conveyances = np.array(conveyances)
    total_conveyance = np.sum(conveyances)

    if total_conveyance > 0:
        Q_per_strip = Q_total * (conveyances / total_conveyance)
    else:
        Q_per_strip = np.zeros(len(conveyances))

    return Q_per_strip


def compute_sediment_load_engelund_gauss(
    subdivisions,
    Qrel,
    dates,
    slope,
    Fi,
    manning_n=0.03,
    to_csv=None,
    method: str = "wilcock_crowe",
    mpm_theta_c: float = 0.047,
):
    """
    Compute sediment load using Engelund-Gauss subdivision method.

    For each time step:
    1. Distribute discharge among vertical strips based on conveyance using Manning's equation
    2. Compute flow depth and velocity for each strip
    3. Calculate sediment transport for each strip
    4. Sum contributions from all strips

    Parameters
    ----------
    subdivisions : DataFrame
        DataFrame with columns: 'width', 'height', 'area', 'x_left', 'x_right'
    Qrel : ndarray
        Scenario discharge time series (m³/s)
    dates : list
        List of datetime objects corresponding to flow rates
    slope : float
        Reach slope (m/m)
    Fi : ndarray
        Fraction of sediment in each phi class (length 18)
    manning_n : float, optional
        Manning's roughness coefficient (default: 0.03)
    to_csv : str, optional
        File path to save the results as CSV
    method : {"wilcock_crowe", "mpm"}, optional
        Sediment transport formula (default: "wilcock_crowe")
    mpm_theta_c : float, optional
        Critical Shields parameter for MPM (default: 0.047)

    Returns
    -------
    pd.DataFrame
        Sediment load per phi class and total (qS)
    """
    sed_range = np.arange(-9.5, 7.5 + 1, 1)
    if dates is None:
        dates = np.arange(len(Qrel))

    # Compute D50 and D84 from grain size distribution
    cumsum = np.cumsum(Fi)
    D50 = 2 ** (-np.interp(0.5, cumsum, sed_range)) / 1000
    D84 = 2 ** (-np.interp(0.84, cumsum, sed_range)) / 1000

    results = []

    for time_idx, Q_total in enumerate(Qrel):
        # Distribute discharge among strips
        Q_strips = distribute_discharge_by_conveyance(subdivisions, Q_total, manning_n)

        # Initialize arrays to accumulate sediment transport
        qS_total_strips = np.zeros(len(sed_range))
        h_weighted = 0.0
        v_weighted = 0.0
        total_area = 0.0

        # Compute sediment transport for each strip
        for strip_idx, strip in subdivisions.iterrows():
            Q_strip = Q_strips[strip_idx]
            B_strip = strip["width"]

            # Skip strips with zero discharge
            if Q_strip <= 0 or B_strip <= 0:
                continue

            # Solve for flow depth in this strip
            try:
                h_strip, Omega_strip, v_strip = steady_flow_solver(
                    B_strip, slope, Q_strip, D84
                )
            except ValueError:
                # Solver failed - assume zero transport for this strip
                h_strip = 0.0
                v_strip = 0.0
                continue

            if h_strip <= 0:
                continue

            # Compute sediment transport for this strip
            if method == "wilcock_crowe":
                qS_strip = wilcock_crowe_2003(Fi, slope, B_strip, h_strip, D50, D84)
            elif method == "mpm":
                qS_strip = meyer_peter_mueller(
                    Fi, slope, B_strip, h_strip, D50, theta_c=mpm_theta_c
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Supported: 'wilcock_crowe', 'mpm'."
                )

            # Accumulate sediment transport from this strip
            qS_total_strips += qS_strip

            # Weighted averages for flow properties
            strip_area = strip["area"]
            h_weighted += h_strip * strip_area
            v_weighted += v_strip * strip_area
            total_area += strip_area

        # Compute weighted average flow properties
        if total_area > 0:
            h_avg = h_weighted / total_area
            v_avg = v_weighted / total_area
        else:
            h_avg = 0.0
            v_avg = 0.0

        # Store results for this time step
        row = {
            "Datetime": dates[time_idx],
            "Q": Q_total,
            "h": h_avg,
            "Omega": total_area,
            "v": v_avg,
        }
        row.update(
            {f"qS_phi_{phi}": qS_total_strips[j] for j, phi in enumerate(sed_range)}
        )
        row["qS_total"] = np.sum(qS_total_strips)
        results.append(row)

    df = pd.DataFrame(results)

    if to_csv is not None:
        df.to_csv(to_csv, index=False)

    return df


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
    Solves for steady flow depth, cross-section area, and velocity in a rectangular channel by finding a root of the steady-flow residual with scipy.optimize.fsolve, a general-purpose nonlinear root-finding algorithm that may use various methods internally. A logarithmic formula is used to estimate the Chézy friction coefficient based on flow depth and sediment size.

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
        Convergence tolerance for fsolve (meters). Default is 1e-6.
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

    def flow_equation(h):
        """
        Residual function for steady uniform flow.
        Returns: g * Rh * slope - (Q / (Omega * C))^2
        """
        # Check for non-physical depth values
        if h[0] <= 0:
            raise ValueError(
                f"Negative or zero flow depth encountered during iteration: h={h[0]:.2e}. "
                "This indicates numerical instability in the flow solver."
            )

        h_eff = h[0]

        # Hydraulic geometry
        Omega = B * h_eff
        Rh = Omega / (B + 2 * h_eff)
        ratio = max(h_eff / D84, 1e-6)

        # Empirical friction coefficient
        denom = (6.5) ** 2 + (2.5) ** 2 * ratio ** (5 / 3)
        C = (6.5 * 2.5 * ratio) / np.sqrt(denom)

        # Flow equation residual
        residual = g * Rh * slope - (Q / (Omega * C)) ** 2

        return residual

    # Initial guess for flow depth
    h0 = max(0.1 * D84, 1e-4)

    # Solve using fsolve
    solution = fsolve(flow_equation, [h0], xtol=tol, maxfev=max_iter, full_output=False)
    h = float(solution[0])

    # Ensure non-negative and finite result
    h = max(h, 0.0)
    if not np.isfinite(h):
        h = 0.0

    # Final calculations
    Omega = B * h
    v = Q / Omega if Omega > 0 else 0.0

    # Ensure finite numeric results
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
    if Fi.shape[0] != dmi.shape[0]:
        raise ValueError(
            f"Fi length ({Fi.shape[0]}) does not match expected phi-class range "
            f"[-9.5, 7.5] with {dmi.shape[0]} classes. "
            f"Ensure grain size distribution from Reach.add_cross_section_info() "
            f"produces correct number of phi classes."
        )

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
