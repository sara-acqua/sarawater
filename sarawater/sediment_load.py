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
    rho_w : float, optional
        Water density (kg/m^3). Default is 1000 kg/m^3.
    rho_s : float, optional
        Sediment density (kg/m^3). Default is 2650 kg/m^3
    g : float, optional
        Gravitational acceleration (m/s^2). Default is 9.81 m/s^2.

    Returns
    -------
    Qsi : ndarray
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
    theta_c=0.047,
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
        If Fi is None, returns a length-1 array with the total transport using D50.
    """
    phi_i = np.maximum(theta_i - theta_c, 0.0)
    qb_i = 8.0 * (phi_i**1.5) * np.sqrt(g * Delta * DMI**3)
    qsi = qb_i * Fi  # availability-weighted fractional transport
    return qsi


def transport_capacity(
    theta_i,
    Fi,
    transport_formula="wilcock_crowe",
    mpm_theta_c=0.047,
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
        Transport formula to use. Default "wilcock_crowe" (Wilcock & Crowe, 2003).
        If "mpm", uses Meyer-Peter & Müller (1948).
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
    transport_formula="wilcock_crowe",
    rho_w=1000,
    rho_s=2650,
    g=9.81,
    mpm_theta_c=0.047,
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
        Sediment transport formula. Default "wilcock_crowe" (Wilcock & Crowe, 2003). If "mpm", uses Meyer-Peter & Müller (1948) formula with availability weighting.
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
            - Columns named 'Qs_phi_<phi>' representing sediment transport rate
            (m³/s) for each phi size class.
            - A 'Qs_total' column representing total sediment transport rate (m³/s).

    to_csv : str, optional
        Path to save the resulting annual sediment volume or mass table as a CSV file.
        If None (default), no file is saved.

    as_dict : bool, default=False
        If True, the function returns the results as a nested Python dictionary in the form:
        {
            year_1: {'Qs_phi_-9.5': value, ..., 'Qs_total': total_value},
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
            {year: {phi_class_name: value, ..., 'Qs_total': total_value}}
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
    phi_cols = [c for c in df.columns if c.startswith("Qs_phi_")]
    total_col = "Qs_total"

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
