import numpy as np
from scipy.optimize import fsolve


def shear_stress(rho_w, g, h, slope):
    """
    Bed shear stress (N/m^2) for wide rectangular channel approx:
        tau = rho_w * g * h * slope
    where h is flow depth (m).
    """
    return rho_w * g * h * slope


def residual_uniFlow_rect(depth, Q, B, ks, slope):
    """
    Residual function for steady uniform flow in rectangular channel. Computes the flow discharge Q_computed for a given water depth h and compares it to the known discharge Q. Returns Q_computed/Q - 1, which equals 0 when correct h is found.
    """
    Omega = B * depth
    Rh = Omega / (B + 2 * depth)
    Q_computed = Omega * ks * Rh ** (2 / 3) * slope**0.5

    return Q_computed / Q - 1


def residual_invEngelund_cs(
    h: float, Q: float, y: np.ndarray, z: np.ndarray, ks: float, slope: float
) -> float:
    """
    Engelund formula residual for finding water surface elevation of a composite cross-section for a given flow discharge.

    Returns Q_computed/Q - 1, which equals 0 when correct h is found.

    Parameters
    ----------
    h : float
        Water surface elevation to test (m)
    Q : float
        Target discharge (m³/s)
    y : ndarray
        Horizontal coordinates along cross-section (m)
    z : ndarray
        Bed elevation at each point (m)
    ks : float
        Strickler coefficient (m^(1/3)/s)
    slope : float
        Channel slope (m/m)

    Returns
    -------
    float
        Residual: Q_Engelund/Q - 1
    """
    Q_Eng = 0.0
    num_points = y.size
    wet_points = z < h

    depth = np.zeros_like(y)
    depth[wet_points] = h - z[wet_points]

    for i in range(num_points - 1):
        depth_L = depth[i]
        depth_R = depth[i + 1]

        dy = y[i + 1] - y[i]
        dz = z[i + 1] - z[i]

        # Handle partial submersion
        if depth_L == 0 and depth_R == 0:
            dy = 0
        elif depth_L == 0 and depth_R > 0:
            dy = dy * (depth_R / (-dz))
        elif depth_L > 0 and depth_R == 0:
            dy = dy * (depth_L / dz)

        depth_avg = 0.5 * (depth_L + depth_R)
        Q_Eng += dy * slope**0.5 * ks * depth_avg ** (5 / 3)

    return Q_Eng / Q - 1


def steady_flow_solver(
    Q: float,
    slope: float,
    ks: float,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    tol: float = 1e-6,
    Qmin: float = 1e-6,
) -> tuple[float, float, float]:
    """
    Solves for steady flow depth, cross-section area, and velocity.

    Uses scipy.optimize.fsolve to find the flow depth that satisfies the Engelund method integrating over cross-section geometry, using Strickler formula for local flow resistance.

    Parameters
    ----------
    Q : float
        Water discharge (cubic meters per second, m^3/s).
    slope : float
        Channel bed slope (dimensionless, m/m).
    ks : float
        Strickler coefficient (m^(1/3)/s).
    y_coords : ndarray, optional
        Horizontal coordinates along cross-section (m).
    z_coords : ndarray, optional
        Bed elevation at each point (m).
    tol : float, optional
        Convergence tolerance for fsolve. Default is 1e-6.
    Qmin : float, optional
        Minimum discharge threshold to avoid numerical issues. If Q<Qmin, skips numerical computations and returns minimum bed elevation as water surface. Default is 1e-6 m^3/s.

    Returns
    -------
    h : float
        Water surface elevation (m).
    Omega : float
        Cross-sectional wetted area (m^2).
    U : float
        Mean flow velocity (m/s).

    Raises
    ------
    ValueError
        If required parameters are missing or invalid for the selected mode.
    """

    # Handle zero or very small discharge
    if Q < Qmin:
        print(
            "Warning: Discharge Q is very small or zero. Returning minimum bed elevation as water surface."
        )
        h = float(np.min(z_coords))
        Omega = 0.0
        U = 0.0
        return h, Omega, U

    # Find water surface elevation using Engelund method
    h0 = np.min(z_coords) + 1.0  # Initial guess

    try:
        solution = fsolve(
            residual_invEngelund_cs,
            h0,
            args=(Q, y_coords, z_coords, ks, slope),
            xtol=tol,
        )
        h = float(solution[0])
    except Exception as e:
        raise ValueError(
            f"Failed to converge for arbitrary cross-section: {e}\n"
            f"Q={Q}, slope={slope}, ks={ks}"
        )

    # Ensure valid result
    if not np.isfinite(h) or h < np.min(z_coords):
        print(
            f"Warning: Non-physical water surface elevation computed: h={h:.2e}. "
            "Returning minimum bed elevation as water surface and zero area and flow velocity."
        )
        h = np.min(z_coords)
        Omega = 0.0
        U = 0.0
        return h, Omega, U

    # Compute wetted area and average flow velocity
    N = y_coords.size
    wet = z_coords < h
    depth = np.zeros_like(y_coords)
    depth[wet] = h - z_coords[wet]

    Omega = 0.0  # wetted area

    for i in range(N - 1):
        depth_L = depth[i]
        depth_R = depth[i + 1]

        dy = y_coords[i + 1] - y_coords[i]
        dz = z_coords[i + 1] - z_coords[i]

        # Handle partial submersion
        if depth_L == 0 and depth_R == 0:
            dy = 0
        elif depth_L == 0 and depth_R > 0:
            dy = dy * (depth_R / (-dz))
        elif depth_L > 0 and depth_R == 0:
            dy = dy * (depth_L / dz)

        Omega += 0.5 * (depth_L + depth_R) * dy

    U = Q / Omega
    return h, Omega, U
