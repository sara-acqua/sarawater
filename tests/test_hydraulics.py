import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.hydraulics import (
    shear_stress,
    steady_flow_solver,
    residual_uniFlow_rect,
    residual_invEngelund_cs,
)


# === Test Data Setup ===

# Channel parameters
B = 15.0  # channel width [m]
slope = 0.002  # bed slope
ks = 20.0  # Strickler coefficient [m^(1/3)/s]

# Create simple rectangular cross-section
y_coords = np.array([0, B])
z_coords = np.array([0, 0])  # Flat bed

# Physical constants
rho_w = 1000.0  # kg/m³
g = 9.81  # m/s²


# === Tests ===


def test_shear_stress_basic():
    """Test basic shear stress calculation."""
    h = 1.0  # 1 meter depth
    tau = shear_stress(rho_w, g, h, slope)

    # Expected: tau = rho_w * g * h * slope = 1000 * 9.81 * 1.0 * 0.002
    expected = rho_w * g * h * slope

    assert np.isclose(tau, expected), f"Expected {expected}, got {tau}"
    assert tau > 0, "Shear stress should be positive for positive inputs"


def test_shear_stress_proportionality():
    """Test that shear stress scales correctly with depth and slope."""
    h = 2.0

    # Double depth should double shear stress
    tau1 = shear_stress(rho_w, g, 1.0, slope)
    tau2 = shear_stress(rho_w, g, 2.0, slope)
    assert np.isclose(tau2, 2 * tau1), "Shear stress should scale linearly with depth"

    # Double slope should double shear stress
    tau3 = shear_stress(rho_w, g, h, 0.001)
    tau4 = shear_stress(rho_w, g, h, 0.002)
    assert np.isclose(tau4, 2 * tau3), "Shear stress should scale linearly with slope"


def test_steady_flow_solver_monotonicity():
    """Test that computed flow depth increases with discharge."""
    Q_values = [0.1, 1, 10, 100]
    h_values = [
        steady_flow_solver(Q, slope, ks, y_coords, z_coords)[0] for Q in Q_values
    ]

    assert all(np.diff(h_values) > 0), "Flow depth should increase monotonically with Q"


def test_numerical_stability_small_h():
    """Check that the solver does not crash for very small flow or roughness."""
    h, A, U, P = steady_flow_solver(0.001, slope, ks, y_coords, z_coords)
    assert h >= 0, "Depth should never be negative"
    assert np.isfinite(h), "Depth should be finite"
    assert np.isfinite(A), "Area should be finite"
    assert np.isfinite(U), "Velocity should be finite"
    assert np.isfinite(P), "Wetted perimeter should be finite"


def test_solver_low_Q_stability_range():
    """
    Ensure that the steady flow solver remains stable and finite for very small discharges.
    """
    # Test a wide range of small discharges (including zero)
    discharges = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

    for Q in discharges:
        h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)

        # Assert non-negative and finite outputs
        assert np.isfinite(h), f"Depth not finite for Q={Q}"
        assert np.isfinite(A), f"Area not finite for Q={Q}"
        assert np.isfinite(U), f"Velocity not finite for Q={Q}"
        assert np.isfinite(P), f"Wetted perimeter not finite for Q={Q}"

        assert h >= 0.0, f"Depth negative for Q={Q}"
        assert A >= 0.0, f"Area negative for Q={Q}"
        assert U >= 0.0, f"Velocity negative for Q={Q}"
        assert P >= 0.0, f"Wetted perimeter negative for Q={Q}"

        # For very small Q, depth and velocity should remain small
        if Q < 1e-2:
            assert h < 0.1, f"Unrealistic depth for very small Q={Q}: {h}"
            assert U < 2.0, f"Unrealistic velocity for very small Q={Q}: {U}"


def test_solver_high_Q_stability_range():
    """
    Ensure that the steady flow solver remains stable and finite for very large discharges.
    """
    # Large discharges to test hydraulic realism
    discharges = [10, 20, 50, 100]

    for Q in discharges:
        h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)

        # Must remain finite and positive
        assert np.isfinite(h), f"Depth not finite for Q={Q}"
        assert np.isfinite(A), f"Area not finite for Q={Q}"
        assert np.isfinite(U), f"Velocity not finite for Q={Q}"
        assert np.isfinite(P), f"Wetted perimeter not finite for Q={Q}"

        assert h > 0.0, f"Depth should be positive for Q={Q}"
        assert A > 0.0, f"Area should be positive for Q={Q}"
        assert U > 0.0, f"Velocity should be positive for Q={Q}"
        assert P > 0.0, f"Wetted perimeter should be positive for Q={Q}"

        # Check physically realistic ranges
        assert h < 50.0, f"Unrealistically large depth for Q={Q}: {h:.2f} m"
        assert U < 15.0, f"Unrealistically high velocity for Q={Q}: {U:.2f} m/s"


def test_steady_flow_solver_continuity():
    """Test that Q = A * U (continuity equation) holds."""
    Q = 5.0
    h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)

    Q_computed = A * U
    assert np.isclose(
        Q, Q_computed, rtol=1e-6
    ), f"Continuity violated: Q={Q}, A*U={Q_computed}"


def test_residual_uniFlow_rect():
    """Test residual function for rectangular channel."""
    from scipy.optimize import fsolve

    Q = 10.0

    # Use fsolve to find the depth that satisfies the residual equation
    h_solution = fsolve(residual_uniFlow_rect, 1.0, args=(Q, B, ks, slope))[0]

    # The residual should be close to zero at the solution
    residual = residual_uniFlow_rect(h_solution, Q, B, ks, slope)

    assert np.abs(residual) < 1e-6, f"Residual should be near zero, got {residual}"

    # Test that depth is physically reasonable
    assert h_solution > 0, "Depth should be positive"
    assert h_solution < 10, "Depth should be reasonable for Q=10 m³/s"


def test_residual_invEngelund_cs():
    """Test residual function for composite cross-section (Engelund method)."""
    Q = 10.0

    # Use steady_flow_solver to find correct water surface elevation
    h_correct, _, U_correct, P_correct = steady_flow_solver(
        Q, slope, ks, y_coords, z_coords
    )

    # The residual should be close to zero at the correct elevation
    residual = residual_invEngelund_cs(h_correct, Q, y_coords, z_coords, ks, slope)

    assert np.abs(residual) < 1e-3, f"Residual should be near zero, got {residual}"


def test_steady_flow_solver_complex_section():
    """Test steady flow solver with a more complex cross-section (trapezoidal)."""
    # Create a simple trapezoidal channel: banks at 45 degrees
    y_trap = np.array([0, 2, 10, 12])
    z_trap = np.array([2, 0, 0, 2])  # V-shaped with flat bottom

    Q = 5.0
    h, A, U, P = steady_flow_solver(Q, slope, ks, y_trap, z_trap)

    # Basic sanity checks
    assert h > np.min(z_trap), "Water surface should be above minimum bed elevation"
    assert h < np.min(z_trap) + 5, "Water surface should be reasonable"
    assert A > 0, "Wetted area should be positive"
    assert P > 0, "Wetted perimeter should be positive"
    assert U > 0, "Velocity should be positive"

    # Check continuity
    assert np.isclose(Q, A * U, rtol=1e-6), "Continuity equation should hold"


def test_steady_flow_solver_symmetry():
    """Test that symmetric cross-sections produce symmetric results."""
    # Create a symmetric trapezoidal channel
    y_sym = np.array([0, 5, 10, 15])
    z_sym = np.array([1, 0, 0, 1])  # Symmetric trapezoid

    Q = 3.0
    h, A, U, P = steady_flow_solver(Q, slope, ks, y_sym, z_sym)

    # Compute depth at each point
    depth = h - z_sym
    depth[depth < 0] = 0

    # For a symmetric channel, depths should be symmetric (allowing small numerical error)
    assert np.allclose(
        depth[0], depth[-1], atol=1e-6
    ), f"Depths should be symmetric: {depth[0]} vs {depth[-1]}"
    assert np.allclose(
        depth[1], depth[-2], atol=1e-6
    ), f"Depths should be symmetric: {depth[1]} vs {depth[-2]}"


def test_steady_flow_solver_zero_discharge():
    """Test behavior with zero discharge."""
    Q = 0.0
    h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)

    # Should return minimum bed elevation and zero area/velocity
    assert h == np.min(z_coords), "Water surface should be at minimum bed elevation"
    assert A == 0, "Area should be zero"
    assert U == 0, "Velocity should be zero"
    assert P == 0, "Wetted perimeter should be zero"


def test_uniform_flow_depth_computation():
    Q = 10.0
    Bwide = 100.0  # Wide channel to approximate uniform flow

    y_coords = np.linspace(0, Bwide, 100)
    z_coords = np.zeros_like(y_coords)  # Flat bed

    uniflow_depth = (Q / (Bwide * ks * slope**0.5)) ** (3 / 5)
    uniflow_h = uniflow_depth + z_coords[0]
    uniflow_U = Q / (Bwide * uniflow_depth)
    uniflow_P = Bwide  # Wetted perimeter for wide rectangular channel

    h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)

    assert np.isclose(
        h, uniflow_h, rtol=1e-3
    ), f"Computed depth {h} does not match expected {uniflow_h}"
    assert np.isclose(
        U, uniflow_U, rtol=1e-3
    ), f"Computed velocity {U} does not match expected {uniflow_U}"
    assert np.isclose(
        P, uniflow_P, rtol=1e-3
    ), f"Computed perimeter {P} does not match expected {uniflow_P}"


def test_uniform_flow_triangular_cross_section():
    """Test the computation of the uniform flow depth for a triangular cross section with 45-degrees banks."""
    rtol = 3e-2  # Relaxed relative tolerance to 3% to account for numerical differences

    Q = 10.0
    y_coords = np.linspace(-10, 10, 1001)
    z_coords = np.abs(y_coords)

    uniflow_depth = (Q / (ks * slope**0.5) * 8 ** (1 / 3)) ** (3 / 8)
    uniflow_h = uniflow_depth + np.min(z_coords)
    uniflow_A = uniflow_h**2
    uniflow_U = Q / uniflow_A
    uniflow_P = 2 * uniflow_h * (2**0.5)

    h, A, U, P = steady_flow_solver(Q, slope, ks, y_coords, z_coords)
    assert np.isclose(
        h, uniflow_h, rtol=rtol
    ), f"Computed depth {h} does not match expected {uniflow_h}"
    assert np.isclose(
        A, uniflow_A, rtol=rtol
    ), f"Computed area {A} does not match expected {uniflow_A}"
    assert np.isclose(
        U, uniflow_U, rtol=3e-2
    ), f"Computed velocity {U} does not match expected {uniflow_U}"
    assert np.isclose(
        P, uniflow_P, rtol=3e-2
    ), f"Computed perimeter {P} does not match expected {uniflow_P}"


if __name__ == "__main__":
    test_shear_stress_basic()
    test_shear_stress_proportionality()
    test_steady_flow_solver_monotonicity()
    test_numerical_stability_small_h()
    test_solver_low_Q_stability_range()
    test_solver_high_Q_stability_range()
    test_steady_flow_solver_continuity()
    test_residual_uniFlow_rect()
    test_residual_invEngelund_cs()
    test_steady_flow_solver_complex_section()
    test_steady_flow_solver_symmetry()
    test_steady_flow_solver_zero_discharge()
    test_uniform_flow_depth_computation()
    print("All hydraulics tests passed successfully.")
