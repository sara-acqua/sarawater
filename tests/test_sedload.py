import sys
import os
import numpy as np
import datetime

# Add parent directory to path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.sediment_load import (
    compute_sediment_load,
    wilcock_crowe_2003,
    meyer_peter_mueller,
    transport_capacity,
    integrate_transport_across_section,
    shields_parameter,
    PHI_RANGE,
    DMI,
)
from sarawater.hydraulics import steady_flow_solver, shear_stress

# === Synthetic Test Data Setup ===
np.random.seed(42)

# Generate discharge time series
dates = [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
Q_series = np.random.lognormal(mean=1.5, sigma=0.5, size=len(dates))  # m³/s

# Sediment fractions (phi classes) - use PHI_RANGE from sediment_load module
Fi = np.exp(-0.5 * ((PHI_RANGE + 2) / 2.5) ** 2)  # normal distribution around phi=-2
Fi /= Fi.sum()  # Normalize to 1

# Channel parameters
B = 15.0  # channel width [m]
slope = 0.002  # bed slope
ks = 20.0  # Strickler coefficient [m^(1/3)/s]

# Create simple rectangular cross-section
y_coords = np.array([0, B])
z_coords = np.array([0, 0])  # Flat bed

# Physical constants
rho_w = 1000.0  # kg/m³
rho_s = 2650.0  # kg/m³
g = 9.81  # m/s²
Delta = (rho_s - rho_w) / rho_w
theta_c = 0.047


# === Tests ===


def test_wilcock_crowe_behavior():
    """Test basic behavior of Wilcock & Crowe 2003 sediment transport model."""
    # Create some test theta values (Shields parameters)
    theta_i = np.full(len(PHI_RANGE), 0.05)  # Shields parameter for each grain class

    qsi = wilcock_crowe_2003(theta_i, Fi)

    # Ensure outputs are positive or zero
    assert np.all(qsi >= 0), "Sediment transport should be non-negative"

    # Ensure some sediment is actually moving
    assert qsi.sum() > 0, "Total sediment transport should be > 0 for typical flow"


def test_compute_sediment_load_output_shape():
    """Test DataFrame output has expected columns and shape."""
    df = compute_sediment_load(
        Q_series[:10], dates[:10], y_coords, z_coords, slope, ks, Fi
    )

    # Check structure
    assert "Datetime" in df.columns
    assert "Q" in df.columns
    assert "h" in df.columns
    assert "Qs_total" in df.columns

    # Check phi columns
    expected_cols = [f"Qs_phi_{phi}" for phi in PHI_RANGE]
    for col in expected_cols:
        assert col in df.columns

    # Check dimensions
    assert len(df) == 10, "DataFrame should have one row per time step"


def test_sediment_load_sensitivity():
    """Test sediment load increases with discharge (approximate monotonicity)."""
    Q_small = np.array([0.1, 1, 10])
    dates_small = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    df_small = compute_sediment_load(
        Q_small, dates_small, y_coords, z_coords, slope, ks, Fi
    )

    Qs_totals = df_small["Qs_total"].values
    assert all(
        np.diff(Qs_totals) > 0
    ), "Total sediment load should increase with discharge"


def test_consistency_between_runs():
    """Test deterministic behavior given same input and random seed."""
    df1 = compute_sediment_load(
        Q_series[:5], dates[:5], y_coords, z_coords, slope, ks, Fi
    )
    df2 = compute_sediment_load(
        Q_series[:5], dates[:5], y_coords, z_coords, slope, ks, Fi
    )

    np.testing.assert_allclose(df1["Qs_total"], df2["Qs_total"], rtol=1e-10)


def test_no_transport_when_zero_flow():
    """Ensure no sediment transport occurs when discharge is zero."""
    Q_zero = np.zeros(3)
    dates_zero = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    df = compute_sediment_load(Q_zero, dates_zero, y_coords, z_coords, slope, ks, Fi)
    assert np.allclose(df["Qs_total"], 0), "Sediment transport must be zero for Q=0"


def test_transport_capacity_selector():
    """Test that transport_capacity function correctly selects between formulas."""
    theta_i = np.full(len(PHI_RANGE), 0.05)

    # Test Wilcock & Crowe selection
    qsi_wc = transport_capacity(theta_i, Fi, transport_formula="wilcock_crowe")
    assert np.all(qsi_wc >= 0), "Wilcock-Crowe transport should be non-negative"
    assert qsi_wc.sum() > 0, "Total transport should be positive"

    # Test MPM selection
    qsi_mpm = transport_capacity(theta_i, Fi, transport_formula="mpm")
    assert np.all(qsi_mpm >= 0), "MPM transport should be non-negative"
    assert qsi_mpm.sum() > 0, "Total transport should be positive"

    # Test invalid formula
    try:
        transport_capacity(theta_i, Fi, transport_formula="invalid")
        assert False, "Should raise ValueError for invalid formula"
    except ValueError as e:
        assert "Unknown transport formula" in str(e)


def test_meyer_peter_mueller_behavior():
    """Test basic behavior of Meyer-Peter & Müller sediment transport model."""
    # Test with Shields parameters above critical threshold
    theta_i = np.full(len(PHI_RANGE), 0.1)  # Above typical theta_c = 0.047

    qsi = meyer_peter_mueller(theta_i, Fi, theta_c=0.047)

    # Ensure outputs are positive or zero
    assert np.all(qsi >= 0), "Sediment transport should be non-negative"

    # Ensure some sediment is moving (theta > theta_c)
    assert qsi.sum() > 0, "Total sediment transport should be > 0 when theta > theta_c"

    # Test with Shields parameters below critical threshold
    theta_i_low = np.full(len(PHI_RANGE), 0.01)  # Below theta_c = 0.047
    qsi_low = meyer_peter_mueller(theta_i_low, Fi, theta_c=0.047)

    # Should be zero or very small when below threshold
    assert qsi_low.sum() < 1e-10, "Transport should be negligible when theta < theta_c"


def test_integrate_transport_across_section():
    """Test integration of unit transport rates across a cross-section."""
    # Create a simple rectangular channel
    y_test = np.array([0, 5, 10, 15])
    z_test = np.array([0, 0, 0, 0])  # Flat bed
    h_test = 1.0  # Water surface elevation

    # Create uniform unit transport rates for 3 grain classes
    # Shape should be (n_classes, N) where N is number of points
    n_classes = 3
    n_points = len(y_test)
    qs_test = np.ones((n_classes, n_points)) * 0.01  # m²/s

    # Integrate
    Qsi = integrate_transport_across_section(qs_test, y_test, z_test, h_test)

    # For a rectangular channel with uniform qs, Qs = qs * width
    expected = 0.01 * 15  # width = 15 m

    assert Qsi.shape == (n_classes,), "Should return one value per grain class"
    for i in range(n_classes):
        assert np.isclose(
            Qsi[i], expected, rtol=0.01
        ), f"Class {i}: expected {expected}, got {Qsi[i]}"


def test_mpm_steady_flow_verification():
    """
    Test MPM formula with steady flow discharge and rectangular cross-section.
    All values from sediment load time series must match theoretical MPM values.

    This test verifies that the compute_sediment_load function correctly implements
    the Meyer-Peter & Müller (1948) formula by comparing computed transport rates
    against theoretical values calculated directly from the formula.
    """
    # Test parameters
    Q_steady = 5.0  # constant discharge [m³/s]
    n_timesteps = 5
    dates_steady = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i)
        for i in range(n_timesteps)
    ]
    Q_series_steady = np.full(n_timesteps, Q_steady)

    # Compute sediment load using MPM formula
    df_mpm = compute_sediment_load(
        Q_series_steady,
        dates_steady,
        y_coords,
        z_coords,
        slope,
        ks,
        Fi,
        transport_formula="mpm",
        mpm_theta_c=theta_c,
        rho_w=rho_w,
        rho_s=rho_s,
        g=g,
    )

    # Verify all timesteps have the same values (steady flow)
    for col in df_mpm.columns:
        if col != "Datetime":
            values = df_mpm[col].values
            # All values should be identical for steady flow
            assert np.allclose(
                values, values[0], rtol=1e-10
            ), f"Column {col} should be constant for steady flow"

    # Now compute the theoretical MPM transport rate manually
    # Step 1: Solve for flow depth
    h_computed = df_mpm["h"].iloc[0]
    depth = h_computed - z_coords

    # Step 2: Compute theoretical transport for each stripe and integrate
    theta_i_theoretical = np.zeros((len(DMI), len(depth)))
    qs_theoretical = np.zeros_like(theta_i_theoretical)

    for stripe_idx, stripe_depth in enumerate(depth):
        tau_b = shear_stress(rho_w, g, stripe_depth, slope)
        theta_i_theoretical[:, stripe_idx] = shields_parameter(
            tau_b, rho_w, rho_s, g, DMI
        )

        # Apply MPM formula: qb_i = 8 * max(theta_i - theta_c, 0)^{3/2} * sqrt(g * Delta * d_i^3)
        phi_i = np.maximum(theta_i_theoretical[:, stripe_idx] - theta_c, 0.0)
        qb_i = 8.0 * (phi_i**1.5) * np.sqrt(g * Delta * DMI**3)
        qs_theoretical[:, stripe_idx] = qb_i * Fi

    # Step 3: Integrate across section to obtain total transport for each grain class
    Qs_theoretical = integrate_transport_across_section(
        qs_theoretical, y_coords, z_coords, h_computed
    )

    # Step 4: Compare computed vs theoretical values for the total transport
    Qs_total_computed = df_mpm["Qs_total"].iloc[0]
    Qs_total_theoretical = Qs_theoretical.sum()

    assert np.isclose(
        Qs_total_computed, Qs_total_theoretical, rtol=1e-8
    ), f"Total transport mismatch: computed={Qs_total_computed}, theoretical={Qs_total_theoretical}"

    # Check per-class transport
    for j, phi in enumerate(PHI_RANGE):
        col_name = f"Qs_phi_{phi}"
        Qs_computed = df_mpm[col_name].iloc[0]
        Qs_theo = Qs_theoretical[j]

        assert np.isclose(
            Qs_computed, Qs_theo, rtol=1e-8
        ), f"Phi class {phi}: computed={Qs_computed}, theoretical={Qs_theo}"

    # Verify all timesteps match the theoretical value (since flow is steady)
    for i in range(n_timesteps):
        assert np.isclose(
            df_mpm["Qs_total"].iloc[i], Qs_total_theoretical, rtol=1e-8
        ), f"Timestep {i} does not match theoretical value"


if __name__ == "__main__":
    test_wilcock_crowe_behavior()
    test_compute_sediment_load_output_shape()
    test_sediment_load_sensitivity()
    test_consistency_between_runs()
    test_no_transport_when_zero_flow()
    test_transport_capacity_selector()
    test_meyer_peter_mueller_behavior()
    test_integrate_transport_across_section()
    test_mpm_steady_flow_verification()
    print("All sediment_load tests passed successfully.")
