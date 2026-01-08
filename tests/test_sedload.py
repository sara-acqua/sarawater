import sys
import os
import numpy as np
import datetime

# Add parent directory to path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.sediment_load import (
    compute_sediment_load,
    steady_flow_solver,
    wilcock_crowe_2003,
)


# === Synthetic Test Data Setup ===
np.random.seed(42)

# Generate discharge time series
dates = [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(365)]
Q_series = np.random.lognormal(mean=1.5, sigma=0.5, size=len(dates))  # m³/s

# Sediment fractions (phi classes)
sed_range = np.arange(-9.5, 7.5 + 1, 1)
Fi = np.exp(-0.5 * ((sed_range + 2) / 2.5) ** 2)
Fi /= Fi.sum()  # Normalize to 1

# Channel parameters
B = 15.0  # channel width [m]
slope = 0.002  # bed slope
D50 = 0.002  # m
D84 = 0.005  # m


# === Tests ===


def test_steady_flow_solver_monotonicity():
    """Test that computed flow depth increases with discharge."""
    Q_values = [0.1, 1, 10, 100]
    h_values = [steady_flow_solver(B, slope, Q, D84)[0] for Q in Q_values]

    assert all(np.diff(h_values) > 0), "Flow depth should increase monotonically with Q"


def test_wilcock_crowe_behavior():
    """Test basic behavior of Wilcock & Crowe 2003 sediment transport model."""
    h = 1.2  # m
    Qsi = wilcock_crowe_2003(Fi, slope, B, h, D50, D84)

    # Ensure outputs are positive or zero
    assert np.all(Qsi >= 0), "Sediment transport should be non-negative"

    # Ensure some sediment is actually moving
    assert Qsi.sum() > 0, "Total sediment transport should be > 0 for typical flow"


def test_compute_sediment_load_output_shape():
    """Test DataFrame output has expected columns and shape."""
    df = compute_sediment_load(Q_series[:10], dates[:10], B, slope, Fi)

    # Check structure
    assert "Datetime" in df.columns
    assert "Q" in df.columns
    assert "h" in df.columns
    assert "qS_total" in df.columns

    # Check phi columns
    expected_cols = [f"qS_phi_{phi}" for phi in sed_range]
    for col in expected_cols:
        assert col in df.columns

    # Check dimensions
    assert len(df) == 10, "DataFrame should have one row per time step"


def test_sediment_load_sensitivity():
    """Test sediment load increases with discharge (approximate monotonicity)."""
    Q_small = [0.1, 1, 10]
    dates_small = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    df_small = compute_sediment_load(Q_small, dates_small, B, slope, Fi)

    qS_totals = df_small["qS_total"].values
    assert all(
        np.diff(qS_totals) > 0
    ), "Total sediment load should increase with discharge"


def test_consistency_between_runs():
    """Test deterministic behavior given same input and random seed."""
    df1 = compute_sediment_load(Q_series[:5], dates[:5], B, slope, Fi)
    df2 = compute_sediment_load(Q_series[:5], dates[:5], B, slope, Fi)

    np.testing.assert_allclose(df1["qS_total"], df2["qS_total"], rtol=1e-10)


def test_no_transport_when_zero_flow():
    """Ensure no sediment transport occurs when discharge is zero."""
    Q_zero = np.zeros(3)
    dates_zero = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(3)
    ]
    df = compute_sediment_load(Q_zero, dates_zero, B, slope, Fi)
    assert np.allclose(df["qS_total"], 0), "Sediment transport must be zero for Q=0"


def test_numerical_stability_small_h():
    """Check that the solver does not crash for very small flow or roughness."""
    h, A, v = steady_flow_solver(B, slope, 0.001, D84)
    assert h >= 0, "Depth should never be negative"
    assert np.isfinite(h), "Depth should be finite"
    assert np.isfinite(v), "Velocity should be finite"


def test_solver_low_Q_stability_range():
    """
    Ensure that the steady flow solver remains stable and finite for very small discharges.
    """

    # Typical channel parameters
    B = 15.0  # width [m]
    slope = 0.002  # slope [-]
    D84 = 0.005  # sediment size [m]

    # Test a wide range of small discharges (including zero)
    discharges = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]

    for Q in discharges:
        h, A, v = steady_flow_solver(B, slope, Q, D84)

        # Assert non-negative and finite outputs
        assert np.isfinite(h), f"Depth not finite for Q={Q}"
        assert np.isfinite(A), f"Area not finite for Q={Q}"
        assert np.isfinite(v), f"Velocity not finite for Q={Q}"

        assert h >= 0.0, f"Depth negative for Q={Q}"
        assert A >= 0.0, f"Area negative for Q={Q}"
        assert v >= 0.0, f"Velocity negative for Q={Q}"

        # For very small Q, depth and velocity should remain small
        if Q < 1e-2:
            assert h < 0.1, f"Unrealistic depth for very small Q={Q}: {h}"
            assert v < 2.0, f"Unrealistic velocity for very small Q={Q}: {v}"


def test_solver_high_Q_stability_range():
    """
    Ensure that the steady flow solver remains stable and finite for very large discharges.
    """

    # Channel geometry and slope representative of mountain and lowland rivers
    B = 15.0  # channel width [m]
    slope = 0.002  # slope [-]
    D84 = 0.005  # characteristic grain size [m]

    # Large discharges to test hydraulic realism
    discharges = [10, 20, 50, 100]

    for Q in discharges:
        h, A, v = steady_flow_solver(B, slope, Q, D84)

        # Must remain finite and positive
        assert np.isfinite(h), f"Depth not finite for Q={Q}"
        assert np.isfinite(A), f"Area not finite for Q={Q}"
        assert np.isfinite(v), f"Velocity not finite for Q={Q}"

        assert h > 0.0, f"Depth should be positive for Q={Q}"
        assert A > 0.0, f"Area should be positive for Q={Q}"
        assert v > 0.0, f"Velocity should be positive for Q={Q}"

        # Check physically realistic ranges
        assert h < 50.0, f"Unrealistically large depth for Q={Q}: {h:.2f} m"
        assert v < 15.0, f"Unrealistically high velocity for Q={Q}: {v:.2f} m/s"


if __name__ == "__main__":
    test_steady_flow_solver_monotonicity()
    test_wilcock_crowe_behavior()
    test_compute_sediment_load_output_shape()
    test_sediment_load_sensitivity()
    test_consistency_between_runs()
    test_no_transport_when_zero_flow()
    test_numerical_stability_small_h()
    test_solver_low_Q_stability_range()
    test_solver_high_Q_stability_range()
    print("All sediment_load tests passed successfully.")
