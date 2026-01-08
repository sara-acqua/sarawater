import sys, os
import numpy as np
import datetime

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.IHA import compute_IHA, compute_IHA_index

# Create test data
dates = [
    datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(365 * 10)
]
# Generate lognormal distribution for natural flow
np.random.seed(42)  # for reproducibility
mu, sigma = 2.0, 1.0  # parameters for lognormal distribution
Qnat = np.random.lognormal(mu, sigma, len(dates))
QS = Qnat * 0.5  # Modified flow at half the natural flow


def test_IHA_groups():
    """Test that all IHA groups are computed correctly"""
    IHA_groups = compute_IHA(Qnat, QS, dates)

    # Test we have all 5 groups
    assert len(IHA_groups) == 5
    assert all([f"Group{i+1}" in IHA_groups for i in range(5)])

    # Test Group 1: Monthly statistics
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    for month in months:
        assert f"mean_{month}" in IHA_groups["Group1"]

    # Test Group 2: Moving averages and base flow
    assert "base_flow" in IHA_groups["Group2"]
    assert "zero_flow_days" in IHA_groups["Group2"]
    for window in [1, 3, 7, 30, 90]:
        assert f"moving_avg_{window}d_min" in IHA_groups["Group2"]
        assert f"moving_avg_{window}d_max" in IHA_groups["Group2"]

    # Test Group 3: Timing of annual extremes
    assert "julian_day_max" in IHA_groups["Group3"]
    assert "julian_day_min" in IHA_groups["Group3"]

    # Test Group 4: Pulse analysis
    assert "low_pulse_count" in IHA_groups["Group4"]
    assert "high_pulse_count" in IHA_groups["Group4"]
    assert "low_pulse_duration" in IHA_groups["Group4"]
    assert "high_pulse_duration" in IHA_groups["Group4"]

    # Test Group 5: Flow variations
    assert "positive_variation_median" in IHA_groups["Group5"]
    assert "negative_variation_median" in IHA_groups["Group5"]
    assert "flow_reversals" in IHA_groups["Group5"]


def test_IARI_computation():
    """Test IARI computation with different weights"""
    # Test with default weights
    _, iari_default = compute_IHA_index(Qnat, QS, dates, index_metric="IARI")
    assert "groups" in iari_default
    assert "aggregated" in iari_default
    assert len(iari_default["groups"]) == 5

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    _, iari_custom = compute_IHA_index(
        Qnat, QS, dates, index_metric="IARI", weights=weights
    )
    assert len(iari_custom["groups"]) == 5

    # Test weight validation
    try:
        invalid_weights = [0.1, 0.2, 0.3, 0.2]  # Only 4 weights
        compute_IHA_index(Qnat, QS, dates, index_metric="IARI", weights=invalid_weights)
        assert False, "Should have raised ValueError for invalid number of weights"
    except ValueError:
        pass

    try:
        invalid_weights = [0.1, 0.2, 0.3, 0.2, 0.1]  # Sum != 1
        compute_IHA_index(Qnat, QS, dates, index_metric="IARI", weights=invalid_weights)
        assert False, "Should have raised ValueError for weights not summing to 1"
    except ValueError:
        pass


def test_IARI_values():
    """Test IARI values for specific scenarios"""
    # Test when QS = Qnat (no alteration)
    _, iari_no_change = compute_IHA_index(Qnat, Qnat, dates, index_metric="IARI")

    # Test when QS = 0 (maximum alteration)
    QS_zero = np.zeros_like(Qnat)
    _, iari_max_change = compute_IHA_index(Qnat, QS_zero, dates, index_metric="IARI")
    assert np.all(iari_max_change["aggregated"] > 0)
    assert np.all(iari_max_change["aggregated"] > iari_no_change["aggregated"])


def test_normalized_IHA_computation():
    """Test normalized IHA computation with different weights"""
    # Test with default weights
    _, niha_default = compute_IHA_index(Qnat, QS, dates, index_metric="normalized_IHA")
    assert "groups" in niha_default
    assert "aggregated" in niha_default
    assert len(niha_default["groups"]) == 5

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    _, niha_custom = compute_IHA_index(
        Qnat, QS, dates, index_metric="normalized_IHA", weights=weights
    )
    assert len(niha_custom["groups"]) == 5

    # Test weight validation
    try:
        invalid_weights = [0.1, 0.2, 0.3, 0.2]  # Only 4 weights
        compute_IHA_index(
            Qnat, QS, dates, index_metric="normalized_IHA", weights=invalid_weights
        )
        assert False, "Should have raised ValueError for invalid number of weights"
    except ValueError:
        pass

    try:
        invalid_weights = [0.1, 0.2, 0.3, 0.2, 0.1]  # Sum != 1
        compute_IHA_index(
            Qnat, QS, dates, index_metric="normalized_IHA", weights=invalid_weights
        )
        assert False, "Should have raised ValueError for weights not summing to 1"
    except ValueError:
        pass


def test_normalized_IHA_values():
    """Test normalized IHA values for specific scenarios"""
    # Test when QS = Qnat (no alteration) - should give values close to 0
    _, niha_no_change = compute_IHA_index(
        Qnat, Qnat, dates, index_metric="normalized_IHA"
    )
    assert np.all(niha_no_change["aggregated"] >= 0)
    # When flows are identical, normalized IHA should be close to 0
    assert np.all(niha_no_change["aggregated"] < 0.01)

    # Test when QS = 0.5 * Qnat (moderate alteration)
    QS_half = Qnat * 0.5
    _, niha_half = compute_IHA_index(
        Qnat, QS_half, dates, index_metric="normalized_IHA"
    )
    assert np.all(niha_half["aggregated"] > 0)
    # Should show alteration since flows are different
    assert np.all(niha_half["aggregated"] > niha_no_change["aggregated"])

    # Test when QS = 0 (maximum alteration)
    QS_zero = np.zeros_like(Qnat) + 0.01  # Add small value to avoid division by zero
    _, niha_max_change = compute_IHA_index(
        Qnat, QS_zero, dates, index_metric="normalized_IHA"
    )
    assert np.all(niha_max_change["aggregated"] > 0)
    # Maximum alteration should have higher values than moderate alteration
    assert np.all(niha_max_change["aggregated"] > niha_half["aggregated"])


def test_index_metric_validation():
    """Test that invalid index_metric raises ValueError"""
    try:
        compute_IHA_index(Qnat, QS, dates, index_metric="invalid_metric")
        assert False, "Should have raised ValueError for invalid index_metric"
    except ValueError as e:
        assert "index_metric must be either 'IARI' or 'normalized_IHA'" in str(e)


def test_IHA_numerical_accuracy():
    """Test numerical accuracy of IHA indicators with synthetic data"""
    # Create simple synthetic data for one year with known properties
    dates_1y = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(365)
    ]

    # Sinusoidal flow with known properties
    t = np.linspace(0, 2 * np.pi, 365)
    Qnat_test = 10 + 5 * np.sin(t)  # Mean = 10, amplitude = 5
    QS_test = 8 + 4 * np.sin(t)  # Mean = 8, amplitude = 4

    IHA_groups = compute_IHA(Qnat_test, QS_test, dates_1y)

    # Test Group 1: Monthly means
    # January mean (days 0-30)
    jan_mean = np.mean(QS_test[:31])
    assert np.isclose(IHA_groups["Group1"]["mean_january"][0], jan_mean, rtol=1e-10)

    # July mean (days 181-211)
    jul_mean = np.mean(QS_test[181:212])
    assert np.isclose(IHA_groups["Group1"]["mean_july"][0], jul_mean, rtol=1e-10)

    # Test Group 2: Moving averages
    # Base flow should be mean of the series
    assert np.isclose(IHA_groups["Group2"]["base_flow"][0], 8.0, rtol=1e-10)

    # Zero flow days (sinusoidal flow ranges from 4 to 12, no zero flow)
    assert IHA_groups["Group2"]["zero_flow_days"][0] == 0

    # 7-day moving average extremes
    moving_avg_7d = np.convolve(QS_test, np.ones(7) / 7, mode="valid")
    assert np.isclose(
        IHA_groups["Group2"]["moving_avg_7d_min"][0], np.min(moving_avg_7d), rtol=1e-10
    )
    assert np.isclose(
        IHA_groups["Group2"]["moving_avg_7d_max"][0], np.max(moving_avg_7d), rtol=1e-10
    )

    # Test Group 3: Timing of extremes
    max_day = np.argmax(QS_test) + 1  # Adding 1 because Julian days start at 1
    min_day = np.argmin(QS_test) + 1
    assert IHA_groups["Group3"]["julian_day_max"][0] == max_day
    assert IHA_groups["Group3"]["julian_day_min"][0] == min_day

    # Test Group 4: Pulse analysis
    # Create simple pulse data
    pulse_dates = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(100)
    ]
    pulse_Qnat = np.ones(100) * 10
    pulse_QS = np.ones(100) * 10
    pulse_QS[10:20] = 5  # Low pulse for 10 days
    pulse_QS[50:70] = 15  # High pulse for 20 days

    pulse_IHA = compute_IHA(pulse_Qnat, pulse_QS, pulse_dates)
    assert pulse_IHA["Group4"]["low_pulse_count"][0] == 1
    assert pulse_IHA["Group4"]["high_pulse_count"][0] == 1
    assert pulse_IHA["Group4"]["low_pulse_duration"][0] == 10
    assert pulse_IHA["Group4"]["high_pulse_duration"][0] == 20

    # Test Group 5: Flow variations
    # Create simple variation data
    var_QS = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    var_dates = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(10)
    ]
    var_Qnat = np.ones_like(var_QS)

    var_IHA = compute_IHA(var_Qnat, var_QS, var_dates)
    assert np.isclose(
        var_IHA["Group5"]["positive_variation_median"][0], 1.0, rtol=1e-10
    )
    assert np.isclose(
        var_IHA["Group5"]["negative_variation_median"][0], -1.0, rtol=1e-10
    )
    assert (
        var_IHA["Group5"]["flow_reversals"][0] == 8
    )  # 8 changes between increasing/decreasing


def test_zero_flow_days():
    """Test zero_flow_days indicator with a time series containing zero-flow days"""
    # Create a simple time series with known zero-flow days
    dates_test = [
        datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(30)
    ]

    # Create flow data with 5 zero-flow days (days 5-9)
    Qnat_test = np.ones(30) * 10.0
    QS_test = np.ones(30) * 10.0
    QS_test[5:10] = 0.0  # 5 days with zero flow

    IHA_groups = compute_IHA(Qnat_test, QS_test, dates_test)

    # Should detect exactly 5 zero-flow days
    assert IHA_groups["Group2"]["zero_flow_days"][0] == 5

    # Test with flow below threshold but not exactly zero
    QS_test2 = np.ones(30) * 10.0
    QS_test2[10:15] = 0.0005  # 5 days below default threshold (0.001)

    IHA_groups2 = compute_IHA(Qnat_test, QS_test2, dates_test)
    assert IHA_groups2["Group2"]["zero_flow_days"][0] == 5

    # Test with custom threshold
    QS_test3 = np.ones(30) * 10.0
    QS_test3[2:8] = 0.5  # 6 days with low flow

    IHA_groups3 = compute_IHA(Qnat_test, QS_test3, dates_test, zero_flow_threshold=1.0)
    assert IHA_groups3["Group2"]["zero_flow_days"][0] == 6


def test_daily_averaging():
    """Test that sub-daily data is correctly averaged to daily resolution"""
    # Create sub-daily (hourly) data for 3 days
    start_date = datetime.datetime(2025, 1, 1, 0, 0, 0)
    hourly_dates = [
        start_date + datetime.timedelta(hours=x) for x in range(72)
    ]  # 3 days * 24 hours

    # Create hourly flow data with known daily averages
    # Day 1: all hours = 10 m³/s, average = 10
    # Day 2: all hours = 20 m³/s, average = 20
    # Day 3: all hours = 30 m³/s, average = 30
    hourly_Qnat = np.concatenate([np.ones(24) * 10, np.ones(24) * 20, np.ones(24) * 30])

    hourly_QS = np.concatenate([np.ones(24) * 8, np.ones(24) * 16, np.ones(24) * 24])

    # Compute IHA with hourly data
    IHA_hourly = compute_IHA(hourly_Qnat, hourly_QS, hourly_dates)

    # Create equivalent daily data
    daily_dates = [
        datetime.datetime(2025, 1, 1),
        datetime.datetime(2025, 1, 2),
        datetime.datetime(2025, 1, 3),
    ]
    daily_Qnat = np.array([10, 20, 30])
    daily_QS = np.array([8, 16, 24])

    # Compute IHA with daily data
    IHA_daily = compute_IHA(daily_Qnat, daily_QS, daily_dates)

    # The results should be identical since hourly data averages to the same daily values
    # Check Group 2: base_flow (which is the mean of the series)
    assert np.isclose(
        IHA_hourly["Group2"]["base_flow"][0],
        IHA_daily["Group2"]["base_flow"][0],
        rtol=1e-10,
    ), f"Base flow mismatch: hourly={IHA_hourly['Group2']['base_flow'][0]}, daily={IHA_daily['Group2']['base_flow'][0]}"

    # Check that the daily-averaged values are correct
    expected_daily_mean = np.mean([8, 16, 24])
    assert np.isclose(
        IHA_hourly["Group2"]["base_flow"][0], expected_daily_mean, rtol=1e-10
    ), f"Base flow should be {expected_daily_mean}, got {IHA_hourly['Group2']['base_flow'][0]}"

    # Test with more complex sub-daily patterns
    # Create hourly data where hours within each day vary
    hourly_dates_varying = [
        start_date + datetime.timedelta(hours=x) for x in range(48)
    ]  # 2 days

    # Day 1: hours vary linearly from 5 to 15 (average = 10)
    day1_hourly = np.linspace(5, 15, 24)
    # Day 2: hours vary linearly from 10 to 30 (average = 20)
    day2_hourly = np.linspace(10, 30, 24)

    hourly_Qnat_varying = np.concatenate([day1_hourly, day2_hourly])
    hourly_QS_varying = hourly_Qnat_varying * 0.8  # 80% of natural flow

    # Compute IHA with varying hourly data
    IHA_varying = compute_IHA(
        hourly_Qnat_varying, hourly_QS_varying, hourly_dates_varying
    )

    # Create daily data with correct averages
    daily_dates_2d = [datetime.datetime(2025, 1, 1), datetime.datetime(2025, 1, 2)]
    daily_Qnat_2d = np.array([10.0, 20.0])  # Averages of the hourly data
    daily_QS_2d = daily_Qnat_2d * 0.8

    IHA_daily_2d = compute_IHA(daily_Qnat_2d, daily_QS_2d, daily_dates_2d)

    # Compare base flow values
    assert np.isclose(
        IHA_varying["Group2"]["base_flow"][0],
        IHA_daily_2d["Group2"]["base_flow"][0],
        rtol=1e-10,
    ), f"Base flow mismatch with varying hourly data: varying={IHA_varying['Group2']['base_flow'][0]}, daily={IHA_daily_2d['Group2']['base_flow'][0]}"

    # Verify the average is correct
    expected_mean = np.mean([8.0, 16.0])  # 80% of [10, 20]
    assert np.isclose(
        IHA_varying["Group2"]["base_flow"][0], expected_mean, rtol=1e-10
    ), f"Base flow should be {expected_mean}, got {IHA_varying['Group2']['base_flow'][0]}"


if __name__ == "__main__":
    test_IHA_groups()
    test_IARI_computation()
    test_IARI_values()
    test_normalized_IHA_computation()
    test_normalized_IHA_values()
    test_index_metric_validation()
    test_IHA_numerical_accuracy()
    test_zero_flow_days()
    test_daily_averaging()
