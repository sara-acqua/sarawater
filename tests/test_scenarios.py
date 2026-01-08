import sys, os
import numpy as np
import datetime

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.scenarios as sc
import sarawater.reach as rch

# Global variable: shared Reach instance for all tests
dates = [
    datetime.datetime(2025, month, 1) for month in range(1, 13)
]  # January to December
Qnat = np.random.randint(1, 100, len(dates))  # Natural flow rate time series
test_scenarios_reach = rch.Reach("Global Reach", dates, Qnat, 50)


def test_scenario_basics():
    # Use the global Reach instance
    sc1 = sc.Scenario("Scenario 1", "This is the first scenario.", test_scenarios_reach)
    assert sc1.name == "Scenario 1"
    assert sc1.description == "This is the first scenario."
    assert np.array_equal(
        sc1.Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert (
        sc1.Qab_max == test_scenarios_reach.Qab_max
    )  # Ensure Qab_max is correctly fetched from the reach
    assert np.array_equal(
        sc1.dates, test_scenarios_reach.dates
    )  # Ensure dates are correctly fetched from the reach


def test_changing_reach_attrs():
    # Use a local Reach instance
    local_reach = rch.Reach("Local Reach", dates, Qnat, 50)
    sc1 = sc.Scenario("Scenario 1", "This is the first scenario.", local_reach)
    assert sc1.Qnat is not None  # Ensure Qnat is not None
    assert sc1.dates is not None  # Ensure dates are not None

    # Change the reach attributes
    new_Qnat = np.linspace(10, 200, num=100)
    new_dates = np.arange(100, 200)  # New dummy dates for the reach
    local_reach.Qnat = new_Qnat
    local_reach.dates = new_dates

    # Check if the scenario reflects the changes in the reach attributes
    assert np.array_equal(sc1.Qnat, new_Qnat)  # Ensure Qnat is updated correctly
    assert np.array_equal(sc1.dates, new_dates)  # Ensure dates are updated correctly


def test_const_scenario_with_dates():

    # Define monthly constant flow rates
    QR_months = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    # Create a ConstScenario with the reach and QR_months
    const_scenario = sc.ConstScenario(
        "Const Scenario with Dates",
        "A constant scenario with monthly flow rates.",
        test_scenarios_reach,
        QR_months,
    )

    # Verify that the QR mapping is carried out properly
    assert np.array_equal(const_scenario.QR, QR_months), "QR mapping failed"
    for i, month in enumerate(range(1, 13)):
        month_mask = np.array([date.month == month for date in dates])
        assert np.all(
            const_scenario.QR[month_mask] == QR_months[i]
        ), f"QR mapping failed for month {month}"


def test_const_scenario():
    # Use the global Reach instance
    sc1 = sc.ConstScenario(
        "Scenario 1", "This is the first scenario.", test_scenarios_reach, [1] * 12
    )
    assert sc1.name == "Scenario 1"
    assert sc1.description == "This is the first scenario."
    assert np.array_equal(
        sc1.Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert sc1.QR_months[0] == 1
    assert len(sc1.QR_months) == 12


def test_prop_scenario():
    # Use the global Reach instance
    sc1 = sc.PropScenario(
        "Scenario 1",
        "This is the first proportional scenario.",
        test_scenarios_reach,
        1,
        0.3,
        0.5,
        50,
    )
    assert sc1.name == "Scenario 1"
    assert sc1.description == "This is the first proportional scenario."
    assert np.array_equal(
        sc1.Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert sc1.QRbase == 1
    assert sc1.c_Qin == 0.3
    assert sc1.QRmin == 0.5
    assert sc1.QRmax == 50


def test_scenario_IARI_computation():
    # Create a scenario with known properties
    QR_months = [10] * 12  # Constant release of 10 units for all months
    test_scenario = sc.ConstScenario(
        "Test Scenario",
        "Test scenario for IARI computation",
        test_scenarios_reach,
        QR_months,
    )

    # Set QS to compute IARI
    test_scenario.compute_QS()

    # Test the compute_IHA_index method with IARI
    IHA, IARI = test_scenario.compute_IHA_index(index_metric="IARI")

    # Test IARI structure
    assert isinstance(IARI, dict)
    assert "groups" in IARI
    assert "aggregated" in IARI
    assert len(IARI["groups"]) == 5
    # Test IHA structure
    assert isinstance(IHA, dict)
    assert all([f"Group{i+1}" in IHA for i in range(5)])
    assert all(isinstance(v, dict) for v in IHA.values())


def test_scenario_normalized_IHA_computation():
    # Create a scenario with known properties
    QR_months = [10] * 12  # Constant release of 10 units for all months
    test_scenario = sc.ConstScenario(
        "Test Scenario",
        "Test scenario for normalized IHA computation",
        test_scenarios_reach,
        QR_months,
    )

    # Set QS to compute normalized IHA
    test_scenario.compute_QS()

    # Test the compute_IHA_index method with normalized_IHA
    IHA, normalized_IHA = test_scenario.compute_IHA_index(index_metric="normalized_IHA")

    # Test normalized_IHA structure
    assert isinstance(normalized_IHA, dict)
    assert "groups" in normalized_IHA
    assert "aggregated" in normalized_IHA
    assert len(normalized_IHA["groups"]) == 5
    # Test IHA structure
    assert isinstance(IHA, dict)
    assert all([f"Group{i+1}" in IHA for i in range(5)])
    assert all(isinstance(v, dict) for v in IHA.values())


def test_scenario_IARI_with_custom_weights():
    # Create a scenario with custom weights
    QR_months = [5] * 12
    test_scenario = sc.ConstScenario(
        "Custom Weights Scenario",
        "Test scenario with custom weights",
        test_scenarios_reach,
        QR_months,
    )

    test_scenario.compute_QS()

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    IHA, IARI = test_scenario.compute_IHA_index(
        index_metric="IARI", index_options={"weights": weights}
    )

    assert "groups" in IARI
    assert "aggregated" in IARI
    assert len(IARI["groups"]) == 5


def test_scenario_normalized_IHA_with_custom_weights():
    # Create a scenario with custom weights
    QR_months = [5] * 12
    test_scenario = sc.ConstScenario(
        "Custom Weights Scenario",
        "Test scenario with custom weights",
        test_scenarios_reach,
        QR_months,
    )

    test_scenario.compute_QS()

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    IHA, normalized_IHA = test_scenario.compute_IHA_index(
        index_metric="normalized_IHA", index_options={"weights": weights}
    )

    assert "groups" in normalized_IHA
    assert "aggregated" in normalized_IHA
    assert len(normalized_IHA["groups"]) == 5


if __name__ == "__main__":
    test_scenario_basics()
    test_const_scenario()
    test_prop_scenario()
