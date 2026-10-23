import sys, os
import numpy as np
import pytest
import datetime
import pandas as pd

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
        sc1.Qabs_max == test_scenarios_reach.Qabs_max
    )  # Ensure Qabs_max is correctly fetched from the reach
    assert np.array_equal(
        np.array(sc1.dates), np.array(test_scenarios_reach.dates)
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
    Qreq_months = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
    ]

    # Create a ConstScenario with the reach and Qreq_months
    const_scenario = sc.ConstScenario(
        "Const Scenario with Dates",
        "A constant scenario with monthly flow rates.",
        test_scenarios_reach,
        Qreq_months,
    )

    # Verify that the Qreq mapping is carried out properly
    assert np.array_equal(const_scenario.Qreq, Qreq_months), "Qreq mapping failed"
    for i, month in enumerate(range(1, 13)):
        month_mask = np.array([date.month == month for date in dates])
        assert np.all(
            const_scenario.Qreq[month_mask] == Qreq_months[i]
        ), f"Qreq mapping failed for month {month}"


def test_const_scenario():
    # Use the global Reach instance
    sc1 = sc.ConstScenario(
        "Scenario 1", "This is the first scenario.", test_scenarios_reach, [1.0] * 12
    )
    assert sc1.name == "Scenario 1"
    assert sc1.description == "This is the first scenario."
    assert np.array_equal(
        sc1.Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert sc1.Qreq_months[0] == 1
    assert len(sc1.Qreq_months) == 12


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
    assert sc1.Qbase == 1
    assert sc1.c_Qin == 0.3
    assert sc1.Qreq_min == 0.5
    assert sc1.Qreq_max == 50


def test_scenario_IARI_computation():
    # Create a scenario with known properties
    Qreq_months = [10.0] * 12  # Constant release of 10 units for all months
    test_scenario = sc.ConstScenario(
        "Test Scenario",
        "Test scenario for IARI computation",
        test_scenarios_reach,
        Qreq_months,
    )

    # Set Qrel to compute IARI
    test_scenario.compute_Qrel()

    # Test the compute_IHA_index method with IARI
    IHA, IARI = test_scenario.compute_IHA_index(index_metric="IARI")

    # Test IARI structure
    assert hasattr(IARI, "groups")
    assert hasattr(IARI, "aggregated")
    assert len(IARI.groups) == 5
    # Test IHA structure
    assert isinstance(IHA, dict)
    assert all([f"Group{i+1}" in IHA for i in range(5)])
    assert all(isinstance(v, dict) for v in IHA.values())


def test_scenario_normalized_IHA_computation():
    # Create a scenario with known properties
    Qreq_months = [10.0] * 12  # Constant release of 10 units for all months
    test_scenario = sc.ConstScenario(
        "Test Scenario",
        "Test scenario for normalized IHA computation",
        test_scenarios_reach,
        Qreq_months,
    )

    # Set Qrel to compute normalized IHA
    test_scenario.compute_Qrel()

    # Test the compute_IHA_index method with normalized_IHA
    IHA, normalized_IHA = test_scenario.compute_IHA_index(index_metric="normalized_IHA")

    # Test normalized_IHA structure
    assert hasattr(normalized_IHA, "groups")
    assert hasattr(normalized_IHA, "aggregated")
    assert len(normalized_IHA.groups) == 5
    # Test IHA structure
    assert isinstance(IHA, dict)
    assert all([f"Group{i+1}" in IHA for i in range(5)])
    assert all(isinstance(v, dict) for v in IHA.values())


def test_scenario_IARI_with_custom_weights():
    # Create a scenario with custom weights
    Qreq_months = [5.0] * 12
    test_scenario = sc.ConstScenario(
        "Custom Weights Scenario",
        "Test scenario with custom weights",
        test_scenarios_reach,
        Qreq_months,
    )

    test_scenario.compute_Qrel()

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    IHA, IARI = test_scenario.compute_IHA_index(
        index_metric="IARI", index_options={"weights": weights}
    )

    assert len(IARI.groups) == 5


def test_scenario_normalized_IHA_with_custom_weights():
    # Create a scenario with custom weights
    Qreq_months = [5.0] * 12
    test_scenario = sc.ConstScenario(
        "Custom Weights Scenario",
        "Test scenario with custom weights",
        test_scenarios_reach,
        Qreq_months,
    )

    test_scenario.compute_Qrel()

    # Test with custom weights
    weights = [0.1, 0.2, 0.3, 0.2, 0.2]
    IHA, normalized_IHA = test_scenario.compute_IHA_index(
        index_metric="normalized_IHA", index_options={"weights": weights}
    )

    assert len(normalized_IHA.groups) == 5


def _make_sediment_reach():
    sed_dates = [
        datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i)
        for i in range(2 * 365)
    ]
    Qnat_sed = np.random.default_rng(1).lognormal(2.0, 0.8, len(sed_dates))
    reach = rch.Reach("Sed Reach", sed_dates, Qnat_sed, 50.0)
    reach.add_cross_section_geometry(0.002, 20, width=10.0)
    reach.add_grain_size_distribution(4.0)
    scenario = sc.ConstScenario("Const", "Const flow", reach, [10] * 12)
    reach.add_scenario(scenario)
    scenario.compute_Qrel()
    return reach, scenario


def test_scenario_sediment_indices_guard_and_computation():
    reach, scenario = _make_sediment_reach()
    assert scenario.sediment_indices is None
    with pytest.raises(ValueError, match="compute_sediment_indices"):
        scenario._require_sediment_indices()

    res = scenario.compute_sediment_indices(threshold_percentile=5, n_bins=20)
    assert scenario.sediment_indices is res
    assert scenario._require_sediment_indices() is res
    assert reach.natural_sediment_load_df is not None
    for value in (res.ITS, res.IEF, res.ISV, res.ISR):
        assert 0.0 <= value <= 1.0


def _make_load_df(dates, Qs_total=1.0):
    return pd.DataFrame(
        {
            "Datetime": dates,
            "Q": np.arange(1.0, len(dates) + 1),
            "Qs_total": np.linspace(1.0, 2.0, len(dates)) * Qs_total,
            "Qs_phi_0.5": np.linspace(1.0, 2.0, len(dates)) * Qs_total,
        }
    )


def test_compute_sediment_load_defaults_to_mpm_and_invalidates_results(monkeypatch):
    _, scenario = _make_sediment_reach()
    calls = []
    result = _make_load_df(scenario.dates)

    def fake_compute_sediment_load(*args, **kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(sc, "compute_sediment_load", fake_compute_sediment_load)
    scenario.annual_sediment_budget = pd.DataFrame()
    scenario.compute_sediment_load()

    assert calls[0]["transport_formula"] == "mpm"
    assert calls[0]["mpm_theta_c"] == 0.047
    assert scenario.sediment_load_df is result
    assert scenario.annual_sediment_budget is None
    assert scenario.sediment_indices is None


def test_annual_budget_reuses_load_unless_formula_options_are_given(monkeypatch):
    _, scenario = _make_sediment_reach()
    scenario.sediment_load_df = _make_load_df(scenario.dates)

    def unexpected_recompute(*args, **kwargs):
        raise AssertionError("Existing sediment load should be reused")

    monkeypatch.setattr(scenario, "compute_sediment_load", unexpected_recompute)
    scenario.compute_annual_sediment_budget()

    calls = []

    def recompute(to_csv=None, **kwargs):
        calls.append(kwargs)
        scenario.sediment_load_df = _make_load_df(scenario.dates, Qs_total=2.0)

    monkeypatch.setattr(scenario, "compute_sediment_load", recompute)
    scenario.compute_annual_sediment_budget(transport_formula="wilcock_crowe")

    assert calls == [{"transport_formula": "wilcock_crowe", "mpm_theta_c": 0.047}]
    assert scenario.annual_sediment_budget is not None


def test_sediment_indices_recompute_both_series_with_requested_options(monkeypatch):
    reach, scenario = _make_sediment_reach()
    calls = []
    scenario_load = _make_load_df(reach.dates)
    natural_load = _make_load_df(reach.dates, Qs_total=2.0)

    def compute_scenario_load(transport_formula, mpm_theta_c):
        calls.append(("scenario", transport_formula, mpm_theta_c))
        scenario.sediment_load_df = scenario_load

    def compute_natural_load(transport_formula, mpm_theta_c):
        calls.append(("natural", transport_formula, mpm_theta_c))
        reach.natural_sediment_load_df = natural_load

    monkeypatch.setattr(scenario, "compute_sediment_load", compute_scenario_load)
    monkeypatch.setattr(reach, "compute_natural_sediment_budget", compute_natural_load)

    scenario.compute_sediment_indices(transport_formula="mpm", mpm_theta_c=0.05)

    assert calls == [
        ("scenario", "mpm", 0.05),
        ("natural", "mpm", 0.05),
    ]


def test_sediment_indices_reuse_existing_series_without_formula_options(monkeypatch):
    reach, scenario = _make_sediment_reach()
    scenario.sediment_load_df = _make_load_df(reach.dates)
    reach.natural_sediment_load_df = _make_load_df(reach.dates, Qs_total=2.0)

    def unexpected_recompute(*args, **kwargs):
        raise AssertionError("Existing sediment load series should be reused")

    monkeypatch.setattr(scenario, "compute_sediment_load", unexpected_recompute)
    monkeypatch.setattr(reach, "compute_natural_sediment_budget", unexpected_recompute)

    result = scenario.compute_sediment_indices()

    assert scenario.sediment_indices is result


if __name__ == "__main__":
    test_scenario_basics()
    test_const_scenario()
    test_prop_scenario()
