import sys, os
import numpy as np
import datetime

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.reach as rch
import sarawater.scenarios as sc

# Global variables: shared natural flow time series and dates
dates = [
    datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(365)
]  # All days in 2025
Qnat = np.random.randint(1, 100, len(dates))  # Natural flow rate time series


def test_reach_basics():
    # Create a reach with the natural flow rate time series and Qab_max as a positional argument
    r1 = rch.Reach("Reach 1", dates, Qnat, 50)

    # Verify the reach attributes
    assert r1.name == "Reach 1"
    assert np.array_equal(r1.dates, dates)  # Ensure dates are correctly set
    assert np.array_equal(r1.Qnat, Qnat)  # Ensure Qnat is correctly set
    assert r1.Qab_max == 50  # Ensure Qab_max is correctly set
    assert len(r1.scenarios) == 0  # Ensure no scenarios are added initially


def test_reach_with_prop_scenario():
    # Create a reach with the natural flow rate time series and Qab_max as a positional argument
    r1 = rch.Reach(
        "Reach 1", dates, Qnat, 50
    )  # Create a proportional scenario and add it to the reach
    sc1 = sc.PropScenario(
        "Scenario 1", "This is the first scenario.", r1, 1, 0.3, 0.5, 50
    )
    r1.add_scenario(sc1)

    # Verify the scenario is added to the reach
    assert len(r1.scenarios) == 1
    assert r1.scenarios[0].name == "Scenario 1"
    assert r1.scenarios[0].description == "This is the first scenario."
    assert np.array_equal(
        r1.scenarios[0].Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert r1.scenarios[0].QRbase == 1
    assert r1.scenarios[0].c_Qin == 0.3
    assert r1.scenarios[0].QRmax == 50
    assert (
        r1.scenarios[0].Qab_max == 50
    )  # Ensure Qab_max is correctly fetched from the reach


def test_reach_with_const_scenario():
    # Create a reach with the natural flow rate time series and Qab_max as a positional argument
    r1 = rch.Reach("Reach 1", dates, Qnat, 50)

    # Create a constant scenario and add it to the reach
    sc1 = sc.ConstScenario("Scenario 2", "This is a constant scenario.", r1, [1] * 12)
    r1.add_scenario(sc1)

    # Verify the scenario is added to the reach
    assert len(r1.scenarios) == 1
    assert r1.scenarios[0].name == "Scenario 2"
    assert r1.scenarios[0].description == "This is a constant scenario."
    assert np.array_equal(
        r1.scenarios[0].Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert r1.scenarios[0].QR_months[0] == 1
    assert len(r1.scenarios[0].QR_months) == 12
    assert (
        r1.scenarios[0].Qab_max == 50
    )  # Ensure Qab_max is correctly fetched from the reach


def test_reach_with_multiple_scenarios():
    # Create a reach with the natural flow rate time series and Qab_max as a positional argument
    r1 = rch.Reach("Reach 1", dates, Qnat, 50)

    # Create multiple scenarios and add them to the reach
    sc1 = sc.ConstScenario("Scenario 1", "This is a constant scenario.", r1, [1] * 12)
    sc2 = sc.PropScenario(
        "Scenario 2", "This is a proportional scenario.", r1, 1, 0.3, 0.5, 50
    )
    r1.add_scenario(sc1)
    r1.add_scenario(sc2)

    # Verify both scenarios are added to the reach
    assert len(r1.scenarios) == 2
    assert r1.scenarios[0].name == "Scenario 1"
    assert r1.scenarios[1].name == "Scenario 2"
    assert np.array_equal(
        r1.scenarios[0].Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach
    assert np.array_equal(
        r1.scenarios[1].Qnat, Qnat
    )  # Ensure Qnat is correctly fetched from the reach


def test_reach_with_ecological_scenario():
    # Create a reach with the natural flow rate time series and Qab_max as a positional argument
    r1 = rch.Reach("Reach 1", dates, Qnat, 50)

    # Create an ecological scenario using the reach method
    r1.add_ecological_flow_scenario(
        "Ecological", "This is an ecological scenario", k=0.2, p=2.0
    )

    # Verify the ecological scenario is properly added and configured
    assert len(r1.scenarios) == 1
    assert r1.scenarios[0].name == "Ecological"
    assert r1.scenarios[0].description == "This is an ecological scenario"
    assert np.array_equal(
        r1.scenarios[0].Qnat, Qnat
    )  # Ensure Qnat is correctly fetched
    assert len(r1.scenarios[0].QR_months) == 12  # Should have 12 monthly values
    assert r1.scenarios[0].Qab_max == 50  # Ensure Qab_max is correctly set
    assert np.all(
        r1.scenarios[0].QR_months >= np.percentile(Qnat, 3)
    )  # Ensure DE is greater than Q97


if __name__ == "__main__":
    test_reach_basics()
    test_reach_with_prop_scenario()
    test_reach_with_const_scenario()
    test_reach_with_multiple_scenarios()
    test_reach_with_ecological_scenario()
