import sys, os
import numpy as np
import datetime
import pandas as pd

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.reach as rch
import sarawater.scenarios as sc

# Global variables: shared natural flow time series and dates
dates = [
    datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(365)
]  # All days in 2025
Qnat = np.random.randint(1, 100, len(dates))  # Natural flow rate time series


def test_reach_basics():
    # Create a reach with the natural flow rate time series and Qabs_max as a positional argument
    r1 = rch.Reach("Reach 1", dates, Qnat, 50)

    # Verify the reach attributes
    assert r1.name == "Reach 1"
    assert np.array_equal(r1.dates, dates)  # Ensure dates are correctly set
    assert np.array_equal(r1.Qnat, Qnat)  # Ensure Qnat is correctly set
    assert r1.Qabs_max == 50  # Ensure Qabs_max is correctly set
    assert len(r1.scenarios) == 0  # Ensure no scenarios are added initially


def test_reach_with_prop_scenario():
    # Create a reach with the natural flow rate time series and Qabs_max as a positional argument
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
    assert r1.scenarios[0].Qbase == 1
    assert r1.scenarios[0].c_Qin == 0.3
    assert r1.scenarios[0].Qreq_max == 50
    assert (
        r1.scenarios[0].Qabs_max == 50
    )  # Ensure Qabs_max is correctly fetched from the reach


def test_reach_with_const_scenario():
    # Create a reach with the natural flow rate time series and Qabs_max as a positional argument
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
    assert r1.scenarios[0].Qreq_months[0] == 1
    assert len(r1.scenarios[0].Qreq_months) == 12
    assert (
        r1.scenarios[0].Qabs_max == 50
    )  # Ensure Qabs_max is correctly fetched from the reach


def test_reach_with_multiple_scenarios():
    # Create a reach with the natural flow rate time series and Qabs_max as a positional argument
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
    # Create a reach with the natural flow rate time series and Qabs_max as a positional argument
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
    assert len(r1.scenarios[0].Qreq_months) == 12  # Should have 12 monthly values
    assert r1.scenarios[0].Qabs_max == 50  # Ensure Qabs_max is correctly set
    assert np.all(
        r1.scenarios[0].Qreq_months >= np.percentile(Qnat, 3)
    )  # Ensure DE is greater than Q97



def test_reach_with_irregular_cross_section():
    """Test reach with irregular cross-section and grain size distribution"""
    # Create a reach
    reach = rch.Reach("Test Reach", dates, Qnat, 50)
    
    # Create an irregular cross-section (trapezoidal-like)
    section_data = pd.DataFrame({
        'x [m]': [0.0, 2.0, 5.0, 8.0, 10.0],
        'y [m]': [0.0, 1.5, 2.0, 1.5, 0.0]
    })
    
    # Create grain size distribution data
    # Simple distribution with a few points
    grain_data = pd.DataFrame({
        'di[mm]': [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        'i(di)': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    })
    
    slope = 0.002
    
    # Add cross-section info
    reach.add_cross_section_info(section_data, slope, grain_data)
    
    # Verify reach properties are set
    assert reach.width > 0, "Width should be positive"
    assert reach.area > 0, "Area should be positive"
    assert reach.slope == slope, "Slope should match input"
    assert reach.height_avg > 0, "Average height should be positive"
    
    # Verify phi percentages
    assert hasattr(reach, 'phi_percentages'), "Phi percentages should be set"
    assert reach.phi_percentages is not None, "Phi percentages should not be None"
    assert len(reach.phi_percentages) == 18, f"Expected 18 phi classes, got {len(reach.phi_percentages)}"
    
    # Verify phi percentages sum to approximately 1.0
    total_fraction = reach.phi_percentages.sum()
    assert abs(total_fraction - 1.0) < 0.01, f"Phi percentages should sum to 1.0, got {total_fraction}"
    
    # Verify subdivision data
    assert hasattr(reach, 'rectangular_section'), "Rectangular section should be set"
    assert reach.rectangular_section is not None, "Rectangular section should not be None"
    assert len(reach.rectangular_section) == 4, "Should have 4 subdivisions for 5 points"
    
    # Verify each subdivision has required fields
    for i, subdiv in reach.rectangular_section.iterrows():
        assert 'width' in subdiv, "Subdivision should have width"
        assert 'height' in subdiv, "Subdivision should have height"
        assert 'area' in subdiv, "Subdivision should have area"
        assert subdiv['width'] > 0, f"Subdivision {i} width should be positive"
        assert subdiv['area'] >= 0, f"Subdivision {i} area should be non-negative"


def test_reach_with_simple_d50():
    """Test reach with simple D50 grain size specification"""
    # Create a reach
    reach = rch.Reach("Test Reach", dates, Qnat, 50)
    
    # Simple rectangular cross-section
    section_data = pd.DataFrame({
        'x [m]': [0.0, 10.0],
        'y [m]': [2.0, 2.0]
    })
    
    # Simple D50 value in mm
    D50 = 10.0  # mm
    slope = 0.001
    
    # Add cross-section info
    reach.add_cross_section_info(section_data, slope, D50)
    
    # Verify reach properties
    assert reach.width == 10.0, "Width should be 10m"
    assert reach.slope == slope, "Slope should match input"
    
    # Verify phi percentages are created correctly
    assert hasattr(reach, 'phi_percentages'), "Phi percentages should be set"
    assert reach.phi_percentages is not None, "Phi percentages should not be None"
    assert len(reach.phi_percentages) == 18, f"Expected 18 phi classes, got {len(reach.phi_percentages)}"
    
    # For simple D50, most of the distribution should be concentrated
    # around the corresponding phi class
    total_fraction = reach.phi_percentages.sum()
    assert abs(total_fraction - 1.0) < 0.01, f"Phi percentages should sum to 1.0, got {total_fraction}"


def test_reach_with_grain_array():
    """Test reach with grain size as 2D array"""
    import numpy as np
    
    # Create a reach
    reach = rch.Reach("Test Reach", dates, Qnat, 50)
    
    # Simple cross-section
    section_data = pd.DataFrame({
        'x [m]': [0.0, 5.0, 10.0],
        'y [m]': [0.0, 1.5, 0.0]
    })
    
    # Grain data as 2D numpy array
    grain_array = np.array([
        [0.0, 0.2, 0.5, 0.8, 1.0],  # i(di) - cumulative fractions
        [1.0, 5.0, 10.0, 20.0, 50.0]  # di[mm] - grain sizes
    ])
    
    slope = 0.002
    
    # Add cross-section info
    reach.add_cross_section_info(section_data, slope, grain_array)
    
    # Verify phi percentages
    assert hasattr(reach, 'phi_percentages'), "Phi percentages should be set"
    assert len(reach.phi_percentages) == 18, f"Expected 18 phi classes, got {len(reach.phi_percentages)}"
    
    total_fraction = reach.phi_percentages.sum()
    assert abs(total_fraction - 1.0) < 0.01, f"Phi percentages should sum to 1.0, got {total_fraction}"


if __name__ == "__main__":
    test_reach_basics()
    test_reach_with_prop_scenario()
    test_reach_with_const_scenario()
    test_reach_with_multiple_scenarios()
    test_reach_with_ecological_scenario()
    test_reach_with_irregular_cross_section()
    test_reach_with_simple_d50()
    test_reach_with_grain_array()
