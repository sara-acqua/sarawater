"""
Test suite for the habitat module (SARAwater.habitat).

This test file follows the workflow described in tutorial_habitat.ipynb to:
1. Create a reach object with discharge data obtained from a csv file
2. Add HQ curves for fish species from HQ_curves_tutorial.txt
3. Create two scenarios:
   - Minimum release scenario (DMV) using the csv file
   - Ecological flow scenario (DE) using default parameters
4. Compute habitat indices (IH) for all scenarios and species
5. Verify that computed IH values are reasonable and compare with expected values

The test validates:
- Proper reach object creation and HQ curve integration
- Scenario creation and flow computation
- Habitat index computation using the habitat module functions
- Individual function testing for compute_h_ucut, compute_IH, and compute_habitat_indices

Expected vs Computed IH values:
- The test compares computed IH values with those in IH_Synopsis.txt
- Some differences are expected due to scenario parameter variations
- The test uses a tolerance of ±0.01 to account for these differences
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.reach as rch
import sarawater.scenarios as sc
from sarawater.habitat import compute_habitat_indices, compute_h_ucut, compute_IH

# Global test data setup
data_dir = os.path.join(os.path.dirname(__file__), "tests_data")

# Read discharge data
stream_df = pd.read_csv(
    os.path.join(data_dir, "daily_discharge_30y.csv"), parse_dates=["Date"]
)
datetime_list = np.array(stream_df["Date"].dt.to_pydatetime()).tolist()
discharge_data = np.array(stream_df["Q"].to_list())

# Read DMV data (minimum release values)
minrel_df = pd.read_csv(
    os.path.join(data_dir, "minimum_flow_requirements.csv"), header=None
)
QR_months = np.array(minrel_df[1].tolist()) / 1000.0  # Convert l/s to m3/s

# Read HQ curves
HQ_curves = pd.read_csv(
    os.path.join(data_dir, "HQ_curves.txt"), sep="\t", header="infer"
)

# Create reach object
Qab_max = 0.2


def setup_reach_with_dmv_scenario():
    """Create a fresh reach object with DMV scenario for testing."""
    # Create initial reach object
    reach = rch.Reach("tutorial_reach", datetime_list, discharge_data, Qab_max)

    # Add HQ curves to reach
    reach.add_HQ_curve(HQ_curves)

    # Create constant scenario (DMV)
    const_scenario = sc.ConstScenario(
        name="DMV",
        description="Minimum release scenario from CSV file",
        reach=reach,
        QR_months=QR_months,
    )

    # Add scenario to reach
    reach.add_scenario(const_scenario)

    # Compute QS for the scenario
    for scenario in reach.scenarios:
        scenario.compute_QS()

    return reach


# Expected IH values retrieved from IH_Synopsis.txt (source: SimStream software)
expected_IH = {"ALTERED_1": 0.07}  # From the synopsis file


def test_hq_curves_addition():
    """Test that HQ curves are properly added to the reach."""
    test_reach = setup_reach_with_dmv_scenario()
    available_curves = test_reach.get_list_available_HQ_curves()
    expected_species = [
        "BROW_A_R",
        "MARB_A_R",
        "TROU_J_R",
    ]

    for species in expected_species:
        assert (
            species in available_curves
        ), f"Species {species} not found in available curves: {available_curves}"

    # Check that HQ data structure is correct
    for species in expected_species:
        hq_data = test_reach.get_HQ_curve(species)
        assert "DIS" in hq_data.columns
        assert species in hq_data.columns
        assert len(hq_data) > 0


def test_habitat_computation():
    """
    Test habitat index computation when no species parameter is provided.
    This verifies that the default behavior computes IH for all available species at once.
    """
    test_reach = setup_reach_with_dmv_scenario()

    # Get available species for verification
    available_curves = test_reach.get_list_available_HQ_curves()

    for scenario in test_reach.scenarios:
        # Test default behavior: compute IH for all species at once
        scenario.compute_IH_for_species()

        # Verify IH was computed for all available species
        assert hasattr(scenario, "IH")
        assert len(scenario.IH) == len(available_curves), (
            f"Expected IH for {len(available_curves)} species, "
            f"but got {len(scenario.IH)}"
        )

        for HQ_name in available_curves:
            assert HQ_name in scenario.IH, f"Missing IH for species {HQ_name}"
            assert "IH" in scenario.IH[HQ_name]

            # Check that IH values are in reasonable range [0, 1]
            ih_value = scenario.IH[HQ_name]["IH"]
            if not np.isnan(ih_value):
                assert 0 <= ih_value <= 1


def test_habitat_index_values():
    """Test that computed IH values are in reasonable range and document the computed values."""
    tolerance = 0.01

    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = None
    for scenario in test_reach.scenarios:
        if scenario.name == "DMV":
            dmv_scenario = scenario
            break

    assert (
        dmv_scenario is not None
    ), f"DMV scenario not found. Available scenarios: {[s.name for s in test_reach.scenarios]}"

    # Ensure computations are done
    dmv_scenario.compute_QS()

    # Get all available species
    available_species = test_reach.get_list_available_HQ_curves()

    # Compute habitat for all available species using default behavior
    dmv_scenario.compute_IH_for_species()

    # Collect computed IH values
    computed_IH_values = {}
    for species in available_species:
        ih_value = dmv_scenario.IH[species]["IH"]
        computed_IH_values[species] = ih_value
        print(f"Computed IH for {species}: {ih_value:.3f}")

    # Get the minimum IH value among all species
    min_IH_species = min(computed_IH_values, key=computed_IH_values.get)
    min_computed_IH = computed_IH_values[min_IH_species]

    print(f"\nMinimum IH value: {min_computed_IH:.3f} (species: {min_IH_species})")

    # Test that all IH values are in reasonable range [0, 1]
    for species, ih_value in computed_IH_values.items():
        assert (
            0 <= ih_value <= 1
        ), f"IH value {ih_value} for {species} is outside valid range [0, 1]"

        # Test that IH is not NaN
        assert not np.isnan(ih_value), f"IH value for {species} is NaN"

    # Expected value from synopsis file (relaxed tolerance for now)
    expected_value = expected_IH["ALTERED_1"]
    print(f"Expected IH from SimStream software: {expected_value:.3f}")

    assert (
        abs(min_computed_IH - expected_value) <= tolerance
    ), f"Minimum computed IH ({min_computed_IH:.3f}) differs from expected ({expected_value:.3f}) by more than {tolerance}"
    # For now, just ensure the computation produces reasonable results
    assert 0 <= min_computed_IH <= 1


def test_compute_h_ucut_function():
    """Test the compute_h_ucut function directly."""
    test_reach = setup_reach_with_dmv_scenario()

    # Use BROW_A_R HQ curve for testing
    hq_data = test_reach.get_HQ_curve("BROW_A_R")
    HQ = hq_data[["DIS", "BROW_A_R"]].values

    # Use a subset of data for faster testing
    test_dates = datetime_list[:365]  # One year
    test_Q = discharge_data[:365]

    # Compute Q97 (3rd percentile)
    Q97 = np.percentile(test_Q, 3)

    # Test reference mode
    UCUT_cum_ref, UCUT_events_ref, H_ref, UCUT_cum_pes_ref, extra_ref = compute_h_ucut(
        HQ, test_dates, test_Q, Q97, mode="reference"
    )

    # Verify outputs
    assert isinstance(H_ref, np.ndarray)
    assert len(H_ref) == len(test_Q)
    assert isinstance(extra_ref, dict)
    assert "Q97" in extra_ref
    assert "H97" in extra_ref

    # Test altered mode
    UCUT_cum_alt, UCUT_events_alt, H_alt, UCUT_cum_pes_alt, extra_alt = compute_h_ucut(
        HQ, test_dates, test_Q * 0.5, Q97, H97_ref=extra_ref["H97"], mode="altered"
    )

    # Verify outputs
    assert isinstance(H_alt, np.ndarray)
    assert len(H_alt) == len(test_Q)


def test_compute_IH_function():
    """Test the compute_IH function directly."""
    # Create some test data
    UCUT_cum_ref = np.array([5, 10, 15, 20])
    UCUT_cum_alt = np.array([8, 12, 18, 25])
    H_ref = np.random.uniform(0.5, 1.0, 100)
    H_alt = np.random.uniform(0.3, 0.8, 100)
    UCUT_events_ref = np.array([4, 3, 2, 1])

    ITH, ISH, IH, HSD = compute_IH(
        UCUT_cum_ref, UCUT_cum_alt, H_ref, H_alt, UCUT_events_ref
    )

    # Verify outputs are in reasonable ranges
    assert 0 <= ISH <= 1
    assert 0 <= ITH <= 1
    assert 0 <= IH <= 1
    assert HSD >= 0


def test_compute_habitat_indices_function():
    """Test the main habitat calculation function."""
    test_reach = setup_reach_with_dmv_scenario()

    # Use BROW_A_R HQ curve
    hq_data = test_reach.get_HQ_curve("BROW_A_R")
    HQ = hq_data[["DIS", "BROW_A_R"]].values

    # Use subset of data
    test_dates = datetime_list[:365]
    Qnat = discharge_data[:365]
    Qalt = Qnat * 0.6  # Simulated altered flow

    # Compute habitat indices
    result = compute_habitat_indices(Qnat, Qalt, HQ, test_dates)

    # Verify all expected keys are present
    expected_keys = [
        "Q97_ref",
        "H97_ref",
        "UCUT_cum_ref",
        "UCUT_events_ref",
        "H_ref",
        "UCUT_cum_alt",
        "UCUT_events_alt",
        "H_alt",
        "ITH",
        "ISH",
        "IH",
        "HSD",
    ]

    for key in expected_keys:
        assert key in result

    # Verify value ranges
    assert 0 <= result["ISH"] <= 1
    assert 0 <= result["ITH"] <= 1
    assert 0 <= result["IH"] <= 1
    assert result["HSD"] >= 0


def test_compute_IH_default_behavior():
    """Test the default behavior of compute_IH_for_species with species=None."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_QS()

    # Get all available species
    available_species = test_reach.get_list_available_HQ_curves()

    # Call compute_IH_for_species with default species=None
    dmv_scenario.compute_IH_for_species()

    # Verify IH was computed for all available species
    assert len(dmv_scenario.IH) == len(available_species), (
        f"Expected IH for {len(available_species)} species, "
        f"but got {len(dmv_scenario.IH)}"
    )

    for species in available_species:
        assert species in dmv_scenario.IH, f"Missing IH for species {species}"
        assert "IH" in dmv_scenario.IH[species]
        ih_value = dmv_scenario.IH[species]["IH"]
        assert (
            0 <= ih_value <= 1
        ), f"IH value {ih_value} for {species} is outside valid range [0, 1]"


def test_compute_IH_single_species():
    """Test compute_IH_for_species with a single species string."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_QS()

    # Call compute_IH_for_species with a single species
    dmv_scenario.compute_IH_for_species(species="BROW_A_R")

    # Verify IH was computed only for the specified species
    assert len(dmv_scenario.IH) == 1
    assert "BROW_A_R" in dmv_scenario.IH
    assert "IH" in dmv_scenario.IH["BROW_A_R"]
    ih_value = dmv_scenario.IH["BROW_A_R"]["IH"]
    assert 0 <= ih_value <= 1


def test_compute_IH_species_list():
    """Test compute_IH_for_species with a list of species."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_QS()

    # Call compute_IH_for_species with a list of species
    species_list = ["BROW_A_R", "TROU_J_R"]
    dmv_scenario.compute_IH_for_species(species=species_list)

    # Verify IH was computed only for the specified species
    assert len(dmv_scenario.IH) == 2
    for species in species_list:
        assert species in dmv_scenario.IH, f"Missing IH for species {species}"
        assert "IH" in dmv_scenario.IH[species]
        ih_value = dmv_scenario.IH[species]["IH"]
        assert 0 <= ih_value <= 1


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Running habitat module tests...")

    test_hq_curves_addition()
    test_habitat_computation()
    test_habitat_index_values()
    test_compute_h_ucut_function()
    test_compute_IH_function()
    test_compute_habitat_indices_function()
    test_compute_IH_default_behavior()
    test_compute_IH_single_species()
    test_compute_IH_species_list()

    print("\n✓ All tests passed!")
