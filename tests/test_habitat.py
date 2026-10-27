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
- Individual function testing for compute_habitat_series, compute_habitat_threshold, compute_ucut, compute_IH, and compute_habitat_indices

Expected vs Computed IH values:
- The test compares computed IH values with those in IH_Synopsis.txt
- Some differences are expected due to scenario parameter variations
- The test uses a tolerance of ±0.01 to account for these differences
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.reach as rch
import sarawater.scenarios as sc
from sarawater.habitat import (
    HabitatIndicesResult,
    UCUTCurve,
    compute_habitat_indices,
    compute_habitat_series,
    compute_habitat_threshold,
    compute_ucut,
    compute_IH,
    resample_HQ_curve,
)

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
Qreq_months = np.array(minrel_df[1].tolist()) / 1000.0  # Convert l/s to m3/s

# Read HQ curves
HQ_curves = pd.read_csv(
    os.path.join(data_dir, "HQ_curves.txt"), sep="\t", header="infer"
)

# Create reach object
Qabs_max = 0.2


def setup_reach_with_dmv_scenario():
    """Create a fresh reach object with DMV scenario for testing."""
    # Create initial reach object
    reach = rch.Reach("tutorial_reach", datetime_list, discharge_data, Qabs_max)

    # Add HQ curves to reach
    reach.add_HQ_curve(HQ_curves)

    # Create constant scenario (DMV)
    const_scenario = sc.ConstScenario(
        name="DMV",
        description="Minimum release scenario from CSV file",
        reach=reach,
        Qreq_months=list(Qreq_months),
    )

    # Add scenario to reach
    reach.add_scenario(const_scenario)

    # Compute Qrel for the scenario
    for scenario in reach.scenarios:
        scenario.compute_Qrel()

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

            # Check that IH values are in reasonable range [0, 1]
            ih_value = scenario.IH[HQ_name].IH
            if not np.isnan(ih_value):
                assert 0 <= ih_value <= 1


def test_habitat_index_values():
    """Test that computed IH values are in reasonable range and that the minimum (among the species) corresponds to the expected value from SimStream."""
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
    dmv_scenario.compute_Qrel()

    # Get all available species
    available_species = test_reach.get_list_available_HQ_curves()

    # Compute habitat for all available species using default behavior
    dmv_scenario.compute_IH_for_species(
        HQ_curve_resampling=True,
    )  # HQ resampling is needed for SimStream comparison

    # Collect computed IH values
    computed_IH_values: dict[str, float] = {}
    for species in available_species:
        ih_value = dmv_scenario.IH[species].IH
        computed_IH_values[species] = ih_value
        print(f"Computed IH for {species}: {ih_value:.3f}")

    # Get the minimum IH value among all species
    min_IH_species = min(
        computed_IH_values, key=lambda species: computed_IH_values[species]
    )
    min_computed_IH = computed_IH_values[min_IH_species]

    print(f"\nMinimum IH value: {min_computed_IH:.3f} (species: {min_IH_species})")

    # Test that all IH values are in reasonable range [0, 1]
    for species, ih_value in computed_IH_values.items():
        assert (
            0 <= ih_value <= 1
        ), f"IH value {ih_value} for {species} is outside valid range [0, 1]"

        # Test that IH is not NaN
        assert not np.isnan(ih_value), f"IH value for {species} is NaN"

    # Expected value from SimStream (relaxed tolerance for now)
    expected_value = expected_IH["ALTERED_1"]
    print(f"Expected IH from SimStream software: {expected_value:.3f}")

    assert (
        abs(min_computed_IH - expected_value) <= tolerance
    ), f"Minimum computed IH ({min_computed_IH:.3f}) differs from expected ({expected_value:.3f}) by more than {tolerance}"
    # For now, just ensure the computation produces reasonable results
    assert 0 <= min_computed_IH <= 1


def test_habitat_building_blocks():
    """Test the habitat series, threshold and UCUT functions directly."""
    test_reach = setup_reach_with_dmv_scenario()

    # Use BROW_A_R HQ curve for testing
    hq_data = test_reach.get_HQ_curve("BROW_A_R")
    HQ = hq_data[["DIS", "BROW_A_R"]].values

    # Use a subset of data for faster testing
    test_Q = discharge_data[:365]
    Q_threshold = np.percentile(test_Q, 3)

    H_series = compute_habitat_series(HQ, test_Q)
    H_threshold = compute_habitat_threshold(HQ, Q_threshold)
    ucut = compute_ucut(H_series, H_threshold)

    assert len(H_series) == len(test_Q)
    assert isinstance(H_threshold, float)
    assert isinstance(ucut, UCUTCurve)
    n_ucut = len(ucut.durations)
    assert len(ucut.cum_days) == n_ucut
    assert len(ucut.cum_freq) == n_ucut
    assert np.all(np.diff(ucut.durations) < 0)
    assert np.all((ucut.cum_freq >= 0) & (ucut.cum_freq <= 1))


def test_resample_HQ_curve():
    HQ = np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 0.0]])
    curve = resample_HQ_curve(HQ, 5)
    assert curve.shape == (5, 2)
    assert np.allclose(curve[:, 0], [0, 2.5, 5, 7.5, 10])
    assert np.allclose(curve[:, 1], [0, 5, 10, 5, 0])
    with pytest.raises(ValueError):
        resample_HQ_curve(HQ, 1)


def test_compute_habitat_threshold():
    HQ = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert compute_habitat_threshold(HQ, 4.2) == 5.0
    assert compute_habitat_threshold(HQ, 11.0) == 0.0


def test_compute_ucut_known_curve():
    """UCUT durations/cumulative days on a hand-computed habitat series."""
    # Under-threshold (H < 5) events of 3, 3, 1 and 2 days
    H_series = np.array([1, 1, 1, 9, 1, 1, 1, 9, 1, 9, 1, 1, 9], dtype=float)
    ucut = compute_ucut(H_series, 5.0)

    assert np.array_equal(ucut.durations, [3, 2, 1])
    # >=3 days: 6; >=2 days: 6 + 2; >=1 day: 6 + 2 + 1
    assert np.allclose(ucut.cum_days, [6, 8, 9])
    assert np.allclose(ucut.cum_freq, np.array([6, 8, 9]) / H_series.size)


def test_compute_ucut_no_events():
    ucut = compute_ucut(np.full(10, 9.0), 5.0)
    assert ucut.durations.size == 0
    assert ucut.cum_days.size == 0
    assert ucut.cum_freq.size == 0


def test_compute_ucut_nan_not_under_threshold():
    ucut = compute_ucut(np.array([1.0, np.nan, 1.0]), 5.0)
    assert np.array_equal(ucut.durations, [1])
    assert np.allclose(ucut.cum_days, [2])
    assert np.allclose(ucut.cum_freq, [2 / 3])


def test_compute_habitat_series_discharge_equal_to_curve_maximum():
    """A discharge equal to the last HQ point has a defined habitat value."""
    HQ = np.array([[0.0, 0.0], [10.0, 10.0]])
    H_series = compute_habitat_series(HQ, np.array([10.0, 12.0]))
    assert H_series[0] == 10.0
    assert np.isnan(H_series[1])


def test_compute_IH_function():
    """Test the compute_IH function directly."""
    # Create some test data
    ucut_ref = UCUTCurve(
        durations=np.array([4, 3, 2, 1]),
        cum_days=np.array([5, 10, 15, 20], dtype=float),
        cum_freq=np.array([5, 10, 15, 20]) / 100,
    )
    ucut_alt = UCUTCurve(
        durations=np.array([4, 3, 2, 1]),
        cum_days=np.array([8, 12, 18, 25], dtype=float),
        cum_freq=np.array([8, 12, 18, 25]) / 100,
    )
    H_ref = np.random.uniform(0.5, 1.0, 100)
    H_alt = np.random.uniform(0.3, 0.8, 100)

    ITH, ISH, IH, HSD = compute_IH(ucut_ref, ucut_alt, H_ref, H_alt)

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
    result = compute_habitat_indices(Qnat, Qalt, HQ)

    assert isinstance(result, HabitatIndicesResult)

    # Verify value ranges
    assert 0 <= result.ISH <= 1
    assert 0 <= result.ITH <= 1
    assert 0 <= result.IH <= 1
    assert result.HSD >= 0


def test_compute_IH_default_behavior():
    """Test the default behavior of compute_IH_for_species with species=None."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_Qrel()

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
        ih_value = dmv_scenario.IH[species].IH
        assert (
            0 <= ih_value <= 1
        ), f"IH value {ih_value} for {species} is outside valid range [0, 1]"


def test_compute_IH_single_species():
    """Test compute_IH_for_species with a single species string."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_Qrel()

    # Call compute_IH_for_species with a single species
    dmv_scenario.compute_IH_for_species(species="BROW_A_R")

    # Verify IH was computed only for the specified species
    assert len(dmv_scenario.IH) == 1
    assert "BROW_A_R" in dmv_scenario.IH
    ih_value = dmv_scenario.IH["BROW_A_R"].IH
    assert 0 <= ih_value <= 1


def test_compute_IH_species_list():
    """Test compute_IH_for_species with a list of species."""
    test_reach = setup_reach_with_dmv_scenario()

    # Get the DMV scenario
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_Qrel()

    # Call compute_IH_for_species with a list of species
    species_list = ["BROW_A_R", "TROU_J_R"]
    dmv_scenario.compute_IH_for_species(species=species_list)

    # Verify IH was computed only for the specified species
    assert len(dmv_scenario.IH) == 2
    for species in species_list:
        assert species in dmv_scenario.IH, f"Missing IH for species {species}"
        ih_value = dmv_scenario.IH[species].IH
        assert 0 <= ih_value <= 1


def test_compute_IH_for_species_hq_resampling_options_forwarded():
    """Test Scenario-level habitat options are forwarded to compute_habitat_indices."""
    test_reach = setup_reach_with_dmv_scenario()
    dmv_scenario = test_reach.scenarios[0]
    dmv_scenario.compute_Qrel()

    species = "BROW_A_R"
    hq_data = test_reach.get_HQ_curve(species)
    HQ = hq_data[["DIS", species]].values

    scenario_result = dmv_scenario.compute_IH_for_species(
        species=species,
        HQ_curve_resampling=True,
        n_resample=9,
    )[species]
    direct_result = compute_habitat_indices(
        dmv_scenario.Qnat,
        dmv_scenario.Qrel,
        HQ,
        HQ_curve_resampling=True,
        n_resample=9,
    )

    assert np.isclose(scenario_result.IH, direct_result.IH, equal_nan=True)
    assert np.isclose(scenario_result.ITH, direct_result.ITH, equal_nan=True)
    assert np.isclose(scenario_result.ISH, direct_result.ISH, equal_nan=True)
    assert np.allclose(scenario_result.H_alt, direct_result.H_alt, equal_nan=True)


if __name__ == "__main__":
    # Run tests when script is executed directly
    print("Running habitat module tests...")

    test_hq_curves_addition()
    test_habitat_computation()
    test_habitat_index_values()
    test_compute_ucut_known_curve()
    test_compute_IH_function()
    test_compute_habitat_indices_function()
    test_compute_IH_default_behavior()
    test_compute_IH_single_species()
    test_compute_IH_species_list()

    print("\n✓ All tests passed!")
