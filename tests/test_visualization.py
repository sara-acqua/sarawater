import sys, os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pytest

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import sarawater.scenarios as sc
import sarawater.reach as rch
from sarawater.visualization import ReachPlotter

# Create test data
dates = [
    datetime.datetime(2025, 1, 1) + datetime.timedelta(days=x) for x in range(365 * 3)
]
Qnat = np.random.lognormal(2.0, 1.0, len(dates))
test_visualization_reach = rch.Reach("Test Reach", dates, Qnat, 50.0)

# Add some test scenarios
sc1 = sc.ConstScenario(
    "Constant Flow", "A constant flow scenario", test_visualization_reach, [10] * 12
)
sc2 = sc.PropScenario(
    "Prop Flow 1",
    "A proportional flow scenario",
    test_visualization_reach,
    5.0,
    0.5,
    6.0,
    20.0,
)
test_visualization_reach.add_scenario(sc1)
test_visualization_reach.add_scenario(sc2)
for scenario in test_visualization_reach.scenarios:
    scenario.compute_Qrel()
    scenario.compute_IHA()


def test_plotter_initialization():
    """Test basic initialization of ReachPlotter"""
    plotter = ReachPlotter(test_visualization_reach)
    assert plotter.reach == test_visualization_reach
    assert plotter.output_dir == "outputs"

    # Test with output directory
    test_output_dir = os.path.join("tests", "test_output")
    plotter = ReachPlotter(test_visualization_reach, test_output_dir)
    assert plotter.output_dir == test_output_dir


def test_plotter_initialization_rejects_none_output_dir():
    """Test ReachPlotter rejects None as output_dir."""
    with pytest.raises(
        ValueError, match="output_dir must be a valid directory path string"
    ):
        ReachPlotter(test_visualization_reach, None)


def test_scenario_discharge_plot():
    """Test scenario discharge plotting"""
    plotter = ReachPlotter(test_visualization_reach)

    # Test plotting without saving
    plotter.plot_scenarios_discharge()

    # Test plotting with date range
    start_date = datetime.datetime(2025, 6, 1)
    end_date = datetime.datetime(2025, 12, 31)
    plotter.plot_scenarios_discharge(start_date=start_date, end_date=end_date)


def test_iha_plots():
    """Test IHA parameter plotting"""
    plotter = ReachPlotter(test_visualization_reach)
    plotter.plot_iha_parameters()


def test_iari_plots():
    """Test IARI group value plotting"""
    plotter = ReachPlotter(test_visualization_reach)

    # Compute IARI values for scenarios
    for scenario in test_visualization_reach.scenarios:
        scenario.compute_IHA_index(index_metric="IARI")

    plotter.plot_iari_groups()


def test_normalized_IHA_plot():
    """Test normalized IHA summary plotting"""
    plotter = ReachPlotter(test_visualization_reach)

    # Compute normalized IHA values for scenarios
    for scenario in test_visualization_reach.scenarios:
        scenario.compute_IHA_index(index_metric="normalized_IHA")

    plotter.plot_nIHA_summary()


def test_monthly_abstraction_plot():
    """Test monthly abstraction volume plotting"""
    plotter = ReachPlotter(test_visualization_reach)

    # Ensure scenarios have computed their abstracted volumes
    for scenario in test_visualization_reach.scenarios:
        scenario.compute_natural_abstracted_volumes()

    plotter.plot_monthly_abstraction()


def test_iari_vs_volume_plot():
    """Test IARI vs volume plotting"""
    plotter = ReachPlotter(test_visualization_reach)

    # Compute IARI values and volumes for scenarios
    for scenario in test_visualization_reach.scenarios:
        scenario.compute_IHA_index(index_metric="IARI")
        scenario.compute_natural_abstracted_volumes()

    # Test plotting without saving
    plotter.plot_iari_vs_volume()


def _make_local_reach():
    """Build a fresh reach with one scenario and no sediment results computed."""
    local_reach = rch.Reach("Local Reach", dates, Qnat, 50.0)
    local_scenario = sc.ConstScenario(
        "Constant Flow", "A constant flow scenario", local_reach, [10] * 12
    )
    local_reach.add_scenario(local_scenario)
    local_scenario.compute_Qrel()
    return local_reach, local_scenario


def test_sediment_load_total_requires_data_for_every_scenario():
    """Test total sediment plotting reports scenarios without computed data."""
    local_reach, _ = _make_local_reach()
    plotter = ReachPlotter(local_reach)
    with pytest.raises(ValueError, match="Constant Flow.*compute_sediment_load"):
        plotter.plot_sediment_load_total()
    with pytest.raises(
        ValueError, match="Constant Flow.*compute_annual_sediment_budget"
    ):
        plotter.plot_annual_sediment_budget_by_class()


def test_scenario_sediment_transport_plot_auto_computes_data():
    """Test scenario sediment plotting computes missing data and respects dates."""
    plt.close("all")
    local_reach, scenario = _make_local_reach()
    local_reach.add_cross_section_geometry(0.002, 20, width=10.0)
    local_reach.add_grain_size_distribution(4.0)
    assert scenario.sediment_load_df is None

    ax = scenario.plot_scenario_sediment_transport(
        start_date=scenario.dates[1], end_date=scenario.dates[3]
    )

    assert scenario.sediment_load_df is not None
    assert len(ax.lines) == 1
    assert len(ax.lines[0].get_xdata()) == 3
    assert ax.lines[0].get_label() == scenario.name
    assert ax.get_ylabel() == "Sediment transport capacity [m³/s]"
    plt.close("all")


def test_sediment_load_plots_use_computed_scenario_data():
    """Test sediment plots read the scenario sediment-load DataFrame."""
    test_visualization_reach.add_cross_section_geometry(0.002, 20, width=10.0)
    test_visualization_reach.add_grain_size_distribution(4.0)
    for scenario in test_visualization_reach.scenarios:
        scenario.compute_sediment_load()
        scenario.compute_annual_sediment_budget()

    plotter = ReachPlotter(test_visualization_reach)
    total_ax = plotter.plot_sediment_load_total(log_scale=False)
    class_ax = plotter.plot_annual_sediment_budget_by_class()
    budget_ax = plotter.plot_sediment_budget_vs_volume()

    assert len(total_ax.lines) == len(test_visualization_reach.scenarios)
    assert total_ax.get_ylabel() == "Total Sediment Load (m³/s)"
    scenario_axes = class_ax.figure.axes[:-1]
    colorbar_ax = class_ax.figure.axes[-1]
    assert len(scenario_axes) == len(test_visualization_reach.scenarios)
    assert all(ax.collections for ax in scenario_axes)
    assert all(
        len(ax.get_yticks()) == len(ax.get_yticklabels()) for ax in scenario_axes
    )
    assert colorbar_ax.get_ylabel() == "Annual sediment budget (m³/year)"
    assert all(ax.get_shared_x_axes().joined(ax, class_ax) for ax in scenario_axes)
    assert test_visualization_reach.natural_annual_sediment_budget is not None
    assert len(budget_ax.containers) == len(test_visualization_reach.scenarios)
    assert (
        budget_ax.get_xlabel() == "Normalized sediment budget (scenario / natural) [-]"
    )
    assert (
        budget_ax.get_ylabel() == r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]"
    )
    scenario = test_visualization_reach.scenarios[0]
    scenario_budget = scenario._require_annual_sediment_budget()
    natural_budget = test_visualization_reach._require_natural_annual_sediment_budget()
    common_years = scenario_budget.index.intersection(natural_budget.index)
    normalized_budget = scenario_budget.loc[common_years, "Qs_total"].to_numpy(
        dtype=float
    ) / natural_budget.loc[common_years, "Qs_total"].to_numpy(dtype=float)
    errorbar = budget_ax.containers[0]
    data_line, _, errorbar_lines = errorbar.lines
    np.testing.assert_allclose(
        np.asarray(data_line.get_xdata(), dtype=float), [np.mean(normalized_budget)]
    )
    volume_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
    np.testing.assert_allclose(
        np.asarray(data_line.get_ydata(), dtype=float), [np.median(volume_values)]
    )
    x_error_segment = errorbar_lines[0].get_segments()[0]
    np.testing.assert_allclose(
        (x_error_segment[1, 0] - x_error_segment[0, 0]) / 2,
        np.std(normalized_budget),
    )


if __name__ == "__main__":
    test_plotter_initialization()
    test_scenario_discharge_plot()
    test_iari_plots()
    test_iha_plots()
    test_monthly_abstraction_plot()
    test_iari_vs_volume_plot()
    plt.show()
