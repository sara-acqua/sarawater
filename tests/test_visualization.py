import sys, os
import numpy as np
import datetime
import matplotlib.pyplot as plt

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
    scenario.compute_QS()
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


if __name__ == "__main__":
    test_plotter_initialization()
    test_scenario_discharge_plot()
    test_iari_plots()
    test_iha_plots()
    test_monthly_abstraction_plot()
    test_iari_vs_volume_plot()
    plt.show()
