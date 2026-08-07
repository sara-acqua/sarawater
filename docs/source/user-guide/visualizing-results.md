## Visualizing the results

The {py:class}`sarawater.visualization.ReachPlotter` class provides comprehensive visualization capabilities for comparing scenarios.

**Creating a Plotter object:**

```python
   from sarawater.visualization import ReachPlotter
   
   plotter = ReachPlotter(reach, output_dir='outputs')
```

The ``output_dir`` argument must be a valid directory path string (``None`` is not supported).

**Available Plots:**

**Discharge and Flow Regime:**

- {py:meth}`sarawater.visualization.ReachPlotter.plot_scenarios_discharge`: Compare discharge time series across scenarios
- {py:meth}`sarawater.visualization.ReachPlotter.plot_cases_duration`: Visualize flow regime case durations (Case 1: Qnat <= Qreq; Case 2: abstraction occurring; Case 3: excess flow)
- {py:meth}`sarawater.visualization.ReachPlotter.plot_cases_duration_month`: Monthly case duration comparison
- {py:meth}`sarawater.visualization.ReachPlotter.plot_monthly_abstraction`: Compare monthly water abstraction volumes

**Hydrologic Alteration:**

- {py:meth}`sarawater.visualization.ReachPlotter.plot_iha_parameters`: Multi-panel plot of all IHA parameters across scenarios
- {py:meth}`sarawater.visualization.ReachPlotter.plot_iari_groups`: IARI values by IHA group
- {py:meth}`sarawater.visualization.ReachPlotter.plot_iari_summary`: Overall IARI comparison across scenarios
- {py:meth}`sarawater.visualization.ReachPlotter.plot_nIHA_summary`: Normalized IHA comparison
- {py:meth}`sarawater.visualization.ReachPlotter.plot_iha_boxplots`: Box plots of IHA parameters showing inter-annual variability
- {py:meth}`sarawater.visualization.ReachPlotter.plot_relative_deviations`: Relative deviations of IHA parameters from natural conditions
- {py:meth}`sarawater.visualization.ReachPlotter.plot_iari_vs_volume`: Trade-off between hydrologic alteration (IARI) and water abstraction

**Habitat Analysis:**

- {py:meth}`sarawater.visualization.ReachPlotter.plot_hq_curves`: Display habitat-discharge curves
- {py:meth}`sarawater.visualization.ReachPlotter.plot_habitat_timeseries`: Compare habitat availability time series
- {py:meth}`sarawater.visualization.ReachPlotter.plot_ucut_curves`: UCUT curves showing duration of habitat stress events
- {py:meth}`sarawater.visualization.ReachPlotter.plot_ih_vs_volume`: Trade-off between habitat alteration (IH) and water abstraction
- {py:meth}`sarawater.visualization.ReachPlotter.plot_nIHA_vs_volume`: Trade-off between normalized IHA and water abstraction

**Plot Options:**

All plotting methods support saving plots to the configured output directory by setting ``save=True``. Additional keyword arguments can be passed to customize the plots (e.g., figure size, colors, labels).