# User Guide
This section provides a comprehensive overview of the main classes and functions available in the `SARAwater` package.

## Reach objects
The {py:class}`sarawater.reach.Reach` class is the central object in SARAwater, representing a river reach with its natural flow time series. It serves as a container for scenarios and associated data.

### Key Attributes:
- ``name``: Name of the reach
- ``dates``: List of datetime objects for the time series
- ``Qnat``: Natural flow rate time series (NumPy array)
- ``Qabs_max``: Maximum water abstraction threshold (m3/s)
- ``scenarios``: List of scenarios added to the reach
- ``IHA_nat``: Natural-flow IHA indicators (computed automatically at construction)

### Main Methods:
- {py:meth}`sarawater.reach.Reach.add_scenario`: Add a scenario to the reach
- {py:meth}`sarawater.reach.Reach.add_ecological_flow_scenario`: Create and add an ecological flow scenario with monthly adjustments
- {py:meth}`sarawater.reach.Reach.add_HQ_curve`: Add habitat-discharge curves for different species/life stages
- {py:meth}`sarawater.reach.Reach.get_list_available_HQ_curves`: Get list of available habitat curves
- {py:meth}`sarawater.reach.Reach.get_HQ_curve`: Retrieve a specific habitat-discharge curve
- {py:meth}`sarawater.reach.Reach.print_scenarios`: Print a list of all scenarios added to the reach
- {py:meth}`sarawater.reach.Reach.export_scenarios_summary`: Export a comprehensive summary table of all scenarios with their parameters and indices

## Scenario objects

Scenarios represent different water management alternatives. SARAwater provides a base {py:class}`sarawater.scenarios.Scenario` class and specialized subclasses:

### {py:class}`sarawater.scenarios.Scenario` (base class)

The parent class for all scenario types, containing shared functionality.

### Key Attributes:

- ``name``: Name of the scenario
- ``description``: Description of the scenario
- ``reach``: Associated {py:class}`sarawater.reach.Reach` object
- ``Qabs_max``: Maximum water abstraction, if different from the reach value (m3/s)
- ``Qreq``: Minimum release flow time series (m3/s)
- ``Qrel``: Released flow rate time series (m3/s)
- ``IHA``: {py:data}`sarawater.IHA.IHAResult` mapping of IHA indicators (``Group1`` to ``Group5``)
- ``IH``: Dictionary mapping species names to {py:class}`sarawater.habitat.HabitatIndicesResult`
- ``IARI``: {py:class}`sarawater.IHA.IHAIndexResult` (set after IARI computation)
- ``normalized_IHA``: {py:class}`sarawater.IHA.IHAIndexResult` (set after normalized IHA computation)

### Main Methods:

- {py:meth}`sarawater.scenarios.Scenario.compute_Qrel`: Calculate the released flow time series based on Qnat, Qreq, and Qabs_max
- {py:meth}`sarawater.scenarios.Scenario.plot_scenario_discharge`: Plot the released discharge time series
- {py:meth}`sarawater.scenarios.Scenario.compute_IHA`: Compute Indicators of Hydrologic Alteration
- {py:meth}`sarawater.scenarios.Scenario.compute_IHA_index`: Compute IHA indices (IARI or normalized IHA)
- {py:meth}`sarawater.scenarios.Scenario.compute_natural_abstracted_volumes`: Calculate water volumes abstracted from the reach
- {py:meth}`sarawater.scenarios.Scenario.compute_IH_for_species`: Compute habitat indices for one or more species
- {py:meth}`sarawater.scenarios.Scenario.cases_duration_for_month`: Compute monthly duration of each flow case (Case 1/2/3)
- {py:meth}`sarawater.scenarios.Scenario.compute_sediment_load`: Compute sediment transport time series for the scenario
- {py:meth}`sarawater.scenarios.Scenario.plot_scenario_sediment_transport`: Plot sediment transport time series
- {py:meth}`sarawater.scenarios.Scenario.compute_annual_sediment_budget`: Compute annual sediment budgets

### {py:class}`sarawater.scenarios.ConstScenario` (constant release)

A scenario with constant monthly flow requirements.

Parameters:
- ``Qreq_months``: List of 12 float values representing monthly constant flow rates (m3/s)

### {py:class}`sarawater.scenarios.PropScenario` (proportional release)

Child class for scenarios with flow requirements proportional to the incoming flow. In scenarios with a proportional flow requirement, $Q_{req}$ is computed as a fraction of the incoming flow discharge, $Q_{in}$, according to the formula:

$$
Q_{req} = Q_{base} + c_{in} \cdot Q_{in}
$$ (eq:Qreq_linear)

where $Q_{base}$ is a base flow requirement (e.g., to maintain minimum ecological conditions), and $c_{in}$ is a coefficient that defines the proportion of the incoming flow to be included in the flow requirement.

This formula is then adjusted to ensure that $Q_{req}$ remains within user-specified minimum and maximum bounds, $Q_{req,min}$ and $Q_{req,max}$, respectively. This step is needed because low flow requirements may cause severe alteration in the downstream reach, while high flow requirements may lead to a very low water abstraction, which might not be sufficient. The complete definition of $Q_{req}$ in proportional scenarios is given by the piecewise function:


$$
Q_{req} = \begin{cases}
   Q_{req,min} & \text{if } Q_{base} + c_{in} \cdot Q_{in} \leq Q_{req,min}\\
   Q_{base} + c_{in} \cdot Q_{in} & \text{if } Q_{req,min} < Q_{base} + c_{in} \cdot Q_{in} < Q_{req,max}\\
   Q_{req,max} & \text{if } Q_{base} + c_{in} \cdot Q_{in} > Q_{req,max}
   \end{cases}
$$ (eq:Qreq_piecewise)

Parameters:
- ``Qbase``: Base flow rate (m3/s)
- ``c_Qin``: Proportionality coefficient (dimensionless)
- ``Qreq_min``: Minimum release constraint (m3/s)
- ``Qreq_max``: Maximum release constraint (m3/s)

## Compute the released flow discharge for each scenario

Given the incoming flow discharge $Q_{nat}$, the flow requirement $Q_{req}$ and the maximum abstractable flow $Q_{abs,max}$, the released flow discharge $Q_{rel}$ is computed as:

$$
Q_{rel} = \begin{cases}
   Q_{nat} & \text{if } Q_{nat} \leq Q_{req}\\
   Q_{req} & \text{if } Q_{req} < Q_{nat} < Q_{req} + Q_{abs,max}\\
   Q_{nat} - Q_{abs,max} & \text{if } Q_{nat} \geq Q_{req} + Q_{abs,max}
   \end{cases}
$$ (eq:Qrel_piecewise)

Where the three cases correspond to the following:

1) The incoming flow $Q_{nat}$ is lower than the flow requirement $Q_{req}$; therefore, no water is abstracted and the released flow discharge $Q_{rel}$ equals the incoming flow. This usually happens in low-flow periods.
2) There is enough incoming flow to satisfy the flow requirement, and the abstracted flow discharge $Q_{abs}$ is lower than the maximum abstractable flow $Q_{abs,max}$. Recall that, according to Equation (1), $Q_{abs} = Q_{nat} - Q_{rel}$ (where, in this case, $Q_{rel} = Q_{req}$). This is the most "common" case, where the flow requirement rule is applied straightforwardly.
3) The incoming flow $Q_{nat}$ is so large that the maximum abstractable flow can be diverted while still releasing a flow rate larger than the flow requirement. This usually happens during flood events.

The piecewise function {eq}`eq:Qrel_piecewise` is implemented in {py:meth}`sarawater.scenarios.Scenario.compute_Qrel`; therefore, to compute the released flow time series for each scenario we can simply write

```python
   scenario.compute_Qrel()
```

## Assessing alterations

SARAwater provides several methods to quantify hydrological, habitat, and sediment transport alterations.

### Hydrologic alteration

**Indicators of Hydrologic Alteration (IHA)**

The IHA framework quantifies changes in flow regime by analyzing 33 parameters grouped into 5 categories:

- **Group 1**: Monthly flow statistics (mean flows for each month)
- **Group 2**: Magnitude and duration of extreme conditions (min/max flows over 1, 3, 7, 30, 90-day windows; base flow; zero-flow days)
- **Group 3**: Timing of extreme conditions (Julian dates of annual min/max flows)
- **Group 4**: Frequency and duration of high/low pulses
- **Group 5**: Rate and frequency of flow changes (rise/fall rates, number of reversals)

**Computing IHA:**

Use {py:meth}`sarawater.scenarios.Scenario.compute_IHA` to calculate IHA indicators for a scenario.

The method returns an {py:data}`sarawater.IHA.IHAResult` mapping with five groups (``Group1`` to ``Group5``), where each group stores yearly arrays for its IHA parameters.

**IHA Indices:**

Two aggregate indices are available:

- **IARI (Index of Alteration of Hydrologic Regime)**: Measures overall deviation from the natural hydrologic regime. When equal to 0 indicates an unaltered condition, while above 0.15 indicates severe alteration.
   - Compute with: {py:meth}`sarawater.scenarios.Scenario.compute_IHA_index` using ``index_metric='IARI'``
  
- **Normalized IHA**: Normalized deviations of IHA parameters.
  
   - Compute with: {py:meth}`sarawater.scenarios.Scenario.compute_IHA_index` using ``index_metric='normalized_IHA'``

{py:meth}`sarawater.scenarios.Scenario.compute_IHA_index` returns a tuple:

1. {py:data}`sarawater.IHA.IHAResult` for altered-flow IHA indicators
2. {py:class}`sarawater.IHA.IHAIndexResult` with:
    - ``groups``: per-group yearly index values
    - ``aggregated``: weighted aggregated yearly values

### Habitat alteration

Habitat alteration is quantified using habitat-discharge (HQ) curves and the UCUT (Under-threshold Cumulative Curve) methodology.

**Habitat Indices:**

The following indices quantify habitat alteration for aquatic species:

- **Q97**: 3rd-percentile reference discharge threshold (stored as ``Q97_ref``)
- **H97**: Habitat availability at Q97 in reference conditions (stored as ``H97_ref``)
- **ISH (Index of Spatial Habitat)**: Measures average habitat reduction (0 = severe loss, 1 = no change)
- **ITH (Index of Temporal Habitat)**: Measures habitat stress duration (0 = severe stress, 1 = no stress)
- **IH (Habitat Index)**: Overall habitat alteration index, minimum of ISH and ITH (0 = severe impact, 1 = no impact)
- **HSD (Habitat Stress Days)**: Cumulative measure of habitat stress events

Habitat outputs are returned as {py:class}`sarawater.habitat.HabitatIndicesResult` objects.

**Computing Habitat Indices:**

1. Add HQ curves to the reach with {py:meth}`sarawater.reach.Reach.add_HQ_curve`
2. Compute indices for a scenario with {py:meth}`sarawater.scenarios.Scenario.compute_IH_for_species`

The method accepts a single species name, a list of species, or ``None`` (computes for all available species).

Lower-level habitat computation utilities are available as:

- {py:func}`sarawater.habitat.compute_habitat_indices`
- {py:func}`sarawater.habitat.compute_h_ucut`
- {py:func}`sarawater.habitat.compute_IH`

### Sediment transport alteration

Sediment transport alteration can be computed at scenario level and analyzed with annual summaries:

- {py:meth}`sarawater.scenarios.Scenario.compute_sediment_load`: computes and stores a sediment transport table (see {py:func}`sarawater.sediment_load.compute_sediment_load`)
- {py:meth}`sarawater.scenarios.Scenario.compute_annual_sediment_budget`: aggregates annual sediment budgets (via {py:func}`sarawater.sediment_load.compute_annual_sediment_volume`)
- {py:meth}`sarawater.scenarios.Scenario.plot_scenario_sediment_transport`: plots time series of sediment transport capacity

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