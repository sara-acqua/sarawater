.. _user_guide:

User Guide
==========

This section provides a comprehensive overview of the main classes and functions available in the `SARAwater` package.

Reach objects
-------------

The ``Reach`` class is the central object in SARAwater, representing a river reach with its natural flow time series. It serves as a container for scenarios and associated data.

**Key Attributes:**

- ``name``: Name of the reach
- ``dates``: List of datetime objects for the time series
- ``Qnat``: Natural flow rate time series (numpy array)
- ``Qabs_max``: Maximum water abstraction threshold (m³/s)
- ``scenarios``: List of scenarios added to the reach
- ``IHA_nat``: Natural flow IHA indicators (computed automatically)

**Main Methods:**

- ``add_scenario(scenario)``: Add a scenario to the reach
- ``add_ecological_flow_scenario(name, description, k=0.2, p=2.0)``: Create and add an ecological flow scenario with monthly adjustments
- ``add_HQ_curve(HQ_curve)``: Add habitat-discharge curves for different species/life stages
- ``get_list_available_HQ_curves()``: Get list of available habitat curves
- ``get_HQ_curve(curve_name)``: Retrieve a specific habitat-discharge curve
- ``print_scenarios()``: Print a list of all scenarios added to the reach
- ``export_scenario_summary()``: Export a comprehensive summary table of all scenarios with their parameters and indices

Scenario objects
----------------

Scenarios represent different water management alternatives. SARAwater provides a base ``Scenario`` class and specialized subclasses:

**Scenario (base class)**

The parent class for all scenario types, containing shared functionality.

**Key Attributes:**

- ``name``: Name of the scenario
- ``description``: Description of the scenario
- ``reach``: Associated Reach object
- ``Qabs_max``: Maximum water abstraction, if different from Reach (m³/s)

- ``Qreq``: Minimum release flow time series (m³/s)
- ``Qrel``: Released flow rate time series (m³/s)
- ``IHA``: Dictionary of IHA indicators
- ``IH``: Dictionary of habitat indices for different species

**Main Methods:**

- ``compute_Qrel()``: Calculate the released flow time series based on Qnat, Qreq, and Qabs_max
- ``plot_scenario_discharge(start_date=None, end_date=None, **kwargs)``: Plot the released discharge time series
- ``compute_IHA(**kwargs)``: Compute Indicators of Hydrologic Alteration
- ``compute_IHA_index(index_metric, index_options={})``: Compute IHA indices (IARI or normalized_IHA)
- ``compute_natural_abstracted_volumes(month_to_season=None)``: Calculate water volumes abstracted from the reach
- ``compute_IH_for_species(species=None, **kwargs)``: Compute habitat indices for one or more species

**ConstScenario (constant release)**

A scenario with constant monthly flow requirements.

**Parameters:**

- ``Qreq_months``: List of 12 float values representing monthly constant flow rates (l/s)

**PropScenario (proportional release)**

Child class for scenarios with flow requirements proportional to the incoming flow. In scenarios with a proportional flow requirement, :math:`Q_{req}` is computed as a fraction of the incoming flow discharge, :math:`Q_{in}`, according to the formula:

.. math::

   Q_{req} = Q_{base} + c_{in} \cdot Q_{in}

where :math:`Q_{base}` is a base flow requirement (e.g., to maintain minimum ecological conditions), and :math:`c_{in}` is a coefficient that defines the proportion of the incoming flow to be included in the flow requirement.

This formula is then adjusted to ensure that :math:`Q_{req}` remains within user-specified minimum and maximum bounds, :math:`Q_{req,min}` and :math:`Q_{req,max}`, respectively. This step is needed because low flow requirements may cause severe alteration in the downstream reach, while high flow requirements may lead to a very low water abstraction, which might not be sufficient. The complete definition of :math:`Q_{req}` in proportional scenarios is given by the piecewise function:

.. math::

   Q_{req} = \begin{cases}
   Q_{req,min} & \text{if } Q_{base} + c_{in} \cdot Q_{in} \leq Q_{req,min}\\
   Q_{base} + c_{in} \cdot Q_{in} & \text{if } Q_{req,min} < Q_{base} + c_{in} \cdot Q_{in} < Q_{req,max}\\
   Q_{req,max} & \text{if } Q_{base} + c_{in} \cdot Q_{in} > Q_{req,max}
   \end{cases}

**Parameters:**

- ``Qbase``: Base flow rate (m³/s)
- ``c_Qin``: Proportionality coefficient (dimensionless)
- ``Qreq_min``: Minimum release constraint (m³/s)
- ``Qreq_max``: Maximum release constraint (m³/s)

Compute the released flow discharge for each scenario
-----------------------------------------------------

Given the incoming flow discharge :math:`Q_{nat}`, the flow requirement :math:`Q_{req}` and the maximum abstractable flow :math:`Q_{abs,max}`, the released flow discharge :math:`Q_{rel}` is computed as:

.. math::

   Q_{rel} = \begin{cases}
   Q_{nat} & \text{if } Q_{nat} \leq Q_{req} & \text{(4.1)}\\
   Q_{req} & \text{if } Q_{req} < Q_{nat} < Q_{req} + Q_{abs,max} & \text{(4.2)}\\
   Q_{nat} - Q_{abs,max} & \text{if } Q_{nat} \geq Q_{req} + Q_{abs,max} & \text{(4.3)}
   \end{cases}

Where the three cases correspond to the following:

* **(4.1)** The incoming flow :math:`Q_{nat}` is lower than the flow requirement :math:`Q_{req}`; therefore, no water is abstracted and the released flow discharge :math:`Q_{rel}` equals the incoming flow. This usually happens in low-flow periods.
* **(4.2)** There is enough incoming flow to satisfy the flow requirement, and the abstracted flow discharge :math:`Q_{abs}` is lower than the maximum abstractable flow :math:`Q_{abs,max}`. Recall that, according to Equation (1), :math:`Q_{abs} = Q_{nat} - Q_{rel}` (where, in this case, :math:`Q_{rel} = Q_{req}`). This is the most "common" case, where the flow requirement rule is applied straightforwardly.
* **(4.3)** The incoming flow :math:`Q_{nat}` is so large that the maximum abstractable flow can be diverted while still releasing a flow rate larger than the flow requirement. This usually happens during flood events.

The piecewise function (4.1-4.3) is implemented in the ``compute_Qrel`` method of the class ``Scenario``; therefore, to compute the released flow time series for each scenario we can simply write
.. code-block:: python

   scenario.compute_Qrel()

Assessing alterations
-----------------------

SARAwater provides several methods to quantify hydrological, habitat, and sediment transport alterations.

Hydrologic alteration
^^^^^^^^^^^^^^^^^^^^^

**Indicators of Hydrologic Alteration (IHA)**

The IHA framework quantifies changes in flow regime by analyzing 33 parameters grouped into 5 categories:

- **Group 1**: Monthly flow statistics (mean flows for each month)
- **Group 2**: Magnitude and duration of extreme conditions (min/max flows over 1, 3, 7, 30, 90-day windows; base flow; zero-flow days)
- **Group 3**: Timing of extreme conditions (Julian dates of annual min/max flows)
- **Group 4**: Frequency and duration of high/low pulses
- **Group 5**: Rate and frequency of flow changes (rise/fall rates, number of reversals)

**Computing IHA:**

Use ``scenario.compute_IHA()`` to calculate IHA indicators for a scenario. The method returns a dictionary with yearly values for each parameter.

**IHA Indices:**

Two aggregate indices are available:

- **IARI (Index of Alteration of Hydrologic Regime)**: Measures overall deviation from the natural hydrologic regime. When equal to 0 indicates an unaltered condition, while above 0.15 indicates severe alteration.
  - Compute with: ``scenario.compute_IHA_index('IARI')``
  
- **Normalized IHA**: Normalized deviations of IHA parameters.
  
  - Compute with: ``scenario.compute_IHA_index('normalized_IHA')``

Habitat alteration
^^^^^^^^^^^^^^^^^^

Habitat alteration is quantified using habitat-discharge (HQ) curves and the UCUT (Under-threshold Cumulative Curve) methodology.

**Habitat Indices:**

The following indices quantify habitat alteration for aquatic species:

- **H97**: Habitat availability at Q97 (low flow threshold)
- **ISH (Index of Spatial Habitat)**: Measures average habitat reduction (0 = severe loss, 1 = no change)
- **ITH (Index of Temporal Habitat)**: Measures habitat stress duration (0 = severe stress, 1 = no stress)
- **IH (Habitat Index)**: Overall habitat alteration index, minimum of ISH and ITH (0 = severe impact, 1 = no impact)
- **HSD (Habitat Stress Days)**: Cumulative measure of habitat stress events

**Computing Habitat Indices:**

1. Add HQ curves to the reach: ``reach.add_HQ_curve(HQ_dataframe)``
2. Compute indices for a scenario: ``scenario.compute_IH_for_species(species='species_name')``

The method accepts a single species name, a list of species, or ``None`` (computes for all available species).

.. Sediment transport alteration
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualizing the results
-----------------------

The ``ReachPlotter`` class provides comprehensive visualization capabilities for comparing scenarios.

**Creating a Plotter:**

.. code-block:: python

   from sarawater.visualization import ReachPlotter
   
   plotter = ReachPlotter(reach, output_dir='outputs')

**Available Plots:**

**Discharge and Flow Regime:**

- ``plot_scenarios_discharge(start_date, end_date, log_scale=True, save=False)``: Compare discharge time series across scenarios
- ``plot_cases_duration(save=False)``: Visualize flow regime case durations (Case 1: Qnat ≤ Qreq; Case 2: abstraction occurring; Case 3: excess flow)
- ``plot_cases_duration_month(month, save=False)``: Monthly case duration comparison
- ``plot_monthly_abstraction(save=False)``: Compare monthly water abstraction volumes

**Hydrologic Alteration:**

- ``plot_iha_parameters(save=False)``: Multi-panel plot of all IHA parameters across scenarios
- ``plot_iari_groups(save=False)``: IARI values by IHA group
- ``plot_iari_summary(save=False)``: Overall IARI comparison across scenarios
- ``plot_nIHA_summary(save=False)``: Normalized IHA comparison
- ``plot_iha_boxplots(save=False)``: Box plots of IHA parameters showing inter-annual variability
- ``plot_relative_deviations(save=False)``: Relative deviations of IHA parameters from natural conditions
- ``plot_iari_vs_volume(save=False)``: Trade-off between hydrologic alteration (IARI) and water abstraction

**Habitat Analysis:**

- ``plot_hq_curves(species, save=False)``: Display habitat-discharge curves for specified species
- ``plot_habitat_timeseries(species, start_date, end_date, save=False)``: Compare habitat availability time series
- ``plot_ucut_curves(species, save=False)``: UCUT curves showing duration of habitat stress events
- ``plot_ih_vs_volume(save=False)``: Trade-off between habitat alteration (IH) and water abstraction
- ``plot_nIHA_vs_volume(save=False)``: Trade-off between normalized IHA and water abstraction

**Plot Options:**

All plotting methods support:

- ``save=False``: Set to ``True`` to save plots to the output directory
- Date filtering with ``start_date`` and ``end_date`` parameters (where applicable)
- Additional matplotlib kwargs can be passed to customize appearance