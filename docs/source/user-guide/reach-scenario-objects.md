# Reach and Scenario Objects

## Reach objects
The {py:class}`sarawater.reach.Reach` class is the central object in SARAwater, representing a river reach with its natural flow time series. It serves as a container for scenarios and associated data.

### Key Attributes
- ``name``: Name of the reach
- ``dates``: List of datetime objects for the time series
- ``Qnat``: Natural flow rate time series (NumPy array)
- ``Qabs_max``: Maximum water abstraction threshold (m3/s)
- ``scenarios``: List of scenarios added to the reach
- ``IHA_nat``: Natural-flow IHA indicators (computed automatically at construction)

### Main Methods
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

### Key Attributes

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

### Main Methods

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

### Compute the released flow discharge for each scenario

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