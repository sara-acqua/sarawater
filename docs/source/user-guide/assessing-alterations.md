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