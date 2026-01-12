.. _background:

Background
==========

SARAwater is a Python package for analyzing different types of alterations in river reaches subject to flow abstraction, under different flow management scenarios. It is designed to help water resource managers and environmental scientists assess the ecological impacts of water abstractions and develop sustainable management strategies.

IHA index
---------
Indicators of Hydrologic Alteration (IHA) is a widely used method for assessing the degree of hydrologic change caused by human activities, such as the construction of water diversions [1]_ [2]_. 

The IHA evaluates 32 or 33 ecologically relevant parameters derived from daily discharge data. These parameters are organized into five distinct groups, each characterizing a different aspect of the flow regime:

1. Magnitude of Monthly Flow: Measures the average water conditions for each month, which impacts habitat availability for aquatic organisms.

2. Magnitude and Duration of Annual Extremes: Examines the intensity and length of the highest and lowest flow events (e.g., 1-, 3-, 7-, 30-, and 90-day annual minima and maxima).

3. Timing of Annual Extremes: Tracks the Julian dates of the annual 1-day maximum and minimum flows, which often serve as critical biological cues for spawning or migration.

4. Frequency and Duration of Pulses: Analyzes how often high and low flow pulses occur and how long they last, which have many influences on ecosystems, for example, they are critical for nutrient exchange and floodplain connectivity.

5. Rate and Frequency of Water Condition Changes: Measures how quickly flow rises or falls and the number of flow reversals.

Usually, IHA has been used to assess pre and post impact conditions, here it is used to assess upstream and downstream of a withdrawal point, hypothezing that upstream is the reference -natural- condition. This assumption could be not exact in some cases, for example when there are multiple withdrawals or discharges along the river reach. In that case, the user should carefully choose the reference natural condition.

In this work we try to assess the ecological impacts of flow alterations in rivers, through a parameter that comes from the IHA analysis and unifies all the 33 parameters. This parameter is called here IHA index, and it can be computed following two approaches.

MesoHABSIM
----------
The MesoHABSIM (Mesohabitat Simulation) model is a tool for simulating the physical habitat conditions in river systems. It allows users to assess the impacts of flow alterations on aquatic habitats by modeling the availability and quality of different habitat types under various flow scenarios. 

MesoHABSIM assesses habitat availability for a specific species or life stage by using the mesoscale as the spatial reference. In this context, “mesoscale” refers to the spatial scale of hydro-morphological units [3]_.
The MesoHABSIM approach is based on three main components:

1. Description of the hydro-morphological and environmental characteristics of mesohabitats, representing how the spatial mosaic of habitat units varies as a function of discharge

2. A statistical biological model describing target-species preferences with respect to the environmental characteristics of mesohabitats

3. Coupling the hydro-morphological description (1) with the biological models (2), enabling the definition of a quantitative relationship between available habitat and discharge

The resulting habitat–discharge relationship (component 3) forms the basis for quantifying release scenarios consistent with environmental flow requirements. 

SARAwater provides functionalities for running MesoHABSIM simulations and analyzing the results, starting from discharge time series and existing habitat–discharge relationships.

Sediment transport
---------


References
----------

.. [1] Richter, B. D., Baumgartner, J. V., Powell, J., & Braun, D. P. (1996). A method for assessing hydrologic alteration within ecosystems. Conservation Biology, 10(4), 1163-1174.
.. [2] Richter, B., Baumgartner, J., Wigington, R., & Braun, D. (1997). How much water does a river need?. Freshwater Biology, 37: 231-249.
.. [3] Parasiewicz, P. (2007). The MesoHABSIM Model Revisited. River Research and Applications, 23, 893–903.
