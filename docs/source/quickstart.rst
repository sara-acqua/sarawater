.. _quickstart:

Quickstart
==========
This quickstart guide will help you get started with `SARAwater` by walking you through a simple example of how to use the package. This guide will show you how to retrieve these two variables and use them to create a :code:`Reach` object. Then, we will consider one "scenario", that is, a minimum flow requirement, to our reach and evaluate its alteration. A more complete example is provided in the :ref:`Tutorial 1<tutorial_1_IHA>`.

The first step in using `SARA` will (almost always) be to import the package and create a :code:`Reach` object. To initialize a :code:`Reach`, we need a flow discharge time series. Specifically, we have to provide the list of dates, provided as a list of :code:`datetime` objects, and a numpy array of discharge values. Moreover, we need to provide a name for the reach and the maximum flow discharge that can be abstracted from our reach. If the latter is not known, we can set it to a large number.

Assuming that your flow discharge data is stored in a CSV file with two columns: "date" and "discharge", you can use the following code to read the data and create a :code:`Reach` object:

.. code-block:: python

   import sarawater as sara
   import pandas as pd

   # Read the CSV file
   df = pd.read_csv("path/to/your/data.csv", parse_dates=["date"])

   # Extract the dates and discharge values
   dates = df["date"].dt.to_pydatetime().tolist()
   discharge = df["discharge"].values

   # Define the maximum abstraction (set to a large number if unknown), in cubic meters per second
   max_abstraction = 1e6

   # Create a Reach object
   my_reach = sara.Reach("My Reach", dates, discharge, max_abstraction)

Let's now add a :code:`Scenario`, that is, a rule that determines the minimum flow that must be released downstream of the abstraction.

The simplest form of :code:`Scenario` consists of prescribing a minimum flow value for each month. This type of scenario can be created using the :code:`ConstScenario` class, which requires us to provide a list of 12 values via the argument :code:`Qreq_months`, one for each month of the year.

The following code creates a scenario that prescribes a minimum flow of 1 cubic meter per second for each month of the year, except for the summer season (june, july, and august), when the minimum flow is set to 0.5 cubic meters per second.

.. code-block:: python

   # Define the minimum flow for each month (in cubic meters per second)
   Qreq_months = [1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1, 1]

   # Create a ConstScenario object
   my_scenario = sara.ConstScenario("My Scenario", Qreq_months)

After creating the scenario, we can add it to our reach using the :code:`add_scenario` method.

.. code-block:: python

   my_reach.add_scenario(my_scenario)

