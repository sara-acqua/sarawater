# Getting Started

Welcome to `SARAwater`! This guide will walk you through installing the package and setting up your first stream reach and scenario.

---

## Installation

### Requirements
* Python 3.11 or higher

### Installing via pip
The easiest way to install `SARAwater` is from PyPI using `pip`:

```bash
pip install sarawater
```

### Installing from Source (Development)
To install the latest development version directly from GitHub:

```bash
git clone https://github.com/sara-acqua/sarawater.git
cd SARAwater
pip install -e .
```

### Verify Installation
Run the following command to check if `SARAwater` was installed correctly:

```bash
python -c "import sarawater as sara; print(sara.__file__)"
```
You should see the path to the `__init__.py` file of the `sarawater` package printed in the terminal.

## Quickstart Example

In this example, we will:
1. Load flow discharge data from a CSV file.
2. Initialize a `Reach` object.
3. Define a minimum flow requirement scenario (`ConstScenario`).
4. Apply the scenario to the reach.

For a more comprehensive example, check out [Tutorial 1](tutorials/tutorial_1_IHA/tutorial_1_IHA.ipynb).

### Step 1: Create a Reach Object
To initialize a `Reach`, you need a list of dates (as `datetime` objects) and a NumPy array of discharge values. 

Assuming your flow data is stored in `data.csv` with "date" and "discharge" columns:

```python
import pandas as pd
import sarawater as sara

# Read the CSV file
df = pd.read_csv("path/to/your/data.csv", parse_dates=["date"])

# Extract dates and discharge values
dates = df["date"].dt.to_pydatetime().tolist()
discharge = df["discharge"].values

# Maximum abstraction capacity in m³/s (set to a large number if unknown)
max_abstraction = 1e6

# Create the Reach object
my_reach = sara.Reach("My Reach", dates, discharge, max_abstraction)
```

### Step 2: Define and Apply a Scenario
A `Scenario` defines rules for downstream flow releases (e.g., minimum flow requirements). 

The code below uses `ConstScenario` to set a minimum flow of **1.0 m³/s** for most months, reduced to **0.5 m³/s** during the summer (June, July, August):

```python
# Monthly minimum flow requirements in m³/s (Jan - Dec)
Qreq_months = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

# Create and attach the scenario
my_scenario = sara.ConstScenario("SR", "Summer Reduction Scenario", my_reach, Qreq_months)
my_reach.add_scenario(my_scenario)
```