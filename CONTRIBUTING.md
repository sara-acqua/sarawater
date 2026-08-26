# Contributing

First off, thank you for considering contributing to SARAwater! We welcome contributions from everyone, whether you are fixing a bug, adding a new feature, or improving our documentation. 

If this is your first time contributing to an open-source Python package, don't worry: this guide will walk you through everything you need to know.

## Reporting Bugs and Requesting Features

You do not need to write code to contribute to SARAwater! If you find a bug, have a question, or want to suggest a new feature, the best place to start are the [GitHub Issues](https://github.com/sara-acqua/SARAwater/issues) of our repository.

* **Bug Reports:** If you encounter an error, please [open an issue](https://github.com/sara-acqua/SARAwater/issues) and include as much detail as possible. Let us know what version of SARAwater you are using, what you were trying to do, and the exact error message you received.
* **Feature Requests:** If you have an idea for a new scenario model or an improvement to the existing package, please [open an issue](https://github.com/sara-acqua/SARAwater/issues) to discuss it with the package maintainers before you spend time writing the code.

> [!TIP]
> If you want to contribute code to fix an issue or add a feature, please read the following sections!

## Getting Started with Code Contributions: How to Fork the Repository, Create a Branch, and submit a Pull Request
> [!TIP]
> If you are new to GitHub and open-source contributions, check out this [GitHub Guide for Beginners](https://guides.github.com/activities/hello-world/) to get familiar with the basics of forking, branching, and making pull requests. Some code editors allow for using git and GitHub directly from the interface, without needing to use the command line. For example, if you are using [VS Code](https://code.visualstudio.com/), you can also check out the [Source Control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview) page, which also includes a short tutorial specific to GitHub, to learn how to manage your contributions directly from the editor.

To start working on the code, you will need your own fork of the repository and a dedicated development environment. SARAwater supports **Python 3.11+**.

1.  **Fork the repository:** Navigate to the [SARAwater GitHub repository](https://github.com/sara-acqua/SARAwater) and click **Fork** in the top-right corner. Choose your personal GitHub account as the destination and create the fork. This creates a copy of the repository that you can modify freely.
2.  **Clone your fork:** Download your fork to your local machine. Replace `YOUR-USERNAME` with your GitHub username and run:
    ```bash
    git clone https://github.com/YOUR-USERNAME/SARAwater.git
    cd SARAwater
    ```
3.  **Connect the official repository as upstream:** Add the main SARAwater repository as an `upstream` remote so you can pull in the latest changes made by others:
    ```bash
    git remote add upstream https://github.com/sara-acqua/SARAwater.git
    ```
    You can verify that both remotes are configured with:
    ```bash
    git remote -v
    ```
4.  **Keep your fork in sync before starting new work:** From time to time, update your local `main` branch from the official repository and push the changes to your fork:
    ```bash
    git checkout main
    git fetch upstream
    git merge upstream/main
    git push origin main
    ```
5.  **Install the package for development:**
    We use `pip` to install the package in "editable" mode (`-e`), along with all the dependencies needed for development and building documentation.
    ```bash
    pip install -e .[dev,docs]
    ```
    It is recommended to run the `pip install` command within a virtual environment (e.g., `venv` or `conda`) to keep the development version of SARAwater and its dependencies isolated from other Python projects on your machine.

### Create a feature branch and submit your contribution

> [!WARNING]
> **Never commit directly to `main`!** Your local and forked `main` branches should remain exact mirrors of the official SARAwater `main` branch. Always create a dedicated feature branch before making changes.

Once your local environment is ready, follow this workflow for each contribution:

1. **Create a feature branch** for your bugfix or new feature:
   ```bash
   git checkout -b feature/scenario-model
   ```
2. **Make your changes** and keep them focused on a single issue or improvement.
3. **Format and test your changes** before submitting them:
   ```bash
   black .
   pytest
   ```
4. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "feat: add new scenario model calculation"
   ```
5. **Push your branch to your fork** and open a pull request:
   ```bash
   git push -u origin feature/scenario-model
   ```
   Then open the repository on GitHub and click **Compare & pull request** to submit your changes for review.

## Contributing to the web Documentation
The core of the SARAwater package documentation is written in Markdown files located in the `docs/source` directory and published on the package website: https://sara-acqua.github.io/sarawater/. To contribute to the documentation, you can edit the `.md` files directly in your branch. Whenever a change in the `.md` files is detected, the documentation will be automatically rebuilt using [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and used to update the package website on GitHub Pages.

This repository relies on the MyST language for all Markdown files, which allows for advanced features such as cross-references, citations, and math equations. You can find more information about MyST in the [official documentation](https://myst-parser.readthedocs.io/en/latest/).

To ensure your documentation is correctly formatted after your edits, you can build the documentation locally. From the root of the repository (that is, the main directory where `pyproject.toml` is located), run:
```bash
cd docs
sphinx-build -M html source build
```
You can then open the generated HTML files in the `docs/build/html` directory in your web browser.

## Coding Guidelines & Architecture

To keep the SARAwater codebase clean, reliable, and easy to maintain, please adhere to the following rules when writing code:

### No Keyboard Inputs
This package is designed to be used programmatically (e.g., in automated pipelines or Jupyter Notebooks). **Do not use `input()` to ask the user for parameter values.** All necessary information, parameters, and configurations must be passed explicitly as arguments when instantiating objects or calling their methods.

### AI and LLM Usage Policy
You are welcome to use AI assistants to help write code, but **do not take LLM-generated code for granted**. As the contributor, you are responsible for the code you submit. A few guidelines to keep in mind when working with AI-generated code:
* Double-check the functionalities that are already implemented in the codebase before writing new code. If you find that the functionality you want to add is already implemented, please use it instead of writing new code.
* Ensure that the coding conventions and docstring guidelines are followed. The repository contains an instruction file ([.github/copilot-instructions.md]) that is read automatically by Github Copilot and that enforces these conventions. If you are using an AI assistant other than Copilot, consider attaching the same file as context to your prompts.

### Formatting
This repository relies on `black` to ensure a consistent code style across the entire project. Before submitting your code, format it by running:
```bash
black .
```

## Docstrings Guidelines for `sarawater` Contributors
All Python code in `sarawater` must be documented using [**NumPy-style docstrings**](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard). These docstrings must follow specific formatting rules, so that they can be effectively handled by `sphinx-autoapi` and published into the official HTML documentation (API reference page).

### Core Principles for Writing Docstrings

1. **reStructuredText Syntax:** Although descriptive docs (`.md` files) use MyST Markdown, **docstrings inside Python files must use reStructuredText (reST)**.
2. **Imperative Mood:** Start summary lines with an imperative command (e.g., `Compute flow requirements...` instead of `Computes...` or `This function calculates...`).
3. **Underline Length:** Section headers (like `Parameters` or `Returns`) must be underlined with hyphens (`-`) matching or exceeding the length of the section title.
4. **Indentation:** Indent parameter and return descriptions by **4 spaces**.

### Summary Line & Extended Description
Begin with a single line describing what the object does. Leave a blank line before adding a detailed explanation or mathematical context if necessary.

```python
def compute_Qrel(Qnat: np.ndarray, Qreq: np.ndarray, Qabs_max: float) -> np.ndarray:
    """Compute the released flow discharge time series.

    Applies the piecewise release rule based on natural incoming flow, 
    environmental flow requirements, and maximum diversion capacity.
    """
```

### Parameters
Format each parameter as `name : type`. Append `, optional` or `, default=value` when applicable.

```python
Parameters
----------
Qnat : np.ndarray
    Natural flow rate time series in m3/s.
Qreq : np.ndarray
    Minimum release flow requirement time series in m3/s.
Qabs_max : float
    Maximum water abstraction threshold in m3/s.
k : float, default=0.2
    Proportionality factor for environmental flow scaling.
```

### Returns / Yields
Specify the return type on the first line, followed by a 4-space indented description.

```python
Returns
-------
np.ndarray
    Time series of released discharge `Qrel` in m3/s.
```

If returning multiple values, list them explicitly:

```python
Returns
-------
IHAIndexResult
    Dataclass containing calculated group scores and total score.
IHAResult
    Dictionary containing raw yearly indicator values per group.
```

### Raises
List exceptions intentionally raised by the function.

```python
Raises
------
ValueError
    If `Qnat` contains negative discharge values or missing dates.
```

### Examples
Provide runnable doctest examples starting with `>>>`.

```python
Examples
--------
import pandas as pd
import numpy as np
import sarawater as sara
minrel_df = pd.read_csv(min_release_filepath, header=None)
Qreq_months = np.array(minrel_df[1].tolist()) / 1000.0
MFR_scenario = sara.ConstScenario(
    name="MFR",
    description="Minimum Flow Requirement scenario from CSV file",
    reach=my_reach,
    Qreq_months=Qreq_months,
)
my_reach.add_scenario(MFR_scenario)
```

### Notes & References
Include theoretical context, governing equations, or literature citations referencing keys from `references.bib`.

```python
Notes
-----
Calculated using the 33-indicator framework described by Richter et al. :cite:p:`richter1996`.
```

---

### Formatting & Cross-Referencing Cheat Sheet

Use standard Sphinx reST roles inside docstrings to generate direct links across the site:

| Content Element | reST Docstring Syntax | Output Rendered in Docs |
| :--- | :--- | :--- |
| **Inline Variable/Code** | `` `variable` `` | Monospaced text |
| **Class Reference** | `:class:`sarawater.Scenario`` | Link to `Scenario` API page |
| **Short Class Reference** | `:class:`~sarawater.Scenario`` | Link displaying only `Scenario` |
| **Function Reference** | `:func:`sarawater.compute_IHA`` | Link to function API page |
| **Method Reference** | `:meth:`sarawater.Scenario.compute_Qrel`` | Link to method API page |
| **External Class (NumPy)** | `:class:`numpy.ndarray`` | Link to external NumPy docs |
| **External Class (Pandas)**| `:class:`pandas.DataFrame`` | Link to external Pandas docs |

---

### Complete Reference Example

```python
def compute_IHA_index(
    Qnat: np.ndarray,
    Qrel: np.ndarray,
    dates: list[datetime.datetime],
    index_metric: str,
    weights: Sequence[float] | None = None,
    IHA_nat: IHAResult | None = None,
    IHA_alt: IHAResult | None = None,
    epsilon: float = 1e-5,
) -> tuple[IHAResult, IHAIndexResult]:
    """Compute the IHA indicators and the related IARI index for each year.

    Parameters
    ----------
    Qnat : np.ndarray
        Natural flow rate time series
    Qrel : np.ndarray
        Released flow rate time series
    dates : list[datetime.datetime]
        List of datetime objects corresponding to flow rates
    index_metric : str
        Name of the index to compute (IARI, normalized_IHA)
    weights : list[float], optional
        List of 5 weights for each group of IHA parameters. Must sum to 1.
        If None, equal weights (0.2) will be used.
    IHA_nat : IHAResult, optional
        Pre-computed IHA for the natural flow series. If provided, it will be used instead of computing it again.
    IHA_alt : IHAResult, optional
        Pre-computed IHA for the altered flow series. If provided, it will be used instead of computing it again.
    epsilon : float, optional
        Small value to prevent division by zero in calculations. Used only if index_metric is "normalized_IHA". Default is 1e-5.

    Returns
    -------
    tuple[IHAResult, IHAIndexResult]
        Tuple containing:

        - The altered-state values of the IHA, stored in a dictionary where each key is a group name and each value is a dictionary of indicators belonging to that group.
        - The altered-state values of the IHA index, stored in a IHAIndexResult object with per-group values and aggregated values.

    Examples
    --------
    >>> import datetime
    >>> import numpy as np
    >>> from sarawater.IHA import compute_IHA_index
    >>> dates = [
    ...     datetime.datetime(2000, 1, 1) + datetime.timedelta(days=i)
    ...     for i in range(365)
    ... ]
    >>> q_nat = np.linspace(10.0, 30.0, 365)
    >>> q_rel = np.maximum(q_nat - 2.0, 0.0)
    >>> iha_res, index_res = compute_IHA_index(
    ...     q_nat,
    ...     q_rel,
    ...     dates,
    ...     index_metric='IARI',
    ... )
    >>> group1_scores = index_res.groups['Group1']
    >>> annual_scores = index_res.aggregated
    >>> sorted(iha_res.keys())
    ['Group1', 'Group2', 'Group3', 'Group4', 'Group5']

    See Also
    --------
    :func:`~sarawater.IHA.compute_IHA` : Base indicator calculation.
    """
    ...
```



## Running Tests

We use `pytest` to ensure that new changes do not break existing functionality. Before opening a Pull Request, verify that all tests pass by running:
```bash
pytest
```