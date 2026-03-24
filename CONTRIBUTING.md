# Contributing to SARAwater

First off, thank you for considering contributing to SARAwater! We welcome contributions from everyone, whether you are fixing a bug, adding a new feature, or improving our documentation. 

If this is your first time contributing to an open-source Python package, don't worry: this guide will walk you through everything you need to know.

## Reporting Bugs and Requesting Features

You do not need to write code to contribute to SARAwater! If you find a bug, have a question, or want to suggest a new feature, the best place to start are the [GitHub Issues](https://github.com/sara-acqua/SARAwater/issues) of our repository.

* **Bug Reports:** If you encounter an error, please [open an issue](https://github.com/sara-acqua/SARAwater/issues) and include as much detail as possible. Let us know what version of SARAwater you are using, what you were trying to do, and the exact error message you received.
* **Feature Requests:** If you have an idea for a new scenario model or an improvement to the existing package, please [open an issue](https://github.com/sara-acqua/SARAwater/issues) to discuss it with the package maintainers before you spend time writing the code.

**If you want to contribute code to fix an issue or add a feature, please read the following sections!**

---

## Getting Started with Code Contributions: Setting Up Your Development Environment

**TIP**: If you are new to GitHub and open-source contributions, check out this [GitHub Guide for Beginners](https://guides.github.com/activities/hello-world/) to get familiar with the basics of forking, branching, and making pull requests. Some code editors allow for using git and GitHub directly from the interface, without needing to use the command line. For example, if you are using [VS Code](https://code.visualstudio.com/), you can also check out the [Source Control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview) page, which also includes a short tutorial specific to GitHub, to learn how to manage your contributions directly from the editor.

To start working on the code, you will need a local copy of the repository and a dedicated development environment. SARAwater supports **Python 3.11+**.

1.  **Fork the repository:** Click the "Fork" button at the top right of the [SARAwater GitHub page](https://github.com/sara-acqua/SARAwater). This creates a copy of the repository in your own GitHub account.
2.  **Clone your fork:** Download your copy to your local machine by cloning it, either using your Editor or the command line. For the latter, replace `YOUR-USERNAME` in the URL with your actual GitHub username and run:
    ```bash
    git clone https://github.com/YOUR-USERNAME/SARAwater.git
    cd SARAwater
    ```
3.  **Connect to the original repository:** Add the main SARAwater repository as an "upstream" remote so you can easily pull in the latest changes made by others:
    ```bash
    git remote add upstream https://github.com/sara-acqua/SARAwater.git
    ```
4.  **Install the package for development:**
    We use `pip` to install the package in "editable" mode (`-e`), along with all the dependencies needed for development and building documentation.
    ```bash
    pip install -e .[dev,docs]
    ```
    It is recommended to run the `pip install` command within a virtual environment (e.g., `venv` or `conda`) to keep the development version of SARAwater and its dependencies isolated from other Python projects on your machine.
5.  **Install Pandoc (for documentation):**
    Building the documentation requires Pandoc. You can download and install it from the [official Pandoc website](https://pandoc.org/installing.html).

## Coding Guidelines & Architecture

To keep the SARAwater codebase clean, reliable, and easy to maintain, please adhere to the following rules when writing code:

### No Keyboard Inputs
This package is designed to be used programmatically (e.g., in automated pipelines or Jupyter Notebooks). **Do not use `input()` to ask the user for parameter values.** All necessary information, parameters, and configurations must be passed explicitly as arguments when instantiating objects or calling their methods.

### AI and LLM Usage Policy
You are welcome to use AI assistants to help write code, but **do not take LLM-generated code for granted**. As the contributor, you are responsible for the code you submit. A few guidelines to keep in mind when working with AI-generated code:
* Double-check the functionalities that are already implemented in the codebase before writing new code. If you find that the functionality you want to add is already implemented, please use it instead of writing new code.
* Make sure to follow the same coding style as the rest of the codebase (e.g., variable naming). If you are unsure about the coding style, check out the existing code and follow it as closely as possible. For instance, do not add a `verbose` keyword argument to control output verbosity.
* Avoid "hidden" numerical approximations or hardcoded values. If a variable has a certain range of acceptable values, `raise` an error if it happens to fall outside of that range, instead of silently using a default value.

### Formatting
This repository relies on `black` to ensure a consistent code style across the entire project. Before submitting your code, format it by running:
```bash
black .
```

## Contributing to the Documentation

### Docstrings
Good documentation is just as important as good code. We use [**NumPy-style docstrings**](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for all new functions, methods, and classes. 

Here is an example of what a standard NumPy-style docstring looks like:

```python
def calculate_velocity(distance, time):
    """
    Calculates the velocity of an object.

    Parameters
    ----------
    distance : float
        The distance traveled in meters.
    time : float
        The time taken in seconds.

    Returns
    -------
    float
        The calculated velocity in meters per second.
    """
    return distance / time
```

### Contributing to the web Documentation
The core of the SARAwater package documentation is written in the [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) (.rst) files located in the `docs/source` directory and published on the package website: https://sara-acqua.github.io/sarawater/. To contribute to the documentation, you can edit the `.rst` files directly. Whenever a change in the `.rst` files is detected, the documentation will be automatically rebuilt using [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and used to update the package website on GitHub Pages.

To ensure your documentation is correctly formatted after your edits, you can build the documentation locally. From the root of the repository (that is, the main directory where `pyproject.toml` is located), run:
```bash
cd docs
sphinx-build -M html source build
```
You can then open the generated HTML files in the `docs/build/html` directory in your web browser.

## Running Tests

We use `pytest` to ensure that new changes do not break existing functionality. Before opening a Pull Request, verify that all tests pass by running:
```bash
pytest
```

## Submitting Your Code

Since direct write access to the main SARAwater repository is restricted to maintainers, you will submit your code by pushing it to your personal fork and opening a Pull Request (PR).

1.  **Update your local branch:** Before making changes, ensure your local `main` branch is up to date with the original repository:
    ```bash
    git checkout main
    git pull upstream main
    ```
2.  **Create a new branch:** Never work directly on the `main` branch. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature/new-scenario-model
    ```
3.  **Commit your changes:** Write clear, descriptive commit messages explaining what you changed and why.
4.  **Push to your fork:** Push your new branch up to your personal GitHub copy (`origin`):
    ```bash
    git push origin feature/new-scenario-model
    ```
5.  **Open a Pull Request:** Go to the main [SARAwater GitHub page](https://github.com/sara-acqua/SARAwater). You should see a prompt to "Compare & pull request" for your recently pushed branch. Click it, fill out the description, and submit\!