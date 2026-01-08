# SARAwater - Scenario-based Alteration of Rivers subject to water Abstraction

[![DOI](https://zenodo.org/badge/1130288528.svg)](https://doi.org/10.5281/zenodo.18183767)

<div class="nav3" style="height:705px;">
    <img src="tutorial/images/SARA_overview.png" alt="Overview of the SARAwater package" width="100%"></a>
</div>

Analysis of different types of alterations in river reaches subject to flow abstractions according to different water withdrawal and release scenarios.

---

For details on how to use the package, including code examples, visit the [Documentation](https://sara-acqua.github.io/sarawater/).

---

## Installation

This package supports Python 3.11+. It can be installed via pip:

```bash
pip install sarawater
```

## Bug reports and feature requests
If you wish to report a bug or you would like to see a new feature added, please [open an issue](https://github.com/sara-acqua/SARAwater/issues). 


## Citing
If you use SARAwater in your research, please cite the release published on Zenodo:

```bibtex
@software{Barile_2026_SARAwater,
  author       = {Barile, Gabriele and
                  Dal Santo, Matteo and
                  Crivellaro, Marta},
  title        = {{SARAwater: Scenario-based Alteration of Rivers subject to water Abstraction}},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {https://doi.org/10.5281/zenodo.18183767},
}
```

## Contributing
To install the package for development and documentation building, clone the repository and run:
```cmd
pip install -e .[dev,docs]
```
During development, tests can be run locally with:
```cmd
pytest
```
while documentation can be built with:
```cmd
cd docs
sphinx-build -M html source build
```
This repository relies on the `black` formatter for code formatting. To format the code locally, run:
```cmd
black
```