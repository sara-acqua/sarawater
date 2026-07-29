# SARAwater — Agent Instructions

**SARAwater** (Scenario-based Alteration of Rivers subject to water Abstraction) is a Python package for analysing hydrological, habitat, and sediment transport alterations in river reaches under different water abstraction scenarios.

## Key commands

```bash
pip install -e .[dev,docs]   # install for development
pytest                        # run all tests
black .                       # format code (required before committing)
cd docs && sphinx-build -M html source build  # build docs (requires Pandoc)
```

## Architecture

The typical workflow is:

1. Create a `Reach` (natural flow time series + max abstraction)
2. Add one or more `Scenario` objects to the reach
3. Call `scenario.compute_Qrel()` to compute released flow
4. Compute alteration indices (IHA, habitat, sediment)
5. Visualise with `ReachPlotter`

### Core modules

| Module | Purpose |
|---|---|
| `reach.py` | `Reach` class — central container for flow data and scenarios |
| `scenarios.py` | `Scenario`, `ConstScenario`, `PropScenario` — water management alternatives |
| `IHA.py` | `compute_IHA()`, `compute_IHA_index()` — hydrologic alteration indicators |
| `habitat.py` | `compute_habitat_indices()` — habitat-discharge curve analysis (UCUT method) |
| `hydraulics.py` | `steady_flow_solver()` — cross-section hydraulics (Manning/Strickler) |
| `sediment_load.py` | `compute_sediment_load()` — bedload transport per grain-size class |
| `visualization.py` | `ReachPlotter` — multi-scenario comparison plots |
| `utils.py` | Shared validators (e.g. `_validate_positive_numeric`) |

### Key classes

**`Reach(name, dates, Qnat, Qabs_max)`**
- `dates`: `list[datetime]`, `Qnat`: `np.ndarray` (non-negative, m³/s)
- Add geometry before sediment analysis: `add_cross_section_geometry()`, `add_grain_size_distribution()`
- Add habitat curves before habitat analysis: `add_HQ_curve()`
- `IHA_nat` is computed automatically at construction

**`ConstScenario(name, description, reach, Qreq_months)`** — 12 monthly constant flow requirements  
**`PropScenario(name, description, reach, Qbase, c_Qin, Qreq_min, Qreq_max)`** — proportional release (`Qreq = Qbase + c_Qin * Qnat`, clamped to [Qreq_min, Qreq_max])

**`ReachPlotter(reach, output_dir='outputs')`** — use `save=True` on individual plot methods to write files.

## Coding conventions

- **Docstrings**: NumPy-style for all public functions, methods, and classes. See [CONTRIBUTING.md](../CONTRIBUTING.md) for an example.
- **Formatter**: `black` — run before every commit.
- **No `input()` calls** — the package is used programmatically; all parameters must be explicit arguments.
- **No `verbose` parameter** — do not add verbosity controls.
- **Validate at boundaries**: `raise ValueError` for out-of-range inputs instead of silently clamping or using defaults. Use `_validate_positive_numeric()` from `utils.py` for positive-numeric checks.
- **No hardcoded magic numbers** — if a value has a physical meaning or a valid range, document it and validate it.
- Units: flow in **m³/s**, lengths in **m**, grain sizes in **mm** (converted internally), sediment density default **2650 kg/m³**.

## Tests

Tests live in `tests/`. Each module has a corresponding `test_<module>.py`. Test data is in `tests/tests_data/`. Run with `pytest` from the repo root.

When adding a feature, add tests to the relevant file. Follow the existing pattern: create a `Reach` and `Scenario` inline in each test function (or use a module-level fixture for shared setup, as in `test_scenarios.py`).

## Documentation

Docs source is in `docs/source/` (reStructuredText + Sphinx). Tutorials are Jupyter notebooks in `tutorial/` and mirrored under `docs/source/tutorials/notebooks/`. See [docs/source/user-guide.rst](../docs/source/user-guide.rst) for the full API narrative.
