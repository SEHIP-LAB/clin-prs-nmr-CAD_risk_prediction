# Python Analysis Modules

The `sources/` directory contains two Python modules used by the analysis notebooks.

| Module | Used by | Purpose |
| --- | --- | --- |
| `metrics.py` | Notebooks 13–15 | Survival prediction metrics, calibration, decision-curve analysis and bootstrap procedures |
| `score106.py` | Notebook 12 | Training and evaluation routines for the Raw106-derived metabolomic score |

## `metrics.py`

The module provides the statistical functions used in the paired score comparison and ethnicity subgroup evaluations. Notebooks 13–15 load it dynamically as part of their analysis workflows.

## `score106.py`

Notebook 12 executes this module after loading the shared configuration and the ordered Raw106 feature manifest. The module uses objects prepared in the notebook execution context and is intended to be run through notebook 12 rather than as a standalone Python script.

The required shared objects are listed in `REQUIRED_SHARED_GLOBALS`. If the execution context is incomplete, the module raises an explanatory error before model training begins.

## Running the analyses

Run the relevant notebook from the `notebooks/` directory after completing its upstream data-preparation steps. From the repository root, `python scripts/preflight.py` can be used to check the analysis prerequisites.

See [Reproducibility](../docs/reproducibility.md) for the workflow and [Software validation](../docs/validation_report.md) for the scope of the automated tests.
