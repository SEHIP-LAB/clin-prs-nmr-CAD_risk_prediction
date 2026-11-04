# CAD Risk Prediction Using Clinical Factors, Polygenic Risk Scores and NMR Metabolomics

This repository contains the analysis code for a UK Biobank study of coronary artery disease (CAD) risk prediction. It includes 19 Jupyter notebooks and two Python modules supporting cohort construction, metabolomic score derivation, survival modelling, prediction-performance evaluation, sensitivity analyses and figure generation.

The analysis integrates clinical predictors, a CAD polygenic risk score (PRS) and NMR metabolomic measurements. Four NMR representations are evaluated:

- A scalar score derived from the Full249 candidate panel (Score249).
- A separately trained scalar score derived from Raw106 measurements (Score106).
- A Spearman-pruned set of NMR metabolites.
- Direct use of Raw106 metabolite predictors.

The repository also includes internal subgroup analyses based on self-reported ethnicity in the UK Biobank.

## Data access

Individual-level UK Biobank data are not included in this repository. Researchers must obtain the appropriate UK Biobank access and prepare authorized data exports that match the input definitions in `notebooks/00_shared_config.ipynb`.

The analysis expects seven input CSV files. Their filenames, order and required fields are defined in notebook 00. The files are linked by the participant identifier `Eid`. Do not upload UK Biobank participant-level data or other restricted research data to this repository.

## Repository structure

| Path | Contents |
| --- | --- |
| `notebooks/` | The 19 analysis and figure-generation notebooks |
| `sources/` | Python modules for survival metrics and Raw106 score derivation |
| `scripts/` | Environment, input and repository checks |
| `config/` | Example environment configuration |
| `docs/` | Notebook index, workflow and validation documentation |
| `tests/` | Automated software tests |

## Installation

The main analysis dependencies are listed in `requirements.txt`. The separate `requirements-validation.txt` file contains dependencies for the automated software tests.

Create an environment and install the analysis dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp config/.env.example .env
```

Set the absolute paths to the authorized input and output directories in `.env`. Keep participant data and generated analysis outputs outside the repository. The example environment file uses Bash syntax; its variables must be exported to the notebook process.

For a Bash shell:

```bash
set -a
source .env
set +a
```

## Input checks and analysis setup

Run the preflight check from the repository root:

```bash
python scripts/preflight.py --data-dir "$CHD_DATA_DIR" --require-ethnicity --require-cuda
```

The preflight check verifies the expected input filenames and CSV headers, along with the requested software and analysis prerequisites. It reads CSV headers; the notebooks perform their own checks on the data and analysis artifacts.

If UK Biobank Field 21000 is provided in a separate export, set `CHD_ETHNICITY_CSV` to that file before running the check.

Notebook 04 uses CUDA by default. For a CPU-only environment, set `CHD_GPU_REQUIRED=0` and omit `--require-cuda` from the preflight command. CPU execution may require substantially more computation time for the larger survival-model analyses.

Categorical NRI is included only when the selected risk-category cut points have a documented basis. Continuous NRI and IDI are evaluated separately.

## Analysis workflow

The notebooks are organized by analysis component. Notebooks 11 and 12 derive the two score representations and dispatch the corresponding modelling workflows in notebooks 02–05.

| Analysis component | Notebooks | Description |
| --- | --- | --- |
| Shared configuration and cohort construction | 00–01 | Input definitions, cohort rules, data partitioning and NMR preparation |
| Score derivation | 11–12 | Full249-based and Raw106-based score derivation |
| Prediction modelling | 02–05 | Incremental predictive value, NMR representation comparison, survival-model comparison, calibration, decision-curve analysis and joint risk stratification |
| Paired score comparison | 13 | Comparison of Score106 and Score249 on matched participants |
| Ethnicity subgroup evaluation | 14–15 | Evaluation of Cox models and prediction models across self-reported ethnicity subgroups |
| Metabolite analyses | 06, 09–10 | Metabolite associations and confirmation analyses |
| Sensitivity analyses | 07–08 | Missingness assessment and Coverage90 sensitivity analysis |
| Figure generation | 16–18 | Main and supplementary figure collections |

For the detailed execution order, run identifiers, dependencies and restart instructions, see [Reproducibility](docs/reproducibility.md). The full notebook list is available in the [Notebook index](docs/file_naming.md).

## Software checks

From the repository root, run:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
```

The automated checks cover notebook structure, Python source compilation, repository references and selected analysis components using synthetic data. These software checks do not replace analyses using the authorized UK Biobank data.

See the [Software validation report](docs/validation_report.md) for the checks and their scope.
