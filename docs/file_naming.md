# Notebook index

The analysis notebooks use two-digit identifiers and lowercase `snake_case` filenames. The numbering groups notebooks by their role in the study; dependencies between analysis stages are described in `reproducibility.md`.

| Notebook | Description |
| --- | --- |
| `00_shared_config.ipynb` | Shared configuration, input definitions and analysis contracts |
| `01_cohort_and_nmr_derivation.ipynb` | Cohort construction, data partitioning and NMR data preparation |
| `02_incremental_value.ipynb` | Incremental predictive value analyses |
| `03_nmr_representation_comparison.ipynb` | Comparison of NMR biomarker representations |
| `04_survival_model_comparison.ipynb` | Comparison of survival prediction models |
| `05_calibration_dca_and_joint_risk.ipynb` | Calibration, decision-curve analysis and joint risk stratification |
| `06_metabolite_associations.ipynb` | Metabolite association and interaction analyses |
| `07_missingness_audit.ipynb` | Missingness assessment and cohort-retention analysis |
| `08_coverage90_sensitivity.ipynb` | Coverage-based sensitivity analysis |
| `09_metabolite_interaction_confirmation.ipynb` | Confirmation analyses for PRS–metabolite interactions |
| `10_glyca_confirmation.ipynb` | GlycA main-effect confirmation analysis |
| `11_nmr_score249_refit.ipynb` | Derivation of the Full249-based NMR score |
| `12_nmr_score106_derivation.ipynb` | Derivation of the Raw106-based NMR score |
| `13_paired_nmr_score_comparison.ipynb` | Paired comparison of NMR score representations |
| `14_ethnicity_cox_evaluation.ipynb` | Cox-model evaluation across population subgroups |
| `15_ethnicity_all_model_evaluation.ipynb` | Evaluation of prediction models across population subgroups |
| `16_manuscript_figures.ipynb` | Generation of the main manuscript figure collection |
| `17_score_and_ethnicity_summary_figures.ipynb` | Summary figures for score and subgroup analyses |
| `18_nmr_and_ethnicity_figures.ipynb` | NMR representation and subgroup figure collection |

Input filenames, output filenames, run identifiers, environment variables, model keys and cache tags are defined by the analysis code. Keep these settings consistent when running dependent notebooks.
