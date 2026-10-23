# icare_risk analysis workflow

This project evaluates ESBL risk at admission using existing clinical scores 
and machine-learning models. Notebooks are orchestration layers; reusable 
code lives under `icare_risk/`.

## Run order

Run notebooks sequentially from a clean kernel. Each notebook reads 
configuration from `analysis.yaml`, reads outputs from earlier stages, 
and writes its own outputs. Do not run later notebooks before their 
required inputs exist.

```text
00_data_audit.ipynb
    └── Audit the admission-level input file; check schema, duplicates, missingness, outcomes, and dates.

01_cohort_and_temporal_split.ipynb
    └── Create the one-row-per-admission analysis cohort and assign train/validation/test periods.

02_clinical_scores.ipynb
    └── Validate or calculate score variables and save the score-enriched dataset.

03_train_models.ipynb
    └── Fit configured logistic-regression and XGBoost models on development data; save model bundles.

04_generate_predictions.ipynb
    └── Apply saved models to admissions and append probability columns to the full admission table.

05_model_performance.ipynb
    └── Evaluate test-set discrimination, threshold metrics, calibration, and bootstrap uncertainty.

06_cross_validation.ipynb
    └── Estimate development-set stability using repeated stratified cross-validation.

07_temporal_performance.ipynb
    └── Evaluate performance by calendar period and assess temporal drift.

08_subgroup_analysis.ipynb
    └── Evaluate model performance in configured demographic, clinical, specialty, and compound subgroups.

09_feature_importance.ipynb
    └── Produce logistic-regression coefficients, tree importance, and permutation-importance tables.

10_shap_analysis.ipynb
    └── Generate global and individual model explanations using SHAP or the configured XGBoost fallback.

11_parsimonious_models.ipynb
    └── Build reduced models using development-derived feature rankings and compare them on the temporal test set.

12_sensitivity_analysis.ipynb
    └── Examine model response to one-at-a-time feature changes and predefined counterfactual scenarios.

13_threshold_analysis.ipynb
    └── Select operating thresholds from development predictions and apply them unchanged to the test set.

14_decision_curve_analysis.ipynb
    └── Compare clinical net benefit across probability thresholds against treat-all and treat-none strategies.

15_final_model_comparison.ipynb
    └── Combine saved results into final comparison and ranking tables for reporting.
```

## Important workflow rule

The notebooks are designed to run forward. If configuration, feature sets, split dates, 
or model definitions change, rerun from the first affected notebook onward. In 
particular:

- Change input/cohort/split settings → rerun `00` onward.
- Change score definitions → rerun `02` onward.
- Change feature sets or model parameters → rerun `03` onward.
- Change only evaluation settings → rerun the relevant evaluation notebooks.
- Change subgroup definitions → rerun `08` onward only if prediction columns and source variables are unchanged.

Each stage writes outputs rather than modifying earlier outputs in place. This makes the 
workflow reproducible and avoids needing to move backward and forward during normal 
execution.

## Main data flow

```text
admission-level input
    ↓
processed cohort with split labels
    ↓
score-enriched cohort
    ↓
fitted model bundles
    ↓
full admission table + model probabilities
    ↓
performance, temporal, subgroup, explainability, threshold, and decision analyses
```

## Configuration

Edit `analysis.yaml` before running the notebooks. It controls:

- Input file and read options
- Column names
- Outcome encoding
- Required validation columns
- Temporal split dates
- Clinical-score source columns
- Feature sets
- Models and hyperparameters
- Output locations
- Explainability settings
- Subgroup definitions

The configuration is commented so that a new user can edit it without 
reading the package implementation first.

## Output directories

```text
results/audit/                 data-quality reports
results/scores/                score summaries
results/metrics/               performance, calibration, CV, and threshold results
results/temporal/              performance over calendar time
results/subgroups/             subgroup results
results/feature_importance/    coefficients, permutation importance, and SHAP outputs
results/decision_curve/        net-benefit results and figures
results/final/                 final model comparison tables
models/fitted/                 serialized fitted model bundles
```

## Reproducibility

Use a clean kernel and run all cells in order. Confirm that the target mapping, 
prediction-time feature exclusions, split dates, and random seeds are unchanged 
before comparing results across runs.

The current target is supervised ESBL classification within the defined 
microbiologically labelled cohort. Performance should not automatically be 
interpreted as performance in all suspected-sepsis admissions unless a broader 
cohort is analysed.
