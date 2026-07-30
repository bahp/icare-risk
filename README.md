# 🏥 Clinical Scoring Evaluation Pipeline

A modular, configuration-driven pipeline to generate synthetic patient 
data, engineer clinical features, and audit predictive scoring systems 
(e.g., INCREMENT-ESBL, Gavaghan, Jones, Holmgren).

---

## 🐳 Environment Setup & Execution

By default, the `make` pipeline routes all commands into a Docker 
container. This ensures the environment is identical across Windows, 
Mac, and Linux, preventing library and path conflicts.

## 1. Standard Docker Workflow (Default)

### Build and start the container in the background

```bash
docker-compose up -d --build
```

### How to Run Commands

The Clinical Pipeline Manager is a unified wrapper that lets you 
run complex data generation and scoring workflows effortlessly. It 
ensures your code runs the exact same way on Mac, Linux, and Windows.

Depending on your operating system, use the corresponding wrapper:
* **Mac/Linux:** Use `make <command>`
* **Windows:** Use `./make.bat <command>` (or double-click `make.bat` for an interactive menu)

**Core Pipeline Commands:**

| Command | Description | Allowed Script Parameters (`ARGS="..."`)                                                       |
|---|---|------------------------------------------------------------------------------------------------|
| `make generate` | Step 1: Create synthetic patients | `--config` *(def: `data_config.yaml`)*                                                         |
| `make features` | Step 2: Clean data & engineer phenotypes | `--data-config` *(def: `data_config.yaml`)*<br>`--feature-config` *(def: `feature_config.yaml`)* |
| `make evaluate` | Step 3: Run AI/ML scoring metrics | `--eval-config` *(def: `eval_config.yaml`)*<br>`--feature-config` *(def: `feature_config.yaml`)* |
| `make thresholds` | Step 4: Check clinical safety | Currently hardcoded: <br> `feature_config.yaml` & `threshold_config.yaml`                      |
| `make validate` | Step 5: Audit specific patient cases | Currently uses hardcoded: <br> `cases.csv`                                       |
| `make all` | Runs Steps 1 through 5 sequentially | Same `ARGS` to ALL scripts 

### Dynamic Configurations 

Because the pipeline is fully dynamic, you do not need to rewrite Python 
code to test new theories. You can pass custom YAML configuration files 
directly through the manager.

#### Example 1: Generating Data with an Experimental Config

```bash
make generate ARGS="--config experimental_data.yaml"
```


### Stop the container when finished

```bash
docker-compose down
```

## 2. Running Locally (Without Docker)

If you have installed the requirements natively:

```bash
pip install -r requirements.txt
```

and wish to run the pipeline directly on your machine's CPU, simply 
append `local` to any command.

This works identically on Mac, Linux, and Windows.

```bash
make generate local
make evaluate local
```

Or with dynamic configuration

```bash
make generate local ARGS="--config experimental_data.yaml"
```

### 💡 Pro Tip for Windows Users

You do not need the `ARGS=""` syntax on Windows. You can chain the arguments directly in your terminal.

```bash
make.bat evaluate local --eval-config eval_strict.yaml
```

---

# 🚀 1. Quick Start: The Pipeline

You can orchestrate the pipeline using the provided `Makefile` (Mac/Linux) or `make.bat` (Windows).

> **Note:** All examples below show the default local commands. If using Docker, remember to add your Docker flag.

## 1.1 Generate Synthetic Data

Generates a synthetic patient cohort with demographics, comorbidities, and time-series vitals/labs based on your YAML configurations.

```bash
make generate
```

To use a custom config:

```bash
make generate ARGS="--config my_data.yaml"
```

---
## 📁 Key Directories

```
config/                    → YAML configuration files controlling:
   ├── data_config.yaml      → Data generation probabilities
   ├── feature_config.yaml   → Feature engineering rules
   └── icd_config.yaml       → ICD coding mappings

src/                       → Core logic modules
   ├── scores.py             → Pure mathematical clinical scoring functions
   ├── features.py           → Pipeline transformation logic
   ├── generators.py         → Synthetic data creation utilities
   ├── metrics.py            → Metrics to evaluate performance
   ├── phenotypes.py         → Special features, phenotypes from findings
   └── utils.py              → Helper functions

scripts/                   → Executable pipeline orchestration scripts
   ├── 01_generate_data.py   → Step 1
   ├── 02_build_features.py  → Step 2
   ├── 03_evaluate_scores.py → Step 3
   └── 05_validate_scores.py → Tool to validate/debug scores

tests/                     → Validation and regression testing
   ├── cases.csv             → "Ground Truth" patient profiles
   └── test_scores.py        → Unit tests validating scoring accuracy
```




## 1. Quick Start: The pipeline

Run all commands from the root of the project directory using the -m (module) flag.

#### 1.1 Generate Synthetic Data

Generates a synthetic patient cohort with demographics, comorbidities, 
and time-series vitals/labs based on your YAML configurations.

```bash
python -m scripts.01_generate_data
```

  - **Reads:** `config/data_config.yaml` and `config/icd_config.yaml` 
  - **Outputs:** `data/synthetic/<timestamp>/`

#### 1.2. Build Features & Compute Scores

Merges raw data, handles missing value imputation, calculates time-windowed 
features, and runs the clinical scoring algorithms across the generated cohort.

```bash
python -m scripts.02_build_features
```

  - **Reads:** `config/feature_config.yaml`
  - **Outputs:** `data/processed/<timestamp>/features_engineered.csv`


#### 1.3. Validate Scores (Audit Report)

Runs specific edge-case patients through the scoring algorithms to generate 
a point-by-point clinical audit trail, ensuring mathematical accuracy. Remember
to Check the `reports/score_validation.log` file to see exactly how points were
awarded for each patient.

```bash
python -m scripts.05_validate_scores
```

  - **Reads:** `tests/cases.csv`
  - **Outputs:**  `reports/score_validation.log`

#### 1.4. Evaluate Model Performance (PENDING)

Evaluates how well the computed scores predict the target label (e.g., `sepsis_case`).
It calculates AUROC, AUPRC, Sensitivity, and Specificity at the mathematically optimal
threshold.

```bash
python -m scripts/c_evaluate_scores.py
```

  - **Outputs:** `outputs/metrics/` (CSVs) and `outputs/plots/` (ROC/PR curves)




## 2. How to Configure the Pipeline

#### 2.1 Configuring Data Generation (`config/data_config.yaml`)

Want to add a new lab test or demographic feature? Just add it to 
this file. The pipeline will automatically generate it, bound it 
within realistic human ranges, and apply missing data masks.

##### Example: Adding a lab test (e.g., Creatinine)

```yaml
ts_config:
  creatinine:
    type: 'float'
    unit: 'mg/dL'
    prob: 0.20          # 80% chance this value is missing (only drawn 20% of the time)
    range: [0.3, 15.0]  # Impossible to generate values outside this range
```

##### Example: Adding a demographic

```yaml
demographics_schema:
  age:
    type: 'int'
    range: [18, 95]
  sex:
    type: 'enumerated'
    values: ['M', 'F']
```

##### Example: Adding a condition

You can now control the "sickness" or "resistance" level of your synthetic
population by editing the `conditions` in `config/data_config.yaml`.

* **To simulate a high-resistance setting:** Increase the probability of `prev_3gcr_culture`.
* **To simulate an elderly cohort:** Increase the probability of `CKD` and `malignancy`.

```bash
conditions:
  hosp_abroad_12m: 0.15  # Adds a binary 1/0 column for hospital abroad.
  mech_vent: 0.3         # Adds a binary 1/0 column for mechanical ventilation
  cardiac_arrest: 0.2    # Adds a binary 1/0 column for cardiac arrest
  prev_3gcr_culture: 0.1
```


#### 2.2 Configuring Features & Scores (`config/feature_config.yaml`)

This file tells the pipeline exactly how to handle the data generated in Step 1.

##### 2.2.1 Base Features (Imputation & Rolling Windows)

Define how to handle missing data and what statistical windows to compute.

```yaml
base_features:
  hr:
    impute: 'ffill'           # Forward-fill missing heart rates
    rolling:
      windows: ['24h', '7D']  # Calculate 24-hour and 7-day stats
      aggs: ['mean', 'max']   # Specifically, the mean and max
```

##### 2.2.2 Computed Features (Simple Math)

You can write simple math directly as strings. The pipeline will compile and calculate it instantly.

```yaml
computed_features:
  shock_index: "hr / sbp"
  derived_diabetes: "(med_insulin_given == 1) or (glucose_24h_max > 200)"
```

##### 2.2.3 Custom Scores (Complex Clinical Logic)

For heavy clinical scoring (like MEWS or INCREMENT-ESBL), write a Python function in
`src/scores.py`, then map the columns to it here.

```yaml
custom_scores:
  increment_esbl_core:
    module: 'scores'                     # Looks in src/scores.py
    function: 'calculate_increment_esbl' # Calls this specific function
    kwargs:                              # Passes these dataset columns as arguments
      age_col: 'age'
      charlson_col: 'charlson_score'
      pitt_col: 'pitt_score'
      inapprop_abx_col: 'inappropriate_abx'
```


## 3. Adding a Brand New Clinical Score

Let’s say your attending physician asks you to test the qSOFA score which is calculated as
follows: Respiratory Rate ≥ 22, Systolic BP ≤ 100, Altered Mental Status.

##### Step 1: Write the logic in `src/scores.py`

```python
# src/scores.py
import pandas as pd
import numpy as np

def calculate_qsofa(df, rr_col='rr', sbp_col='sbp', ams_col='altered_mental_status'):
    score = pd.Series(0, index=df.index)
    
    if rr_col in df.columns:
        score += np.where(df[rr_col] >= 22, 1, 0)
    if sbp_col in df.columns:
        score += np.where(df[sbp_col] <= 100, 1, 0)
    if ams_col in df.columns:
        score += np.where(df[ams_col] == 1, 1, 0)
        
    return score
```


##### Step 2: Tell the YAML to use it in `config/feature_config.yaml`

```yaml
custom_scores:
  qsofa_score:
    module: 'src.scores'
    function: 'calculate_qsofa'
    kwargs:
      rr_col: 'rr'
      sbp_col: 'sbp'
      ams_col: 'gcs_less_than_15'  # Assuming you engineered a GCS flag!
```

Note that if you do not want to engineer the `gcs_less_than_15`, you could compute just
the `gcs_score` and then inside the calculate_qsofa function check whether the value is
less than 15.


##### Step 3: Run validation, build features and evaluate

```bash
python -m scripts.05_validate_scores
python -m scripts.02_build_features
python -m scripts.03_evaluate_scores
```

You will instantly see your new `qsofa_Score` evaluated against all the existing
scores in your `outputs/` folder!


## 4. Testing and Validation

We use `pytest` to ensure clinical scores (Charlson, Pitt, INCREMENT-ESBL)
match the literature exactly and remain stable as the codebase evolves.

##### 4.1 Running Tests

Run all tests from the project root directory:

```bash
py -m pytest tests
```

`pytest` will automatically:

- Discover all `test_*.py` files inside the `tests/` folder  
- Execute each test function  
- Provide a clean summary of passed/failed tests  
- Show detailed tracebacks if failures occur  

##### 4.2 Running Specific Tests (Advanced)

If you are debugging a specific score or a single patient case, you don't need to run the
entire test suite. You can use the `-k` (keyword) flag to isolate exactly what you want to test:

```bash
# Run ONLY the CSV-based scoring tests (all patients)
python -m pytest tests/test_scores.py -k "test_all_scores_from_csv"             

# Run ONLY the CSV test for a SPECIFIC patient (e.g., Row index 2)
python -m pytest tests/test_scores.py -k "test_all_scores_from_csv and index2"
```



