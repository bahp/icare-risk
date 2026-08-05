# Pipeline Architecture & Orchestration

<div class="grid cards" markdown>

- 🗄️ **Data Generation**

    ---

    Learn how the synthetic clinical data factory builds patients, vitals, and pharmacy records.

    [Read the Data Generation Guide →](01-data-generation.md)

- 🧠 **Feature Pipeline**

    ---

    Discover how raw data is translated into predictive phenotypes and rolling windows.

    [Read the Feature Pipeline Guide →](02-feature-building.md)

- 📊 **Pipeline Evaluation**

    ---

    View evaluation metrics, performance curves, and cohort analyses.

    [Read the Evaluation Guide →](04-pipeline-evaluation.md)

</div>

---

This document outlines the consolidated folder structure of the DM-ESBL clinical validation pipeline and explains how to utilize the unified cross-platform automation suite to execute workflows seamlessly.

## Environment Initialization

Before running any pipeline steps, initialize the isolated container infrastructure. This ensures all dependencies, libraries, and Python environments match production standards exactly, eliminating cross-platform configuration errors.

```bash
# Clone, build, and launch the isolated pipeline container network in detached mode
docker-compose up -d --build
```

Once the container status is active, you can utilize the orchestration suite below to run commands directly inside the containerized system environment.

---

## 1. Folder Structure

The project strictly separates configuration blueprints, core modular logic, execution dispatchers, and generated artifacts to ensure complete auditability, reproducibility, and isolation of experimental setups.

```text
dm-esbl/
├── assets/                     # Static clinical knowledge bases and maps
│   └── res195-comorbidity-cci-gold.csv
├── config/                     # Blueprints controlling pipeline execution (YAML)
│   ├── code_search.yaml        # Keywords to search for interesting codes
│   ├── data_config.yaml        # Synthetic data generation schemas
│   ├── feature_config.yaml     # Feature windowing and phenotype definitions
│   ├── eval_config.yaml        # Cohort stratification and evaluation rules
│   └── threshold_config.yaml   # Literature-recommended validation cutoffs
├── data/                       # Data storage layer (Git ignored)
│   ├── synthetic/              # Raw synthetic cohorts (Step 1 output)
│   └── processed/              # Patient-level analytics (Step 2 output)
├── outputs/                    # Metrics, plots, and experiment artifacts
│   └── <run_timestamp>/
│       ├── metrics/            # Performance reports
│       └── plots/              # ROC, PR, longitudinal analyses
├── reports/                    # Validation logs and audit trails
│   ├── score_validation.log
│   └── code_search_results.txt
├── scripts/                    # Pipeline entry points
│   ├── 01_generate_data_v2.py
│   ├── 02_build_features_icare.py
│   ├── 03_evaluate_scores_v2.py
│   ├── 04_evaluate_thresholds.py
│   ├── 05_validate_scores.py
│   └── 06_find_clinical_codes.py
├── src/                        # Core production logic
│   ├── generators.py           # Cohort simulation algorithms
│   ├── features.py             # Feature engineering pipeline
│   ├── phenotypes.py           # Clinical rule extraction
│   ├── scores.py               # Clinical score calculations
│   ├── metrics.py              # Statistical evaluation engines
│   └── utils.py                # Shared helper utilities
├── tests/                      # Regression testing suite
│   ├── cases.csv
│   ├── test_phenotypes.py
│   └── test_scores.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── make.bat
```

---

## 2. The Unified Orchestration Suite

The project provides dual managers (`Makefile` and `make.bat`) that expose identical commands across Linux, macOS, and Windows. This abstraction allows the same workflow regardless of host operating system.

!!! note "📋 CLI Target Overrides"

    Running `make` or `.\make.bat` without a target automatically displays the built-in help menu showing all available automation commands.

| Unix | Windows | Description |
|------|----------|-------------|
| `make all` | `.\make.bat all` | Runs the complete pipeline (Steps 1–5). |
| `make generate` | `.\make.bat generate` | Step 1: Generate synthetic clinical cohorts. |
| `make features` | `.\make.bat features` | Step 2: Build features, rolling windows, phenotypes, and scores. |
| `make evaluate` | `.\make.bat evaluate` | Step 3: Evaluate predictive performance metrics. |
| `make thresholds` | `.\make.bat thresholds` | Step 4: Perform literature-based threshold validation. |
| `make validate` | `.\make.bat validate` | Step 5: Validate gold-standard patient cases. |
| `make search` | `.\make.bat search` | Utility: Search clinical coding classifications. |
| `make test` | `.\make.bat test` | Utility: Execute the Pytest regression suite. |
| `make clean` | `.\make.bat clean` | Utility: Remove generated reports and temporary artifacts. |

---

## 3. Runtime Environments

The orchestration layer supports both containerized and native execution.

### A. Containerized Environment (Default)

By default, every automation target executes inside the Docker container to ensure dependency isolation and reproducibility.

```bash
# Initial setup
docker-compose up -d --build

# Execute pipeline stages
make generate
make features
```

### B. Native Host Environment

To execute directly on the local machine, append the `local` keyword.

```bash
# Linux / macOS
make generate local

# Windows
.\make.bat features local
```

---

## 4. Parameterized Argument Injection

Pipeline scripts accept runtime configuration overrides without modifying the default YAML configuration files.

### Unix (using `ARGS`)

```bash
make generate ARGS="--config experimental_data_blueprint.yaml"

make evaluate local ARGS="--eval-config strict_sepsis_matrix.yaml"
```

### Windows

```bash
.\make.bat generate --config experimental_data_blueprint.yaml

.\make.bat evaluate local --eval-config strict_sepsis_matrix.yaml
```