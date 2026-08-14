# Pipeline Architecture & Orchestration



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


??? abstract "📁 View Full Project Structure"

    ```text
    dm-esbl/
    ├── assets/                     # Static clinical knowledge bases and maps
    │   └── lookups
    │      └── ICD10.csv              # File with ICD10 codes and descriptions
    │      └── res195.csv             # File with res195 codes and descriptions
    ├── data/                       # Data storage layer (Git ignored)
    │   ├── external/                 # Raw external data
    │   ├── synthetic/                # Raw synthetic ddata
    │   └── processed/                # Processed data
    ├── docs/                       # Documentation
    │   ├── stylesheets/               # stylesheets
    │   ├── api/                       # Documentation API pages
    │   ├── xx-<filename>.md           # Documentation page xx
    │   └── index.md                   # Documentation index
    ├── local/                      # Local development (more on this later)
    ├── notebooks/                  # Notebooks
    ├── outputs/                    # Metrics, plots, and experiment artifacts
    │   └── <run_timestamp>/            # Run <timestamp>
    │       ├── metrics/                    # Performance reports
    │       └── plots/                      # ROC, PR, longitudinal analyses
    ├── src/
    │   └── icare_risk/
    │       ├── __init__.py
    │       ├── config/                         # Blueprints controlling pipeline execution (YAML)
    │       │   ├── code_search.yaml                # Keywords to search for clinical concepts
    │       │   ├── data_config.yaml                # Synthetic data generation schemas[cite: 1]
    │       │   ├── eval_config.yaml                # Cohort stratification and evaluation rules
    │       │   ├── feature_config.yaml             # Feature windowing and phenotype definitions
    │       │   └── threshold_config.yaml           # Literature-recommended validation cutoffs
    │       ├── scripts/                        # Step-by-step pipeline execution entry points[cite: 2]
    │       │   ├── __init__.py
    │       │   ├── a_generate_data.py              # Step 1: Generate synthetic iCARE cohort[cite: 2]
    │       │   ├── b_build_features_icare.py       # Step 2: Build clinical features and phenotypes[cite: 2]
    │       │   ├── c_evaluate_scores.py            # Step 3: Compute clinical scores[cite: 2]
    │       │   ├── d_evaluate_thresholds.py        # Step 4: Evaluate stewardship thresholds[cite: 2]
    │       │   ├── e_validate_scores.py            # Step 5: Run clinical audit and validation[cite: 2]
    │       │   ├── f_find_clinical_codes.py        # Utility: Search clinical concepts and codes[cite: 2]
    │       │   └── g_sandbox.py                    # Development sandbox runner[cite: 2]
    │       ├── features.py                     # Feature engineering pipeline engine
    │       ├── generators.py                   # Cohort simulation and data generation logic
    │       ├── metrics.py                      # Statistical evaluation and performance engines
    │       ├── phenotypes.py                   # Clinical rule and phenotype extractors[cite: 1]
    │       ├── scores.py                       # Clinical score calculation logic[cite: 1]
    │       └── utils.py                        # Shared helper utilities and I/O handlers
    └── tests/
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

!!! note "CLI Target Overrides"

    Running `make` or `.\make.bat` without a target automatically displays the built-in help menu showing all available automation commands.

| Unix | Windows | Description |
|------|----------|-------------|
| `make all` | `.\make.bat all` | Runs the complete pipeline (Steps 1–5). |
| `make generate` | `.\make.bat generate` | Step 1: Generate synthetic clinical cohorts. |
| `make features` | `.\make.bat features` | Step 2: Build features, phenotypes, and scores. |
| `make evaluate` | `.\make.bat evaluate` | Step 3: Evaluate predictive performance metrics. |
| `make thresholds` | `.\make.bat thresholds` | Step 4: Perform literature-based threshold validation. |
| `make validate` | `.\make.bat validate` | Step 5: Validate gold-standard patient cases. |
| `make search` | `.\make.bat search` | Utility: Search clinical coding classifications. |
| `make test` | `.\make.bat test` | Utility: Execute the Pytest regression suite. |
| `make clean` | `.\make.bat clean` | Utility: Remove generated reports and artifacts. |

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

### C. Azure ML Environment

See (ref)

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