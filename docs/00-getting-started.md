# Tutorial: Quickstart & Local Development

This tutorial guides you through installing the `icare-risk` package, executing the core clinical 
pipeline, and establishing a local workspace to prototype custom phenotypes.

By the end of this guide, you will have generated a synthetic patient cohort, processed their 
clinical features, and successfully injected your own experimental logic into the pipeline.

!!! tip 
    Think of the `icare-risk` package as a complete clinical factory.

    The command-line tools are the **machines** that manufacture your datasets and analyses, 
    while the configuration files are the **recipe books** that determine exactly what gets built.
    By creating your own local workspace, you can safely experiment with new clinical ideas without 
    modifying the production package.

---

## 1. Installation (Getting Everything Ready)

Install the package into your active Python environment. This automatically registers all 
command-line interfaces (CLIs) so they are available from your terminal.

```bash
pip install icare-risk
```

After installation, the following commands become available:

| Command               | Purpose                                    |
|-----------------------|--------------------------------------------|
| `icare-risk-generate` | Generate synthetic clinical datasets       |
| `icare-risk-features` | Engineer clinical features and risk scores |
| `icare-risk-evaluate` | Evaluate predictive performance            |
| `icare-risk-validate` | Validate the scores with detailed reports  |
| `icare-risk-search`   | Search clinical codes from lookup tables   |

---

## 2. Running the Clinical Pipeline

The complete pipeline consists of three independent stages. Each stage 
consumes the output of the previous one.

### A. Generate Synthetic Clinical Data

Create a realistic synthetic hospital cohort including admissions, laboratory results, vital signs, diagnoses, and medications.

```bash
icare-risk-generate
```

This command produces timestamped synthetic datasets that mimic a real hospital database while containing no patient-identifiable information.

---

### B. Engineer Clinical Features

Transform the raw datasets into machine-learning ready features.

During this stage the pipeline automatically:

- Cleans and validates the raw data
- Pivots longitudinal time-series measurements
- Calculates validated clinical scores
- Applies custom feature definitions
- Produces the final processed dataset

```bash
icare-risk-features
```

---

### C. Evaluate Predictive Performance

Evaluate your generated features using predefined patient cohorts and prediction windows.

The evaluation stage automatically computes metrics such as:

- AUROC
- AUPRC
- Sensitivity
- Specificity
- Calibration statistics

```bash
icare-risk-evaluate
```

---

Upon completion, the pipeline automatically creates a timestamped project directory containing:

- Synthetic datasets
- Processed feature tables
- Evaluation summaries
- Performance reports
- Generated metadata

---

## 3. Creating a Local Workspace (Your Development Sandbox)

Rather than modifying the production package directly, `icare-risk` encourages local development.

Create the following directory structure inside your own project:

```text
my-project/
│
│── data/
│   ├── synthetic/
│   └── processed/
│
└── local/
    ├── config/
    │    ├── local_data_config.yaml
    │    └── local_feature_config.yaml
    ├── features/
    └── scores/
```

Each directory has a specific purpose.

| Directory | Purpose |
|-----------|---------|
| `data/` | Default destination for generated synthetic and processed datasets. |
| `config/` | Stores your customized YAML configuration files. |
| `local/` | Contains experimental Python modules and custom clinical logic. |

Keeping local code separate allows you to safely prototype new ideas without affecting the production implementation.

---

## 4. Building a Custom Clinical Feature (Extending the Pipeline)

One of the main strengths of `icare-risk` is that you can inject your own clinical logic directly into the feature engineering pipeline.

Let's build a simple example.

### A. Define Your Feature

Create a Python module such as:

```text
local/features/stress.py
```

Inside it, define your custom feature function.

```python
import pandas as pd


def derive_local_stress_phenotype(df, hr_col="HR", temp_col="Temp", **kwargs):
    """
    Flags patients presenting with both
    tachycardia and fever.
    """

    flag = pd.Series(0, index=df.index)

    if hr_col in df.columns and temp_col in df.columns:
        hr = pd.to_numeric(df[hr_col], errors="coerce")
        temp = pd.to_numeric(df[temp_col], errors="coerce")

        flag.loc[(hr > 100) & (temp > 38.0)] = 1

    return flag.values
```

This function simply returns:

- **1** if both conditions are present
- **0** otherwise

---

### B. Register the Feature

Next, tell the pipeline where your function lives.

Open:

```text
config/local_feature_config.yaml
```

and register the feature.

```yaml
custom_features:
  local_stress_flag:
    module: "local.features.stress"
    function: "derive_local_stress_phenotype"

    kwargs:
      hr_col: "hr"
      temp_col: "temp"
```

The configuration acts as the bridge between the YAML blueprint and your Python implementation.

---

### C. Execute Only Your Feature

You do not need to rebuild every feature during development.

Instead, execute only your experimental phenotype using the `--only` option.

```bash
icare-risk-features \
    --feature-config config/local_feature_config.yaml \
    --only local_stress_flag
```

This dramatically speeds up development by isolating your custom feature from the rest.
---

## 5. Recommended Development Workflow (Iterate Quickly)

A typical development cycle looks like this:

1. Generate synthetic data.
2. Write a new feature inside `local/features/`.
3. Register it in `local_feature_config.yaml`.
4. Execute only your feature using `--only`.
5. Inspect the output.
6. Refine your logic.
7. Repeat until satisfied.
8. Promote the feature into the production package when ready.

This workflow allows rapid experimentation while keeping the core `icare-risk` codebase clean and stable.

---

## Quick Reference

| Task | Command |
|------|---------|
| Install package | `pip install icare-risk` |
| Generate synthetic data | `icare-risk-generate` |
| Build features | `icare-risk-features` |
| Evaluate models | `icare-risk-evaluate` |
| Run only one feature | `icare-risk-features --only local_stress_flag` |
| Local feature configuration | `config/local_feature_config.yaml` |
| Local Python modules | `local/features/` |

---