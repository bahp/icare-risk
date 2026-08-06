# Local Feature & Score Development

This guide outlines the workflow for creating, prototyping, debugging, and testing custom clinical features or risk scores locally within your repository before promoting them to the core codebase.

The recommended approach is to keep experimental logic and configuration separate from production code while you develop and validate your feature. Once the feature is stable and validated, it can be promoted into the main pipeline.

---

## 1. Recommended Local Directory Structure

To keep your experimental logic separate from production code, maintain a clean separation for your local overrides:

```text
local/
|── __init__.py
├── config/
├── features/
│   |── __init__.py
│   └── stress.py
├── scores/
│   |── __init__.py
│   └── triage.py
├── tests/
│   |── __init__.py
│   └── test_local_logic.py
└── notebooks/
    └── pipeline_sandbox.ipynb       # Interactive prototyping notebook
```

This allows you to iterate on new features without modifying the production 
phenotype module or the master feature configuration.

---

## 2. Step-by-Step Workflow

### Step 1: Write Your Custom Logic

Create or edit your local module, for example:

```text
local/features/stress.py
```

Write your feature extraction function in this module.

The function should accept the primary DataFrame and `**kwargs`, including any 
contextual DataFrames that may be required by more complex features.

```python
# local/features/stress.py

import pandas as pd
import numpy as np

def derive_local_stress_phenotype(df, hr_col='HR', temp_col='Temp', **kwargs):
    """
    A locally defined phenotype for testing the pipeline.
    Flags patients (1/0) who have both a high heart rate (>100) and fever (>38.0).
    """
    # Start by assuming the condition is absent (0)
    flag = pd.Series(0, index=df.index)

    # Safely evaluate if the required columns exist
    if hr_col in df.columns and temp_col in df.columns:
        hr = pd.to_numeric(df[hr_col], errors='coerce')
        temp = pd.to_numeric(df[temp_col], errors='coerce')

        # Apply the clinical logic
        flag.loc[(hr > 100) & (temp > 38.0)] = 1

    return flag.values
```

Keeping experimental functions means that you can modify and test them without affecting 
the production implementation.

---

### Step 2: Register It in Your Local Config

Add your new feature to:

```text
local/config/local_feature_config.yaml
```

under `custom_features`:

```yaml
# config/local_feature_config.yaml

custom_features:
  local_stress_flag:
    module: "local.features.stress"
    function: "derive_local_stress_phenotype"
    kwargs:
      hr_col: "hr"
      tamp_col: "temp"
```

The configuration connects the feature name used by the pipeline to the Python module 
and function that implement it. You can also use `kwargs` to expose parameters that you 
want to adjust during experimentation without modifying the Python function itself.

---

## 3. Prototyping & Interactive Testing

When actively writing and tweaking a feature that relies on complex relational tables, such 
as microbiology or pharmacy contexts, use an interactive notebook:

```text
notebooks/pipeline_sandbox.ipynb
```

This allows you to test the feature against real data without running the complete feature-building script after every change.

### Load the Data

```python
import pandas as pd

from icare_risk.utils import load_yaml_config, get_latest_data_dir
from icare_risk.features import FeaturePipeline
from icare_risk.scripts.b_build_features_icare import (
    prepare_icare_ts,
    LazyContextDict,
)

# 1. Load data configs and find dataset
data_config = load_yaml_config("data_config.yaml")
latest_dir = get_latest_data_dir(loaded_config=data_config)

# 2. Load base tables
df_episodes = pd.read_csv(
    latest_dir / 'icare_episodes_anon.csv'
).rename(columns={'SUBJECT': 'patient_id'})

df_vitals = pd.read_csv(
    latest_dir / 'icare_vital_signs_anon.csv',
    parse_dates=['OBSERVATION_PERFORMED_DT']
)

df_labs = pd.read_csv(
    latest_dir / 'icare_pathology_blood_anon.csv',
    parse_dates=['SAMPLE_COLLECTED_DT']
)

df_ts = prepare_icare_ts(
    df_vitals,
    df_labs,
    data_config
)
```

### Set Up Lazy Contexts

If the feature requires additional relational tables, configure them using `LazyContextDict`:

```python
contexts = LazyContextDict({
    'problems': latest_dir / 'icare_problems_anon.csv',
    'pharmacy': latest_dir / 'icare_pharmacy_prescribing_anon.csv'
})
```

This allows contextual data to be loaded when required rather than loading every table into memory immediately.

### Test the Feature Interactively

Import your local feature and run it directly against the prepared time-series DataFrame:

```python
from icare_risk.local_phenotypes import derive_local_stress_indicator

df_ts['local_stress_flag'] = derive_local_stress_indicator(
    df_ts,
    hr_col='hr_24h_max',
    threshold=95
)

display(
    df_ts[
        [
            'patient_id',
            'date',
            'hr_24h_max',
            'local_stress_flag'
        ]
    ].head(10)
)
```

This gives you a fast feedback loop for checking:

* Whether the required columns are available.
* Whether the feature produces the expected values.
* Whether missing values are handled correctly.
* Whether thresholds and parameters behave as expected.
* Whether the resulting feature aligns with the intended clinical logic.

---

## 4. Targeted Debugging with the Pipeline (`--only`)

Once your function works as expected in the notebook, verify how it integrates with the full feature pipeline.

Instead of waiting for every feature and score to be generated, use the target isolation flag:

```text
--only
```

This loads the full pipeline infrastructure and connects the context files normally, but computes only the feature or score you specify.

### Running a Single Feature from the Terminal

Execute the build script using your local configuration and target only the new feature:

```bash
python scripts/b_build_features_icare.py \
    --feature-config config/local_feature_config.yaml \
    --only local_stress_flag
```

This is useful for checking that:

* The feature is correctly registered.
* The module can be imported.
* The configuration is valid.
* Required context tables are available.
* The feature integrates correctly with the pipeline.
* The output is generated with the expected structure.

### Running Multiple Target Features Simultaneously

If you want to test your local feature alongside an existing score, such as checking `local_stress_flag` alongside `pitt_score`, comma-separate the targets:

```bash
python scripts/b_build_features_icare.py \
    --feature-config config/local_feature_config.yaml \
    --only local_stress_flag,pitt_score
```

This is particularly useful when validating how your new feature behaves alongside existing features or scores.

---

## 5. Promotion to Production

Once your local feature meets the required validation checks and you are satisfied with its behaviour, promote it into the production codebase.

### Step 1: Move the Implementation

Move the validated function from:

```text
src/icare_risk/local_phenotypes.py
```

into:

```text
src/icare_risk/phenotypes.py
```

For example:

```python
# src/icare_risk/phenotypes.py

def derive_local_stress_indicator(df, **kwargs):
    ...
```

Rename the function if necessary so that it follows the naming conventions used by the production codebase.

### Step 2: Move the Configuration

Transfer the corresponding configuration block from:

```text
config/local_feature_config.yaml
```

into:

```text
config/feature_config.yaml
```

For example:

```yaml
custom_features:
  local_stress_flag:
    module: 'icare_risk.phenotypes'
    function: 'derive_local_stress_indicator'
    kwargs:
      hr_col: 'hr_24h_max'
      threshold: 100
```

### Step 3: Run the Full Pipeline

After promoting the feature, run the complete, unconditioned pipeline build:

```bash
python scripts/b_build_features_icare.py
```

Running the full pipeline is important because a feature that works correctly in isolation may still interact with other features, scores, context tables, or pipeline stages in unexpected ways.

---

## 6. Recommended Development Cycle

The overall workflow can be summarized as:

```text
                    ┌─────────────────────────┐
                    │  Define feature logic   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ local_phenotypes.py     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ local_feature_config    │
                    │         .yaml           │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Test interactively in  │
                    │  pipeline_sandbox.ipynb │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Test with --only       │
                    │  against pipeline       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                       ┌───────────────────┐
                       │  Feature valid?   │
                       └─────────┬─────────┘
                                 │
                         Yes ────┴──── No
                          │             │
                          ▼             └──► Iterate locally
               ┌─────────────────────┐
               │ Promote to          │
               │ production code     │
               └──────────┬──────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │ Run full pipeline   │
               └─────────────────────┘
```

---

## Quick Reference

| Stage                      | Location / Command                         |
| -------------------------- | ------------------------------------------ |
| Experimental phenotype     | `src/icare_risk/local_phenotypes.py`       |
| Experimental configuration | `config/local_feature_config.yaml`         |
| Interactive testing        | `notebooks/pipeline_sandbox.ipynb`         |
| Test one feature           | `--only local_stress_flag`                 |
| Test multiple features     | `--only local_stress_flag,pitt_score`      |
| Production phenotype       | `src/icare_risk/phenotypes.py`             |
| Production configuration   | `config/feature_config.yaml`               |
| Full pipeline              | `python scripts/b_build_features_icare.py` |

---

## Key Principle

> **Develop locally, test in isolation, validate in the pipeline, then promote to production.**

Keeping experimental features isolated makes it easier to iterate quickly while reducing the risk of introducing incomplete or untested logic into the core feature pipeline.
