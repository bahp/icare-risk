# 📖 Guide: Pipeline Execution

Now that your YAML configuration files are set up, it is time to run the pipeline.

The architecture is split into **three distinct steps** to ensure modularity and reproducibility:

1. **Data Generation (`01_generate_data_v2.py`)** — Reads the blueprints and builds the raw, messy hospital database.
2. **Feature Engineering (`02_build_features_icare.py`)** — Cleans the messy database, pivots the time-series data, and calculates advanced clinical scores.
3. **Model Evaluation (`03_evaluate_scores_v2.py`)** — Tests the accuracy of those clinical scores across different patient subgroups and timeframes.

---

## 1. The Golden Rule of Execution

> ⚠️ **Important:** Because the pipeline relies on shared resources, such as the `src/` directory and the `config/` files, you must **always run these scripts from the root directory of your project**.

Furthermore, you must run them as **Python modules using the `-m` flag**.

### ⚙️ Why the `-m` Flag?

If you run:

```bash
python scripts/01_generate_data_v2.py
```

Python assumes that the `scripts/` folder is the center of the universe.

This can result in an error such as:

```text
ModuleNotFoundError: No module named 'src'
```

Instead, run:

```bash
python -m scripts.01_generate_data_v2
```

from the **project root**.

This tells Python:

> "Hey, load the whole project into memory first, then execute this specific script."

This ensures that your relative paths and imports work correctly.

---

# 2. Step 1: Running the Data Generator

The first script acts as the **hospital simulator**.

It reads:

```text
data_config_v2.yaml
```

and uses it to:

* Generate a cohort of synthetic patients
* Build interconnected relational tables
* Create tables such as Episodes, Pharmacy, and Vitals

### The Command

Open your terminal, make sure you are in the **project root folder**, and run:

```bash
python -m scripts.01_generate_data
```

### What to Expect

When the script runs, it prints a log to your terminal showing exactly which tables it is building.

It automatically creates a **new timestamped directory**, so previous datasets are not overwritten.

Example output:

```text
==================================================
🧬 [Step 1] Dynamic Synthetic Data Generation
==================================================

📂 TARGET SAVE DIRECTORY: /your/path/data/synthetic/2026-03-25_103015

Generating configured tables for 100 patients...
 -> Building ICARE_EPISODES_ANON [relational]...
 -> Building ICARE_VITAL_SIGNS_ANON [eav_timeseries]...
 -> Building ICARE_PHARMACY_PRESCRIBING_ANON [relational]...

✅ Success! All data saved dynamically to: data/synthetic/2026-03-25_103015
```

### Output

The generated raw data is stored under:

```text
data/synthetic/<timestamp>/
```

For example:

```text
data/synthetic/2026-03-25_103015/
```

---

# 3. Step 2: Running the Feature Builder

Once the raw data exists, it is time to engineer the features.

This script automatically finds the **most recently created folder** in:

```text
data/synthetic/
```

It then:

1. Loads the raw CSV files
2. Pivots the long EAV (Entity-Attribute-Value) tables into wide time-series formats
3. Processes the data according to `feature_config.yaml`
4. Runs the Feature Pipeline
5. Generates phenotypes and clinical scores
6. Saves the engineered dataset

### The Command

From the project root folder, run:

```bash
python -m scripts.02_build_features_icare
```

### What to Expect

The script performs several processing stages.

First, it pivots the long EAV tables into wide time-series formats.

For example:

```text
Vitals EAV
    │
    ├── Heart Rate
    ├── Temperature
    ├── Blood Pressure
    └── ...
          │
          ▼
Wide Time-Series Data
```

It then triggers the Feature Pipeline.

Example output:

```text
==================================================
⚙️  [Step 2] Feature Engineering: ICARE Edition
==================================================

📂 Loading data from run: 2026-03-25_103015
✅ Raw ICARE tables loaded successfully.
  -> Vitals pivoted. Shape: (1440, 6)
  -> Labs pivoted. Shape: (350, 4)
  -> Combined Time-Series Shape: (1790, 10)
  -> Static (Episodes) rows: 100

🚀 Starting Feature Pipeline...
Processing base features...
  🔍 Computing phenotype: is_aki...
     ✅ is_aki added. Unique values: [0 1]
  🔍 Computing score: charlson_score...
     ✅ charlson_score added. Unique values: [3 unique values]

✨ SUCCESS! Engineered 45 features across 1790 rows.
📍 Saved to: data/processed/2026-03-25_103015/features_engineered_icare.csv
```

### Output

The engineered dataset is saved under:

```text
data/processed/<timestamp>/
```

For example:

```text
data/processed/2026-03-25_103015/features_engineered_icare.csv
```

The timestamp matches the corresponding synthetic data run.

---

# 4. Step 3: Running the Model Evaluator

Finally, we need to test whether the clinical scores you generated actually work.

This script is driven entirely by:

```text
eval_config.yaml
```

It can slice your dataset by:

### Time

For example:

* Admission
* 24 hours
* 48 hours
* Other configured milestones

### Patient Subgroups

For example:

* All patients
* Elderly diabetics
* Other configured cohorts

This allows you to generate a comprehensive performance report across different patient groups and timeframes.

### The Command

From the project root, run:

```bash
python -m scripts.03_evaluate_scores_v2
```

### What to Expect

The script acts as a **"Dumb Dispatcher"**, executing whatever experiments you have defined in the YAML configuration.

It loops through your configured:

* Cohorts
* Timeframes
* Clinical scores

and evaluates each combination.

Example output:

```text
==================================================
📊 [Step 3] Configuration-Driven Model Evaluation
==================================================

🎯 Target Label: 'ground_truth'
🧪 Evaluating Scores: mews_score, pitt_score, increment_esbl_score

🏥 === RUNNING COHORT: ALL PATIENTS (100 Patients) ===
  -> [Admission] Evaluating first clinical record...
  -> [Milestone 24h] Evaluating records near hour 24...

🏥 === RUNNING COHORT: ELDERLY DIABETICS (24 Patients) ===
  -> [Admission] Evaluating first clinical record...
  -> [Milestone 24h] Evaluating records near hour 24...

  📈 Generating AUROC Over Time (up to 168h)...

==================================================
🏆 MASTER EXPERIMENT SUMMARY
==================================================
Strategy                         Score_Name      AUROC  AUPRC
All Patients - Admission         mews_score      0.82   0.65
All Patients - Admission         pitt_score      0.88   0.71
Elderly Diabetics - Admission    mews_score      0.75   0.55

✅ All metrics and plots saved to: outputs
```

### Evaluation Outputs

The evaluator generates:

* Performance metrics
* Stratified results
* ROC curves
* Temporal AUROC plots
* A master summary

All outputs are saved under:

```text
outputs/
```

---

# 5. Understanding the Folder Architecture

After running all three scripts successfully, your project root will be organized into separate configuration, data, processing, and evaluation directories.

The pipeline uses matching timestamps for the synthetic and processed data, while evaluation results are stored separately under `outputs/`.

```text
📦 project_root/
┣ 📂 config/
┃ ┣ 📜 data_config_v2.yaml                 <-- Recipe for Step 1
┃ ┣ 📜 feature_config.yaml                 <-- Recipe for Step 2
┃ ┗ 📜 eval_config.yaml                    <-- Recipe for Step 3
┃
┣ 📂 data/
┃ ┣ 📂 synthetic/
┃ ┃ ┗ 📂 2026-03-25_103015/                <-- Output of Step 1 (Raw)
┃ ┃   ┣ 📜 icare_episodes_anon.csv
┃ ┃   ┗ 📜 icare_vital_signs_anon.csv
┃ │
┃ ┗ 📂 processed/
┃   ┗ 📂 2026-03-25_103015/                <-- Output of Step 2 (Engineered)
┃     ┗ 📜 features_engineered_icare.csv
┃
┣ 📂 outputs/                               <-- Output of Step 3 (Evaluation)
┃ ┣ 📂 metrics/
┃ ┃ ┣ 📜 master_summary.csv                 <-- Master performance table
┃ ┃ ┣ 📜 all_patients_admission_metrics.csv
┃ ┃ ┗ 📜 elderly_diabetics_admission_metrics.csv
┃ │
┃ ┗ 📂 plots/
┃   ┣ 🖼️ all_patients_admission_roc.png
┃   ┗ 🖼️ temporal_auroc.png                <-- Longitudinal AUROC plot
│
┣ 📂 scripts/
┃ ┣ 📜 01_generate_data_v2.py
┃ ┣ 📜 02_build_features_icare.py
┃ ┗ 📜 03_evaluate_scores_v2.py
│
┗ 📂 src/
```

---

# 6. Complete Pipeline Flow

The entire pipeline follows a simple sequential structure:

```text
┌──────────────────────────────┐
│ 1. DATA GENERATION           │
│                              │
│ data_config_v2.yaml          │
│          │                   │
│          ▼                   │
│ Synthetic Hospital Data      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. FEATURE ENGINEERING       │
│                              │
│ feature_config.yaml          │
│          │                   │
│          ▼                   │
│ Clean + Pivot + Features     │
│ Phenotypes + Clinical Scores │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. MODEL EVALUATION          │
│                              │
│ eval_config.yaml             │
│          │                   │
│          ▼                   │
│ Cohorts + Timeframes         │
│ AUROC + AUPRC + Plots        │
└──────────────┬───────────────┘
               │
               ▼
       📊 Evaluation Outputs
```

---

## Quick Reference

| Step                       | Purpose                             | Configuration         | Command                                     | Main Output                   |
| -------------------------- | ----------------------------------- | --------------------- | ------------------------------------------- | ----------------------------- |
| **1. Data Generation**     | Generate synthetic hospital data    | `data_config_v2.yaml` | `python -m scripts.01_generate_data`        | `data/synthetic/<timestamp>/` |
| **2. Feature Engineering** | Clean, pivot, and engineer features | `feature_config.yaml` | `python -m scripts.02_build_features_icare` | `data/processed/<timestamp>/` |
| **3. Model Evaluation**    | Evaluate clinical scores            | `eval_config.yaml`    | `python -m scripts.03_evaluate_scores_v2`   | `outputs/`                    |

### ⚠️ Remember

Always:

1. Open your terminal in the **project root**.
2. Use Python's **module syntax** with `-m`.
3. Run the steps in order.
4. Keep the generated timestamps aligned between synthetic and processed data.
5. Check `outputs/` for the final metrics and plots.

> 🎉 **You're Ready for Publication!**
>
> The `master_summary.csv` contains the stratified metrics, while the `plots/` directory holds the generated curves ready to be used in a clinical research paper or dashboard.
