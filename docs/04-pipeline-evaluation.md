# Model Evaluation & Stratification

Welcome to **Step 3 of the pipeline**!

You have generated raw hospital data and engineered complex clinical features. Now we must ask the most important question:

> **Are these clinical scores actually accurate?**

The evaluation pipeline is entirely **configuration-driven**. Instead of rewriting Python code every time you want to change an experiment, you define your experiment in `eval_config.yaml`.

> 💡 **The "God Config" Concept:** In the past, changing an experiment required rewriting Python code. Now, this pipeline is 100% configuration-driven.
>
> The Python script acts as a **"Dumb Dispatcher."** It simply reads your instructions from `eval_config.yaml` and builds the exact experiment you ask for.

Your `eval_config.yaml` acts as the **Mission Control panel** for the evaluation pipeline.

It is divided into four main configuration sections:

1. Global Settings
2. Time-Slice Strategies
3. Longitudinal Analyses
4. Subgroup Stratification

---

# 1. Global Settings (The Target)

This section defines the core rules of the experiment.

It answers two fundamental questions:

* **What are we trying to predict?**
* **How are we measuring success?**

```yaml
experiment:
  target_label: 'ground_truth'
  scores_to_evaluate: 'all'
  metrics_to_compute: 'all'
```

### `target_label`

The exact name of the column containing the **true outcome**.

For example:

```text
sepsis_case
```

or:

```text
ground_truth
```

The evaluation engine uses this column as the reference against which the clinical scores are tested.

---

### `scores_to_evaluate`

Defines which clinical scores should be evaluated.

You can use:

```yaml
scores_to_evaluate: 'all'
```

to automatically evaluate every custom score you have built.

Alternatively, you can specify an explicit list:

```yaml
scores_to_evaluate:
  - 'mews_score'
  - 'pitt_score'
```

This is useful when you want to isolate a particular experiment.

---

### `metrics_to_compute`

Defines which performance metrics should be calculated and included in the CSV summaries.

Examples include:

* AUROC
* Sensitivity
* PPV
* Other configured performance metrics

Using:

```yaml
metrics_to_compute: 'all'
```

allows the pipeline to calculate all available metrics.

---

# 2. Time-Slice Strategies

A patient's health can change drastically from **Day 1 to Day 5**.

Therefore, the accuracy of a clinical score can also change depending on **when** you evaluate it.

The pipeline supports different "snapshots" in time.

For example:

```yaml
time_slices:
  admission:
    run: true
    plots_to_generate: ['roc_curve', 'pr_curve']

  milestones:
    run: true
    hours: [24, 48, 72]
```

The evaluation can therefore ask questions such as:

> "How well does this score predict the outcome when the patient first arrives?"

or:

> "How well does the score perform 24, 48, or 72 hours after admission?"

---

## A. Admission

The **Admission** strategy grabs the very first recorded row for the patient.

This is particularly useful for evaluating triage-oriented tools such as:

```text
MEWS
```

Conceptually:

```text
Patient Admission
       │
       ▼
First Recorded Row
       │
       ▼
Clinical Score
       │
       ▼
Evaluate Against Outcome
```

---

## B. Peak

The **Peak** strategy grabs the absolute highest (worst) score recorded during the patient's entire stay.

This is useful for retrospective severity scales such as:

```text
Pitt Bacteremia Score
```

Instead of asking:

> "How sick was the patient when they arrived?"

it asks:

> "How severe did the patient's condition become during their stay?"

---

## C. Milestones

Milestones evaluate patients at specific numbers of hours after admission.

For example:

```yaml
milestones:
  run: true
  hours: [24, 48, 72]
```

This evaluates the score at:

```text
24 hours
48 hours
72 hours
```

after admission.

This allows you to study how predictive performance changes as the patient's clinical course develops.

---

## ⚠️ Continuous Mode Warning

There is an important difference between evaluating selected snapshots and evaluating every available time point.

Setting:

```yaml
continuous: true
```

evaluates **every single 4-hour interval independently**.

This can be useful for debugging, but it can artificially inflate your accuracy metrics.

For example, suppose one sick patient has 30 rows of observations.

If all 30 predictions are treated as independent observations, that one patient effectively contributes 30 "correct" or "incorrect" predictions.

> ⚠️ **Use continuous mode carefully when interpreting performance metrics.**

---

# 3. Longitudinal Analyses (Performance Over Time)

Sometimes a single snapshot is not enough.

Instead, you may want to see how model performance changes across an entire week.

For example:

> "Does the AUROC improve or degrade as the patient's hospital stay progresses?"

This is what the **longitudinal analysis** is designed to answer.

```yaml
longitudinal:
  auroc_over_time:
    run: true
    bin_hours: 24
    max_hours: 168
```

### `run`

Controls whether the longitudinal AUROC analysis is executed.

```yaml
run: true
```

enables it.

---

### `bin_hours`

Defines the size of each time bucket.

For example:

```yaml
bin_hours: 24
```

means the pipeline divides the patient's stay into **24-hour intervals**.

Conceptually:

```text
Admission
    │
    ├── 0–24h
    ├── 24–48h
    ├── 48–72h
    ├── 72–96h
    └── ...
```

The AUROC is then calculated for each time bucket.

---

### `max_hours`

Defines how far into the patient's stay the longitudinal analysis should continue.

For example:

```yaml
max_hours: 168
```

means the analysis extends up to:

```text
168 hours = 7 days
```

The result is a line graph showing how AUROC changes over time.

---

# 4. Subgroup Stratification (The Multi-Filter Engine)

A score might appear to be **85% accurate overall**, but that does not necessarily mean it performs equally well for every patient population.

For example:

> What if the score is 85% accurate overall, but only 50% accurate for elderly patients?

To investigate this, the pipeline allows you to define **Cohorts** using configurable filters.

> ⚙️ **How It Works:** The Python engine reads your filters and translates string operators, such as `">="`, into actual mathematical operations such as `operator.ge`.
>
> It then uses those rules to slice the dataset before performing the evaluation.

---

## Example: Elderly Diabetics

```yaml
subgroup_analysis:
  run: true

  cohorts:
    - name: "Elderly Diabetics"
      filters:

        # Filter 1: Must be older than 65
        - column: "AGE_AT_ADMISSION"
          operator: ">="
          value: 65

        # AND Filter 2: Must have diabetes
        - column: "hx_diabetes_uncomp"
          operator: "=="
          value: 1
```

This creates a cohort called:

```text
Elderly Diabetics
```

The cohort requires **both conditions** to be satisfied.

### Filter 1 — Age

```yaml
column: "AGE_AT_ADMISSION"
operator: ">="
value: 65
```

This means:

```text
AGE_AT_ADMISSION >= 65
```

### Filter 2 — Diabetes

```yaml
column: "hx_diabetes_uncomp"
operator: "=="
value: 1
```

This means:

```text
hx_diabetes_uncomp == 1
```

Therefore, a patient must satisfy:

```text
Age >= 65
        AND
History of diabetes = 1
```

to enter the cohort.

---

## How Cohorts Are Evaluated

When you run the evaluation script, the pipeline takes the selected sub-population and passes it independently through every active **Time-Slice Strategy**.

For example:

```text
Elderly Diabetics
       │
       ├── Admission
       │
       ├── Peak
       │
       ├── 24h
       │
       ├── 48h
       │
       └── 72h
```

This allows you to compare performance across both:

* **Patient populations**
* **Time points**

---

# 5. Running the Script

Once your `eval_config.yaml` is configured, run the evaluation pipeline from the project root.

```bash
python -m scripts.03_evaluate_scores
```

The script reads the configuration and acts as the **Dumb Dispatcher**.

It determines:

1. Which target label to use
2. Which clinical scores to evaluate
3. Which metrics to calculate
4. Which time-slice strategies to run
5. Which longitudinal analyses to perform
6. Which patient subgroups to evaluate

---

# 6. Understanding the Outputs

The evaluation engine automatically organizes the results into the `outputs/` directory.

The `src/metrics.py` engine also sanitizes cohort names so they can safely be used in filenames.

For example:

```text
"Elderly (>= 65)"
```

can become:

```text
elderly_65
```

The resulting folder structure looks like this:

```text
📦 project_root/
 ┗ 📂 outputs/
   ┣ 📂 metrics/
   ┃ ┣ 📜 master_summary.csv
   ┃ ┣ 📜 all_patients_admission_metrics.csv
   ┃ ┗ 📜 elderly_diabetics_peak_metrics.csv
   │
   ┗ 📂 plots/
     ┣ 🖼️ all_patients_admission_roc.png
     ┣ 🖼️ elderly_diabetics_peak_pr.png
     ┗ 🖼️ temporal_auroc.png
```

---

## Metrics

The `metrics/` directory contains the numerical evaluation results.

### `master_summary.csv`

This is the main summary file containing the results across:

* Clinical scores
* Cohorts
* Evaluation strategies
* Time points

It is the most useful file for comparing experiments.

Other files may contain more specific subsets of the evaluation.

For example:

```text
all_patients_admission_metrics.csv
```

contains the evaluation results for all patients using the Admission strategy.

Similarly:

```text
elderly_diabetics_peak_metrics.csv
```

contains the Peak evaluation for the Elderly Diabetics cohort.

---

## Plots

The `plots/` directory contains the visual evaluation outputs.

Examples include:

```text
all_patients_admission_roc.png
```

A ROC curve for all patients evaluated at admission.

```text
elderly_diabetics_peak_pr.png
```

A precision-recall curve for the Elderly Diabetics cohort evaluated using the Peak strategy.

```text
temporal_auroc.png
```

A longitudinal line graph showing AUROC performance over time.

---

# 7. The Master Summary

> 🎉 **Master Summary:** The `master_summary.csv` is your **golden ticket**.

This file allows you to compare experiments in one place.

You can open it in Excel and filter by the **Strategy** column.

For example, you could compare how a score performs for different patient populations at different stages of admission:

```text
Strategy
   │
   ├── All Patients - Admission
   ├── Young Patients - Admission
   ├── Elderly Patients - Admission
   ├── All Patients - Peak
   └── Elderly Diabetics - Peak
```

This makes it possible to quickly identify differences in clinical score performance across patient groups and evaluation strategies.

---

# 8. Complete Evaluation Flow

The entire evaluation process can be thought of as a series of filters and experiments:

```text
                 Engineered Dataset
                         │
                         ▼
               ┌───────────────────┐
               │  Target Label     │
               │  ground_truth     │
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ Scores to Evaluate│
               │ MEWS / Pitt / ... │
               └─────────┬─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Patient Cohorts   │
              │                     │
              │ All Patients        │
              │ Elderly Diabetics   │
              │ Other cohorts       │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Time Strategies   │
              │                     │
              │ Admission           │
              │ Peak                │
              │ 24h / 48h / 72h    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    Performance      │
              │                     │
              │ AUROC               │
              │ Sensitivity         │
              │ PPV                 │
              │ Other metrics       │
              └──────────┬──────────┘
                         │
                         ▼
                📊 Metrics + Plots
```

---

# Quick Reference

| Area           | Configuration                          | Purpose                                        |
| -------------- | -------------------------------------- | ---------------------------------------------- |
| Target outcome | `target_label`                         | Defines the true outcome column                |
| Scores         | `scores_to_evaluate`                   | Selects which clinical scores to test          |
| Metrics        | `metrics_to_compute`                   | Defines which performance metrics to calculate |
| Admission      | `time_slices.admission`                | Evaluates the first recorded patient row       |
| Peak           | `time_slices.peak`                     | Evaluates the patient's worst recorded score   |
| Milestones     | `time_slices.milestones`               | Evaluates scores at specified hours            |
| Continuous     | `continuous: true`                     | Evaluates every available time interval        |
| Longitudinal   | `longitudinal.auroc_over_time`         | Tracks AUROC over time                         |
| Time buckets   | `bin_hours`                            | Defines the size of longitudinal windows       |
| Maximum time   | `max_hours`                            | Defines how far the analysis extends           |
| Subgroups      | `subgroup_analysis`                    | Enables cohort-based evaluation                |
| Cohort filters | `filters`                              | Defines which patients belong to a cohort      |
| Metrics output | `outputs/metrics/`                     | Stores numerical results                       |
| Plot output    | `outputs/plots/`                       | Stores ROC, PR, and temporal plots             |
| Master results | `master_summary.csv`                   | Central summary of evaluation results          |
| Run evaluation | `python -m scripts.03_evaluate_scores` | Executes the evaluation pipeline               |

---

## ⚠️ Important Things to Remember

1. **The evaluation is configuration-driven.** Change `eval_config.yaml` rather than rewriting the Python dispatcher.
2. **Always define the correct `target_label`.** This is the ground truth against which your scores are evaluated.
3. **Be deliberate about time slices.** Admission, Peak, and Milestones answer different clinical questions.
4. **Use Continuous Mode carefully.** Evaluating every row can artificially inflate performance metrics.
5. **Use subgroup analysis to investigate performance differences.** Overall performance can hide poor performance in specific patient populations.
6. **Check both metrics and plots.** Numerical summaries and visualizations provide complementary views of performance.
7. **Use `master_summary.csv` as your central comparison table.**

> 🎯 **The Goal**
>
> The purpose of this step is not simply to find the highest AUROC.
>
> The goal is to understand **when**, **where**, and **for whom** your clinical scores perform well — and where their limitations may lie.
