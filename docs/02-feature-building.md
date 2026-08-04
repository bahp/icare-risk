# Feature & Phenotype Builder

Welcome to the **Feature Pipeline**!

Raw hospital data is incredibly messy—patients miss lab tests, doctors use different diagnosis codes, and vitals are recorded at random times. Before we can build predictive AI models, we must translate this raw data into clean, mathematical signals.

!!! tip "For Dummies"
    Think of `feature_config.yaml` as the **control panel**.
    
    Instead of writing hundreds of lines of complex Python code to clean data and search for 
    ICD-10 codes, you simply declare what you want in this file. The background Python scripts 
    act as an automated factory that reads your rules and builds the final dataset.

The pipeline operates in **four strict phases**, moving from simple cleaning to complex medical algorithms:

1. Base Features
2. Computed Features
3. Custom Phenotypes
4. Final Clinical Scores

---

## 1. Phase 1: Base Features (Time-Series Cleaning)

This section of the YAML file handles raw numbers that change over time, such as:

* Heart rate
* Blood pressure
* Temperature
* Laboratory measurements

It fixes missing data and calculates trends automatically using the logic inside `features.py`.

```yaml id="m7t2ak"
base_features:
  hr:
    impute: 'ffill'
    missing_indicator: false
    delta: true
    rolling:
      windows: ['24h', '7D']
      aggs: ['mean', 'max']
```

### Exhaustive Parameter Breakdown

#### `impute`

Defines how the system handles a blank spot where a nurse did not record a vital.

**`'ffill'` — Forward Fill**

Takes the last known value and carries it forward.

For example, if HR was 80 at 8 AM, the system assumes it is still roughly 80 at 9 AM.

**`'constant'`**

Replaces missing data with a specific baseline number.

This is used together with:

```yaml
fill_value: 5.0
```

---

#### `missing_indicator: true/false`

If set to `true`, this creates a brand-new column indicating whether the original measurement was missing.

For example:

```text
crp_is_missing
```

The resulting column contains `1`s and `0`s.

!!! info "Why this matters"

    The fact that a doctor forgot to order a test can itself be a strong signal.


> 💡 **Why this matters:** The fact that a doctor forgot to order a test can itself be a strong signal.

---

#### `delta: true`

Calculates the mathematical difference between the current reading and the previous reading.

This is useful for detecting sudden changes, such as:

* Sudden spikes in blood pressure
* Sudden drops in vital signs
* Rapid changes in other time-series measurements

---

#### `rolling`

Creates summary windows over previous measurements.

For example:

```yaml
rolling:
  windows: ['24h']
  aggs: ['max', 'min']
```

This tells the system to look back over the last 24 hours for **each specific patient**, preventing cross-contamination between patients.

It can then create columns such as:

```text
hr_24h_max
hr_24h_min
```

---

## 2. Phase 2: Computed Features (Fast Math)

Once the base features are clean, we can perform basic mathematical operations on them.

This section uses the Pandas `df.eval()` engine, which allows you to write simple mathematical equations as plain-text strings.

```yaml id="6ynp0v"
computed_features:
  # Simple division for clinical ratios
  shock_index: "hr / sbp"

  # Boolean Logic (Returns 1 for True, 0 for False)
  qsofa_score: "(rr >= 22) + (sbp <= 100)"
```

### 🧮 How Boolean Math Works

In Python:

```text
True  = 1
False = 0
```

By putting equations in parentheses and adding them together, you can instantly build simple scores.

For example:

```text
qsofa_score: "(rr >= 22) + (sbp <= 100)"
```

If a patient has:

```text
Respiratory Rate = 25
Systolic BP       = 120
```

Then:

```text
(rr >= 22)  → True  → 1
(sbp <= 100) → False → 0
```

Therefore:

```text
qsofa_score = 1
```

---

## 3. Phase 3: Custom Features (Phenotypes)

A **Phenotype** is a clinical state.

We cannot use simple mathematics to determine whether a patient has conditions such as:

* Metastatic cancer
* Diabetes
* Previous myocardial infarction

Instead, we need to search through information such as:

* Clinical text
* Pharmacy records
* Billing codes
* ICD-10 codes

For example:

```yaml id="p8u0yc"
custom_features:
  hx_mi:  # History of Myocardial Infarction
    module: 'src.phenotypes'
    function: 'derive_historical_condition'
    kwargs:
      target_codes: ['I21', 'I22', 'I25.2', '323..00', 'G30..00']
```

### How It Works

#### 1. `module` & `function`

These tell the system exactly which Python script and function to run.

In this example:

```text
module   → src.phenotypes
function → derive_historical_condition
```

The pipeline therefore runs:

```python
derive_historical_condition()
```

---

#### 2. `kwargs` — Keyword Arguments

This is where you pass specific instructions to the Python function.

The function searches through the patient's entire medical history, such as the `ICARE_PROBLEMS_ANON` table.

If it finds any of the specified codes using prefix matching, it places a:

```text
1
```

in the `hx_mi` column.

---

### ⚙️ Complex Example: Temporal Pharmacy Search

Consider a rule such as `has_vasopressors`.

This rule calls:

```text
has_medication_in_window
```

and passes:

```text
window_hours: 24
```

along with a list of:

```text
target_meds
```

The Python code uses the `_get_prescriptions_in_window` helper to temporally align the patient's vitals with the pharmacy database.

It then checks whether any of the specified drugs, such as `epinephrine`, were administered within exactly 24 hours of that specific moment in time.

The result is:

```text
1 → medication found
0 → medication not found
```

---

## 4. Phase 4: Final Clinical Scores

This is the top of the pyramid.

Now that we have:

1. Clean vitals
2. Computed mathematical features
3. Complex `0/1` phenotypes

we can calculate validated medical scores such as:

* Charlson Comorbidity Index
* INCREMENT-ESBL

For example:

```yaml id="q5zj8k"
custom_scores:
  charlson_quan_score:
    module: 'src.scores'
    function: 'calculate_charlson_quan'
    kwargs:
      age_col: 'AGE_AT_ADMISSION'
      mi_col: 'hx_mi'
      chf_col: 'hx_chf'
      # ... other mappings
```

### The Golden Rule

The `kwargs` here map the **names of the columns you generated in Phase 3** to the **variables expected by the medical calculator**.

In other words:

> "Hey Calculator, when you need to know if the patient has a history of MI, look inside the column named `hx_mi`."

This separation keeps the feature-generation layer independent from the clinical score calculation.

---

# 5. Tutorial: Adding a New Feature (Acute Kidney Injury)

Let's walk through an exhaustive example.

We want to create a new phenotype called **Acute Kidney Injury (AKI)**.

A patient is flagged for AKI if:

1. They have the ICD-10 code `N17`
2. **OR** their rolling Creatinine lab test is severely elevated, for example `> 1.5 mg/dL`

---

## Step 1: Update `feature_config.yaml`

First, we need to make sure Creatinine is being processed as a base feature so that rolling windows are available.

Then we declare our new custom phenotype.

```yaml id="w9b3fx"
# 1. Add Creatinine to Base Features so we get rolling windows
base_features:
  creatinine:
    impute: 'ffill'
    rolling:
      windows: ['24h']
      aggs: ['max']

# 2. Add the new AKI phenotype to Custom Features
custom_features:
  is_aki:  # This will be the name of the new column
    module: 'src.phenotypes'
    function: 'derive_aki_status'
    kwargs:
      creatinine_col: 'creatinine_24h_max'
      target_codes: ['N17']
```

The important part is that `creatinine_col` points to the rolling feature generated in Phase 1:

```text
creatinine_24h_max
```

---

## Step 2: Write the Logic in `src/phenotypes.py`

Next, open `phenotypes.py` and write the `derive_aki_status` function.

The example uses built-in helper functions to keep the code clean and fast.

```python id="j5w8r2"
def derive_aki_status(df, **kwargs):
    """
    Derives Acute Kidney Injury (1=Yes, 0=No).
    Logic: ICD-10 code 'N17' OR Creatinine > 1.5.
    """

    # Start by assuming no AKI (fill column with 0s)
    flag = pd.Series(0, index=df.index)

    # Rule 1: Check for Historical ICD-10 Codes
    target_codes = kwargs.get('target_codes', [])

    has_code = _patient_has_historical_codes(
        df=df,
        context_df=kwargs.get('context_dfs', {}).get('problems'),
        patient_col='SUBJECT',
        code_col='PROBLEM_CODE',
        target_codes=target_codes
    )

    flag.loc[has_code] = 1

    # Rule 2: Check the rolling Lab Value Proxy
    creat_col = kwargs.get('creatinine_col')

    if creat_col in df.columns:
        creat = pd.to_numeric(df[creat_col], errors='coerce')
        flag.loc[creat > 1.5] = 1

    return flag.values
```

### What the Function Does

The function starts by assuming that every patient is **not** flagged:

```python
flag = pd.Series(0, index=df.index)
```

It then applies two independent rules.

### Rule 1 — Historical ICD-10 Code

The function searches for the configured target codes:

```python
target_codes = kwargs.get('target_codes', [])
```

It then uses:

```python
_patient_has_historical_codes(...)
```

to search the patient's historical problems.

If the `N17` code is found:

```text
is_aki = 1
```

### Rule 2 — Creatinine Threshold

The function retrieves the configured rolling feature:

```python
creat_col = kwargs.get('creatinine_col')
```

It then converts the values to numeric values and checks:

```python
creat > 1.5
```

If the patient's rolling creatinine exceeds `1.5`:

```text
is_aki = 1
```

The two rules therefore work as an **OR** condition.

---

## Step 3: What the Output Looks Like

When you run the pipeline, the system will automatically:

1. Pull the data
2. Apply the rolling window to creatinine
3. Execute the new `derive_aki_status` function
4. Append the `is_aki` column to the final dataset

The resulting data could look like this:

| patient_id | date             | creatinine | creatinine_24h_max | is_aki | Explanation                                                                       |
| ---------: | ---------------- | ---------: | -----------------: | -----: | --------------------------------------------------------------------------------- |
|        101 | 2024-01-01 08:00 |        0.9 |                0.9 |      0 | Healthy patient, normal creatinine, no ICD codes.                                 |
|        102 | 2024-01-01 12:00 |        1.6 |                1.6 |  **1** | Flagged! Creatinine is > 1.5.                                                     |
|        103 | 2024-01-02 08:00 |        1.1 |                1.1 |  **1** | Flagged! Labs are normal, but patient has the `N17` ICD-10 code in their history. |

---

# 6. Under the Hood: The Python Backend Rules

If you are a developer looking to add new medical rules, you must adhere to the naming conventions and architectural rules defined by the pipeline.

---

## 1. `phenotypes.py` — The Translation Layer

This file extracts messy data and turns it into clean `1`s and `0`s.

> **Important:** Do not put final predictive score mathematics here.

### `has_*`

Used for **past medical history**.

For example:

```text
has_diabetes
```

These functions always return:

```text
1 or 0
```

---

### `is_*`

Used for **acute states during the visit**.

For example:

```text
is_mechanically_ventilated
```

These functions always return:

```text
1 or 0
```

---

### `derive_*`

Used for **continuous variables or categories**.

For example:

* Extracting an age
* Categorizing fever into 0, 1, or 2 points
* Other derived clinical variables

---

## 2. `scores.py` — The Rule Engine

This file only does mathematics.

> **Important:** Do not search for ICD-10 codes here.

The file relies heavily on a custom function called:

```python
evaluate_score
```

This allows you to build a transparent "rulebook" for a clinical score.

For example, the INCREMENT-ESBL score can be represented as a list of rules:

```python id="v4q6sj"
rules = [
    {
        'desc': 'Age > 50',
        'col': age_col,
        'condition': df[age_col] > 50,
        'points': 3
    },
    {
        'desc': 'Charlson > 3',
        'col': charlson_col,
        'condition': df[charlson_col] > 3,
        'points': 4
    },
]

return evaluate_score(df, rules, score_name, verbose, logger)
```

### ✅ Why We Use This Rule Engine

Clinical scores require auditing.

If a doctor asks:

> "Why did my patient get an INCREMENT score of 7?"

the `evaluate_score` engine automatically prints a clean trace such as:

```text
[+] Age > 50 (Value: 72) +3 Points
```

This makes the AI pipeline transparent and explainable.

---

## Quick Reference

| Phase             | Purpose                                  | Main Configuration  |
| ----------------- | ---------------------------------------- | ------------------- |
| **Phase 1**       | Clean and transform time-series data     | `base_features`     |
| **Phase 2**       | Perform mathematical calculations        | `computed_features` |
| **Phase 3**       | Derive clinical phenotypes               | `custom_features`   |
| **Phase 4**       | Calculate validated clinical scores      | `custom_scores`     |
| Phenotype backend | Translate messy clinical data into flags | `src.phenotypes`    |
| Score backend     | Apply transparent clinical scoring rules | `src.scores`        |
| Score engine      | Evaluate and trace clinical rules        | `evaluate_score`    |

### Feature Pipeline at a Glance

```text
Raw Hospital Data
       │
       ▼
┌─────────────────────┐
│ Phase 1             │
│ Base Features       │
│ Cleaning + Rolling  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Phase 2             │
│ Computed Features   │
│ Fast Mathematical   │
│ Operations           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Phase 3             │
│ Custom Phenotypes   │
│ Clinical States     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Phase 4             │
│ Clinical Scores     │
│ Validated Rules     │
└──────────┬──────────┘
           │
           ▼
      Model-Ready Data
```

---
