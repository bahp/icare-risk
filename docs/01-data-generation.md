# Synthetic Clinical Data Generator

This tool allows you to generate fake patient data. Instead of hard-coding values, 
the system uses a flexible **blueprint** (`data_config.yaml`) to build a relational 
database of patients, vitals, labs, and medications.

!!! tip "For Dummies" 
    Think of this tool as a fully automated factory.

    The `data_config.yaml` serves as the master **recipe book** that dictates
    exactly what the factory should build, while the Python scripts act sas the
    **machines** that actually manufacture the data and build it.


---

## 1. Global Parameters (The "Big Picture")

At the very top of your `data_config.yaml` file, you will find the `data_source`,
`paths` and `generation_params`. This latter dictates the size and scope of your 
fake hospital.

```yaml
data_source: "synthetic"            # Generate synthetic data

paths:
  synthetic_dir: "data/synthetic"   # Path to save synthetic data
  external_dir: "data/external"
  processed_dir: "data/processed"

generation_params:
  n_patients: 100           # How many fake patients to create
  days: 10                  # How many days of data to simulate
  freq: '4h'                # How often to check vitals (every 4 hours)
  output_format: 'tidy'     # The shape of the final data
  default_missing_rate: 0.5 # 50% chance a test isn't taken (makes data realistic)
```

---

## 2. The Clinical Dictionary (The "Menu")

Before building tables, we define universal medical concepts under `clinical_concepts`. 
This ensures the generator never creates impossible scenarios (like a body temperature 
of 100°C) and standardizes coding.

Concepts are grouped into:

* `vitals`
* `labs`
* `neuro`

Each concept defines:

| Property  | Description                                                                                      |
|-----------|--------------------------------------------------------------------------------------------------|
| `code`    | The LOINC code (e.g., `"LOINC-8867-4"` for heart rate)                                           |
| `name`    | The human-readable string (e.g. `"Heart Rate"`)                                                  |
| `unit`    | Measurement unit (e.g., `'bpm'`, `'mg/dL'`).                                                     |
| `prob`    | The probability this specific test is ordered/recorded. This overrides the default missing rate. |
| `range`   | The absolute `[min, max]` bounds for generated values.                                           |

---

## 3. Table Architectures

!!! info "Under the Hood"
    The `tables` section dictates how CSVs are structured and how they relate.

Let's break down the core architectural rules.

### A. Table Types & Row Counts

Every table must define its basic structural behavior.

#### `type: "relational"`

A standard spreadsheet where each row is a distinct event, (e.g. admission or a prescribed drug).

#### `type: "eav_timeseries"`

Entity-Attribute-Value format.

This is used for repeating measurements such as heart rate over time. It makes one long table where the **test name**
and **result** are stacked in rows rather than spread across many different columns.

#### `rows_per_patient_range: [1, 5]`

This tells the generator:

> "For every fake patient, generate anywhere between 1 and 5 random rows for this table."

Some patients might get 1 medical problem, while others might get 5.

#### `source: "clinical_concepts.vitals"`

Used **only** in EAV tables.

It tells the table which dictionary concepts to use to generate the test values.

---

### B. The `map_to` Keyword (The Translator)

In `eav_timeseries` tables, such as vitals or lab results, the generator produces raw data first.

You use `map_to` to tell the generator exactly which column in your final CSV should hold which piece of data.

```yaml
OBSERVATION_CODE:
  map_to: "concept.code"  # Takes the LOINC code from the dictionary

OBSERVATION_NAME:
  map_to: "concept.name"  # Takes the human-readable name

OBSERVATION_RESULT_CLEAN:
  map_to: "value"         # Takes the actual randomly generated number (e.g., 98.6)
```

---

## 4. Column Types Deep-Dive

Inside the `schema` of a table, you define your columns. Here is what each column 
configuration does with soeme examples included.

### 1. The Combo Meal: `categorical_tuple`

This is crucial for clinical accuracy. If a patient has diabetes, their diagnosis code should 
always match the description. If you generated them separately, you might accidentally give a 
patient an Asthma code with a Diabetes description. A `categorical_tuple` locks them together.

```yaml
PROBLEM_TUPLE:
  type: "categorical_tuple"
  columns: [ "PROBLEM_CODE", "PROBLEM_DESC" ]
  values:
    # If the generator picks line 1, it inserts both "E11.9" and Diabetes text safely.
    - [ "E11.9", "Type 2 diabetes mellitus" ]
    - [ "I10", "Essential (primary) hypertension" ]
    - [ "J44.9", "Chronic obstructive pulmonary disease" ]
```

---

### 2. Time Travel Prevention: `date` and `date_offset`

Medical data must follow a timeline. You can't resolve a medical problem before you've 
diagnosed it. The generator uses offsets to enforce this logic.

```yaml
PROBLEM_DT_TM:
  type: "date"
  start: "2020-01-01"
  end: "2024-01-01"  # Picks a random baseline date in this window.

UPDATE_DT_TM:
  type: "date_offset"
  base_col: "PROBLEM_DT_TM" # Look at the problem date we just generated...
  days_range: [ 0, 90 ]     # ...and add anywhere from 0 to 90 days to it.
```

!!! success "Result"
    If the problem started on January 1st, the update date is mathematically guaranteed to happen between
    January 1st and April 1st. No time paradoxes!

---

### 3. The V.I.P. Pass: `foreign_key`

This is how tables talk to each other.

If you generate a prescription, it needs to belong to a valid hospital visit.

```yaml
ENCNTR_ID:
  type: "foreign_key"
  source_table: "ICARE_EPISODES_ANON"
```

This tells the script:

> "Go look at the Episodes table, find a valid Encounter ID for this specific patient, and paste it here."

---

### 4. Simple Generators

The generator also provides several simple column types:

| Type         | Description                                                                    |
|--------------|--------------------------------------------------------------------------------|
| `unique_id`  | Generates a random 7-digit number (e.g., `8472910`).                           |
| `enumerated` | Randomly selects a value from a defined list, such as `["yes", "no", "none"]`. |
| `boolean`    | Flips a coin. `probability: 0.70` means 70% chance of generating `1` (True).   |
| `int`        | Generates random whole numbers within a provided `range: [min, max]`.          |
| `float`      | Generates random decimal numbers within a provided `range: [min, max]`.        |

---

## 5. The Configured Tables

The `data_config.yaml` currently builds **6 interconnected tables** for your cohort.

| # | Table Name                            | Description                                                                                 |
|---|---------------------------------------|---------------------------------------------------------------------------------------------|
| 1 | **`ICARE_EPISODES_ANON`**             | The anchor table. Records admissions, discharges, age, and deprivation deciles.             |
| 2 | **`ICARE_MICROBIOLOGY_ANON`**         | Tracks blood/urine cultures, organism growth (e.g., *E. coli*, MRSA), and sensitivities.    |
| 3 | **`ICARE_VITAL_SIGNS_ANON`**          | Dense time-series of heart rate, temperature, blood pressure, etc., tracked longitudinally. |
| 4 | **`ICARE_PROBLEMS_ANON`**             | ICD-10 diagnostic history (diabetes, sepsis, CKD) with onset and resolution dates.          |
| 5 | **`ICARE_PHARMACY_PRESCRIBING_ANON`** | Medication orders (antibiotics, pressors, fluids) with routes and dosages.                  |
| 6 | **`ICARE_PATHOLOGY_BLOOD_ANON`**      | Lab test results (creatinine, CRP, lactate) featuring delayed result timestamps.            |

---

## 6. How Data is Created

The Python backend (`src/generators.py`) executes the YAML blueprint sequentially.

It establishes parent tables, such as **Episodes**, first so child tables, such as **Pharmacy** and **Vitals**, can
safely inherit foreign keys.

It then applies:

* Biological variance
* Bounds logic
* Missingness masks

This produces highly realistic synthetic datasets.

---

## 7. Running the Tool

Use the unified orchestration suite to execute data generation.

This automatically routes through Docker, or runs locally if specified, and outputs 
your CSVs to a timestamped folder in `data/synthetic/`.

### Default — Docker

```bash
# Runs in Docker Container
make generate
```

### Native Execution

```bash
# Runs on Host CPU
make generate local
```

!!! note "Note for Windows users"
    Use `.\make.bat generate`.

---

## Quick Reference

| Area                         | Key Configuration / Command              |
|------------------------------|------------------------------------------|
| Number of patients           | `generation_params.n_patients`           |
| Simulation duration          | `generation_params.days`                 |
| Measurement frequency        | `generation_params.freq`                 |
| Missing data                 | `generation_params.default_missing_rate` |
| Clinical concepts            | `clinical_concepts`                      |
| Standard relational tables   | `type: "relational"`                     |
| Time-series tables           | `type: "eav_timeseries"`                 |
| Map generated values         | `map_to`                                 |
| Keep related values together | `categorical_tuple`                      |
| Generate dependent dates     | `date_offset`                            |
| Connect tables               | `foreign_key`                            |
| Generate with Docker         | `make generate`                          |
| Generate locally             | `make generate local`                    |
| Windows                      | `.\make.bat generate`                    |

---
