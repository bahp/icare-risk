# Pipeline Validation & Quality Assurance

In clinical machine learning, a single logic error can have cascading effects. If a function mistakenly categorizes a patient's temperature or misreads a medication history, the final risk score will be entirely compromised.

**Checking that each individual step is working correctly is the most critical part of the pipeline.**

!!! danger "Garbage In, Garbage Out"

    A predictive model is only as intelligent as the features it is trained on. Before 
    evaluating model performance (AUROC, AUPRC, etc.), you must first guarantee that every 
    clinical score and phenotype is being calculated correctly.

---

## 1. Core Components to Validate

The pipeline transforms raw clinical data through three major processing stages. Each should be reviewed and tested independently.

### `features.py`

Responsible for feature engineering.

Verify that:

- Rolling windows correctly calculate statistics (e.g., 24-hour maximum heart rate).
- Missing values are handled appropriately.
- Aggregations produce expected patient-level features.

### `phenotypes.py`

Responsible for clinical phenotype extraction.

Verify that:

- ICD-10 regular expressions correctly identify diagnoses.
- Medication mappings correctly convert raw text into binary indicators.
- Clinical rules consistently return `0` or `1`.

### `scores.py`

Responsible for clinical risk score calculation.

Verify that:

- Every scoring algorithm assigns points correctly.
- Thresholds match published clinical definitions.
- Final score totals are mathematically accurate.

---

## 2. Clinical Score Validation

The repository includes a dedicated validation workflow for confirming that clinical scoring algorithms produce the expected outputs.

Validation uses two files:

- `tests/cases.csv`
- `scripts/05_validate_scores.py`

!!! note "Validation Strategy"

    `cases.csv` contains synthetic test patients with manually calculated expected scores (for example `exp_increment` and `exp_gavaghan`).

    During validation, every patient is processed through the scoring engine and the computed values are compared against the manually verified expected values.

---

## 3. Running Validation

Execute the validation script from the project root.

```bash
python -m scripts.05_validate_scores
```

---

## 4. Reading the Validation Report

After execution, a detailed audit report is generated:

```text
reports/score_validation.log
```

The report explains exactly how each score was calculated and provides a full point-by-point breakdown.

!!! success "Expected Result"

    Look for **✅ MATCH** indicators.

    If a **❌ MISMATCH** appears, the report identifies the patient, score, and calculation that failed, making it straightforward to locate the underlying issue in `scores.py`.

Example output:

```text
▶ CASE ID: 1 | High-Risk Tertiary
============================================================
  [INCREMENT-ESBL Breakdown]

    [+] Age > 50                  (Value: 60)   +3
    [+] Charlson > 3              (Value: 5)    +4
    [+] Pitt Score >= 6           (Value: 7)    +3
    [+] SIRS >= 2                 (Value: 3)    +4
    [+] Non-Urinary Source        (Value: 1)    +3
    [+] Non-E. coli               (Value: 1)    +2
    [+] Inappropriate Abx         (Value: 1)    +2
    ---------------------------------------------
    [=] TOTAL COMPUTED:           21

    ✅ MATCH: Expected 21, got 21
```

---

## 5. Creating New Unit Tests

Whenever new phenotypes or scoring systems are added, corresponding unit tests should also be implemented.

This prevents logic drift, where later modifications unintentionally alter existing clinical behaviour.

### Two Levels of Testing

| Test Type | Target File | Purpose |
|------------|-------------|---------|
| **Phenotype Tests** | `test_phenotypes.py` | Verify binary clinical logic (e.g. *Is 91 bpm greater than 90?*) |
| **Score Tests** | `test_scores.py` | Verify score calculations, weighting, and rule hierarchy |

---

## 6. Example: Testing a New Phenotype

Suppose a new phenotype function named `is_hypotensive()` is added.

### Step 1 — Create Edge Cases

Construct a small DataFrame containing values immediately above and below the clinical threshold.

For an SBP threshold of **90 mmHg**, test:

- 89
- 90
- 120

```python
def test_is_hypotensive_logic():
    data = pd.DataFrame({
        "sbp": [89, 90, 120]
    })

    result = is_hypotensive(data, sbp_col="sbp")

    assert result.iloc[0] == 1  # 89 mmHg
    assert result.iloc[1] == 0  # 90 mmHg
    assert result.iloc[2] == 0  # 120 mmHg
```

---

### Step 2 — Test Keyword Arguments

Many phenotype functions locate columns through `**kwargs`.

Always pass explicit column names during testing so the function does not fall back to production defaults (such as `sbp_24h_min`) that do not exist in the mock dataset.

!!! note "Testing Tip"

    When testing clinical scores in `test_scores.py`, use the `score_configs` fixture.

    This ensures your unit tests use the same scoring weights defined in `config/feature_config.yaml`.

---

## 7. Best Practices

- Test both minimum and maximum scoring scenarios.
- Include threshold edge cases (e.g. 89, 90, and 91).
- Verify overlapping rules return logical OR (`|`) behaviour rather than summing duplicate flags.
- Ensure phenotype functions return a `pd.Series` rather than a `numpy.ndarray`, allowing use of `.iloc` and `.loc` in unit tests.
- Keep expected values manually verified so automated validation always has a trusted reference.