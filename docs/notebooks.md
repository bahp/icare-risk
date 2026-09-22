# 📚 Tutorial Notebooks

Welcome to the **iCARE-RISK** interactive tutorials! You can download any of the Jupyter Notebooks below to run locally in your Python environment or execute on platforms like Google Colab or JupyterLab.

---

<div class="grid cards" markdown>

-   **01. Quickstart Guide**

    ---

    An end-to-end walkthrough of the iCARE-RISK pipeline setup, model initialization, and basic prediction workflow on clinical data.

    [Download 01_quickstart.ipynb](notebooks/tutorials/01_quickstart.ipynb){ .md-button .md-button--primary }

-   **02. History Batch Processing**

    ---

    Learn how to execute historical cohort extractions, handle large scale data, and perform batch risk scoring efficiently.

    [Download 02_history_batch.ipynb](notebooks/tutorials/02_history_batch.ipynb){ .md-button }

-   **05a. Charlson Comorbidity Score**

    ---

    Detailed guide on calculating the Charlson Comorbidity Index using ICD diagnoses codes and extracting risk weights.

    [Download 05_score_charlson.ipynb](notebooks/tutorials/05_score_charlson.ipynb){ .md-button }

-   **05b. Pitt Bacteremia Score**

    ---

    Calculate the Pitt Bacteremia Score for bloodstream infection severity assessment using clinical variables.

    [Download 05_score_pitt.ipynb](notebooks/tutorials/05_score_pitt.ipynb){ .md-button }

-   **05c. SIRS Criteria**

    ---

    Evaluate Systemic Inflammatory Response Syndrome (SIRS) criteria across temporal vital sign and lab measurements.

    [Download 05_score_sirs.ipynb](notebooks/tutorials/05_score_sirs.ipynb){ .md-button }

-   **10. Package Utilities**

    ---

    Explore core package utilities for data validation, temporal feature engineering, and internal data structures.

    [Download 10_pkg_utils.ipynb](notebooks/tutorials/10_pkg_utils.ipynb){ .md-button }

-   **11. Package Tools**

    ---

    Advanced helper tools for risk score benchmarking, pipeline logging, and output evaluation metrics.

    [Download 11_pkg_tools.ipynb](notebooks/tutorials/11_pkg_tools.ipynb){ .md-button }

</div>

---

### 💡 How to Run These Locally

1. Make sure you have `icare-risk` installed in your Python environment:
   ```bash
   pip install icare-risk