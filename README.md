# Are You a Fraud? — Credit-card fraud classification

[![Python checks](https://github.com/ZekeH43/Project-2/actions/workflows/python-checks.yml/badge.svg)](https://github.com/ZekeH43/Project-2/actions/workflows/python-checks.yml)

A Python machine-learning project comparing **K-Nearest Neighbors, Gradient Boosting, and XGBoost** on transaction data. A majority-class baseline shows why high accuracy alone can be misleading when fraud is uncommon.

The project includes a runnable experiment, a guided notebook, input validation, regression tests, and a saved result snapshot. It demonstrates data preparation, scikit-learn pipelines, classification metrics, and reproducible experiments. It is an educational analysis, not a deployed fraud-prevention service.

## Run the experiment

Prerequisites: **Python 3.11 or 3.12** and Git. The dataset is already included; no API keys or GPU are required. Run these commands from a terminal.

### Windows PowerShell

```powershell
git clone https://github.com/ZekeH43/Project-2.git
cd Project-2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m fraud_detection
```

Using the virtual environment's Python directly avoids changing PowerShell's activation policy. If Python 3.12 is not installed, install it first or use `py -3.11` when creating the environment.

### Linux, WSL, or macOS

With Python 3.12 and its `venv` support installed:

```bash
git clone https://github.com/ZekeH43/Project-2.git
cd Project-2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m fraud_detection
```

Use `python3.11` instead if that is your installed supported version. The commands below use `python` from the activated environment; in PowerShell, substitute `.\.venv\Scripts\python.exe`.

The default run samples **10,000 rows**, trains on **7,500**, evaluates on **2,500**, prints the metric table, and writes `artifacts/metrics.json`. It reads the full source CSV before sampling, so reducing the sample size reduces training work but not CSV-loading memory. Generated files in `artifacts/` are ignored by Git.

```bash
# Repeat the original project's larger experiment, now with randomized sampling.
python -m fraud_detection --sample-size 20000 --output artifacts/metrics-20000.json

# Use the smaller historical CSV for a quick demonstration.
python -m fraud_detection --data Resources/credit_card_data_2.csv

# Choose particular models or see all options.
python -m fraud_detection --models baseline knn
python -m fraud_detection --help
```

`--all-rows` uses the entire CSV and can make KNN evaluation slow. A sample size greater than the file's row count uses all available rows. Custom CSVs must match the [documented schema](docs/DATA.md). Each run replaces the JSON file at its chosen output path.

## Explore the notebook

Install the additional notebook dependencies in the same environment:

```bash
python -m pip install -r requirements-notebook.txt
python -m jupyter lab Final_module.ipynb
```

Choose **Restart Kernel and Run All Cells**. The [notebook](Final_module.ipynb) explains the data, model comparison, confusion matrices, training-data correlations, and Gradient Boosting feature importance. It calls the tested Python modules rather than maintaining a second copy of the training code.

To execute it without opening a browser:

```bash
python -m jupyter nbconvert --to notebook --execute Final_module.ipynb --output executed.ipynb --output-dir artifacts --ExecutePreprocessor.timeout=300
```

## Results from the default experiment

The [saved JSON](reports/metrics.json) was generated with seed **42**, a stratified 10,000-row sample, and a 25% holdout. The holdout contains **218 fraud** and **2,282 non-fraud** transactions. Precision, recall, and F1 below refer to **fraud (class 1)**; values are rounded to four decimals.

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.9128 | 0.0000 | 0.0000 | 0.0000 | 0.0872 |
| K-Nearest Neighbors | 0.9912 | 0.9623 | 0.9358 | 0.9488 | 0.9558 |
| Gradient Boosting | 0.9976 | 0.9775 | 0.9954 | 0.9864 | 0.9995 |
| XGBoost | 0.9972 | 0.9907 | 0.9771 | 0.9838 | 0.9983 |

The baseline gets **91.28% accuracy while catching no fraud**. On this particular split, Gradient Boosting misses fewer fraud cases, while XGBoost produces fewer false alarms. These are descriptive results from one dataset and one split; they do not establish which model would work best in a live system. The snapshot also includes balanced accuracy, ROC AUC, full classification reports, and confusion matrices.

Recreate the snapshot with:

```bash
python -m fraud_detection --output reports/metrics.json
```

The JSON records the dataset's SHA-256 hash, sample settings, class counts, and numerical-library versions. Direct dependencies are pinned; transitive dependencies can differ by environment. Small numerical differences may occur across library/platform versions.

## How the code works

| File | Purpose |
| --- | --- |
| `fraud_detection/data.py` | Validate the CSV, sample reproducibly, and create a stratified train/test split. |
| `fraud_detection/models.py` | Build fresh models, fit on training data, and calculate holdout metrics. |
| `fraud_detection/__main__.py` | Parse command-line options, print results, and save the experiment record. |
| `Final_module.ipynb` | Walk through the analysis with plots and explanations. |
| `tests/` | Test input failures, split isolation, model fitting, metric calculations, and the CLI. |
| `docs/WALKTHROUGH.md` | Explain the design and provide interview practice prompts. |
| `docs/DATA.md` | Describe fields, file provenance, and historical assets. |
| `reports/metrics.json` | Preserve the measured result snapshot. |

## Tests and checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

GitHub Actions runs the tests, code checks, and a full notebook execution on Python **3.11 and 3.12**. Regression tests use synthetic fixtures to check behavior, not to claim model performance. The notebook and saved report use the committed transaction CSV.

## Corrections and limitations

- The original notebook reversed `classification_report` arguments. The current code uses actual labels first, so precision, recall, and support have their intended meanings.
- The original scaler was fitted on training data. KNN now preserves that behavior in a pipeline with a regression test; the two tree models do not need scaling.
- Sampling now draws across the file and preserves class proportions instead of using the first 10,000 or 20,000 rows. The corrected results should not be compared as though they used the old split.
- XGBoost is a separate model, not proof that Gradient Boosting avoided overfitting. No cross-validation or hyperparameter search has been performed in this version.
- Dataset origin, version, and license were not recorded in the original repository. Real-world provenance is unverified. There are no timestamps or customer IDs for chronological or grouped evaluation.
- Repeated model or threshold selection using this holdout would bias its scores. Future tuning should use training-only cross-validation and a fresh final test set.
- Tree feature importance reflects the fitted model and can favor continuous features; it does not show causation. No production deployment or calibrated risk scores are provided.

The original presentation files and smaller data exports remain as historical project assets. Their old screenshots and conclusions are not the current result snapshot. Git history preserves the original notebook and README.

## Learning references

- [scikit-learn: classification report argument order](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn: average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
