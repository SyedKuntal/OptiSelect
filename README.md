# OptiSelect - Multi-Dataset Smart Feature Selection

Group 1 - Optimization. A Streamlit dashboard that compares a classifier trained on **all features**
with the same classifier trained on an **automatically selected subset** (SelectKBest or RFE), on five datasets.

## Quick start

```bash
cd optiselect
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                 # the dashboard
python generate_results.py           # (optional) all experiments -> results/ (tables + all figures)
```

In the app: pick **Dataset -> Classifier -> Selection method -> Number of features (k) -> Run Experiment**.
You get test accuracy, macro-F1, the chosen feature names, training time, a comparison against the
all-features baseline, and the five mandatory figures (also saved as PNGs in `results/`).
The slider's upper bound follows the selected dataset.

## Datasets

| # | Dataset | Features | Source | Notes | Contributor |
|---|---------|----------|--------|-------|-------------|
| 1 | Iris | 4 | scikit-learn | 3 classes | *(name to be added)* |
| 2 | Wine | 13 | scikit-learn | 3 classes | Syed Kuntal |
| 3 | Breast Cancer | 30 | scikit-learn | binary | *(name to be added)* |
| 4 | Heart Disease | 13 | UCI (Cleveland file) | 303 rows, 6 missing values (`?`) imputed inside the Pipeline; target 0 vs 1-4 -> binary | *(name to be added)* |
| 5 | Wine Quality | 11 | UCI | target **quality >= 6 -> "Good"**; Red (default) / White / Combined selectable | Syed Kuntal |

Raw files live in `data/` (`heart_disease/processed.cleveland.data`, `wine_quality/winequality-*.csv`).
Wine Quality files are `;`-separated and contain exact duplicate rows (kept; this can make test scores
slightly optimistic - mention it in the report's limitations).

## Method (identical for every dataset)

- One stratified 80/20 split, `random_state=42`.
- One `Pipeline`: `SimpleImputer(median)` -> `StandardScaler` -> selector -> classifier. All steps are fitted on
  training data only (no leakage).
- k is compared by 5-fold stratified CV on the **training split**. The test set is used once per reported
  configuration and never to choose k or features.
- SelectKBest = filter (ANOVA F-score). RFE = wrapper (Logistic Regression estimator, one feature removed per step).
- "CV-best k" = highest mean CV accuracy; ties go to the smaller k.
- Classifiers: Logistic Regression, Decision Tree, Random Forest (default hyper-parameters, `random_state=42`).
- Training time = one `pipeline.fit` on the training split (imputer + scaler + selector + model).
  The dots in Figure 5 use CV fit time per fold, which is approximate because folds run in parallel.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (tabs: Experiment, Data overview, Manual selection, Results table, About) |
| `experiments.py` | Pipeline, CV sweep, baseline vs optimized evaluation, captions built from measured numbers |
| `dataset_loaders.py` | One loader per dataset + the `DATASETS` registry |
| `plots.py` | The 5 mandatory figures + 2 data-overview charts; saves PNGs |
| `generate_results.py` | Runs everything from the command line; writes `results/comparison_table.csv` |
| `results/` | Figures (PNG), `comparison_table.csv`, `accuracy_vs_k_all.csv`, `environment.txt` |

Figure files: `<dataset>__<model>__<method>_k<k>__figN_*.png` (Figures 1 and 5 show both methods, so
they have no method/k in the name).

## Adding or changing a dataset

1. In `dataset_loaders.py` write `_load_xxx(variant=None) -> Dataset` (X as a DataFrame, y as 0..C-1 integers,
   class names, and short notes on the target and preprocessing).
2. Add one `DatasetSpec(...)` line to `DATASETS` (member number, contributor name, suggested k values, default k).
3. Run `python generate_results.py --datasets "<name>"` - the table and figures update; other rows are kept.

## The five figures (what to explain in the report)

1. Accuracy vs number of features - where accuracy peaks.
2. Baseline vs optimized - whether fewer features preserve performance.
3. Feature scores - which inputs matter most.
4. Confusion matrix - which classes are confused.
5. Accuracy vs runtime - the speed / performance trade-off.

Every number in `results/` and in the app is measured by running this code; nothing is typed in by hand.
