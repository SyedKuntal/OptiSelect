"""
generate_results.py - run every experiment from the command line and save measured results.

    python generate_results.py                         # all 5 datasets x 3 models  (full table)
    python generate_results.py --datasets Wine "Wine Quality"
    python generate_results.py --datasets Iris --models "Logistic Regression"
    python generate_results.py --wq-variant White      # Wine Quality: Red (default) | White | Combined

For every (dataset, model) it
  * sweeps k = 1..n for SelectKBest and RFE with 5-fold CV on the training split,
  * picks the CV-best k for each method (ties -> fewer features; the test set is never used),
  * evaluates All features, SelectKBest(k*) and RFE(k*) once on the held-out test set,
  * saves the 5 figures per method into results/, and
  * writes results/comparison_table.csv (rows for datasets/models not re-run are kept)
    and results/accuracy_vs_k_all.csv.
Every number in these files is measured by running the code - nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import pandas as pd
import sklearn

from dataset_loaders import DATASET_NAMES, DATASETS
from experiments import (METHOD_NAMES, MODEL_NAMES, compute_sweep, best_k, get_split, run_experiment)
from plots import make_all_figures

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TABLE_PATH = RESULTS_DIR / "comparison_table.csv"
SWEEP_PATH = RESULTS_DIR / "accuracy_vs_k_all.csv"


def table_row(res: dict, tag: str) -> dict:
    """tag: 'All features' or a method name."""
    r = res["baseline"] if tag == "All features" else res["optimized"]
    k = res["n_features"] if tag == "All features" else res["k"]
    feats = res["all_features"] if tag == "All features" else r["features"]
    return {
        "Dataset": res["label"], "Model": res["model"], "Method": tag,
        "k / total": f"{k}/{res['n_features']}",
        "CV accuracy mean": round(r["cv_mean"], 4), "CV accuracy SD": round(r["cv_std"], 4),
        "CV accuracy (mean +/- SD)": f"{r['cv_mean']:.3f} +/- {r['cv_std']:.3f}",
        "Test accuracy": round(r["test_acc"], 4), "Macro-F1": round(r["test_f1"], 4),
        "Train time (s)": round(r["train_time"], 4),
        "Selected features": "; ".join(feats) if tag != "All features" else "all",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASET_NAMES, choices=DATASET_NAMES)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--wq-variant", default="Red", choices=DATASETS["Wine Quality"].variants,
                    help="Which Wine Quality data to use (default: Red)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows, sweeps = [], []
    t_all = time.perf_counter()
    for name in args.datasets:
        variant = args.wq_variant if DATASETS[name].variants else None
        _, X_train, _, y_train, _ = get_split(name, variant)
        for model in args.models:
            t0 = time.perf_counter()
            sweep = compute_sweep(X_train, y_train, model)
            first = True
            for method in METHOD_NAMES:
                res = run_experiment(name, model, method, best_k(sweep, method), variant=variant, sweep=sweep)
                make_all_figures(res, RESULTS_DIR)
                if first:
                    rows.append(table_row(res, "All features"))
                    first = False
                rows.append(table_row(res, method))
            sw = sweep.copy()
            sw.insert(0, "Model", model)
            sw.insert(0, "Dataset", res["label"])
            sweeps.append(sw)
            print(f"{res['label']:<26} {model:<20} done in {time.perf_counter() - t0:6.1f}s")

    new = pd.DataFrame(rows)
    new_sw = pd.concat(sweeps, ignore_index=True)
    # Keep earlier rows for datasets/models that were not re-run this time.
    if TABLE_PATH.exists():
        old = pd.read_csv(TABLE_PATH)
        keys = set(zip(new["Dataset"], new["Model"]))
        old = old[[(d, m) not in keys for d, m in zip(old["Dataset"], old["Model"])]]
        new = pd.concat([old, new], ignore_index=True)
    if SWEEP_PATH.exists():
        old = pd.read_csv(SWEEP_PATH)
        keys = set(zip(new_sw["Dataset"], new_sw["Model"]))
        old = old[[(d, m) not in keys for d, m in zip(old["Dataset"], old["Model"])]]
        new_sw = pd.concat([old, new_sw], ignore_index=True)
    order = {n: i for i, n in enumerate(DATASET_NAMES)}
    new = new.assign(_o=new["Dataset"].map(lambda s: order.get(s.split(" (")[0], 99)))
    new = new.sort_values("_o", kind="stable").drop(columns="_o")
    new.to_csv(TABLE_PATH, index=False)
    new_sw.to_csv(SWEEP_PATH, index=False)

    (RESULTS_DIR / "environment.txt").write_text(
        f"python {platform.python_version()}\nscikit-learn {sklearn.__version__}\n"
        f"pandas {pd.__version__}\nrandom_state=42, test_size=0.2, cv=StratifiedKFold(5, shuffle=True)\n")
    print(f"\nSaved {TABLE_PATH.name}, {SWEEP_PATH.name} and figures in {RESULTS_DIR}  "
          f"(total {time.perf_counter() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
