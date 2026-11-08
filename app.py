"""
OptiSelect - Smart Feature Selection dashboard.
Run with:   streamlit run app.py
"""
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dataset_loaders import DATASETS, DATASET_NAMES, load_dataset
from experiments import (METHOD_NAMES, MODEL_NAMES, compute_sweep, describe_results, get_split,
                         run_experiment, run_manual)
from plots import fig_class_distribution, fig_correlation, make_all_figures

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TABLE_PATH = RESULTS_DIR / "comparison_table.csv"

st.set_page_config(page_title="OptiSelect AI", layout="wide")


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_sweep(dataset: str, variant, model: str) -> pd.DataFrame:
    """The k = 1..n cross-validation sweep (training split only). Cached per dataset/model."""
    _, X_train, _, y_train, _ = get_split(dataset, variant)
    return compute_sweep(X_train, y_train, model)


def zip_figures(paths: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths.values():
            z.write(p, arcname=Path(p).name)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("OptiSelect AI")
st.sidebar.caption("Feature Selection Dashboard")
st.sidebar.subheader("Controls")

dataset = st.sidebar.selectbox("Dataset", DATASET_NAMES)
spec = DATASETS[dataset]

variant = None
if spec.variants:
    variant = st.sidebar.selectbox(f"{dataset} data", spec.variants,
                                   help="Red / White wine files from UCI, or both stacked together.")
ds = load_dataset(dataset, variant)
n_features = ds.X.shape[1]

model_name = st.sidebar.selectbox("Classifier", MODEL_NAMES)
method = st.sidebar.selectbox("Selection method", METHOD_NAMES,
                              help="SelectKBest = filter (ANOVA F-score). RFE = wrapper (recursive elimination "
                                   "with Logistic Regression).")
# The slider's upper bound follows the selected dataset.
k = int(st.sidebar.slider("Number of features (k)", 1, n_features, min(spec.default_k, n_features),
                          key=f"k_{dataset}"))
st.sidebar.caption(f"{dataset} has {n_features} input features. Example sizes from the guide: "
                   f"{', '.join(map(str, spec.suggested_sizes))}.")
run_clicked = st.sidebar.button("Run Experiment", type="primary")

if run_clicked:
    with st.spinner(f"Running {model_name} on {ds.label}: sweeping k = 1..{n_features} for both methods..."):
        sweep = cached_sweep(dataset, variant, model_name)
        result = run_experiment(dataset, model_name, method, k, variant=variant, sweep=sweep)
        paths = make_all_figures(result, RESULTS_DIR)
    st.session_state["result"] = result
    st.session_state["paths"] = paths


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("OptiSelect - Smart Feature Selection")
st.caption("Compare a classifier trained on ALL features with one trained on an automatically selected subset. "
           "Imputation, scaling and selection are fitted on training data only.")

tab_run, tab_data, tab_manual, tab_table, tab_about = st.tabs(
    ["Experiment", "Data overview", "Manual selection", "Results table", "About"])


# ------------------------------- Experiment tab ----------------------------- #
with tab_run:
    st.subheader(f"Dataset preview - {ds.label} ({len(ds.X)} rows, {n_features} features)")
    preview = ds.X.head(8).copy()
    preview["Class"] = ds.y.head(8).map(lambda i: ds.class_names[i])
    st.dataframe(preview)

    st.markdown("**Model overview**")
    c = st.columns(4)
    c[0].metric("Dataset samples", len(ds.X))
    c[1].metric("Input features", n_features)
    c[2].metric("Selected features (k)", k)
    c[3].markdown(f"<div style='font-size:14px'>Model</div><div style='font-size:22px;line-height:1.6'>"
                  f"{model_name}</div>", unsafe_allow_html=True)
    st.caption(f"Contributor: {spec.contributor}")

    result = st.session_state.get("result")
    paths = st.session_state.get("paths")
    if result is None:
        st.info("Choose a dataset, classifier, method and k in the sidebar, then click **Run Experiment**.")
    else:
        base, opt = result["baseline"], result["optimized"]
        st.divider()
        st.subheader(f"Results: {result['label']} | {result['model']} | "
                     f"{result['method']} | k = {result['k']} of {result['n_features']}")
        if (result["dataset"], result["variant"], result["model"], result["method"], result["k"]) != \
                (dataset, variant, model_name, method, k):
            st.caption("Showing the last run. The sidebar has changed since - click Run Experiment to update.")

        m = st.columns(4)
        m[0].metric("Test accuracy", f"{opt['test_acc']:.3f}",
                    f"{opt['test_acc'] - base['test_acc']:+.3f} vs all features")
        m[1].metric("Test macro-F1", f"{opt['test_f1']:.3f}",
                    f"{opt['test_f1'] - base['test_f1']:+.3f} vs all features")
        m[2].metric("Training time", f"{opt['train_time']:.3f} s",
                    f"{opt['train_time'] - base['train_time']:+.3f} s vs all features", delta_color="inverse")
        m[3].metric("CV accuracy (mean ± SD)", f"{opt['cv_mean']:.3f} ± {opt['cv_std']:.3f}")

        st.markdown(f"**Chosen features ({len(opt['features'])}):** "
                    + "  ".join(f"`{f}`" for f in opt["features"]))
        kb = result["cv_best_k"]
        st.info("Best k by 5-fold cross-validation on the training split (ties go to fewer features): "
                + " | ".join(f"**{mm}: k = {kb[mm]}**" for mm in METHOD_NAMES)
                + ". The test set was not used to choose k.")

        table = pd.DataFrame({
            "Setup": [f"All features ({result['n_features']})", f"{result['method']} ({result['k']})"],
            "CV accuracy (mean ± SD)": [f"{base['cv_mean']:.3f} ± {base['cv_std']:.3f}",
                                        f"{opt['cv_mean']:.3f} ± {opt['cv_std']:.3f}"],
            "Test accuracy": [round(base["test_acc"], 4), round(opt["test_acc"], 4)],
            "Test macro-F1": [round(base["test_f1"], 4), round(opt["test_f1"], 4)],
            "Training time (s)": [round(base["train_time"], 4), round(opt["train_time"], 4)],
        })
        st.dataframe(table, hide_index=True)
        st.caption(f"Split: {result['n_train']} train / {result['n_test']} test (stratified, random_state=42). "
                   f"Target: {ds.target_note}")

        caps = describe_results(result)
        names = ["fig1_accuracy_vs_k", "fig2_baseline_vs_optimized", "fig3_feature_scores",
                 "fig4_confusion_matrix", "fig5_accuracy_vs_runtime"]
        left, right = st.columns(2)
        for i, key in enumerate(names):
            with (left if i % 2 == 0 else right):
                st.image(str(paths[key]))
                st.caption(caps[key.split("_")[0]])
        st.download_button("Download all 5 figures (ZIP)", zip_figures(paths),
                           file_name=f"{Path(paths[names[0]]).name.split('__fig')[0]}_figures.zip",
                           mime="application/zip")
        st.caption(f"PNG files are also saved in: {RESULTS_DIR}")


# ------------------------------- Data overview tab -------------------------- #
with tab_data:
    st.subheader(f"{ds.label}: what the data looks like")
    c = st.columns(4)
    c[0].metric("Rows", len(ds.X))
    c[1].metric("Input features", n_features)
    c[2].metric("Classes", len(ds.class_names))
    c[3].metric("Missing values", int(ds.X.isna().sum().sum()))
    st.markdown(f"**Source / preprocessing:** {ds.data_note}")
    st.markdown(f"**Target:** {ds.target_note}")
    miss = ds.X.isna().sum()
    if miss.sum() > 0:
        st.markdown("**Missing values per feature (imputed inside the Pipeline with the training median):**")
        st.dataframe(miss[miss > 0].rename("missing").to_frame())
    left, right = st.columns(2)
    with left:
        fig = fig_class_distribution(ds)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig = fig_correlation(ds)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("**Summary statistics**")
    st.dataframe(ds.X.describe().T)


# ------------------------------- Manual selection tab ----------------------- #
with tab_manual:
    st.subheader("Choose input features yourself")
    st.write("Tick the features to keep, then compare against all features using the identical split and "
             "cross-validation protocol. Uses the dataset and classifier from the sidebar.")
    cols = st.columns(3)
    chosen = []
    for i, f in enumerate(ds.X.columns):
        if cols[i % 3].checkbox(f, value=i < min(3, n_features), key=f"man_{dataset}_{variant}_{f}"):
            chosen.append(f)
    if st.button("Run Feature Optimization", key="manual_run"):
        if not chosen:
            st.warning("Tick at least one feature.")
        else:
            with st.spinner("Training..."):
                man = run_manual(dataset, model_name, chosen, variant=variant)
            a, b = man["manual"], man["all"]
            st.markdown("**Selected feature output:** " + "  ".join(f"`{f}`" for f in chosen))
            st.dataframe(pd.DataFrame({
                "Setup": [f"Your {len(chosen)} features", f"All {n_features} features"],
                "CV accuracy (mean ± SD)": [f"{a['cv_mean']:.3f} ± {a['cv_std']:.3f}",
                                            f"{b['cv_mean']:.3f} ± {b['cv_std']:.3f}"],
                "Test accuracy": [round(a["test_acc"], 4), round(b["test_acc"], 4)],
                "Test macro-F1": [round(a["test_f1"], 4), round(b["test_f1"], 4)],
                "Training time (s)": [round(a["train_time"], 4), round(b["train_time"], 4)],
            }), hide_index=True)


# ------------------------------- Results table tab -------------------------- #
with tab_table:
    st.subheader("Final comparison table (measured values)")
    if TABLE_PATH.exists():
        full = pd.read_csv(TABLE_PATH)
        shown = full.drop(columns=["CV accuracy mean", "CV accuracy SD"], errors="ignore")
        c1, c2 = st.columns(2)
        ds_pick = c1.multiselect("Dataset", sorted(full["Dataset"].unique()))
        md_pick = c2.multiselect("Model", sorted(full["Model"].unique()))
        if ds_pick:
            shown = shown[shown["Dataset"].isin(ds_pick)]
        if md_pick:
            shown = shown[shown["Model"].isin(md_pick)]
        st.dataframe(shown, hide_index=True)
        st.download_button("Download table (CSV)", full.to_csv(index=False), "comparison_table.csv", "text/csv")
        st.caption("For each dataset and model, k for SelectKBest and RFE is the CV-best k on the training split. "
                   "Test scores come from one evaluation on the held-out 20%.")
    else:
        st.info("No table yet. Create it by running  `python generate_results.py`  in the project folder "
                "(it runs every dataset, classifier and method and saves measured results only).")


# ------------------------------- About tab ---------------------------------- #
with tab_about:
    st.subheader("About OptiSelect")
    st.markdown(
        "**Group 1 - Optimization.** One dashboard that compares classification with all features against "
        "automatically selected feature subsets.\n\n"
        "**Protocol (same for every dataset)**\n"
        "- One stratified 80/20 train/test split, `random_state=42`.\n"
        "- A single scikit-learn `Pipeline`: median imputer, standard scaler, feature selector, classifier. "
        "Every step is fitted on training data only, so nothing leaks from the test set.\n"
        "- k is compared with 5-fold stratified cross-validation on the training split. "
        "The held-out test set is used once per reported configuration.\n"
        "- Methods: **SelectKBest** (filter, ANOVA F-score) and **RFE** (wrapper, Logistic Regression estimator).\n"
        "- Baseline: the same pipeline with all features.\n")
    st.markdown("**Dataset contributors**")
    st.dataframe(pd.DataFrame([{"Member": s.member_no, "Dataset": s.name, "Contributor": s.contributor,
                                "Source": s.source} for s in DATASETS.values()]), hide_index=True)
