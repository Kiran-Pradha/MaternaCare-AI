"""
Phase 2 — Baseline Classifier (Random Forest + XGBoost)
=========================================================
Trains baseline (non-GA-tuned) classifiers on the Maternal Health Risk dataset,
benchmarks against published results, and adds a trimester-wise evaluation.

IMPORTANT — Trimester simulation note:
The public UCI/Kaggle Maternal Health Risk dataset does NOT include a real
trimester/gestational-age field. To demonstrate the trimester-wise evaluation
methodology described in our proposal, this script assigns a SIMULATED
trimester (1, 2, or 3) to each record. This is still a placeholder — document
this limitation explicitly in your report — but it is NOT purely random.

The simulation is feature-informed: SystolicBP and BS (blood sugar) are used
as proxies, since both clinically tend to trend higher later in pregnancy for
at-risk cases (e.g. gestational hypertension, gestational diabetes). Records
with higher BP/BS are given a higher probability (not certainty) of being
assigned to Trimester 3, and lower BP/BS records a higher probability of
Trimester 1 — sampled probabilistically (via softmax-style weighting) so the
assignment stays noisy and realistic rather than a deterministic giveaway.

This makes the trimester-wise breakdown below reflect a genuine, explainable
pattern grounded in real physiology, rather than pure sampling noise. If you
later find or collect a dataset with real gestational-age/trimester info,
swap out `simulate_trimester()` for the real column.

Usage:
    python src/models/baseline.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase2_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")

sns.set_style("whitegrid")
PALETTE = ["#9C3B63", "#D9799E", "#FBE1EB"]

# Published benchmark from literature (Random Forest, 10-fold CV) — see docs/ write-up references
PUBLISHED_RF_ACCURACY = 0.88


def find_label_column(df):
    for candidate in ["RiskLevel", "Risk Level", "risk_level", "RiskLevel "]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a risk label column automatically. "
        f"Available columns: {list(df.columns)}. Please update find_label_column()."
    )


def simulate_trimester(df, seed=42):
    """
    Feature-informed placeholder trimester assignment — see module docstring.

    Uses SystolicBP and BS (blood sugar) as proxies: higher values push the
    probability toward Trimester 3, lower values toward Trimester 1, with
    Trimester 2 as a stable baseline. Sampling is probabilistic (softmax-style
    weighting), not deterministic — so this is a plausible simulation, not a
    trivial leak.
    """
    rng = np.random.default_rng(seed)

    if "SystolicBP" not in df.columns or "BS" not in df.columns:
        raise KeyError(
            f"simulate_trimester() expects 'SystolicBP' and 'BS' columns. "
            f"Available columns: {list(df.columns)}. Update the proxy columns if your "
            f"dataset uses different names."
        )

    bp_z = (df["SystolicBP"] - df["SystolicBP"].mean()) / df["SystolicBP"].std()
    bs_z = (df["BS"] - df["BS"].mean()) / df["BS"].std()
    propensity = 0.5 * bp_z + 0.5 * bs_z  # higher => more physiologically "later-pregnancy-like"

    # Temperature controls how strongly propensity influences the outcome —
    # lower = noisier/more random, higher = more deterministic. 0.6 keeps it
    # a genuine tendency rather than a giveaway.
    temperature = 0.6

    trimester = np.empty(len(df), dtype=int)
    for i, p in enumerate(propensity.values):
        weights = np.array([
            np.exp(-temperature * p),  # Trimester 1: favored by low propensity
            np.exp(0.0),                # Trimester 2: neutral baseline
            np.exp(temperature * p),   # Trimester 3: favored by high propensity
        ])
        probs = weights / weights.sum()
        trimester[i] = rng.choice([1, 2, 3], p=probs)

    return trimester


def load_data():
    if not os.path.exists(MATERNAL_CSV):
        raise FileNotFoundError(
            f"Dataset not found at {MATERNAL_CSV}. "
            f"Complete Phase 1 first (download + place the CSV) before running Phase 2."
        )
    df = pd.read_csv(MATERNAL_CSV)
    label_col = find_label_column(df)

    df["Trimester"] = simulate_trimester(df)

    X = df.drop(columns=[label_col, "Trimester"])
    y_raw = df[label_col]
    trimester = df["Trimester"]

    # Encode labels with a fixed, clinically sensible order: low < mid < high
    order_map = {}
    for val in y_raw.unique():
        v = str(val).strip().lower()
        if "low" in v:
            order_map[val] = 0
        elif "mid" in v:
            order_map[val] = 1
        elif "high" in v:
            order_map[val] = 2
    if len(order_map) == y_raw.nunique():
        y = y_raw.map(order_map)
        class_names = ["Low Risk", "Mid Risk", "High Risk"]
    else:
        # fallback to generic LabelEncoder if labels don't match expected wording
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = list(le.classes_)

    return X, y, trimester, class_names


def evaluate_model(name, model, X_test, y_test, class_names):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_weighted = f1_score(y_test, preds, average="weighted")
    recall_high = recall_score(y_test, preds, labels=[2], average="macro", zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"High-Risk Recall: {recall_high:.4f}  (most clinically important metric)")
    print("\nFull classification report:")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
    plt.close()

    return {"accuracy": acc, "f1_weighted": f1_weighted, "high_risk_recall": recall_high}


def trimester_wise_evaluation(model, X_test, y_test, trimester_test, class_names, model_name):
    print(f"\n--- Trimester-wise breakdown ({model_name}) ---")
    print("[NOTE: Trimester values are SIMULATED (feature-informed via BP/BS) — see module docstring]")
    results = {}
    for tri in sorted(trimester_test.unique()):
        mask = trimester_test == tri
        if mask.sum() < 5:
            print(f"Trimester {tri}: too few samples ({mask.sum()}) — skipping")
            continue
        preds = model.predict(X_test[mask])
        acc = accuracy_score(y_test[mask], preds)
        f1 = f1_score(y_test[mask], preds, average="weighted")
        print(f"Trimester {tri}: n={mask.sum():3d}  Accuracy={acc:.4f}  Weighted F1={f1:.4f}")
        results[f"trimester_{tri}"] = {"n": int(mask.sum()), "accuracy": acc, "f1_weighted": f1}
    return results


def main():
    print("=" * 60)
    print("PHASE 2 — BASELINE CLASSIFIER")
    print("=" * 60)

    X, y, trimester, class_names = load_data()
    print(f"\nDataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Classes: {class_names}")

    X_train, X_test, y_train, y_test, tri_train, tri_test = train_test_split(
        X, y, trimester, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    all_results = {}

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    all_results["random_forest"] = evaluate_model("Baseline Random Forest", rf, X_test, y_test, class_names)
    all_results["random_forest"]["trimester"] = trimester_wise_evaluation(
        rf, X_test, y_test, tri_test, class_names, "Random Forest"
    )
    joblib.dump(rf, os.path.join(MODELS_DIR, "baseline_random_forest.joblib"))

    # ---- XGBoost ----
    xgb = XGBClassifier(
        n_estimators=100, eval_metric="mlogloss", random_state=42
    )
    xgb.fit(X_train, y_train)
    all_results["xgboost"] = evaluate_model("Baseline XGBoost", xgb, X_test, y_test, class_names)
    all_results["xgboost"]["trimester"] = trimester_wise_evaluation(
        xgb, X_test, y_test, tri_test, class_names, "XGBoost"
    )
    joblib.dump(xgb, os.path.join(MODELS_DIR, "baseline_xgboost.joblib"))

    # ---- Benchmark against literature ----
    print("\n" + "=" * 60)
    print("BENCHMARK AGAINST PUBLISHED LITERATURE")
    print("=" * 60)
    rf_acc = all_results["random_forest"]["accuracy"]
    diff = rf_acc - PUBLISHED_RF_ACCURACY
    print(f"Published Random Forest accuracy (10-fold CV, prior literature): {PUBLISHED_RF_ACCURACY:.4f}")
    print(f"Our baseline Random Forest accuracy (single 80/20 split):        {rf_acc:.4f}")
    print(f"Difference: {diff:+.4f}")
    if abs(diff) <= 0.05:
        print("-> Within a reasonable range of the published benchmark. Pipeline looks validated.")
    else:
        print("-> Meaningful gap from the published benchmark — worth double-checking preprocessing "
              "before moving to Phase 3 (e.g., feature scaling, train/test split method, or "
              "consider k-fold cross-validation instead of a single split for a fairer comparison).")

    # ---- Save results summary ----
    summary_path = os.path.join(OUT_DIR, "phase2_baseline_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to: {summary_path}")
    print(f"Confusion matrices saved to: {OUT_DIR}/")
    print(f"Trained models saved to: {MODELS_DIR}/")
    print("\nNext step: Phase 3 — Genetic Algorithm Optimization (src/models/ga_optimize.py)")


if __name__ == "__main__":
    main()