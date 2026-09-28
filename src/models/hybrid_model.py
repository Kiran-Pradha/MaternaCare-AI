"""
hybrid_model.py — Review 1 remark: "Run your hybrid model on single dataset
to check accuracy"
MaternaCare AI

Defines and evaluates ONE coherent hybrid model — combining the Phase 7
rules engine's clinical-flag logic with a supervised ML classifier — on the
single real dataset (UCI/Kaggle Maternal Health Risk, 1,014 records), with
proper stratified k-fold cross-validation so the number reported is honest
(no train/test leakage, no cherry-picked split).

Why "hybrid" and what it actually does
----------------------------------------
This project's headline framing is a *hybrid* decision-support system:
data-driven (ML) + knowledge-driven (rules grounded in WHO/NICE/ACOG
thresholds). Previously these lived in separate scripts (baseline.py for
ML, rules_engine.py for rules) without a single combined accuracy number
tying them together — which is what the review panel's remark is asking
to see fixed.

The fusion strategy implemented here, in order of priority:

1. HARD SAFETY OVERRIDE: if the clinical-flag logic (data_validation.py's
   flag_clinical_concerns, the same logic the rules engine uses) detects
   a SEVERE flag (severe hypertension >=160/110, or hypoglycemia, or
   fever), force the prediction to "high risk" regardless of what the ML
   model says. Rationale: a single dangerously abnormal vital should never
   be averaged away by a model that's otherwise doing fine on the bulk of
   cases — this is a patient-safety design choice, not just an accuracy
   optimization, and is worth keeping explicit in the write-up even where
   it costs a little raw accuracy vs. a pure ML approach.
2. Otherwise, SOFT FUSION: the ML classifier's predicted class
   probabilities are blended with a rule-derived risk score (count and
   severity of non-severe flags, mapped to a soft probability distribution
   over the three risk tiers) via a weighted average, then argmax'd.
3. If there are no flags and the ML model is confident, the hybrid
   prediction reduces to the ML prediction.

Evaluation
----------
Stratified 5-fold cross-validation (every fold sees a representative mix of
all 3 risk classes), repeated with 3 different random seeds and averaged,
to get a stable estimate rather than a single lucky/unlucky split. Reports
accuracy, macro-F1, and a confusion matrix for three variants so the
hybrid's contribution is visible against its parts:
  - ML-only (RandomForest, matching Phase 2's baseline approach)
  - Rules-only (flag-count heuristic, no learning)
  - Hybrid (the fusion above)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from data_validation import flag_clinical_concerns, VITAL_RANGES  # noqa: E402

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
CLASSES = ["low risk", "mid risk", "high risk"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Rule-based component
# ---------------------------------------------------------------------------

SEVERE_FLAGS = {"severe_hypertension", "hypoglycemia", "fever"}


def rule_based_probabilities(row: pd.Series) -> np.ndarray:
    """
    Turn a row's clinical flags into a soft probability distribution over
    [low, mid, high] risk. Not a learned model — a transparent, auditable
    heuristic that mirrors what a clinician's mental checklist would do:
    more / more severe flags -> more mass on higher risk tiers.
    """
    flags = flag_clinical_concerns(row)
    n_flags = len(flags)
    has_severe = any(f in SEVERE_FLAGS for f in flags)

    if has_severe:
        return np.array([0.02, 0.08, 0.90])
    if n_flags >= 2:
        return np.array([0.05, 0.35, 0.60])
    if n_flags == 1:
        return np.array([0.20, 0.55, 0.25])
    return np.array([0.70, 0.25, 0.05])


def rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    probs = np.array([rule_based_probabilities(row) for _, row in df.iterrows()])
    return probs.argmax(axis=1)


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------

def hybrid_predict(df: pd.DataFrame, ml_probs: np.ndarray, rule_weight: float = 0.35) -> np.ndarray:
    """
    Combine ML predicted probabilities with rule-based probabilities.
    `ml_probs` must be aligned row-for-row with `df` (same order, same index
    reset). Applies the hard severe-flag override first, then soft fusion.
    """
    preds = np.zeros(len(df), dtype=int)
    for i, (_, row) in enumerate(df.iterrows()):
        flags = flag_clinical_concerns(row)
        if any(f in SEVERE_FLAGS for f in flags):
            preds[i] = CLASS_TO_IDX["high risk"]
            continue
        rule_p = rule_based_probabilities(row)
        fused = rule_weight * rule_p + (1 - rule_weight) * ml_probs[i]
        preds[i] = fused.argmax()
    return preds


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
    return df


def run_single_dataset_evaluation(csv_path: str, n_splits: int = 5, seeds=(42, 7, 123)) -> dict:
    df = load_dataset(csv_path)
    X = df[FEATURES].values
    y = df["RiskLevel"].map(CLASS_TO_IDX).values

    results = {"ml_only": [], "rules_only": [], "hybrid": []}
    all_confusion = {"ml_only": np.zeros((3, 3), int), "rules_only": np.zeros((3, 3), int), "hybrid": np.zeros((3, 3), int)}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            df_test = df.iloc[test_idx].reset_index(drop=True)

            clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
            clf.fit(X_train, y_train)
            ml_probs = clf.predict_proba(X_test)
            ml_preds = ml_probs.argmax(axis=1)

            rule_preds = rule_based_predict(df_test)

            hybrid_preds = hybrid_predict(df_test, ml_probs)

            for name, preds in [("ml_only", ml_preds), ("rules_only", rule_preds), ("hybrid", hybrid_preds)]:
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="macro")
                results[name].append((acc, f1))
                all_confusion[name] += confusion_matrix(y_test, preds, labels=[0, 1, 2])

    summary = {}
    for name, vals in results.items():
        accs = [v[0] for v in vals]
        f1s = [v[1] for v in vals]
        summary[name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "n_folds_total": len(accs),
            "confusion_matrix": all_confusion[name].tolist(),
        }
    return summary


def format_report(summary: dict, csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    lines = []
    lines.append("=" * 72)
    lines.append("HYBRID MODEL — SINGLE-DATASET EVALUATION")
    lines.append(f"Dataset: {os.path.basename(csv_path)} ({len(df)} records, real, not synthetic)")
    lines.append("Method: stratified 5-fold CV x 3 seeds = 15 folds, no leakage")
    lines.append("=" * 72)
    for name, label in [("ml_only", "ML-only (RandomForest)"),
                         ("rules_only", "Rules-only (clinical-flag heuristic)"),
                         ("hybrid", "HYBRID (rules + ML fusion, severe-flag override)")]:
        s = summary[name]
        lines.append(f"\n{label}")
        lines.append(f"  Accuracy : {s['mean_accuracy']*100:.2f}% (+/- {s['std_accuracy']*100:.2f}) across {s['n_folds_total']} folds")
        lines.append(f"  Macro F1 : {s['mean_macro_f1']*100:.2f}% (+/- {s['std_macro_f1']*100:.2f})")
        cm = np.array(s["confusion_matrix"])
        lines.append(f"  Confusion matrix (rows=true, cols=pred, order=[low, mid, high]):")
        for row in cm:
            lines.append(f"    {row.tolist()}")
    return "\n".join(lines)


def run_weight_sensitivity_sweep(csv_path: str, weights=None, n_splits: int = 5, seeds=(42, 7, 123)) -> pd.DataFrame:
    """
    Sweep the rule_weight fusion parameter to find where (if anywhere) the
    hybrid actually beats ML-only, rather than assuming a single arbitrary
    weight. This mirrors the rigor already applied elsewhere in this
    project (Phase 3's GA hyperparameter sweeps) — report the honest
    result of the sweep, including if it shows the naive 50/50-ish fusion
    hurts and only a very light rule weight (or none, i.e. pure
    hard-override-only) is actually justified by the data.
    """
    if weights is None:
        weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]

    df = load_dataset(csv_path)
    X = df[FEATURES].values
    y = df["RiskLevel"].map(CLASS_TO_IDX).values

    rows = []
    for w in weights:
        accs = []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                df_test = df.iloc[test_idx].reset_index(drop=True)

                clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
                clf.fit(X_train, y_train)
                ml_probs = clf.predict_proba(X_test)

                hybrid_preds = hybrid_predict(df_test, ml_probs, rule_weight=w)
                accs.append(accuracy_score(y_test, hybrid_preds))
        rows.append({"rule_weight": w, "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs))})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        project_root / "data" / "maternal_health_risk.csv",
        project_root / "data" / "raw" / "maternal_health_risk.csv",
        project_root / "data" / "raw" / "anemia_dataset.csv",
    ]
    csv_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    if not csv_path.exists():
        raise FileNotFoundError(
            "Could not find a maternal health dataset. Expected one of: "
            + ", ".join(str(p) for p in candidate_paths)
        )

    summary = run_single_dataset_evaluation(str(csv_path))
    print(format_report(summary, str(csv_path)))

    print("\n" + "=" * 72)
    print("RULE-WEIGHT SENSITIVITY SWEEP (finding the honest optimum, not assuming one)")
    print("=" * 72)
    sweep = run_weight_sensitivity_sweep(str(csv_path))
    print(sweep.to_string(index=False))
    best = sweep.loc[sweep["mean_accuracy"].idxmax()]
    print(f"\nBest rule_weight found: {best['rule_weight']} -> {best['mean_accuracy']*100:.2f}% accuracy")