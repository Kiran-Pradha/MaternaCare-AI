"""
Phase 3g -- Diagnostic: WHY do trend features degrade performance?
======================================================================================
The sensitivity sweep (Phase 3f) found the trend-feature penalty is remarkably
stable (~ -0.12) across a 5x change in noise level and a 4x change in drift
magnitude. That invariance is itself the clue: if the problem were that trend
features are NOISY, then making them noisier should change the effect. It does not.

What IS invariant to rescaling? Uniqueness. Multiplying a continuous column by a
constant leaves every value just as distinct from its neighbours as before.
Near-unique continuous columns are a known failure mode for tree ensembles: they
offer excellent-looking in-sample splits that isolate individual training rows,
so the model memorises rather than generalises.

This script runs four tests to confirm or refute that explanation:

  TEST 1 -- Train vs. test gap.
      Memorisation has a signature: near-perfect TRAINING accuracy with much
      lower test accuracy. If the trend model shows a large gap and the
      latest-only model does not, that is the fingerprint.

  TEST 2 -- Random-noise control (the decisive test).
      Add 10 columns of pure random numbers -- unrelated to the label, to the
      patient, to anything -- to the latest-only feature set. If THIS reproduces
      a similar ~12% drop, then the cause is structural (near-unique continuous
      columns) and has nothing to do with trend features or label conflicts.

  TEST 3 -- max_features setting.
      An earlier fix set max_features=None (consider ALL features at every
      split). If memorisation is the mechanism, that fix made things WORSE, by
      forcing every split to consider the memorisable columns instead of
      sometimes sampling past them. Compares None vs 'sqrt'.

  TEST 4 -- Feature uniqueness measurement.
      Directly counts how many distinct values each feature takes. Confirms
      whether trend features really are near-unique compared to raw vitals.

Self-contained script (no cross-file imports).

Usage:
    python src/models/trend_degradation_diagnostic.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3g_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")

sns.set_style("whitegrid")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES = ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
PATTERNS = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]
N_VISITS = 4
NOISE_PCT = 0.05
DRIFT_RANGE = (0.5, 1.0)


def find_label_column(df):
    for c in ["RiskLevel", "Risk Level", "risk_level"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find risk label column. Columns: {list(df.columns)}")


def encode_labels(y_raw):
    order_map = {}
    for val in y_raw.unique():
        v = str(val).strip().lower()
        if "low" in v:
            order_map[val] = 0
        elif "mid" in v:
            order_map[val] = 1
        elif "high" in v:
            order_map[val] = 2
    return y_raw.map(order_map)


def load_data():
    df = pd.read_csv(MATERNAL_CSV)
    label_col = find_label_column(df)
    y = encode_labels(df[label_col])
    X = df.drop(columns=[label_col])
    return X, y


def pattern_shape(pattern, t):
    if pattern == "Stable":
        return 0.0
    if pattern == "Gradually Worsening":
        return -(1 - t)
    if pattern == "Sudden Deterioration":
        return -(1 - 1 / (1 + np.exp(-12 * (t - 0.75))))
    if pattern == "Improving":
        return (1 - t)


def build_trend_dataset(X, y, seed=RANDOM_SEED):
    """Simulates visits and engineers trend features (same logic as Phase 4)."""
    rng = np.random.default_rng(seed)
    feature_stds = {f: X[f].std() for f in FEATURES}
    pattern_probs = {
        0: [0.40, 0.15, 0.10, 0.35], 1: [0.25, 0.30, 0.20, 0.25], 2: [0.10, 0.35, 0.40, 0.15],
    }
    records = []
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    for pos, row in X_reset.iterrows():
        risk_class = int(y_reset.iloc[pos])
        pattern = rng.choice(PATTERNS, p=pattern_probs[risk_class])
        visits = {f: [] for f in FEATURES}
        for i in range(N_VISITS):
            t = i / (N_VISITS - 1)
            for feat in FEATURES:
                final_val = row[feat]
                if i == N_VISITS - 1:
                    val = final_val
                else:
                    drift = feature_stds[feat] * rng.uniform(*DRIFT_RANGE)
                    noise = rng.normal(0, feature_stds[feat] * NOISE_PCT)
                    val = final_val + pattern_shape(pattern, t) * drift + noise
                visits[feat].append(val)
        rec = {"RiskLevel": risk_class}
        for feat in FEATURES:
            vals = np.array(visits[feat])
            rec[f"{feat}_latest"] = vals[-1]
            rec[f"{feat}_rate_of_change"] = (vals[-1] - vals[0]) / (len(vals) - 1)
            rec[f"{feat}_slope"] = np.polyfit(range(len(vals)), vals, 1)[0]
        records.append(rec)
    return pd.DataFrame(records)


def train_test_gap(X_feat, y_lab, max_features=None, label=""):
    """Returns (train_acc, test_acc) averaged over 5 stratified folds."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_accs, test_accs = [], []
    for tr_idx, te_idx in cv.split(X_feat, y_lab):
        clf = RandomForestClassifier(n_estimators=100, max_features=max_features,
                                      random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X_feat.iloc[tr_idx], y_lab.iloc[tr_idx])
        train_accs.append(accuracy_score(y_lab.iloc[tr_idx], clf.predict(X_feat.iloc[tr_idx])))
        test_accs.append(accuracy_score(y_lab.iloc[te_idx], clf.predict(X_feat.iloc[te_idx])))
    return float(np.mean(train_accs)), float(np.mean(test_accs))


def main():
    print("=" * 70)
    print("PHASE 3g -- WHY DO TREND FEATURES DEGRADE PERFORMANCE?")
    print("=" * 70)

    X, y = load_data()
    print(f"Loaded {len(X)} records")

    trend_df = build_trend_dataset(X, y)
    latest_cols = [f"{f}_latest" for f in FEATURES]
    trend_cols = latest_cols + [f"{f}_rate_of_change" for f in FEATURES] + [f"{f}_slope" for f in FEATURES]

    X_latest = trend_df[latest_cols]
    X_trend = trend_df[trend_cols]
    yy = trend_df["RiskLevel"]

    results = {}

    # ---------------- TEST 1: train vs test gap ----------------
    print("\n" + "=" * 70)
    print("TEST 1 -- TRAIN vs TEST GAP (memorisation fingerprint)")
    print("=" * 70)
    print("A large train-test gap for the trend model, but not the latest-only model,")
    print("indicates memorisation rather than genuine learning.\n")

    lat_train, lat_test = train_test_gap(X_latest, yy, max_features=None)
    tr_train, tr_test = train_test_gap(X_trend, yy, max_features=None)

    print(f"Latest-only   : train={lat_train:.4f}  test={lat_test:.4f}  gap={lat_train - lat_test:+.4f}")
    print(f"Latest+trend  : train={tr_train:.4f}  test={tr_test:.4f}  gap={tr_train - tr_test:+.4f}")
    gap_increase = (tr_train - tr_test) - (lat_train - lat_test)
    print(f"\nGap increase from adding trend features: {gap_increase:+.4f}")
    if gap_increase > 0.05:
        print("-> Consistent with MEMORISATION: trend features let the model fit training")
        print("   rows it cannot generalise from.")
    else:
        print("-> Gap did not widen much; memorisation is NOT the main mechanism.")

    results["test1_train_test_gap"] = {
        "latest_train": lat_train, "latest_test": lat_test, "latest_gap": lat_train - lat_test,
        "trend_train": tr_train, "trend_test": tr_test, "trend_gap": tr_train - tr_test,
        "gap_increase": gap_increase,
    }

    # ---------------- TEST 2: random-noise control (decisive) ----------------
    print("\n" + "=" * 70)
    print("TEST 2 -- RANDOM-NOISE CONTROL (the decisive test)")
    print("=" * 70)
    print("Adds 10 columns of PURE RANDOM numbers to the latest-only features.")
    print("These carry no information whatsoever about the label.")
    print("If this reproduces a similar drop, the cause is structural -- near-unique")
    print("continuous columns -- not trend features specifically.\n")

    rng = np.random.default_rng(RANDOM_SEED)
    X_random = X_latest.copy()
    for i in range(10):
        X_random[f"pure_random_{i}"] = rng.normal(0, 1, len(X_random))

    _, rand_test = train_test_gap(X_random, yy, max_features=None)
    trend_drop = tr_test - lat_test
    random_drop = rand_test - lat_test

    print(f"Latest-only            : test={lat_test:.4f}")
    print(f"Latest + trend features: test={tr_test:.4f}   (drop: {trend_drop:+.4f})")
    print(f"Latest + PURE RANDOM   : test={rand_test:.4f}   (drop: {random_drop:+.4f})")

    if random_drop < -0.03 and abs(random_drop - trend_drop) < 0.06:
        print("\n-> CONFIRMED: pure random columns cause a comparable drop.")
        print("   The degradation is a STRUCTURAL property of adding near-unique continuous")
        print("   columns to a tree ensemble on this dataset -- NOT a property of trend")
        print("   features, and NOT caused by label conflicts.")
    elif random_drop > -0.03:
        print("\n-> NOT CONFIRMED: random columns barely hurt, but trend features do.")
        print("   Something specific to the trend features is responsible -- most likely")
        print("   that they encode the simulation's random per-patient drift draw, which")
        print("   correlates with the label just enough in-sample to mislead the model.")
    else:
        print("\n-> PARTIAL: both hurt, but by clearly different amounts. Structural")
        print("   dilution explains part of the effect, but not all of it.")

    results["test2_random_control"] = {
        "latest_test": lat_test, "trend_test": tr_test, "random_test": rand_test,
        "trend_drop": trend_drop, "random_drop": random_drop,
    }

    # ---------------- TEST 3: max_features setting ----------------
    print("\n" + "=" * 70)
    print("TEST 3 -- max_features: None vs 'sqrt'")
    print("=" * 70)
    print("An earlier fix set max_features=None. If memorisation is the mechanism,")
    print("that fix made things WORSE, by forcing every split to consider the")
    print("memorisable columns instead of sometimes sampling past them.\n")

    _, trend_test_sqrt = train_test_gap(X_trend, yy, max_features="sqrt")
    _, latest_test_sqrt = train_test_gap(X_latest, yy, max_features="sqrt")

    drop_none = tr_test - lat_test
    drop_sqrt = trend_test_sqrt - latest_test_sqrt

    print(f"max_features=None : latest={lat_test:.4f}  trend={tr_test:.4f}  drop={drop_none:+.4f}")
    print(f"max_features=sqrt : latest={latest_test_sqrt:.4f}  trend={trend_test_sqrt:.4f}  drop={drop_sqrt:+.4f}")
    if drop_sqrt > drop_none + 0.01:
        print("\n-> The max_features=None change made the problem WORSE. 'sqrt' is the better")
        print("   setting here, and the earlier fix should be reverted.")
    elif drop_none > drop_sqrt + 0.01:
        print("\n-> max_features=None is genuinely better here; the earlier fix helped.")
    else:
        print("\n-> Both settings behave similarly; max_features is not the deciding factor.")

    results["test3_max_features"] = {
        "none_latest": lat_test, "none_trend": tr_test, "none_drop": drop_none,
        "sqrt_latest": latest_test_sqrt, "sqrt_trend": trend_test_sqrt, "sqrt_drop": drop_sqrt,
    }

    # ---------------- TEST 4: feature uniqueness ----------------
    print("\n" + "=" * 70)
    print("TEST 4 -- FEATURE UNIQUENESS")
    print("=" * 70)
    print("Counts distinct values per feature. A ratio near 1.0 means almost every")
    print("patient has their own unique value -- ideal for memorisation.\n")

    uniqueness = {}
    for col in trend_cols:
        n_unique = trend_df[col].nunique()
        ratio = n_unique / len(trend_df)
        uniqueness[col] = {"n_unique": int(n_unique), "ratio": float(ratio)}

    latest_ratios = [uniqueness[c]["ratio"] for c in latest_cols]
    derived_cols = [c for c in trend_cols if c not in latest_cols]
    derived_ratios = [uniqueness[c]["ratio"] for c in derived_cols]

    print(f"{'Feature':<32} {'distinct':>10} {'ratio':>8}")
    print("-" * 52)
    for col in latest_cols:
        print(f"{col:<32} {uniqueness[col]['n_unique']:>10} {uniqueness[col]['ratio']:>8.3f}")
    print("-" * 52)
    for col in derived_cols:
        print(f"{col:<32} {uniqueness[col]['n_unique']:>10} {uniqueness[col]['ratio']:>8.3f}")

    print(f"\nMean uniqueness ratio -- raw vitals     : {np.mean(latest_ratios):.3f}")
    print(f"Mean uniqueness ratio -- trend features : {np.mean(derived_ratios):.3f}")
    if np.mean(derived_ratios) > 0.9 and np.mean(latest_ratios) < 0.5:
        print("\n-> Trend features are near-unique per patient while raw vitals are heavily")
        print("   repeated. This is exactly the asymmetry that drives memorisation, and it")
        print("   explains why rescaling noise/drift changed nothing: rescaling a continuous")
        print("   column does not make its values any less distinct.")

    results["test4_uniqueness"] = {
        "per_feature": uniqueness,
        "mean_ratio_raw_vitals": float(np.mean(latest_ratios)),
        "mean_ratio_trend_features": float(np.mean(derived_ratios)),
    }

    # ---------------- Plot ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(["Latest-only", "Latest+trend"], [lat_train, tr_train], color="#FBE1EB",
                label="Train", edgecolor="#9C3B63")
    axes[0].bar(["Latest-only", "Latest+trend"], [lat_test, tr_test], color="#9C3B63", label="Test")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test 1: Train vs Test Gap")
    axes[0].legend()

    axes[1].bar(["Latest-only", "+ trend", "+ pure random"],
                [lat_test, tr_test, rand_test], color=["#FBE1EB", "#9C3B63", "#D9799E"])
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Test 2: Random-Noise Control")
    axes[1].axhline(y=lat_test, color="gray", linestyle="--", linewidth=1)

    axes[2].bar(["raw vitals", "trend features"],
                [np.mean(latest_ratios), np.mean(derived_ratios)], color=["#FBE1EB", "#9C3B63"])
    axes[2].set_ylabel("Mean uniqueness ratio")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Test 4: Feature Uniqueness")

    plt.suptitle("Diagnostic: Why Trend Features Degrade Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "trend_degradation_diagnostic.png"), dpi=150)
    plt.close()

    summary_path = os.path.join(OUT_DIR, "phase3g_diagnostic_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {summary_path}")
    print(f"Plot saved to: {OUT_DIR}/trend_degradation_diagnostic.png")


if __name__ == "__main__":
    main()