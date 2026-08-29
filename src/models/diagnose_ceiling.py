"""
Phase 3b — Ceiling Diagnostic
================================
Investigates WHY GA-optimization plateaued at the same accuracy regardless of
hyperparameters. Checks two things:

  1. Label conflicts: records with identical (or near-identical) feature
     values but DIFFERENT risk labels. If these exist, no classifier can get
     both right — this is a hard, irreducible error floor baked into the
     data itself, not a modeling failure.

  2. Visual class overlap: a 2D PCA projection of the feature space, colored
     by risk level. If classes visibly overlap in this projection, that's
     further evidence of genuine ambiguity between adjacent risk categories
     (e.g. Mid vs. High), rather than a solvable optimization problem.

Usage:
    python src/models/diagnose_ceiling.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import load_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
PINK_PALETTE = ["#FBE1EB", "#D9799E", "#9C3B63"]  # Low -> Mid -> High


def check_exact_duplicates(X, y, class_names):
    print("\n" + "=" * 60)
    print("1. EXACT DUPLICATE FEATURE CHECK")
    print("=" * 60)

    df = X.copy()
    df["_label"] = y.values

    # Group by all feature columns, look for groups with >1 distinct label
    feature_cols = [c for c in df.columns if c != "_label"]
    grouped = df.groupby(feature_cols)["_label"].nunique()
    conflicting_groups = grouped[grouped > 1]

    n_conflicting_rows = df.set_index(feature_cols).index.isin(conflicting_groups.index).sum()

    print(f"Total records: {len(df)}")
    print(f"Feature combinations with >1 distinct risk label: {len(conflicting_groups)}")
    print(f"Total rows involved in these conflicts: {n_conflicting_rows} "
          f"({100 * n_conflicting_rows / len(df):.1f}% of the dataset)")

    if len(conflicting_groups) > 0:
        print("\nExample conflicting feature combinations (same features, different labels):")
        example_keys = conflicting_groups.index[:3]
        for key in example_keys:
            mask = True
            for col, val in zip(feature_cols, key if isinstance(key, tuple) else (key,)):
                mask = mask & (df[col] == val)
            subset = df[mask]
            labels_seen = [class_names[l] for l in subset["_label"].unique()]
            print(f"  Features {dict(zip(feature_cols, key if isinstance(key, tuple) else (key,)))} "
                  f"-> labeled as: {labels_seen} ({len(subset)} occurrences)")

    # ---- Theoretical maximum achievable accuracy ----
    # For each conflicting group, a deterministic classifier can at best predict
    # the MAJORITY label within that group — every minority-labeled duplicate is
    # then guaranteed to be misclassified, no matter how good the model is.
    # This gives a hard upper bound on achievable accuracy for this dataset.
    irreducible_errors = 0
    for group_key in conflicting_groups.index:
        mask = True
        for col, val in zip(feature_cols, group_key if isinstance(group_key, tuple) else (group_key,)):
            mask = mask & (df[col] == val)
        group_labels = df.loc[mask, "_label"]
        majority_count = group_labels.value_counts().max()
        group_size = len(group_labels)
        irreducible_errors += (group_size - majority_count)  # minority-labeled rows: unavoidably wrong

    theoretical_max_accuracy = (len(df) - irreducible_errors) / len(df)
    print(f"\nIrreducible errors (minority-labeled duplicates within conflicting groups): {irreducible_errors}")
    print(f"THEORETICAL MAXIMUM ACCURACY on this dataset (any model, any tuning): "
          f"{theoretical_max_accuracy:.4f} ({theoretical_max_accuracy*100:.1f}%)")
    print("-> No classifier, however well-tuned, can exceed this ceiling — the label conflicts "
          "make a portion of the dataset fundamentally unresolvable.")

    return {
        "total_records": len(df),
        "conflicting_feature_combinations": int(len(conflicting_groups)),
        "rows_involved_in_conflicts": int(n_conflicting_rows),
        "pct_involved": float(100 * n_conflicting_rows / len(df)),
        "irreducible_errors": int(irreducible_errors),
        "theoretical_max_accuracy": float(theoretical_max_accuracy),
    }


def check_near_duplicates(X, y, class_names, tolerance_pct=0.02):
    """
    Looser check: rounds continuous features to reduce float-precision noise,
    catching 'near-identical' cases that exact matching would miss.
    """
    print("\n" + "=" * 60)
    print("2. NEAR-DUPLICATE CHECK (rounded features)")
    print("=" * 60)

    df = X.copy()
    df["_label"] = y.values

    rounded = X.copy()
    for col in rounded.select_dtypes(include="number").columns:
        # Round to ~2% of the feature's range, collapsing tiny differences
        col_range = rounded[col].max() - rounded[col].min()
        step = max(col_range * tolerance_pct, 1e-9)
        rounded[col] = (rounded[col] / step).round() * step

    rounded["_label"] = y.values
    feature_cols = [c for c in rounded.columns if c != "_label"]
    grouped = rounded.groupby(feature_cols)["_label"].nunique()
    conflicting_groups = grouped[grouped > 1]
    n_conflicting_rows = rounded.set_index(feature_cols).index.isin(conflicting_groups.index).sum()

    print(f"Feature combinations (rounded to ~{tolerance_pct*100:.0f}% precision) with conflicting labels: "
          f"{len(conflicting_groups)}")
    print(f"Rows involved: {n_conflicting_rows} ({100 * n_conflicting_rows / len(df):.1f}% of the dataset)")

    return {
        "conflicting_near_duplicate_groups": int(len(conflicting_groups)),
        "rows_involved": int(n_conflicting_rows),
        "pct_involved": float(100 * n_conflicting_rows / len(df)),
    }


def visualize_class_overlap(X, y, class_names):
    print("\n" + "=" * 60)
    print("3. VISUAL CLASS OVERLAP (PCA projection)")
    print("=" * 60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()

    print(f"2D PCA explains {explained*100:.1f}% of total feature variance "
          f"(a rough approximation — real overlap may involve more dimensions).")

    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name,
                    color=PINK_PALETTE[i % len(PINK_PALETTE)], alpha=0.6, s=30, edgecolor="white", linewidth=0.3)
    plt.title(f"Risk Class Overlap — 2D PCA Projection\n({explained*100:.1f}% variance explained)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Risk Level")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "class_overlap_pca.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    return {"pca_variance_explained": float(explained)}


def main():
    print("=" * 60)
    print("PHASE 3b — CEILING DIAGNOSTIC")
    print("=" * 60)

    X, y, trimester, class_names = load_data()
    X_features = X  # trimester already excluded by load_data()

    exact = check_exact_duplicates(X_features, y, class_names)
    near = check_near_duplicates(X_features, y, class_names)
    overlap = visualize_class_overlap(X_features, y, class_names)

    print("\n" + "=" * 60)
    print("SUMMARY / INTERPRETATION")
    print("=" * 60)
    if exact["rows_involved_in_conflicts"] > 0 or near["rows_involved"] > 0:
        print(f"Found genuine label conflicts affecting {near['pct_involved']:.1f}% of records "
              f"(near-duplicate check). This provides direct evidence that a portion of the GA "
              f"optimization ceiling is due to irreducible ambiguity in the dataset itself, not a "
              f"modeling limitation.")
    else:
        print("No exact or near-duplicate label conflicts found. The performance ceiling is more likely "
              "due to genuine (but non-identical) overlap between adjacent risk classes in feature space "
              "— see the PCA visualization for visual evidence of this overlap.")

    print(f"\nUse docs/phase3_outputs/class_overlap_pca.png directly in your report/slides as evidence "
          f"for why hyperparameter tuning alone couldn't push past the observed accuracy ceiling.")

    import json
    summary = {"exact_duplicates": exact, "near_duplicates": near, "pca_overlap": overlap}
    with open(os.path.join(OUT_DIR, "ceiling_diagnostic_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {os.path.join(OUT_DIR, 'ceiling_diagnostic_results.json')}")


if __name__ == "__main__":
    main()