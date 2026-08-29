"""
Phase 1 — Dataset Acquisition & EDA
=====================================
Run this after placing the two raw CSVs in data/raw/:
    - data/raw/maternal_health_risk.csv
    - data/raw/anemia_dataset.csv

Usage:
    python src/data/eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")
ANEMIA_CSV = os.path.join(RAW_DIR, "anemia_dataset.csv")

sns.set_style("whitegrid")


def check_file_exists(path, friendly_name):
    if not os.path.exists(path):
        print(f"[MISSING] {friendly_name} not found at: {path}")
        print(f"          Please download it and place it there (see README.md Phase 1 section).")
        return False
    return True


def run_maternal_eda():
    print("\n" + "=" * 60)
    print("MATERNAL HEALTH RISK DATASET — EDA")
    print("=" * 60)

    if not check_file_exists(MATERNAL_CSV, "Maternal Health Risk dataset"):
        return None

    df = pd.read_csv(MATERNAL_CSV)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nSummary statistics:\n{df.describe()}")

    # Try to find the risk label column (commonly named 'RiskLevel')
    label_col = None
    for candidate in ["RiskLevel", "Risk Level", "risk_level", "RiskLevel "]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col:
        print(f"\nClass distribution ({label_col}):")
        print(df[label_col].value_counts())

        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x=label_col, hue=label_col, order=df[label_col].value_counts().index,
                      palette="RdPu", legend=False)
        plt.title("Maternal Risk Level — Class Distribution")
        plt.xlabel("Risk Level")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "maternal_class_distribution.png"), dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/maternal_class_distribution.png")
    else:
        print("\n[WARNING] Could not auto-detect the risk label column. "
              "Check df.columns and update this script's label_col logic.")

    # Histograms for numeric features
    numeric_cols = df.select_dtypes(include="number").columns
    n = len(numeric_cols)
    if n > 0:
        fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(14, 3.5 * ((n + 2) // 3)))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color="#9C3B63")
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "maternal_feature_histograms.png"), dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/maternal_feature_histograms.png")

    # Correlation heatmap
    if n > 1:
        plt.figure(figsize=(7, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="RdPu", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "maternal_correlation_heatmap.png"), dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/maternal_correlation_heatmap.png")

    return df


def run_anemia_eda():
    print("\n" + "=" * 60)
    print("ANEMIA / CBC DATASET — EDA")
    print("=" * 60)

    if not check_file_exists(ANEMIA_CSV, "Anemia/CBC dataset"):
        return None

    df = pd.read_csv(ANEMIA_CSV)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nSummary statistics:\n{df.describe()}")

    numeric_cols = df.select_dtypes(include="number").columns
    n = len(numeric_cols)
    if n > 0:
        fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3, figsize=(14, 3.5 * ((n + 2) // 3)))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color="#D9799E")
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "anemia_feature_histograms.png"), dpi=150)
        plt.close()
        print(f"Saved: {OUT_DIR}/anemia_feature_histograms.png")

    return df


if __name__ == "__main__":
    maternal_df = run_maternal_eda()
    anemia_df = run_anemia_eda()

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    if maternal_df is not None and anemia_df is not None:
        print(f"All plots saved to: {OUT_DIR}/")
        print("\nNext step: Phase 2 — Baseline Classifier (src/models/baseline.py)")
    else:
        print("Some datasets were missing — download them first, then re-run this script.")
