"""
Phase 3c — GA-Optimized Ensemble Blending
=============================================
Individual hyperparameter tuning (Phase 3) plateaued at the same accuracy
regardless of configuration — the ceiling diagnostic (Phase 3b) showed this
is largely due to genuine label ambiguity in the data, but also revealed
~7 percentage points of real headroom between the baseline accuracy and the
theoretical maximum. This script tries to close some of that gap using a
different mechanism: blending two DIFFERENT model families instead of tuning
one at a time.

How this works:
  1. Load the GA-optimized Random Forest and XGBoost (from Phase 3).
  2. Get out-of-fold predicted probabilities for both models on the training
     set (via cross_val_predict) — this avoids overfitting the blend weights
     to data the models have already seen.
  3. A Genetic Algorithm searches for the best PER-CLASS blend weight
     (e.g. maybe RF is more reliable for "Low Risk" while XGBoost is more
     reliable for "High Risk" — the optimal mix can differ by class), rather
     than a single blanket weight for all classes.
  4. The final blend is evaluated on the held-out test set, compared against
     each model alone and a naive unweighted 50/50 blend, and validated with
     repeated cross-validation for robustness.

Usage:
    python src/models/ga_ensemble.py

Requires Phase 2 (baseline.py) and ideally Phase 3 (ga_optimize.py) to have
been run first. Falls back gracefully to default hyperparameters if GA
results aren't found.
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.model_selection import (
    train_test_split, cross_val_predict, RepeatedStratifiedKFold, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from xgboost import XGBClassifier
from deap import base, creator, tools

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import load_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

RANDOM_SEED = 42
POP_SIZE = 30
N_GEN = 20
CX_PROB = 0.6
MUT_PROB = 0.3
CV_FOLDS = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_best_params():
    """Loads GA-optimized hyperparameters from Phase 3 results if available,
    otherwise falls back to sensible defaults with a warning."""
    results_path = os.path.join(OUT_DIR, "phase3_ga_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        rf_params = results.get("ga_random_forest", {}).get("best_params")
        xgb_params = results.get("ga_xgboost", {}).get("best_params")
        if rf_params and xgb_params:
            print("Loaded GA-optimized hyperparameters from Phase 3 results.")
            return rf_params, xgb_params

    print("[WARNING] Phase 3 GA results not found — using default hyperparameters instead. "
          "Run src/models/ga_optimize.py first for the full hybrid pipeline.")
    return (
        dict(n_estimators=100, random_state=RANDOM_SEED),
        dict(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_SEED),
    )


def build_rf(params):
    clean = {k: v for k, v in params.items() if k != "eval_metric"}
    return RandomForestClassifier(**clean)


def build_xgb(params):
    return XGBClassifier(**params)


# ---- DEAP setup for per-class blend weight search ----
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def clip_individual(ind):
    for i in range(len(ind)):
        ind[i] = min(max(ind[i], 0.0), 1.0)
    return ind


def blend_proba(rf_proba, xgb_proba, weights):
    """weights: array of length n_classes, one blend ratio per class."""
    blended = np.zeros_like(rf_proba)
    for c in range(rf_proba.shape[1]):
        blended[:, c] = weights[c] * rf_proba[:, c] + (1 - weights[c]) * xgb_proba[:, c]
    return blended


def run_blend_ga(oof_rf_proba, oof_xgb_proba, y_train, n_classes):
    def fitness_fn(individual):
        blended = blend_proba(oof_rf_proba, oof_xgb_proba, individual)
        preds = np.argmax(blended, axis=1)
        return (f1_score(y_train, preds, average="weighted"),)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_classes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    history = [max(population, key=lambda i: i.fitness.values[0]).fitness.values[0]]

    print(f"\nGeneration 0 | Best fitness (weighted F1): {history[0]:.4f}")

    for gen in range(1, N_GEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(c1, c2)
                clip_individual(c1)
                clip_individual(c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(m)
                clip_individual(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        best = max(population, key=lambda i: i.fitness.values[0])
        history.append(best.fitness.values[0])
        if gen % 5 == 0 or gen == N_GEN:
            print(f"Generation {gen} | Best fitness (weighted F1): {best.fitness.values[0]:.4f}")

    best_individual = max(population, key=lambda i: i.fitness.values[0])
    return list(best_individual), history


def main():
    print("=" * 60)
    print("PHASE 3c — GA-OPTIMIZED ENSEMBLE BLENDING")
    print("=" * 60)

    X, y, trimester, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    n_classes = len(class_names)

    rf_params, xgb_params = load_best_params()

    # ---- Out-of-fold probabilities (honest, non-leaked signal for weight search) ----
    print("\nComputing out-of-fold predictions for blend-weight search...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_rf_proba = cross_val_predict(build_rf(rf_params), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)
    oof_xgb_proba = cross_val_predict(build_xgb(xgb_params), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)

    print(f"\n{'=' * 60}\nGA SEARCH — Per-Class Blend Weights\n{'=' * 60}")
    best_weights, history = run_blend_ga(oof_rf_proba, oof_xgb_proba, y_train.values, n_classes)
    print(f"\nBest per-class blend weights found (weight = RF share, 1-weight = XGBoost share):")
    for cname, w in zip(class_names, best_weights):
        print(f"  {cname}: {w:.3f} RF / {1-w:.3f} XGBoost")

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history, color="#9C3B63", linewidth=2, marker="o", markersize=3)
    plt.title("GA Convergence — Ensemble Blend Weights")
    plt.xlabel("Generation")
    plt.ylabel("Best Weighted F1 (OOF)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ga_convergence_ensemble_blend.png"), dpi=150)
    plt.close()

    # ---- Final models trained on full training set ----
    final_rf = build_rf(rf_params)
    final_rf.fit(X_train, y_train)
    final_xgb = build_xgb(xgb_params)
    final_xgb.fit(X_train, y_train)

    test_rf_proba = final_rf.predict_proba(X_test)
    test_xgb_proba = final_xgb.predict_proba(X_test)

    def eval_preds(name, preds):
        acc = accuracy_score(y_test, preds)
        f1w = f1_score(y_test, preds, average="weighted")
        recall_high = recall_score(y_test, preds, labels=[2], average="macro", zero_division=0)
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}  |  Weighted F1: {f1w:.4f}  |  High-Risk Recall: {recall_high:.4f}")
        return {"accuracy": acc, "f1_weighted": f1w, "high_risk_recall": recall_high}

    print("\n" + "=" * 60)
    print("TEST SET COMPARISON")
    print("=" * 60)

    results = {}
    results["rf_alone"] = eval_preds("Random Forest alone (GA-tuned)", np.argmax(test_rf_proba, axis=1))
    results["xgb_alone"] = eval_preds("XGBoost alone (GA-tuned)", np.argmax(test_xgb_proba, axis=1))

    naive_blend = 0.5 * test_rf_proba + 0.5 * test_xgb_proba
    results["naive_50_50_blend"] = eval_preds("Naive 50/50 blend", np.argmax(naive_blend, axis=1))

    ga_blend = blend_proba(test_rf_proba, test_xgb_proba, best_weights)
    results["ga_optimized_blend"] = eval_preds("GA-optimized per-class blend", np.argmax(ga_blend, axis=1))
    results["ga_optimized_blend"]["weights_per_class"] = dict(zip(class_names, [float(w) for w in best_weights]))

    best_single = max(results["rf_alone"]["accuracy"], results["xgb_alone"]["accuracy"])
    improvement = results["ga_optimized_blend"]["accuracy"] - best_single
    print(f"\nGA blend vs. best individual model: {improvement:+.4f}")

    # ---- Robustness check via repeated cross-validation ----
    print("\n" + "=" * 60)
    print("CROSS-VALIDATED ROBUSTNESS CHECK")
    print("=" * 60)
    print("(Retrains fresh RF/XGB per fold, applies the GA-found weights, "
          "checks whether the blend's advantage generalizes beyond this one split)")

    cv_robust = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    blend_scores, rf_scores, xgb_scores = [], [], []

    for train_idx, test_idx in cv_robust.split(X, y):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        rf_fold = build_rf(rf_params)
        rf_fold.fit(Xtr, ytr)
        xgb_fold = build_xgb(xgb_params)
        xgb_fold.fit(Xtr, ytr)

        rf_p = rf_fold.predict_proba(Xte)
        xgb_p = xgb_fold.predict_proba(Xte)
        blend_p = blend_proba(rf_p, xgb_p, best_weights)

        blend_scores.append(accuracy_score(yte, np.argmax(blend_p, axis=1)))
        rf_scores.append(accuracy_score(yte, np.argmax(rf_p, axis=1)))
        xgb_scores.append(accuracy_score(yte, np.argmax(xgb_p, axis=1)))

    blend_scores, rf_scores, xgb_scores = map(np.array, (blend_scores, rf_scores, xgb_scores))
    best_individual_scores = np.maximum(rf_scores, xgb_scores)

    print(f"RF alone:        {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")
    print(f"XGBoost alone:   {xgb_scores.mean():.4f} +/- {xgb_scores.std():.4f}")
    print(f"GA blend:        {blend_scores.mean():.4f} +/- {blend_scores.std():.4f}")

    t_stat, p_value = stats.ttest_rel(blend_scores, best_individual_scores)
    print(f"\nPaired t-test (blend vs. best-individual-per-fold): t={t_stat:.3f}, p={p_value:.4f}", end="  ")
    significant = p_value < 0.05 and blend_scores.mean() > best_individual_scores.mean()
    if significant:
        print("-> Statistically significant improvement from ensembling.")
    else:
        print("-> Not a statistically significant improvement.")

    results["cv_robustness"] = {
        "rf_cv_mean": float(rf_scores.mean()), "rf_cv_std": float(rf_scores.std()),
        "xgb_cv_mean": float(xgb_scores.mean()), "xgb_cv_std": float(xgb_scores.std()),
        "blend_cv_mean": float(blend_scores.mean()), "blend_cv_std": float(blend_scores.std()),
        "t_statistic": float(t_stat), "p_value": float(p_value), "significant": bool(significant),
    }

    # ---- Comparison bar chart ----
    plt.figure(figsize=(7, 5))
    labels = ["RF alone", "XGBoost alone", "Naive 50/50", "GA-optimized blend"]
    values = [results["rf_alone"]["accuracy"], results["xgb_alone"]["accuracy"],
              results["naive_50_50_blend"]["accuracy"], results["ga_optimized_blend"]["accuracy"]]
    colors = ["#D9799E", "#D9799E", "#FBE1EB", "#9C3B63"]
    bars = plt.bar(labels, values, color=colors, edgecolor="white")
    plt.axhline(y=best_single, color="gray", linestyle="--", linewidth=1, label="Best individual model")
    plt.ylim(min(values) - 0.05, max(values) + 0.05)
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison — Individual vs. Ensemble Blend")
    plt.legend()
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ensemble_comparison.png"), dpi=150)
    plt.close()

    summary_path = os.path.join(OUT_DIR, "phase3c_ensemble_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump({"rf": final_rf, "xgb": final_xgb, "weights": best_weights},
                os.path.join(MODELS_DIR, "ga_ensemble_blend.joblib"))

    print(f"\nFull results saved to: {summary_path}")
    print(f"Plots saved to: {OUT_DIR}/")
    print(f"Ensemble model saved to: {MODELS_DIR}/ga_ensemble_blend.joblib")
    print("\nNext step: Phase 4 — Longitudinal Trajectory Simulation (src/data/trajectory_simulation.py)")


if __name__ == "__main__":
    main()