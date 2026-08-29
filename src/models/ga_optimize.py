"""
Phase 3 — Genetic Algorithm Optimization
===========================================
Uses a Genetic Algorithm (via DEAP) to search for better hyperparameters for
Random Forest and XGBoost than the Phase 2 baseline (default settings),
then compares GA-optimized vs. baseline performance on the same test set.

How the GA works (see project write-up / slides for the full explanation):
  1. Each "individual" is a vector of values in [0,1], one per hyperparameter.
  2. decode_rf()/decode_xgb() map those [0,1] values to real hyperparameter
     ranges (e.g. n_estimators between 50-300).
  3. Fitness = mean weighted F1 score from 5-fold cross-validation on the
     TRAINING set only (never the test set, to avoid leaking test info into
     model selection).
  4. Each generation: evaluate fitness -> select the fittest -> crossover
     (blend pairs) -> mutate (small random tweaks) -> repeat.
  5. The best individual found is decoded, trained on the full training set,
     and evaluated once on the held-out test set for a fair final comparison.

Usage:
    python src/models/ga_optimize.py

Requires Phase 2 to have been run first (uses the same data loading logic
and compares against models/baseline_random_forest.joblib /
models/baseline_xgboost.joblib if present).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from deap import base, creator, tools

# Reuse the exact data loading / trimester simulation logic from Phase 2
# so both phases are evaluated on identical train/test splits.
from baseline import load_data

# ---- Paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

# ---- GA settings ----
# Widened search (regularization, class balancing) needs a bit more room to
# converge than the initial version — these settings do a more thorough
# search than the first pass. Still adjustable if runtime is a concern.
POP_SIZE = 40
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3
CV_FOLDS = 5
RANDOM_SEED = 42

# Cross-validated final comparison settings (more robust than a single
# 80/20 split, especially with a small test set where accuracy can only
# move in coarse steps).
FINAL_CV_SPLITS = 5
FINAL_CV_REPEATS = 5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# Hyperparameter decoding: [0,1]-valued genes -> real params
# =========================================================

def decode_rf(individual):
    # Focused search around the strongest region already observed in the baseline/
    # class-balanced medical-risk setting, rather than the full broad space.
    n_estimators = int(round(120 + individual[0] * (260 - 120)))
    max_depth = int(round(8 + individual[1] * (28 - 8)))
    min_samples_split = int(round(2 + individual[2] * (8 - 2)))
    min_samples_leaf = int(round(1 + individual[3] * (3 - 1)))
    max_features_options = ["sqrt", "log2", None]
    idx = min(int(individual[4] * len(max_features_options)), len(max_features_options) - 1)
    max_features = max_features_options[idx]
    class_weight_options = [None, "balanced", "balanced_subsample"]
    cw_idx = min(int(individual[5] * len(class_weight_options)), len(class_weight_options) - 1)
    class_weight = class_weight_options[cw_idx]
    criterion_options = ["gini", "entropy"]
    crit_idx = min(int(individual[6] * len(criterion_options)), len(criterion_options) - 1)
    criterion = criterion_options[crit_idx]
    return dict(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, class_weight=class_weight, criterion=criterion,
        random_state=RANDOM_SEED,
    )


def decode_xgb(individual):
    # Search a tighter region around proven high-performing XGBoost settings for
    # moderately sized tabular data with class imbalance and strong regularization.
    n_estimators = int(round(100 + individual[0] * (400 - 100)))
    max_depth = int(round(3 + individual[1] * (12 - 3)))
    learning_rate = round(0.05 + individual[2] * (0.25 - 0.05), 4)
    subsample = round(0.7 + individual[3] * (1.0 - 0.7), 3)
    colsample_bytree = round(0.6 + individual[4] * (1.0 - 0.6), 3)
    reg_alpha = round(individual[5] * 2.0, 3)          # 0-2: mild L1 regularization
    reg_lambda = round(0.5 + individual[6] * (4.0 - 0.5), 3)  # 0.5-4.0: moderate L2
    min_child_weight = int(round(1 + individual[7] * (8 - 1)))
    return dict(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda, min_child_weight=min_child_weight,
        eval_metric="mlogloss", random_state=RANDOM_SEED,
    )


# =========================================================
# DEAP setup
# =========================================================

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def make_toolbox(n_genes, fitness_fn):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def clip_individual(ind):
    for i in range(len(ind)):
        ind[i] = min(max(ind[i], 0.0), 1.0)
    return ind


def run_ga(decode_fn, model_builder, X_train, y_train, n_genes, label):
    """Runs the GA and returns (best_individual, best_params, fitness_history)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def fitness_fn(individual):
        params = decode_fn(individual)
        model = model_builder(params)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
            return (scores.mean(),)
        except Exception:
            return (0.0,)  # invalid hyperparameter combo -> worst fitness

    toolbox = make_toolbox(n_genes, fitness_fn)
    population = toolbox.population(n=POP_SIZE)

    print(f"\n{'=' * 60}\nGA OPTIMIZATION — {label}\n{'=' * 60}")
    print(f"Population: {POP_SIZE}  |  Generations: {N_GEN}  |  CV folds: {CV_FOLDS}")

    fitness_history = []
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    best_so_far = max(population, key=lambda ind: ind.fitness.values[0])
    fitness_history.append(best_so_far.fitness.values[0])
    print(f"Generation 0 | Best fitness (weighted F1): {best_so_far.fitness.values[0]:.4f}")

    for gen in range(1, N_GEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                clip_individual(child1)
                clip_individual(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                clip_individual(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        best_so_far = max(population, key=lambda ind: ind.fitness.values[0])
        fitness_history.append(best_so_far.fitness.values[0])

        if gen % 5 == 0 or gen == N_GEN:
            print(f"Generation {gen} | Best fitness (weighted F1): {best_so_far.fitness.values[0]:.4f}")

    best_individual = max(population, key=lambda ind: ind.fitness.values[0])
    best_params = decode_fn(best_individual)
    print(f"\nBest hyperparameters found for {label}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_individual, best_params, fitness_history


def plot_convergence(history, label, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history, color="#9C3B63", linewidth=2, marker="o", markersize=3)
    plt.title(f"GA Convergence — {label}")
    plt.xlabel("Generation")
    plt.ylabel("Best Weighted F1 (CV)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def evaluate_and_save(name, model, X_test, y_test, class_names):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_weighted = f1_score(y_test, preds, average="weighted")
    recall_high = recall_score(y_test, preds, labels=[2], average="macro", zero_division=0)

    print(f"\n--- {name} (test set) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"High-Risk Recall: {recall_high:.4f}")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(OUT_DIR, f"confusion_matrix_{fname}.png"), dpi=150)
    plt.close()

    return {"accuracy": acc, "f1_weighted": f1_weighted, "high_risk_recall": recall_high}


def load_baseline_metrics():
    """Loads baseline models saved by Phase 2, re-evaluates them for a fair side-by-side."""
    rf_path = os.path.join(MODELS_DIR, "baseline_random_forest.joblib")
    xgb_path = os.path.join(MODELS_DIR, "baseline_xgboost.joblib")
    if not (os.path.exists(rf_path) and os.path.exists(xgb_path)):
        print("\n[WARNING] Baseline models not found in models/. Run Phase 2 (src/models/baseline.py) first "
              "for a proper baseline-vs-GA comparison. Continuing with GA-only results.")
        return None, None
    return joblib.load(rf_path), joblib.load(xgb_path)


def cross_validated_comparison(baseline_builder, ga_builder, X, y, label):
    """
    Compares baseline (default hyperparams) vs. GA-optimized model using
    Repeated Stratified K-Fold cross-validation on the FULL dataset, rather
    than a single 80/20 split. This is much less sensitive to the coarse
    quantization of a small test set (e.g. 203 samples => accuracy can only
    move in ~0.5% steps), and gives a mean +/- std across many folds instead
    of a single number.

    A paired t-test on the fold-wise accuracy scores (same folds for both
    models, since they share the same cv object/seed) checks whether any
    observed difference is statistically meaningful or just noise.
    """
    cv = RepeatedStratifiedKFold(n_splits=FINAL_CV_SPLITS, n_repeats=FINAL_CV_REPEATS, random_state=RANDOM_SEED)

    print(f"\n--- Cross-validated comparison — {label} "
          f"({FINAL_CV_SPLITS}-fold x {FINAL_CV_REPEATS} repeats = {FINAL_CV_SPLITS * FINAL_CV_REPEATS} folds) ---")

    baseline_scores = cross_val_score(baseline_builder(), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    ga_scores = cross_val_score(ga_builder(), X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print(f"Baseline accuracy:      {baseline_scores.mean():.4f} +/- {baseline_scores.std():.4f}")
    print(f"GA-optimized accuracy:  {ga_scores.mean():.4f} +/- {ga_scores.std():.4f}")
    print(f"Mean improvement:       {ga_scores.mean() - baseline_scores.mean():+.4f}")

    t_stat, p_value = stats.ttest_rel(ga_scores, baseline_scores)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}", end="  ")
    if p_value < 0.05:
        direction = "better" if ga_scores.mean() > baseline_scores.mean() else "worse"
        print(f"-> Statistically significant difference (GA is {direction}, p < 0.05)")
    else:
        print("-> Not statistically significant (p >= 0.05) — cannot conclude a real difference "
              "beyond random fold-to-fold variation.")

    return {
        "baseline_cv_mean": float(baseline_scores.mean()),
        "baseline_cv_std": float(baseline_scores.std()),
        "ga_cv_mean": float(ga_scores.mean()),
        "ga_cv_std": float(ga_scores.std()),
        "mean_improvement": float(ga_scores.mean() - baseline_scores.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def main():
    print("=" * 60)
    print("PHASE 3 — GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 60)

    X, y, trimester, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    baseline_rf, baseline_xgb = load_baseline_metrics()

    results = {}

    # ---- GA-optimize Random Forest ----
    _, rf_params, rf_history = run_ga(
        decode_rf,
        lambda p: RandomForestClassifier(**p),
        X_train, y_train, n_genes=7, label="Random Forest"
    )
    plot_convergence(rf_history, "Random Forest", "ga_convergence_random_forest.png")
    ga_rf = RandomForestClassifier(**rf_params)
    ga_rf.fit(X_train, y_train)
    results["ga_random_forest"] = evaluate_and_save("GA-Optimized Random Forest", ga_rf, X_test, y_test, class_names)
    results["ga_random_forest"]["best_params"] = rf_params
    joblib.dump(ga_rf, os.path.join(MODELS_DIR, "ga_optimized_random_forest.joblib"))

    # ---- GA-optimize XGBoost ----
    _, xgb_params, xgb_history = run_ga(
        decode_xgb,
        lambda p: XGBClassifier(**p),
        X_train, y_train, n_genes=8, label="XGBoost"
    )
    plot_convergence(xgb_history, "XGBoost", "ga_convergence_xgboost.png")
    ga_xgb = XGBClassifier(**xgb_params)
    ga_xgb.fit(X_train, y_train)
    results["ga_xgboost"] = evaluate_and_save("GA-Optimized XGBoost", ga_xgb, X_test, y_test, class_names)
    results["ga_xgboost"]["best_params"] = xgb_params
    joblib.dump(ga_xgb, os.path.join(MODELS_DIR, "ga_optimized_xgboost.joblib"))

    # ---- Baseline vs GA comparison ----
    print("\n" + "=" * 60)
    print("BASELINE vs. GA-OPTIMIZED COMPARISON")
    print("=" * 60)

    comparison = {}
    if baseline_rf is not None:
        base_metrics = evaluate_and_save("Baseline Random Forest (recheck)", baseline_rf, X_test, y_test, class_names)
        comparison["random_forest"] = {
            "baseline_accuracy": base_metrics["accuracy"],
            "ga_accuracy": results["ga_random_forest"]["accuracy"],
            "accuracy_improvement": results["ga_random_forest"]["accuracy"] - base_metrics["accuracy"],
            "baseline_high_risk_recall": base_metrics["high_risk_recall"],
            "ga_high_risk_recall": results["ga_random_forest"]["high_risk_recall"],
        }
        print(f"\nRandom Forest: baseline={base_metrics['accuracy']:.4f} -> "
              f"GA-optimized={results['ga_random_forest']['accuracy']:.4f} "
              f"({comparison['random_forest']['accuracy_improvement']:+.4f})")

    if baseline_xgb is not None:
        base_metrics = evaluate_and_save("Baseline XGBoost (recheck)", baseline_xgb, X_test, y_test, class_names)
        comparison["xgboost"] = {
            "baseline_accuracy": base_metrics["accuracy"],
            "ga_accuracy": results["ga_xgboost"]["accuracy"],
            "accuracy_improvement": results["ga_xgboost"]["accuracy"] - base_metrics["accuracy"],
            "baseline_high_risk_recall": base_metrics["high_risk_recall"],
            "ga_high_risk_recall": results["ga_xgboost"]["high_risk_recall"],
        }
        print(f"XGBoost:       baseline={base_metrics['accuracy']:.4f} -> "
              f"GA-optimized={results['ga_xgboost']['accuracy']:.4f} "
              f"({comparison['xgboost']['accuracy_improvement']:+.4f})")

    results["comparison"] = comparison

    # ---- Cross-validated comparison (more robust than the single split above) ----
    print("\n" + "=" * 60)
    print("CROSS-VALIDATED COMPARISON (more robust than a single 80/20 split)")
    print("=" * 60)

    cv_comparison = {}
    cv_comparison["random_forest"] = cross_validated_comparison(
        lambda: RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        lambda: RandomForestClassifier(**rf_params),
        X, y, "Random Forest"
    )
    cv_comparison["xgboost"] = cross_validated_comparison(
        lambda: XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_SEED),
        lambda: XGBClassifier(**xgb_params),
        X, y, "XGBoost"
    )
    results["cv_comparison"] = cv_comparison

    summary_path = os.path.join(OUT_DIR, "phase3_ga_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFull results saved to: {summary_path}")
    print(f"Convergence plots + confusion matrices saved to: {OUT_DIR}/")
    print(f"GA-optimized models saved to: {MODELS_DIR}/")
    print("\nNext step: Phase 4 — Longitudinal Trajectory Simulation (src/data/trajectory_simulation.py)")


if __name__ == "__main__":
    main()