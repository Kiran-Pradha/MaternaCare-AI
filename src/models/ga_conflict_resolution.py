"""
Phase 3d — GA-Based Automated Label-Conflict Resolution
============================================================
Phase 3b identified that 21.2% of the dataset consists of records sharing
identical feature values but conflicting risk labels — an irreducible source
of error for any classifier. This script goes a step further: rather than
just reporting the conflicts, a Genetic Algorithm LEARNS what to do about
each conflicting group.

This is a genuinely new algorithm, not a repeat of Phase 3's GA-tuned
classifier or Phase 5's GA-tuned clustering: it is purpose-built to resolve
a data-quality problem this project itself discovered, and no cited paper
performs this specific operation on this dataset.

Chromosome design:
  - One gene PER CONFLICTING GROUP (not per row) — e.g. 35 genes for 35
    conflicting groups, not 215 genes for 215 rows. This keeps the search
    space small (3^35 instead of 2^215) and every gene individually
    interpretable in the final report.
  - Each gene takes one of three actions for its group:
        0 = DROP the entire group from training
        1 = KEEP MAJORITY ONLY — remove the minority-labeled rows within
            that group, keep the majority-labeled ones
        2 = KEEP AS-IS — no change (status quo)

Fitness function:
  - Standard k-fold cross-validation. For each fold, the chosen actions are
    applied ONLY to that fold's TRAINING rows — the test fold is always left
    untouched with its real, unmodified labels, since that mirrors what the
    model will face at real inference time.
  - Fitness = mean cross-validated weighted F1 across folds, with a small
    penalty proportional to how much data is discarded (mainly to break
    ties in favor of retaining more data, not to dominate the objective).

Comparison: the GA-optimized curation strategy is benchmarked against two
naive extremes — keep everything (Phase 2 baseline) and drop all conflicting
groups entirely — to test whether the GA found something genuinely better
than either simple extreme.

Usage:
    python src/models/ga_conflict_resolution.py
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from deap import base, creator, tools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import load_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3d_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

RANDOM_SEED = 42
POP_SIZE = 30
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3
GA_CV_FOLDS = 3          # lighter CV during the GA search itself, for speed
FINAL_CV_SPLITS = 5      # full rigor for the final reported comparison
FINAL_CV_REPEATS = 5

DROP, KEEP_MAJORITY, KEEP_AS_IS = 0, 1, 2
ACTION_NAMES = {DROP: "Drop group", KEEP_MAJORITY: "Keep majority only", KEEP_AS_IS: "Keep as-is"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def identify_conflict_groups(X, y):
    """
    Finds groups of rows with identical feature values but >1 distinct label.
    Returns:
      group_id: array, length n_samples, -1 if not in a conflicting group,
                otherwise an index into `groups`.
      groups: list of dicts, one per conflicting group, with member row
              indices and each label's count.
    """
    df = X.copy()
    df["_label"] = y.values
    df["_row"] = np.arange(len(df))

    feature_cols = [c for c in df.columns if c not in ("_label", "_row")]
    grouped = df.groupby(feature_cols)

    group_id = np.full(len(df), -1, dtype=int)
    groups = []
    for _, sub in grouped:
        if sub["_label"].nunique() > 1:
            gid = len(groups)
            member_rows = sub["_row"].values
            label_counts = sub["_label"].value_counts().to_dict()
            majority_label = sub["_label"].value_counts().idxmax()
            groups.append({
                "gid": gid, "rows": member_rows.tolist(),
                "label_counts": label_counts, "majority_label": int(majority_label),
            })
            group_id[member_rows] = gid

    return group_id, groups


def curate_training_set(train_idx, y, group_id, actions):
    """Applies actions with fold-local majority-label computation (no leakage)."""
    train_idx = np.array(train_idx)
    y_train_all = y.values[train_idx] if hasattr(y, "values") else y[train_idx]

    group_to_local_rows = {}
    for pos, idx in enumerate(train_idx):
        g = group_id[idx]
        if g == -1:
            continue
        group_to_local_rows.setdefault(g, []).append((idx, y_train_all[pos]))

    fold_majority = {}
    for g, members in group_to_local_rows.items():
        labels = [lbl for _, lbl in members]
        if labels:
            vals, counts = np.unique(labels, return_counts=True)
            fold_majority[g] = vals[np.argmax(counts)]

    kept = []
    for pos, idx in enumerate(train_idx):
        g = group_id[idx]
        if g == -1:
            kept.append(idx)
            continue
        action = actions[g]
        if action == DROP:
            continue
        elif action == KEEP_MAJORITY:
            maj = fold_majority.get(g)
            if maj is not None and y_train_all[pos] == maj:
                kept.append(idx)
        else:  # KEEP_AS_IS
            kept.append(idx)

    return np.array(kept, dtype=int)


if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def run_ga(X, y, group_id, n_groups):
    cv = StratifiedKFold(n_splits=GA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    def fitness_fn(individual):
        actions = list(individual)
        fold_scores = []
        n_dropped_total = 0
        n_train_total = 0

        for train_idx, test_idx in cv.split(X_arr, y_arr):
            curated_idx = curate_training_set(train_idx, y, group_id, actions)
            n_dropped_total += len(train_idx) - len(curated_idx)
            n_train_total += len(train_idx)

            if len(curated_idx) < 20 or len(np.unique(y_arr[curated_idx])) < 3:
                fold_scores.append(0.0)
                continue

            clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
            clf.fit(X_arr[curated_idx], y_arr[curated_idx])
            preds = clf.predict(X_arr[test_idx])
            fold_scores.append(f1_score(y_arr[test_idx], preds, average="weighted"))

        mean_f1 = np.mean(fold_scores)
        drop_fraction = n_dropped_total / max(n_train_total, 1)
        penalty = 0.02 * drop_fraction
        return (mean_f1 - penalty,)

    toolbox = base.Toolbox()
    toolbox.register("attr_action", random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_action, n=n_groups)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    history = [max(population, key=lambda i: i.fitness.values[0]).fitness.values[0]]
    print(f"\n{'=' * 60}\nGA SEARCH — Label Conflict Resolution ({n_groups} groups)\n{'=' * 60}")
    print(f"Generation 0 | Best fitness: {history[0]:.4f}")

    for gen in range(1, N_GEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        best = max(population, key=lambda i: i.fitness.values[0])
        history.append(best.fitness.values[0])
        if gen % 5 == 0 or gen == N_GEN:
            print(f"Generation {gen} | Best fitness: {best.fitness.values[0]:.4f}")

    best_individual = list(max(population, key=lambda i: i.fitness.values[0]))

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history, color="#9C3B63", linewidth=2, marker="o", markersize=3)
    plt.title("GA Convergence — Label Conflict Resolution")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (CV Weighted F1 minus data-loss penalty)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ga_convergence_conflict_resolution.png"), dpi=150)
    plt.close()

    return best_individual, history


def evaluate_strategy(name, actions, X, y, group_id):
    """Full rigor: 5x5 repeated CV comparison for a given action strategy."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    X_arr = X.values if hasattr(X, "values") else X
    y_arr = y.values if hasattr(y, "values") else y

    cv = RepeatedStratifiedKFold(n_splits=FINAL_CV_SPLITS, n_repeats=FINAL_CV_REPEATS, random_state=RANDOM_SEED)
    scores = []
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        curated_idx = curate_training_set(train_idx, y, group_id, actions) if actions is not None else train_idx
        if len(curated_idx) < 20 or len(np.unique(y_arr[curated_idx])) < 3:
            scores.append(0.0)
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        clf.fit(X_arr[curated_idx], y_arr[curated_idx])
        preds = clf.predict(X_arr[test_idx])
        scores.append(accuracy_score(y_arr[test_idx], preds))

    scores = np.array(scores)
    print(f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores


def main():
    print("=" * 60)
    print("PHASE 3d — GA-BASED LABEL CONFLICT RESOLUTION")
    print("=" * 60)

    X, y, trimester, class_names = load_data()
    group_id, groups = identify_conflict_groups(X, y)
    n_groups = len(groups)
    n_conflicting_rows = int((group_id != -1).sum())

    print(f"\nDataset: {len(X)} rows, {X.shape[1]} features")
    print(f"Conflicting groups found: {n_groups}")
    print(f"Rows involved in conflicts: {n_conflicting_rows} ({100*n_conflicting_rows/len(X):.1f}%)")

    if n_groups == 0:
        print("\nNo conflicting groups found in this dataset — nothing for the GA to resolve. Exiting.")
        return

    best_actions, history = run_ga(X, y, group_id, n_groups)

    action_counts = {DROP: 0, KEEP_MAJORITY: 0, KEEP_AS_IS: 0}
    for a in best_actions:
        action_counts[a] += 1
    print(f"\nGA-found strategy — action breakdown across {n_groups} groups:")
    for action, count in action_counts.items():
        print(f"  {ACTION_NAMES[action]}: {count} groups ({100*count/n_groups:.1f}%)")

    print("\n" + "=" * 60)
    print("FINAL COMPARISON — 5x5 Repeated Cross-Validation")
    print("=" * 60)

    baseline_actions = [KEEP_AS_IS] * n_groups
    drop_all_actions = [DROP] * n_groups

    baseline_scores = evaluate_strategy("Baseline (keep everything)", baseline_actions, X, y, group_id)
    drop_all_scores = evaluate_strategy("Naive (drop all conflicting groups)", drop_all_actions, X, y, group_id)
    ga_scores = evaluate_strategy("GA-optimized curation", best_actions, X, y, group_id)

    print(f"\nGA vs. Baseline:  {ga_scores.mean() - baseline_scores.mean():+.4f}")
    print(f"GA vs. Drop-all:  {ga_scores.mean() - drop_all_scores.mean():+.4f}")

    t_base, p_base = stats.ttest_rel(ga_scores, baseline_scores)
    t_drop, p_drop = stats.ttest_rel(ga_scores, drop_all_scores)
    print(f"\nPaired t-test GA vs. Baseline:  t={t_base:.3f}, p={p_base:.4f}"
          f"  {'(significant)' if p_base < 0.05 else '(not significant)'}")
    print(f"Paired t-test GA vs. Drop-all:  t={t_drop:.3f}, p={p_drop:.4f}"
          f"  {'(significant)' if p_drop < 0.05 else '(not significant)'}")

    plt.figure(figsize=(7, 5))
    labels = ["Baseline\n(keep all)", "Naive\n(drop all conflicts)", "GA-optimized\ncuration"]
    means = [baseline_scores.mean(), drop_all_scores.mean(), ga_scores.mean()]
    stds = [baseline_scores.std(), drop_all_scores.std(), ga_scores.std()]
    colors = ["#D9799E", "#F2AFC6", "#9C3B63"]
    plt.bar(labels, means, yerr=stds, color=colors, capsize=5, edgecolor="white")
    plt.ylabel("Cross-Validated Accuracy")
    plt.title("GA-Optimized Label-Conflict Resolution vs. Naive Strategies")
    for i, m in enumerate(means):
        plt.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "conflict_resolution_comparison.png"), dpi=150)
    plt.close()

    results = {
        "n_groups": n_groups,
        "n_conflicting_rows": n_conflicting_rows,
        "action_breakdown": {ACTION_NAMES[k]: v for k, v in action_counts.items()},
        "best_actions": best_actions,
        "baseline_cv": {"mean": float(baseline_scores.mean()), "std": float(baseline_scores.std())},
        "drop_all_cv": {"mean": float(drop_all_scores.mean()), "std": float(drop_all_scores.std())},
        "ga_optimized_cv": {"mean": float(ga_scores.mean()), "std": float(ga_scores.std())},
        "ga_vs_baseline": {"t_statistic": float(t_base), "p_value": float(p_base), "significant": bool(p_base < 0.05)},
        "ga_vs_drop_all": {"t_statistic": float(t_drop), "p_value": float(p_drop), "significant": bool(p_drop < 0.05)},
    }

    summary_path = os.path.join(OUT_DIR, "phase3d_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    joblib.dump({"group_id": group_id, "groups": groups, "best_actions": best_actions},
                os.path.join(MODELS_DIR, "ga_conflict_resolution.joblib"))

    print(f"\nFull results saved to: {summary_path}")
    print(f"Plots saved to: {OUT_DIR}/")
    print("\nNext step: back to Phase 6 - Explainability (SHAP), or continue with remaining novelty additions.")


if __name__ == "__main__":
    main()