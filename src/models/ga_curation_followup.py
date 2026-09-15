"""
Phase 3e -- Follow-Up: Does Conflict-Resolution Curation Help GA-Tuned Classifiers
And the Simulated Longitudinal Dataset?
======================================================================================
Two questions, answered empirically:

  1. Phase 3 (GA hyperparameter tuning) plateaued at 85.22% on the ORIGINAL
     (uncurated) data. Phase 3d (conflict-resolution curation) reached 85.90%
     using a DEFAULT (non-GA-tuned) Random Forest. This script asks: if we
     GA-tune a classifier on the CURATED data, does it beat both?

  2. Phase 4 (longitudinal trajectory simulation) found that adding trend
     features significantly HURT risk classification (-10.2%, p=5.6e-16),
     traced to the same label-conflict mechanism. This script asks: if we
     curate the base data first, does the trend-feature penalty shrink or
     disappear?

Self-contained script (no cross-file imports), so it can be run standalone.

Usage:
    python src/models/ga_curation_followup.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from deap import base, creator, tools
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3e_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")

sns.set_style("whitegrid")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# Shared data loading + conflict identification
# =========================================================

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


def identify_conflict_groups(X, y):
    """Groups of rows with identical features but different labels."""
    df = X.copy()
    df["_label"] = y.values
    df["_idx"] = np.arange(len(df))
    feature_cols = [c for c in X.columns]
    groups = []
    for _, group_df in df.groupby(feature_cols):
        if group_df["_label"].nunique() > 1:
            groups.append({"indices": group_df["_idx"].tolist()})
    return groups


# =========================================================
# GA: find best per-group curation action (reproduces Phase 3d)
# =========================================================

ACTIONS = ["drop", "keep_majority", "keep_as_is"]

if not hasattr(creator, "FitnessMaxCuration"):
    creator.create("FitnessMaxCuration", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualCuration"):
    creator.create("IndividualCuration", list, fitness=creator.FitnessMaxCuration)


def curate_fold(train_idx, X, y, groups, chromosome):
    """Applies curation actions to a fold's training indices only. Majority
    label is computed from the training portion only (no leakage)."""
    train_idx_set = set(train_idx)
    exclude = set()
    for gene, group in zip(chromosome, groups):
        group_train = [i for i in group["indices"] if i in train_idx_set]
        if not group_train:
            continue
        action = ACTIONS[gene]
        if action == "drop":
            exclude.update(group_train)
        elif action == "keep_majority":
            labels = y.loc[group_train]
            majority_label = labels.value_counts().idxmax()
            for i in group_train:
                if y.loc[i] != majority_label:
                    exclude.add(i)
        # keep_as_is -> no exclusion
    final_train = [i for i in train_idx if i not in exclude]
    return final_train


def find_best_curation(X, y, groups, pop_size=25, n_gen=15, cv_splits=3):
    n_genes = len(groups)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_splits_list = list(cv.split(X, y))

    def fitness_fn(chromosome):
        scores = []
        for train_idx, test_idx in cv_splits_list:
            curated_train = curate_fold(list(train_idx), X, y, groups, chromosome)
            if len(curated_train) < 20:
                return (0.0,)
            X_tr, y_tr = X.loc[curated_train], y.loc[curated_train]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
            clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(f1_score(y_te, preds, average="weighted"))
        return (np.mean(scores),)

    toolbox = base.Toolbox()
    toolbox.register("attr_action", lambda: random.randint(0, 2))
    toolbox.register("individual", tools.initRepeat, creator.IndividualCuration, toolbox.attr_action, n=n_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    print(f"Generation 0 | Best fitness: {max(i.fitness.values[0] for i in population):.4f}")
    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                del m.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        population[:] = offspring
        if gen % 5 == 0 or gen == n_gen:
            print(f"Generation {gen} | Best fitness: {max(i.fitness.values[0] for i in population):.4f}")

    best = max(population, key=lambda i: i.fitness.values[0])
    return list(best)


# =========================================================
# GA hyperparameter tuning (reproduces Phase 3, on given train data)
# =========================================================

if not hasattr(creator, "FitnessMaxTune"):
    creator.create("FitnessMaxTune", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualTune"):
    creator.create("IndividualTune", list, fitness=creator.FitnessMaxTune)


def decode_rf(individual):
    return dict(
        n_estimators=int(round(50 + individual[0] * 250)),
        max_depth=int(round(3 + individual[1] * 27)),
        min_samples_split=int(round(2 + individual[2] * 8)),
        min_samples_leaf=int(round(1 + individual[3] * 4)),
        random_state=RANDOM_SEED,
    )


def ga_tune_rf(X_train, y_train, pop_size=20, n_gen=12, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    def fitness_fn(individual):
        params = decode_rf(individual)
        clf = RandomForestClassifier(**params)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        return (scores.mean(),)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.IndividualTune, toolbox.attr_float, n=4)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(c1, c2)
                for c in (c1, c2):
                    for i in range(len(c)):
                        c[i] = min(max(c[i], 0.0), 1.0)
                del c1.fitness.values
                del c2.fitness.values
        for m in offspring:
            if random.random() < 0.3:
                toolbox.mutate(m)
                for i in range(len(m)):
                    m[i] = min(max(m[i], 0.0), 1.0)
                del m.fitness.values
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        population[:] = offspring

    best = max(population, key=lambda i: i.fitness.values[0])
    return decode_rf(best)


# =========================================================
# EXPERIMENT 1: GA-tuned classifier on curated vs. uncurated data
# =========================================================

def experiment_1(X, y, groups, best_actions):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 -- GA-Tuned Classifier: Curated vs. Uncurated")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    print("\n(a) GA-tuning on UNCURATED training data...")
    params_uncurated = ga_tune_rf(X_train, y_train)
    clf_uncurated = RandomForestClassifier(**params_uncurated)
    clf_uncurated.fit(X_train, y_train)
    acc_uncurated = accuracy_score(y_test, clf_uncurated.predict(X_test))
    print(f"    Test accuracy (GA-tuned, uncurated): {acc_uncurated:.4f}")

    print("\n(b) Applying conflict-resolution curation to training data...")
    curated_train_idx = curate_fold(list(X_train.index), X, y, groups, best_actions)
    X_train_curated = X.loc[curated_train_idx]
    y_train_curated = y.loc[curated_train_idx]
    print(f"    Training set size: {len(X_train)} -> {len(X_train_curated)} after curation "
          f"({len(X_train) - len(X_train_curated)} rows removed)")

    print("(c) GA-tuning on CURATED training data...")
    params_curated = ga_tune_rf(X_train_curated, y_train_curated)
    clf_curated = RandomForestClassifier(**params_curated)
    clf_curated.fit(X_train_curated, y_train_curated)
    acc_curated = accuracy_score(y_test, clf_curated.predict(X_test))
    print(f"    Test accuracy (GA-tuned, curated): {acc_curated:.4f}")

    print("\n(d) Cross-validated robustness check (5-fold x 3-repeat)...")
    cv_robust = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    uncurated_scores, curated_scores = [], []
    for train_idx, test_idx in cv_robust.split(X, y):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        clf_u = RandomForestClassifier(**params_uncurated)
        clf_u.fit(Xtr, ytr)
        uncurated_scores.append(accuracy_score(yte, clf_u.predict(Xte)))

        curated_idx_fold = curate_fold(list(Xtr.index), X, y, groups, best_actions)
        clf_c = RandomForestClassifier(**params_curated)
        clf_c.fit(X.loc[curated_idx_fold], y.loc[curated_idx_fold])
        curated_scores.append(accuracy_score(yte, clf_c.predict(Xte)))

    uncurated_scores, curated_scores = np.array(uncurated_scores), np.array(curated_scores)
    t_stat, p_val = stats.ttest_rel(curated_scores, uncurated_scores)
    print(f"    GA-tuned + uncurated: {uncurated_scores.mean():.4f} +/- {uncurated_scores.std():.4f}")
    print(f"    GA-tuned + curated:   {curated_scores.mean():.4f} +/- {curated_scores.std():.4f}")
    print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f} "
          f"{'(significant)' if p_val < 0.05 else '(not significant)'}")

    plt.figure(figsize=(6.5, 5))
    vals = [uncurated_scores.mean(), curated_scores.mean()]
    bars = plt.bar(["GA-tuned\n(uncurated)", "GA-tuned\n(curated)"], vals, color=["#D9799E", "#9C3B63"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.4f}", ha="center")
    plt.ylabel("Cross-Validated Accuracy")
    plt.title("Experiment 1: GA Tuning -- Curated vs. Uncurated Data")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "experiment1_curated_vs_uncurated_tuning.png"), dpi=150)
    plt.close()

    return {
        "test_split_accuracy": {"uncurated": float(acc_uncurated), "curated": float(acc_curated)},
        "cv_comparison": {
            "uncurated_mean": float(uncurated_scores.mean()), "uncurated_std": float(uncurated_scores.std()),
            "curated_mean": float(curated_scores.mean()), "curated_std": float(curated_scores.std()),
            "t_statistic": float(t_stat), "p_value": float(p_val), "significant": bool(p_val < 0.05),
        },
        "training_rows_removed": len(X_train) - len(X_train_curated),
    }


# =========================================================
# EXPERIMENT 2: Effect on the simulated longitudinal dataset
# =========================================================

N_VISITS = 4
FEATURES = ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
PATTERNS = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]


def pattern_shape(pattern, t):
    if pattern == "Stable":
        return 0.0
    if pattern == "Gradually Worsening":
        return -(1 - t)
    if pattern == "Sudden Deterioration":
        return -(1 - 1 / (1 + np.exp(-12 * (t - 0.75))))
    if pattern == "Improving":
        return (1 - t)


def simulate_visits(df, y, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    feature_stds = {f: df[f].std() for f in FEATURES}
    pattern_probs = {
        0: [0.40, 0.15, 0.10, 0.35], 1: [0.25, 0.30, 0.20, 0.25], 2: [0.10, 0.35, 0.40, 0.15],
    }
    records = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        risk_class = int(y.iloc[pos])
        pattern = rng.choice(PATTERNS, p=pattern_probs[risk_class])
        for i in range(N_VISITS):
            t = i / (N_VISITS - 1)
            visit = {"VisitNumber": i + 1, "PatientID": pos}
            for feat in FEATURES:
                final_val = row[feat]
                if i == N_VISITS - 1:
                    val = final_val
                else:
                    drift = feature_stds[feat] * rng.uniform(0.5, 1.0)
                    val = final_val + pattern_shape(pattern, t) * drift + rng.normal(0, feature_stds[feat] * 0.05)
                visit[feat] = val
            visit["RiskLevel"] = risk_class
            records.append(visit)
    return pd.DataFrame(records)


def engineer_trend_features(long_df):
    records = []
    for pid, group in long_df.groupby("PatientID"):
        group = group.sort_values("VisitNumber")
        rec = {"PatientID": pid, "RiskLevel": group["RiskLevel"].iloc[0]}
        for feat in FEATURES:
            vals = group[feat].values
            rec[f"{feat}_latest"] = vals[-1]
            rec[f"{feat}_rate_of_change"] = (vals[-1] - vals[0]) / (len(vals) - 1)
            rec[f"{feat}_slope"] = np.polyfit(range(len(vals)), vals, 1)[0]
        records.append(rec)
    return pd.DataFrame(records)


def experiment_2(X, y, groups, best_actions):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 -- Effect of Curation on the Longitudinal Dataset")
    print("=" * 60)

    def run_trend_vs_latest(base_X, base_y, label):
        long_df = simulate_visits(base_X.reset_index(drop=True), base_y.reset_index(drop=True))
        trend_df = engineer_trend_features(long_df)

        latest_cols = [f"{f}_latest" for f in FEATURES]
        trend_cols = latest_cols + [f"{f}_rate_of_change" for f in FEATURES] + [f"{f}_slope" for f in FEATURES]

        Xl = trend_df[latest_cols]
        Xt = trend_df[trend_cols]
        yy = trend_df["RiskLevel"]

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
        latest_scores = cross_val_score(
            RandomForestClassifier(n_estimators=150, max_features=None, random_state=RANDOM_SEED),
            Xl, yy, cv=cv, scoring="accuracy", n_jobs=-1
        )
        trend_scores = cross_val_score(
            RandomForestClassifier(n_estimators=150, max_features=None, random_state=RANDOM_SEED),
            Xt, yy, cv=cv, scoring="accuracy", n_jobs=-1
        )
        t_stat, p_val = stats.ttest_rel(trend_scores, latest_scores)
        print(f"\n[{label}] Latest-only: {latest_scores.mean():.4f} +/- {latest_scores.std():.4f}")
        print(f"[{label}] Latest+trend: {trend_scores.mean():.4f} +/- {trend_scores.std():.4f}")
        print(f"[{label}] Difference: {trend_scores.mean() - latest_scores.mean():+.4f}  "
              f"(t={t_stat:.3f}, p={p_val:.4f}, {'significant' if p_val < 0.05 else 'not significant'})")
        return {
            "latest_mean": float(latest_scores.mean()), "latest_std": float(latest_scores.std()),
            "trend_mean": float(trend_scores.mean()), "trend_std": float(trend_scores.std()),
            "difference": float(trend_scores.mean() - latest_scores.mean()),
            "t_statistic": float(t_stat), "p_value": float(p_val), "significant": bool(p_val < 0.05),
        }

    print("\n--- On UNCURATED base data (reproduces Phase 4) ---")
    result_uncurated = run_trend_vs_latest(X, y, "Uncurated")

    print("\n--- On CURATED base data ---")
    curated_idx = curate_fold(list(X.index), X, y, groups, best_actions)
    X_curated = X.loc[curated_idx]
    y_curated = y.loc[curated_idx]
    print(f"Base dataset size: {len(X)} -> {len(X_curated)} after curation")
    result_curated = run_trend_vs_latest(X_curated, y_curated, "Curated")

    plt.figure(figsize=(7.5, 5))
    labels = ["Uncurated\nLatest-only", "Uncurated\nLatest+Trend", "Curated\nLatest-only", "Curated\nLatest+Trend"]
    vals = [result_uncurated["latest_mean"], result_uncurated["trend_mean"],
            result_curated["latest_mean"], result_curated["trend_mean"]]
    colors = ["#FBE1EB", "#D9799E", "#FBE1EB", "#9C3B63"]
    bars = plt.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("Cross-Validated Accuracy")
    plt.title("Experiment 2: Trend-Feature Effect -- Uncurated vs. Curated Base Data")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "experiment2_longitudinal_curation_effect.png"), dpi=150)
    plt.close()

    return {"uncurated": result_uncurated, "curated": result_curated,
            "rows_removed_from_base": len(X) - len(X_curated)}


def main():
    print("=" * 60)
    print("PHASE 3e -- CONFLICT-RESOLUTION FOLLOW-UP EXPERIMENTS")
    print("=" * 60)

    X, y = load_data()
    print(f"Loaded {len(X)} records")

    groups = identify_conflict_groups(X, y)
    print(f"Identified {len(groups)} conflicting groups "
          f"({sum(len(g['indices']) for g in groups)} rows)")

    print("\nFinding best curation strategy (GA search)...")
    best_actions = find_best_curation(X, y, groups)
    action_names = [ACTIONS[a] for a in best_actions]
    print(f"Action breakdown: {pd.Series(action_names).value_counts().to_dict()}")

    exp1_results = experiment_1(X, y, groups, best_actions)
    exp2_results = experiment_2(X, y, groups, best_actions)

    all_results = {
        "n_groups": len(groups),
        "best_actions": best_actions,
        "action_breakdown": pd.Series(action_names).value_counts().to_dict(),
        "experiment_1_ga_tuning": exp1_results,
        "experiment_2_longitudinal": exp2_results,
    }
    summary_path = os.path.join(OUT_DIR, "phase3e_followup_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}\nAll results saved to: {summary_path}")
    print(f"Plots saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()