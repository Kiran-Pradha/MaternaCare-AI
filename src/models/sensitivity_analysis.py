"""
Phase 3f -- Sensitivity Analysis: Is the Trend-Feature Conclusion Robust
to Simulation Design Choices?
======================================================================================
The longitudinal trajectory simulation (Phase 4) has three design choices that were
picked somewhat arbitrarily:
    1. N_VISITS       -- how many visits to simulate per patient (we used 4)
    2. NOISE_PCT       -- how much random noise per visit, as % of feature std (we used 5%)
    3. DRIFT_RANGE     -- how far earlier visits can deviate from the final value,
                          as a multiplier on feature std (we used 0.5x-1.0x)

This script asks: if we had picked DIFFERENT reasonable values for these, would our
conclusion about trend features change? If the answer is "no, the conclusion is
consistent across a range of reasonable choices," that is real evidence the finding
is a property of the DATA and METHOD, not an artifact of one arbitrary parameter
setting. If the answer is "yes, it flips around," that is important to know and
report honestly too.

This script runs on the CURATED dataset (post conflict-resolution), since that is
the more defensible final version of the pipeline.

Self-contained script (no cross-file imports).

Usage:
    python src/models/sensitivity_analysis.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from deap import base, creator, tools
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase3f_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")

sns.set_style("whitegrid")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

FEATURES = ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
PATTERNS = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]

DEFAULT_N_VISITS = 4
DEFAULT_NOISE_PCT = 0.05
DEFAULT_DRIFT_RANGE = (0.5, 1.0)

N_VISITS_SWEEP = [3, 4, 5, 6]
NOISE_PCT_SWEEP = [0.02, 0.05, 0.10]
DRIFT_RANGE_SWEEP = [(0.3, 0.6), (0.5, 1.0), (0.7, 1.3)]

SEEDS_PER_CONFIG = 3


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
    df = X.copy()
    df["_label"] = y.values
    df["_idx"] = np.arange(len(df))
    feature_cols = [c for c in X.columns]
    groups = []
    for _, group_df in df.groupby(feature_cols):
        if group_df["_label"].nunique() > 1:
            groups.append({"indices": group_df["_idx"].tolist()})
    return groups


ACTIONS = ["drop", "keep_majority", "keep_as_is"]

if not hasattr(creator, "FitnessMaxCuration"):
    creator.create("FitnessMaxCuration", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualCuration"):
    creator.create("IndividualCuration", list, fitness=creator.FitnessMaxCuration)


def curate_fold(train_idx, X, y, groups, chromosome):
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
    return [i for i in train_idx if i not in exclude]


def find_best_curation(X, y, groups, pop_size=15, n_gen=8, cv_splits=3):
    # Lighter than the Phase 3d settings on purpose: this is only a fallback for
    # when no cached curation strategy is available. The strategy found here is
    # used identically across all sweep configurations, so small differences in
    # the search do not bias the sensitivity comparison (which is what we care
    # about here -- relative effect across parameters, not absolute accuracy).
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
            # 50 trees (not 100) during the search -- enough to rank candidate
            # strategies against each other, roughly 2x faster per evaluation.
            clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
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
    best = max(population, key=lambda i: i.fitness.values[0])
    return list(best)


def pattern_shape(pattern, t):
    if pattern == "Stable":
        return 0.0
    if pattern == "Gradually Worsening":
        return -(1 - t)
    if pattern == "Sudden Deterioration":
        return -(1 - 1 / (1 + np.exp(-12 * (t - 0.75))))
    if pattern == "Improving":
        return (1 - t)


def simulate_visits(df, y, n_visits, noise_pct, drift_range, seed):
    rng = np.random.default_rng(seed)
    feature_stds = {f: df[f].std() for f in FEATURES}
    pattern_probs = {
        0: [0.40, 0.15, 0.10, 0.35], 1: [0.25, 0.30, 0.20, 0.25], 2: [0.10, 0.35, 0.40, 0.15],
    }
    records = []
    for pos, (idx, row) in enumerate(df.iterrows()):
        risk_class = int(y.iloc[pos])
        pattern = rng.choice(PATTERNS, p=pattern_probs[risk_class])
        for i in range(n_visits):
            t = i / (n_visits - 1)
            visit = {"VisitNumber": i + 1, "PatientID": pos}
            for feat in FEATURES:
                final_val = row[feat]
                if i == n_visits - 1:
                    val = final_val
                else:
                    drift_mag = feature_stds[feat] * rng.uniform(*drift_range)
                    noise = rng.normal(0, feature_stds[feat] * noise_pct)
                    val = final_val + pattern_shape(pattern, t) * drift_mag + noise
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


def evaluate_config(X_curated, y_curated, n_visits, noise_pct, drift_range, seed):
    long_df = simulate_visits(X_curated.reset_index(drop=True), y_curated.reset_index(drop=True),
                               n_visits, noise_pct, drift_range, seed)
    trend_df = engineer_trend_features(long_df)

    latest_cols = [f"{f}_latest" for f in FEATURES]
    trend_cols = latest_cols + [f"{f}_rate_of_change" for f in FEATURES] + [f"{f}_slope" for f in FEATURES]

    Xl, Xt, yy = trend_df[latest_cols], trend_df[trend_cols], trend_df["RiskLevel"]

    # 5 folds x 1 repeat (not 2) and 60 trees: we average across SEEDS_PER_CONFIG
    # different simulation seeds anyway, so repeated-CV within each seed is
    # redundant variance-reduction at roughly double the cost.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    latest_scores = cross_val_score(
        RandomForestClassifier(n_estimators=60, max_features=None, random_state=RANDOM_SEED),
        Xl, yy, cv=cv, scoring="accuracy", n_jobs=-1
    )
    trend_scores = cross_val_score(
        RandomForestClassifier(n_estimators=60, max_features=None, random_state=RANDOM_SEED),
        Xt, yy, cv=cv, scoring="accuracy", n_jobs=-1
    )
    diff = trend_scores.mean() - latest_scores.mean()
    t_stat, p_val = stats.ttest_rel(trend_scores, latest_scores)
    return diff, p_val, latest_scores.mean(), trend_scores.mean()


def run_sweep(X_curated, y_curated, param_name, values, fixed_params):
    print(f"\n--- Sweeping {param_name} ({len(values)} values x {SEEDS_PER_CONFIG} seeds) ---", flush=True)
    results = []
    for val in values:
        params = dict(fixed_params)
        params[param_name] = val
        diffs, pvals = [], []
        print(f"  {param_name}={val} ...", end="", flush=True)
        for seed_offset in range(SEEDS_PER_CONFIG):
            seed = RANDOM_SEED + seed_offset
            diff, p_val, latest_mean, trend_mean = evaluate_config(
                X_curated, y_curated, params["n_visits"], params["noise_pct"], params["drift_range"], seed
            )
            diffs.append(diff)
            pvals.append(p_val)
            print(".", end="", flush=True)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        frac_significant = np.mean([p < 0.05 for p in pvals])
        print(f" mean diff={mean_diff:+.4f} (std={std_diff:.4f}), "
              f"significant in {frac_significant*100:.0f}% of seeds", flush=True)
        results.append({
            "param_value": str(val), "mean_diff": float(mean_diff), "std_diff": float(std_diff),
            "frac_significant": float(frac_significant), "individual_diffs": [float(d) for d in diffs],
        })
    return results


def load_cached_curation(n_groups):
    """
    The curation strategy was already found in Phase 3d / 3e -- re-running that
    GA search here is pure wasted compute (it is the single most expensive step
    in this script). Look for a previously-saved result and reuse it.

    Checks, in order:
      1. docs/phase3f_outputs/cached_curation.json  (this script's own cache)
      2. docs/phase3d_outputs/phase3d_conflict_resolution_results.json
      3. docs/phase3e_outputs/phase3e_followup_results.json
    """
    candidates = [
        os.path.join(OUT_DIR, "cached_curation.json"),
        os.path.join(PROJECT_ROOT, "docs", "phase3d_outputs", "phase3d_conflict_resolution_results.json"),
        os.path.join(PROJECT_ROOT, "docs", "phase3e_outputs", "phase3e_followup_results.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            actions = data.get("best_actions")
            if actions and len(actions) == n_groups:
                actions = [int(a) for a in actions]
                print(f"Reusing cached curation strategy from: {os.path.relpath(path, PROJECT_ROOT)}")
                return actions
            elif actions:
                print(f"  (Found cached actions in {os.path.basename(path)} but length "
                      f"{len(actions)} != {n_groups} groups -- ignoring.)")
        except Exception as e:
            print(f"  (Could not read {os.path.basename(path)}: {e})")
    return None


def main():
    print("=" * 60)
    print("PHASE 3f -- SENSITIVITY ANALYSIS")
    print("=" * 60)

    X, y = load_data()
    groups = identify_conflict_groups(X, y)
    print(f"Loaded {len(X)} records, {len(groups)} conflicting groups")

    best_actions = load_cached_curation(len(groups))
    if best_actions is None:
        print("\nNo cached curation found -- running GA search "
              "(this is the slowest step; results will be cached for next time)...")
        best_actions = find_best_curation(X, y, groups)
        with open(os.path.join(OUT_DIR, "cached_curation.json"), "w") as f:
            json.dump({"best_actions": best_actions}, f)
        print(f"Cached curation strategy to {OUT_DIR}/cached_curation.json")

    curated_idx = curate_fold(list(X.index), X, y, groups, best_actions)
    X_curated, y_curated = X.loc[curated_idx], y.loc[curated_idx]
    print(f"Curated dataset: {len(X)} -> {len(X_curated)} rows")

    fixed_params = {"n_visits": DEFAULT_N_VISITS, "noise_pct": DEFAULT_NOISE_PCT, "drift_range": DEFAULT_DRIFT_RANGE}

    n_visits_results = run_sweep(X_curated, y_curated, "n_visits", N_VISITS_SWEEP, fixed_params)
    noise_results = run_sweep(X_curated, y_curated, "noise_pct", NOISE_PCT_SWEEP, fixed_params)
    drift_results = run_sweep(X_curated, y_curated, "drift_range", DRIFT_RANGE_SWEEP, fixed_params)

    all_diffs = (
        [r["mean_diff"] for r in n_visits_results] +
        [r["mean_diff"] for r in noise_results] +
        [r["mean_diff"] for r in drift_results]
    )
    all_positive = all(d > 0 for d in all_diffs)
    all_negative = all(d < 0 for d in all_diffs)

    print("\n" + "=" * 60)
    print("OVERALL CONSISTENCY CHECK")
    print("=" * 60)
    print(f"Range of trend-feature effect across {len(all_diffs)} configurations: "
          f"{min(all_diffs):+.4f} to {max(all_diffs):+.4f}")
    if all_positive:
        print("-> CONSISTENT: trend features helped in every configuration tested.")
    elif all_negative:
        print("-> CONSISTENT: trend features hurt in every configuration tested.")
    else:
        print("-> INCONSISTENT: the direction of the trend-feature effect changes depending on "
              "simulation parameters -- this should be reported as a real limitation, not glossed over.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    sweep_data = [
        ("N_VISITS", N_VISITS_SWEEP, n_visits_results),
        ("Noise %", [f"{v*100:.0f}%" for v in NOISE_PCT_SWEEP], noise_results),
        ("Drift Range", [f"{v[0]}-{v[1]}x" for v in DRIFT_RANGE_SWEEP], drift_results),
    ]
    for ax, (title, xvals, results) in zip(axes, sweep_data):
        means = [r["mean_diff"] for r in results]
        stds = [r["std_diff"] for r in results]
        ax.bar([str(v) for v in xvals], means, yerr=stds, color="#9C3B63", capsize=4)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(title)
    axes[0].set_ylabel("Trend-Feature Effect\n(Accuracy Difference)")
    plt.suptitle("Sensitivity Analysis: Trend-Feature Effect Across Simulation Parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sensitivity_analysis.png"), dpi=150)
    plt.close()

    results = {
        "n_visits_sweep": n_visits_results,
        "noise_pct_sweep": noise_results,
        "drift_range_sweep": drift_results,
        "overall": {
            "min_diff": float(min(all_diffs)), "max_diff": float(max(all_diffs)),
            "all_positive": bool(all_positive), "all_negative": bool(all_negative),
            "consistent_direction": bool(all_positive or all_negative),
        },
    }
    summary_path = os.path.join(OUT_DIR, "phase3f_sensitivity_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {summary_path}")
    print(f"Plot saved to: {OUT_DIR}/sensitivity_analysis.png")


if __name__ == "__main__":
    main()