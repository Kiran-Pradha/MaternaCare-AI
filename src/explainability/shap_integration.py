"""
Phase 6 -- SHAP Explainability Integration
======================================================================================
Adds explainability on top of the risk classifier: for every prediction, SHAP
(SHapley Additive exPlanations) attributes the outcome to specific input features,
so a frontline health worker sees not just "High Risk" but WHY.

Model used: a Random Forest trained on the conflict-curated dataset (reusing the
curation strategy found in Phase 3d/3e/3f, loaded from cache if available -- no
expensive re-optimization here, since this phase's focus is explainability, not
further tuning).

What this script produces:
  1. A global summary plot (which features matter most, on average, across all
     predictions) -- both a bar chart and a beeswarm-style detail plot.
  2. Individual waterfall explanations for three example patients (one per risk
     class), showing exactly which features pushed the prediction up or down.
  3. A human-readable "top factors" sentence generator for each example patient
     -- this is the actual output format a health worker would see in the demo
     app (Phase 9), not a chart.
  4. A sanity check: do the top global features match clinical expectation
     (BP, blood sugar should dominate)?

Usage:
    python src/explainability/shap_integration.py
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from deap import base, creator, tools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase6_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MATERNAL_CSV = os.path.join(RAW_DIR, "maternal_health_risk.csv")

sns.set_style("whitegrid")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = ["Low Risk", "Mid Risk", "High Risk"]


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
    if not os.path.exists(MATERNAL_CSV):
        raise FileNotFoundError(f"Dataset not found at {MATERNAL_CSV}. Complete Phase 1 first.")
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

if not hasattr(creator, "FitnessMaxCuration6"):
    creator.create("FitnessMaxCuration6", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualCuration6"):
    creator.create("IndividualCuration6", list, fitness=creator.FitnessMaxCuration6)


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
            clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            scores.append(f1_score(y_te, preds, average="weighted"))
        return (np.mean(scores),)

    toolbox = base.Toolbox()
    toolbox.register("attr_action", lambda: random.randint(0, 2))
    toolbox.register("individual", tools.initRepeat, creator.IndividualCuration6, toolbox.attr_action, n=n_genes)
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


def load_cached_curation(n_groups):
    """Reuse a previously-found curation strategy instead of re-searching."""
    candidates = [
        os.path.join(OUT_DIR, "cached_curation.json"),
        os.path.join(PROJECT_ROOT, "docs", "phase3f_outputs", "cached_curation.json"),
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
                print(f"Reusing cached curation strategy from: {os.path.relpath(path, PROJECT_ROOT)}")
                return [int(a) for a in actions]
        except Exception:
            continue
    return None


def train_final_model(X, y):
    groups = identify_conflict_groups(X, y)
    print(f"Identified {len(groups)} conflicting groups")

    best_actions = load_cached_curation(len(groups))
    if best_actions is None:
        print("No cached curation found -- running a lightweight GA search "
              "(this is a fallback; ideally reuse Phase 3d's result)...")
        best_actions = find_best_curation(X, y, groups)
        with open(os.path.join(OUT_DIR, "cached_curation.json"), "w") as f:
            json.dump({"best_actions": best_actions}, f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    curated_train_idx = curate_fold(list(X_train.index), X, y, groups, best_actions)
    X_train_curated = X.loc[curated_train_idx]
    y_train_curated = y.loc[curated_train_idx]
    print(f"Training set: {len(X_train)} -> {len(X_train_curated)} rows after curation")

    clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=3,
                                  random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train_curated, y_train_curated)

    test_acc = accuracy_score(y_test, clf.predict(X_test))
    test_f1 = f1_score(y_test, clf.predict(X_test), average="weighted")
    print(f"Final model -- test accuracy: {test_acc:.4f}, weighted F1: {test_f1:.4f}")

    return clf, X_train_curated, X_test, y_test, {"test_accuracy": test_acc, "test_f1": test_f1}


def get_shap_values(explainer, X_sample):
    """Handles both SHAP API shapes: list-per-class, or a single (n, features, classes) array."""
    raw = explainer.shap_values(X_sample)
    if isinstance(raw, list):
        return raw
    if raw.ndim == 3:
        return [raw[:, :, c] for c in range(raw.shape[2])]
    return [raw]


def plot_global_summary(shap_values_per_class, X_test, feature_names):
    mean_abs = np.zeros(len(feature_names))
    for class_shap in shap_values_per_class:
        mean_abs += np.abs(class_shap).mean(axis=0)
    mean_abs /= len(shap_values_per_class)

    order = np.argsort(mean_abs)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_vals = mean_abs[order]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features[::-1], sorted_vals[::-1], color="#9C3B63")
    plt.xlabel("Mean |SHAP value| (average across all 3 risk classes)")
    plt.title("Global Feature Importance -- What Drives Risk Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_global_summary.png"), dpi=150)
    plt.close()

    return dict(zip(sorted_features, [float(v) for v in sorted_vals]))


def plot_beeswarm(explainer, X_test, class_idx, class_name):
    try:
        sv = explainer(X_test)
        if len(sv.shape) == 3:
            sv_class = sv[:, :, class_idx]
        else:
            sv_class = sv
        plt.figure()
        shap.plots.beeswarm(sv_class, show=False, max_display=10)
        plt.title(f"SHAP Detail -- {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_beeswarm_{class_name.replace(' ', '_').lower()}.png"), dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"  (Beeswarm plot skipped for {class_name}: {e})")
        return False


def explain_patient(shap_values_per_class, X_test, patient_idx, predicted_class, feature_names, expected_values):
    class_shap = shap_values_per_class[predicted_class][patient_idx]
    feature_vals = X_test.iloc[patient_idx]

    contributions = list(zip(feature_names, class_shap, feature_vals))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = contributions[:3]

    sentences = []
    for feat, shap_val, actual_val in top3:
        direction = "increased" if shap_val > 0 else "decreased"
        sentences.append(
            f"{feat} = {actual_val:.1f} {direction} the model's confidence in this "
            f"{CLASS_NAMES[predicted_class]} prediction (SHAP contribution: {shap_val:+.3f})"
        )

    base_val = expected_values[predicted_class] if hasattr(expected_values, "__len__") else expected_values

    return {
        "predicted_class": CLASS_NAMES[predicted_class],
        "base_value": float(base_val),
        "top_factors": sentences,
        "full_contributions": [(f, float(s), float(v)) for f, s, v in contributions],
    }


def plot_waterfall(explainer, X_test, patient_idx, predicted_class, class_name):
    try:
        sv = explainer(X_test)
        if len(sv.shape) == 3:
            sv_single = sv[patient_idx, :, predicted_class]
        else:
            sv_single = sv[patient_idx]
        plt.figure()
        shap.plots.waterfall(sv_single, show=False, max_display=10)
        plt.title(f"Why this patient was predicted {class_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"shap_waterfall_{class_name.replace(' ', '_').lower()}.png"), dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"  (Waterfall plot skipped: {e})")
        return False


def main():
    print("=" * 60)
    print("PHASE 6 -- SHAP EXPLAINABILITY")
    print("=" * 60)

    X, y = load_data()
    print(f"Loaded {len(X)} records")

    clf, X_train_curated, X_test, y_test, model_metrics = train_final_model(X, y)
    feature_names = list(X.columns)

    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(clf)
    shap_values_per_class = get_shap_values(explainer, X_test)
    print(f"SHAP values computed for {len(shap_values_per_class)} classes x {len(X_test)} test samples")

    print("\n--- Global Feature Importance ---")
    global_importance = plot_global_summary(shap_values_per_class, X_test, feature_names)
    for feat, val in sorted(global_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<15}: {val:.4f}")

    top_feature = max(global_importance, key=global_importance.get)
    clinically_expected = {"SystolicBP", "DiastolicBP", "BS"}
    if top_feature in clinically_expected:
        print(f"\n-> Sanity check PASSED: top feature ({top_feature}) matches clinical expectation "
              f"(BP/blood sugar should dominate risk prediction).")
    else:
        print(f"\n-> Sanity check FLAG: top feature ({top_feature}) is not one of the usually-expected "
              f"drivers (BP/blood sugar) -- worth a closer look before presenting.")

    print("\n--- Generating detail plots per class ---")
    for i, cname in enumerate(CLASS_NAMES):
        plot_beeswarm(explainer, X_test, i, cname)

    print("\n--- Example Patient Explanations ---")
    preds = clf.predict(X_test)
    example_explanations = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        matches = np.where(preds == class_idx)[0]
        if len(matches) == 0:
            print(f"  No test patients predicted as {class_name} -- skipping example.")
            continue
        patient_idx = int(matches[0])
        expl = explain_patient(shap_values_per_class, X_test, patient_idx, class_idx,
                                feature_names, explainer.expected_value)
        example_explanations[class_name] = expl

        print(f"\n[{class_name}] Patient #{patient_idx}:")
        for sentence in expl["top_factors"]:
            print(f"    - {sentence}")

        plot_waterfall(explainer, X_test, patient_idx, class_idx, class_name)

    results = {
        "model_metrics": model_metrics,
        "global_feature_importance": global_importance,
        "top_feature": top_feature,
        "example_explanations": example_explanations,
    }
    summary_path = os.path.join(OUT_DIR, "phase6_shap_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {summary_path}")
    print(f"Plots saved to: {OUT_DIR}/")
    print("\nNext step: Phase 7 -- Clinical Recommendation Rules Engine")


if __name__ == "__main__":
    main()