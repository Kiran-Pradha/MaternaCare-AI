"""
Phase 5 — GA-Optimized K-Means (GKA-style) Hybrid Clustering
================================================================
Groups patients by anemia/iron-deficiency risk using CBC (Complete Blood
Count) indices, via a hybrid clustering technique: a Genetic Algorithm
searches for good INITIAL CENTROID positions, which K-Means then refines —
following the published "Genetic K-Means Algorithm" (GKA) pattern, rather
than K-Means' default random/k-means++ initialization.

Why this matters:
  - Plain K-Means is sensitive to its starting centroids — a bad random
    start can converge to a worse, less meaningful final clustering.
  - GA doesn't do the clustering itself — it searches for better starting
    points, then K-Means does the actual refinement. This is the same
    "GA optimizes, the other algorithm executes" pattern used for the
    GA-tuned classifier in Phase 3 — this is what makes it a genuine hybrid,
    not just two techniques used side by side.

Dataset expected: data/raw/anemia_dataset.csv with columns approximately:
    Gender, Hemoglobin, MCH, MCHC, MCV
(See README.md Phase 1 section for where to download this.)

Usage:
    python src/clustering/ga_kmeans.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from deap import base, creator, tools

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase5_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ANEMIA_CSV = os.path.join(RAW_DIR, "anemia_dataset.csv")

sns.set_style("whitegrid")
PINK_SEQUENCE = ["#FBE1EB", "#D9799E", "#9C3B63", "#4A2438"]

RANDOM_SEED = 42
K_CLUSTERS = 3  # Low / Moderate / Severe risk tiers
POP_SIZE = 30
N_GEN = 25
CX_PROB = 0.6
MUT_PROB = 0.3

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# Data loading
# =========================================================

def find_column(df, candidates, friendly_name):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find a column for '{friendly_name}'. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}. Update find_column() calls in this script "
        f"if your dataset uses different column names."
    )


def load_anemia_data():
    if not os.path.exists(ANEMIA_CSV):
        raise FileNotFoundError(
            f"Dataset not found at {ANEMIA_CSV}. Download it first (see README.md Phase 1 section)."
        )
    df = pd.read_csv(ANEMIA_CSV)

    hb_col = find_column(df, ["Hemoglobin", "hemoglobin", "HB", "Hb"], "Hemoglobin")
    mch_col = find_column(df, ["MCH", "mch"], "MCH")
    mchc_col = find_column(df, ["MCHC", "mchc"], "MCHC")
    mcv_col = find_column(df, ["MCV", "mcv"], "MCV")
    gender_col = None
    for c in ["Gender", "gender", "Sex", "sex"]:
        if c in df.columns:
            gender_col = c
            break

    feature_cols = [hb_col, mch_col, mchc_col, mcv_col]
    if gender_col:
        feature_cols.append(gender_col)

    print(f"Using columns: {feature_cols}")
    df_clean = df[feature_cols].dropna().reset_index(drop=True)
    df_clean = df_clean.rename(columns={hb_col: "Hemoglobin", mch_col: "MCH", mchc_col: "MCHC", mcv_col: "MCV"})
    if gender_col:
        df_clean = df_clean.rename(columns={gender_col: "Gender"})
        if df_clean["Gender"].dtype == object:
            df_clean["Gender"] = df_clean["Gender"].astype("category").cat.codes

    print(f"Loaded {len(df_clean)} records after dropping missing values")
    return df_clean


# =========================================================
# Step 1: Determine a reasonable K (elbow + silhouette), for reference
# =========================================================

def explore_k(X_scaled):
    print("\n" + "=" * 60)
    print("EXPLORING CLUSTER COUNT (elbow method + silhouette score)")
    print("=" * 60)

    inertias, silhouettes = [], []
    k_range = range(2, 7)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(list(k_range), inertias, marker="o", color="#9C3B63", linewidth=2)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")

    ax2.plot(list(k_range), silhouettes, marker="o", color="#D9799E", linewidth=2)
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score by K")
    ax2.axvline(x=K_CLUSTERS, color="gray", linestyle="--", linewidth=1,
                label=f"Chosen K={K_CLUSTERS} (Low/Moderate/Severe)")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "k_selection.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")
    print(f"\nUsing K={K_CLUSTERS} for clinical interpretability (Low / Moderate / Severe risk tiers), "
          f"consistent with the project's supplement-guidance mapping — even if a different K "
          f"scored marginally higher on silhouette.")


# =========================================================
# Step 2: Plain K-Means baseline
# =========================================================

def run_plain_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)  # default k-means++ init, best of 10 random starts
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"\nPlain K-Means (k-means++ init, best of 10 random starts): silhouette = {sil:.4f}")
    return km, labels, sil


# =========================================================
# Step 3: GA-optimized centroid initialization (GKA-style hybrid)
# =========================================================

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def run_ga_kmeans(X_scaled, k, n_features):
    n_genes = k * n_features
    mins = X_scaled.min(axis=0)
    maxs = X_scaled.max(axis=0)

    def random_gene(i):
        feat_idx = i % n_features
        return random.uniform(mins[feat_idx], maxs[feat_idx])

    def fitness_fn(individual):
        centroids = np.array(individual).reshape(k, n_features)
        try:
            km = KMeans(n_clusters=k, init=centroids, n_init=1, max_iter=300, random_state=RANDOM_SEED)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < k:
                return (-1.0,)  # a cluster vanished — invalid solution
            return (silhouette_score(X_scaled, labels),)
        except Exception:
            return (-1.0,)

    def clip(ind):
        for i in range(len(ind)):
            feat_idx = i % n_features
            ind[i] = min(max(ind[i], mins[feat_idx]), maxs[feat_idx])
        return ind

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                      lambda: [random_gene(i) for i in range(n_genes)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_fn)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP_SIZE)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    history = [max(population, key=lambda i: i.fitness.values[0]).fitness.values[0]]
    print(f"\n{'=' * 60}\nGA SEARCH — Centroid Initialization\n{'=' * 60}")
    print(f"Generation 0 | Best silhouette: {history[0]:.4f}")

    for gen in range(1, N_GEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(c1, c2)
                clip(c1)
                clip(c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(m)
                clip(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = offspring
        best = max(population, key=lambda i: i.fitness.values[0])
        history.append(best.fitness.values[0])
        if gen % 5 == 0 or gen == N_GEN:
            print(f"Generation {gen} | Best silhouette: {best.fitness.values[0]:.4f}")

    best_individual = max(population, key=lambda i: i.fitness.values[0])
    best_centroids = np.array(best_individual).reshape(k, n_features)

    final_km = KMeans(n_clusters=k, init=best_centroids, n_init=1, max_iter=300, random_state=RANDOM_SEED)
    final_labels = final_km.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, final_labels)

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(history)), history, color="#9C3B63", linewidth=2, marker="o", markersize=3)
    plt.title("GA Convergence — Centroid Initialization")
    plt.xlabel("Generation")
    plt.ylabel("Best Silhouette Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ga_convergence_clustering.png"), dpi=150)
    plt.close()

    return final_km, final_labels, final_sil


# =========================================================
# Step 4: Interpret clusters + map to supplement guidance
# =========================================================

def interpret_clusters(df, labels, scaler, feature_cols):
    df = df.copy()
    df["Cluster"] = labels

    cluster_summary = df.groupby("Cluster")["Hemoglobin"].agg(["mean", "count"]).sort_values("mean")
    print("\n" + "=" * 60)
    print("CLUSTER INTERPRETATION (sorted by mean Hemoglobin, ascending)")
    print("=" * 60)
    print(cluster_summary)
    print("\n[Reference: WHO general anemia thresholds — non-pregnant women: "
          "Severe <8 g/dL, Moderate 8-10.9 g/dL, Mild 11-11.9 g/dL, Normal >=12 g/dL. "
          "Pregnant women: Severe <7 g/dL, Moderate 7-9.9 g/dL, Mild 10-10.9 g/dL. "
          "Compare your cluster means against these to assign final clinical labels appropriate "
          "to your population — the ordinal ranking below (Severe/Moderate/Low) is relative to "
          "the clusters found, not an absolute clinical diagnosis.]")

    tier_names = ["Severe Risk", "Moderate Risk", "Low Risk"] if len(cluster_summary) == 3 else \
                 [f"Tier {i+1} (lowest Hb first)" for i in range(len(cluster_summary))]
    cluster_to_tier = dict(zip(cluster_summary.index, tier_names))
    df["RiskTier"] = df["Cluster"].map(cluster_to_tier)

    supplement_guidance = {
        "Severe Risk": "Refer for clinical evaluation; iron supplementation under medical supervision; investigate underlying cause.",
        "Moderate Risk": "Begin iron supplementation + dietary counseling (iron-rich foods); recheck Hemoglobin in 4 weeks.",
        "Low Risk": "Routine dietary guidance; no supplementation needed; recheck at next scheduled visit.",
    }
    df["SupplementGuidance"] = df["RiskTier"].map(
        lambda t: supplement_guidance.get(t, "Clinical review recommended — tier not explicitly mapped.")
    )

    print("\nSample mapping (first 5 records):")
    print(df[["Hemoglobin", "MCH", "MCHC", "MCV", "RiskTier", "SupplementGuidance"]].head())

    return df, cluster_to_tier


def plot_cluster_pca(X_scaled, cluster_to_tier, cluster_labels_raw):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum()

    # Fixed color-per-tier mapping (not tied to arbitrary cluster ID order),
    # so darker = more severe consistently, regardless of which raw cluster
    # index K-Means happened to assign to which tier.
    tier_colors = {
        "Severe Risk": "#4A2438",
        "Moderate Risk": "#9C3B63",
        "Low Risk": "#F2AFC6",
    }
    fallback_colors = PINK_SEQUENCE

    plt.figure(figsize=(7, 6))
    unique_clusters = sorted(set(cluster_labels_raw))
    for i, c in enumerate(unique_clusters):
        mask = cluster_labels_raw == c
        tier = cluster_to_tier[c]
        color = tier_colors.get(tier, fallback_colors[i % len(fallback_colors)])
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=tier,
                    color=color, alpha=0.6, s=30, edgecolor="white", linewidth=0.3)

    # Order legend by severity (Severe -> Moderate -> Low) rather than plot order
    handles, labels = plt.gca().get_legend_handles_labels()
    order_priority = {"Severe Risk": 0, "Moderate Risk": 1, "Low Risk": 2}
    sorted_pairs = sorted(zip(handles, labels), key=lambda hl: order_priority.get(hl[1], 99))
    handles, labels = zip(*sorted_pairs)

    plt.title(f"Iron-Deficiency Risk Clusters — 2D PCA Projection\n({explained*100:.1f}% variance explained)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles, labels, title="Risk Tier")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "cluster_pca_visualization.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    print("=" * 60)
    print("PHASE 5 — GA-OPTIMIZED K-MEANS HYBRID CLUSTERING")
    print("=" * 60)

    df = load_anemia_data()
    feature_cols = [c for c in ["Hemoglobin", "MCH", "MCHC", "MCV", "Gender"] if c in df.columns]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    explore_k(X_scaled)

    print("\n" + "=" * 60)
    print(f"COMPARISON: Plain K-Means vs. GA-Optimized K-Means (K={K_CLUSTERS})")
    print("=" * 60)

    plain_km, plain_labels, plain_sil = run_plain_kmeans(X_scaled, K_CLUSTERS)
    ga_km, ga_labels, ga_sil = run_ga_kmeans(X_scaled, K_CLUSTERS, X_scaled.shape[1])

    improvement = ga_sil - plain_sil
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Plain K-Means silhouette score:        {plain_sil:.4f}")
    print(f"GA-optimized K-Means silhouette score: {ga_sil:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    if improvement > 0.01:
        print("-> GA-optimized initialization found a meaningfully better clustering than "
              "random/k-means++ initialization.")
    elif improvement > -0.01:
        print("-> Comparable performance — GA found an equally good clustering, though not "
              "a clear improvement on this run.")
    else:
        print("-> Plain K-Means performed better on this run — GA search may need more "
              "generations/population, or K-Means++ initialization was already strong for this data.")

    # Use whichever clustering performed better for the final interpretation
    if ga_sil >= plain_sil:
        final_labels, final_km, which = ga_labels, ga_km, "GA-optimized"
    else:
        final_labels, final_km, which = plain_labels, plain_km, "plain"
    print(f"\nUsing {which} K-Means result for cluster interpretation (higher silhouette score).")

    df_result, cluster_to_tier = interpret_clusters(df, final_labels, scaler, feature_cols)
    plot_cluster_pca(X_scaled, cluster_to_tier, final_labels)

    results = {
        "plain_kmeans_silhouette": float(plain_sil),
        "ga_kmeans_silhouette": float(ga_sil),
        "improvement": float(improvement),
        "used_for_final_interpretation": which,
        "cluster_tiers": {str(k): v for k, v in cluster_to_tier.items()},
    }
    summary_path = os.path.join(OUT_DIR, "phase5_clustering_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    df_result.to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "anemia_risk_tiers.csv"), index=False)
    joblib.dump({"model": final_km, "scaler": scaler, "feature_cols": feature_cols, "cluster_to_tier": cluster_to_tier},
                os.path.join(MODELS_DIR, "ga_kmeans_anemia.joblib"))

    print(f"\nFull results saved to: {summary_path}")
    print(f"Labeled data saved to: data/processed/anemia_risk_tiers.csv")
    print(f"Plots saved to: {OUT_DIR}/")
    print(f"Model saved to: {MODELS_DIR}/ga_kmeans_anemia.joblib")
    print("\nNext step: Phase 6 — Explainability (SHAP) (src/explainability/shap_integration.py)")


if __name__ == "__main__":
    main()