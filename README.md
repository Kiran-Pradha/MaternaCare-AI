# MaternaCare AI

A hybrid, explainable, longitudinal clinical decision-support pipeline for maternal health risk assessment.

## Project purpose

MaternaCare AI aims to support maternal health screening in low-resource and frontline clinical settings by combining:

- structured risk prediction from demographic and clinical vitals,
- longitudinal trajectory reasoning for changes over time,
- clustering for anemia and severity grouping,
- explainable model outputs, and
- transparent clinical recommendation rules.

The project is designed not only to predict a risk label, but also to explain why the risk is high or moderate, show how a patient is changing over time, and give actionable guidance that a health worker can understand.

---

## Why this project exists

Maternal morbidity and anemia risk are clinically important, but many real-world settings still rely on simple point-in-time assessment. A single snapshot can miss important patterns such as:

- a patient whose blood pressure is worsening over several visits,
- a patient whose vitals are stable but still clinically high risk,
- a patient whose anemia severity places them in a more urgent management tier,
- a clinician needing a clear explanation and recommendation, not just a model probability.

This project tackles that challenge by building a full pipeline, not by relying on a single model alone.

---

## High-level project story

The project began with a standard machine-learning workflow:

1. Understand the dataset and identify the target variable.
2. Train baseline classifiers.
3. Improve the model using optimization and diagnostics.
4. Test whether time-based trend information adds value.
5. Build patient groupings using clustering.
6. Add explainability and rule-based guidance.
7. Prepare a voice-driven interface for field use.

This makes the project a complete review-friendly research pipeline rather than a single isolated experiment.

---

## Repository structure

```text
maternacare-ai/
├── app/                            # Application-facing workflow and UI logic
├── data/
│   ├── raw/                        # Original datasets
│   └── processed/                  # Cleaned and engineered output datasets
├── docs/
│   ├── eda_outputs/                # EDA plots and summaries
│   ├── phase2_outputs/             # Baseline model output artifacts
│   ├── phase3_outputs/             # GA tuning, ceiling diagnostics, ensemble outputs
│   ├── phase3d_outputs/            # Label conflict resolution outputs
│   ├── phase3f_outputs/            # Sensitivity analysis outputs
│   ├── phase3g_outputs/            # Trend-degradation diagnostic outputs
│   ├── phase4_outputs/             # Trajectory simulation and trend-feature outputs
│   ├── phase5_outputs/             # Clustering and patient-tier outputs
│   ├── phase6_outputs/             # SHAP explainability outputs
│   ├── phase7_outputs/             # Rules-engine example outputs
│   └── ...
├── models/
│   ├── baseline_random_forest.joblib
│   ├── baseline_xgboost.joblib
│   ├── ga_optimized_random_forest.joblib
│   ├── ga_optimized_xgboost.joblib
│   ├── ga_conflict_resolution.joblib
│   ├── ga_ensemble_blend.joblib
│   ├── ga_kmeans_anemia.joblib
│   ├── risk_classifier_with_trends.joblib
│   └── trajectory_pattern_classifier.joblib
├── notebooks/
├── src/
│   ├── clustering/
│   ├── data/
│   ├── explainability/
│   ├── models/
│   ├── rules/
│   ├── voice/
│   └── __init__.py
├── README.md
├── requirements.txt
├── vitals_parser.py
└── .gitignore
```

---

## Completed project phases

## Phase 1: Data acquisition and exploratory analysis

### Why we do it

Before building any model, we must understand the data quality, feature space, label structure, class balance, missing values, and possible inconsistencies. In clinical datasets, this stage is essential because weak data preparation can bias the entire project.

### How we do it

The project loads two key datasets from `data/raw/`:

- maternal health risk dataset
- anemia / CBC dataset

The EDA script in `src/data/eda.py`:

- checks that required files exist,
- prints dataset shape and preview rows,
- summarizes missing values,
- reports descriptive statistics,
- identifies candidate target columns,
- visualizes class distributions,
- creates histograms for numeric variables,
- generates correlation heatmaps,
- saves the outputs to `docs/eda_outputs/`.

### Key outcome

This phase creates the foundation for all later phases by confirming what the data looks like and what we can trust.

---

## Phase 2: Baseline model benchmarking

### Why we do it

A baseline is necessary to answer a fundamental question: how well do standard models perform on this problem without additional optimization? This gives us a starting point and prevents us from mistaking weak improvements for genuine gains.

### How we do it

The code in `src/models/baseline.py` and `src/models/phase2_baseline.py`:

- loads the maternal dataset,
- applies the same preprocessing logic used across later model phases,
- trains baseline classifiers such as Random Forest and XGBoost,
- compares performance using test-set evaluation,
- saves the trained baseline models into `models/`,
- writes model summaries to `docs/phase2_outputs/`.

### Key outcome

The baseline models define the performance floor. Every later improvement is interpreted relative to this benchmark.

---

## Phase 3: Model optimization and diagnostic refinement

This is the largest and most important model-improvement section of the project.

### Why we do it

A standard model may reach a decent score, but there may still be room for improvement. The project therefore explores multiple dimensions of improvement:

- better hyperparameters,
- ensemble blending,
- label-quality diagnostics,
- curation of noisy/conflicting labels,
- sensitivity checks around methodology assumptions,
- understanding why some features hurt performance.

### 3a. Genetic Algorithm optimization

#### Why

The project does not want to rely on manually guessed hyperparameters. Instead, it searches systematically for better model settings.

#### How

`src/models/ga_optimize.py` uses DEAP-based genetic algorithms to:

- encode model hyperparameters into gene-like structures,
- evaluate performance on cross-validation folds,
- select the best-performing candidates,
- crossover and mutate the candidates across generations,
- decode the best chromosome into actual model parameters,
- compare optimized models against the baseline.

This is a classic GA style search done in a medical ML setting to maximize weighted F1 and generalization.

### 3b. Ceiling diagnostic

#### Why

Before trying more complex changes, the team checks whether the task has a performance ceiling. If a model is fundamentally limited by data quality or overlap between classes, optimizing further will not yield strong gains.

#### How

`src/models/diagnose_ceiling.py` examines whether the dataset is already near a theoretical upper bound and whether the class structure is too overlapping for improved algorithms to help much.

This diagnostic helps avoid false optimism and keeps the project honest about what is actually learnable from the data.

### 3c. GA-optimized ensemble blending

#### Why

Individual optimized models often perform similarly, but combining them can sometimes capture more diverse decision boundaries.

#### How

`src/models/ga_ensemble.py`:

- loads the GA-tuned models,
- searches for blending weights,
- trains a blended ensemble,
- compares the ensemble against the individual models,
- stores results under `docs/phase3_outputs/`.

### 3d. Label-conflict resolution via GA

#### Why

Medical datasets can contain contradictory or noisy labels. If the target labels conflict internally, a model may learn unstable boundaries or be forced to fit contradictory evidence.

#### How

`src/models/ga_conflict_resolution.py` creates a GA-based conflict-resolution workflow to:

- identify records with inconsistent or contradictory labeling patterns,
- propose curation or conflict-handling actions,
- evaluate whether correcting these labels improves downstream learning,
- save results to `docs/phase3d_outputs/`.

This is not just pre-processing; it is a research experiment in label quality and data cleaning.

### 3e. Follow-up validation of curation impact

#### Why

After identifying label conflicts, the project asks a critical question: does the curation actually help, or does it just change the training data superficially?

#### How

`src/models/ga_curation_followup.py` compares the effect of conflict-resolution curation on the GA-tuned classifier and related settings. This creates a deeper validation loop rather than assuming that one cleanup step is always good.

### 3f. Sensitivity analysis of trend-feature choices

#### Why

When adding engineered features, especially trend features, the project must check whether the findings are robust or only artifacts of one chosen setup.

#### How

`src/models/sensitivity_analysis.py` tests how sensitive the result is to design choices, including simulation assumptions and curation strategies. This directly addresses whether the observed effects are stable and scientifically defensible.

### 3g. Diagnostic of why trend features degrade performance

#### Why

This is a crucial insight in the project: sometimes engineered trend features look clinically sensible but worsen the model instead of helping it.

#### How

`src/models/trend_degradation_diagnostic.py` investigates the root cause of that effect by:

- simulating visits and trend features,
- checking how trend signals interact with the underlying label structure,
- identifying whether the model is exploiting artifacts or memorizing patterns rather than learning generalizable relationships.

This phase is especially important because it demonstrates scientific honesty: the project does not force a trend-based model just because it sounds useful. It tests whether the improvement is real.

### Key outcome of Phase 3

This is the phase where the project shifts from “can we predict risk?” to “what is actually driving model behavior, and how can we make it more robust?”

---

## Phase 4: Longitudinal trajectory simulation

### Why we do it

The original maternal dataset contains one record per patient, not a multi-visit time series. But maternal risk is inherently longitudinal. To investigate whether change over time adds value, the project simulates plausible patient visit histories.

### How we do it

`src/data/trajectory_simulation.py` takes each real patient record and turns it into a multi-visit synthetic trajectory. It:

- uses the original record as the final visit,
- assigns a clinical trajectory pattern such as Stable, Gradually Worsening, Sudden Deterioration, or Improving,
- generates earlier visits by applying feature-dependent drift and realistic noise,
- engineers trend features such as slope and change-over-time signals,
- trains a trajectory pattern classifier and compares the trend-aware model with the baseline model,
- saves outputs to `docs/phase4_outputs/`.

### Why this is scientifically useful

The key question is whether trend features genuinely help prediction or whether they simply add noise. The project explicitly models this to test the real value of longitudinal reasoning.

### Important note

This is a simulated longitudinal dataset, not a real multi-visit medical dataset. The script is designed to demonstrate and test the method even though the original public dataset is cross-sectional.

---

## Phase 5: Hybrid clustering for anemia and risk stratification

### Why we do it

A single classification score is not enough for a frontline clinical system. It is useful to group patients into clinically interpretable risk tiers, especially for anemia-related parameters.

### How we do it

`src/clustering/ga_kmeans.py` performs a GA-optimized K-Means workflow:

- loads anemia-related CBC variables,
- standardizes the features,
- explores how many clusters make sense,
- uses GA to search for better centroid initializations,
- lets K-Means refine the final cluster assignments,
- validates clustering quality using silhouette score and visualization,
- builds clinically meaningful anemia tiers such as Low, Moderate, and Severe risk.

This is a hybrid approach because the Genetic Algorithm is used to optimize the cluster initialization, while K-Means performs the actual clustering refinement.

### Key outcome

The clustering stage gives the system a clinically interpretable tiering layer that can support management recommendations and explainability.

---

## Phase 6: Explainability with SHAP

### Why we do it

A good clinical decision-support system must explain decisions. Machine learning can identify risk, but clinicians need transparency: which feature mattered most, and why?

### How we do it

`src/explainability/shap_integration.py`:

- loads a trained model,
- computes SHAP values on the test data,
- identifies the top contributing features for each risk case,
- produces explanation artifacts for review,
- stores outputs under `docs/phase6_outputs/`.

This connects the model to the human decision-making layer by translating a black-box prediction into understandable feature-level evidence.

### Key outcome

This phase makes the system more trustworthy and reviewable for clinical interpretation.

---

## Phase 7: Transparent clinical rules engine

### Why we do it

An explainable model is helpful, but in a clinical setting a rules engine is often even more defensible. Rules make the reasoning explicit, auditable, and traceable to clinical thresholds.

### How we do it

`src/rules/rules_engine.py` builds a deterministic recommendation engine that combines:

- risk level,
- trajectory pattern,
- vital-sign thresholds,
- anemia tier,
- SHAP-based feature importance context,
- evidence from guideline-style thresholds.

The engine returns action recommendations such as:

- urgent referral,
- close monitoring,
- standard follow-up,
- dietary or supplementation guidance,
- escalation for severe hypertension or elevated temperature.

The rules are intentionally not pure ML output; they are designed to be explicit, interpretable, and clinically grounded.

### Key outcome

This is the safety-oriented layer that turns model insight into practical guidance for health workers.

---

## Phase 8: Voice-based vitals intake and interaction

### Why we do it

In many real-world healthcare settings, data entry happens verbally. A healthcare worker may speak a patient's vital signs instead of typing them into a form. The project therefore includes a voice pipeline to support that workflow.

### How we do it

The modules in `src/voice/` implement the pipeline:

- `vitals_parser.py` parses spoken or text-based vitals into structured values,
- `vitals_extractor.py` extracts relevant fields from transcript content,
- `tts_offline.py` generates offline spoken responses,
- `voice_pipeline.py` orchestrates the flow end-to-end,
- `stt_vosk_integration.py` prepares speech-to-text integration for networked environments,
- `tts_gtts_integration.py` supports online TTS options if needed.

The architecture is designed so that a transcript can be processed into structured vitals, confirmed back to the user, and then handed to the risk/rules engine.

### Key outcome

This phase moves the project toward a realistic field-use interface, especially for settings where digital forms are limited or unavailable.

---

## Current status of the project

The project has completed the major research and prototype pipeline from data understanding through optimization, trend analysis, clustering, explainability, rule-based recommendation, and voice intake.

The main completed areas are:

- Phase 1: EDA and dataset understanding
- Phase 2: baseline ML benchmarking
- Phase 3: GA optimization and diagnostics
- Phase 4: longitudinal simulation and trajectory features
- Phase 5: GA-optimized clustering for anemia tiers
- Phase 6: explainability
- Phase 7: clinical rules engine
- Phase 8: voice pipeline groundwork

The project is therefore already substantial and review-ready as a full research pipeline, even though final application integration and polishing remain.

---

## Important project findings so far

Across the work completed so far, several important conclusions have emerged:

- Baseline models provide a clear benchmark and are necessary for fair comparison.
- Genetic search can improve the model search space beyond fixed hand-tuned settings.
- Data quality and label consistency matter a great deal for medical tasks.
- Trend features can be helpful in principle, but are not automatically beneficial in small structured datasets.
- Explainability matters as much as predictive performance.
- Transparent clinical rules are essential for safety and accountability.
- Voice input can extend the system toward real field deployment.

These findings are important because they show the project is not only producing a predictive model; it is building an understandable clinical support system.

---

## How to run the project

From the repository root:

```bash
# create and activate a virtual environment if needed
python -m venv .venv
.venv\Scripts\activate

# install requirements
pip install -r requirements.txt

# run EDA
python src/data/eda.py

# run baseline modeling
python src/models/baseline.py

# run GA optimization
python src/models/ga_optimize.py

# run trajectory simulation
python src/data/trajectory_simulation.py

# run clustering
python src/clustering/ga_kmeans.py

# run rules engine demo
python src/rules/rules_engine.py

# run voice pipeline demo using transcript input
python src/voice/voice_pipeline.py
```

Note: some voice components may require a networked environment or external audio model dependencies to be fully executed in a live setting.

---

## Outputs and evidence folder

Most project evidence is saved under `docs/` and `models/`. These artifacts serve as review material and support the narrative of the study.

Examples include:

- EDA plots and summary outputs
- baseline model scores
- GA optimization results
- ensemble and conflict-resolution performance
- trajectory simulation outputs
- clustering diagnostics
- SHAP explanation summaries
- example rules-engine recommendations

These files are not just supporting material; they form the project artifact trail that shows the progression of the research.

---

## Review guidance

This README is intentionally written for project review, viva presentation, and assessment. It emphasizes:

- what was done,
- why each phase was necessary,
- how each stage was implemented,
- what the project learned,
- how the pieces fit together as a full pipeline.

A reviewer should be able to read this document and understand the progression from raw data -> baseline model -> optimization -> trajectory reasoning -> clustering -> explainability -> rules -> voice interface.

---

## Final note

MaternaCare AI is already far beyond a simple prototype. It represents a layered research and engineering pipeline that combines prediction, robustness testing, interpretability, and field-use readiness.

The work completed so far provides a strong foundation for the next stage of polishing, integration, and final deployment-oriented development.
