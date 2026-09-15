# MaternaCare AI

A hybrid, explainable, longitudinal decision-support system for maternal health risk assessment in low-resource settings.

## Project status

This project has already progressed through several major phases of the pipeline and includes working model experiments, optimization routines, diagnostic tests, and generated outputs.

### Completed so far

- Data ingestion and exploratory analysis for the maternal health dataset
- Baseline model benchmarking and comparison
- Conflict-resolution / curation workflow for duplicate or contradictory labels
- Genetic algorithm (GA) optimization for model selection and conflict handling
- Ensemble and blended modeling experiments
- Longitudinal trajectory simulation and trend-feature engineering
- Sensitivity analysis of simulation design choices
- Diagnostic analysis explaining why trend features may degrade performance
- Clustering and segmentation analysis for patient patterns
- Output artifacts saved under the docs/ folder for review and presentation use

## Repository structure

```text
maternacare-ai/
├── app/                              # Demo / application layer
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Cleaned, engineered, and simulated outputs
├── docs/
│   ├── eda_outputs/                  # EDA plots and summaries
│   ├── phase2_outputs/               # Baseline model outputs
│   ├── phase3_outputs/               # Phase 3 diagnostic/model outputs
│   ├── phase3d_outputs/              # Conflict-resolution / curation outputs
│   ├── phase3f_outputs/              # Sensitivity results
│   ├── phase3g_outputs/              # Trend-degradation diagnostic outputs
│   ├── phase4_outputs/               # Trajectory simulation results
│   ├── phase5_outputs/               # Clustering and segmentation outputs
│   └── phase6_outputs/               # Additional project artifacts
├── models/                           # Saved trained joblib models
├── notebooks/                        # Research notebooks and experiments
├── src/
│   ├── clustering/                   # GA + clustering logic
│   ├── data/                         # Data loading, EDA, trajectory simulation
│   ├── explainability/               # Explainability-related components
│   ├── models/                       # Baseline, GA-optimized, and diagnostic models
│   ├── rules/                        # Recommendation logic
│   ├── voice/                        # Voice interface components
│   └── __init__.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Key project milestones completed

### Phase 1: Data preparation and exploratory analysis
- Dataset loading and review
- Risk-label normalization
- Exploratory summaries and visual inspection
- Output saved under docs/eda_outputs/

### Phase 2: Baseline modeling
- Baseline classifier training and evaluation
- Model comparison and benchmark results
- Outputs stored in docs/phase2_outputs/

### Phase 3: Model refinement and diagnostic checks
- Conflict group identification and resolution strategy
- GA-based curation search for noisy contradictory labels
- Ceiling diagnostic analysis for model upper bounds
- Ensemble and blended-model experiments
- Sensitivity evaluation of simulation assumptions
- Diagnostic study explaining performance degradation from trend features

Generated outputs include:
- docs/phase3_outputs/
- docs/phase3d_outputs/
- docs/phase3f_outputs/
- docs/phase3g_outputs/

### Phase 4: Longitudinal trajectory modeling
- Simulated patient visit trajectories
- Trend feature engineering using visit-level changes
- Trajectory-based classification outputs
- Saved results in docs/phase4_outputs/

### Phase 5: Clustering and patient grouping
- Cluster analysis of maternal risk patterns
- Robustness checks and visualization
- Saved results in docs/phase5_outputs/

## Representative model and analysis outputs

The project currently includes trained artifacts and diagnostics such as:

- baseline_random_forest.joblib
- baseline_xgboost.joblib
- ga_optimized_random_forest.joblib
- ga_optimized_xgboost.joblib
- ga_conflict_resolution.joblib
- ga_ensemble_blend.joblib
- ga_kmeans_anemia.joblib
- trajectory_pattern_classifier.joblib
- risk_classifier_with_trends.joblib

and corresponding reports in docs/phase*_outputs/.

## Execution notes

The project is designed to run from the repository root.

Common commands:

```bash
# Activate virtual environment (Windows PowerShell)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run EDA
python src/data/eda.py

# Run sensitivity analysis
python src/models/sensitivity_analysis.py

# Run trend degradation diagnostic
python src/models/trend_degradation_diagnostic.py
```

## Current interpretation of findings

The project has already established that:

- baseline and optimized models can achieve strong classification performance,
- conflict resolution is important for cleaning contradictory labels,
- trend features can create memorization effects in small structured datasets,
- sensitivity analysis shows the negative trend-feature effect is robust across tested simulation settings,
- clustering and trajectory analysis provide additional structure for patient-level interpretation.

## Next steps

Planned continuation of the project includes:

- finalizing the main model selection narrative,
- integrating explainability and rule-based recommendations,
- polishing the project write-up and presentation materials,
- preparing the application layer and final user-facing workflow.

## Reference documents

The project includes several supporting materials in docs/:

- project roadmap
- zeroth review presentation
- project write-up
- output summaries and figures for each major phase

This README reflects the state of the project as completed so far and will continue to be updated as additional work is finalized.
