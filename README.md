# MaternaCare AI

A Hybrid, Explainable, Longitudinal Decision-Support System for Maternal Health Risk in Low-Resource Settings.

## Project Structure

```
maternacare-ai/
├── data/
│   ├── raw/              # Original downloaded datasets (never edit these directly)
│   └── processed/        # Cleaned/engineered datasets used for modeling
├── notebooks/            # Exploratory notebooks (EDA, experiments)
├── src/
│   ├── data/              # Data loading, cleaning, trajectory simulation
│   ├── models/            # Baseline + GA-optimized classifiers
│   ├── clustering/         # GA + K-Means iron-deficiency module
│   ├── explainability/    # SHAP integration
│   ├── rules/              # Recommendation rules engine
│   └── voice/              # Speech-to-text / text-to-speech interface
├── app/                   # Streamlit demo app
├── models/                # Saved trained model files (.pkl, .joblib)
├── docs/                  # Project write-up, roadmap, presentation, references
└── requirements.txt
```

## Team Setup (VS Code)

Each team member should follow these steps once, on their own machine:

1. **Clone the repo** (after it's created on GitHub — see below):
   ```bash
   git clone https://github.com/<your-username>/maternacare-ai.git
   cd maternacare-ai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **In VS Code**: open the project folder (`File > Open Folder`), then select the `venv` interpreter (`Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv/`).

5. **Install the recommended VS Code extensions**: Python (Microsoft), Jupyter.

## Creating the GitHub Repo (do this once, as a team)

1. One team member goes to https://github.com/new, creates a repository named `maternacare-ai` (keep it **Private** if you don't want it public until submission).
2. In that same folder on your machine (this folder), run:
   ```bash
   git init
   git add .
   git commit -m "Initial project structure"
   git branch -M main
   git remote add origin https://github.com/<your-username>/maternacare-ai.git
   git push -u origin main
   ```
3. Go to the repo on GitHub → **Settings → Collaborators** → add your other two teammates by their GitHub usernames/emails.
4. Everyone else then just does `git clone <the repo URL>` (step 1 above) instead of `git init`.

## Datasets Needed (Phase 1)

Download these manually and place them in `data/raw/`:

1. **UCI Maternal Health Risk Dataset**
   Source: Kaggle — search "Maternal Health Risk Data Set" (uploaded by csafrit2 / originally UCI).
   Save as: `data/raw/maternal_health_risk.csv`

2. **Anemia / CBC Dataset**
   Source: Kaggle — search "Anemia Dataset" (Hemoglobin, MCH, MCHC, MCV columns).
   Save as: `data/raw/anemia_dataset.csv`

> Tip: If you have a Kaggle account, install the Kaggle CLI (`pip install kaggle`), place your `kaggle.json` API token in `~/.kaggle/`, then you can download via `kaggle datasets download -d <dataset-slug>` instead of the browser.

## Running Phase 1 (EDA)

Once both CSVs are in `data/raw/`, run:
```bash
python src/data/eda.py
```
This will print dataset summaries and save plots to `docs/eda_outputs/`.

## Workflow / Git Practice

- Never commit directly to `main` for anything beyond initial setup. Create a branch per feature:
  ```bash
  git checkout -b <yourname>/baseline-classifier
  ```
- Push your branch and open a Pull Request on GitHub so teammates can review before merging.
- Pull the latest `main` before starting new work each day: `git pull origin main`.

## Team Roles (reference)

- **Member 1** — Data & Core Model: baseline classifier, GA-optimization, trimester/trajectory modeling (`src/data/`, `src/models/`)
- **Member 2** — Explainability, Recommendation & Clustering: SHAP, rules engine, GA+K-Means clustering (`src/explainability/`, `src/rules/`, `src/clustering/`)
- **Member 3** — Interface & Integration: voice interface, demo app, documentation (`src/voice/`, `app/`, `docs/`)

## Reference Docs

See `docs/` for the full project write-up, 12-phase roadmap PDF, and Zeroth Review presentation.
