"""
trajectory_classifier.py -- gives the synthetic trajectories their job
MaternaCare AI

WHY THIS EXISTS
---------------
The real dataset has ONE visit per patient, so it can say nothing about how a
patient's readings change over time. The rules engine (Phase 7) needs exactly
that: a trajectory pattern -- Stable / Gradually Worsening / Sudden
Deterioration / Improving -- to combine with the risk level.

So the two data sources have two different jobs:

    real dataset (1,014 single visits)  ->  risk model: "how risky is THIS visit?"
    synthetic trajectories              ->  this classifier: "how are the visits CHANGING?"

HOW IT WORKS
------------
1. Simulate many multi-visit patients with trajectory_simulation_v2.py (visit 1 is
   a real row; later visits are simulated), each labelled with the pattern that
   was planted: stable, worsening, improving, or sudden.
2. Turn each trajectory into features that describe its SHAPE (total change, last
   step, slope, biggest step, acceleration of the last step), measured in units
   of each vital's real-world standard deviation. Nothing depends on the absolute
   level of the vitals -- that is the risk model's job.
3. Train a Random Forest to recover the planted pattern. Trajectories of 2, 3 and
   4 visits are all used, so it works for 1-3 previous visits plus the current one.

HONEST LIMITS (say these out loud when presenting)
--------------------------------------------------
* The labels are the patterns WE planted, so accuracy here measures "can the
  classifier recover the simulation's patterns", not "does it work on real
  patients". It is a proof of concept until real follow-up data exists.
* The size of the planted trends (see trajectory_simulation_v2.py) are assumptions.
* With only two visits (one step) a slow trend cannot be told from noise; the
  evaluation below reports accuracy per trajectory length for that reason.

Run it to see the evaluation:   py src/models/trajectory_classifier.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from trajectory_simulation_v2 import _load_clean_real_data, generate_trajectories  # noqa: E402

VITALS = ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
LABEL_TO_PATTERN = {
    "stable": "Stable",
    "worsening": "Gradually Worsening",
    "improving": "Improving",
    "sudden": "Sudden Deterioration",
}
PATTERNS = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]
LENGTHS = (2, 3, 4)          # total visits = previous visits (1-3) + the current one
# Gradual trends are simulated at 1x to 4x the base drift (roughly 2.5-10 mmHg of systolic BP per
# visit), so "steady climb" is learned as a SHAPE, not as one fixed size. An assumption.
TREND_STRENGTH_RANGE = (1.0, 4.0)


def trajectory_features(visits, real_std):
    """Describe the SHAPE of a trajectory.

    visits   : list of dicts (oldest first, current visit last), each with the VITALS keys
    real_std : {vital: standard deviation in the real dataset}, used to put every vital
               on the same scale
    """
    X = np.array([[float(v[c]) for c in VITALS] for v in visits])
    std = np.array([real_std[c] for c in VITALS])
    steps = np.diff(X, axis=0) / std                    # (n-1, 5), in std units
    total = (X[-1] - X[0]) / std
    last = steps[-1]
    t = np.arange(len(X))
    slope = np.array([np.polyfit(t, X[:, j] / std[j], 1)[0] for j in range(X.shape[1])])
    max_abs = np.abs(steps).max(axis=0)
    accel = last - steps[:-1].mean(axis=0) if len(steps) > 1 else last
    return np.concatenate([total, last, slope, max_abs, accel, [len(X)]])


def _trajectories_from_frame(synth):
    """Split the long simulator output into (visits, label) pairs, one per patient."""
    out = []
    for _, g in synth.sort_values(["patient_id", "visit_number"]).groupby("patient_id"):
        visits = g[VITALS].to_dict("records")
        out.append((visits, g["trend_label"].iloc[0]))
    return out


def _simulate(real, n_patients, length, seed):
    return _trajectories_from_frame(
        generate_trajectories(real, n_patients=n_patients, n_visits=length, seed=seed,
                              risk_informed_fraction=0.5, sudden_fraction=0.25,
                              trend_strength_range=TREND_STRENGTH_RANGE))


class TrajectoryClassifier:
    """Predicts the trajectory pattern from a patient's visits."""

    def __init__(self, n_patients_per_length=600, seed=42):
        self.n_patients_per_length = n_patients_per_length
        self.seed = seed
        self.model = None
        self.real_std = None
        self.n_train = 0

    def fit(self, real):
        """`real` = path to the dataset CSV, or the loaded DataFrame (standard column names)."""
        clean = _load_clean_real_data(real)
        self.real_std = {c: float(clean[c].std()) for c in VITALS}
        X, y = [], []
        for L in LENGTHS:
            for visits, label in _simulate(real, self.n_patients_per_length, L, self.seed + L):
                X.append(trajectory_features(visits, self.real_std))
                y.append(LABEL_TO_PATTERN[label])
        self.model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=self.seed, n_jobs=-1)
        self.model.fit(np.array(X), np.array(y))
        self.n_train = len(y)
        return self

    def predict(self, visits):
        """-> (pattern, {pattern: probability}). Fewer than 2 visits -> ("Stable", None)."""
        if len(visits) < 2:
            return "Stable", None
        x = trajectory_features(visits[-max(LENGTHS):], self.real_std).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        probs = {p: 0.0 for p in PATTERNS}
        probs.update({c: float(p) for c, p in zip(self.model.classes_, proba)})
        return max(probs, key=probs.get), probs

    def evaluate(self, real, n_test_per_length=400, seed=999):
        """Accuracy on FRESH simulated patients the model has never seen (different seed,
        different real-row anchors), reported overall and per trajectory length."""
        res = {"per_length": {}, "n_test": 0}
        y_all, p_all = [], []
        for L in LENGTHS:
            Xt, yt = [], []
            for visits, label in _simulate(real, n_test_per_length, L, seed + L):
                Xt.append(trajectory_features(visits, self.real_std))
                yt.append(LABEL_TO_PATTERN[label])
            pred = self.model.predict(np.array(Xt))
            res["per_length"][L] = float(accuracy_score(yt, pred))
            y_all += yt
            p_all += list(pred)
        res["n_test"] = len(y_all)
        res["accuracy"] = float(accuracy_score(y_all, p_all))
        res["macro_f1"] = float(f1_score(y_all, p_all, average="macro"))
        res["majority_baseline"] = float(pd.Series(y_all).value_counts(normalize=True).max())
        res["confusion"] = confusion_matrix(y_all, p_all, labels=PATTERNS).tolist()
        return res


def format_evaluation(res):
    lines = ["=" * 70, "TRAJECTORY-PATTERN CLASSIFIER -- evaluation on fresh simulated patients", "=" * 70,
             f"Test patients: {res['n_test']}  (never seen in training; trajectories of 2, 3 and 4 visits)",
             f"Accuracy : {res['accuracy'] * 100:.1f}%   (always guessing the most common pattern: "
             f"{res['majority_baseline'] * 100:.1f}%)",
             f"Macro F1 : {res['macro_f1'] * 100:.1f}%", "", "Accuracy by number of visits:"]
    for L, a in res["per_length"].items():
        lines.append(f"  {L} visits ({L - 1} previous + current): {a * 100:.1f}%")
    lines += ["", "Confusion matrix (rows = planted pattern, columns = predicted):",
              "  order: " + ", ".join(PATTERNS)]
    for name, row in zip(PATTERNS, res["confusion"]):
        lines.append(f"  {name:22s} {row}")
    lines += ["", "Reminder: these are patterns the simulation planted, so this is a proof of concept,",
              "not evidence about real patients."]
    return "\n".join(lines)


if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data",
                       "maternal_health_risk_data.csv")
    clf = TrajectoryClassifier().fit(csv)
    print(f"Trained on {clf.n_train} simulated trajectories.\n")
    print(format_evaluation(clf.evaluate(csv)))