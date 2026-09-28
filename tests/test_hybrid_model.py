"""test_hybrid_model.py — requires the real CSV to be present at
data/maternal_health_risk_data.csv. Runs a lightweight (small n_splits,
single seed) version of the evaluation to keep test runtime short; the
full multi-seed sweep is run separately as the actual deliverable, not as
a unit test."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
from hybrid_model import (
    load_dataset, rule_based_probabilities, rule_based_predict,
    hybrid_predict, run_single_dataset_evaluation, CLASS_TO_IDX,
)
import pandas as pd
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "maternal_health_risk_data.csv")


def _require_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[SKIP] real dataset not present at {CSV_PATH} — hybrid model tests need it")
        return False
    return True


def test_rule_probabilities_sum_to_one():
    row = pd.Series({"Age": 30, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76})
    probs = rule_based_probabilities(row)
    assert abs(probs.sum() - 1.0) < 1e-9
    print("[PASS] rule-based probability distribution sums to 1")


def test_severe_flag_pushes_toward_high_risk():
    severe_row = pd.Series({"Age": 30, "SystolicBP": 170, "DiastolicBP": 115, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76})
    probs = rule_based_probabilities(severe_row)
    assert probs.argmax() == CLASS_TO_IDX["high risk"]
    print("[PASS] severe clinical flags push rule-based prediction to high risk")


def test_hybrid_predict_shape():
    if not _require_csv():
        return
    df = load_dataset(CSV_PATH).head(20).reset_index(drop=True)
    fake_ml_probs = np.tile([0.33, 0.33, 0.34], (len(df), 1))
    preds = hybrid_predict(df, fake_ml_probs)
    assert len(preds) == len(df)
    assert set(preds.tolist()) <= {0, 1, 2}
    print("[PASS] hybrid_predict returns correctly-shaped, valid-class predictions")


def test_single_dataset_evaluation_runs_and_beats_chance():
    if not _require_csv():
        return
    # lightweight: 1 seed, 3 folds, just to confirm the harness runs correctly
    # and produces a sane (well above 33% chance-level for 3 balanced-ish classes) result
    summary = run_single_dataset_evaluation(CSV_PATH, n_splits=3, seeds=(42,))
    assert summary["ml_only"]["mean_accuracy"] > 0.5, "ML-only should clear chance level by a wide margin"
    assert summary["hybrid"]["mean_accuracy"] > 0.5
    assert summary["rules_only"]["mean_accuracy"] > 0.0
    print(f"[PASS] evaluation harness runs end-to-end; "
          f"ML-only={summary['ml_only']['mean_accuracy']*100:.1f}%, "
          f"hybrid={summary['hybrid']['mean_accuracy']*100:.1f}%, "
          f"rules-only={summary['rules_only']['mean_accuracy']*100:.1f}%")


if __name__ == "__main__":
    test_rule_probabilities_sum_to_one()
    test_severe_flag_pushes_toward_high_risk()
    test_hybrid_predict_shape()
    test_single_dataset_evaluation_runs_and_beats_chance()
    print("\nAll hybrid_model tests passed (or skipped if CSV absent).")
