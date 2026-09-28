"""test_trajectory_simulation_v2.py — requires the real CSV to be present
at data/maternal_health_risk_data.csv (bootstraps anchors from it)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
from trajectory_simulation_v2 import generate_trajectories, validate_synthetic_dataset, FEATURES

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "maternal_health_risk_data.csv")


def _require_csv():
    if not os.path.exists(CSV_PATH):
        print(f"[SKIP] real dataset not present at {CSV_PATH} — trajectory tests need it")
        return False
    return True


def test_generation_shape():
    if not _require_csv():
        return
    df = generate_trajectories(CSV_PATH, n_patients=20, n_visits=4, seed=1)
    assert df["patient_id"].nunique() == 20
    assert len(df) == 20 * 4
    for col in FEATURES + ["patient_id", "visit_number", "trend_label"]:
        assert col in df.columns
    print("[PASS] generated dataset has correct shape and columns")


def test_reproducibility():
    if not _require_csv():
        return
    df1 = generate_trajectories(CSV_PATH, n_patients=15, n_visits=4, seed=99)
    df2 = generate_trajectories(CSV_PATH, n_patients=15, n_visits=4, seed=99)
    assert df1.equals(df2), "same seed should produce identical output"
    print("[PASS] generation is reproducible given a fixed seed")


def test_validation_passes_on_generated_data():
    if not _require_csv():
        return
    df = generate_trajectories(CSV_PATH, n_patients=300, n_visits=4, seed=42)
    report = validate_synthetic_dataset(df, CSV_PATH)
    assert report["plausibility_check"]["pass"], "generated data should never be implausible by construction"
    assert report["visit1_distribution_check_all_pass"], "Visit-1 should statistically match real data"
    assert report["visit_to_visit_jump_check"]["pass"], "no implausible visit-to-visit jumps expected"
    assert report["overall_pass"]
    print("[PASS] full validation suite passes on a freshly generated dataset")


def test_age_constant_within_patient():
    if not _require_csv():
        return
    df = generate_trajectories(CSV_PATH, n_patients=10, n_visits=4, seed=5)
    for pid, group in df.groupby("patient_id"):
        assert group["Age"].nunique() == 1, "Age should not change across a handful of visits"
    print("[PASS] Age stays constant within each synthetic patient's trajectory")


def test_trend_labels_present():
    if not _require_csv():
        return
    df = generate_trajectories(CSV_PATH, n_patients=200, n_visits=4, risk_informed_fraction=0.4, seed=7)
    labels = set(df["trend_label"].unique())
    assert labels <= {"stable", "worsening", "improving"}
    assert "stable" in labels  # with 200 patients and 0.4 informed fraction, both should appear
    print(f"[PASS] trend labels present: {labels}")


if __name__ == "__main__":
    test_generation_shape()
    test_reproducibility()
    test_validation_passes_on_generated_data()
    test_age_constant_within_patient()
    test_trend_labels_present()
    print("\nAll trajectory_simulation_v2 tests passed (or skipped if CSV absent).")
