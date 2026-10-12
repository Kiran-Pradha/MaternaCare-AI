"""test_trajectory_classifier.py -- the synthetic trajectories' job: training and checking the
trajectory-pattern classifier. Needs data/maternal_health_risk_data.csv."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
sys.path.insert(0, os.path.join(ROOT, "src", "data"))
CSV = os.path.join(ROOT, "data", "maternal_health_risk_data.csv")

from trajectory_simulation_v2 import generate_trajectories, validate_synthetic_dataset  # noqa: E402
from trajectory_classifier import TrajectoryClassifier, PATTERNS, trajectory_features  # noqa: E402


def test_sudden_patients_are_generated_and_valid():
    df = generate_trajectories(CSV, n_patients=300, n_visits=4, seed=3, sudden_fraction=0.25,
                               trend_strength_range=(1.0, 4.0))
    labels = set(df["trend_label"])
    assert labels == {"stable", "worsening", "improving", "sudden"}, labels
    report = validate_synthetic_dataset(df, CSV)
    assert report["overall_pass"], report
    # a sudden patient's biggest systolic move is the LAST step
    sudden_ids = df.loc[df.trend_label == "sudden", "patient_id"].unique()[:50]
    n_last = 0
    for pid in sudden_ids:
        steps = df[df.patient_id == pid].sort_values("visit_number")["SystolicBP"].diff().dropna().tolist()
        n_last += steps[-1] == max(steps)
    assert n_last / len(sudden_ids) > 0.8
    print("[PASS] 'sudden' patients are generated, pass validation, and jump at the last visit")


def test_default_generation_unchanged():
    a = generate_trajectories(CSV, n_patients=50, n_visits=4, seed=11)
    assert set(a["trend_label"]) <= {"stable", "worsening", "improving"}
    print("[PASS] default generation still produces only the original three patterns")


def test_features_work_for_2_3_4_visits():
    std = {c: 1.0 for c in ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]}
    v = dict(SystolicBP=120, DiastolicBP=80, BS=7.0, BodyTemp=98.0, HeartRate=76)
    sizes = {len(trajectory_features([v] * n, std)) for n in (2, 3, 4)}
    assert len(sizes) == 1
    print("[PASS] the same number of features for 2, 3 and 4 visits")


def test_classifier_beats_guessing_and_handles_clear_cases():
    clf = TrajectoryClassifier(n_patients_per_length=300).fit(CSV)
    res = clf.evaluate(CSV, n_test_per_length=200)
    assert res["accuracy"] > res["majority_baseline"] + 0.3, res
    flat = [dict(SystolicBP=120, DiastolicBP=80, BS=7.0, BodyTemp=98.0, HeartRate=76)] * 3
    assert clf.predict(flat)[0] == "Stable"
    jump = [dict(SystolicBP=112, DiastolicBP=72, BS=7.0, BodyTemp=98.0, HeartRate=72)] * 2 + \
           [dict(SystolicBP=150, DiastolicBP=98, BS=11.5, BodyTemp=99.0, HeartRate=90)]
    assert clf.predict(jump)[0] == "Sudden Deterioration"
    assert clf.predict(flat[:1]) == ("Stable", None)       # one visit: nothing to compare
    pattern, probs = clf.predict(flat)
    assert abs(sum(probs.values()) - 1) < 1e-6 and set(probs) == set(PATTERNS)
    print(f"[PASS] classifier: {res['accuracy'] * 100:.0f}% on fresh simulated patients "
          f"(guessing: {res['majority_baseline'] * 100:.0f}%); flat -> Stable, big jump -> Sudden")


if __name__ == "__main__":
    if not os.path.exists(CSV):
        print("[SKIP] dataset not found at", CSV)
        sys.exit(0)
    test_sudden_patients_are_generated_and_valid()
    test_default_generation_unchanged()
    test_features_work_for_2_3_4_visits()
    test_classifier_beats_guessing_and_handles_clear_cases()
    print("\nAll trajectory_classifier tests passed.")
