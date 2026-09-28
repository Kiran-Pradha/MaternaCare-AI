"""test_data_validation.py — offline, deterministic, no data download needed
(uses small inline dataframes plus the real CSV if present)."""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
from data_validation import (
    validate_dataframe, summarize_validation, clip_to_plausible,
    flag_clinical_concerns, VITAL_RANGES,
)


def test_plausibility_flags_impossible_heart_rate():
    df = pd.DataFrame([
        {"Age": 30, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 7},
        {"Age": 30, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76},
    ])
    validated = validate_dataframe(df)
    assert validated.loc[0, "any_implausible"] == True
    assert validated.loc[1, "any_implausible"] == False
    print("[PASS] impossible HeartRate=7 correctly flagged, normal value not flagged")


def test_plausibility_flags_impossible_age():
    df = pd.DataFrame([
        {"Age": 70, "SystolicBP": 120, "DiastolicBP": 80, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76},
    ])
    validated = validate_dataframe(df)
    assert validated.loc[0, "Age_implausible"] == True
    print("[PASS] implausible maternal age flagged")


def test_clip_to_plausible():
    assert clip_to_plausible(500, "SystolicBP") == VITAL_RANGES["SystolicBP"].plausible_hi
    assert clip_to_plausible(-10, "SystolicBP") == VITAL_RANGES["SystolicBP"].plausible_lo
    assert clip_to_plausible(120, "SystolicBP") == 120
    print("[PASS] clip_to_plausible clamps correctly at both ends and passes through in-range values")


def test_clinical_flags():
    severe_row = pd.Series({"Age": 30, "SystolicBP": 165, "DiastolicBP": 115, "BS": 7.0, "BodyTemp": 98.0, "HeartRate": 76})
    flags = flag_clinical_concerns(severe_row)
    assert "severe_hypertension" in flags

    normal_row = pd.Series({"Age": 28, "SystolicBP": 110, "DiastolicBP": 70, "BS": 5.0, "BodyTemp": 98.0, "HeartRate": 75})
    flags = flag_clinical_concerns(normal_row)
    assert flags == []
    print("[PASS] clinical flags correctly identify severe hypertension and correctly find nothing for a normal row")


def test_real_dataset_validation_if_present():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "maternal_health_risk_data.csv")
    if not os.path.exists(csv_path):
        print("[SKIP] real dataset not present at", csv_path)
        return
    df = pd.read_csv(csv_path)
    validated = validate_dataframe(df)
    n_bad = int(validated["any_implausible"].sum())
    # known result from manual inspection: 10 implausible rows (8 age, 2 heart rate)
    assert n_bad == 10, f"expected 10 implausible rows in real dataset, got {n_bad}"
    print(f"[PASS] real dataset validation matches known result: {n_bad} implausible rows")


if __name__ == "__main__":
    test_plausibility_flags_impossible_heart_rate()
    test_plausibility_flags_impossible_age()
    test_clip_to_plausible()
    test_clinical_flags()
    test_real_dataset_validation_if_present()
    print("\nAll data_validation tests passed.")
