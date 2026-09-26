"""
Unit tests for src/voice/vitals_extractor.py

Run with:  python -m pytest tests/test_vitals_extractor.py -v
or simply: python tests/test_vitals_extractor.py
No audio, no network, no model files required.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.voice.vitals_extractor import extract_vitals, format_missing_fields_prompt


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    return condition


def test_full_digit_transcript():
    t = ("Patient age is 28. BP is 120 over 80. Blood sugar is 7.5. "
         "Temperature is 98.6. Heart rate is 76.")
    r = extract_vitals(t)
    ok = (
        r.vitals.get("Age") == 28.0 and
        r.vitals.get("SystolicBP") == 120.0 and
        r.vitals.get("DiastolicBP") == 80.0 and
        r.vitals.get("BS") == 7.5 and
        r.vitals.get("BodyTemp") == 98.6 and
        r.vitals.get("HeartRate") == 76.0 and
        r.is_complete()
    )
    return check("full digit transcript, all 6 fields", ok)


def test_partial_transcript_reports_missing():
    t = "Age is 30. Systolic BP is 140. Blood sugar 6."
    r = extract_vitals(t)
    ok = (
        r.vitals.get("Age") == 30.0 and
        r.vitals.get("SystolicBP") == 140.0 and
        "DiastolicBP" in r.unmatched_fields and
        "BodyTemp" in r.unmatched_fields and
        "HeartRate" in r.unmatched_fields and
        not r.is_complete()
    )
    return check("partial transcript flags unmatched fields", ok)


def test_out_of_range_value_is_flagged_not_dropped():
    t = "Age is 28. Heart rate is 400. Blood sugar 7."
    r = extract_vitals(t)
    ok = (
        r.vitals.get("HeartRate") == 400.0 and
        "HeartRate" in r.flags
    )
    return check("out-of-range value kept but flagged", ok)


def test_hemoglobin_and_alt_phrasing():
    t = "Hb is 9.2 and the patient is 24 years old"
    r = extract_vitals(t)
    ok = (
        r.vitals.get("Hemoglobin") == 9.2 and
        r.vitals.get("Age") == 24.0
    )
    return check("hemoglobin + 'years old' phrasing", ok)


def test_missing_field_prompt_wording():
    t = "Age 22, BS 5.5"
    r = extract_vitals(t)
    prompt = format_missing_fields_prompt(r)
    ok = prompt is not None and "systolic" in prompt.lower()
    return check("missing-field follow-up prompt generated", ok)


def test_no_false_positive_on_empty_transcript():
    r = extract_vitals("")
    ok = r.vitals == {} and not r.is_complete()
    return check("empty transcript yields no vitals", ok)


def test_spoken_number_words():
    t = "age is twenty eight, blood sugar seven point five"
    r = extract_vitals(t)
    ok = r.vitals.get("Age") == 28.0 and r.vitals.get("BS") == 7.5
    return check("spoken number words normalized to digits", ok)


if __name__ == "__main__":
    tests = [
        test_full_digit_transcript,
        test_partial_transcript_reports_missing,
        test_out_of_range_value_is_flagged_not_dropped,
        test_hemoglobin_and_alt_phrasing,
        test_missing_field_prompt_wording,
        test_no_false_positive_on_empty_transcript,
        test_spoken_number_words,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    sys.exit(0 if all(results) else 1)
