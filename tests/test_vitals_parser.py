"""
test_vitals_parser.py — Phase 8 test suite
MaternaCare AI

Fully offline, no audio/network required. Run with:
    python3 -m pytest tests/test_vitals_parser.py -v
or:
    python3 tests/test_vitals_parser.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "voice"))

from vitals_parser import parse_vitals_from_text, format_confirmation_prompt, _words_to_number


CASES = [
    # (transcript, expected_dict_subset)
    (
        "Patient age thirty two, BP is 140 over 90, blood sugar 7.5, "
        "temperature ninety eight point six, heart rate 88",
        dict(Age=32.0, SystolicBP=140.0, DiastolicBP=90.0, BS=7.5, BodyTemp=98.6, HeartRate=88.0),
    ),
    (
        "blood pressure 120/80",
        dict(SystolicBP=120.0, DiastolicBP=80.0),
    ),
    (
        "systolic 150 diastolic 95",
        dict(SystolicBP=150.0, DiastolicBP=95.0),
    ),
    (
        "her blood sugar is seven point eight mmol",
        dict(BS=7.8),
    ),
    (
        "pulse is 76 bpm",
        dict(HeartRate=76.0),
    ),
    (
        "hemoglobin 9.2 g/dl",
        dict(Hemoglobin=9.2),
    ),
    (
        "age is forty five years",
        dict(Age=45.0),
    ),
    (
        # partial transcript — only some vitals mentioned
        "temperature is 99.1",
        dict(BodyTemp=99.1),
    ),
    (
        # noisy/garbled — should not crash, should find nothing
        "um the the patient uh seems fine today",
        dict(),
    ),
]

OUT_OF_RANGE_CASE = "BP is 400 over 300, heart rate 5"


def run_word_number_tests():
    assert _words_to_number("ninety eight point six") == 98.6
    assert _words_to_number("thirty two") == 32
    assert _words_to_number("seven") == 7
    assert _words_to_number("not a number") is None
    print("[PASS] word-number conversion")


def run_extraction_tests():
    for transcript, expected in CASES:
        parsed = parse_vitals_from_text(transcript)
        result = parsed.as_dict()
        for key, expected_val in expected.items():
            actual_val = result[key]
            assert actual_val == expected_val, (
                f"FAIL on '{transcript}': expected {key}={expected_val}, got {actual_val}"
            )
        # fields not in `expected` should remain None for this transcript
        for key in ("Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "Hemoglobin"):
            if key not in expected:
                assert result[key] is None, (
                    f"FAIL on '{transcript}': expected {key}=None, got {result[key]}"
                )
    print(f"[PASS] {len(CASES)} extraction phrasing/variant cases")


def run_range_warning_test():
    parsed = parse_vitals_from_text(OUT_OF_RANGE_CASE)
    assert parsed.SystolicBP == 400.0
    assert parsed.DiastolicBP == 300.0
    assert parsed.HeartRate == 5.0
    assert len(parsed.warnings) == 3, f"expected 3 range warnings, got {len(parsed.warnings)}: {parsed.warnings}"
    print("[PASS] out-of-range values flagged with warnings, not dropped")


def run_confirmation_prompt_test():
    parsed = parse_vitals_from_text("age 32, BP 140 over 90")
    prompt = format_confirmation_prompt(parsed)
    assert "age 32" in prompt
    assert "systolic 140" in prompt
    assert "Still need" in prompt  # BS, BodyTemp, HeartRate missing
    empty = parse_vitals_from_text("hello there")
    assert format_confirmation_prompt(empty) == "I did not catch any vitals. Please repeat."
    print("[PASS] confirmation prompt generation (found + missing + empty cases)")


def run_missing_fields_test():
    parsed = parse_vitals_from_text("age 32")
    missing = parsed.missing_fields()
    assert set(missing) == {"SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"}
    print("[PASS] missing_fields() correctly reports unfound required vitals")


if __name__ == "__main__":
    run_word_number_tests()
    run_extraction_tests()
    run_range_warning_test()
    run_confirmation_prompt_test()
    run_missing_fields_test()
    print("\nAll Phase 8 vitals_parser tests passed.")
