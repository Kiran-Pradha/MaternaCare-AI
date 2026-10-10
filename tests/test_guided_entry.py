"""test_guided_entry.py — tests the guided voice flow with a pretend
microphone (no audio, no Vosk, no network needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "voice"))
from guided_entry import run_guided_session
from vitals_parser import parse_field_value


def _session(transcripts, typed=None, confirms=None, max_tries=3):
    """Run a session where the 'microphone' returns `transcripts` in order."""
    t, ty, co = list(transcripts), list(typed or []), list(confirms or [])
    heard_calls = []

    def record_fn(field, seconds):
        heard_calls.append(field)
        return t.pop(0)

    def ask_fn(prompt):
        if "speak" in prompt:                       # "Press Enter, then speak ..."
            return ty.pop(0) if ty else ""
        return co.pop(0) if co else ""              # the confirmation question

    spoken = []
    result = run_guided_session(record_fn=record_fn, ask_fn=ask_fn, say_fn=spoken.append,
                                max_tries=max_tries, out_dir="/tmp/maternacare_test")
    return result, heard_calls, spoken


def test_happy_path():
    result, _, _ = _session(["thirty two", "one four zero ninety", "seven point five",
                             "nine eight point six", "eighty eight"])
    v = result["vitals"]
    assert (v["Age"], v["SystolicBP"], v["DiastolicBP"], v["BS"], v["BodyTemp"], v["HeartRate"]) == \
           (32, 140, 90, 7.5, 98.6, 88)
    assert result["ready_for_rules_engine"] is True
    print("[PASS] guided happy path: all six vitals captured, ready for rules engine")


def test_retry_after_celsius_answer():
    result, calls, spoken = _session(["thirty two", "one twenty over eighty", "six point two",
                                      "thirty seven", "ninety eight point six", "seventy six"])
    assert result["vitals"]["BodyTemp"] == 98.6
    assert calls.count("BodyTemp") == 2
    assert any("Celsius" in s for s in spoken)
    print("[PASS] Celsius answer rejected with a hint, then re-asked and accepted")


def test_person_can_reject_what_was_heard():
    # first heart-rate answer is understood as 88 but the person says 'n'
    result, calls, _ = _session(["thirty two", "one twenty over eighty", "six point two",
                                 "ninety eight point six", "eighty eight", "seventy"],
                                confirms=["", "", "", "", "n", ""])
    assert result["vitals"]["HeartRate"] == 70
    print("[PASS] pressing n discards a misheard value and asks again")


def test_typed_value_instead_of_speaking():
    result, calls, _ = _session(["thirty two", "six point two", "ninety eight point six", "seventy six"],
                                typed=["", "120/80", "", "", ""])
    v = result["vitals"]
    assert (v["SystolicBP"], v["DiastolicBP"]) == (120, 80)
    assert "BP" not in calls          # mic was not used for BP
    print("[PASS] typed '120/80' accepted without using the microphone")


def test_unconfirmable_field_blocks_rules_engine():
    # BP never heard properly in 3 tries -> must NOT be ready
    result, _, _ = _session(["thirty two", "one forty", "one forty", "one forty",
                             "six point two", "ninety eight point six", "seventy six"])
    assert result["vitals"]["SystolicBP"] is None
    assert result["ready_for_rules_engine"] is False
    print("[PASS] an unconfirmed vital leaves ready_for_rules_engine False")


def test_field_parser_edge_cases():
    assert parse_field_value("one four zero ninety", "BP") == ((140.0, 90.0), None)
    assert parse_field_value("ninety over one forty", "BP")[0] is None      # top < bottom
    assert parse_field_value("", "Age")[0] is None
    assert parse_field_value("thirty two forty", "Age")[0] is None          # two numbers for one field
    assert parse_field_value("eight", "HeartRate")[0] is None               # out of range
    print("[PASS] single-field parser rejects empty, ambiguous, reversed and out-of-range answers")


if __name__ == "__main__":
    test_happy_path()
    test_retry_after_celsius_answer()
    test_person_can_reject_what_was_heard()
    test_typed_value_instead_of_speaking()
    test_unconfirmable_field_blocks_rules_engine()
    test_field_parser_edge_cases()
    print("\nAll guided_entry tests passed.")
