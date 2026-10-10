"""
guided_entry.py — Phase 8: guided voice entry of vitals
MaternaCare AI

The computer asks for ONE vital at a time. Because each answer is just a
number, nothing depends on the recognizer catching words like "temperature"
or "over" (the exact things the free-form mode kept missing).

For every vital:
    1. the computer says (and prints) what it needs
    2. you press Enter and SPEAK the number  -- or type it instead
    3. it shows what it understood and you confirm with Enter (or type n to redo)

Nothing out of range is accepted, and nothing reaches the rules engine until
every vital has been confirmed by a person.

Run it:   py src/voice/guided_entry.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from vitals_parser import (  # noqa: E402
    GUIDED_FIELDS, ParsedVitals, parse_field_value, format_confirmation_prompt,
)


def _default_say(text: str) -> None:
    print(f"\n>> {text}")
    try:
        from tts_offline import speak_offline
        speak_offline(text)
    except Exception:
        pass  # no speaker / no TTS engine: the printed line is enough


def _default_record(field: str, seconds: int) -> str:
    from stt_vosk_integration import transcribe_microphone, NUMBER_VOCABULARY
    return transcribe_microphone(duration_seconds=seconds, vocabulary=NUMBER_VOCABULARY)


def _show(field: str, value) -> str:
    if field == "BP":
        return f"blood pressure {value[0]:g} over {value[1]:g}"
    names = {"Age": "age", "BS": "blood sugar", "BodyTemp": "temperature", "HeartRate": "heart rate"}
    return f"{names.get(field, field)} {value:g}"


def run_guided_session(record_fn=None, ask_fn=None, say_fn=None,
                       seconds: int = 6, max_tries: int = 3, out_dir: str = None) -> dict:
    """
    record_fn(field, seconds) -> transcript   (default: Vosk microphone)
    ask_fn(prompt)            -> typed text   (default: input)
    say_fn(text)                              (default: print + offline voice)
    The three are injectable so the whole flow can be tested without a mic.
    """
    record_fn = record_fn or _default_record
    ask_fn = ask_fn or input
    say_fn = say_fn or _default_say
    out_dir = out_dir or os.path.join(tempfile.gettempdir(), "maternacare_voice")

    vitals = ParsedVitals()
    transcripts = []

    for field, description in GUIDED_FIELDS:
        accepted = False
        for _ in range(max_tries):
            say_fn(f"Please say the {description}.")
            typed = ask_fn("Press Enter, then speak  (or type the value and press Enter): ").strip()
            transcript = typed if typed else record_fn(field, seconds)
            print(f"   heard: {transcript!r}")

            value, error = parse_field_value(transcript, field)
            if error:
                say_fn(error + " Let's try again.")
                continue

            reply = ask_fn(f"   I understood {_show(field, value)}.  Enter = correct, n = redo: ").strip().lower()
            if reply in ("", "y", "yes"):
                transcripts.append(f"{field}: {transcript}")
                if field == "BP":
                    vitals.SystolicBP, vitals.DiastolicBP = value
                else:
                    setattr(vitals, field, value)
                accepted = True
                break
        if not accepted:
            say_fn(f"Could not confirm {description}; leaving it empty.")

    vitals.raw_transcript = " | ".join(transcripts)
    confirmation_text = format_confirmation_prompt(vitals)
    say_fn(confirmation_text)

    wav_path = None
    try:
        from tts_offline import save_offline_wav
        os.makedirs(out_dir, exist_ok=True)
        wav_path = save_offline_wav(confirmation_text, os.path.join(out_dir, "confirmation.wav"))
    except Exception:
        pass

    return {
        "vitals": vitals.as_dict(),
        "confirmation_text": confirmation_text,
        "confirmation_wav": wav_path,
        "ready_for_rules_engine": len(vitals.missing_fields()) == 0 and not vitals.warnings,
    }


if __name__ == "__main__":
    print("MaternaCare AI - guided voice entry")
    print("Speak clearly, one number at a time. You can also type a value instead of speaking.")
    result = run_guided_session()
    print("\n=== RESULT ===")
    for key, val in result["vitals"].items():
        if key not in ("warnings", "raw_transcript") and val is not None:
            print(f"  {key}: {val:g}")
    print("  ready_for_rules_engine:", result["ready_for_rules_engine"])
