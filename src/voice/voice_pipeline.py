"""
voice_pipeline.py — Phase 8: Voice Interface (Orchestration)
MaternaCare AI

Wires the voice interface into the rest of the pipeline:

    audio in --(STT: Vosk, online-machine)--> transcript text
        --(vitals_parser, tested offline)--> ParsedVitals
        --(confirmation prompt --(TTS: offline espeak or online gTTS)--> spoken back)
        --(rules_engine.py from Phase 7)--> risk tier + explanation
        --(TTS)--> spoken risk explanation

This module only depends on vitals_parser and tts_offline directly (both
fully tested in this sandbox). It imports stt_vosk_integration and
rules_engine lazily / defensively so this file stays importable and
partially testable even before both of those are wired up on a networked
machine with the rest of the repo checked out.

`run_from_transcript()` is the entry point already fully testable here —
it's the "audio in" step alone that needs the team's networked machine.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vitals_parser import parse_vitals_from_text, format_confirmation_prompt, ParsedVitals
from tts_offline import save_offline_wav


def run_from_transcript(transcript: str, out_dir: str = "/tmp/maternacare_voice") -> dict:
    """
    Fully testable half of the pipeline: takes an already-transcribed
    string (as Vosk would produce) and runs it through parsing +
    confirmation-prompt generation + offline TTS rendering.

    Returns a dict with the parsed vitals, the confirmation text, and the
    path to the rendered confirmation .wav.
    """
    os.makedirs(out_dir, exist_ok=True)

    vitals = parse_vitals_from_text(transcript)
    confirmation_text = format_confirmation_prompt(vitals)
    confirmation_wav = os.path.join(out_dir, "confirmation.wav")
    save_offline_wav(confirmation_text, confirmation_wav)

    result = {
        "transcript": transcript,
        "vitals": vitals.as_dict(),
        "confirmation_text": confirmation_text,
        "confirmation_wav": confirmation_wav,
        "ready_for_rules_engine": len(vitals.missing_fields()) == 0,
    }
    return result


def run_from_wav(wav_path: str, out_dir: str = "/tmp/maternacare_voice") -> dict:
    """
    Full pipeline entry point: audio file -> transcript (Vosk) -> vitals ->
    confirmation. This requires stt_vosk_integration's model to be present,
    which requires internet access this sandbox doesn't have — so this
    function is written and ready but should be smoke-tested by the team on
    a networked machine before Phase 9 integration.
    """
    from stt_vosk_integration import transcribe_wav

    transcript = transcribe_wav(wav_path)
    return run_from_transcript(transcript, out_dir=out_dir)


def speak_risk_explanation(risk_tier: str, explanation: str, out_dir: str = "/tmp/maternacare_voice") -> str:
    """
    Given a risk tier + explanation string from Phase 7's rules_engine.py,
    render it to speech offline. Kept as a thin, separate function so
    Phase 9 integration can call it right after invoking the rules engine
    without needing to know anything about the TTS backend.
    """
    os.makedirs(out_dir, exist_ok=True)
    text = f"Risk level: {risk_tier}. {explanation}"
    out_path = os.path.join(out_dir, "risk_explanation.wav")
    save_offline_wav(text, out_path)
    return out_path


if __name__ == "__main__":
    demo_transcript = (
        "Patient age twenty eight, blood pressure 150 over 95, "
        "blood sugar 8.2, temperature 99.5, heart rate 92"
    )
    result = run_from_transcript(demo_transcript)
    print("Parsed vitals:", result["vitals"])
    print("Confirmation text:", result["confirmation_text"])
    print("Confirmation WAV:", result["confirmation_wav"],
          f"({os.path.getsize(result['confirmation_wav'])} bytes)")
    print("Ready for rules engine:", result["ready_for_rules_engine"])
