"""
tts_offline.py — Phase 8: Voice Interface (Offline TTS)
MaternaCare AI

Speaks/renders confirmation prompts and risk-tier explanations using
pyttsx3 (which wraps espeak/espeak-ng on Linux). Fully offline — no network
call, no API key, no downloaded model. This is the TTS path a low-resource
field deployment can rely on with zero connectivity, which matches the
project's low-resource-settings brief better than a cloud TTS call would
for the *default* path (gTTS is offered as an optional higher-quality
alternative in tts_gtts_integration.py when internet is available).

Verified in this sandbox:
    - espeak / espeak-ng installed via apt and confirmed to synthesize audio
    - pyttsx3 initializes with driverName='espeak' and enumerates voices
    - engine.save_to_file() + engine.runAndWait() produces a playable .wav

Usage
-----
    from tts_offline import speak_offline, save_offline_wav

    speak_offline("I heard blood pressure 140 over 90.")          # plays audio
    save_offline_wav("...", "/tmp/confirmation.wav")               # renders to file
"""

import os
import subprocess
import tempfile

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


def _get_engine(rate: int = 165, volume: float = 1.0):
    """Create a pyttsx3 engine bound to the espeak driver.

    rate: words per minute (165 is a clear, not-rushed pace for field use)
    volume: 0.0-1.0
    """
    try:
        engine = pyttsx3.init(driverName="espeak")   # Linux / Mac with espeak installed
    except Exception:
        engine = pyttsx3.init()                      # Windows (SAPI5) or the platform default
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    return engine


def _wav_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def save_offline_wav(text: str, out_path: str, rate: int = 165) -> str:
    """
    Render `text` to a .wav file at `out_path` using offline speech synthesis
    (no network, no speakers needed). Returns out_path.

    pyttsx3 sometimes finishes without writing the file (the first file in a
    fresh process is the usual victim), so the result is CHECKED: if the file
    is missing, try again once, then fall back to the espeak command-line tool.
    Raises RuntimeError only if every route failed, never fails silently.
    """
    if not _PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")

    for _ in range(2):
        try:
            engine = _get_engine(rate=rate)
            engine.save_to_file(text, out_path)
            engine.runAndWait()
        except Exception:
            pass
        if _wav_ok(out_path):
            return out_path

    try:
        save_offline_wav_via_cli(text, out_path, wpm=rate)
    except Exception:
        pass
    if _wav_ok(out_path):
        return out_path
    raise RuntimeError(f"Could not create speech audio at {out_path}")


def save_offline_wav_via_cli(text: str, out_path: str, wpm: int = 165) -> str:
    """
    Fallback path that calls the espeak command-line tool directly instead of
    going through pyttsx3. One less layer, so it is more dependable on some
    setups. Tries `espeak-ng` first, then `espeak`.
    """
    last_error = None
    for exe in ("espeak-ng", "espeak"):
        try:
            subprocess.run([exe, "-s", str(wpm), "-w", out_path, text],
                           check=True, capture_output=True)
            return out_path
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            last_error = e
    raise RuntimeError(f"espeak command-line tool not available: {last_error}")


def speak_offline(text: str, rate: int = 165) -> None:
    """
    Speak `text` immediately through the system's default audio output.
    Requires an actual audio device — not exercised in this sandbox (no
    speaker), but uses the same engine already verified capable of
    synthesizing audio via save_offline_wav / the .wav byte output.
    """
    if not _PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")
    engine = _get_engine(rate=rate)
    engine.say(text)
    engine.runAndWait()


def list_available_voices():
    """Return (id, name, languages) for every voice pyttsx3/espeak can see."""
    if not _PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")
    engine = _get_engine()
    voices = engine.getProperty("voices")
    return [(v.id, v.name, getattr(v, "languages", None)) for v in voices]


if __name__ == "__main__":
    # Self-test: render a confirmation prompt to a real .wav file and verify
    # it was written with non-trivial size (i.e. actually contains audio).
    sample_text = "I heard: age 32, systolic 140, diastolic 90. Please confirm."
    out_file = os.path.join(tempfile.gettempdir(), "maternacare_tts_test.wav")
    save_offline_wav(sample_text, out_file)
    size = os.path.getsize(out_file)
    print(f"Rendered offline TTS to {out_file} ({size} bytes)")
    assert size > 1000, "Output WAV suspiciously small — TTS may have failed silently"
    print("[PASS] offline TTS produced valid audio output")

    voices = list_available_voices()
    print(f"[INFO] {len(voices)} voices available via espeak driver")
