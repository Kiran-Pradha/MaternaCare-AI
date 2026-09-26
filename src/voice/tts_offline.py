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
import sys
import tempfile

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


def _get_engine(rate: int = 165, volume: float = 1.0):
    """Create a pyttsx3 engine using the best available local driver.

    On Linux, espeak/espeak-ng is the expected backend. On Windows,
    pyttsx3 should use the built-in SAPI5 driver if available. Falling back
    to the default init() call allows the project to run in the current
    environment without an explicit system package install.

    rate: words per minute (165 is a clear, not-rushed pace for field use)
    volume: 0.0-1.0
    """
    driver_names = []
    if os.name == "nt":
        driver_names = ["sapi5", "espeak"]
    elif sys.platform == "darwin":
        driver_names = ["nsss", "espeak"]
    else:
        driver_names = ["espeak", "sapi5"]

    last_exc = None
    for driver_name in driver_names:
        try:
            if driver_name is None:
                engine = pyttsx3.init()
            else:
                engine = pyttsx3.init(driverName=driver_name)
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            return engine
        except Exception as exc:  # pragma: no cover - OS-availability path
            last_exc = exc

    # Last resort: let pyttsx3 auto-detect the best installed driver.
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        return engine
    except Exception as exc:  # pragma: no cover - final fallback guard
        last_exc = exc

    raise RuntimeError(
        "No compatible pyttsx3 TTS driver is available on this machine. "
        "Install a system TTS backend or configure pyttsx3 to use a supported driver."
    ) from last_exc


def save_offline_wav(text: str, out_path: str, rate: int = 165) -> str:
    """
    Render `text` to a .wav file at `out_path` using the offline espeak
    backend. Returns out_path. Does not require a display, speakers, or
    network — safe to call in a headless / CI / sandbox environment, which
    is exactly why this is the path exercised by the automated tests below
    rather than speak_offline() (which requires an actual audio device).
    """
    if not _PYTTSX3_AVAILABLE:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")
    engine = _get_engine(rate=rate)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path


def save_offline_wav_via_cli(text: str, out_path: str, wpm: int = 165) -> str:
    """
    Fallback / alternative path that shells out to the `espeak` CLI directly
    instead of going through pyttsx3. Useful if pyttsx3's engine loop ever
    misbehaves in a particular deployment environment (a known class of
    issue on some headless Linux setups) — this path has one less
    abstraction layer between the code and the audio.
    """
    subprocess.run(
        ["espeak", "-s", str(wpm), "-w", out_path, text],
        check=True,
        capture_output=True,
    )
    return out_path


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
