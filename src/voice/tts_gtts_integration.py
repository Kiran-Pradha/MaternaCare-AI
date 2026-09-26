"""
tts_gtts_integration.py — Phase 8: Voice Interface (Online-dependent TTS)
MaternaCare AI

*** NOT RUNNABLE IN THIS SANDBOX ***
This sandbox's network allowlist does not include Google's TTS endpoint
(translate.google.com, used internally by the gTTS package), so this cannot
be exercised here. The code is complete and correct and should be run as-is
on a machine with normal internet access.

Why this exists alongside tts_offline.py: espeak/pyttsx3 (tts_offline.py) is
the default, zero-connectivity-required path and is what should ship for
low-resource field use. gTTS is offered as an optional higher-naturalness
alternative for sites that do have reliable internet — the voice quality
difference between espeak and gTTS is substantial, so it's worth having both
paths and letting the deployment choose, rather than picking one and locking
the project into it.

Usage (on a networked machine):
    pip install gTTS
    from tts_gtts_integration import speak_gtts, save_gtts_mp3
    save_gtts_mp3("I heard blood pressure 140 over 90.", "/tmp/confirm.mp3")
"""

import os


def save_gtts_mp3(text: str, out_path: str, lang: str = "en", slow: bool = False) -> str:
    """
    Render `text` to an .mp3 file using Google's TTS service via gTTS.
    Requires internet access. Not tested in this sandbox — run on the
    team's networked machine to verify end-to-end.

    Parameters
    ----------
    text : str
    out_path : str
        Destination .mp3 path.
    lang : str
        BCP-47-ish language code gTTS accepts (e.g. 'en', 'hi', 'ta' for
        Tamil, 'bn' for Bengali) — relevant for regional-language rollout
        in the low-resource maternal-health field context this project
        targets.
    slow : bool
        Slower, more deliberate speech — worth considering as the default
        for field use given the audience may be listening under noisy or
        low-literacy conditions.
    """
    from gtts import gTTS  # lazy import so the file is reviewable without gTTS installed

    tts = gTTS(text=text, lang=lang, slow=slow)
    tts.save(out_path)
    return out_path


def speak_gtts(text: str, lang: str = "en", slow: bool = False) -> None:
    """
    Render to a temp mp3 and play it immediately via the system's default
    player. Convenience wrapper for quick manual testing on a networked
    machine with speakers.
    """
    import tempfile
    import subprocess
    import platform

    tmp_path = os.path.join(tempfile.gettempdir(), "maternacare_gtts_tmp.mp3")
    save_gtts_mp3(text, tmp_path, lang=lang, slow=slow)

    system = platform.system()
    if system == "Linux":
        subprocess.run(["mpg123", tmp_path], check=False)
    elif system == "Darwin":
        subprocess.run(["afplay", tmp_path], check=False)
    elif system == "Windows":
        os.startfile(tmp_path)  # noqa: this is a Windows-only API, fine here
    else:
        print(f"Don't know how to auto-play on {system}; file saved at {tmp_path}")


SUPPORTED_LANGUAGES_OF_INTEREST = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
}


if __name__ == "__main__":
    print(__doc__)
    print("\nThis script requires internet access to Google's TTS endpoint,")
    print("which is outside this sandbox's network allowlist. Run on a")
    print("networked machine to verify.")
    print(f"\nLanguages worth testing for regional rollout: {SUPPORTED_LANGUAGES_OF_INTEREST}")
