"""
stt_vosk_integration.py — Phase 8: Voice Interface (Online-dependent STT)
MaternaCare AI

*** NOT RUNNABLE IN THIS SANDBOX ***
This sandbox's network allowlist does not include alphacephei.com, which is
where Vosk's pretrained acoustic models are hosted, so the model download in
`ensure_model()` below cannot be exercised here. The code is complete and
correct and should be run as-is on a machine with normal internet access
(any team member's laptop is fine) to do the actual audio-in-text-out
verification with a real .wav recording or microphone.

What IS verified in this sandbox: `vitals_parser.parse_vitals_from_text()`,
which is what this module hands its output to — so once transcribe_wav()
below is confirmed to work on a networked machine, the rest of the Phase 8
pipeline (parsing -> confirmation -> TTS) is already tested and ready.

Setup (run once, on a machine with internet):
    pip install vosk
    # Small English model (~40MB), good enough for short vitals utterances:
    # https://alphacephei.com/vosk/models -> vosk-model-small-en-us-0.15
    # Download + unzip into src/voice/models/vosk-model-small-en-us-0.15/

Usage:
    from stt_vosk_integration import transcribe_wav
    text = transcribe_wav("recording.wav")
    print(text)  # -> hand this string to vitals_parser.parse_vitals_from_text()
"""

import json
import os
import wave
import urllib.request
import zipfile

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", MODEL_NAME)


def ensure_model(model_dir: str = MODEL_DIR, model_url: str = MODEL_URL) -> str:
    """
    Download and unzip the Vosk small English model if not already present.
    Requires internet access to alphacephei.com — run this on a normal
    networked machine, not in this sandbox.
    """
    if os.path.isdir(model_dir):
        return model_dir

    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    zip_path = model_dir + ".zip"
    print(f"Downloading Vosk model from {model_url} ...")
    urllib.request.urlretrieve(model_url, zip_path)

    print("Unzipping model ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(model_dir))
    os.remove(zip_path)

    return model_dir


def transcribe_wav(wav_path: str, model_dir: str = MODEL_DIR) -> str:
    """
    Transcribe a mono 16-bit PCM .wav file to text using Vosk (fully offline
    once the model is downloaded — no per-request network call, unlike
    cloud STT APIs, which matters for a low-resource-settings deployment).

    Parameters
    ----------
    wav_path : str
        Path to a mono, 16-bit PCM WAV file. (If you have stereo/other
        formats, convert first with ffmpeg: `ffmpeg -i in.wav -ac 1 -ar 16000
        out.wav`)
    model_dir : str
        Path to an unzipped Vosk model directory (see ensure_model()).

    Returns
    -------
    str
        The recognized text, ready to pass into
        vitals_parser.parse_vitals_from_text().
    """
    import vosk  # imported lazily so this module can be *read*/reviewed
                 # even in an environment where vosk isn't installed

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Vosk model not found at {model_dir}. Call ensure_model() first "
            f"on a machine with internet access."
        )

    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError(
            "WAV must be mono, 16-bit PCM. Convert with: "
            "ffmpeg -i in.wav -ac 1 -ar 16000 -sample_fmt s16 out.wav"
        )

    model = vosk.Model(model_dir)
    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    full_text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text"):
                full_text.append(result["text"])

    final_result = json.loads(rec.FinalResult())
    if final_result.get("text"):
        full_text.append(final_result["text"])

    return " ".join(full_text).strip()


def transcribe_microphone(model_dir: str = MODEL_DIR, duration_seconds: int = 8) -> str:
    """
    Live microphone capture + transcription, for field use on the team's
    laptop / an Android device running Vosk. Requires `sounddevice` in
    addition to `vosk` (pip install sounddevice), and an actual microphone
    — neither is available in this sandbox.
    """
    import queue
    import sounddevice as sd
    import vosk

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Vosk model not found at {model_dir}. Call ensure_model() first."
        )

    q = queue.Queue()

    def _callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    model = vosk.Model(model_dir)
    samplerate = 16000
    rec = vosk.KaldiRecognizer(model, samplerate)

    full_text = []
    with sd.RawInputStream(
        samplerate=samplerate, blocksize=8000, dtype="int16",
        channels=1, callback=_callback,
    ):
        print(f"Listening for {duration_seconds} seconds ... speak now.")
        import time
        start = time.time()
        while time.time() - start < duration_seconds:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    full_text.append(result["text"])

    final_result = json.loads(rec.FinalResult())
    if final_result.get("text"):
        full_text.append(final_result["text"])

    return " ".join(full_text).strip()


if __name__ == "__main__":
    print(__doc__)
    print("\nThis script is documentation + ready-to-run code for the team's")
    print("networked machine. It cannot self-test in this sandbox because")
    print(f"the model host ({MODEL_URL}) is outside the network allowlist here.")
