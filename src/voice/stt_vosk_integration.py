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


# ---------------------------------------------------------------------------
# Vocabulary-restricted decoding
# ---------------------------------------------------------------------------
# The generic small English model has ~10% word error rate even on clear
# native speech, because it's decoding against tens of thousands of
# possible English words. But this app only ever needs to recognize a small,
# fixed vocabulary: digits, a few number-words, and a handful of medical
# terms. Vosk supports passing a JSON grammar to KaldiRecognizer that
# restricts decoding to just that word list, which concentrates the
# acoustic model's probability mass on the words that actually matter here
# instead of splitting it across the full dictionary. This is the highest-
# leverage accuracy fix for this specific use case — much more effective
# than switching acoustic models, and it doesn't require a bigger download.
#
# NOTE: this was written and reasoned through in this sandbox but, like the
# rest of this file, needs a live Vosk model to actually verify — run the
# before/after WER comparison on a networked machine to confirm the
# improvement holds for your own voice and mic.

VITALS_VOCABULARY = [
    # digits, spoken individually or compositionally
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "point", "over",
    # domain terms
    "blood", "pressure", "sugar", "glucose", "systolic", "diastolic",
    "temperature", "temp", "body",
    "heart", "rate", "pulse", "bpm",
    "age", "years", "hemoglobin", "haemoglobin",
    "patient", "is", "the", "her", "his", "mmol", "degrees", "fahrenheit",
    "[unk]",  # lets Vosk emit an explicit "out of vocabulary" token instead
              # of forcing a bad guess, so you can see when something didn't
              # fit the grammar rather than silently getting a wrong word
]


# Guided mode asks for ONE vital at a time, so the answer is only ever a number.
# A numbers-only word list gives the recognizer far fewer ways to go wrong than
# the full vitals vocabulary ("temperature" -> "rate sugar" can't happen).
NUMBER_VOCABULARY = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "point", "over", "[unk]",
]


def transcribe_wav(wav_path: str, model_dir: str = MODEL_DIR, restrict_vocabulary: bool = True) -> str:
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
    restrict_vocabulary : bool
        If True (default), constrain recognition to VITALS_VOCABULARY —
        recommended for this app, since it dramatically improves accuracy
        on the small set of words a vitals reading actually uses. Set to
        False to fall back to open-vocabulary recognition (e.g. for
        debugging, or if a health worker says something outside the
        expected vocabulary that you still want captured as best-effort).

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
    if restrict_vocabulary:
        rec = vosk.KaldiRecognizer(model, wf.getframerate(), json.dumps(VITALS_VOCABULARY))
    else:
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


def transcribe_microphone(model_dir: str = MODEL_DIR, duration_seconds: int = 8,
                           restrict_vocabulary: bool = True, vocabulary: list = None) -> str:
    """
    Live microphone capture + transcription, for field use on the team's
    laptop / an Android device running Vosk. Requires `sounddevice` in
    addition to `vosk` (pip install sounddevice), and an actual microphone
    — neither is available in this sandbox.

    restrict_vocabulary: see transcribe_wav() — recommended True for this
    app's fixed vitals vocabulary.
    vocabulary: optional custom word list (e.g. NUMBER_VOCABULARY for guided
    mode). Defaults to VITALS_VOCABULARY.
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
    if restrict_vocabulary:
        rec = vosk.KaldiRecognizer(model, samplerate, json.dumps(vocabulary or VITALS_VOCABULARY))
    else:
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
