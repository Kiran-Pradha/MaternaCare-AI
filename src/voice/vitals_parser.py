"""
vitals_parser.py — Phase 8: Voice Interface (Text-Layer Component)
MaternaCare AI

Extracts structured vitals (Age, SystolicBP, DiastolicBP, BS, BodyTemp,
HeartRate) from free-form transcribed speech text.

DESIGN RATIONALE
-----------------
This module is deliberately decoupled from the STT engine that produces the
text. Vosk (offline) and any other STT backend both ultimately hand back a
plain string transcript — so all of the "intelligence" of understanding what
a health worker said lives here, in pure string/regex logic, and is fully
unit-testable without a live microphone, an audio file, or network access to
download a Vosk acoustic model. This was a deliberate scoping decision made
because this sandbox's network allowlist does not include the Vosk model
host, so audio-in-audio-out could not be exercised end-to-end here. The
parser is the part of the voice pipeline whose correctness matters most
(get the wrong BP number and everything downstream — rules engine, SHAP
explanation, risk tier — is wrong), so it is the part built and tested most
rigorously in this phase.

Handles:
- Multiple phrasings ("BP is 140 over 90", "blood pressure 140/90", "systolic 140 diastolic 90")
- Units spoken or omitted ("blood sugar 7.5", "blood sugar is 7 point 5 mmol")
- Word-numbers ("temperature is ninety eight point six")
- Heart rate synonyms (pulse, heart rate, HR)
- Age, hemoglobin (for the anemia tier) as a bonus field
- Partial transcripts (only some vitals mentioned) — returns whatever it found
- Basic clinical range validation with warnings (does not silently accept
  physiologically impossible values from a mis-heard transcript)
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# Word-number support (STT engines sometimes emit "ninety eight" instead of
# "98", especially with smaller offline models / noisy audio)
# ---------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(text: str) -> Optional[float]:
    """Convert a short word-number phrase like 'ninety eight point six' to 98.6."""
    text = text.strip().lower()
    if "point" in text:
        whole_part, _, frac_part = text.partition("point")
        whole = _words_to_number(whole_part)
        if whole is None:
            return None
        frac_digits = []
        for w in frac_part.strip().split():
            if w in _ONES:
                frac_digits.append(str(_ONES[w]))
            else:
                return None
        if not frac_digits:
            return whole
        return float(f"{int(whole)}.{''.join(frac_digits)}")

    tokens = text.split()
    total = 0
    matched = False
    for tok in tokens:
        if tok in _TENS:
            total += _TENS[tok]
            matched = True
        elif tok in _ONES:
            total += _ONES[tok]
            matched = True
        elif tok in ("and",):
            continue
        else:
            return None
    return float(total) if matched else None


_NUMBER_WORD_PATTERN = re.compile(
    r"\b((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*)+"
    r"(?:point\s+(?:zero|one|two|three|four|five|six|seven|eight|nine)(?:\s+"
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine))*)?)\b",
    re.IGNORECASE,
)


def _normalize_word_numbers(text: str) -> str:
    """Replace spoken word-numbers in the text with digit strings, e.g.
    'ninety eight point six' -> '98.6'. Leaves digit numbers untouched."""
    def _sub(m):
        val = _words_to_number(m.group(1))
        if val is None:
            return m.group(0)
        if val == int(val):
            return str(int(val))
        return str(val)

    return _NUMBER_WORD_PATTERN.sub(_sub, text)


# ---------------------------------------------------------------------------
# Clinical validity ranges — used only to flag likely mis-transcriptions,
# never to silently discard data. A rules_engine.py downstream integration
# should treat `warnings` as "confirm with health worker before using".
# ---------------------------------------------------------------------------

VALID_RANGES = {
    "Age": (10, 60),           # years, maternal context
    "SystolicBP": (70, 220),   # mmHg
    "DiastolicBP": (40, 140),  # mmHg
    "BS": (2.0, 25.0),         # blood sugar, mmol/L
    "BodyTemp": (90.0, 108.0), # Fahrenheit (matches source dataset's units)
    "HeartRate": (30, 200),    # bpm
    "Hemoglobin": (3.0, 20.0), # g/dL
}


@dataclass
class ParsedVitals:
    Age: Optional[float] = None
    SystolicBP: Optional[float] = None
    DiastolicBP: Optional[float] = None
    BS: Optional[float] = None
    BodyTemp: Optional[float] = None
    HeartRate: Optional[float] = None
    Hemoglobin: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    raw_transcript: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)

    def fields_found(self) -> List[str]:
        return [k for k, v in asdict(self).items()
                if k not in ("warnings", "raw_transcript") and v is not None]

    def missing_fields(self, required=("Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate")) -> List[str]:
        return [f for f in required if getattr(self, f) is None]


_NUM = r"(\d+(?:\.\d+)?)"

_PATTERNS = {
    # BP: handle "140 over 90", "140/90", "systolic 140 diastolic 90" separately
    "bp_combined": re.compile(
        rf"(?:blood\s*pressure|bp)\D{{0,10}}{_NUM}\s*(?:over|/|-)\s*{_NUM}", re.IGNORECASE
    ),
    "bp_bare_combined": re.compile(rf"{_NUM}\s*(?:over|/)\s*{_NUM}\s*(?:mm\s*hg)?", re.IGNORECASE),
    "systolic": re.compile(rf"systolic\D{{0,10}}{_NUM}", re.IGNORECASE),
    "diastolic": re.compile(rf"diastolic\D{{0,10}}{_NUM}", re.IGNORECASE),
    "age": re.compile(rf"\bage\D{{0,10}}{_NUM}\s*(?:years?)?", re.IGNORECASE),
    "bs": re.compile(
        rf"(?:blood\s*sugar|glucose|bs)\D{{0,10}}{_NUM}\s*(?:mmol\s*/?\s*l)?", re.IGNORECASE
    ),
    "temp": re.compile(
        rf"(?:body\s*temp(?:erature)?|temp(?:erature)?)\D{{0,10}}{_NUM}\s*(?:f|fahrenheit)?", re.IGNORECASE
    ),
    "hr": re.compile(
        rf"(?:heart\s*rate|pulse|hr)\D{{0,10}}{_NUM}\s*(?:bpm|beats)?", re.IGNORECASE
    ),
    "hb": re.compile(
        rf"(?:hemoglobin|haemoglobin|hb)\D{{0,10}}{_NUM}\s*(?:g\s*/?\s*dl)?", re.IGNORECASE
    ),
}


def parse_vitals_from_text(transcript: str) -> ParsedVitals:
    """
    Extract vitals from a transcribed utterance.

    Parameters
    ----------
    transcript : str
        Raw text as returned by an STT engine (e.g. Vosk's result['text']).

    Returns
    -------
    ParsedVitals
        Dataclass with whichever fields were found (others remain None),
        plus a list of human-readable warnings for out-of-range values.
    """
    original = transcript
    text = _normalize_word_numbers(transcript)
    result = ParsedVitals(raw_transcript=original)

    # --- Blood pressure (combined systolic/diastolic forms first) ---
    m = _PATTERNS["bp_combined"].search(text) or _PATTERNS["bp_bare_combined"].search(text)
    if m:
        result.SystolicBP = float(m.group(1))
        result.DiastolicBP = float(m.group(2))
    else:
        m_sys = _PATTERNS["systolic"].search(text)
        m_dia = _PATTERNS["diastolic"].search(text)
        if m_sys:
            result.SystolicBP = float(m_sys.group(1))
        if m_dia:
            result.DiastolicBP = float(m_dia.group(1))

    # --- Age ---
    m = _PATTERNS["age"].search(text)
    if m:
        result.Age = float(m.group(1))

    # --- Blood sugar ---
    m = _PATTERNS["bs"].search(text)
    if m:
        result.BS = float(m.group(1))

    # --- Body temperature ---
    m = _PATTERNS["temp"].search(text)
    if m:
        result.BodyTemp = float(m.group(1))

    # --- Heart rate / pulse ---
    m = _PATTERNS["hr"].search(text)
    if m:
        result.HeartRate = float(m.group(1))

    # --- Hemoglobin (anemia-tier bonus field) ---
    m = _PATTERNS["hb"].search(text)
    if m:
        result.Hemoglobin = float(m.group(1))

    # --- Range validation (warn, never silently drop) ---
    for field_name, (lo, hi) in VALID_RANGES.items():
        val = getattr(result, field_name)
        if val is not None and not (lo <= val <= hi):
            result.warnings.append(
                f"{field_name}={val} is outside plausible range [{lo}, {hi}] — "
                f"please confirm with health worker, possible mis-transcription."
            )

    return result


def format_confirmation_prompt(vitals: ParsedVitals) -> str:
    """
    Build a short spoken confirmation string, meant to be handed to the TTS
    layer so the health worker can verbally confirm what was captured before
    it's passed on to the rules engine / ML model.
    """
    parts = []
    label_map = [
        ("Age", "age {v:.0f}"),
        ("SystolicBP", "systolic {v:.0f}"),
        ("DiastolicBP", "diastolic {v:.0f}"),
        ("BS", "blood sugar {v:g}"),
        ("BodyTemp", "temperature {v:g}"),
        ("HeartRate", "heart rate {v:.0f}"),
        ("Hemoglobin", "hemoglobin {v:g}"),
    ]
    for field_name, template in label_map:
        val = getattr(vitals, field_name)
        if val is not None:
            parts.append(template.format(v=val))

    if not parts:
        return "I did not catch any vitals. Please repeat."

    missing = vitals.missing_fields()
    msg = "I heard: " + ", ".join(parts) + "."
    if missing:
        msg += " Still need: " + ", ".join(missing) + "."
    if vitals.warnings:
        msg += " Warning: some values look unusual, please double check."
    return msg


if __name__ == "__main__":
    demo = "Patient age thirty two, BP is 140 over 90, blood sugar 7.5, temperature ninety eight point six, heart rate 88"
    parsed = parse_vitals_from_text(demo)
    print(parsed.as_dict())
    print(format_confirmation_prompt(parsed))
