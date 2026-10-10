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


_DIGIT_WORDS = {w: v for w, v in _ONES.items() if v <= 9}
_NUMBER_VOCAB = set(_ONES) | set(_TENS) | {"point", "hundred"}
_PUNCT = ",.;:!?"


def _parse_number_word_run(words):
    """Turn a run of spoken number words into a list of numbers.

    One run can hold SEVERAL numbers, e.g. when the recognizer drops the word
    "over": "one four zero ninety" is really 140 and 90. Grammar, left to right:

      * tens word (+ optional single digit)  -> ONE number   "ninety eight" = 98
      * teen word                            -> ONE number   "twelve" = 12
      * run of single digit words            -> ONE number   "one four zero" = 140
        (digits read out one by one are JOINED, never added)
      * "one"/"two" + tens word              -> hundreds     "one twenty" = 120
      * digit + "hundred" (+ tens/ones)      -> hundreds     "one hundred five" = 105
      * "point" + digit words                -> decimal      "seven point five" = 7.5

    Returns None if a word doesn't fit the grammar.
    """
    numbers, i, n = [], 0, len(words)
    pending_hundred = False

    def push(val):
        nonlocal pending_hundred
        if pending_hundred:
            numbers[-1] += val
            pending_hundred = False
        else:
            numbers.append(float(val))

    while i < n:
        w = words[i]
        if w == "point":
            j, digits = i + 1, []
            while j < n and words[j] in _DIGIT_WORDS:
                digits.append(str(_DIGIT_WORDS[words[j]]))
                j += 1
            if not digits:
                return None
            frac = "".join(digits)
            if numbers:
                numbers[-1] = float(f"{int(numbers[-1])}.{frac}")
            else:
                numbers.append(float(f"0.{frac}"))
            pending_hundred = False
            i = j
        elif w == "hundred":
            if not numbers or numbers[-1] >= 10 or pending_hundred:
                return None
            numbers[-1] *= 100
            pending_hundred = True
            i += 1
        elif w in _TENS:
            val = _TENS[w]
            if i + 1 < n and words[i + 1] in _DIGIT_WORDS and _DIGIT_WORDS[words[i + 1]] >= 1:
                val += _DIGIT_WORDS[words[i + 1]]
                i += 1
            push(val)
            i += 1
        elif w in _ONES and _ONES[w] >= 10:
            push(_ONES[w])
            i += 1
        elif w in _DIGIT_WORDS:
            if pending_hundred:                       # "one hundred five"
                push(_DIGIT_WORDS[w])
                i += 1
                continue
            digits = []
            while i < n and words[i] in _DIGIT_WORDS:
                digits.append(str(_DIGIT_WORDS[words[i]]))
                i += 1
            if len(digits) == 1 and digits[0] in ("1", "2") and i < n and words[i] in _TENS:
                val = _TENS[words[i]]                 # "one twenty (eight)"
                i += 1
                if i < n and words[i] in _DIGIT_WORDS and _DIGIT_WORDS[words[i]] >= 1:
                    val += _DIGIT_WORDS[words[i]]
                    i += 1
                numbers.append(float(int(digits[0]) * 100 + val))
            else:
                numbers.append(float("".join(digits)))
        else:
            return None
    return numbers


def _words_to_number(text: str) -> Optional[float]:
    """Convert a phrase that holds exactly ONE spoken number to that number
    ('ninety eight point six' -> 98.6, 'one four zero' -> 140). Returns None
    if the phrase isn't a number or holds more than one."""
    nums = _parse_number_word_run(text.strip().lower().split())
    return nums[0] if nums and len(nums) == 1 else None


def _fmt_number(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def _normalize_word_numbers(text: str) -> str:
    """Replace spoken number words with digits, e.g. 'ninety eight point six'
    -> '98.6' and 'one four zero ninety' -> '140 90'. Other words and
    existing digits are left untouched."""
    tokens = text.split()
    out, i = [], 0
    while i < len(tokens):
        if tokens[i].lower().strip(_PUNCT) in _NUMBER_VOCAB:
            j, run = i, []
            while j < len(tokens) and tokens[j].lower().strip(_PUNCT) in _NUMBER_VOCAB:
                run.append(tokens[j].lower().strip(_PUNCT))
                j += 1
                if tokens[j - 1][-1] in _PUNCT:   # punctuation ends the run
                    break
            nums = _parse_number_word_run(run)
            if nums:
                out.extend(_fmt_number(x) for x in nums)
            else:
                out.extend(tokens[i:j])
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


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
        rf"\b(?:blood\s*pressure|bp)\D{{0,10}}{_NUM}\s*(?:over|/|-)\s*{_NUM}", re.IGNORECASE
    ),
    "bp_bare_combined": re.compile(rf"{_NUM}\s*(?:over|/)\s*{_NUM}\s*(?:mm\s*hg)?", re.IGNORECASE),
    # speech recognizers often drop the word "over": "blood pressure 140 90"
    "bp_no_over": re.compile(rf"\b(?:blood\s*pressure|bp)\D{{0,10}}{_NUM}\s+{_NUM}\b", re.IGNORECASE),
    "systolic": re.compile(rf"systolic\D{{0,10}}{_NUM}", re.IGNORECASE),
    "diastolic": re.compile(rf"diastolic\D{{0,10}}{_NUM}", re.IGNORECASE),
    "age": re.compile(rf"\bage\D{{0,10}}{_NUM}\s*(?:years?)?", re.IGNORECASE),
    "bs": re.compile(
        rf"\b(?:blood\s*sugar|glucose|bs)\D{{0,10}}{_NUM}\s*(?:mmol\s*/?\s*l)?", re.IGNORECASE
    ),
    "temp": re.compile(
        rf"\b(?:body\s*temp(?:erature)?|temp(?:erature)?)\D{{0,10}}{_NUM}\s*(?:f|fahrenheit)?", re.IGNORECASE
    ),
    "hr": re.compile(
        rf"\b(?:heart\s*rate|pulse|hr)\D{{0,10}}{_NUM}\s*(?:bpm|beats)?", re.IGNORECASE
    ),
    "hb": re.compile(
        rf"\b(?:hemoglobin|haemoglobin|hb)\D{{0,10}}{_NUM}\s*(?:g\s*/?\s*dl)?", re.IGNORECASE
    ),
}


def _bp_pair_ok(m) -> bool:
    """True if a regex match holds a believable (systolic, diastolic) pair."""
    if not m:
        return False
    sys_, dia = float(m.group(1)), float(m.group(2))
    return 60 <= sys_ <= 260 and 30 <= dia <= 160 and sys_ > dia


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
    elif _bp_pair_ok(_PATTERNS["bp_no_over"].search(text)):
        m2 = _PATTERNS["bp_no_over"].search(text)
        result.SystolicBP = float(m2.group(1))
        result.DiastolicBP = float(m2.group(2))
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


# ---------------------------------------------------------------------------
# Guided mode: the computer asks for ONE vital at a time, so the answer is just
# a number and no keyword ("temperature", "over") has to be recognized.
# ---------------------------------------------------------------------------

GUIDED_FIELDS = [
    ("Age", "age in years"),
    ("BP", "blood pressure, top number then bottom number"),
    ("BS", "blood sugar"),
    ("BodyTemp", "body temperature in Fahrenheit"),
    ("HeartRate", "heart rate"),
]


def parse_field_value(transcript: str, field: str):
    """Parse the answer to a single question.

    field is one of Age, BS, BodyTemp, HeartRate, Hemoglobin, or "BP".
    Returns (value, None) on success, where value is a float (for "BP" a
    (systolic, diastolic) tuple), or (None, reason) if the answer can't be
    trusted. Nothing out of range is ever accepted here: the health worker is
    asked again instead.
    """
    text = _normalize_word_numbers(transcript or "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]

    if field == "BP":
        if len(nums) != 2:
            return None, "I need two numbers: the top number, then the bottom number."
        sys_, dia = nums
        for name, v in (("SystolicBP", sys_), ("DiastolicBP", dia)):
            lo, hi = VALID_RANGES[name]
            if not lo <= v <= hi:
                return None, f"{v:g} looks wrong for {name} (expected {lo}-{hi})."
        if sys_ <= dia:
            return None, "The top number must be bigger than the bottom number."
        return (sys_, dia), None

    if not nums:
        return None, "I did not catch a number."
    if len(nums) > 1:
        return None, "I heard more than one number: " + ", ".join(f"{x:g}" for x in nums) + "."
    v = nums[0]
    lo, hi = VALID_RANGES[field]
    if not lo <= v <= hi:
        hint = ""
        if field == "BodyTemp" and 34 <= v <= 42:
            hint = " That looks like Celsius; please say the temperature in Fahrenheit."
        return None, f"{v:g} looks wrong for {field} (expected {lo:g}-{hi:g})." + hint
    return v, None


if __name__ == "__main__":
    demo = "Patient age thirty two, BP is 140 over 90, blood sugar 7.5, temperature ninety eight point six, heart rate 88"
    parsed = parse_vitals_from_text(demo)
    print(parsed.as_dict())
    print(format_confirmation_prompt(parsed))
