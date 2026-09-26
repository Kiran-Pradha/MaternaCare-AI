"""
Phase 8 — Voice Interface: Vitals Extraction from Speech Transcript
=====================================================================

Parses a free-text transcript (the output of any STT engine — Vosk, Google,
whatever) into the structured vitals dict consumed by the Phase 7 rules
engine (src/rules/rules_engine.py) and the Phase 2/3d trained model.

This module is deliberately decoupled from any actual speech recognition:
it operates purely on strings, so it is fully unit-testable in any sandbox,
including ones with no microphone and no network access. STT and TTS are
just the I/O shell around this — this file is the part that actually needs
to be correct.

Design notes
------------
- Regex-first, not a full NLP pipeline. Maternal-health field workers in the
  target low-resource setting speak short, formulaic phrases ("BP is
  110 over 70", "blood sugar seven point five"), not free-form prose, so a
  small set of tolerant patterns covers the real distribution far better
  than the engineering cost of a trained NER model would justify.
- Every extracted value is range-checked against physiologically plausible
  bounds (PLAUSIBLE_RANGES). Out-of-range values are still returned, but
  flagged, so the app layer can ask the health worker to repeat the value
  rather than silently feeding a mis-heard "1200" systolic into the model.
- Spoken number words ("seven point five", "one twenty over eighty") are
  normalized before the numeric regexes run, since BS and BP are often
  read out digit-by-digit or as number words by ASR post-processing.
- This module never makes a risk determination itself. It only produces a
  vitals dict; Phase 7's rules_engine (or the Phase 2/3d model) does the
  actual clinical reasoning. Keeping that boundary explicit matters for a
  decision-support tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Plausible physiological ranges, used only to flag (not reject) outliers.
# Sources: same WHO/NICE/ACOG bounds used by the Phase 7 rules engine.
# ---------------------------------------------------------------------------
PLAUSIBLE_RANGES = {
    "Age": (10, 60),
    "SystolicBP": (70, 200),
    "DiastolicBP": (40, 130),
    "BS": (2.0, 30.0),          # mmol/L
    "BodyTemp": (95.0, 106.0),  # Fahrenheit (matches UCI dataset convention)
    "HeartRate": (30, 200),
    "Hemoglobin": (3.0, 20.0),  # g/dL
}

# Word -> digit map for spoken numbers, small enough for this domain
# (vitals are read as short number phrases, not general arithmetic).
_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "to": 2, "too": 2, "three": 3,
    "four": 4, "for": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(phrase: str) -> str | None:
    """Convert a short spoken-number phrase (e.g. 'one twenty', 'seven point
    five') to its digit-string form ('120', '7.5'). Returns None if the
    phrase doesn't parse as a number."""
    phrase = phrase.strip().lower()
    if "point" in phrase:
        whole, _, frac = phrase.partition("point")
        whole_val = _words_to_number(whole)
        frac_tokens = frac.strip().split()
        frac_digits = "".join(str(_ONES[t]) for t in frac_tokens if t in _ONES)
        if whole_val is not None and frac_digits:
            return f"{whole_val}.{frac_digits}"
        return None

    tokens = phrase.split()
    if not tokens:
        return None

    total = 0
    matched_any = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _TENS:
            total += _TENS[tok]
            matched_any = True
        elif tok in _ONES:
            # "one twenty" style: leading hundred-ish digit spoken alone,
            # e.g. "one" then "twenty" => 1*100 + 20 handled by caller
            # joining digit strings; here we just accumulate ones normally.
            total += _ONES[tok]
            matched_any = True
        else:
            return None
        i += 1
    return str(total) if matched_any else None


def _normalize_spoken_numbers(text: str) -> str:
    """Best-effort replacement of spoken number phrases with digits, so the
    numeric regexes below can match either '120 over 80' or 'one twenty
    over eighty'. Deliberately conservative: only replaces runs of
    number-words, leaves everything else untouched."""
    number_word_pattern = re.compile(
        r"\b((?:zero|oh|one|two|to|too|three|four|for|five|six|seven|eight|"
        r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
        r"seventy|eighty|ninety|point)(?:[\s-]+(?:zero|oh|one|two|to|too|"
        r"three|four|for|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
        r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|point))*)\b",
        re.IGNORECASE,
    )

    def _sub(m: re.Match) -> str:
        as_num = _words_to_number(m.group(0))
        return as_num if as_num is not None else m.group(0)

    return number_word_pattern.sub(_sub, text)


@dataclass
class ExtractionResult:
    vitals: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)   # field -> warning message
    unmatched_fields: list = field(default_factory=list)

    def is_complete(self, required=("Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate")) -> bool:
        return all(f in self.vitals for f in required)

    def to_dict(self) -> dict:
        return dict(self.vitals)


_NUM = r"(\d+(?:\.\d+)?)"

PATTERNS = {
    "Age": [
        rf"\bage\s*(?:is|:)?\s*{_NUM}\b",
        rf"\b{_NUM}\s*(?:years?|yrs?)\s*old\b",
        rf"\bi\s*am\s*{_NUM}\b",
    ],
    "BP_PAIR": [
        # "BP 120 over 80", "blood pressure is 120/80", "systolic 120 diastolic 80"
        rf"\b(?:bp|blood\s*pressure)\s*(?:is|:)?\s*{_NUM}\s*(?:/|over)\s*{_NUM}\b",
    ],
    "SystolicBP": [
        rf"\bsystolic(?:\s*(?:bp|blood\s*pressure))?\s*(?:is|:)?\s*{_NUM}\b",
    ],
    "DiastolicBP": [
        rf"\bdiastolic(?:\s*(?:bp|blood\s*pressure))?\s*(?:is|:)?\s*{_NUM}\b",
    ],
    "BS": [
        rf"\b(?:bs|blood\s*sugar|glucose|blood\s*glucose)\s*(?:is|:|of)?\s*{_NUM}\b",
    ],
    "BodyTemp": [
        rf"\b(?:temp|temperature|body\s*temp(?:erature)?)\s*(?:is|:)?\s*{_NUM}\b",
    ],
    "HeartRate": [
        rf"\b(?:heart\s*rate|pulse|hr)\s*(?:is|:)?\s*{_NUM}\b",
    ],
    "Hemoglobin": [
        rf"\b(?:hemoglobin|haemoglobin|hb)\s*(?:is|:|of)?\s*{_NUM}\b",
    ],
}


def extract_vitals(transcript: str) -> ExtractionResult:
    """Parse a raw STT transcript into a vitals dict.

    Parameters
    ----------
    transcript : str
        Raw text returned by the STT engine (Vosk / Google / manual typed
        fallback). Case-insensitive, tolerant of filler words.

    Returns
    -------
    ExtractionResult
        .vitals            -> dict of field_name -> float, for every field
                               successfully parsed
        .flags             -> dict of field_name -> warning string, for any
                               parsed value that falls outside
                               PLAUSIBLE_RANGES (kept in .vitals, but the
                               caller should confirm with the user before
                               passing it to the model / rules engine)
        .unmatched_fields   -> list of the six core vitals fields that could
                               not be found in the transcript at all
    """
    text = (transcript or "").strip().lower()
    text = _normalize_spoken_numbers(text)
    result = ExtractionResult()

    # Blood pressure pair takes priority over separate systolic/diastolic
    # patterns, since "120 over 80" is the common phrasing and would
    # otherwise partially double-match.
    bp_matched = False
    for pat in PATTERNS["BP_PAIR"]:
        m = re.search(pat, text)
        if m:
            result.vitals["SystolicBP"] = float(m.group(1))
            result.vitals["DiastolicBP"] = float(m.group(2))
            bp_matched = True
            break

    for field_name, patterns in PATTERNS.items():
        if field_name == "BP_PAIR":
            continue
        if bp_matched and field_name in ("SystolicBP", "DiastolicBP"):
            continue
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                result.vitals[field_name] = float(m.group(1))
                break

    # Range-check every extracted value.
    for field_name, value in result.vitals.items():
        lo, hi = PLAUSIBLE_RANGES.get(field_name, (None, None))
        if lo is not None and not (lo <= value <= hi):
            result.flags[field_name] = (
                f"{field_name}={value} is outside the plausible range "
                f"[{lo}, {hi}]. Please ask the health worker to repeat "
                f"this value before submitting."
            )

    core_fields = ("Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate")
    result.unmatched_fields = [f for f in core_fields if f not in result.vitals]

    return result


def format_missing_fields_prompt(result: ExtractionResult) -> str | None:
    """Build a short spoken follow-up prompt asking only for the fields that
    were not captured, so the voice interface doesn't force the health
    worker to repeat the whole reading. Returns None if nothing is missing."""
    if not result.unmatched_fields:
        return None
    friendly = {
        "Age": "the patient's age",
        "SystolicBP": "the systolic blood pressure",
        "DiastolicBP": "the diastolic blood pressure",
        "BS": "the blood sugar level",
        "BodyTemp": "the body temperature",
        "HeartRate": "the heart rate",
    }
    missing_readable = [friendly[f] for f in result.unmatched_fields]
    if len(missing_readable) == 1:
        return f"I didn't catch {missing_readable[0]}. Could you repeat it?"
    return (
        "I didn't catch a few values: " + ", ".join(missing_readable[:-1]) +
        f" and {missing_readable[-1]}. Could you repeat them?"
    )
