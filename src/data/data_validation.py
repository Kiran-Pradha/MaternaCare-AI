"""
data_validation.py — Review 1 remark: "make sure the range for different
parameters are sensible"
MaternaCare AI

Defines clinically-grounded plausible ranges for every vital used in this
project, and provides functions to validate any dataframe (the real UCI
dataset, GA-curated data, or synthetic trajectory data) against them.

Where these ranges come from
-----------------------------
Two distinct kinds of range are defined, and it matters not to conflate
them:

1. PLAUSIBLE_RANGE — the widest range a value could physiologically be and
   still belong to a living pregnant patient whose reading wasn't a sensor
   error. Anything outside this is treated as a probable data-quality
   defect (mis-transcription, sensor fault), not a real extreme patient.
   Sources: standard adult/obstetric physiology references (WHO, ACOG,
   NICE hypertension-in-pregnancy guidance) and the observed range of any
   value that is physiologically survivable.

2. CLINICAL_FLAG_THRESHOLD — the point at which a value itself becomes a
   risk signal (independent of whether it's plausible). E.g. a systolic BP
   of 145 is entirely plausible (a real, unwell patient) AND already a
   hypertension flag. These thresholds are what rules_engine-style logic
   should key off; they are looser/different from PLAUSIBLE_RANGE and
   serve a different purpose.

This distinction matters directly for the review panel's remark: mixing
"is this value real" checks with "is this value risky" checks was the kind
of range-sensibility problem worth catching before Phase 9 — a value can
fail one check, the other, both, or neither.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


@dataclass
class VitalRange:
    plausible_lo: float
    plausible_hi: float
    flag_lo: float          # below this = a low-side clinical concern (e.g. hypotension, bradycardia)
    flag_hi: float          # above this = a high-side clinical concern (e.g. hypertension, tachycardia)
    unit: str
    source_note: str


# ---------------------------------------------------------------------------
# Vital-by-vital definitions
# ---------------------------------------------------------------------------
# BP thresholds: ACOG/NICE define hypertension in pregnancy as >=140/90 mmHg,
# severe hypertension as >=160/110. Plausible physiological range for a
# living patient is roughly 60-260 systolic, 30-150 diastolic (Kaldi/IoT
# cuffs on this dataset don't go outside 70-160/49-100 in practice, but the
# *validity* envelope should be wider than the *observed* envelope, since
# a validity check should catch genuinely dangerous readings, not just
# "outside what we happened to see").
#
# BS (blood glucose, mmol/L): normal fasting ~4-5.4, gestational diabetes
# diagnostic threshold (WHO) fasting >=5.1, severe hyperglycemia >11.1,
# hypoglycemia <3.0 is a medical emergency. Plausible survivable range
# roughly 1.5-33 mmol/L (below/above that is incompatible with
# consciousness for more than brief periods).
#
# BodyTemp (Fahrenheit): normal ~97-99, fever (infection/sepsis risk in
# pregnancy) >=100.4, hypothermia <95. Plausible survivable range ~90-108.
#
# HeartRate (bpm): normal resting adult ~60-100, pregnancy can run slightly
# elevated. Bradycardia <60, tachycardia >100 (>120 more clinically
# significant). Plausible survivable range ~30-220 (an HR of 7, as found
# in 2 duplicate rows of the real dataset, is not survivable and is a
# sensor/entry error, not an extreme patient).
#
# Age (years, maternal): dataset's own observed range is 10-70; the low end
# reflects real-world early adolescent pregnancy which does occur and is
# itself a recognized risk factor, so it is not excluded, but is flagged.
# The realistic reproductive-range ceiling is roughly 55.

VITAL_RANGES: Dict[str, VitalRange] = {
    "Age": VitalRange(
        plausible_lo=10, plausible_hi=60,
        flag_lo=18, flag_hi=35,  # <18 and >=35 are both recognized obstetric risk-age bands
        unit="years",
        source_note="Adolescent (<18) and advanced maternal age (>=35) are "
                     "both established risk factors (ACOG); plausible floor "
                     "reflects earliest recorded pregnancies, ceiling reflects "
                     "practical fertility limits.",
    ),
    "SystolicBP": VitalRange(
        plausible_lo=60, plausible_hi=220,
        flag_lo=90, flag_hi=140,  # <90 hypotension concern, >=140 hypertension (ACOG/NICE)
        unit="mmHg",
        source_note="Hypertension in pregnancy: >=140/90 (ACOG/NICE); severe >=160/110.",
    ),
    "DiastolicBP": VitalRange(
        plausible_lo=30, plausible_hi=150,
        flag_lo=60, flag_hi=90,
        unit="mmHg",
        source_note="Paired with SystolicBP threshold above.",
    ),
    "BS": VitalRange(
        plausible_lo=1.5, plausible_hi=33.0,
        flag_lo=3.0, flag_hi=7.8,  # WHO gestational diabetes screening thresholds
        unit="mmol/L",
        source_note="WHO gestational diabetes fasting threshold ~5.1; random/2h "
                     "values used here follow the dataset's own convention "
                     "(non-fasting), flag_hi set at 7.8 as a conservative "
                     "elevated-glucose marker; <3.0 is hypoglycemia, a medical emergency.",
    ),
    "BodyTemp": VitalRange(
        plausible_lo=90.0, plausible_hi=108.0,
        flag_lo=96.0, flag_hi=100.4,  # 100.4F = 38C, standard fever threshold
        unit="Fahrenheit",
        source_note="100.4F (38C) is the standard clinical fever threshold, "
                     "relevant for infection/sepsis risk in pregnancy.",
    ),
    "HeartRate": VitalRange(
        plausible_lo=30, plausible_hi=220,
        flag_lo=60, flag_hi=100,
        unit="bpm",
        source_note="Normal adult resting range 60-100bpm; pregnancy can run "
                     "the upper end of normal or slightly above. A reading of "
                     "7bpm (present in this dataset) is not survivable and is "
                     "a sensor/entry artifact, not a real extreme patient.",
    ),
}


def validate_dataframe(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Check every row of `df` against PLAUSIBLE ranges (not flag thresholds —
    this function is about data quality, not clinical risk).

    Returns a copy of df with one added boolean column per checked vital,
    `{col}_implausible`, plus a summary `any_implausible` column.
    """
    columns = columns or list(VITAL_RANGES.keys())
    out = df.copy()
    any_bad = pd.Series(False, index=df.index)

    for col in columns:
        if col not in df.columns or col not in VITAL_RANGES:
            continue
        r = VITAL_RANGES[col]
        bad = (df[col] < r.plausible_lo) | (df[col] > r.plausible_hi)
        out[f"{col}_implausible"] = bad
        any_bad = any_bad | bad

    out["any_implausible"] = any_bad
    return out


def summarize_validation(df: pd.DataFrame, columns: List[str] = None) -> str:
    """Human-readable summary of how many rows/values failed plausibility checks."""
    validated = validate_dataframe(df, columns)
    n_total = len(df)
    n_bad_rows = int(validated["any_implausible"].sum())

    lines = [f"Validated {n_total} rows against clinically-grounded plausible ranges."]
    lines.append(f"Rows with at least one implausible value: {n_bad_rows} "
                 f"({100 * n_bad_rows / n_total:.2f}%)")

    for col in (columns or list(VITAL_RANGES.keys())):
        flag_col = f"{col}_implausible"
        if flag_col in validated.columns:
            n = int(validated[flag_col].sum())
            if n > 0:
                r = VITAL_RANGES[col]
                bad_vals = df.loc[validated[flag_col], col].tolist()
                lines.append(
                    f"  - {col}: {n} implausible value(s) outside "
                    f"[{r.plausible_lo}, {r.plausible_hi}] {r.unit} -> {bad_vals}"
                )
    return "\n".join(lines)


def clip_to_plausible(value: float, column: str) -> float:
    """Clip a single value into the plausible range for `column`. Used by the
    synthetic trajectory generator to guarantee every generated value is
    physiologically valid by construction, not just by post-hoc filtering."""
    r = VITAL_RANGES[column]
    return max(r.plausible_lo, min(r.plausible_hi, value))


def flag_clinical_concerns(row: pd.Series) -> List[str]:
    """
    Return a list of human-readable clinical flags for a single patient row,
    using CLINICAL_FLAG_THRESHOLD-style logic (separate from plausibility).
    This is the shared logic the rules engine (Phase 7) and the hybrid model
    (this remark's deliverable) both key off, kept here as one source of
    truth so the two don't drift out of sync with each other.
    """
    flags = []
    r = VITAL_RANGES

    if row["SystolicBP"] >= r["SystolicBP"].flag_hi or row["DiastolicBP"] >= r["DiastolicBP"].flag_hi:
        severity = "severe" if (row["SystolicBP"] >= 160 or row["DiastolicBP"] >= 110) else "elevated"
        flags.append(f"{severity}_hypertension")
    if row["SystolicBP"] < r["SystolicBP"].flag_lo:
        flags.append("hypotension")

    if row["BS"] >= r["BS"].flag_hi:
        flags.append("hyperglycemia")
    if row["BS"] < r["BS"].flag_lo:
        flags.append("hypoglycemia")

    if row["BodyTemp"] >= r["BodyTemp"].flag_hi:
        flags.append("fever")
    if row["BodyTemp"] < r["BodyTemp"].flag_lo:
        flags.append("hypothermia")

    if row["HeartRate"] > r["HeartRate"].flag_hi:
        flags.append("tachycardia")
    if row["HeartRate"] < r["HeartRate"].flag_lo:
        flags.append("bradycardia")

    if row["Age"] < r["Age"].flag_lo:
        flags.append("adolescent_pregnancy")
    if row["Age"] >= r["Age"].flag_hi:
        flags.append("advanced_maternal_age")

    return flags


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "data" / "maternal_health_risk.csv",
        project_root / "data" / "raw" / "maternal_health_risk.csv",
        project_root / "data" / "raw" / "anemia_dataset.csv",
    ]

    csv_path = next((p for p in candidates if p.exists()), candidates[0])
    if not csv_path.exists():
        raise FileNotFoundError(
            "Could not find a maternal health dataset. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    df = pd.read_csv(csv_path)
    print(f"Using dataset: {csv_path}")
    print(summarize_validation(df))
    print()
    print("Example clinical flags for first 5 rows:")
    for i in range(min(5, len(df))):
        print(f"  row {i}: {flag_clinical_concerns(df.iloc[i])}")