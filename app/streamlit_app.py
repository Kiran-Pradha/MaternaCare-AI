"""
MaternaCare AI -- Streamlit demo app (Phase 9: integration)
=============================================================================
One page that walks a health worker through:

    1. enter vitals   (type them, speak them, or load an example)
    2. confirm them   (nothing is assessed until a person ticks the box)
    3. get an assessment:
         ML risk level  ->  trajectory pattern  ->  Phase 7 rules engine
         -> urgency, recommended action, flags, anemia guidance, SHAP reasons

Run from the repository root:

    streamlit run app/streamlit_app.py

Pieces and where they come from
-------------------------------
* Rules engine ........ src/rules/rules_engine.py        (Phase 7, unchanged)
* Voice entry ......... src/voice/*                       (Phase 8; optional)
* Risk model .......... a Random Forest trained at start-up on the dataset in data/
                        (see load_model(): swap in your GA-curated model there)
* Explanations ........ SHAP TreeExplainer on that model (Phase 6 style)
* Trajectory .......... detect_trajectory() below: a transparent stand-in for the
                        Phase 4 pattern classifier (see the note on that function)
"""

import json
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
for _p in (ROOT, os.path.join(ROOT, "src", "voice")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.rules.rules_engine import generate_recommendation  # noqa: E402

try:                                    # Phase 8 text parsing (pure Python, no audio)
    from vitals_parser import parse_field_value, parse_vitals_from_text  # noqa: E402
    PARSER_OK = True
except Exception:                       # app still works with typed input only
    PARSER_OK = False

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
CLASS_ORDER = ["Low Risk", "Mid Risk", "High Risk"]
PATTERNS = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]

# widget key -> (label, min, max, step, is_int)   (same limits as the voice parser)
FIELDS = {
    "age":  ("Age (years)",                  10, 60, 1, True),
    "sbp":  ("Systolic BP (mmHg)",           70, 220, 1, True),
    "dbp":  ("Diastolic BP (mmHg)",          40, 140, 1, True),
    "bs":   ("Blood sugar (mmol/L)",         2.0, 25.0, 0.1, False),
    "temp": ("Body temperature (\u00b0F)",   90.0, 108.0, 0.1, False),
    "hr":   ("Heart rate (bpm)",             30, 200, 1, True),
    "hb":   ("Hemoglobin (g/dL)",            3.0, 20.0, 0.1, False),
}
PARSER_TO_KEY = {"Age": "age", "SystolicBP": "sbp", "DiastolicBP": "dbp", "BS": "bs",
                 "BodyTemp": "temp", "HeartRate": "hr", "Hemoglobin": "hb"}

# Thresholds for the trajectory detector (placeholders -- tune on real follow-up data)
SUDDEN_STEP = 0.30      # risk score jumps by this much between the last two visits
GRADUAL_TOTAL = 0.15    # risk score drifts by this much from first to last visit

# urgency (0-4 from the rules engine) -> (streamlit box, headline)
URGENCY_STYLE = {
    0: ("success", "Routine care"),
    1: ("success", "Routine follow-up"),
    2: ("warning", "Follow-up needed"),
    3: ("error", "Urgent: act within days"),
    4: ("error", "EMERGENCY: act now"),
}

EXAMPLES = {
    "Lower-risk patient": dict(age=25, sbp=112, dbp=74, bs=7.2, temp=98.0, hr=70, use_hb=True, hb=12.0, n_prev=0),
    "Mid-risk patient": dict(age=27, sbp=128, dbp=82, bs=9.0, temp=98.4, hr=76, use_hb=True, hb=10.5, n_prev=0),
    "High-risk, rising readings": dict(
        age=30, sbp=162, dbp=108, bs=14.5, temp=99.1, hr=92, use_hb=True, hb=8.5, n_prev=2,
        pv0=(118, 76, 7.0, 98.0, 74), pv1=(135, 86, 9.5, 98.4, 82)),
}


# --------------------------------------------------------------------------
# Model, explanations, trajectory
# --------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]
_COLUMN_ALIASES = {  # normalised header (lower-case, letters+digits only) -> canonical name
    "age": "Age", "systolicbp": "SystolicBP", "diastolicbp": "DiastolicBP", "bs": "BS",
    "bodytemp": "BodyTemp", "heartrate": "HeartRate", "risklevel": "RiskLevel",
    "bodytemperature": "BodyTemp", "bloodsugar": "BS", "risk": "RiskLevel",
}
_SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages", ".idea"}


def _canonical_columns(columns):
    """Map whatever headers the CSV has onto the standard names, ignoring case, spaces, _ and -."""
    mapping = {}
    for c in columns:
        key = "".join(ch for ch in str(c).lower() if ch.isalnum())
        if key in _COLUMN_ALIASES:
            mapping[c] = _COLUMN_ALIASES[key]
    return mapping


def read_dataset(path, nrows=None):
    """Read a CSV and return it with standard column names, or None if it isn't the dataset."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, nrows=nrows)
            break
        except Exception:
            df = None
    if df is None:
        return None
    df = df.rename(columns=_canonical_columns(df.columns))
    return df if set(REQUIRED_COLUMNS).issubset(df.columns) else None


def find_dataset():
    """Look for the Maternal Health Risk CSV: first under data/ (any depth), then anywhere in
    the project. Returns (path_or_None, checked) where `checked` lists every CSV looked at and
    the columns it had, so a failure can be explained instead of silent."""
    checked = []
    roots = [os.path.join(ROOT, "data"), ROOT]
    seen = set()
    for top in roots:
        if not os.path.isdir(top):
            continue
        for folder, dirs, files in os.walk(top):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for name in sorted(files):
                if not name.lower().endswith(".csv"):
                    continue
                path = os.path.join(folder, name)
                if path in seen:
                    continue
                seen.add(path)
                df = read_dataset(path, nrows=3)
                if df is not None:
                    return path, checked
                try:
                    cols = list(pd.read_csv(path, nrows=1, encoding="utf-8-sig").columns)
                except Exception as e:
                    cols = [f"(unreadable: {type(e).__name__})"]
                checked.append((os.path.relpath(path, ROOT), cols))
    return None, checked


@st.cache_resource(show_spinner="Training the risk model on the maternal-health dataset ...")
def load_model(csv_path):
    """Random Forest on the UCI Maternal Health Risk data.

    TO USE YOUR OWN MODEL (e.g. the GA-curated, GA-tuned RF from Phase 3d/3e):
    return it here as (classifier, n_rows). It needs predict_proba, a classes_
    attribute holding "Low Risk" / "Mid Risk" / "High Risk", and to be trained on
    a DataFrame with the columns in FEATURES.
    """
    df = read_dataset(csv_path)
    df["RiskLevel"] = df["RiskLevel"].str.strip().str.title()
    # drop the implausible rows found in the data audit (Age > 60, heart rate < 30)
    df = df[(df["Age"] <= 60) & (df["HeartRate"] >= 30)]
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(df[FEATURES], df["RiskLevel"])
    return clf, len(df)


@st.cache_resource(show_spinner=False)
def load_explainer(_clf):
    try:
        import shap
        return shap.TreeExplainer(_clf)
    except Exception:
        return None


def predict_probs(clf, row):
    p = clf.predict_proba(pd.DataFrame([row])[FEATURES])[0]
    classes = list(clf.classes_)
    return {c: float(p[classes.index(c)]) for c in CLASS_ORDER}


def shap_top_factors(explainer, clf, row, risk_level, k=3):
    """[(feature, shap_value, actual_value), ...] for the predicted class."""
    if explainer is None:
        return None
    try:
        sv = explainer.shap_values(pd.DataFrame([row])[FEATURES])
        idx = list(clf.classes_).index(risk_level)
        if isinstance(sv, list):
            vals = np.asarray(sv[idx])[0]
        else:
            sv = np.asarray(sv)
            vals = sv[0, :, idx] if sv.ndim == 3 else sv[0]
        order = np.argsort(-np.abs(vals))[:k]
        return [(FEATURES[i], float(vals[i]), float(row[FEATURES[i]])) for i in order]
    except Exception:
        return None


def detect_trajectory(clf, visits):
    """Classify the pattern across visits (oldest first, current visit last).

    NOTE: this is a transparent stand-in for the Phase 4 pattern-recovery
    classifier, which was not available to this app. Each visit gets a risk score
    from the model (P(High) + 0.5 * P(Mid)); the pattern comes from how that score
    moves. The two thresholds at the top of this file are placeholders and should
    be tuned on real follow-up data before any real use.
    """
    scores = []
    for v in visits:
        p = predict_probs(clf, v)
        scores.append(p["High Risk"] + 0.5 * p["Mid Risk"])
    if len(scores) < 2:
        return "Stable", scores
    last_step = scores[-1] - scores[-2]
    total = scores[-1] - scores[0]
    if last_step >= SUDDEN_STEP:
        return "Sudden Deterioration", scores
    if total >= GRADUAL_TOTAL:
        return "Gradually Worsening", scores
    if total <= -GRADUAL_TOTAL:
        return "Improving", scores
    return "Stable", scores


def anemia_tier_from_hb(hb):
    """WHO thresholds for pregnancy: <7 severe, 7-9.9 moderate, >=10 low."""
    if hb is None:
        return None
    if hb < 7:
        return "Severe Risk"
    if hb < 10:
        return "Moderate Risk"
    return "Low Risk"


# --------------------------------------------------------------------------
# Voice helpers (all optional)
# --------------------------------------------------------------------------
def mic_status():
    """(ready, reason). Needs vosk + sounddevice + the downloaded Vosk model."""
    if not PARSER_OK:
        return False, "voice parser files not found in src/voice"
    try:
        import vosk  # noqa: F401
        import sounddevice  # noqa: F401
        from stt_vosk_integration import MODEL_DIR
    except Exception as e:
        return False, f"missing package ({type(e).__name__}); run: pip install vosk sounddevice"
    if not os.path.isdir(MODEL_DIR):
        return False, "Vosk model not downloaded yet (run ensure_model() once)"
    return True, ""


def _set_field(key, value):
    _, lo, hi, _, is_int = FIELDS[key]
    st.session_state[key] = int(round(value)) if is_int else float(value)


def cb_speak(field):
    """Record 6 s from this computer's microphone and fill the matching box(es)."""
    from stt_vosk_integration import transcribe_microphone, NUMBER_VOCABULARY
    try:
        transcript = transcribe_microphone(duration_seconds=6, vocabulary=NUMBER_VOCABULARY)
    except Exception as e:
        st.session_state["voice_msg"] = ("error", f"Microphone problem: {e}")
        return
    value, err = parse_field_value(transcript, field)
    if err:
        st.session_state["voice_msg"] = ("error", f"Heard \u201c{transcript}\u201d. {err} Please try again.")
        return
    if field == "BP":
        _set_field("sbp", value[0])
        _set_field("dbp", value[1])
        shown = f"{value[0]:g}/{value[1]:g}"
    else:
        key = {"Age": "age", "BS": "bs", "BodyTemp": "temp", "HeartRate": "hr", "Hemoglobin": "hb"}[field]
        _set_field(key, value)
        if key == "hb":
            st.session_state["use_hb"] = True
        shown = f"{value:g}"
    st.session_state["voice_msg"] = (
        "success", f"Heard \u201c{transcript}\u201d \u2192 {shown}. Please check the box below.")
    st.session_state["confirmed"] = False


def cb_fill_sentence():
    parsed = parse_vitals_from_text(st.session_state.get("sentence", ""))
    filled, skipped = [], []
    for name in parsed.fields_found():
        key = PARSER_TO_KEY[name]
        val = getattr(parsed, name)
        _, lo, hi, _, _ = FIELDS[key]
        if lo <= val <= hi:
            _set_field(key, val)
            if key == "hb":
                st.session_state["use_hb"] = True
            filled.append(name)
        else:
            skipped.append(f"{name}={val:g}")
    if filled or skipped:
        msg = "Filled: " + (", ".join(filled) if filled else "nothing") + "."
        if skipped:
            msg += " Ignored out-of-range: " + ", ".join(skipped) + "."
        st.session_state["voice_msg"] = ("success" if filled else "error", msg + " Please check the boxes below.")
        st.session_state["confirmed"] = False
    else:
        st.session_state["voice_msg"] = ("error", "No vitals found in that sentence.")


def cb_example(name):
    ex = EXAMPLES[name]
    for k in ("age", "sbp", "dbp", "bs", "temp", "hr", "hb"):
        if k in ex:
            st.session_state[k] = ex[k]
    st.session_state["use_hb"] = ex["use_hb"]
    st.session_state["n_prev"] = ex["n_prev"]
    for i in range(3):
        if f"pv{i}" in ex:
            for suffix, v in zip(("sbp", "dbp", "bs", "temp", "hr"), ex[f"pv{i}"]):
                st.session_state[f"pv{i}_{suffix}"] = v
    st.session_state["confirmed"] = False
    st.session_state["voice_msg"] = ("info", f"Loaded example: {name}. Check the values, then tick the box.")


def speak_text(text):
    """Offline text-to-speech -> WAV bytes the browser can play (or None)."""
    try:
        from tts_offline import save_offline_wav
        path = os.path.join(tempfile.gettempdir(), "maternacare_app_readout.wav")
        save_offline_wav(text, path)
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="MaternaCare AI", page_icon="\U0001F930", layout="wide")

# default values (set once; widgets keep their own state afterwards)
_defaults = dict(age=28, sbp=120, dbp=80, bs=7.0, temp=98.0, hr=76, use_hb=False, hb=11.5,
                 n_prev=0, confirmed=False, sentence="age 32, blood pressure 140 over 90, "
                 "blood sugar 7.5, temperature 98.6, heart rate 88")
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)
for i in range(3):
    for suffix, v in zip(("sbp", "dbp", "bs", "temp", "hr"), (118, 76, 7.0, 98.0, 74)):
        st.session_state.setdefault(f"pv{i}_{suffix}", v)

csv_path, _checked = find_dataset()

with st.sidebar:
    st.header("MaternaCare AI")
    st.caption("Hybrid, explainable, longitudinal decision support for maternal health risk in "
               "low-resource settings.")
    st.markdown(
        "**How an assessment is made**\n\n"
        "1. A machine-learning model estimates the risk level from the vitals.\n"
        "2. The pattern across visits (stable, worsening, ...) is determined.\n"
        "3. A rule-based engine (WHO / NICE / ACOG thresholds) turns risk + pattern + "
        "vital-sign flags into a concrete action.\n"
        "4. SHAP shows which vitals drove the ML estimate."
    )
    st.warning("Prototype for a capstone project. It supports, and never replaces, "
               "clinical judgement. Not a medical device.")

if csv_path is None:
    st.title("\U0001F930 MaternaCare AI")
    st.error("I could not find the Maternal Health Risk dataset.")
    st.markdown(
        f"I looked in `{os.path.join(ROOT, 'data')}` (and the rest of the project folder) for a CSV "
        "with these columns: **" + ", ".join(REQUIRED_COLUMNS) + "**."
    )
    if _checked:
        st.markdown("CSV files I found, but they did not match:")
        for rel, cols in _checked:
            st.code(f"{rel}\ncolumns: {cols}")
    else:
        st.markdown("I found no CSV files at all.")
    st.markdown("**Fix:** copy the dataset CSV into the `data` folder, or upload it here:")
    up = st.file_uploader("Maternal Health Risk CSV", type="csv")
    if up is not None:
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        target = os.path.join(ROOT, "data", "maternal_health_risk_data.csv")
        with open(target, "wb") as f:
            f.write(up.getbuffer())
        if read_dataset(target) is None:
            os.remove(target)
            st.error("That file does not have the expected columns, so it was not saved.")
        else:
            st.rerun()
    st.stop()

clf, n_train = load_model(csv_path)
explainer = load_explainer(clf)
with st.sidebar:
    st.caption(f"Risk model: Random Forest trained on {n_train} records "
               f"(about 84% cross-validated accuracy in our experiments).")

st.title("\U0001F930 MaternaCare AI")
st.caption("Enter the patient's vitals, confirm them, and get a risk assessment with a recommended action.")

# ---- 1. vitals ------------------------------------------------------------
st.subheader("1. Patient vitals")

ex_cols = st.columns(len(EXAMPLES) + 1)
ex_cols[0].markdown("**Try an example:**")
for col, name in zip(ex_cols[1:], EXAMPLES):
    col.button(name, key=f"ex_{name}", on_click=cb_example, args=(name,))

with st.expander("\U0001F3A4 Voice entry (optional)", expanded=False):
    ready, reason = mic_status()
    if ready:
        st.caption("Click a button, wait a second, then say just the number (for blood pressure: "
                   "the top number, then the bottom number). Recording lasts 6 seconds and uses "
                   "the microphone of the computer running this app.")
        bcols = st.columns(5)
        for col, (label, field) in zip(bcols, [("Age", "Age"), ("Blood pressure", "BP"),
                                               ("Blood sugar", "BS"), ("Temperature", "BodyTemp"),
                                               ("Heart rate", "HeartRate")]):
            col.button(f"Speak {label.lower()}", key=f"spk_{field}", on_click=cb_speak, args=(field,))
    else:
        st.info(f"Microphone entry is not set up on this computer: {reason}. "
                "You can still type values or use the sentence box below.")
    if PARSER_OK:
        st.text_input("Or type / paste a whole sentence", key="sentence")
        st.button("Fill the form from this sentence", key="fill_sentence", on_click=cb_fill_sentence)
    msg = st.session_state.get("voice_msg")
    if msg:
        getattr(st, msg[0])(msg[1])

c1, c2, c3 = st.columns(3)
with c1:
    for k in ("age", "sbp", "dbp"):
        lab, lo, hi, step, is_int = FIELDS[k]
        st.number_input(lab, min_value=lo, max_value=hi, step=step, key=k)
with c2:
    for k in ("bs", "temp", "hr"):
        lab, lo, hi, step, is_int = FIELDS[k]
        st.number_input(lab, min_value=lo, max_value=hi, step=step, key=k,
                        format=None if is_int else "%.1f")
with c3:
    st.checkbox("Hemoglobin result available (anemia check)", key="use_hb")
    lab, lo, hi, step, _ = FIELDS["hb"]
    st.number_input(lab, min_value=lo, max_value=hi, step=step, key="hb", format="%.1f",
                    disabled=not st.session_state["use_hb"])

# ---- 2. previous visits / trajectory ---------------------------------------
st.subheader("2. Previous visits (optional)")
st.caption("If you add previous visits, replace the boxes' starting numbers with the patient's REAL "
           "earlier readings. The trajectory, and so the urgency, is calculated from them.")
t1, t2 = st.columns([1, 2])
with t1:
    st.selectbox("Number of previous visits", [0, 1, 2, 3], key="n_prev")
with t2:
    traj_mode = st.radio("How should the trajectory be determined?",
                         ["Auto-detect from previous visits", "I will choose it"],
                         horizontal=True, key="traj_mode")
    manual_pattern = None
    if traj_mode == "I will choose it":
        manual_pattern = st.selectbox("Trajectory pattern", PATTERNS, key="manual_pattern")

n_prev = st.session_state["n_prev"]
prev_visits = []
for i in range(n_prev):
    st.markdown(f"**Previous visit {i + 1}**" + (" (oldest)" if i == 0 and n_prev > 1 else ""))
    cols = st.columns(5)
    row = {"Age": st.session_state["age"]}
    for col, (suffix, name) in zip(cols, [("sbp", "SystolicBP"), ("dbp", "DiastolicBP"), ("bs", "BS"),
                                          ("temp", "BodyTemp"), ("hr", "HeartRate")]):
        lab, lo, hi, step, is_int = FIELDS[suffix]
        row[name] = col.number_input(lab, min_value=lo, max_value=hi, step=step,
                                     key=f"pv{i}_{suffix}", format=None if is_int else "%.1f")
    prev_visits.append(row)

# ---- 3. confirm --------------------------------------------------------------
st.subheader("3. Confirm and assess")
st.checkbox("I have checked these values against the patient. (Typed or spoken values must be "
            "confirmed by a person before they are assessed.)", key="confirmed")
if not st.session_state["confirmed"]:
    st.info("Tick the box above to run the assessment.")
    st.stop()

# ---- assessment --------------------------------------------------------------
current = {
    "Age": st.session_state["age"], "SystolicBP": st.session_state["sbp"],
    "DiastolicBP": st.session_state["dbp"], "BS": st.session_state["bs"],
    "BodyTemp": st.session_state["temp"], "HeartRate": st.session_state["hr"],
}
if current["SystolicBP"] <= current["DiastolicBP"]:
    st.error("Systolic BP must be higher than diastolic BP. Please correct the values.")
    st.stop()

probs = predict_probs(clf, current)
risk_level = max(probs, key=probs.get)

if manual_pattern:
    pattern, scores, how = manual_pattern, None, "chosen manually"
elif n_prev == 0:
    pattern, scores, how = "Stable", None, "no previous visits entered, so assumed Stable"
else:
    pattern, scores = detect_trajectory(clf, prev_visits + [current])
    how = "auto-detected from the visits entered"

top = shap_top_factors(explainer, clf, current, risk_level)
hb = st.session_state["hb"] if st.session_state["use_hb"] else None
rec = generate_recommendation(
    risk_level, pattern,
    vitals={k: float(v) for k, v in current.items() if k != "Age"},
    anemia_tier=anemia_tier_from_hb(hb),
    shap_top_factors=top,
)

st.divider()
st.header("Assessment")

box, headline = URGENCY_STYLE[rec["overall_urgency"]]
prim = rec["primary_recommendation"]
# The engine's overall urgency is the MAX of the risk x trajectory row and any vital-sign
# flag. When a flag is what raised it, say so, instead of showing a headline that
# contradicts the timeframe of the table row underneath it.
driving = ([f for f in rec["vital_flags"] if f["urgency"] == rec["overall_urgency"]]
           if rec["overall_urgency"] > prim["urgency"] else [])
if driving:
    reasons = " ".join(f"**{f['flag']}:** {f['detail']}" for f in driving)
    getattr(st, box)(f"**{headline}.** {reasons}  \n"
                     f"**Basis:** {driving[0]['basis']}  \n"
                     f"**Pathway from risk level \u00d7 trajectory:** {prim['action']} ({prim['timeframe']}).")
else:
    getattr(st, box)(f"**{headline}.** {prim['action']}  \n**When:** {prim['timeframe']}")

m1, m2, m3 = st.columns(3)
m1.metric("Predicted risk level", risk_level)
m2.metric("Trajectory", rec["trajectory_pattern"])
m3.metric("Overall urgency", f"{rec['overall_urgency']} / 4")

if scores:
    st.caption(f"Trajectory was auto-detected from the {n_prev} previous visit(s) entered in section 2. "
               "If those numbers are not the patient's real earlier readings, the trajectory and urgency "
               "are not valid. Open \u201cHow the trajectory was decided\u201d below to check.")

left, right = st.columns(2)
with left:
    st.markdown("#### Recommended action")
    st.markdown(f"**{prim['action']}**  \n{prim['timeframe']}")
    st.caption(f"Basis: {prim['basis']}")

    if rec["vital_flags"]:
        st.markdown("#### Vital-sign flags")
        for f in rec["vital_flags"]:
            st.markdown(f"- **{f['flag']}**: {f['detail']}")
            st.caption(f"Basis: {f['basis']}")
    else:
        st.markdown("#### Vital-sign flags")
        st.write("None: all vitals are below the rule thresholds.")

    if "anemia_recommendation" in rec:
        a = rec["anemia_recommendation"]
        st.markdown(f"#### Anemia check ({a['tier']})")
        st.write(a["action"])
        st.caption(f"Basis: {a['basis']}")

with right:
    st.markdown("#### How confident is the model?")
    for _c in CLASS_ORDER:
        st.write(f"{_c}: **{probs[_c]:.0%}**")
        st.progress(min(max(probs[_c], 0.0), 1.0))

    st.markdown("#### Why this risk level? (SHAP)")
    if top:
        for feat, sv, val in top:
            direction = "raised" if sv > 0 else "lowered"
            st.markdown(f"- **{feat} = {val:g}** {direction} the chance of *{risk_level}* ({sv:+.2f})")
        st.caption("SHAP shows how much each vital pushed the model towards this risk level.")
    else:
        st.caption("SHAP explanations are not available (the shap package is not installed).")

with st.expander(f"How the trajectory was decided ({how})"):
    if scores:
        labels = [f"Visit {i + 1}" for i in range(len(scores) - 1)] + ["Current"]
        _used = pd.DataFrame(prev_visits + [current], index=labels)[
            ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
        _used["risk score"] = [round(x, 2) for x in scores]
        st.dataframe(_used)
        st.line_chart(pd.Series(scores, index=labels, name="risk score"))
        st.caption("Risk score per visit = P(High) + 0.5 \u00d7 P(Mid), from the ML model. "
                   f"A jump of \u2265 {SUDDEN_STEP:g} in the last step means sudden deterioration; "
                   f"a drift of \u2265 {GRADUAL_TOTAL:g} overall means gradually worsening / improving. "
                   "These thresholds are placeholders to be tuned on real follow-up data.")
    else:
        st.write(f"Pattern: **{pattern}**, {how}.")

st.markdown("#### Summary for the health worker")
st.write(rec["explanation"])

# ---- read aloud + download --------------------------------------------------
a1, a2 = st.columns(2)
with a1:
    if st.button("\U0001F50A Read the recommendation aloud", key="read_aloud"):
        spoken = (f"Risk level {risk_level}. Trajectory {rec['trajectory_pattern']}. "
                  f"Recommended action: {prim['action']}. When: {prim['timeframe']}.")
        if rec["vital_flags"]:
            spoken += " Flags: " + ", ".join(f["flag"] for f in rec["vital_flags"]) + "."
        wav = speak_text(spoken)
        if wav:
            st.audio(wav, format="audio/wav", autoplay=True)
        else:
            st.warning("Offline text-to-speech is not available on this computer.")
with a2:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "vitals": current,
        "hemoglobin": hb,
        "previous_visits": prev_visits,
        "model_probabilities": probs,
        "trajectory_how": how,
        "recommendation": rec,
    }
    st.download_button("\u2B07\uFE0F Download assessment (JSON)", json.dumps(report, indent=2, default=str),
                       file_name="maternacare_assessment.json", mime="application/json")