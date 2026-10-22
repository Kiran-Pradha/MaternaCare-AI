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
* Trajectory .......... src/models/trajectory_classifier.py: a classifier trained on the
                        SYNTHETIC multi-visit trajectories (the only place "change across
                        visits" exists, since the real data has one visit per patient).
                        Falls back to a simple heuristic if that module is unavailable.
"""

import html
import os
import re
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
for _p in (ROOT, APP_DIR, os.path.join(ROOT, "src", "voice"), os.path.join(ROOT, "src", "data"),
           os.path.join(ROOT, "src", "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.rules.rules_engine import generate_recommendation  # noqa: E402

try:                                    # Phase 8 text parsing (pure Python, no audio)
    from vitals_parser import parse_field_value, parse_vitals_from_text  # noqa: E402
    PARSER_OK = True
except Exception:                       # app still works with typed input only
    PARSER_OK = False

try:                                    # trajectory classifier (trained on synthetic trajectories)
    from trajectory_classifier import TrajectoryClassifier  # noqa: E402
    TRAJ_OK = True
except Exception:
    TRAJ_OK = False

try:                                    # printable PDF report (pip install reportlab)
    from report import build_pdf  # noqa: E402
    REPORT_OK = True
except Exception:
    REPORT_OK = False

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


@st.cache_resource(show_spinner="Training the trajectory classifier on simulated multi-visit patients ...")
def load_trajectory_model(csv_path):
    """Train on synthetic trajectories (visit 1 = a real row, later visits simulated) and
    evaluate on fresh simulated patients. Returns (classifier, evaluation) or (None, None)."""
    if not TRAJ_OK:
        return None, None
    try:
        real = read_dataset(csv_path)
        tc = TrajectoryClassifier().fit(real)
        return tc, tc.evaluate(real)
    except Exception:
        return None, None


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


VOICE_PROMPTS = {   # field -> (name, what the computer says)
    "Age": ("age", "Please say the age in years."),
    "BP": ("blood pressure", "Please say the blood pressure. First the top number, then the bottom number."),
    "BS": ("blood sugar", "Please say the blood sugar."),
    "BodyTemp": ("temperature", "Please say the body temperature in Fahrenheit."),
    "HeartRate": ("heart rate", "Please say the heart rate."),
    "Hemoglobin": ("hemoglobin", "Please say the hemoglobin."),
}
FIELD_TO_KEY = {"Age": "age", "BS": "bs", "BodyTemp": "temp", "HeartRate": "hr", "Hemoglobin": "hb"}


def cb_start_voice(job):
    """Button callback: only REMEMBER what was asked. The work happens at the top of the next
    run (run_voice_job), before the input boxes exist, so it is allowed to fill them."""
    st.session_state["voice_job"] = job


def say(text):
    """Speak through this computer's speakers (offline), if spoken prompts are switched on."""
    if not st.session_state.get("spoken_prompts", True):
        return
    try:
        from tts_offline import speak_offline
        speak_offline(text)
        time.sleep(0.3)          # let the sound die away so the microphone doesn't hear it
    except Exception:
        pass


def _capture(field, tries=2):
    """Say what is needed, listen for 6 seconds, parse. -> (value, transcript, error_text)."""
    from stt_vosk_integration import transcribe_microphone, NUMBER_VOCABULARY
    name, prompt = VOICE_PROMPTS[field]
    transcript, error = "", ""
    for attempt in range(tries):
        st.toast(prompt.replace("Please say", "Say"), icon="\U0001F3A4")   # visible even if scrolled down
        say(prompt if attempt == 0 else f"Sorry, I did not catch that. {prompt}")
        try:
            transcript = transcribe_microphone(duration_seconds=6, vocabulary=NUMBER_VOCABULARY)
        except Exception as e:
            return None, "", f"Microphone problem: {e}"
        value, err = parse_field_value(transcript, field)
        if not err:
            return value, transcript, ""
        error = f"Heard \u201c{transcript}\u201d. {err}"
    return None, transcript, error


def run_voice_job(job):
    """job = one field ("Age", "BP", "BS", "BodyTemp", "HeartRate", "Hemoglobin") or "ALL"."""
    ready, reason = mic_status()
    if not ready:
        st.session_state["voice_msg"] = ("error", f"Microphone entry is not set up: {reason}.")
        return
    if job == "ALL":
        fields = ["Age", "BP", "BS", "BodyTemp", "HeartRate"] + (["Hemoglobin"] if st.session_state.get("ask_hb") else [])
    else:
        fields = [job]
    done, failed = [], []
    for field in fields:
        name = VOICE_PROMPTS[field][0]
        value, transcript, err = _capture(field)
        if err:
            failed.append(f"{name} ({err})")
            say(f"Sorry, I could not understand the {name}.")
            continue
        if field == "BP":
            _set_field("sbp", value[0])
            _set_field("dbp", value[1])
            shown = f"{value[0]:g} over {value[1]:g}"
        else:
            _set_field(FIELD_TO_KEY[field], value)
            if field == "Hemoglobin":
                st.session_state["use_hb"] = True
            shown = f"{value:g}"
        done.append(f"{name} {shown}")
        say(f"I heard {shown}.")
    st.session_state["confirmed"] = False
    if done:
        say("Please check the values on the screen, then tick the confirmation box.")
    text = ("Filled: " + ", ".join(done) + "." if done else "Nothing was filled.")
    if failed:
        text += " Not understood: " + "; ".join(failed) + ". You can type those values instead."
    text += " Please check the boxes below."
    st.session_state["voice_msg"] = ("success" if done and not failed else "warning" if done else "error", text)


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
# Assessment helpers (HTML snippets; every dynamic value is escaped)
# --------------------------------------------------------------------------
UR_PALETTE = {   # urgency -> (light tint, strong colour)
    0: ("#EAF6EF", "#1F7A4D"),
    1: ("#EAF6EF", "#1F7A4D"),
    2: ("#FFF5DF", "#B36B00"),
    3: ("#FDEBDD", "#C2410C"),
    4: ("#FDE8E6", "#B42318"),
}
RISK_COLOR = {"Low Risk": "#1F7A4D", "Mid Risk": "#B36B00", "High Risk": "#B42318"}
TRAJ_COLOR = {"Stable": "#2F5D9B", "Gradually Worsening": "#B36B00",
              "Sudden Deterioration": "#B42318", "Improving": "#1F7A4D"}
ANEMIA_COLOR = {"Low Risk": "#1F7A4D", "Moderate Risk": "#B36B00", "Severe Risk": "#B42318"}


def _e(text):
    """Escape text for HTML, keep **bold** and line breaks."""
    t = html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return t.replace("\n", "<br>")


def _banner(headline, urg, lines_html):
    bg, col = UR_PALETTE[urg]
    segs = "".join(f'<span class="mc-seg{" on" if i < urg else ""}"></span>' for i in range(4))
    return (f'<div class="mc-hbanner" style="border-color:{col};">'
            f'<div class="mc-hbanner-top" style="background:{col};"><span class="mc-hbanner-title">{_e(headline)}</span>'
            f'<span class="mc-hbanner-level"><span class="mc-seg-label">Urgency {urg} of 4</span>{segs}</span></div>'
            f'<div class="mc-hbanner-body" style="background:{bg};">{lines_html}</div></div>')


def _strip(cells):
    """cells = [(label, value, sub, colour)] -> three result cards."""
    return '<div class="mc-strip">' + "".join(
        f'<div class="mc-cell" style="border-top-color:{c};background:{c}12;">'
        f'<div class="mc-cell-label">{_e(l)}</div><div class="mc-cell-value" style="color:{c};">{_e(v)}</div>'
        f'<div class="mc-cell-sub">{_e(sub)}</div></div>'
        for l, v, sub, c in cells) + "</div>"


def _panel(title, body, meta=None, meta_color=None):
    m = ""
    if meta:
        st_ = (f' style="color:{meta_color};border-color:{meta_color}66;background:{meta_color}12;"'
               if meta_color else "")
        m = f'<span class="mc-meta"{st_}>{_e(meta)}</span>'
    return (f'<div class="mc-panel"><div class="mc-panel-head"><span class="mc-panel-title">{_e(title)}</span>{m}</div>'
            f'<div class="mc-panel-body">{body}</div></div>')


def _bar_rows(items):
    """items = [(label, fraction 0-1, colour, is_predicted)]"""
    rows = []
    for label, frac, color, pred in items:
        frac = min(max(frac, 0.0), 1.0)
        bg = f"background:{color}14;" if pred else ""
        lab = f"<b>{_e(label)}</b>" if pred else _e(label)
        rows.append(f'<div class="mc-bar-row" style="{bg}"><div class="mc-bar-top"><span>{lab}</span>'
                    f'<span class="mc-bar-pct" style="color:{color};">{frac:.0%}</span></div>'
                    f'<div class="mc-track"><div class="mc-fill" style="width:{frac * 100:.0f}%;background:{color};"></div></div></div>')
    return "".join(rows)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="MaternaCare AI", page_icon="\U0001FA7A", layout="wide",
                   initial_sidebar_state="collapsed")

# Clinical teal-and-blue look. The base colours come from .streamlit/config.toml (read when
# Streamlit STARTS); this CSS adds the finishing touches. All text uses the same font as the
# rest of the page, so the assessment looks like the other sections.
THEME_TOML = """[theme]
base = "light"
primaryColor = "#0F766E"
backgroundColor = "#F5FAFA"
secondaryBackgroundColor = "#E3F2F1"
textColor = "#12343B"
font = "sans serif"
"""
THEME_CSS = """
.stApp { background: linear-gradient(180deg, #F5FAFA 0%, #EAF4F6 100%); }
h1, h2, h3, h4, h5 { color: #0B4F55 !important; }
.mc-hero { background: linear-gradient(120deg, #D9EFEE 0%, #DCE8F6 100%); border: 1px solid #BCD6D9;
           border-left: 6px solid #0F766E; border-radius: 14px; padding: 22px 30px; margin: 0 0 14px 0;
           box-shadow: 0 3px 12px rgba(11, 79, 85, 0.10); }
.mc-hero-title { font-size: 2.2rem; font-weight: 700; color: #0B4F55; line-height: 1.2; }
.mc-hero-sub { font-size: 1.05rem; color: #31565C; margin-top: 6px; }
button { border-radius: 10px !important; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #CFE3E5; border-radius: 12px;
                           padding: 12px 18px; box-shadow: 0 2px 8px rgba(11, 79, 85, 0.08); }
[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #CFE3E5 !important; border-radius: 12px; }
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { display: none !important; }

/* ---- assessment (inherits the page font so it matches the other sections) ---- */
.mc-hbanner, .mc-cell, .mc-panel { font-family: inherit; }
.mc-hbanner { border: 1px solid; border-radius: 12px; overflow: hidden; margin: 0 0 16px 0; box-shadow: 0 3px 12px rgba(11, 79, 85, 0.10); }
.mc-hbanner-top { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px 18px;
                  padding: 14px 22px; color: #FFFFFF; }
.mc-hbanner-title { font-size: 1.45rem; font-weight: 700; letter-spacing: 0.1px; }
.mc-hbanner-level { font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; white-space: nowrap; }
.mc-seg { display: inline-block; width: 24px; height: 6px; border-radius: 3px; background: rgba(255, 255, 255, 0.35); margin-left: 5px; }
.mc-seg.on { background: #FFFFFF; }
.mc-seg-label { margin-right: 6px; }
.mc-hbanner-body { padding: 14px 22px 16px 22px; color: #17323A; line-height: 1.6; font-size: 0.98rem; }
.mc-hbanner-body div + div { margin-top: 4px; }
.mc-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 0 0 16px 0; }
.mc-cell { border: 1px solid #CFE3E5; border-top-width: 4px; border-radius: 10px; padding: 14px 18px 16px 18px;
           box-shadow: 0 2px 8px rgba(11, 79, 85, 0.07); }
.mc-cell-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #4F6F75; }
.mc-cell-value { font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin-top: 6px; }
.mc-cell-sub { font-size: 0.85rem; color: #5B7479; margin-top: 4px; }
.mc-panel { background: #FFFFFF; border: 1px solid #CFE3E5; border-radius: 12px; margin: 0 0 16px 0; overflow: hidden;
            box-shadow: 0 2px 10px rgba(11, 79, 85, 0.07); }
.mc-panel-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 18px;
                 background: linear-gradient(90deg, #E6F4F3 0%, #E8F0FA 100%); border-bottom: 1px solid #CFE3E5;
                 border-left: 5px solid #0F766E; }
.mc-panel-title { font-size: 1.1rem; font-weight: 700; color: #0B4F55; }
.mc-meta { background: #FFFFFF; border: 1px solid #BCD6D9; border-radius: 999px; padding: 2px 12px; font-size: 0.78rem;
           font-weight: 600; color: #0B4F55; white-space: nowrap; }
.mc-panel-body { padding: 16px 18px; color: #17323A; font-size: 0.97rem; line-height: 1.6; }
.mc-action { font-size: 1.3rem; font-weight: 700; color: #0B4F55; line-height: 1.35; }
.mc-when { display: inline-flex; align-items: center; gap: 10px; background: #EAF5F5; border: 1px solid #CFE3E5;
           border-radius: 8px; padding: 6px 12px; margin-top: 12px; font-weight: 500; }
.mc-when-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #0F766E; }
.mc-note { font-size: 0.84rem; color: #5B7479; margin-top: 12px; line-height: 1.5; }
.mc-flag { border-left: 4px solid; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; }
.mc-flag:last-child { margin-bottom: 0; }
.mc-flag-top { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
.mc-flag-title { font-weight: 700; }
.mc-flag-level { font-size: 0.78rem; font-weight: 700; white-space: nowrap; }
.mc-flag-basis { font-size: 0.82rem; color: #5B7479; margin-top: 4px; }
.mc-bar-row { border-radius: 10px; padding: 8px 12px; margin-bottom: 6px; }
.mc-bar-row:last-child { margin-bottom: 0; }
.mc-bar-top { display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 6px; }
.mc-bar-pct { font-weight: 700; font-size: 1.05rem; }
.mc-track { background: #E1EEEF; border-radius: 6px; height: 10px; overflow: hidden; }
.mc-fill { height: 100%; border-radius: 6px; }
.mc-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.mc-table th { text-align: left; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
               color: #4F6F75; padding: 0 10px 8px 0; border-bottom: 2px solid #CFE3E5; }
.mc-table td { padding: 10px 10px 10px 0; border-bottom: 1px solid #E6F0F1; color: #17323A; vertical-align: middle; }
.mc-table tr:last-child td { border-bottom: none; }
.mc-shap-cell { display: flex; align-items: center; gap: 10px; }
.mc-shap-track { flex: 1; background: #E6F0F1; border-radius: 5px; height: 8px; overflow: hidden; min-width: 60px; }
.mc-shap-fill { height: 100%; border-radius: 5px; }
.mc-shap-num { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 44px; text-align: right; }
.mc-summary { font-size: 1.02rem; line-height: 1.7; }
"""
st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


def ensure_theme_file():
    """Streamlit takes its colours from .streamlit/config.toml in the folder it was started from.
    If the light theme is not active, create that file (never overwriting an existing one).
    Returns None (all fine), "created", "exists" or "failed"."""
    try:
        if st.get_option("theme.base") == "light":
            return None
    except Exception:
        pass
    path = os.path.join(os.getcwd(), ".streamlit", "config.toml")
    if os.path.exists(path):
        return "exists"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(THEME_TOML)
        return "created"
    except Exception:
        return "failed"


_theme_state = ensure_theme_file()

# default values (set once; widgets keep their own state afterwards)
_defaults = dict(age=28, sbp=120, dbp=80, bs=7.0, temp=98.0, hr=76, use_hb=False, hb=11.5,
                 n_prev=0, confirmed=False, spoken_prompts=True, ask_hb=False, patient_label="", health_worker="", sentence="age 32, blood pressure 140 over 90, "
                 "blood sugar 7.5, temperature 98.6, heart rate 88")
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)
for i in range(3):
    for suffix, v in zip(("sbp", "dbp", "bs", "temp", "hr"), (118, 76, 7.0, 98.0, 74)):
        st.session_state.setdefault(f"pv{i}_{suffix}", v)

_job = st.session_state.pop("voice_job", None)
if _job:
    run_voice_job(_job)          # must run BEFORE the input boxes are drawn so it can fill them

csv_path, _checked = find_dataset()

if csv_path is None:
    st.title("\U0001FA7A MaternaCare AI")
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
traj_clf, traj_eval = load_trajectory_model(csv_path)
explainer = load_explainer(clf)

st.markdown(
    '<div class="mc-hero"><div class="mc-hero-title">\U0001FA7A MaternaCare AI</div>'
    '<div class="mc-hero-sub">AI-assisted maternal risk assessment and clinical decision support '
    'for frontline health workers</div></div>',
    unsafe_allow_html=True)

# ---- 1. vitals ------------------------------------------------------------
st.subheader("1. Patient vitals")

st.markdown("#### Voice entry (optional)")
with st.container(border=True):
    ready, reason = mic_status()
    if ready:
        st.checkbox("\U0001F50A Speak the questions aloud", key="spoken_prompts")
        g1, g2 = st.columns([2, 1])
        g1.button("\U0001F399\uFE0F Ask me every question (hands-free)", key="spk_ALL", on_click=cb_start_voice,
                  args=("ALL",), type="primary")
        g2.checkbox("Also ask for hemoglobin", key="ask_hb")
        st.markdown("Or fill one value at a time:")
        bcols = st.columns(6)
        for col, (label, field) in zip(bcols, [("Age", "Age"), ("Blood pressure", "BP"),
                                               ("Blood sugar", "BS"), ("Temperature", "BodyTemp"),
                                               ("Heart rate", "HeartRate"), ("Hemoglobin", "Hemoglobin")]):
            col.button(f"Speak {label.lower()}", key=f"spk_{field}", on_click=cb_start_voice, args=(field,))
    else:
        st.info(f"Microphone entry is not set up on this computer: {reason}. "
                "You can still type values or use the sentence box below.")
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

traj_probs, scores = None, None
if manual_pattern:
    pattern, how = manual_pattern, "chosen manually"
elif n_prev == 0:
    pattern, how = "Stable", "no previous visits entered, so assumed Stable"
elif traj_clf is not None:
    pattern, traj_probs = traj_clf.predict(prev_visits + [current])
    how = "auto-detected by the trajectory classifier from the visits entered"
else:
    pattern, scores = detect_trajectory(clf, prev_visits + [current])
    how = "auto-detected by the fallback heuristic from the visits entered"

top = shap_top_factors(explainer, clf, current, risk_level)
hb = st.session_state["hb"] if st.session_state["use_hb"] else None
rec = generate_recommendation(
    risk_level, pattern,
    vitals={k: float(v) for k, v in current.items() if k != "Age"},
    anemia_tier=anemia_tier_from_hb(hb),
    shap_top_factors=top,
)

st.divider()
st.subheader("4. Assessment")

prim = rec["primary_recommendation"]
urg = rec["overall_urgency"]
_, headline = URGENCY_STYLE[urg]
ur_bg, ur_col = UR_PALETTE[urg]

# The engine's overall urgency is the MAX of the risk x trajectory row and any vital-sign
# flag. When a flag is what raised it, say so, instead of showing a headline that
# contradicts the timeframe of the table row underneath it.
driving = ([f for f in rec["vital_flags"] if f["urgency"] == urg] if urg > prim["urgency"] else [])
if driving:
    _lines = "".join(f'<div><b>{_e(f["flag"])}:</b> {_e(f["detail"])}</div>' for f in driving)
    _lines += f'<div><b>Basis:</b> {_e(driving[0]["basis"])}</div>'
    _lines += (f'<div><b>Pathway from risk level \u00d7 trajectory:</b> {_e(prim["action"])} '
               f'({_e(prim["timeframe"])})</div>')
else:
    _lines = f'<div>{_e(prim["action"])}</div><div><b>When:</b> {_e(prim["timeframe"])}</div>'
st.markdown(_banner(headline, urg, _lines), unsafe_allow_html=True)

_how_txt = how[:1].upper() + how[1:]
st.markdown(_strip([
    ("Predicted risk level", risk_level, f"Model probability {probs[risk_level]:.0%}",
     RISK_COLOR.get(risk_level, "#12343B")),
    ("Trajectory", rec["trajectory_pattern"], _how_txt, TRAJ_COLOR.get(rec["trajectory_pattern"], "#12343B")),
    ("Overall urgency", f"{urg} of 4", headline, ur_col),
]), unsafe_allow_html=True)

if traj_probs or scores:
    _conf = ""
    if traj_probs:
        _top = traj_probs[pattern]
        _conf = f" Pattern confidence: {_top:.0%}" + (" (LOW: treat the trajectory with caution)." if _top < 0.5 else ".")
    st.caption(f"Trajectory was auto-detected from the {n_prev} previous visit(s) entered in section 2."
               f"{_conf} If those numbers are not the patient's real earlier readings, the trajectory "
               "and urgency are not valid. Open \u201cHow the trajectory was decided\u201d below to check.")

left, right = st.columns(2)

with left:
    st.markdown(_panel(
        "Recommended action",
        f'<div class="mc-action">{_e(prim["action"])}</div>'
        f'<div class="mc-when"><span class="mc-when-label">When</span><span>{_e(prim["timeframe"])}</span></div>'
        f'<div class="mc-note">Basis: {_e(prim["basis"])}</div>',
        meta=f"Urgency {prim['urgency']} of 4", meta_color=UR_PALETTE[prim["urgency"]][1]),
        unsafe_allow_html=True)

    if rec["vital_flags"]:
        _body = "".join(
            f'<div class="mc-flag" style="border-left-color:{UR_PALETTE[f["urgency"]][1]};'
            f'background:{UR_PALETTE[f["urgency"]][0]};">'
            f'<div class="mc-flag-top"><span class="mc-flag-title">{_e(f["flag"])}</span>'
            f'<span class="mc-flag-level" style="color:{UR_PALETTE[f["urgency"]][1]};">Urgency {f["urgency"]} of 4</span></div>'
            f'<div>{_e(f["detail"])}</div><div class="mc-flag-basis">Basis: {_e(f["basis"])}</div></div>'
            for f in rec["vital_flags"])
        _meta, _mcol = f"{len(rec['vital_flags'])} flagged", "#B42318"
    else:
        _body = "None. All vitals are below the rule thresholds."
        _meta, _mcol = "No flags", "#1F7A4D"
    st.markdown(_panel("Vital-sign flags", _body, meta=_meta, meta_color=_mcol), unsafe_allow_html=True)

    if "anemia_recommendation" in rec:
        a = rec["anemia_recommendation"]
        st.markdown(_panel(
            "Anemia check",
            f'<div>{_e(a["action"])}</div><div class="mc-note">Basis: {_e(a["basis"])}</div>',
            meta=a["tier"], meta_color=ANEMIA_COLOR.get(a["tier"], "#0B4F55")), unsafe_allow_html=True)

with right:
    _bars = _bar_rows([(c, probs[c], RISK_COLOR[c], c == risk_level) for c in CLASS_ORDER])
    st.markdown(_panel("Model confidence", _bars, meta=f"Predicted: {risk_level}",
                       meta_color=RISK_COLOR[risk_level]), unsafe_allow_html=True)

    if top:
        _max = max(abs(sv) for _, sv, _ in top) or 1.0
        _trs = ""
        for feat, sv, val in top:
            up = sv > 0
            _col = "#C2410C" if up else "#0F766E"
            _trs += (f'<tr><td><b>{_e(feat)}</b></td><td>{val:g}</td>'
                     f'<td style="color:{_col};font-weight:700;">{"Raises" if up else "Lowers"}</td>'
                     f'<td><div class="mc-shap-cell"><div class="mc-shap-track"><div class="mc-shap-fill" '
                     f'style="width:{abs(sv) / _max * 100:.0f}%;background:{_col};"></div></div>'
                     f'<span class="mc-shap-num">{sv:+.2f}</span></div></td></tr>')
        _body = ('<table class="mc-table"><tr><th>Vital</th><th>Value</th>'
                 f'<th>Effect on {_e(risk_level)}</th><th>Contribution</th></tr>{_trs}</table>')
    else:
        _body = '<div class="mc-note">SHAP explanations are not available (the shap package is not installed).</div>'
    st.markdown(_panel("Why this risk level", _body, meta="SHAP"), unsafe_allow_html=True)

st.markdown(_panel("Summary for the health worker", f'<div class="mc-summary">{_e(rec["explanation"])}</div>'),
            unsafe_allow_html=True)

st.markdown("#### How the trajectory was decided")
st.caption(how[:1].upper() + how[1:] + ".")
with st.container(border=True):
    if traj_probs:
        _labels = [f"Previous visit {i + 1}" for i in range(len(prev_visits))] + ["Current visit"]
        st.dataframe(pd.DataFrame(prev_visits + [current], index=_labels)[
            ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]])
        st.markdown("Probability of each pattern, from the trajectory classifier:")
        for _pat in PATTERNS:
            st.write(f"{_pat}: **{traj_probs[_pat]:.0%}**")
            st.progress(min(max(traj_probs[_pat], 0.0), 1.0))
        _len_acc = traj_eval["per_length"].get(n_prev + 1)
        st.caption(
            f"The classifier was trained on {traj_clf.n_train} SIMULATED patients (visit 1 taken from a real "
            f"record, later visits simulated), and looks only at how the readings change between visits. On "
            f"fresh simulated patients it was {traj_eval['accuracy']:.0%} accurate overall"
            + (f" ({_len_acc:.0%} for {n_prev + 1}-visit trajectories)" if _len_acc else "") +
            ". Those are patterns the simulation planted, so this is a proof of concept until it is "
            "checked on real follow-up data. Slow trends are the hardest to tell apart from a stable patient.")
    elif scores:
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

# ---- printable report, read aloud, download -----------------------------------
st.markdown("#### Printable report")
p1, p2 = st.columns(2)
p1.text_input("Patient name or ID (printed on the report only, never stored)", key="patient_label")
p2.text_input("Health worker", key="health_worker")

stamp = datetime.now()
report_data = {
    "generated_at": stamp.strftime("%d %b %Y, %H:%M"),
    "patient_label": st.session_state["patient_label"],
    "health_worker": st.session_state["health_worker"],
    "vitals": current,
    "hemoglobin": hb,
    "previous_visits": prev_visits,
    "risk_level": risk_level,
    "model_probabilities": probs,
    "trajectory": {"pattern": pattern, "how": how,
                   "confidence": traj_probs[pattern] if traj_probs else None, "probabilities": traj_probs},
    "urgency": {"level": rec["overall_urgency"], "headline": headline},
    "recommendation": rec,
    "shap": top,
}

b1, b2 = st.columns(2)
with b1:
    if REPORT_OK:
        try:
            st.download_button("\U0001F5A8\uFE0F Download printable report (PDF)", build_pdf(report_data),
                               file_name=f"maternacare_report_{stamp:%Y%m%d_%H%M}.pdf", mime="application/pdf",
                               key="dl_pdf", type="primary")
        except Exception as e:
            st.error(f"Could not build the PDF report: {e}")
    else:
        st.info("To enable the printable PDF report, run:  pip install reportlab  (then restart the app).")
with b2:
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
st.caption("Open the downloaded PDF and print it, or save it to the patient's file.")