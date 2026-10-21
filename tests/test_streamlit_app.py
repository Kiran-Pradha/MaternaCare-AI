"""test_streamlit_app.py -- runs the real Streamlit page headlessly (no browser, no
microphone) and checks the main behaviours.

Run from the repository root:   py tests\\test_streamlit_app.py
Needs: streamlit, scikit-learn, pandas, and the dataset CSV in data/.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP = os.path.join(ROOT, "app", "streamlit_app.py")

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("[SKIP] streamlit is not installed (pip install streamlit)")
    sys.exit(0)


def fresh():
    return AppTest.from_file(APP, default_timeout=120).run()


def metrics(at):
    return {m.label: m.value for m in at.metric}


def set_high_risk_readings(at):
    for key, value in dict(age=30, sbp=162, dbp=108, bs=14.5, temp=99.1, hr=92,
                           use_hb=True, hb=8.5, n_prev=2).items():
        at.session_state[key] = value
    for i, values in enumerate(((118, 76, 7.0, 98.0, 74), (135, 86, 9.5, 98.4, 82))):
        for suffix, value in zip(("sbp", "dbp", "bs", "temp", "hr"), values):
            at.session_state[f"pv{i}_{suffix}"] = value


def test_waits_for_confirmation():
    at = fresh()
    assert not at.exception, at.exception
    assert len(at.metric) == 0, "assessment must not run before the confirmation box is ticked"
    assert not [button for button in at.button if button.key.startswith("ex_")], \
        "example buttons should not appear in the UI"
    text = " ".join(x.value for x in at.markdown) + " ".join(x.value for x in at.caption)
    assert "Try an example" not in text
    assert "REAL earlier readings" not in text
    assert not [text_input for text_input in at.text_input if text_input.key == "sentence"]
    assert not [button for button in at.button if button.key == "fill_sentence"]
    print("[PASS] nothing is assessed until a person confirms the values")


def test_high_risk_example():
    at = fresh()
    set_high_risk_readings(at)
    at.run()
    at.checkbox(key="confirmed").check().run()
    assert not at.exception, at.exception
    m = metrics(at)
    assert m["Predicted risk level"] == "High Risk"
    assert m["Overall urgency"] == "4 / 4"
    assert m["Trajectory"] in ("Gradually Worsening", "Sudden Deterioration")
    text = " ".join(x.value for x in at.markdown)
    assert "Severe hypertension" in text           # 162/108 meets the ACOG severe threshold
    print("[PASS] high-risk example: High Risk, rising trajectory, urgency 4/4, severe-BP flag listed")


def test_low_risk_example():
    at = fresh()
    for key, value in dict(age=25, sbp=112, dbp=74, bs=7.2, temp=98.0, hr=70,
                           use_hb=True, hb=12.0, n_prev=0).items():
        at.session_state[key] = value
    at.run()
    at.checkbox(key="confirmed").check().run()
    assert not at.exception, at.exception
    assert metrics(at)["Predicted risk level"] == "Low Risk"
    assert metrics(at)["Overall urgency"] == "0 / 4"
    print("[PASS] low-risk example: Low Risk, urgency 0/4")


def test_sudden_jump_detected():
    at = fresh()
    at.session_state["n_prev"] = 2
    for i in range(2):
        for suffix, v in zip(("sbp", "dbp", "bs", "temp", "hr"), (112, 72, 7.0, 98.0, 72)):
            at.session_state[f"pv{i}_{suffix}"] = v
    for k, v in dict(sbp=150, dbp=98, bs=11.5, temp=99.0, hr=90).items():
        at.session_state[k] = v
    at.session_state["confirmed"] = True
    at.run()
    assert not at.exception, at.exception
    assert metrics(at)["Trajectory"] == "Sudden Deterioration"
    print("[PASS] stable history then a big jump is detected as Sudden Deterioration")


def test_bad_bp_is_blocked():
    at = fresh()
    at.session_state["sbp"], at.session_state["dbp"] = 80, 90
    at.session_state["confirmed"] = True
    at.run()
    assert len(at.metric) == 0 and any("Systolic" in e.value for e in at.error)
    print("[PASS] systolic below diastolic is rejected")


def _run_visits(prev, cur):
    at = fresh()
    at.session_state["n_prev"] = len(prev)
    for i, v in enumerate(prev):
        for suffix, x in zip(("sbp", "dbp", "bs", "temp", "hr"), v):
            at.session_state[f"pv{i}_{suffix}"] = x
    for k, x in zip(("sbp", "dbp", "bs", "temp", "hr"), cur):
        at.session_state[k] = x
    at.session_state["confirmed"] = True
    at.run()
    assert not at.exception, at.exception
    return metrics(at)["Trajectory"]


def test_trajectory_shapes():
    assert _run_visits([(120, 80, 7.0, 98.0, 76)] * 3, (120, 80, 7.0, 98.0, 76)) == "Stable"
    assert _run_visits([(112, 72, 7.0, 98.0, 72), (120, 78, 7.8, 98.2, 76), (128, 83, 8.6, 98.4, 80)],
                       (137, 89, 9.5, 98.7, 85)) == "Gradually Worsening"
    assert _run_visits([(150, 98, 11.0, 99.0, 90), (142, 93, 10.0, 98.8, 86), (134, 88, 9.0, 98.5, 82)],
                       (126, 82, 8.0, 98.2, 78)) == "Improving"
    print("[PASS] flat -> Stable, steady climb -> Gradually Worsening, steady fall -> Improving")


# ---- voice entry, tested with a pretend microphone and speaker --------------------------
class FakeVoice:
    """Installs fake vosk / sounddevice / stt / tts modules so the voice buttons can run without
    audio hardware. `heard` = what the pretend microphone returns, in order; `spoken` collects
    everything the computer 'says'."""

    def __init__(self, heard):
        import types
        self.spoken, self.queue = [], list(heard)
        stt = types.ModuleType("stt_vosk_integration")
        stt.MODEL_DIR = os.path.dirname(__file__)                    # any folder that exists
        stt.NUMBER_VOCABULARY = []
        stt.transcribe_microphone = lambda duration_seconds=6, vocabulary=None, **k: self.queue.pop(0)
        tts = types.ModuleType("tts_offline")
        tts.speak_offline = lambda text, rate=165: self.spoken.append(text)

        def _no_wav(*a, **k):
            raise RuntimeError("no audio device in tests")
        tts.save_offline_wav = _no_wav
        self.fakes = {"vosk": types.ModuleType("vosk"), "sounddevice": types.ModuleType("sounddevice"),
                      "stt_vosk_integration": stt, "tts_offline": tts}
        self.saved = {}

    def __enter__(self):
        for name, mod in self.fakes.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, old in self.saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_voice_single_field_speaks_prompt_and_fills_value():
    with FakeVoice(["thirty two"]) as v:
        at = fresh()
        at.button(key="spk_Age").click().run()
        assert not at.exception, at.exception
        assert at.session_state["age"] == 32
        assert v.spoken[0] == "Please say the age in years.", v.spoken
        assert at.session_state["confirmed"] is False
    print("[PASS] 'Speak age': the computer says 'Please say the age in years.' and fills 32")


def test_voice_hemoglobin():
    with FakeVoice(["nine point five"]) as v:
        at = fresh()
        at.button(key="spk_Hemoglobin").click().run()
        assert not at.exception, at.exception
        assert at.session_state["hb"] == 9.5 and at.session_state["use_hb"] is True
        assert v.spoken[0] == "Please say the hemoglobin."
    print("[PASS] 'Speak hemoglobin' fills 9.5 and switches the anemia check on")


def test_voice_ask_every_question():
    heard = ["three two", "one forty over ninety", "seven point five", "nine eight point six", "eighty eight"]
    with FakeVoice(heard) as v:
        at = fresh()
        at.button(key="spk_ALL").click().run()
        assert not at.exception, at.exception
        s_ = at.session_state
        assert (s_["age"], s_["sbp"], s_["dbp"], s_["bs"], s_["temp"], s_["hr"]) == (32, 140, 90, 7.5, 98.6, 88)
        asked = [t for t in v.spoken if t.startswith("Please say")]
        assert asked == ["Please say the age in years.",
                         "Please say the blood pressure. First the top number, then the bottom number.",
                         "Please say the blood sugar.", "Please say the body temperature in Fahrenheit.",
                         "Please say the heart rate."], asked
        assert not s_["use_hb"]
    print("[PASS] hands-free mode asks all five questions aloud and fills all five values")


def test_voice_ask_every_question_with_hemoglobin():
    heard = ["three two", "one forty over ninety", "seven point five", "nine eight point six",
             "eighty eight", "ten point two"]
    with FakeVoice(heard) as v:
        at = fresh()
        at.checkbox(key="ask_hb").check().run()
        at.button(key="spk_ALL").click().run()
        assert not at.exception, at.exception
        assert at.session_state["hb"] == 10.2 and at.session_state["use_hb"] is True
        assert "Please say the hemoglobin." in v.spoken
    print("[PASS] with 'Also ask for hemoglobin' ticked, the sixth question is asked and filled")


def test_voice_not_understood_leaves_value_and_retries():
    with FakeVoice(["blah", "blah"]) as v:
        at = fresh()
        before = at.session_state["age"]
        at.button(key="spk_Age").click().run()
        assert not at.exception, at.exception
        assert at.session_state["age"] == before
        assert any(t.startswith("Sorry, I did not catch that.") for t in v.spoken)   # asked a second time
        kind, text = at.session_state["voice_msg"]
        assert kind == "error" and "Nothing was filled" in text
    print("[PASS] an answer that can't be understood is retried once, then leaves the box unchanged")


def test_voice_prompts_can_be_switched_off():
    with FakeVoice(["thirty two"]) as v:
        at = fresh()
        at.checkbox(key="spoken_prompts").uncheck().run()
        at.button(key="spk_Age").click().run()
        assert at.session_state["age"] == 32 and v.spoken == []
    print("[PASS] with spoken prompts off the computer stays silent but voice entry still works")


def test_printable_report_button_and_theme_file():
    at = fresh()
    set_high_risk_readings(at)
    at.run()
    at.checkbox(key="confirmed").check().run()
    assert not at.exception, at.exception
    labels = [str(getattr(d, "label", "")) for d in at.get("download_button")]
    assert any("PDF" in l for l in labels), labels
    import tomllib
    with open(os.path.join(ROOT, ".streamlit", "config.toml"), "rb") as f:
        theme = tomllib.load(f)["theme"]
    assert theme["base"] == "light" and theme["primaryColor"].startswith("#")
    print("[PASS] a printable-PDF download button is offered, and the light maternal theme file is valid")


if __name__ == "__main__":
    if not os.path.isdir(os.path.join(ROOT, "data")):
        print("[SKIP] no data/ folder")
        sys.exit(0)
    test_waits_for_confirmation()
    test_high_risk_example()
    test_low_risk_example()
    test_sudden_jump_detected()
    test_trajectory_shapes()
    test_voice_single_field_speaks_prompt_and_fills_value()
    test_voice_hemoglobin()
    test_voice_ask_every_question()
    test_voice_ask_every_question_with_hemoglobin()
    test_voice_not_understood_leaves_value_and_retries()
    test_voice_prompts_can_be_switched_off()
    test_printable_report_button_and_theme_file()
    test_bad_bp_is_blocked()
    print("\nAll Streamlit app tests passed.")
