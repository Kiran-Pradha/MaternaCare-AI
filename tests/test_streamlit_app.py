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


def test_waits_for_confirmation():
    at = fresh()
    assert not at.exception, at.exception
    assert len(at.metric) == 0, "assessment must not run before the confirmation box is ticked"
    print("[PASS] nothing is assessed until a person confirms the values")


def test_high_risk_example():
    at = fresh()
    at.button(key="ex_High-risk, rising readings").click().run()
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
    at.button(key="ex_Lower-risk patient").click().run()
    at.checkbox(key="confirmed").check().run()
    assert not at.exception, at.exception
    assert metrics(at)["Predicted risk level"] == "Low Risk"
    assert metrics(at)["Overall urgency"] == "0 / 4"
    print("[PASS] low-risk example: Low Risk, urgency 0/4")


def test_sentence_fills_form():
    at = fresh()
    at.text_input(key="sentence").set_value(
        "blood pressure one four zero over ninety heart rate eighty eight")
    at.button(key="fill_sentence").click().run()
    s = at.session_state
    assert (s["sbp"], s["dbp"], s["hr"]) == (140, 90, 88)
    assert s["confirmed"] is False
    print("[PASS] a spoken-style sentence fills BP 140/90 and HR 88, and un-confirms the form")


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


if __name__ == "__main__":
    if not os.path.isdir(os.path.join(ROOT, "data")):
        print("[SKIP] no data/ folder")
        sys.exit(0)
    test_waits_for_confirmation()
    test_high_risk_example()
    test_low_risk_example()
    test_sentence_fills_form()
    test_sudden_jump_detected()
    test_trajectory_shapes()
    test_bad_bp_is_blocked()
    print("\nAll Streamlit app tests passed.")
