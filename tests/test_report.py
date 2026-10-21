"""test_report.py -- the printable PDF report (needs: pip install reportlab)."""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

try:
    from report import build_pdf
except ImportError:
    print("[SKIP] reportlab is not installed (pip install reportlab)")
    sys.exit(0)
from src.rules.rules_engine import generate_recommendation


def _data(risk="Low Risk", pattern="Stable", vit=None, hb=None, prev=None, shap=None, label="", worker=""):
    vit = vit or dict(Age=26, SystolicBP=112, DiastolicBP=74, BS=7.2, BodyTemp=98.0, HeartRate=70)
    rec = generate_recommendation(risk, pattern, {k: float(v) for k, v in vit.items() if k != "Age"},
                                  None if hb is None else ("Severe Risk" if hb < 7 else "Moderate Risk" if hb < 10 else "Low Risk"),
                                  shap)
    return dict(generated_at="12 Oct 2026, 10:00", patient_label=label, health_worker=worker, vitals=vit,
                hemoglobin=hb, previous_visits=prev or [], risk_level=risk,
                model_probabilities={"Low Risk": 0.8, "Mid Risk": 0.15, "High Risk": 0.05},
                trajectory=dict(pattern=pattern, how="no previous visits entered, so assumed Stable",
                                confidence=None, probabilities=None),
                urgency=dict(level=rec["overall_urgency"], headline="Routine care"),
                recommendation=rec, shap=shap)


def _pages(pdf):
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def test_plain_report_is_one_page():
    pdf = build_pdf(_data())
    assert pdf.startswith(b"%PDF") and _pages(pdf) == 1
    print("[PASS] a plain low-risk report is a valid, single-page PDF")


def test_full_report_builds():
    vit = dict(Age=30, SystolicBP=162, DiastolicBP=108, BS=14.5, BodyTemp=99.1, HeartRate=92)
    prev = [dict(SystolicBP=118, DiastolicBP=76, BS=7.0, BodyTemp=98.0, HeartRate=74)] * 2
    pdf = build_pdf(_data("High Risk", "Sudden Deterioration", vit, 8.5, prev,
                          [("SystolicBP", 0.43, 162.0), ("BS", 0.2, 14.5)], "ANC-2291", "Priya S."))
    assert pdf.startswith(b"%PDF") and 1 <= _pages(pdf) <= 2
    print("[PASS] a fully loaded high-risk report (previous visits, anemia, flags, SHAP) builds")


def test_non_latin_names_do_not_crash():
    pdf = build_pdf(_data(label="\u0b85\u0ba9\u0bbf\u0ba4\u0bbe / Meena \u00d1", worker="\u0baa\u0bbf\u0bb0\u0bbf\u0baf\u0bbe"))
    assert pdf.startswith(b"%PDF")
    print("[PASS] Tamil / accented characters in names don't crash the report")


if __name__ == "__main__":
    test_plain_report_is_one_page()
    test_full_report_builds()
    test_non_latin_names_do_not_crash()
    print("\nAll report tests passed.")
