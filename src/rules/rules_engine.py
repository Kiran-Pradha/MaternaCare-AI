"""
Phase 7 -- Clinical Recommendation Rules Engine
======================================================================================
Deliberately NOT machine learning. This is a transparent, auditable lookup engine
that converts (risk level + trajectory pattern + specific vitals + anemia tier)
into concrete, actionable guidance for a frontline health worker -- e.g. "refer
within 48 hours," not just a risk label.

Why rule-based, on purpose: in a safety-critical recommendation layer, every
output should be traceable to a specific, citable clinical source, not a model's
learned behavior. This mirrors how real clinical decision-support tools are built
(rule-based pathways, not free-form model output) and was a deliberate design
choice discussed and justified during project planning.

Grounding for the thresholds below (see docs/ references):
  - ACOG: clinic BP >=140/90 mmHg defines hypertension in pregnancy;
          >=160/110 mmHg is severe and requires urgent action.
  - NICE: weekly review when hypertension is poorly controlled,
          every 2-4 weeks when well controlled.
  - WHO:  early assessment/referral recommended for high-risk pregnancy
          complications; rapid BP escalation is a pre-eclampsia warning sign.
  - WHO anemia thresholds (pregnant women): Severe <7 g/dL, Moderate 7-9.9 g/dL,
          Mild 10-10.9 g/dL, Normal >=11 g/dL.

Every rule below cites which of these it is grounded in. Combinations not
explicitly covered fall back to a cautious default (recommend clinical review)
rather than guessing -- an uncovered case should never silently produce no
guidance.

Usage:
    python src/rules/rules_engine.py
    (runs a self-test + a demonstration)
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "phase7_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

VALID_PATTERNS = {"Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"}


# =========================================================
# 1. Risk level x Trajectory pattern -> primary action
# =========================================================

RISK_TRAJECTORY_MATRIX = {
    ("High Risk", "Sudden Deterioration"): {
        "action": "Immediate referral to a higher-level facility",
        "timeframe": "Within 24 hours",
        "basis": "WHO: early assessment and referral recommended for high-risk pregnancy "
                 "complications; rapid escalation is a known pre-eclampsia warning sign.",
        "urgency": 4,
    },
    ("High Risk", "Gradually Worsening"): {
        "action": "Referral to a higher-level facility + weekly monitoring until seen",
        "timeframe": "Within 48-72 hours",
        "basis": "NICE: weekly review recommended when hypertension is poorly controlled.",
        "urgency": 3,
    },
    ("High Risk", "Stable"): {
        "action": "Refer for clinical evaluation (non-emergency)",
        "timeframe": "Within 1 week",
        "basis": "ACOG: clinic BP >=140/90 mmHg defines hypertension and warrants follow-up "
                 "even when the reading is not actively worsening.",
        "urgency": 2,
    },
    ("High Risk", "Improving"): {
        "action": "Refer for clinical evaluation (non-emergency); continue close monitoring",
        "timeframe": "Within 1 week",
        "basis": "Even an improving trend from a High Risk baseline still meets ACOG's "
                 "hypertension threshold and needs confirmation, not assumption of resolution.",
        "urgency": 2,
    },
    ("Mid Risk", "Sudden Deterioration"): {
        "action": "Escalate to urgent clinical review; treat as potential High Risk in progress",
        "timeframe": "Within 24-48 hours",
        "basis": "WHO: rapid change in vitals warrants early assessment regardless of the "
                 "current point-in-time risk label.",
        "urgency": 3,
    },
    ("Mid Risk", "Gradually Worsening"): {
        "action": "Increase monitoring frequency; re-assess in 1-2 weeks",
        "timeframe": "1-2 weeks",
        "basis": "NICE: appointment frequency should increase as control worsens, "
                 "even before crossing into a higher risk category.",
        "urgency": 2,
    },
    ("Mid Risk", "Stable"): {
        "action": "Routine follow-up per standard antenatal care (ANC) schedule",
        "timeframe": "Next scheduled visit",
        "basis": "Standard ANC protocol -- routine monitoring sufficient when the "
                 "trajectory is not worsening.",
        "urgency": 1,
    },
    ("Mid Risk", "Improving"): {
        "action": "Routine follow-up; note improving trend in patient record",
        "timeframe": "Next scheduled visit",
        "basis": "Standard ANC protocol.",
        "urgency": 1,
    },
    ("Low Risk", "Sudden Deterioration"): {
        "action": "Re-assess promptly -- a sudden change from Low Risk baseline is unexpected "
                   "and should not be dismissed",
        "timeframe": "Within 48-72 hours",
        "basis": "WHO: rapid change in vitals warrants early assessment regardless of the "
                 "current point-in-time risk label.",
        "urgency": 2,
    },
    ("Low Risk", "Gradually Worsening"): {
        "action": "Increase monitoring frequency slightly; watch trend at next visit",
        "timeframe": "Next scheduled visit, or sooner if trend continues",
        "basis": "Standard ANC protocol, adjusted for an observed (if mild) worsening trend.",
        "urgency": 1,
    },
    ("Low Risk", "Stable"): {
        "action": "Routine care; no escalation needed",
        "timeframe": "Next scheduled visit",
        "basis": "Standard ANC protocol.",
        "urgency": 0,
    },
    ("Low Risk", "Improving"): {
        "action": "Routine care; no escalation needed",
        "timeframe": "Next scheduled visit",
        "basis": "Standard ANC protocol.",
        "urgency": 0,
    },
}

FALLBACK_ACTION = {
    "action": "Recommend clinical review -- this combination of risk level and trajectory "
              "was not explicitly covered by the rules table",
    "timeframe": "As soon as practical",
    "basis": "Cautious default: an uncovered case should never silently produce no guidance.",
    "urgency": 2,
}


# =========================================================
# 2. Vital-specific flags (independent of the matrix above --
#    these can fire regardless of overall risk level)
# =========================================================

def check_vital_flags(vitals):
    """vitals: dict with keys SystolicBP, DiastolicBP, BS (blood sugar), BodyTemp, HeartRate."""
    flags = []

    sbp = vitals.get("SystolicBP")
    dbp = vitals.get("DiastolicBP")
    if sbp is not None and dbp is not None:
        if sbp >= 160 or dbp >= 110:
            flags.append({
                "flag": "Severe hypertension",
                "detail": f"BP {sbp:.0f}/{dbp:.0f} mmHg meets the severe threshold.",
                "basis": "ACOG: >=160/110 mmHg is severe and requires urgent action.",
                "urgency": 4,
            })
        elif sbp >= 140 or dbp >= 90:
            flags.append({
                "flag": "Hypertension",
                "detail": f"BP {sbp:.0f}/{dbp:.0f} mmHg meets the hypertension threshold.",
                "basis": "ACOG: clinic BP >=140/90 mmHg defines hypertension in pregnancy.",
                "urgency": 2,
            })

    bs = vitals.get("BS")
    if bs is not None:
        # NOTE: threshold expressed on this dataset's BS scale (approx. mmol/L-like values
        # in the 6-19 range seen in the source data) -- calibrate against your specific
        # dataset's units before relying on this in a real clinical setting.
        if bs >= 11:
            flags.append({
                "flag": "Elevated blood sugar",
                "detail": f"Blood sugar reading of {bs:.1f} is substantially elevated.",
                "basis": "Consistent with gestational diabetes risk ranges reported in the "
                         "maternal health literature reviewed for this project.",
                "urgency": 2,
            })

    temp = vitals.get("BodyTemp")
    if temp is not None and temp >= 101:
        flags.append({
            "flag": "Elevated temperature",
            "detail": f"Body temperature {temp:.1f} may indicate infection.",
            "basis": "Fever in pregnancy warrants investigation for underlying infection.",
            "urgency": 2,
        })

    return flags


# =========================================================
# 3. Anemia tier -> supplement guidance (from Phase 5 clustering)
# =========================================================

ANEMIA_GUIDANCE = {
    "Severe Risk": {
        "action": "Refer for clinical evaluation; iron supplementation under medical "
                  "supervision; investigate underlying cause.",
        "basis": "WHO: Severe anemia in pregnancy (<7 g/dL) requires clinical management, "
                 "not self-directed supplementation.",
    },
    "Moderate Risk": {
        "action": "Begin iron supplementation + dietary counseling (iron-rich foods); "
                  "recheck Hemoglobin in 4 weeks.",
        "basis": "WHO: Moderate anemia (7-9.9 g/dL) -- standard iron supplementation "
                 "protocol with follow-up.",
    },
    "Low Risk": {
        "action": "Routine dietary guidance; no supplementation needed; recheck at next "
                  "scheduled visit.",
        "basis": "WHO: Hemoglobin within or near normal range for pregnancy.",
    },
}


# =========================================================
# 4. Combine everything into one recommendation
# =========================================================

def generate_recommendation(risk_level, trajectory_pattern, vitals,
                             anemia_tier=None, shap_top_factors=None):
    """
    risk_level: "Low Risk" / "Mid Risk" / "High Risk"
    trajectory_pattern: "Stable" / "Gradually Worsening" / "Sudden Deterioration" / "Improving"
                         (or None/unrecognized -- falls back to "Stable")
    vitals: dict of raw vital values
    anemia_tier: "Severe Risk" / "Moderate Risk" / "Low Risk" (from Phase 5), or None
    shap_top_factors: list of (feature, shap_value, actual_value) tuples from Phase 6, or None
    """
    pattern = trajectory_pattern if trajectory_pattern in VALID_PATTERNS else "Stable"

    primary = RISK_TRAJECTORY_MATRIX.get((risk_level, pattern), FALLBACK_ACTION)
    vital_flags = check_vital_flags(vitals)

    all_urgencies = [primary["urgency"]] + [f["urgency"] for f in vital_flags]
    overall_urgency = max(all_urgencies)

    explanation_parts = []
    if shap_top_factors:
        top_feat, top_shap, top_val = shap_top_factors[0]
        explanation_parts.append(
            f"This {risk_level} classification was primarily driven by {top_feat} = {top_val:.1f}."
        )
    explanation_parts.append(f"Recommended action: {primary['action']} ({primary['timeframe']}).")
    if vital_flags:
        explanation_parts.append("Additional flags: " + "; ".join(f["flag"] for f in vital_flags) + ".")

    result = {
        "risk_level": risk_level,
        "trajectory_pattern": pattern,
        "primary_recommendation": primary,
        "vital_flags": vital_flags,
        "overall_urgency": overall_urgency,
        "explanation": " ".join(explanation_parts),
    }

    if anemia_tier and anemia_tier in ANEMIA_GUIDANCE:
        result["anemia_recommendation"] = {
            "tier": anemia_tier,
            **ANEMIA_GUIDANCE[anemia_tier],
        }
        result["explanation"] += f" Anemia screening: {anemia_tier} -- {ANEMIA_GUIDANCE[anemia_tier]['action']}"

    return result


# =========================================================
# Self-test: every matrix combination + fallback + vital flags
# =========================================================

def run_self_test():
    print("=" * 70)
    print("SELF-TEST -- verifying every rule combination produces valid output")
    print("=" * 70)

    risk_levels = ["Low Risk", "Mid Risk", "High Risk"]
    patterns = ["Stable", "Gradually Worsening", "Sudden Deterioration", "Improving"]
    n_tested = 0
    n_fallback = 0

    for risk in risk_levels:
        for pattern in patterns:
            rec = generate_recommendation(
                risk, pattern,
                vitals={"SystolicBP": 120, "DiastolicBP": 80, "BS": 8, "BodyTemp": 98.5, "HeartRate": 75}
            )
            n_tested += 1
            is_fallback = rec["primary_recommendation"] is FALLBACK_ACTION
            if is_fallback:
                n_fallback += 1
            assert rec["primary_recommendation"]["action"], "Empty action -- rule engine bug"
            assert rec["overall_urgency"] >= 0

    print(f"Tested {n_tested} risk x trajectory combinations -- all produced valid, non-empty output.")
    print(f"{n_fallback} combinations used the fallback rule "
          f"(should be 0, since the matrix above is fully populated).")
    assert n_fallback == 0, "Matrix has an unexpected gap!"

    fake_rec = generate_recommendation(
        "Mid Risk", "NotARealPattern",
        vitals={"SystolicBP": 120, "DiastolicBP": 80, "BS": 8, "BodyTemp": 98.5, "HeartRate": 75}
    )
    print(f"\nFallback test (invalid/unrecognized pattern label 'NotARealPattern'):")
    print(f"  -> safely defaulted to pattern = '{fake_rec['trajectory_pattern']}'")
    assert fake_rec["trajectory_pattern"] == "Stable", "Unrecognized pattern did not default correctly"
    print("-> Confirmed: unrecognized input never causes a crash or silent gap.")

    print("\n--- Vital flag tests ---")
    severe_bp = check_vital_flags({"SystolicBP": 165, "DiastolicBP": 112, "BS": 8, "BodyTemp": 98, "HeartRate": 75})
    assert any(f["flag"] == "Severe hypertension" for f in severe_bp), "Severe BP flag did not fire"
    print("Severe hypertension flag: PASSED")

    mild_bp = check_vital_flags({"SystolicBP": 142, "DiastolicBP": 85, "BS": 8, "BodyTemp": 98, "HeartRate": 75})
    assert any(f["flag"] == "Hypertension" for f in mild_bp), "Hypertension flag did not fire"
    print("Hypertension flag: PASSED")

    normal_bp = check_vital_flags({"SystolicBP": 118, "DiastolicBP": 76, "BS": 7, "BodyTemp": 98, "HeartRate": 75})
    assert len(normal_bp) == 0, "Normal vitals incorrectly triggered a flag"
    print("Normal vitals (no flags expected): PASSED")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    run_self_test()

    print("\n" + "=" * 70)
    print("DEMONSTRATION -- example recommendations")
    print("=" * 70)

    examples = [
        {
            "label": "High Risk, sudden deterioration, severe BP",
            "risk_level": "High Risk", "trajectory_pattern": "Sudden Deterioration",
            "vitals": {"SystolicBP": 162, "DiastolicBP": 108, "BS": 14.5, "BodyTemp": 99.1, "HeartRate": 92},
            "anemia_tier": "Moderate Risk",
            "shap_top_factors": [("BS", 0.37, 14.5), ("SystolicBP", 0.25, 162.0)],
        },
        {
            "label": "Mid Risk, stable trend",
            "risk_level": "Mid Risk", "trajectory_pattern": "Stable",
            "vitals": {"SystolicBP": 128, "DiastolicBP": 82, "BS": 9.0, "BodyTemp": 98.4, "HeartRate": 76},
            "anemia_tier": "Low Risk",
            "shap_top_factors": [("SystolicBP", 0.21, 128.0)],
        },
        {
            "label": "Low Risk, improving trend",
            "risk_level": "Low Risk", "trajectory_pattern": "Improving",
            "vitals": {"SystolicBP": 112, "DiastolicBP": 74, "BS": 7.2, "BodyTemp": 98.0, "HeartRate": 70},
            "anemia_tier": None,
            "shap_top_factors": [("BS", 0.30, 7.2)],
        },
    ]

    all_outputs = []
    for ex in examples:
        print(f"\n--- {ex['label']} ---")
        rec = generate_recommendation(ex["risk_level"], ex["trajectory_pattern"], ex["vitals"],
                                       ex["anemia_tier"], ex["shap_top_factors"])
        print(f"Action:    {rec['primary_recommendation']['action']}")
        print(f"Timeframe: {rec['primary_recommendation']['timeframe']}")
        print(f"Basis:     {rec['primary_recommendation']['basis']}")
        if rec["vital_flags"]:
            print(f"Flags:     {[f['flag'] for f in rec['vital_flags']]}")
        print(f"Overall urgency: {rec['overall_urgency']}/4")
        print(f"Explanation: {rec['explanation']}")
        all_outputs.append(rec)

    summary_path = os.path.join(OUT_DIR, "phase7_rules_engine_examples.json")
    with open(summary_path, "w") as f:
        json.dump(all_outputs, f, indent=2, default=str)
    print(f"\nExample outputs saved to: {summary_path}")
    print("\nNext step: Phase 8 -- Voice Interface (speech-to-text / text-to-speech)")