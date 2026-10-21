"""
report.py -- printable PDF report for one MaternaCare AI assessment
=============================================================================
build_pdf(data) -> bytes   (an A4 PDF, ready to download and print)

`data` is a plain dict (see build_report_data() in streamlit_app.py for how the app fills it):

    generated_at, patient_label, health_worker      -> strings (the last two may be empty)
    vitals            {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}
    hemoglobin        float or None
    previous_visits   [{SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}, ...]  (oldest first)
    risk_level, probabilities {"Low Risk": 0.2, ...}
    trajectory        {"pattern", "how", "confidence" (or None), "probabilities" (or None)}
    urgency           {"level": 0-4, "headline": str}
    recommendation    the rules engine's output dict
    shap              [(feature, shap_value, actual_value), ...] or None

Needs reportlab (pip install reportlab). Uses only the built-in Helvetica font, so text is
cleaned to plain Latin characters first.
"""

import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

PINK = colors.HexColor("#C2185B")
PLUM = colors.HexColor("#4A2545")
BLUSH = colors.HexColor("#FDE7EF")
SOFT_LINE = colors.HexColor("#F0B8CC")
GREY = colors.HexColor("#6B5B66")

# urgency level -> (background, text colour)
URGENCY_COLOURS = {
    0: (colors.HexColor("#E6F4EA"), colors.HexColor("#1B5E20")),
    1: (colors.HexColor("#E6F4EA"), colors.HexColor("#1B5E20")),
    2: (colors.HexColor("#FFF4D6"), colors.HexColor("#7A4F00")),
    3: (colors.HexColor("#FDE3DC"), colors.HexColor("#A3341A")),
    4: (colors.HexColor("#FBD0D0"), colors.HexColor("#8C1515")),
}

DISCLAIMER_LINES = ("MaternaCare AI is a prototype decision-support tool built for a capstone project.",
                    "It supports, and never replaces, clinical judgement, and it is not a medical device.")

_REPLACEMENTS = {"\u2265": ">=", "\u2264": "<=", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
                 "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u2192": "->", "\u2022": "-"}


def _t(value):
    """Plain, escaped text that Helvetica can print."""
    s = "" if value is None else str(value)
    for k, v in _REPLACEMENTS.items():
        s = s.replace(k, v)
    s = s.encode("latin-1", "replace").decode("latin-1")
    return escape(s)


def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["body"] = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
                               leading=13, textColor=PLUM)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=8, leading=10.5, textColor=GREY)
    s["h2"] = ParagraphStyle("h2", parent=s["body"], fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                             textColor=PINK, spaceBefore=10, spaceAfter=4)
    s["cell"] = ParagraphStyle("cell", parent=s["body"], fontSize=9, leading=11.5)
    s["cellb"] = ParagraphStyle("cellb", parent=s["cell"], fontName="Helvetica-Bold")
    s["head"] = ParagraphStyle("head", parent=s["cell"], fontName="Helvetica-Bold", textColor=colors.white)
    s["title"] = ParagraphStyle("title", parent=s["body"], fontName="Helvetica-Bold", fontSize=18, leading=22,
                                textColor=colors.white)
    s["subtitle"] = ParagraphStyle("subtitle", parent=s["body"], fontSize=10, leading=13, textColor=colors.white)
    s["banner"] = ParagraphStyle("banner", parent=s["body"], fontSize=10.5, leading=14.5)
    s["foot"] = ParagraphStyle("foot", parent=s["small"], alignment=TA_CENTER)
    return s


def _grid(data, widths, header=True, st=None):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, SOFT_LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), PINK)]
        for r in range(2, len(data), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), BLUSH))
    else:
        for r in range(len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), BLUSH))
    t.setStyle(TableStyle(style + (st or [])))
    return t


def _flag_text(flags):
    return "; ".join(f["flag"] for f in flags) if flags else "None"


def build_pdf(d):
    S = _styles()
    W = A4[0] - 36 * mm                       # usable width
    rec = d["recommendation"]
    prim = rec["primary_recommendation"]
    traj = d["trajectory"]
    vit = d["vitals"]
    story = []

    # ---- header band -----------------------------------------------------
    head = Table([[Paragraph("MaternaCare AI", S["title"]),
                   Paragraph(_t(d["generated_at"]), ParagraphStyle("r", parent=S["subtitle"], alignment=2))],
                  [Paragraph("Maternal Risk Assessment Report", S["subtitle"]), ""]],
                 colWidths=[W * 0.62, W * 0.38])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PINK), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, 0), 9),
                              ("BOTTOMPADDING", (0, -1), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [head, Spacer(1, 6)]

    # ---- who ---------------------------------------------------------------
    who = Table([[Paragraph("<b>Patient name / ID</b>", S["cell"]), Paragraph(_t(d.get("patient_label")) or "&nbsp;", S["cell"]),
                  Paragraph("<b>Health worker</b>", S["cell"]), Paragraph(_t(d.get("health_worker")) or "&nbsp;", S["cell"])]],
                colWidths=[W * 0.20, W * 0.30, W * 0.18, W * 0.32])
    who.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, SOFT_LINE), ("BACKGROUND", (0, 0), (0, 0), BLUSH),
                             ("BACKGROUND", (2, 0), (2, 0), BLUSH), ("TOPPADDING", (0, 0), (-1, -1), 7),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
    story += [who, Spacer(1, 8)]

    # ---- urgency banner ----------------------------------------------------
    level = d["urgency"]["level"]
    bg, fg = URGENCY_COLOURS.get(level, URGENCY_COLOURS[2])
    banner_style = ParagraphStyle("b", parent=S["banner"], textColor=fg)
    drivers = [f for f in rec["vital_flags"] if f["urgency"] == rec["overall_urgency"]] \
        if rec["overall_urgency"] > prim["urgency"] else []
    text = f"<b>{_t(d['urgency']['headline'])}</b> (urgency {level} / 4)<br/>"
    if drivers:
        text += "<b>Reason:</b> " + _t(" ".join(f"{f['flag']}: {f['detail']}" for f in drivers)) + "<br/>"
        text += f"<b>Pathway from risk level and trajectory:</b> {_t(prim['action'])} ({_t(prim['timeframe'])})"
    else:
        text += f"<b>{_t(prim['action'])}</b><br/><b>When:</b> {_t(prim['timeframe'])}"
    ban = Table([[Paragraph(text, banner_style)]], colWidths=[W])
    ban.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 1.2, fg),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [ban]

    # ---- vitals ----------------------------------------------------------------
    story.append(Paragraph("Vital signs recorded at this visit", S["h2"]))
    hb = d.get("hemoglobin")
    rows = [[Paragraph(h, S["head"]) for h in ("Measurement", "Value", "Unit")],
            ["Age", f"{vit['Age']:g}", "years"],
            ["Blood pressure", f"{vit['SystolicBP']:g} / {vit['DiastolicBP']:g}", "mmHg (systolic / diastolic)"],
            ["Blood sugar", f"{vit['BS']:g}", "mmol/L"],
            ["Body temperature", f"{vit['BodyTemp']:g}", "degrees Fahrenheit"],
            ["Heart rate", f"{vit['HeartRate']:g}", "beats per minute"],
            ["Hemoglobin", f"{hb:g}" if hb is not None else "not measured", "g/dL" if hb is not None else ""]]
    rows = [rows[0]] + [[Paragraph(_t(c), S["cell"]) for c in r] for r in rows[1:]]
    story.append(_grid(rows, [W * 0.30, W * 0.25, W * 0.45]))

    prev = d.get("previous_visits") or []
    if prev:
        story.append(Paragraph("Previous visits used for the trajectory (oldest first)", S["h2"]))
        pr = [[Paragraph(h, S["head"]) for h in ("Visit", "BP (mmHg)", "Blood sugar", "Temp (F)", "Heart rate")]]
        for i, v in enumerate(prev, 1):
            pr.append([Paragraph(_t(c), S["cell"]) for c in
                       (f"Previous {i}", f"{v['SystolicBP']:g} / {v['DiastolicBP']:g}", f"{v['BS']:g}",
                        f"{v['BodyTemp']:g}", f"{v['HeartRate']:g}")])
        story.append(_grid(pr, [W * 0.2, W * 0.2, W * 0.2, W * 0.2, W * 0.2]))

    # ---- assessment -----------------------------------------------------------
    story.append(Paragraph("Assessment", S["h2"]))
    probs = d["model_probabilities"]
    ptxt = ", ".join(f"{k} {v:.0%}" for k, v in probs.items())
    conf = traj.get("confidence")
    ttxt = f"<b>{_t(traj['pattern'])}</b> ({_t(traj['how'])})"
    if conf is not None:
        ttxt += f"; pattern confidence {conf:.0%}"
    ar = [[Paragraph("<b>Predicted risk level</b>", S["cell"]), Paragraph(f"<b>{_t(d['risk_level'])}</b>  ({_t(ptxt)})", S["cell"])],
          [Paragraph("<b>Trajectory across visits</b>", S["cell"]), Paragraph(ttxt, S["cell"])],
          [Paragraph("<b>Vital-sign flags</b>", S["cell"]), Paragraph(_t(_flag_text(rec["vital_flags"])), S["cell"])],
          [Paragraph("<b>Overall urgency</b>", S["cell"]), Paragraph(f"{level} / 4", S["cell"])]]
    story.append(_grid(ar, [W * 0.30, W * 0.70], header=False))

    # ---- recommended action ----------------------------------------------------
    block = [Paragraph("Recommended action", S["h2"]),
             Paragraph(f"<b>{_t(prim['action'])}</b>", S["body"]),
             Paragraph(f"<b>When:</b> {_t(prim['timeframe'])}", S["body"]),
             Paragraph(f"Basis: {_t(prim['basis'])}", S["small"])]
    story.append(KeepTogether(block))

    if rec["vital_flags"]:
        fl = [Paragraph("Vital-sign flags in detail", S["h2"])]
        for f in rec["vital_flags"]:
            fl.append(Paragraph(f"- <b>{_t(f['flag'])}:</b> {_t(f['detail'])}", S["body"]))
            fl.append(Paragraph(f"Basis: {_t(f['basis'])}", S["small"]))
        story.append(KeepTogether(fl))

    if "anemia_recommendation" in rec:
        a = rec["anemia_recommendation"]
        story.append(KeepTogether([Paragraph(f"Anemia check ({_t(a['tier'])})", S["h2"]),
                                   Paragraph(_t(a["action"]), S["body"]),
                                   Paragraph(f"Basis: {_t(a['basis'])}", S["small"])]))

    if d.get("shap"):
        sh = [Paragraph("Main factors behind the risk estimate (SHAP)", S["h2"])]
        for feat, sv, val in d["shap"]:
            word = "raised" if sv > 0 else "lowered"
            sh.append(Paragraph(f"- <b>{_t(feat)} = {val:g}</b> {word} the chance of {_t(d['risk_level'])} ({sv:+.2f})", S["body"]))
        sh.append(Paragraph("SHAP shows how much each vital pushed the model towards this risk level.", S["small"]))
        story.append(KeepTogether(sh))

    story.append(KeepTogether([Paragraph("Summary", S["h2"]), Paragraph(_t(rec["explanation"]), S["body"])]))

    # ---- sign-off ---------------------------------------------------------------
    story.append(Spacer(1, 14))
    sign = Table([[Paragraph("Reviewed by:", S["cell"]), "", Paragraph("Signature:", S["cell"]), "",
                   Paragraph("Date:", S["cell"]), ""]],
                 colWidths=[W * 0.14, W * 0.22, W * 0.14, W * 0.22, W * 0.10, W * 0.18], rowHeights=[22])
    sign.setStyle(TableStyle([("LINEBELOW", (1, 0), (1, 0), 0.6, PLUM), ("LINEBELOW", (3, 0), (3, 0), 0.6, PLUM),
                              ("LINEBELOW", (5, 0), (5, 0), 0.6, PLUM), ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    story.append(KeepTogether([sign]))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(SOFT_LINE)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawCentredString(A4[0] / 2, 11.5 * mm, DISCLAIMER_LINES[0])
        canvas.drawCentredString(A4[0] / 2, 8 * mm, DISCLAIMER_LINES[1])
        canvas.drawRightString(A4[0] - 18 * mm, 8 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                      bottomMargin=22 * mm, title="MaternaCare AI - Maternal Risk Assessment Report",
                      author="MaternaCare AI").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
