"""
report.py -- printable PDF report for one MaternaCare AI assessment
=============================================================================
build_pdf(data) -> bytes   (an A4 PDF, ready to download and print)

`data` is a plain dict (see report_data in streamlit_app.py for how the app fills it):

    generated_at, patient_label, health_worker      -> strings (the last two may be empty)
    vitals            {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}
    hemoglobin        float or None
    previous_visits   [{SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}, ...]  (oldest first)
    risk_level, model_probabilities {"Low Risk": 0.2, ...}
    trajectory        {"pattern", "how", "confidence" (or None), "probabilities" (or None)}
    urgency           {"level": 0-4, "headline": str}
    recommendation    the rules engine's output dict
    shap              [(feature, shap_value, actual_value), ...] or None

Needs reportlab (pip install reportlab). Uses only the built-in Helvetica font, so text is
cleaned to plain Latin characters first.

Colours follow the dashboard: teal theme, with green / amber / red for Low / Mid / High risk.
"""

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

# ---- dashboard palette -------------------------------------------------------
TEAL = colors.HexColor("#0F766E")        # primary
DEEP = colors.HexColor("#0B4F55")        # headings
INK = colors.HexColor("#12343B")         # body text
MUTED = colors.HexColor("#5B7479")       # notes
LINE = colors.HexColor("#CFE3E5")        # borders
TINT = colors.HexColor("#E3F2F1")        # section bars
ZEBRA = colors.HexColor("#F0F8F8")       # alternate table rows
TRACK = colors.HexColor("#E1EEEF")       # empty part of a bar

GREEN = colors.HexColor("#1F7A4D")
AMBER = colors.HexColor("#B36B00")
ORANGE = colors.HexColor("#C2410C")
RED = colors.HexColor("#B42318")
BLUE = colors.HexColor("#2F5D9B")

RISK_COLOUR = {"Low Risk": GREEN, "Mid Risk": AMBER, "High Risk": RED}
ANEMIA_COLOUR = {"Low Risk": GREEN, "Moderate Risk": AMBER, "Severe Risk": RED}
TRAJ_COLOUR = {"Stable": BLUE, "Gradually Worsening": AMBER, "Sudden Deterioration": RED, "Improving": GREEN}
URGENCY_COLOUR = {0: GREEN, 1: GREEN, 2: AMBER, 3: ORANGE, 4: RED}
CLASS_ORDER = ["Low Risk", "Mid Risk", "High Risk"]

_REPLACEMENTS = {"\u2265": ">=", "\u2264": "<=", "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
                 "\u201c": '"', "\u201d": '"', "\u00d7": "x", "\u2192": "->", "\u2022": "-"}


def _plain(value):
    s = "" if value is None else str(value)
    for k, v in _REPLACEMENTS.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _t(value):
    """Plain, escaped text that Helvetica can print."""
    return escape(_plain(value))


def _hex(c):
    return "#" + c.hexval()[2:]


def _tint(c, amount=0.12):
    """A very light version of colour c (for backgrounds)."""
    return colors.Color(1 - (1 - c.red) * amount, 1 - (1 - c.green) * amount, 1 - (1 - c.blue) * amount)


def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["body"] = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=10,
                               leading=14.5, textColor=INK)
    s["small"] = ParagraphStyle("small", parent=s["body"], fontSize=8.5, leading=11.5, textColor=MUTED)
    s["h2"] = ParagraphStyle("h2", parent=s["body"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                             textColor=DEEP)
    s["cell"] = ParagraphStyle("cell", parent=s["body"], fontSize=9.5, leading=12.5)
    s["head"] = ParagraphStyle("head", parent=s["cell"], fontName="Helvetica-Bold", textColor=colors.white)
    s["label"] = ParagraphStyle("label", parent=s["body"], fontName="Helvetica-Bold", fontSize=7.5, leading=10,
                                textColor=MUTED)
    s["big"] = ParagraphStyle("big", parent=s["body"], fontName="Helvetica-Bold", fontSize=15, leading=18)
    s["action"] = ParagraphStyle("action", parent=s["body"], fontName="Helvetica-Bold", fontSize=12.5,
                                 leading=17, textColor=DEEP)
    s["title"] = ParagraphStyle("title", parent=s["body"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                                textColor=colors.white)
    s["subtitle"] = ParagraphStyle("subtitle", parent=s["body"], fontSize=10.5, leading=14, textColor=colors.white)
    s["banner_title"] = ParagraphStyle("banner_title", parent=s["body"], fontName="Helvetica-Bold", fontSize=13.5,
                                       leading=17, textColor=colors.white)
    s["banner_level"] = ParagraphStyle("banner_level", parent=s["banner_title"], fontSize=10, alignment=TA_RIGHT)
    s["banner"] = ParagraphStyle("banner", parent=s["body"], fontSize=10.5, leading=15)
    return s


S = _styles()


# ---- building blocks -----------------------------------------------------------
def _section(title, W):
    """A shaded heading bar with a teal edge, like the dashboard panel headers."""
    t = Table([[Paragraph(_t(title), S["h2"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, 0), 4, TEAL),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [Spacer(1, 12), t, Spacer(1, 6)]


def _grid(data, widths, header=True, extra=None):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), TEAL))
        for r in range(2, len(data), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    else:
        for r in range(0, len(data), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def _bar(frac, colour, width, height=9):
    frac = min(max(frac, 0.0), 1.0)
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=TRACK, strokeColor=None, rx=height / 2, ry=height / 2))
    if frac > 0:
        d.add(Rect(0, 0, max(width * frac, height), height, fillColor=colour, strokeColor=None,
                   rx=height / 2, ry=height / 2))
    return d


def _box(flowables, colour, W, fill=None):
    """A tinted box with a thick coloured left edge."""
    t = Table([[flowables]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill or _tint(colour)),
                           ("LINEBEFORE", (0, 0), (0, 0), 4, colour), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def _flag_text(flags):
    return "; ".join(f["flag"] for f in flags) if flags else "None"


def _result_strip(cells, W):
    """cells = [(label, value, sub, colour)] -> three side-by-side result cards."""
    gap = 4 * mm
    cw = (W - 2 * gap) / 3
    row, style = [], [("VALIGN", (0, 0), (-1, -1), "TOP")]
    for i, (label, value, sub, col) in enumerate(cells):
        c = i * 2
        row.append([Paragraph(_t(label.upper()), S["label"]),
                    Paragraph(f'<font color="{_hex(col)}">{_t(value)}</font>', S["big"]),
                    Paragraph(_t(sub), S["small"])])
        if i < 2:
            row.append("")
        style += [("BACKGROUND", (c, 0), (c, 0), _tint(col)), ("LINEABOVE", (c, 0), (c, 0), 3.5, col),
                  ("BOX", (c, 0), (c, 0), 0.5, LINE), ("TOPPADDING", (c, 0), (c, 0), 7),
                  ("BOTTOMPADDING", (c, 0), (c, 0), 8), ("LEFTPADDING", (c, 0), (c, 0), 9),
                  ("RIGHTPADDING", (c, 0), (c, 0), 7)]
    t = Table([row], colWidths=[cw, gap, cw, gap, cw])
    t.setStyle(TableStyle(style))
    return t


def build_pdf(d):
    W = A4[0] - 36 * mm                       # usable width
    rec = d["recommendation"]
    prim = rec["primary_recommendation"]
    traj = d["trajectory"]
    vit = d["vitals"]
    level = d["urgency"]["level"]
    risk = d["risk_level"]
    probs = d["model_probabilities"]
    generated = _plain(d.get("generated_at") or datetime.now().strftime("%d %b %Y, %H:%M"))
    story = []

    # ---- header band --------------------------------------------------------------
    head = Table([[Paragraph("MaternaCare AI", S["title"]),
                   Paragraph(_t(generated), ParagraphStyle("r", parent=S["subtitle"], alignment=TA_RIGHT))],
                  [Paragraph("Maternal Risk Assessment Report", S["subtitle"]), ""]],
                 colWidths=[W * 0.62, W * 0.38])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TEAL), ("LEFTPADDING", (0, 0), (-1, -1), 14),
                              ("RIGHTPADDING", (0, 0), (-1, -1), 14), ("TOPPADDING", (0, 0), (-1, 0), 12),
                              ("BOTTOMPADDING", (0, -1), (-1, -1), 12), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [head, Spacer(1, 8)]

    # ---- who ---------------------------------------------------------------------
    who = Table([[Paragraph("<b>Patient name / ID</b>", S["cell"]), Paragraph(_t(d.get("patient_label")) or "&nbsp;", S["cell"]),
                  Paragraph("<b>Health worker</b>", S["cell"]), Paragraph(_t(d.get("health_worker")) or "&nbsp;", S["cell"])]],
                colWidths=[W * 0.20, W * 0.30, W * 0.18, W * 0.32])
    who.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (0, 0), TINT),
                             ("BACKGROUND", (2, 0), (2, 0), TINT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                             ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    story += [who, Spacer(1, 10)]

    # ---- urgency banner ---------------------------------------------------------------
    ucol = URGENCY_COLOUR.get(level, AMBER)
    drivers = [f for f in rec["vital_flags"] if f["urgency"] == rec["overall_urgency"]] \
        if rec["overall_urgency"] > prim["urgency"] else []
    if drivers:
        body = "<br/>".join(f"<b>{_t(f['flag'])}:</b> {_t(f['detail'])}" for f in drivers)
        body += (f"<br/><b>Pathway from risk level and trajectory:</b> {_t(prim['action'])} "
                 f"({_t(prim['timeframe'])})")
    else:
        body = f"{_t(prim['action'])}<br/><b>When:</b> {_t(prim['timeframe'])}"
    ban = Table([[Paragraph(_t(d["urgency"]["headline"]), S["banner_title"]),
                  Paragraph(f"Urgency {level} of 4", S["banner_level"])],
                 [Paragraph(body, S["banner"]), ""]], colWidths=[W * 0.70, W * 0.30])
    ban.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ucol), ("BACKGROUND", (0, 1), (-1, 1), _tint(ucol)),
                             ("SPAN", (0, 1), (1, 1)), ("BOX", (0, 0), (-1, -1), 1, ucol),
                             ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                             ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                             ("TOPPADDING", (0, 1), (-1, 1), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 10)]))
    story.append(ban)
    story.append(Spacer(1, 10))

    # ---- key results ---------------------------------------------------------------
    how = _plain(traj.get("how", ""))
    how = how[:1].upper() + how[1:]
    story.append(_result_strip([
        ("Predicted risk level", risk, f"Model probability {probs.get(risk, 0):.0%}", RISK_COLOUR.get(risk, DEEP)),
        ("Trajectory", traj["pattern"], how, TRAJ_COLOUR.get(traj["pattern"], DEEP)),
        ("Overall urgency", f"{level} of 4", d["urgency"]["headline"], ucol),
    ], W))

    # ---- model confidence + colour key ----------------------------------------------
    story += _section("Risk level and model confidence", W)
    bar_w = W * 0.50
    rows, style = [], []
    for i, cls in enumerate(CLASS_ORDER):
        col = RISK_COLOUR[cls]
        p = probs.get(cls, 0.0)
        is_pred = cls == risk
        label = f"<b>{_t(cls)}</b>" if is_pred else _t(cls)
        rows.append([Paragraph(f'<font color="{_hex(col)}">{label}</font>', S["cell"]), _bar(p, col, bar_w),
                     Paragraph(f'<font color="{_hex(col)}"><b>{p:.0%}</b></font>', S["cell"]),
                     Paragraph(f'<font color="{_hex(col)}"><b>PREDICTED</b></font>' if is_pred else "", S["small"])])
        if is_pred:
            style.append(("BACKGROUND", (0, i), (-1, i), _tint(col, 0.14)))
    conf = Table(rows, colWidths=[W * 0.17, bar_w + 8, W * 0.10, W - W * 0.27 - bar_w - 8])
    conf.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                              ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE)] + style))
    story.append(conf)
    story.append(Spacer(1, 6))
    key = Table([[Paragraph("<b>Colour key</b>", S["small"]),
                  Paragraph(f'<font color="{_hex(GREEN)}"><b>Low Risk</b></font>', S["cell"]),
                  Paragraph(f'<font color="{_hex(AMBER)}"><b>Mid Risk</b></font>', S["cell"]),
                  Paragraph(f'<font color="{_hex(RED)}"><b>High Risk</b></font>', S["cell"])]],
                colWidths=[W * 0.16, W * 0.2, W * 0.2, W * 0.2], hAlign="LEFT")
    key.setStyle(TableStyle([("BACKGROUND", (1, 0), (1, 0), _tint(GREEN)), ("BACKGROUND", (2, 0), (2, 0), _tint(AMBER)),
                             ("BACKGROUND", (3, 0), (3, 0), _tint(RED)), ("BOX", (1, 0), (1, 0), 0.8, GREEN),
                             ("BOX", (2, 0), (2, 0), 0.8, AMBER), ("BOX", (3, 0), (3, 0), 0.8, RED),
                             ("ALIGN", (1, 0), (-1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story.append(key)

    # ---- recommended action -------------------------------------------------------
    story += _section("Recommended action", W)
    pcol = URGENCY_COLOUR.get(prim["urgency"], TEAL)
    story.append(KeepTogether([_box([
        Paragraph(_t(prim["action"]), S["action"]), Spacer(1, 4),
        Paragraph(f"<b>When:</b> {_t(prim['timeframe'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                  f"<font color=\"{_hex(pcol)}\"><b>Urgency {prim['urgency']} of 4</b></font>", S["body"]),
        Spacer(1, 3), Paragraph(f"Basis: {_t(prim['basis'])}", S["small"])], pcol, W)]))

    # ---- flags ------------------------------------------------------------------------
    story += _section("Vital-sign flags", W)
    if rec["vital_flags"]:
        for f in rec["vital_flags"]:
            fc = URGENCY_COLOUR.get(f["urgency"], AMBER)
            story.append(KeepTogether([_box([
                Paragraph(f"<b>{_t(f['flag'])}</b> &nbsp;-&nbsp; <font color=\"{_hex(fc)}\"><b>Urgency {f['urgency']} of 4</b></font>", S["body"]),
                Paragraph(_t(f["detail"]), S["body"]),
                Paragraph(f"Basis: {_t(f['basis'])}", S["small"])], fc, W), Spacer(1, 5)]))
    else:
        story.append(_box([Paragraph("<b>No flags.</b> All vitals are below the rule thresholds.", S["body"])], GREEN, W))

    # ---- anemia -----------------------------------------------------------------------
    if "anemia_recommendation" in rec:
        a = rec["anemia_recommendation"]
        acol = ANEMIA_COLOUR.get(a["tier"], TEAL)
        story += _section("Anemia check", W)
        story.append(KeepTogether([_box([
            Paragraph(f"<font color=\"{_hex(acol)}\"><b>{_t(a['tier'])}</b></font>", S["action"]),
            Paragraph(_t(a["action"]), S["body"]),
            Paragraph(f"Basis: {_t(a['basis'])}", S["small"])], acol, W)]))

    # ---- vitals ----------------------------------------------------------------------
    story += _section("Vital signs recorded at this visit", W)
    hb = d.get("hemoglobin")
    vrows = [[Paragraph(h, S["head"]) for h in ("Measurement", "Value", "Unit")],
             ["Age", f"{vit['Age']:g}", "years"],
             ["Blood pressure", f"{vit['SystolicBP']:g} / {vit['DiastolicBP']:g}", "mmHg (systolic / diastolic)"],
             ["Blood sugar", f"{vit['BS']:g}", "mmol/L"],
             ["Body temperature", f"{vit['BodyTemp']:g}", "degrees Fahrenheit"],
             ["Heart rate", f"{vit['HeartRate']:g}", "beats per minute"],
             ["Hemoglobin", f"{hb:g}" if hb is not None else "not measured", "g/dL" if hb is not None else ""]]
    vrows = [vrows[0]] + [[Paragraph(f"<b>{_t(r[0])}</b>", S["cell"]), Paragraph(_t(r[1]), S["cell"]),
                           Paragraph(_t(r[2]), S["cell"])] for r in vrows[1:]]
    story.append(_grid(vrows, [W * 0.30, W * 0.25, W * 0.45]))

    prev = d.get("previous_visits") or []
    if prev:
        story += _section("Previous visits used for the trajectory (oldest first)", W)
        pr = [[Paragraph(h, S["head"]) for h in ("Visit", "BP (mmHg)", "Blood sugar", "Temp (F)", "Heart rate")]]
        for i, v in enumerate(prev, 1):
            pr.append([Paragraph(_t(c), S["cell"]) for c in
                       (f"Previous {i}", f"{v['SystolicBP']:g} / {v['DiastolicBP']:g}", f"{v['BS']:g}",
                        f"{v['BodyTemp']:g}", f"{v['HeartRate']:g}")])
        story.append(_grid(pr, [W * 0.2] * 5))

    # ---- trajectory ----------------------------------------------------------------------
    story += _section("Trajectory across visits", W)
    tcol = TRAJ_COLOUR.get(traj["pattern"], DEEP)
    ttxt = f"<font color=\"{_hex(tcol)}\"><b>{_t(traj['pattern'])}</b></font> ({_t(traj['how'])})"
    if traj.get("confidence") is not None:
        ttxt += f"<br/>Pattern confidence: {traj['confidence']:.0%}"
    story.append(_box([Paragraph(ttxt, S["body"])], tcol, W))

    # ---- SHAP ------------------------------------------------------------------------------
    if d.get("shap"):
        story += _section("Main factors behind the risk estimate (SHAP)", W)
        mx = max(abs(sv) for _, sv, _ in d["shap"]) or 1.0
        sh = [[Paragraph("Vital", S["head"]), Paragraph("Value", S["head"]),
               Paragraph(f"Effect on {_t(risk)}", S["head"]), Paragraph("Contribution", S["head"]), ""]]
        for feat, sv, val in d["shap"]:
            ec = ORANGE if sv > 0 else TEAL
            sh.append([Paragraph(f"<b>{_t(feat)}</b>", S["cell"]), Paragraph(f"{val:g}", S["cell"]),
                       Paragraph(f'<font color="{_hex(ec)}"><b>{"Raises" if sv > 0 else "Lowers"}</b></font>', S["cell"]),
                       _bar(abs(sv) / mx, ec, W * 0.2), Paragraph(f"{sv:+.2f}", S["cell"])])
        story.append(KeepTogether([_grid(sh, [W * 0.27, W * 0.15, W * 0.22, W * 0.26, W * 0.10],
                                         extra=[("SPAN", (3, 0), (4, 0))])]))

    # ---- summary ---------------------------------------------------------------------------
    story += _section("Summary", W)
    story.append(KeepTogether([_box([Paragraph(_t(rec["explanation"]), S["body"])], TEAL, W)]))

    # ---- sign-off ----------------------------------------------------------------------------
    story.append(Spacer(1, 22))
    sign = Table([[Paragraph("Reviewed by:", S["cell"]), "", Paragraph("Signature:", S["cell"]), "",
                   Paragraph("Date:", S["cell"]), ""]],
                 colWidths=[W * 0.14, W * 0.22, W * 0.14, W * 0.22, W * 0.10, W * 0.18], rowHeights=[22])
    sign.setStyle(TableStyle([("LINEBELOW", (1, 0), (1, 0), 0.6, INK), ("LINEBELOW", (3, 0), (3, 0), 0.6, INK),
                              ("LINEBELOW", (5, 0), (5, 0), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    story.append(KeepTogether([sign]))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10.5 * mm, f"Report generated on {generated}")
        canvas.drawRightString(A4[0] - 18 * mm, 10.5 * mm, f"Page {doc.page}")
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                      bottomMargin=22 * mm, title="MaternaCare AI - Maternal Risk Assessment Report",
                      author="MaternaCare AI").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()