"""
SHIPGUARD AI - Automated Inspection Report Generator
Generates comprehensive, professional maritime PDF inspection reports using ReportLab.
Includes embedded annotated image, defect breakdown, severity ranking, risk score, and statutory disclaimers.
"""

import os
import io
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)


# Navy Maritime Palette for PDF
COLOR_PRIMARY_NAVY = colors.HexColor("#0a192f")
COLOR_SECONDARY_BLUE = colors.HexColor("#1e3a8a")
COLOR_CYAN_ACCENT = colors.HexColor("#0284c7")
COLOR_LIGHT_BG = colors.HexColor("#f8fafc")
COLOR_BORDER = colors.HexColor("#cbd5e1")
COLOR_TEXT_DARK = colors.HexColor("#0f172a")
COLOR_MUTED = colors.HexColor("#64748b")


def generate_pdf_report(
    inspection_data: Dict[str, Any],
    annotated_image_rgb: Optional[np.ndarray] = None,
    output_path: Optional[str] = None
) -> bytes:
    """
    Builds a downloadable PDF report for a completed ship inspection.

    Args:
        inspection_data: Dictionary containing detections, severity, risk, and metadata
        annotated_image_rgb: NumPy array of the annotated inspection image
        output_path: Optional file path to save PDF directly to disk

    Returns:
        PDF document content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=COLOR_PRIMARY_NAVY
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=COLOR_CYAN_ACCENT
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=COLOR_PRIMARY_NAVY
    )
    body_text = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=COLOR_TEXT_DARK
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=COLOR_MUTED
    )

    story = []

    # 1. Header Banner Table
    timestamp_str = inspection_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    header_data = [
        [
            Paragraph("<b>SHIPGUARD AI</b><br/><font size=8 color='#0284c7'>INDIGENOUS CONTACTLESS SHIP INSPECTION SYSTEM</font>", title_style),
            Paragraph(f"<b>INSPECTION REPORT</b><br/>Generated: {timestamp_str}<br/>Status: <b>RECORDED</b>", ParagraphStyle('RightMeta', parent=body_text, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_CYAN_ACCENT, spaceBefore=4, spaceAfter=10))

    # 2. Key Inspection Executive Metrics Table
    total_defects = inspection_data.get("total_defects", len(inspection_data.get("detections", [])))
    severity = inspection_data.get("overall_severity", "N/A")
    risk_score = inspection_data.get("risk_score", 0)
    recommended_action = inspection_data.get("recommended_action", "Routine monitoring")
    highest_conf = inspection_data.get("highest_confidence", 0.0)
    total_area_pct = inspection_data.get("total_defect_area_pct", 0.0)

    # Color code severity badge
    sev_bg = colors.HexColor("#e63946") if severity == "CRITICAL" else \
             colors.HexColor("#fb8500") if severity == "HIGH" else \
             colors.HexColor("#ffb703") if severity == "MEDIUM" else colors.HexColor("#2ec4b6")

    summary_cards = [
        [
            Paragraph(f"<b>TOTAL DEFECTS</b><br/><font size=16 color='#0a192f'><b>{total_defects}</b></font>", body_text),
            Paragraph(f"<b>SEVERITY</b><br/><font size=14 color='{sev_bg.hexval()}'><b>{severity}</b></font>", body_text),
            Paragraph(f"<b>RISK SCORE</b><br/><font size=16 color='#0a192f'><b>{risk_score}/100</b></font>", body_text),
            Paragraph(f"<b>PEAK CONFIDENCE</b><br/><font size=14 color='#0a192f'><b>{highest_conf*100:.1f}%</b></font>", body_text)
        ]
    ]
    summary_table = Table(summary_cards, colWidths=[135, 135, 135, 135])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # 3. Recommended Action Callout Box
    action_box = [
        [
            Paragraph(f"<b>RECOMMENDED ACTION:</b> {recommended_action.upper()}<br/>"
                      f"<font size=8 color='#475569'>Cumulative defect area: {total_area_pct:.1f}% of inspected frame | "
                      f"Action Level: {severity}</font>", body_text)
        ]
    ]
    action_table = Table(action_box, colWidths=[540])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_CYAN_ACCENT),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(action_table)
    story.append(Spacer(1, 12))

    # 4. Embedded Annotated Image (if provided)
    if annotated_image_rgb is not None:
        story.append(Paragraph("<b>Annotated Inspection Visualization</b>", section_heading))
        story.append(Spacer(1, 4))
        
        # Save annotated image into in-memory buffer for ReportLab
        pil_img = Image.fromarray(annotated_image_rgb)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)

        # Scale image neatly to fit within 540 pt width maintaining aspect ratio
        orig_w, orig_h = pil_img.size
        target_w = 480.0
        target_h = min(220.0, (orig_h / orig_w) * target_w)
        
        rl_img = RLImage(img_buffer, width=target_w, height=target_h)
        story.append(rl_img)
        story.append(Spacer(1, 10))

    # 5. Localized Defects Table
    detections = inspection_data.get("detections", [])
    story.append(Paragraph("<b>Localized Structural Defect Registry</b>", section_heading))
    story.append(Spacer(1, 4))

    if detections:
        defect_rows = [["#", "Defect Class", "Confidence", "Surface Area", "Severity Level", "Bounding Box [x1,y1,x2,y2]"]]
        for i, d in enumerate(detections[:12], 1):
            bbox_str = f"[{int(d['bbox'][0])},{int(d['bbox'][1])},{int(d['bbox'][2])},{int(d['bbox'][3])}]"
            d_sev = d.get("severity", "EVALUATED")
            defect_rows.append([
                str(i),
                d.get("class_name", "defect").upper(),
                f"{d.get('confidence', 0.0)*100:.1f}%",
                f"{d.get('relative_area_pct', 0.0):.1f}%",
                d_sev,
                bbox_str
            ])
        
        defect_table = Table(defect_rows, colWidths=[24, 116, 75, 75, 90, 160])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY_NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT_BG]),
            ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(defect_table)
    else:
        story.append(Paragraph("No surface anomalies detected within configured confidence criteria.", body_text))

    story.append(Spacer(1, 10))

    # 6. Explainable Risk Factors Rationale
    reasons = inspection_data.get("explanation_reasons", [])
    if reasons:
        story.append(Paragraph("<b>Risk Calculation Breakdown (Explainable Rules)</b>", section_heading))
        story.append(Spacer(1, 4))
        for r in reasons:
            story.append(Paragraph(f"• {r}", body_text))
        story.append(Spacer(1, 8))

    # 7. Statutory Maritime Disclaimer
    story.append(KeepTogether([
        HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceBefore=4, spaceAfter=6),
        Paragraph("<b>STATUTORY NOTICE & SYSTEM SCOPE</b>", ParagraphStyle('DisclH', parent=disclaimer_style, fontName='Helvetica-Bold')),
        Paragraph(
            "SHIPGUARD AI is an indigenous computer-vision decision-support prototype developed for contactless "
            "structural screening. Detection outputs, severity grades, and prioritization scores are generated algorithmically "
            "and do not constitute statutory certification, ultrasound structural thickness measurement, or formal endorsement "
            "by an authorized classification society (e.g., DNV, Lloyd's Register, Indian Register of Shipping). "
            "Physical inspection by certified naval architects and marine surveyors is required for all high-risk findings.",
            disclaimer_style
        )
    ]))

    doc.build(story)

    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    else:
        buffer.seek(0)
        return buffer.getvalue()
