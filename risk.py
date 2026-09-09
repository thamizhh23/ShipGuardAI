"""
SHIPGUARD AI - Risk Assessment & Maintenance Recommendation Engine
Computes an explainable 0-100 inspection prioritization risk score based strictly on actual model detections.
Strictly labeled as 'Inspection Prioritization Risk' (NOT a certified structural failure prediction).
"""

from typing import List, Dict, Any, Tuple
from src.severity import SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL


# Recommended Maritime Actions
ACTION_ROUTINE = "Routine monitoring"
ACTION_SCHEDULED = "Schedule detailed inspection"
ACTION_PRIORITY = "Priority inspection recommended"
ACTION_IMMEDIATE = "Immediate expert inspection recommended"

ACTION_DETAILS = {
    ACTION_ROUTINE: {
        "timeframe": "Next scheduled dry-dock / periodic maintenance window",
        "scope": "Surface visual re-check and routine anti-corrosive coating washdown.",
        "icon": "🟢"
    },
    ACTION_SCHEDULED: {
        "timeframe": "Within 30 calendar days",
        "scope": "Ultrasonic thickness measurement (UTM) or manual dye-penetrant inspection of highlighted zones.",
        "icon": "🟡"
    },
    ACTION_PRIORITY: {
        "timeframe": "Within 7 calendar days or prior to next offshore voyage",
        "scope": "Targeted structural NDT (non-destructive testing), pitting depth evaluation, and surface prep/recoating.",
        "icon": "🟠"
    },
    ACTION_IMMEDIATE: {
        "timeframe": "Immediate / Halt hull loading operations if structural boundary affected",
        "scope": "Comprehensive certified marine surveyor inspection for potential structural compromise or hull fracture.",
        "icon": "🔴"
    }
}

DISCLAIMER_TEXT = (
    "DISCLAIMER: SHIPGUARD AI is an indigenous AI-assisted decision-support system designed to prioritize "
    "contactless visual inspection workflows. The generated Risk Score and Recommended Actions do not substitute "
    "for statutory survey, classification society certification, or qualified marine engineering clearance."
)


class RiskEngine:
    """
    Computes a transparent 0-100 Inspection Prioritization Risk score.
    Decomposes the score into contributing mathematical components.
    """

    LABEL = "Inspection Prioritization Risk"

    @classmethod
    def calculate_risk(
        cls,
        detections: List[Dict[str, Any]],
        overall_severity: str,
        total_defect_area_pct: float
    ) -> Dict[str, Any]:
        """
        Calculates the risk score from 0 to 100 with full explainability.

        Formula components:
        1. Defect Quantity Score (0 - 25 pts)
        2. Defect Confidence Weight (0 - 20 pts)
        3. Cumulative Surface Defect Area (0 - 25 pts)
        4. Defect Severity & Class Hazard (0 - 30 pts)
        """
        if not detections:
            return {
                "score": 0,
                "severity": "CLEAN",
                "recommended_action": ACTION_ROUTINE,
                "action_details": ACTION_DETAILS[ACTION_ROUTINE],
                "score_breakdown": {
                    "defect_count_contribution": 0.0,
                    "confidence_contribution": 0.0,
                    "area_contribution": 0.0,
                    "severity_contribution": 0.0
                },
                "explanation_reasons": ["No defects detected in this inspection segment."],
                "label": cls.LABEL,
                "disclaimer": DISCLAIMER_TEXT
            }

        reasons: List[str] = []
        n_defects = len(detections)

        # 1. Defect Count Contribution (0 to 25 pts)
        # Scaled up to 6 defects for max count score
        count_score = min(25.0, (n_defects / 6.0) * 25.0)
        reasons.append(f"{n_defects} defect(s) detected (+{count_score:.1f} pts)")

        # 2. Confidence Contribution (0 to 20 pts)
        # Average confidence of detected defects
        avg_conf = sum(d.get("confidence", 0.0) for d in detections) / n_defects
        conf_score = avg_conf * 20.0
        reasons.append(f"Average model detection confidence of {avg_conf*100:.1f}% (+{conf_score:.1f} pts)")

        # 3. Surface Area Contribution (0 to 25 pts)
        # 30% area or more yields full 25 pts
        area_score = min(25.0, (total_defect_area_pct / 30.0) * 25.0)
        reasons.append(f"Cumulative defect area covers {total_defect_area_pct:.1f}% of frame (+{area_score:.1f} pts)")

        # 4. Severity & Defect Hazard Contribution (0 to 30 pts)
        sev_base_points = {
            SEVERITY_LOW: 8.0,
            SEVERITY_MEDIUM: 16.0,
            SEVERITY_HIGH: 24.0,
            SEVERITY_CRITICAL: 30.0
        }
        severity_score = sev_base_points.get(overall_severity, 8.0)
        reasons.append(f"Overall severity graded as {overall_severity} (+{severity_score:.1f} pts)")

        # Compounding multiplier for critical defect co-occurrence
        has_critical = any(d.get("severity") == SEVERITY_CRITICAL for d in detections)
        has_crack = any("crack" in d.get("class_name", "").lower() for d in detections)
        compounding_bonus = 0.0
        if has_critical and has_crack:
            compounding_bonus = 5.0
            reasons.append("Compounding hazard alert: Co-occurrence of crack fractures with severe degradation (+5.0 pts)")

        total_raw = count_score + conf_score + area_score + severity_score + compounding_bonus
        final_score = int(round(min(100.0, max(0.0, total_raw))))

        # Determine Recommended Action based on Score Thresholds
        if final_score >= 80 or overall_severity == SEVERITY_CRITICAL:
            recommended_action = ACTION_IMMEDIATE
        elif final_score >= 55 or overall_severity == SEVERITY_HIGH:
            recommended_action = ACTION_PRIORITY
        elif final_score >= 30 or overall_severity == SEVERITY_MEDIUM:
            recommended_action = ACTION_SCHEDULED
        else:
            recommended_action = ACTION_ROUTINE

        return {
            "score": final_score,
            "overall_severity": overall_severity,
            "recommended_action": recommended_action,
            "action_details": ACTION_DETAILS[recommended_action],
            "score_breakdown": {
                "defect_count_contribution": round(count_score, 1),
                "confidence_contribution": round(conf_score, 1),
                "area_contribution": round(area_score, 1),
                "severity_contribution": round(severity_score, 1)
            },
            "explanation_reasons": reasons,
            "label": cls.LABEL,
            "disclaimer": DISCLAIMER_TEXT
        }
