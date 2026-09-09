"""
SHIPGUARD AI - Explainable Severity Assessment Engine
Provides transparent, rule-based severity grading for marine structural visual defects.
Strictly labeled as an 'AI-assisted inspection prioritization score' (not a certified marine survey).
"""

from typing import List, Dict, Any


# Standard Severity Levels
SEVERITY_LOW = "LOW"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

SEVERITY_RANKS = {
    SEVERITY_LOW: 1,
    SEVERITY_MEDIUM: 2,
    SEVERITY_HIGH: 3,
    SEVERITY_CRITICAL: 4
}

SEVERITY_COLORS = {
    SEVERITY_LOW: "#2ec4b6",       # Teal / Green
    SEVERITY_MEDIUM: "#ffb703",    # Amber Yellow
    SEVERITY_HIGH: "#fb8500",      # Deep Orange
    SEVERITY_CRITICAL: "#e63946"   # Crimson Red
}


class SeverityEngine:
    """
    Transparent, explainable rule-based severity evaluator.
    Examines defect class, confidence level, and relative surface area.
    """

    LABEL = "AI-assisted inspection prioritization score"

    @classmethod
    def evaluate_defect(cls, defect: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assesses severity for a single localized defect.

        Factors considered:
        1. Inherent defect hazard class (e.g. severe-corrosion, crack vs mild-corrosion)
        2. Defect relative area (defect footprint relative to total frame)
        3. Model detection confidence
        """
        cls_name = defect.get("class_name", "").lower()
        conf = defect.get("confidence", 0.0)
        rel_area_pct = defect.get("relative_area_pct", 0.0)

        reasons: List[str] = []
        severity = SEVERITY_LOW

        # Rule Set 1: Class-based baseline hazard
        is_crack = "crack" in cls_name
        is_severe_corrosion = "severe-corrosion" in cls_name or "corroded-part" in cls_name
        is_moderate_corrosion = "moderate-corrosion" in cls_name or "iron rust" in cls_name or "corrosion" in cls_name
        is_mild = "mild-corrosion" in cls_name or "rust" in cls_name

        if is_crack:
            if rel_area_pct > 10.0 or conf > 0.65:
                severity = SEVERITY_CRITICAL
                reasons.append(f"Structural fatigue crack spanning {rel_area_pct:.1f}% area with high confidence ({conf*100:.0f}%)")
            elif rel_area_pct > 3.0 or conf > 0.35:
                severity = SEVERITY_HIGH
                reasons.append(f"Confirmed surface fracture/crack ({conf*100:.0f}% confidence)")
            else:
                severity = SEVERITY_MEDIUM
                reasons.append(f"Detected minor crack fissure ({conf*100:.0f}% confidence)")

        elif is_severe_corrosion:
            if rel_area_pct > 15.0:
                severity = SEVERITY_CRITICAL
                reasons.append(f"Extensive severe corrosion covering {rel_area_pct:.1f}% of hull surface")
            elif rel_area_pct > 5.0 or conf > 0.50:
                severity = SEVERITY_HIGH
                reasons.append(f"Significant localized severe corrosion ({conf*100:.0f}% confidence)")
            else:
                severity = SEVERITY_MEDIUM
                reasons.append(f"Localized deep corrosion patch ({conf*100:.0f}% confidence)")

        elif is_moderate_corrosion:
            if rel_area_pct > 20.0:
                severity = SEVERITY_HIGH
                reasons.append(f"Large surface corrosion zone ({rel_area_pct:.1f}% of frame)")
            elif rel_area_pct > 5.0:
                severity = SEVERITY_MEDIUM
                reasons.append(f"Moderate oxidation footprint ({rel_area_pct:.1f}% area)")
            else:
                severity = SEVERITY_LOW
                reasons.append("Small localized corrosion spot")

        elif is_mild:
            if rel_area_pct > 25.0:
                severity = SEVERITY_MEDIUM
                reasons.append(f"Broad superficial rust coverage ({rel_area_pct:.1f}% area)")
            else:
                severity = SEVERITY_LOW
                reasons.append("Superficial surface patina / mild oxidation")
        else:
            # Default fallback for unclassified marine defects
            if rel_area_pct > 15.0:
                severity = SEVERITY_MEDIUM
                reasons.append(f"Unclassified visual anomaly with significant area ({rel_area_pct:.1f}%)")
            else:
                severity = SEVERITY_LOW
                reasons.append("Low-grade surface visual irregularity")

        # Confidence modifier
        if conf < 0.25 and severity in {SEVERITY_HIGH, SEVERITY_CRITICAL}:
            # Downgrade one level if detection confidence is borderline to prevent false alarms
            previous = severity
            severity = SEVERITY_MEDIUM if severity == SEVERITY_HIGH else SEVERITY_HIGH
            reasons.append(f"Downgraded from {previous} due to exploratory confidence ({conf*100:.1f}%)")

        reason_text = "; ".join(reasons) if reasons else "Standard structural defect evaluation."

        return {
            "severity": severity,
            "rank": SEVERITY_RANKS[severity],
            "color": SEVERITY_COLORS[severity],
            "reason": reason_text,
            "label": cls.LABEL
        }

    @classmethod
    def evaluate_inspection(cls, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates the aggregate overall severity for an entire ship inspection session.

        Aggregates individual defect severity levels, accounting for defect density
        and compounding multiple-defect risks.
        """
        if not detections:
            return {
                "overall_severity": "CLEAN",
                "rank": 0,
                "color": "#06d6a0",
                "summary_reason": "No structural defects detected within detection confidence limits.",
                "severity_counts": {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 0, SEVERITY_HIGH: 0, SEVERITY_CRITICAL: 0},
                "evaluated_defects": [],
                "label": cls.LABEL
            }

        evaluated = []
        severity_counts = {
            SEVERITY_LOW: 0,
            SEVERITY_MEDIUM: 0,
            SEVERITY_HIGH: 0,
            SEVERITY_CRITICAL: 0
        }

        for d in detections:
            eval_res = cls.evaluate_defect(d)
            merged = {**d, **eval_res}
            evaluated.append(merged)
            severity_counts[eval_res["severity"]] += 1

        # Aggregation Logic
        reasons: List[str] = []
        if severity_counts[SEVERITY_CRITICAL] >= 1:
            overall = SEVERITY_CRITICAL
            reasons.append(f"{severity_counts[SEVERITY_CRITICAL]} critical structural defect(s) localized")
        elif severity_counts[SEVERITY_HIGH] >= 2:
            overall = SEVERITY_CRITICAL
            reasons.append(f"Compounding risk: {severity_counts[SEVERITY_HIGH]} high-severity defects localized in proximity")
        elif severity_counts[SEVERITY_HIGH] == 1:
            overall = SEVERITY_HIGH
            reasons.append("1 high-severity defect identified requiring priority attention")
        elif severity_counts[SEVERITY_MEDIUM] >= 3:
            overall = SEVERITY_HIGH
            reasons.append(f"Compounding risk: Cluster of {severity_counts[SEVERITY_MEDIUM]} moderate defects detected")
        elif severity_counts[SEVERITY_MEDIUM] >= 1:
            overall = SEVERITY_MEDIUM
            reasons.append(f"{severity_counts[SEVERITY_MEDIUM]} moderate defect(s) detected")
        else:
            overall = SEVERITY_LOW
            reasons.append("Only low-grade superficial anomalies detected")

        return {
            "overall_severity": overall,
            "rank": SEVERITY_RANKS[overall],
            "color": SEVERITY_COLORS[overall],
            "summary_reason": "; ".join(reasons),
            "severity_counts": severity_counts,
            "evaluated_defects": evaluated,
            "label": cls.LABEL
        }
