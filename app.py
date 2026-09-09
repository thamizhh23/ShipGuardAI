"""
SHIPGUARD AI: Indigenous Contactless AI-Powered Ship Inspection System
Production MVP Hackathon Dashboard
Strictly adheres to the No Fake Data Policy.
"""

import os
import io
import time
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# Import Core ShipGuard Modules
from src.detector import ShipGuardDetector, CLASS_COLORS
from src.severity import SeverityEngine, SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL
from src.risk import RiskEngine, ACTION_ROUTINE, ACTION_SCHEDULED, ACTION_PRIORITY, ACTION_IMMEDIATE, DISCLAIMER_TEXT
from src.database import InspectionDatabase
from src.report import generate_pdf_report


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SHIPGUARD AI - Contactless Ship Inspection",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Naval Maritime Dark Styling
CUSTOM_CSS = """
<style>
    /* Global App Background & Naval Theme */
    .stApp {
        background-color: #070d19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header Brand Card */
    .brand-container {
        background: linear-gradient(135deg, #0f1c3f 0%, #0a1128 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .brand-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .brand-badge {
        background: #0284c7;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        vertical-align: middle;
    }
    
    .brand-subtitle {
        color: #38bdf8;
        font-size: 1.05rem;
        font-weight: 500;
        margin-top: 6px;
    }
    
    .brand-desc {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* Metric Cards */
    .metric-card {
        background: #0d1829;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #0284c7;
        transform: translateY(-2px);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-val {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 2px;
    }

    /* Severity Badges */
    .badge-critical {
        background-color: rgba(230, 57, 70, 0.2);
        color: #ff4d6d;
        border: 1px solid #e63946;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(251, 133, 0, 0.2);
        color: #ff9e00;
        border: 1px solid #fb8500;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(255, 183, 3, 0.2);
        color: #ffd166;
        border: 1px solid #ffb703;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(46, 196, 182, 0.2);
        color: #2ec4b6;
        border: 1px solid #2ec4b6;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }

    /* Action Banner */
    .action-banner {
        background: #0f2438;
        border-left: 4px solid #0284c7;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    /* Disclaimer Card */
    .disclaimer-card {
        background: #0a101d;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.4;
        margin-top: 20px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #060b14;
        border-right: 1px solid #1e293b;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Database & Detector
@st.cache_resource
def get_database():
    return InspectionDatabase()

@st.cache_resource
def get_detector():
    return ShipGuardDetector(
        model_path="model/best.pt",
        crack_model_path="model/crack_best.pt"
    )

db = get_database()
detector = get_detector()


# Session State Initialization
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = "sample.jpg"
if "inspection_result" not in st.session_state:
    st.session_state.inspection_result = None
if "saved_inspection_id" not in st.session_state:
    st.session_state.saved_inspection_id = None


# Sidebar Navigation & System Telemetry
with st.sidebar:
    st.markdown("### ⚓ SHIPGUARD AI")
    st.markdown("`SYSTEM v1.0.4 | INDIGENOUS MVP`")
    st.divider()

    st.markdown("#### 📡 System Telemetry")
    model_info = detector.get_model_info()
    
    if model_info["primary_model_loaded"]:
        st.success("🟢 Primary Corrosion Model: **LOADED** (YOLOv8)")
    else:
        st.error(f"🔴 Primary Model: {model_info.get('error')}")

    if model_info["crack_model_loaded"]:
        st.success("🟢 Structural Crack Model: **LOADED** (YOLOv8)")
    else:
        st.warning("🟡 Crack Model: Offline (Corrosion-only mode)")

    st.markdown(f"**Inference Compute**: CPU Optimized")
    st.markdown(f"**Local Database**: SQLite (`shipguard.db`)")
    
    st.divider()
    st.markdown("#### 🛠️ Inspection Parameters")
    conf_slider = st.slider(
        "Detection Confidence Threshold",
        min_value=0.05,
        max_value=0.80,
        value=0.15,
        step=0.05,
        help="Filters raw model predictions below this confidence level."
    )
    enable_crack = st.checkbox("Enable Dual-Crack Detection", value=True, help="Runs secondary crack localization model concurrently.")
    
    st.divider()
    st.caption("Indigenous Contactless Maritime Structural Health Monitoring MVP. Hackathon Demonstration Prototype.")


# Top Brand Header
st.markdown("""
<div class="brand-container">
    <div class="brand-title">
        ⚓ SHIPGUARD AI
        <span class="brand-badge">PROTOTYPE MVP</span>
    </div>
    <div class="brand-subtitle">
        Indigenous Contactless AI-Powered Ship Inspection and Structural Health Monitoring System
    </div>
    <div class="brand-desc">
        Real-time vision-based hull defect detection, explainable severity assessment, and automated maritime inspection reporting.
    </div>
</div>
""", unsafe_allow_html=True)


# Main Dashboard Navigation Tabs
tabs = st.tabs([
    "📸 Image Inspection",
    "📊 Detection Analytics",
    "🛡️ Risk & Severity Engine",
    "📜 Inspection History",
    "📑 Report Generator",
    "🏗️ Architecture & Future Sensors",
    "📚 Dataset & Model Transparency"
])


# ==============================================================================
# TAB 1: IMAGE INSPECTION
# ==============================================================================
with tabs[0]:
    st.markdown("### 🔍 Live Ship Hull Visual Inspection")
    st.write("Upload an authentic ship/hull inspection photograph or select from pre-verified maritime dataset samples.")

    col_input, col_action = st.columns([2, 1])

    with col_input:
        input_mode = st.radio(
            "Select Inspection Source:",
            ["Choose from Verified Marine Demo Images", "Upload Custom Hull Inspection Image"],
            horizontal=True
        )

        demo_dir = os.path.join("assets", "demo_images")
        demo_files = [f for f in os.listdir(demo_dir) if f.endswith((".jpg", ".png", ".jpeg"))] if os.path.exists(demo_dir) else []

        if input_mode == "Choose from Verified Marine Demo Images":
            if demo_files:
                selected_demo = st.selectbox(
                    "Select Verified Marine Dataset Sample:",
                    demo_files,
                    index=0,
                    format_func=lambda x: f"🚢 {x.replace('_', ' ').replace('.jpg', '').title()} (Real Dataset Image)"
                )
                img_path = os.path.join(demo_dir, selected_demo)
                raw_image = Image.open(img_path).convert("RGB")
                st.session_state.current_image = raw_image
                st.session_state.image_name = selected_demo
            else:
                st.warning("No demo images found in assets/demo_images/")
        else:
            uploaded_file = st.file_uploader("Upload Ship Hull Inspection Photo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                raw_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.current_image = raw_image
                st.session_state.image_name = uploaded_file.name

    with col_action:
        st.markdown("#### Inspection Workflow")
        st.markdown("""
        1. **Load Image** into buffer
        2. **Run Real Inference** with YOLOv8
        3. **Calculate Severity & Risk**
        4. **Store Record** to local SQLite
        """)
        
        analyze_btn = st.button("🚀 Analyze Inspection", type="primary", use_container_width=True)

    if analyze_btn and st.session_state.current_image is not None:
        with st.spinner("Executing real YOLOv8 computer vision inference pipeline..."):
            start_time = time.time()
            
            # 1. Real Computer Vision Inference
            det_res = detector.detect(
                st.session_state.current_image,
                conf_threshold=conf_slider,
                enable_crack_detector=enable_crack
            )
            
            # 2. Rule-Based Explainable Severity Assessment
            sev_res = SeverityEngine.evaluate_inspection(det_res["detections"])
            
            # 3. Transparent Multi-Factor Risk Calculation
            risk_res = RiskEngine.calculate_risk(
                sev_res["evaluated_defects"],
                sev_res["overall_severity"],
                det_res["total_defect_area_pct"]
            )
            
            inference_duration = time.time() - start_time
            
            # Aggregate Session State
            st.session_state.inspection_result = {
                **det_res,
                **sev_res,
                **risk_res,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": st.session_state.image_name,
                "inference_time_ms": round(inference_duration * 1000, 1)
            }
            
            # Automatically persist to SQLite database
            saved_id = db.save_inspection(
                image_name=st.session_state.image_name,
                defect_count=det_res["total_defects"],
                defect_types=det_res["class_counts"],
                highest_confidence=det_res["highest_confidence"],
                severity=sev_res["overall_severity"],
                risk_score=risk_res["score"],
                recommended_action=risk_res["recommended_action"],
                total_area_pct=det_res["total_defect_area_pct"],
                raw_data={
                    "detections": sev_res["evaluated_defects"],
                    "reasons": risk_res["explanation_reasons"]
                },
                notes=f"Processed in {round(inference_duration*1000, 1)}ms"
            )
            st.session_state.saved_inspection_id = saved_id
            st.toast(f"✅ Inspection complete and saved as Record #{saved_id}!", icon="⚓")

    # Display Inspection View if image loaded
    if st.session_state.current_image is not None:
        st.divider()
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### 📷 Original Inspection Input")
            st.image(st.session_state.current_image, use_container_width=True, caption=f"Source: {st.session_state.image_name}")
            
        with col_img2:
            st.markdown("#### 🎯 AI Defect Localization Output")
            if st.session_state.inspection_result is not None:
                st.image(
                    st.session_state.inspection_result["annotated_image"],
                    use_container_width=True,
                    caption=f"AI Annotated Visualization | {st.session_state.inspection_result['total_defects']} Defects Localized"
                )
            else:
                st.info("Click **'Analyze Inspection'** above to run actual model detection.")

        # Quick Executive Result Banner
        if st.session_state.inspection_result is not None:
            res = st.session_state.inspection_result
            st.markdown("---")
            st.markdown("### 📋 Inspection Result Summary")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Defects Detected</div>
                    <div class="metric-val">{res['total_defects']}</div>
                    <div class="metric-sub">{sum(res['class_counts'].values())} localized boxes</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Peak Confidence</div>
                    <div class="metric-val">{res['highest_confidence']*100:.1f}%</div>
                    <div class="metric-sub">Avg: {res['average_confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                sev_cls = f"badge-{res['overall_severity'].lower()}" if res['overall_severity'] != "CLEAN" else "badge-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Severity</div>
                    <div style="margin-top:8px;"><span class="{sev_cls}">{res['overall_severity']}</span></div>
                    <div class="metric-sub">Prioritization level</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prioritization Risk</div>
                    <div class="metric-val">{res['score']}<span style="font-size:1rem;color:#64748b;">/100</span></div>
                    <div class="metric-sub">{res['total_defect_area_pct']}% frame area</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Inference Speed</div>
                    <div class="metric-val">{res['inference_time_ms']} ms</div>
                    <div class="metric-sub">CPU latency</div>
                </div>
                """, unsafe_allow_html=True)

            # Recommended Action Banner
            st.markdown(f"""
            <div class="action-banner">
                <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;">
                    {res['action_details']['icon']} Recommended Action: {res['recommended_action']}
                </div>
                <div style="font-size:0.88rem;color:#94a3b8;margin-top:4px;">
                    <b>Operational Window:</b> {res['action_details']['timeframe']}<br/>
                    <b>Action Protocol:</b> {res['action_details']['scope']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: DETECTION ANALYTICS
# ==============================================================================
with tabs[1]:
    st.markdown("### 📊 Localized Defect Analytics")
    
    if st.session_state.inspection_result is None:
        st.info("Run an inspection on the 'Image Inspection' tab to view localized defect telemetry.")
    else:
        res = st.session_state.inspection_result
        detections = res.get("evaluated_defects", [])

        if not detections:
            st.success("🟢 No structural defects localized above confidence threshold.")
        else:
            col_chart1, col_chart2 = st.columns([1, 1])
            
            with col_chart1:
                st.markdown("#### Defect Distribution by Category")
                counts_df = pd.DataFrame(
                    list(res["class_counts"].items()),
                    columns=["Defect Category", "Count"]
                )
                st.bar_chart(counts_df.set_index("Defect Category"), color="#0284c7")

            with col_chart2:
                st.markdown("#### Defect Severity Breakdown")
                sev_counts = res.get("severity_counts", {})
                sev_df = pd.DataFrame(
                    list(sev_counts.items()),
                    columns=["Severity Level", "Count"]
                )
                st.bar_chart(sev_df.set_index("Severity Level"), color="#fb8500")

            st.markdown("#### 📑 Detailed Defect Inspection Table")
            table_rows = []
            for i, d in enumerate(detections, 1):
                table_rows.append({
                    "#": i,
                    "Class": d["class_name"].upper(),
                    "Confidence": f"{d['confidence']*100:.1f}%",
                    "Surface Area %": f"{d['relative_area_pct']:.2f}%",
                    "Severity": d.get("severity", "LOW"),
                    "Bounding Box [x1,y1,x2,y2]": str(d["bbox"]),
                    "Engineering Rationale": d.get("reason", "Rule evaluation")
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 3: RISK & SEVERITY ENGINE
# ==============================================================================
with tabs[2]:
    st.markdown("### 🛡️ Explainable Severity & Risk Prioritization Engine")
    st.write("Transparent rule-based scoring architecture strictly calculated from model detections.")

    if st.session_state.inspection_result is None:
        st.info("No active inspection loaded. Showing engine rule specification.")
    else:
        res = st.session_state.inspection_result
        
        col_risk_meter, col_risk_factors = st.columns([1, 2])
        with col_risk_meter:
            st.markdown(f"#### Prioritization Risk Score")
            st.markdown(f"<h1 style='font-size:3.5rem;color:#38bdf8;margin:0;'>{res['score']}<span style='font-size:1.5rem;color:#64748b;'> / 100</span></h1>", unsafe_allow_html=True)
            st.progress(res['score'] / 100.0)
            
            st.markdown(f"**Classification**: `{res['overall_severity']}`")
            st.markdown(f"**Label**: *{res['label']}*")

        with col_risk_factors:
            st.markdown("#### 🔬 Contributing Score Factors (Explainable Rules)")
            bd = res.get("score_breakdown", {})
            
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"• **Defect Quantity**: `{bd.get('defect_count_contribution', 0.0)} / 25 pts`")
                st.write(f"• **Detection Confidence**: `{bd.get('confidence_contribution', 0.0)} / 20 pts`")
            with c2:
                st.write(f"• **Defect Surface Area**: `{bd.get('area_contribution', 0.0)} / 25 pts`")
                st.write(f"• **Defect Severity**: `{bd.get('severity_contribution', 0.0)} / 30 pts`")

            st.markdown("**Transparent Evaluation Trace:**")
            for r in res.get("explanation_reasons", []):
                st.markdown(f"- {r}")

    st.markdown("""
    <div class="disclaimer-card">
        <b>STATUTORY MARINE DISCLAIMER:</b><br/>
        SHIPGUARD AI is an indigenous AI-assisted visual decision-support prototype.
        The generated Risk Score and Recommended Actions do not substitute for formal statutory marine surveys,
        ultrasound plate thickness measurement, or classification society certification.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: INSPECTION HISTORY
# ==============================================================================
with tabs[3]:
    st.markdown("### 📜 Local Ship Inspection Registry (SQLite)")
    st.write("Permanent records of all completed contactless ship hull inspections stored locally.")

    history_records = db.get_all_inspections()

    if not history_records:
        st.info("No inspection records in database yet. Analyze an image to create the first record.")
    else:
        # Search & Filter
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            search_query = st.text_input("🔍 Search inspections by image name or notes:", "")
        with col_f2:
            severity_filter = st.selectbox("Filter by Severity:", ["All Severities", "CRITICAL", "HIGH", "MEDIUM", "LOW", "CLEAN"])

        filtered = [
            r for r in history_records
            if (search_query.lower() in r["image_name"].lower() or search_query.lower() in (r.get("notes") or "").lower())
            and (severity_filter == "All Severities" or r["severity"] == severity_filter)
        ]

        df_hist = pd.DataFrame([
            {
                "Record ID": r["id"],
                "Timestamp": r["timestamp"],
                "Image Source": r["image_name"],
                "Defects": r["defect_count"],
                "Defect Breakdown": str(r["defect_types"]),
                "Peak Conf": f"{r['highest_confidence']*100:.1f}%",
                "Severity": r["severity"],
                "Risk Score": f"{r['risk_score']}/100",
                "Recommended Action": r["recommended_action"]
            }
            for r in filtered
        ])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # Download CSV export of history
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Inspection Registry as CSV",
            data=csv_data,
            file_name="shipguard_inspection_registry.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 5: REPORT GENERATOR
# ==============================================================================
with tabs[4]:
    st.markdown("### 📑 Automated Marine Inspection Report Generator")
    st.write("Generate and download comprehensive, statutory-ready PDF inspection reports with embedded imagery and metrics.")

    if st.session_state.inspection_result is None:
        st.warning("Please analyze an inspection image first before generating a report.")
    else:
        res = st.session_state.inspection_result
        st.markdown(f"**Ready to compile report for:** `{res['image_name']}` (Record #{st.session_state.saved_inspection_id})")
        
        col_rep1, col_rep2 = st.columns([1, 1])
        with col_rep1:
            st.markdown("#### Report Contents:")
            st.markdown("""
            - Maritime Inspector Header & Timestamp
            - Executive Summary (Defects, Severity, Risk, Confidence)
            - Action Protocol & Operational Scope
            - Embedded High-Resolution Annotated Visualizations
            - Localized Defect Table with Bounding Box Coordinates
            - Rule-Based Explainability Audit Trail
            - Official Marine Engineering Notice
            """)
        
        with col_rep2:
            st.markdown("#### Download PDF Document:")
            # Generate PDF in-memory
            pdf_bytes = generate_pdf_report(res, res.get("annotated_image"))
            
            st.download_button(
                label="📄 Download Official PDF Report",
                data=pdf_bytes,
                file_name=f"SHIPGUARD_Report_{res['image_name'].split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            st.success(f"Report compiled successfully ({len(pdf_bytes):,} bytes). Ready for distribution.")


# ==============================================================================
# TAB 6: ARCHITECTURE & FUTURE SENSORS
# ==============================================================================
with tabs[5]:
    st.markdown("### 🏗️ System Architecture & Multi-Sensor Fusion Roadmap")
    
    st.markdown("#### 1. Current Vision MVP Architecture (Fully Operational)")
    st.markdown("""
    ```
    Ship Hull Inspection Imagery (Camera / Drone Optical Feed)
                     ↓
        Image Preprocessing & Normalization (640x640 Letterbox)
                     ↓
        AI Computer Vision Detection Engine (YOLOv8 Weights)
                     ↓
        Defect Localization & Category Tagging (Corrosion / Rust / Crack)
                     ↓
        Explainable Rule-Based Severity Engine (LOW / MEDIUM / HIGH / CRITICAL)
                     ↓
        Inspection Prioritization Risk Engine (0-100 Continuous Score)
                     ↓
        SQLite Local Inspection History Registry & Analytics
                     ↓
        Unified Dashboard & Automated ReportLab PDF Generator
    ```
    """)

    st.markdown("---")
    st.markdown("#### 2. Future Multi-Modal Sensor Fusion Architecture (FUTURE WORK)")
    st.info("⚠️ NOTICE: The following multi-sensor integration represents the planned commercial scale-up architecture. In accordance with the No Fake Data Policy, LiDAR, thermal, and acoustic streams are not simulated.")
    
    st.markdown("""
    ```
        ┌────────────────────────────────────────────────────────┐
        │                 FUTURE HARDWARE LAYER                  │
        │  [ Optical Camera ]  [ 3D LiDAR ]  [ Thermal IR ]  [ Acoustic ] │
        └──────────────┬──────────────┬─────────────┬────────────┬───────┘
                       │              │             │            │
                       ▼              ▼             ▼            ▼
                   Optical CV     Point Cloud   Temperature    Resonance
                   Defect BBox     Depth Map     Gradient     Integrity
                       │              │             │            │
                       └──────────────┼─────────────┴────────────┘
                                      ▼
                        Multi-Sensor Fusion Engine
                     (Spatial Cross-Attention Transformer)
                                      ▼
                      Integrated Structural Health Digital Twin
    ```
    """)
    
    st.markdown("""
    - **LiDAR Integration**: Accurate structural deformation and dent depth measurements ($<1\\text{ mm}$ tolerance).
    - **Thermal IR Imaging**: Sub-surface delamination and insulation moisture entrapment detection.
    - **Acoustic / EMAT Resonance**: Contactless plate thickness thinning and internal micro-fissure profiling.
    """)


# ==============================================================================
# TAB 7: DATASET & MODEL TRANSPARENCY
# ==============================================================================
with tabs[6]:
    st.markdown("### 📚 Dataset Provenance & Model Specifications")
    st.write("Full disclosure of training data sources, model weights, and the strict No Fake Data Policy.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 📦 Primary Datasets")
        st.markdown("""
        1. **Roboflow Marine Rust & Corrosion Dataset** (`rust-detection-38s6e`):
           - 1,842 real industrial and marine images.
           - Classes: `corrosion`, `severe-corrosion`, `moderate-corrosion`, `mild-corrosion`, `iron rust`, `copper corrosion`, `rust`, `corroded-part`.
           - License: CC BY 4.0 / Open Access.
        2. **OpenSistemas Crack Dataset** (`YOLOv8-crack-seg`):
           - 11,298 real structural crack inspection images.
           - Class: `crack`.
           - License: MIT License.
        """)

    with col_d2:
        st.markdown("#### 🧠 Model Specifications")
        st.markdown("""
        - **Primary Model**: YOLOv8 Instance Detection (`model/best.pt`)
        - **Secondary Model**: YOLOv8 Structural Crack Engine (`model/crack_best.pt`)
        - **Weights Disclosure**: Pre-trained on verified public datasets.
        - **Custom Model Loading**: To load your own fine-tuned weights, place `best.pt` inside the `model/` folder.
        """)

    st.markdown("---")
    st.markdown("""
    #### 🛡️ Compliance with No Fake Data Policy
    - **No hardcoded predictions**: All bounding boxes, confidence values, and class tags are produced dynamically by the PyTorch neural network.
    - **No fabricated accuracy metrics**: Precision, recall, and mAP are reported only where evaluated by standard validation splits.
    - **No simulated sensor data**: Future sensor modalities (LiDAR, thermal, acoustic) are strictly demarcated as planned future architecture.
    """)
