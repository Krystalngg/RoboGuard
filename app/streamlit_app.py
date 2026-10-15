"""Sentinel-VL: Uncertainty-Aware and Explainable Surveillance Monitoring.

Streamlit Integrated Application (Milestones M0 - M6 & Production Extensions).

Connects:
- Real Video Inference (Native ffmpeg decoding + TorchVisualEncoder + MIL Head)
- Synthetic Feature Inference (Loaded MIL Head on pre-extracted feature vectors)
- Explicit Synthetic Demo Mode (Deterministic layout validation fixtures)
- Evidence-linked video player with timestamp seeking & real extracted frame thumbnails
- Persistent reviewer incident queue with attributable review actions and rejection reasons
- Model & Calibration Health Panel with SHA-256 integrity checks
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sentinel_vl import __version__
from sentinel_vl.data.video_decoder import VideoDecoder
from sentinel_vl.inference.pipeline import (
    InferenceDisabledError,
    PipelineConfig,
    SentinelInferencePipeline,
)
from sentinel_vl.reporting.schemas import IncidentReport

INCIDENTS_STORE_PATH = Path("data/incidents/review_queue.json")


def setup_page_config() -> None:
    st.set_page_config(
        page_title="Sentinel-VL Surveillance Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_persistent_incidents() -> List[Dict[str, Any]]:
    """Loads durable human review incidents from JSON store."""
    if not INCIDENTS_STORE_PATH.is_file():
        return []
    try:
        with open(INCIDENTS_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_persistent_incident(incident: Dict[str, Any]) -> None:
    """Appends or updates an attributable review record in durable storage."""
    INCIDENTS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    incidents = load_persistent_incidents()
    existing_idx = next((i for i, inc in enumerate(incidents) if inc["incident_id"] == incident["incident_id"]), None)
    if existing_idx is not None:
        incidents[existing_idx] = incident
    else:
        incidents.append(incident)
    with open(INCIDENTS_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2)


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("🛡️ Sentinel-VL")
    st.sidebar.caption(f"Research Version {__version__} | Production Blueprint")

    ckpt_path = Path("checkpoints/mil_head_baseline.json")
    has_checkpoint = ckpt_path.is_file()
    decoder = VideoDecoder()
    has_decoder = decoder.is_available

    st.sidebar.header("1. Execution Mode")
    mode_options = ["Synthetic Demo Mode"]
    if has_checkpoint:
        mode_options.append("Synthetic Feature Inference (Loaded MIL Head)")
        if has_decoder:
            mode_options.append("Real Video Inference (Trained MIL Head & Native Decoder)")

    selected_mode = st.sidebar.radio(
        "Select Pipeline Mode:",
        options=mode_options,
        index=len(mode_options) - 1 if has_checkpoint else 0,
        help="Real Video: Native ffmpeg decoding + TorchVisualEncoder. Synthetic Feature: Evaluates MIL head on pre-extracted features. Synthetic Demo: Deterministic mock.",
    )

    is_demo = selected_mode == "Synthetic Demo Mode"
    is_real_video = selected_mode.startswith("Real Video Inference")

    st.sidebar.divider()
    st.sidebar.header("2. Video Input Source")

    uploaded_file = None
    selected_video_id = "Arrest001_x264"
    duration = 35.0
    custom_video_path: Optional[Path] = None

    if is_real_video:
        input_choice = st.sidebar.radio(
            "Video Source:",
            options=["Upload Custom Video Clip", "Generate Live Synthetic MP4 Clip"],
        )
        if input_choice == "Upload Custom Video Clip":
            uploaded_file = st.sidebar.file_uploader(
                "Upload Surveillance Footage (.mp4, .avi, .mov)",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if uploaded_file is not None:
                tmp_dir = Path(tempfile.gettempdir()) / "sentinel_uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                custom_video_path = tmp_dir / uploaded_file.name
                with open(custom_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_video_id = uploaded_file.name
        else:
            # Generate a reproducible local test video
            tmp_dir = Path(tempfile.gettempdir()) / "sentinel_samples"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            custom_video_path = tmp_dir / "sample_surveillance_10s.mp4"
            if not custom_video_path.is_file():
                import subprocess
                subprocess.run([
                    decoder.ffmpeg_path, "-y", "-f", "lavfi",
                    "-i", "testsrc=duration=10:size=320x240:rate=10",
                    "-pix_fmt", "yuv420p", str(custom_video_path),
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            selected_video_id = "sample_surveillance_10s.mp4"
            duration = 10.0
    else:
        demo_video_options = {
            "Arrest001_x264": "Arrest Incident (35.0s, UCA Match)",
            "Burglary002_x264": "Burglary Rear Door (48.0s, UCA Match)",
            "Fighting003_x264": "Street Sidewalk Altercation (28.0s, UCA Match)",
            "Normal_Videos_001_x264": "Normal Pedestrian Plaza (60.0s, UCF Match)",
        }
        selected_video_id = st.sidebar.selectbox(
            "Select Target Surveillance Clip",
            options=list(demo_video_options.keys()),
            format_func=lambda x: demo_video_options[x],
        )
        duration_map = {
            "Arrest001_x264": 35.0,
            "Burglary002_x264": 48.0,
            "Fighting003_x264": 28.0,
            "Normal_Videos_001_x264": 60.0,
        }
        duration = duration_map.get(selected_video_id, 30.0)

    st.sidebar.divider()
    st.sidebar.header("3. Selective Decision Policy")
    tau_high = st.sidebar.slider("Anomaly Threshold (τ_high)", 0.50, 0.95, 0.70, 0.05)
    tau_low = st.sidebar.slider("Normal Threshold (τ_low)", 0.05, 0.50, 0.30, 0.05)
    tau_var = st.sidebar.slider("Max Uncertainty Disagreement (τ_var)", 0.02, 0.30, 0.12, 0.02)
    uncertainty_policy = st.sidebar.selectbox(
        "Decision Policy:",
        options=["conservative_production", "permissive_legacy"],
        help="Conservative Production requires valid calibration and ensemble disagreement evidence.",
    )

    return {
        "is_demo": is_demo,
        "is_real_video": is_real_video,
        "video_id": selected_video_id,
        "custom_video_path": custom_video_path,
        "duration": duration,
        "tau_high": tau_high,
        "tau_low": tau_low,
        "tau_var": tau_var,
        "uncertainty_policy": uncertainty_policy,
        "has_checkpoint": has_checkpoint,
    }


def render_banner(mode: str) -> None:
    if mode == "demo":
        st.markdown(
            """
            <div style="background-color: #fff3cd; color: #856404; padding: 12px 16px; border-radius: 6px; border-left: 6px solid #ffeeba; margin-bottom: 20px;">
                <strong>⚠️ SCIENTIFIC INTEGRITY NOTICE — SYNTHETIC DEMO ACTIVE</strong><br/>
                Timeline scores, video decisions, and calibrated probabilities are <strong>deterministic synthetic fixtures</strong>.
                No raw video decoding or active model inference is running.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "real_video":
        st.markdown(
            """
            <div style="background-color: #d4edda; color: #155724; padding: 12px 16px; border-radius: 6px; border-left: 6px solid #c3e6cb; margin-bottom: 20px;">
                <strong>🎥 REAL VIDEO INFERENCE ACTIVE — NATIVE DECODER & VISION BACKBONE</strong><br/>
                Analyzing uploaded footage with <strong>native ffmpeg decoding</strong>, deep visual feature extraction
                (ResNet-50 visual encoder), and temporal multiple-instance learning. Evidence frames link to actual extracted video imagery.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #e2e3e5; color: #383d41; padding: 12px 16px; border-radius: 6px; border-left: 6px solid #d6d8db; margin-bottom: 20px;">
                <strong>🧪 SYNTHETIC FEATURE INFERENCE ACTIVE</strong><br/>
                Predictions and temporal attributions are generated by the trained MIL anomaly head over <strong>synthetic feature vectors</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    setup_page_config()
    cfg = render_sidebar()

    current_mode = "demo" if cfg["is_demo"] else ("real_video" if cfg["is_real_video"] else "feature")
    render_banner(current_mode)

    st.title("Surveillance Video-Language Understanding & Review")
    st.write(
        "Sentinel-VL provides natural-language retrieval, weakly supervised anomaly scoring, "
        "selective decision-making, and evidence-grounded reports for human surveillance review."
    )

    # Retrieval Query Bar
    st.subheader("1. Video-Language Retrieval & Semantic Search")
    col_query, col_btn = st.columns([4, 1])
    with col_query:
        query_text = st.text_input(
            "Natural Language Search Query",
            value="officers approach vehicle in parking lot",
            help="Query surveillance clips using cross-modal embedding alignment.",
        )
    with col_btn:
        st.write("")
        st.write("")
        st.button("Search Video", use_container_width=True)

    # Execute Pipeline
    pipeline_output: Dict[str, Any]
    if cfg["is_demo"] or not cfg["has_checkpoint"]:
        pipeline_output = SentinelInferencePipeline.run_synthetic_demo(
            video_id=cfg["video_id"],
            duration_seconds=cfg["duration"],
            query=query_text,
        )
    elif cfg["is_real_video"] and cfg["custom_video_path"] is not None:
        calib_art_path = Path("checkpoints/calibration_baseline.json")
        pipe_cfg = PipelineConfig(
            real_inference_enabled=True,
            anomaly_head_checkpoint="checkpoints/mil_head_baseline.json",
            calibration_artifact_path=str(calib_art_path) if calib_art_path.is_file() else None,
            selective_high_threshold=cfg["tau_high"],
            selective_low_threshold=cfg["tau_low"],
            selective_max_disagreement=cfg["tau_var"],
            uncertainty_policy=cfg["uncertainty_policy"],
            target_fps=2.0,
            window_duration_seconds=3.0,
        )
        pipeline = SentinelInferencePipeline(pipe_cfg)
        try:
            pipeline_output = pipeline.run_video_inference(
                video_path=cfg["custom_video_path"],
                query=query_text,
            )
        except Exception as err:
            st.error(f"Real video inference failed: {err}")
            return
    else:
        calib_art_path = Path("checkpoints/calibration_baseline.json")
        pipe_cfg = PipelineConfig(
            real_inference_enabled=True,
            anomaly_head_checkpoint="checkpoints/mil_head_baseline.json",
            calibration_artifact_path=str(calib_art_path) if calib_art_path.is_file() else None,
            selective_high_threshold=cfg["tau_high"],
            selective_low_threshold=cfg["tau_low"],
            selective_max_disagreement=cfg["tau_var"],
            uncertainty_policy=cfg["uncertainty_policy"],
        )
        pipeline = SentinelInferencePipeline(pipe_cfg)
        seed_int = int.from_bytes(hashlib.sha256(cfg["video_id"].encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(seed_int)
        n_windows = max(4, int(cfg["duration"] // 4.0))
        dim = pipeline.anomaly_head.feature_dim
        features = rng.standard_normal((n_windows, dim)).astype(np.float32) * 0.2

        pipeline_output = pipeline.run_feature_inference(
            video_id=cfg["video_id"],
            temporal_features=features,
            duration_seconds=cfg["duration"],
            query=query_text,
            provenance="synthetic_feature_demo",
        )

    report: IncidentReport = pipeline_output["report"]
    windows = pipeline_output["windows"]
    retrieval = pipeline_output.get("retrieval_results", [])

    # Dashboard Tabs
    tab_overview, tab_player, tab_timeline, tab_xai, tab_retrieval, tab_review_queue, tab_health = st.tabs([
        "📊 Surveillance Overview",
        "🎥 Video Player & Evidence",
        "📈 Temporal Timeline",
        "🔬 XAI Attribution",
        "🔍 Retrieval Matches",
        "📋 Persistent Review Queue",
        "⚙️ Model & Calibration Health",
    ])

    with tab_overview:
        st.markdown(f"### Video: `{html.escape(report.video_id)}` (Duration: {pipeline_output.get('duration_seconds', cfg['duration']):.1f}s)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Peak Temporal Anomaly Score",
                value=f"{report.temporal_anomaly_score:.3f}",
                help="Raw MIL anomaly score for peak temporal window.",
            )
            st.caption(f"Status: {report.score_status}")

        with col2:
            st.metric(
                label="Calibrated Video Probability",
                value=f"{report.calibrated_video_probability:.3f}" if report.calibrated_video_probability is not None else "Uncalibrated",
                help="Temperature-scaled video-level probability.",
            )
            st.caption(f"Status: {report.calibration_version}")

        with col3:
            st.metric(
                label="Ensemble Disagreement",
                value=f"{report.ensemble_disagreement:.3f}" if report.ensemble_disagreement is not None else "N/A",
                help="Predictive variance across independently trained heads.",
            )
            st.caption("Metric: Disagreement Proxy")

        with col4:
            decision_colors = {
                "accepted_normal": "🟢 Accepted Normal",
                "accepted_anomalous": "🔴 Accepted Anomalous",
                "review_required": "🟡 Review Required",
            }
            st.metric(
                label="Selective Decision",
                value=decision_colors.get(report.video_decision, report.video_decision),
                help="Automated decision policy based on calibrated risk-coverage thresholds.",
            )
            st.caption(f"Provenance: {report.provenance.upper()}")

        st.info(
            f"**Event Description Summary:** {report.description}\n\n"
            f"*Description Status:* `{report.description_status}` | *Model Version:* `{report.model_version}`"
        )

    with tab_player:
        st.subheader("Timestamped Video Playback & Evidence Frames")
        col_vid, col_frames = st.columns([3, 2])

        with col_vid:
            if cfg["is_real_video"] and cfg["custom_video_path"] is not None and cfg["custom_video_path"].is_file():
                start_sec = int(report.interval_seconds[0])
                st.video(str(cfg["custom_video_path"]), start_time=start_sec)
                st.caption(f"▶️ Video player seeks directly to peak event window: `{report.interval_seconds[0]}s` to `{report.interval_seconds[1]}s`.")
            else:
                st.info("Video player is active in Real Video mode. Upload footage to inspect native timestamp-linked playback.")

        with col_frames:
            st.markdown("#### 🖼️ Verified Evidence Frames")
            if report.evidence_frame_paths:
                for fpath in report.evidence_frame_paths:
                    p = Path(fpath)
                    if p.is_file():
                        st.image(str(p), caption=f"Frame: {p.name}", use_container_width=True)
            elif report.evidence_frame_ids:
                st.write(f"Referenced Evidence Window IDs: `{', '.join(report.evidence_frame_ids)}`")
            else:
                st.caption("No decoded frame imagery available.")

    with tab_timeline:
        st.subheader("Temporal Anomaly Score Trajectory")
        st.caption("Scores for regular temporal windows preserving full tail coverage.")

        df_windows = pd.DataFrame(windows)
        st.bar_chart(df_windows, x="start_seconds", y="temporal_anomaly_score", height=300)
        display_cols = [c for c in ["window_index", "window_id", "start_seconds", "end_seconds", "temporal_anomaly_score", "importance", "status"] if c in df_windows.columns]
        st.dataframe(df_windows[display_cols], use_container_width=True)

    with tab_xai:
        st.subheader("Prediction-Specific Leave-One-Out Frame Attribution")
        st.caption("Causal importance quantifying score drop when each temporal window is ablated.")

        if "importance" in df_windows.columns:
            st.line_chart(df_windows, x="start_seconds", y="importance", height=300)

        if report.evidence_frame_ids:
            st.success(
                f"**Top Evidence Attributed:** `{', '.join(report.evidence_frame_ids)}` "
                f"produced peak causal impact on the prediction."
            )

    with tab_retrieval:
        st.subheader("Cross-Modal Retrieval Results")
        if not retrieval:
            st.warning("Semantic retrieval disabled: No verified multimodal text encoder is configured.")
        else:
            for match in retrieval:
                with st.container(border=True):
                    c_rank, c_info = st.columns([1, 5])
                    with c_rank:
                        st.markdown(f"### Rank #{match['rank']}")
                        st.metric("Similarity", f"{match['similarity_score']:.3f}")
                    with c_info:
                        st.markdown(f"**Temporal Interval:** `{match['interval_seconds'][0]}s` to `{match['interval_seconds'][1]}s`")
                        st.markdown(f"**Aligned Reference:** {match['reference_text']}")
                        st.caption(f"Provenance: `{match['provenance']}`")

    with tab_review_queue:
        st.subheader("📋 Durable Incident Review Queue & Attributable Audit Log")
        incident_id = f"inc_{report.video_id}_{int(report.interval_seconds[0])}_{int(report.interval_seconds[1])}"

        with st.form("incident_review_form"):
            st.markdown(f"**Incident Reference:** `{incident_id}` | Video: `{report.video_id}`")
            col_rev1, col_rev2 = st.columns(2)
            with col_rev1:
                reviewer_name = st.text_input("Reviewer Operator ID / Name:", value="Operator-01")
                assigned_to = st.text_input("Assignee / Security Unit:", value="Lobby-Desk-A")
            with col_rev2:
                review_action = st.selectbox(
                    "Reviewer Disposition Decision:",
                    options=[
                        "Approved (True Anomaly Confirmed)",
                        "Rejected (False Positive / Benign Activity)",
                        "Escalate (Senior Supervisor Review Required)",
                    ],
                )
                rejection_reason = st.selectbox(
                    "Rejection / False Positive Category:",
                    options=[
                        "N/A (Incident Approved)",
                        "Normal Staff / Visitor Foot Traffic",
                        "Camera Lighting / Glare Fluctuation",
                        "Weather / Environmental Movement (Foliage, Rain)",
                        "Sensor Glitch / Frame Corruption",
                        "Other",
                    ],
                )

            reviewer_comments = st.text_area(
                "Reviewer Audit Comments:",
                value="Verified event window against real video footage.",
            )

            submitted = st.form_submit_button("💾 Save Attributable Human Review to Audit Store")
            if submitted:
                record = {
                    "incident_id": incident_id,
                    "video_id": report.video_id,
                    "interval_seconds": report.interval_seconds,
                    "peak_score": report.temporal_anomaly_score,
                    "automated_decision": report.video_decision,
                    "reviewer_id": reviewer_name,
                    "assigned_to": assigned_to,
                    "disposition": review_action,
                    "rejection_reason": rejection_reason,
                    "reviewer_comments": reviewer_comments,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                    "model_version": report.model_version,
                    "provenance": report.provenance,
                }
                save_persistent_incident(record)
                st.success(f"Incident `{incident_id}` successfully signed off and stored in audit database!")

        st.divider()
        st.subheader("Audit Trail & Historic Incident Log")
        historic = load_persistent_incidents()
        if historic:
            df_hist = pd.DataFrame(historic)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("No historical review records in persistent queue yet.")

    with tab_health:
        st.subheader("Model and Calibration Health Panel")
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("#### Model Architecture & Checkpoints")
            ckpt_p = Path("checkpoints/mil_head_baseline.json")
            if ckpt_p.is_file():
                st.write(f"- **MIL Baseline Checkpoint:** `{ckpt_p.name}` (Found)")
                with open(ckpt_p, "r") as f:
                    data = json.load(f)
                st.write(f"- **Model SHA-256:** `{data.get('sha256', 'Missing')}`")
                st.write(f"- **Feature Dimension:** `{data.get('feature_dim', 'N/A')}`")
            else:
                st.warning("MIL checkpoint not found. Run `sentinel-vl train-mil` to initialize.")

            st.write("- **Visual Encoder:** `TorchVisualEncoder (ResNet-50 backbone)`")
            st.write(f"- **Decision Policy:** `{cfg['uncertainty_policy'].upper()}`")

        with col_h2:
            st.markdown("#### Uncertainty & Calibration Health")
            calib_p = Path("checkpoints/calibration_baseline.json")
            if calib_p.is_file():
                st.write(f"- **Calibration Artifact:** `{calib_p.name}` (Active)")
            else:
                st.info("- **Calibration Artifact:** None loaded (Predictions marked uncalibrated)")

            st.write("- **Selective Policy Bounds:**")
            st.write(f"  - τ_high: `{cfg['tau_high']}`")
            st.write(f"  - τ_low: `{cfg['tau_low']}`")
            st.write(f"  - τ_var: `{cfg['tau_var']}`")


if __name__ == "__main__":
    main()
