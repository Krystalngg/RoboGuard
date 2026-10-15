"""Grounded Incident Report Generator and Validator (Milestone M5).

Scientific rules:
- Generated text is labeled "generated_unreviewed" and never presented as verified fact.
- Descriptions do not claim unverified visual actions; when no VLM is active, honest score-based descriptions are emitted.
- Evidence frame references must be verified against actual candidate frame IDs.
- Provides conservative fallback report if confidence is low or evidence frames are missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from sentinel_vl.reporting.schemas import IncidentReport


class GroundedReportGenerator:
    """Generates structured incident reports grounded in model evidence."""

    @staticmethod
    def generate_report(
        video_id: str,
        peak_interval: Tuple[float, float],
        temporal_anomaly_score: float,
        video_decision: str,
        evidence_frame_ids: Optional[List[str]] = None,
        candidate_frame_pool: Optional[List[str]] = None,
        observed_action_description: Optional[str] = None,
        calibrated_probability: Optional[float] = None,
        disagreement: Optional[float] = None,
        calibration_version: str = "uncalibrated",
        model_version: str = "sentinel-vl-m5",
        provenance: str = "synthetic_demo",
        evidence_frame_paths: Optional[List[str]] = None,
        provenance_details: Optional[Dict[str, str]] = None,
    ) -> IncidentReport:
        """Constructs and validates an honest incident report without fabricated visual actions."""
        start_sec, end_sec = peak_interval

        # Validate evidence frames against candidate pool if provided
        valid_evidence: List[str] = []
        if evidence_frame_ids and candidate_frame_pool:
            pool_set = set(candidate_frame_pool)
            valid_evidence = [fid for fid in evidence_frame_ids if fid in pool_set]

        # If evidence frames were specified but none were valid, fall back conservatively
        if evidence_frame_ids and candidate_frame_pool and not valid_evidence:
            description = (
                f"Notice: Elevated anomaly activity detected between {start_sec:.1f}s and {end_sec:.1f}s. "
                "Evidence frame verification incomplete; human review required."
            )
            report_decision = "review_required"
        elif observed_action_description and observed_action_description.strip():
            description = (
                f"Observable Event ({start_sec:.1f}s - {end_sec:.1f}s): {observed_action_description.strip()}"
            )
            if valid_evidence:
                description += f" (Referenced frames: [{', '.join(valid_evidence)}])"
            report_decision = video_decision
        elif valid_evidence:
            description = (
                f"Observable Event Notice ({start_sec:.1f}s - {end_sec:.1f}s): The model assigned an elevated "
                f"anomaly score ({temporal_anomaly_score:.3f}). No visual event description is available "
                f"(VLM reporting is not configured). Referenced frames: [{', '.join(valid_evidence)}]."
            )
            report_decision = video_decision
        else:
            description = (
                f"The model assigned an elevated anomaly score ({temporal_anomaly_score:.3f}) "
                f"to interval [{start_sec:.1f}s, {end_sec:.1f}s]. "
                "No visual event description is available (VLM reporting is not configured)."
            )
            report_decision = video_decision

        report = IncidentReport(
            video_id=video_id,
            interval_seconds=(start_sec, end_sec),
            temporal_anomaly_score=round(temporal_anomaly_score, 4),
            score_status="weakly_supervised_not_segment_calibrated",
            video_decision=report_decision,
            calibration_version=calibration_version,
            description=description,
            evidence_frame_ids=valid_evidence,
            description_status="generated_unreviewed",
            model_version=model_version,
            provenance=provenance,
            calibrated_video_probability=calibrated_probability,
            ensemble_disagreement=disagreement,
            evidence_frame_paths=evidence_frame_paths if valid_evidence else None,
            provenance_details=provenance_details,
        )

        if valid_evidence and candidate_frame_pool:
            report.validate_evidence_frames(candidate_frame_pool)

        return report
