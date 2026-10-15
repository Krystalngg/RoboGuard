"""Incident reporting and structured explanation schemas for Sentinel-VL.

Strict integrity rules:
- Generated text is labeled "generated_unreviewed" by default; never presented as verified finding.
- Temporal anomaly score is explicitly labeled as uncalibrated or weakly supervised.
- Strict runtime validation on bounds, finite numbers, enums, intervals, and strict JSON serialization.
- No legal or definitive conclusions ("guilty", "crime confirmed", "safe") permitted.
- Evidence frame IDs must be validated against available frame keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import html
import json
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

VALID_DECISIONS = {"accepted_normal", "accepted_anomalous", "review_required"}
VALID_DESCRIPTION_STATUSES = {"generated_unreviewed", "human_reviewed", "rejected"}
VALID_PROVENANCES = {"real_model", "synthetic_demo", "feature_test", "synthetic_feature_demo"}
PROHIBITED_PHRASES = [
    "crime confirmed",
    "criminal detected",
    "guilty",
    "perpetrator identified",
    "safe guaranteed",
]


@dataclass
class IncidentReport:
    """Structured report for an analyzed video segment."""
    video_id: str
    interval_seconds: Tuple[float, float]
    temporal_anomaly_score: float
    score_status: str  # e.g., "weakly_supervised_not_segment_calibrated"
    video_decision: Literal["accepted_normal", "accepted_anomalous", "review_required"]
    calibration_version: str
    description: str
    evidence_frame_ids: List[str]
    description_status: Literal["generated_unreviewed", "human_reviewed", "rejected"] = "generated_unreviewed"
    model_version: str = "sentinel-vl-m0"
    provenance: Literal["real_model", "synthetic_demo", "feature_test", "synthetic_feature_demo"] = "synthetic_demo"
    calibrated_video_probability: Optional[float] = None
    ensemble_disagreement: Optional[float] = None
    evidence_frame_paths: Optional[List[str]] = None
    provenance_details: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        # Runtime string check
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValueError("video_id must not be empty.")

        # Interval bounds
        if not hasattr(self.interval_seconds, "__len__") or len(self.interval_seconds) != 2:
            raise ValueError("interval_seconds must contain exactly [start, end].")
        start, end = float(self.interval_seconds[0]), float(self.interval_seconds[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"Interval bounds must be finite numbers, got ({start}, {end})")
        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval: start={start}, end={end}")

        # Anomaly score check
        if not isinstance(self.temporal_anomaly_score, (int, float)) or not math.isfinite(self.temporal_anomaly_score):
            raise ValueError(f"temporal_anomaly_score must be a finite float, got {self.temporal_anomaly_score}")
        if not (0.0 <= self.temporal_anomaly_score <= 1.0):
            raise ValueError(f"temporal_anomaly_score must be in [0, 1], got {self.temporal_anomaly_score}")

        # Calibrated probability check
        if self.calibrated_video_probability is not None:
            if not isinstance(self.calibrated_video_probability, (int, float)) or not math.isfinite(self.calibrated_video_probability):
                raise ValueError(f"calibrated_video_probability must be finite, got {self.calibrated_video_probability}")
            if not (0.0 <= self.calibrated_video_probability <= 1.0):
                raise ValueError(f"calibrated_video_probability must be in [0, 1], got {self.calibrated_video_probability}")

        # Disagreement check
        if self.ensemble_disagreement is not None:
            if not isinstance(self.ensemble_disagreement, (int, float)) or not math.isfinite(self.ensemble_disagreement):
                raise ValueError(f"ensemble_disagreement must be finite, got {self.ensemble_disagreement}")
            if self.ensemble_disagreement < 0.0:
                raise ValueError(f"ensemble_disagreement must be non-negative, got {self.ensemble_disagreement}")

        # Enums
        if self.video_decision not in VALID_DECISIONS:
            raise ValueError(f"video_decision must be one of {sorted(VALID_DECISIONS)}, got '{self.video_decision}'")
        if self.description_status not in VALID_DESCRIPTION_STATUSES:
            raise ValueError(
                f"description_status must be one of {sorted(VALID_DESCRIPTION_STATUSES)}, got '{self.description_status}'"
            )
        if self.provenance not in VALID_PROVENANCES:
            raise ValueError(f"provenance must be one of {sorted(VALID_PROVENANCES)}, got '{self.provenance}'")

        # Prohibit inflammatory or legally conclusive labels
        desc_lower = self.description.lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in desc_lower:
                raise ValueError(
                    f"Prohibited label '{phrase}' found in description. "
                    "Sentinel-VL generates objective visual event descriptions only."
                )

    def validate_evidence_frames(self, available_frame_ids: List[str]) -> None:
        """Validates that all evidence frame IDs exist in the provided candidate pool."""
        available_set = set(available_frame_ids)
        missing = [fid for fid in self.evidence_frame_ids if fid not in available_set]
        if missing:
            raise ValueError(f"Evidence frame IDs do not exist in candidate pool: {missing}")

    def to_dict(self, escape_html: bool = False) -> Dict[str, Any]:
        """Converts report to dictionary with optional HTML escaping for dashboard security."""
        data = asdict(self)
        if escape_html:
            data["description"] = html.escape(self.description)
            data["video_id"] = html.escape(self.video_id)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to formatted JSON string with strict NaN/Inf prohibition."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)
