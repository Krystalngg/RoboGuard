"""Unit tests for auditability, model integrity safeguards, and schema bounds."""

import json
import math
from pathlib import Path
import pytest

from sentinel_vl.inference.pipeline import SentinelInferencePipeline
from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.reporting.schemas import IncidentReport
from sentinel_vl.uncertainty.calibration import (
    CalibrationArtifact,
    HeadEnsemble,
    SelectiveDecisionService,
)


def test_head_ensemble_rejects_duplicate_heads() -> None:
    head1 = MILAnomalyHead(feature_dim=32, seed=42)
    # head2 with exact same weights
    head2 = MILAnomalyHead(feature_dim=32, seed=42)

    with pytest.raises(ValueError, match="HeadEnsemble rejects duplicate models with identical weights/hashes"):
        HeadEnsemble(heads=[head1, head2])


def test_calibration_artifact_compatibility_rejection() -> None:
    artifact = CalibrationArtifact(
        temperature=1.2,
        aggregation_definition="top_k_mean",
        head_checkpoint_hashes=["hash_model_A"],
        feature_dim=512,
    )

    # Valid check passes
    artifact.validate_compatibility(
        expected_aggregation="top_k_mean",
        expected_head_hashes=["hash_model_A"],
        expected_feature_dim=512,
    )

    # Incompatible aggregation definition rejected
    with pytest.raises(ValueError, match="Incompatible calibration artifact aggregation"):
        artifact.validate_compatibility(expected_aggregation="max_pooling")

    # Incompatible checkpoint hash rejected
    with pytest.raises(ValueError, match="Incompatible calibration artifact: current head hash not in artifact"):
        artifact.validate_compatibility(expected_head_hashes=["hash_model_DIFFERENT"])

    # Incompatible feature dimension rejected
    with pytest.raises(ValueError, match="Incompatible feature dimension"):
        artifact.validate_compatibility(expected_feature_dim=256)


def test_checkpoint_checksum_pinning(tmp_path: Path) -> None:
    head = MILAnomalyHead(feature_dim=16, seed=10)
    ckpt_path = tmp_path / "pinned_ckpt.json"
    actual_hash = head.save_checkpoint(ckpt_path)

    # Correct pinned hash loads cleanly
    loaded = MILAnomalyHead.load_checkpoint(ckpt_path, expected_checksum=actual_hash)
    assert loaded.feature_dim == 16

    # Mismatched pinned hash raises error
    wrong_hash = "0" * 64
    with pytest.raises(ValueError, match="pinned checksum mismatch"):
        MILAnomalyHead.load_checkpoint(ckpt_path, expected_checksum=wrong_hash)


def test_checkpoint_missing_checksum_rejected(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "unsigned_ckpt.json"
    payload = {
        "model_type": "MILAnomalyHead",
        "version": "1.0.0",
        "feature_dim": 8,
        "bias": 0.0,
        "weights": [0.0] * 8,
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(ValueError, match="lacks required 'sha256' field"):
        MILAnomalyHead.load_checkpoint(ckpt_path, verify_checksum=True)


def test_incident_report_strict_runtime_validation() -> None:
    # Non-finite score rejected
    with pytest.raises(ValueError, match="temporal_anomaly_score must be a finite float"):
        IncidentReport(
            video_id="vid_1",
            interval_seconds=(0.0, 4.0),
            temporal_anomaly_score=float("nan"),
            score_status="uncalibrated",
            video_decision="review_required",
            calibration_version="none",
            description="Normal activity.",
            evidence_frame_ids=[],
        )

    # Invalid decision enum rejected
    with pytest.raises(ValueError, match="video_decision must be one of"):
        IncidentReport(
            video_id="vid_1",
            interval_seconds=(0.0, 4.0),
            temporal_anomaly_score=0.5,
            score_status="uncalibrated",
            video_decision="arbitrary_unvalidated_status",  # Invalid!
            calibration_version="none",
            description="Normal activity.",
            evidence_frame_ids=[],
        )

    # Non-finite probability rejected
    with pytest.raises(ValueError, match="calibrated_video_probability must be finite"):
        IncidentReport(
            video_id="vid_1",
            interval_seconds=(0.0, 4.0),
            temporal_anomaly_score=0.5,
            score_status="uncalibrated",
            video_decision="review_required",
            calibration_version="none",
            description="Normal activity.",
            evidence_frame_ids=[],
            calibrated_video_probability=float("inf"),
        )


def test_conservative_decision_service_requires_uncertainty_evidence() -> None:
    service = SelectiveDecisionService(
        policy="conservative_production",
        require_calibration=True,
        require_disagreement=True,
    )

    # Uncalibrated -> MUST require review
    assert service.decide(0.95, disagreement=0.01, is_calibrated=False) == "review_required"
    assert service.decide(0.05, disagreement=0.01, is_calibrated=False) == "review_required"

    # Missing disagreement (no ensemble) -> MUST require review
    assert service.decide(0.95, disagreement=None, is_calibrated=True) == "review_required"
    assert service.decide(0.05, disagreement=None, is_calibrated=True) == "review_required"

    # Fully calibrated with valid disagreement -> automated decision
    assert service.decide(0.85, disagreement=0.02, is_calibrated=True) == "accepted_anomalous"
    assert service.decide(0.15, disagreement=0.02, is_calibrated=True) == "accepted_normal"


def test_synthetic_demo_tail_coverage() -> None:
    # 35.0-second demo with 4.0-second windows must cover the full 35.0 seconds
    demo = SentinelInferencePipeline.run_synthetic_demo(video_id="tail_test", duration_seconds=35.0)
    windows = demo["windows"]
    assert len(windows) == 9  # 8 * 4.0 = 32.0, plus 9th window covering 32.0 to 35.0
    assert windows[-1]["start_seconds"] == 32.0
    assert windows[-1]["end_seconds"] == 35.0
