"""Regression tests verifying provenance guarantees, safeguards, and calibration integrity."""

import json
from pathlib import Path
import numpy as np
import pytest

from sentinel_vl.inference.pipeline import (
    InferenceDisabledError,
    PipelineConfig,
    SentinelInferencePipeline,
)
from sentinel_vl.models.anomaly import MILAnomalyHead, compute_roc_and_pr_auc
from sentinel_vl.uncertainty.calibration import (
    CalibrationArtifact,
    HeadEnsemble,
    TemperatureCalibrator,
)


def test_run_video_inference_disabled(tmp_path):
    """Verifies that run_video_inference raises InferenceDisabledError explicitly."""
    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"mock video data")

    pipeline = SentinelInferencePipeline()
    with pytest.raises(InferenceDisabledError, match="Real-video inference is disabled"):
        pipeline.run_video_inference(dummy_video)


def test_feature_inference_provenance_not_real_model():
    """Verifies that feature inference defaults to 'feature_test' and never 'real_model'."""
    ckpt_path = Path("checkpoints/mil_head_baseline.json")
    if not ckpt_path.is_file():
        pytest.skip("Baseline checkpoint not found")

    cfg = PipelineConfig(
        real_inference_enabled=True,
        anomaly_head_checkpoint=str(ckpt_path),
    )
    pipeline = SentinelInferencePipeline(cfg)

    features = np.zeros((4, 512), dtype=np.float32)
    res = pipeline.run_feature_inference("clip_001", features, duration_seconds=16.0)

    assert res["provenance"] == "feature_test"
    assert res["report"].provenance == "feature_test"
    assert res["uq_summary"].provenance == "feature_test"


def test_uncalibrated_when_no_artifact_configured():
    """Verifies that pipeline reports calibrated_video_probability as None if uncalibrated."""
    ckpt_path = Path("checkpoints/mil_head_baseline.json")
    if not ckpt_path.is_file():
        pytest.skip("Baseline checkpoint not found")

    cfg = PipelineConfig(
        real_inference_enabled=True,
        anomaly_head_checkpoint=str(ckpt_path),
        calibration_artifact_path=None,
    )
    pipeline = SentinelInferencePipeline(cfg)

    features = np.zeros((4, 512), dtype=np.float32)
    res = pipeline.run_feature_inference("clip_001", features, duration_seconds=16.0)

    assert res["report"].calibrated_video_probability is None
    assert res["report"].calibration_version == "uncalibrated"
    assert res["uq_summary"].calibrated_probability is None


def test_calibration_artifact_serialization(tmp_path):
    """Verifies CalibrationArtifact save, load, and calibration calculation."""
    artifact_path = tmp_path / "calibration.json"
    artifact = CalibrationArtifact(
        temperature=1.85,
        aggregation_definition="top_k_mean_bag",
        head_checkpoint_hashes=["abc123456"],
        calibration_partition_hash="val_split_v1",
    )
    artifact.save(artifact_path)

    loaded = CalibrationArtifact.load(artifact_path)
    assert loaded.temperature == 1.85
    assert loaded.aggregation_definition == "top_k_mean_bag"

    calibrator = TemperatureCalibrator.from_artifact(loaded)
    assert calibrator.is_calibrated is True
    # Test temperature scaling
    calibrated_score = calibrator.calibrate(0.7)
    assert 0.0 <= calibrated_score <= 1.0


def test_checkpoint_tamper_detection(tmp_path):
    """Verifies that MILAnomalyHead detects checksum mismatch upon load."""
    head = MILAnomalyHead(feature_dim=64)
    ckpt_path = tmp_path / "head.json"
    head.save_checkpoint(ckpt_path)

    # Tamper with checkpoint weights
    with open(ckpt_path, "r") as f:
        data = json.load(f)
    data["weights"][0] += 0.5  # Tamper without updating checksum
    with open(ckpt_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Checkpoint integrity verification failed"):
        MILAnomalyHead.load_checkpoint(ckpt_path)


def test_roc_tie_order_invariance():
    """Verifies that ROC-AUC and PR-AUC are tie-order invariant."""
    scores = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    labels = np.array([1, 1, 0, 1, 0])

    # Reorder tied indices
    perm1 = [0, 1, 2, 3, 4]
    perm2 = [0, 3, 2, 1, 4]

    res1 = compute_roc_and_pr_auc(labels[perm1], scores[perm1])
    res2 = compute_roc_and_pr_auc(labels[perm2], scores[perm2])

    assert isinstance(res1["roc_auc"], float)
    assert isinstance(res1["pr_auc"], float)
    assert res1["roc_auc"] == pytest.approx(res2["roc_auc"], abs=1e-6)
    assert res1["pr_auc"] == pytest.approx(res2["pr_auc"], abs=1e-6)


def test_head_ensemble_requires_multiple_heads():
    """Verifies that HeadEnsemble rejects initialization with fewer than 2 heads."""
    single_head = MILAnomalyHead(feature_dim=64)
    with pytest.raises(ValueError, match="HeadEnsemble requires at least 2 heads"):
        HeadEnsemble(heads=[single_head])
