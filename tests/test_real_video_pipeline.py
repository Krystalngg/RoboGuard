"""Unit and integration tests for real video-to-feature inference pipeline.

Verifies:
1. Native timestamped decoding and visual feature extraction from actual video files.
2. Full tail coverage for clips with non-multiple durations (e.g. 35s clips).
3. Evidence frame linkage pointing to real stored image frames on disk.
4. Conservative decision policy requiring human review when uncalibrated or without ensemble.
5. Semantic retrieval disabled safely when no verified text encoder is available.
"""

from pathlib import Path
import subprocess
import numpy as np
import pytest

from sentinel_vl.data.video_decoder import VideoDecoder
from sentinel_vl.inference.pipeline import PipelineConfig, SentinelInferencePipeline
from sentinel_vl.models.anomaly import MILAnomalyHead


@pytest.fixture
def real_test_video(tmp_path: Path) -> Path:
    """Generates an authentic 3.5-second MP4 test video using ffmpeg."""
    decoder = VideoDecoder()
    if not decoder.is_available:
        pytest.skip("ffmpeg / ffprobe not available on PATH")

    clip_path = tmp_path / "test_clip_3.5s.mp4"
    cmd = [
        decoder.ffmpeg_path,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=3.5:size=160x120:rate=10",
        "-pix_fmt", "yuv420p",
        str(clip_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        pytest.skip(f"Failed to generate test video: {res.stderr.decode()}")

    return clip_path


def test_real_video_inference_execution(real_test_video: Path, tmp_path: Path) -> None:
    # 1. Create a trained MIL head checkpoint
    ckpt_path = tmp_path / "mil_ckpt.json"
    head = MILAnomalyHead(feature_dim=512, seed=42)
    head.save_checkpoint(ckpt_path)

    # 2. Configure pipeline with real video inference enabled
    config = PipelineConfig(
        real_inference_enabled=True,
        anomaly_head_checkpoint=str(ckpt_path),
        target_fps=2.0,
        window_duration_seconds=2.0,
        device="cpu",
    )
    pipeline = SentinelInferencePipeline(config)

    frames_dir = tmp_path / "extracted_frames"
    output = pipeline.run_video_inference(
        real_test_video,
        query="person entering door",
        output_frames_dir=frames_dir,
    )

    assert output["provenance"] == "real_model"
    assert output["video_id"] == real_test_video.stem
    assert output["duration_seconds"] == pytest.approx(3.5, abs=0.2)

    # Verify tail coverage: 3.5s video with 2.0s windows must have 2 windows:
    # Window 0: 0.0s to 2.0s
    # Window 1: 2.0s to 3.5s (partial final window preserved!)
    windows = output["windows"]
    assert len(windows) == 2
    assert windows[0]["start_seconds"] == 0.0
    assert windows[0]["end_seconds"] == 2.0
    assert windows[1]["start_seconds"] == 2.0
    assert windows[1]["end_seconds"] == pytest.approx(3.5, abs=0.2)

    # Verify conservative uncertainty policy:
    # Uncalibrated and without ensemble -> decision MUST be 'review_required'
    report = output["report"]
    assert report.video_decision == "review_required"
    assert report.score_status == "weakly_supervised_not_segment_calibrated"
    assert report.calibration_version == "uncalibrated"

    # Verify evidence frame linkage to real image files on disk
    assert len(report.evidence_frame_ids) > 0
    assert report.evidence_frame_paths is not None
    assert len(report.evidence_frame_paths) > 0
    for p_str in report.evidence_frame_paths:
        frame_file = Path(p_str)
        assert frame_file.is_file(), f"Referenced evidence frame does not exist: {p_str}"

    # Verify semantic retrieval is safely disabled rather than emitting fake embeddings
    assert output["retrieval_results"] == []
    assert output["provenance_details"]["retrieval"] == "disabled_no_verified_text_encoder"
