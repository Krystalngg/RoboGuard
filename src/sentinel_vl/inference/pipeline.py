"""Integrated Inference Pipeline for Sentinel-VL.

Connects:
- Raw video decoding with ffmpeg/ffprobe and media stream metadata verification
- Versioned deep visual encoder (TorchVisualEncoder) for spatial feature extraction
- Temporal window aggregation with exact timestamp preservation and partial window coverage
- Anomaly scoring via MIL head or HeadEnsemble
- Video-level probability calibration with artifact compatibility validation
- Predictive disagreement & conservative selective decision service
- Attribution tied to the actual decision function (standalone head or ensemble)
- Grounded incident report generation with real evidence frame linkage
- Disabling semantic retrieval when verified text encoder is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.data.video_decoder import (
    TimestampedFrame,
    VideoDecoder,
    VideoDecoderError,
    VideoMetadata,
)
from sentinel_vl.explainability.attribution import FrameAttributionService
from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.models.encoders import SemanticTextEncoder, TorchVisualEncoder
from sentinel_vl.models.retrieval import (
    MeanTemporalAggregator,
    SyntheticFeatureGenerator,
    VideoTextRetrievalHead,
)
from sentinel_vl.reporting.grounding import GroundedReportGenerator
from sentinel_vl.reporting.schemas import IncidentReport
from sentinel_vl.uncertainty.calibration import (
    CalibrationArtifact,
    HeadEnsemble,
    SelectiveDecisionService,
    TemperatureCalibrator,
)
from sentinel_vl.uncertainty.contracts import UQPrediction


class InferenceDisabledError(RuntimeError):
    """Raised when inference is attempted without configured checkpoints or unsupported inputs."""
    pass


@dataclass
class PipelineConfig:
    """Configuration parameters for Sentinel-VL pipeline execution."""
    real_inference_enabled: bool = False
    anomaly_head_checkpoint: Optional[str] = None
    ensemble_checkpoints: Optional[List[str]] = None
    calibration_artifact_path: Optional[str] = None
    calibrator_temperature: Optional[float] = None
    visual_encoder_weights_path: Optional[str] = None
    selective_high_threshold: float = 0.70
    selective_low_threshold: float = 0.30
    selective_max_disagreement: float = 0.12
    uncertainty_policy: str = "conservative_production"
    require_calibration: bool = True
    require_disagreement: bool = True
    device: str = "cpu"
    target_fps: float = 1.0
    window_duration_seconds: float = 4.0


class SentinelInferencePipeline:
    """Inference pipeline executing MIL scoring, UQ, attribution, and reporting."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.anomaly_head: Optional[MILAnomalyHead] = None
        self.ensemble: Optional[HeadEnsemble] = None
        self.calibrator = TemperatureCalibrator(temperature=None)
        self.decision_service = SelectiveDecisionService(
            high_threshold=self.config.selective_high_threshold,
            low_threshold=self.config.selective_low_threshold,
            max_disagreement=self.config.selective_max_disagreement,
            policy=self.config.uncertainty_policy,
            require_calibration=self.config.require_calibration,
            require_disagreement=self.config.require_disagreement,
        )
        self.retrieval_head = VideoTextRetrievalHead()
        self.decoder = VideoDecoder()
        self.temporal_aggregator = MeanTemporalAggregator(normalize=True)
        self.visual_encoder: Optional[TorchVisualEncoder] = None
        self.text_encoder = SemanticTextEncoder(is_configured=False)

        # Load anomaly head checkpoint if configured and enabled
        if self.config.real_inference_enabled:
            if self.config.anomaly_head_checkpoint:
                ckpt_path = Path(self.config.anomaly_head_checkpoint)
                if not ckpt_path.is_file():
                    raise InferenceDisabledError(f"Configured checkpoint does not exist: {ckpt_path}")
                self.anomaly_head = MILAnomalyHead.load_checkpoint(ckpt_path)

            # Load multi-head ensemble if multiple checkpoints configured
            if self.config.ensemble_checkpoints and len(self.config.ensemble_checkpoints) >= 2:
                for p in self.config.ensemble_checkpoints:
                    if not Path(p).is_file():
                        raise InferenceDisabledError(f"Configured ensemble checkpoint does not exist: {p}")
                self.ensemble = HeadEnsemble.load_from_checkpoints(self.config.ensemble_checkpoints)

            # Initialize visual encoder
            try:
                feature_dim = self.anomaly_head.feature_dim if self.anomaly_head else 512
                self.visual_encoder = TorchVisualEncoder(
                    feature_dim=feature_dim,
                    weights_path=self.config.visual_encoder_weights_path,
                    device=self.config.device,
                )
            except Exception:
                self.visual_encoder = None

        # Load calibration artifact or temperature if configured with compatibility validation
        if self.config.calibration_artifact_path:
            calib_path = Path(self.config.calibration_artifact_path)
            if calib_path.is_file():
                artifact = CalibrationArtifact.load(calib_path)
                if self.anomaly_head is not None:
                    head_hash = self.anomaly_head.compute_weights_hash()
                    artifact.validate_compatibility(
                        expected_aggregation="top_k_mean",
                        expected_head_hashes=[head_hash],
                        expected_feature_dim=self.anomaly_head.feature_dim,
                    )
                self.calibrator = TemperatureCalibrator.from_artifact(artifact)
        elif self.config.calibrator_temperature is not None:
            self.calibrator = TemperatureCalibrator(temperature=self.config.calibrator_temperature)

    def run_video_inference(
        self,
        video_path: str | Path,
        query: Optional[str] = None,
        output_frames_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Executes full raw video inference pipeline:

        Video file -> Metadata validation -> Timestamped frame decoding ->
        Visual feature encoding -> Temporal window aggregation ->
        MIL / Ensemble anomaly scoring -> Calibration -> Conservative decision ->
        Attribution & frame evidence -> Structured Grounded Incident Report.
        """
        if not self.config.real_inference_enabled:
            raise InferenceDisabledError(
                "Real-video inference is disabled: pipeline config has real_inference_enabled=False. "
                "Set config.real_inference_enabled=True and configure a valid checkpoint."
            )

        if self.anomaly_head is None:
            raise InferenceDisabledError("Real-video inference is disabled: no anomaly head checkpoint loaded.")

        if self.visual_encoder is None:
            raise InferenceDisabledError("Real-video inference is disabled: visual encoder is not initialized.")

        v_path = Path(video_path).resolve()
        if not v_path.is_file():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        if not self.decoder.is_available:
            raise InferenceDisabledError("ffmpeg and ffprobe must be installed on PATH for video inference.")

        # 1. Probing media metadata
        meta = self.decoder.probe(v_path)
        if meta.duration_seconds <= 0.0:
            raise VideoDecoderError(f"Unable to determine valid positive duration for '{v_path}'.")

        # 2. Extract regularly spaced timestamped frames
        frames = self.decoder.extract_regular_frames(
            v_path,
            target_fps=self.config.target_fps,
            output_dir=output_frames_dir,
        )
        if not frames:
            raise VideoDecoderError(f"No frames could be extracted from video '{v_path}'.")

        # 3. Partition into temporal feature windows with full tail coverage
        w_duration = float(self.config.window_duration_seconds)
        total_duration = float(meta.duration_seconds)
        n_windows = max(1, math.ceil(total_duration / w_duration))

        window_features: List[np.ndarray] = []
        windows_meta: List[Dict[str, Any]] = []
        window_to_frames: Dict[int, List[TimestampedFrame]] = {}

        for w_idx in range(n_windows):
            start_sec = round(w_idx * w_duration, 2)
            end_sec = round(min(total_duration, (w_idx + 1) * w_duration), 2)

            # Match extracted frames belonging to interval
            if w_idx == n_windows - 1:
                w_frames = [f for f in frames if start_sec <= f.timestamp_seconds <= end_sec]
            else:
                w_frames = [f for f in frames if start_sec <= f.timestamp_seconds < end_sec]

            if not w_frames:
                # Nearest frame fallback
                nearest = min(frames, key=lambda f: abs(f.timestamp_seconds - (start_sec + end_sec) / 2.0))
                w_frames = [nearest]

            window_to_frames[w_idx] = w_frames

            # Extract visual features for frames in window
            frame_feats = self.visual_encoder.encode_frames([f.image_path for f in w_frames])
            # Aggregate frame embeddings into window embedding
            win_emb = self.temporal_aggregator.forward(frame_feats)
            window_features.append(win_emb)

            windows_meta.append({
                "window_index": w_idx,
                "window_id": f"win_{w_idx:03d}",
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "frame_count": len(w_frames),
                "referenced_frames": [f.image_path.name for f in w_frames],
            })

        temporal_features = np.stack(window_features, axis=0)

        # 4. Anomaly scoring & decision attribution
        win_scores = self.anomaly_head.predict_window_scores(temporal_features)
        bag_score = self.anomaly_head.aggregate_video_score(win_scores)
        peak_idx = int(np.argmax(win_scores))
        peak_score = float(win_scores[peak_idx])
        peak_interval = (windows_meta[peak_idx]["start_seconds"], windows_meta[peak_idx]["end_seconds"])

        # 5. Uncertainty Quantification (Ensemble and Calibration)
        if self.ensemble is not None:
            mean_bag, disagreement, ens_win_scores = self.ensemble.predict_bag(temporal_features)
            score_to_calibrate = mean_bag
            model_for_attribution = self.ensemble
            anomaly_provenance = "real_model_ensemble"
        else:
            mean_bag = bag_score
            disagreement = None
            score_to_calibrate = bag_score
            model_for_attribution = self.anomaly_head
            anomaly_provenance = "real_model_single_head"

        if self.calibrator.is_calibrated:
            calibrated_prob = self.calibrator.calibrate(score_to_calibrate)
            calib_version = f"video_temp_{self.calibrator.temperature}"
            calib_provenance = "video_temperature_calibrated"
            is_calibrated = True
        else:
            calibrated_prob = None
            calib_version = "uncalibrated"
            calib_provenance = "uncalibrated"
            is_calibrated = False

        score_for_decision = calibrated_prob if calibrated_prob is not None else mean_bag
        decision = self.decision_service.decide(
            score_for_decision,
            disagreement=disagreement,
            is_calibrated=is_calibrated,
        )

        uq_summary = UQPrediction(
            raw_score=round(mean_bag, 4),
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            decision=decision,
            granularity="video_level_calibrated" if is_calibrated else "segment_level_uncalibrated",
            calibration_version=calib_version,
            provenance="real_model",
        )

        # 6. Prediction-Specific Attribution to the actual decision function
        importances = FrameAttributionService.compute_leave_one_out_importance(
            model_for_attribution, temporal_features
        )
        for idx, imp in enumerate(importances):
            windows_meta[idx]["importance"] = imp
            windows_meta[idx]["temporal_anomaly_score"] = win_scores[idx]
            windows_meta[idx]["status"] = "evaluated_video_window"

        # Evidence frame linkage pointing to real stored images on disk
        sorted_window_idxs = np.argsort(-np.array(importances))
        top_window_idx = int(sorted_window_idxs[0])
        top_frames = window_to_frames[top_window_idx]

        evidence_frame_ids = [f.image_path.name for f in top_frames[:3]]
        evidence_frame_paths = [str(f.image_path) for f in top_frames[:3]]
        candidate_frame_pool = [f.image_path.name for f in frames]

        # 7. Natural language cross-modal retrieval
        retrieval_results = []
        if query:
            if self.text_encoder and self.text_encoder.is_available:
                q_emb = self.text_encoder.encode_text(query, dim=self.anomaly_head.feature_dim)
                clip_sims = self.retrieval_head.rank_candidates_for_query(
                    q_emb, temporal_features, top_k=min(3, n_windows)
                )
                for rank_pos, (c_idx, sim_score) in enumerate(clip_sims, 1):
                    w_info = windows_meta[c_idx]
                    retrieval_results.append({
                        "rank": rank_pos,
                        "interval_seconds": [w_info["start_seconds"], w_info["end_seconds"]],
                        "similarity_score": round(sim_score, 4),
                        "query": query,
                        "reference_text": f"Evaluated surveillance window {w_info['window_id']}.",
                        "provenance": "real_model",
                    })
                retrieval_provenance = "real_multimodal_text_encoder"
            else:
                retrieval_provenance = "disabled_no_verified_text_encoder"
        else:
            retrieval_provenance = "none_requested"

        provenance_details = {
            "media_provenance": "decoded_video_frames",
            "visual_encoder": self.visual_encoder.encoder_version,
            "anomaly_scoring": anomaly_provenance,
            "calibration": calib_provenance,
            "retrieval": retrieval_provenance,
            "reporting": "grounded_unreviewed",
        }

        # 8. Honest Grounded Report
        report = GroundedReportGenerator.generate_report(
            video_id=v_path.stem,
            peak_interval=peak_interval,
            temporal_anomaly_score=peak_score,
            video_decision=decision,
            evidence_frame_ids=evidence_frame_ids,
            candidate_frame_pool=candidate_frame_pool,
            observed_action_description=None,
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            calibration_version=calib_version,
            model_version=f"sentinel-vl-m6-video-engine [{self.visual_encoder.encoder_version}]",
            provenance="real_model",
            evidence_frame_paths=evidence_frame_paths,
            provenance_details=provenance_details,
        )

        return {
            "provenance": "real_model",
            "provenance_details": provenance_details,
            "video_id": v_path.stem,
            "duration_seconds": total_duration,
            "metadata": meta,
            "windows": windows_meta,
            "uq_summary": uq_summary,
            "report": report,
            "retrieval_results": retrieval_results,
            "extracted_frames": frames,
        }

    def run_feature_inference(
        self,
        video_id: str,
        temporal_features: np.ndarray,
        duration_seconds: float,
        query: Optional[str] = None,
        candidate_reference_texts: Optional[List[str]] = None,
        provenance: str = "feature_test",
    ) -> Dict[str, Any]:
        """Runs inference over pre-extracted temporal feature vectors."""
        if not self.config.real_inference_enabled:
            raise InferenceDisabledError(
                "Real model inference is disabled in pipeline config. "
                "Set config.real_inference_enabled=True and configure a valid checkpoint."
            )

        if self.anomaly_head is None:
            raise InferenceDisabledError("No anomaly head checkpoint loaded.")

        t_windows = len(temporal_features)
        if t_windows == 0:
            raise ValueError("temporal_features array must not be empty.")

        # Ensure caller cannot fake 'real_model' without video decoding provenance
        if provenance == "real_model":
            provenance = "feature_test"

        window_duration = duration_seconds / float(t_windows)

        # 1. Anomaly scoring
        win_scores = self.anomaly_head.predict_window_scores(temporal_features)
        bag_score = self.anomaly_head.aggregate_video_score(win_scores)
        peak_score = float(np.max(win_scores))
        peak_idx = int(np.argmax(win_scores))
        peak_interval = (
            round(peak_idx * window_duration, 1),
            round(min(duration_seconds, (peak_idx + 1) * window_duration), 1),
        )

        # 2. Uncertainty Quantification: Ensemble and Calibration
        if self.ensemble is not None:
            mean_bag, disagreement, _ = self.ensemble.predict_bag(temporal_features)
            score_to_calibrate = mean_bag
            model_for_attribution = self.ensemble
            anomaly_prov = "real_model_ensemble"
        else:
            mean_bag = bag_score
            disagreement = None
            score_to_calibrate = bag_score
            model_for_attribution = self.anomaly_head
            anomaly_prov = "real_model_single_head"

        if self.calibrator.is_calibrated:
            calibrated_prob = self.calibrator.calibrate(score_to_calibrate)
            calib_version = f"video_temp_{self.calibrator.temperature}"
            is_calibrated = True
        else:
            calibrated_prob = None
            calib_version = "uncalibrated"
            is_calibrated = False

        score_for_decision = calibrated_prob if calibrated_prob is not None else mean_bag
        decision = self.decision_service.decide(
            score_for_decision,
            disagreement=disagreement,
            is_calibrated=is_calibrated,
        )

        uq_summary = UQPrediction(
            raw_score=round(mean_bag, 4),
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            decision=decision,
            granularity="video_level_calibrated" if is_calibrated else "segment_level_uncalibrated",
            calibration_version=calib_version,
            provenance=provenance,
        )

        # 3. Explainability: Window attribution using window_id instead of fake frame IDs
        importances = FrameAttributionService.compute_leave_one_out_importance(
            model_for_attribution, temporal_features
        )
        evidence_window_ids = [f"win_{idx:03d}" for idx in np.argsort(-np.array(importances))[:2]]
        candidate_window_pool = [f"win_{idx:03d}" for idx in range(t_windows)]

        # 4. Grounded Reporting
        provenance_details = {
            "media_provenance": "caller_feature_vectors",
            "anomaly_scoring": anomaly_prov,
            "calibration": "calibrated" if is_calibrated else "uncalibrated",
            "retrieval": "disabled_or_mock" if query else "none",
        }

        report = GroundedReportGenerator.generate_report(
            video_id=video_id,
            peak_interval=peak_interval,
            temporal_anomaly_score=peak_score,
            video_decision=decision,
            evidence_frame_ids=evidence_window_ids,
            candidate_frame_pool=candidate_window_pool,
            observed_action_description=None,
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            calibration_version=calib_version,
            model_version="sentinel-vl-m6-feature-engine",
            provenance=provenance,
            provenance_details=provenance_details,
        )

        # 5. Retrieval ranking
        retrieval_results = []
        if query:
            if self.text_encoder and self.text_encoder.is_available:
                q_emb = self.text_encoder.encode_text(query, dim=self.anomaly_head.feature_dim)
                clip_sims = self.retrieval_head.rank_candidates_for_query(
                    q_emb, temporal_features, top_k=min(3, t_windows)
                )
                ret_prov = "real_multimodal_text_encoder"
            else:
                # Engineering smoke test simulation: explicitly labeled as synthetic hash query
                q_emb = SyntheticFeatureGenerator.generate_embedding(query, dim=self.anomaly_head.feature_dim)
                clip_sims = self.retrieval_head.rank_candidates_for_query(
                    q_emb, temporal_features, top_k=min(3, t_windows)
                )
                ret_prov = "synthetic_hash_query_simulation"

            for rank_pos, (c_idx, sim_score) in enumerate(clip_sims, 1):
                c_start = round(c_idx * window_duration, 1)
                c_end = round(min(duration_seconds, (c_idx + 1) * window_duration), 1)
                ref_txt = (
                    candidate_reference_texts[c_idx]
                    if candidate_reference_texts and c_idx < len(candidate_reference_texts)
                    else f"Segment evaluation window win_{c_idx:03d}."
                )
                retrieval_results.append({
                    "rank": rank_pos,
                    "interval_seconds": [c_start, c_end],
                    "similarity_score": round(sim_score, 4),
                    "query": query,
                    "reference_text": ref_txt,
                    "provenance": provenance,
                })

        windows = [
            {
                "window_index": i,
                "window_id": f"win_{i:03d}",
                "start_seconds": round(i * window_duration, 1),
                "end_seconds": round(min(duration_seconds, (i + 1) * window_duration), 1),
                "temporal_anomaly_score": win_scores[i],
                "importance": importances[i],
                "status": "evaluated_window",
            }
            for i in range(t_windows)
        ]

        return {
            "provenance": provenance,
            "provenance_details": provenance_details,
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "windows": windows,
            "uq_summary": uq_summary,
            "report": report,
            "retrieval_results": retrieval_results,
        }

    @staticmethod
    def run_synthetic_demo(
        video_id: str = "demo_surveillance_clip_001",
        duration_seconds: float = 35.0,
        query: Optional[str] = "person running away",
    ) -> Dict[str, Any]:
        """Generates explicitly labeled synthetic demonstration outputs with full tail coverage."""
        window_duration = 4.0
        num_windows = max(1, math.ceil(duration_seconds / window_duration))
        windows = []
        for idx in range(num_windows):
            start = round(idx * window_duration, 1)
            end = round(min(duration_seconds, (idx + 1) * window_duration), 1)
            center_norm = abs((idx / max(1, num_windows - 1)) - 0.5) * 2.0
            mock_score = round(max(0.08, min(0.88, 0.88 - 0.70 * center_norm)), 3)
            windows.append({
                "window_index": idx,
                "window_id": f"win_{idx:03d}",
                "start_seconds": start,
                "end_seconds": end,
                "temporal_anomaly_score": mock_score,
                "importance": round(mock_score * 0.4, 3),
                "status": "synthetic_mock_window",
            })

        max_window = max(windows, key=lambda w: w["temporal_anomaly_score"])
        peak_score = max_window["temporal_anomaly_score"]

        if peak_score > 0.75:
            decision = "review_required"
        elif peak_score > 0.5:
            decision = "accepted_anomalous"
        else:
            decision = "accepted_normal"

        uq_summary = UQPrediction(
            raw_score=peak_score,
            calibrated_probability=round(peak_score * 0.92, 3),
            disagreement=0.085,
            decision=decision,
            granularity="video_level_calibrated",
            calibration_version="mock_video_calibration_v0",
            provenance="synthetic_demo",
        )

        mock_report = IncidentReport(
            video_id=video_id,
            interval_seconds=(max_window["start_seconds"], max_window["end_seconds"]),
            temporal_anomaly_score=peak_score,
            score_status="weakly_supervised_not_segment_calibrated",
            video_decision=decision,
            calibration_version="mock_video_calibration_v0",
            description=(
                f"Synthetic Demo: Peak score ({peak_score:.3f}) observed between {max_window['start_seconds']}s and "
                f"{max_window['end_seconds']}s. (Synthetic mock fixture; no visual event description active)."
            ),
            evidence_frame_ids=["mock_frame_012", "mock_frame_016"],
            description_status="generated_unreviewed",
            model_version="sentinel-vl-m0-synthetic",
            provenance="synthetic_demo",
            calibrated_video_probability=uq_summary.calibrated_probability,
            ensemble_disagreement=uq_summary.disagreement,
            provenance_details={"mode": "synthetic_mock_demo"},
        )

        mock_retrieval = [
            {
                "rank": 1,
                "interval_seconds": [max_window["start_seconds"], max_window["end_seconds"]],
                "similarity_score": 0.842,
                "query": query or "person running",
                "reference_text": "Synthetic mock match: query simulation.",
                "provenance": "synthetic_demo",
            },
            {
                "rank": 2,
                "interval_seconds": [0.0, 4.0],
                "similarity_score": 0.315,
                "query": query or "person running",
                "reference_text": "Synthetic mock match: query simulation.",
                "provenance": "synthetic_demo",
            },
        ]

        return {
            "provenance": "synthetic_demo",
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "windows": windows,
            "uq_summary": uq_summary,
            "report": mock_report,
            "retrieval_results": mock_retrieval,
        }
