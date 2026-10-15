"""Uncertainty Quantification, Calibration, and Selective Decision Service (Milestone M4).

Scientific requirements:
- Head ensemble over fixed features with multiple independently trained/initialized heads.
- Temperature scaling T > 0 fitted strictly on validation/calibration split.
- Uses consistent video-level aggregation (top-k mean bag score) across training, calibration, and inference.
- Saves and loads versioned calibration artifacts.
- If no valid calibration artifact is loaded, predictions are explicitly labeled 'uncalibrated' (calibrated_prob = None).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.uncertainty.contracts import (
    UQPrediction,
    compute_binary_entropy,
    compute_risk_and_coverage,
)


@dataclass
class CalibrationArtifact:
    """Versioned artifact capturing temperature scaling fit provenance and model compatibility."""
    temperature: float
    aggregation_definition: str = "top_k_mean"
    calibrator_version: str = "1.0.0"
    head_checkpoint_hashes: List[str] = field(default_factory=list)
    calibration_partition_hash: Optional[str] = None
    encoder_version: Optional[str] = None
    feature_dim: Optional[int] = None
    top_k_ratio: Optional[float] = None
    min_k: Optional[int] = None
    ensemble_composition: Optional[List[str]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {self.temperature}")

    def validate_compatibility(
        self,
        expected_aggregation: str = "top_k_mean",
        expected_head_hashes: Optional[List[str]] = None,
        expected_encoder_version: Optional[str] = None,
        expected_feature_dim: Optional[int] = None,
        expected_top_k_ratio: Optional[float] = None,
        expected_min_k: Optional[int] = None,
    ) -> None:
        """Validates that this calibration artifact matches model and pipeline requirements."""
        if self.aggregation_definition != expected_aggregation:
            raise ValueError(
                f"Incompatible calibration artifact aggregation: expected '{expected_aggregation}', "
                f"got '{self.aggregation_definition}'"
            )
        if expected_head_hashes and self.head_checkpoint_hashes:
            if not any(h in self.head_checkpoint_hashes for h in expected_head_hashes):
                raise ValueError(
                    f"Incompatible calibration artifact: current head hash not in artifact "
                    f"checkpoint hashes {self.head_checkpoint_hashes}"
                )
        if expected_encoder_version and self.encoder_version and self.encoder_version != expected_encoder_version:
            raise ValueError(
                f"Incompatible encoder version in calibration artifact: expected '{expected_encoder_version}', "
                f"got '{self.encoder_version}'"
            )
        if expected_feature_dim and self.feature_dim and self.feature_dim != expected_feature_dim:
            raise ValueError(
                f"Incompatible feature dimension in calibration artifact: expected {expected_feature_dim}, "
                f"got {self.feature_dim}"
            )
        if expected_top_k_ratio is not None and self.top_k_ratio is not None and abs(self.top_k_ratio - expected_top_k_ratio) > 1e-4:
            raise ValueError(
                f"Incompatible top_k_ratio in calibration artifact: expected {expected_top_k_ratio}, "
                f"got {self.top_k_ratio}"
            )
        if expected_min_k is not None and self.min_k is not None and self.min_k != expected_min_k:
            raise ValueError(
                f"Incompatible min_k in calibration artifact: expected {expected_min_k}, "
                f"got {self.min_k}"
            )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class HeadEnsemble:
    """Ensemble of independently trained/initialized anomaly prediction heads over fixed representations."""

    def __init__(
        self,
        heads: Optional[List[MILAnomalyHead]] = None,
        feature_dim: int = 512,
        num_heads: int = 3,
        seed: int = 42,
    ) -> None:
        if heads is not None:
            if len(heads) < 2:
                raise ValueError("HeadEnsemble requires at least 2 heads to quantify predictive disagreement.")
            # Verify model independence: reject duplicate weights/hashes
            hashes = [h.compute_weights_hash() for h in heads]
            if len(set(hashes)) < len(heads):
                raise ValueError(
                    "HeadEnsemble rejects duplicate models with identical weights/hashes. "
                    "Distinct independently trained models are required to quantify predictive disagreement."
                )
            self.heads = heads
        else:
            self.heads = [
                MILAnomalyHead(feature_dim=feature_dim, seed=seed + i * 17)
                for i in range(max(2, num_heads))
            ]

    @classmethod
    def load_from_checkpoints(cls, checkpoint_paths: List[str | Path]) -> HeadEnsemble:
        """Loads ensemble from a list of checkpoint JSON files."""
        if len(checkpoint_paths) < 2:
            raise ValueError("Ensemble loading requires at least 2 checkpoint paths.")
        heads = [MILAnomalyHead.load_checkpoint(p) for p in checkpoint_paths]
        return cls(heads=heads)

    def predict_bag(self, temporal_features: np.ndarray) -> Tuple[float, float, List[float]]:
        """Predicts aggregated video score, predictive disagreement, and mean window scores across ensemble.

        Returns:
            (mean_bag_score, disagreement, window_mean_scores)
        """
        all_window_scores = []
        all_bag_scores = []
        head_entropies = []

        for head in self.heads:
            win_scores = head.predict_window_scores(temporal_features)
            bag_s = head.aggregate_video_score(win_scores)
            all_window_scores.append(win_scores)
            all_bag_scores.append(bag_s)
            head_entropies.append(compute_binary_entropy(bag_s))

        # Mean top-k bag score across ensemble
        mean_bag = float(np.mean(all_bag_scores))

        # Shannon entropy of ensemble mean
        mean_entropy = compute_binary_entropy(mean_bag)

        # Disagreement proxy = H(p_bar) - mean(H(p_m))
        disagreement = max(0.0, float(mean_entropy - np.mean(head_entropies)))

        # Mean window scores across ensemble
        mean_win_scores = np.mean(all_window_scores, axis=0).tolist()
        mean_win_scores = [round(float(s), 4) for s in mean_win_scores]

        return round(mean_bag, 4), round(disagreement, 4), mean_win_scores


class TemperatureCalibrator:
    """Post-hoc temperature scaling for video-level binary anomaly predictions."""

    def __init__(self, temperature: Optional[float] = None) -> None:
        if temperature is not None and temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature) if temperature is not None else None

    @property
    def is_calibrated(self) -> bool:
        return self.temperature is not None and self.temperature > 0.0

    @staticmethod
    def _logit(p: float) -> float:
        p_clipped = max(min(p, 1.0 - 1e-7), 1e-7)
        return float(np.log(p_clipped / (1.0 - p_clipped)))

    def calibrate(self, raw_bag_score: float) -> Optional[float]:
        """Applies temperature scaling: p_T = sigmoid(logit(p) / T). Returns None if uncalibrated."""
        if not self.is_calibrated or self.temperature is None:
            return None
        z = self._logit(raw_bag_score)
        scaled_z = np.clip(z / self.temperature, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-scaled_z))
        return round(float(prob), 4)

    def fit(
        self,
        val_bag_scores: List[float],
        val_labels: List[int],
        lr: float = 0.05,
        epochs: int = 50,
    ) -> float:
        """Optimizes temperature T on validation calibration partition using NLL."""
        if len(val_bag_scores) != len(val_labels) or len(val_bag_scores) == 0:
            raise ValueError("Validation scores and labels must be non-empty and of equal length.")

        logits = np.array([self._logit(s) for s in val_bag_scores], dtype=np.float32)
        y = np.array(val_labels, dtype=np.float32)

        init_t = self.temperature if self.temperature else 1.0
        log_t = float(np.log(max(init_t, 0.1)))

        for _ in range(epochs):
            t = np.exp(log_t)
            scaled = np.clip(logits / t, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-scaled))

            grad_log_t = np.sum((sig - y) * (-logits / t))
            log_t -= lr * float(np.clip(grad_log_t, -5.0, 5.0))

        self.temperature = round(float(np.exp(log_t)), 4)
        return self.temperature

    def compute_ece(
        self,
        calibrated_probs: List[float],
        labels: List[int],
        num_bins: int = 5,
    ) -> float:
        """Computes Expected Calibration Error (ECE)."""
        probs = np.array(calibrated_probs)
        y = np.array(labels)
        n = len(probs)
        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0

        for b in range(num_bins):
            in_bin = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1] if b < num_bins - 1 else probs <= bin_edges[b + 1])
            count = np.sum(in_bin)
            if count > 0:
                bin_acc = float(np.mean(y[in_bin]))
                bin_conf = float(np.mean(probs[in_bin]))
                ece += (count / n) * abs(bin_acc - bin_conf)

        return round(float(ece), 4)

    def to_artifact(
        self,
        checkpoint_hashes: Optional[List[str]] = None,
        partition_hash: Optional[str] = None,
    ) -> CalibrationArtifact:
        if not self.is_calibrated or self.temperature is None:
            raise ValueError("Cannot create artifact from an uncalibrated calibrator.")
        return CalibrationArtifact(
            temperature=self.temperature,
            aggregation_definition="top_k_mean",
            head_checkpoint_hashes=checkpoint_hashes or [],
            calibration_partition_hash=partition_hash,
        )

    @classmethod
    def from_artifact(cls, artifact: CalibrationArtifact) -> TemperatureCalibrator:
        return cls(temperature=artifact.temperature)


class SelectiveDecisionService:
    """Decides between automated positive/negative decisions and human review abstention."""

    def __init__(
        self,
        high_threshold: float = 0.70,
        low_threshold: float = 0.30,
        max_disagreement: float = 0.12,
        policy: str = "conservative_production",
        require_calibration: bool = True,
        require_disagreement: bool = True,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_disagreement = max_disagreement
        self.policy = policy
        self.require_calibration = require_calibration
        self.require_disagreement = require_disagreement

    def decide(
        self,
        score_or_calibrated_prob: float,
        disagreement: Optional[float] = None,
        is_calibrated: bool = True,
    ) -> str:
        """Returns 'accepted_normal', 'accepted_anomalous', or 'review_required'."""
        if self.policy == "conservative_production":
            if self.require_calibration and not is_calibrated:
                return "review_required"
            if self.require_disagreement and disagreement is None:
                return "review_required"

        if disagreement is not None and disagreement > self.max_disagreement:
            return "review_required"

        if score_or_calibrated_prob >= self.high_threshold:
            return "accepted_anomalous"
        elif score_or_calibrated_prob <= self.low_threshold:
            return "accepted_normal"
        else:
            return "review_required"

    def evaluate_risk_coverage(
        self,
        scores_or_probs: List[float],
        disagreements: List[float],
        labels: List[int],
    ) -> Dict[str, Any]:
        """Evaluates policy coverage, error risk, and review rate."""
        decisions = [
            self.decide(p, d) for p, d in zip(scores_or_probs, disagreements)
        ]
        accepted_mask = [d != "review_required" for d in decisions]
        predicted_binary = [
            1 if d == "accepted_anomalous" else 0 for d in decisions
        ]

        cov, risk, rev_rate = compute_risk_and_coverage(
            accepted_mask=accepted_mask,
            ground_truth=labels,
            predicted_labels=predicted_binary,
        )

        return {
            "coverage": cov,
            "risk": risk,
            "review_rate": rev_rate,
            "total_samples": len(labels),
            "accepted_count": sum(accepted_mask),
            "review_count": len(labels) - sum(accepted_mask),
        }
