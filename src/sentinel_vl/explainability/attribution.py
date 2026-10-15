"""Prediction-Specific Explainability and Attribution (Milestone M5).

Scientific rules:
- Attribution is tied to a specific named model output (e.g., MIL anomaly bag score).
- Frame attribution is evaluated by perturbation (leave-one-out and removal degradation)
  against random removal controls, not merely visualized with saliency heatmaps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

from sentinel_vl.models.anomaly import MILAnomalyHead


class FrameAttributionService:
    """Computes and evaluates temporal attribution for video anomaly predictions."""

    @staticmethod
    def _get_bag_score(model: Any, features: np.ndarray) -> float:
        if hasattr(model, "predict_bag"):
            mean_bag, _, _ = model.predict_bag(features)
            return float(mean_bag)
        win_scores = model.predict_window_scores(features)
        return float(model.aggregate_video_score(win_scores))

    @staticmethod
    def compute_leave_one_out_importance(
        model: Any,
        temporal_features: np.ndarray,
    ) -> List[float]:
        """Calculates causal importance of each temporal window via leave-one-out perturbation.

        Importance I_j = bag_score(full) - bag_score(features without window j).
        Attributes the actual model or ensemble decision function.
        """
        if temporal_features.ndim != 2:
            raise ValueError(f"Expected 2D array (T, D), got {temporal_features.shape}")

        t_windows = len(temporal_features)
        if t_windows <= 1:
            return [1.0]

        # Base prediction
        base_bag_score = FrameAttributionService._get_bag_score(model, temporal_features)

        importances: List[float] = []

        for j in range(t_windows):
            # Mask window j
            mask = np.ones(t_windows, dtype=bool)
            mask[j] = False
            perturbed_features = temporal_features[mask]

            p_bag_score = FrameAttributionService._get_bag_score(model, perturbed_features)

            # Drop in score when window j is removed
            drop = base_bag_score - p_bag_score
            importances.append(round(float(drop), 4))

        return importances

    @staticmethod
    def evaluate_removal_versus_random(
        model: Any,
        temporal_features: np.ndarray,
        remove_k: int = 2,
        num_random_trials: int = 10,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Compares score degradation when removing top-k attributed windows vs random windows.

        A faithful attribution method should exhibit significantly greater drop when
        removing top-k important windows than when removing random windows of equal size.
        """
        t_windows = len(temporal_features)
        if t_windows <= remove_k:
            return {
                "top_k_drop": 0.0,
                "mean_random_drop": 0.0,
                "faithfulness_ratio": 1.0,
                "faithful": True,
            }

        base_bag = FrameAttributionService._get_bag_score(model, temporal_features)

        # 1. Top-k removal
        importances = FrameAttributionService.compute_leave_one_out_importance(model, temporal_features)
        top_indices = np.argsort(-np.array(importances))[:remove_k]

        mask_top = np.ones(t_windows, dtype=bool)
        mask_top[top_indices] = False
        feat_no_top = temporal_features[mask_top]
        bag_no_top = FrameAttributionService._get_bag_score(model, feat_no_top)
        top_k_drop = max(0.0, base_bag - bag_no_top)

        # 2. Random removal controls
        rng = np.random.RandomState(seed)
        random_drops: List[float] = []

        for _ in range(num_random_trials):
            rand_indices = rng.choice(t_windows, size=remove_k, replace=False)
            mask_rand = np.ones(t_windows, dtype=bool)
            mask_rand[rand_indices] = False
            feat_no_rand = temporal_features[mask_rand]
            bag_no_rand = FrameAttributionService._get_bag_score(model, feat_no_rand)
            random_drops.append(max(0.0, base_bag - bag_no_rand))

        mean_random_drop = float(np.mean(random_drops))
        faithfulness_ratio = round(top_k_drop / max(mean_random_drop, 1e-6), 2)

        return {
            "top_k_drop": round(top_k_drop, 4),
            "mean_random_drop": round(mean_random_drop, 4),
            "faithfulness_ratio": faithfulness_ratio,
            "faithful": top_k_drop >= mean_random_drop,
        }
