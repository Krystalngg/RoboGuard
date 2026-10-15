"""Visual and text feature encoders for real video processing and multimodal alignment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np

from sentinel_vl.models.base import BaseVisualEncoder

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    HAS_PIL = False

try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TorchVisualEncoder(BaseVisualEncoder):
    """Deep visual feature extractor built on a frozen ResNet backbone."""

    def __init__(
        self,
        feature_dim: int = 512,
        weights_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        if not HAS_TORCH or not HAS_PIL:
            raise RuntimeError("PyTorch, torchvision, and Pillow are required for TorchVisualEncoder.")

        self.feature_dim = feature_dim
        self.encoder_version = f"resnet50-proj{feature_dim}-v1"
        self.device = torch.device(device)

        # Standard ImageNet pre-processing
        self.transform = tv_transforms.Compose([
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        resnet = tv_models.resnet50()

        # Check default cache if weights_path not supplied
        if weights_path is None:
            default_cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
            if default_cached.is_file():
                weights_path = default_cached

        if weights_path is not None and Path(weights_path).is_file():
            ckpt = torch.load(str(weights_path), map_location="cpu")
            resnet.load_state_dict(ckpt)

        resnet.eval()
        modules = list(resnet.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules).to(self.device)

        # Orthogonal projection to target feature dimension
        self.proj = torch.nn.Linear(2048, feature_dim, bias=False).to(self.device)
        torch.manual_seed(42)
        torch.nn.init.orthogonal_(self.proj.weight)
        self.proj.eval()

    def encode_frames(
        self,
        frames: List[Union[str, Path, Image.Image, np.ndarray]],
    ) -> np.ndarray:
        """Extracts normalized spatial embeddings for a list of image paths or PIL images.

        Returns:
            np.ndarray of shape (num_frames, feature_dim)
        """
        if not frames:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        tensors: List[torch.Tensor] = []
        for f in frames:
            if isinstance(f, (str, Path)):
                img = Image.open(f).convert("RGB")
            elif isinstance(f, np.ndarray):
                img = Image.fromarray(f).convert("RGB")
            elif isinstance(f, Image.Image):
                img = f.convert("RGB")
            else:
                raise TypeError(f"Unsupported frame type: {type(f)}")

            t = self.transform(img)
            tensors.append(t)

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(batch).squeeze(-1).squeeze(-1)
            projected = self.proj(feat)
            norm = projected.norm(p=2, dim=-1, keepdim=True)
            normalized = projected / torch.clamp(norm, min=1e-12)

        return normalized.cpu().numpy().astype(np.float32)


class SemanticTextEncoder:
    """Interface for verified cross-modal text embeddings.

    Explicitly marks availability: if no verified text model weights are configured,
    semantic search is safely marked unavailable rather than producing random hash vectors.
    """

    def __init__(self, is_configured: bool = False, model_name: Optional[str] = None) -> None:
        self.is_configured = is_configured
        self.model_name = model_name or "none"

    @property
    def is_available(self) -> bool:
        return self.is_configured

    def encode_text(self, text: str, dim: int = 512) -> np.ndarray:
        """Encodes text query into embedding space. Raises if encoder is not configured."""
        if not self.is_available:
            raise RuntimeError(
                "Semantic text encoder is not configured. Real video retrieval requires a verified "
                "multimodal text encoder in the same embedding space."
            )
        # Placeholder for plugged-in text model
        return np.zeros((dim,), dtype=np.float32)
