"""Typed data schemas and validation contracts for Sentinel-VL.

Provides strict invariant checks for video records, captions, anomaly windows,
manifests, and dataset split partitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


class ValidationError(ValueError):
    """Raised when data records violate schema invariants or interval bounds."""
    pass


VALID_DECODING_STATUSES = {"unverified", "ok", "corrupted", "missing", "decode_failed"}


@dataclass
class VideoRecord:
    """Represents a validated video file record in a dataset manifest."""
    video_id: str
    source_path: str
    duration_seconds: Optional[float] = None
    decoding_status: str = "unverified"
    fps: Optional[float] = None
    resolution: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValidationError("video_id must be a non-empty string")
        if not isinstance(self.source_path, str) or not self.source_path.strip():
            raise ValidationError("source_path must be a non-empty string")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValidationError("duration_seconds must be positive")
        if self.decoding_status not in VALID_DECODING_STATUSES:
            raise ValidationError(
                f"decoding_status must be one of {sorted(VALID_DECODING_STATUSES)}, "
                f"got '{self.decoding_status}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VideoRecord:
        return cls(**data)


@dataclass
class CaptionSegment:
    """Represents a natural-language description segment anchored in a temporal window."""
    segment_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    caption: str
    provenance: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, str) or not self.segment_id.strip():
            raise ValidationError("segment_id must be a non-empty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValidationError("video_id must be a non-empty string")
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValidationError("caption must be a non-empty string")
        if self.start_seconds < 0.0:
            raise ValidationError(f"start_seconds must be non-negative, got {self.start_seconds}")
        if self.end_seconds <= self.start_seconds:
            raise ValidationError(
                f"end_seconds ({self.end_seconds}) must be strictly greater than start_seconds ({self.start_seconds})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaptionSegment:
        return cls(**data)


@dataclass
class AnomalyWindow:
    """Represents a temporal window annotated with weakly supervised or ground truth anomaly labels."""
    window_id: str
    video_id: str
    start_seconds: float
    end_seconds: float
    label: Optional[int] = None
    label_provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id.strip():
            raise ValidationError("window_id must be a non-empty string")
        if not isinstance(self.video_id, str) or not self.video_id.strip():
            raise ValidationError("video_id must be a non-empty string")
        if self.start_seconds < 0.0:
            raise ValidationError(f"start_seconds must be non-negative, got {self.start_seconds}")
        if self.end_seconds <= self.start_seconds:
            raise ValidationError(
                f"end_seconds ({self.end_seconds}) must be strictly greater than start_seconds ({self.start_seconds})"
            )
        if self.label is not None and self.label not in (0, 1):
            raise ValidationError(f"label must be 0, 1, or None, got {self.label}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnomalyWindow:
        return cls(**data)


@dataclass
class DataManifest:
    """Dataset manifest consolidating videos, captions, and anomaly windows."""
    manifest_version: str = "1.0.0"
    videos: Dict[str, VideoRecord] = field(default_factory=dict)
    captions: List[CaptionSegment] = field(default_factory=list)
    anomaly_windows: List[AnomalyWindow] = field(default_factory=list)
    excluded_videos: Dict[str, str] = field(default_factory=dict)

    def validate_integrity(self) -> None:
        """Verifies foreign key references and temporal interval bounds against video durations."""
        for cap in self.captions:
            if cap.video_id not in self.videos:
                raise ValidationError(
                    f"Caption segment '{cap.segment_id}' references unknown video '{cap.video_id}'"
                )
            v = self.videos[cap.video_id]
            if v.duration_seconds is not None and cap.end_seconds > v.duration_seconds:
                raise ValidationError(
                    f"Caption segment '{cap.segment_id}' interval end {cap.end_seconds} "
                    f"exceeds video duration {v.duration_seconds}"
                )

        for win in self.anomaly_windows:
            if win.video_id not in self.videos:
                raise ValidationError(
                    f"Anomaly window '{win.window_id}' references unknown video '{win.video_id}'"
                )
            v = self.videos[win.video_id]
            if v.duration_seconds is not None and win.end_seconds > v.duration_seconds:
                raise ValidationError(
                    f"Anomaly window '{win.window_id}' interval end {win.end_seconds} "
                    f"exceeds video duration {v.duration_seconds}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "videos": {vid: v.to_dict() for vid, v in self.videos.items()},
            "captions": [c.to_dict() for c in self.captions],
            "anomaly_windows": [w.to_dict() for w in self.anomaly_windows],
            "excluded_videos": dict(self.excluded_videos),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DataManifest:
        videos = {vid: VideoRecord.from_dict(v) for vid, v in data.get("videos", {}).items()}
        captions = [CaptionSegment.from_dict(c) for c in data.get("captions", [])]
        anomaly_windows = [AnomalyWindow.from_dict(w) for w in data.get("anomaly_windows", [])]
        excluded = data.get("excluded_videos", {})
        return cls(
            manifest_version=data.get("manifest_version", "1.0.0"),
            videos=videos,
            captions=captions,
            anomaly_windows=anomaly_windows,
            excluded_videos=excluded,
        )


@dataclass
class SplitPartition:
    """Represents isolated video split assignments for train, validation, and test splits."""
    name: str
    train_video_ids: Set[str] = field(default_factory=set)
    val_video_ids: Set[str] = field(default_factory=set)
    test_video_ids: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.train_video_ids = set(self.train_video_ids)
        self.val_video_ids = set(self.val_video_ids)
        self.test_video_ids = set(self.test_video_ids)

    def validate_disjointness(self) -> None:
        """Enforces video-level isolation: no video may belong to more than one split."""
        train_val = self.train_video_ids & self.val_video_ids
        train_test = self.train_video_ids & self.test_video_ids
        val_test = self.val_video_ids & self.test_video_ids

        if train_val or train_test or val_test:
            leaks = []
            if train_val:
                leaks.append(f"train_vs_val: {sorted(train_val)}")
            if train_test:
                leaks.append(f"train_vs_test: {sorted(train_test)}")
            if val_test:
                leaks.append(f"val_vs_test: {sorted(val_test)}")
            raise ValidationError(
                f"Data leakage detected in partition '{self.name}': " + "; ".join(leaks)
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "train": sorted(self.train_video_ids),
            "val": sorted(self.val_video_ids),
            "test": sorted(self.test_video_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SplitPartition:
        return cls(
            name=data.get("name", "custom_split"),
            train_video_ids=set(data.get("train", [])),
            val_video_ids=set(data.get("val", [])),
            test_video_ids=set(data.get("test", [])),
        )
