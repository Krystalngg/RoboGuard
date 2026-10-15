"""Sentinel-VL Data Module: Schemas, Manifests, Adapters, Split Auditing, and Decoding."""

from sentinel_vl.data.manifest import ManifestManager
from sentinel_vl.data.schemas import (
    AnomalyWindow,
    CaptionSegment,
    DataManifest,
    SplitPartition,
    ValidationError,
    VideoRecord,
)
from sentinel_vl.data.split_auditor import AuditResult, SplitAuditor
from sentinel_vl.data.uca_adapter import UCAAdapter
from sentinel_vl.data.video_decoder import (
    TimestampedFrame,
    VideoDecoder,
    VideoDecoderError,
    VideoMetadata,
)

__all__ = [
    "AnomalyWindow",
    "AuditResult",
    "CaptionSegment",
    "DataManifest",
    "ManifestManager",
    "SplitAuditor",
    "SplitPartition",
    "TimestampedFrame",
    "UCAAdapter",
    "ValidationError",
    "VideoDecoder",
    "VideoDecoderError",
    "VideoMetadata",
    "VideoRecord",
]
