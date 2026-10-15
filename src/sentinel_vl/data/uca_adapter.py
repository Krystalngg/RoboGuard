"""UCA (UCF-Crime Annotation / CVPR 2024) Data Adapter for Sentinel-VL.

Parses authentic video annotations and maps them into internal VideoRecord
and CaptionSegment schemas with temporal validity checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sentinel_vl.data.schemas import (
    CaptionSegment,
    DataManifest,
    ValidationError,
    VideoRecord,
)


class UCAAdapter:
    """Adapter for parsing authentic CVPR 2024 UCA surveillance annotations."""

    IS_VERIFIED: bool = True
    VERIFIED_SOURCE: str = "CVPR 2024 (UCF-Crime Annotation benchmark)"

    @classmethod
    def parse_json_content(
        cls,
        data: Dict[str, Any],
    ) -> Tuple[Dict[str, VideoRecord], List[CaptionSegment]]:
        """Parses structured JSON annotations into VideoRecord and CaptionSegment instances."""
        videos: Dict[str, VideoRecord] = {}
        captions: List[CaptionSegment] = []

        for video_id, info in data.items():
            duration = float(info["duration"]) if "duration" in info and info["duration"] is not None else None
            timestamps = info.get("timestamps", [])
            sentences = info.get("sentences", [])

            if len(timestamps) != len(sentences):
                raise ValidationError(
                    f"timestamp count ({len(timestamps)}) != sentence count ({len(sentences)}) "
                    f"for video '{video_id}'"
                )

            videos[video_id] = VideoRecord(
                video_id=video_id,
                source_path=f"data/videos/{video_id}.mp4",
                duration_seconds=duration,
                decoding_status="unverified",
            )

            for idx, (ts, sentence) in enumerate(zip(timestamps, sentences)):
                segment_id = f"{video_id}_seg_{idx:03d}"
                start_sec = float(ts[0])
                end_sec = float(ts[1])
                captions.append(
                    CaptionSegment(
                        segment_id=segment_id,
                        video_id=video_id,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                        caption=str(sentence).strip(),
                        provenance="uca_human_reference",
                    )
                )

        return videos, captions

    @classmethod
    def parse_txt_content(
        cls,
        text: str,
    ) -> Tuple[Dict[str, VideoRecord], List[CaptionSegment]]:
        """Parses line-delimited text annotations (`<video_id> <start> <end> ##<caption>`)."""
        videos: Dict[str, VideoRecord] = {}
        captions: List[CaptionSegment] = []
        video_durations: Dict[str, float] = {}

        for line_idx, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            if "##" not in line:
                raise ValidationError(
                    f"missing '##' description delimiter on line {line_idx}: '{line}'"
                )

            meta_part, caption_part = line.split("##", 1)
            tokens = meta_part.strip().split()
            if len(tokens) < 3:
                raise ValidationError(
                    f"Invalid metadata format on line {line_idx}: expected '<video_id> <start> <end>', "
                    f"got '{meta_part}'"
                )

            video_id = tokens[0]
            try:
                start_sec = float(tokens[1])
                end_sec = float(tokens[2])
            except ValueError as err:
                raise ValidationError(f"Invalid timestamp on line {line_idx}: {err}") from err

            caption_text = caption_part.strip()
            segment_id = f"{video_id}_txt_{line_idx:03d}"

            captions.append(
                CaptionSegment(
                    segment_id=segment_id,
                    video_id=video_id,
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    caption=caption_text,
                    provenance="uca_human_reference",
                )
            )

            current_max = video_durations.get(video_id, 0.0)
            if end_sec > current_max:
                video_durations[video_id] = end_sec

        for video_id, max_end in video_durations.items():
            videos[video_id] = VideoRecord(
                video_id=video_id,
                source_path=f"data/videos/{video_id}.mp4",
                duration_seconds=round(max_end, 1),
                decoding_status="unverified",
            )

        return videos, captions

    @classmethod
    def parse_file(cls, path: str | Path) -> DataManifest:
        """Parses a JSON or TXT file into a validated DataManifest."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Annotation file not found: {p}")

        suffix = p.suffix.lower()
        if suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            videos, captions = cls.parse_json_content(data)
        elif suffix in (".txt", ".csv"):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            videos, captions = cls.parse_txt_content(content)
        else:
            # Try JSON first, fallback to TXT
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                data = json.loads(content)
                videos, captions = cls.parse_json_content(data)
            except (json.JSONDecodeError, AttributeError):
                videos, captions = cls.parse_txt_content(content)

        manifest = DataManifest(
            manifest_version="1.0.0",
            videos=videos,
            captions=captions,
            anomaly_windows=[],
        )
        manifest.validate_integrity()
        return manifest
