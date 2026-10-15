"""Native timestamp-aware video decoding and metadata probing using ffmpeg/ffprobe."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import List, Optional


class VideoDecoderError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails to decode media or inspect streams."""
    pass


@dataclass
class VideoMetadata:
    """Technical container and stream metadata for a verified video file."""
    path: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    total_frames: int


@dataclass
class TimestampedFrame:
    """An extracted image frame anchored with its exact temporal timestamp."""
    frame_index: int
    timestamp_seconds: float
    image_path: Path


class VideoDecoder:
    """Extracts timestamped video frames and inspects media stream properties."""

    def __init__(
        self,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe_path = ffprobe_path or shutil.which("ffprobe") or "ffprobe"

    @property
    def is_available(self) -> bool:
        """Checks if both ffmpeg and ffprobe are executable on PATH."""
        return bool(shutil.which(self.ffmpeg_path) and shutil.which(self.ffprobe_path))

    @staticmethod
    def _parse_frame_rate(fps_str: str) -> float:
        """Parses rational or decimal frame rate strings like '30/1' or '30000/1001'."""
        if not fps_str or not isinstance(fps_str, str):
            return 0.0
        try:
            if "/" in fps_str:
                num_s, den_s = fps_str.split("/", 1)
                num, den = float(num_s), float(den_s)
                if den == 0.0:
                    return 0.0
                return round(num / den, 2)
            val = float(fps_str)
            return round(val, 2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def probe(self, video_path: str | Path) -> VideoMetadata:
        """Inspects video stream parameters using ffprobe."""
        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not self.is_available:
            raise VideoDecoderError("ffprobe is not available on PATH.")

        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
            "-of", "json",
            str(path),
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            info = json.loads(res.stdout.decode("utf-8"))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as err:
            raise VideoDecoderError(f"Failed to probe video '{path}': {err}") from err

        streams = info.get("streams", [])
        if not streams:
            raise VideoDecoderError(f"No video streams found in '{path}'")

        v_stream = streams[0]
        width = int(v_stream.get("width", 0))
        height = int(v_stream.get("height", 0))

        # Determine FPS
        fps = self._parse_frame_rate(v_stream.get("avg_frame_rate", ""))
        if fps <= 0.0:
            fps = self._parse_frame_rate(v_stream.get("r_frame_rate", ""))
        if fps <= 0.0:
            fps = 30.0

        # Determine duration
        duration = 0.0
        if "duration" in v_stream and v_stream["duration"]:
            try:
                duration = float(v_stream["duration"])
            except ValueError:
                duration = 0.0
        if duration <= 0.0 and "format" in info and "duration" in info["format"]:
            try:
                duration = float(info["format"]["duration"])
            except ValueError:
                duration = 0.0

        # Determine total frames
        total_frames = 0
        if "nb_frames" in v_stream and v_stream["nb_frames"]:
            try:
                total_frames = int(v_stream["nb_frames"])
            except ValueError:
                total_frames = 0
        if total_frames <= 0:
            total_frames = int(round(duration * fps))

        return VideoMetadata(
            path=path,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=round(duration, 3),
            total_frames=total_frames,
        )

    def extract_regular_frames(
        self,
        video_path: str | Path,
        target_fps: float = 1.0,
        output_dir: Optional[str | Path] = None,
    ) -> List[TimestampedFrame]:
        """Extracts regularly spaced image frames from a video file at target_fps.

        Returns:
            List of TimestampedFrame records with precise timestamp positions.
        """
        if target_fps <= 0.0:
            raise ValueError("target_fps must be positive")

        path = Path(video_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        if not self.is_available:
            raise VideoDecoderError("ffmpeg is not available on PATH.")

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = Path(tempfile.mkdtemp(prefix="sentinel_frames_"))

        frame_pattern = str(out_path / "frame_%04d.jpg")

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(path),
            "-vf", f"fps={target_fps}",
            "-q:v", "2",
            frame_pattern,
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as err:
            raise VideoDecoderError(f"ffmpeg frame extraction failed: {err.stderr.decode('utf-8', errors='replace')}") from err

        frame_files = sorted(out_path.glob("frame_*.jpg"))
        results: List[TimestampedFrame] = []

        for idx, img_path in enumerate(frame_files):
            timestamp = round(idx / float(target_fps), 3)
            results.append(
                TimestampedFrame(
                    frame_index=idx,
                    timestamp_seconds=timestamp,
                    image_path=img_path,
                )
            )

        return results
