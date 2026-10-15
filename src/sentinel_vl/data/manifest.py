"""Manifest management, persistence, disk file verification, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple
from sentinel_vl.data.schemas import DataManifest
from sentinel_vl.data.uca_adapter import UCAAdapter


class ManifestManager:
    """Manages serializing, loading, disk integrity verification, and conversion of manifests."""

    @staticmethod
    def save_manifest(manifest: DataManifest, path: str | Path) -> None:
        """Saves a validated DataManifest to disk in JSON format."""
        manifest.validate_integrity()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    @staticmethod
    def load_manifest(path: str | Path) -> DataManifest:
        """Loads a DataManifest from disk and validates internal consistency."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = DataManifest.from_dict(data)
        manifest.validate_integrity()
        return manifest

    @staticmethod
    def verify_files_on_disk(
        manifest: DataManifest,
        base_dir: Optional[str | Path] = None,
    ) -> Tuple[int, List[str]]:
        """Verifies that video source files referenced by the manifest physically exist on disk.

        Returns:
            Tuple of (found_count, missing_video_ids)
        """
        found_count = 0
        missing_ids: List[str] = []

        base = Path(base_dir) if base_dir else Path.cwd()

        for vid, record in manifest.videos.items():
            p = Path(record.source_path)
            if not p.is_absolute():
                p = base / p

            if p.is_file():
                found_count += 1
            else:
                missing_ids.append(vid)

        return found_count, missing_ids

    @staticmethod
    def build_manifest_from_uca(source_path: str | Path) -> DataManifest:
        """Builds a DataManifest from authentic UCA annotations."""
        return UCAAdapter.parse_file(source_path)
