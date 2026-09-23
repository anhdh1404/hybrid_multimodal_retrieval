from __future__ import annotations

from pathlib import Path

import pandas as pd


class Flickr30KDataset:
    """Loader for a Flickr30K-style images directory and results.csv."""

    def __init__(self, image_dir: str | Path, captions_csv: str | Path):
        self.image_dir = Path(image_dir)
        self.captions_csv = Path(captions_csv)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.captions_csv.exists():
            raise FileNotFoundError(f"CSV not found: {self.captions_csv}")

        self.captions = pd.read_csv(self.captions_csv)

    def image_paths(self) -> list[Path]:
        return sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )

    def __len__(self) -> int:
        return len(self.image_paths())
