"""Provider contract — mirrors spec §3 (spek-teknis-sam-studio-engine.md).

One deliberate deviation from the spec sketch: ``segment`` takes an
``out_dir`` so implementations have somewhere to write mask PNGs. The
spec sketch shows ``segment(req)`` only; file outputs need a directory,
and every caller in this package passes it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

Mode = Literal["sheet", "subject", "parts"]

_MODES = ("sheet", "subject", "parts")


@dataclass(frozen=True)
class SegmentationRequest:
    image_path: str
    prompt: str  # text concept, e.g. "every sticker on this sheet"
    mode: Mode = "sheet"
    max_objects: int = 64

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if not 1 <= self.max_objects <= 512:
            raise ValueError(f"max_objects must be 1..512, got {self.max_objects}")
        if not Path(self.image_path).is_file():
            raise ValueError(f"image not found: {self.image_path}")


@dataclass(frozen=True)
class SegmentedObject:
    index: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in original pixels
    mask_png_path: str  # raw mask from the provider, grayscale, full-image size
    area_ratio: float  # mask area / image area
    score: float | None = None  # provider confidence, when supplied
    # Filled by the job runner (post-processing), not by the provider:
    cutout_png_path: str | None = None
    status: str = "ok"  # ok | review | failed

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"invalid bbox {self.bbox}")
        if not 0.0 <= self.area_ratio <= 1.0:
            raise ValueError(f"area_ratio out of range: {self.area_ratio}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score out of range: {self.score}")


class SegmentationProvider(Protocol):
    async def segment(
        self, req: SegmentationRequest, out_dir: Path
    ) -> list[SegmentedObject]: ...
