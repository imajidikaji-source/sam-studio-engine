"""Fake segmentation provider — deterministic stand-in for the real API.

Grid masks only, no randomness: the same image + mode always yields the
same objects. Lets the UI, job service and tests run without API access
(spec §9 step 1: "skeleton provider + contract + fake provider").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .contract import SegmentedObject, SegmentationRequest


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)


class FakeSegmentationProvider:
    """Deterministic grid segmentation."""

    async def segment(
        self, req: SegmentationRequest, out_dir: Path
    ) -> list[SegmentedObject]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(req.image_path) as image:
            img_w, img_h = image.size
        if req.mode == "subject":
            boxes = [(img_w // 10, img_h // 10, img_w * 8 // 10, img_h * 8 // 10)]
        elif req.mode == "parts":
            boxes = [
                (0, 0, img_w // 2, img_h // 2),
                (img_w // 2, 0, img_w - img_w // 2, img_h // 2),
                (0, img_h // 2, img_w // 2, img_h - img_h // 2),
                (img_w // 2, img_h // 2, img_w - img_w // 2, img_h - img_h // 2),
            ]
        else:  # sheet: 3x2 grid
            cols, rows = 3, 2
            cw, ch = img_w // cols, img_h // rows
            boxes = [
                (c * cw, r * ch, cw if c < cols - 1 else img_w - c * cw,
                 ch if r < rows - 1 else img_h - r * ch)
                for r in range(rows)
                for c in range(cols)
            ]
        boxes = boxes[: req.max_objects]
        objects: list[SegmentedObject] = []
        for index, (x, y, w, h) in enumerate(boxes):
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            mask[y : y + h, x : x + w] = 1
            mask_path = out_dir / f"mask_{index:02d}.png"
            _write_mask(mask_path, mask)
            area = int(mask.sum())
            objects.append(
                SegmentedObject(
                    index=index,
                    bbox=(x, y, w, h),
                    mask_png_path=str(mask_path),
                    area_ratio=area / (img_w * img_h),
                    score=None,
                )
            )
        return objects
