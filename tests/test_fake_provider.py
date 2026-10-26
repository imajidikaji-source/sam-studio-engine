import asyncio
from pathlib import Path

from PIL import Image

from sam_engine.contract import SegmentationRequest
from sam_engine.fake_provider import FakeSegmentationProvider


def test_fake_sheet_grid(sheet_image, tmp_path):
    provider = FakeSegmentationProvider()
    req = SegmentationRequest(image_path=sheet_image, prompt="every sticker")
    objects = asyncio.run(provider.segment(req, tmp_path / "masks"))
    assert len(objects) == 6
    for obj in objects:
        assert Path(obj.mask_png_path).is_file()
        with Image.open(obj.mask_png_path) as mask:
            assert mask.size == (300, 200)
    total_ratio = sum(obj.area_ratio for obj in objects)
    assert abs(total_ratio - 1.0) < 1e-6


def test_fake_deterministic(sheet_image, tmp_path):
    provider = FakeSegmentationProvider()
    req = SegmentationRequest(image_path=sheet_image, prompt="every sticker")
    first = asyncio.run(provider.segment(req, tmp_path / "a"))
    second = asyncio.run(provider.segment(req, tmp_path / "b"))
    assert [o.bbox for o in first] == [o.bbox for o in second]


def test_fake_subject_and_cap(sheet_image, tmp_path):
    provider = FakeSegmentationProvider()
    subject = asyncio.run(provider.segment(
        SegmentationRequest(image_path=sheet_image, prompt="main object",
                            mode="subject"),
        tmp_path / "s"))
    assert len(subject) == 1
    capped = asyncio.run(provider.segment(
        SegmentationRequest(image_path=sheet_image, prompt="x", max_objects=2),
        tmp_path / "c"))
    assert len(capped) == 2
