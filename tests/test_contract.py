import pytest

from sam_engine.contract import SegmentedObject, SegmentationRequest


def test_valid_request(sheet_image):
    req = SegmentationRequest(image_path=sheet_image, prompt="every sticker")
    assert req.mode == "sheet"
    assert req.max_objects == 64


def test_empty_prompt_rejected(sheet_image):
    with pytest.raises(ValueError):
        SegmentationRequest(image_path=sheet_image, prompt="  ")


def test_bad_mode_rejected(sheet_image):
    with pytest.raises(ValueError):
        SegmentationRequest(image_path=sheet_image, prompt="x", mode="bogus")


def test_max_objects_bounds(sheet_image):
    with pytest.raises(ValueError):
        SegmentationRequest(image_path=sheet_image, prompt="x", max_objects=0)
    with pytest.raises(ValueError):
        SegmentationRequest(image_path=sheet_image, prompt="x", max_objects=513)


def test_missing_image_rejected():
    with pytest.raises(ValueError):
        SegmentationRequest(image_path="/nope/missing.png", prompt="x")


def test_segmented_object_validation():
    with pytest.raises(ValueError):
        SegmentedObject(index=0, bbox=(0, 0, 0, 10), mask_png_path="m.png",
                        area_ratio=0.5)
    with pytest.raises(ValueError):
        SegmentedObject(index=0, bbox=(0, 0, 10, 10), mask_png_path="m.png",
                        area_ratio=1.5)
    with pytest.raises(ValueError):
        SegmentedObject(index=0, bbox=(0, 0, 10, 10), mask_png_path="m.png",
                        area_ratio=0.5, score=2.0)
    obj = SegmentedObject(index=0, bbox=(0, 0, 10, 10), mask_png_path="m.png",
                          area_ratio=0.5, score=0.9)
    assert obj.status == "ok"
