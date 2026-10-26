from PIL import Image

from sam_engine.contract import SegmentedObject
from sam_engine.postprocess import MIN_AREA_RATIO, filter_noise, make_cutout


def test_make_cutout_transparent_and_trimmed(sheet_image, tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (300, 200), 0)
    for x in range(50, 150):
        for y in range(50, 150):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    out_path, bbox = make_cutout(sheet_image, str(mask_path),
                                 tmp_path / "cut.png", feather_px=0, padding=16)
    with Image.open(out_path) as cut:
        assert cut.mode == "RGBA"
        # 100x100 region + 16px padding each side, clamped to image.
        assert cut.size == (132, 132)
        # Corners outside the mask are transparent.
        assert cut.getpixel((0, 0))[3] == 0
        assert cut.getpixel((66, 66))[3] == 255
    assert bbox == (34, 34, 132, 132)


def test_filter_noise():
    objs = [
        SegmentedObject(index=0, bbox=(0, 0, 10, 10), mask_png_path="m.png",
                        area_ratio=0.9),
        SegmentedObject(index=1, bbox=(0, 0, 1, 1), mask_png_path="m.png",
                        area_ratio=MIN_AREA_RATIO / 2),
    ]
    assert [o.index for o in filter_noise(objs)] == [0]
