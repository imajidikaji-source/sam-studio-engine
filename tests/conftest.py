"""Shared fixtures: synthetic images generated with PIL (no external assets)."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sheet_image(tmp_path: Path) -> str:
    """A 300x200 sheet with 6 colored cells (fake-provider grid friendly)."""
    path = tmp_path / "sheet.png"
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    k = 0
    for row in range(2):
        for col in range(3):
            draw.rectangle(
                [col * 100, row * 100, (col + 1) * 100 - 1, (row + 1) * 100 - 1],
                fill=colors[k],
            )
            k += 1
    image.save(path)
    return str(path)


@pytest.fixture
def tiny_image(tmp_path: Path) -> str:
    path = tmp_path / "tiny.png"
    Image.new("RGB", (10, 10), "white").save(path)
    return str(path)
