#!/usr/bin/env python3
"""Generate the Sprint 2 sticker test set (10 synthetic sheets).

Writes tests/fixtures/sam/sticker-*.png + ground-truth.json. Shapes are
drawn with PIL only — no user assets, no external downloads. Expected
counts are the shapes actually drawn; the hard cases (touching, tiny,
white-on-white, shadows) are deliberate edge cases from the concept doc.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "sam"
W, H = 600, 400


def star_points(cx, cy, r_outer, r_inner, points=5):
    pts = []
    for i in range(points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi * i / points - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def draw_shape(draw, kind, box, color):
    x0, y0, x1, y1 = box
    if kind == "circle":
        draw.ellipse(box, fill=color)
    elif kind == "rect":
        draw.rectangle(box, fill=color)
    elif kind == "triangle":
        draw.polygon([(x0 + (x1 - x0) // 2, y0), (x0, y1), (x1, y1)], fill=color)
    elif kind == "star":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        r = min(x1 - x0, y1 - y0) // 2
        draw.polygon(star_points(cx, cy, r, r // 2), fill=color)


def sheet(name, shapes, shadow=False, white_on_white=False):
    """shapes: list of (kind, box, color). Returns expected count."""
    base = Image.new("RGB", (W, H), "white")
    if shadow:
        shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        for kind, box, _color in shapes:
            shifted = (box[0] + 8, box[1] + 8, box[2] + 8, box[3] + 8)
            draw_shape(shadow_draw, kind, shifted, (120, 120, 120, 255))
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(6))
        base = Image.alpha_composite(base.convert("RGBA"), shadow_layer).convert("RGB")
    draw = ImageDraw.Draw(base)
    for kind, box, color in shapes:
        if white_on_white:
            draw_shape(draw, kind, box, (255, 255, 255))
            # Faint outline only — the hard case.
            outline = ImageDraw.Draw(base)
            if kind == "circle":
                outline.ellipse(box, outline=(215, 215, 215), width=2)
            elif kind == "rect":
                outline.rectangle(box, outline=(215, 215, 215), width=2)
            else:
                outline.polygon(
                    [(box[0] + (box[2] - box[0]) // 2, box[1]), (box[0], box[3]),
                     (box[2], box[3])], outline=(215, 215, 215), width=2)
        else:
            draw_shape(draw, kind, box, color)
    base.save(OUT / f"{name}.png")
    return len(shapes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    colors = ["red", "blue", "green", "orange", "purple", "teal",
              "magenta", "brown", "navy"]
    kinds = ["circle", "rect", "triangle", "star"]
    ground_truth = []

    def grid_sheet(name, rows, cols, notes=""):
        shapes = []
        cw, ch = W // cols, H // rows
        k = 0
        for r in range(rows):
            for c in range(cols):
                pad = 18
                box = (c * cw + pad, r * ch + pad,
                       (c + 1) * cw - pad, (r + 1) * ch - pad)
                shapes.append((kinds[k % len(kinds)], box, colors[k % len(colors)]))
                k += 1
        count = sheet(name, shapes)
        ground_truth.append({"file": f"{name}.png", "expected_objects": count,
                             "notes": notes or f"clean {rows}x{cols} grid"})

    grid_sheet("sticker-01", 2, 2)
    grid_sheet("sticker-02", 2, 3)
    grid_sheet("sticker-03", 2, 3)
    grid_sheet("sticker-04", 3, 3)
    grid_sheet("sticker-05", 2, 4)
    grid_sheet("sticker-06", 3, 3)

    # Touching objects: shapes share edges / overlap slightly.
    touching = [
        ("circle", (60, 60, 220, 220), "red"),
        ("rect", (200, 80, 360, 240), "blue"),
        ("triangle", (330, 60, 480, 220), "green"),
        ("star", (140, 230, 300, 380), "orange"),
    ]
    ground_truth.append({"file": "sticker-07.png",
                         "expected_objects": sheet("sticker-07", touching),
                         "notes": "touching/overlapping objects (edge case)"})

    # Tiny objects (<2% of image area each) + 2 normal ones.
    tiny = [
        ("circle", (40, 40, 70, 70), "red"),
        ("rect", (150, 50, 180, 80), "blue"),
        ("triangle", (300, 40, 335, 75), "green"),
        ("star", (450, 50, 485, 85), "orange"),
        ("circle", (80, 250, 115, 285), "purple"),
        ("rect", (400, 260, 435, 295), "teal"),
        ("circle", (200, 150, 320, 270), "navy"),
        ("rect", (380, 150, 520, 290), "magenta"),
    ]
    ground_truth.append({"file": "sticker-08.png",
                         "expected_objects": sheet("sticker-08", tiny),
                         "notes": "6 tiny objects (<2% area) + 2 normal (edge case)"})

    # White-on-white: white shapes, faint outline only.
    white_shapes = [
        ("circle", (60, 60, 200, 200), "white"),
        ("rect", (260, 70, 420, 210), "white"),
        ("triangle", (120, 240, 280, 380), "white"),
        ("star", (350, 230, 500, 380), "white"),
    ]
    ground_truth.append({"file": "sticker-09.png",
                         "expected_objects": sheet("sticker-09", white_shapes,
                                                   white_on_white=True),
                         "notes": "HARD: white shapes on white background, faint outline"})

    # Soft drop shadows.
    shadow_shapes = [
        ("circle", (60, 60, 200, 200), "red"),
        ("rect", (260, 70, 420, 210), "blue"),
        ("triangle", (120, 240, 280, 380), "green"),
        ("star", (350, 230, 500, 380), "orange"),
        ("circle", (430, 40, 540, 150), "purple"),
    ]
    ground_truth.append({"file": "sticker-10.png",
                         "expected_objects": sheet("sticker-10", shadow_shapes,
                                                   shadow=True),
                         "notes": "soft drop shadows (edge case)"})

    (OUT / "ground-truth.json").write_text(
        json.dumps(ground_truth, indent=2) + "\n")
    total = sum(entry["expected_objects"] for entry in ground_truth)
    print(f"wrote {len(ground_truth)} sheets, {total} expected objects -> {OUT}")


if __name__ == "__main__":
    main()
