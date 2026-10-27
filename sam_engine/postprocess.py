"""Local post-processing (spec §5, Sprint 1 subset).

Implemented now: noise filtering by area ratio + mask -> transparent PNG
cutout (feather + trim + padding). SVG silhouette export lands with the
Sheet Splitter sprint, via meta-sam-parser's ``decode_mask_to_svg_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .contract import SegmentedObject

MIN_AREA_RATIO = 0.0005  # spec §5 step 1: drop components < 0.05% of image


def filter_noise(
    objects: list[SegmentedObject], min_area_ratio: float = MIN_AREA_RATIO
) -> list[SegmentedObject]:
    return [obj for obj in objects if obj.area_ratio >= min_area_ratio]


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected foreground component (BFS)."""
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best: list[tuple[int, int]] = []
    for start_y in range(height):
        for start_x in range(width):
            if binary[start_y, start_x] == 0 or visited[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            component: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and binary[ny, nx] and not visited[ny, nx]):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) > len(best):
                best = component
    out = np.zeros_like(binary)
    for y, x in best:
        out[y, x] = 1
    return out


def _trace_contour(binary: np.ndarray) -> list[tuple[int, int]]:
    """Moore-Neighbour boundary tracing of the outer contour.

    Returns pixel-corner coordinates (x, y) of the boundary polygon.
    Empty input -> empty list.
    """
    height, width = binary.shape
    start: tuple[int, int] | None = None
    for y in range(height):
        for x in range(width):
            if binary[y, x]:
                start = (y, x)
                break
        if start is not None:
            break
    if start is None:
        return []
    # 8-neighbourhood clockwise starting east: E, SE, S, SW, W, NW, N, NE.
    directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    contour: list[tuple[int, int]] = []
    current = start
    backtrack_dir = 7  # enter from the north (as if coming from NE)
    while True:
        contour.append((current[1], current[0]))  # (x, y)
        # Search neighbours clockwise starting after the backtrack direction.
        found: tuple[int, int] | None = None
        found_dir = -1
        for step in range(8):
            direction = directions[(backtrack_dir + 1 + step) % 8]
            ny, nx = current[0] + direction[0], current[1] + direction[1]
            if 0 <= ny < height and 0 <= nx < width and binary[ny, nx]:
                found = (ny, nx)
                found_dir = (backtrack_dir + 1 + step) % 8
                break
        if found is None:
            break
        # Backtrack direction for the next pixel: opposite side of entry.
        backtrack_dir = (found_dir + 4) % 8
        if found == start and len(contour) > 1:
            break
        if found == start:
            break
        current = found
        if len(contour) > width * height + 10:  # safety valve
            break
    return contour


def _douglas_peucker(points: list[tuple[int, int]], tolerance: float) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return points

    def perpendicular_distance(point, line_start, line_end) -> float:
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return float(((x - x1) ** 2 + (y - y1) ** 2) ** 0.5)
        return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / (dx * dx + dy * dy) ** 0.5

    def simplify(sub: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(sub) <= 2:
            return sub
        farthest = max(
            range(1, len(sub) - 1),
            key=lambda i: perpendicular_distance(sub[i], sub[0], sub[-1]),
        )
        if perpendicular_distance(sub[farthest], sub[0], sub[-1]) <= tolerance:
            return [sub[0], sub[-1]]
        left = simplify(sub[: farthest + 1])
        right = simplify(sub[farthest:])
        return left[:-1] + right

    return simplify(points)


def mask_png_to_svg(
    mask_png_path: str | Path,
    out_svg_path: str | Path,
    tolerance: float = 1.5,
) -> Path:
    """Trace a mask PNG to a single-color SVG silhouette path.

    Uses the largest connected component only (stray specks are dropped).
    Raises ValueError on an empty mask.
    """
    out_svg_path = Path(out_svg_path)
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(mask_png_path) as mask_image:
        width, height = mask_image.size
        binary = (np.asarray(mask_image.convert("L")) > 127).astype(np.uint8)
    binary = _largest_component(binary)
    contour = _trace_contour(binary)
    if not contour:
        raise ValueError(f"empty mask: {mask_png_path}")
    simplified = _douglas_peucker(contour, tolerance)
    if len(simplified) < 3:
        simplified = contour[:3] if len(contour) >= 3 else contour
    path_data = "M " + " L ".join(f"{x} {y}" for x, y in simplified) + " Z"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}">'
        f'<path d="{path_data}" fill="black"/></svg>\n'
    )
    out_svg_path.write_text(svg)
    return out_svg_path


def _smooth_mask_edges(mask: Image.Image) -> Image.Image:
    """Soften raster stair-steps on mask edges (supersampling).

    SAM masks are raster: diagonal curves come out as visible steps at
    full resolution, and a 1px feather only blurs the steps instead of
    removing them. Upscaling 2x with LANCZOS, blurring at that scale,
    then downscaling moves the effective edge by less than a pixel
    while replacing steps with a smooth gradient. Verified on the
    mosque test image (2026-10-27): dome curves smooth, thin minaret
    spires intact — unlike a plain feather 3, which haloes and erodes
    thin structures.
    """
    big = mask.resize((mask.width * 2, mask.height * 2), Image.LANCZOS)
    big = big.filter(ImageFilter.GaussianBlur(2))
    return big.resize(mask.size, Image.LANCZOS)


def make_cutout(
    image_path: str,
    mask_png_path: str,
    out_path: str | Path,
    feather_px: int = 1,
    padding: int = 16,
) -> tuple[Path, tuple[int, int, int, int]]:
    """Apply a grayscale mask to an image and save a trimmed transparent PNG.

    Returns (out_path, trimmed bbox x,y,w,h including padding, clamped to
    the image bounds).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        img_w, img_h = image.size
    with Image.open(mask_png_path) as mask_image:
        mask = mask_image.convert("L")
    if mask.size != (img_w, img_h):
        mask = mask.resize((img_w, img_h))
    # Geometry (bbox/trim) comes from the RAW mask so smoothing never
    # shifts or grows the cutout; only the alpha edge is smoothed.
    bbox = mask.getbbox()
    if bbox is None:
        # Empty mask: save a fully transparent 1x1 PNG rather than failing.
        out = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        out.save(out_path)
        return out_path, (0, 0, 1, 1)
    mask = _smooth_mask_edges(mask)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
    rgba.putalpha(mask)
    x0 = max(0, bbox[0] - padding)
    y0 = max(0, bbox[1] - padding)
    x1 = min(img_w, bbox[2] + padding)
    y1 = min(img_h, bbox[3] + padding)
    rgba.crop((x0, y0, x1, y1)).save(out_path)
    return out_path, (x0, y0, x1 - x0, y1 - y0)
