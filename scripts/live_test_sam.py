#!/usr/bin/env python3
"""First live SAM 3.1 segmentation test (Sprint 1 verification).

Uses the Meta Model API connector credential via the surrogate mechanism
(same pattern as meta-api-probe.py) — the key never appears here, in
arguments, or in logs. One image, ~$0.0025, covered by free credits.

Saves the test image, raw output_text, mask PNGs and cutouts under
live-test-out/ next to this script's project root.
"""

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request, read_json_response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw

from sam_engine.meta_provider import extract_output_text
from sam_engine.postprocess import make_cutout
from sam_engine.sam_text import decode_record_mask, parse_output_text

BASE = "https://api.meta.ai"
HOSTS = ["api.meta.ai"]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise RuntimeError("redirect refused, verify the endpoint")


def make_test_image(path: Path) -> None:
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([50, 50, 150, 150], fill="red")
    draw.rectangle([220, 60, 340, 180], fill="blue")
    draw.polygon([(100, 220), (200, 220), (150, 130)], fill="green")
    image.save(path)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "live-test-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "test-sheet.png"
    make_test_image(image_path)
    data_url = ("data:image/png;base64,"
                + base64.b64encode(image_path.read_bytes()).decode())
    payload = {
        "model": "sam-3.1",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "geometric shapes"},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        "metadata": {"include_confidence": "true"},
    }
    req = urllib.request.Request(
        BASE + "/v1/responses",
        data=json.dumps(payload).encode(),
        headers={**UA, "Content-Type": "application/json"},
        method="POST",
    )
    add_surrogate_to_request(req, "custom.meta-model-api", allowed_hosts=HOSTS)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=180) as resp:
            status, body = resp.status, read_json_response(resp)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "{}")
        except Exception:
            body = {}
        status = e.code
    print("HTTP", status)
    if status >= 400:
        print(json.dumps(body)[:2000])
        return 1
    output_text = extract_output_text(body)
    if output_text is None:
        print("NO output_text lane. Body keys:", list(body.keys())[:20])
        print(json.dumps(body)[:2000])
        return 1
    (out_dir / "output_text.txt").write_text(output_text)
    print("output_text chars:", len(output_text))
    records, diagnostics = parse_output_text(output_text)
    print("records:", len(records), "diagnostics:", diagnostics)
    with Image.open(image_path) as image:
        img_w, img_h = image.size
    import numpy as np

    for index, record in enumerate(records):
        raster = decode_record_mask(record)
        raster_arr = np.frombuffer(raster, dtype=np.uint8).reshape(
            record.mask_height, record.mask_width)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        y1 = min(img_h, record.y + record.mask_height)
        x1 = min(img_w, record.x + record.mask_width)
        full_mask[record.y:y1, record.x:x1] = raster_arr[: y1 - record.y, : x1 - record.x]
        mask_path = out_dir / f"mask_{index:02d}.png"
        Image.fromarray((full_mask * 255).astype(np.uint8), mode="L").save(mask_path)
        cutout_path, _ = make_cutout(str(image_path), str(mask_path),
                                     out_dir / f"cutout_{index:02d}.png")
        print(f"  obj {record.object_id}: box=({record.x},{record.y},"
              f"{record.w},{record.h}) conf={record.confidence} "
              f"mask={record.mask_width}x{record.mask_height} "
              f"area={int(full_mask.sum()) / (img_w * img_h):.3f} -> {cutout_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
