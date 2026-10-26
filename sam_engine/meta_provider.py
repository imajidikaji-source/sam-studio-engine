"""Meta Model API provider — SAM 3.1 via the hosted Responses API.

What is VERIFIED (official meta-sam protocol, protocol/sam3.md):
- Base URL default ``https://api.meta.ai/v1``; model id ``sam-3.1``
  (both confirmed by the official api-playground README).
- Output grammar: one output_text lane, one line per frame, image = frame
  0, ``<0f>id<|box;...|><|mask;...|>`` tokens, mask data ``H,W,payload``
  with base85 arithmetic-coded payload (``~`` lossless / ``!`` one_bit).
- Input: ``input_text`` is a noun phrase (our prompt) OR object prompts,
  never both. ``metadata.include_confidence = "true"`` asks for ``c``.
- Mask decoding via the official ``meta-sam-parser`` package.

What is NOT yet verified and must be confirmed on the first live call:
the exact Responses request envelope (field names for the inline image).
This module follows the OpenAI-compatible Responses convention
(``input`` with ``input_text`` + ``input_image`` parts carrying a data URL).
If the first live call rejects the envelope, adjust ONLY this module —
the parser, decoder and job service are envelope-independent.

UPDATE 2026-10-26 (live test, scripts/live_test_sam.py): the envelope is
CONFIRMED — HTTP 200, status completed, images_processed=1, output_text
with 3 records for a 3-shape synthetic image. Prompt wording MATTERS:
concrete noun phrases work (``geometric shapes`` -> 3 objects with
confidences ~0.74-0.77); abstract quantifiers (``every shape``,
``all objects``) and comma-lists of nouns returned EMPTY output_text.
Sheet Splitter default prompts must be concrete nouns (validate the
exact sticker-sheet prompt in Sprint 2, e.g. ``stickers``).

Billing: a 402 maps to ``BillingNotConfiguredError`` (the error seen on
2026-10-23 before a payment method existed). No call is made without an
API key; the key comes from ``META_MODEL_API_KEY`` or the constructor,
never from a file in this repo.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .contract import SegmentedObject, SegmentationRequest
from .sam_text import decode_record_mask, parse_output_text


class MetaApiError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class BillingNotConfiguredError(MetaApiError):
    """HTTP 402 — no usable billing/credits on the account."""


def extract_output_text(payload: dict) -> str | None:
    """Find the output_text lane in a Responses API response body."""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    return None


class MetaModelApiProvider:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.meta.ai/v1",
        model: str = "sam-3.1",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 180.0,
    ):
        if not api_key:
            raise ValueError("Meta Model API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = http_client
        self.timeout = timeout
        # SAM 3.1 hosted image price, developer.meta.com model page.
        self.cost_per_image_usd = 2.50 / 1000

    async def segment(
        self, req: SegmentationRequest, out_dir: Path
    ) -> list[SegmentedObject]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_bytes = Path(req.image_path).read_bytes()
        mime, _ = mimetypes.guess_type(req.image_path)
        data_url = (
            f"data:{mime or 'image/png'};base64,"
            f"{base64.b64encode(image_bytes).decode()}"
        )
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": req.prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "metadata": {"include_confidence": "true"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        close_client = self._client is None
        response = None
        last_error: Exception | None = None
        try:
            # Transient disconnects happen on this network (see AGENTS.md:
            # one retry usually connects). Retry transport errors only —
            # HTTP error statuses are real answers, not retries.
            for attempt in range(3):
                try:
                    response = await client.post(
                        f"{self.base_url}/responses", json=body, headers=headers
                    )
                    break
                except httpx.TransportError as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))
            if response is None:
                raise MetaApiError(
                    f"Meta Model API unreachable after retries: {last_error}"
                )
        finally:
            if close_client:
                await client.aclose()
        if response.status_code == 402:
            raise BillingNotConfiguredError(
                "Billing verification failed (HTTP 402). Check payment method / credits."
            )
        if response.status_code >= 400:
            raise MetaApiError(
                f"Meta Model API error {response.status_code}: "
                f"{response.text[:500]}",
                status_code=response.status_code,
            )
        output_text = extract_output_text(response.json())
        if output_text is None:
            raise MetaApiError("Response contained no output_text lane")
        records, _diagnostics = parse_output_text(output_text)
        with Image.open(req.image_path) as image:
            img_w, img_h = image.size
        objects: list[SegmentedObject] = []
        for index, record in enumerate(records[: req.max_objects]):
            raster = decode_record_mask(record)
            raster_arr = np.frombuffer(raster, dtype=np.uint8).reshape(
                record.mask_height, record.mask_width
            )
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # Mask raster is box-sized, positioned at the box origin.
            y0, x0 = record.y, record.x
            y1 = min(img_h, y0 + record.mask_height)
            x1 = min(img_w, x0 + record.mask_width)
            full_mask[y0:y1, x0:x1] = raster_arr[: y1 - y0, : x1 - x0]
            mask_path = out_dir / f"mask_{index:02d}.png"
            Image.fromarray((full_mask * 255).astype(np.uint8), mode="L").save(
                mask_path
            )
            area = int(full_mask.sum())
            objects.append(
                SegmentedObject(
                    index=index,
                    bbox=(record.x, record.y, record.w, record.h),
                    mask_png_path=str(mask_path),
                    area_ratio=area / (img_w * img_h),
                    score=record.confidence,
                )
            )
        return objects
