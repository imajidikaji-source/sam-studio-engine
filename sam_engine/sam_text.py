"""Parser for the SAM 3.1 Responses API output-text grammar.

Wire format verified from the official protocol document
(meta-models/meta-sam, protocol/sam3.md):
- One line per frame; image requests use frame 0 only (``<0f>``).
- A line is a frame marker followed by comma-separated segments; each
  segment is an object id + exactly one ``box`` token + one ``mask`` token.
- Box: ``<|box;x1;y1;x2;y2;w;h;c?|>`` with INCLUSIVE x2/y2; normalized
  boxes here are half-open (w = x2 - x1 + 1, h = y2 - y1 + 1).
- Mask: ``<|mask;x=0;y=0;c?;data=H,W,payload|>``. The payload's first
  character is the encoding marker: ``~`` lossless, ``!`` one_bit. The
  payload alphabet excludes ``,`` so splitting segments on commas is safe.
- ``c`` (confidence) is optional per token and ranges 0..1.

Mask raster decoding uses the official ``meta-sam-parser`` package
(arithmetic-coded base85 payloads are NOT reimplemented here).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FRAME_RE = re.compile(r"^<(\d+)f(?:;[^>]*)?>")
_TOKEN_RE = re.compile(r"<\|([A-Za-z][A-Za-z0-9_.-]*)(;[^|]*)?\|>")


@dataclass(frozen=True)
class ParsedRecord:
    object_id: str
    x: int
    y: int
    w: int
    h: int
    confidence: float | None
    mask_width: int
    mask_height: int
    mask_encoding: str  # "lossless" | "one_bit"
    mask_payload: str  # includes the leading marker character


def _parse_fields(body: str | None) -> dict[str, str | None]:
    fields: dict[str, str | None] = {}
    if not body:
        return fields
    for part in body[1:].split(";"):  # body starts with ';'
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        else:
            fields[part] = None
    return fields


def _parse_confidence(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return 0.0 if value == 0 else value


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_output_text(text: str) -> tuple[list[ParsedRecord], list[str]]:
    """Parse one Responses output_text lane (image: frame 0 only).

    Returns (records, diagnostics). Malformed lines are skipped with a
    diagnostic instead of failing the whole parse; valid records are kept.
    """
    records: list[ParsedRecord] = []
    diagnostics: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        frame_match = _FRAME_RE.match(line)
        if not frame_match:
            diagnostics.append(f"malformed line (no frame marker): {line[:80]}")
            continue
        frame = int(frame_match.group(1))
        if frame != 0:
            diagnostics.append(f"skipped non-zero frame {frame} (image mode)")
            continue
        rest = line[frame_match.end():]
        if not rest:
            diagnostics.append("malformed line (bare frame header)")
            continue
        # Split segments ONLY on commas that separate segments: a comma
        # followed by an object id + token opener. Commas inside a mask
        # data tuple (H,W,payload) must not split.
        rest = rest.lstrip(",")
        segments = re.split(r",(?=\d+<\|)", rest)
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            record, error = _parse_segment(segment)
            if error:
                diagnostics.append(error)
            elif record is not None:
                records.append(record)
    return records, diagnostics


def _parse_segment(segment: str) -> tuple[ParsedRecord | None, str | None]:
    id_match = re.match(r"^(\d+)", segment)
    if not id_match:
        return None, f"malformed segment (no object id): {segment[:80]}"
    object_id = id_match.group(1)
    tokens = _TOKEN_RE.findall(segment[id_match.end():])
    box_fields: dict[str, str | None] | None = None
    mask_fields: dict[str, str | None] | None = None
    for name, body in tokens:
        if name == "box" and box_fields is None:
            box_fields = _parse_fields(body)
        elif name == "mask" and mask_fields is None:
            mask_fields = _parse_fields(body)
    if box_fields is None or mask_fields is None:
        return None, f"segment {object_id} missing box or mask token"
    x1 = _parse_int(box_fields.get("x1"))
    y1 = _parse_int(box_fields.get("y1"))
    x2 = _parse_int(box_fields.get("x2"))
    y2 = _parse_int(box_fields.get("y2"))
    img_w = _parse_int(box_fields.get("w"))
    img_h = _parse_int(box_fields.get("h"))
    if None in (x1, y1, x2, y2, img_w, img_h):
        return None, f"segment {object_id} box missing required fields"
    assert x1 is not None and y1 is not None and x2 is not None
    assert y2 is not None and img_w is not None and img_h is not None
    if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1 or x2 >= img_w or y2 >= img_h:
        return None, f"segment {object_id} box outside image bounds"
    if mask_fields.get("x") != "0" or mask_fields.get("y") != "0":
        return None, f"segment {object_id} mask x/y must be 0"
    data = mask_fields.get("data")
    if data is None:
        return None, f"segment {object_id} mask missing data"
    parts = data.split(",", 2)
    if len(parts) != 3:
        return None, f"segment {object_id} mask data not H,W,payload"
    try:
        mask_h, mask_w = int(parts[0]), int(parts[1])
    except ValueError:
        return None, f"segment {object_id} mask dimensions not integers"
    payload = parts[2]
    if not payload or payload[0] not in ("~", "!"):
        return None, f"segment {object_id} mask payload missing encoding marker"
    encoding = "lossless" if payload[0] == "~" else "one_bit"
    confidence = _parse_confidence(mask_fields.get("c"))
    if confidence is None:
        confidence = _parse_confidence(box_fields.get("c"))
    return (
        ParsedRecord(
            object_id=object_id,
            x=x1,
            y=y1,
            w=x2 - x1 + 1,
            h=y2 - y1 + 1,
            confidence=confidence,
            mask_width=mask_w,
            mask_height=mask_h,
            mask_encoding=encoding,
            mask_payload=payload,
        ),
        None,
    )


def decode_record_mask(record: ParsedRecord) -> bytes:
    """Decode a parsed mask payload to a row-major raster (0/1 bytes).

    Uses the official meta-sam-parser decoder. Raises ValueError on a
    structurally invalid payload.
    """
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError as exc:  # pragma: no cover - dependency documented
        raise RuntimeError("meta-sam-parser is required to decode SAM masks") from exc
    mask = SegmentationMask(
        encoding=record.mask_encoding,  # type: ignore[arg-type]
        payload=record.mask_payload,
        width=record.mask_width,
        height=record.mask_height,
    )
    try:
        return decode_mask_to_raster(mask)
    except Exception as exc:
        raise ValueError(f"invalid SAM mask payload: {exc}") from exc
