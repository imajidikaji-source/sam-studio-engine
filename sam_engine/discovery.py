"""Discovery-first loop (Sprint 4 chunk 1).

Flow: upload -> vision gate -> object list -> user picks -> extract.
This inverts the Sprint 3 goal-driven loop: no goal is needed, because
the list comes from the image itself.

1. A vision model scans the image and returns a verdict:
   ``{processable, reason, objects}``. If ``processable`` is false, the
   job ends as ``rejected`` BEFORE any SAM call — the whole point of
   the gate is that hopeless images never cost a segmentation call.
2. Each listed concept is segmented (default mode ``subject``: one
   call per concept). Concepts the user did not select are skipped,
   so cost scales with the selection, not the list.
3. Each extracted object gets a cutout + SVG + SEMANTIC metadata: the
   name comes from the vision concept (e.g. ``watch face``), fixing
   Sprint 3's positional metadata (``bottom-right-product``).

Known limitation (honest): a concept with several instances in the
image (e.g. three domes) may return partial coverage in subject mode
— the picker shows what was actually found per concept, and the user
picks from that, not from the promise.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image

from .agent import MuseSparkLLM
from .contract import SegmentationProvider, SegmentationRequest
from .postprocess import make_cutout, mask_png_to_svg


@dataclass
class GateVerdict:
    processable: bool
    reason: str
    objects: list[str] = field(default_factory=list)


class VisionGateError(Exception):
    pass


class VisionGate:
    """Interface for the scan step (testable without the API)."""

    async def scan(self, image_path: str) -> GateVerdict:
        raise NotImplementedError


class ScriptedVision(VisionGate):
    """Deterministic test/demo vision gate: canned verdicts."""

    def __init__(self, verdict: GateVerdict):
        self.verdict = verdict
        self.calls: list[str] = []

    async def scan(self, image_path: str) -> GateVerdict:
        self.calls.append(image_path)
        return self.verdict


GATE_PROMPT = (
    "You are a preflight gate for an image segmentation app. Judge "
    "whether this image can be segmented into distinct objects. List "
    "distinct FOREGROUND objects only — never list the background, "
    "backdrop, or scenery itself. Reply with ONLY JSON: "
    "{\"processable\": true/false, \"reason\": \"<one sentence>\", "
    "\"objects\": [<names of distinct objects>]}"
)


class MuseSparkVision(VisionGate):
    """Muse Spark 1.3 as vision gate (image input verified live 2026-10-27)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.meta.ai/v1",
        model: str = "muse-spark-1.3",
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

    @staticmethod
    def _parse_verdict(text: str) -> GateVerdict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        data = json.loads(cleaned)
        return GateVerdict(
            processable=bool(data["processable"]),
            reason=str(data.get("reason", "")),
            objects=[str(o) for o in data.get("objects", [])],
        )

    def _small_data_url(self, image_path: str) -> str:
        """Downscale for the vision call (listing needs shape, not pixels)."""
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((1024, 1024))
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(
            buf.getvalue()
        ).decode()

    async def scan(self, image_path: str) -> GateVerdict:
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": GATE_PROMPT},
                        {"type": "input_image",
                         "image_url": self._small_data_url(image_path)},
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        close_client = self._client is None
        try:
            last_error: Exception | None = None
            response = None
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
                raise VisionGateError(
                    f"vision gate unreachable after retries: {last_error}"
                )
            if response.status_code >= 400:
                raise VisionGateError(
                    f"vision gate HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            text = MuseSparkLLM.extract_text(response.json())
            try:
                return self._parse_verdict(text)
            except (ValueError, KeyError, TypeError):
                pass
            # One repair retry with a stricter instruction.
            body["input"][0]["content"][0]["text"] = (
                GATE_PROMPT + " Your previous reply was not valid JSON. "
                "Reply with ONLY the JSON object."
            )
            response = await client.post(
                f"{self.base_url}/responses", json=body, headers=headers
            )
            if response.status_code >= 400:
                raise VisionGateError(
                    f"vision gate HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            try:
                return self._parse_verdict(
                    MuseSparkLLM.extract_text(response.json())
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise VisionGateError(
                    f"unparseable vision gate response: {exc}"
                ) from exc
        finally:
            if close_client:
                await client.aclose()


@dataclass
class DiscoveryObject:
    concept: str
    index: int
    bbox: tuple[int, int, int, int]
    area_ratio: float
    score: float | None
    status: str  # ok | review
    cutout_file: str | None = None
    svg_file: str | None = None
    metadata_name: str = ""
    metadata_title: str = ""


@dataclass
class DiscoveryResult:
    status: str  # completed | rejected
    verdict: GateVerdict
    objects: list[DiscoveryObject] = field(default_factory=list)
    sam_calls: int = 0
    failure_reason: str | None = None


async def run_discovery_job(
    image_path: str,
    provider: SegmentationProvider,
    vision: VisionGate,
    out_dir: str | Path,
    selected: list[str] | None = None,
    mode: str = "subject",
    max_objects: int = 64,
) -> DiscoveryResult:
    """Run the discovery-first flow; writes cutouts + metadata.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = await vision.scan(image_path)
    if not verdict.processable:
        return DiscoveryResult(
            status="rejected",
            verdict=verdict,
            failure_reason=verdict.reason,
        )
    concepts = verdict.objects
    if selected is not None:
        wanted = set(selected)
        concepts = [c for c in concepts if c in wanted]
    objects: list[DiscoveryObject] = []
    sam_calls = 0
    for concept in concepts:
        raw = await provider.segment(
            SegmentationRequest(
                image_path=image_path,
                prompt=concept,
                mode=mode,  # type: ignore[arg-type]
                max_objects=max_objects,
            ),
            out_dir / f"concept-{concept.replace(' ', '-')}" / "masks",
        )
        sam_calls += 1
        for obj in raw:
            cutout_path, _ = make_cutout(
                image_path, obj.mask_png_path,
                out_dir / "cutouts"
                / f"{concept.replace(' ', '-')}_{obj.index:02d}.png",
            )
            try:
                svg_path = mask_png_to_svg(
                    obj.mask_png_path,
                    out_dir / "svg"
                    / f"{concept.replace(' ', '-')}_{obj.index:02d}.svg",
                )
            except ValueError:
                svg_path = None
            objects.append(
                DiscoveryObject(
                    concept=concept,
                    index=obj.index,
                    bbox=obj.bbox,
                    area_ratio=obj.area_ratio,
                    score=obj.score,
                    status="review" if obj.area_ratio < 0.005 else "ok",
                    cutout_file=str(cutout_path),
                    svg_file=str(svg_path) if svg_path else None,
                    metadata_name=concept.lower().replace(" ", "-"),
                    metadata_title=concept,
                )
            )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "verdict": {
                    "processable": verdict.processable,
                    "reason": verdict.reason,
                    "objects": verdict.objects,
                },
                "selected": selected,
                "sam_calls": sam_calls,
                "objects": [
                    {
                        "concept": o.concept,
                        "index": o.index,
                        "bbox": o.bbox,
                        "area_ratio": o.area_ratio,
                        "score": o.score,
                        "status": o.status,
                        "cutout_file": o.cutout_file,
                        "svg_file": o.svg_file,
                        "metadata_name": o.metadata_name,
                        "metadata_title": o.metadata_title,
                    }
                    for o in objects
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return DiscoveryResult(
        status="completed", verdict=verdict, objects=objects,
        sam_calls=sam_calls,
    )
