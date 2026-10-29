"""Muse Spark agent loop (Sprint 3, spec §6).

The loop turns a user GOAL into segmented, named, metadata-bearing
assets without a manual prompt:

1. LLM formulates a short concrete-noun prompt from the goal.
2. The segmentation provider runs; deterministic QA checks the masks:
   - 0 objects -> failure "empty"
   - any object covering > 85% of a sheet -> failure "merged suspected"
     (spec §6: a sheet object that big is almost certainly several
     objects fused into one mask)
3. On failure the LLM proposes an ALTERNATIVE prompt using the failure
   context, up to MAX_RETRIES retries. After that the job ends as
   "needs_review" with the best attempt attached — never a silent
   success, never a silent failure.
4. On success the LLM writes per-object metadata (name, title,
   keywords, category) as JSON; a bad-JSON answer gets one repair
   retry, then metadata is recorded as failed (explicitly) while the
   objects themselves still complete.

LLM access is a protocol (``AgentLLM``) so the loop is fully testable
without the API: tests and demos use ``ScriptedLLM``. The production
implementation is ``MuseSparkLLM`` (Meta Model API, model
``muse-spark-1.3``). Its Responses-API envelope was VERIFIED LIVE on
2026-10-27: a probe call returned exactly 'OK', and three live agent
loops completed first-try (mosque sheet, sticker sheet, mosque subject
with a named goal). One live loop with a VAGUE goal ('main subject of
this illustration', subject mode) correctly exhausted 4 attempts and
ended as needs_review — the LLM cannot see the image, so a goal that
names no concrete subject is unanswerable by text-only prompting. That
is designed behavior, not a bug: in the product flow the formulated
prompt is previewed before approval (Sprint 4), so a wrong guess is
caught in one tap. Metadata is likewise positional/generic today: the
LLM only receives bbox/area/score, not the image, so names like
'bottom-right-product' describe geometry, not depicted content.
Semantic naming needs image input to the LLM or a vision pass — a
post-hackathon improvement, not a v1 promise.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .contract import SegmentedObject, SegmentationProvider
from .postprocess import make_cutout, mask_png_to_svg

MAX_RETRIES = 3  # spec §6: retry maks 3x (initial attempt + 3 retries)
MERGED_AREA_RATIO = 0.85  # spec §6: sheet object > 85% = merged suspected


class AgentLLM:
    """Minimal LLM interface the loop needs (testable without the API)."""

    async def complete(self, prompt: str) -> str:
        raise NotImplementedError


class ScriptedLLM(AgentLLM):
    """Deterministic test/demo LLM: canned answers in call order.

    ``answers`` maps a prompt-substring key to a reply; the special key
    ``"*"`` is the fallback. If no key matches, the fallback (or "")
    is returned.
    """

    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        # Longest matching key wins, so a specific context (e.g. a retry
        # marker) beats the generic goal key that is also present.
        matches = [
            key for key in self.answers
            if key != "*" and key in prompt
        ]
        if matches:
            return self.answers[max(matches, key=len)]
        return self.answers.get("*", "")


class MuseSparkError(Exception):
    pass


class MuseSparkLLM(AgentLLM):
    """Muse Spark via the Meta Model API Responses endpoint.

    ENVELOPE STATUS: VERIFIED live 2026-10-27 (probe returned 'OK';
    three live loops completed). The parser still tries every plausible
    text shape and raises MuseSparkError if none yields text, so a
    future shape mismatch fails loudly instead of silently producing
    empty metadata.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.meta.ai/v1",
        model: str = "muse-spark-1.3",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise ValueError("Meta Model API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = http_client
        self.timeout = timeout

    @staticmethod
    def extract_text(payload: dict) -> str:
        """Pull assistant text out of a Responses-API payload, defensively."""
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        chunks: list[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for block in item.get("content", []) or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("output_text", "text"):
                    text = block.get("text", "")
                    if isinstance(text, str):
                        chunks.append(text)
        if chunks:
            return "".join(chunks)
        raise MuseSparkError(f"no text found in Muse Spark response: {payload!r}"[:500])

    async def complete(self, prompt: str) -> str:
        body = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
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
            else:
                raise MuseSparkError(
                    f"Muse Spark unreachable after retries: {last_error}"
                )
            if response.status_code == 402:
                raise MuseSparkError("billing not configured (HTTP 402)")
            if response.status_code >= 400:
                raise MuseSparkError(
                    f"Muse Spark HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            return self.extract_text(response.json())
        finally:
            if close_client:
                await client.aclose()


@dataclass
class ObjectMetadata:
    name: str
    title: str
    keywords: list[str] = field(default_factory=list)
    category: str = ""


@dataclass
class AgentObject:
    index: int
    bbox: tuple[int, int, int, int]
    area_ratio: float
    score: float | None
    status: str  # ok | review (from jobs.py review threshold)
    cutout_file: str | None = None
    svg_file: str | None = None
    metadata: ObjectMetadata | None = None


@dataclass
class Attempt:
    prompt: str
    n_objects: int
    issues: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    goal: str
    status: str  # completed | needs_review
    final_prompt: str
    attempts: list[Attempt] = field(default_factory=list)
    objects: list[AgentObject] = field(default_factory=list)
    # ok | partial | failed | deferred | not_attempted. "deferred" is
    # Sprint 4 chunk 3: preview must not spend LLM metadata calls, so
    # a deferred run skips metadata until attach_metadata() runs after
    # approve.
    metadata_status: str = "not_attempted"
    failure_reason: str | None = None


def qa_issues(objects: list[SegmentedObject], mode: str) -> list[str]:
    """Deterministic mask-quality checks (spec §6)."""
    issues: list[str] = []
    if not objects:
        issues.append("empty: no objects detected")
        return issues
    if mode == "sheet":
        for obj in objects:
            if obj.area_ratio > MERGED_AREA_RATIO:
                issues.append(
                    f"merged suspected: object {obj.index} covers "
                    f"{obj.area_ratio:.0%} of the sheet"
                )
    return issues


PROMPT_FORMULATION_TEMPLATE = (
    "You segment images with a vision model that only understands short "
    "concrete visual noun phrases (for example 'geometric shapes' or "
    "'mosque'). It does NOT understand sales-medium words like 'stickers' "
    "or abstract quantifiers like 'every shape' — those return nothing. "
    "User goal: {goal}\n"
    "{failure_context}"
    "Reply with ONLY the noun phrase, no explanation, no quotes."
)

METADATA_TEMPLATE = (
    "For each segmented object below, write commercial metadata as a JSON "
    "array. Each element: {{\"index\": <int>, \"name\": <short lowercase "
    "slug>, \"title\": <human title>, \"keywords\": [<up to 5 words>], "
    "\"category\": <one word>}}. Objects (index, bbox x/y/w/h, area "
    "ratio, score):\n{objects}\n"
    "Reply with ONLY the JSON array, no explanation, no markdown fences."
)


def _parse_metadata(text: str, n_objects: int) -> dict[int, ObjectMetadata]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("metadata is not a JSON array")
    out: dict[int, ObjectMetadata] = {}
    for item in data:
        idx = int(item["index"])
        out[idx] = ObjectMetadata(
            name=str(item.get("name", f"object-{idx}")),
            title=str(item.get("title", "")),
            keywords=[str(k) for k in item.get("keywords", [])][:5],
            category=str(item.get("category", "")),
        )
    return out


async def _generate_metadata(
    llm: AgentLLM, objects_desc: str, n_objects: int,
) -> tuple[dict[int, ObjectMetadata], str]:
    """One metadata LLM call + one repair retry on bad JSON.

    Returns (metadata by index, status). Raises nothing: unparseable
    output after the repair retry yields ({}, "failed").
    """
    try:
        metadata = _parse_metadata(
            await llm.complete(METADATA_TEMPLATE.format(objects=objects_desc)),
            n_objects,
        )
        return metadata, "ok" if len(metadata) == n_objects else "partial"
    except (ValueError, KeyError, TypeError):
        pass
    try:
        metadata = _parse_metadata(
            await llm.complete(
                METADATA_TEMPLATE.format(objects=objects_desc)
                + "\nYour previous reply was not valid JSON. "
                "Reply with ONLY the JSON array."
            ),
            n_objects,
        )
        return metadata, "ok" if len(metadata) == n_objects else "partial"
    except (ValueError, KeyError, TypeError):
        return {}, "failed"


def _write_agent_metadata(
    out_dir: Path,
    goal: str,
    status: str,
    final_prompt: str,
    attempts: list[Attempt],
    metadata_status: str,
    agent_objects: list[AgentObject],
) -> None:
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "goal": goal,
                "status": status,
                "final_prompt": final_prompt,
                "attempts": [
                    {"prompt": a.prompt, "n_objects": a.n_objects,
                     "issues": a.issues}
                    for a in attempts
                ],
                "metadata_status": metadata_status,
                "objects": [
                    {
                        "index": o.index,
                        "bbox": o.bbox,
                        "area_ratio": o.area_ratio,
                        "score": o.score,
                        "status": o.status,
                        "cutout_file": o.cutout_file,
                        "svg_file": o.svg_file,
                        "metadata": (
                            {
                                "name": o.metadata.name,
                                "title": o.metadata.title,
                                "keywords": o.metadata.keywords,
                                "category": o.metadata.category,
                            }
                            if o.metadata
                            else None
                        ),
                    }
                    for o in agent_objects
                ],
            },
            indent=2,
        )
        + "\n"
    )


async def attach_metadata(
    llm: AgentLLM, result: AgentResult, out_dir: str | Path,
) -> AgentResult:
    """Generate LLM metadata for a deferred run — the post-approve step.

    Only runs on a completed run whose metadata is still "deferred";
    any other state is returned unchanged (never double-generates).
    """
    if result.status != "completed" or result.metadata_status != "deferred":
        return result
    out_dir = Path(out_dir)
    objects_desc = "\n".join(
        f"- {o.index}: bbox={o.bbox}, area={o.area_ratio:.3f}, "
        f"score={o.score}"
        for o in result.objects
    )
    metadata, status = await _generate_metadata(
        llm, objects_desc, len(result.objects)
    )
    for obj in result.objects:
        obj.metadata = metadata.get(obj.index)
    result.metadata_status = status
    _write_agent_metadata(
        out_dir, result.goal, result.status, result.final_prompt,
        result.attempts, result.metadata_status, result.objects,
    )
    return result


async def run_agent_job(
    image_path: str,
    goal: str,
    provider: SegmentationProvider,
    llm: AgentLLM,
    out_dir: str | Path,
    mode: str = "sheet",
    max_objects: int = 64,
    defer_metadata: bool = False,
) -> AgentResult:
    """Run the full agent loop; writes cutouts + metadata.json to out_dir.

    With ``defer_metadata=True`` (Sprint 4 chunk 3) the loop completes
    segmentation + QA + assets but makes NO metadata LLM call — preview
    must not spend a metadata call on a sheet the user may reject.
    Metadata is then generated post-approve via ``attach_metadata``.
    """
    from .contract import SegmentationRequest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[Attempt] = []
    failure_context = ""
    best_objects: list[SegmentedObject] = []
    final_prompt = ""

    for attempt_no in range(MAX_RETRIES + 1):
        prompt = await llm.complete(
            PROMPT_FORMULATION_TEMPLATE.format(
                goal=goal, failure_context=failure_context
            )
        )
        prompt = prompt.strip().strip('"').strip()
        final_prompt = prompt
        raw = await provider.segment(
            SegmentationRequest(
                image_path=image_path,
                prompt=prompt,
                mode=mode,  # type: ignore[arg-type]
                max_objects=max_objects,
            ),
            out_dir / f"attempt{attempt_no}" / "masks",
        )
        issues = qa_issues(raw, mode)
        attempts.append(Attempt(prompt=prompt, n_objects=len(raw), issues=issues))
        if not issues:
            best_objects = raw
            break
        best_objects = raw
        failure_context = (
            f"The previous prompt {prompt!r} failed QA: "
            f"{'; '.join(issues)}. Propose a DIFFERENT concrete visual "
            f"noun phrase for the same goal.\n"
        )
    else:
        # Loop exhausted without break: needs review, never silent success.
        return AgentResult(
            goal=goal,
            status="needs_review",
            final_prompt=final_prompt,
            attempts=attempts,
            objects=[],
            metadata_status="not_attempted",
            failure_reason="; ".join(attempts[-1].issues),
        )

    # Success path: assets + metadata. With defer_metadata (Sprint 4
    # chunk 3) the metadata LLM call is skipped entirely here — preview
    # must not spend a metadata call on a sheet the user may reject.
    # Metadata is generated post-approve via attach_metadata().
    metadata: dict[int, ObjectMetadata] = {}
    if defer_metadata:
        metadata_status = "deferred"
    else:
        objects_desc = "\n".join(
            f"- {o.index}: bbox={o.bbox}, area={o.area_ratio:.3f}, "
            f"score={o.score}"
            for o in best_objects
        )
        metadata, metadata_status = await _generate_metadata(
            llm, objects_desc, len(best_objects)
        )

    agent_objects: list[AgentObject] = []
    for obj in best_objects:
        cutout_path, _ = make_cutout(
            image_path, obj.mask_png_path,
            out_dir / "cutouts" / f"obj_{obj.index:02d}.png",
        )
        try:
            svg_path = mask_png_to_svg(
                obj.mask_png_path, out_dir / "svg" / f"obj_{obj.index:02d}.svg"
            )
        except ValueError:
            svg_path = None
        agent_objects.append(
            AgentObject(
                index=obj.index,
                bbox=obj.bbox,
                area_ratio=obj.area_ratio,
                score=obj.score,
                status="review" if obj.area_ratio < 0.005 else "ok",
                cutout_file=str(cutout_path),
                svg_file=str(svg_path) if svg_path else None,
                metadata=metadata.get(obj.index),
            )
        )

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "goal": goal,
                "status": "completed",
                "final_prompt": final_prompt,
                "attempts": [
                    {"prompt": a.prompt, "n_objects": a.n_objects,
                     "issues": a.issues}
                    for a in attempts
                ],
                "metadata_status": metadata_status,
                "objects": [
                    {
                        "index": o.index,
                        "bbox": o.bbox,
                        "area_ratio": o.area_ratio,
                        "score": o.score,
                        "status": o.status,
                        "cutout_file": o.cutout_file,
                        "svg_file": o.svg_file,
                        "metadata": (
                            {
                                "name": o.metadata.name,
                                "title": o.metadata.title,
                                "keywords": o.metadata.keywords,
                                "category": o.metadata.category,
                            }
                            if o.metadata
                            else None
                        ),
                    }
                    for o in agent_objects
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return AgentResult(
        goal=goal,
        status="completed",
        final_prompt=final_prompt,
        attempts=attempts,
        objects=agent_objects,
        metadata_status=metadata_status,
    )
