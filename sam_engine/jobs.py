"""In-memory job store + runner (spec §4, Sprint 1 subset + Sprint 2 batch).

Two-phase flow (preview -> approve -> credit commit) is structural: the
job runs, the user previews objects, approve finalizes. The Worker credit
client (``/credits/balance`` + ``/credits/consume``, idempotency key =
job_id) lands in Sprint 4 — approve() today only records approval and
never charges anything. ``credit_estimate`` is informational (1 credit per
sheet, spec §8).

Sprint 2 additions: per-job API cost log (``api_calls``,
``cost_usd_estimate`` from the provider's ``cost_per_image_usd``) and
batch grouping (one batch = many sheet jobs, one summary).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .contract import SegmentedObject, SegmentationProvider, SegmentationRequest
from .postprocess import filter_noise, make_cutout, mask_png_to_svg

REVIEW_AREA_RATIO = 0.005  # spec §6: < 0.5% of image -> needs review


@dataclass
class ObjectResult:
    index: int
    bbox: tuple[int, int, int, int]
    area_ratio: float
    score: float | None
    status: str  # ok | review | failed
    mask_file: str
    cutout_file: str | None = None
    svg_file: str | None = None


@dataclass
class Job:
    id: str
    image_path: str
    prompt: str
    mode: str
    max_objects: int
    status: str = "queued"  # queued | running | completed | failed | cancelled
    objects: list[ObjectResult] = field(default_factory=list)
    error: str | None = None
    approved: bool = False
    credit_estimate: int = 1
    cancel_requested: bool = False
    events: list[dict] = field(default_factory=list)
    api_calls: int = 0
    cost_usd_estimate: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass
class Batch:
    id: str
    prompt: str
    mode: str
    job_ids: list[str] = field(default_factory=list)


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queues: dict[str, asyncio.Queue] = {}
        self.batches: dict[str, Batch] = {}

    def create(self, req: SegmentationRequest) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            image_path=req.image_path,
            prompt=req.prompt,
            mode=req.mode,
            max_objects=req.max_objects,
        )
        self.jobs[job.id] = job
        self.queues[job.id] = asyncio.Queue()
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def emit(self, job: Job, event: dict) -> None:
        job.events.append(event)
        queue = self.queues.get(job.id)
        if queue is not None:
            queue.put_nowait(event)


async def run_job(
    store: JobStore,
    provider: SegmentationProvider,
    job: Job,
    base_dir: Path,
) -> None:
    job_dir = Path(base_dir) / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"
    job.started_at = time.time()
    store.emit(job, {"type": "started", "job_id": job.id})
    try:
        req = SegmentationRequest(
            image_path=job.image_path,
            prompt=job.prompt,
            mode=job.mode,  # type: ignore[arg-type]
            max_objects=job.max_objects,
        )
        raw_objects = await provider.segment(req, job_dir / "masks")
        job.api_calls += 1
        job.cost_usd_estimate += getattr(provider, "cost_per_image_usd", 0.0)
        if job.cancel_requested:
            job.status = "cancelled"
            store.emit(job, {"type": "cancelled", "job_id": job.id})
            return
        if not raw_objects:
            job.status = "failed"
            job.error = "no objects detected"
            store.emit(job, {"type": "failed", "job_id": job.id, "error": job.error})
            return
        kept = filter_noise(raw_objects)
        if not kept:
            job.status = "failed"
            job.error = "all objects filtered as noise"
            store.emit(job, {"type": "failed", "job_id": job.id, "error": job.error})
            return
        store.emit(
            job,
            {"type": "segmented", "job_id": job.id, "objects": len(raw_objects),
             "kept": len(kept)},
        )
        for obj in kept:
            if job.cancel_requested:
                job.status = "cancelled"
                store.emit(job, {"type": "cancelled", "job_id": job.id})
                return
            cutout_path, _bbox = make_cutout(
                job.image_path, obj.mask_png_path, job_dir / "cutouts" / f"obj_{obj.index:02d}.png"
            )
            try:
                svg_path = mask_png_to_svg(
                    obj.mask_png_path, job_dir / "svg" / f"obj_{obj.index:02d}.svg"
                )
            except ValueError:
                svg_path = None
            status = "review" if obj.area_ratio < REVIEW_AREA_RATIO else "ok"
            job.objects.append(
                ObjectResult(
                    index=obj.index,
                    bbox=obj.bbox,
                    area_ratio=obj.area_ratio,
                    score=obj.score,
                    status=status,
                    mask_file=obj.mask_png_path,
                    cutout_file=str(cutout_path),
                    svg_file=str(svg_path) if svg_path else None,
                )
            )
            store.emit(
                job,
                {"type": "object_done", "job_id": job.id, "index": obj.index,
                 "status": status},
            )
        job.status = "completed"
        store.emit(
            job,
            {"type": "completed", "job_id": job.id, "objects": len(job.objects)},
        )
    except Exception as exc:  # job failure must not crash the service
        job.status = "failed"
        job.error = str(exc)
        store.emit(job, {"type": "failed", "job_id": job.id, "error": job.error})
    finally:
        if job.status in ("completed", "failed", "cancelled"):
            job.finished_at = time.time()
