"""SAM Studio engine — segmentation provider layer + job service.

Sprint 1 scope (Meta AI Hackathon, Oct 26-27 2026):
- provider contract (spec §3)
- fake provider (spec §9 step 1)
- Meta Model API provider, SAM 3.1 (spec §3/§9 step 3, envelope pending
  first live-call confirmation — see meta_provider.py)
- FastAPI job service: POST /sam/jobs, GET /sam/jobs/{id},
  WS /sam/jobs/{id}/events, approve, cancel, per-object file download
- local post-processing: mask -> transparent PNG cutout (trim + padding)

Out of Sprint 1 scope (later sprints): agent QA loop (Sprint 3),
credit Worker wiring (Sprint 4), full UI (Sprint 4), SVG silhouette
export (Sprint 2, via meta-sam-parser's decode_mask_to_svg_path).
"""

from .contract import Mode, SegmentedObject, SegmentationProvider, SegmentationRequest

__all__ = ["Mode", "SegmentedObject", "SegmentationProvider", "SegmentationRequest"]
