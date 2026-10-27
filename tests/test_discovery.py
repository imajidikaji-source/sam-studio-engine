import asyncio

import pytest

from sam_engine.contract import SegmentationProvider
from sam_engine.discovery import (
    GateVerdict,
    MuseSparkVision,
    ScriptedVision,
    VisionGateError,
    run_discovery_job,
)
from sam_engine.fake_provider import FakeSegmentationProvider


def run(coro):
    return asyncio.run(coro)


def test_rejected_gate_makes_zero_sam_calls(sheet_image, tmp_path):
    class CountingProvider(SegmentationProvider):
        def __init__(self):
            self.calls = 0

        async def segment(self, req, out_dir):
            self.calls += 1
            return []

    provider = CountingProvider()
    vision = ScriptedVision(GateVerdict(
        processable=False, reason="blurry", objects=[]))
    result = run(run_discovery_job(
        sheet_image, provider, vision, tmp_path / "disc"))
    assert result.status == "rejected"
    assert result.failure_reason == "blurry"
    assert result.sam_calls == 0
    assert provider.calls == 0


def test_completed_with_selection_filter(sheet_image, tmp_path):
    vision = ScriptedVision(GateVerdict(
        processable=True, reason="ok", objects=["shapes", "other"]))
    result = run(run_discovery_job(
        sheet_image, FakeSegmentationProvider(), vision,
        tmp_path / "disc", selected=["shapes"]))
    assert result.status == "completed"
    assert result.sam_calls == 1  # only the selected concept was segmented
    assert all(o.concept == "shapes" for o in result.objects)
    assert result.objects[0].metadata_name == "shapes"
    assert (tmp_path / "disc" / "metadata.json").exists()


def test_verdict_parsing():
    v = MuseSparkVision._parse_verdict(
        '{"processable": true, "reason": "ok", "objects": ["a", "b"]}')
    assert v.processable and v.objects == ["a", "b"]
    v = MuseSparkVision._parse_verdict(
        '```json\n{"processable": false, "reason": "x", "objects": []}\n```')
    assert not v.processable
    with pytest.raises(ValueError):
        MuseSparkVision._parse_verdict("not json")
