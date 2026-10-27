import asyncio
import json

import pytest

from sam_engine.agent import (
    MAX_RETRIES,
    AgentResult,
    ScriptedLLM,
    qa_issues,
    run_agent_job,
)
from sam_engine.contract import SegmentedObject, SegmentationProvider
from sam_engine.fake_provider import FakeSegmentationProvider

META_JSON = json.dumps([
    {"index": i, "name": f"shape-{i}", "title": f"Shape {i}",
     "keywords": ["shape", "sticker"], "category": "shapes"}
    for i in range(6)
])


def run(coro):
    return asyncio.run(coro)


def test_qa_rules():
    assert qa_issues([], "sheet") == ["empty: no objects detected"]
    big = SegmentedObject(index=0, bbox=(0, 0, 100, 100),
                          mask_png_path="m.png", area_ratio=0.9)
    assert any("merged" in i for i in qa_issues([big], "sheet"))
    # Subject mode: one big object is expected, not merged.
    assert qa_issues([big], "subject") == []


def test_loop_success_first_try(sheet_image, tmp_path):
    llm = ScriptedLLM({"User goal": "shapes", "*": META_JSON})
    result = run(run_agent_job(
        sheet_image, "sticker sheet for kids",
        FakeSegmentationProvider(), llm, tmp_path / "agent"))
    assert result.status == "completed"
    assert len(result.attempts) == 1
    assert result.final_prompt == "shapes"
    assert len(result.objects) == 6
    assert result.metadata_status == "ok"
    assert result.objects[0].metadata.name == "shape-0"
    assert result.objects[0].cutout_file is not None
    assert (tmp_path / "agent" / "metadata.json").exists()


def test_loop_retries_on_empty_then_succeeds(sheet_image, tmp_path):
    class EmptyThenGrid(SegmentationProvider):
        def __init__(self):
            self.calls = 0
            self.fake = FakeSegmentationProvider()

        async def segment(self, req, out_dir):
            self.calls += 1
            if self.calls == 1:
                return []
            return await self.fake.segment(req, out_dir)

    llm = ScriptedLLM({
        "User goal": "shapes",
        "DIFFERENT concrete": "geometric shapes",
        "*": META_JSON,
    })
    result = run(run_agent_job(
        sheet_image, "sticker sheet", EmptyThenGrid(), llm,
        tmp_path / "agent"))
    assert result.status == "completed"
    assert len(result.attempts) == 2
    assert result.attempts[0].issues == ["empty: no objects detected"]
    assert result.attempts[1].prompt == "geometric shapes"


def test_loop_needs_review_after_max_retries(sheet_image, tmp_path):
    class AlwaysEmpty(SegmentationProvider):
        async def segment(self, req, out_dir):
            return []

    llm = ScriptedLLM({"*": "shapes"})
    result = run(run_agent_job(
        sheet_image, "impossible goal", AlwaysEmpty(), llm,
        tmp_path / "agent"))
    assert result.status == "needs_review"
    assert len(result.attempts) == MAX_RETRIES + 1
    assert result.objects == []
    assert result.failure_reason is not None


def test_loop_metadata_failure_is_explicit_not_silent(sheet_image, tmp_path):
    llm = ScriptedLLM({"User goal": "shapes", "*": "not json at all"})
    result = run(run_agent_job(
        sheet_image, "sticker sheet", FakeSegmentationProvider(), llm,
        tmp_path / "agent"))
    assert result.status == "completed"  # objects are fine
    assert result.metadata_status == "failed"
    assert result.objects[0].metadata is None


def test_muse_spark_extract_text_shapes():
    from sam_engine.agent import MuseSparkLLM

    assert MuseSparkLLM.extract_text({"output_text": "hi"}) == "hi"
    payload = {"output": [{"content": [
        {"type": "output_text", "text": "hello "},
        {"type": "output_text", "text": "world"},
    ]}]}
    assert MuseSparkLLM.extract_text(payload) == "hello world"
    with pytest.raises(Exception):
        MuseSparkLLM.extract_text({"output": []})
