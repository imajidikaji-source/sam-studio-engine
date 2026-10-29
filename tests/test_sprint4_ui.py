"""Sprint 4 chunk 3: UI page + licence plumbing.

Covers: license_id carried on jobs, approve consumes exactly 1 credit,
double approve never double-charges, failed sheets charge 0, balance
gate (402 + Approve disabled semantics), discovery picker custom
concepts with the fake provider, upload + UI routes, and goal-driven
LLM metadata deferred until after approve. Fake provider only — no
live Meta API calls anywhere in this file.
"""

import asyncio
import base64
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from sam_engine.agent import (
    ScriptedLLM,
    attach_metadata,
    run_agent_job,
)
from sam_engine.app import create_app
from sam_engine.credits import CreditStore
from sam_engine.discovery import GateVerdict, ScriptedVision
from sam_engine.fake_provider import FakeSegmentationProvider


def run(coro):
    return asyncio.run(coro)


def _client(tmp_path, **kwargs):
    kwargs.setdefault("provider", FakeSegmentationProvider())
    kwargs.setdefault("base_dir", tmp_path / "jobs")
    return TestClient(create_app(**kwargs))


def _completed_job(client, image_path, **over):
    body = {"image_path": image_path, "prompt": "shapes", "mode": "sheet"}
    body.update(over)
    created = client.post("/sam/jobs", json=body)
    assert created.status_code == 200
    job_id = created.json()["job_id"]
    status = None
    for _ in range(100):
        resp = client.get(f"/sam/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    return status


def test_license_id_carried_and_default(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        job = _completed_job(client, sheet_image, license_id="lic-123")
        assert job["license_id"] == "lic-123"
        defaulted = _completed_job(client, sheet_image)
        assert defaulted["license_id"] == "demo"


def test_batch_license_id_carried(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        resp = client.post("/sam/batch", json={
            "image_paths": [sheet_image], "prompt": "shapes",
            "license_id": "lic-batch",
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_ids"][0]
        assert client.get(f"/sam/jobs/{job_id}").json()["license_id"] == "lic-batch"


def test_approve_consumes_exactly_one_and_double_approve_once(
    sheet_image, tmp_path
):
    store = CreditStore()
    store.grant("demo", "machine-1", 5, "pro", "grant-1")
    with _client(tmp_path, credit_store=store) as client:
        job = _completed_job(client, sheet_image)
        approved = client.post(f"/sam/jobs/{job['job_id']}/approve")
        assert approved.status_code == 200
        assert approved.json()["approved"] is True
        assert approved.json()["credit_consumed"] is True
        assert approved.json()["credit_entry_id"] is not None
        assert client.get("/credits/balance/demo").json()["balance"] == 4
        again = client.post(f"/sam/jobs/{job['job_id']}/approve")
        assert again.status_code == 200
        assert client.get("/credits/balance/demo").json()["balance"] == 4


def test_balance_gate_blocks_approve(sheet_image, tmp_path):
    store = CreditStore()  # demo balance is 0
    with _client(tmp_path, credit_store=store) as client:
        job = _completed_job(client, sheet_image)
        resp = client.post(f"/sam/jobs/{job['job_id']}/approve")
        assert resp.status_code == 402
        assert "insufficient balance" in resp.json()["detail"]
        status = client.get(f"/sam/jobs/{job['job_id']}").json()
        assert status["approved"] is False
        assert status["credit_consumed"] is False
        assert client.get("/credits/balance/demo").json()["balance"] == 0


def test_failed_sheet_consumes_zero(sheet_image, tmp_path):
    class FailingProvider:
        async def segment(self, req, out_dir):
            raise RuntimeError("boom")

    store = CreditStore()
    store.grant("demo", "machine-1", 5, "pro", "grant-1")
    with _client(
        tmp_path, provider=FailingProvider(), credit_store=store
    ) as client:
        created = client.post("/sam/jobs", json={
            "image_path": sheet_image, "prompt": "shapes",
        })
        job_id = created.json()["job_id"]
        status = None
        for _ in range(100):
            status = client.get(f"/sam/jobs/{job_id}").json()
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        assert status["status"] == "failed"
        resp = client.post(f"/sam/jobs/{job_id}/approve")
        assert resp.status_code == 409
        assert client.get("/credits/balance/demo").json()["balance"] == 5


def test_upload_route(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        raw = Path(sheet_image).read_bytes()
        resp = client.post("/sam/upload", json={
            "filename": "sheet.png",
            "image_base64": base64.b64encode(raw).decode(),
        })
        assert resp.status_code == 200
        assert Path(resp.json()["image_path"]).is_file()
        bad = client.post("/sam/upload", json={
            "filename": "x.png", "image_base64": "!!!not-base64!!!",
        })
        assert bad.status_code == 400
        not_image = client.post("/sam/upload", json={
            "filename": "x.png",
            "image_base64": base64.b64encode(b"hello").decode(),
        })
        assert not_image.status_code == 400


def test_ui_route_markers(tmp_path):
    with _client(tmp_path) as client:
        resp = client.get("/sam/ui")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text
        for marker in (
            "SAM Studio", "tambah konsep sendiri", "Approve", "Saldo",
            "goal-driven", "discovery-first", "Ekspor",
        ):
            assert marker in html


def test_discover_scan_configured_and_unconfigured(sheet_image, tmp_path):
    vision = ScriptedVision(GateVerdict(
        processable=True, reason="test gate", objects=["dome", "minaret"],
    ))
    with _client(tmp_path, vision=vision) as client:
        scan = client.post("/sam/discover/scan", json={
            "image_path": sheet_image,
        })
        assert scan.status_code == 200
        assert scan.json()["gate"] == "configured"
        assert scan.json()["objects"] == ["dome", "minaret"]
    with _client(tmp_path) as client:
        scan = client.post("/sam/discover/scan", json={
            "image_path": sheet_image,
        })
        assert scan.status_code == 200
        assert scan.json()["gate"] == "not_configured"
        assert scan.json()["objects"] == []


def test_discover_run_custom_concept_survives_selection(
    sheet_image, tmp_path
):
    vision = ScriptedVision(GateVerdict(
        processable=True, reason="test gate", objects=["dome", "minaret"],
    ))
    with _client(tmp_path, vision=vision) as client:
        run_resp = client.post("/sam/discover/run", json={
            "image_path": sheet_image,
            "selected": ["dome", "menara-baru"],
        })
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert data["status"] == "completed"
        concepts = {o["concept"] for o in data["objects"]}
        assert "dome" in concepts
        assert "menara-baru" in concepts  # custom concept segmented
        assert "minaret" not in concepts  # unpicked gate concept skipped
        assert data["sam_calls"] == 2
        # Semantic names come from the concept, not a position.
        assert all(o["metadata_title"] == o["concept"] for o in data["objects"])
        first = data["objects"][0]
        file_resp = client.get(first["file_url"])
        assert file_resp.status_code == 200
        assert file_resp.headers["content-type"] == "image/png"
        mask_resp = client.get(first["mask_url"])
        assert mask_resp.status_code == 200
        assert client.get(first["svg_url"]).status_code == 200


def test_discover_run_without_vision_uses_typed_concepts(
    sheet_image, tmp_path
):
    with _client(tmp_path) as client:
        run_resp = client.post("/sam/discover/run", json={
            "image_path": sheet_image, "selected": ["foo"],
        })
        assert run_resp.status_code == 200
        assert run_resp.json()["status"] == "completed"
        assert run_resp.json()["objects"][0]["concept"] == "foo"


def test_discover_approve_consumes_once_and_gate_blocks(
    sheet_image, tmp_path
):
    vision = ScriptedVision(GateVerdict(
        processable=True, reason="test gate", objects=["dome"],
    ))
    store = CreditStore()
    store.grant("demo", "machine-1", 3, "pro", "grant-1")
    with _client(tmp_path, vision=vision, credit_store=store) as client:
        run_resp = client.post("/sam/discover/run", json={
            "image_path": sheet_image, "selected": ["dome"],
        })
        disc_id = run_resp.json()["discovery_id"]
        approved = client.post(f"/sam/discover/{disc_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["approved"] is True
        assert client.get("/credits/balance/demo").json()["balance"] == 2
        again = client.post(f"/sam/discover/{disc_id}/approve")
        assert again.status_code == 200
        assert client.get("/credits/balance/demo").json()["balance"] == 2

    # No balance -> 402, no charge.
    store2 = CreditStore()
    with _client(tmp_path, vision=vision, credit_store=store2) as client:
        run_resp = client.post("/sam/discover/run", json={
            "image_path": sheet_image, "selected": ["dome"],
        })
        disc_id = run_resp.json()["discovery_id"]
        resp = client.post(f"/sam/discover/{disc_id}/approve")
        assert resp.status_code == 402
        assert client.get("/credits/balance/demo").json()["balance"] == 0

    # Rejected discovery cannot approve (0 credits by construction).
    rejected_vision = ScriptedVision(GateVerdict(
        processable=False, reason="blurry", objects=[],
    ))
    with _client(tmp_path, vision=rejected_vision) as client:
        run_resp = client.post("/sam/discover/run", json={
            "image_path": sheet_image, "selected": ["dome"],
        })
        assert run_resp.json()["status"] == "rejected"
        disc_id = run_resp.json()["discovery_id"]
        assert client.post(
            f"/sam/discover/{disc_id}/approve"
        ).status_code == 409


def test_job_file_kinds(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        job = _completed_job(client, sheet_image)
        job_id = job["job_id"]
        assert client.get(
            f"/sam/jobs/{job_id}/objects/0/file"
        ).status_code == 200
        assert client.get(
            f"/sam/jobs/{job_id}/objects/0/file?kind=mask"
        ).status_code == 200
        assert client.get(
            f"/sam/jobs/{job_id}/objects/0/file?kind=svg"
        ).status_code == 200
        assert client.get(
            f"/sam/jobs/{job_id}/objects/0/file?kind=bogus"
        ).status_code == 400
        assert client.get(
            f"/sam/jobs/{job_id}/objects/99/file"
        ).status_code == 404


META_JSON = json.dumps([
    {"index": i, "name": f"shape-{i}", "title": f"Shape {i}",
     "keywords": ["shape"], "category": "shapes"}
    for i in range(6)
])


def test_goal_metadata_deferred_until_after_approve(
    sheet_image, tmp_path
):
    llm = ScriptedLLM({"User goal": "shapes", "*": META_JSON})
    out_dir = tmp_path / "agent-deferred"
    result = run(run_agent_job(
        sheet_image, "sticker sheet for kids",
        FakeSegmentationProvider(), llm, out_dir,
        defer_metadata=True,
    ))
    assert result.status == "completed"
    assert result.metadata_status == "deferred"
    assert all(o.metadata is None for o in result.objects)
    # No metadata LLM call happened during the run: formulation calls
    # only (the metadata template never appears in the call log).
    assert not any(
        "commercial metadata" in call for call in llm.calls
    )
    assert json.loads(
        (out_dir / "metadata.json").read_text()
    )["metadata_status"] == "deferred"

    # Post-approve step fills metadata exactly once.
    llm2 = ScriptedLLM({"commercial metadata": META_JSON})
    result = run(attach_metadata(llm2, result, out_dir))
    assert result.metadata_status == "ok"
    assert result.objects[0].metadata.name == "shape-0"
    n_calls = len(llm2.calls)
    result = run(attach_metadata(llm2, result, out_dir))
    assert result.metadata_status == "ok"
    assert len(llm2.calls) == n_calls  # second attach is a no-op
