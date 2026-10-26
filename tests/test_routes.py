import time

from fastapi.testclient import TestClient

from sam_engine.app import create_app
from sam_engine.fake_provider import FakeSegmentationProvider


def _client(tmp_path):
    return TestClient(
        create_app(provider=FakeSegmentationProvider(), base_dir=tmp_path / "jobs")
    )


def test_job_lifecycle(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        created = client.post("/sam/jobs", json={
            "image_path": sheet_image, "prompt": "every sticker", "mode": "sheet",
        })
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
        assert len(status["objects"]) == 6
        assert all(obj["status"] == "ok" for obj in status["objects"])
        approved = client.post(f"/sam/jobs/{job_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["approved"] is True
        file_resp = client.get(f"/sam/jobs/{job_id}/objects/0/file")
        assert file_resp.status_code == 200
        assert file_resp.headers["content-type"] == "image/png"


def test_job_errors(sheet_image, tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/sam/jobs/nope").status_code == 404
        assert client.post("/sam/jobs/nope/approve").status_code == 404
        bad = client.post("/sam/jobs", json={
            "image_path": sheet_image, "prompt": "x", "mode": "bogus",
        })
        assert bad.status_code == 400
        missing = client.post("/sam/jobs", json={
            "image_path": "/nope.png", "prompt": "x",
        })
        assert missing.status_code == 400
