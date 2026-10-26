import time

from fastapi.testclient import TestClient
from PIL import Image

from sam_engine.app import create_app
from sam_engine.fake_provider import FakeSegmentationProvider
from sam_engine.postprocess import mask_png_to_svg


def test_svg_silhouette_from_mask(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask = Image.new("L", (100, 80), 0)
    for x in range(20, 60):
        for y in range(10, 50):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    out = mask_png_to_svg(mask_path, tmp_path / "silhouette.svg")
    text = out.read_text()
    assert 'viewBox="0 0 100 80"' in text
    assert "<path" in text and text.rstrip().endswith("</svg>")
    # A rectangle simplifies to few points, far fewer than the raw contour.
    assert text.count(" L ") < 20


def test_svg_empty_mask_rejected(tmp_path):
    import pytest

    mask_path = tmp_path / "empty.png"
    Image.new("L", (50, 50), 0).save(mask_path)
    with pytest.raises(ValueError):
        mask_png_to_svg(mask_path, tmp_path / "out.svg")


def test_batch_endpoint(sheet_image, tmp_path):
    with TestClient(
        create_app(provider=FakeSegmentationProvider(), base_dir=tmp_path / "jobs")
    ) as client:
        created = client.post("/sam/batch", json={
            "image_paths": [sheet_image, sheet_image],
            "prompt": "every sticker",
        })
        assert created.status_code == 200
        batch_id = created.json()["batch_id"]
        assert len(created.json()["job_ids"]) == 2
        summary = None
        for _ in range(200):
            resp = client.get(f"/sam/batch/{batch_id}")
            assert resp.status_code == 200
            summary = resp.json()
            if summary["running"] == 0:
                break
            time.sleep(0.05)
        assert summary["completed"] == 2
        assert summary["total_objects"] == 12
        assert summary["total_cost_usd_estimate"] == 0.0
        assert summary["avg_duration_s"] is not None
        assert client.get("/sam/batch/nope").status_code == 404


def test_job_cost_log_with_billable_provider(sheet_image, tmp_path):
    import asyncio

    from sam_engine.contract import SegmentationRequest
    from sam_engine.jobs import JobStore, run_job

    class BillableFake(FakeSegmentationProvider):
        cost_per_image_usd = 0.0025

    store = JobStore()
    job = store.create(SegmentationRequest(image_path=sheet_image, prompt="x"))
    asyncio.run(run_job(store, BillableFake(), job, tmp_path / "jobs"))
    assert job.status == "completed"
    assert job.api_calls == 1
    assert abs(job.cost_usd_estimate - 0.0025) < 1e-9
    assert job.duration_s is not None and job.duration_s >= 0
