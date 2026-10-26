import asyncio

import httpx
import pytest

from sam_engine.contract import SegmentationRequest
from sam_engine.meta_provider import (
    BillingNotConfiguredError,
    MetaApiError,
    MetaModelApiProvider,
)


def _mock_client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url="https://api.meta.ai")


def test_meta_402_maps_to_billing_error(sheet_image, tmp_path):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.path == "/v1/responses"
        body = request.read()
        assert b"sam-3.1" in body
        return httpx.Response(402, json={"error": "billing_not_configured"})

    provider = MetaModelApiProvider(api_key="test-key",
                                    http_client=_mock_client(handler))
    req = SegmentationRequest(image_path=sheet_image, prompt="every sticker")
    with pytest.raises(BillingNotConfiguredError):
        asyncio.run(provider.segment(req, tmp_path))


def test_meta_missing_output_text_errors(sheet_image, tmp_path):
    def handler(request):
        return httpx.Response(200, json={"output": []})

    provider = MetaModelApiProvider(api_key="test-key",
                                    http_client=_mock_client(handler))
    req = SegmentationRequest(image_path=sheet_image, prompt="x")
    with pytest.raises(MetaApiError):
        asyncio.run(provider.segment(req, tmp_path))


def test_meta_success_path_with_mocked_decode(sheet_image, tmp_path, monkeypatch):
    # One box 100x100 at origin on a 300x200 image, mask payload mocked.
    output_text = ("<0f>0<|box;x1=0;y1=0;x2=99;y2=99;w=300;h=200;c=0.9|>"
                   "<|mask;x=0;y=0;c=0.9;data=100,100,!!!!!!!!!!|>")

    def handler(request):
        return httpx.Response(200, json={
            "output": [{"content": [{"type": "output_text", "text": output_text}]}]
        })

    import sam_engine.meta_provider as meta_module

    def fake_decode(record):
        assert record.mask_width == 100 and record.mask_height == 100
        return bytes([1]) * (100 * 100)

    monkeypatch.setattr(meta_module, "decode_record_mask", fake_decode)
    provider = MetaModelApiProvider(api_key="test-key",
                                    http_client=_mock_client(handler))
    req = SegmentationRequest(image_path=sheet_image, prompt="red square")
    objects = asyncio.run(provider.segment(req, tmp_path))
    assert len(objects) == 1
    obj = objects[0]
    assert obj.bbox == (0, 0, 100, 100)
    assert obj.score == 0.9
    assert abs(obj.area_ratio - (100 * 100) / (300 * 200)) < 1e-9


def test_meta_requires_key():
    with pytest.raises(ValueError):
        MetaModelApiProvider(api_key="")
