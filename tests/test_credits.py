import pytest
from fastapi.testclient import TestClient

from sam_engine.app import create_app
from sam_engine.credits import (
    CreditStore,
    InsufficientBalanceError,
    RefundError,
    UnknownFeatureError,
)
from sam_engine.fake_provider import FakeSegmentationProvider


def _store() -> CreditStore:
    return CreditStore()


def test_grant_and_balance():
    s = _store()
    s.grant("lic-1", "mach-1", 50, "solo", "key-g1")
    bal = s.balance("lic-1")
    assert bal == {"license_id": "lic-1", "balance": 50,
                   "grants_total": 50, "consumed_total": 0}
    assert s.balance("unknown")["balance"] == 0


def test_grant_idempotent_same_key_and_same_tier():
    s = _store()
    e1 = s.grant("lic-1", "mach-1", 50, "solo", "key-g1")
    e2 = s.grant("lic-1", "mach-1", 50, "solo", "key-g1")
    assert e1.entry_id == e2.entry_id
    e3 = s.grant("lic-1", "mach-1", 50, "solo", "key-g2")
    assert e3.entry_id == e1.entry_id  # same tier never grants twice
    assert s.balance("lic-1")["balance"] == 50
    # Upgrade adds a difference grant under a new tier.
    s.grant("lic-1", "mach-1", 30, "agency", "key-g3")
    assert s.balance("lic-1")["balance"] == 80


def test_estimate_and_unknown_feature():
    s = _store()
    assert s.estimate("sam_studio", 3) == 3
    with pytest.raises(UnknownFeatureError):
        s.estimate("nope", 1)


def test_consume_idempotent_and_insufficient():
    s = _store()
    s.grant("lic-1", "mach-1", 2, "solo", "key-g1")
    e1 = s.consume("lic-1", "sam_studio", 1, "job-1", "key-c1")
    assert e1.amount == -1
    assert s.balance("lic-1")["balance"] == 1
    e2 = s.consume("lic-1", "sam_studio", 1, "job-1", "key-c1")
    assert e2.entry_id == e1.entry_id
    assert s.balance("lic-1")["balance"] == 1  # retry charged nothing
    s.consume("lic-1", "sam_studio", 1, "job-2", "key-c2")
    assert s.balance("lic-1")["balance"] == 0
    with pytest.raises(InsufficientBalanceError):
        s.consume("lic-1", "sam_studio", 1, "job-3", "key-c3")
    assert s.balance("lic-1")["balance"] == 0
    assert len(s.ledger) == 3  # grant + 2 consumes, failed consume wrote nothing


def test_refund_once_and_unknown_ref():
    s = _store()
    s.grant("lic-1", "mach-1", 5, "solo", "key-g1")
    s.consume("lic-1", "sam_studio", 2, "job-9", "key-c9")
    assert s.balance("lic-1")["balance"] == 3
    r1 = s.refund("job-9", "key-r9")
    assert r1.amount == 2
    assert s.balance("lic-1")["balance"] == 5
    r2 = s.refund("job-9", "key-r9")
    assert r2.entry_id == r1.entry_id
    assert s.balance("lic-1")["balance"] == 5
    with pytest.raises(RefundError):
        s.refund("job-missing", "key-rx")


def test_reconcile_after_mixed_sequence():
    s = _store()
    s.grant("lic-1", "mach-1", 100, "solo", "key-g1")
    s.grant("lic-2", "mach-2", 10, "trial", "key-g2", is_trial=True)
    for i in range(7):
        s.consume("lic-1", "sam_studio", 1, f"job-{i}", f"key-c{i}")
    s.refund("job-3", "key-r3")
    assert s.reconcile() == []


def test_credit_routes(tmp_path):
    with TestClient(create_app(
            provider=FakeSegmentationProvider(),
            base_dir=tmp_path / "jobs")) as client:
        g = client.post("/credits/grant", json={
            "license_id": "lic-1", "machine_id": "mach-1", "amount": 10,
            "tier": "solo", "idempotency_key": "key-g1"})
        assert g.status_code == 200
        est = client.get("/credits/estimate",
                         params={"feature_id": "sam_studio", "units": 3})
        assert est.json()["cost"] == 3
        c = client.post("/credits/consume", json={
            "license_id": "lic-1", "feature_id": "sam_studio", "units": 3,
            "ref_id": "job-1", "idempotency_key": "key-c1"})
        assert c.status_code == 200
        bal = client.get("/credits/balance/lic-1").json()
        assert bal["balance"] == 7
        poor = client.post("/credits/consume", json={
            "license_id": "lic-1", "feature_id": "sam_studio", "units": 99,
            "ref_id": "job-2", "idempotency_key": "key-c2"})
        assert poor.status_code == 402
        r = client.post("/credits/refund", json={
            "ref_id": "job-1", "idempotency_key": "key-r1"})
        assert r.status_code == 200
        assert client.get("/credits/balance/lic-1").json()["balance"] == 10
        assert client.get("/credits/estimate",
                          params={"feature_id": "nope"}).status_code == 404
