"""Studio-wide credit system — Sprint 4 chunk 2, reference implementation.

Implements the semantics of ``spek-sistem-kredit-omnistock.md`` (Worker-side
wallet + append-only ledger + tariff registry):

- One wallet per license_id (bound to a machine_id), usable across all
  API-cost features. Credits never expire, never transfer between
  licenses, never convert back to money.
- Every mutation is a ledger entry with a UNIQUE idempotency key, so
  retries, double-clicks, and crashes can never double-charge.
- Tariffs live in the registry: changing a price changes the registry,
  never the client.
- Two-phase flow: ``estimate`` before the job (refuse early when the
  balance is short, before any paid API call), ``consume`` exactly once
  at commit (approve). Failure or cancel before commit = no consume.
- ``refund`` restores a consumed amount exactly once, only for a ref_id
  that was consumed and never refunded.

SCOPE HONESTY: this is the engine-side reference implementation with an
in-memory store, built so the logic, the idempotency rules, and the API
shape are verified by tests and demonstrable from the public repo. The
production Cloudflare Worker ports these exact semantics onto durable
storage (KV for fast balance reads + ledger records, D1 if volume demands)
and enforces license-token + machine-id authentication at the edge —
authentication deliberately does NOT live in this module, because the
engine demo service is not the system of record. Do not deploy this
in-memory store as the real wallet.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


class UnknownFeatureError(Exception):
    pass


class InsufficientBalanceError(Exception):
    pass


class RefundError(Exception):
    pass


@dataclass(frozen=True)
class Tariff:
    feature_id: str
    display_name: str
    tariff_credits: int
    tariff_unit: str  # "per_sheet" | "per_image" | "per_job"


@dataclass
class Wallet:
    license_id: str
    machine_id: str
    balance: int = 0
    grants_total: int = 0
    consumed_total: int = 0
    is_trial: bool = False


@dataclass
class LedgerEntry:
    entry_id: str
    license_id: str
    type: str  # grant | consume | topup | adjust | refund
    amount: int  # signed: grants/topup/refund positive, consume negative
    feature_id: str | None
    ref_id: str | None
    idempotency_key: str
    created_at: str
    note: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreditStore:
    """Thread-safe in-memory wallet + ledger + registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.wallets: dict[str, Wallet] = {}
        self.ledger: list[LedgerEntry] = []
        self._by_key: dict[str, LedgerEntry] = {}
        self._granted_tiers: dict[tuple[str, str], LedgerEntry] = {}
        self.registry: dict[str, Tariff] = {
            "sam_studio": Tariff(
                feature_id="sam_studio",
                display_name="SAM Studio — Sheet Splitter & Cutout",
                tariff_credits=1,
                tariff_unit="per_sheet",
            )
        }

    def register_tariff(self, tariff: Tariff) -> None:
        with self._lock:
            self.registry[tariff.feature_id] = tariff

    def estimate(self, feature_id: str, units: int = 1) -> int:
        tariff = self.registry.get(feature_id)
        if tariff is None:
            raise UnknownFeatureError(f"unknown feature: {feature_id}")
        if units < 1:
            raise ValueError("units must be >= 1")
        return tariff.tariff_credits * units

    def balance(self, license_id: str) -> dict:
        wallet = self.wallets.get(license_id)
        if wallet is None:
            return {
                "license_id": license_id,
                "balance": 0,
                "grants_total": 0,
                "consumed_total": 0,
            }
        return {
            "license_id": license_id,
            "balance": wallet.balance,
            "grants_total": wallet.grants_total,
            "consumed_total": wallet.consumed_total,
        }

    def _append(
        self, license_id: str, type_: str, amount: int,
        feature_id: str | None, ref_id: str | None,
        idempotency_key: str, note: str = "",
    ) -> LedgerEntry:
        entry = LedgerEntry(
            entry_id=uuid.uuid4().hex,
            license_id=license_id,
            type=type_,
            amount=amount,
            feature_id=feature_id,
            ref_id=ref_id,
            idempotency_key=idempotency_key,
            created_at=_now(),
            note=note,
        )
        self.ledger.append(entry)
        self._by_key[idempotency_key] = entry
        return entry

    def grant(
        self, license_id: str, machine_id: str, amount: int, tier: str,
        idempotency_key: str, is_trial: bool = False, note: str = "",
    ) -> LedgerEntry:
        """One-time activation grant. Idempotent twice over: same key
        returns the original entry; same (license, tier) never grants
        twice (an upgrade adds a *difference* grant under a new tier)."""
        with self._lock:
            if idempotency_key in self._by_key:
                return self._by_key[idempotency_key]
            if (license_id, tier) in self._granted_tiers:
                return self._granted_tiers[(license_id, tier)]
            wallet = self.wallets.get(license_id)
            if wallet is None:
                wallet = Wallet(
                    license_id=license_id, machine_id=machine_id,
                    is_trial=is_trial,
                )
                self.wallets[license_id] = wallet
            wallet.balance += amount
            wallet.grants_total += amount
            entry = self._append(
                license_id, "grant", amount, None, None,
                idempotency_key, note or f"tier:{tier}",
            )
            self._granted_tiers[(license_id, tier)] = entry
            return entry

    def consume(
        self, license_id: str, feature_id: str, units: int, ref_id: str,
        idempotency_key: str,
    ) -> LedgerEntry:
        """Charge at commit time. Idempotent by key; never lets the
        balance go negative; never writes on failure (caller checks
        first and only calls on commit)."""
        with self._lock:
            if idempotency_key in self._by_key:
                return self._by_key[idempotency_key]
            cost = self.estimate(feature_id, units)
            wallet = self.wallets.get(license_id)
            balance = wallet.balance if wallet else 0
            if balance < cost:
                raise InsufficientBalanceError(
                    f"balance {balance} < cost {cost} for {feature_id}"
                )
            if wallet is None:
                wallet = Wallet(license_id=license_id, machine_id="")
                self.wallets[license_id] = wallet
            wallet.balance -= cost
            wallet.consumed_total += cost
            return self._append(
                license_id, "consume", -cost, feature_id, ref_id,
                idempotency_key,
            )

    def refund(self, ref_id: str, idempotency_key: str) -> LedgerEntry:
        """Restore a consumed amount exactly once. Only a ref_id with a
        prior consume and no prior refund qualifies."""
        with self._lock:
            if idempotency_key in self._by_key:
                return self._by_key[idempotency_key]
            consume_entry = None
            for entry in self.ledger:
                if entry.ref_id == ref_id and entry.type == "consume":
                    consume_entry = entry
                if entry.ref_id == ref_id and entry.type == "refund":
                    return entry  # already refunded: idempotent
            if consume_entry is None:
                raise RefundError(f"no consume found for ref_id {ref_id}")
            wallet = self.wallets[consume_entry.license_id]
            amount = -consume_entry.amount
            wallet.balance += amount
            wallet.consumed_total -= amount
            return self._append(
                consume_entry.license_id, "refund", amount,
                consume_entry.feature_id, ref_id, idempotency_key,
            )

    def reconcile(self) -> list[str]:
        """Check every wallet balance equals its ledger sum. Returns a
        list of mismatch descriptions (empty = clean)."""
        with self._lock:
            sums: dict[str, int] = {}
            for entry in self.ledger:
                sums[entry.license_id] = sums.get(entry.license_id, 0) + entry.amount
            problems = []
            for license_id, wallet in self.wallets.items():
                if sums.get(license_id, 0) != wallet.balance:
                    problems.append(
                        f"{license_id}: wallet {wallet.balance} != "
                        f"ledger {sums.get(license_id, 0)}"
                    )
            return problems
