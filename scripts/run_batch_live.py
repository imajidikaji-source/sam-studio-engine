#!/usr/bin/env python3
"""Sprint 2 live batch run: 10 sticker sheets through real SAM 3.1.

Drives the job runner (same code path as the service) with the Meta
provider. The API key comes from the connector surrogate (never printed,
never stored) — the script only runs where the connector is registered.
Cost: 10 images x $0.0025 = $0.025 from free credits.

Writes sprint2-metrics.json next to the fixtures + per-sheet job outputs
under batch-out/.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from sam_engine.contract import SegmentationRequest
from sam_engine.jobs import JobStore, run_job
from sam_engine.meta_provider import MetaModelApiProvider

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "sam"
PROMPT = "shapes"


async def main() -> int:
    # Workaround: this VM's NO_PROXY contains bracketed IPv6 entries
    # ([::1], [fd8b:...]) that httpx's env-proxy parsing chokes on
    # ("Invalid port: ':1]'"). Strip bracketed entries for this process
    # only; the egress proxy itself still comes from the environment.
    import os

    for var in ("NO_PROXY", "no_proxy"):
        if var in os.environ:
            os.environ[var] = ",".join(
                h for h in os.environ[var].split(",")
                if h.strip() and not h.strip().startswith("[")
            )
    surrogate = dynamic_credential_entry("custom.meta-model-api")["surrogate"]
    client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {surrogate}"},
        timeout=300.0,
    )
    provider = MetaModelApiProvider(api_key=surrogate, http_client=client)
    store = JobStore()
    ground_truth = json.loads((FIXTURES / "ground-truth.json").read_text())
    results = []
    out_root = ROOT / "batch-out"
    for entry in ground_truth:
        image_path = str(FIXTURES / entry["file"])
        job = store.create(SegmentationRequest(
            image_path=image_path, prompt=PROMPT, mode="sheet"))
        started = time.time()
        await run_job(store, provider, job, out_root)
        elapsed = time.time() - started
        detected = len(job.objects)
        expected = entry["expected_objects"]
        results.append({
            "file": entry["file"],
            "expected": expected,
            "detected": detected,
            "status": job.status,
            "error": job.error,
            "duration_s": round(elapsed, 1),
            "cost_usd": job.cost_usd_estimate,
            "notes": entry["notes"],
            "objects": [{"bbox": o.bbox, "score": o.score,
                         "area_ratio": round(o.area_ratio, 4),
                         "status": o.status} for o in job.objects],
        })
        print(f"{entry['file']}: expected={expected} detected={detected} "
              f"status={job.status} {elapsed:.1f}s err={job.error}")
    await client.aclose()
    total_expected = sum(r["expected"] for r in results)
    total_detected = sum(r["detected"] for r in results)
    completed = [r for r in results if r["status"] == "completed"]
    clean = [r for r in results if "clean" in r["notes"]]
    clean_expected = sum(r["expected"] for r in clean)
    clean_detected = sum(r["detected"] for r in clean)
    metrics = {
        "prompt": PROMPT,
        "sheets": len(results),
        "total_expected": total_expected,
        "total_detected": total_detected,
        "overall_detection_rate": round(total_detected / total_expected, 3)
        if total_expected else 0,
        "clean_detection_rate": round(clean_detected / clean_expected, 3)
        if clean_expected else 0,
        "avg_duration_s": round(
            sum(r["duration_s"] for r in completed) / len(completed), 1)
        if completed else None,
        "total_cost_usd": round(sum(r["cost_usd"] for r in results), 4),
        "sheets_completed": len(completed),
        "results": results,
    }
    (ROOT / "sprint2-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "results"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
