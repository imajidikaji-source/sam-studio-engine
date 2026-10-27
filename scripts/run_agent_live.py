#!/usr/bin/env python3
"""Sprint 3 live verification: Muse Spark envelope probe + agent loop runs.

1. Tiny Muse Spark call to confirm the Responses-API text shape.
2. Agent loop on the mosque illustration (goal -> prompt -> mask -> metadata).
3. Agent loop on a sticker sheet (retry path if the LLM first says 'stickers').

Writes agent-live-results.json + per-run outputs under agent-live-out/.
The API key comes from the connector surrogate (never printed/stored).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from sam_engine.agent import MuseSparkLLM, run_agent_job
from sam_engine.meta_provider import MetaModelApiProvider

ROOT = Path(__file__).resolve().parent.parent


async def main() -> int:
    for var in ("NO_PROXY", "no_proxy"):
        if var in os.environ:
            os.environ[var] = ",".join(
                h for h in os.environ[var].split(",")
                if h.strip() and not h.strip().startswith("[")
            )
    surrogate = dynamic_credential_entry("custom.meta-model-api")["surrogate"]
    client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {surrogate}"}, timeout=300.0
    )
    llm = MuseSparkLLM(api_key=surrogate, http_client=client)
    provider = MetaModelApiProvider(api_key=surrogate, http_client=client)

    # 1. Envelope probe.
    probe = await llm.complete("Reply with exactly: OK")
    print(f"envelope probe -> {probe!r}")

    out_root = ROOT / "agent-live-out"
    results = {"envelope_probe": probe, "runs": []}

    # 2. Mosque illustration.
    mosque = "/home/hatch/workspace/user/files/1000090737_13_71le.png"
    res = await run_agent_job(
        mosque, "mosque illustration for a sticker",
        provider, llm, out_root / "mosque", mode="sheet")
    print(f"mosque: status={res.status} prompt={res.final_prompt!r} "
          f"attempts={len(res.attempts)} objects={len(res.objects)} "
          f"metadata={res.metadata_status}")
    for a in res.attempts:
        print(f"   attempt prompt={a.prompt!r} n={a.n_objects} issues={a.issues}")
    for o in res.objects:
        print(f"   obj{o.index}: meta={o.metadata}")
    results["runs"].append({
        "image": "mosque", "status": res.status,
        "final_prompt": res.final_prompt,
        "attempts": [{"prompt": a.prompt, "n": a.n_objects,
                      "issues": a.issues} for a in res.attempts],
        "metadata_status": res.metadata_status,
        "objects": [{"index": o.index, "bbox": o.bbox,
                     "metadata": (o.metadata.__dict__ if o.metadata else None)}
                    for o in res.objects],
    })

    # 3. Sticker sheet (retry path lives here if the LLM says 'stickers').
    sheet = str(ROOT / "tests" / "fixtures" / "sam" / "sticker-01.png")
    res = await run_agent_job(
        sheet, "sticker sheet for kids",
        provider, llm, out_root / "sticker", mode="sheet")
    print(f"sticker: status={res.status} prompt={res.final_prompt!r} "
          f"attempts={len(res.attempts)} objects={len(res.objects)} "
          f"metadata={res.metadata_status}")
    for a in res.attempts:
        print(f"   attempt prompt={a.prompt!r} n={a.n_objects} issues={a.issues}")
    for o in res.objects:
        print(f"   obj{o.index}: meta={o.metadata}")
    results["runs"].append({
        "image": "sticker-01", "status": res.status,
        "final_prompt": res.final_prompt,
        "attempts": [{"prompt": a.prompt, "n": a.n_objects,
                      "issues": a.issues} for a in res.attempts],
        "metadata_status": res.metadata_status,
        "objects": [{"index": o.index, "bbox": o.bbox,
                     "metadata": (o.metadata.__dict__ if o.metadata else None)}
                    for o in res.objects],
    })
    (ROOT / "agent-live-results.json").write_text(json.dumps(results, indent=2))
    await client.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
