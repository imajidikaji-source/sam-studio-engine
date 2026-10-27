# SAM Studio Engine

Segmentation engine built during the **Meta Global AI Developer Hackathon**
(Oct 26 – Nov 4, 2026) for **OmniStock SAM Studio** — a SAM 3.1 engine that
splits composite sticker/clipart sheets into transparent PNG assets.

This repository contains **only the engine module built during the event**
(provider contract, providers, post-processing, job service). The rest of
OmniStock Studio stays private; per the hackathon rules, startups may bring
a product, but only work from the event counts.

## What is here (Sprint 1–2)

- `sam_engine/contract.py` — `SegmentationRequest` / `SegmentedObject` /
  `SegmentationProvider` (spec §3).
- `sam_engine/fake_provider.py` — deterministic grid provider for
  development and tests (no API needed).
- `sam_engine/meta_provider.py` — SAM 3.1 via Meta Model API, with
  transport-error retries (transient disconnects happen; HTTP error
  statuses are NOT retried).
- `sam_engine/sam_text.py` — parser for the SAM output-text grammar.
- `sam_engine/postprocess.py` — mask → transparent PNG cutout
  (feather, trim, padding) + mask → SVG silhouette (contour trace with
  Douglas-Peucker simplification; single-color silhouette only).
- `sam_engine/jobs.py`, `sam_engine/app.py` — job store + FastAPI service:
  `POST /sam/jobs`, `GET /sam/jobs/{id}`, `WS /sam/jobs/{id}/events`,
  `POST /sam/jobs/{id}/approve`, `POST /sam/jobs/{id}/cancel`,
  `GET /sam/jobs/{id}/objects/{idx}/file`, plus batch
  `POST /sam/batch` and `GET /sam/batch/{id}` (aggregate status,
  detection totals, total cost estimate, average duration).
- `sam_engine/agent.py` — Sprint 3 Muse Spark agent loop
  (`run_agent_job`): goal → LLM prompt → segmentation → deterministic
  mask QA (empty / merged suspected) → retry max 3x → `needs_review`
  on exhaustion, never silent; per-object metadata JSON
  (`metadata.json`). LLM is a protocol: `MuseSparkLLM` (live,
  `muse-spark-1.3`, envelope verified 2026-10-27) or `ScriptedLLM`
  (tests/demos).
- `sam_engine/discovery.py` — Sprint 4 discovery-first loop
  (`run_discovery_job`): vision gate (`MuseSparkVision`, image input
  verified 2026-10-27) returns `{processable, reason, objects}` —
  rejected images cost ZERO SAM calls — then one segmentation call
  per selected concept, cutout + SVG + metadata whose name comes from
  the vision concept (semantic, not positional). Honest limits: the
  vision list varies run to run (a short list is possible), and
  multi-instance concepts may return partial coverage — the picker UI
  must offer "add your own concept" as fallback.
- Per-job API cost log: `api_calls`, `cost_usd_estimate`
  ($0.0025/image for SAM 3.1), `duration_s` on every job.
- `scripts/make_sticker_sheets.py` — regenerates the 10-sheet synthetic
  test set + `tests/fixtures/sam/ground-truth.json` (the PNG fixtures
  themselves are generated, not committed — run the script to recreate
  them). `scripts/run_batch_live.py` — live batch runner used for
  Sprint 2 metrics.

## Verified vs pending (honest labels)

**Verified from official sources** (Meta `meta-sam` repo, `protocol/sam3.md`
and the api-playground README) **and from a live call on 2026-10-26**:
- Base URL `https://api.meta.ai/v1`, model `sam-3.1`, price $2.50/1k images.
- Output grammar: Responses API `output_text` lane, one line per frame,
  image = frame 0, `<0f>id<|box;...|><|mask;...|>` tokens, mask data
  `H,W,payload` with base85 arithmetic-coded payload (`~` lossless /
  `!` one_bit), decoded with the official `meta-sam-parser` package.
- Input is a noun phrase (`input_text`); `metadata.include_confidence`
  asks for `c` values.
- Request envelope confirmed live: `POST /v1/responses` with
  `input_text` + `input_image` (data URL) returns HTTP 200, status
  completed, `images_processed: 1`, and real masks.
- **Prompt wording matters:** concrete noun phrases work
  (`geometric shapes` → 3/3 objects, confidences ~0.74–0.77 on a synthetic
  3-shape image); abstract quantifiers (`every shape`, `all objects`) and
  comma-lists of nouns returned EMPTY output. Sprint 2 confirmed the
  same on sticker sheets: the words `stickers`, `sticker`,
  `colorful stickers`, `cartoon stickers` ALL returned 0 objects, while
  `shapes` returned 4/4 and `geometric shapes` 4/4 on the same sheet.
  SAM 3.1 segments on the visual concept, not the medium — Sheet
  Splitter prompts must name what the objects look like, not what
  they are sold as.

## Sprint 2 batch result (live, 2026-10-27)

10 synthetic sticker sheets, prompt `shapes`, via the real SAM 3.1 API
(full detail in `sprint2-metrics.json`):

- Detection: **63/63 objects (100%)** — including the deliberate edge
  cases: touching objects 4/4, tiny objects (<2% area) 8/8, soft drop
  shadows 5/5, and white-on-white shapes 4/4.
- Speed: **average 33.8 s/sheet** vs the <30 s target — the average is
  skewed by one 176.8 s sheet (transport retries); the median sheet
  took ~20 s, so the target is within reach once retry latency is
  tightened, not a model-speed problem.
- Cost: **$0.025 total** for 10 sheets ($0.0025 each), from free credits.
- Caveat (honest): the 100% is a detection *count* vs the drawn shapes;
  mask precision (IoU against ground truth) is not scored yet — that
  lands with the 30-sheet set in Sprint 5.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -e .[dev]
# Fake provider (no API key):
SAM_PROVIDER=fake .venv/bin/uvicorn sam_engine.app:create_app --factory --port 8000
# Meta provider (needs key + billing/credits):
SAM_PROVIDER=meta META_MODEL_API_KEY=... .venv/bin/uvicorn sam_engine.app:create_app --factory --port 8000
```

Note: `create_app` is a factory; uvicorn needs `--factory` as shown.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Roadmap

- Sprint 2: done (batch endpoint, SVG silhouette, cost log, 10-sheet
  live batch 63/63).
- Sprint 3: done early (agent loop + Muse Spark live verified +
  metadata; subject mode via same path; XMP/CSV wiring deferred to
  product integration).
- Sprint 4: credit Worker wiring (1 credit per approved sheet) + UI +
  discovery-first picker mode (chunk 1, engine loop, done 2026-10-27:
  gate + list + per-concept segmentation + selection; live-verified
  on mosque — domes and minarets segmented individually — and watch).
- Sprint 5: 30-sheet test set, QA gate, 3-minute demo video, submission.
