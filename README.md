# Catalyst Scanner — Architecture Prototype

Demo of a real-time U.S. stock **news-catalyst analysis and alerting** pipeline for discretionary day trading.
**Decision support only. It never places trades.**

This prototype proves the architecture against canned, real-sourced test cases. Live feeds are
deliberately deferred (see [Scope](#scope-deferred-not-missed)).

## Quickstart (Windows, Python 3.11+)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # defaults to local LM Studio
python -m pytest -q             # 21 deterministic/pipeline tests, no LLM needed
python run_demo.py --fresh      # run all populated test cases end to end
python run_demo.py --provider anthropic --fresh   # needs ANTHROPIC_API_KEY in .env
```

`run_demo.py` flags: `--provider lmstudio|anthropic`, `--fresh` (re-analyze; clears **only that provider's** cache),
`--case <id-prefix>`, `-v`.

## Dashboard (Google Finance-style UI)

```powershell
streamlit run app.py
```

- **Ticker strip:** price, day change and an assessment-coloured dot. Click a card to open that ticker.
- **Price chart:** 1M / 6M / YTD / 1Y / 5Y ranges, with the catalyst marked on the event date.
- **Key stats**, plus a **Move check**. This deterministic check compares the move the source claims with the close-to-close move in the price data.
- **Catalyst analysis:** assessment banner, the 12-field alert, reasoning, evidence, risk factors and data caveats, plus the test-case PASS/FAIL.
- **Scorecard tab:** Claude vs Gemma side by side.
- Opening the page reads saved analyses only (instant). LLM calls happen only when you press **Run analysis** in the sidebar.
- Prices come from **yfinance, which is demo-only** (unofficial access). It sits behind `PriceSource` in `scanner/prices.py`; production would use Alpaca market data.

### Move check findings
The price data does **not** support several of the fixture's claimed moves:

| Ticker | Fixture claim | Price data |
|---|---|---|
| NBIS | +17.3% on Aug 12, 2026 | +34.1% (the date matches, the size doesn't) |
| RCAT | ~+22% on Aug 6, 2026 | +0.1%; no day in Aug 2026 above +8.8% |
| CELC | +660% since end of July 2026 | about −9% (even the same window in 2025 is +259%) |

Re-check these against the original sources before the client demo. If the dates are wrong, add correct
`published_at` values.

## Deploying publicly (Streamlit Community Cloud), NVIDIA NIM only

The public app uses **NVIDIA NIM** (`google/gemma-4-31b-it` on NVIDIA's free hosted catalog). Your Anthropic key
**never leaves your machine**:

1. Push this folder to GitHub. `.env`, `.streamlit/secrets.toml` and `data/*.db` are gitignored. Check with `git status` before pushing.
2. At share.streamlit.io, create the app from the repo with main file `app.py`. Under Advanced settings, pick Python 3.12.
3. In the app's **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real `nvapi-` key.

Safety controls:
- `ENABLED_PROVIDERS = "nim"`: Claude and LM Studio are hidden, and a crafted `?p=anthropic` URL falls back to NIM.
  Leaving `ANTHROPIC_API_KEY` out of the secrets is the real guarantee.
- **Run analysis** only analyzes cases with no saved result, so there is at most one NIM call per test case (4 today).
- **Re-run all** spends credits on every click and is locked behind `RERUN_PASSCODE`.
- Streamlit Cloud storage is **temporary**: SQLite resets when the app restarts. Press *Run analysis* once after each deploy or restart.
- yfinance can be rate-limited from cloud IPs. The app degrades to "No price data" instead of crashing.

## Architecture

```
NewsSource.fetch()  ──►  rules.evaluate()  ──►  Store.find_duplicate()  ──►  classify_catalyst()
 (fixture today;         deterministic:          exact fingerprint +          provider-agnostic LLM,
  Alpaca / EDGAR /       % move, window,         fuzzy cross-wire match;      schema-validated output,
  wires later)           cap tier, freshness,    analysis cache               retry + backoff
                         dilution keywords
                                                                                      │
   Store.should_alert()  ◄──  alerting.build_alert()  ◄──  rules.apply_guardrails() ◄─┘
   duplicate-alert           exact client format,          hard rules the LLM cannot override
   prevention                deterministic Watch rec
```

| Module | Responsibility | LLM? |
|---|---|---|
| `scanner/ingestion.py` | `NewsSource` interface; `FixtureNewsSource`; stubs for Alpaca/EDGAR | no |
| `scanner/rules.py` | % move parsing, move window, cap tier, freshness, dilution/reverse-split detection, dedup fingerprints, monitor window + NYSE holidays, **guardrails** | no |
| `scanner/classifier.py` | `classify_catalyst()`; `LMStudioProvider`, `AnthropicProvider`, `NvidiaNimProvider` | **yes** |
| `scanner/storage.py` | SQLite: seen news, per-story source list, analysis cache, sent alerts | no |
| `scanner/alerting.py` | Alert build/format; Watch recommendation from assessment + cap tier | no |
| `scanner/evaluation.py` | Parses `expected_classification` into explicit PASS/FAIL checks | no |
| `scanner/pipeline.py` | Orchestration; degrades gracefully on provider outage | — |
| `run_demo.py` | CLI display | — |

### Key design decisions
- **Model-independent.** Every provider returns the same Pydantic `CatalystAnalysis`. LM Studio uses
  `response_format=json_schema`. Anthropic uses a forced tool call. Adding OpenAI or Gemini is one subclass.
  Invalid output is rejected, and the validator error is fed back to the model on retry.
- **Verdict-last schema.** Evidence and risk factors come before the assessment field, so the model
  lists the negatives before it commits to a verdict.
- **Answer key never reaches the AI.** `reasoning_notes` and `expected_classification` are kept off `NewsEvent`.
- **Deterministic guardrails override the AI:**
  1. A stale or unverifiable timestamp forces **NO FRESH CATALYST**. This is the "no false catalyst matching" rule.
  2. Dilution keywords (offering, warrants, ATM, S-1/S-3/424B, PIPE, convertibles) force `Recent dilution: Yes`.
  3. A financing or offering event can never be **STRONG CATALYST**.
- **Source provenance.** Secondary coverage (Barchart, StocksToTrade) is flagged as "original release not verified".
  Cross-wire duplicates merge into one story that lists every source, first-seen source first.
- **Alert dedup.** A story re-alerts only on a meaningful change: a new story, a classification change, or a further move of 10 points or more.

## Test cases (`data/test_cases.json`)

Cases with null `ticker/company/headline/expected_classification/source` are **skipped automatically**.
To add one, fill those fields. Optional fields the pipeline already understands:

| Field | Effect |
|---|---|
| `published_at` | ISO-8601 **with offset**, e.g. `"2026-08-12T08:31:00-04:00"`. Enables a real news age and the freshness guardrail |
| `body` | Full release or filing text. The AI analyzes this instead of the headline alone |
| `market_cap_usd` | Deterministic small/large-cap tier (< $2B = small) instead of the category hint |

`expected_classification` grading: text **starting with** `STRONG CATALYST`, `POTENTIAL PUFFY`/`PUFFY`,
`MIXED`, or `NO FRESH CATALYST` asserts that exact assessment. `Recent dilution: Yes` asserts the
dilution flag. `not itself a bullish` asserts the assessment is not STRONG.

## Current results: same pipeline, two providers

| Case | Gemma 4 12B QAT (local) | Claude Sonnet 5 (API) |
|---|---|---|
| CELC Phase 3 data | PASS (STRONG) | PASS (STRONG, confidence 0.75) |
| UPXI registered direct offering | PASS (PUFFY/WEAK, dilution Yes) | PASS (PUFFY/WEAK, dilution Yes) |
| NBIS $1B+ contract | PASS (STRONG); category mislabelled "M&A" | PASS (STRONG, Material Contract); binding "Not verified" |
| RCAT miss + reaffirmed guide | **FAIL**: STRONG, judges that the partnership outweighs the miss | PASS (MIXED): "reaffirmation, not a raise" |
| **Score / latency** | **3/4, ~7–10 s per case** (thinking off) | **4/4, ~5–7 s per case** |

Gemma 4 thinking mode is controlled by `LMSTUDIO_REASONING_EFFORT`. The default `none` is fast. With thinking on,
cases took 190 s or more and hit the 120 s timeout.

Claude also flagged, without being told, that the CELC and NBIS sources are secondary or promotional articles,
and lowered its confidence because of it. Both runs are headline-only.

Other behaviour shown: cross-run alert suppression, the analysis cache, and graceful handling of a provider outage.

## Scope (deferred, not missed)

| Item | Status / where it plugs in |
|---|---|
| Trade Ideas | Post-hire discovery: confirm whether the subscription exposes a supported export or API before rebuilding movers |
| Alpaca real-time news (WebSocket) | `AlpacaNewsSource` stub. Push-based, so breaking news skips the 15-min scan |
| Benzinga API, wire RSS, company IR | More `NewsSource` subclasses. Cross-wire dedup is already in `Store.find_duplicate` |
| SEC EDGAR | `EdgarFilingSource` stub. Latest-filings feed for 8-K/6-K/S-1/S-3/424B, used to verify and enrich |
| Market data | No live prices or market cap. Move % is parsed from fixture text |
| Calendar | 2026 NYSE full-day holidays hardcoded. No early closes (use `exchange_calendars` later) |
| Reliability | Retry with backoff in the AI layer only. No heartbeat, no process supervisor, no alert channel (console only) |
| Full-text fetch | Headline-only today. Fetching and parsing the release or filing body is the biggest quality lever |
