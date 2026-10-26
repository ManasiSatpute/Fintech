# WhatsApp Bond Trade Extraction

Turns unstructured WhatsApp broker chats into structured bond-trade records
(bond, buyer/seller, quantity, yield, settlement, status) using a hybrid of
deterministic rules and an LLM. It is an **assistive extraction system**: it does
not execute trades.

Branch with the latest work: `pipeline-improvements`.

---

## 1. Where the project stands

**Done (prompts 0 to 10 and 13 to 15, plus follow-up fixes)**

- Central settings in `config.py` (every tunable overridable by environment variable).
- Parser with stable chat identity (`chat_key` from group id / counterpart number), stable message ids, day-first/month-first date detection, no silent `datetime.now()` fallback, reply links and media kept.
- Noise is tagged, not dropped. Short acknowledgements ("ok", "closed") are never discarded, because they can confirm a trade.
- Weighted relevance score instead of a hard ISIN gate. Issuer names live in `issuer_aliases.json`.
- Per-chat grouping where a reply link beats the time gap.
- Numbered LLM snippets mask contact details before the LLM, while preserving trade data; local placeholders restore model results after validation.
- `lexicon.json` with English / Hinglish / Marathi trade cues.
- `normalize.py`: quantity (Cr / L / units / ranges), yield vs coupon, bps, T+n and "Mon VD" settlement dates, maturity parsing, ISIN checksum.
- New extraction schema (several deals per window, evidence line numbers, nothing defaulted) and validation (evidence check, stricter CONCLUDED rule, confidence, `needs_review`).
- Partial-ISIN handling: anything that is not a full valid ISIN moves to `partial_isin_text`.
- Yield cleanup: sign stripped, out-of-range yields (outside about 3 to 25) nulled and flagged, "ISIN - number" treated as units.
- Extraction and database views show chat names, status and review badges, confidence, evidence, missing fields, quantities, amounts, settlement details, and report counts.
- Last test run: **99 tests passing** (plus 15 subtests).

**Not done yet (in order)**

1. Make a closed trade reach `CONCLUDED` (see Known issues, item 1).
2. `CANCELLED` handling for explicit refusals (see Known issues, item 2).
3. **Prompt 11, database schema** (new statuses and fields; `/store` is blocked until this is done).
4. Prompt 12, API behaviour (remove the 8-context cap, per-context error handling).
5. Prompt 16, before/after comparison against the saved baseline.
6. Primary / secondary bond identity ladder and LLM-based context assignment (the mentor's reviews, section 8).

Full details for each are in section 9.

---

## 2. Quick start (Windows, PowerShell)

### Prerequisites

- Python 3.13 (3.11+ should work).
- [Ollama](https://ollama.com/download) installed (this provides the `ollama` command).
- Git.

### One-time setup

```powershell
# 1. virtual environment (keep it OUTSIDE OneDrive to avoid sync slowness)
python -m venv C:\venvs\fintech
C:\venvs\fintech\Scripts\Activate.ps1        # prompt should start with (fintech)

# 2. install what the project actually uses
pip install fastapi uvicorn python-multipart pydantic ollama langchain-ollama pytest
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

> `requirements.txt` in the repo is a dump of a whole Python environment (it includes
> unrelated packages and a Linux-only `uvloop`). Do not install it as is. Replace it with
> `pip freeze > requirements.txt` from inside the venv once the app runs cleanly. Do not add `alembic`; the
> project does not use it.

### The model

Default model is `gemma4:31b-cloud`. The `-cloud` tag means it runs on Ollama's servers:

```powershell
ollama --version
ollama signin                 # needs an Ollama account
ollama run gemma4:31b-cloud "say hello"
```

Chat text is sent to Ollama's servers with a cloud model. Confirm this is acceptable for the chats you use.

To use a local model instead, set `MODEL_NAME` (for example `qwen2.5:7b` after `ollama pull qwen2.5:7b`).
**Use the same model for every before/after comparison**, or you will be measuring the model change and not the pipeline.

### Run the tests

```powershell
python -m pytest -q
```

### Run the app

Set these in the same terminal first, then start the server:

```powershell
$env:MAX_CONTEXTS="none"                     # process every context (default cap is 8)
$env:OWN_SENDER_NAMES="<owner display name>" # comma-separated if more than one
python -m uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>, upload a chat file (or load a sample), and run extraction.
A full day of chats takes several minutes, since each context is a separate model call.

Environment variables only last for the terminal window they were set in.

### Baseline report

```powershell
python scripts\baseline_report.py data\real_chat.jsonl
```

Writes a timestamped JSON report to `baseline/` (git-ignored): message counts by chat, system-filter drops, windows, relevant windows, contexts sent to the model, contexts cut off by the cap, and each deal's raw model output next to its validated version. Falls back to a mock model (and says so in `llm_mode`) if Ollama is unreachable. A mock-mode report is not a valid baseline.

Run a concise extraction smoke report with aggregate counts only:

```powershell
python scripts\smoke_run.py data\real_chat.jsonl
```

---

## 3. Configuration (`config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `gemma4:31b-cloud` | Ollama model |
| `MAX_CONTEXTS` | `8` | Max contexts sent to the model. `none` = no limit |
| `OWN_SENDER_NAMES` | empty | Owner's display names (comma-separated). Used to work out which side the owner is on |
| `GAP_MINUTES` | `15` | Time gap that starts a new window |
| `DAY_FIRST_DEFAULT` | `true` | Date order when a file is ambiguous |
| `LLM_RETRIES` | `1` | Retries on malformed model output |
| `MARKET_HOLIDAYS` | empty | ISO dates (comma-separated) skipped when counting business days |
| `PII_MASKING_ENABLED` | `true` | Mask contact details and account fragments before the model |
| `MASK_PERSON_NAMES` | `false` | Also mask person names |
| `RELEVANCE_THRESHOLD` | `3` | Minimum relevance score for a window to reach the model |
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.7` | Below this, a deal is flagged `needs_review` |
| `ALLOWED_ORIGINS` | localhost only | CORS origins |

---

## 4. Project layout

| Path | Purpose |
|---|---|
| `app.py` | FastAPI app: upload, extract, store, view data, samples |
| `pipeline.py` | Parsing, noise tagging, relevance, grouping, snippet building, LLM call, validation |
| `normalize.py` | Pure functions: quantity, yield, bps, settlement dates, maturity, ISIN checks |
| `db.py` | SQLite storage (`bond_deals.db`, git-ignored). **Schema update pending (prompt 11)** |
| `config.py` | All settings |
| `lexicon.json` | Multilingual trade cues (agreement, pass, pending, relative days). Needs review by a Hindi/Marathi speaker |
| `issuer_aliases.json` | Issuer name aliases. Add public issuer names only |
| `extraction_examples.json` | Few-shot examples for the extraction prompt (synthetic) |
| `scripts/baseline_report.py` | Baseline / comparison report |
| `tests/` | Unit tests (parser, normalizers, schema, validation, lexicon, baseline script) |
| `static/` | Web UI (`index.html`, `style.css`) |
| `testFiles/` | Sanitised sample chats |
| `data/` | **Git-ignored.** Real chats live here only |
| `baseline/` | **Git-ignored.** Generated reports |
| `AGENTS.md`, `.github/copilot-instructions.md` | Rules for AI coding assistants (same content). Keep them in sync |
| `WhatsApp_pipeline.py`, `check_sql_data.py` | Old prototype / helper. `app.py` does not use them |

### API endpoints

`GET /` UI, `POST /upload`, `POST /extract/{upload_id}`, `POST /store/{upload_id}`, `GET /data`, `GET /samples`, `POST /load-sample/{filename}`.

---

## 5. How the pipeline works

1. **Parse** JSONL or plain WhatsApp export into messages with `chat_key`, `chat_name`, `chat_type`, `is_own`, `is_media`, reply text, stable `message_id`.
2. **Tag noise** (system events, greetings, acks). Nothing is silently dropped; counts are reported.
3. **Hints and relevance score** from deterministic signals (ISIN with checksum, amounts, yields, action words, settlement words, issuer aliases, bond vocabulary).
4. **Group per chat** into windows; reply links beat the time gap; orphan acknowledgements do not open windows.
5. **Build the snippet**: numbered lines (`#n`), `[OWN]` tag, quoted replies, phone numbers masked.
6. **LLM extraction** into a Pydantic schema (`{"deals": [...]}`), temperature 0, retries, evidence line numbers per field, nothing defaulted.
7. **Validate**: evidence check (a value must appear in its cited lines or it becomes null), normalisers, CONCLUDED rules, confidence, `needs_review`.
8. **Store** to SQLite (blocked until prompt 11).

**Design principle:** the LLM interprets meaning; deterministic code records facts. Unknown means null, never a default.

---

## 6. Domain rules (from the mentor notes)

- **Direction words:** "I have an offer" = sell. "I have a bid" = buy. "Any offer?" = the asker wants to buy. "Any bid?" = the asker wants to sell.
- **Yield:** record YTM (yield to maturity). Mostly the number quoted is YTM, not the coupon. The percentage before the issuer name is the coupon.
- **Quantity** may be an amount (for example 25 Cr) and is indicative. Quantity = amount / face value. Numbers up to the thousands are usually unit counts.
- **Settlement:** identify T+0 / T+1 / T+2 from words such as "today" and "tomorrow". "Mon VD" means the next Monday value date.
- **Bond identity:** primary = ISIN (unique per bond). Secondary (when no ISIN) = issuer name, maturity, coupon, credit rating, sector (PSU / NBFC). A transaction counts as complete only if the ISIN is captured. Open question: how a trade agreed on secondary details is held until the ISIN is resolved.
- **Completion:** all mentioned terms are clear and the other side confirms (for example "okay", "done", "closed").
- **Cancellation:** an explicit "pass" means cancelled. No reply means the offer stays initiated (not cancelled).
- **No official cut-off time.** Chats can run around the clock.
- **Only buyer and seller** are modelled.
- **Models:** open-weight models are preferred (the mentor mentioned Groq).
- **Out of scope:** audit trails, role IDs, dashboard. Manual verification that a trade really happened is expected before a record is trusted.

---

## 7. Data and privacy rules (read before touching data)

- The chats are confidential. **Real chats must stay in `data/`** (git-ignored). Never commit them.
- `baseline/` reports contain chat excerpts. They are git-ignored; do not paste them anywhere.
- Committed fixtures in `testFiles/` and `chat_2.txt` are anonymised. Add only synthetic or anonymised data.
- Do not print or quote the contents of `data/` or `baseline/` in logs, chat tools or issues. Report counts and field names only.
- Before pushing: `git status --short` and `git ls-files data baseline`. Stage files by name, not `git add .`.
- The repo is currently public. Make it private (and add the mentor as a collaborator) unless the data owner has approved publishing.
- A cloud LLM receives chat text. Keep masking on and confirm approval for real chats.

---

## 8. Mentor reviews to implement

1. **Not every message has an ISIN.** A conversation may begin with issuer, maturity or other details. Handle this when grouping messages.
2. **Do not identify context by ISIN.** Send a batch of messages to an LLM and ask whether each message belongs to an existing context or a different one.
3. **Identify the bond in two ways.** Primary: the ISIN, if present. Secondary: issuer name, maturity, credit rating, coupon and similar details.

---

## 9. Work plan for the remaining items

Each item is one self-contained task. Do them in order, run `python -m pytest -q` after each, and keep `scripts/baseline_report.py` and its tests working.

**A. Make a closed trade reach CONCLUDED (do first).**
Currently no trade in the real chat is CONCLUDED, including a clear "ok lets close at 10.08" followed by "closed". Diagnose in the browser (F12, Network tab, `extract` response): look at `warnings` and `missing_fields` for that deal.
- If `conclusion_downgraded` and `later_acceptance_evidence` appear: in `_conclusion_check` in `pipeline.py`, the acceptance is compared against the highest cited line of every term. Restrict `term_fields` to `["ytm", "price"]` and compare against the **earliest** cited yield/price line (`min`, not `max`). Quantity and settlement are often restated after the close.
- If there is no `conclusion_downgraded`: the model returned NEGOTIATING itself. Add to the status rules in the extraction prompt that a close accepted by the other side is CONCLUDED even if quantity or settlement is restated afterwards, plus a synthetic few-shot example.
Add a unit test with synthetic line numbers (yield cited on lines 2 and 9, acceptance on line 4).

**B. CANCELLED for explicit refusals.**
A refusal ("pass", "no sir", "no requirement", "nahi", "rehne do", see `lexicon.json`) in direct reply to a specific offer or inquiry (by reply link or next message in the same chat) sets that deal to CANCELLED with the refusal line as `status_evidence`. A refusal that cannot be tied to an offer stays UNCERTAIN. Silence stays INITIATED.

**C. Prompt 11, database schema (`db.py`), in two passes.**
- Pass 1: `schema_version` table and a development reset helper; `messages` gets `chat_key`, `chat_name`, `chat_type`, `direction`, `is_own`, `is_media`, `media_kind`, `replying_to_text`, `noise_label`, hashed sender id; insert with `INSERT OR IGNORE` on the stable `message_id`.
- Pass 2: `transactions` allows INITIATED / NEGOTIATING / CONCLUDED / CANCELLED / UNCERTAIN, nullable or OTHER settlement, plus `context_id`, `chat_key`, `owner_side`, `buyer_text`, `seller_text`, `quantity_text`, `amount_inr`, `units_assumed`, `ytm_is_two_way`, `settlement_text`, `maturity_text`, `maturity_ym`, `partial_isin_text`, `rfq_id`, `package_note`, `evidence_json`, `missing_fields_json`, `confidence`, `needs_review`, `prompt_version`, `model_name`; new `transaction_messages` join table (one broadcast can belong to several deals). Update `persist_upload` and `fetch_database_data`, then remove the "cannot store the new extraction format" guard in `app.py`.
- SQLite cannot alter CHECK constraints, so recreate the tables (delete any old `bond_deals.db` first). Leave the bond identity / dedupe logic alone; it is being reworked (item I).

**D. Prompt 12, `app.py`.** Remove the top-8 cap and ISIN-first sorting (process all contexts chronologically). Wrap each context in its own try/except and collect errors with the raw model output. `/extract` returns a full report (messages, media, system, skipped, contexts, skipped_low_relevance, processed, failed, deals by status, `needs_review` count, token usage). Add a TTL for in-memory upload sessions. Pass chat info and message ids through so `/store` links messages through the join table.

**E. Prompt 13, `privacy.py` (done).** Mask phones, `@numeric` mentions, URLs, emails, bank or account fragments and sender numbers before the model. Person names are optional (`MASK_PERSON_NAMES`, default false). Do **not** mask ISINs, yields, quantities, dates or RFQ ids. Keep an in-memory placeholder map and restore model results locally after validation.

**F. Prompt 14, tests and `scripts/smoke_run.py` (done).** Offline tests with a mocked LLM and synthetic fixtures (never real data). `smoke_run.py` follows the `/extract` pipeline and prints aggregate counts only: messages, chats, skipped, media, contexts, reply-linked, unresolved replies, `skipped_low_relevance`, deals by status, `needs_review`, warnings by type, and token/context-window totals.

**G. Prompt 15, UI (`static/`) (done).** Extraction and database views show chat name, status badges for all statuses, `needs_review` and confidence, missing fields, evidence line numbers, quantity as said and normalized amount, settlement text and date, plus extraction report counts.

**H. Prompt 16, before/after.** Re-run the baseline script on the same real file with the same model and produce a side-by-side table (contexts, dropped or skipped messages, contexts sent to the model, deals by status, `needs_review`). List messages that were previously dropped and are now kept.

**I. Primary / secondary bond identity and LLM context assignment (mentor reviews).**
- Context assignment: for each new message give the LLM the active contexts (short cards: bond details so far plus last few messages) and ask: same context / new context / not trade-related, with a confidence. Time gap becomes a hint only. Low confidence goes to review. A multi-bond message updates several contexts. Keep the assigner behind the `assign_contexts` function in `pipeline.py`.
- Identity ladder: validated ISIN, then partial ISIN, then structured name, then issuer plus maturity plus coupon plus rating (via `issuer_aliases.json`). Output levels: ISIN-confirmed, uniquely resolved by secondary details, partial, ambiguous (several candidates; never guess), unknown. Carry identity forward within a context so later "Bond Unspecified" messages inherit it. The same bond can appear under different issuer names in different chats, with the same ISIN.
- Cross-chat linking of one deal (broadcast, negotiation, internal dealing, ops) is a later step above per-chat contexts.

---

## 10. Known issues

1. No deal reaches CONCLUDED yet (item A).
2. Explicit refusals stay UNCERTAIN (item B).
3. `/store` fails with "cannot store the new extraction format" until item C.
4. One conversation can be split into several records, including several "Bond Unspecified / UNCERTAIN" rows (item I).
5. The same broadcast posted to several chats creates separate records (needs cross-chat linking).
6. When the owner acts as an intermediary (a client bids, the owner asks a dealing group for a seller), the side can flip between runs. This needs a decision from the mentor on how to represent the broker's two legs.
7. A package offer ("package offer at 8.4%") copies the same yield onto every bond; it belongs in `package_note`.
8. Isolated one-letter acknowledgements and bare ack messages have no bond context on their own.
9. Platform trades (price lives on the exchange RFQ platform, not in the chat) show no yield in the chat. Decide what status these get.
10. Open decisions for the mentor: broker two-leg representation, settlement stored as date plus label, post-trade steps ("reported", "accepted", "validated") in or out of scope, how long an unanswered offer stays INITIATED, bond master data availability for face value and ISIN lookup, what masking is required before text goes to a hosted LLM.

---

## 11. Working conventions

- Follow `AGENTS.md` (also `.github/copilot-instructions.md`). In short: no hardcoded sample-chat content in code, domain data in JSON config; deterministic code is pure functions with tests; never invent values; never silently drop data; keep existing function names and signatures; add tests for every behaviour change; do not touch bond identity / dedupe in `db.py` outside item I.
- Work on a feature branch. Commit specific files, never `git add .`.
- When using an AI assistant on this repo, tell it to follow `AGENTS.md` and to report counts only from `data/` and `baseline/`.

---

## 12. Troubleshooting


| Symptom | Cause / fix |
|---|---|
| `ollama` command not found | Install Ollama from ollama.com/download, then open a **new** terminal |
| Model not found / auth error | `ollama signin`; confirm with `ollama run gemma4:31b-cloud "say hello"` |
| `llm_mode: mock` in a baseline report | Ollama unreachable or the `ollama` / `langchain-ollama` packages are missing. Fix, then re-run |
| `ModuleNotFoundError` | The venv is not active. Re-run the `Activate.ps1` line |
| `uvloop does not support Windows` while installing `requirements.txt` | Do not install that file; use the package list in section 2 |
| Only 8 contexts processed | Set `$env:MAX_CONTEXTS="none"` before starting the server |
| Sides (buyer / seller) look wrong | Set `OWN_SENDER_NAMES` to the owner's display name exactly as in the chat |
| Port already in use | Close the old `uvicorn` window or add `--port 8001` |
| `/store` error about the new extraction format | Expected until item C (prompt 11) is done |