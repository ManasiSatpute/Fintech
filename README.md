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
- Phase 1 SQLite hardening: versioned in-place migrations, verified per-connection foreign keys, separate identification status, and a duplicate-checked partial unique ISIN index.
- Phase 2 bond reference index: a configurable external CSV is loaded into immutable in-memory ISIN and issuer indexes; the extraction pipeline does not use it yet.
- Phase 3 source-based ISIN extraction: original message evidence, normalized candidates, structure and checksum validation; no pipeline integration or bond matching.
- Phase 4 standalone bond matching: source-supported exact ISIN first, then exact normalized issuer plus evidenced corroborating details; ambiguous or conflicting results never select arbitrarily.
- Last test run: **153 tests passing** (plus 22 subtests).

**Not done yet (in order)**

1. Make a closed trade reach `CONCLUDED` (see Known issues, item 1).
2. `CANCELLED` handling for explicit refusals (see Known issues, item 2).
3. Prompt 12, API behaviour (remove the 8-context cap, per-context error handling).
4. Prompt 16, before/after comparison against the saved baseline.
5. LLM-based context assignment and carrying resolved identity across messages (the mentor's reviews, section 8).

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
| `db.py` | SQLite storage (`bond_deals.db`, git-ignored) with versioned schema migrations |
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
8. **Store** extracted trade records and message links in SQLite.

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
- **Out of scope:** role IDs and dashboard. Identification decisions are audited separately; manual verification that a trade really happened is still expected before a record is trusted.

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

**C. Phase 1 SQLite database hardening (`db.py`, implemented).**
- Schema version 5 migrates earlier supported databases in place; migrations are transactional and do not replace the database or delete rows.
- Foreign-key enforcement is enabled and verified on every project database connection. `transactions.identification_status` is independent of the existing trade `status`; `transactions.bond_id` may be null until a unique bond match is persisted.
- A unique partial index covers non-null `bonds.isin`. If duplicate ISIN groups exist, migration stops and rolls back without changing rows. Back up the database, review duplicates locally, and resolve each manually: correct erroneous ISINs only from an authoritative source; for confirmed duplicate records representing the same bond, reassign dependent transactions to a reviewed canonical bond ID and set the duplicate row's ISIN to null while retaining the row. Verify foreign keys and the absence of duplicate non-null ISINs, then rerun application initialization. Never delete the duplicate bond rows as an automatic migration step.
- Version 5 rebuilds `transactions` transactionally to remove `NOT NULL` from `bond_id`, retaining its columns, rows, indexes, and child references; it validates foreign keys before completing.
- `identification_audit` stores distinct identification decisions, their method, reason, raw/normalized issuer values, ISIN, candidates and source evidence. Repeating the same decision for a transaction is idempotent.

**D. Phase 2 in-memory bond reference index (`bond_reference.py`, implemented).**
- Configure the external file with `BOND_REFERENCE_CSV`; the default is `ISIN_DETAILS.csv` beside `config.py`. The CSV stays outside SQLite and remains excluded from Git.
- `get_bond_reference_index()` lazily loads and reuses an immutable snapshot. `reload_bond_reference_index()` explicitly builds and atomically replaces it. `BondReferenceIndex.load(path)` builds an independent snapshot for tools and tests.
- Exact ISIN lookup normalizes case and whitespace. Issuer candidate retrieval normalizes Unicode, case and whitespace. Original values remain available separately from normalized lookup keys.
- Duplicate identifiers are counted. Identical rows remain lookupable; conflicting rows return no single-record ISIN match and remain available as candidates for review. Missing headers, unreadable files, and files with no valid records raise `BondReferenceError`.
- This phase does not make identification decisions or connect to the extraction pipeline.

**Phase 3 source-based ISIN extraction and validation (`isin_utils.py`, implemented).**
- Extract ISIN-shaped candidates from one original WhatsApp message, retain the raw candidate and source-message span, normalize case and whitespace/hyphen separators, then validate ISO 6166 structure and checksum.
- Results distinguish no identifier, one valid identifier, invalid format, checksum failure, multiple candidates, and an LLM candidate unsupported by its supplied source evidence. Multiple candidates are never resolved by selecting one.
- LLM candidates are checked against the original message and, if supplied, only explicitly linked messages with evidence referring back to the original message. Callers must not pass an unfiltered chat window.
- This module does not perform bond matching, persist records, or integrate into extraction.

**Phase 4 primary and secondary bond matching (`bond_matching.py`, implemented).**
- A single source-supported, checksum-valid Phase 3 ISIN is looked up exactly in the Phase 2 index. Results distinguish an exact primary match from a valid ISIN absent from the reference.
- Secondary matching requires an exact normalized issuer key and at least one source-supported maturity, coupon/rate, or exact security-description detail. All comparable supplied fields must agree; unknown reference fields do not corroborate a candidate. Fuzzy issuer similarity and issuer-only matches are not accepted.
- Multiple candidates are returned as ambiguous. If primary and unique secondary evidence point to different securities, the result is a conflict and selects neither. Invalid, checksum-failing, unsupported, or unrelated-message identifier evidence is never used for exact lookup.
- The reference data has no credit-rating column. Alias-to-legal-name resolution is not applied because the current alias file does not establish exact mappings to reference issuer records.
- This remains an independent matching utility: no extraction-pipeline integration, persistence, audit writes, or schema changes.

**Phase 5 SQLite persistence and identification audit (`db.py`, implemented).**
- `persist_identification_result()` accepts a Phase 4 result and persists the status, method, explanation, candidate references and source evidence in one transaction. Only unique primary ISIN and secondary issuer matches can create or link an authoritative reference bond.
- No-match, ambiguous, conflicting, invalid or unsupported outcomes do not create a bond link. Existing non-placeholder bond links are not overwritten by a conflicting decision.
- Identical decisions for the same transaction share a deterministic audit key and do not create duplicate audit rows. A failed audit write rolls back bond and transaction changes.
- `persist_upload()` supports deferred bond creation so Phase 6 can leave transactions unresolved rather than creating an `UNKNOWN` placeholder. Phase 5 introduced the persistence API; Phase 6 below wires it into the application flow.

**Phase 6 application integration (`pipeline.py`, `app.py`, `static/`, implemented).**
- Each validated extracted deal is matched using only original source messages cited by that deal's identity/issuer/term evidence. ISIN candidates are extracted and validated from those messages; unsupported identifiers and multiple candidates are not selected.
- The configured reference index is loaded through its process cache. Missing/invalid reference data is surfaced as `reference_unavailable` while extraction remains available for review.
- `/store` persists messages, transactions, links, reference bonds and identification audits in one SQLite transaction. A failed identification write rolls back the upload writes as well. Only unique supported matches create/link a reference bond; other outcomes remain unlinked and are audited. Repeated stores use stable transaction IDs and idempotent audit decisions; changed results on an already-stored upload are skipped and reported in the `identification_skipped` response count rather than applied to an existing transaction.
- The extraction view displays match status/reason and the transaction view displays persisted identification status. Message rows include all linked transaction IDs from the join table.
- Clean price is accepted only when its cited source line explicitly labels the value as a price; yield, coupon and amount are never used to calculate or validate it.

**Recovering uploads partially identified by an earlier Phase 6 version.**
- Earlier versions committed upload transactions before attempting each identification write. A failed later identification write could therefore leave a stored upload with null identification statuses or missing matching audit rows. Current retries detect any existing transaction with the source hash and intentionally skip the whole upload; they do not repair that historical state.
- To identify affected rows, run this read-only query against a database backup or an approved read-only connection:
  ```sql
  SELECT source_hash,
         COUNT(*) AS transaction_count,
         SUM(CASE
               WHEN t.identification_status IS NULL
                 OR NOT EXISTS (
                   SELECT 1
                   FROM identification_audit a
                   WHERE a.transaction_id = t.transaction_id
                     AND a.identification_status = t.identification_status
                 )
               THEN 1 ELSE 0
             END) AS transactions_missing_matching_audit
  FROM transactions t
  WHERE source_hash IS NOT NULL
  GROUP BY source_hash
  HAVING transactions_missing_matching_audit > 0;
  ```
- This query detects existing transactions with no identification status or no audit row matching that status. It cannot detect transactions that were never inserted because the database has no upload manifest recording the expected transaction IDs.
- For recovery, first make and verify a database backup. Re-process the original upload with the current extraction code in an isolated database and compare its deterministic transaction IDs and source evidence with the affected stored rows. Do not use `update_existing=True` as an automatic repair. Apply a reviewed missing result only to its verified transaction, using the normal identification persistence API after confirming that its bond link is null and no prior audit exists. If a bond link or any prior audit exists, stop for manual review. Never delete audit records or replace an established link as part of recovery.

**E. Prompt 12, `app.py`.** Remove the top-8 cap and ISIN-first sorting (process all contexts chronologically). Wrap each context in its own try/except and collect errors with the raw model output. `/extract` returns a full report (messages, media, system, skipped, contexts, skipped_low_relevance, processed, failed, deals by status, `needs_review` count, token usage). Add a TTL for in-memory upload sessions. Pass chat info and message ids through so `/store` links messages through the join table.

**F. Prompt 13, `privacy.py` (done).** Mask phones, `@numeric` mentions, URLs, emails, bank or account fragments and sender numbers before the model. Person names are optional (`MASK_PERSON_NAMES`, default false). Do **not** mask ISINs, yields, quantities, dates or RFQ ids. Keep an in-memory placeholder map and restore model results locally after validation.

**G. Prompt 14, tests and `scripts/smoke_run.py` (done).** Offline tests with a mocked LLM and synthetic fixtures (never real data). `smoke_run.py` follows the `/extract` pipeline and prints aggregate counts only: messages, chats, skipped, media, contexts, reply-linked, unresolved replies, `skipped_low_relevance`, deals by status, `needs_review`, warnings by type, and token/context-window totals.

**H. Prompt 15, UI (`static/`) (done).** Extraction and database views show chat name, status badges for all statuses, `needs_review` and confidence, missing fields, evidence line numbers, quantity as said and normalized amount, settlement text and date, plus extraction report counts.

**I. Prompt 16, before/after.** Re-run the baseline script on the same real file with the same model and produce a side-by-side table (contexts, dropped or skipped messages, contexts sent to the model, deals by status, `needs_review`). List messages that were previously dropped and are now kept.

**J. LLM context assignment and identity carry-forward (mentor reviews).**
- Context assignment: for each new message give the LLM the active contexts (short cards: bond details so far plus last few messages) and ask: same context / new context / not trade-related, with a confidence. Time gap becomes a hint only. Low confidence goes to review. A multi-bond message updates several contexts. Keep the assigner behind the `assign_contexts` function in `pipeline.py`.
- Identity carry-forward: build on Phase 4's source-grounded primary and secondary match results; later carry confirmed identity within a context so later "Bond Unspecified" messages inherit it. Partial-identifier matching and safe alias-to-legal-name mappings need separate definition before integration. The same bond can appear under different issuer names in different chats, with the same ISIN.
- Cross-chat linking of one deal (broadcast, negotiation, internal dealing, ops) is a later step above per-chat contexts.

---

## 10. Known issues

1. No deal reaches CONCLUDED yet (item A).
2. Explicit refusals stay UNCERTAIN (item B).
3. One conversation can be split into several records, including several "Bond Unspecified / UNCERTAIN" rows (item J).
4. The same broadcast posted to several chats creates separate records (needs cross-chat linking).
5. When the owner acts as an intermediary (a client bids, the owner asks a dealing group for a seller), the side can flip between runs. This needs a decision from the mentor on how to represent the broker's two legs.
6. A package offer ("package offer at 8.4%") copies the same yield onto every bond; it belongs in `package_note`.
7. Isolated one-letter acknowledgements and bare ack messages have no bond context on their own.
8. Platform trades (price lives on the exchange RFQ platform, not in the chat) show no yield in the chat. Decide what status these get.
9. Open decisions for the mentor: broker two-leg representation, settlement stored as date plus label, post-trade steps ("reported", "accepted", "validated") in or out of scope, how long an unanswered offer stays INITIATED, bond master data availability for face value and ISIN lookup, what masking is required before text goes to a hosted LLM.

---

## 11. Working conventions

- Follow `AGENTS.md` (also `.github/copilot-instructions.md`). In short: no hardcoded sample-chat content in code, domain data in JSON config; deterministic code is pure functions with tests; never invent values; never silently drop data; keep existing function names and signatures; add tests for every behaviour change; do not touch bond identity / dedupe in `db.py` outside item J.
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