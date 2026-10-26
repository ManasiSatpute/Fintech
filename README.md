# WhatsApp Bond Trade Extraction

A local FastAPI application that turns WhatsApp exports into reviewable bond-trade
records. Deterministic parsing, validation, source evidence, and reference matching
surround an LLM that interprets the conversation. The application is assistive: it
does not execute trades, and extracted records require human review.

## Current behavior

- `POST /upload` parses a WhatsApp text or JSON export and retains it in an
  in-memory upload session.
- `POST /extract/{upload_id}` groups messages, calls the configured Ollama model,
  validates cited evidence, and attempts conservative bond identification.
- Bond identification uses only the source messages cited for the extracted
  identity fields. A primary ISIN must be source-supported, structurally valid,
  and checksum-valid. A contradictory source-supported issuer blocks linking.
  Secondary matching requires an exact normalized issuer and at least one
  corroborating, source-supported reference detail. Ambiguous, invalid, conflicting,
  unsupported, and unmatched cases do not receive an automatic reference-bond link.
- Clean price is retained only when a cited line explicitly labels the value as a
  price. Yield, coupon, and trade amount are not used to calculate or infer price.
- `POST /store/{upload_id}` persists messages, transactions, links, bonds, and
  identification audits in one SQLite transaction. A persistence failure rolls
  back the upload. Repeating a stored upload skips its extraction results and
  reports that they were not applied.
- `GET /data` returns persisted transaction, bond, and message records to the UI.
  `GET /samples` lists available sample files, and
  `POST /load-sample/{filename}` loads one by name.

Upload sessions are in memory and expire after the configured TTL; after a server
restart, an upload must be submitted again. The API has no authentication or
authorization. Run it only on a trusted local machine; do not expose it to a
network or treat it as a production service.

## Requirements and installation

The repository's `requirements.txt` is an environment snapshot, not a minimal
application dependency manifest. It includes unrelated packages and a platform-
specific `uvloop` entry; do not install it wholesale for this application.

Python was tested in the project's current Python 3.13 environment. Other Python
versions have not been certified here.

On Windows PowerShell:

```powershell
python -m venv C:\venvs\fintech
C:\venvs\fintech\Scripts\Activate.ps1
python -m pip install fastapi uvicorn python-multipart pydantic ollama langchain-ollama pytest httpx
```

Ollama must also be installed and available to the application. The default model
is `gemma4:31b-cloud`; select an installed local model with `MODEL_NAME` if the
chat data must not be sent to a hosted model. With the cloud model, chat text is
processed by Ollama's servers. PII masking is enabled by default but is not a
guarantee of de-identification. Use only synthetic or explicitly approved data.

## Reference CSV

Set `BOND_REFERENCE_CSV` to the reference file path. Its default is
`ISIN_DETAILS.csv` beside `config.py`. The file is read into an immutable in-memory
index; it is not copied into SQLite. Required headers are:

```text
ISIN,NAME_OF_ISSUER,SECURITY_DESCRIPTION,MATURITY_DATE,INTEREST_RATE,CURRENCY
```

If the reference file is unavailable or invalid, extraction can continue, but
identification is marked unavailable and records must be reviewed. The index is a
process snapshot; restart the application or explicitly reload it after changing
the CSV.

## Configuration

These defaults are defined in `config.py`:

| Environment variable | Default | Purpose |
|---|---|---|
| `MODEL_NAME` | `gemma4:31b-cloud` | Ollama model |
| `MAX_CONTEXTS` | `none` | Maximum contexts per extraction; `none` means no cap |
| `OWN_SENDER_NAMES` | empty | Comma-separated names used to identify the owner's side |
| `GAP_MINUTES` | `15` | Time gap used when grouping messages |
| `DAY_FIRST_DEFAULT` | `true` | Date-order fallback for ambiguous dates |
| `LLM_RETRIES` | `1` | Retry count for failed or invalid model output |
| `MARKET_HOLIDAYS` | empty | Comma-separated ISO dates for settlement calculations |
| `PII_MASKING_ENABLED` | `true` | Mask configured contact/account patterns before inference |
| `MASK_PERSON_NAMES` | `false` | Also mask detected person names |
| `RELEVANCE_THRESHOLD` | `3` | Minimum relevance score for a context |
| `REVIEW_CONFIDENCE_THRESHOLD` | `0.7` | Confidence below which a deal needs review |
| `ALLOWED_ORIGINS` | localhost origins | CORS allowlist |
| `MODEL_CONTEXT_LENGTH_FALLBACK` | `262144` | Context length if Ollama metadata is unavailable |
| `BOND_REFERENCE_CSV` | `ISIN_DETAILS.csv` beside `config.py` | Reference CSV path |
| `UPLOAD_SESSION_TTL_SECONDS` | `3600` | In-memory upload-session lifetime |

The SQLite database path is currently the relative path `bond_deals.db` in the
process working directory. It is not configured by an environment variable.
Importing `app.py` initializes the database and applies supported migrations.
**Do not start the app in a directory containing a valuable database unless that
database is intentionally the target.** For a demo, use a disposable project copy
or other isolated working directory and a synthetic reference CSV.

## Start the application

Run from the project directory (or an isolated copy of it), so the relative
`static/` and sample-file paths resolve:

```powershell
$env:MODEL_NAME = "gemma4:31b-cloud"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The default host binds locally. Do not use a
public-facing host or disable masking for sensitive data without an approved
deployment and privacy review.

## Database initialization and migrations

The app initializes SQLite on import. New databases receive the current schema;
recognized older schema versions are upgraded by versioned migrations. Foreign
keys are enabled and verified on project connections. The schema has nullable
transaction bond links, a unique partial index for non-null bond ISINs, and
identification audit records.

Migration of a real database is a persistent change. Back up the file first and
do not test migrations against the production or personal database. Migration
failure due to duplicate non-null bond ISINs requires manual review; automated
migration does not delete duplicate bond rows.

Older application versions could leave an upload only partly identified. Current
duplicate-upload handling intentionally skips an upload if its source hash is
already stored; it does not repair historical partial writes. The database has
no manifest of expected upload rows, so missing transactions cannot all be
detected. On a verified backup or an approved read-only connection, this query
counts existing transactions whose identification status is missing or lacks a
matching audit row:

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

This cannot detect expected transactions that were never inserted. For recovery,
verify a backup, reprocess the original upload in an isolated database, compare
the deterministic transaction IDs and audit evidence, and have a reviewer approve
any manual repair. Never overwrite established links or delete audit rows as an
automatic recovery step.

## Tests and database isolation

Run the suite from the repository root:

```powershell
python -m pytest -q
```

Pytest loads `tests/conftest.py`, redirects SQLite's bound default database paths
before app import, and rejects file-backed SQLite connections outside OS temporary
storage. In-memory connections are allowed. The guard applies to the pytest run;
it does not protect manually launched scripts or the application outside pytest.
Use only the guarded pytest command for automated validation.

## Data handling

- Use synthetic fixtures for development and testing. Never commit private chats,
  generated reports containing excerpts, credentials, or local database files.
- PII masking is a best-effort pattern-based safeguard; it may miss sensitive
  content. The default cloud model sends chat text to hosted inference.
- Inference and reference-loading failures return generic client messages with
  safe error categories and correlation IDs where available. Server logs are
  intended for controlled local diagnostics and should still be access-restricted.
- This project has not had a browser-level demo verified as part of this README
  update.
