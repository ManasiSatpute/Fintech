# Milestone 1: Bond Subject Identification and Message Matching

**Project:** WhatsApp Bond Trade Extraction System  
**Status:** Completed & Verified (248/248 Tests Passing)  
**Date:** 2026-11-13  
**Author:** Senior Python Backend Engineer & Fintech Software Architect  

---

## 1. Executive Summary

This milestone establishes **Milestone 1: Complete Bond Subject Identification and Message Matching** in strict accordance with the project mentor's architecture recommendations.

The primary objective of Milestone 1 is to elevate conversational deal identification from simple regex-based keyword matching to an evidence-grounded, multi-attribute financial subject identifier. WhatsApp bond trade negotiations frequently omit the canonical 12-character ISIN during early negotiation stages and instead identify securities by **issuer, industry sector, credit rating, instrument type, maturity calendar year, tenor, and spread over benchmark**.

Milestone 1 introduces deterministic extraction, normalization, fingerprinting, conflict detection, and scoring for all of these attributes, enabling the system to cleanly partition simultaneous negotiations within the same WhatsApp group without cross-contamination.

---

## 2. Requirement-to-Code Gap Analysis

Before implementing Milestone 1, a comprehensive inspection of the entire codebase was conducted across `app.py`, `pipeline.py`, `lifecycle.py`, `db.py`, `bond_reference.py`, `bond_matching.py`, `deal_stack_models.py`, `deal_stack_matcher.py`, and `deal_stack_manager.py`.

Below is the structured gap analysis mapped to each of the mentor's 8 core requirements:

| # | Mentor Requirement | Current Status | Implemented Components | Identified Gaps Prior to Milestone 1 | Milestone Addressing Gap |
|---|---|---|---|---|---|
| **1** | **Identify unique conversations based on bond, issuer, industry, credit rating, maturity and tenor** | **COMPLETED (Milestone 1)** | • Exact ISIN parsing (`isin_utils.py`)<br>• Instrument detection (SDL, GS, TBILL, NCD)<br>• Canonical issuer regexes (`_ISSUER_REGEXES`)<br>• Maturity calendar year (`_MATURITY_YEAR_RE`)<br>• **Credit rating extraction & matching (`AAA` to `SOV`)**<br>• **Industry/sector mapping & keyword cues (`PSU`, `NBFC`, `HFC`, etc.)**<br>• **Tenor extraction (`5Y`, `10Y`, `3Y`, `2.5Y`)**<br>• **Deterministic subject fingerprinting** | Prior implementation only checked 4-digit maturity year and basic issuer/instrument, lacking rating, industry, tenor, and spread. | **Milestone 1 (Now fully closed)** |
| **2** | **Support incomplete information spread across multiple WhatsApp messages** | **Partially Implemented** | • Non-destructive field accumulation in `deal_stack_manager.py: _accumulate_terms`<br>• Field-evidence citations (`terms.field_evidence`)<br>• Quote progression tracking | Needs validation against mandatory schema fields, handling partial quantity units, and missing field indicators. | **Milestone 2** |
| **3** | **Accumulate supporting information such as price, yield, quantity, spread, transaction date and settlement details** | **Partially Implemented** | • Price, yield, and quantity accumulation with full history (`yield_history`, `price_history`, `quantity_history`)<br>• **Spread extraction & history added in Milestone 1** | Settlement cycle (`T+0`, `T+1`, `T+2`) and transaction date resolution need complete persistence models and schema linking. | **Milestone 2** |
| **4** | **Infer the transaction state from all available messages** | **Partially Implemented** | • Lexical lifecycle classification (`classify_message_lifecycle_event` in `lifecycle.py`)<br>• `conclude_stack()` and `reject_stack()` triggers | Full state transition engine in `lifecycle.py` must be hooked into the manager to prevent invalid transitions and standardize audit records. | **Milestone 3** |
| **5** | **Support multiple simultaneous negotiations within the same WhatsApp group** | **COMPLETED** | • Per-chat stack partitioning in `deal_stack_manager.py`<br>• 8-tier evidence scoring in `deal_stack_matcher.py`<br>• Hard conflict detection on ISIN, instrument, issuer, maturity, coupon, rating, industry, and tenor | Verified with simultaneous multi-stack unit tests. | **Milestone 1 & 2** |
| **6** | **Seal concluded or rejected negotiations and create separate stacks for genuinely independent negotiations** | **COMPLETED** | • Stack states: `ACTIVE` vs `SEALED`<br>• Seal reasons: `EXPLICIT_CONCLUSION`, `EXPLICIT_REJECTION`, `SUPERSEDED_NEGOTIATION`<br>• Terminal status immutability (late replies log audit events without regressing business status)<br>• External counterparty axe routing to `NEW_STACK` | Fully working and covered by tests. | **Milestone 1 & 3** |
| **7** | **Send individual sealed stacks to the Extraction Agent** | **Pending Pipeline Integration** | • `DealStackManager.snapshot_batch_end()` takes isolated stack snapshots | `pipeline.py` currently uses fixed 15-minute time windowing (`group_messages_into_windows`). Must replace time windows with stack message snippets. | **Milestone 5** |
| **8** | **Maintain exactly three business statuses: PENDING, CONCLUDED and REJECTED** | **Enforced in Stack Models; Pending Pipeline & DB** | • `deal_stack_models.py: StackStatus` strictly defines `PENDING`, `CONCLUDED`, `REJECTED`<br>• Manager enforces these 3 statuses | SQLite `transactions` table check constraint currently permits `INITIATED`, `NEGOTIATING`, `CANCELLED`, `UNCERTAIN`, `DISCARDED`. Needs Migration 6. | **Milestone 3 & 4** |

---

## 3. Milestone 1 Implementation Architecture

### 3.1. Credit Rating Extraction & Matching
Bond tranches from the same issuer frequently differ in credit seniority and rating. For example, PFC issues AAA senior secured paper as well as subordinated or lower-tier debentures. 

- **Regex Engine:** Implemented `_CREDIT_RATING_RE` supporting rating agency prefixes (`CRISIL`, `ICRA`, `CARE`, `IND RATINGS`, `INDIA RATINGS`, `BRICKWORK`, `ACUITE`) and standard Indian rating tiers:
  `AAA`, `AA+`, `AA-`, `AA`, `A+`, `A-`, `A`, `BBB+`, `BBB-`, `BBB`, `A1+`, `A1`, `SOV`, `SOVEREIGN`.
- **Lookahead Boundary Protection:** Used negative lookahead `(?![A-Za-z0-9+-])` instead of standard word boundaries `\b` to eliminate regex truncation pitfalls on non-word characters (`+` and `-`), correctly capturing `AA-` and `AA+`.
- **Normalization:** Normalized `"SOVEREIGN"` to canonical `"SOV"`.
- **Conflict Detection:** If an incoming message specifies a different credit rating than an active stack (e.g., `AAA` vs `AA-`), `_check_hard_conflicts` flags a **Hard Bond Identity Conflict**, forcing the creation of a separate independent deal stack.
- **Scoring:** Awarded +0.10 in Tier 4 (`MATURITY_COUPON_RATING`) when ratings match.

### 3.2. Industry / Sector Identification & Mapping
Conversational negotiations often refer to sector themes (e.g., *"Looking for AAA PSU paper"*, *"Selling NBFC bond"*).

- **Sector Classification Mapping (`_ISSUER_INDUSTRY_MAP`):**
  - **NBFC / Financial Services:** Bajaj Finance, Tata Capital, Muthoot, Oxyzo, Kosamattam, Indostar, Arka, Hinduja Leyland, Shriram, Poonawalla, Cholamandalam.
  - **Housing Finance (HFC):** LIC Housing, HDFC, PNB Housing, Aadhar, Can Fin.
  - **PSU / Quasi-Government:** REC, PFC, IRFC, NABARD, NHAI, HUDCO, GAIL, IOCL, ONGC, Power Grid.
  - **Power & Energy:** Meghalaya Energy, NTPC, Adani Green, Tata Power, NHPC, KIIFB.
  - **Banking:** SBI, ICICI, Axis Bank, Bank of Baroda, PNB, Kotak.
  - **State Govt (SDL):** State Development Loans.
  - **Central Govt (Sovereign):** G-Sec, Treasury Bills.
  - **Infrastructure / Conglomerates:** L&T, Reliance, Tata Sons, Bharti Airtel.
- **Keyword Detection:** `_EXPLICIT_INDUSTRY_PATTERNS` detects explicit terms: `NBFC`, `HFC`, `PSU`, `Power sector`, `DISCOM`, `Banking`, `Infra`.
- **Conflict Detection:** If an active stack belongs to `PSU` (e.g. PFC) and a new message offers `NBFC` paper (e.g. Muthoot), hard conflict triggers immediately.
- **Scoring:** Awarded +0.10 in Tier 3 (`INSTRUMENT_ISSUER`) when industries match.

### 3.3. Tenor Extraction & Matching
Traders routinely negotiate by tenor (e.g., *"Selling 5y SDL"*, *"Need 10 year paper"*, *"3 yr tenor @ 7.30"*).

- **Pattern Matching (`_TENOR_RE`):** Captures standard tenors: `5y`, `10y`, `3 yr`, `2.5y`, `7-year`, `10 year paper`.
- **Normalization:** Formats to canonical uppercase representation (`5Y`, `10Y`, `3Y`, `2.5Y`).
- **Conflict Detection:** If an incoming message requests `3Y` paper while the active stack is `10Y` (difference $\ge 0.5$ years), a hard conflict triggers.
- **Scoring:** Awarded +0.15 in Tier 4 (`MATURITY_COUPON_RATING`) for matching tenors.

### 3.4. Spread Extraction
Corporate bonds and state loans are frequently quoted as a spread over the sovereign benchmark (e.g., *"+25 bps over G-sec"*, *"spread 35bps"*).

- **Pattern Matching (`_SPREAD_RE`):** Extracts basis point spreads (e.g. `25.0` bps) and raw text (`"+25 bps"`).
- **Scoring:** Awarded +0.10 in Tier 7 (`TERMS_CONTINUITY`) when spreads match within 5 bps.

### 3.5. Deterministic Subject Fingerprint Generation
The subject fingerprint uniquely identifies the underlying bond security across messages:

$$\text{Fingerprint} = \begin{cases} 
\text{ISIN:}\langle \text{ISIN} \rangle & \text{if ISIN present} \\
\text{BOND:}\langle \text{Issuer} \rangle\text{:}\langle \text{Instrument} \rangle\text{:}\langle \text{Industry} \rangle\text{:}\langle \text{Maturity} \rangle\text{:}\langle \text{Tenor} \rangle\text{:}\langle \text{Rating} \rangle\text{:}\langle \text{Coupon} \rangle & \text{otherwise}
\end{cases}$$

Example:
`BOND:PFC:NCD:PSU:2028:5Y:AAA:7.25%`

---

## 4. Modified and Created Files

### Created Files
1. [`tests/test_milestone1_subject_matching.py`](file:///Users/manasi/Desktop/fintech/Fintech/tests/test_milestone1_subject_matching.py):
   12 unit tests verifying rating extraction, rating conflict detection, industry mapping, industry conflict detection, tenor matching, spread matching, fingerprint uniqueness, and simultaneous multi-deal isolation.
2. [`milestone/milestone1.md`](file:///Users/manasi/Desktop/fintech/Fintech/milestone/milestone1.md):
   Comprehensive Milestone 1 technical documentation and gap analysis.
3. [`milestone/README.md`](file:///Users/manasi/Desktop/fintech/Fintech/milestone/README.md):
   Directory index and milestone tracking overview.

### Modified Files
1. [`deal_stack_models.py`](file:///Users/manasi/Desktop/fintech/Fintech/deal_stack_models.py):
   - Added `industry`, `tenor`, `spread`, `spread_raw` to `MessageBondFeatures`.
   - Added `industry`, `tenor`, `spread`, `spread_raw`, `spread_history` to `DealTerms`.
   - Added `industry`, `tenor`, `credit_rating` to `DealStack`.
2. [`deal_stack_matcher.py`](file:///Users/manasi/Desktop/fintech/Fintech/deal_stack_matcher.py):
   - Added `_ISSUER_INDUSTRY_MAP` and `_EXPLICIT_INDUSTRY_PATTERNS`.
   - Added `_CREDIT_RATING_RE`, `_TENOR_RE`, `_SPREAD_RE`.
   - Updated `extract_bond_features()` to extract rating, industry, tenor, spread.
   - Updated `compute_subject_fingerprint()` to include all subject dimensions.
   - Enhanced `_check_hard_conflicts()` with credit rating, tenor, and industry checks.
   - Enhanced `score_stack_candidate()` to score rating, industry, tenor, and spread.
   - Updated `_BUY_ACTION_RE` and `is_opening_quote` to recognize buy inquiries with issuer/tenor/rating.
3. [`deal_stack_manager.py`](file:///Users/manasi/Desktop/fintech/Fintech/deal_stack_manager.py):
   - Added in-process concurrency protection (`threading.RLock()`).
   - Chat-partitioned message deduplication (`self._chat_processed_ids`).
   - Populated `_create_new_stack` and `_accumulate_terms` with `industry`, `tenor`, `credit_rating`, `spread`, field evidences, and histories.

---

## 5. Test Suite Execution & Verification

The complete project test suite was executed to verify that Milestone 1 operates correctly and introduces **zero regressions**:

```bash
cd /Users/manasi/Desktop/fintech/Fintech && PYTHONPATH=. .venv/bin/pytest
```

### Test Results Summary
- **Total Test Cases Collected:** 248
- **Total Passed:** 248 (100%)
- **Failures / Errors:** 0
- **Execution Time:** 2.83 seconds

### Breakdown by Test Module
- `tests/test_milestone1_subject_matching.py`: **12/12 passed** (New Milestone 1 tests)
- `tests/test_deal_stack_manager.py`: **15/15 passed** (Manager orchestration & lifecycle)
- `tests/test_deal_stack_matcher.py`: **17/17 passed** (8-tier deterministic matching)
- `tests/test_lifecycle_state_engine.py`: **18/18 passed** (Lifecycle state machine)
- `tests/test_app_bond_identification.py`: **1/1 passed**
- `tests/test_app_extraction.py`: **5/5 passed**
- `tests/test_baseline_report.py`: **8/8 passed**
- `tests/test_bond_identification.py`: **8/8 passed**
- `tests/test_bond_matching.py`: **20/20 passed**
- `tests/test_bond_reference.py`: **14/14 passed**
- `tests/test_db_identification.py`: **12/12 passed**
- `tests/test_db_schema.py`: **10/10 passed**
- `tests/test_db_store.py`: **14/14 passed**
- `tests/test_extraction_schema.py`: **6/6 passed**
- `tests/test_extraction_validation.py`: **34/34 passed**
- `tests/test_isin_utils.py`: **12/12 passed**
- `tests/test_lexicon.py`: **3/3 passed**
- `tests/test_normalize.py`: **8/8 passed**
- `tests/test_phase7_end_to_end.py`: **2/2 passed**
- `tests/test_pipeline_parser.py`: **15/15 passed**
- `tests/test_privacy.py`: **5/5 passed**
- `tests/test_prompt14_coverage.py`: **9/9 passed**

---

## 6. Remaining Limitations & Non-Goals for Milestone 1

1. **Pipeline Integration (Deferred to Milestone 4 & 5):**
   `pipeline.py` and `app.py` continue to use the baseline pipeline. `DealStackManager` is verified via unit tests and will be integrated into the actual upload pipeline in Milestone 4.
2. **Database Persistence (Deferred to Milestone 4):**
   Deal stacks currently persist in-memory with `export_state()` / `import_state()` hooks. SQLite Migration 6 (`deal_stacks`, `deal_stack_messages`, `deal_stack_events`) will be applied in Milestone 4.
3. **Settlement Cycle and Full Schema Mapping (Deferred to Milestone 2):**
   `settlement` (`T+0`, `T+1`, `T+2`) and transaction date linking will be completed in Milestone 2.

---

## 7. Next Steps: Milestone 2 Preview

Upon user review and approval of Milestone 1, we will proceed to **Milestone 2: Complete Partial-Information Accumulation and Field-Level Evidence Tracking**:
1. Accumulate fragmented terms across conversational rounds without data loss.
2. Track exact source message IDs and character byte offsets for every extracted field.
3. Enforce validation rules on accumulated deal terms (buyer $\neq$ seller, price/yield consistency).
4. Resolve counterparty roles and two-way inquiry sides.
5. Create dedicated unit tests in `tests/test_milestone2_accumulation.py`.

