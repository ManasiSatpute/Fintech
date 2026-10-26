"""
FastAPI application for Bond Trade Extraction.
Connects the Chat-style UI to the Python backend with:
- /upload
- /extract/{upload_id}
- /store/{upload_id}
- /data
- /samples
- /load-sample/{filename}
"""

import hashlib
import logging
import os
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import config
import db
import pipeline
import privacy
from bond_matching import BondMatchResult
from bond_reference import BondReferenceError, get_bond_reference_index
from bond_identification import identification_summary

app = FastAPI(title="Bond Trade Extraction App", version="1.0.0")
logger = logging.getLogger(__name__)


class _ContextExtractionFailure(RuntimeError):
    def __init__(self, error_category: str, correlation_id: str) -> None:
        super().__init__("Context extraction failed.")
        self.error_category = error_category
        self.correlation_id = correlation_id

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache for uploads
UPLOAD_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _prune_expired_upload_sessions(now: float) -> None:
    expired_ids = [
        key
        for key, session in UPLOAD_SESSIONS.items()
        if now - session.get("_created_monotonic", 0)
        >= config.UPLOAD_SESSION_TTL_SECONDS
    ]
    for expired_id in expired_ids:
        del UPLOAD_SESSIONS[expired_id]


def _store_upload_session(upload_id: str, session: Dict[str, Any]) -> None:
    now = time.monotonic()
    _prune_expired_upload_sessions(now)
    session["_created_monotonic"] = now
    UPLOAD_SESSIONS[upload_id] = session


def _get_upload_session(upload_id: str) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    _prune_expired_upload_sessions(now)
    return UPLOAD_SESSIONS.get(upload_id)


# Ensure static folder exists
os.makedirs("static", exist_ok=True)
db.init_db()


@app.get("/")
def get_index():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Step 1 & 2: Receive WhatsApp .txt or .json export, parse raw messages,
    return parsed messages immediately to frontend (Step 3).
    """
    try:
        content_bytes = await file.read()
        raw_text = content_bytes.decode("utf-8", errors="replace")
    except Exception:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    upload_id = str(uuid.uuid4())
    filename = file.filename or "chat_export.txt"

    parsed_messages, parse_report = pipeline.parse_raw_chat_content(
        raw_text,
        chat_id=upload_id,
        return_report=True,
    )

    _store_upload_session(upload_id, {
        "upload_id": upload_id,
        "filename": filename,
        "source_hash": hashlib.sha256(content_bytes).hexdigest(),
        "raw_text": raw_text,
        "parsed_messages": parsed_messages,
        "parse_report": parse_report,
        "extractions": [],
        "token_usage": {},
        "context_window": {},
        "context_message_maps": {},
        "prompt_version": pipeline.PROMPT_VERSION,
        "model_name": pipeline.MODEL_NAME,
    })

    return {
        "upload_id": upload_id,
        "filename": filename,
        "parsed_count": len(parsed_messages),
        "parse_report": parse_report,
        "messages": [
            {
                "message_id": m["message_id"],
                "sender": m["sender"],
                "timestamp": m["timestamp"],
                "message_text": m["message_text"],
                "group_name": m.get("group_name", "Default"),
            }
            for m in parsed_messages
        ],
    }


@app.post("/extract/{upload_id}")
async def extract_trades(upload_id: str):
    """
    Steps 4-7:
    - Step 4: Clean noise & group into trade contexts
    - Step 5: Ollama extraction via langchain-ollama (gemma4:31b-cloud)
    - Step 6: Real token usage & context window calculation
    - Step 7: Business-rule validation
    """
    session = _get_upload_session(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found. Please upload a file first.")

    parsed_messages = session["parsed_messages"]
    contexts, skipped_low_relevance = pipeline.group_into_trade_contexts(
        parsed_messages,
        return_skipped=True,
    )
    contexts.sort(key=lambda context: context["start"])
    extractions = []
    identification_results: List[Optional[BondMatchResult]] = []
    context_errors = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    max_single_call_tokens = 0

    selected_contexts = contexts
    if config.MAX_CONTEXTS is not None:
        selected_contexts = selected_contexts[:config.MAX_CONTEXTS]
    skipped_context_limit = len(contexts) - len(selected_contexts)
    context_message_maps = {
        ctx["context_id"]: ctx["message_number_map"]
        for ctx in selected_contexts
    }
    processed_contexts = 0
    failed_contexts = 0
    reference_index = None
    reference_error = None
    if selected_contexts:
        try:
            reference_index = get_bond_reference_index()
        except BondReferenceError:
            logger.error(
                "Bond reference data is unavailable; check BOND_REFERENCE_CSV."
            )
            reference_error = (
                "Bond reference data is unavailable. Check BOND_REFERENCE_CSV."
            )

    for ctx in selected_contexts:
        extraction_result: Dict[str, Any] = {}
        tokens: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            extraction_result, tokens = await run_in_threadpool(
                pipeline.call_llm_extraction,
                ctx["snippet_text"],
            )
            prompt_tk = tokens.get("prompt_tokens", 0)
            compl_tk = tokens.get("completion_tokens", 0)
            tot_tk = tokens.get("total_tokens", prompt_tk + compl_tk)
            total_prompt_tokens += prompt_tk
            total_completion_tokens += compl_tk
            max_single_call_tokens = max(max_single_call_tokens, tot_tk)

            if extraction_result.get("extraction_failed"):
                raise _ContextExtractionFailure(
                    error_category=extraction_result.get("error_category")
                    or "model_extraction_failed",
                    correlation_id=extraction_result.get("correlation_id")
                    or str(uuid.uuid4()),
                )

            context_datetime = max(
                (
                    message["dt"]
                    for message in ctx["messages"]
                    if isinstance(message.get("dt"), datetime)
                ),
                default=None,
            )
            for deal in extraction_result.get("deals", []):
                if not isinstance(deal, dict):
                    raise ValueError("Model returned a non-object deal")
                validated_deal = pipeline.validate_extraction(
                    deal,
                    context_snippet=ctx["snippet_text"],
                    message_datetime=context_datetime,
                )
                if validated_deal.get("extraction_failed"):
                    validated_deal.pop("raw_text", None)
                    validated_deal["error"] = (
                        "A model deal could not be validated."
                    )
                    validated_deal["error_category"] = "invalid_extracted_deal"
                    validated_deal["correlation_id"] = str(uuid.uuid4())
                validated_deal = privacy.restore_pii(
                    validated_deal,
                    ctx.get("privacy_map", {}),
                )
                validated_deal.update({
                    "context_id": ctx["context_id"],
                    "chat_key": ctx["chat_key"],
                    "chat_name": ctx["chat_name"],
                    "chat_type": ctx["chat_type"],
                    "message_ids": ctx["message_ids"],
                    "group_name": ctx.get("group_name", "Default"),
                    "message_indices": ctx["message_indices"],
                    "message_number_map": ctx["message_number_map"],
                    "snippet_text": ctx["snippet_text"],
                    "call_tokens": tokens,
                    "prompt_version": extraction_result["prompt_version"],
                    "model_name": pipeline.MODEL_NAME,
                })
                match_result = pipeline.identify_extracted_bond(
                    validated_deal,
                    parsed_messages,
                    ctx["message_number_map"],
                    reference_index,
                    reference_error,
                )
                validated_deal["identification"] = identification_summary(
                    match_result
                )
                identification_results.append(match_result)
                extractions.append(validated_deal)
            processed_contexts += 1
        except Exception as error:
            error_category = (
                error.error_category
                if isinstance(error, _ContextExtractionFailure)
                else "context_processing_failed"
            )
            correlation_id = (
                error.correlation_id
                if isinstance(error, _ContextExtractionFailure)
                else str(uuid.uuid4())
            )
            logger.warning(
                "Context extraction failed: category=%s correlation_id=%s",
                error_category,
                correlation_id,
            )
            failed_contexts += 1
            context_error = {
                "context_id": ctx["context_id"],
                "error": "Extraction failed for this context.",
                "error_category": error_category,
                "correlation_id": correlation_id,
                "raw_llm_output": None,
                "call_tokens": tokens,
                "chat_key": ctx["chat_key"],
                "chat_name": ctx["chat_name"],
                "chat_type": ctx["chat_type"],
                "message_ids": ctx["message_ids"],
            }
            context_errors.append(context_error)
            identification_results.append(None)
            extractions.append({
                "extraction_failed": True,
                **context_error,
                "chat_key": ctx["chat_key"],
                "chat_name": ctx["chat_name"],
                "chat_type": ctx["chat_type"],
                "message_ids": ctx["message_ids"],
                "message_indices": ctx["message_indices"],
                "message_number_map": ctx["message_number_map"],
                "prompt_version": pipeline.PROMPT_VERSION,
                "model_name": pipeline.MODEL_NAME,
            })

    # Step 6: Query real context window length & compute usage
    max_context_length = pipeline.get_model_context_length()
    overall_total_tokens = total_prompt_tokens + total_completion_tokens

    # Context window percentage based on max prompt/context used in an invocation against model limit
    ctx_used = max_single_call_tokens if max_single_call_tokens > 0 else overall_total_tokens
    ctx_percentage = round((ctx_used / max_context_length) * 100, 4) if max_context_length else 0.0

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": overall_total_tokens,
    }

    context_window = {
        "used_tokens": ctx_used,
        "max_tokens": max_context_length,
        "percentage": ctx_percentage,
        "model_name": pipeline.MODEL_NAME,
    }
    deals_by_status = dict(sorted(Counter(
        deal.get("status") or "UNKNOWN"
        for deal in extractions
        if not deal.get("extraction_failed")
    ).items()))
    identification_by_status = dict(sorted(Counter(
        deal["identification"]["status"]
        for deal in extractions
        if not deal.get("extraction_failed")
        and isinstance(deal.get("identification"), dict)
        and deal["identification"].get("status")
    ).items()))
    parse_report = session.get("parse_report", {})
    system_count = sum(
        bool(message.get("is_system"))
        for message in parsed_messages
    )

    # Save to session
    session["extractions"] = extractions
    session["identification_results"] = identification_results
    session["token_usage"] = token_usage
    session["context_window"] = context_window
    session["context_message_maps"] = context_message_maps
    session["prompt_version"] = pipeline.PROMPT_VERSION
    session["model_name"] = pipeline.MODEL_NAME
    session["context_errors"] = context_errors
    session["context_chat_info"] = {
        ctx["context_id"]: {
            "chat_key": ctx["chat_key"],
            "chat_name": ctx["chat_name"],
            "chat_type": ctx["chat_type"],
            "message_ids": ctx["message_ids"],
        }
        for ctx in selected_contexts
    }

    return {
        "upload_id": upload_id,
        "trade_contexts_count": len(selected_contexts),
        "contexts_detected": len(contexts),
        "total_contexts_detected": len(contexts),
        "skipped_low_relevance": len(skipped_low_relevance),
        "skipped_context_limit": skipped_context_limit,
        "messages_parsed": len(parsed_messages),
        "media": parse_report.get("media", 0),
        "system": system_count,
        "skipped_on_parse": parse_report.get("skipped", 0),
        "processed": processed_contexts,
        "failed": failed_contexts,
        "context_errors": context_errors,
        "deals_by_status": deals_by_status,
        "identification_by_status": identification_by_status,
        "bond_reference_error": reference_error,
        "needs_review_count": sum(
            bool(deal.get("needs_review"))
            for deal in extractions
            if not deal.get("extraction_failed")
        ),
        "extractions": extractions,
        "prompt_version": pipeline.PROMPT_VERSION,
        "token_usage": token_usage,
        "context_window": context_window,
    }


@app.post("/store/{upload_id}")
async def store_extracted_data(upload_id: str, update_existing: bool = False):
    """
    Step 8 & 9: Persist parsed messages, distinct bonds, and transactions
    into SQLite database in a single atomic transaction.
    """
    session = _get_upload_session(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found.")

    parsed_messages = session.get("parsed_messages", [])
    extractions = session.get("extractions", [])

    try:
        session_identification_results = session.get(
            "identification_results",
            [],
        )
        identification_results = []
        for index, deal in enumerate(extractions):
            if deal.get("extraction_failed"):
                continue
            if index >= len(session_identification_results):
                raise RuntimeError(
                    "Identification result is missing for an extracted deal."
                )
            match_result = session_identification_results[index]
            if not isinstance(match_result, BondMatchResult):
                raise RuntimeError(
                    "Identification result is unavailable for an extracted deal."
                )
            identification_results.append(match_result)

        counts = db.persist_upload(
            db_path=db.DB_PATH,
            chat_id=upload_id,
            parsed_messages=parsed_messages,
            validated_deals=extractions,
            context_message_maps=session.get("context_message_maps", {}),
            prompt_version=session.get("prompt_version"),
            model_name=session.get("model_name"),
            source_hash=session.get("source_hash"),
            update_existing=update_existing,
            defer_bond_creation=True,
            identification_results=identification_results,
        )

        identification_counts: Dict[str, int] = counts.get(
            "identification_by_status",
            {},
        )

        messages_inserted = counts.get(
            "messages_inserted",
            counts.get("messages_stored", 0),
        )
        messages_already_present = counts.get("messages_already_present", 0)
        transactions_inserted = counts.get(
            "transactions_inserted",
            counts.get("transactions_stored", 0),
        )
        transactions_updated = counts.get("transactions_updated", 0)
        transactions_skipped = counts.get("transactions_skipped", 0)
        already_stored = counts.get("already_stored", False)
        if already_stored:
            message = (
                "This file was already stored; no records were changed. "
                f"{messages_already_present} messages already present "
                f"and {transactions_skipped} transactions skipped."
            )
        else:
            message = (
                f"Stored successfully — {messages_inserted} messages "
                f"inserted, {messages_already_present} already present, "
                f"{transactions_inserted} transactions inserted, "
                f"{transactions_updated} updated, and "
                f"{transactions_skipped} skipped."
            )
        return {
            "success": True,
            "upload_id": upload_id,
            "already_stored": already_stored,
            "messages_inserted": messages_inserted,
            "messages_already_present": messages_already_present,
            "transactions_inserted": transactions_inserted,
            "transactions_updated": transactions_updated,
            "transactions_skipped": transactions_skipped,
            "messages_stored": counts.get("messages_stored", messages_inserted),
            "bonds_stored": counts.get("bonds_stored", 0),
            "transactions_stored": counts.get(
                "transactions_stored",
                transactions_inserted + transactions_updated,
            ),
            "identification_by_status": identification_counts,
            "identification_skipped": counts.get("identification_skipped", 0),
            "message": message,
        }
    except Exception:
        logger.exception("Database storage failed")
        raise HTTPException(
            status_code=500,
            detail="Database storage failed.",
        )


@app.get("/data")
def get_database_records():
    """Retrieve all current records in SQLite for interactive database inspector."""
    try:
        return db.fetch_database_data(db.DB_PATH)
    except Exception:
        logger.exception("Failed to fetch database data")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch database data.",
        )


@app.get("/samples")
def get_sample_files():
    """List available sample files for quick 1-click testing in UI."""
    samples = []
    candidates = [
        {"name": "chat_single_deal.txt", "path": "testFiles/chat_single_deal.txt", "description": "Single deal (6.65% GS 2026, 5 Cr @ 6.465)"},
        {"name": "chat_multiple_deals.txt", "path": "testFiles/chat_multiple_deals.txt", "description": "Multiple deals (7.18% SDL 2028 & 6.79% GS 2027)"},
        {"name": "chat_real_world.json", "path": "testFiles/chat_real_world.json", "description": "Real-world multi-channel JSON (Oxyzo, NHAI, HUDCO, etc.)"},
        {"name": "chat_noise_and_deal.txt", "path": "testFiles/chat_noise_and_deal.txt", "description": "Small talk + single deal (8.25% SDL 2029)"},
        {"name": "chat_ambiguous.txt", "path": "testFiles/chat_ambiguous.txt", "description": "Unconfirmed deal (Neither said 'done')"},
        {"name": "chat_no_deal.txt", "path": "testFiles/chat_no_deal.txt", "description": "Pure chit-chat (No bond discussion)"},
        {"name": "chat_2.txt", "path": "chat_2.txt", "description": "Original chat_2.txt (6.65% GS 2026)"},
    ]

    for c in candidates:
        if os.path.exists(c["path"]):
            samples.append({
                "name": c["name"],
                "path": c["path"],
                "description": c["description"],
                "size": os.path.getsize(c["path"]),
            })

    return {"samples": samples}


@app.post("/load-sample/{filename}")
def load_sample_file(filename: str):
    """Load a sample file directly by name without having to upload via file picker."""
    safe_name = os.path.basename(filename)
    candidates = [f"testFiles/{safe_name}", safe_name]
    found_path = None
    for p in candidates:
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        raise HTTPException(status_code=404, detail=f"Sample file {safe_name} not found.")

    with open(found_path, "rb") as f:
        content_bytes = f.read()
    raw_text = content_bytes.decode("utf-8", errors="replace")

    upload_id = str(uuid.uuid4())
    parsed_messages, parse_report = pipeline.parse_raw_chat_content(
        raw_text,
        chat_id=upload_id,
        return_report=True,
    )

    _store_upload_session(upload_id, {
        "upload_id": upload_id,
        "filename": safe_name,
        "source_hash": hashlib.sha256(content_bytes).hexdigest(),
        "raw_text": raw_text,
        "parsed_messages": parsed_messages,
        "parse_report": parse_report,
        "extractions": [],
        "token_usage": {},
        "context_window": {},
        "context_message_maps": {},
        "prompt_version": pipeline.PROMPT_VERSION,
        "model_name": pipeline.MODEL_NAME,
    })

    return {
        "upload_id": upload_id,
        "filename": safe_name,
        "parsed_count": len(parsed_messages),
        "parse_report": parse_report,
        "messages": [
            {
                "message_id": m["message_id"],
                "sender": m["sender"],
                "timestamp": m["timestamp"],
                "message_text": m["message_text"],
                "group_name": m.get("group_name", "Default"),
            }
            for m in parsed_messages
        ],
    }


# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")
