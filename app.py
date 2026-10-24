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

import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import db
import pipeline

app = FastAPI(title="Bond Trade Extraction App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache for uploads
UPLOAD_SESSIONS: Dict[str, Dict[str, Any]] = {}

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    upload_id = str(uuid.uuid4())
    filename = file.filename or "chat_export.txt"

    parsed_messages, parse_report = pipeline.parse_raw_chat_content(
        raw_text,
        chat_id=upload_id,
        return_report=True,
    )

    UPLOAD_SESSIONS[upload_id] = {
        "upload_id": upload_id,
        "filename": filename,
        "raw_text": raw_text,
        "parsed_messages": parsed_messages,
        "parse_report": parse_report,
        "extractions": [],
        "token_usage": {},
        "context_window": {},
    }

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
    session = UPLOAD_SESSIONS.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found. Please upload a file first.")

    parsed_messages = session["parsed_messages"]
    if not parsed_messages:
        return {
            "upload_id": upload_id,
            "prompt_version": pipeline.PROMPT_VERSION,
            "trade_contexts_count": 0,
            "extractions": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "context_window": {
                "used_tokens": 0,
                "max_tokens": config.MODEL_CONTEXT_LENGTH_FALLBACK,
                "percentage": 0.0,
            },
        }

    # 4. Clean + route + group into trade contexts
    contexts, skipped_low_relevance = pipeline.group_into_trade_contexts(
        parsed_messages,
        return_skipped=True,
    )

    if not contexts:
        return {
            "upload_id": upload_id,
            "prompt_version": pipeline.PROMPT_VERSION,
            "trade_contexts_count": 0,
            "skipped_low_relevance": len(skipped_low_relevance),
            "message": "No bond-relevant trade negotiation contexts detected in chat.",
            "extractions": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "context_window": {
                "used_tokens": 0,
                "max_tokens": pipeline.get_model_context_length(),
                "percentage": 0.0,
            },
        }

    # 5. Extract via LLM for each context
    extractions = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    max_single_call_tokens = 0

    selected_contexts = (
        contexts
        if config.MAX_CONTEXTS is None
        else contexts[:config.MAX_CONTEXTS]
    )

    for ctx in selected_contexts:
        extraction_result, tokens = pipeline.call_llm_extraction(ctx["snippet_text"])

        prompt_tk = tokens.get("prompt_tokens", 0)
        compl_tk = tokens.get("completion_tokens", 0)
        tot_tk = tokens.get("total_tokens", prompt_tk + compl_tk)

        total_prompt_tokens += prompt_tk
        total_completion_tokens += compl_tk
        if tot_tk > max_single_call_tokens:
            max_single_call_tokens = tot_tk

        if extraction_result.get("extraction_failed"):
            failed_result = dict(extraction_result)
            failed_result["context_id"] = ctx["context_id"]
            failed_result["group_name"] = ctx.get("group_name", "Default")
            failed_result["message_indices"] = ctx["message_indices"]
            failed_result["snippet_text"] = ctx["snippet_text"]
            failed_result["call_tokens"] = tokens
            extractions.append(failed_result)
            continue

        for deal in extraction_result.get("deals", []):
            if not isinstance(deal, dict):
                continue
            # Step 7: Validate business rules
            context_datetime = max(
                (
                    message["dt"]
                    for message in ctx["messages"]
                    if isinstance(message.get("dt"), datetime)
                ),
                default=None,
            )
            validated_deal = pipeline.validate_extraction(
                deal,
                context_snippet=ctx["snippet_text"],
                message_datetime=context_datetime,
            )
            validated_deal["context_id"] = ctx["context_id"]
            validated_deal["group_name"] = ctx.get("group_name", "Default")
            validated_deal["message_indices"] = ctx["message_indices"]
            validated_deal["snippet_text"] = ctx["snippet_text"]
            validated_deal["call_tokens"] = tokens
            validated_deal["prompt_version"] = extraction_result["prompt_version"]

            extractions.append(validated_deal)

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

    # Save to session
    session["extractions"] = extractions
    session["token_usage"] = token_usage
    session["context_window"] = context_window

    return {
        "upload_id": upload_id,
        "trade_contexts_count": len(selected_contexts),
        "total_contexts_detected": len(contexts),
        "skipped_low_relevance": len(skipped_low_relevance),
        "extractions": extractions,
        "prompt_version": pipeline.PROMPT_VERSION,
        "token_usage": token_usage,
        "context_window": context_window,
    }


@app.post("/store/{upload_id}")
async def store_extracted_data(upload_id: str):
    """
    Step 8 & 9: Persist parsed messages, distinct bonds, and transactions
    into SQLite database in a single atomic transaction.
    """
    session = UPLOAD_SESSIONS.get(upload_id)
    if not session:
        raise HTTPException(status_code=404, detail="Upload session not found.")

    parsed_messages = session.get("parsed_messages", [])
    extractions = session.get("extractions", [])
    if any(
        "bond_identity_text" in deal or "extraction_failed" in deal
        for deal in extractions
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "The current database schema cannot store the new extraction "
                "format yet; no extracted deals were stored."
            ),
        )

    try:
        counts = db.persist_upload(
            db_path=db.DB_PATH,
            chat_id=upload_id,
            parsed_messages=parsed_messages,
            validated_deals=extractions,
        )

        return {
            "success": True,
            "upload_id": upload_id,
            "messages_stored": counts["messages_stored"],
            "bonds_stored": counts["bonds_stored"],
            "transactions_stored": counts["transactions_stored"],
            "message": (
                f"Stored successfully — {counts['messages_stored']} messages, "
                f"{counts['bonds_stored']} bonds, {counts['transactions_stored']} "
                f"transactions saved to the database."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database storage failed: {e}")


@app.get("/data")
def get_database_records():
    """Retrieve all current records in SQLite for interactive database inspector."""
    try:
        return db.fetch_database_data(db.DB_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch database data: {e}")


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

    with open(found_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    upload_id = str(uuid.uuid4())
    parsed_messages, parse_report = pipeline.parse_raw_chat_content(
        raw_text,
        chat_id=upload_id,
        return_report=True,
    )

    UPLOAD_SESSIONS[upload_id] = {
        "upload_id": upload_id,
        "filename": safe_name,
        "raw_text": raw_text,
        "parsed_messages": parsed_messages,
        "parse_report": parse_report,
        "extractions": [],
        "token_usage": {},
        "context_window": {},
    }

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
