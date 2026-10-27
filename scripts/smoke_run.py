"""Run extraction on one chat export and print aggregate counts only."""

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_smoke(input_path: Path) -> Dict[str, Any]:
    """Run the same extraction path as /extract and return a safe summary."""
    import app

    upload_id = str(uuid.uuid4())
    raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    messages, parse_report = app.pipeline.parse_raw_chat_content(
        raw_text,
        chat_id=upload_id,
        return_report=True,
    )
    app._store_upload_session(upload_id, {
        "upload_id": upload_id,
        "filename": input_path.name,
        "raw_text": raw_text,
        "parsed_messages": messages,
        "parse_report": parse_report,
        "extractions": [],
        "token_usage": {},
        "context_window": {},
        "context_message_maps": {},
        "prompt_version": app.pipeline.PROMPT_VERSION,
        "model_name": app.pipeline.MODEL_NAME,
    })
    try:
        extraction_report = asyncio.run(app.extract_trades(upload_id))
    finally:
        app.UPLOAD_SESSIONS.pop(upload_id, None)

    windows = app.pipeline.assign_contexts(messages)

    def warning_type(warning: Any) -> str:
        text = str(warning)
        if re.fullmatch(r"[a-z][a-z0-9_]*", text):
            return text
        if text.casefold().startswith("yield "):
            return "yield_out_of_expected_range"
        return "other"

    warnings = Counter(
        warning_type(warning)
        for deal in extraction_report["extractions"]
        for warning in deal.get("warnings") or []
    )
    return {
        "messages_parsed": extraction_report["messages_parsed"],
        "chats": len({message["chat_key"] for message in messages}),
        "skipped": extraction_report["skipped_on_parse"],
        "skipped_on_parse": extraction_report["skipped_on_parse"],
        "media": extraction_report["media"],
        "system": extraction_report["system"],
        "trade_contexts_count": extraction_report["trade_contexts_count"],
        "contexts_detected": extraction_report["contexts_detected"],
        "total_contexts_detected": extraction_report["total_contexts_detected"],
        "reply_linked": sum(window["reply_linked_count"] for window in windows),
        "unresolved_replies": sum(
            len(window["unresolved_replies"]) for window in windows
        ),
        "skipped_low_relevance": extraction_report["skipped_low_relevance"],
        "skipped_context_limit": extraction_report["skipped_context_limit"],
        "processed": extraction_report["processed"],
        "failed": extraction_report["failed"],
        "context_errors": len(extraction_report["context_errors"]),
        "deals_by_status": extraction_report["deals_by_status"],
        "needs_review_count": extraction_report["needs_review_count"],
        "warnings_by_type": dict(sorted(warnings.items())),
        "prompt_version": extraction_report["prompt_version"],
        "token_usage": extraction_report["token_usage"],
        "context_window": extraction_report["context_window"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extraction and print aggregate counts only."
    )
    parser.add_argument("chat_file", type=Path)
    args = parser.parse_args()
    input_path = args.chat_file.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    print(json.dumps(run_smoke(input_path), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
