"""Run the current extraction pipeline and write a comparison report."""

import argparse
import copy
import importlib
import importlib.util
import json
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module_missing(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is None
    try:
        return importlib.util.find_spec(name) is None
    except (ValueError, ImportError):
        return False


def _load_pipeline() -> Tuple[Any, List[str]]:
    """Load pipeline code even when its optional Ollama adapters are absent."""
    missing_modules = []
    for module_name in ("ollama", "langchain_ollama"):
        if _module_missing(module_name):
            missing_modules.append(module_name)

    if "ollama" in missing_modules:
        ollama_stub = types.ModuleType("ollama")

        class MissingOllamaClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("The ollama package is not installed")

        ollama_stub.Client = MissingOllamaClient
        sys.modules["ollama"] = ollama_stub

    if "langchain_ollama" in missing_modules:
        langchain_ollama_stub = types.ModuleType("langchain_ollama")

        class MissingChatOllama:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("The langchain-ollama package is not installed")

        langchain_ollama_stub.ChatOllama = MissingChatOllama
        sys.modules["langchain_ollama"] = langchain_ollama_stub

    return importlib.import_module("pipeline"), missing_modules


pipeline, MISSING_LLM_MODULES = _load_pipeline()
import config


def _count_values(values: List[Any]) -> List[Dict[str, Any]]:
    counts = Counter(json.dumps(value, sort_keys=True) for value in values)
    return [
        {"value": json.loads(value), "count": count}
        for value, count in sorted(counts.items())
    ]


def _read_chat_types(raw_text: str) -> Tuple[List[Any], int, int]:
    chat_types = []
    invalid_json_lines = 0
    non_object_json_lines = 0
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid_json_lines += 1
            continue
        if not isinstance(value, dict):
            non_object_json_lines += 1
            continue
        chat_types.append(value.get("chat_type"))
    return chat_types, invalid_json_lines, non_object_json_lines


def _window_counts(messages: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Count assigned windows before and after the current relevance gate."""
    dropped_by_system_filter = sum(
        pipeline.is_system_message(message) for message in messages
    )
    windows = pipeline.assign_contexts(messages)
    windows_relevant = sum(
        pipeline.is_bond_trade_relevant(window["messages"])
        for window in windows
    )
    return dropped_by_system_filter, len(windows), windows_relevant


def _select_contexts(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select chronologically, applying only an explicitly configured dev cap."""
    ordered = sorted(contexts, key=lambda context: context.get("start", ""))
    return ordered if config.MAX_CONTEXTS is None else ordered[:config.MAX_CONTEXTS]


def _mock_extraction(_: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    return {
        "deals": [],
        "prompt_version": pipeline.PROMPT_VERSION,
    }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _llm_mode() -> Tuple[bool, Optional[str]]:
    if MISSING_LLM_MODULES:
        return True, "Missing optional modules: " + ", ".join(MISSING_LLM_MODULES)
    try:
        pipeline.ollama.Client().list()
    except Exception as error:
        return True, f"Ollama is unavailable: {error}"
    return False, None


def create_report(input_path: Path) -> Dict[str, Any]:
    raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    chat_types, invalid_json_lines, non_object_json_lines = _read_chat_types(raw_text)
    messages = pipeline.parse_raw_chat_content(raw_text, chat_id=input_path.stem)
    chat_types_match_parsed = len(chat_types) == len(messages)
    parsed_chat_types = chat_types if chat_types_match_parsed else [None] * len(messages)

    dropped, windows_formed, windows_relevant = _window_counts(messages)
    contexts, skipped_low_relevance = pipeline.group_into_trade_contexts(
        messages,
        return_skipped=True,
    )
    if windows_relevant != len(contexts):
        raise RuntimeError(
            "Window metrics do not match the current pipeline grouping "
            f"({windows_relevant} relevant windows counted, {len(contexts)} returned)"
        )

    selected_contexts = _select_contexts(contexts)
    use_mock, fallback_reason = _llm_mode()
    extract = _mock_extraction if use_mock else pipeline.call_llm_extraction

    extracted_deals = []
    non_object_outputs = []
    for context in selected_contexts:
        result, tokens = extract(context["snippet_text"])
        if result.get("extraction_failed"):
            non_object_outputs.append(result)
            continue
        deals = result.get("deals", [])
        for raw_deal in deals:
            if not isinstance(raw_deal, dict):
                non_object_outputs.append(raw_deal)
                continue
            context_datetime = max(
                (
                    message["dt"]
                    for message in context["messages"]
                    if isinstance(message.get("dt"), datetime)
                ),
                default=None,
            )
            validated_deal = pipeline.validate_extraction(
                copy.deepcopy(raw_deal),
                context_snippet=context["snippet_text"],
                message_datetime=context_datetime,
            )
            extracted_deals.append({
                "context_id": context["context_id"],
                "group_name": context.get("group_name"),
                "raw_llm_json": raw_deal,
                "validated_deal": validated_deal,
                "call_tokens": tokens,
            })

    return {
        "source_file": str(input_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": pipeline.MODEL_NAME,
        "prompt_version": pipeline.PROMPT_VERSION,
        "llm_mode": "mock" if use_mock else "ollama",
        "llm_fallback_reason": fallback_reason,
        "messages_parsed": len(messages),
        "counts_by_chat_type": _count_values(parsed_chat_types),
        "chat_type_counts_mapped_to_parsed_messages": chat_types_match_parsed,
        "counts_by_group_name": _count_values(
            [message.get("group_name") for message in messages]
        ),
        "invalid_jsonl_lines": invalid_json_lines,
        "non_object_jsonl_lines": non_object_json_lines,
        "system_filter_dropped": dropped,
        "windows_formed": windows_formed,
        "windows_passed_relevance": windows_relevant,
        "skipped_low_relevance": len(skipped_low_relevance),
        "contexts_selected_for_extraction": len(selected_contexts),
        "contexts_sent_to_llm": 0 if use_mock else len(selected_contexts),
        "mock_extraction_calls": len(selected_contexts) if use_mock else 0,
        "contexts_cut_off_by_top_8": len(contexts) - len(selected_contexts),
        "contexts_cut_off_by_max_contexts": len(contexts) - len(selected_contexts),
        "extracted_deals": extracted_deals,
        "non_object_llm_outputs": non_object_outputs,
        "non_object_llm_output_count": len(non_object_outputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current pipeline on a JSONL chat and save a baseline report."
    )
    parser.add_argument("chat_jsonl", type=Path, help="Path to the JSONL chat file")
    args = parser.parse_args()
    input_path = args.chat_jsonl.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    report = create_report(input_path)
    output_dir = ROOT / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"{input_path.stem}_baseline_{timestamp}.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Baseline report written to {output_path}")
    print(
        f"Parsed {report['messages_parsed']} messages; "
        f"{report['windows_passed_relevance']} relevant windows; "
        f"{report['contexts_selected_for_extraction']} extraction calls "
        f"({report['llm_mode']} mode)."
    )


if __name__ == "__main__":
    main()
