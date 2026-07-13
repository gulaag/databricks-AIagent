"""
The Tech Engineer Study Group agent — a single, unified implementation.

One class, ``TechEngineerAgent``, authored on the Databricks-recommended
``mlflow.pyfunc.ResponsesAgent`` interface. It replaces the three earlier
implementations (autonomous PyFunc, ResponsesAgent, and conversational
orchestrator) with no loss of functionality. It supports three modes:

  - ``auto``  (default) — hands-off: search → draft → post → log in one call.
  - ``draft``           — propose a complete announcement and refine it from
                          natural-language feedback; posting is WITHHELD so a
                          human approves first. The same mode serves the initial
                          proposal and every refinement (the conversation history
                          distinguishes them).
  - ``send``            — post an already-approved draft (WYSIWYG) and audit-log it.

The mode is selected per request via ``custom_inputs={"mode": ...}`` (defaulting
to ``auto``), so the *same* deployed endpoint drives the autonomous demo AND the
human-in-the-loop conversational demo.

For notebook ergonomics there are thin, stateful convenience methods
(``autonomous`` / ``propose`` / ``refine`` / ``send``) that keep the running
conversation on the instance and call the very same ``predict`` path — so the
notebook exercises exactly the code that serves in production.

Hardening (applies to the LLM-driven ``auto`` and ``draft`` modes):
  - every tool call runs in try/except; malformed JSON, missing args, or a tool
    exception are fed back to the LLM as an error result — the request never 500s;
  - the LLM call retries transient failures with exponential backoff;
  - retrieved content is treated as untrusted data (indirect-injection guard);
  - a wall-clock deadline + max-iteration cap always return a clean message.

Auth: the LLM call goes through ``mlflow.deployments`` and Vector Search / SQL
through the Databricks SDK. Both pick up the M2M OAuth that Model Serving injects
from the model's declared ``resources`` — no ``DATABRICKS_TOKEN`` at serving time.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Generator

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from src.agent.prompts import (
    AUTONOMOUS_SYSTEM_PROMPT,
    DRAFT_SYSTEM_PROMPT,
    SEARCH_TOOLS,
    TOOL_DEFINITIONS,
    UNTRUSTED_CONTENT_GUARD,
    WORKFLOW_STEPS,
)
from src.tools.logger import log_agent_action
from src.tools.messaging import post_to_channel
from src.tools.search import search_knowledge_base

_MAX_ITERATIONS = 6
_WALL_TIME_SECONDS = 150
_LLM_MAX_ATTEMPTS = 4
_LLM_BASE_BACKOFF_SECONDS = 1.5
_LLM_MAX_BACKOFF_SECONDS = 15.0

_DONE = "response.output_item.done"
_VALID_MODES = {"auto", "draft", "send"}


def _to_chat_messages(input_items: list[dict]) -> list[dict]:
    """Convert Responses-format input items to ChatCompletion messages.

    Prefers MLflow's official ``to_chat_completions_input`` helper; falls back to
    a minimal pass-through for plain ``role``/``content`` messages so the agent
    keeps working even if the helper is renamed across MLflow versions.
    """
    try:
        from mlflow.types.responses import to_chat_completions_input

        return to_chat_completions_input(input_items)
    except Exception:  # noqa: BLE001 — defensive fallback, see docstring
        messages: list[dict] = []
        for item in input_items:
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": item.get("role", "user"), "content": content})
            elif item.get("type") == "message" and isinstance(content, list):
                text = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
                messages.append({"role": item.get("role", "user"), "content": text})
        return messages or [{"role": "user", "content": ""}]


def _last_assistant_text(items: list[dict]) -> str:
    """Return the text of the most recent assistant message in a Responses/CC list."""
    for item in reversed(items):
        if item.get("role") == "assistant":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    return ""


class TechEngineerAgent(ResponsesAgent):
    """Unified session-announcement agent (auto / draft / send)."""

    def __init__(self) -> None:
        # Runtime config from env vars. Model Serving injects these from the
        # endpoint's env vars / secrets; notebooks set them before constructing.
        self._endpoint = os.environ.get(
            "DATABRICKS_FM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
        )
        self._index_name = os.environ.get(
            "VS_INDEX_NAME", "main.tech_engineer.sessions_vs_index"
        )
        self._log_table = os.environ.get(
            "LOG_TABLE_NAME", "main.tech_engineer.agent_action_log"
        )
        self._warehouse_id = os.environ.get("SQL_WAREHOUSE_ID", "")
        self._webhook = os.environ.get("WEBHOOK_URL", "")
        # Conversational state — used ONLY by the notebook convenience methods.
        # The serving path (predict/predict_stream) is stateless: it reads the
        # full history from request.input and never touches this.
        self._input_items: list[dict] = []

    # ==================================================================
    # ResponsesAgent interface (the deployable, stateless contract)
    # ==================================================================
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item for event in self.predict_stream(request) if event.type == _DONE
        ]
        return ResponsesAgentResponse(output=outputs)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        custom_inputs = request.custom_inputs or {}
        mode = custom_inputs.get("mode", "auto") if isinstance(custom_inputs, dict) else "auto"
        if mode not in _VALID_MODES:
            mode = "auto"
        input_items = [item.model_dump() for item in request.input]

        if mode == "send":
            yield from self._run_send(input_items, custom_inputs)
        else:
            yield from self._run_llm(input_items, mode)

    # ==================================================================
    # Mode: auto / draft  (LLM-driven tool loop)
    # ==================================================================
    def _run_llm(
        self, input_items: list[dict], mode: str
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if mode == "draft":
            system = DRAFT_SYSTEM_PROMPT + "\n\n" + UNTRUSTED_CONTENT_GUARD
            tools = SEARCH_TOOLS
        else:  # auto
            system = AUTONOMOUS_SYSTEM_PROMPT + "\n\n" + UNTRUSTED_CONTENT_GUARD
            tools = TOOL_DEFINITIONS

        messages: list[dict] = [{"role": "system", "content": system}]
        messages += _to_chat_messages(input_items)
        yield from self._tool_loop(messages, tools)

    def _tool_loop(
        self, messages: list[dict], tools: list[dict]
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        deadline = time.monotonic() + _WALL_TIME_SECONDS

        for _ in range(_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                yield self._text_event(
                    f"申し訳ありません。処理が{_WALL_TIME_SECONDS}秒を超えたため中断しました。"
                )
                return

            try:
                assistant = self._call_llm(messages, tools)
            except Exception as exc:  # noqa: BLE001 — degrade gracefully, never 500
                yield self._text_event(
                    f"申し訳ありません。言語モデルの呼び出しに失敗しました（{exc}）。"
                )
                return

            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []

            if not tool_calls:
                final_text = self._enforce_citations(assistant.get("content") or "", messages)
                yield self._text_event(final_text)
                return

            for tool_call in tool_calls:
                call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                fn = tool_call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)

                yield ResponsesAgentStreamEvent(
                    type=_DONE,
                    item=self.create_function_call_item(
                        id=f"fc_{uuid.uuid4().hex[:12]}",
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                    ),
                )

                result = self._dispatch_tool(name, arguments)

                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result}
                )
                yield ResponsesAgentStreamEvent(
                    type=_DONE,
                    item=self.create_function_call_output_item(
                        call_id=call_id, output=result
                    ),
                )

        yield self._text_event(
            "申し訳ありません。反復回数の上限に達したため、処理を完了できませんでした。"
        )

    # ==================================================================
    # Mode: send  (deterministic WYSIWYG post + audit log)
    # ==================================================================
    def _run_send(
        self, input_items: list[dict], custom_inputs: Any
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        approved = ""
        if isinstance(custom_inputs, dict):
            approved = custom_inputs.get("approved_text") or ""
        if not approved:
            approved = _last_assistant_text(input_items)

        if not approved.strip():
            yield self._text_event(
                "エラー: 承認された案内文が見つかりません。先に propose / refine を実行してください。"
            )
            return

        post_status, log_status = self._post_and_log(approved)
        yield self._text_event(f"投稿結果: {post_status}\nログ: {log_status}")

    def _post_and_log(self, approved: str) -> tuple[str, str]:
        """Post the approved announcement and write an audit record. Never raises."""
        try:
            post_status = post_to_channel(message=approved, webhook_url=self._webhook)
        except Exception as exc:  # noqa: BLE001
            post_status = f"ERROR: post_to_channel raised: {exc}"
        try:
            log_status = log_agent_action(
                action_name="post_to_channel",
                input_payload={"message": approved[:500]},
                output_payload={"result": post_status},
                table_name=self._log_table,
                status="SUCCESS" if post_status.startswith("SUCCESS") else "FAILURE",
                warehouse_id=self._warehouse_id or None,
            )
        except Exception as exc:  # noqa: BLE001
            log_status = f"ERROR: log_agent_action raised: {exc}"
        return post_status, log_status

    # ==================================================================
    # LLM call (with retry) and tool dispatch (never raises)
    # ==================================================================
    @mlflow.trace(span_type=SpanType.LLM)
    def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call the Databricks Foundation Model endpoint, retrying transient errors."""
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        last_exc: Exception | None = None

        for attempt in range(_LLM_MAX_ATTEMPTS):
            try:
                response = client.predict(
                    endpoint=self._endpoint,
                    inputs={
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "max_tokens": 2048,
                        "temperature": 0.2,
                    },
                )
                return response["choices"][0]["message"]
            except Exception as exc:  # noqa: BLE001 — retry a bounded number of times
                last_exc = exc
                if attempt == _LLM_MAX_ATTEMPTS - 1:
                    break
                time.sleep(
                    min(_LLM_BASE_BACKOFF_SECONDS * (2**attempt), _LLM_MAX_BACKOFF_SECONDS)
                )

        raise RuntimeError(f"LLM call failed after {_LLM_MAX_ATTEMPTS} attempts: {last_exc}")

    @mlflow.trace(span_type=SpanType.TOOL)
    def _dispatch_tool(self, name: str, arguments: str) -> str:
        """Route a tool call to its implementation, converting all failures to text."""
        try:
            args: dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return (
                f"ERROR: could not parse arguments for tool '{name}' as JSON ({exc}). "
                "Re-issue the call with valid JSON arguments."
            )

        try:
            if name == "search_knowledge_base":
                if "query" not in args:
                    return "ERROR: search_knowledge_base requires a 'query' argument."
                results = search_knowledge_base(
                    query=args["query"],
                    index_name=self._index_name,
                    num_results=args.get("num_results", 5),
                    similarity_threshold=args.get("similarity_threshold", 0.6),
                )
                return json.dumps(results, ensure_ascii=False)

            if name == "post_to_channel":
                if "message" not in args:
                    return "ERROR: post_to_channel requires a 'message' argument."
                return post_to_channel(message=args["message"], webhook_url=self._webhook)

            if name == "log_agent_action":
                return log_agent_action(
                    action_name=args.get("action_name", "unknown"),
                    input_payload=args.get("input_payload", {}),
                    output_payload=args.get("output_payload", {}),
                    table_name=self._log_table,
                    status=args.get("status", "SUCCESS"),
                    warehouse_id=self._warehouse_id or None,
                )

            return f"ERROR: Unknown tool '{name}'."
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the request
            return f"ERROR: tool '{name}' raised an exception: {exc}"

    # ==================================================================
    # Output helpers
    # ==================================================================
    def _text_event(self, text: str) -> ResponsesAgentStreamEvent:
        """Wrap assistant text as a Responses ``message`` output item."""
        return ResponsesAgentStreamEvent(
            type=_DONE,
            item=self.create_text_output_item(text=text, id=f"msg_{uuid.uuid4().hex[:12]}"),
        )

    def _enforce_citations(self, answer: str, messages: list[dict]) -> str:
        """Append ``[Source: …]`` tags if search was used but the LLM omitted them."""
        retrieval_used = any(
            tc.get("function", {}).get("name") == "search_knowledge_base"
            for msg in messages
            for tc in (msg.get("tool_calls") or [])
        )
        if not retrieval_used or "[Source:" in answer:
            return answer
        sources = self._sources_from_messages(messages)
        if sources:
            answer += "\n\n[Source: " + ", ".join(sources) + "]"
        return answer

    @staticmethod
    def _sources_from_messages(messages: list[dict]) -> list[str]:
        """Gather unique source_file values from search-result tool messages."""
        sources: list[str] = []
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            try:
                parsed = json.loads(msg.get("content") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict):
                        src = (row.get("metadata") or {}).get("source_file")
                        if src and src not in sources:
                            sources.append(src)
        return sources

    @staticmethod
    def _final_text(output_items: list) -> str:
        """Concatenate assistant `message` text from output items (dict or typed)."""
        texts: list[str] = []
        for item in output_items:
            d = item if isinstance(item, dict) else item.model_dump()
            if d.get("type") == "message":
                for part in d.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        texts.append(part.get("text", ""))
        return "\n".join(texts)

    @staticmethod
    def _sources_from_output(output_items: list) -> list[str]:
        """Gather source_file values from function_call_output items (search results)."""
        sources: list[str] = []
        for item in output_items:
            d = item if isinstance(item, dict) else item.model_dump()
            if d.get("type") != "function_call_output":
                continue
            try:
                parsed = json.loads(d.get("output") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict):
                        src = (row.get("metadata") or {}).get("source_file")
                        if src and src not in sources:
                            sources.append(src)
        return sources

    # ==================================================================
    # Notebook convenience API (stateful) — drives the interactive demo.
    # Each method calls the same predict() path as production.
    # ==================================================================
    def _request(self, input_items: list[dict], mode: str) -> ResponsesAgentRequest:
        return ResponsesAgentRequest(input=input_items, custom_inputs={"mode": mode})

    def reset(self) -> None:
        """Clear the conversational draft state (start a fresh session)."""
        self._input_items = []

    def current_draft(self) -> str | None:
        draft = _last_assistant_text(self._input_items)
        return draft or None

    def autonomous(self, request: str) -> str:
        """Hands-off run: search → draft → post → log. Returns the final message."""
        out = self.predict(self._request([{"role": "user", "content": request}], "auto"))
        return self._final_text(out.output)

    def propose(self, request: str) -> dict[str, Any]:
        """Search + draft a complete announcement (posts nothing). Starts a session."""
        self._input_items = [{"role": "user", "content": request}]
        out = self.predict(self._request(self._input_items, "draft"))
        draft = self._final_text(out.output)
        self._input_items.append({"role": "assistant", "content": draft})
        return {"steps": WORKFLOW_STEPS, "draft": draft, "sources": self._sources_from_output(out.output)}

    def refine(self, feedback: str) -> dict[str, Any]:
        """Revise the current draft from natural-language feedback (repeatable)."""
        if not self._input_items:
            return {"draft": None, "message": "先に propose(...) を実行してください。"}
        if not feedback or not feedback.strip():
            return {"draft": self.current_draft(), "message": "修正指示が空のため、変更はありません。"}
        self._input_items.append({"role": "user", "content": feedback})
        out = self.predict(self._request(self._input_items, "draft"))
        draft = self._final_text(out.output)
        self._input_items.append({"role": "assistant", "content": draft})
        return {"draft": draft, "sources": self._sources_from_output(out.output)}

    def send(self) -> dict[str, str]:
        """Post the current approved draft and write an audit log."""
        approved = self.current_draft()
        if not approved:
            return {
                "post_status": "ERROR: no draft to send — run propose() first.",
                "log_status": "skipped",
            }
        post_status, log_status = self._post_and_log(approved)
        return {"post_status": post_status, "log_status": log_status}
