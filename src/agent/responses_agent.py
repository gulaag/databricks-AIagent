"""
MLflow ``ResponsesAgent`` implementation of the Tech Engineer action agent.

This is the *Databricks-recommended* authoring interface
(``mlflow.pyfunc.ResponsesAgent``). Building the agent this way gives the
deployed endpoint out-of-the-box compatibility with the AI Playground, Agent
Evaluation, and Agent Monitoring, a standard streaming contract, and an
automatically-inferred model signature.

Why we wrap a ChatCompletion loop
---------------------------------
The canonical ``ResponsesAgent`` example calls the OpenAI *Responses* API. Our
LLM is a Databricks Foundation Model (Llama-3.3-70B) served over the
*ChatCompletion* API, which is the contract we already run reliably in
production. So we keep that proven tool-use loop and wrap it in the
``ResponsesAgent`` interface, converting message formats at the boundary. This
is exactly the "wrap any existing agent" pattern the interface is designed for.

Hardening over a naive loop
---------------------------
  - Every tool call runs inside try/except; a bad tool call (malformed JSON,
    missing arg, tool exception) is fed back to the LLM as an error result so it
    can recover, instead of 500-ing the whole request.
  - The LLM call is retried with exponential backoff on transient errors.
  - Retrieved knowledge-base content is treated as untrusted data (indirect
    prompt-injection guard, see ``UNTRUSTED_CONTENT_GUARD``).
  - A wall-clock deadline and a max-iteration cap bound every request; on either
    limit the agent returns a clean message rather than raising.

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

from src.agent.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS, UNTRUSTED_CONTENT_GUARD
from src.tools.logger import log_agent_action
from src.tools.messaging import post_to_channel
from src.tools.search import search_knowledge_base

_MAX_ITERATIONS = 6
_WALL_TIME_SECONDS = 150
_LLM_MAX_ATTEMPTS = 4
_LLM_BASE_BACKOFF_SECONDS = 1.5
_LLM_MAX_BACKOFF_SECONDS = 15.0

# The autonomous serving agent gets the full system prompt plus the untrusted-
# content guard, since it feeds retrieved documents back into its own context.
_RESPONSES_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + UNTRUSTED_CONTENT_GUARD

_DONE = "response.output_item.done"


def _to_chat_messages(input_items: list[dict]) -> list[dict]:
    """Convert Responses-format input items to ChatCompletion messages.

    Prefers MLflow's official ``to_chat_completions_input`` helper. Falls back to
    a minimal pass-through for plain ``role``/``content`` messages (the common
    single-turn case) so the agent keeps working even if the helper is renamed
    across MLflow versions.
    """
    try:
        from mlflow.types.responses import to_chat_completions_input

        return to_chat_completions_input(input_items)
    except Exception:  # noqa: BLE001 — defensive fallback, see docstring
        messages: list[dict] = []
        for item in input_items:
            content = item.get("content")
            if isinstance(content, str):
                messages.append(
                    {"role": item.get("role", "user"), "content": content}
                )
            elif item.get("type") == "message" and isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
                messages.append({"role": item.get("role", "user"), "content": text})
        return messages or [{"role": "user", "content": ""}]


class TechEngineerResponsesAgent(ResponsesAgent):
    """Autonomous session-announcement agent, authored as an MLflow ResponsesAgent."""

    def __init__(self) -> None:
        # Runtime config comes from environment variables. Model Serving injects
        # these from the endpoint's env vars / secrets; notebooks set them before
        # constructing the agent for a local smoke test.
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

    # ------------------------------------------------------------------
    # ResponsesAgent interface
    # ------------------------------------------------------------------
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
        messages: list[dict] = [{"role": "system", "content": _RESPONSES_SYSTEM_PROMPT}]
        messages += _to_chat_messages([item.model_dump() for item in request.input])
        yield from self._run_tool_loop(messages)

    # ------------------------------------------------------------------
    # Tool-use loop (ChatCompletion under the hood)
    # ------------------------------------------------------------------
    def _run_tool_loop(
        self, messages: list[dict]
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        deadline = time.monotonic() + _WALL_TIME_SECONDS

        for _ in range(_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                yield self._text_event(
                    f"申し訳ありません。処理が{_WALL_TIME_SECONDS}秒を超えたため中断しました。"
                )
                return

            try:
                assistant = self._call_llm(messages)
            except Exception as exc:  # noqa: BLE001 — degrade gracefully, never 500
                yield self._text_event(
                    f"申し訳ありません。言語モデルの呼び出しに失敗しました（{exc}）。"
                )
                return

            messages.append(assistant)
            tool_calls = assistant.get("tool_calls") or []

            if not tool_calls:
                final_text = self._enforce_citations(
                    assistant.get("content") or "", messages
                )
                yield self._text_event(final_text)
                return

            for tool_call in tool_calls:
                call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                fn = tool_call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                arguments = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)

                # Surface the model's tool *intent* as a Responses output item.
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

                # Feed the result back to the LLM (ChatCompletion tool message)…
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "content": result}
                )
                # …and surface it as a Responses output item.
                yield ResponsesAgentStreamEvent(
                    type=_DONE,
                    item=self.create_function_call_output_item(
                        call_id=call_id, output=result
                    ),
                )

        yield self._text_event(
            "申し訳ありません。反復回数の上限に達したため、処理を完了できませんでした。"
        )

    # ------------------------------------------------------------------
    # LLM call with retry
    # ------------------------------------------------------------------
    @mlflow.trace(span_type=SpanType.LLM)
    def _call_llm(self, messages: list[dict]) -> dict:
        """Call the Databricks Foundation Model endpoint, retrying transient errors.

        Returns the assistant message dict (``{"role", "content", "tool_calls"?}``).
        Raises only after exhausting all attempts.
        """
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        last_exc: Exception | None = None

        for attempt in range(_LLM_MAX_ATTEMPTS):
            try:
                response = client.predict(
                    endpoint=self._endpoint,
                    inputs={
                        "messages": messages,
                        "tools": TOOL_DEFINITIONS,
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
                backoff = min(
                    _LLM_BASE_BACKOFF_SECONDS * (2**attempt), _LLM_MAX_BACKOFF_SECONDS
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"LLM call failed after {_LLM_MAX_ATTEMPTS} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Tool dispatch — ALWAYS returns a string, NEVER raises
    # ------------------------------------------------------------------
    @mlflow.trace(span_type=SpanType.TOOL)
    def _dispatch_tool(self, name: str, arguments: str) -> str:
        """Route a tool call to its implementation, converting all failures to text.

        Returning an error string (rather than raising) lets the LLM see what
        went wrong and self-correct on the next turn — the request never crashes.
        """
        try:
            args: dict[str, Any] = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return (
                f"ERROR: could not parse arguments for tool '{name}' as JSON "
                f"({exc}). Re-issue the call with valid JSON arguments."
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _text_event(self, text: str) -> ResponsesAgentStreamEvent:
        """Wrap final assistant text as a Responses ``message`` output item."""
        return ResponsesAgentStreamEvent(
            type=_DONE,
            item=self.create_text_output_item(
                text=text, id=f"msg_{uuid.uuid4().hex[:12]}"
            ),
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
        sources = self._collect_sources(messages)
        if sources:
            answer += "\n\n[Source: " + ", ".join(sources) + "]"
        return answer

    @staticmethod
    def _collect_sources(messages: list[dict]) -> list[str]:
        """Gather unique ``source_file`` values from search-result tool messages."""
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
                    if not isinstance(row, dict):
                        continue
                    src = (row.get("metadata") or {}).get("source_file")
                    if src and src not in sources:
                        sources.append(src)
        return sources
