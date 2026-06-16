"""
MLflow PyFunc model orchestrator for the Tech Engineer Study Group agent.

This module defines the AgentModel class, which is the deployable unit.
It is registered to Unity Catalog via mlflow.pyfunc.log_model and served
through Databricks Model Serving.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import requests

from src.agent.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
from src.tools.logger import log_agent_action
from src.tools.search import search_knowledge_base
from src.tools.teams import post_to_teams

_MAX_ITERATIONS = 5
_WALL_TIME_SECONDS = 150


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper that orchestrates the agentic tool-use loop.

    The model accepts a user message, enters a ReAct-style tool-use loop
    against the Databricks Foundation Model API, and continues until the
    LLM returns a final text answer (no more tool calls).
    """

    def __init__(self) -> None:
        self._endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
        self._teams_webhook: str = ""
        self._index_name: str = "main.tech_engineer.sessions_vs_index"
        self._log_table: str = "main.tech_engineer.agent_action_log"

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load runtime configuration from artifact + env vars; crash on misconfiguration."""
        self._endpoint = os.environ.get("DATABRICKS_FM_ENDPOINT", self._endpoint)
        self._teams_webhook = os.environ.get("TEAMS_WEBHOOK_URL", "")
        self._index_name = os.environ.get("VS_INDEX_NAME", self._index_name)
        self._log_table = os.environ.get("LOG_TABLE_NAME", self._log_table)

        try:
            config_path = (context.artifacts or {}).get("agent_config")
            if config_path:
                with open(config_path) as f:
                    cfg = json.load(f)
                self._endpoint = cfg.get("llm_endpoint", self._endpoint)
                self._index_name = cfg.get("vs_index_name", self._index_name)
                self._log_table = cfg.get("log_table_name", self._log_table)
        except Exception as exc:
            raise RuntimeError(
                f"load_context: failed to read agent_config artifact: {exc}"
            ) from exc

        assert self._endpoint, "load_context: DATABRICKS_FM_ENDPOINT is not set"
        assert self._index_name, "load_context: VS_INDEX_NAME is not set"

    @property
    def resources(self) -> list:
        """Declare UC resource dependencies for M2M OAuth injection in Model Serving."""
        from mlflow.models.resources import (
            DatabricksServingEndpoint,
            DatabricksVectorSearchIndex,
        )
        return [
            DatabricksVectorSearchIndex(index_name=self._index_name),
            DatabricksServingEndpoint(endpoint_name=self._endpoint),
        ]

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame | dict[str, Any] | list,
    ) -> dict[str, Any]:
        """Run the agentic tool-use loop for a single user request.

        Args:
            context: MLflow context object (unused at inference time).
            model_input: Accepts three shapes:
                - ``pd.DataFrame`` with a ``"messages"`` column (serving path)
                - ``dict`` with ``"messages"`` key (notebook testing path)
                - ``list`` of message dicts (Databricks AI Playground path)

        Returns:
            OpenAI-compatible envelope:
            ``{"choices": [{"message": {"role": "assistant", "content": str}}]}``.
        """
        if isinstance(model_input, pd.DataFrame):
            messages: list[dict] = model_input["messages"].iloc[0]
        elif isinstance(model_input, list):
            messages = model_input
        elif isinstance(model_input, dict):
            messages = model_input.get("messages", [])
            if not messages:
                raw = model_input.get("input", "")
                if isinstance(raw, str) and raw:
                    messages = [{"role": "user", "content": raw}]
        else:
            raise ValueError(f"Unsupported model_input type: {type(model_input)}")

        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ]

        with mlflow.start_run(nested=True):
            final_answer = self._run_tool_loop(conversation)

        return {"choices": [{"message": {"role": "assistant", "content": final_answer}}]}

    def _run_tool_loop(self, conversation: list[dict]) -> str:
        """Execute the ReAct tool-use loop until the LLM stops calling tools."""
        deadline = time.monotonic() + _WALL_TIME_SECONDS

        for iteration in range(_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                return f"ERROR: Agent timed out after {_WALL_TIME_SECONDS} seconds."

            response = self._call_llm(conversation)
            message = response["choices"][0]["message"]
            conversation.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return self._enforce_citations(message.get("content", ""), conversation)

            for tc in tool_calls:
                tool_result = self._dispatch_tool(tc)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"[System: Iteration {iteration + 1} of {_MAX_ITERATIONS}. "
                        "Continue following the execution order in your instructions. "
                        "If you have retrieved context and drafted the announcement, "
                        "call post_to_teams next, then log_agent_action.]"
                    ),
                }
            )

        return "ERROR: Agent exceeded maximum iteration limit."

    def _enforce_citations(self, answer: str, conversation: list[dict]) -> str:
        """Append [Source:] tags if search was used but the LLM omitted them."""
        retrieval_was_used = any(
            tc.get("function", {}).get("name") == "search_knowledge_base"
            for msg in conversation
            for tc in (msg.get("tool_calls") or [])
        )
        if not retrieval_was_used or "[Source:" in answer:
            return answer

        sources: list[str] = []
        for msg in conversation:
            if msg.get("role") == "tool":
                try:
                    tool_result = json.loads(msg["content"])
                    if isinstance(tool_result, list):
                        for r in tool_result:
                            src = r.get("metadata", {}).get("source_file")
                            if src and src not in sources:
                                sources.append(src)
                except (json.JSONDecodeError, AttributeError):
                    pass

        if sources:
            answer += "\n\n[Source: " + ", ".join(sources) + "]"
        return answer

    def _call_llm(self, messages: list[dict]) -> dict:
        """POST to the Databricks Foundation Model API chat completions endpoint."""
        token = os.environ.get("DATABRICKS_TOKEN", "")
        host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
        url = f"{host}/serving-endpoints/{self._endpoint}/invocations"

        payload = {
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "max_tokens": 2048,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _dispatch_tool(self, tool_call: dict) -> Any:
        """Route a tool_call object to the correct Python function."""
        name = tool_call["function"]["name"]
        args: dict = json.loads(tool_call["function"].get("arguments", "{}"))

        if name == "search_knowledge_base":
            return search_knowledge_base(
                query=args["query"],
                index_name=self._index_name,
                num_results=args.get("num_results", 5),
                similarity_threshold=args.get("similarity_threshold", 0.6),
            )

        if name == "post_to_teams":
            return post_to_teams(
                agenda_content=args["agenda_content"],
                webhook_url=self._teams_webhook,
            )

        if name == "log_agent_action":
            return log_agent_action(
                action_name=args["action_name"],
                input_payload=args.get("input_payload", {}),
                output_payload=args.get("output_payload", {}),
                table_name=self._log_table,
                status=args.get("status", "SUCCESS"),
            )

        return f"ERROR: Unknown tool '{name}'."
