"""
MLflow PyFunc model orchestrator for the Tech Engineer Study Group agent.

This module defines the AgentModel class, which is the deployable unit.
It is registered to Unity Catalog via mlflow.pyfunc.log_model and served
through Databricks Model Serving.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
import requests

from src.agent.prompts import SYSTEM_PROMPT, TOOL_DEFINITIONS
from src.tools.logger import log_agent_action
from src.tools.search import search_knowledge_base
from src.tools.teams import post_to_teams


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper that orchestrates the agentic tool-use loop.

    The model accepts a user message, enters a ReAct-style tool-use loop
    against the Databricks Foundation Model API, and continues until the
    LLM returns a final text answer (no more tool calls).
    """

    def __init__(self) -> None:
        self._endpoint: str | None = None
        self._teams_webhook: str | None = None
        self._index_name: str | None = None
        self._log_table: str | None = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load runtime configuration from MLflow model artifacts or UC secrets."""
        self._endpoint = os.environ.get(
            "DATABRICKS_FM_ENDPOINT",
            "databricks-meta-llama-3-3-70b-instruct",
        )
        self._teams_webhook = os.environ.get("TEAMS_WEBHOOK_URL", "")
        self._index_name = os.environ.get(
            "VS_INDEX_NAME",
            "main.tech_engineer.sessions_vs_index",
        )
        self._log_table = os.environ.get(
            "LOG_TABLE_NAME",
            "main.tech_engineer.agent_action_log",
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame | dict[str, Any],
    ) -> str:
        """Run the agentic tool-use loop for a single user request.

        Args:
            context: MLflow context object (unused at inference time).
            model_input: Either a DataFrame with a ``"messages"`` column or a
                dict with key ``"messages"`` containing a list of chat messages.

        Returns:
            The final assistant response as a plain string.
        """
        if isinstance(model_input, pd.DataFrame):
            messages: list[dict] = model_input["messages"].iloc[0]
        else:
            messages = model_input.get("messages", [])

        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ]

        with mlflow.start_run(nested=True):
            return self._run_tool_loop(conversation)

    def _run_tool_loop(self, conversation: list[dict]) -> str:
        """Execute the ReAct tool-use loop until the LLM stops calling tools."""
        max_iterations = 10

        for _ in range(max_iterations):
            response = self._call_llm(conversation)
            message = response["choices"][0]["message"]
            conversation.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return message.get("content", "")

            for tc in tool_calls:
                tool_result = self._dispatch_tool(tc)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        return "ERROR: Agent exceeded maximum iteration limit."

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
