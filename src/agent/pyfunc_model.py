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

from src.agent.prompts import (
    DRAFT_SYSTEM_PROMPT,
    SEARCH_TOOLS,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
)
from src.tools.logger import log_agent_action
from src.tools.messaging import post_to_channel
from src.tools.search import search_knowledge_base

_MAX_ITERATIONS = 5
_WALL_TIME_SECONDS = 150


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper that orchestrates the agentic tool-use loop.

    The model accepts a user message, enters a ReAct-style tool-use loop
    against the Databricks Foundation Model API, and continues until the
    LLM returns a final text answer (no more tool calls).

    Auth note: the LLM call goes through ``mlflow.deployments`` and Vector
    Search / SQL through the Databricks SDK. Both pick up M2M OAuth that
    Model Serving injects from the ``resources`` declaration — so no
    DATABRICKS_TOKEN secret is required at serving time.
    """

    def __init__(self) -> None:
        self._endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
        self._webhook: str = ""
        self._index_name: str = "main.tech_engineer.sessions_vs_index"
        self._log_table: str = "main.tech_engineer.agent_action_log"
        self._warehouse_id: str = ""

    def configure(
        self,
        endpoint: str | None = None,
        index_name: str | None = None,
        log_table: str | None = None,
        warehouse_id: str | None = None,
        webhook: str | None = None,
    ) -> "AgentModel":
        """Set configuration explicitly before logging or for direct notebook runs.

        Returns self so the call can be chained into log_model / resources.
        """
        if endpoint:
            self._endpoint = endpoint
        if index_name:
            self._index_name = index_name
        if log_table:
            self._log_table = log_table
        if warehouse_id is not None:
            self._warehouse_id = warehouse_id
        if webhook is not None:
            self._webhook = webhook
        return self

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load runtime configuration from artifact + env vars; crash on misconfiguration."""
        self._endpoint = os.environ.get("DATABRICKS_FM_ENDPOINT", self._endpoint)
        self._webhook = os.environ.get("WEBHOOK_URL", self._webhook)
        self._index_name = os.environ.get("VS_INDEX_NAME", self._index_name)
        self._log_table = os.environ.get("LOG_TABLE_NAME", self._log_table)
        self._warehouse_id = os.environ.get("SQL_WAREHOUSE_ID", self._warehouse_id)

        # Only read the artifact if it resolves to a real local file. During
        # signature inference at log time the path may be unresolved; we must
        # not crash there. At serving time MLflow resolves it to a local path.
        config_path = (context.artifacts or {}).get("agent_config")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
            except Exception as exc:
                raise RuntimeError(
                    f"load_context: failed to parse agent_config artifact: {exc}"
                ) from exc
            self._endpoint = cfg.get("llm_endpoint", self._endpoint)
            self._index_name = cfg.get("vs_index_name", self._index_name)
            self._log_table = cfg.get("log_table_name", self._log_table)
            self._warehouse_id = cfg.get("warehouse_id", self._warehouse_id)

        assert self._endpoint, "load_context: DATABRICKS_FM_ENDPOINT is not set"
        assert self._index_name, "load_context: VS_INDEX_NAME is not set"

    @property
    def resources(self) -> list:
        """Declare UC resource dependencies for M2M OAuth injection in Model Serving."""
        from mlflow.models.resources import (
            DatabricksServingEndpoint,
            DatabricksSQLWarehouse,
            DatabricksVectorSearchIndex,
        )
        res: list = [
            DatabricksVectorSearchIndex(index_name=self._index_name),
            DatabricksServingEndpoint(endpoint_name=self._endpoint),
        ]
        if self._warehouse_id:
            res.append(DatabricksSQLWarehouse(warehouse_id=self._warehouse_id))
        return res

    def predict(self, context, model_input):
        """Run the autonomous agent for each request and return the answers.

        (No type hints on the signature: MLflow tries to derive a schema from them
        and warns unless they are list[...]; we supply an explicit signature instead.)

        Uses a simple, serving-robust contract: a ``query`` string in, a response
        string out (row-aligned). This is the shape MLflow's scoring server handles
        most reliably (``{"dataframe_records": [{"query": "..."}]}`` ->
        ``{"predictions": ["..."]}``).

        Args:
            context: MLflow context object (unused at inference time).
            model_input: One of:
                - ``pd.DataFrame`` with a ``"query"`` column (serving path)
                - ``dict`` ``{"query": "..."}`` or ``{"query": ["...", ...]}``
                - ``dict`` ``{"messages": [...]}`` (chat-style; last user msg used)
                - ``list`` of strings, or a single ``str``

        Returns:
            A list of response strings, one per input row.
        """
        queries = self._to_queries(model_input)
        return [self._answer_one(q) for q in queries]

    def _to_queries(self, model_input: Any) -> list[str]:
        """Normalise any supported input shape into a list of query strings."""
        if isinstance(model_input, pd.DataFrame):
            col = "query" if "query" in model_input.columns else model_input.columns[0]
            return [str(v) for v in model_input[col].tolist()]
        if isinstance(model_input, str):
            return [model_input]
        if isinstance(model_input, list):
            if model_input and isinstance(model_input[0], dict):
                return [self._last_user_content(model_input)]
            return [str(v) for v in model_input]
        if isinstance(model_input, dict):
            if "query" in model_input:
                v = model_input["query"]
                return [str(x) for x in v] if isinstance(v, list) else [str(v)]
            if "messages" in model_input:
                return [self._last_user_content(model_input["messages"])]
        raise ValueError(f"Unsupported model_input type: {type(model_input)}")

    @staticmethod
    def _last_user_content(messages: list[dict]) -> str:
        """Extract the most recent user message content from a chat list."""
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return messages[-1].get("content", "") if messages else ""

    def _answer_one(self, query: str) -> str:
        """Run the full autonomous tool loop for a single query string."""
        conversation: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        return self._run_tool_loop(conversation, TOOL_DEFINITIONS)

    # ------------------------------------------------------------------
    # Human-in-the-loop API (used by the demo notebook)
    # ------------------------------------------------------------------
    def draft_announcement(self, user_request: str) -> dict[str, Any]:
        """Draft an announcement WITHOUT posting it.

        The agent searches the knowledge base and writes the announcement, but the
        posting tool is withheld — nothing is sent. The returned text is exactly
        what ``send_announcement`` will post once a human approves it.

        Returns:
            ``{"message": <draft text>, "sources": [<source_file>, ...]}``.
        """
        conversation: list[dict] = [
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]
        message = self._run_tool_loop(conversation, SEARCH_TOOLS)
        return {"message": message, "sources": self._collect_sources(conversation)}

    def send_announcement(self, message: str) -> dict[str, str]:
        """Post an already-approved announcement and log the action.

        Call this only after a human has reviewed the draft from
        ``draft_announcement``. Posts exactly the text passed in (WYSIWYG).
        """
        post_status = post_to_channel(message=message, webhook_url=self._webhook)
        log_status = log_agent_action(
            action_name="post_to_channel",
            input_payload={"message": message[:500]},
            output_payload={"result": post_status},
            table_name=self._log_table,
            status="SUCCESS" if post_status.startswith("SUCCESS") else "FAILURE",
            warehouse_id=self._warehouse_id or None,
        )
        return {"post_status": post_status, "log_status": log_status}

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------
    def _run_tool_loop(self, conversation: list[dict], tools: list[dict]) -> str:
        """Execute the ReAct tool-use loop until the LLM stops calling tools."""
        deadline = time.monotonic() + _WALL_TIME_SECONDS

        for iteration in range(_MAX_ITERATIONS):
            if time.monotonic() > deadline:
                return f"ERROR: Agent timed out after {_WALL_TIME_SECONDS} seconds."

            response = self._call_llm(conversation, tools)
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
                        "Continue following your instructions. When the announcement is "
                        "ready, complete the required steps and give your final response.]"
                    ),
                }
            )

        return "ERROR: Agent exceeded maximum iteration limit."

    def _collect_sources(self, conversation: list[dict]) -> list[str]:
        """Gather unique source_file values from search results in the conversation."""
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
        return sources

    def _enforce_citations(self, answer: str, conversation: list[dict]) -> str:
        """Append [Source:] tags if search was used but the LLM omitted them."""
        retrieval_was_used = any(
            tc.get("function", {}).get("name") == "search_knowledge_base"
            for msg in conversation
            for tc in (msg.get("tool_calls") or [])
        )
        if not retrieval_was_used or "[Source:" in answer:
            return answer

        sources = self._collect_sources(conversation)
        if sources:
            answer += "\n\n[Source: " + ", ".join(sources) + "]"
        return answer

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
        """Call the Databricks Foundation Model endpoint via the deployments client.

        Using ``get_deploy_client("databricks")`` (rather than a raw token POST)
        means auth is handled by the platform: notebook credentials during
        development, and injected M2M OAuth at serving time.
        """
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        return client.predict(
            endpoint=self._endpoint,
            inputs={
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 2048,
                "temperature": 0.2,
            },
        )

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

        if name == "post_to_channel":
            return post_to_channel(
                message=args["message"],
                webhook_url=self._webhook,
            )

        if name == "log_agent_action":
            return log_agent_action(
                action_name=args["action_name"],
                input_payload=args.get("input_payload", {}),
                output_payload=args.get("output_payload", {}),
                table_name=self._log_table,
                status=args.get("status", "SUCCESS"),
                warehouse_id=self._warehouse_id or None,
            )

        return f"ERROR: Unknown tool '{name}'."
