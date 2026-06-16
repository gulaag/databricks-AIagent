"""
Unity Catalog Delta table action logger.

Writes a structured audit record for every tool execution performed by the
agent. Records land in a UC-governed Delta table and are queryable via SQL
for compliance reporting and MLflow experiment cross-referencing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql import Row


@mlflow.trace(name="log_agent_action", span_type="TOOL")
def log_agent_action(
    action_name: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    table_name: str,
    run_id: str | None = None,
    status: str = "SUCCESS",
) -> str:
    """Append a structured action log record to a Unity Catalog Delta table.

    Args:
        action_name: Human-readable name of the tool/action being logged,
            e.g. ``"post_to_teams"`` or ``"search_knowledge_base"``.
        input_payload: Dict of inputs passed to the action.
        output_payload: Dict of outputs returned by the action.
        table_name: Fully-qualified UC table name, e.g.
            ``catalog.schema.agent_action_log``.
        run_id: Optional MLflow run ID for cross-referencing traces.
        status: Execution status string — ``"SUCCESS"`` or ``"FAILURE"``.

    Returns:
        A status string confirming the write or describing the failure.
    """
    try:
        spark = SparkSession.builder.getOrCreate()

        record = Row(
            logged_at=datetime.now(timezone.utc).isoformat(),
            mlflow_run_id=run_id or mlflow.active_run().info.run_id
            if mlflow.active_run()
            else None,
            action_name=action_name,
            status=status,
            input_json=json.dumps(input_payload, ensure_ascii=False),
            output_json=json.dumps(output_payload, ensure_ascii=False),
        )

        df = spark.createDataFrame([record])
        df.write.format("delta").mode("append").saveAsTable(table_name)

        return f"SUCCESS: Action '{action_name}' logged to {table_name}."

    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Failed to log action '{action_name}': {str(exc)}"
