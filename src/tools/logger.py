"""
Unity Catalog Delta table action logger.

Writes a structured audit record for every tool execution performed by the
agent. Records land in a UC-governed Delta table and are queryable via SQL
for compliance reporting and MLflow experiment cross-referencing.

Two write paths:
  - SQL Statement Execution API (when a warehouse_id is provided). This is the
    only path that works inside a Model Serving container, which has no Spark
    session. It also self-provisions the target table on first write.
  - Spark ``saveAsTable`` (fallback for notebook / cluster execution, where no
    warehouse id is configured).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import mlflow

_TABLE_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
  logged_at     STRING,
  mlflow_run_id STRING,
  action_name   STRING,
  status        STRING,
  input_json    STRING,
  output_json   STRING
) USING DELTA
"""


def _current_run_id(explicit: str | None) -> str | None:
    """Resolve the MLflow run id, preferring an explicit value."""
    if explicit:
        return explicit
    active = mlflow.active_run()
    return active.info.run_id if active else None


def _log_via_sql(table_name: str, warehouse_id: str, record: dict[str, Any]) -> str:
    """Append a record using the Databricks SQL Statement Execution API.

    Parameterised statements keep JSON payloads safe from SQL injection and
    escaping issues. Works at serving time via injected M2M OAuth.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementParameterListItem

    w = WorkspaceClient()

    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=_TABLE_DDL_TEMPLATE.format(table=table_name),
        wait_timeout="30s",
    )

    insert_sql = (
        f"INSERT INTO {table_name} "
        "(logged_at, mlflow_run_id, action_name, status, input_json, output_json) "
        "VALUES (:logged_at, :mlflow_run_id, :action_name, :status, :input_json, :output_json)"
    )
    params = [
        StatementParameterListItem(name=key, value=value)
        for key, value in record.items()
    ]
    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=insert_sql,
        parameters=params,
        wait_timeout="30s",
    )
    return f"SUCCESS: Action '{record['action_name']}' logged to {table_name} (SQL)."


def _log_via_spark(table_name: str, record: dict[str, Any]) -> str:
    """Append a record using Spark (notebook / cluster fallback)."""
    from pyspark.sql import Row, SparkSession

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame([Row(**record)])
    df.write.format("delta").mode("append").saveAsTable(table_name)
    return f"SUCCESS: Action '{record['action_name']}' logged to {table_name} (Spark)."


@mlflow.trace(name="log_agent_action", span_type="TOOL")
def log_agent_action(
    action_name: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    table_name: str,
    run_id: str | None = None,
    status: str = "SUCCESS",
    warehouse_id: str | None = None,
) -> str:
    """Append a structured action log record to a Unity Catalog Delta table.

    Args:
        action_name: Name of the tool/action being logged, e.g. ``"post_to_channel"``.
        input_payload: Dict of inputs passed to the action.
        output_payload: Dict of outputs returned by the action.
        table_name: Fully-qualified UC table name, e.g.
            ``catalog.schema.agent_action_log``.
        run_id: Optional MLflow run ID for cross-referencing traces.
        status: Execution status — ``"SUCCESS"`` or ``"FAILURE"``.
        warehouse_id: SQL warehouse to use for the statement-execution write path.
            When provided (e.g. at serving time), it is used; otherwise the Spark
            path is used (notebook / cluster).

    Returns:
        A status string confirming the write or describing the failure. Never
        raises, so a logging failure cannot break the agent's main flow.
    """
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": _current_run_id(run_id),
        "action_name": action_name,
        "status": status,
        "input_json": json.dumps(input_payload, ensure_ascii=False),
        "output_json": json.dumps(output_payload, ensure_ascii=False),
    }

    try:
        if warehouse_id:
            return _log_via_sql(table_name, warehouse_id, record)
        return _log_via_spark(table_name, record)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: Failed to log action '{action_name}': {str(exc)}"
