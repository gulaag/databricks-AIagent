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


def _current_run_id(explicit: str | None) -> str | None:
    """Resolve the MLflow run id, preferring an explicit value."""
    if explicit:
        return explicit
    active = mlflow.active_run()
    return active.info.run_id if active else None


def _run_sql_to_completion(w, warehouse_id, statement, parameters=None, poll_timeout_s=120):
    """Execute a statement and BLOCK until it reaches a terminal state.

    Crucial for correctness: ``execute_statement`` with a ``wait_timeout`` returns
    a non-terminal (PENDING/RUNNING) response when a cold warehouse is still
    starting, and returns a FAILED response *without raising* on e.g. a permission
    error. Reporting success on either is wrong — the row never lands. So we poll
    to a terminal state and raise unless it actually SUCCEEDED, which lets the
    caller surface a truthful FAILURE instead of a false SUCCESS.
    """
    import time as _time

    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        parameters=parameters,
        wait_timeout="30s",
    )
    deadline = _time.time() + poll_timeout_s
    while resp.status and resp.status.state and resp.status.state.value in ("PENDING", "RUNNING"):
        if _time.time() > deadline:
            raise RuntimeError(
                f"SQL statement did not finish within {poll_timeout_s}s "
                f"(state={resp.status.state.value})"
            )
        _time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)

    state = resp.status.state.value if (resp.status and resp.status.state) else "UNKNOWN"
    if state != "SUCCEEDED":
        detail = (
            resp.status.error.message
            if (resp.status and resp.status.error)
            else f"terminal state {state}"
        )
        raise RuntimeError(f"SQL statement failed ({state}): {detail}")
    return resp


def _log_via_sql(table_name: str, warehouse_id: str, record: dict[str, Any]) -> str:
    """Append a record using the Databricks SQL Statement Execution API.

    Parameterised statements keep JSON payloads safe from SQL injection and
    escaping issues. Works at serving time via injected M2M OAuth — provided the
    serving endpoint's service principal has been granted MODIFY on the table.
    Each statement is run to a verified terminal state, so a permission error or
    a cold-warehouse timeout is surfaced as a real FAILURE, not a false success.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementParameterListItem

    w = WorkspaceClient()

    # Deliberately NO "CREATE TABLE IF NOT EXISTS" here. The audit table is
    # provisioned once by the notebook / setup path; requiring CREATE TABLE would
    # force a broad schema-level grant on the serving service principal. This path
    # therefore needs only USE CATALOG + USE SCHEMA + MODIFY on the table.
    insert_sql = (
        f"INSERT INTO {table_name} "
        "(logged_at, mlflow_run_id, action_name, status, input_json, output_json) "
        "VALUES (:logged_at, :mlflow_run_id, :action_name, :status, :input_json, :output_json)"
    )
    params = [
        StatementParameterListItem(name=key, value=value)
        for key, value in record.items()
    ]
    _run_sql_to_completion(w, warehouse_id, insert_sql, parameters=params)
    return f"SUCCESS: Action '{record['action_name']}' logged to {table_name} (SQL)."


def _log_via_spark(table_name: str, record: dict[str, Any]) -> str:
    """Append a record using Spark (notebook / cluster fallback).

    Uses an explicit all-STRING (nullable) schema instead of inferring it from a
    single row. Inference raises CANNOT_DETERMINE_TYPE when any value is None
    (e.g. mlflow_run_id with no active run), which was silently dropping every
    notebook log write. All audit columns are strings, so this is exact.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StringType, StructField, StructType

    spark = SparkSession.builder.getOrCreate()
    schema = StructType([StructField(name, StringType(), True) for name in record])
    df = spark.createDataFrame([tuple(record.values())], schema=schema)
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
