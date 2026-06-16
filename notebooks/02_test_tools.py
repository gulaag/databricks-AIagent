# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Isolated Tool Testing
# MAGIC
# MAGIC **Purpose:** Validate each tool independently before wiring them into the agent.
# MAGIC Run cells individually. Each section is self-contained.

# COMMAND ----------

# MAGIC %pip install mlflow requests databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

# Add project root to path so local src imports resolve
sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent")

# ---------------------------------------------------------------------------
# Configuration — set before running
# ---------------------------------------------------------------------------
VS_INDEX_NAME = "main.tech_engineer.sessions_vs_index"
LOG_TABLE_NAME = "main.tech_engineer.agent_action_log"

# Leave empty to log via Spark (works in any notebook). Set to a SQL warehouse
# ID to exercise the same statement-execution path used at serving time.
SQL_WAREHOUSE_ID = ""

# Teams webhook URL — retrieve from UC secrets at runtime:
# TEAMS_WEBHOOK_URL = dbutils.secrets.get(scope="agent_secrets", key="teams_webhook_url")
TEAMS_WEBHOOK_URL = ""  # Replace with your test webhook URL

# COMMAND ----------

# MAGIC %md ## Test 1 — Vector Search retrieval

# COMMAND ----------

import mlflow
from src.tools.search import search_knowledge_base

mlflow.set_experiment("/Users/digvijay@arsaga.jp/agent-tool-tests")

with mlflow.start_run(run_name="test_search"):
    results = search_knowledge_base(
        query="Databricks Unity Catalog RBAC data governance",
        index_name=VS_INDEX_NAME,
        num_results=3,
    )
    mlflow.log_metric("num_results_returned", len(results))
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(r["chunk_text"][:300])
        print("Metadata:", r["metadata"])

# COMMAND ----------

# MAGIC %md ## Test 2 — Teams webhook (uses a test/dry-run channel)

# COMMAND ----------

from src.tools.teams import post_to_teams

test_agenda = """
【テスト投稿】Tech Engineer 勉強会 — Databricks AI Agent 入門

■ 日時: 2024年X月X日（月）18:00〜19:00
■ テーマ: Databricksを用いたAI Agent構築入門

アジェンダ:
1. 00:00〜00:10 オープニング・前回振り返り
2. 00:10〜00:30 Databricks Agent Framework 概要
3. 00:30〜00:50 ライブデモ: ツール実行とMLflow Tracing
4. 00:50〜01:00 Q&A・次回予告

※ このメッセージはテスト投稿です。
"""

if TEAMS_WEBHOOK_URL:
    with mlflow.start_run(run_name="test_teams_post"):
        result = post_to_teams(agenda_content=test_agenda, webhook_url=TEAMS_WEBHOOK_URL)
        mlflow.log_param("post_result", result)
        print(result)
else:
    print("SKIPPED: Set TEAMS_WEBHOOK_URL to run this test.")

# COMMAND ----------

# MAGIC %md ## Test 3 — Action logger (writes to Delta table)

# COMMAND ----------

from src.tools.logger import log_agent_action

with mlflow.start_run(run_name="test_logger"):
    result = log_agent_action(
        action_name="test_log_entry",
        input_payload={"test_key": "test_value"},
        output_payload={"status": "verified"},
        table_name=LOG_TABLE_NAME,
        status="SUCCESS",
        warehouse_id=SQL_WAREHOUSE_ID or None,
    )
    print(result)

# COMMAND ----------

# MAGIC %md ## Verify — Query the action log table

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {LOG_TABLE_NAME} ORDER BY logged_at DESC LIMIT 10"))
