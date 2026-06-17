# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Run the Action Agent (interactive demo)
# MAGIC
# MAGIC This is the live demo. Type a request, run the agent end-to-end:
# MAGIC
# MAGIC **understand → decompose → search knowledge → draft a 1-hour agenda →
# MAGIC post to the test Slack channel → log the run.**
# MAGIC
# MAGIC No Model Serving endpoint required — the agent runs directly here, which is
# MAGIC cheaper and simpler to present. (Notebook 03 shows how it would be deployed
# MAGIC as a REST endpoint if/when that's needed.)
# MAGIC
# MAGIC **Prerequisite:** run `01_data_ingestion.py` first so the Vector Search index exists.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-vectorsearch databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys

# Repo root on the path so `src` imports resolve. Adjust if your repo folder differs.
sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent")

CATALOG = "main"
SCHEMA = "tech_engineer"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.sessions_vs_index"
LOG_TABLE_NAME = f"{CATALOG}.{SCHEMA}.agent_action_log"

# Webhook read from a secret — never hard-coded.
WEBHOOK_URL = dbutils.secrets.get(scope="agent_secrets", key="slack_webhook_url")

# COMMAND ----------

# MAGIC %md ## Step 1 — Enter the request
# MAGIC Edit the text widget at the top of the notebook, or keep the sample below.

# COMMAND ----------

dbutils.widgets.text(
    "user_request",
    "来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。"
    "1時間枠のアジェンダを作成し、社内向けの案内文を作成して、"
    "テスト用チャンネルに投稿してください。",
    "User request",
)

user_request = dbutils.widgets.get("user_request")
print("Request:\n" + user_request)

# COMMAND ----------

# MAGIC %md ## Step 2 — Build the agent

# COMMAND ----------

import mlflow

from src.agent.pyfunc_model import AgentModel

mlflow.set_experiment("/Users/digvijay@arsaga.jp/agent-demo-runs")

agent = AgentModel().configure(
    endpoint=LLM_ENDPOINT,
    index_name=VS_INDEX_NAME,
    log_table=LOG_TABLE_NAME,
    webhook=WEBHOOK_URL,
    # warehouse_id left unset: logging uses Spark here in the notebook.
)


class _Ctx:
    artifacts = {}


agent.load_context(_Ctx())
print("Agent ready.")

# COMMAND ----------

# MAGIC %md ## Step 3 — Run the agent (this posts to Slack)
# MAGIC Open the MLflow run/trace from the cell output to watch each tool call:
# MAGIC `search_knowledge_base` → `post_to_channel` → `log_agent_action`.

# COMMAND ----------

with mlflow.start_run(run_name="agent-demo"):
    result = agent.predict(_Ctx(), {"messages": [{"role": "user", "content": user_request}]})

answer = result["choices"][0]["message"]["content"]
print(answer)

# COMMAND ----------

# MAGIC %md ## Step 4 — Show the execution log (the audit trail)

# COMMAND ----------

display(
    spark.sql(
        f"SELECT logged_at, action_name, status FROM {LOG_TABLE_NAME} "
        "ORDER BY logged_at DESC LIMIT 10"
    )
)
