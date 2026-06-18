# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Run the Action Agent (interactive demo, with human approval)
# MAGIC
# MAGIC The agent runs in two phases, with a **confirmation gate** in between:
# MAGIC
# MAGIC 1. **Draft** — understand the request → search the knowledge base → write the
# MAGIC    agenda + announcement. **Nothing is posted yet.**
# MAGIC 2. **Review** — you read exactly what will be sent.
# MAGIC 3. **Send** — only if you set the confirmation widget to `yes`, the agent posts
# MAGIC    the announcement to the Slack test channel and logs the action.
# MAGIC
# MAGIC No Model Serving endpoint required — the agent runs directly here.
# MAGIC
# MAGIC **Prerequisite:** run `01_data_ingestion.py` first so the Vector Search index exists.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-vectorsearch databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Repo root on the path so `src` imports resolve. Adjust if your repo folder differs.
sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent")

CATALOG = "main"
SCHEMA = "tech_engineer"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.sessions_vs_index"
LOG_TABLE_NAME = f"{CATALOG}.{SCHEMA}.agent_action_log"

# Resolve workspace host + token so the agent's LLM and Vector Search clients work
# when run locally in this notebook.
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# Webhook read from a secret — never hard-coded.
WEBHOOK_URL = dbutils.secrets.get(scope="agent_secrets", key="slack_webhook_url")

# COMMAND ----------

# MAGIC %md ## Step 1 — Enter the request
# MAGIC Edit the `user_request` widget at the top of the notebook, or keep the sample.

# COMMAND ----------

dbutils.widgets.text(
    "user_request",
    "来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。"
    "1時間枠のアジェンダを作成し、社内向けの案内文を作成して、"
    "テスト用チャンネルに投稿してください。",
    "User request",
)
dbutils.widgets.dropdown("confirm_send", "no", ["no", "yes"], "Confirm: send to Slack?")

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

# MAGIC %md ## Step 3 — DRAFT (nothing is posted)
# MAGIC The agent searches and writes the announcement. Review the output below — this
# MAGIC is the **exact** text that will be sent if you approve.

# COMMAND ----------

with mlflow.start_run(run_name="agent-draft"):
    draft = agent.draft_announcement(user_request)

print("================ DRAFT (review before sending) ================\n")
print(draft["message"])
print("\n================ sources used ================")
print(draft["sources"] or "(none)")

# COMMAND ----------

# MAGIC %md ## Step 4 — CONFIRM & SEND
# MAGIC To send: set the **`Confirm: send to Slack?`** widget to `yes`, then run this cell.
# MAGIC It posts exactly the draft shown above. To revise instead, edit the request and
# MAGIC re-run Step 3.

# COMMAND ----------

if dbutils.widgets.get("confirm_send") == "yes":
    with mlflow.start_run(run_name="agent-send"):
        result = agent.send_announcement(draft["message"])
    print(result["post_status"])
    print(result["log_status"])
else:
    print("Not sent. Review the draft above, set 'Confirm: send to Slack?' = yes, then re-run.")

# COMMAND ----------

# MAGIC %md ## Step 5 — Execution log (the audit trail)

# COMMAND ----------

display(
    spark.sql(
        f"SELECT logged_at, action_name, status FROM {LOG_TABLE_NAME} "
        "ORDER BY logged_at DESC LIMIT 10"
    )
)
