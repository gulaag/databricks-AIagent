# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Conversational Agent (propose → refine → approve → act)
# MAGIC
# MAGIC This is the "digital coworker" demo. You give a **one-liner**; the agent does the work:
# MAGIC
# MAGIC 1. **Propose** — searches past sessions, then drafts a COMPLETE announcement.
# MAGIC    It *proposes* missing details (date/time/venue) marked `[仮]` instead of interrogating you.
# MAGIC 2. **Refine** — talk to it in plain language (*"move to 2pm", "add a hands-on block",
# MAGIC    "make it 45 min", "more casual"*). Re-run as many times as you like.
# MAGIC 3. **Approve & send** — when you're happy, it posts to Slack and writes an audit log.
# MAGIC
# MAGIC Nothing is posted until you approve. **Prerequisite:** run `01_data_ingestion.py` first.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-vectorsearch databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Databricks auto-adds the repo root to sys.path; this is a harmless explicit fallback.
sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent")

CATALOG = "main"
SCHEMA = "tech_engineer"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.sessions_vs_index"
LOG_TABLE_NAME = f"{CATALOG}.{SCHEMA}.agent_action_log"

# Resolve workspace host + token so the agent's LLM and Vector Search clients work locally.
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# Webhook read from a secret — never hard-coded.
WEBHOOK_URL = dbutils.secrets.get(scope="agent_secrets", key="slack_webhook_url")

# COMMAND ----------

# MAGIC %md ## Step 1 — Build the agent

# COMMAND ----------

import mlflow

from src.agent.conversation import ConversationalAgent

mlflow.set_experiment("/Users/digvijay@arsaga.jp/agent-conversational")

agent = ConversationalAgent(
    endpoint=LLM_ENDPOINT,
    index_name=VS_INDEX_NAME,
    log_table=LOG_TABLE_NAME,
    webhook=WEBHOOK_URL,
    # warehouse_id left unset: logging uses Spark here in the notebook.
)
print("Agent ready.")

# COMMAND ----------

# MAGIC %md ## Step 2 — Propose
# MAGIC Give a one-liner in the `request` widget. The agent proposes a full plan + draft.
# MAGIC Nothing is posted.

# COMMAND ----------

dbutils.widgets.text(
    "request",
    "来週のTech Engineer共有会で「Databricks AI Agent入門」を1時間でやりたい",
    "Request (one-liner)",
)
request = dbutils.widgets.get("request")

result = agent.propose(request)

print("=== PLAN — what the agent will do ===")
for i, step in enumerate(result["steps"], 1):
    print(f"  {i}. {step}")
print("\n=== PROPOSED ANNOUNCEMENT (review — not yet sent) ===\n")
print(result["draft"])
print("\n=== sources used ===")
print(result["sources"] or "(none — drafted from general knowledge)")

# COMMAND ----------

# MAGIC %md ## Step 3 — Refine (optional, repeatable)
# MAGIC Type an instruction in the `feedback` widget and re-run this cell. Examples:
# MAGIC `日時を6月25日 18:00に確定` ・ `ハンズオンの時間を15分追加` ・ `もっとカジュアルに` ・ `45分に短縮`.
# MAGIC Leave it blank to skip.

# COMMAND ----------

dbutils.widgets.text("feedback", "", "Refinement (blank = skip)")
feedback = dbutils.widgets.get("feedback")

out = agent.refine(feedback)
if out.get("message"):
    print(out["message"])
if out.get("draft"):
    print("\n=== UPDATED ANNOUNCEMENT ===\n")
    print(out["draft"])

# COMMAND ----------

# MAGIC %md ## Step 4 — Approve & send
# MAGIC Set **`Confirm: send to Slack?` → `yes`** and run. It posts exactly the latest draft above.

# COMMAND ----------

dbutils.widgets.dropdown("confirm_send", "no", ["no", "yes"], "Confirm: send to Slack?")

if dbutils.widgets.get("confirm_send") == "yes":
    res = agent.send()
    print(res["post_status"])
    print(res["log_status"])
else:
    print("Not sent. Review/refine above, set 'Confirm: send to Slack?' = yes, then re-run.")

# COMMAND ----------

# MAGIC %md ## Step 5 — Execution log (audit trail)

# COMMAND ----------

display(
    spark.sql(
        f"SELECT logged_at, action_name, status FROM {LOG_TABLE_NAME} "
        "ORDER BY logged_at DESC LIMIT 10"
    )
)
