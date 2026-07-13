# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Agent Demo (one agent, two ways to drive it)
# MAGIC
# MAGIC A single agent (`TechEngineerAgent`) powers both demos below. Same code, same
# MAGIC deployed endpoint — only the *mode* differs:
# MAGIC
# MAGIC 1. **Conversational (human-in-the-loop):** `propose → refine → approve → send`.
# MAGIC    The agent proposes a complete announcement, you refine it in plain language,
# MAGIC    and nothing is posted until you approve. *(This is the star of the demo.)*
# MAGIC 2. **Autonomous (hands-off):** one request → search → draft → post → log, in a
# MAGIC    single call. This is exactly what the deployed endpoint / AI Playground runs.
# MAGIC
# MAGIC **Prerequisite:** run `01_data_ingestion.py` first so the Vector Search index exists.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.1" databricks-vectorsearch databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Databricks auto-adds the repo root to sys.path; this is a harmless explicit fallback.
sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-Aiagent")

CATALOG = "main"
SCHEMA = "tech_engineer"

# Configure the agent via env vars (same contract as serving). SQL_WAREHOUSE_ID is
# left unset so notebook logging uses the Spark path.
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DATABRICKS_FM_ENDPOINT"] = "databricks-meta-llama-3-3-70b-instruct"
os.environ["VS_INDEX_NAME"] = f"{CATALOG}.{SCHEMA}.sessions_vs_index"
os.environ["LOG_TABLE_NAME"] = f"{CATALOG}.{SCHEMA}.agent_action_log"
os.environ["SQL_WAREHOUSE_ID"] = ""
# Webhook read from a secret — never hard-coded.
os.environ["WEBHOOK_URL"] = dbutils.secrets.get(scope="agent_secrets", key="slack_webhook_url")

LOG_TABLE_NAME = os.environ["LOG_TABLE_NAME"]

# COMMAND ----------

# MAGIC %md ## Build the agent

# COMMAND ----------

import mlflow

from src.agent.agent import TechEngineerAgent

mlflow.set_experiment("/Users/digvijay@arsaga.jp/agent-demo")

agent = TechEngineerAgent()
print("Agent ready.")


def show_audit_log(n: int = 10):
    """Freshly query and display the most recent audit-log rows.

    Called right after send / autonomous so the updated trail appears inline —
    no need to re-run a separate cell during the demo.
    """
    display(
        spark.sql(
            f"SELECT logged_at, action_name, status FROM {LOG_TABLE_NAME} "
            f"ORDER BY logged_at DESC LIMIT {n}"
        )
    )

# COMMAND ----------

# MAGIC %md # Demo 1 — Conversational (propose → refine → approve → send)

# COMMAND ----------

# MAGIC %md ## Step 1 — Propose
# MAGIC Give a one-liner in the `request` widget. The agent proposes a full plan + draft.
# MAGIC Nothing is posted.

# COMMAND ----------

dbutils.widgets.text(
    "request",
    "来週のTech Engineer共有会で「Databricks AI Agent入門」を1時間でやりたい",
    "Request (one-liner)",
)
request = dbutils.widgets.get("request")

agent.reset()
result = agent.propose(request)

print("=== PLAN — what the agent will do ===")
for i, step in enumerate(result["steps"], 1):
    print(f"  {i}. {step}")
print("\n=== PROPOSED ANNOUNCEMENT (review — not yet sent) ===\n")
print(result["draft"])
print("\n=== sources used ===")
print(result["sources"] or "(none — drafted from general knowledge)")

# COMMAND ----------

# MAGIC %md ## Step 2 — Refine (optional, repeatable)
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

# MAGIC %md ## Step 3 — Approve & send
# MAGIC Set **`Confirm: send to Slack?` → `yes`** and run. It posts exactly the latest draft above.

# COMMAND ----------

dbutils.widgets.dropdown("confirm_send", "no", ["no", "yes"], "Confirm: send to Slack?")

if dbutils.widgets.get("confirm_send") == "yes":
    res = agent.send()
    print(res["post_status"])
    print(res["log_status"])
    # Auto-display the refreshed audit trail — the just-logged action appears here
    # immediately, so there is nothing to re-run during the demo.
    print("\n=== 実行ログ（最新・自動更新） ===")
    show_audit_log()
else:
    print("Not sent. Review/refine above, set 'Confirm: send to Slack?' = yes, then re-run.")

# COMMAND ----------

# MAGIC %md # Demo 2 — Autonomous (hands-off, one call)
# MAGIC The **same agent**, run end-to-end with no approval gate: it searches, drafts,
# MAGIC posts, and logs by itself. This is what the deployed endpoint does. Because it
# MAGIC posts, it is behind a confirm widget.

# COMMAND ----------

dbutils.widgets.text(
    "auto_request",
    "来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマに、"
    "1時間枠のアジェンダを作成し、案内文を作って、テスト用チャンネルに投稿してください。",
    "Autonomous request",
)
dbutils.widgets.dropdown("confirm_autonomous", "no", ["no", "yes"], "Run autonomous (will post)?")

if dbutils.widgets.get("confirm_autonomous") == "yes":
    agent.reset()
    final = agent.autonomous(dbutils.widgets.get("auto_request"))
    print("=== Autonomous run complete — final message ===\n")
    print(final)
    print("\n=== 実行ログ（最新・自動更新） ===")
    show_audit_log()
else:
    print("Skipped. Set 'Run autonomous (will post)?' = yes to run the hands-off mode.")

# COMMAND ----------

# MAGIC %md # Execution log (the audit trail)
# MAGIC This same view is shown automatically right after `send` / `autonomous` above —
# MAGIC re-run this cell only if you want to refresh it independently.

# COMMAND ----------

show_audit_log()
