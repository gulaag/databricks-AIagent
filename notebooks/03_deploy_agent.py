# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Register Agent to Unity Catalog & Deploy to Model Serving
# MAGIC
# MAGIC **Purpose:** Log the AgentModel to Unity Catalog via MLflow, then deploy it
# MAGIC to a Databricks Model Serving endpoint.
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01 and 02 must have been run successfully.

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
import mlflow
import mlflow.pyfunc

sys.path.insert(0, "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "main"
SCHEMA = "tech_engineer"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.tech_engineer_agent"
SERVING_ENDPOINT_NAME = "tech-engineer-agent-endpoint"
EXPERIMENT_PATH = "/Users/digvijay@arsaga.jp/agent-deployment"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md ## Step 1 — Log model to Unity Catalog

# COMMAND ----------

from src.agent.pyfunc_model import AgentModel

pip_requirements = [
    "mlflow",
    "requests",
    "databricks-vectorsearch",
    "pyspark",
]

with mlflow.start_run(run_name="agent-registration") as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="agent_model",
        python_model=AgentModel(),
        pip_requirements=pip_requirements,
        registered_model_name=MODEL_NAME,
        await_registration_for=300,
    )
    print(f"Model logged. Run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md ## Step 2 — Set the registered model as Champion alias

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
latest_version = client.get_registered_model(MODEL_NAME).latest_versions[0].version

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=latest_version,
)
print(f"Set alias 'champion' -> version {latest_version} of {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Create or update the Model Serving endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)

w = WorkspaceClient()

served_model = ServedModelInput(
    model_name=MODEL_NAME,
    model_version=str(latest_version),
    workload_size=ServedModelInputWorkloadSize.SMALL,
    scale_to_zero_enabled=True,
    environment_vars={
        "DATABRICKS_HOST": "{{secrets/agent_secrets/databricks_host}}",
        "DATABRICKS_TOKEN": "{{secrets/agent_secrets/databricks_token}}",
        "TEAMS_WEBHOOK_URL": "{{secrets/agent_secrets/teams_webhook_url}}",
        "VS_INDEX_NAME": "main.tech_engineer.sessions_vs_index",
        "LOG_TABLE_NAME": "main.tech_engineer.agent_action_log",
    },
)

endpoint_config = EndpointCoreConfigInput(served_models=[served_model])

existing = [e.name for e in w.serving_endpoints.list()]
if SERVING_ENDPOINT_NAME in existing:
    w.serving_endpoints.update_config(
        name=SERVING_ENDPOINT_NAME, served_models=[served_model]
    )
    print(f"Updated endpoint: {SERVING_ENDPOINT_NAME}")
else:
    w.serving_endpoints.create(
        name=SERVING_ENDPOINT_NAME,
        config=endpoint_config,
    )
    print(f"Created endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Smoke test the live endpoint

# COMMAND ----------

import requests
import json

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

test_payload = {
    "messages": [
        {
            "role": "user",
            "content": (
                "来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。"
                "1時間枠のアジェンダを作成し、社内向けの案内文を作って、"
                "テスト用Teamsチャンネルに投稿してください。"
            ),
        }
    ]
}

response = requests.post(
    url=f"https://{host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=test_payload,
    timeout=120,
)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
