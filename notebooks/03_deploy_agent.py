# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Register the Agent to Unity Catalog & Deploy to Model Serving
# MAGIC
# MAGIC **Purpose:** Smoke-test the unified agent locally, log it to Unity Catalog via
# MAGIC MLflow using the **`ResponsesAgent`** interface (the Databricks-recommended way
# MAGIC to author agents), then deploy it to a Model Serving endpoint.
# MAGIC
# MAGIC One agent, three modes (selected per request via `custom_inputs={"mode": ...}`):
# MAGIC `auto` (autonomous), `draft` (propose/refine, posts nothing), `send` (post approved).
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01 and 02 must have been run successfully.

# COMMAND ----------

# MAGIC %pip install -U "mlflow>=3.1" databricks-sdk databricks-vectorsearch --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil
import sys
import time

import mlflow
import mlflow.pyfunc

# Repo root is used only to import src here; Databricks also auto-adds it to the
# path. The model packaging derives every path from the src import, so it is
# independent of the exact Git-folder name.
REPO_ROOT = "/Workspace/Users/digvijay@arsaga.jp/databricks-Aiagent"
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "main"
SCHEMA = "tech_engineer"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.tech_engineer_agent"
SERVING_ENDPOINT_NAME = "tech-engineer-agent-endpoint"
EXPERIMENT_PATH = "/Users/digvijay@arsaga.jp/agent-deployment"

LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
VS_INDEX_NAME = "main.tech_engineer.sessions_vs_index"
LOG_TABLE_NAME = "main.tech_engineer.agent_action_log"

# SQL warehouse used for serving-time action logging (the serving container has
# no Spark session). Wiring it here makes the audit log work end-to-end from the
# deployed endpoint, not just from notebooks. Serverless warehouse auto-starts on
# demand and auto-stops when idle.
SQL_WAREHOUSE_ID = "c222bab9769cec3a"

# Resolve workspace host + token so the agent's LLM and Vector Search clients work
# when the agent is run locally here (the smoke tests). At serving time these are
# provided automatically by injected M2M OAuth, so this is notebook-only.
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# The agent reads its config from env vars. For the local smoke test:
#  - point retrieval + logging at the right places,
#  - leave SQL_WAREHOUSE_ID empty so notebook logging uses the Spark path,
#  - leave WEBHOOK_URL unset so NOTHING can be posted during registration.
os.environ["DATABRICKS_FM_ENDPOINT"] = LLM_ENDPOINT
os.environ["VS_INDEX_NAME"] = VS_INDEX_NAME
os.environ["LOG_TABLE_NAME"] = LOG_TABLE_NAME
os.environ["SQL_WAREHOUSE_ID"] = ""
os.environ.pop("WEBHOOK_URL", None)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md ## Step 1 — Pre-registration smoke tests (must all pass before log_model)
# MAGIC
# MAGIC Exercises the agent locally in `auto` mode (knowledge / retrieval / refusal —
# MAGIC none of which post) and `draft` mode (propose, which never posts). Nothing is
# MAGIC sent to Slack during registration.

# COMMAND ----------

from src.agent.agent import TechEngineerAgent

smoke_agent = TechEngineerAgent()
failures = []

# --- auto mode: information, retrieval, and out-of-scope refusal ---
AUTO_TESTS = [
    {
        "query": "Databricks Unity Catalogについて教えてください。",
        "signals": ["Unity Catalog"],
        "desc": "Standard knowledge query",
    },
    {
        "query": "過去のセッションでデータガバナンスについて話しましたか？",
        "signals": ["Source", "ガバナンス", "見つかりません"],
        "desc": "Retrieval answer is cited or gracefully empty",
    },
    {
        "query": "Pythonのクイックソートを実装してください。",
        "signals": ["対応できません"],
        "desc": "Out-of-scope refusal",
    },
]

for i, test in enumerate(AUTO_TESTS):
    try:
        content = smoke_agent.autonomous(test["query"])
        if not any(sig in content for sig in test["signals"]):
            failures.append(
                f"[auto] Test {i + 1} ({test['desc']}): expected one of "
                f"{test['signals']}.\nGot: {content[:300]}"
            )
        else:
            print(f"PASS — [auto] {test['desc']}")
    except Exception as exc:
        failures.append(f"[auto] Test {i + 1} ({test['desc']}): Exception: {exc}")

# --- draft mode: propose a complete announcement (posts nothing) ---
try:
    result = smoke_agent.propose(
        "来週のTech Engineer共有会で「Databricks AI Agent入門」を1時間でやりたい。"
    )
    draft = result.get("draft") or ""
    assert len(draft) > 50, f"draft unexpectedly short: {draft!r}"
    print(f"PASS — [draft] proposed {len(draft)} chars; sources={result.get('sources')}")
except Exception as exc:
    failures.append(f"[draft] propose: Exception: {exc}")

assert not failures, "Pre-registration smoke tests failed:\n" + "\n\n".join(failures)
print("\nAll smoke tests passed. Proceeding with model registration.")

# COMMAND ----------

# MAGIC %md ## Step 2 — Log the agent to Unity Catalog (models-from-code)
# MAGIC
# MAGIC `ResponsesAgent` models are logged via the *models-from-code* pattern: we point
# MAGIC `python_model` at `agent_entry.py` (which calls `mlflow.models.set_model`) and
# MAGIC ship the `src` package via `code_paths`. MLflow infers the signature from the
# MAGIC `ResponsesAgent` schema automatically — we do not pass one.

# COMMAND ----------

pip_requirements = [
    "mlflow>=3.1",
    "requests",
    "databricks-vectorsearch",
    "databricks-sdk",
]

# Stage src to local disk for code_paths (MLflow refuses to copy from a /Workspace
# Repo path). Derive src's real location from the import, copy only .py files, and
# stage agent_entry.py next to it (from the same repo root the import resolves to,
# so this is independent of the Git-folder name).
import src as _src_pkg

SRC_DIR = (
    os.path.dirname(_src_pkg.__file__)
    if getattr(_src_pkg, "__file__", None)
    else list(_src_pkg.__path__)[0]
)
STAGE_DIR = "/tmp/agent_code"
LOCAL_SRC = os.path.join(STAGE_DIR, "src")
ENTRY_FILE = os.path.join(STAGE_DIR, "agent_entry.py")

shutil.rmtree(STAGE_DIR, ignore_errors=True)
for _root, _dirs, _files in os.walk(SRC_DIR):
    _dirs[:] = [d for d in _dirs if d != "__pycache__"]
    _rel = os.path.relpath(_root, SRC_DIR)
    _dest = LOCAL_SRC if _rel == "." else os.path.join(LOCAL_SRC, _rel)
    os.makedirs(_dest, exist_ok=True)
    for _fn in _files:
        if _fn.endswith(".py"):
            shutil.copyfile(os.path.join(_root, _fn), os.path.join(_dest, _fn))

_repo_root = os.path.dirname(SRC_DIR)
shutil.copyfile(os.path.join(_repo_root, "agent_entry.py"), ENTRY_FILE)
print(f"Staged src -> {LOCAL_SRC}: {sorted(os.listdir(LOCAL_SRC))}")
print(f"Staged entry -> {ENTRY_FILE} (from repo root {_repo_root})")

# Declare the UC resources the agent depends on so Model Serving injects M2M OAuth.
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksVectorSearchIndex,
)

resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=VS_INDEX_NAME),
]
if SQL_WAREHOUSE_ID:
    resources.append(DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID))

# A non-posting example request (Responses format) for the input example.
input_example = {
    "input": [
        {"role": "user", "content": "Databricks AI Agentとは何かを一言で教えてください。"}
    ]
}

with mlflow.start_run(run_name="agent-registration") as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent_model",
        python_model=ENTRY_FILE,          # models-from-code entry (calls set_model)
        code_paths=[LOCAL_SRC],           # ship the src package for imports at load
        pip_requirements=pip_requirements,
        registered_model_name=MODEL_NAME,
        resources=resources,
        input_example=input_example,
        await_registration_for=300,
    )
    print(f"Model logged. Run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered version: {model_info.registered_model_version}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Set the registered model as Champion alias

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
latest_version = model_info.registered_model_version

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=latest_version,
)
print(f"Set alias 'champion' -> version {latest_version} of {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## Step 4 — Create or update the Model Serving endpoint
# MAGIC
# MAGIC The agent reads config from env vars. `WEBHOOK_URL` comes from a secret so it is
# MAGIC never baked into the artifact or printed. `SQL_WAREHOUSE_ID` makes the audit log
# MAGIC work from the serving container (which has no Spark).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)

w = WorkspaceClient()

env_vars = {
    "WEBHOOK_URL": "{{secrets/agent_secrets/slack_webhook_url}}",
    "DATABRICKS_FM_ENDPOINT": LLM_ENDPOINT,
    "VS_INDEX_NAME": VS_INDEX_NAME,
    "LOG_TABLE_NAME": LOG_TABLE_NAME,
}
if SQL_WAREHOUSE_ID:
    env_vars["SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

served_model = ServedModelInput(
    model_name=MODEL_NAME,
    model_version=str(latest_version),
    workload_size=ServedModelInputWorkloadSize.SMALL,
    scale_to_zero_enabled=True,
    environment_vars=env_vars,
)

endpoint_config = EndpointCoreConfigInput(served_models=[served_model])

existing = [e.name for e in w.serving_endpoints.list()]
if SERVING_ENDPOINT_NAME in existing:
    w.serving_endpoints.update_config(
        name=SERVING_ENDPOINT_NAME, served_models=[served_model]
    )
    print(f"Updated endpoint: {SERVING_ENDPOINT_NAME}")
else:
    w.serving_endpoints.create(name=SERVING_ENDPOINT_NAME, config=endpoint_config)
    print(f"Created endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md ## Step 5 — Wait for endpoint readiness before live test

# COMMAND ----------


def _wait_for_endpoint(wc: WorkspaceClient, endpoint_name: str, timeout_s: int = 2400) -> None:
    """Poll until the endpoint reports ready=READY, or fail fast on UPDATE_FAILED."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ep = wc.serving_endpoints.get(name=endpoint_name)
        state = ep.state.config_update.value if ep.state else "UNKNOWN"
        ready = ep.state.ready.value if ep.state else "NOT_READY"
        print(f"  config_update={state} | ready={ready}")
        if ready == "READY":
            print(f"Endpoint {endpoint_name} is ready.")
            return
        if state == "UPDATE_FAILED":
            raise RuntimeError(
                f"Endpoint {endpoint_name} update FAILED. Check Serving > "
                f"{endpoint_name} > build/service logs in the UI."
            )
        time.sleep(30)
    raise TimeoutError(f"Endpoint {endpoint_name} did not become ready within {timeout_s}s")


_wait_for_endpoint(w, SERVING_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md ## Step 6 — Live endpoint smoke test
# MAGIC
# MAGIC `ResponsesAgent` endpoints accept the Responses request shape directly
# MAGIC (`{"input": [...]}`) and return `{"output": [...]}`.
# MAGIC
# MAGIC **NOTE:** this asks the agent to post, so it **will** post to the Slack test
# MAGIC channel (default `auto` mode) and write an audit-log row via the SQL warehouse.

# COMMAND ----------

import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

test_payload = {
    "input": [
        {
            "role": "user",
            "content": (
                "来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。"
                "1時間枠のアジェンダを作成し、社内向けの案内文を作って、"
                "テスト用Slackチャンネルに投稿してください。"
            ),
        }
    ]
}

response = requests.post(
    url=f"https://{host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=test_payload,
    timeout=180,
)

print(f"Status: {response.status_code}")
body = response.json()
print(json.dumps(body, ensure_ascii=False, indent=2))

if response.status_code == 200 and "output" in body:
    texts = [
        part.get("text", "")
        for item in body["output"]
        if item.get("type") == "message"
        for part in (item.get("content") or [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    print("\n=== Agent final message ===\n" + "\n".join(texts))
