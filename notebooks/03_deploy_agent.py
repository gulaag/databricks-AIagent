# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Register the ResponsesAgent to Unity Catalog & Deploy to Model Serving
# MAGIC
# MAGIC **Purpose:** Smoke-test the agent locally, log it to Unity Catalog via MLflow
# MAGIC using the **`ResponsesAgent`** interface (the Databricks-recommended way to
# MAGIC author agents), then deploy it to a Model Serving endpoint.
# MAGIC
# MAGIC **Why `ResponsesAgent`?** It gives the deployed endpoint out-of-the-box
# MAGIC compatibility with the AI Playground, Agent Evaluation, and Agent Monitoring,
# MAGIC a standard streaming contract, and an automatically-inferred signature.
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01 and 02 must have been run successfully.
# MAGIC **Cell order:** local smoke tests → log_model (models-from-code) → alias →
# MAGIC deploy → wait → live test.

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

# Repo root: used both to import src here AND to package src into the model so
# the serving container can import it (otherwise: ModuleNotFoundError: 'src').
REPO_ROOT = "/Workspace/Users/digvijay@arsaga.jp/databricks-AIagent"
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
# no Spark session). Set this to your warehouse ID, e.g. "abc123def456". Leave
# empty to skip the SQL log path (the agent still runs; logging degrades gracefully).
SQL_WAREHOUSE_ID = ""

# Resolve workspace host + token so the agent's LLM and Vector Search clients work
# when the agent is run locally here (the smoke tests). At serving time these are
# provided automatically by injected M2M OAuth, so this is notebook-only.
os.environ["DATABRICKS_HOST"] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# The agent reads its config from env vars (see TechEngineerResponsesAgent.__init__).
# Set them for the local smoke test so retrieval + logging target the right places.
# NOTE: WEBHOOK_URL is intentionally LEFT UNSET here so that nothing can be posted
# to Slack during registration. The deployed endpoint receives the webhook from a
# secret (Step 4); the live post happens only in Step 6.
os.environ["DATABRICKS_FM_ENDPOINT"] = LLM_ENDPOINT
os.environ["VS_INDEX_NAME"] = VS_INDEX_NAME
os.environ["LOG_TABLE_NAME"] = LOG_TABLE_NAME
os.environ["SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID
os.environ.pop("WEBHOOK_URL", None)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md ## Step 1 — Pre-registration smoke tests (must all pass before log_model)
# MAGIC
# MAGIC These exercise the `ResponsesAgent` locally with the Responses request shape
# MAGIC (`{"input": [{"role": "user", "content": "..."}]}`). The queries are
# MAGIC information / refusal cases — none ask the agent to post — so nothing is sent.

# COMMAND ----------

from mlflow.types.responses import ResponsesAgentRequest

from src.agent.responses_agent import TechEngineerResponsesAgent


def _final_text(output_items: list) -> str:
    """Concatenate the text of every assistant `message` item in the output."""
    texts: list[str] = []
    for item in output_items:
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return "\n".join(texts)


SMOKE_TESTS = [
    {
        "query": "Databricks Unity Catalogについて教えてください。",
        "expected_signals": ["Unity Catalog"],
        "description": "Standard knowledge query",
    },
    {
        "query": "過去のセッションでデータガバナンスについて話しましたか？",
        # Pass if the answer either cites a source or gracefully reports no match.
        "expected_signals": ["Source", "ガバナンス", "見つかりません"],
        "description": "Retrieval answer is cited or gracefully empty",
    },
    {
        "query": "Pythonのクイックソートを実装してください。",
        "expected_signals": ["対応できません"],
        "description": "Out-of-scope refusal",
    },
]

smoke_agent = TechEngineerResponsesAgent()

failures = []
for i, test in enumerate(SMOKE_TESTS):
    try:
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": test["query"]}]
        )
        response = smoke_agent.predict(request)
        content = _final_text(response.output)
        if not any(sig in content for sig in test["expected_signals"]):
            failures.append(
                f"Test {i + 1} ({test['description']}): "
                f"Expected one of {test['expected_signals']} in response.\n"
                f"Got: {content[:300]}"
            )
        else:
            print(f"PASS — Test {i + 1}: {test['description']}")
    except Exception as exc:
        failures.append(f"Test {i + 1} ({test['description']}): Exception: {exc}")

assert not failures, "Pre-registration smoke tests failed:\n" + "\n\n".join(failures)
print("\nAll smoke tests passed. Proceeding with model registration.")

# COMMAND ----------

# MAGIC %md ## Step 2 — Log the agent to Unity Catalog (models-from-code)
# MAGIC
# MAGIC `ResponsesAgent` models are logged via the *models-from-code* pattern: we
# MAGIC point `python_model` at `agent_entry.py` (which calls `mlflow.models.set_model`)
# MAGIC and ship the `src` package via `code_paths`. MLflow infers the signature from
# MAGIC the `ResponsesAgent` schema automatically — we do not pass one.

# COMMAND ----------

# Serving container needs no Spark: the LLM call goes through mlflow.deployments,
# Vector Search and SQL logging through databricks-sdk. All auth via M2M OAuth.
pip_requirements = [
    "mlflow>=3.1",
    "requests",
    "databricks-vectorsearch",
    "databricks-sdk",
]

# Stage src to local disk for code_paths. MLflow refuses to copy from a /Workspace
# Repo path (it may contain notebook objects), and the repo's real on-disk path can
# differ from any hard-coded guess. So derive src's actual location from the import,
# then copy only .py files into a clean local dir. Stage agent_entry.py alongside
# it (NOT inside src) and use that file as the models-from-code entry point.
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
# agent_entry.py sits at the repo root, next to src/. Derive that root from the
# imported package location rather than the hard-coded REPO_ROOT, so this works
# regardless of the exact Git-folder name/casing on any given workspace.
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

# A non-posting example request used for signature/example purposes.
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

# Use the version returned by log_model rather than latest_versions[0], which is
# deprecated under Unity Catalog and not reliably ordered.
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
# MAGIC The agent reads its config from env vars. `WEBHOOK_URL` comes from a secret
# MAGIC so the webhook is never baked into the artifact or printed in logs.

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
# Only include SQL_WAREHOUSE_ID when set — an empty env-var value can be rejected.
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
    w.serving_endpoints.create(
        name=SERVING_ENDPOINT_NAME,
        config=endpoint_config,
    )
    print(f"Created endpoint: {SERVING_ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md ## Step 5 — Wait for endpoint readiness before live test

# COMMAND ----------


def _wait_for_endpoint(wc: WorkspaceClient, endpoint_name: str, timeout_s: int = 2400) -> None:
    """Poll until the endpoint reports ready=READY, or fail fast on UPDATE_FAILED.

    First-time serving builds (container + deps + compute) can take 10-30 min,
    hence the long timeout. A failed update is surfaced immediately instead of
    waiting out the clock.
    """
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
    raise TimeoutError(
        f"Endpoint {endpoint_name} did not become ready within {timeout_s}s"
    )


_wait_for_endpoint(w, SERVING_ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md ## Step 6 — Live endpoint smoke test
# MAGIC
# MAGIC `ResponsesAgent` endpoints accept the Responses request shape directly
# MAGIC (`{"input": [...]}`) and return `{"output": [...]}`.
# MAGIC
# MAGIC **NOTE:** the deployed agent is AUTONOMOUS — this request asks it to post, so
# MAGIC it **will** post to the Slack test channel. The human-approval demo lives in
# MAGIC notebooks 04 / 05.

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
