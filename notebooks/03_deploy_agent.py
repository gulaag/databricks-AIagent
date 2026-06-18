# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Register Agent to Unity Catalog & Deploy to Model Serving
# MAGIC
# MAGIC **Purpose:** Smoke-test the agent locally, log it to Unity Catalog via MLflow,
# MAGIC then deploy to a Databricks Model Serving endpoint.
# MAGIC
# MAGIC **Prerequisites:** Notebooks 01 and 02 must have been run successfully.
# MAGIC **Cell order:** pre-registration smoke tests → log_model → alias → deploy → wait → live test

# COMMAND ----------

# MAGIC %pip install mlflow databricks-sdk databricks-vectorsearch --quiet
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

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md ## Step 1 — Pre-registration smoke tests (must all pass before log_model)

# COMMAND ----------

from src.agent.pyfunc_model import AgentModel


class _MockContext:
    artifacts = {}


SMOKE_TESTS = [
    {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Databricks Unity Catalogについて教えてください。",
                }
            ]
        },
        "expected_signals": ["Unity Catalog"],
        "description": "Standard knowledge query",
    },
    {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "過去のセッションでデータガバナンスについて話しましたか？",
                }
            ]
        },
        # Pass if the answer either cites a source or gracefully reports no match.
        # Avoids a brittle hard dependency on specific index content.
        "expected_signals": ["Source", "ガバナンス", "見つかりません"],
        "description": "Retrieval answer is cited or gracefully empty",
    },
    {
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Pythonのクイックソートを実装してください。",
                }
            ]
        },
        "expected_signals": ["対応できません"],
        "description": "Out-of-scope refusal",
    },
]

# Smoke-test instance is configured but kept separate from the instance we log,
# so no inference-time state can leak into the pickled model artifact.
smoke_agent = AgentModel().configure(
    endpoint=LLM_ENDPOINT,
    index_name=VS_INDEX_NAME,
    log_table=LOG_TABLE_NAME,
    warehouse_id=SQL_WAREHOUSE_ID,
)
smoke_agent.load_context(_MockContext())

failures = []
for i, test in enumerate(SMOKE_TESTS):
    try:
        result = smoke_agent.predict(_MockContext(), test["input"])
        content = result["choices"][0]["message"]["content"]
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

# MAGIC %md ## Step 2 — Log model to Unity Catalog

# COMMAND ----------

# Serving container needs no Spark: the LLM call goes through mlflow.deployments,
# Vector Search and SQL logging through databricks-sdk. All auth via M2M OAuth.
pip_requirements = [
    "mlflow",
    "requests",
    "databricks-vectorsearch",
    "databricks-sdk",
]

agent_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "vs_index_name": VS_INDEX_NAME,
    "log_table_name": LOG_TABLE_NAME,
    "warehouse_id": SQL_WAREHOUSE_ID,
}

# Fresh, explicitly-configured instance — never run through predict(), so the
# pickled artifact carries no inference-time state. Its resources include the
# SQL warehouse only when SQL_WAREHOUSE_ID is set.
logged_agent = AgentModel().configure(
    endpoint=LLM_ENDPOINT,
    index_name=VS_INDEX_NAME,
    log_table=LOG_TABLE_NAME,
    warehouse_id=SQL_WAREHOUSE_ID,
)

# Build the signature explicitly from representative examples. infer_signature
# does NOT invoke the model, so it can't trip over load_context — and Unity
# Catalog requires every model to carry a signature.
from mlflow.models import infer_signature

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "来週のTech Engineer共有会のアジェンダを作成してください。",
        }
    ]
}
output_example = {
    "choices": [
        {"message": {"role": "assistant", "content": "（サンプル応答）"}}
    ]
}
signature = infer_signature(input_example, output_example)

# Stage src to local disk for code_paths. MLflow refuses to copy from a /Workspace
# Repo path (it may contain notebook objects), and the repo's real on-disk path can
# differ from any hard-coded guess. So derive src's actual location from the import,
# then copy only .py files (content only, no metadata) into a clean local dir.
import src as _src_pkg

SRC_DIR = (
    os.path.dirname(_src_pkg.__file__)
    if getattr(_src_pkg, "__file__", None)
    else list(_src_pkg.__path__)[0]
)
LOCAL_SRC = "/tmp/agent_code/src"
shutil.rmtree("/tmp/agent_code", ignore_errors=True)
for _root, _dirs, _files in os.walk(SRC_DIR):
    _dirs[:] = [d for d in _dirs if d != "__pycache__"]
    _rel = os.path.relpath(_root, SRC_DIR)
    _dest = LOCAL_SRC if _rel == "." else os.path.join(LOCAL_SRC, _rel)
    os.makedirs(_dest, exist_ok=True)
    for _fn in _files:
        if _fn.endswith(".py"):
            shutil.copyfile(os.path.join(_root, _fn), os.path.join(_dest, _fn))
print(f"Staged src from {SRC_DIR} -> {LOCAL_SRC}: {sorted(os.listdir(LOCAL_SRC))}")

with mlflow.start_run(run_name="agent-registration") as run:
    # Write config to a local path and hand that path to log_model, which copies
    # it into the model artifacts. A runs:/ URI would not resolve during logging.
    tmp_config_path = f"/tmp/agent_config_{run.info.run_id}.json"
    with open(tmp_config_path, "w") as f:
        json.dump(agent_config, f)

    model_info = mlflow.pyfunc.log_model(
        name="agent_model",
        python_model=logged_agent,
        # Ship the src package inside the model so the serving container can
        # import src.agent.* / src.tools.* at load time (staged to local disk).
        code_paths=[LOCAL_SRC],
        pip_requirements=pip_requirements,
        registered_model_name=MODEL_NAME,
        await_registration_for=300,
        artifacts={"agent_config": tmp_config_path},
        resources=logged_agent.resources,
        signature=signature,
        input_example=input_example,
    )
    os.unlink(tmp_config_path)
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
        "WEBHOOK_URL": "{{secrets/agent_secrets/slack_webhook_url}}",
        "VS_INDEX_NAME": VS_INDEX_NAME,
        "LOG_TABLE_NAME": LOG_TABLE_NAME,
        "SQL_WAREHOUSE_ID": SQL_WAREHOUSE_ID,
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

# COMMAND ----------

import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

test_payload = {
    "messages": [
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
    timeout=120,
)

print(f"Status: {response.status_code}")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))
