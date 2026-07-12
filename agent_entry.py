"""
Models-from-code entry point for logging the ResponsesAgent to Unity Catalog.

MLflow executes this file both at log time and when the model loads inside the
serving container. Its only job is to (a) construct the agent and (b) register
it via ``mlflow.models.set_model``. All runtime configuration (LLM endpoint,
Vector Search index, log table, SQL warehouse, webhook) is read from environment
variables inside the agent, which Model Serving injects from the endpoint's env
vars and secrets — so nothing environment-specific is baked into the artifact.

Logged with:
    mlflow.pyfunc.log_model(
        name="agent_model",
        python_model="<this file>",
        code_paths=["<staged src/>"],
        resources=[...],
        ...
    )
"""

import mlflow

from src.agent.responses_agent import TechEngineerResponsesAgent

AGENT = TechEngineerResponsesAgent()
mlflow.models.set_model(AGENT)
