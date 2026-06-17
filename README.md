# Databricks Action Agent — Tech Engineer Study Group

Enterprise-grade action agent built on Databricks. Given a plain-language
request, the agent retrieves context from past session transcripts via
Vector Search, drafts a Japanese-language announcement, posts it to a chat
channel (Slack or Microsoft Teams), and logs every action to a Unity Catalog
Delta table.

---

## Architecture

```
User Request
    │
    ▼
AgentModel (MLflow PyFunc)
    │
    ├── search_knowledge_base ──► Databricks Vector Search (UC-governed index)
    │
    ├── post_to_channel ────────► Slack / Microsoft Teams (Incoming Webhook)
    │
    └── log_agent_action ───────► Unity Catalog Delta Table (audit log)
                                        │
                                        └── MLflow Tracing (every span recorded)
```

**Foundation Model:** Databricks Meta-Llama-3.3-70B-Instruct (pay-per-token, no training on proprietary data)  
**Embedding Model:** Databricks GTE-Large-EN  
**Governance:** All tools, data, and secrets managed through Unity Catalog

---

## Repository Structure

```
ai_agent_demo/
├── src/
│   ├── tools/
│   │   ├── messaging.py      # Slack / Teams webhook sender (auto-detected)
│   │   ├── search.py         # Vector Search retrieval
│   │   └── logger.py         # UC Delta table action logger
│   └── agent/
│       ├── prompts.py        # System prompt and tool schemas
│       └── pyfunc_model.py   # MLflow PyFunc orchestrator
└── notebooks/
    ├── 01_data_ingestion.py  # PDF → Delta → Vector Search index
    ├── 02_test_tools.py      # Isolated tool validation
    └── 03_deploy_agent.py    # UC model registration + serving endpoint
```

---

## Setup

### 1. Clone into Databricks Workspace

In a Databricks workspace terminal or via Repos:

```bash
git clone https://github.com/gulaag/databricks-AIagent.git
```

### 2. Upload Source Documents

Upload past Tech Engineer session PDFs to the Unity Catalog volume:

```
/Volumes/main/tech_engineer/session_documents/
```

### 3. Configure Unity Catalog Secrets

```bash
databricks secrets create-scope agent_secrets

# Incoming webhook for the test channel (Slack or Teams). Never committed.
databricks secrets put-secret agent_secrets slack_webhook_url
```

Auth to the LLM endpoint, Vector Search index, and SQL warehouse is handled by
M2M OAuth that Model Serving injects from the model's declared `resources` — so
no `databricks_token` secret is needed.

### 4. Run Notebooks in Order

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.py` | Parse PDF/Markdown/text, build Vector Search index |
| `02_test_tools.py` | Validate each tool independently |
| `03_deploy_agent.py` | Register model to UC, deploy serving endpoint |

---

## Key Databricks Features Used

| Feature | Role in this project |
|---|---|
| **Unity Catalog** | Governs the Vector Search index, Delta log table, and secrets |
| **Vector Search** | Semantic retrieval over past session transcripts and docs |
| **MLflow Tracing** | Records every LLM call, tool execution, and token count |
| **Model Serving** | Hosts the PyFunc agent as a scalable REST endpoint |
| **Foundation Models** | LLM inference without data leaving the corporate perimeter |

---

## Example Request

```
来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。
1時間枠のアジェンダを作成し、社内向けの案内文を作って、
テスト用Slackチャンネルに投稿してください。
```

The agent will:
1. Search past sessions for relevant Databricks and governance content
2. Draft a 1-hour agenda incorporating retrieved context
3. Post the announcement to the test channel (Slack or Teams)
4. Log all actions to `main.tech_engineer.agent_action_log`
