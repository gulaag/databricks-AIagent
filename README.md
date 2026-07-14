# Databricks Action Agent — Tech Engineer Study Group

Internal tech-validation demo: an **action agent** on Databricks (not FAQ/RAG chat).
Given a plain-language request, the agent searches past session knowledge, drafts a
Japanese 1-hour study-session agenda, asks for approval (Playground default), posts
to a test Slack/Teams channel via webhook, and records execution via MLflow Tracing
(and a UC Delta audit table where the runtime identity allows writes).

**GitHub:** https://github.com/gulaag/databricks-AIagent  
**Serving endpoint:** `tech-engineer-agent-endpoint`  
**UC model:** `main.tech_engineer.tech_engineer_agent`

---

## FAQ/RAG vs this agent

| | FAQ / RAG chatbot | This action agent |
|---|---|---|
| Main output | Answer text | **Business action** (channel post) |
| Tools | Retrieve (+ generate) | Retrieve → draft → **post** → audit/trace |
| Human control | Usually none | Playground default: draft → ask → post on approval |
| Success metric | Answer quality | Did the announcement land in the test channel? |

---

## Architecture

```
User (AI Playground / notebook / REST)
    │
    ▼
Model Serving  tech-engineer-agent-endpoint
    │
    ▼
TechEngineerAgent  (MLflow ResponsesAgent)
    │
    ├── search_knowledge_base ──► Vector Search  main.tech_engineer.sessions_vs_index
    ├── post_to_channel ────────► Slack / Teams incoming webhook (secret)
    └── log_agent_action ───────► UC Delta  main.tech_engineer.agent_action_log
                                         │
                                         └── MLflow Tracing (spans for LLM + tools)
```

| Piece | Value |
|---|---|
| LLM | `databricks-meta-llama-3-3-70b-instruct` |
| Embeddings | `databricks-gte-large-en` |
| Modes | `chat` (endpoint default), `auto`, `draft`, `send` |
| Auth at serving | Automatic auth passthrough via MLflow `resources` (M2M OAuth) |
| Webhook secret | `agent_secrets/slack_webhook_url` |

### Modes

| Mode | Behavior |
|---|---|
| `chat` | **Default for Playground / endpoint.** Search → draft → ask → post only after explicit approval |
| `auto` | Hands-off: search → draft → post → log in one call |
| `draft` | Propose / refine only (no post) |
| `send` | Post an already-approved text (WYSIWYG) + audit |

---

## Repository structure

```
├── agent_entry.py              # MLflow models-from-code entry (set_model)
├── src/
│   ├── agent/
│   │   ├── agent.py            # TechEngineerAgent (ResponsesAgent)
│   │   └── prompts.py          # System prompts + tool schemas
│   └── tools/
│       ├── search.py           # Vector Search retrieval
│       ├── messaging.py        # Slack / Teams webhook poster
│       ├── logger.py           # UC Delta audit writer
│       └── guardrails.py       # Secret scan + duplicate-post protection
└── notebooks/
    ├── 01_data_ingestion.py    # Docs → Delta chunks → Vector Search index
    ├── 02_test_tools.py        # Isolated tool checks
    ├── 03_deploy_agent.py      # Register UC model + deploy / update endpoint
    └── 04_agent_demo.py        # Notebook demo (chat / auto / draft / send)
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/gulaag/databricks-AIagent.git
```

Workspace Repo path used in this project: `/Users/digvijay@arsaga.jp/databricks-Aiagent`

### 2. Source documents

Upload PDFs / Markdown into:

```
/Volumes/main/tech_engineer/session_documents/
```

### 3. Secrets

```bash
databricks secrets create-scope agent_secrets
databricks secrets put-secret agent_secrets slack_webhook_url
```

LLM, Vector Search, and SQL warehouse auth at serving time come from declared
MLflow `resources` (no PAT required for those).

### 4. Run notebooks in order

| Notebook | Purpose |
|---|---|
| `01_data_ingestion.py` | Build `session_chunks` + Vector Search index |
| `02_test_tools.py` | Validate search / messaging / logger tools |
| `03_deploy_agent.py` | Log model to UC, deploy `tech-engineer-agent-endpoint` |
| `04_agent_demo.py` | End-to-end demo (approval gate + autonomous) |

Then open **AI Playground** against `tech-engineer-agent-endpoint` for the front-end demo.

---

## Example request

```
来週のTech Engineer共有会で「Databricks AI Agent入門」をテーマにしたいです。
1時間枠のアジェンダを作成し、社内向けの案内文を作って、
テスト用Slackチャンネルに投稿してください。
```

Expected Playground flow (`chat` mode):

1. Search the knowledge index  
2. Show a Japanese draft agenda / announcement (with `[仮]` placeholders if needed)  
3. Ask for confirmation  
4. On approval → post to the test channel  
5. Record the run in MLflow Tracing (and Delta audit when runtime grants allow)

---

## Known limitations (demo / future work)

1. **Serving-side Delta audit writes (`MODIFY`)**  
   Model Serving automatic auth grants the system runtime SP **`SELECT` only** on
   `DatabricksTable` resources. `INSERT` into `agent_action_log` from the endpoint
   therefore fails with `PERMISSION_DENIED ... MODIFY` unless a grantable service
   principal / OBO path is set up (often needs admin once).  
   **Demo workaround:** use MLflow Tracing on serving; show Delta log rows from the
   notebook path (runs as the user).

2. **Calendar / Teams meeting invites**  
   Not implemented. Current integration is **incoming webhook message** only.
   Real calendar invites need Microsoft Graph (or similar) app registration and
   typically admin consent → tracked as future work.

3. **Knowledge quality**  
   Index is mostly scraped Databricks docs plus a thin session KB; retrieval can
   include noisy chunks. Good enough for demo; not production RAG ops.

---

## Key Databricks features used

| Feature | Role |
|---|---|
| **Unity Catalog** | Models, Vector Search index, Delta tables, secrets governance |
| **Vector Search** | Semantic retrieval over session / docs chunks |
| **MLflow ResponsesAgent + Tracing** | Agent interface, Playground-ready I/O, span-level observability |
| **Model Serving** | Hosts the agent as a scalable endpoint (AI Playground front door) |
| **Foundation Model APIs** | Llama 3.3 70B instruct for tool-calling loop |

---

## Manager deliverables (this validation)

| Deliverable | Status / where |
|---|---|
| Working demo | Endpoint + Playground + Slack test channel |
| Demo overview | Slides (to create) + this README |
| Architecture / flow diagrams | Slides (to create); ASCII diagram above |
| Pros / cons of Databricks agents | Slides (to create) |
| Real-project caveats | Slides + **Known limitations** above |
