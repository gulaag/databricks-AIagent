"""
Conversational, plan-and-execute agent for the Tech Engineer Study Group.

Pattern: propose -> refine -> approve -> act (not a Q&A chatbot).

  - propose(request): retrieve grounded context, then draft a COMPLETE announcement.
    The agent proposes missing details (date/time/venue) with a "[仮]" marker rather
    than interrogating the user.
  - refine(feedback): revise the current draft from natural-language feedback. Call
    as many times as needed; state is kept on the instance.
  - send(): post the approved draft to the channel and write an audit log.

Orchestration (search -> draft -> post -> log) is deterministic and reliable; the
LLM provides the intelligence (planning, drafting, revising). This is intentionally
robust for a live demo.

Auth: the LLM call uses mlflow.deployments (notebook credentials locally; injected
M2M OAuth if ever run inside serving). Vector Search / SQL use the same ambient auth.
"""

from __future__ import annotations

from typing import Any

from src.tools.logger import log_agent_action
from src.tools.messaging import post_to_channel
from src.tools.search import search_knowledge_base

# The fixed workflow the agent follows — shown to the user as its "plan".
WORKFLOW_STEPS = [
    "依頼を理解し、必要な作業を分解する",
    "過去セッション資料を Vector Search で検索する",
    "1時間枠のアジェンダ＋案内文を作成する（不足項目は [仮] として提案）",
    "内容を確認・修正する（人間が承認）",
    "承認後、Slack へ投稿し、実行ログを保存する",
]

_PROPOSE_SYSTEM = """You are a planning assistant for the int.[CoE] Tech Engineer Study Group.

From the user's request and the reference context, PROPOSE a complete, ready-to-review
session announcement. Write in Japanese unless the request is in English and asks for English.

Include:
- タイトル
- 開催概要（日時・場所・対象者）。未指定の項目は妥当な候補を *提案* し「[仮]」と明記する
  （例: 「日時: [仮] 来週木曜 18:00–19:00 / 会場: [仮] 5F会議室（Zoom併用）」）。
- 1時間枠のタイムテーブル付きアジェンダ
- 過去セッションを踏まえた「議論トピック案」

Rules:
- Propose, do NOT interrogate. Never ask the user questions. Fill any gaps with sensible
  "[仮]" suggestions they can change later.
- Ground claims in the reference context and cite as [Source: <file>]. Never invent facts
  that are not in the context. If the context says no past material was found, draft from
  general Databricks knowledge and keep it conservative (no fabricated citations).
- Slack-friendly formatting: emoji section markers (例: 📅, 🕐, 📝), "・" bullets, and
  *single asterisks* for emphasis. Do NOT use Markdown headings (#), tables, or **double
  asterisks** — Slack renders them literally.
- Output ONLY the announcement text — no preamble, no explanation, no mention of tools.
"""

_REFINE_SYSTEM = """You are revising a draft session announcement based on the user's feedback.

Apply the requested changes and return the COMPLETE updated announcement — not a diff and
not a summary of what changed. Keep the same language and the Slack-friendly formatting
(emoji markers, "・" bullets, *single asterisks*; no Markdown headings/tables). Preserve
[Source: <file>] citations where still relevant. If the user specifies a value for a "[仮]"
placeholder, replace it with their value and remove the "[仮]" mark.

Output ONLY the updated announcement text — no preamble, no explanation.
"""


class ConversationalAgent:
    """Stateful propose -> refine -> approve -> act agent for notebook-driven use."""

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        log_table: str,
        webhook: str,
        warehouse_id: str | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._index_name = index_name
        self._log_table = log_table
        self._webhook = webhook
        self._warehouse_id = warehouse_id or None

        self._current_draft: str | None = None
        self._sources: list[str] = []
        self._last_request: str | None = None

    # -- LLM helper -----------------------------------------------------------
    def _chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
        """Call the Databricks Foundation Model endpoint for a chat completion."""
        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        resp = client.predict(
            endpoint=self._endpoint,
            inputs={
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        content = resp["choices"][0]["message"].get("content") or ""
        return content.strip()

    # -- 1. Propose -----------------------------------------------------------
    def propose(self, request: str) -> dict[str, Any]:
        """Search for context, then draft a complete announcement. Posts nothing.

        Returns ``{"steps": [...], "draft": str, "sources": [...]}``.
        """
        self._last_request = request

        results = search_knowledge_base(
            query=request, index_name=self._index_name, num_results=5
        )
        self._sources = []
        context_parts: list[str] = []
        for r in results:
            meta = r.get("metadata") or {}
            src = meta.get("source_file")
            if src and src not in self._sources:
                self._sources.append(src)
            context_parts.append(f"[{src or 'unknown'}] {r.get('chunk_text', '')}")
        context = (
            "\n\n".join(context_parts)
            if context_parts
            else "(関連する過去セッション資料は見つかりませんでした。)"
        )

        user_msg = f"依頼:\n{request}\n\n参考コンテキスト:\n{context}"
        draft = self._chat(
            [
                {"role": "system", "content": _PROPOSE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        self._current_draft = draft
        return {"steps": WORKFLOW_STEPS, "draft": self._current_draft, "sources": self._sources}

    # -- 2. Refine ------------------------------------------------------------
    def refine(self, feedback: str) -> dict[str, Any]:
        """Revise the current draft from natural-language feedback.

        Returns ``{"draft": str}`` (or a message if there is no draft yet).
        """
        if not self._current_draft:
            return {"draft": None, "message": "先に propose(...) を実行してください。"}
        if not feedback or not feedback.strip():
            return {"draft": self._current_draft, "message": "修正指示が空のため、変更はありません。"}

        user_msg = f"現在の案:\n{self._current_draft}\n\n修正指示:\n{feedback}"
        revised = self._chat(
            [
                {"role": "system", "content": _REFINE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        self._current_draft = revised
        return {"draft": self._current_draft}

    # -- 3. Send (act) --------------------------------------------------------
    def send(self) -> dict[str, str]:
        """Post the current (approved) draft and write an audit log."""
        if not self._current_draft:
            return {
                "post_status": "ERROR: no draft to send — run propose() first.",
                "log_status": "skipped",
            }

        post_status = post_to_channel(message=self._current_draft, webhook_url=self._webhook)
        log_status = log_agent_action(
            action_name="post_to_channel",
            input_payload={
                "request": self._last_request,
                "draft_preview": self._current_draft[:500],
            },
            output_payload={"result": post_status, "sources": self._sources},
            table_name=self._log_table,
            status="SUCCESS" if post_status.startswith("SUCCESS") else "FAILURE",
            warehouse_id=self._warehouse_id,
        )
        return {"post_status": post_status, "log_status": log_status}

    # -- convenience ----------------------------------------------------------
    @property
    def current_draft(self) -> str | None:
        return self._current_draft
