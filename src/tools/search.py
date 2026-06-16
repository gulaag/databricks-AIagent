"""
Databricks Vector Search retrieval tool.

Queries a Unity Catalog-governed Vector Search index to retrieve
semantically similar chunks from past Tech Engineer sessions and
Databricks AI documentation.
"""

from __future__ import annotations

from typing import Any

import mlflow
from databricks.vector_search.client import VectorSearchClient


def _probe_index_columns(index: Any) -> list[str]:
    """Detect available column names from the VS index at runtime.

    Tries index.describe() first; falls back to a 1-result probe query.
    This prevents hard-coded column lists from 400-ing on schema changes.
    """
    try:
        desc = index.describe()
        cols = (
            desc.get("delta_sync_index_spec", {}).get("columns_to_sync") or []
        )
        if cols:
            return [c["name"] for c in cols if "name" in c]
    except Exception:  # noqa: BLE001
        pass

    try:
        probe = index.similarity_search(query_text="probe", num_results=1)
        manifest_cols = probe.get("manifest", {}).get("columns", [])
        return [c.get("name") for c in manifest_cols if c.get("name")]
    except Exception:  # noqa: BLE001
        return ["chunk_text", "source_file", "chunk_index", "session_date", "topic_tags"]


def _parse_vs_results(raw: dict, col_names: list[str]) -> list[dict[str, Any]]:
    """Convert a raw VS API response into a normalised list of result dicts."""
    results = []
    for hit in raw.get("result", {}).get("data_array", []):
        row = dict(zip(col_names, hit))
        chunk_text = row.pop("chunk_text", "")
        score = row.pop("score", 0.0)
        results.append(
            {
                "chunk_text": chunk_text,
                "score": score,
                "metadata": {k: v for k, v in row.items()},
            }
        )
    return results


@mlflow.trace(name="search_knowledge_base", span_type="RETRIEVER")
def search_knowledge_base(
    query: str,
    index_name: str,
    num_results: int = 5,
    similarity_threshold: float = 0.6,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Query the Databricks Vector Search index for relevant context chunks.

    Args:
        query: The natural-language search query from the agent.
        index_name: Fully-qualified UC index name, e.g.
            ``catalog.schema.tech_engineer_sessions_index``.
        num_results: Maximum number of chunks to return (default 5).
        similarity_threshold: Minimum similarity score 0.0–1.0 (default 0.6).
            If no results meet this threshold, retries without it and flags
            results with ``metadata["fallback_retrieval"] = True``.
        filters: Optional dict of metadata filters applied server-side.

    Returns:
        A list of result dicts, each containing:
        ``{"chunk_text": str, "score": float, "metadata": dict}``.
        Returns an empty list on unrecoverable error.
    """
    try:
        client = VectorSearchClient(disable_notice=True)
        index = client.get_index(index_name=index_name)

        col_names = _probe_index_columns(index)

        query_kwargs: dict[str, Any] = {
            "query_text": query,
            "columns": col_names,
            "num_results": num_results,
            "score_threshold": similarity_threshold,
        }
        if filters:
            query_kwargs["filters_json"] = filters

        raw = index.similarity_search(**query_kwargs)
        results = _parse_vs_results(raw, col_names)

        if not results and similarity_threshold > 0.0:
            fallback_kwargs = {k: v for k, v in query_kwargs.items() if k != "score_threshold"}
            raw = index.similarity_search(**fallback_kwargs)
            results = _parse_vs_results(raw, col_names)
            for r in results:
                r["metadata"]["fallback_retrieval"] = True

        return results

    except Exception as exc:  # noqa: BLE001
        # No mlflow.log_param here: at serving time there is no active run, so
        # logging a param would itself raise. The @mlflow.trace span captures
        # the exception context; we degrade gracefully to an empty result set.
        print(f"search_knowledge_base error: {exc}")
        return []
