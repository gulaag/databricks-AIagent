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


@mlflow.trace(name="search_knowledge_base", span_type="RETRIEVER")
def search_knowledge_base(
    query: str,
    index_name: str,
    num_results: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Query the Databricks Vector Search index for relevant context chunks.

    Args:
        query: The natural-language search query from the agent.
        index_name: Fully-qualified UC index name, e.g.
            ``catalog.schema.tech_engineer_sessions_index``.
        num_results: Maximum number of chunks to return (default 5).
        filters: Optional dictionary of metadata filters applied server-side,
            e.g. ``{"source_type": "session_transcript"}``.

    Returns:
        A list of result dicts, each containing at minimum:
        ``{"chunk_text": str, "score": float, "metadata": dict}``.
        Returns an empty list on error so the agent can continue gracefully.
    """
    try:
        client = VectorSearchClient(disable_notice=True)
        index = client.get_index(index_name=index_name)

        query_kwargs: dict[str, Any] = {
            "query_text": query,
            "columns": ["chunk_text", "source_file", "session_date", "topic_tags"],
            "num_results": num_results,
        }
        if filters:
            query_kwargs["filters_json"] = filters

        raw = index.similarity_search(**query_kwargs)

        results = []
        for hit in raw.get("result", {}).get("data_array", []):
            columns = raw.get("manifest", {}).get("columns", [])
            col_names = [c.get("name") for c in columns]
            row = dict(zip(col_names, hit))
            results.append(
                {
                    "chunk_text": row.get("chunk_text", ""),
                    "score": row.get("score", 0.0),
                    "metadata": {
                        "source_file": row.get("source_file"),
                        "session_date": row.get("session_date"),
                        "topic_tags": row.get("topic_tags"),
                    },
                }
            )
        return results

    except Exception as exc:  # noqa: BLE001
        mlflow.log_param("search_error", str(exc))
        return []
