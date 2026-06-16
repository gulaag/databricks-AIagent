# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion & Vector Search Index Build
# MAGIC
# MAGIC **Purpose:** Process past Tech Engineer session PDFs and Databricks AI documentation
# MAGIC into a Unity Catalog Vector Search index.
# MAGIC
# MAGIC **Run order:** Execute cells top-to-bottom on a single-node cluster with
# MAGIC `databricks-vectorsearch` and `pypdf` installed.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch pypdf mlflow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
import mlflow

spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Configuration — update these values before running
# ---------------------------------------------------------------------------
CATALOG = "main"
SCHEMA = "tech_engineer"
SOURCE_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/session_documents"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
VS_ENDPOINT_NAME = "tech_engineer_vs_endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.sessions_vs_index"
DELTA_TABLE = f"{CATALOG}.{SCHEMA}.session_chunks"

# COMMAND ----------

# MAGIC %md ## Step 1 — Create Unity Catalog schema and volume (idempotent)

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.session_documents"
    " COMMENT 'Raw PDF uploads for Tech Engineer sessions'"
)

print(f"Schema ready: {CATALOG}.{SCHEMA}")
print(f"Upload PDFs to: {SOURCE_VOLUME}")

# COMMAND ----------

# MAGIC %md ## Step 2 — Parse PDFs and write chunks to Delta

# COMMAND ----------

from pypdf import PdfReader
import re


def extract_chunks_from_pdf(pdf_path: str, chunk_size: int = 500) -> list[dict]:
    """Split a PDF into fixed-size text chunks with metadata."""
    reader = PdfReader(pdf_path)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append(
            {
                "chunk_text": chunk_text,
                "source_file": Path(pdf_path).name,
                "session_date": None,
                "topic_tags": None,
            }
        )
    return chunks


all_chunks: list[dict] = []
pdf_files = list(Path(SOURCE_VOLUME).glob("*.pdf"))
print(f"Found {len(pdf_files)} PDF(s) in {SOURCE_VOLUME}")

for pdf_path in pdf_files:
    chunks = extract_chunks_from_pdf(str(pdf_path))
    all_chunks.extend(chunks)
    print(f"  {pdf_path.name}: {len(chunks)} chunks")

chunks_df = spark.createDataFrame(all_chunks).withColumn(
    "chunk_id", monotonically_increasing_id().cast("string")
)
chunks_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(DELTA_TABLE)

print(f"\nWrote {chunks_df.count()} total chunks to {DELTA_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Enable Change Data Feed (required for Vector Search sync)

# COMMAND ----------

spark.sql(
    f"ALTER TABLE {DELTA_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# COMMAND ----------

# MAGIC %md ## Step 4 — Create Vector Search endpoint and index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)

# Create endpoint if it doesn't exist
existing_endpoints = [e["name"] for e in vs_client.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT_NAME not in existing_endpoints:
    vs_client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created VS endpoint: {VS_ENDPOINT_NAME}")
else:
    print(f"VS endpoint already exists: {VS_ENDPOINT_NAME}")

# COMMAND ----------

# Create Delta Sync index (auto-syncs when the Delta table is updated)
try:
    vs_client.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME,
        source_table_name=DELTA_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    print(f"Created VS index: {VS_INDEX_NAME}")
except Exception as exc:
    if "already exists" in str(exc).lower():
        print(f"VS index already exists: {VS_INDEX_NAME}")
    else:
        raise

# COMMAND ----------

# MAGIC %md ## Step 5 — Trigger initial sync and verify

# COMMAND ----------

index = vs_client.get_index(index_name=VS_INDEX_NAME)
index.sync()
print("Sync triggered. Check status with: index.describe()")
print(index.describe())
