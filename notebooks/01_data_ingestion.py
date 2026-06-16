# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion & Vector Search Index Build
# MAGIC
# MAGIC **Purpose:** Process past Tech Engineer session PDFs and Databricks AI documentation
# MAGIC into a Unity Catalog Vector Search index.
# MAGIC
# MAGIC **Run order:** Execute cells top-to-bottom on a single-node cluster.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch pypdf mlflow tiktoken --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import hashlib
import os
import random
import re
import time
from pathlib import Path

import mlflow
import requests
import tiktoken
from pypdf import PdfReader
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

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

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64
CHUNK_VERSION = "v1"

CHUNKS_SCHEMA = StructType([
    StructField("chunk_id", StringType(), False),
    StructField("chunk_text", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("chunk_index", IntegerType(), True),
    StructField("session_date", StringType(), True),
    StructField("topic_tags", StringType(), True),
    StructField("version", StringType(), True),
])

DATABRICKS_HOST = spark.conf.get("spark.databricks.workspaceUrl")
DATABRICKS_TOKEN = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

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


def normalize_text(text: str) -> str:
    """Strip ANSI escape codes and non-printable control characters."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def extract_chunks_from_pdf(
    pdf_path: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[dict]:
    """Split a PDF into token-aware overlapping chunks with deterministic IDs."""
    reader = PdfReader(pdf_path)
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    full_text = normalize_text(raw_text)

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full_text)
    stride = chunk_tokens - overlap_tokens
    source_file = Path(pdf_path).name

    chunks = []
    for chunk_index, start in enumerate(range(0, len(tokens), stride)):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        chunk_text = enc.decode(window)
        chunk_id = hashlib.sha256(
            f"{source_file}|{chunk_index}|{chunk_text[:120]}|{CHUNK_VERSION}".encode()
        ).hexdigest()
        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "session_date": None,
                "topic_tags": None,
                "version": CHUNK_VERSION,
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

chunks_df = spark.createDataFrame(all_chunks, schema=CHUNKS_SCHEMA)

# COMMAND ----------

# MAGIC %md ### Pre-write data quality checks

# COMMAND ----------

null_count = chunks_df.filter(
    col("chunk_id").isNull() | col("chunk_text").isNull()
).count()
assert null_count == 0, f"Data quality failure: {null_count} rows with null PK or chunk_text"

dup_count = (
    chunks_df.groupBy("chunk_id").count().filter(col("count") > 1).count()
)
assert dup_count == 0, f"Data quality failure: {dup_count} duplicate chunk_ids in this batch"

print(f"Quality checks passed. {chunks_df.count()} chunks ready to merge.")

# COMMAND ----------

# MAGIC %md ### Create Delta table (idempotent) and MERGE chunks

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {DELTA_TABLE} (
  chunk_id     STRING NOT NULL,
  chunk_text   STRING,
  source_file  STRING,
  chunk_index  INT,
  session_date STRING,
  topic_tags   STRING,
  version      STRING
)
USING DELTA
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

chunks_df.createOrReplaceTempView("new_chunks_staging")

spark.sql(f"""
MERGE INTO {DELTA_TABLE} AS target
USING new_chunks_staging AS source
ON target.chunk_id = source.chunk_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
""")

print(f"MERGE complete. Table: {DELTA_TABLE}")

# COMMAND ----------

# MAGIC %md ## Step 3 — Create Vector Search endpoint and index

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)

existing_endpoints = [e["name"] for e in vs_client.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT_NAME not in existing_endpoints:
    vs_client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created VS endpoint: {VS_ENDPOINT_NAME}")
else:
    print(f"VS endpoint already exists: {VS_ENDPOINT_NAME}")

# COMMAND ----------

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

# MAGIC %md ## Step 4 — Trigger sync and poll until online

# COMMAND ----------


def _retryable_call(fn, retries: int = 8, base_sleep: float = 1.5, max_sleep: float = 20.0):
    """Call fn with exponential backoff + jitter. Raises on final failure."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            sleep_time = min(base_sleep * (2 ** attempt) + random.uniform(0, 1), max_sleep)
            print(f"Attempt {attempt + 1} failed ({exc}). Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)


def _trigger_vs_sync(host: str, token: str, index_name: str) -> dict:
    """POST directly to the VS sync REST endpoint (bypasses SDK hang issue)."""
    url = f"https://{host}/api/2.0/vector-search/indexes/{index_name.replace('.', '/')}/sync"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_vs_status(host: str, token: str, index_name: str, timeout_s: int = 600) -> None:
    """Poll VS index status until ONLINE or timeout. Handles both API response shapes."""
    url = f"https://{host}/api/2.0/vector-search/indexes/{index_name.replace('.', '/')}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=30
        )
        resp.raise_for_status()
        body = resp.json()
        status = (
            body.get("status", {}).get("detailed_state")
            or body.get("result", {}).get("status", "UNKNOWN")
        )
        print(f"VS index status: {status}")
        if "ONLINE" in str(status).upper():
            print(f"VS index is online: {VS_INDEX_NAME}")
            return
        time.sleep(15)
    raise TimeoutError(f"VS index did not come online within {timeout_s}s")


_retryable_call(lambda: _trigger_vs_sync(DATABRICKS_HOST, DATABRICKS_TOKEN, VS_INDEX_NAME))
_poll_vs_status(DATABRICKS_HOST, DATABRICKS_TOKEN, VS_INDEX_NAME)
