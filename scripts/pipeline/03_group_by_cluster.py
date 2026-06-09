#!/usr/bin/env python3
"""STEP 3 - Group molecules by cluster with DuckDB.

STEP 2 (02_assign_clusters.py) wrote parquets of (id, smiles, cluster_id) in arbitrary
cluster order. To build one map per cluster we need each cluster's molecules
together. We use DuckDB to rewrite everything sorted by cluster_id and
Hive-partitioned into buckets.

Why buckets instead of one directory per cluster?
    50k-100k clusters means 50k-100k output directories/files, which exhausts
    file handles and OOMs the writer. Instead we partition by
    `cluster_id // bucket_size` (default 100 -> ~500-1000 buckets) and keep the
    data sorted by cluster_id inside each bucket, so querying a single cluster
    still skips almost everything via parquet row-group statistics.

Query one cluster later:
    SELECT id, smiles
    FROM read_parquet('OUT/bucket=*/*.parquet', hive_partitioning=true)
    WHERE cluster_id = 4242

Usage:
    python scripts/pipeline/03_group_by_cluster.py \
        --labels OUT/labels/ \
        --output OUT/clustered/ \
        --memory-limit 32GB
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import duckdb

from _common import fmt_time, log


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True, help="Folder of (id,smiles,cluster_id) parquets")
    p.add_argument("--output", required=True, help="Hive-partitioned output folder")
    p.add_argument("--bucket-size", type=int, default=100,
                   help="Clusters per bucket directory (default 100)")
    p.add_argument("--memory-limit", default="32GB")
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    pattern = str(Path(args.labels) / "*.parquet")

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute("SET preserve_insertion_order = false")  # allow streaming/spilling

    n_rows = con.execute(f"SELECT count(*) FROM read_parquet('{pattern}')").fetchone()[0]
    log(f"Partitioning {n_rows:,} rows by cluster_id (bucket size {args.bucket_size})")

    t0 = time.perf_counter()
    con.execute(f"""
        COPY (
            SELECT id, smiles, cluster_id,
                   (cluster_id // {args.bucket_size}) AS bucket
            FROM read_parquet('{pattern}')
            ORDER BY cluster_id
        ) TO '{out}' (
            FORMAT PARQUET,
            PARTITION_BY (bucket),
            OVERWRITE_OR_IGNORE,
            COMPRESSION 'zstd',
            ROW_GROUP_SIZE 100000
        )
    """)
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)

    n_buckets = sum(1 for d in out.iterdir() if d.is_dir() and d.name.startswith("bucket="))
    log(f"Done: {n_buckets:,} buckets in {fmt_time(time.perf_counter()-t0)} -> {out}")


if __name__ == "__main__":
    main()
