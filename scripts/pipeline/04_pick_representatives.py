#!/usr/bin/env python3
"""STEP 4 - Pick one representative molecule per cluster.

The primary map shows one point per cluster. We choose, for each cluster, the
molecule closest to that cluster's MQN median (a stable "most typical member").

Because STEP 3 wrote the data sorted by cluster_id, each cluster is a
contiguous block. We stream the buckets in order and keep the first <=N
molecules per cluster (N=500 is plenty to estimate a median), compute their
MQN, and pick the nearest-to-median per cluster.

Output: a CSV with columns  id, smiles, cluster_id  (one row per cluster).

Usage:
    python scripts/pipeline/04_pick_representatives.py \
        --clustered OUT/clustered/ \
        --output OUT/representatives.csv
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _common import fmt_time, log, mqn_aligned

SAMPLE_PER_CLUSTER = 500
BATCH = 1_000_000


def sorted_parquet_files(clustered_dir: Path) -> list[Path]:
    """All parquet files, ordered by bucket number so the stream is cluster-sorted."""
    buckets = sorted(clustered_dir.glob("bucket=*"),
                     key=lambda d: int(d.name.split("=")[1]))
    files: list[Path] = []
    for b in buckets:
        files.extend(sorted(b.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No bucket=*/*.parquet under {clustered_dir}")
    return files


def sample_per_cluster(clustered_dir: Path, n_per_cluster: int) -> pd.DataFrame:
    """Stream cluster-sorted parquet, keep first <=N rows per cluster_id."""
    files = sorted_parquet_files(clustered_dir)
    log(f"  Streaming {len(files)} parquet files ...")

    seen: dict[int, int] = {}          # cluster_id -> rows collected so far
    ids: list[str] = []
    smiles: list[str] = []
    cids: list[int] = []
    total = 0

    for fi, fpath in enumerate(files):
        for batch in pq.ParquetFile(fpath).iter_batches(
                batch_size=BATCH, columns=["id", "smiles", "cluster_id"]):
            cid_arr = batch.column("cluster_id").to_numpy()
            total += len(cid_arr)

            # cluster_id is sorted, so clusters appear as contiguous segments.
            starts = np.concatenate([[0], np.where(np.diff(cid_arr) != 0)[0] + 1])
            ends = np.concatenate([starts[1:], [len(cid_arr)]])

            id_col, smi_col = batch.column("id"), batch.column("smiles")
            for s, e in zip(starts, ends):
                cid = int(cid_arr[s])
                remaining = n_per_cluster - seen.get(cid, 0)
                if remaining <= 0:
                    continue
                take = min(e - s, remaining)
                for j in range(s, s + take):
                    ids.append(id_col[j].as_py())
                    smiles.append(smi_col[j].as_py())
                    cids.append(cid)
                seen[cid] = seen.get(cid, 0) + take

        if (fi + 1) % 50 == 0 or fi == len(files) - 1:
            log(f"    {fi+1}/{len(files)} files  read={total:,}  "
                f"sampled={len(ids):,}  clusters={len(seen):,}")

    return pd.DataFrame({"id": ids, "smiles": smiles, "cluster_id": cids})


def pick_representatives(df: pd.DataFrame, nproc: int) -> pd.DataFrame:
    """For each cluster, the molecule nearest to the cluster's MQN median."""
    log("  Computing MQN for sampled molecules ...")
    idx, fps = mqn_aligned(df["smiles"].tolist(), nproc)
    df = df.iloc[idx].reset_index(drop=True)          # keep only molecules that parsed
    fps = fps.astype(np.float32)

    log(f"  Selecting representatives across {df['cluster_id'].nunique():,} clusters ...")
    cids = df["cluster_id"].to_numpy()
    ids = df["id"].to_numpy()
    smi = df["smiles"].to_numpy()

    reps: list[tuple[str, str, int]] = []
    for cid in np.unique(cids):
        mask = cids == cid
        cfps = fps[mask]
        median = np.median(cfps, axis=0)
        best = np.argmin(np.sum((cfps - median) ** 2, axis=1))
        reps.append((str(ids[mask][best]), str(smi[mask][best]), int(cid)))

    return pd.DataFrame(reps, columns=["id", "smiles", "cluster_id"]).sort_values("cluster_id")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clustered", required=True, help="Stage 4 bucketed output folder")
    p.add_argument("--output", required=True, help="representatives.csv path")
    p.add_argument("--per-cluster", type=int, default=SAMPLE_PER_CLUSTER)
    p.add_argument("--nproc", type=int, default=None)
    args = p.parse_args()

    t0 = time.perf_counter()
    log("Step 1: sample molecules per cluster ...")
    df = sample_per_cluster(Path(args.clustered), args.per_cluster)

    log("Step 2: pick representatives ...")
    reps = pick_representatives(df, args.nproc or os.cpu_count())

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    reps.to_csv(args.output, index=False)
    log(f"Saved {len(reps):,} representatives -> {args.output} "
        f"({fmt_time(time.perf_counter()-t0)})")


if __name__ == "__main__":
    main()
