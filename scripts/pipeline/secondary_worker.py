#!/usr/bin/env python3
"""Distributed worker - build secondary (drill-down) TMAPs for a *list of cluster ids*.

This is the unit of work you fan out to make the K (~50k) per-cluster maps in
parallel. STEP 6 (`06_secondary_maps.py`) walks *contiguous bucket ranges* on
one machine; this script instead takes an explicit **list of cluster ids**, so
you can hand each node / process / array-task an arbitrary slice of the work
however your scheduler splits it. The per-cluster output is identical
(`cluster_<id>.html`, same Morgan/Jaccard layout + property colour layers as the
primary map), so the two are interchangeable and both feed the primary-map links.

Why DuckDB-direct (and not a CSV per cluster):
  The clusters already live in a hive-partitioned parquet store
  (`bucket=<cluster_id // BUCKET_SIZE>/`, zstd). DuckDB reads only the bucket
  files that hold the requested ids - no second full pass, no 50k throwaway
  CSVs, no extra disk. The expensive part of each map is the Morgan-FP + TMAP
  layout (~seconds/cluster), which dwarfs the I/O, so materialising CSVs would
  buy nothing and cost a full re-export. We therefore read straight from the
  partitioned store and group the requested ids by bucket so each bucket file is
  opened at most once.

How to distribute (we don't prescribe a scheduler):
  1. Get the full id list once, e.g.
       duckdb -noheader -list \\
         -c "SELECT DISTINCT cluster_id FROM read_parquet('clustered/**/*.parquet')" \\
         > all_ids.txt
     (or just run this script once with --all to discover + build them all).
  2. Split all_ids.txt into N chunks - contiguous or round-robin, your call -
     one chunk per worker (split(1), awk, your job array's task id, ...).
  3. Launch N copies of this script, each with its own --cluster-ids-file against
     the same --clustered store and --output folder. Workers share no state and
     are resumable (an existing cluster_<id>.html is skipped), so re-running a
     chunk only fills the gaps.

  The map build is CPU-bound (RDKit fingerprints + the in-memory HNSW/Jaccard
  layout), so scale by processes / cores / nodes - a GPU does not help here.

Usage:
    # one explicit slice (e.g. a SLURM array task or GNU parallel chunk):
    python scripts/pipeline/secondary_worker.py \
        --clustered OUT/clustered/ \
        --output    OUT/site/ \
        --cluster-ids-file chunk_007.txt \
        --max-points 2000

    # discover + build everything on one box (== STEP 6):
    python scripts/pipeline/secondary_worker.py \
        --clustered OUT/clustered/ \
        --output    OUT/site/ --all
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import duckdb

from _common import add_property_colors, fmt_time, log

MIN_POINTS = 3  # TMAP needs a few points to build a graph


def build_one(cid: int, df, out_dir: Path, max_points: int, k_neighbors: int) -> bool:
    """Build cluster_<cid>.html from a dataframe of (id, smiles). True if written.

    Same builder as Stage 7: a Morgan/ECFP + Jaccard layout (HNSW under the
    hood) exposes the *substructural* spread within an already-MQN-similar
    cluster, with the same property colour layers as the primary map.
    """
    from tmap import TMAP
    from tmap.utils import fingerprints_from_smiles

    out_html = out_dir / f"cluster_{cid}.html"
    if out_html.exists():
        return False
    if len(df) < MIN_POINTS:
        return False
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    smiles = df["smiles"].tolist()
    ids = df["id"].astype(str).tolist()

    fps = fingerprints_from_smiles(smiles, fp_type="morgan", n_bits=2048, radius=2)
    if len(fps) != len(smiles):
        # A dropped SMILES would desync fps from the id/SMILES labels; skip
        # rather than mislabel (rare - these already parsed for MQN in Stage 3).
        log(f"    cluster {cid}: {len(smiles)-len(fps)} SMILES dropped, skipping")
        return False

    n_neighbors = min(k_neighbors, len(smiles) - 1)
    viz = TMAP(metric="jaccard", n_neighbors=n_neighbors).fit(fps).to_tmapviz(include_edges=True)
    viz.title = f"Cluster {cid}"
    viz.background_color = "#FFFFFF"
    viz.add_smiles(smiles)
    viz.add_label("ID", ids)
    add_property_colors(viz, smiles)  # same colour axes as the primary map
    viz.write_html(str(out_html))
    return True


def load_cluster_ids(args, clustered: Path) -> list[int]:
    """Resolve cluster ids from --cluster-ids-file / --cluster-ids / --all."""
    if args.cluster_ids_file:
        text = Path(args.cluster_ids_file).read_text()
        ids = [int(tok) for tok in text.split() if tok.strip()]
        log(f"Loaded {len(ids):,} cluster ids from {args.cluster_ids_file}")
        return ids
    if args.cluster_ids:
        ids = [int(tok) for tok in args.cluster_ids.replace(",", " ").split()]
        log(f"Loaded {len(ids):,} cluster ids from --cluster-ids")
        return ids
    # --all: discover every cluster id present in the store.
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT cluster_id FROM read_parquet('{clustered}/**/*.parquet') "
        "ORDER BY cluster_id"
    ).fetchall()
    con.close()
    ids = [int(r[0]) for r in rows]
    log(f"Discovered {len(ids):,} cluster ids in {clustered}")
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clustered", required=True, help="Stage 4 bucketed parquet folder")
    p.add_argument("--output", required=True, help="Folder for cluster_<id>.html files")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cluster-ids-file", help="Text file of cluster ids (one per line)")
    src.add_argument("--cluster-ids", help="Comma/space separated cluster ids")
    src.add_argument("--all", action="store_true", help="Build every cluster in the store")
    p.add_argument("--bucket-size", type=int, default=100,
                   help="Must match Stage 4 (bucket = cluster_id // bucket_size). Default 100")
    p.add_argument("--max-points", type=int, default=2000,
                   help="Down-sample clusters larger than this (default 2000)")
    p.add_argument("--k-neighbors", type=int, default=20)
    args = p.parse_args()

    clustered = Path(args.clustered)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_ids = load_cluster_ids(args, clustered)
    if not cluster_ids:
        log("No cluster ids to build - nothing to do.")
        return

    # Group requested ids by their bucket so each bucket parquet is read once.
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for cid in cluster_ids:
        by_bucket[cid // args.bucket_size].append(cid)
    log(f"Building {len(cluster_ids):,} clusters across {len(by_bucket):,} buckets "
        f"-> {out_dir}")

    con = duckdb.connect()
    written = skipped = missing = 0
    t0 = time.perf_counter()

    for bi, bucket in enumerate(sorted(by_bucket)):
        bdir = clustered / f"bucket={bucket}"
        cids = by_bucket[bucket]
        if not bdir.exists():
            missing += len(cids)
            log(f"  bucket {bucket}: dir absent, skipping {len(cids)} requested ids")
            continue
        # Read only the requested clusters from this one bucket file.
        id_list = ",".join(str(c) for c in cids)
        df = con.execute(
            f"SELECT id, smiles, cluster_id FROM read_parquet('{bdir}/*.parquet') "
            f"WHERE cluster_id IN ({id_list})"
        ).df()
        for cid, group in df.groupby("cluster_id"):
            if build_one(int(cid), group, out_dir, args.max_points, args.k_neighbors):
                written += 1
            else:
                skipped += 1
        log(f"  bucket {bi+1}/{len(by_bucket)} (bucket={bucket})  "
            f"written={written:,} skipped={skipped:,}  {fmt_time(time.perf_counter()-t0)}")

    con.close()
    tail = f", {missing:,} ids in absent buckets" if missing else ""
    log(f"Done: {written:,} maps written, {skipped:,} skipped/existing{tail} "
        f"in {fmt_time(time.perf_counter()-t0)}")


if __name__ == "__main__":
    main()
