#!/usr/bin/env python3
"""STEP 6 - Build one secondary TMAP per cluster (the drill-down maps).

For each cluster we pull its members (via DuckDB, one bucket at a time so each
bucket is read only once), optionally down-sample to keep the HTML light, and
lay them out with Morgan/ECFP + Jaccard. Within a cluster the molecules are
already MQN-similar, so an ECFP layout exposes the *substructural* detail that
MQN can't - which is what you want when eyeballing a cluster.

Each map is written as `cluster_<id>.html`, matching the link the primary map
points to (STEP 5). Skips clusters whose HTML already exists, so it is
resumable; use --start-bucket/--end-bucket to split the work across processes.

Note: this is by far the longest step - one HTML per cluster (tens of
thousands of files at K=50k). Down-sampling (--max-points) keeps each file
small; the public site caps clusters at ~800 points for the same reason.

Usage:
    python scripts/pipeline/06_secondary_maps.py \
        --clustered OUT/clustered/ \
        --output OUT/site/ \
        --max-points 2000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from _common import add_property_colors, fmt_time, log

MIN_POINTS = 3  # TMAP needs a few points to build a graph


def build_one(cid: int, df, out_dir: Path, max_points: int, k_neighbors: int) -> bool:
    """Build cluster_<cid>.html from a dataframe of (id, smiles). True if written."""
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

    # Morgan/ECFP + Jaccard: substructural similarity within the cluster.
    fps = fingerprints_from_smiles(smiles, fp_type="morgan", n_bits=2048, radius=2)
    if len(fps) != len(smiles):
        # A dropped SMILES would desync the fps from the id/SMILES labels; skip
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clustered", required=True, help="Stage 4 bucketed output folder")
    p.add_argument("--output", required=True, help="Folder for cluster_<id>.html files")
    p.add_argument("--max-points", type=int, default=2000,
                   help="Down-sample clusters larger than this (default 2000)")
    p.add_argument("--k-neighbors", type=int, default=20)
    p.add_argument("--start-bucket", type=int, default=0)
    p.add_argument("--end-bucket", type=int, default=None,
                   help="Exclusive; process buckets [start, end) for parallel runs")
    args = p.parse_args()

    clustered = Path(args.clustered)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = sorted(clustered.glob("bucket=*"), key=lambda d: int(d.name.split("=")[1]))
    end = args.end_bucket if args.end_bucket is not None else len(buckets)
    buckets = [b for b in buckets if args.start_bucket <= int(b.name.split("=")[1]) < end]
    log(f"Generating secondary maps for {len(buckets)} buckets -> {out_dir}")

    con = duckdb.connect()
    written = skipped = 0
    t0 = time.perf_counter()

    for bi, bucket in enumerate(buckets):
        # Read one bucket (~100 clusters) once, then split by cluster in pandas.
        df = con.execute(
            f"SELECT id, smiles, cluster_id FROM read_parquet('{bucket}/*.parquet')"
        ).df()
        for cid, group in df.groupby("cluster_id"):
            if build_one(int(cid), group, out_dir, args.max_points, args.k_neighbors):
                written += 1
            else:
                skipped += 1

        log(f"  bucket {bi+1}/{len(buckets)} ({bucket.name})  "
            f"written={written:,} skipped={skipped:,}  {fmt_time(time.perf_counter()-t0)}")

    con.close()
    log(f"Done: {written:,} maps written, {skipped:,} skipped/existing "
        f"in {fmt_time(time.perf_counter()-t0)}")


if __name__ == "__main__":
    main()
