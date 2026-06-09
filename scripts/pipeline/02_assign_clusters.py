"""Assign a cluster to every molecule, using pre-trained PQEncoder + PQKMeans.

Reads any text file (ONE SMILES per line) or a folder of such files, computes
MQN fingerprints, encodes them to PQ codes, predicts the cluster of each
molecule, and writes parquet files with three columns:

    id (string) | smiles (string) | cluster_id (int32)

The id travels WITH each molecule, so a label is never mixed up - even when a
SMILES fails to parse, that row is dropped together with its id (no silent
shift). If your file has no id column we make one up (the molecule's line
number). If it has one, point --id-col at it (and --smiles-col at the SMILES).

Uses the GPU for the heavy steps when available (--device auto).

In this pipeline this is STEP 2 (assign): the full pass over ALL molecules,
using the models trained in STEP 1 (01_train_models.py).

Usage:
    # plain SMILES file (ids are made up = line numbers):
    python scripts/pipeline/02_assign_clusters.py --input molecules.smi --output out/ \
        --encoder models/encoder.joblib --clusterer models/clusterer.joblib

    # input is a folder of files, "id<TAB>smiles" (keep the real ids):
    python scripts/pipeline/02_assign_clusters.py --input shards/ --output out/ \
        --encoder models/encoder.joblib --clusterer models/clusterer.joblib \
        --smiles-col 1 --id-col 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chelombus import PQEncoder, DataStreamer, FingerprintCalculator
from chelombus.clustering.PyQKmeans import PQKMeans


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Assign clusters to SMILES with pre-trained models")
    parser.add_argument("--input", required=True, help="SMILES file, or a folder of files")
    parser.add_argument("--output", required=True, help="Output folder for parquet files")
    parser.add_argument("--encoder", default="models/encoder.joblib")
    parser.add_argument("--clusterer", default="models/clusterer.joblib")
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="Molecules processed (and saved) per parquet file")
    parser.add_argument("--smiles-col", type=int, default=0,
                        help="0-based column holding the SMILES (default 0)")
    parser.add_argument("--id-col", type=int, default=None,
                        help="0-based column holding the id (default: make one up)")
    parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks that already have an output file")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading encoder from {args.encoder}")
    encoder = PQEncoder.load(args.encoder)
    print(f"Loading clusterer from {args.clusterer}")
    clusterer = PQKMeans.load(args.clusterer)
    print(f"  k={clusterer.k:,} clusters, m={encoder.m} subvectors")

    stream = DataStreamer()
    fp_calc = FingerprintCalculator()

    total_molecules = 0
    start = time.perf_counter()

    # with_ids=True -> each chunk is a list of (id, smiles) tuples.
    for i, chunk in enumerate(stream.parse_input(
        args.input, chunksize=args.chunksize, smiles_col=args.smiles_col,
        id_col=args.id_col, with_ids=True, verbose=0,
    )):
        out_file = os.path.join(args.output, f"chunk_{i:05d}.parquet")
        if args.resume and os.path.exists(out_file):
            total_molecules += len(chunk)
            continue

        t0 = time.perf_counter()
        ids = [c[0] for c in chunk]
        smiles = [c[1] for c in chunk]

        # MQN fingerprints. return_valid_idx tells us which SMILES parsed, so we
        # keep id / smiles / cluster_id aligned even if some fail.
        valid_idx, fps = fp_calc.FingerprintFromSmiles(smiles, "mqn", return_valid_idx=True)
        if len(valid_idx) == 0:
            total_molecules += len(chunk)
            continue
        ids_ok = [ids[j] for j in valid_idx]
        smiles_ok = [smiles[j] for j in valid_idx]

        # MQN -> PQ codes -> cluster label (GPU when available).
        codes = encoder.transform(fps.astype(np.float32), verbose=0, device=args.device)
        labels = clusterer.predict(codes, device=args.device).astype(np.int32)

        table = pa.table({
            "id": pa.array(ids_ok, type=pa.string()),
            "smiles": pa.array(smiles_ok, type=pa.string()),
            "cluster_id": pa.array(labels, type=pa.int32()),
        })
        pq.write_table(table, out_file, compression="zstd")

        total_molecules += len(chunk)
        rate = total_molecules / (time.perf_counter() - start)
        print(
            f"\rChunk {i:>5d} | {total_molecules:>12,} molecules | "
            f"{rate:,.0f} mol/s | chunk: {time.perf_counter()-t0:.1f}s",
            end="", flush=True,
        )

        del chunk, fps, codes, labels, table

    print(f"\n\nDone: {total_molecules:,} molecules in {format_time(time.perf_counter()-start)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
