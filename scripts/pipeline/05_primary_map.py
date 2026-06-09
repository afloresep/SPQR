#!/usr/bin/env python3
"""STEP 5 - Build the primary (representatives) TMAP, with links to clusters.

One point per cluster, laid out by MQN so the map reads in interpretable
chemical coordinates (size, polarity, ...). Two things make it a *nested* map:

  * `add_label(...)` puts the compound id and cluster id into each point's card
    (the panel shown on click) - this is the metadata layer.
  * `configure_card(links=...)` adds an "Open cluster map" button whose URL,
    `cluster_{Cluster}.html`, is filled in per point. Serve the secondary maps
    (STEP 6) in the same folder and the button opens the right one.

Output: representatives_TMAP.html

Usage:
    python scripts/pipeline/05_primary_map.py \
        --representatives OUT/representatives.csv \
        --output OUT/site/representatives_TMAP.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import add_property_colors, log


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--representatives", required=True, help="representatives.csv")
    p.add_argument("--output", default="representatives_TMAP.html")
    p.add_argument("--title", default="Enamine - cluster representatives")
    p.add_argument("--k-neighbors", type=int, default=21)
    args = p.parse_args()

    from tmap import TMAP
    from tmap.utils import fingerprints_from_smiles

    # Read id as string: Enamine ids are 19-20 digit integers that overflow
    # pandas' int64 inference (silently downcast to float64, losing the exact id).
    df = (pd.read_csv(args.representatives, dtype={"id": str})
          .sort_values("cluster_id").reset_index(drop=True))
    smiles = df["smiles"].tolist()
    ids = df["id"].astype(str).tolist()
    clusters = df["cluster_id"].astype(str).tolist()
    log(f"Building primary TMAP for {len(smiles):,} cluster representatives")

    # MQN + Euclidean: dense descriptor, interpretable property axes.
    fps = fingerprints_from_smiles(smiles, fp_type="mqn")
    if len(fps) != len(smiles):
        raise RuntimeError(
            f"tmap dropped {len(smiles)-len(fps)} unparseable SMILES; representatives "
            "should already be valid - clean the CSV before plotting to avoid desync.")

    model = TMAP(metric="euclidean", n_neighbors=args.k_neighbors).fit(fps)
    viz = model.to_tmapviz(include_edges=True)
    viz.title = args.title
    viz.background_color = "#FFFFFF"



    # Tooltip / card content.
    viz.add_label("Cluster", clusters)  # cluster id (also used in the link URL)
    viz.add_label("ID", ids)            # compound id (metadata)
    viz.add_smiles(smiles)              # structure drawing

    # Colour layers: interpretable MQN-derived properties. Shared with the
    # secondary maps (_common.add_property_colors) so both expose identical axes.
    add_property_colors(viz, smiles)



    # The nesting: clicking a representative opens its cluster's secondary map.
    viz.configure_card(
        title_column="Cluster",     # heading: the cluster number
        subtitle_column="ID",       # italic line below: the compound id
        fields=["ID", "Cluster"],
        links=[{"label": "Open cluster map", "url": "cluster_{Cluster}.html"}],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    viz.write_html(str(out))
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
