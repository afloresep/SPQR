"""Shared helpers for the nested-TMAP map scripts in this folder.

These small utilities are used by the map-building stages (partition,
representatives, primary/secondary TMAPs). The heavy compute - fingerprints,
encoder/clusterer training, and cluster assignment - is done by the main
chelombus scripts (see this folder's README); those read SMILES through
``chelombus.DataStreamer`` and carry an id per molecule.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Put the repo root on sys.path so `import chelombus...` works when these
# scripts are run directly (this file -> repo root is parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MQN_DIM = 42


# ── logging / formatting ──────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fmt_time(s: float) -> str:
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}h {int(m)}m {sec:.0f}s"


# ── id-aware MQN (used to pick cluster representatives in stage 5) ──────────
# Pool workers must be module-level so they can be pickled.

def _mqn_idx(arg: tuple[int, str]) -> tuple[int, list] | None:
    """(input_index, smiles) -> (input_index, mqn list) or None on parse fail."""
    i, smiles = arg
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return i, list(rdMolDescriptors.MQNs_(mol))
    except Exception:
        return None


def mqn_aligned(
    smiles: list[str], nproc: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MQN for *smiles*, reporting which inputs survived.

    Returns:
        valid_idx : int64 [n]    positions in *smiles* that parsed OK
        fps       : int16 [n,42] their MQN fingerprints, row-aligned to valid_idx
    """
    nproc = nproc or os.cpu_count()
    with Pool(processes=nproc) as pool:
        results = pool.map(_mqn_idx, enumerate(smiles), chunksize=10_000)

    results = [r for r in results if r is not None]
    if not results:
        return np.empty(0, dtype=np.int64), np.empty((0, MQN_DIM), dtype=np.int16)

    idx = np.fromiter((r[0] for r in results), dtype=np.int64, count=len(results))
    fps = np.array([r[1] for r in results], dtype=np.int16)
    return idx, fps


# ── shared TMAP colour layers (viz scripts only) ───────────────────────────
# Imported lazily inside the function so non-viz callers don't need tmap.

def add_property_colors(viz, smiles: list[str]) -> None:
    """Attach the standard MQN-property colour layers to a TmapViz.

    Used by BOTH the primary map and the per-cluster secondary maps so the two
    expose identical, interpretable axes. Wide-range descriptors are continuous
    (viridis gradient); small-integer counts and formal charge are discrete, so
    they go on as categorical=True - tmapviz renders 3.0 as "3", gives NaN its
    own colour, and draws a colour-per-value legend (tab10/tab20) instead of a
    misleading continuous ramp.
    """
    from tmap.utils import molecular_properties, AVAILABLE_PROPERTIES
    props = molecular_properties(smiles, properties=list(AVAILABLE_PROPERTIES))

    # Continuous (gradient) layers.
    viz.add_color_layout("Molecular weight", props["mw"])
    viz.add_color_layout("LogP", props["logp"])
    viz.add_color_layout("TPSA", props["tpsa"])
    viz.add_color_layout("Fraction Csp3", props["fraction_csp3"])
    viz.add_color_layout("QED", props["qed"])
    viz.add_color_layout("H-bond acceptors", props["hba"])
    viz.add_color_layout("Rotatable bonds", props["n_rotatable_bonds"])
    viz.add_color_layout("Heavy atoms", props["n_heavy_atoms"])
    viz.add_color_layout("Heteroatoms", props["n_heteroatoms"])

    # Categorical (discrete legend) layers - small-cardinality counts / charge.
    viz.add_color_layout("Rings", props["n_rings"], categorical=True, color="tab20")
    viz.add_color_layout("Aromatic rings", props["n_aromatic_rings"], categorical=True, color="tab10")
    viz.add_color_layout("H-bond donors", props["hbd"], categorical=True, color="tab10")
    viz.add_color_layout("Formal charge", props["formal_charge"], categorical=True, color="tab10")
