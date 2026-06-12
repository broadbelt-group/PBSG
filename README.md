# PBSG — Polymer Bead Sequence Graphs

A coarse-grained, machine-interpretable polymer representation that explicitly accounts for stereochemistry.

PBSG represents a polymer as a sequence graph of **beads**, where each bead is a repeat unit. Instead of modeling every atom, each repeat unit is collapsed into a single node carrying precomputed chemical, topological, and flexibility features. Stereochemistry is encoded directly: chiral beads store distinct **R** and **S** feature sets, and chain-level tacticity is generated from physically meaningful parameters (meso-diad probability `Pm` and enantiomeric excess `ee`). The result is a representation that is far more compact than an atomistic graph while retaining the chemical and stereochemical detail needed for property prediction.

## Why PBSG

PBSG was developed to predict polymer melting temperature `Tm` from structure. In our work, PBSG graphs outperformed both atomistic molecular graphs and traditional molecular fingerprints, while remaining substantially more computationally efficient than atomistic graphs. A flattened (pooled) form of PBSG also outperformed conventional fingerprints as a feature vector for tree-based models. See [Citation](#citation) for details.

## Key ideas

- **A bead is a repeat unit.** Repeat units are written as SMILES with two connection points (`[*]`) marking where the backbone continues.
- **Features are cached.** The first time a bead is registered, its features are computed and stored in a local registry, so they never need to be recomputed for subsequent polymers that contain the same repeat unit.
- **Stereochemistry is a priority.** Chiral beads carry separate R and S feature sets. In a generated chain, a side value of `+1` corresponds to R and `-1` to S; achiral beads use `0`. Chain stereochemistry/tacticity is produced from `Pm` (probability of meso addition) and/or `ee`.
- **Reactivity ratios can be input for random copolymers.** Reactivity ratios can be used to build random copolymer chains when provided.

## Installation

```bash
pip install pbsg
```

PBSG depends on `torch` and `torch-geometric`. On most platforms `pip` will resolve these automatically, but in some environments (particular CUDA builds, Apple Silicon, etc.) it is more reliable to install PyTorch and PyG first, following their official instructions, and then install PBSG:

```bash
pip install torch torch-geometric
pip install pbsg
```

> **NumPy note:** use `torch >= 2.3` if you are on NumPy 2.x. Older torch builds were compiled against NumPy 1.x and will emit a binary-compatibility warning (and can crash) under NumPy 2.

## Quickstart

The typical workflow takes a `DataFrame` of polymers, builds PBSG graphs, and hands them to your model.

```python
import pandas as pd

from pbsg.polymer import polymers_from_df
from pbsg.ml import to_pyg
from pbsg.ml.pyg import architecture_onehot
from pbsg.polymer.parser import parse_smiles_list

# 1. Load your data
df = pd.read_csv("polymers.csv")

# 2. Build the bead list from repeat-unit SMILES, and the architecture encoding
df["beads_list"]          = df["repeat_units"].apply(parse_smiles_list)
df["architecture_onehot"] = df.apply(architecture_onehot, axis=1)

# 3. Generate PBSG polymer graphs (chains of `total_chain_length` beads)
polymers = polymers_from_df(df, total_chain_length=100)

# 4. Convert each polymer to a PyTorch Geometric `Data` object
graphs = [to_pyg(p, stereo=True) for p in polymers]

# `graphs` can now be batched with torch_geometric.loader.DataLoader
# and fed to a GNN (e.g. GINE / GATv2), or pooled into feature vectors.
```

You can also build a single graph directly from a compact string, where `(+)`/`(-)` denote R/S beads:

```python
from pbsg.polymer.parser import from_string
from pbsg.ml import to_pyg

graph = from_string("A(+)-B-A(-)-C")
```

### Embedding options

`to_pyg` supports three bead-embedding modes:

- `to_pyg(p)` — default; PCA-compressed fingerprints + descriptors (uses the shipped PCA models).
- `to_pyg(p, fixed=True)` — fixed-length Morgan + RDKit fingerprints + descriptors (no PCA).
- `to_pyg(p, pca=False)` — full (uncompressed) fingerprints + descriptors.

Pass `stereo=True` to make Morgan fingerprints stereo-sensitive for chiral beads.

## Input format

PBSG reads a `DataFrame` with (a subset of) the following columns:

| Column | Meaning |
|---|---|
| `repeat_units` | Dot-separated repeat-unit SMILES, e.g. `"[*]C(C)C[*]"` or `"A.B"` for a copolymer. Each unit has two `[*]` connection points. |
| `beads_list` | List of repeat-unit SMILES (produced by `parse_smiles_list`). |
| `sequence_type` | `"block"`, `"alternating"`, or `"statistical"` / `"random"`. |
| `lengths` | Block lengths (block) or molar fractions (statistical/alternating). |
| `reactivity_ratios` | Reactivity ratios for statistical copolymers. |
| `spatiality_rules` | Per-bead stereo rule: `0` (achiral), `1` (R), `-1` (S), `"alt"`, or `"ran"`. |
| `Pm` | Probability of meso addition (sets tacticity). |
| `ee` | Enantiomeric excess. |
| `stereo_class` | `isotactic`, `syndiotactic`, `heterotactic`, `atactic`, `achiral`, or `unknown`. |
| `architecture_onehot` | Architecture encoding (produced by `architecture_onehot`). |
| `Tm`, `has_Tm`, `Tg`, `has_Tg`, `Xc` | Optional property targets, attached to each polymer graph. |

## The bead registry

Registered bead features are cached on disk at `~/.pbsg/bead_registry.json`. This directory is created lazily — only the first time a bead is registered — so importing the package has no side effects. The cache persists across sessions and across projects, so repeat units are only ever featurized once.

## Public API

```python
from pbsg.beads import get_or_register_bead, delete_bead, registry, BeadType
from pbsg.polymer import (
    generate_polymer, polymers_from_df,
    PolymerGraph, BeadNode, from_string, from_df,
)
from pbsg.polymer.parser import parse_smiles_list
from pbsg.polymer.generators import smiles_from_graphs
from pbsg.ml import to_pyg, build_bead_embeddings
from pbsg.ml.pyg import architecture_onehot
```

| Function | Purpose |
|---|---|
| `parse_smiles_list(s)` | Split a dot-separated repeat-unit SMILES string into a list. |
| `get_or_register_bead(smiles)` | Featurize a repeat unit (or fetch it from the registry). |
| `generate_polymer(...)` | Build a single `PolymerGraph` from bead types and sequence rules. |
| `polymers_from_df(df, total_chain_length)` | Build a list of `PolymerGraph`s from a DataFrame. |
| `to_pyg(graph, ...)` | Convert a `PolymerGraph` to a PyTorch Geometric `Data` object. |
| `smiles_from_graphs(df, total_chain_length)` | Reconstruct polymer SMILES from generated chains. |
| `architecture_onehot(row)` | One-hot encode polymer architecture from a DataFrame row. |

## Citation

This work is associated with the following manuscript (in preparation):

> **Stereochemistry-Aware Classification and Regression of Polymer Melting Points Employing Coarse-Grained Representation.**
> Maya L. Milrod, Kevin M. Shebek, A. Nolan Wilson, Eugene Y.-X. Chen, Tobin J. Marks, and Linda J. Broadbelt.

A BibTeX entry and DOI will be added upon publication.

## License

MIT — see [LICENSE](LICENSE).
