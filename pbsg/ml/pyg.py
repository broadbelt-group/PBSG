import os
import logging
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from pbsg.beads import get_or_register_bead

logger = logging.getLogger(__name__)

_dir = os.path.dirname(os.path.abspath(__file__))
_pca_morgan = None
_pca_rdkit = None


def _get_pca():
    """Lazily load the PCA models on first use (not at import time)."""
    global _pca_morgan, _pca_rdkit
    if _pca_morgan is None:
        import joblib
        _pca_morgan = joblib.load(os.path.join(_dir, "pca_morgan.pkl"))
        _pca_rdkit = joblib.load(os.path.join(_dir, "pca_rdkit.pkl"))
    return _pca_morgan, _pca_rdkit


def build_bead_embeddings_pca(smiles, side=0, stereo=False) -> torch.Tensor:
    bead = get_or_register_bead(smiles)
    if bead is None:
        logger.warning("Could not register bead for SMILES: %s", smiles)
        return None

    # only morgan is stereo-sensitive
    if stereo and bead.smiles_CW is not None and bead.smiles_CCW is not None:
        if side == 1 and bead.morgan_fp_CW is not None:
            morgan_raw = np.array(bead.morgan_fp_CW).reshape(1, -1)
        elif side == -1 and bead.morgan_fp_CCW is not None:
            morgan_raw = np.array(bead.morgan_fp_CCW).reshape(1, -1)
        else:
            morgan_raw = np.array(bead.morgan_fp).reshape(1, -1)
    else:
        morgan_raw = np.array(bead.morgan_fp).reshape(1, -1)

    # rdkit always uses base - not stereo sensitive
    rdkit_raw = np.array(bead.rdkit_fp).reshape(1, -1)

    pca_morgan, pca_rdkit = _get_pca()
    morgan_comp = torch.tensor(pca_morgan.transform(morgan_raw)[0], dtype=torch.float32)
    rdkit_comp = torch.tensor(pca_rdkit.transform(rdkit_raw)[0], dtype=torch.float32)

    descriptors = torch.tensor([
        bead.MolWt, bead.LogP, bead.TPSA,
        bead.RotBonds, bead.Rings, bead.HBD, bead.HBA,
        bead.Heteroatoms, bead.NHOH, bead.AliphaticRings, bead.AmideBonds, bead.AromaticRings,
        bead.backbone_atoms, bead.sidechain_atoms, bead.avg_branch_length,
        bead.flexibility, bead.side_bulkiness_frac, bead.kier_flex
    ], dtype=torch.float64)

    embedding = torch.cat([morgan_comp, rdkit_comp, descriptors])
    return embedding


_gen_fixed_morgan = GetMorganGenerator(radius=2, fpSize=64, includeChirality=True)


def build_bead_embeddings_fixed(smiles, side=0, stereo=False, fp_bits=64) -> torch.Tensor:
    bead = get_or_register_bead(smiles)
    if bead is None:
        logger.warning("Could not register bead for SMILES: %s", smiles)
        return None

    # Stereo-sensitive Morgan FP
    if stereo and bead.smiles_CW is not None and bead.smiles_CCW is not None:
        if side == 1 and bead.morgan_fp_CW is not None:
            smi = bead.smiles_CW
        elif side == -1 and bead.morgan_fp_CCW is not None:
            smi = bead.smiles_CCW
        else:
            smi = smiles
    else:
        smi = smiles

    mol_stereo = Chem.MolFromSmiles(smi)
    mol_base = Chem.MolFromSmiles(smiles)

    if mol_stereo is not None:
        morgan_arr = _gen_fixed_morgan.GetFingerprintAsNumPy(mol_stereo).astype(np.float32)
    else:
        morgan_arr = np.zeros(fp_bits, dtype=np.float32)

    if mol_base is not None:
        rdkit_arr = np.array(Chem.RDKFingerprint(mol_base, fpSize=fp_bits), dtype=np.float32)
    else:
        rdkit_arr = np.zeros(fp_bits, dtype=np.float32)

    morgan_vec = torch.tensor(morgan_arr, dtype=torch.float32)
    rdkit_vec = torch.tensor(rdkit_arr, dtype=torch.float32)

    descriptors = torch.tensor([
        bead.MolWt, bead.LogP, bead.TPSA,
        bead.RotBonds, bead.Rings, bead.HBD, bead.HBA,
        bead.Heteroatoms, bead.NHOH, bead.AliphaticRings,
        bead.AmideBonds, bead.AromaticRings,
        bead.backbone_atoms, bead.sidechain_atoms,
        bead.avg_branch_length, bead.flexibility,
        bead.side_bulkiness_frac, bead.kier_flex
    ], dtype=torch.float32)

    return torch.cat([morgan_vec, rdkit_vec, descriptors])


def build_bead_embeddings(smiles, side=0, stereo=False) -> torch.Tensor:
    bead = get_or_register_bead(smiles)
    if bead is None:
        logger.warning("Could not register bead for SMILES: %s", smiles)
        return None

    # only morgan is stereo-sensitive
    if stereo and bead.smiles_CW is not None and bead.smiles_CCW is not None:
        if side == 1 and bead.morgan_fp_CW is not None:
            morgan_raw = np.array(bead.morgan_fp_CW).reshape(1, -1)
        elif side == -1 and bead.morgan_fp_CCW is not None:
            morgan_raw = np.array(bead.morgan_fp_CCW).reshape(1, -1)
        else:
            morgan_raw = np.array(bead.morgan_fp).reshape(1, -1)
    else:
        morgan_raw = np.array(bead.morgan_fp).reshape(1, -1)

    # rdkit always uses base - not stereo sensitive
    rdkit_raw = np.array(bead.rdkit_fp).reshape(1, -1)

    morgan_vec = torch.tensor(morgan_raw[0], dtype=torch.float32)
    rdkit_vec = torch.tensor(rdkit_raw[0], dtype=torch.float32)

    descriptors = torch.tensor([
        bead.MolWt, bead.LogP, bead.TPSA,
        bead.RotBonds, bead.Rings, bead.HBD,
        bead.Heteroatoms, bead.NHOH, bead.AliphaticRings, bead.AmideBonds,
        bead.backbone_atoms, bead.sidechain_atoms, bead.avg_branch_length,
        bead.flexibility, bead.kier_flex
    ], dtype=torch.float32)

    return torch.cat([morgan_vec, rdkit_vec, descriptors])


def side_to_onehot(side):
    if side == 1:
        return torch.tensor([1., 0., 0.])
    elif side == -1:
        return torch.tensor([0., 1., 0.])
    else:  # 0 or None
        return torch.tensor([0., 0., 1.])


def edge_stereo(side_i, side_j):
    if side_i == 0 or side_j == 0:
        return 0.5   # one has no stereochem
    return 1.0 if side_i == side_j else 0.0  # same or opposite


def architecture_onehot(row):
    arch = str(row.get("polymer_type", "")).lower()
    if arch == "copolymer":
        arch = str(row.get("copolymer_type", "")).lower()
    arch_map = {"alternating": 0, "block": 1, "random": 2, "homopolymer": 3}
    vec = np.zeros(4, dtype=np.float32)
    if arch in arch_map:
        vec[arch_map[arch]] = 1.0
    return vec

def classify_stereo(row):
    pm = row['Pm']
    if pd.notna(pm):
        if pm > 0.80: return 'isotactic'
        if pm < 0.20: return 'syndiotactic'
        if 0.40 <= pm <= 0.60: return 'atactic'
        return 'stereo-irregular'  # 0.20-0.40 or 0.60-0.80

    s = row['spatiality_rules']
    if isinstance(s, list) and len(s) > 0:
        if s == [1]: return 'isotactic'
        if s == [-1]: return 'isotactic'
        if s == ['alt']: return 'syndiotactic'
        if s == [1, -1]:
            return 'syndiotactic' if row['polymer_type'] == 'alternating' else 'stereo-irregular'
        if s == ['ran']: return 'atactic'

    ee = row['ee']
    if pd.notna(ee):
        if ee < 0.20: return 'atactic'
        if ee > 0.80: return 'isotactic'
        return 'stereo-irregular'

    return 'achiral'


def stereo_onehot(ste):
    stereo_map = {"isotactic": 0, "syndiotactic": 1, "stereo-irregular": 2,
                  "atactic": 3, "achiral": 4, "unknown": 5}
    vec = np.zeros(6, dtype=np.float32)
    if ste in stereo_map:
        vec[stereo_map[ste]] = 1.0
    return vec


def pm_from_sides(sides: list) -> float:
    """
    Compute meso diad fraction from bead side assignments.
    Pm = meso_diads / (meso_diads + racemo_diads)

    Ignores achiral beads (side=0) — only counts stereocenters.
    Returns 0.5 if no stereocenters present (achiral polymer).
    """
    meso = racemo = 0
    for i in range(len(sides) - 1):
        si, sj = sides[i], sides[i + 1]
        if si == 0 or sj == 0:
            continue
        if si == sj:
            meso += 1
        else:
            racemo += 1
    total = meso + racemo
    return 0.5 if total == 0 else meso / total


def sequence_features(beads: list, sides: list) -> np.ndarray:
    """
    Two sequence-level scalar features for the metadata vector:
      Px    : cross-diad fraction (0=homopolymer, 1=alternating, ~0.5=random)
      asym  : |x1 - 0.5| composition asymmetry (0=equimolar, 0.5=homopolymer)
    """
    # Px
    same = cross = 0
    for i in range(len(beads) - 1):
        if beads[i] == beads[i + 1]:
            same += 1
        else:
            cross += 1
    total = same + cross
    px = cross / total if total > 0 else 0.0

    # unique monomer fraction asymmetry
    if len(beads) == 0:
        asym = 0.0
    else:
        counts = Counter(beads)
        n_most_common = counts.most_common(1)[0][1]
        x1 = n_most_common / len(beads)
        asym = abs(x1 - 0.5)

    return np.array([px, asym], dtype=np.float32)


def to_pyg(polymer_graph, stereo=False, use_architecture=True,
           pca=True, fixed=False, pm=None, stereo_class=None) -> Data:
    x = []
    for node in polymer_graph.nodes:
        if fixed:
            emb = build_bead_embeddings_fixed(node.bead_id, side=node.side, stereo=stereo)
        elif pca:
            emb = build_bead_embeddings_pca(node.bead_id, side=node.side, stereo=stereo)
        else:
            emb = build_bead_embeddings(node.bead_id, side=node.side, stereo=stereo)
        if emb is None:
            return None
        side_enc = side_to_onehot(node.side)
        x.append(torch.cat([emb, side_enc]))

    edges = torch.tensor(polymer_graph.edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edges, edges.flip(0)], dim=1)

    edge_attr_fwd = []
    edge_attr_rev = []
    for i, j in polymer_graph.edges:
        val = edge_stereo(polymer_graph.nodes[i].side, polymer_graph.nodes[j].side)
        edge_attr_fwd.append([val])
        edge_attr_rev.append([val])
    edge_attr = torch.tensor(edge_attr_fwd + edge_attr_rev, dtype=torch.float32)

    data = Data(x=torch.stack(x), edge_index=edge_index, edge_attr=edge_attr)

    sides = [node.side for node in polymer_graph.nodes]
    beads = [node.bead_id for node in polymer_graph.nodes]
    seq_vals = sequence_features(beads, sides)
    pm_val = np.float32(float(pm_from_sides(sides)))
    arch_vec = polymer_graph.arch
    stereo_vec = stereo_onehot(polymer_graph.stereo_class)
    meta = np.concatenate([arch_vec, stereo_vec, [pm_val], seq_vals])
    data.architecture = torch.tensor(meta, dtype=torch.float32).unsqueeze(0)  # [1, META_DIM]

    return data