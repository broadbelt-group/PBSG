import torch
from torch_geometric.data import Data
from rdkit import Chem


def atom_features(atom):
    return [
        atom.GetDegree(),                         # degree
        atom.GetFormalCharge(),                   # formal charge
        atom.GetTotalNumHs(),                     # total H
        atom.GetValence(Chem.ValenceType.IMPLICIT),
        float(atom.GetIsAromatic()),
        float(atom.GetHybridization()),
        float(atom.GetChiralTag()),
        atom.GetValence(Chem.ValenceType.EXPLICIT),
        float(atom.GetNoImplicit()),
        atom.GetNumRadicalElectrons(),
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalValence()
    ]


def bond_features(bond):
    return [
        float(bond.GetBondType()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing())
    ]


def mol_to_graph(mol, target=None):
    # ---- Nodes ----
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(atom_features(atom))
    x = torch.tensor(node_features, dtype=torch.float)

    # ---- Edges ----
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        features = bond_features(bond)

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        edge_attrs.append(features)
        edge_attrs.append(features)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # ---- Target ----
    if target is not None:
        y = torch.tensor([target], dtype=torch.long)
    else:
        y = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)