import random
import logging

import numpy as np
import pandas as pd
from rdkit import Chem

from .graph import BeadNode, PolymerGraph
from .parser import from_df
from pbsg.beads import get_or_register_bead
from pbsg.beads.registry import registry

logger = logging.getLogger(__name__)


def generate_polymer(
    bead_types: list[str],
    lengths: list[float] = None,
    sequence_type: str = "block",        # "block", "alternating", "statistical"
    reactivity_ratios: list[float] = None,
    spatiality_rules: list = None,       # default = 0 for all
    Pm: float = None,
    ee: float = None,
    total_chain_length: int = 100,
    seed: int | None = None
):
    """
    Generate a polymer with bead spacing and stereochemistry rules.

    Parameters
    ----------
    bead_types : list[str]
        List of bead IDs.
    lengths : list[float]
        - block: absolute beads per block
        - statistical/random/alternating: relative composition (molar fraction)
    sequence_type : str
        "block", "alternating", "statistical"
    reactivity_ratios : list[float]
        - statistical: reactivity ratios for each bead type
    spatiality_rules : list
        Per bead type rules. Options:
        - 0 -> no side chain / non-stereogenic
        - 1 -> same side (+1)
        - -1 -> same side (-1)
        - "alt" -> alternating +1/-1
        - "ran" -> random +/-1
        Default = 0 for all
    Pm : float
        - for polymers with Pm, the probability of meso addition
    ee : float
        - for polymers with ee% given
    total_chain_length : int
        Total number of beads in the polymer
    seed : int | None
        Random seed for reproducibility
    """
    if seed is None:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

    # default spatiality rules = 0
    if spatiality_rules is None:
        spatiality_rules = [0] * len(bead_types)

    if sequence_type == "block":
        chain = []
        sides = []
        lengthss = []
        if lengths is not None and not (isinstance(lengths, float) and np.isnan(lengths)):
            lengthss = [l * total_chain_length for l in lengths]
        else:
            lengthss = [total_chain_length]
        if Pm is not None:
            sides = _generate_meso(Pm, total_chain_length)
        else:
            for i, (bead_id, length) in enumerate(zip(bead_types, lengthss)):
                rule = spatiality_rules[i]
                block_sides = _generate_sides(int(length), rule, ee=ee)
                sides.extend(block_sides)
        for bead_id, length in zip(bead_types, lengthss):
            chain.extend([bead_id] * int(length))

    elif sequence_type == "alternating":
        # strictly alternate bead types
        chain = []
        sides = []
        max_len = total_chain_length
        bead1, bead2 = bead_types
        for i in range(max_len):
            if i % 2 == 0:
                bead = bead1
            else:
                bead = bead2
            chain.append(bead)
        if Pm is not None:
            sides = _generate_meso(Pm, max_len)
        else:
            sides = _generate_sides(max_len, spatiality_rules=spatiality_rules, ee=ee)

    elif sequence_type in ["statistical", "random"]:
        if reactivity_ratios is not None and not (isinstance(reactivity_ratios, float) and np.isnan(reactivity_ratios)):
            if lengths is not None:
                total = sum(lengths)
                weights = [l / total for l in lengths]
            else:
                weights = [0.5, 0.5]
            # fractions
            f1, f2 = weights
            m1, m2 = bead_types
            r1, r2 = reactivity_ratios

            # Calculate transition probabilities
            p_11 = (r1 * f1) / (r1 * f1 + f2)  # P(M1 -> M1)
            p_12 = 1 - p_11                     # P(M1 -> M2)
            p_22 = (r2 * f2) / (f1 + r2 * f2)  # P(M2 -> M2)
            p_21 = 1 - p_22                     # P(M2 -> M1)

            chain = []
            last_monomer = m1  # Start with M1
            chain.append(last_monomer)

            for i in range(total_chain_length - 1):
                r = random.random()

                if last_monomer == m1:
                    # Last monomer was M1, use p_11 and p_12
                    if r < p_11:
                        next_monomer = m1  # M1 -> M1
                    else:
                        next_monomer = m2  # M1 -> M2
                else:
                    # Last monomer was M2, use p_21 and p_22
                    if r < p_21:
                        next_monomer = m1  # M2 -> M1
                    else:
                        next_monomer = m2  # M2 -> M2

                chain.append(next_monomer)
                last_monomer = next_monomer

            # assign sides
            sides = []
            if Pm is not None:
                sides = _generate_meso(Pm, total_chain_length)
            else:
                for bead_id in chain:
                    idx = bead_types.index(bead_id)
                    rule = spatiality_rules[idx]
                    side = _generate_sides(1, rule, ee=ee)[0]
                    sides.append(side)

        else:
            # scale lengths to fractions if sum != 1
            total = sum(lengths)
            weights = [l / total for l in lengths]
            # determine number of beads per type
            num_beads_per_type = [int(w * total_chain_length) for w in weights]
            # flatten to chain
            chain = []
            for bead_id, n in zip(bead_types, num_beads_per_type):
                chain.extend([bead_id] * n)
            random.shuffle(chain)

            # assign sides
            sides = []
            if Pm is not None:
                sides = _generate_meso(Pm, total_chain_length)
            else:
                for bead_id in chain:
                    idx = bead_types.index(bead_id)
                    rule = spatiality_rules[idx]
                    side = _generate_sides(1, rule, ee=ee)[0]
                    sides.append(side)
    else:
        raise ValueError(f"Unknown sequence_type: {sequence_type}")

    nodes = [BeadNode(bead_id=b, side=s) for b, s in zip(chain, sides)]
    return PolymerGraph.linear(nodes)


def _generate_sides(length: int, rule, index: int = 0, ee=None):
    """
    Helper to generate side values for a bead block.
    """
    if rule == 0:
        return [0] * length
    elif rule == "ran" and ee is not None:
        frac = (1 + ee) / 2
        return [+1 if random.random() < frac else -1 for _ in range(length)]
    elif ee is not None:
        frac = (1 + ee) / 2
        return [+1 if random.random() < frac else -1 for _ in range(length)]
    elif rule == "ran":
        return [random.choice([+1, -1]) for _ in range(length)]
    elif rule == "alt":
        start = +1 if index % 2 == 0 else -1
        return [start if i % 2 == 0 else -start for i in range(length)]
    else:  # 1 or -1
        return [rule] * length


def _generate_meso(Pm, length: int):
    """
    Helper to generate side values of a chain if Pm is given.
    """
    prev_side = random.choice([+1, -1])
    sides = [prev_side]
    for i in range(1, length):
        side = prev_side if random.random() < Pm else -prev_side
        sides.append(side)
        prev_side = side
    return sides


def polymers_from_df(df, total_chain_length, seed) -> list:
    """
    Generate list of PolymerGraph objects from dataframe.
    """
    polymer_df = from_df(df)
    polymers = []

    for idx, row in polymer_df.iterrows():
        try:
            polymer = generate_polymer(
                bead_types=row['beads_list'],
                lengths=row.get('lengths'),
                sequence_type=row.get('sequence_type', 'block'),
                reactivity_ratios=row.get('reactivity_ratios'),
                spatiality_rules=row.get('spatiality_rules'),
                Pm=row.get('Pm'),
                ee=row.get('ee'),
                total_chain_length=total_chain_length,
                seed=seed,
            )

            # Attach thermal properties
            polymer.Xc = row.get('Xc')
            polymer.Tg = row.get('Tg')
            polymer.Tm = row.get('Tm')
            polymer.has_Tm = row.get('has_Tm')
            polymer.has_Tg = row.get('has_Tg')
            polymer.ID = row.get('ID')
            polymer.arch = row.get('architecture_onehot')
            polymer.ratio = row.get('comonomer_ratios')
            polymer.stereo_class = row.get('stereo_class')

            polymers.append(polymer)
        except Exception as e:
            logger.warning("Error generating polymer at row %s: %s", idx, e)

    return polymers


def smiles_from_graphs(df, total_chain_length, seed):
    """Generate polymer SMILES dataframe."""
    polymers = polymers_from_df(df, total_chain_length, seed=seed)
    for p in polymers:
        p.p_smiles = _build_polymer(p, registry)
    rows = []
    for p in polymers:
        rows.append({
            "p_smiles": p.p_smiles,
            "Tm": p.Tm,
            "has_Tm": p.has_Tm
        })
    df_smiles = pd.DataFrame(rows)
    df_smiles.dropna(subset=['p_smiles'], inplace=True)
    return df_smiles


def _connect_two_units(mol1, mol2):
    """Connect mol1 and mol2 at their dummy atoms (*)."""
    combo = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combo)

    dummy_atoms = [a for a in combo.GetAtoms() if a.GetAtomicNum() == 0]

    if len(dummy_atoms) != 4:
        raise ValueError("Each repeat unit must contain exactly two '*' atoms")

    # First two dummy atoms belong to mol1, next two to mol2
    d1 = dummy_atoms[1]  # right dummy of mol1
    d2 = dummy_atoms[2]  # left dummy of mol2

    n1 = d1.GetNeighbors()[0].GetIdx()
    n2 = d2.GetNeighbors()[0].GetIdx()

    emol.AddBond(n1, n2, Chem.BondType.SINGLE)

    # Remove dummy atoms (remove highest index first!)
    for idx in sorted([d1.GetIdx(), d2.GetIdx()], reverse=True):
        emol.RemoveAtom(idx)

    return emol.GetMol()


def _cap_with_methyl(mol):
    """Replace terminal '*' atoms with CH3."""
    emol = Chem.EditableMol(mol)
    dummy_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]

    for dummy in sorted(dummy_atoms, key=lambda x: x.GetIdx(), reverse=True):
        neighbor = dummy.GetNeighbors()[0].GetIdx()

        # Add carbon
        c_idx = emol.AddAtom(Chem.Atom(6))
        emol.AddBond(neighbor, c_idx, Chem.BondType.SINGLE)

        emol.RemoveAtom(dummy.GetIdx())

    return emol.GetMol()


def _build_polymer(polymer, registry):
    fragments = []
    for node in polymer.nodes:
        bead = get_or_register_bead(node.bead_id)
        if bead is None:
            logger.warning("Could not register bead: %s", node.bead_id)
            return None
        if node.side == 1:
            fragments.append(bead.smiles_CW)
        elif node.side == -1:
            fragments.append(bead.smiles_CCW)
        else:
            fragments.append(bead.smiles_CW)
    mols = [Chem.MolFromSmiles(s) for s in fragments]

    polymer = mols[0]
    for next_unit in mols[1:]:
        polymer = _connect_two_units(polymer, next_unit)

    polymer = _cap_with_methyl(polymer)

    Chem.SanitizeMol(polymer)
    polymer_smiles = Chem.MolToSmiles(polymer)
    return polymer_smiles