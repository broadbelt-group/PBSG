import re
import pandas as pd
from .graph import BeadNode, PolymerGraph


def from_df(df) -> pd.DataFrame:
    """
    Parse dataframe into polymer-ready format.

    Returns DataFrame with the relevant polymer columns (bead list, lengths,
    sequence type, reactivity ratios, spatiality rules, Pm, ee, Xc, Tg, Tm, etc.)
    that are present in the input.
    """
    polymer_df = df.copy()

    keep_columns = [
        'ID',
        'beads_list',
        'lengths',
        'sequence_type',
        'reactivity_ratios',
        'spatiality_rules',
        'Pm',
        'ee',
        'Xc',
        'Tg',
        'Tm',
        'has_Tm',
        'has_Tg',
        'architecture_onehot',
        'comonomer_ratios',
        'stereo_class',
        'stereo_onehot'
    ]

    # Only keep columns that exist
    keep_columns = [col for col in keep_columns if col in polymer_df.columns]
    polymer_df = polymer_df[keep_columns]

    return polymer_df


def from_string(s: str) -> PolymerGraph:
    """
    Parse polymer string like:
    A(+)-B-A(-)-C
    """
    tokens = s.split("-")
    nodes = []

    for t in tokens:
        m = re.match(r"(\w+)(?:\(([+-])\))?", t)
        bead = m.group(1)
        side = m.group(2)

        side_val = None
        if side == "+":
            side_val = +1
        elif side == "-":
            side_val = -1

        nodes.append(BeadNode(bead_id=bead, side=side_val))

    return PolymerGraph.linear(nodes)

def parse_smiles_list(value: str):
    """
    Parse repeat unit SMILES from a dot-separated string.
      - Single homopolymer repeat unit : "CC"
      - Multiple repeat units          : "CC.CCO.c1ccccc1"
    Returns a list of SMILES
    """
    value = str(value).strip()
    units = [u.strip() for u in value.split(".") if u.strip()]
    return units