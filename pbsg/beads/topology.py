from rdkit import Chem

def get_branch_stats(mol):
    if not mol: return 0, 0
    branch_points = [atom for atom in mol.GetAtoms() if atom.GetDegree() > 2]
    num_branch_points = len(branch_points)
    return num_branch_points

def analyze_repeat_unit(mol: Chem.Mol) -> dict:
    """
    Identify backbone vs sidechain atoms using polymer connection points (*).
    """
    star_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_atoms) != 2:
        raise ValueError(f"Repeat unit must have exactly 2 star atoms (*), found {len(star_atoms)}")
    

    path = Chem.rdmolops.GetShortestPath(mol, star_atoms[0], star_atoms[1])
    backbone_atoms = set(path)
    all_atoms = set(range(mol.GetNumAtoms()))
    num_atoms=len(all_atoms)
    num_backbone = len(backbone_atoms) - 2
    num_sidechain = len(all_atoms - backbone_atoms)
    num_branches = get_branch_stats(mol)
    avg_branch_length = num_sidechain / num_branches if num_branches > 0 else 0


    return {
        "backbone_atoms": num_backbone/num_atoms,
        "sidechain_atoms": num_sidechain/num_atoms,
        "branching_points": num_branches/num_atoms,
        "avg_branch_length": avg_branch_length,
        "num_sidechain_atoms": num_sidechain,
        "num_backbone_atoms": num_backbone,
    }
