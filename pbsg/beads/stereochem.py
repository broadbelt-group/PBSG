from rdkit import Chem

def assign_stereo_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    
    # skip 4 stereocenters; hope to add this handling later
    if len(centers) > 2:
        return None, None, True  # True = skip this bead
    
    mol_CW = Chem.RWMol(mol)
    mol_CCW = Chem.RWMol(mol)
    
    # apply same chirality to all centers
    for atom_idx, _ in centers:
        mol_CW.GetAtomWithIdx(atom_idx).SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
        )
        mol_CCW.GetAtomWithIdx(atom_idx).SetChiralTag(
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )
    
    smiles_CW  = Chem.MolToSmiles(mol_CW)
    smiles_CCW = Chem.MolToSmiles(mol_CCW)
    
    return smiles_CW, smiles_CCW, False 