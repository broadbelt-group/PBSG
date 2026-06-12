from rdkit import Chem
from rdkit.Chem import Lipinski 



def compute_flexibility(mol: Chem.Mol, topology: dict) -> float:

    """
    Compute a polymer-relevant flexibility scores for a repeat unit.
    """

    # Rotatable bonds (RDKit definition)
    n_rot = Lipinski.NumRotatableBonds(mol)
    n_tot_bonds = mol.GetNumBonds()
    n_backbone = topology.get("num_backbone_atoms") or 0
    n_sidechain = topology.get("num_sidechain_atoms") or 0
    branching = topology.get("branching_points", 0) or 0

    bulkiness = n_sidechain / max(1, branching)  # avg sidechain size

    # Normalize side-chain influence
    sidechain_frac = n_sidechain / max(1, n_backbone) #number of side chain atoms per backbone atom. 
    branching_frac = branching / max(1, n_backbone) #number of branching points per backbone atom. 
    rot_frac = n_rot / max(1, n_tot_bonds) #fraction of rotatable bonds. More rotatable bonds = more flexible
    side_bulkiness_frac=bulkiness/max(1, n_backbone) #avg side chain size over num backbone atoms

    # Simple physically motivated score
    flexibility = max(0.0, n_rot/n_tot_bonds)

    # Clamp to reasonable range
    return {
        "flexibility": flexibility,
        "side_bulkiness_frac": side_bulkiness_frac,
    }
