from rdkit import Chem
from .bead import BeadType
from .rdkit_features import compute_fingerprints, compute_descriptors
from .utils import make_bead_id
from .registry import registry
from .topology import analyze_repeat_unit
from .flexibility import compute_flexibility
from .stereochem import assign_stereo_smiles


def get_or_register_bead(smiles: str, alias: str | None = None) -> BeadType:
    """
    Get a bead from registry or register a new one from SMILES.
    Stereochemistry is implicit in the SMILES.
    """
    smiles_CW, smiles_CCW, skip = assign_stereo_smiles(smiles)

    if skip:
        return None

    mol=Chem.MolFromSmiles(smiles)
    mol_CW = Chem.MolFromSmiles(smiles_CW)
    mol_CCW=Chem.MolFromSmiles(smiles_CCW)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol, canonical=True)
    stereogenic = bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    bead_id = make_bead_id(canonical, stereogenic)

    if registry.has(bead_id):
        return registry.get(bead_id)
    
    morgan_fp, rdkit_fp=compute_fingerprints(mol)

    morgan_fp_CW, rdkit_fp_CW = compute_fingerprints(mol_CW)
    morgan_fp_CCW, rdkit_fp_CCW = compute_fingerprints(mol_CCW)
    descriptors = compute_descriptors(mol)
    topology = analyze_repeat_unit(mol)
    flexibility = compute_flexibility(mol, topology)

    bead = BeadType(
        bead_id=bead_id,
        smiles=canonical,
        smiles_CW=smiles_CW,
        smiles_CCW=smiles_CCW,
        alias=alias,
        stereogenic=stereogenic,
        morgan_fp=morgan_fp,
        rdkit_fp=rdkit_fp,
        morgan_fp_CW=morgan_fp_CW,
        morgan_fp_CCW=morgan_fp_CCW,
        rdkit_fp_CW=rdkit_fp_CW,
        rdkit_fp_CCW=rdkit_fp_CCW,
        MolWt=descriptors["MolWt"],
        LogP=descriptors["LogP"],
        TPSA=descriptors["TPSA"],
        RotBonds=descriptors["RotBonds"],
        Rings=descriptors["Rings"],
        HBD=descriptors["HBD"],
        HBA=descriptors["HBA"],
        Heteroatoms=descriptors["Heteroatoms"],
        NHOH=descriptors["NHOH"],
        AliphaticRings=descriptors["AliphaticRings"],
        AmideBonds=descriptors["AmideBonds"],
        AromaticRings=descriptors["AromaticRings"],
        kier_flex=descriptors["kier_flex"],
        backbone_atoms=topology["backbone_atoms"],
        sidechain_atoms=topology["sidechain_atoms"],
        avg_branch_length=topology["avg_branch_length"],
        flexibility=flexibility["flexibility"],
        side_bulkiness_frac=flexibility["side_bulkiness_frac"],
    )

    registry.register(bead)
    return bead

def delete_bead(bead_id: str):
    registry.delete_bead(bead_id)
