from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
import numpy as np

def compute_fingerprints(mol):
    gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True, fpSize=2048)
    morgan = gen.GetFingerprint(mol)
    rdkit_fp = Chem.RDKFingerprint(mol, fpSize=2048)

    return (
        np.array(morgan),
        np.array(rdkit_fp)
    )

def compute_descriptors(mol):
    all_atoms = set(range(mol.GetNumAtoms()))
    num_atoms=len(all_atoms)-2
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol)/num_atoms,
        "Rings": Descriptors.RingCount(mol)/num_atoms,
        "HBD": Descriptors.NumHDonors(mol)/num_atoms,
        "HBA": Descriptors.NumHAcceptors(mol)/num_atoms,
        "Heteroatoms": (Descriptors.NumHeteroatoms(mol)-2)/num_atoms, #exclude * end groups.
        "NHOH": Descriptors.NHOHCount(mol)/num_atoms,
        "AliphaticRings": Descriptors.NumAliphaticRings(mol)/num_atoms,
        "AmideBonds": Descriptors.NumAmideBonds(mol)/num_atoms,
        "AromaticRings": Descriptors.NumAromaticRings(mol)/num_atoms,
        "kier_flex": rdMolDescriptors.CalcHallKierAlpha(mol)
    }

