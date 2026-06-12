from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple, Any, Dict


@dataclass(frozen=True)
class BeadType:
    """
    Polymer repeat unit (bead) with chemistry and polymer-aware features.
    """

    bead_id: str
    smiles: str
    smiles_CW:  str
    smiles_CCW: str
    stereogenic: bool

    # Fingerprints
    morgan_fp: np.ndarray
    rdkit_fp: np.ndarray
    morgan_fp_CW:  np.ndarray
    morgan_fp_CCW: np.ndarray 
    rdkit_fp_CW:   np.ndarray 
    rdkit_fp_CCW:  np.ndarray 

    #Chemical Descriptors
    MolWt: float
    LogP: float
    TPSA: float
    RotBonds: float
    Rings: float
    HBD: float
    HBA: float
    Heteroatoms: float
    NHOH: float
    AliphaticRings: float
    AmideBonds: float
    AromaticRings: float
    kier_flex: float

    # Topology
    backbone_atoms: float
    sidechain_atoms: float
    avg_branch_length: float

    # Flexibility
    flexibility: float
    side_bulkiness_frac: float

    # Optional alias
    alias: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize BeadType to JSON-serializable dict.

        - numpy arrays are converted to lists
        - tuples/lists kept as lists for JSON
        """
        return {
            "bead_id": self.bead_id,
            "smiles": self.smiles,
            "smiles_CW": self.smiles_CW,
            "smiles_CCW": self.smiles_CCW,
            "alias": self.alias,
            "stereogenic": bool(self.stereogenic),
            "morgan_fp": self.morgan_fp.tolist() if hasattr(self.morgan_fp, "tolist") else list(self.morgan_fp),
            "rdkit_fp": self.rdkit_fp.tolist() if hasattr(self.rdkit_fp, "tolist") else list(self.rdkit_fp),
            "morgan_fp_CW": self.morgan_fp_CW.tolist() if hasattr(self.morgan_fp_CW, "tolist") else list(self.morgan_fp_CW),
            "morgan_fp_CCW": self.morgan_fp_CCW.tolist() if hasattr(self.morgan_fp_CCW, "tolist") else list(self.morgan_fp_CCW),
            "rdkit_fp_CW": self.rdkit_fp_CW.tolist() if hasattr(self.rdkit_fp_CW, "tolist") else list(self.rdkit_fp_CW),
            "rdkit_fp_CCW": self.rdkit_fp_CCW.tolist() if hasattr(self.rdkit_fp_CCW, "tolist") else list(self.rdkit_fp_CCW),
            "MolWt": float(self.MolWt),
            "LogP": float(self.LogP),
            "TPSA": float(self.TPSA),
            "RotBonds": float(self.RotBonds),
            "Rings": float(self.Rings),
            "HBD": float(self.HBD),
            "HBA": float(self.HBA),
            "Heteroatoms": float(self.Heteroatoms),
            "kier_flex": float(self.kier_flex),
            "backbone_atoms": float(self.backbone_atoms),
            "sidechain_atoms": float(self.sidechain_atoms),
            "flexibility": float(self.flexibility),
            "side_bulkiness_frac": float(self.side_bulkiness_frac),
            "avg_branch_length": float(self.avg_branch_length),
            "NHOH": float(self.NHOH),
            "AliphaticRings": float(self.AliphaticRings),
            "AmideBonds": float(self.AmideBonds),
            "AromaticRings": float(self.AromaticRings)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BeadType":
        """Restore BeadType from dict produced by to_dict.

        - Lists for fingerprints are converted back to numpy arrays
        - backbone/sidechain keep lists (or numbers) but converted to tuples when appropriate
        """
        morgan_fp = np.array(d["morgan_fp"]) if d.get("morgan_fp") is not None else np.array([])
        rdkit_fp = np.array(d["rdkit_fp"]) if d.get("rdkit_fp") is not None else np.array([])

        morgan_CW = np.array(d["morgan_fp_CW"]) if d.get("morgan_fp_CW") is not None else np.array([])
        morgan_CCW = np.array(d["morgan_fp_CCW"]) if d.get("morgan_fp_CCW") is not None else np.array([])

        rdkit_CW = np.array(d["rdkit_fp_CW"]) if d.get("rdkit_fp_CW") is not None else np.array([])
        rdkit_CCW = np.array(d["rdkit_fp_CCW"]) if d.get("rdkit_fp_CCW") is not None else np.array([])

        ba = float(d.get("backbone_atoms", 0))
        sa = float(d.get("sidechain_atoms", 0))

        return cls(
            bead_id=d["bead_id"],
            smiles=d["smiles"],
            smiles_CW=d["smiles_CW"],
            smiles_CCW=d["smiles_CCW"],
            stereogenic=bool(d.get("stereogenic", False)),
            rdkit_fp=rdkit_fp,
            morgan_fp=morgan_fp,
            morgan_fp_CW=morgan_CW,
            morgan_fp_CCW=morgan_CCW,
            rdkit_fp_CW=rdkit_CW,
            rdkit_fp_CCW=rdkit_CCW,
            MolWt=d.get("MolWt", 0.0),
            LogP=d.get("LogP", 0.0),
            TPSA=d.get("TPSA", 0.0),
            RotBonds=d.get("RotBonds", 0),
            Rings=d.get("Rings", 0),
            HBD=d.get("HBD", 0),
            HBA=d.get("HBA", 0),
            Heteroatoms=d.get("Heteroatoms", 0),
            NHOH=d.get("NHOH", 0),
            AliphaticRings=d.get("AliphaticRings", 0),
            AmideBonds=d.get("AmideBonds", 0),
            AromaticRings=d.get("AromaticRings",0),
            kier_flex=d.get("kier_flex", 0),
            backbone_atoms=ba,
            sidechain_atoms=sa,
            avg_branch_length=float(d.get("avg_branch_length", 0.0)),
            flexibility=float(d.get("flexibility", 0.0)),
            side_bulkiness_frac=float(d.get("side_bulkiness_frac", 0.0)),
            alias=d.get("alias"),
        )
