from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict
from .bead import BeadType

logger = logging.getLogger(__name__)

REGISTRY_PATH = Path.home() / ".pbsg" / "bead_registry.json"


class BeadRegistry:
    """
    Persistent registry of bead types.
    """

    def __init__(self):
        self._beads: Dict[str, BeadType] = {}
        self.load()

    def has(self, bead_id: str) -> bool:
        return bead_id in self._beads

    def get(self, bead_id: str) -> BeadType | None:
        return self._beads.get(bead_id)

    def find_by_smiles(self, smiles: str) -> BeadType | None:
        for bead in self._beads.values():
            if bead.smiles == smiles:
                return bead
        return None

    def register(self, bead: BeadType):
        self._beads[bead.bead_id] = bead
        self.save()

    def save(self):
        # Create the registry directory lazily, only when first writing.
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._beads.items()},
                f,
                indent=2,
            )

    def load(self):
        if not REGISTRY_PATH.exists():
            return
        if REGISTRY_PATH.stat().st_size == 0:
            # Empty file, skip loading
            return
        with open(REGISTRY_PATH) as f:
            data = json.load(f)
        for bead_id, bead_data in data.items():
            self._beads[bead_id] = BeadType.from_dict(bead_data)

    def delete_bead(self, bead_id: str):
        if bead_id not in self._beads:
            logger.warning("Bead ID %s not found in registry.", bead_id)
            return
        del self._beads[bead_id]
        self.save()
        logger.info("Bead ID %s deleted from registry.", bead_id)


# singleton instance
registry = BeadRegistry()