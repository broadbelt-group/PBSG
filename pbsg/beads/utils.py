import hashlib

def make_bead_id(smiles: str, stereogenic: bool) -> str:
    """
    Generate a stable bead ID from canonical SMILES and stereogenicity.
    """
    key = f"{smiles}|{stereogenic}"
    h = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"PB_{h}"
