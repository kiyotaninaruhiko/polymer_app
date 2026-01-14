"""
Atom Attribution Module

Utilities for calculating atom-level importance/contributions from various molecular representations.
"""
from typing import Optional
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def get_morgan_atom_contributions(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
    bit_weights: Optional[dict] = None
) -> dict:
    """
    Get atom contributions from Morgan fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        bit_weights: Optional dict of {bit_index: weight} for SHAP-like attribution
        
    Returns:
        dict with:
            - atom_contributions: list of floats (contribution per atom)
            - bit_info: dict of {bit: [(atom_idx, radius), ...]}
            - mol: RDKit mol object
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Get fingerprint with bit info
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bit_info)
    
    # Initialize atom contributions
    n_atoms = mol.GetNumAtoms()
    atom_contributions = np.zeros(n_atoms)
    atom_bit_counts = np.zeros(n_atoms)  # Count of bits per atom
    
    # Calculate contributions
    for bit_idx, atom_info_list in bit_info.items():
        # Get weight for this bit (default 1.0 if no weights provided)
        weight = 1.0
        if bit_weights is not None and bit_idx in bit_weights:
            weight = bit_weights[bit_idx]
        
        for atom_idx, r in atom_info_list:
            atom_contributions[atom_idx] += weight
            atom_bit_counts[atom_idx] += 1
    
    # Normalize by bit count (optional)
    # atom_contributions = atom_contributions / (atom_bit_counts + 1e-8)
    
    # Normalize to 0-1 range
    if atom_contributions.max() > 0:
        atom_contributions = atom_contributions / atom_contributions.max()
    
    return {
        "atom_contributions": atom_contributions.tolist(),
        "bit_info": {int(k): v for k, v in bit_info.items()},
        "mol": mol,
        "n_atoms": n_atoms
    }


def get_atompair_contributions(
    smiles: str,
    n_bits: int = 2048
) -> dict:
    """
    Get atom contributions from AtomPair fingerprint.
    
    Returns contributions showing how often each atom participates in atom pairs.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    from rdkit.Chem import rdMolDescriptors
    
    # Get atom pair fingerprint with info
    info = {}
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits, atomPairParameters=info)
    
    n_atoms = mol.GetNumAtoms()
    atom_contributions = np.zeros(n_atoms)
    
    # Each atom pair involves 2 atoms
    for bit_idx, pairs in info.items():
        for atom1, atom2, dist in pairs:
            atom_contributions[atom1] += 1
            atom_contributions[atom2] += 1
    
    # Normalize
    if atom_contributions.max() > 0:
        atom_contributions = atom_contributions / atom_contributions.max()
    
    return {
        "atom_contributions": atom_contributions.tolist(),
        "mol": mol,
        "n_atoms": n_atoms
    }


def get_gnn_node_importance(
    smiles: str,
    model,
    device: str = "cpu"
) -> dict:
    """
    Get atom importance from GNN node embeddings.
    
    Uses the L2 norm of node embeddings as a proxy for importance.
    """
    import torch
    from providers.molclr import smiles_to_molclr_graph
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    graph = smiles_to_molclr_graph(smiles)
    if graph is None:
        raise ValueError(f"Could not convert to graph: {smiles}")
    
    # Create batch for single molecule
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph])
    
    # Get device
    if device == "cuda" and torch.cuda.is_available():
        batch = batch.cuda()
    elif device == "mps" and torch.backends.mps.is_available():
        batch = batch.to("mps")
    
    model.eval()
    
    # Get node embeddings before pooling
    with torch.no_grad():
        # For MolCLR models, we need to get intermediate node representations
        h = model.x_embedding1(batch.x[:, 0]) + model.x_embedding2(batch.x[:, 1])
        
        for layer in range(model.num_layer):
            h = model.gnns[layer](h, batch.edge_index)
            h = model.batch_norms[layer](h)
            if layer < model.num_layer - 1:
                h = torch.relu(h)
        
        # Node importance = L2 norm of node embedding
        node_importance = torch.norm(h, dim=1).cpu().numpy()
    
    # Normalize to 0-1
    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()
    
    return {
        "atom_contributions": node_importance.tolist(),
        "mol": mol,
        "n_atoms": mol.GetNumAtoms()
    }


def get_atom_contributions(
    smiles: str,
    method: str = "morgan",
    **kwargs
) -> dict:
    """
    Unified function to get atom contributions.
    
    Args:
        smiles: SMILES string
        method: "morgan", "atompair", or "gnn"
        **kwargs: Method-specific arguments
    
    Returns:
        dict with atom_contributions and mol
    """
    if method == "morgan":
        return get_morgan_atom_contributions(smiles, **kwargs)
    elif method == "atompair":
        return get_atompair_contributions(smiles, **kwargs)
    elif method == "gnn":
        return get_gnn_node_importance(smiles, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
