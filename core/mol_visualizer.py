"""
Molecular Visualizer Module

Utilities for drawing molecules with atom-level highlighting and color gradients.
"""
import io
import base64
from typing import Optional, Union
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def weights_to_colors(
    weights: list,
    colormap: str = "RdYlGn"
) -> dict:
    """
    Convert weights to RGB colors using matplotlib colormap.
    
    Args:
        weights: List of weights (0-1 normalized)
        colormap: Matplotlib colormap name
        
    Returns:
        dict of {atom_idx: (r, g, b)}
    """
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
    except ImportError:
        # Fallback: simple red-green gradient
        def simple_cmap(val):
            # Red (low) to Green (high)
            return (1 - val, val, 0, 1)
        cmap = simple_cmap
    
    colors = {}
    for i, w in enumerate(weights):
        rgba = cmap(w) if callable(cmap) else cmap(w)
        colors[i] = (rgba[0], rgba[1], rgba[2])
    
    return colors


def draw_mol_with_weights(
    mol: "Chem.Mol",
    atom_weights: list,
    width: int = 400,
    height: int = 300,
    colormap: str = "coolwarm",
    show_atom_indices: bool = False
) -> bytes:
    """
    Draw molecule with atoms colored by weights.
    
    Args:
        mol: RDKit Mol object
        atom_weights: List of weights per atom (0-1 normalized)
        width: Image width
        height: Image height
        colormap: Matplotlib colormap name
        show_atom_indices: Show atom indices on the image
        
    Returns:
        PNG image as bytes
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available")
    
    # Ensure 2D coordinates
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    
    # Convert weights to colors
    atom_colors = weights_to_colors(atom_weights, colormap)
    
    # Create drawer
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    
    # Configure drawing options
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()  # Black atoms for better contrast
    if show_atom_indices:
        opts.addAtomIndices = True
    
    # Highlight all atoms with their colors
    highlight_atoms = list(range(mol.GetNumAtoms()))
    highlight_colors = atom_colors
    
    # Also highlight bonds connecting important atoms
    highlight_bonds = []
    bond_colors = {}
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        # Bond importance = average of connected atoms
        bond_weight = (atom_weights[begin_idx] + atom_weights[end_idx]) / 2
        bond_idx = bond.GetIdx()
        highlight_bonds.append(bond_idx)
        rgba = weights_to_colors([bond_weight], colormap)[0]
        bond_colors[bond_idx] = rgba
    
    # Draw
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def draw_mol_with_weights_svg(
    mol: "Chem.Mol",
    atom_weights: list,
    width: int = 400,
    height: int = 300,
    colormap: str = "coolwarm"
) -> str:
    """
    Draw molecule with atoms colored by weights (SVG format).
    
    Returns:
        SVG string
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit not available")
    
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    
    atom_colors = weights_to_colors(atom_weights, colormap)
    
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    
    highlight_atoms = list(range(mol.GetNumAtoms()))
    
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors
    )
    drawer.FinishDrawing()
    
    return drawer.GetDrawingText()


def generate_attribution_image(
    smiles: str,
    atom_weights: list,
    width: int = 400,
    height: int = 300,
    colormap: str = "coolwarm",
    format: str = "png"
) -> Union[bytes, str]:
    """
    Generate attribution visualization for a SMILES string.
    
    Args:
        smiles: SMILES string
        atom_weights: List of weights per atom
        width, height: Image dimensions
        colormap: Color scheme ("coolwarm", "RdYlGn", "viridis")
        format: "png" or "svg"
        
    Returns:
        Image bytes (PNG) or string (SVG)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Validate weights
    if len(atom_weights) != mol.GetNumAtoms():
        raise ValueError(f"Weight count ({len(atom_weights)}) != atom count ({mol.GetNumAtoms()})")
    
    if format == "svg":
        return draw_mol_with_weights_svg(mol, atom_weights, width, height, colormap)
    else:
        return draw_mol_with_weights(mol, atom_weights, width, height, colormap)


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string for HTML embedding."""
    return base64.b64encode(image_bytes).decode()


def get_colorbar_html(colormap: str = "coolwarm", width: int = 200) -> str:
    """
    Generate HTML for a colorbar legend.
    """
    return f"""
    <div style="display: flex; align-items: center; margin-top: 10px;">
        <span style="margin-right: 10px;">Low</span>
        <div style="width: {width}px; height: 20px; background: linear-gradient(to right, 
            rgb(0, 0, 255), rgb(255, 255, 255), rgb(255, 0, 0));
            border-radius: 3px;"></div>
        <span style="margin-left: 10px;">High</span>
    </div>
    """
