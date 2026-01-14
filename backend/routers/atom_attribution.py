"""
Atom Attribution Router - API endpoints for atom-level importance visualization
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.atom_attribution import get_atom_contributions
from core.mol_visualizer import generate_attribution_image, image_to_base64

router = APIRouter(prefix="/api/atom-attribution", tags=["atom-attribution"])


class AtomAttributionRequest(BaseModel):
    """Request for atom attribution calculation"""
    smiles: str = Field(..., description="SMILES string")
    method: str = Field(default="morgan", description="Attribution method: morgan, atompair")
    radius: int = Field(default=2, description="Morgan FP radius (if method=morgan)")
    n_bits: int = Field(default=2048, description="Number of bits")


class AtomAttributionResponse(BaseModel):
    """Response with atom contributions and visualization"""
    smiles: str
    method: str
    n_atoms: int
    atom_contributions: list[float]
    image_base64: Optional[str] = None


@router.post("", response_model=AtomAttributionResponse)
async def calculate_atom_attribution(request: AtomAttributionRequest):
    """Calculate atom-level importance/attribution"""
    try:
        if request.method == "morgan":
            result = get_atom_contributions(
                request.smiles,
                method="morgan",
                radius=request.radius,
                n_bits=request.n_bits
            )
        elif request.method == "atompair":
            result = get_atom_contributions(
                request.smiles,
                method="atompair",
                n_bits=request.n_bits
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        # Generate visualization
        image_bytes = generate_attribution_image(
            request.smiles,
            result["atom_contributions"],
            width=400,
            height=300,
            colormap="coolwarm",
            format="png"
        )
        image_b64 = image_to_base64(image_bytes)
        
        return AtomAttributionResponse(
            smiles=request.smiles,
            method=request.method,
            n_atoms=result["n_atoms"],
            atom_contributions=result["atom_contributions"],
            image_base64=image_b64
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image")
async def get_attribution_image(
    smiles: str,
    method: str = "morgan",
    colormap: str = "coolwarm",
    width: int = 400,
    height: int = 300
):
    """Get atom attribution as PNG image"""
    try:
        result = get_atom_contributions(smiles, method=method)
        image_bytes = generate_attribution_image(
            smiles,
            result["atom_contributions"],
            width=width,
            height=height,
            colormap=colormap,
            format="png"
        )
        return Response(content=image_bytes, media_type="image/png")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
