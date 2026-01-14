"""
Parse Router - API endpoints for SMILES parsing
"""
from fastapi import APIRouter
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.parsing import parse_smiles_input
from backend.schemas import ParseRequest, ParseResponse, ParsedRecord

router = APIRouter(prefix="/api/parse", tags=["parse"])


@router.post("", response_model=ParseResponse)
async def parse_smiles(request: ParseRequest):
    """Parse SMILES text and validate"""
    result = parse_smiles_input(
        text=request.smiles_text,
        canonicalize=request.canonicalize,
        wildcard_mode=request.wildcard_mode
    )
    
    records = []
    for r in result.records:
        records.append(ParsedRecord(
            input_id=r.input_id,
            input_smiles_raw=r.input_smiles_raw,
            smiles_normalized=r.smiles_normalized,
            parse_status=r.parse_status.value if hasattr(r.parse_status, 'value') else str(r.parse_status),
            error_message=r.error_message,
            has_polymer_wildcard=r.has_polymer_wildcard
        ))
    
    return ParseResponse(
        total_count=result.total_count,
        success_count=result.success_count,
        error_count=result.error_count,
        polymer_count=result.polymer_count,
        records=records
    )
