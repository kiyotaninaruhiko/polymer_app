"""
Pydantic Schemas for API
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


# ============ Parse Schemas ============

class ParseRequest(BaseModel):
    """Request for SMILES parsing"""
    smiles_text: str = Field(..., description="SMILES text (one per line or CSV format)")
    canonicalize: bool = Field(default=True, description="Canonicalize SMILES")
    wildcard_mode: str = Field(default="replace", description="Wildcard handling: replace, skip, error")


class ParsedRecord(BaseModel):
    """Single parsed SMILES record"""
    input_id: str
    input_smiles_raw: str
    smiles_normalized: Optional[str] = None
    parse_status: str
    error_message: str = ""
    has_polymer_wildcard: bool = False


class ParseResponse(BaseModel):
    """Response for SMILES parsing"""
    total_count: int
    success_count: int
    error_count: int
    polymer_count: int
    records: list[ParsedRecord]


# ============ Provider Schemas ============

class ParamSpec(BaseModel):
    """Parameter specification"""
    name: str
    type: str
    default: Any
    description: str
    options: Optional[list] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ProviderInfo(BaseModel):
    """Provider information"""
    name: str
    display_name: str
    kind: str  # numeric, fingerprint, embedding
    supports_polymer_smiles: bool
    version: str
    params_schema: list[ParamSpec]


class ProvidersResponse(BaseModel):
    """Response for providers list"""
    providers: list[ProviderInfo]


# ============ Descriptor Schemas ============

class DescriptorRequest(BaseModel):
    """Request for descriptor generation"""
    smiles_list: list[str] = Field(..., description="List of SMILES to process")
    providers: list[str] = Field(..., description="List of provider names to use")
    params: dict[str, dict] = Field(default_factory=dict, description="Provider-specific parameters")
    use_cache: bool = Field(default=True, description="Use caching")


class DescriptorResult(BaseModel):
    """Result for a single provider"""
    provider_name: str
    display_name: str
    kind: str
    success_count: int
    error_count: int
    execution_time_seconds: float
    feature_columns: list[str]
    features: list[dict]  # List of feature dictionaries
    meta: list[dict]  # List of meta dictionaries


class DescriptorResponse(BaseModel):
    """Response for descriptor generation"""
    results: dict[str, DescriptorResult]


# ============ Export Schemas ============

class ExportRequest(BaseModel):
    """Request for exporting results"""
    provider_name: str
    format: str = Field(default="csv", description="Export format: csv, parquet, json")
