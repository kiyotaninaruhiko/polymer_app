"""
Descriptors Router - API endpoints for descriptor generation
"""
from fastapi import APIRouter, HTTPException
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.parsing import parse_smiles_input
from core.cache import get_cache
from providers.registry import ProviderRegistry, register_all_providers
from backend.schemas import DescriptorRequest, DescriptorResponse, DescriptorResult

router = APIRouter(prefix="/api/descriptors", tags=["descriptors"])


def init_providers():
    """Initialize providers if not already done"""
    if not ProviderRegistry.get_names():
        register_all_providers()


@router.post("/generate", response_model=DescriptorResponse)
async def generate_descriptors(request: DescriptorRequest):
    """Generate descriptors for given SMILES using specified providers"""
    init_providers()
    
    # Parse SMILES first
    smiles_text = "\n".join(request.smiles_list)
    parse_result = parse_smiles_input(smiles_text)
    valid_records = parse_result.get_valid_records()
    
    if not valid_records:
        raise HTTPException(status_code=400, detail="No valid SMILES provided")
    
    # Get cache
    cache = get_cache() if request.use_cache else None
    
    results = {}
    
    for provider_name in request.providers:
        provider = ProviderRegistry.get(provider_name)
        if not provider:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        # Get provider params
        params = request.params.get(provider_name, {})
        
        # Filter records for compatibility
        if not provider.supports_polymer_smiles:
            records_to_process = [r for r in valid_records if r.is_rdkit_compatible()]
        else:
            records_to_process = valid_records
        
        if not records_to_process:
            results[provider_name] = DescriptorResult(
                provider_name=provider.name,
                display_name=provider.display_name,
                kind=provider.kind,
                success_count=0,
                error_count=len(valid_records),
                execution_time_seconds=0,
                feature_columns=[],
                features=[],
                meta=[]
            )
            continue
        
        # Check cache
        smiles_list = [r.smiles_normalized or r.input_smiles_raw for r in records_to_process]
        cache_key = cache.make_key(smiles_list, provider.name, params) if cache else None
        
        if cache and cache.has(cache_key):
            result_data = cache.get(cache_key)
        else:
            # Run featurization
            result_data = provider.featurize(records_to_process, params)
            if cache:
                cache.set(cache_key, result_data)
        
        # Convert to response format
        features = result_data.features_df.to_dict(orient="records") if not result_data.features_df.empty else []
        meta = result_data.meta_df.to_dict(orient="records") if not result_data.meta_df.empty else []
        
        results[provider_name] = DescriptorResult(
            provider_name=provider.name,
            display_name=provider.display_name,
            kind=provider.kind,
            success_count=result_data.success_count,
            error_count=result_data.error_count,
            execution_time_seconds=result_data.execution_time_seconds,
            feature_columns=list(result_data.features_df.columns) if not result_data.features_df.empty else [],
            features=features,
            meta=meta
        )
    
    return DescriptorResponse(results=results)
