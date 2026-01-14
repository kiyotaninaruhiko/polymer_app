"""
Providers Router - API endpoints for provider information
"""
from fastapi import APIRouter
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from providers.registry import ProviderRegistry, register_all_providers
from backend.schemas import ProviderInfo, ProvidersResponse, ParamSpec

router = APIRouter(prefix="/api/providers", tags=["providers"])


def init_providers():
    """Initialize providers if not already done"""
    if not ProviderRegistry.get_names():
        register_all_providers()


@router.get("", response_model=ProvidersResponse)
async def list_providers():
    """List all available descriptor providers"""
    init_providers()
    
    providers = []
    for provider in ProviderRegistry.list_all():
        params = [
            ParamSpec(
                name=p.name,
                type=p.type,
                default=p.default,
                description=p.description,
                options=p.options,
                min_value=p.min_value,
                max_value=p.max_value
            )
            for p in provider.params_schema()
        ]
        
        providers.append(ProviderInfo(
            name=provider.name,
            display_name=provider.display_name,
            kind=provider.kind,
            supports_polymer_smiles=provider.supports_polymer_smiles,
            version=provider.version,
            params_schema=params
        ))
    
    return ProvidersResponse(providers=providers)


@router.get("/{name}", response_model=ProviderInfo)
async def get_provider(name: str):
    """Get information about a specific provider"""
    init_providers()
    
    provider = ProviderRegistry.get(name)
    if not provider:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    
    params = [
        ParamSpec(
            name=p.name,
            type=p.type,
            default=p.default,
            description=p.description,
            options=p.options,
            min_value=p.min_value,
            max_value=p.max_value
        )
        for p in provider.params_schema()
    ]
    
    return ProviderInfo(
        name=provider.name,
        display_name=provider.display_name,
        kind=provider.kind,
        supports_polymer_smiles=provider.supports_polymer_smiles,
        version=provider.version,
        params_schema=params
    )


@router.get("/by-kind/{kind}")
async def list_providers_by_kind(kind: str):
    """List providers by kind (numeric, fingerprint, embedding)"""
    init_providers()
    
    providers = ProviderRegistry.list_by_kind(kind)
    result = []
    
    for provider in providers:
        params = [
            ParamSpec(
                name=p.name,
                type=p.type,
                default=p.default,
                description=p.description,
                options=p.options,
                min_value=p.min_value,
                max_value=p.max_value
            )
            for p in provider.params_schema()
        ]
        
        result.append(ProviderInfo(
            name=provider.name,
            display_name=provider.display_name,
            kind=provider.kind,
            supports_polymer_smiles=provider.supports_polymer_smiles,
            version=provider.version,
            params_schema=params
        ))
    
    return {"providers": result}
