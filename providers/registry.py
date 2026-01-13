"""
Provider Registry

Manages registration and discovery of descriptor providers.
"""
from typing import Dict, Type, Optional
from .base import DescriptorProvider


class ProviderRegistry:
    """Registry for descriptor providers"""
    
    _providers: Dict[str, DescriptorProvider] = {}
    
    @classmethod
    def register(cls, provider: DescriptorProvider) -> None:
        """Register a provider instance"""
        cls._providers[provider.name] = provider
    
    @classmethod
    def get(cls, name: str) -> Optional[DescriptorProvider]:
        """Get a provider by name"""
        return cls._providers.get(name)
    
    @classmethod
    def list_all(cls) -> list[DescriptorProvider]:
        """List all registered providers"""
        return list(cls._providers.values())
    
    @classmethod
    def list_by_kind(cls, kind: str) -> list[DescriptorProvider]:
        """List providers by kind (numeric, fingerprint, embedding)"""
        return [p for p in cls._providers.values() if p.kind == kind]
    
    @classmethod
    def get_names(cls) -> list[str]:
        """Get list of all provider names"""
        return list(cls._providers.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (for testing)"""
        cls._providers.clear()


def register_all_providers() -> None:
    """Register all available providers"""
    # Import providers here to avoid circular imports
    from .rdkit2d import RDKit2DProvider
    from .morgan import MorganFingerprintProvider
    from .transformer_embed import TransformerEmbedProvider
    from .gnn_embed import GNNEmbedProvider
    
    # Register instances
    ProviderRegistry.register(RDKit2DProvider())
    ProviderRegistry.register(MorganFingerprintProvider())
    ProviderRegistry.register(TransformerEmbedProvider())
    ProviderRegistry.register(GNNEmbedProvider())

