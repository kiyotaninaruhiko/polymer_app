"""
Caching Module

Hash-based caching for descriptor generation results.
"""
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CacheKey:
    """Cache key based on input + settings"""
    smiles_hash: str
    provider_name: str
    params_hash: str
    
    def to_string(self) -> str:
        return f"{self.provider_name}_{self.smiles_hash}_{self.params_hash}"


class DescriptorCache:
    """
    Cache for descriptor generation results.
    
    Uses hash of (SMILES list + provider + params) as key.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".poly_pred_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_smiles(self, smiles_list: list[str]) -> str:
        """Create hash of SMILES list"""
        content = "\n".join(sorted(smiles_list))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _hash_params(self, params: dict) -> str:
        """Create hash of parameters"""
        content = json.dumps(params, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def make_key(
        self, 
        smiles_list: list[str], 
        provider_name: str, 
        params: dict
    ) -> CacheKey:
        """Create a cache key"""
        return CacheKey(
            smiles_hash=self._hash_smiles(smiles_list),
            provider_name=provider_name,
            params_hash=self._hash_params(params)
        )
    
    def _get_cache_path(self, key: CacheKey) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key.to_string()}.pkl"
    
    def get(self, key: CacheKey) -> Optional[Any]:
        """
        Retrieve cached result.
        
        Returns:
            Cached FeaturizeResult or None if not found
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    
    def set(self, key: CacheKey, value: Any) -> None:
        """Store result in cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass  # Silently fail on cache write errors
    
    def has(self, key: CacheKey) -> bool:
        """Check if key exists in cache"""
        return self._get_cache_path(key).exists()
    
    def clear(self) -> int:
        """
        Clear all cache files.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        return count
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            "cache_dir": str(self.cache_dir),
            "num_entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }


# Global cache instance
_cache: Optional[DescriptorCache] = None


def get_cache() -> DescriptorCache:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = DescriptorCache()
    return _cache
