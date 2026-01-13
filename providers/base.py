"""
Descriptor Provider Base Class and Common Types

Defines the abstract interface for all descriptor providers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Any, Optional
import pandas as pd
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of SMILES validation by a provider"""
    valid_indices: list[int] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)
    messages: dict[int, str] = field(default_factory=dict)  # idx -> error message
    
    @property
    def all_valid(self) -> bool:
        return len(self.invalid_indices) == 0


@dataclass
class FeaturizeResult:
    """Result of featurization by a provider"""
    features_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    run_meta: dict = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    success_count: int = 0
    error_count: int = 0
    
    def to_combined_df(self) -> pd.DataFrame:
        """Combine meta and features into single DataFrame"""
        if self.meta_df.empty:
            return self.features_df
        if self.features_df.empty:
            return self.meta_df
        return pd.concat([self.meta_df, self.features_df], axis=1)


@dataclass
class ParamSpec:
    """Specification for a provider parameter (for UI generation)"""
    name: str
    type: Literal["int", "float", "str", "bool", "select"]
    default: Any
    description: str
    options: Optional[list] = None  # For select type
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class DescriptorProvider(ABC):
    """
    Abstract base class for descriptor providers.
    
    All descriptor providers (RDKit 2D, Morgan FP, Transformer) 
    must implement this interface.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for UI"""
        pass
    
    @property
    @abstractmethod
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        """Type of descriptor produced"""
        pass
    
    @property
    @abstractmethod
    def supports_polymer_smiles(self) -> bool:
        """Whether this provider can handle polymer SMILES with wildcards"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version string"""
        pass
    
    @abstractmethod
    def params_schema(self) -> list[ParamSpec]:
        """
        Returns parameter specifications for UI generation.
        Each ParamSpec defines a configurable parameter.
        """
        pass
    
    @abstractmethod
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        """
        Validate that SMILES can be processed by this provider.
        
        Args:
            smiles_list: List of SMILES strings to validate
            
        Returns:
            ValidationResult with valid/invalid indices
        """
        pass
    
    @abstractmethod
    def featurize(
        self, 
        records: list, 
        params: dict
    ) -> FeaturizeResult:
        """
        Generate descriptors for input records.
        
        Args:
            records: List of ParsedSMILES records
            params: Configuration parameters
            
        Returns:
            FeaturizeResult with features DataFrame and metadata
        """
        pass
    
    def get_run_metadata(self, params: dict) -> dict:
        """Generate run metadata for reproducibility"""
        return {
            "provider_name": self.name,
            "provider_kind": self.kind,
            "provider_version": self.version,
            "provider_params": params,
            "run_timestamp": datetime.now().isoformat()
        }
