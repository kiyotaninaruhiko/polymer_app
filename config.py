"""
Polymer Descriptor App - Configuration
"""
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime

APP_VERSION = "1.0.0"
APP_NAME = "Polymer SMILES Descriptor Generator"

# Polymer wildcard handling modes
WILDCARD_MODES = Literal["replace", "skip", "error"]

# Default wildcard replacement: * -> [*] (RDKit compatible dummy atom)
DEFAULT_WILDCARD_REPLACEMENT = "[*]"

@dataclass
class WildcardConfig:
    """Configuration for polymer wildcard (*) handling"""
    mode: WILDCARD_MODES = "replace"
    replacement: str = DEFAULT_WILDCARD_REPLACEMENT

@dataclass
class RDKit2DConfig:
    """RDKit 2D descriptor configuration"""
    descriptor_set: Literal["basic", "full"] = "basic"
    
    # Basic descriptors
    BASIC_DESCRIPTORS = [
        "MolWt", "ExactMolWt", "HeavyAtomMolWt",
        "NumHDonors", "NumHAcceptors", "NumHeteroatoms",
        "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
        "RingCount", "TPSA", "LabuteASA",
        "MolLogP", "MolMR",
        "FractionCSP3", "NumValenceElectrons"
    ]

@dataclass
class MorganConfig:
    """Morgan fingerprint configuration"""
    radius: int = 2
    n_bits: int = 2048
    use_counts: bool = False

@dataclass 
class TransformerEmbedConfig:
    """Transformer embedding configuration"""
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    pooling: Literal["cls", "mean", "max"] = "mean"
    max_length: int = 512
    device: str = "cpu"
    batch_size: int = 32

# Model presets for UI
MODEL_PRESETS = {
    "ChemBERTa-zinc": {
        "model_name": "seyonec/ChemBERTa-zinc-base-v1",
        "description": "ChemBERTa pretrained on ZINC dataset"
    },
    "ChemBERTa-pubchem": {
        "model_name": "seyonec/PubChem10M_SMILES_BPE_450k", 
        "description": "ChemBERTa pretrained on PubChem"
    },
    "MoLFormer": {
        "model_name": "ibm/MoLFormer-XL-both-10pct",
        "description": "MoLFormer large-scale molecular model"
    },
    "PolyNC": {
        "model_name": "hkqiu/PolyNC",
        "description": "PolyNC: Natural and chemical language model for polymer properties"
    }
}

@dataclass
class RunMetadata:
    """Metadata for each descriptor generation run"""
    app_version: str = APP_VERSION
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    provider_name: str = ""
    provider_version: str = ""
    provider_params: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "app_version": self.app_version,
            "run_timestamp": self.run_timestamp,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "provider_params": self.provider_params
        }
