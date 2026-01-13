"""
MACCS Keys Provider

Generates MACCS structural keys (166 bits) using RDKit.
"""
import time
from typing import Literal
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


class MACCSKeysProvider(DescriptorProvider):
    """MACCS structural keys provider (166 bits)"""
    
    @property
    def name(self) -> str:
        return "maccs_keys"
    
    @property
    def display_name(self) -> str:
        return "MACCS Keys (166-bit)"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "fingerprint"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False  # Requires RDKit-parseable SMILES
    
    @property
    def version(self) -> str:
        if RDKIT_AVAILABLE:
            from rdkit import rdBase
            return rdBase.rdkitVersion
        return "unknown"
    
    def params_schema(self) -> list[ParamSpec]:
        return []  # MACCS has no configurable parameters
    
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        result = ValidationResult()
        
        if not RDKIT_AVAILABLE:
            for i in range(len(smiles_list)):
                result.invalid_indices.append(i)
                result.messages[i] = "RDKit not available"
            return result
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                result.valid_indices.append(i)
            else:
                result.invalid_indices.append(i)
                result.messages[i] = "Invalid SMILES"
        
        return result
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        start_time = time.time()
        
        meta_rows = []
        fingerprints = []
        success_count = 0
        error_count = 0
        
        if not RDKIT_AVAILABLE:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "RDKit not available"
                })
                error_count += 1
            
            return FeaturizeResult(
                features_df=pd.DataFrame(),
                meta_df=pd.DataFrame(meta_rows),
                run_meta=self.get_run_metadata(params),
                execution_time_seconds=time.time() - start_time,
                success_count=0,
                error_count=error_count
            )
        
        for record in records:
            smiles = record.smiles_normalized or record.input_smiles_raw
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                try:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fp_array = np.array(fp)
                    fingerprints.append(fp_array)
                    
                    meta_rows.append({
                        "input_id": record.input_id,
                        "input_smiles_raw": record.input_smiles_raw,
                        "smiles_normalized": record.smiles_normalized,
                        "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                        "descriptor_status": "OK",
                        "descriptor_error": ""
                    })
                    success_count += 1
                except Exception as e:
                    fingerprints.append(None)
                    meta_rows.append({
                        "input_id": record.input_id,
                        "input_smiles_raw": record.input_smiles_raw,
                        "smiles_normalized": record.smiles_normalized,
                        "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                        "descriptor_status": "FAILED",
                        "descriptor_error": str(e)
                    })
                    error_count += 1
            else:
                fingerprints.append(None)
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "Could not parse SMILES"
                })
                error_count += 1
        
        # Create DataFrame
        if fingerprints and any(fp is not None for fp in fingerprints):
            n_bits = 167  # MACCS keys have 167 bits (0-166)
            cols = [f"maccs_{i:03d}" for i in range(n_bits)]
            
            rows = []
            for fp in fingerprints:
                if fp is not None:
                    rows.append({cols[i]: int(fp[i]) for i in range(n_bits)})
                else:
                    rows.append({col: None for col in cols})
            
            features_df = pd.DataFrame(rows)
        else:
            features_df = pd.DataFrame()
        
        return FeaturizeResult(
            features_df=features_df,
            meta_df=pd.DataFrame(meta_rows),
            run_meta=self.get_run_metadata(params),
            execution_time_seconds=time.time() - start_time,
            success_count=success_count,
            error_count=error_count
        )
