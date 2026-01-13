"""
AtomPair and Topological Torsion Fingerprint Providers

Generates atom pair and topological torsion fingerprints using RDKit.
"""
import time
from typing import Literal
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.AtomPairs import Pairs, Torsions
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


class AtomPairFPProvider(DescriptorProvider):
    """Atom Pair fingerprint provider"""
    
    @property
    def name(self) -> str:
        return "atompair_fp"
    
    @property
    def display_name(self) -> str:
        return "Atom Pair Fingerprint"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "fingerprint"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False
    
    @property
    def version(self) -> str:
        if RDKIT_AVAILABLE:
            from rdkit import rdBase
            return rdBase.rdkitVersion
        return "unknown"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="nBits",
                type="select",
                default=2048,
                description="Number of bits in fingerprint",
                options=[512, 1024, 2048, 4096]
            ),
            ParamSpec(
                name="minLength",
                type="int",
                default=1,
                description="Minimum path length",
                min_value=1,
                max_value=5
            ),
            ParamSpec(
                name="maxLength",
                type="int",
                default=30,
                description="Maximum path length",
                min_value=5,
                max_value=50
            )
        ]
    
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
        
        n_bits = params.get("nBits", 2048)
        min_length = params.get("minLength", 1)
        max_length = params.get("maxLength", 30)
        
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
                    fp = Pairs.GetHashedAtomPairFingerprintAsBitVect(
                        mol, 
                        nBits=n_bits,
                        minLength=min_length,
                        maxLength=max_length
                    )
                    fp_array = np.zeros(n_bits, dtype=int)
                    DataStructs.ConvertToNumpyArray(fp, fp_array)
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
            cols = [f"ap_{i:04d}" for i in range(n_bits)]
            
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


class TopologicalTorsionFPProvider(DescriptorProvider):
    """Topological Torsion fingerprint provider"""
    
    @property
    def name(self) -> str:
        return "torsion_fp"
    
    @property
    def display_name(self) -> str:
        return "Topological Torsion FP"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "fingerprint"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False
    
    @property
    def version(self) -> str:
        if RDKIT_AVAILABLE:
            from rdkit import rdBase
            return rdBase.rdkitVersion
        return "unknown"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="nBits",
                type="select",
                default=2048,
                description="Number of bits in fingerprint",
                options=[512, 1024, 2048, 4096]
            ),
            ParamSpec(
                name="targetSize",
                type="int",
                default=4,
                description="Target size of torsion paths",
                min_value=2,
                max_value=7
            )
        ]
    
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
        
        n_bits = params.get("nBits", 2048)
        target_size = params.get("targetSize", 4)
        
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
                    fp = Torsions.GetHashedTopologicalTorsionFingerprintAsBitVect(
                        mol, 
                        nBits=n_bits,
                        targetSize=target_size
                    )
                    fp_array = np.zeros(n_bits, dtype=int)
                    DataStructs.ConvertToNumpyArray(fp, fp_array)
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
            cols = [f"tt_{i:04d}" for i in range(n_bits)]
            
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
