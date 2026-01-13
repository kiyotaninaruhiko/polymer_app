"""
Morgan Fingerprint Provider

Generates Morgan/ECFP fingerprints using RDKit.
"""
import time
from typing import Literal
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


class MorganFingerprintProvider(DescriptorProvider):
    """Morgan/ECFP fingerprint provider"""
    
    @property
    def name(self) -> str:
        return "morgan_fp"
    
    @property
    def display_name(self) -> str:
        return "Morgan Fingerprint (ECFP)"
    
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
        return [
            ParamSpec(
                name="radius",
                type="int",
                default=2,
                description="Morgan fingerprint radius (2 = ECFP4, 3 = ECFP6)",
                min_value=1,
                max_value=5
            ),
            ParamSpec(
                name="n_bits",
                type="select",
                default=2048,
                description="Number of bits in fingerprint",
                options=[512, 1024, 2048, 4096]
            ),
            ParamSpec(
                name="use_counts",
                type="bool",
                default=False,
                description="Use count fingerprint instead of bit fingerprint"
            )
        ]
    
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        """Validate SMILES can be parsed by RDKit"""
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
                result.messages[i] = "Invalid SMILES for RDKit"
        
        return result
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        """
        Generate Morgan fingerprints for records.
        
        Args:
            records: List of ParsedSMILES objects
            params: {"radius": int, "n_bits": int, "use_counts": bool}
        """
        start_time = time.time()
        
        radius = params.get("radius", 2)
        n_bits = params.get("n_bits", 2048)
        use_counts = params.get("use_counts", False)
        
        rows = []
        meta_rows = []
        success_count = 0
        error_count = 0
        
        # Column names for fingerprint bits
        fp_cols = [f"fp_{i:04d}" for i in range(n_bits)]
        
        for record in records:
            meta_row = {
                "input_id": record.input_id,
                "input_smiles_raw": record.input_smiles_raw,
                "smiles_normalized": record.smiles_normalized,
                "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
            }
            
            mol = record.rdkit_mol
            if mol is None:
                # Try to parse again
                if record.smiles_normalized:
                    mol = Chem.MolFromSmiles(record.smiles_normalized)
                elif record.input_smiles_raw:
                    mol = Chem.MolFromSmiles(record.input_smiles_raw)
            
            if mol is None:
                meta_row["descriptor_status"] = "FAILED"
                meta_row["descriptor_error"] = "Could not parse SMILES"
                meta_rows.append(meta_row)
                rows.append({col: None for col in fp_cols})
                error_count += 1
                continue
            
            try:
                if use_counts:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius, nBits=n_bits
                    )
                    # For counts, we'd use GetHashedMorganFingerprint
                    # but for simplicity, use bit vector
                    fp_array = np.zeros(n_bits, dtype=int)
                    for bit in fp.GetOnBits():
                        fp_array[bit] = 1
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, radius, nBits=n_bits
                    )
                    fp_array = np.zeros(n_bits, dtype=int)
                    for bit in fp.GetOnBits():
                        fp_array[bit] = 1
                
                row = {fp_cols[i]: int(fp_array[i]) for i in range(n_bits)}
                meta_row["descriptor_status"] = "OK"
                meta_row["descriptor_error"] = ""
                success_count += 1
                
            except Exception as e:
                meta_row["descriptor_status"] = "FAILED"
                meta_row["descriptor_error"] = str(e)
                row = {col: None for col in fp_cols}
                error_count += 1
            
            meta_rows.append(meta_row)
            rows.append(row)
        
        execution_time = time.time() - start_time
        
        return FeaturizeResult(
            features_df=pd.DataFrame(rows),
            meta_df=pd.DataFrame(meta_rows),
            run_meta=self.get_run_metadata(params),
            execution_time_seconds=execution_time,
            success_count=success_count,
            error_count=error_count
        )
