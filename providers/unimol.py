"""
Uni-Mol Embedding Provider

Generates 3D structure-aware molecular embeddings using Uni-Mol.
Requires unimol_tools package.

Note: This is a simplified implementation. Full Uni-Mol requires:
- CUDA GPU for efficient inference
- Pre-trained model weights
"""
import time
from typing import Literal, Optional
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from unimol_tools import UniMolRepr
    UNIMOL_AVAILABLE = True
except ImportError:
    UNIMOL_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


class UniMolProvider(DescriptorProvider):
    """Uni-Mol 3D molecular embedding provider"""
    
    def __init__(self):
        self._model = None
    
    @property
    def name(self) -> str:
        return "unimol"
    
    @property
    def display_name(self) -> str:
        return "Uni-Mol (3D Embedding)"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "embedding"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False  # Requires 3D conformer generation
    
    @property
    def version(self) -> str:
        if UNIMOL_AVAILABLE:
            try:
                import unimol_tools
                return getattr(unimol_tools, '__version__', 'unknown')
            except:
                return "unknown"
        return "not installed"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="use_gpu",
                type="bool",
                default=False,
                description="Use GPU for inference (requires CUDA)"
            )
        ]
    
    def _load_model(self, use_gpu: bool):
        """Load Uni-Mol model"""
        if self._model is None:
            self._model = UniMolRepr(data_type='molecule', remove_hs=False, use_gpu=use_gpu)
        return self._model
    
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        result = ValidationResult()
        
        if not UNIMOL_AVAILABLE:
            for i in range(len(smiles_list)):
                result.invalid_indices.append(i)
                result.messages[i] = "Uni-Mol not available. Install with: pip install unimol_tools"
            return result
        
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
        
        use_gpu = params.get("use_gpu", False)
        
        meta_rows = []
        all_embeddings = []
        success_count = 0
        error_count = 0
        
        # Check availability
        if not UNIMOL_AVAILABLE:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "Uni-Mol not available. Install with: pip install unimol_tools"
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
        
        # Load model
        try:
            model = self._load_model(use_gpu)
        except Exception as e:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": f"Failed to load Uni-Mol: {str(e)}"
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
        
        # Process SMILES
        smiles_list = [
            record.smiles_normalized or record.input_smiles_raw 
            for record in records
        ]
        
        try:
            # Get embeddings for all SMILES
            reprs = model.get_repr(smiles_list)
            
            # Handle different return types (dict with 'cls_repr' or direct list/array)
            if isinstance(reprs, dict) and 'cls_repr' in reprs:
                embeddings = reprs['cls_repr']
            else:
                embeddings = reprs
            
            for i, record in enumerate(records):
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "OK",
                    "descriptor_error": ""
                })
                all_embeddings.append(embeddings[i])
                success_count += 1
                
        except Exception as e:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": f"Inference error: {str(e)}"
                })
                all_embeddings.append(None)
                error_count += 1
        
        # Create DataFrame
        if all_embeddings and any(e is not None for e in all_embeddings):
            embed_dim = next(e.shape[0] for e in all_embeddings if e is not None)
            cols = [f"unimol_{i:04d}" for i in range(embed_dim)]
            
            rows = []
            for emb in all_embeddings:
                if emb is not None:
                    rows.append({cols[i]: float(emb[i]) for i in range(embed_dim)})
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
