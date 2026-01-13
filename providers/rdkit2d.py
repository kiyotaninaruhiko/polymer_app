"""
RDKit 2D Descriptor Provider

Generates 2D molecular descriptors using RDKit.
"""
import time
from typing import Literal
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


# Basic descriptor set
BASIC_DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "ExactMolWt": Descriptors.ExactMolWt,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "RingCount": Descriptors.RingCount,
    "TPSA": Descriptors.TPSA,
    "LabuteASA": Descriptors.LabuteASA,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
}

# Full descriptor set (all RDKit descriptors)
def get_full_descriptors() -> dict:
    """Get all available RDKit descriptors"""
    return {name: func for name, func in Descriptors.descList}


class RDKit2DProvider(DescriptorProvider):
    """RDKit 2D molecular descriptor provider"""
    
    @property
    def name(self) -> str:
        return "rdkit_2d"
    
    @property
    def display_name(self) -> str:
        return "RDKit 2D Descriptors"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "numeric"
    
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
                name="descriptor_set",
                type="select",
                default="basic",
                description="Descriptor set to compute",
                options=["basic", "full"]
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
        Compute RDKit 2D descriptors for records.
        
        Args:
            records: List of ParsedSMILES objects
            params: {"descriptor_set": "basic" | "full"}
        """
        start_time = time.time()
        
        descriptor_set = params.get("descriptor_set", "basic")
        descriptors = BASIC_DESCRIPTORS if descriptor_set == "basic" else get_full_descriptors()
        
        rows = []
        meta_rows = []
        success_count = 0
        error_count = 0
        
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
                rows.append({name: None for name in descriptors.keys()})
                error_count += 1
                continue
            
            # Calculate descriptors
            row = {}
            try:
                for name, func in descriptors.items():
                    try:
                        row[name] = func(mol)
                    except Exception as e:
                        row[name] = None
                
                meta_row["descriptor_status"] = "OK"
                meta_row["descriptor_error"] = ""
                success_count += 1
            except Exception as e:
                meta_row["descriptor_status"] = "FAILED"
                meta_row["descriptor_error"] = str(e)
                row = {name: None for name in descriptors.keys()}
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
