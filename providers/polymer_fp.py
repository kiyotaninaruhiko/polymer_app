"""
Polymer Fingerprint Provider

Generates polymer-specific fingerprints based on monomer-level descriptors.
This implements a simple polymer fingerprint based on:
- Monomer unit features (RDKit descriptors)
- Polymer structural features (repeat unit count, end groups)
"""
import time
from typing import Literal, Optional
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


# Monomer-level descriptors for polymer fingerprint
MONOMER_DESCRIPTORS = [
    ('MolWt', Descriptors.MolWt),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('FractionCSP3', Descriptors.FractionCSP3),
    ('TPSA', Descriptors.TPSA),
    ('MolLogP', Descriptors.MolLogP),
    ('NumAliphaticRings', Descriptors.NumAliphaticRings),
] if RDKIT_AVAILABLE else []


def extract_polymer_features(smiles: str) -> Optional[dict]:
    """Extract polymer-specific features from SMILES"""
    if not RDKIT_AVAILABLE:
        return None
    
    # Check for polymer wildcard
    has_wildcard = '*' in smiles
    
    # Try to parse SMILES (with wildcard handling)
    clean_smiles = smiles.replace('*', '[*]') if has_wildcard else smiles
    mol = Chem.MolFromSmiles(clean_smiles)
    
    if mol is None:
        return None
    
    features = {}
    
    # Count wildcards (attachment points)
    wildcard_count = smiles.count('*')
    features['num_attachment_points'] = wildcard_count
    features['is_linear'] = 1 if wildcard_count == 2 else 0
    features['is_branched'] = 1 if wildcard_count > 2 else 0
    features['is_monofunctional'] = 1 if wildcard_count == 1 else 0
    
    # Basic monomer descriptors
    for name, func in MONOMER_DESCRIPTORS:
        try:
            features[f'monomer_{name}'] = func(mol)
        except:
            features[f'monomer_{name}'] = 0.0
    
    # Ring information
    ring_info = mol.GetRingInfo()
    features['num_rings'] = ring_info.NumRings()
    
    # Atom type counts
    atom_counts = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
    
    for symbol in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'P', 'Si']:
        features[f'count_{symbol}'] = atom_counts.get(symbol, 0)
    
    # Morgan fingerprint for structural features (compressed)
    try:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
        morgan_array = np.zeros(256, dtype=int)
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
        for i in range(256):
            features[f'morgan_{i:03d}'] = morgan_array[i]
    except:
        for i in range(256):
            features[f'morgan_{i:03d}'] = 0
    
    return features


class PolymerFingerprintProvider(DescriptorProvider):
    """Polymer-specific fingerprint provider"""
    
    @property
    def name(self) -> str:
        return "polymer_fp"
    
    @property
    def display_name(self) -> str:
        return "Polymer Fingerprint (PFP)"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "fingerprint"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return True  # This provider is designed for polymer SMILES
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="include_morgan",
                type="bool",
                default=True,
                description="Include Morgan fingerprint bits"
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
            clean_smiles = smiles.replace('*', '[*]')
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is not None:
                result.valid_indices.append(i)
            else:
                result.invalid_indices.append(i)
                result.messages[i] = "Invalid SMILES"
        
        return result
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        start_time = time.time()
        
        include_morgan = params.get("include_morgan", True)
        
        meta_rows = []
        all_features = []
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
            smiles = record.input_smiles_raw  # Use raw SMILES to preserve wildcards
            features = extract_polymer_features(smiles)
            
            if features is not None:
                # Filter morgan bits if not included
                if not include_morgan:
                    features = {k: v for k, v in features.items() if not k.startswith('morgan_')}
                
                all_features.append(features)
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "OK",
                    "descriptor_error": ""
                })
                success_count += 1
            else:
                all_features.append(None)
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "Could not extract polymer features"
                })
                error_count += 1
        
        # Create DataFrame
        if all_features and any(f is not None for f in all_features):
            # Get all columns from first valid entry
            first_valid = next(f for f in all_features if f is not None)
            columns = list(first_valid.keys())
            
            rows = []
            for feat in all_features:
                if feat is not None:
                    rows.append({col: feat.get(col) for col in columns})
                else:
                    rows.append({col: None for col in columns})
            
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
