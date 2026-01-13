"""
SMILES Parsing and Validation Module

Handles:
- Multi-line SMILES input parsing
- SMILES validation using RDKit
- Canonicalization
- Polymer wildcard (*) handling
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ParseStatus(Enum):
    """Status of SMILES parsing"""
    OK = "OK"
    INVALID_SMILES = "INVALID_SMILES"
    EMPTY = "EMPTY"
    POLYMER_WILDCARD = "POLYMER_WILDCARD"
    RDKIT_INCOMPATIBLE = "RDKIT_INCOMPATIBLE"


@dataclass
class ParsedSMILES:
    """Result of parsing a single SMILES string"""
    input_id: str
    input_smiles_raw: str
    smiles_normalized: Optional[str] = None
    parse_status: ParseStatus = ParseStatus.OK
    error_message: str = ""
    has_polymer_wildcard: bool = False
    rdkit_mol: Optional[object] = None  # RDKit Mol object
    
    def is_valid(self) -> bool:
        return self.parse_status == ParseStatus.OK
    
    def is_rdkit_compatible(self) -> bool:
        return self.rdkit_mol is not None


@dataclass
class ParseResult:
    """Result of parsing multiple SMILES"""
    records: list[ParsedSMILES] = field(default_factory=list)
    total_count: int = 0
    success_count: int = 0
    error_count: int = 0
    polymer_count: int = 0
    
    def get_valid_records(self) -> list[ParsedSMILES]:
        return [r for r in self.records if r.is_valid()]
    
    def get_rdkit_compatible(self) -> list[ParsedSMILES]:
        return [r for r in self.records if r.is_rdkit_compatible()]


def contains_polymer_wildcard(smiles: str) -> bool:
    """Check if SMILES contains polymer wildcard (*)"""
    # Match * that is not inside brackets [*]
    # This is a simplified check - * at polymer connection points
    return '*' in smiles


def handle_polymer_wildcard(smiles: str, mode: str = "replace", replacement: str = "[*]") -> tuple[str, bool]:
    """
    Handle polymer wildcard (*) in SMILES
    
    Args:
        smiles: Input SMILES string
        mode: "replace" | "skip" | "error"
        replacement: Replacement for * (default: [*])
    
    Returns:
        Tuple of (processed_smiles, was_modified)
    """
    if not contains_polymer_wildcard(smiles):
        return smiles, False
    
    if mode == "replace":
        # Replace bare * with replacement (but not [*] which is already valid)
        # Be careful not to replace * inside brackets
        # Simple approach: replace * not preceded by [ 
        processed = re.sub(r'(?<!\[)\*(?!\])', replacement, smiles)
        return processed, True
    elif mode == "skip":
        return smiles, False
    elif mode == "error":
        raise ValueError(f"Polymer wildcard (*) found in SMILES: {smiles}")
    else:
        return smiles, False


def validate_smiles(smiles: str) -> tuple[bool, Optional[object], str]:
    """
    Validate SMILES string using RDKit
    
    Returns:
        Tuple of (is_valid, mol_object, error_message)
    """
    if not RDKIT_AVAILABLE:
        return False, None, "RDKit not available"
    
    if not smiles or not smiles.strip():
        return False, None, "Empty SMILES"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None, "Invalid SMILES: RDKit could not parse"
        return True, mol, ""
    except Exception as e:
        return False, None, f"SMILES parsing error: {str(e)}"


def canonicalize_smiles(smiles: str, mol: Optional[object] = None) -> Optional[str]:
    """
    Canonicalize SMILES string
    
    Args:
        smiles: Input SMILES
        mol: Pre-parsed RDKit mol object (optional)
    
    Returns:
        Canonical SMILES or None if invalid
    """
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def parse_smiles_input(
    text: str,
    canonicalize: bool = True,
    wildcard_mode: str = "replace",
    wildcard_replacement: str = "[*]"
) -> ParseResult:
    """
    Parse multi-line SMILES input
    
    Supports:
    - One SMILES per line
    - CSV format: id,smiles
    
    Args:
        text: Input text (multiple lines)
        canonicalize: Whether to canonicalize SMILES
        wildcard_mode: How to handle polymer wildcard (*)
        wildcard_replacement: Replacement for * in replace mode
    
    Returns:
        ParseResult with all parsed records
    """
    result = ParseResult()
    lines = text.strip().split('\n')
    
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        result.total_count += 1
        
        # Try to parse as CSV (id,smiles)
        if ',' in line:
            parts = line.split(',', 1)
            input_id = parts[0].strip()
            smiles_raw = parts[1].strip() if len(parts) > 1 else ""
        else:
            input_id = f"mol_{idx + 1:04d}"
            smiles_raw = line
        
        # Create record
        record = ParsedSMILES(
            input_id=input_id,
            input_smiles_raw=smiles_raw
        )
        
        # Check for empty
        if not smiles_raw:
            record.parse_status = ParseStatus.EMPTY
            record.error_message = "Empty SMILES"
            result.error_count += 1
            result.records.append(record)
            continue
        
        # Check for polymer wildcard
        record.has_polymer_wildcard = contains_polymer_wildcard(smiles_raw)
        if record.has_polymer_wildcard:
            result.polymer_count += 1
        
        # Handle polymer wildcard
        try:
            processed_smiles, was_modified = handle_polymer_wildcard(
                smiles_raw, wildcard_mode, wildcard_replacement
            )
        except ValueError as e:
            record.parse_status = ParseStatus.POLYMER_WILDCARD
            record.error_message = str(e)
            result.error_count += 1
            result.records.append(record)
            continue
        
        # Validate with RDKit
        is_valid, mol, error_msg = validate_smiles(processed_smiles)
        
        if not is_valid:
            # If has polymer wildcard and RDKit can't parse, mark as incompatible (not error)
            if record.has_polymer_wildcard:
                record.parse_status = ParseStatus.RDKIT_INCOMPATIBLE
                record.error_message = f"RDKit incompatible (polymer SMILES): {error_msg}"
            else:
                record.parse_status = ParseStatus.INVALID_SMILES
                record.error_message = error_msg
                result.error_count += 1
            result.records.append(record)
            continue
        
        # Canonicalize if requested
        if canonicalize and mol:
            canonical = canonicalize_smiles(processed_smiles, mol)
            if canonical:
                record.smiles_normalized = canonical
            else:
                record.smiles_normalized = processed_smiles
        else:
            record.smiles_normalized = processed_smiles
        
        record.rdkit_mol = mol
        record.parse_status = ParseStatus.OK
        result.success_count += 1
        result.records.append(record)
    
    return result


def parse_csv_file(content: str) -> ParseResult:
    """
    Parse CSV file content
    
    Expected format:
    - Header row (optional, detected if first column is 'id' or 'smiles')
    - Columns: id, smiles (or just smiles)
    """
    lines = content.strip().split('\n')
    if not lines:
        return ParseResult()
    
    # Check for header
    first_line = lines[0].lower()
    if 'smiles' in first_line or 'id' in first_line:
        lines = lines[1:]  # Skip header
    
    return parse_smiles_input('\n'.join(lines))
