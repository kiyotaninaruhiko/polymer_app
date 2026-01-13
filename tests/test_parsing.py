"""
Tests for SMILES parsing module
"""
import pytest
from core.parsing import (
    parse_smiles_input,
    validate_smiles,
    contains_polymer_wildcard,
    handle_polymer_wildcard,
    canonicalize_smiles,
    ParseStatus
)


class TestContainsPolymerWildcard:
    """Tests for polymer wildcard detection"""
    
    def test_no_wildcard(self):
        assert contains_polymer_wildcard("CCO") == False
        assert contains_polymer_wildcard("c1ccccc1") == False
    
    def test_with_wildcard(self):
        assert contains_polymer_wildcard("*CC*") == True
        assert contains_polymer_wildcard("*C(C)(C)C*") == True
    
    def test_bracket_wildcard(self):
        # [*] is also a wildcard
        assert contains_polymer_wildcard("[*]CC[*]") == True


class TestHandlePolymerWildcard:
    """Tests for wildcard handling"""
    
    def test_replace_mode(self):
        smiles, modified = handle_polymer_wildcard("*CC*", mode="replace")
        assert "[*]" in smiles
        assert modified == True
    
    def test_skip_mode(self):
        smiles, modified = handle_polymer_wildcard("*CC*", mode="skip")
        assert smiles == "*CC*"
        assert modified == False
    
    def test_error_mode(self):
        with pytest.raises(ValueError):
            handle_polymer_wildcard("*CC*", mode="error")
    
    def test_no_wildcard(self):
        smiles, modified = handle_polymer_wildcard("CCO", mode="replace")
        assert smiles == "CCO"
        assert modified == False


class TestValidateSmiles:
    """Tests for SMILES validation"""
    
    def test_valid_smiles(self):
        is_valid, mol, error = validate_smiles("CCO")
        assert is_valid == True
        assert mol is not None
        assert error == ""
    
    def test_invalid_smiles(self):
        is_valid, mol, error = validate_smiles("not_a_smiles")
        assert is_valid == False
        assert mol is None
        assert "Invalid" in error or error != ""
    
    def test_empty_smiles(self):
        is_valid, mol, error = validate_smiles("")
        assert is_valid == False
        assert mol is None


class TestCanonicalizeSmiles:
    """Tests for SMILES canonicalization"""
    
    def test_canonicalize(self):
        # Different representations should give same canonical form
        canonical1 = canonicalize_smiles("OCC")
        canonical2 = canonicalize_smiles("CCO")
        assert canonical1 == canonical2
    
    def test_invalid_smiles(self):
        result = canonicalize_smiles("not_a_smiles")
        assert result is None


class TestParseSmilesinput:
    """Tests for multi-line SMILES parsing"""
    
    def test_basic_parsing(self):
        input_text = """CCO
c1ccccc1
CC(C)O"""
        result = parse_smiles_input(input_text)
        
        assert result.total_count == 3
        assert result.success_count == 3
        assert result.error_count == 0
    
    def test_with_ids(self):
        input_text = """mol1,CCO
mol2,c1ccccc1"""
        result = parse_smiles_input(input_text)
        
        assert result.total_count == 2
        assert result.records[0].input_id == "mol1"
        assert result.records[1].input_id == "mol2"
    
    def test_mixed_valid_invalid(self):
        input_text = """CCO
invalid_smiles
c1ccccc1"""
        result = parse_smiles_input(input_text)
        
        assert result.total_count == 3
        assert result.success_count == 2
        assert result.error_count == 1
    
    def test_polymer_smiles(self):
        input_text = """*CC*
CCO"""
        result = parse_smiles_input(input_text, wildcard_mode="replace")
        
        assert result.polymer_count == 1
        assert result.records[0].has_polymer_wildcard == True
    
    def test_empty_lines_ignored(self):
        input_text = """CCO

c1ccccc1

"""
        result = parse_smiles_input(input_text)
        
        assert result.total_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
