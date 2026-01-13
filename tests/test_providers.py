"""
Tests for descriptor providers
"""
import pytest
import numpy as np

from core.parsing import parse_smiles_input
from providers.registry import ProviderRegistry, register_all_providers
from providers.rdkit2d import RDKit2DProvider
from providers.morgan import MorganFingerprintProvider


class TestRDKit2DProvider:
    """Tests for RDKit 2D descriptor provider"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = RDKit2DProvider()
    
    def test_provider_properties(self):
        assert self.provider.name == "rdkit_2d"
        assert self.provider.kind == "numeric"
        assert self.provider.supports_polymer_smiles == False
    
    def test_validate(self):
        smiles_list = ["CCO", "c1ccccc1", "invalid"]
        result = self.provider.validate(smiles_list)
        
        assert 0 in result.valid_indices
        assert 1 in result.valid_indices
        assert 2 in result.invalid_indices
    
    def test_featurize_basic(self):
        input_text = "CCO\nc1ccccc1"
        parse_result = parse_smiles_input(input_text)
        records = parse_result.get_valid_records()
        
        result = self.provider.featurize(records, {"descriptor_set": "basic"})
        
        assert result.success_count == 2
        assert not result.features_df.empty
        assert "MolWt" in result.features_df.columns
        assert "TPSA" in result.features_df.columns
    
    def test_reproducibility(self):
        """Same input should give same output"""
        input_text = "CCO\nc1ccccc1"
        parse_result = parse_smiles_input(input_text)
        records = parse_result.get_valid_records()
        
        result1 = self.provider.featurize(records, {"descriptor_set": "basic"})
        result2 = self.provider.featurize(records, {"descriptor_set": "basic"})
        
        # Compare DataFrames
        assert result1.features_df.equals(result2.features_df)


class TestMorganFingerprintProvider:
    """Tests for Morgan fingerprint provider"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.provider = MorganFingerprintProvider()
    
    def test_provider_properties(self):
        assert self.provider.name == "morgan_fp"
        assert self.provider.kind == "fingerprint"
    
    def test_fingerprint_dimensions(self):
        input_text = "CCO\nc1ccccc1"
        parse_result = parse_smiles_input(input_text)
        records = parse_result.get_valid_records()
        
        # Test with different bit sizes
        for n_bits in [512, 1024, 2048]:
            result = self.provider.featurize(records, {"radius": 2, "n_bits": n_bits})
            assert len(result.features_df.columns) == n_bits
    
    def test_fingerprint_values(self):
        """Fingerprints should be binary (0 or 1)"""
        input_text = "CCO"
        parse_result = parse_smiles_input(input_text)
        records = parse_result.get_valid_records()
        
        result = self.provider.featurize(records, {"radius": 2, "n_bits": 2048})
        
        values = result.features_df.values.flatten()
        unique_values = set(values[~np.isnan(values)])
        assert unique_values.issubset({0, 1})
    
    def test_reproducibility(self):
        """Same input should give same fingerprint"""
        input_text = "CCO"
        parse_result = parse_smiles_input(input_text)
        records = parse_result.get_valid_records()
        
        params = {"radius": 2, "n_bits": 2048}
        result1 = self.provider.featurize(records, params)
        result2 = self.provider.featurize(records, params)
        
        assert result1.features_df.equals(result2.features_df)


class TestProviderRegistry:
    """Tests for provider registry"""
    
    def test_register_all(self):
        ProviderRegistry.clear()
        register_all_providers()
        
        names = ProviderRegistry.get_names()
        assert "rdkit_2d" in names
        assert "morgan_fp" in names
        assert "transformer_embed" in names
    
    def test_get_provider(self):
        ProviderRegistry.clear()
        register_all_providers()
        
        provider = ProviderRegistry.get("rdkit_2d")
        assert provider is not None
        assert provider.name == "rdkit_2d"
    
    def test_list_by_kind(self):
        ProviderRegistry.clear()
        register_all_providers()
        
        numeric = ProviderRegistry.list_by_kind("numeric")
        assert len(numeric) >= 1
        
        fingerprint = ProviderRegistry.list_by_kind("fingerprint")
        assert len(fingerprint) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
