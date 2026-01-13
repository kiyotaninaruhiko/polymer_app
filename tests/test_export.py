"""
Tests for export functionality
"""
import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd

from export_io.export import export_csv, export_parquet, export_json, get_export_filename


class TestExportCSV:
    """Tests for CSV export"""
    
    def test_export_csv(self):
        features_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [4.0, 5.0, 6.0]
        })
        meta_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "smiles": ["CCO", "C", "CC"]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = export_csv(features_df, meta_df, output_path)
            
            # Check file exists
            assert result_path.exists()
            
            # Check content
            loaded_df = pd.read_csv(result_path)
            assert len(loaded_df) == 3
            assert "id" in loaded_df.columns
            assert "feat1" in loaded_df.columns
        finally:
            output_path.unlink()
    
    def test_export_csv_without_meta(self):
        features_df = pd.DataFrame({
            "feat1": [1.0, 2.0]
        })
        meta_df = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = export_csv(features_df, meta_df, output_path, include_meta=False)
            loaded_df = pd.read_csv(result_path)
            assert len(loaded_df.columns) == 1
        finally:
            output_path.unlink()


class TestExportParquet:
    """Tests for Parquet export"""
    
    def test_export_parquet(self):
        features_df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [4.0, 5.0, 6.0]
        })
        meta_df = pd.DataFrame({
            "id": ["a", "b", "c"]
        })
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = export_parquet(features_df, meta_df, output_path)
            
            # Check file exists
            assert result_path.exists()
            
            # Check content
            loaded_df = pd.read_parquet(result_path)
            assert len(loaded_df) == 3
        finally:
            output_path.unlink()


class TestExportJSON:
    """Tests for JSON export"""
    
    def test_export_json(self):
        features_df = pd.DataFrame({
            "feat1": [1.0, 2.0]
        })
        meta_df = pd.DataFrame({
            "id": ["a", "b"]
        })
        run_meta = {"model": "test", "version": "1.0"}
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = export_json(features_df, meta_df, run_meta, output_path)
            
            # Check file exists
            assert result_path.exists()
            
            # Check content
            with open(result_path) as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "data" in data
            assert data["metadata"]["model"] == "test"
            assert len(data["data"]) == 2
        finally:
            output_path.unlink()


class TestGetExportFilename:
    """Tests for filename generation"""
    
    def test_csv_filename(self):
        filename = get_export_filename("rdkit_2d", "csv")
        assert filename.startswith("descriptors_rdkit_2d_")
        assert filename.endswith(".csv")
    
    def test_parquet_filename(self):
        filename = get_export_filename("morgan_fp", "parquet")
        assert "morgan_fp" in filename
        assert filename.endswith(".parquet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
