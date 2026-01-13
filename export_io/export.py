"""
Export Module

Handles exporting results to CSV, Parquet, and JSON formats.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import pandas as pd


def export_csv(
    features_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_path: Union[str, Path],
    include_meta: bool = True
) -> Path:
    """
    Export results to CSV format.
    
    Args:
        features_df: DataFrame with feature columns
        meta_df: DataFrame with metadata columns
        output_path: Output file path
        include_meta: Whether to include metadata columns
    
    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    
    if include_meta and not meta_df.empty:
        combined_df = pd.concat([meta_df, features_df], axis=1)
    else:
        combined_df = features_df
    
    combined_df.to_csv(output_path, index=False)
    return output_path


def export_parquet(
    features_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    output_path: Union[str, Path],
    include_meta: bool = True
) -> Path:
    """
    Export results to Parquet format.
    
    Parquet is recommended for high-dimensional data (fingerprints, embeddings).
    
    Args:
        features_df: DataFrame with feature columns
        meta_df: DataFrame with metadata columns
        output_path: Output file path
        include_meta: Whether to include metadata columns
    
    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    
    if include_meta and not meta_df.empty:
        combined_df = pd.concat([meta_df, features_df], axis=1)
    else:
        combined_df = features_df
    
    combined_df.to_parquet(output_path, index=False)
    return output_path


def export_json(
    features_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    run_meta: dict,
    output_path: Union[str, Path]
) -> Path:
    """
    Export results to JSON format with full metadata.
    
    Args:
        features_df: DataFrame with feature columns
        meta_df: DataFrame with metadata columns
        run_meta: Run metadata dictionary
        output_path: Output file path
    
    Returns:
        Path to the exported file
    """
    output_path = Path(output_path)
    
    # Combine data
    if not meta_df.empty:
        combined_df = pd.concat([meta_df, features_df], axis=1)
    else:
        combined_df = features_df
    
    # Create JSON structure
    export_data = {
        "metadata": {
            "export_timestamp": datetime.now().isoformat(),
            "row_count": len(combined_df),
            "feature_columns": list(features_df.columns),
            "meta_columns": list(meta_df.columns),
            **run_meta
        },
        "data": combined_df.to_dict(orient="records")
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def get_export_filename(
    provider_name: str,
    format: str,
    timestamp: Optional[datetime] = None
) -> str:
    """
    Generate a timestamped export filename.
    
    Args:
        provider_name: Name of the descriptor provider
        format: Export format (csv, parquet, json)
        timestamp: Optional timestamp (defaults to now)
    
    Returns:
        Formatted filename
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"descriptors_{provider_name}_{ts_str}.{format}"
