"""
Transformer Embedding Provider

Generates molecular embeddings using Transformer models (e.g., ChemBERTa, MoLFormer).
"""
import time
from typing import Literal, Optional
import pandas as pd
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


class TransformerEmbedProvider(DescriptorProvider):
    """Transformer-based molecular embedding provider"""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_name = None
    
    @property
    def name(self) -> str:
        return "transformer_embed"
    
    @property
    def display_name(self) -> str:
        return "Transformer Embedding"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "embedding"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return True  # Transformers can process any string
    
    @property
    def version(self) -> str:
        if TRANSFORMERS_AVAILABLE:
            import transformers
            return transformers.__version__
        return "unknown"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="model_name",
                type="str",
                default="seyonec/ChemBERTa-zinc-base-v1",
                description="HuggingFace model ID or local path"
            ),
            ParamSpec(
                name="pooling",
                type="select",
                default="mean",
                description="Pooling strategy for sequence embeddings",
                options=["cls", "mean", "max"]
            ),
            ParamSpec(
                name="max_length",
                type="int",
                default=512,
                description="Maximum sequence length",
                min_value=64,
                max_value=1024
            ),
            ParamSpec(
                name="device",
                type="select",
                default="cpu",
                description="Device for inference",
                options=["cpu", "cuda", "mps"]
            ),
            ParamSpec(
                name="batch_size",
                type="int",
                default=32,
                description="Batch size for inference",
                min_value=1,
                max_value=128
            )
        ]
    
    def _load_model(self, model_name: str, device: str) -> tuple:
        """Load model and tokenizer if not already loaded"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library not available. Install with: pip install transformers torch")
        
        if self._model is not None and self._current_model_name == model_name:
            return self._tokenizer, self._model
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            self._model.eval()
            
            # Move to device
            if device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            elif device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
            
            self._current_model_name = model_name
            return self._tokenizer, self._model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
    
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        """Transformers can process any string, so all are valid"""
        result = ValidationResult()
        for i in range(len(smiles_list)):
            result.valid_indices.append(i)
        return result
    
    def _pool_embeddings(
        self, 
        hidden_states: "torch.Tensor", 
        attention_mask: "torch.Tensor",
        pooling: str
    ) -> np.ndarray:
        """Apply pooling to get fixed-size embeddings"""
        if pooling == "cls":
            # Use CLS token embedding
            embeddings = hidden_states[:, 0, :]
        elif pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif pooling == "max":
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask_expanded == 0] = -1e9
            embeddings = torch.max(hidden_states, dim=1)[0]
        else:
            embeddings = hidden_states[:, 0, :]
        
        return embeddings.cpu().numpy()
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        """
        Generate transformer embeddings for records.
        
        Args:
            records: List of ParsedSMILES objects
            params: Model configuration
        """
        start_time = time.time()
        
        model_name = params.get("model_name", "seyonec/ChemBERTa-zinc-base-v1")
        pooling = params.get("pooling", "mean")
        max_length = params.get("max_length", 512)
        device = params.get("device", "cpu")
        batch_size = params.get("batch_size", 32)
        
        meta_rows = []
        all_embeddings = []
        success_count = 0
        error_count = 0
        
        # Check if transformers available
        if not TRANSFORMERS_AVAILABLE:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "transformers library not available"
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
            tokenizer, model = self._load_model(model_name, device)
        except Exception as e:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": str(e)
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
        
        # Get device
        model_device = next(model.parameters()).device
        
        # Process in batches
        smiles_list = [
            record.smiles_normalized or record.input_smiles_raw 
            for record in records
        ]
        
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_smiles = smiles_list[i:i + batch_size]
            
            try:
                # Tokenize
                inputs = tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # Get embeddings - handle encoder-decoder models
                with torch.no_grad():
                    # Check if model is encoder-decoder (like T5, PolyNC)
                    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
                        # Use only the encoder for embeddings
                        encoder_outputs = model.encoder(**inputs)
                        hidden_states = encoder_outputs.last_hidden_state
                    elif hasattr(model, 'get_encoder'):
                        # Alternative: use get_encoder() method
                        encoder = model.get_encoder()
                        encoder_outputs = encoder(**inputs)
                        hidden_states = encoder_outputs.last_hidden_state
                    else:
                        # Standard encoder-only model (BERT, RoBERTa, etc.)
                        outputs = model(**inputs)
                        hidden_states = outputs.last_hidden_state
                
                # Pool embeddings
                batch_embeddings = self._pool_embeddings(
                    hidden_states, 
                    inputs["attention_mask"],
                    pooling
                )
                
                # Add results
                for j, record in enumerate(batch_records):
                    meta_rows.append({
                        "input_id": record.input_id,
                        "input_smiles_raw": record.input_smiles_raw,
                        "smiles_normalized": record.smiles_normalized,
                        "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                        "descriptor_status": "OK",
                        "descriptor_error": ""
                    })
                    all_embeddings.append(batch_embeddings[j])
                    success_count += 1
                    
            except Exception as e:
                for record in batch_records:
                    meta_rows.append({
                        "input_id": record.input_id,
                        "input_smiles_raw": record.input_smiles_raw,
                        "smiles_normalized": record.smiles_normalized,
                        "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                        "descriptor_status": "FAILED",
                        "descriptor_error": str(e)
                    })
                    all_embeddings.append(None)
                    error_count += 1
        
        # Create features DataFrame
        if all_embeddings and any(e is not None for e in all_embeddings):
            # Get embedding dimension from first valid embedding
            embed_dim = next(e.shape[0] for e in all_embeddings if e is not None)
            emb_cols = [f"emb_{i:04d}" for i in range(embed_dim)]
            
            rows = []
            for emb in all_embeddings:
                if emb is not None:
                    rows.append({emb_cols[i]: float(emb[i]) for i in range(embed_dim)})
                else:
                    rows.append({col: None for col in emb_cols})
            
            features_df = pd.DataFrame(rows)
        else:
            features_df = pd.DataFrame()
        
        execution_time = time.time() - start_time
        
        return FeaturizeResult(
            features_df=features_df,
            meta_df=pd.DataFrame(meta_rows),
            run_meta=self.get_run_metadata(params),
            execution_time_seconds=execution_time,
            success_count=success_count,
            error_count=error_count
        )


class ChemBERTaZincProvider(TransformerEmbedProvider):
    """ChemBERTa-zinc embedding provider"""
    
    MODEL_ID = "seyonec/ChemBERTa-zinc-base-v1"
    
    @property
    def name(self) -> str:
        return "chemberta_zinc"
    
    @property
    def display_name(self) -> str:
        return "ChemBERTa-zinc"
    
    def params_schema(self) -> list[ParamSpec]:
        # Remove model_name from params since it's fixed
        return [p for p in super().params_schema() if p.name != "model_name"]
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        params = dict(params)
        params["model_name"] = self.MODEL_ID
        return super().featurize(records, params)


class ChemBERTaPubchemProvider(TransformerEmbedProvider):
    """ChemBERTa-pubchem embedding provider"""
    
    MODEL_ID = "seyonec/PubChem10M_SMILES_BPE_450k"
    
    @property
    def name(self) -> str:
        return "chemberta_pubchem"
    
    @property
    def display_name(self) -> str:
        return "ChemBERTa-pubchem"
    
    def params_schema(self) -> list[ParamSpec]:
        return [p for p in super().params_schema() if p.name != "model_name"]
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        params = dict(params)
        params["model_name"] = self.MODEL_ID
        return super().featurize(records, params)


class MoLFormerProvider(TransformerEmbedProvider):
    """MoLFormer embedding provider"""
    
    MODEL_ID = "ibm/MoLFormer-XL-both-10pct"
    
    @property
    def name(self) -> str:
        return "molformer"
    
    @property
    def display_name(self) -> str:
        return "MoLFormer"
    
    def params_schema(self) -> list[ParamSpec]:
        return [p for p in super().params_schema() if p.name != "model_name"]
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        params = dict(params)
        params["model_name"] = self.MODEL_ID
        return super().featurize(records, params)


class PolyNCProvider(TransformerEmbedProvider):
    """PolyNC embedding provider (polymer-specific)"""
    
    MODEL_ID = "hkqiu/PolyNC"
    
    @property
    def name(self) -> str:
        return "polync"
    
    @property
    def display_name(self) -> str:
        return "PolyNC"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return True  # PolyNC is specifically designed for polymers
    
    def params_schema(self) -> list[ParamSpec]:
        return [p for p in super().params_schema() if p.name != "model_name"]
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        params = dict(params)
        params["model_name"] = self.MODEL_ID
        return super().featurize(records, params)
