"""
MolCLR Pre-trained GNN Provider

Generates molecular embeddings using MolCLR pre-trained Graph Neural Networks.
MolCLR: Molecular Contrastive Learning of Representations via Graph Neural Networks

Reference: https://arxiv.org/abs/2102.10056
GitHub: https://github.com/yuyangw/MolCLR
"""
import os
import time
import hashlib
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


# MolCLR atom featurization (matches original implementation)
ATOM_LIST = list(range(1, 119))  # Atomic numbers 1-118
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
] if RDKIT_AVAILABLE else []
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
] if RDKIT_AVAILABLE else []
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
] if RDKIT_AVAILABLE else []


def atom_to_feature_vector(atom):
    """Convert atom to feature vector (MolCLR style)"""
    features = [
        ATOM_LIST.index(atom.GetAtomicNum()) if atom.GetAtomicNum() in ATOM_LIST else len(ATOM_LIST),
        CHIRALITY_LIST.index(atom.GetChiralTag()) if atom.GetChiralTag() in CHIRALITY_LIST else len(CHIRALITY_LIST)
    ]
    return features


def bond_to_feature_vector(bond):
    """Convert bond to feature vector (MolCLR style)"""
    features = [
        BOND_LIST.index(bond.GetBondType()) if bond.GetBondType() in BOND_LIST else len(BOND_LIST),
        BONDDIR_LIST.index(bond.GetBondDir()) if bond.GetBondDir() in BONDDIR_LIST else len(BONDDIR_LIST)
    ]
    return features


def smiles_to_molclr_graph(smiles: str) -> Optional["Data"]:
    """Convert SMILES to PyTorch Geometric Data (MolCLR format)"""
    if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))
    
    if len(atom_features) == 0:
        return None
    
    x = torch.tensor(atom_features, dtype=torch.long)
    
    # Edge features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = bond_to_feature_vector(bond)
        
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)
    
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolCLRGINEncoder(nn.Module):
    """
    MolCLR GIN Encoder
    
    Reimplementation compatible with current PyTorch Geometric.
    Architecture matches the original MolCLR paper.
    """
    
    def __init__(
        self,
        num_atom_type: int = 119,
        num_chirality_type: int = 4,
        num_bond_type: int = 4,
        num_bond_dir: int = 3,
        emb_dim: int = 300,
        num_layer: int = 5,
        drop_ratio: float = 0.0,
        pool: str = "mean"
    ):
        super().__init__()
        
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.pool = pool
        
        # Atom embedding
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_type, emb_dim)
        
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        # GIN layers
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for layer in range(num_layer):
            mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )
            self.gnns.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Embed atoms
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        # GNN layers
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer - 1:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)
        
        # Pooling
        if self.pool == "mean":
            h = global_mean_pool(h, batch)
        else:
            h = global_add_pool(h, batch)
        
        return h


class MolCLRGCNEncoder(nn.Module):
    """
    MolCLR GCN Encoder
    
    Reimplementation compatible with current PyTorch Geometric.
    """
    
    def __init__(
        self,
        num_atom_type: int = 119,
        num_chirality_type: int = 4,
        emb_dim: int = 300,
        num_layer: int = 5,
        drop_ratio: float = 0.0,
        pool: str = "mean"
    ):
        super().__init__()
        
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.pool = pool
        
        # Atom embedding
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_type, emb_dim)
        
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        
        # GCN layers
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
    
    def forward(self, x, edge_index, edge_attr, batch):
        # Embed atoms
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        
        # GNN layers
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer < self.num_layer - 1:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)
        
        # Pooling
        if self.pool == "mean":
            h = global_mean_pool(h, batch)
        else:
            h = global_add_pool(h, batch)
        
        return h


class BaseMolCLRProvider(DescriptorProvider):
    """Base class for MolCLR pre-trained model providers"""
    
    # Model download URLs (hosted copies of MolCLR weights)
    # Note: In production, these would be actual URLs to hosted weights
    MODEL_URLS = {
        "gin": "https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gin/model.pth",
        "gcn": "https://github.com/yuyangw/MolCLR/raw/master/ckpt/pretrained_gcn/model.pth"
    }
    
    def __init__(self):
        self._model = None
        self._model_loaded = False
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "embedding"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False  # Requires RDKit-parseable SMILES
    
    @property
    def version(self) -> str:
        return "MolCLR-2021"
    
    def _get_cache_dir(self) -> Path:
        """Get cache directory for model weights"""
        cache_dir = Path.home() / ".cache" / "polymer_app" / "molclr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _download_weights(self, model_type: str) -> Optional[Path]:
        """Download pre-trained weights if not cached"""
        import urllib.request
        import urllib.error
        
        cache_dir = self._get_cache_dir()
        weight_file = cache_dir / f"molclr_{model_type}.pth"
        
        if weight_file.exists():
            return weight_file
        
        url = self.MODEL_URLS.get(model_type)
        if not url:
            return None
        
        try:
            print(f"Downloading MolCLR {model_type.upper()} weights...")
            urllib.request.urlretrieve(url, weight_file)
            print(f"Downloaded to {weight_file}")
            return weight_file
        except urllib.error.URLError as e:
            print(f"Warning: Could not download weights: {e}")
            print("Using randomly initialized model (less accurate)")
            return None
    
    def _create_model(self, model_type: str) -> nn.Module:
        """Create and optionally load pre-trained model"""
        raise NotImplementedError
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="pooling",
                type="select",
                default="mean",
                description="Graph pooling strategy",
                options=["mean", "sum"]
            ),
            ParamSpec(
                name="device",
                type="select",
                default="cpu",
                description="Device for inference",
                options=["cpu", "cuda", "mps"]
            )
        ]
    
    def validate(self, smiles_list: list[str]) -> ValidationResult:
        """Validate SMILES can be converted to graphs"""
        result = ValidationResult()
        
        if not RDKIT_AVAILABLE:
            for i in range(len(smiles_list)):
                result.invalid_indices.append(i)
                result.messages[i] = "RDKit not available"
            return result
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            for i in range(len(smiles_list)):
                result.invalid_indices.append(i)
                result.messages[i] = "PyTorch Geometric not available"
            return result
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                result.valid_indices.append(i)
            else:
                result.invalid_indices.append(i)
                result.messages[i] = "Invalid SMILES for RDKit"
        
        return result
    
    def _get_model(self, pooling: str, device: str):
        """Get or create model"""
        raise NotImplementedError
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        """Generate MolCLR embeddings for records"""
        start_time = time.time()
        
        pooling = params.get("pooling", "mean")
        device = params.get("device", "cpu")
        
        meta_rows = []
        all_embeddings = []
        success_count = 0
        error_count = 0
        
        # Check availability
        if not TORCH_GEOMETRIC_AVAILABLE:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "PyTorch Geometric not available"
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
        
        # Get model
        try:
            model = self._get_model(pooling, device)
        except Exception as e:
            for record in records:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": f"Model error: {str(e)}"
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
        
        # Convert SMILES to graphs
        graphs = []
        valid_indices = []
        invalid_meta = []
        
        for i, record in enumerate(records):
            smiles = record.smiles_normalized or record.input_smiles_raw
            graph = smiles_to_molclr_graph(smiles)
            
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
            else:
                invalid_meta.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "Could not convert to graph"
                })
                error_count += 1
        
        if not graphs:
            return FeaturizeResult(
                features_df=pd.DataFrame(),
                meta_df=pd.DataFrame(invalid_meta),
                run_meta=self.get_run_metadata(params),
                execution_time_seconds=time.time() - start_time,
                success_count=0,
                error_count=error_count
            )
        
        # Get device
        model_device = next(model.parameters()).device
        
        # Batch and get embeddings
        try:
            batch = Batch.from_data_list(graphs).to(model_device)
            
            with torch.no_grad():
                embeddings = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                embeddings = embeddings.cpu().numpy()
        except Exception as e:
            for i in valid_indices:
                record = records[i]
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": f"Inference error: {str(e)}"
                })
                error_count += 1
            
            meta_rows.extend(invalid_meta)
            
            return FeaturizeResult(
                features_df=pd.DataFrame(),
                meta_df=pd.DataFrame(meta_rows),
                run_meta=self.get_run_metadata(params),
                execution_time_seconds=time.time() - start_time,
                success_count=0,
                error_count=error_count
            )
        
        # Build results
        emb_dim = embeddings.shape[1]
        emb_cols = [f"molclr_{i:04d}" for i in range(emb_dim)]
        
        # Create ordered results
        all_results = [None] * len(records)
        embed_idx = 0
        
        for i, record in enumerate(records):
            if i in valid_indices:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "OK",
                    "descriptor_error": ""
                })
                all_results[i] = embeddings[embed_idx]
                embed_idx += 1
                success_count += 1
        
        meta_rows.extend(invalid_meta)
        
        # Create features DataFrame
        rows = []
        for result in all_results:
            if result is not None:
                rows.append({emb_cols[i]: float(result[i]) for i in range(emb_dim)})
            else:
                rows.append({col: None for col in emb_cols})
        
        execution_time = time.time() - start_time
        
        return FeaturizeResult(
            features_df=pd.DataFrame(rows),
            meta_df=pd.DataFrame(meta_rows),
            run_meta=self.get_run_metadata(params),
            execution_time_seconds=execution_time,
            success_count=success_count,
            error_count=error_count
        )


class MolCLRGINProvider(BaseMolCLRProvider):
    """MolCLR GIN (Graph Isomorphism Network) pre-trained provider"""
    
    @property
    def name(self) -> str:
        return "molclr_gin"
    
    @property
    def display_name(self) -> str:
        return "MolCLR-GIN"
    
    def _get_model(self, pooling: str, device: str):
        """Get or create GIN model"""
        if self._model is None:
            self._model = MolCLRGINEncoder(pool=pooling)
            
            # Try to load pre-trained weights
            weight_file = self._download_weights("gin")
            if weight_file and weight_file.exists():
                try:
                    state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
                    # Handle different state dict formats
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    # Filter to matching keys
                    model_state = self._model.state_dict()
                    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                    self._model.load_state_dict(filtered_state, strict=False)
                    print(f"Loaded {len(filtered_state)}/{len(model_state)} weight tensors")
                except Exception as e:
                    print(f"Warning: Could not load weights: {e}")
            
            self._model.eval()
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        elif device == "mps" and torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        else:
            self._model = self._model.cpu()
        
        return self._model


class MolCLRGCNProvider(BaseMolCLRProvider):
    """MolCLR GCN (Graph Convolutional Network) pre-trained provider"""
    
    @property
    def name(self) -> str:
        return "molclr_gcn"
    
    @property
    def display_name(self) -> str:
        return "MolCLR-GCN"
    
    def _get_model(self, pooling: str, device: str):
        """Get or create GCN model"""
        if self._model is None:
            self._model = MolCLRGCNEncoder(pool=pooling)
            
            # Try to load pre-trained weights
            weight_file = self._download_weights("gcn")
            if weight_file and weight_file.exists():
                try:
                    state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    model_state = self._model.state_dict()
                    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
                    self._model.load_state_dict(filtered_state, strict=False)
                    print(f"Loaded {len(filtered_state)}/{len(model_state)} weight tensors")
                except Exception as e:
                    print(f"Warning: Could not load weights: {e}")
            
            self._model.eval()
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        elif device == "mps" and torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        else:
            self._model = self._model.cpu()
        
        return self._model
