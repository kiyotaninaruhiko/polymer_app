"""
GNN Embedding Provider

Generates molecular embeddings using Graph Neural Networks (GIN, GCN, etc.).
Requires PyTorch Geometric.
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
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from .base import DescriptorProvider, ValidationResult, FeaturizeResult, ParamSpec


# Atom features for graph construction
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # Atomic number
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'num_hs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'hybridization': ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
    'is_conjugated': [False, True],
}


def one_hot_encoding(value, allowable_set):
    """One-hot encode a value"""
    encoding = [0] * len(allowable_set)
    if value in allowable_set:
        encoding[allowable_set.index(value)] = 1
    return encoding


def get_atom_features(atom) -> list:
    """Get atom features as a list"""
    features = []
    features += one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    features += one_hot_encoding(str(atom.GetChiralTag()), ATOM_FEATURES['chirality'])
    features += one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree'])
    features += one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    features += one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    features += one_hot_encoding(str(atom.GetHybridization()), ATOM_FEATURES['hybridization'])
    features += one_hot_encoding(atom.GetIsAromatic(), ATOM_FEATURES['is_aromatic'])
    features += one_hot_encoding(atom.IsInRing(), ATOM_FEATURES['is_in_ring'])
    return features


def get_bond_features(bond) -> list:
    """Get bond features as a list"""
    features = []
    features += one_hot_encoding(str(bond.GetBondType()), BOND_FEATURES['bond_type'])
    features += one_hot_encoding(str(bond.GetStereo()), BOND_FEATURES['stereo'])
    features += one_hot_encoding(bond.GetIsConjugated(), BOND_FEATURES['is_conjugated'])
    return features


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data object"""
    if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    if len(atom_features) == 0:
        return None
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get edge indices and features
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feat = get_bond_features(bond)
        
        # Add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)
    
    if len(edge_index) == 0:
        # No bonds - single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, len(BOND_FEATURES['bond_type']) + len(BOND_FEATURES['stereo']) + len(BOND_FEATURES['is_conjugated'])), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SimpleGIN(nn.Module):
    """Simple Graph Isomorphism Network for molecular embeddings"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Project to output dimension
        x = self.output_proj(x)
        
        return x


class GNNEmbedProvider(DescriptorProvider):
    """GNN-based molecular embedding provider"""
    
    def __init__(self):
        self._model = None
        self._input_dim = None
    
    @property
    def name(self) -> str:
        return "gnn_embed"
    
    @property
    def display_name(self) -> str:
        return "GNN Embedding (GIN)"
    
    @property
    def kind(self) -> Literal["numeric", "fingerprint", "embedding"]:
        return "embedding"
    
    @property
    def supports_polymer_smiles(self) -> bool:
        return False  # Requires RDKit-parseable SMILES
    
    @property
    def version(self) -> str:
        if TORCH_GEOMETRIC_AVAILABLE:
            import torch_geometric
            return torch_geometric.__version__
        return "not installed"
    
    def params_schema(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="hidden_dim",
                type="select",
                default=256,
                description="Hidden dimension of GNN layers",
                options=[64, 128, 256, 512]
            ),
            ParamSpec(
                name="output_dim",
                type="select",
                default=256,
                description="Output embedding dimension",
                options=[64, 128, 256, 512]
            ),
            ParamSpec(
                name="num_layers",
                type="int",
                default=3,
                description="Number of GNN layers",
                min_value=1,
                max_value=6
            ),
            ParamSpec(
                name="device",
                type="select",
                default="cpu",
                description="Device for inference",
                options=["cpu", "cuda", "mps"]
            )
        ]
    
    def _get_model(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, device: str):
        """Get or create model"""
        if self._model is None or self._input_dim != input_dim:
            self._model = SimpleGIN(input_dim, hidden_dim, output_dim, num_layers)
            self._model.eval()
            self._input_dim = input_dim
            
            # Initialize with random weights (for consistent embeddings, use trained model)
            # In production, you would load pre-trained weights here
        
        # Move to device
        if device == "cuda" and torch.cuda.is_available():
            self._model = self._model.cuda()
        elif device == "mps" and torch.backends.mps.is_available():
            self._model = self._model.to("mps")
        else:
            self._model = self._model.cpu()
        
        return self._model
    
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
    
    def featurize(self, records: list, params: dict) -> FeaturizeResult:
        """Generate GNN embeddings for records"""
        start_time = time.time()
        
        hidden_dim = params.get("hidden_dim", 256)
        output_dim = params.get("output_dim", 256)
        num_layers = params.get("num_layers", 3)
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
                    "descriptor_error": "PyTorch Geometric not available. Install with: pip install torch-geometric"
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
        
        for i, record in enumerate(records):
            smiles = record.smiles_normalized or record.input_smiles_raw
            graph = smiles_to_graph(smiles)
            
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
            else:
                meta_rows.append({
                    "input_id": record.input_id,
                    "input_smiles_raw": record.input_smiles_raw,
                    "smiles_normalized": record.smiles_normalized,
                    "parse_status": record.parse_status.value if hasattr(record.parse_status, 'value') else str(record.parse_status),
                    "descriptor_status": "FAILED",
                    "descriptor_error": "Could not convert to graph"
                })
                all_embeddings.append(None)
                error_count += 1
        
        if not graphs:
            return FeaturizeResult(
                features_df=pd.DataFrame(),
                meta_df=pd.DataFrame(meta_rows),
                run_meta=self.get_run_metadata(params),
                execution_time_seconds=time.time() - start_time,
                success_count=0,
                error_count=error_count
            )
        
        # Get input dimension from first graph
        input_dim = graphs[0].x.shape[1]
        
        # Get model
        try:
            model = self._get_model(input_dim, hidden_dim, output_dim, num_layers, device)
        except Exception as e:
            for i in valid_indices:
                record = records[i]
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
        
        # Batch graphs
        model_device = next(model.parameters()).device
        batch = Batch.from_data_list(graphs).to(model_device)
        
        # Get embeddings
        try:
            with torch.no_grad():
                embeddings = model(batch.x, batch.edge_index, batch.batch)
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
            
            return FeaturizeResult(
                features_df=pd.DataFrame(),
                meta_df=pd.DataFrame(meta_rows),
                run_meta=self.get_run_metadata(params),
                execution_time_seconds=time.time() - start_time,
                success_count=0,
                error_count=error_count
            )
        
        # Build results in correct order
        embed_idx = 0
        final_embeddings = []
        
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
                final_embeddings.append(embeddings[embed_idx])
                embed_idx += 1
                success_count += 1
            else:
                final_embeddings.append(None)
        
        # Create features DataFrame
        emb_cols = [f"gnn_{i:04d}" for i in range(output_dim)]
        rows = []
        for emb in final_embeddings:
            if emb is not None:
                rows.append({emb_cols[i]: float(emb[i]) for i in range(output_dim)})
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
