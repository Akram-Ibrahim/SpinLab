"""
Graph Neural Network Hamiltonian for SpinLab.

This module demonstrates how to integrate PyTorch-based GNN models
as Hamiltonian terms in SpinLab simulations.
"""

import numpy as np
from typing import Optional, Dict, Any, Union
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .hamiltonian import HamiltonianTerm


class GNNHamiltonianTerm(HamiltonianTerm):
    """
    Graph Neural Network based Hamiltonian term.
    
    This allows using machine-learned interactions in SpinLab simulations.
    """
    
    def __init__(
        self,
        model: Optional['torch.nn.Module'] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_fast: bool = True
    ):
        """
        Initialize GNN Hamiltonian term.
        
        Args:
            model: Pre-loaded PyTorch model
            model_path: Path to saved model file
            device: Device for computation ("cpu", "cuda", "auto")
            use_fast: Whether to use optimized operations
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        super().__init__(use_fast=use_fast)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            raise ValueError("Either model or model_path must be provided")
        
        self.model.eval()
        
        print(f"🧠 GNN Hamiltonian loaded on {self.device}")
    
    def calculate_energy(
        self,
        spins: np.ndarray,
        positions: np.ndarray,
        neighbor_array: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """
        Calculate energy using GNN model.
        
        Args:
            spins: (n_spins, 3) spin configuration
            positions: (n_spins, 3) atomic positions  
            neighbor_array: (n_spins, max_neighbors) neighbor indices
            
        Returns:
            Total energy from GNN
        """
        with torch.no_grad():
            # Convert to PyTorch tensors
            spin_tensor = torch.from_numpy(spins).float().to(self.device)
            pos_tensor = torch.from_numpy(positions).float().to(self.device)
            
            # Prepare graph data
            graph_data = self._prepare_graph_data(
                spin_tensor, pos_tensor, neighbor_array
            )
            
            # Forward pass through GNN
            energy = self.model(graph_data)
            
            return energy.cpu().item()
    
    def calculate_forces(
        self,
        spins: np.ndarray,
        positions: np.ndarray,
        neighbor_array: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate forces (negative gradients) using GNN.
        
        Args:
            spins: (n_spins, 3) spin configuration
            positions: (n_spins, 3) atomic positions
            neighbor_array: (n_spins, max_neighbors) neighbor indices
            
        Returns:
            (n_spins, 3) force vectors
        """
        # Enable gradients for force calculation
        spin_tensor = torch.from_numpy(spins).float().to(self.device)
        pos_tensor = torch.from_numpy(positions).float().to(self.device)
        
        spin_tensor.requires_grad_(True)
        
        # Prepare graph data
        graph_data = self._prepare_graph_data(
            spin_tensor, pos_tensor, neighbor_array
        )
        
        # Forward pass
        energy = self.model(graph_data)
        
        # Calculate gradients
        energy.backward()
        forces = -spin_tensor.grad.cpu().numpy()
        
        return forces
    
    def _prepare_graph_data(
        self,
        spins: torch.Tensor,
        positions: torch.Tensor,
        neighbor_array: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare graph data for GNN model.
        
        This method should be customized based on your specific GNN architecture.
        """
        # Basic graph data structure
        graph_data = {
            'node_features': torch.cat([spins, positions], dim=1),  # Combine spins and positions
            'edge_index': self._get_edge_index(neighbor_array),
            'batch': torch.zeros(spins.size(0), dtype=torch.long, device=self.device)
        }
        
        return graph_data
    
    def _get_edge_index(self, neighbor_array: Optional[np.ndarray]) -> torch.Tensor:
        """Convert neighbor array to PyTorch Geometric edge_index format."""
        if neighbor_array is None:
            # Create fully connected graph (fallback)
            n_nodes = neighbor_array.shape[0] if neighbor_array is not None else 0
            edge_index = torch.combinations(torch.arange(n_nodes), r=2).T
        else:
            # Convert neighbor array to edge list
            edges = []
            for i in range(neighbor_array.shape[0]):
                for j_idx in range(neighbor_array.shape[1]):
                    j = neighbor_array[i, j_idx]
                    if j >= 0:  # Valid neighbor
                        edges.append([i, j])
            
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).T
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return edge_index


class SimpleSpinGNN(nn.Module):
    """
    Example GNN architecture for spin systems.
    
    This is a demonstration - replace with your actual GNN architecture.
    """
    
    def __init__(
        self,
        node_dim: int = 6,  # 3 for spins + 3 for positions
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1
    ):
        super().__init__()
        
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False
            print("⚠️  PyTorch Geometric not available. Install with: pip install torch-geometric")
            return
        
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.global_pool = global_mean_pool
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the GNN."""
        if not self.has_pyg:
            raise ImportError("PyTorch Geometric required for this model")
        
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        batch = graph_data['batch']
        
        # Encode node features
        x = self.node_encoder(x)
        x = torch.relu(x)
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = torch.relu(x)
        
        # Global pooling to get graph-level representation
        x = self.global_pool(x, batch)
        
        # Predict energy
        energy = self.energy_head(x)
        
        return energy.squeeze()


# Example usage integration with existing Hamiltonian
def create_hybrid_hamiltonian(traditional_terms: Dict, gnn_model_path: str):
    """
    Create a hybrid Hamiltonian combining traditional terms with GNN.
    
    Args:
        traditional_terms: Dictionary of traditional Hamiltonian parameters
        gnn_model_path: Path to trained GNN model
        
    Returns:
        Hamiltonian with both traditional and GNN terms
    """
    from .hamiltonian import Hamiltonian
    
    # Create traditional Hamiltonian
    hamiltonian = Hamiltonian()
    
    # Add traditional terms
    if 'exchange' in traditional_terms:
        hamiltonian.add_exchange(**traditional_terms['exchange'])
    
    if 'anisotropy' in traditional_terms:
        hamiltonian.add_single_ion_anisotropy(**traditional_terms['anisotropy'])
    
    # Add GNN term
    gnn_term = GNNHamiltonianTerm(model_path=gnn_model_path)
    hamiltonian.add_custom_term("gnn_interaction", gnn_term)
    
    return hamiltonian