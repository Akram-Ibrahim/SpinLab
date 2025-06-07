"""
Neighbor finding utilities for spin systems.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from ase import Atoms
from ase.neighborlist import neighbor_list
try:
    from matscipy.neighbours import neighbour_list
    HAS_MATSCIPY = True
except ImportError:
    HAS_MATSCIPY = False


class NeighborFinder:
    """
    Utility class for finding neighbors in crystal structures.
    
    Supports both orthogonal and non-orthogonal unit cells using
    ASE's neighbor_list or matscipy for better handling of 
    non-orthogonal cells.
    """
    
    def __init__(self, structure: Atoms):
        """
        Initialize neighbor finder.
        
        Args:
            structure: ASE Atoms object
        """
        self.structure = structure.copy()
        self.positions = structure.get_positions()
        self.cell = structure.get_cell()
        self.n_atoms = len(structure)
        
        # Check if cell is orthogonal
        self.is_orthogonal = self._check_orthogonal()
    
    def _check_orthogonal(self) -> bool:
        """Check if the unit cell is orthogonal."""
        cell_matrix = self.cell.array
        
        # Check if off-diagonal elements are close to zero
        off_diag = np.abs(cell_matrix - np.diag(np.diag(cell_matrix)))
        return np.all(off_diag < 1e-6)
    
    def find_neighbors(
        self, 
        cutoff: float, 
        max_neighbors: Optional[int] = None,
        exclude_self: bool = True
    ) -> np.ndarray:
        """
        Find neighbors within cutoff distance.
        
        Args:
            cutoff: Cutoff distance for neighbors
            max_neighbors: Maximum number of neighbors per site
            exclude_self: Whether to exclude self-interactions
            
        Returns:
            Array of neighbor indices for each site
        """
        if HAS_MATSCIPY and not self.is_orthogonal:
            return self._find_neighbors_matscipy(cutoff, max_neighbors, exclude_self)
        else:
            return self._find_neighbors_ase(cutoff, max_neighbors, exclude_self)
    
    def _find_neighbors_ase(
        self, 
        cutoff: float, 
        max_neighbors: Optional[int], 
        exclude_self: bool
    ) -> np.ndarray:
        """Find neighbors using ASE neighbor_list."""
        i_indices, j_indices, distances = neighbor_list(
            'ijd', self.structure, cutoff, self_interaction=not exclude_self
        )
        
        return self._process_neighbor_lists(
            i_indices, j_indices, distances, max_neighbors
        )
    
    def _find_neighbors_matscipy(
        self, 
        cutoff: float, 
        max_neighbors: Optional[int], 
        exclude_self: bool
    ) -> np.ndarray:
        """Find neighbors using matscipy for non-orthogonal cells."""
        i_indices, j_indices, distances = neighbour_list(
            'ijd', self.structure, cutoff
        )
        
        if exclude_self:
            # Remove self-interactions
            mask = i_indices != j_indices
            i_indices = i_indices[mask]
            j_indices = j_indices[mask]
            distances = distances[mask]
        
        return self._process_neighbor_lists(
            i_indices, j_indices, distances, max_neighbors
        )
    
    def _process_neighbor_lists(
        self, 
        i_indices: np.ndarray, 
        j_indices: np.ndarray, 
        distances: np.ndarray,
        max_neighbors: Optional[int]
    ) -> np.ndarray:
        """Process raw neighbor lists into structured format."""
        # Group neighbors by site
        neighbors_dict = {}
        for i, j, d in zip(i_indices, j_indices, distances):
            if i not in neighbors_dict:
                neighbors_dict[i] = []
            neighbors_dict[i].append((j, d))
        
        # Sort by distance and limit number of neighbors
        max_nn = 0
        for site in neighbors_dict:
            # Sort by distance
            neighbors_dict[site].sort(key=lambda x: x[1])
            
            # Limit number of neighbors
            if max_neighbors is not None:
                neighbors_dict[site] = neighbors_dict[site][:max_neighbors]
            
            max_nn = max(max_nn, len(neighbors_dict[site]))
        
        # Create structured array with padding
        neighbor_array = np.full((self.n_atoms, max_nn), -1, dtype=int)
        
        for site in range(self.n_atoms):
            if site in neighbors_dict:
                nn_list = [neighbor[0] for neighbor in neighbors_dict[site]]
                neighbor_array[site, :len(nn_list)] = nn_list
        
        return neighbor_array
    
    def find_neighbors_by_shell(
        self, 
        cutoffs: List[float], 
        max_neighbors: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Find neighbors organized by coordination shells.
        
        Args:
            cutoffs: List of cutoff distances for each shell
            max_neighbors: Maximum neighbors per shell (optional)
            
        Returns:
            List of neighbor arrays for each shell
        """
        if max_neighbors is None:
            max_neighbors = [None] * len(cutoffs)
        
        if len(cutoffs) != len(max_neighbors):
            raise ValueError("cutoffs and max_neighbors must have same length")
        
        # Get all neighbors up to maximum cutoff
        max_cutoff = max(cutoffs)
        all_neighbors = self.find_neighbors(max_cutoff, exclude_self=True)
        
        # Get distances for filtering
        i_indices, j_indices, distances = neighbor_list(
            'ijd', self.structure, max_cutoff, self_interaction=False
        )
        
        # Create distance lookup
        distance_dict = {}
        for i, j, d in zip(i_indices, j_indices, distances):
            if i not in distance_dict:
                distance_dict[i] = {}
            distance_dict[i][j] = d
        
        # Organize by shells
        shell_neighbors = []
        
        for shell_idx, (cutoff, max_nn) in enumerate(zip(cutoffs, max_neighbors)):
            shell_array = np.full_like(all_neighbors, -1)
            
            for site in range(self.n_atoms):
                shell_nn = []
                
                # Get neighbors in current shell
                for nn_idx in range(all_neighbors.shape[1]):
                    neighbor = all_neighbors[site, nn_idx]
                    if neighbor == -1:
                        break
                    
                    # Check if neighbor is in current shell
                    dist = distance_dict.get(site, {}).get(neighbor, float('inf'))
                    
                    # Determine shell bounds
                    min_dist = cutoffs[shell_idx - 1] if shell_idx > 0 else 0
                    max_dist = cutoff
                    
                    if min_dist < dist <= max_dist:
                        shell_nn.append((neighbor, dist))
                
                # Sort by distance and limit
                shell_nn.sort(key=lambda x: x[1])
                if max_nn is not None:
                    shell_nn = shell_nn[:max_nn]
                
                # Fill array
                for nn_idx, (neighbor, _) in enumerate(shell_nn):
                    shell_array[site, nn_idx] = neighbor
            
            shell_neighbors.append(shell_array)
        
        return shell_neighbors
    
    def get_coordination_number(self, cutoff: float) -> np.ndarray:
        """Get coordination number for each site."""
        neighbors = self.find_neighbors(cutoff)
        
        coordination = np.zeros(self.n_atoms, dtype=int)
        for site in range(self.n_atoms):
            coordination[site] = np.sum(neighbors[site] != -1)
        
        return coordination
    
    def get_neighbor_distances(
        self, 
        cutoff: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neighbor indices and corresponding distances.
        
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        i_indices, j_indices, distances = neighbor_list(
            'ijd', self.structure, cutoff, self_interaction=False
        )
        
        # Group by site
        neighbors_dict = {}
        distances_dict = {}
        
        for i, j, d in zip(i_indices, j_indices, distances):
            if i not in neighbors_dict:
                neighbors_dict[i] = []
                distances_dict[i] = []
            neighbors_dict[i].append(j)
            distances_dict[i].append(d)
        
        # Create arrays
        max_nn = max(len(neighbors_dict.get(i, [])) for i in range(self.n_atoms))
        
        neighbor_array = np.full((self.n_atoms, max_nn), -1, dtype=int)
        distance_array = np.full((self.n_atoms, max_nn), np.inf)
        
        for site in range(self.n_atoms):
            if site in neighbors_dict:
                nn_list = neighbors_dict[site]
                dist_list = distances_dict[site]
                
                # Sort by distance
                sorted_pairs = sorted(zip(nn_list, dist_list), key=lambda x: x[1])
                
                for idx, (neighbor, dist) in enumerate(sorted_pairs):
                    neighbor_array[site, idx] = neighbor
                    distance_array[site, idx] = dist
        
        return neighbor_array, distance_array
    
    def analyze_structure(self, max_cutoff: float = 6.0) -> dict:
        """
        Analyze the structure and suggest appropriate cutoffs.
        
        Args:
            max_cutoff: Maximum distance to consider
            
        Returns:
            Dictionary with structure analysis
        """
        # Get all distances up to max_cutoff
        i_indices, j_indices, distances = neighbor_list(
            'ijd', self.structure, max_cutoff, self_interaction=False
        )
        
        # Histogram of distances
        hist, bins = np.histogram(distances, bins=100)
        
        # Find peaks (coordination shells)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
        
        shell_distances = bins[peaks]
        
        analysis = {
            'suggested_cutoffs': shell_distances.tolist(),
            'distance_histogram': (hist, bins),
            'coordination_shells': len(shell_distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'is_orthogonal': self.is_orthogonal
        }
        
        return analysis