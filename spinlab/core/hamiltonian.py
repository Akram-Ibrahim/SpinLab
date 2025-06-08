"""
Flexible Hamiltonian class for defining spin interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from abc import ABC, abstractmethod

from .fast_ops import (
    fast_exchange_energy, fast_anisotropic_exchange_energy,
    fast_single_ion_anisotropy_energy, fast_zeeman_energy,
    fast_dmi_energy, fast_effective_field, HAS_NUMBA
)


class HamiltonianTerm(ABC):
    """Abstract base class for Hamiltonian terms."""
    
    @abstractmethod
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate energy contribution from this term."""
        pass
    
    @abstractmethod
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field on a specific site."""
        pass


class ExchangeTerm(HamiltonianTerm):
    """Isotropic Heisenberg exchange interaction."""
    
    def __init__(self, J: float, neighbor_shell: str = "shell_1", use_fast: bool = True):
        """
        Initialize exchange term.
        
        Args:
            J: Exchange coupling constant (eV)
            neighbor_shell: Which neighbor shell to use
            use_fast: Whether to use Numba acceleration
        """
        self.J = J
        self.neighbor_shell = neighbor_shell
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate exchange energy: -J * Σ Si · Sj"""
        if self.neighbor_shell not in neighbors:
            return np.zeros(len(spins))
        
        neighbor_array = neighbors[self.neighbor_shell]
        
        if self.use_fast:
            # Use fast Numba implementation
            return fast_exchange_energy(spins, neighbor_array, self.J)
        else:
            # Fallback NumPy implementation
            # Get neighbor spins
            neighbor_spins = spins[neighbor_array]  # Shape: (n_sites, n_neighbors, 3)
            
            # Calculate dot products
            site_spins = spins[:, np.newaxis, :]  # Shape: (n_sites, 1, 3)
            dot_products = np.sum(site_spins * neighbor_spins, axis=2)
            
            # Sum over neighbors and multiply by -J/2 (factor of 2 to avoid double counting)
            energy = -self.J / 2.0 * np.sum(dot_products, axis=1)
            
            return energy
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field: H_i = J * Σ_j Sj"""
        if self.neighbor_shell not in neighbors:
            return np.zeros(3)
        
        neighbor_array = neighbors[self.neighbor_shell]
        
        if self.use_fast:
            # Use fast implementation for all sites, then extract one
            all_fields = fast_effective_field(spins, neighbor_array, self.J)
            return all_fields[site_idx]
        else:
            # Fallback implementation
            neighbor_indices = neighbor_array[site_idx]
            # Filter out invalid neighbors (marked with -1)
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            neighbor_spins = spins[valid_neighbors]
            
            field = self.J * np.sum(neighbor_spins, axis=0)
            
            return field


class AnisotropicExchangeTerm(HamiltonianTerm):
    """Anisotropic exchange interaction with full coupling matrix."""
    
    def __init__(
        self, 
        coupling_matrix: np.ndarray, 
        neighbor_shell: str = "shell_1",
        use_fast: bool = True
    ):
        """
        Initialize anisotropic exchange term.
        
        Args:
            coupling_matrix: 3x3 coupling matrix (eV)
            neighbor_shell: Which neighbor shell to use
            use_fast: Whether to use Numba acceleration
        """
        self.coupling_matrix = np.array(coupling_matrix, dtype=np.float64)
        self.neighbor_shell = neighbor_shell
        self.use_fast = use_fast and HAS_NUMBA
        
        if self.coupling_matrix.shape != (3, 3):
            raise ValueError("Coupling matrix must be 3x3")
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate anisotropic exchange energy."""
        if self.neighbor_shell not in neighbors:
            return np.zeros(len(spins))
        
        neighbor_array = neighbors[self.neighbor_shell]
        
        if self.use_fast:
            # Use fast Numba implementation
            return fast_anisotropic_exchange_energy(spins, neighbor_array, self.coupling_matrix)
        else:
            # Fallback NumPy implementation
            neighbor_spins = spins[neighbor_array]
            
            energies = np.zeros(len(spins))
            
            for i in range(len(spins)):
                site_spin = spins[i]
                valid_neighbors = neighbor_array[i][neighbor_array[i] >= 0]
                for j in valid_neighbors:
                    j_spin = spins[j]
                    # Energy = -Si · J · Sj
                    energy_contrib = -np.dot(site_spin, np.dot(self.coupling_matrix, j_spin))
                    energies[i] += energy_contrib
            
            return energies / 2.0  # Avoid double counting
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field from anisotropic exchange."""
        if self.neighbor_shell not in neighbors:
            return np.zeros(3)
        
        neighbor_indices = neighbors[self.neighbor_shell][site_idx]
        neighbor_spins = spins[neighbor_indices]
        
        field = np.zeros(3)
        for j_spin in neighbor_spins:
            field += np.dot(self.coupling_matrix, j_spin)
        
        return field


class SingleIonAnisotropyTerm(HamiltonianTerm):
    """Single-ion anisotropy term."""
    
    def __init__(self, K: float, axis: np.ndarray = np.array([0, 0, 1]), use_fast: bool = True):
        """
        Initialize single-ion anisotropy.
        
        Args:
            K: Anisotropy constant (eV)
            axis: Easy axis direction (default: z)
            use_fast: Whether to use Numba acceleration
        """
        self.K = K
        self.axis = np.array(axis, dtype=np.float64) / np.linalg.norm(axis)
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate single-ion anisotropy energy: -K * (Si · axis)^2"""
        if self.use_fast:
            # Use fast Numba implementation
            return fast_single_ion_anisotropy_energy(spins, self.K, self.axis)
        else:
            # Fallback NumPy implementation
            dot_products = np.dot(spins, self.axis)
            return -self.K * dot_products**2
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate field from single-ion anisotropy."""
        spin = spins[site_idx]
        dot_product = np.dot(spin, self.axis)
        field = 2 * self.K * dot_product * self.axis
        
        return field


class ZeemanTerm(HamiltonianTerm):
    """Zeeman interaction with external magnetic field."""
    
    def __init__(
        self, 
        B_field: np.ndarray, 
        g_factor: float = 2.0, 
        mu_B: float = 5.78838e-5,
        use_fast: bool = True
    ):
        """
        Initialize Zeeman term.
        
        Args:
            B_field: Magnetic field vector [Bx, By, Bz] (Tesla)
            g_factor: Landé g-factor
            mu_B: Bohr magneton (eV/T)
            use_fast: Whether to use Numba acceleration
        """
        self.B_field = np.array(B_field, dtype=np.float64)
        self.g_factor = g_factor
        self.mu_B = mu_B
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate Zeeman energy: -μ·B = -g*μB*Si·B"""
        if self.use_fast:
            # Use fast Numba implementation
            return fast_zeeman_energy(spins, self.B_field, self.g_factor)
        else:
            # Fallback NumPy implementation
            factor = -self.g_factor * self.mu_B
            return factor * np.dot(spins, self.B_field)
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate field from external magnetic field."""
        return self.g_factor * self.mu_B * self.B_field


class ElectricFieldTerm(HamiltonianTerm):
    """Electric field coupling (for systems with spin-charge coupling)."""
    
    def __init__(self, E_field: np.ndarray, gamma: float):
        """
        Initialize electric field term.
        
        Args:
            E_field: Electric field vector (V/Å)
            gamma: Coupling constant (e·Å)
        """
        self.E_field = np.array(E_field)
        self.gamma = gamma
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate electric field coupling energy."""
        return -self.gamma * np.dot(spins, self.E_field)
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate field from electric field coupling."""
        return self.gamma * self.E_field


class DMITerm(HamiltonianTerm):
    """Dzyaloshinskii-Moriya interaction."""
    
    def __init__(self, D_vector: np.ndarray, neighbor_shell: str = "shell_1", use_fast: bool = True):
        """
        Initialize DMI term.
        
        Args:
            D_vector: DM vector (eV)
            neighbor_shell: Which neighbor shell to use
            use_fast: Whether to use Numba acceleration
        """
        self.D_vector = np.array(D_vector, dtype=np.float64)
        self.neighbor_shell = neighbor_shell
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate DMI energy: D · (Si × Sj)"""
        if self.neighbor_shell not in neighbors:
            return np.zeros(len(spins))
        
        neighbor_array = neighbors[self.neighbor_shell]
        
        if self.use_fast:
            # Use fast Numba implementation
            return fast_dmi_energy(spins, neighbor_array, self.D_vector)
        else:
            # Fallback NumPy implementation
            neighbor_spins = spins[neighbor_array]
            
            energies = np.zeros(len(spins))
            
            for i in range(len(spins)):
                site_spin = spins[i]
                valid_neighbors = neighbor_array[i][neighbor_array[i] >= 0]
                for j in valid_neighbors:
                    j_spin = spins[j]
                    cross_product = np.cross(site_spin, j_spin)
                    energy_contrib = np.dot(self.D_vector, cross_product)
                    energies[i] += energy_contrib
            
            return energies / 2.0  # Avoid double counting
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field from DMI."""
        if self.neighbor_shell not in neighbors:
            return np.zeros(3)
        
        neighbor_indices = neighbors[self.neighbor_shell][site_idx]
        neighbor_spins = spins[neighbor_indices]
        
        field = np.zeros(3)
        for j_spin in neighbor_spins:
            # H_i = D × Sj (contribution to site i from neighbor j)
            field += np.cross(self.D_vector, j_spin)
        
        return field


class KitaevTerm(HamiltonianTerm):
    """Kitaev interaction with bond-directional coupling."""
    
    def __init__(
        self, 
        K_couplings: Dict[str, float], 
        neighbor_shell: str = "shell_1",
        bond_directions: Optional[Dict[tuple, str]] = None,
        use_fast: bool = True
    ):
        """
        Initialize Kitaev term.
        
        Args:
            K_couplings: Dict of Kitaev couplings {"x": Kx, "y": Ky, "z": Kz} (eV)
            neighbor_shell: Which neighbor shell to use
            bond_directions: Dict mapping (i,j) -> "x"/"y"/"z" for bond directions
            use_fast: Whether to use Numba acceleration
        """
        self.K_couplings = K_couplings
        self.neighbor_shell = neighbor_shell
        self.bond_directions = bond_directions or {}
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate Kitaev energy: Σ_γ K_γ Σ_{⟨ij⟩_γ} s_i^γ s_j^γ"""
        if self.neighbor_shell not in neighbors:
            return np.zeros(len(spins))
        
        neighbor_array = neighbors[self.neighbor_shell]
        energies = np.zeros(len(spins))
        
        for i in range(len(spins)):
            neighbor_indices = neighbor_array[i]
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                # Determine bond direction
                bond_key = (min(i, j), max(i, j))
                bond_dir = self.bond_directions.get(bond_key, "z")  # Default to z
                
                if bond_dir in self.K_couplings:
                    K_gamma = self.K_couplings[bond_dir]
                    gamma_idx = {"x": 0, "y": 1, "z": 2}[bond_dir]
                    
                    # K_γ s_i^γ s_j^γ
                    energy_contrib = K_gamma * spins[i, gamma_idx] * spins[j, gamma_idx]
                    energies[i] += energy_contrib
        
        return energies / 2.0  # Avoid double counting
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field from Kitaev interactions."""
        if self.neighbor_shell not in neighbors:
            return np.zeros(3)
        
        neighbor_indices = neighbors[self.neighbor_shell][site_idx]
        valid_neighbors = neighbor_indices[neighbor_indices >= 0]
        field = np.zeros(3)
        
        for j in valid_neighbors:
            bond_key = (min(site_idx, j), max(site_idx, j))
            bond_dir = self.bond_directions.get(bond_key, "z")
            
            if bond_dir in self.K_couplings:
                K_gamma = self.K_couplings[bond_dir]
                gamma_idx = {"x": 0, "y": 1, "z": 2}[bond_dir]
                
                # H_i^γ = K_γ s_j^γ (field component only in γ direction)
                field[gamma_idx] += K_gamma * spins[j, gamma_idx]
        
        return field


class ClusterExpansionTerm(HamiltonianTerm):
    """
    Comprehensive cluster expansion Hamiltonian with sublattice resolution.
    
    Implements:
        H = Σ_{k,L,M} J_k^{LM} Σ_{⟨ij⟩_k^{LM}} s_i · s_j                    (isotropic exchange)
            + Σ_{k,L,M,γ} K_{k,γ}^{LM} Σ_{⟨ij⟩_k^{LM,γ}} s_i^γ s_j^γ      (Kitaev interactions)
            + D_z Σ_{⟨ij⟩_1} ẑ · (s_i × s_j)                               (DMI)
            + Σ_L A^L Σ_{i∈L} (s_i^z)^2                                     (single-ion per sublattice)
    
    Where L,M label sublattices and k labels neighbor shells.
    """
    
    def __init__(
        self,
        shell_list: List[int],
        sublattice_indices: Optional[np.ndarray] = None,
        J_params: Optional[Dict[str, float]] = None,
        K_params: Optional[Dict[str, float]] = None,
        A_params: Optional[Dict[str, float]] = None,
        D_z: Optional[float] = None,
        bond_directions: Optional[Dict[tuple, str]] = None,
        use_fast: bool = True
    ):
        """
        Initialize cluster expansion term.
        
        Args:
            shell_list: List of neighbor shells to include [1, 2, 3, ...]
            sublattice_indices: Array [N] assigning each site to sublattice (0, 1, 2, ...)
            J_params: Isotropic exchange parameters {"J1_AA": value, "J1_AB": value, ...}
            K_params: Kitaev parameters {"K1_x_AA": value, "K1_y_AB": value, ...}
            A_params: Single-ion anisotropy {"A_A": value, "A_B": value, ...}
            D_z: DMI parameter (scalar)
            bond_directions: Dict mapping (i,j) -> "x"/"y"/"z" for Kitaev bonds
            use_fast: Whether to use Numba acceleration
        """
        self.shell_list = shell_list
        self.sublattice_indices = sublattice_indices
        self.J_params = J_params or {}
        self.K_params = K_params or {}
        self.A_params = A_params or {}
        self.D_z = D_z or 0.0
        self.bond_directions = bond_directions or {}
        self.use_fast = use_fast and HAS_NUMBA
        
        # Determine number of sublattices
        if sublattice_indices is not None:
            self.n_sublattices = int(np.max(sublattice_indices)) + 1
            self.sublattice_names = [chr(ord('A') + i) for i in range(self.n_sublattices)]
        else:
            self.n_sublattices = 1
            self.sublattice_names = ['A']
    
    def _get_sublattice_pair_name(self, sub_i: int, sub_j: int) -> str:
        """Get sublattice pair name (e.g., 'AA', 'AB')."""
        name_i = self.sublattice_names[sub_i]
        name_j = self.sublattice_names[sub_j]
        return f"{name_i}{name_j}"
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Calculate cluster expansion energy."""
        N = len(spins)
        energies = np.zeros(N)
        
        # Isotropic exchange contributions
        energies += self._calculate_exchange_energy(spins, neighbors)
        
        # Kitaev contributions
        energies += self._calculate_kitaev_energy(spins, neighbors)
        
        # Single-ion anisotropy contributions
        energies += self._calculate_single_ion_energy(spins)
        
        # DMI contributions
        energies += self._calculate_dmi_energy(spins, neighbors)
        
        return energies
    
    def _calculate_exchange_energy(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate isotropic exchange energy with sublattice resolution."""
        N = len(spins)
        energies = np.zeros(N)
        
        for shell_k in self.shell_list:
            shell_name = f"shell_{shell_k}"
            if shell_name not in neighbors:
                continue
            
            neighbor_array = neighbors[shell_name]
            
            for i in range(N):
                sub_i = self.sublattice_indices[i] if self.sublattice_indices is not None else 0
                neighbor_indices = neighbor_array[i]
                valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                
                for j in valid_neighbors:
                    sub_j = self.sublattice_indices[j] if self.sublattice_indices is not None else 0
                    
                    # Get parameter name
                    pair_name = self._get_sublattice_pair_name(sub_i, sub_j)
                    param_name = f"J{shell_k}_{pair_name}"
                    
                    if param_name in self.J_params:
                        J_k = self.J_params[param_name]
                        dot_product = np.dot(spins[i], spins[j])
                        energies[i] += J_k * dot_product
        
        return energies / 2.0  # Avoid double counting
    
    def _calculate_kitaev_energy(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate Kitaev energy with sublattice resolution."""
        N = len(spins)
        energies = np.zeros(N)
        
        for shell_k in self.shell_list:
            shell_name = f"shell_{shell_k}"
            if shell_name not in neighbors:
                continue
            
            neighbor_array = neighbors[shell_name]
            
            for i in range(N):
                sub_i = self.sublattice_indices[i] if self.sublattice_indices is not None else 0
                neighbor_indices = neighbor_array[i]
                valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                
                for j in valid_neighbors:
                    sub_j = self.sublattice_indices[j] if self.sublattice_indices is not None else 0
                    
                    # Determine bond direction
                    bond_key = (min(i, j), max(i, j))
                    bond_dir = self.bond_directions.get(bond_key, "z")
                    
                    # Get parameter name
                    pair_name = self._get_sublattice_pair_name(sub_i, sub_j)
                    param_name = f"K{shell_k}_{bond_dir}_{pair_name}"
                    
                    if param_name in self.K_params:
                        K_gamma = self.K_params[param_name]
                        gamma_idx = {"x": 0, "y": 1, "z": 2}[bond_dir]
                        
                        # K_γ s_i^γ s_j^γ
                        energy_contrib = K_gamma * spins[i, gamma_idx] * spins[j, gamma_idx]
                        energies[i] += energy_contrib
        
        return energies / 2.0  # Avoid double counting
    
    def _calculate_single_ion_energy(self, spins: np.ndarray) -> np.ndarray:
        """Calculate single-ion anisotropy energy with sublattice resolution."""
        N = len(spins)
        energies = np.zeros(N)
        
        for i in range(N):
            sub_i = self.sublattice_indices[i] if self.sublattice_indices is not None else 0
            sublattice_name = self.sublattice_names[sub_i]
            param_name = f"A_{sublattice_name}"
            
            if param_name in self.A_params:
                A_L = self.A_params[param_name]
                # A^L (s_i^z)^2
                energies[i] = A_L * spins[i, 2]**2
        
        return energies
    
    def _calculate_dmi_energy(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate DMI energy (typically no sublattice resolution)."""
        N = len(spins)
        energies = np.zeros(N)
        
        if abs(self.D_z) < 1e-12:  # No DMI
            return energies
        
        shell_1_name = "shell_1"
        if shell_1_name not in neighbors:
            return energies
        
        neighbor_array = neighbors[shell_1_name]
        
        for i in range(N):
            neighbor_indices = neighbor_array[i]
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                # D_z ẑ · (s_i × s_j)
                cross_product = np.cross(spins[i], spins[j])
                z_component = cross_product[2]
                energies[i] += self.D_z * z_component
        
        return energies / 2.0  # Avoid double counting
    
    def calculate_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Calculate effective field at a specific site."""
        field = np.zeros(3)
        
        # Exchange field contributions
        field += self._calculate_exchange_field(spins, neighbors, site_idx)
        
        # Kitaev field contributions
        field += self._calculate_kitaev_field(spins, neighbors, site_idx)
        
        # Single-ion field contributions
        field += self._calculate_single_ion_field(spins, site_idx)
        
        # DMI field contributions
        field += self._calculate_dmi_field(spins, neighbors, site_idx)
        
        return field
    
    def _calculate_exchange_field(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray], site_idx: int) -> np.ndarray:
        """Calculate exchange field at site_idx."""
        field = np.zeros(3)
        sub_i = self.sublattice_indices[site_idx] if self.sublattice_indices is not None else 0
        
        for shell_k in self.shell_list:
            shell_name = f"shell_{shell_k}"
            if shell_name not in neighbors:
                continue
            
            neighbor_indices = neighbors[shell_name][site_idx]
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                sub_j = self.sublattice_indices[j] if self.sublattice_indices is not None else 0
                pair_name = self._get_sublattice_pair_name(sub_i, sub_j)
                param_name = f"J{shell_k}_{pair_name}"
                
                if param_name in self.J_params:
                    J_k = self.J_params[param_name]
                    field += J_k * spins[j]
        
        return field
    
    def _calculate_kitaev_field(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray], site_idx: int) -> np.ndarray:
        """Calculate Kitaev field at site_idx."""
        field = np.zeros(3)
        sub_i = self.sublattice_indices[site_idx] if self.sublattice_indices is not None else 0
        
        for shell_k in self.shell_list:
            shell_name = f"shell_{shell_k}"
            if shell_name not in neighbors:
                continue
            
            neighbor_indices = neighbors[shell_name][site_idx]
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                sub_j = self.sublattice_indices[j] if self.sublattice_indices is not None else 0
                
                # Determine bond direction
                bond_key = (min(site_idx, j), max(site_idx, j))
                bond_dir = self.bond_directions.get(bond_key, "z")
                
                pair_name = self._get_sublattice_pair_name(sub_i, sub_j)
                param_name = f"K{shell_k}_{bond_dir}_{pair_name}"
                
                if param_name in self.K_params:
                    K_gamma = self.K_params[param_name]
                    gamma_idx = {"x": 0, "y": 1, "z": 2}[bond_dir]
                    
                    # H_i^γ = K_γ s_j^γ (field component only in γ direction)
                    field[gamma_idx] += K_gamma * spins[j, gamma_idx]
        
        return field
    
    def _calculate_single_ion_field(self, spins: np.ndarray, site_idx: int) -> np.ndarray:
        """Calculate single-ion field at site_idx."""
        field = np.zeros(3)
        sub_i = self.sublattice_indices[site_idx] if self.sublattice_indices is not None else 0
        sublattice_name = self.sublattice_names[sub_i]
        param_name = f"A_{sublattice_name}"
        
        if param_name in self.A_params:
            A_L = self.A_params[param_name]
            # H_i^z = 2 A^L s_i^z
            field[2] = 2 * A_L * spins[site_idx, 2]
        
        return field
    
    def _calculate_dmi_field(self, spins: np.ndarray, neighbors: Dict[str, np.ndarray], site_idx: int) -> np.ndarray:
        """Calculate DMI field at site_idx."""
        field = np.zeros(3)
        
        if abs(self.D_z) < 1e-12:  # No DMI
            return field
        
        shell_1_name = "shell_1"
        if shell_1_name not in neighbors:
            return field
        
        neighbor_indices = neighbors[shell_1_name][site_idx]
        valid_neighbors = neighbor_indices[neighbor_indices >= 0]
        z_hat = np.array([0, 0, 1])
        
        for j in valid_neighbors:
            # H_i = D_z (ẑ × s_j)
            cross_product = np.cross(z_hat, spins[j])
            field += self.D_z * cross_product
        
        return field


class Hamiltonian:
    """
    Flexible Hamiltonian class that combines multiple interaction terms.
    """
    
    def __init__(self):
        """Initialize empty Hamiltonian."""
        self.terms: List[HamiltonianTerm] = []
        self.term_names: List[str] = []
    
    def add_term(self, term: HamiltonianTerm, name: str):
        """Add a Hamiltonian term."""
        self.terms.append(term)
        self.term_names.append(name)
    
    def add_exchange(
        self, 
        J: float, 
        neighbor_shell: str = "shell_1",
        name: Optional[str] = None
    ):
        """Add isotropic exchange term."""
        if name is None:
            name = f"exchange_{neighbor_shell}"
        
        term = ExchangeTerm(J, neighbor_shell)
        self.add_term(term, name)
    
    def add_anisotropic_exchange(
        self,
        coupling_matrix: np.ndarray,
        neighbor_shell: str = "shell_1", 
        name: Optional[str] = None
    ):
        """Add anisotropic exchange term."""
        if name is None:
            name = f"anisotropic_exchange_{neighbor_shell}"
        
        term = AnisotropicExchangeTerm(coupling_matrix, neighbor_shell)
        self.add_term(term, name)
    
    def add_single_ion_anisotropy(
        self, 
        K: float, 
        axis: np.ndarray = np.array([0, 0, 1]),
        name: str = "single_ion_anisotropy"
    ):
        """Add single-ion anisotropy term."""
        term = SingleIonAnisotropyTerm(K, axis)
        self.add_term(term, name)
    
    def add_zeeman(
        self, 
        B_field: np.ndarray, 
        g_factor: float = 2.0,
        name: str = "zeeman"
    ):
        """Add Zeeman term."""
        term = ZeemanTerm(B_field, g_factor)
        self.add_term(term, name)
    
    def add_electric_field(
        self, 
        E_field: np.ndarray, 
        gamma: float,
        name: str = "electric_field"
    ):
        """Add electric field coupling term."""
        term = ElectricFieldTerm(E_field, gamma)
        self.add_term(term, name)
    
    def add_dmi(
        self, 
        D_vector: np.ndarray, 
        neighbor_shell: str = "shell_1",
        name: Optional[str] = None
    ):
        """Add DMI term."""
        if name is None:
            name = f"dmi_{neighbor_shell}"
        
        term = DMITerm(D_vector, neighbor_shell)
        self.add_term(term, name)
    
    def add_kitaev(
        self,
        K_couplings: Dict[str, float],
        neighbor_shell: str = "shell_1",
        bond_directions: Optional[Dict[tuple, str]] = None,
        name: Optional[str] = None
    ):
        """Add Kitaev interaction term."""
        if name is None:
            name = f"kitaev_{neighbor_shell}"
        
        term = KitaevTerm(K_couplings, neighbor_shell, bond_directions)
        self.add_term(term, name)
    
    def add_cluster_expansion(
        self,
        shell_list: List[int],
        sublattice_indices: Optional[np.ndarray] = None,
        J_params: Optional[Dict[str, float]] = None,
        K_params: Optional[Dict[str, float]] = None,
        A_params: Optional[Dict[str, float]] = None,
        D_z: Optional[float] = None,
        bond_directions: Optional[Dict[tuple, str]] = None,
        name: str = "cluster_expansion"
    ):
        """
        Add comprehensive cluster expansion term.
        
        Args:
            shell_list: List of neighbor shells to include [1, 2, 3, ...]
            sublattice_indices: Array [N] assigning each site to sublattice (0, 1, 2, ...)
            J_params: Isotropic exchange parameters {"J1_AA": value, "J1_AB": value, ...}
            K_params: Kitaev parameters {"K1_x_AA": value, "K1_y_AB": value, ...}
            A_params: Single-ion anisotropy {"A_A": value, "A_B": value, ...}
            D_z: DMI parameter (scalar)
            bond_directions: Dict mapping (i,j) -> "x"/"y"/"z" for Kitaev bonds
            name: Name for this term
        """
        term = ClusterExpansionTerm(
            shell_list=shell_list,
            sublattice_indices=sublattice_indices,
            J_params=J_params,
            K_params=K_params,
            A_params=A_params,
            D_z=D_z,
            bond_directions=bond_directions
        )
        self.add_term(term, name)
    
    def calculate_energy(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray
    ) -> float:
        """Calculate total energy of the system."""
        total_energy = 0.0
        
        for term in self.terms:
            site_energies = term.calculate_energy(spins, neighbors, positions)
            total_energy += np.sum(site_energies)
        
        return total_energy
    
    def calculate_site_energies(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray
    ) -> np.ndarray:
        """Calculate energy for each site."""
        n_sites = len(spins)
        site_energies = np.zeros(n_sites)
        
        for term in self.terms:
            term_energies = term.calculate_energy(spins, neighbors, positions)
            site_energies += term_energies
        
        return site_energies
    
    def calculate_effective_field(
        self, 
        spins: np.ndarray, 
        neighbors: Dict[str, np.ndarray],
        positions: np.ndarray,
        site_idx: int
    ) -> np.ndarray:
        """Calculate effective field at a specific site."""
        total_field = np.zeros(3)
        
        for term in self.terms:
            field_contrib = term.calculate_field(
                spins, neighbors, positions, site_idx
            )
            total_field += field_contrib
        
        return total_field
    
    def get_term(self, name: str) -> Optional[HamiltonianTerm]:
        """Get a specific Hamiltonian term by name."""
        try:
            idx = self.term_names.index(name)
            return self.terms[idx]
        except ValueError:
            return None
    
    def remove_term(self, name: str):
        """Remove a Hamiltonian term."""
        try:
            idx = self.term_names.index(name)
            del self.terms[idx]
            del self.term_names[idx]
        except ValueError:
            raise ValueError(f"Term '{name}' not found")
    
    def __repr__(self) -> str:
        return f"Hamiltonian(terms={self.term_names})"