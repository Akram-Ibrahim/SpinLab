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