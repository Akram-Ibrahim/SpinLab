"""
Flexible Hamiltonian class for defining spin interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from abc import ABC, abstractmethod

from .fast_ops import (
    exchange_energy, single_ion_anisotropy_energy, 
    magnetic_field_energy, dmi_energy, exchange_effective_field, 
    local_site_energy, local_energy_change,
    metropolis_step, monte_carlo_sweep_full,
    HAS_NUMBA
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
            return exchange_energy(spins, neighbor_array, self.J)
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
            all_fields = exchange_effective_field(spins, neighbor_array, self.J)
            return all_fields[site_idx]
        else:
            # Fallback implementation
            neighbor_indices = neighbor_array[site_idx]
            # Filter out invalid neighbors (marked with -1)
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            neighbor_spins = spins[valid_neighbors]
            
            field = self.J * np.sum(neighbor_spins, axis=0)
            
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
            return single_ion_anisotropy_energy(spins, self.K, self.axis)
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


class MagneticFieldTerm(HamiltonianTerm):
    """Magnetic field interaction with external field."""
    
    def __init__(
        self, 
        B_field: np.ndarray, 
        g_factor: float = 2.0, 
        mu_B: float = 5.78838e-5,
        use_fast: bool = True
    ):
        """
        Initialize magnetic field term.
        
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
        """Calculate magnetic field energy: -μ·B = -g*μB*Si·B"""
        if self.use_fast:
            # Use fast Numba implementation
            return magnetic_field_energy(spins, self.B_field, self.g_factor)
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
            return dmi_energy(spins, neighbor_array, self.D_vector)
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


class Hamiltonian:
    """
    Flexible Hamiltonian class that combines multiple interaction terms.
    
    Supports both single-lattice and sublattice-resolved parameters.
    Parameters can come from any source: cluster expansion fitting, DFT, experiments, etc.
    """
    
    def __init__(self, sublattices: Optional[Dict[str, List[int]]] = None):
        """
        Initialize Hamiltonian.
        
        Args:
            sublattices: Optional dict mapping sublattice names to site indices
                        e.g., {"A": [0, 2, 4], "B": [1, 3, 5]} for bipartite lattice
        """
        self.terms: List[HamiltonianTerm] = []
        self.term_names: List[str] = []
        self.sublattices = sublattices or {}
        self._validate_sublattices()
    
    def _validate_sublattices(self):
        """Validate sublattice definitions."""
        if not self.sublattices:
            return
        
        # Check for overlapping sites
        all_sites = []
        for sublattice_sites in self.sublattices.values():
            all_sites.extend(sublattice_sites)
        
        if len(all_sites) != len(set(all_sites)):
            raise ValueError("Sublattices contain overlapping sites")
    
    def set_sublattices(self, sublattices: Dict[str, List[int]]):
        """Set or update sublattice definitions."""
        self.sublattices = sublattices
        self._validate_sublattices()
    
    def add_term(self, term: HamiltonianTerm, name: str):
        """Add a Hamiltonian term."""
        self.terms.append(term)
        self.term_names.append(name)
    
    def add_exchange(
        self, 
        J: Union[float, Dict[str, float]], 
        neighbor_shell: str = "shell_1",
        sublattice_pairs: Optional[Dict[tuple, str]] = None,
        name: Optional[str] = None
    ):
        """
        Add isotropic exchange term(s).
        
        Args:
            J: Exchange coupling. Can be:
               - float: Single coupling for all interactions
               - Dict[str, float]: Sublattice-resolved couplings like {"AA": J_AA, "AB": J_AB}
            neighbor_shell: Which neighbor shell to use
            sublattice_pairs: Optional mapping of (sublattice1, sublattice2) -> interaction_type
            name: Optional name override
        """
        if isinstance(J, dict):
            # Sublattice-resolved exchange
            for pair_key, coupling in J.items():
                pair_name = name or f"exchange_{pair_key}_{neighbor_shell}"
                term = ExchangeTerm(coupling, neighbor_shell)
                self.add_term(term, pair_name)
        else:
            # Single exchange coupling
            if name is None:
                name = f"exchange_{neighbor_shell}"
            term = ExchangeTerm(J, neighbor_shell)
            self.add_term(term, name)
    
    
    def add_single_ion_anisotropy(
        self, 
        K: Union[float, Dict[str, float]], 
        axis: Union[np.ndarray, Dict[str, np.ndarray]] = np.array([0, 0, 1]),
        name: Optional[str] = None
    ):
        """
        Add single-ion anisotropy term(s).
        
        Args:
            K: Anisotropy constant. Can be:
               - float: Single value for all sites
               - Dict[str, float]: Sublattice-resolved values like {"A": K_A, "B": K_B}
            axis: Easy axis. Can be:
                  - np.ndarray: Single axis for all sites  
                  - Dict[str, np.ndarray]: Sublattice-resolved axes
            name: Optional name override
        """
        if isinstance(K, dict):
            # Sublattice-resolved anisotropy
            for sublattice, K_val in K.items():
                # Get axis for this sublattice
                if isinstance(axis, dict):
                    sublattice_axis = axis.get(sublattice, np.array([0, 0, 1]))
                else:
                    sublattice_axis = axis
                
                sublattice_name = name or f"single_ion_anisotropy_{sublattice}"
                term = SingleIonAnisotropyTerm(K_val, sublattice_axis)
                self.add_term(term, sublattice_name)
        else:
            # Single anisotropy for all sites
            if name is None:
                name = "single_ion_anisotropy"
            term = SingleIonAnisotropyTerm(K, axis)
            self.add_term(term, name)
    
    def add_magnetic_field(
        self, 
        B_field: np.ndarray, 
        g_factor: float = 2.0,
        name: str = "magnetic_field"
    ):
        """Add magnetic field term."""
        term = MagneticFieldTerm(B_field, g_factor)
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
    
    def add_interactions_from_parameters(
        self,
        parameters: Dict[str, Any],
        neighbor_shells: Optional[List[str]] = None
    ):
        """
        Add multiple interactions from fitted parameters dictionary.
        
        This is the main interface for parameters obtained from cluster expansion
        fitting, DFT calculations, or experimental fits.
        
        Args:
            parameters: Dict containing interaction parameters, e.g.:
                       {
                           "exchange": {"AA": -0.1, "AB": 0.05},  # Sublattice-resolved
                           "single_ion_anisotropy": {"A": 0.02, "B": -0.01},
                           "kitaev": {"x": 0.03, "y": 0.03, "z": 0.04},
                           "magnetic_field": [0, 0, 0.1],
                           "dmi": [0.01, 0, 0]
                       }
            neighbor_shells: List of neighbor shells to consider
        """
        if neighbor_shells is None:
            neighbor_shells = ["shell_1"]
        
        # Add exchange interactions
        if "exchange" in parameters:
            for shell in neighbor_shells:
                self.add_exchange(
                    parameters["exchange"], 
                    neighbor_shell=shell,
                    name=f"exchange_{shell}"
                )
        
        # Add single-ion anisotropy
        if "single_ion_anisotropy" in parameters:
            self.add_single_ion_anisotropy(
                parameters["single_ion_anisotropy"],
                name="single_ion_anisotropy"
            )
        
        # Add Kitaev interactions
        if "kitaev" in parameters:
            for shell in neighbor_shells:
                self.add_kitaev(
                    parameters["kitaev"],
                    neighbor_shell=shell,
                    name=f"kitaev_{shell}"
                )
        
        # Add DMI
        if "dmi" in parameters:
            for shell in neighbor_shells:
                self.add_dmi(
                    parameters["dmi"],
                    neighbor_shell=shell,
                    name=f"dmi_{shell}"
                )
        
        # Add magnetic field
        if "magnetic_field" in parameters:
            self.add_magnetic_field(
                parameters["magnetic_field"],
                name="magnetic_field"
            )
        
        # Add electric field
        if "electric_field" in parameters:
            self.add_electric_field(
                parameters["electric_field"]["field"],
                parameters["electric_field"]["gamma"],
                name="electric_field"
            )
    
    def get_sublattice_info(self) -> Dict[str, Any]:
        """Get information about sublattice setup."""
        return {
            "sublattices": self.sublattices,
            "n_sublattices": len(self.sublattices),
            "sublattice_names": list(self.sublattices.keys()) if self.sublattices else [],
            "has_sublattices": bool(self.sublattices)
        }
    
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