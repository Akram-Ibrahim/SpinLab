"""
PairHamiltonian class implementing fitted cluster expansion model.

This class provides an analytic implementation of the cluster expansion
Hamiltonian with fitted parameters, including Kitaev interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.hamiltonian import HamiltonianTerm


class PairHamiltonian:
    """
    Analytic Hamiltonian for cluster expansion with fitted parameters.
    
    Implements:
        H = Σ_{k=1..n} J_k Σ_{⟨ij⟩_k} s_i · s_j                    (isotropic exchange)
            + Σ_{k=1..n} Σ_{γ=x,y,z} K_k^γ Σ_{⟨ij⟩_k^γ} s_i^γ s_j^γ  (Kitaev interactions)
            + D_z Σ_{⟨ij⟩_1} ẑ · (s_i × s_j)                       (first-neighbor DMI)
            + A Σ_i (s_i^z)^2                                       (single-ion anisotropy)
    
    Where ⟨ij⟩_k^γ denotes bonds in shell k with direction γ (for Kitaev terms).
    """
    
    def __init__(
        self,
        parameters: np.ndarray,
        shell_list: List[int],
        pair_tables: Dict[str, np.ndarray],
        include_single_ion: bool = True,
        include_dmi: bool = True,
        include_kitaev: bool = False,
        kitaev_bond_directions: Optional[Dict[str, Dict[int, str]]] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize PairHamiltonian with fitted parameters.
        
        Args:
            parameters: Fitted parameters θ = [J₁, J₂, ..., Jₙ, K₁ˣ, K₁ʸ, K₁ᶻ, ..., A?, D_z?]
            shell_list: List of shell indices [1, 2, 3, ...]
            pair_tables: Dict mapping "shell_k" -> neighbor indices [N, max_neighbors]
            include_single_ion: Whether single-ion anisotropy A is included
            include_dmi: Whether DM interaction D_z is included
            include_kitaev: Whether Kitaev interactions are included
            kitaev_bond_directions: Dict specifying bond directions for Kitaev terms
            feature_names: Names of parameters (for debugging)
        """
        self.shell_list = shell_list
        self.pair_tables = pair_tables
        self.include_single_ion = include_single_ion
        self.include_dmi = include_dmi
        self.include_kitaev = include_kitaev
        self.kitaev_bond_directions = kitaev_bond_directions or {}
        self.feature_names = feature_names
        
        # Parse parameters
        n_shells = len(shell_list)
        n_kitaev = 3 * len(shell_list) if include_kitaev else 0  # 3 components (x,y,z) per shell
        expected_params = n_shells + n_kitaev + int(include_single_ion) + int(include_dmi)
        
        if len(parameters) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(parameters)}")
        
        # Extract J parameters (isotropic exchange)
        self.J_params = parameters[:n_shells]
        
        # Extract Kitaev parameters
        param_idx = n_shells
        if include_kitaev:
            self.K_params = parameters[param_idx:param_idx + n_kitaev].reshape(n_shells, 3)
            param_idx += n_kitaev
        else:
            self.K_params = np.zeros((n_shells, 3))
        
        # Extract A parameter
        if include_single_ion:
            self.A_param = parameters[param_idx]
            param_idx += 1
        else:
            self.A_param = 0.0
        
        # Extract D_z parameter
        if include_dmi:
            self.D_z_param = parameters[param_idx]
        else:
            self.D_z_param = 0.0
        
        # Validate pair tables
        for shell_k in shell_list:
            shell_name = f"shell_{shell_k}"
            if shell_name not in pair_tables:
                raise ValueError(f"Missing neighbor table for {shell_name}")
        
        self._print_summary()
    
    def _print_summary(self):
        """Print summary of initialized Hamiltonian."""
        print(f"🔧 PairHamiltonian initialized:")
        print(f"   Exchange shells: {self.shell_list}")
        print(f"   J parameters: {self.J_params}")
        
        if self.include_kitaev:
            print(f"   Kitaev interactions enabled:")
            for i, shell_k in enumerate(self.shell_list):
                Kx, Ky, Kz = self.K_params[i]
                print(f"     Shell {shell_k}: Kx={Kx:.6f}, Ky={Ky:.6f}, Kz={Kz:.6f}")
        
        if self.include_single_ion:
            print(f"   Single-ion A: {self.A_param:.6f}")
        if self.include_dmi:
            print(f"   DMI D_z: {self.D_z_param:.6f}")
    
    def energy(self, spins: np.ndarray) -> float:
        """
        Calculate total energy of spin configuration.
        
        Args:
            spins: Spin configuration [N, 3]
            
        Returns:
            total_energy: Total energy (scalar)
        """
        total_energy = 0.0
        
        # Isotropic exchange contributions
        for i, shell_k in enumerate(self.shell_list):
            J_k = self.J_params[i]
            shell_name = f"shell_{shell_k}"
            neighbor_array = self.pair_tables[shell_name]
            
            exchange_energy = self._exchange_energy_shell(spins, neighbor_array, J_k)
            total_energy += exchange_energy
        
        # Kitaev contributions
        if self.include_kitaev:
            for i, shell_k in enumerate(self.shell_list):
                shell_name = f"shell_{shell_k}"
                neighbor_array = self.pair_tables[shell_name]
                
                # For each component (x, y, z)
                for gamma in range(3):
                    K_gamma = self.K_params[i, gamma]
                    kitaev_energy = self._kitaev_energy_shell(
                        spins, neighbor_array, K_gamma, gamma, shell_name
                    )
                    total_energy += kitaev_energy
        
        # Single-ion anisotropy
        if self.include_single_ion:
            anisotropy_energy = self._single_ion_energy(spins, self.A_param)
            total_energy += anisotropy_energy
        
        # DM interaction
        if self.include_dmi:
            shell_1_name = f"shell_1"
            if shell_1_name in self.pair_tables:
                neighbor_array = self.pair_tables[shell_1_name]
                dmi_energy = self._dmi_energy(spins, neighbor_array, self.D_z_param)
                total_energy += dmi_energy
        
        return total_energy
    
    def effective_field(self, spins: np.ndarray) -> np.ndarray:
        """
        Calculate effective field H_i = -∂H/∂s_i for all sites.
        
        Args:
            spins: Spin configuration [N, 3]
            
        Returns:
            fields: Effective fields [N, 3]
        """
        N = spins.shape[0]
        fields = np.zeros((N, 3))
        
        # Isotropic exchange field contributions
        for i, shell_k in enumerate(self.shell_list):
            J_k = self.J_params[i]
            shell_name = f"shell_{shell_k}"
            neighbor_array = self.pair_tables[shell_name]
            
            exchange_field = self._exchange_field_shell(spins, neighbor_array, J_k)
            fields += exchange_field
        
        # Kitaev field contributions
        if self.include_kitaev:
            for i, shell_k in enumerate(self.shell_list):
                shell_name = f"shell_{shell_k}"
                neighbor_array = self.pair_tables[shell_name]
                
                # For each component (x, y, z)
                for gamma in range(3):
                    K_gamma = self.K_params[i, gamma]
                    kitaev_field = self._kitaev_field_shell(
                        spins, neighbor_array, K_gamma, gamma, shell_name
                    )
                    fields += kitaev_field
        
        # Single-ion anisotropy field
        if self.include_single_ion:
            anisotropy_field = self._single_ion_field(spins, self.A_param)
            fields += anisotropy_field
        
        # DM interaction field
        if self.include_dmi:
            shell_1_name = f"shell_1"
            if shell_1_name in self.pair_tables:
                neighbor_array = self.pair_tables[shell_1_name]
                dmi_field = self._dmi_field(spins, neighbor_array, self.D_z_param)
                fields += dmi_field
        
        return fields
    
    def torques(self, spins: np.ndarray) -> np.ndarray:
        """
        Calculate torques τ_i = -∂H/∂s_i (alias for effective_field).
        
        Args:
            spins: Spin configuration [N, 3]
            
        Returns:
            torques: Torques [N, 3]
        """
        return self.effective_field(spins)
    
    def _exchange_energy_shell(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        J_k: float
    ) -> float:
        """Calculate isotropic exchange energy for one shell: J_k Σ_{⟨ij⟩_k} s_i · s_j"""
        energy = 0.0
        N = spins.shape[0]
        
        for i in range(N):
            site_spin = spins[i]
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                neighbor_spin = spins[j]
                dot_product = np.dot(site_spin, neighbor_spin)
                energy += J_k * dot_product
        
        # Divide by 2 to avoid double counting
        return energy / 2.0
    
    def _kitaev_energy_shell(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        K_gamma: float,
        gamma: int,
        shell_name: str
    ) -> float:
        """
        Calculate Kitaev energy for one shell and component: K_γ Σ_{⟨ij⟩_k^γ} s_i^γ s_j^γ
        
        Args:
            spins: Spin configuration [N, 3]
            neighbor_array: Neighbor indices [N, max_neighbors]
            K_gamma: Kitaev coupling for component gamma
            gamma: Spin component index (0=x, 1=y, 2=z)
            shell_name: Name of shell (for bond direction lookup)
        """
        energy = 0.0
        N = spins.shape[0]
        
        for i in range(N):
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                # Check if this bond should contribute to gamma component
                if self._is_kitaev_bond(i, j, gamma, shell_name):
                    # K_γ s_i^γ s_j^γ
                    si_gamma = spins[i, gamma]
                    sj_gamma = spins[j, gamma]
                    energy += K_gamma * si_gamma * sj_gamma
        
        # Divide by 2 to avoid double counting
        return energy / 2.0
    
    def _is_kitaev_bond(self, i: int, j: int, gamma: int, shell_name: str) -> bool:
        """
        Determine if bond i-j should contribute to Kitaev component gamma.
        
        This is a simplified implementation. In real systems, you would determine
        the bond direction based on crystal structure and assign:
        - x-bonds contribute to s_i^x s_j^x
        - y-bonds contribute to s_i^y s_j^y  
        - z-bonds contribute to s_i^z s_j^z
        
        For now, we implement a simple scheme where all bonds contribute to all components
        (this can be customized based on your specific lattice).
        """
        # Simple default: all bonds contribute to all components
        # This can be overridden by providing kitaev_bond_directions
        
        if shell_name in self.kitaev_bond_directions:
            bond_info = self.kitaev_bond_directions[shell_name]
            if (i, j) in bond_info:
                bond_type = bond_info[(i, j)]
                component_names = ['x', 'y', 'z']
                return bond_type == component_names[gamma]
        
        # Default: equal contribution to all components
        return True
    
    def _single_ion_energy(self, spins: np.ndarray, A: float) -> float:
        """Calculate single-ion anisotropy energy: A Σ_i (s_i^z)^2"""
        z_components = spins[:, 2]  # z is index 2
        return A * np.sum(z_components**2)
    
    def _dmi_energy(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        D_z: float
    ) -> float:
        """Calculate DMI energy: D_z Σ_{⟨ij⟩_1} ẑ · (s_i × s_j)"""
        energy = 0.0
        N = spins.shape[0]
        
        for i in range(N):
            site_spin = spins[i]
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                neighbor_spin = spins[j]
                cross_product = np.cross(site_spin, neighbor_spin)
                z_component = cross_product[2]  # ẑ · (s_i × s_j)
                energy += D_z * z_component
        
        # Divide by 2 to avoid double counting
        return energy / 2.0
    
    def _exchange_field_shell(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        J_k: float
    ) -> np.ndarray:
        """Calculate exchange field for one shell: H_i = J_k Σ_j s_j"""
        N = spins.shape[0]
        fields = np.zeros((N, 3))
        
        for i in range(N):
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            if len(valid_neighbors) > 0:
                neighbor_spins = spins[valid_neighbors]
                # H_i = J_k Σ_j s_j (sum over neighbors)
                fields[i] = J_k * np.sum(neighbor_spins, axis=0)
        
        return fields
    
    def _kitaev_field_shell(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        K_gamma: float,
        gamma: int,
        shell_name: str
    ) -> np.ndarray:
        """Calculate Kitaev field for one shell and component."""
        N = spins.shape[0]
        fields = np.zeros((N, 3))
        
        for i in range(N):
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                if self._is_kitaev_bond(i, j, gamma, shell_name):
                    # ∂/∂s_i^α [K_γ s_i^γ s_j^γ] = K_γ δ_{αγ} s_j^γ
                    # So field has component only in direction gamma
                    sj_gamma = spins[j, gamma]
                    fields[i, gamma] += K_gamma * sj_gamma
        
        return fields
    
    def _single_ion_field(self, spins: np.ndarray, A: float) -> np.ndarray:
        """Calculate single-ion anisotropy field: H_i = 2A (s_i^z) ẑ"""
        N = spins.shape[0]
        fields = np.zeros((N, 3))
        
        z_components = spins[:, 2]  # z components
        fields[:, 2] = 2 * A * z_components  # Field only in z direction
        
        return fields
    
    def _dmi_field(
        self, 
        spins: np.ndarray, 
        neighbor_array: np.ndarray, 
        D_z: float
    ) -> np.ndarray:
        """Calculate DMI field: H_i = D_z Σ_j (ẑ × s_j)"""
        N = spins.shape[0]
        fields = np.zeros((N, 3))
        
        z_hat = np.array([0, 0, 1])  # ẑ unit vector
        
        for i in range(N):
            neighbor_indices = neighbor_array[i]
            
            # Filter valid neighbors
            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
            
            for j in valid_neighbors:
                neighbor_spin = spins[j]
                # H_i = D_z (ẑ × s_j)
                cross_product = np.cross(z_hat, neighbor_spin)
                fields[i] += D_z * cross_product
        
        return fields
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """Get parameters as a dictionary for easy access."""
        params = {}
        
        # Isotropic exchange parameters
        for i, shell_k in enumerate(self.shell_list):
            params[f"J{shell_k}"] = self.J_params[i]
        
        # Kitaev parameters
        if self.include_kitaev:
            for i, shell_k in enumerate(self.shell_list):
                params[f"K{shell_k}_x"] = self.K_params[i, 0]
                params[f"K{shell_k}_y"] = self.K_params[i, 1]
                params[f"K{shell_k}_z"] = self.K_params[i, 2]
        
        # Single-ion anisotropy
        if self.include_single_ion:
            params["A"] = self.A_param
        
        # DM interaction
        if self.include_dmi:
            params["D_z"] = self.D_z_param
        
        return params
    
    def summary(self) -> str:
        """Return a summary string of the Hamiltonian."""
        lines = ["PairHamiltonian Summary:"]
        lines.append(f"  Shells: {self.shell_list}")
        
        # Exchange terms
        for i, shell_k in enumerate(self.shell_list):
            lines.append(f"  J{shell_k} = {self.J_params[i]:10.6f} eV")
        
        # Kitaev terms
        if self.include_kitaev:
            lines.append("  Kitaev interactions:")
            for i, shell_k in enumerate(self.shell_list):
                Kx, Ky, Kz = self.K_params[i]
                lines.append(f"    Shell {shell_k}: Kx={Kx:8.6f}, Ky={Ky:8.6f}, Kz={Kz:8.6f} eV")
        
        # Single-ion anisotropy
        if self.include_single_ion:
            lines.append(f"  A  = {self.A_param:10.6f} eV")
        
        # DM interaction
        if self.include_dmi:
            lines.append(f"  D_z = {self.D_z_param:10.6f} eV")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        n_params = len(self.J_params) + (3 * len(self.shell_list) if self.include_kitaev else 0) + int(self.include_single_ion) + int(self.include_dmi)
        return f"PairHamiltonian(shells={self.shell_list}, kitaev={self.include_kitaev}, n_params={n_params})"