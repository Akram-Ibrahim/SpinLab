"""
Utility class for easily building cluster expansion Hamiltonians.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from ase import Atoms

from ..core.hamiltonian import Hamiltonian


class ClusterExpansionBuilder:
    """
    Builder class for constructing cluster expansion Hamiltonians with sublattice resolution.
    
    This class provides a user-friendly interface for setting up complex magnetic Hamiltonians
    with isotropic exchange, Kitaev interactions, DMI, and single-ion anisotropy across
    multiple neighbor shells and sublattices.
    
    Example usage:
        ```python
        # Create builder
        builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
        
        # Set sublattices (optional)
        builder.set_sublattices([0, 1, 0, 1])  # Bipartite lattice
        
        # Add isotropic exchange
        builder.add_exchange("J1_AA", -1.0, shell=1, sublattices=("A", "A"))
        builder.add_exchange("J1_AB", 2.0, shell=1, sublattices=("A", "B"))
        
        # Add Kitaev interactions
        builder.add_kitaev("K1_x_AB", 0.5, shell=1, component="x", sublattices=("A", "B"))
        
        # Add single-ion anisotropy
        builder.add_single_ion("A_A", 0.1, sublattice="A")
        
        # Add DMI
        builder.add_dmi(0.2)
        
        # Build Hamiltonian
        hamiltonian = builder.build()
        ```
    """
    
    def __init__(self, structure: Atoms, shell_list: List[int], auto_bond_directions: bool = True):
        """
        Initialize cluster expansion builder.
        
        Args:
            structure: ASE Atoms object with lattice vectors and atomic positions
            shell_list: List of neighbor shells to include [1, 2, 3, ...]
            auto_bond_directions: Whether to automatically determine bond directions for Kitaev
        """
        self.structure = structure
        self.shell_list = shell_list
        self.auto_bond_directions = auto_bond_directions
        
        # Parameters storage
        self.J_params: Dict[str, float] = {}
        self.K_params: Dict[str, float] = {}
        self.A_params: Dict[str, float] = {}
        self.D_z: Optional[float] = None
        
        # Sublattice configuration
        self.sublattice_indices: Optional[np.ndarray] = None
        self.sublattice_names: List[str] = ["A"]
        self.n_sublattices: int = 1
        
        # Bond directions for Kitaev interactions
        self.bond_directions: Dict[tuple, str] = {}
        
        if auto_bond_directions:
            self._auto_determine_bond_directions()
    
    def set_sublattices(self, sublattice_indices: Union[List[int], np.ndarray], names: Optional[List[str]] = None):
        """
        Set sublattice assignment for each atomic site.
        
        Args:
            sublattice_indices: Array/list assigning each site to sublattice (0, 1, 2, ...)
            names: Optional custom names for sublattices (default: A, B, C, ...)
        """
        self.sublattice_indices = np.array(sublattice_indices, dtype=int)
        self.n_sublattices = int(np.max(self.sublattice_indices)) + 1
        
        if names is not None:
            if len(names) != self.n_sublattices:
                raise ValueError(f"Need {self.n_sublattices} sublattice names, got {len(names)}")
            self.sublattice_names = names
        else:
            self.sublattice_names = [chr(ord('A') + i) for i in range(self.n_sublattices)]
        
        print(f"Set {self.n_sublattices} sublattices: {self.sublattice_names}")
    
    def add_exchange(
        self, 
        name: str, 
        value: float, 
        shell: int, 
        sublattices: Optional[Tuple[str, str]] = None
    ):
        """
        Add isotropic exchange parameter.
        
        Args:
            name: Parameter name (e.g., "J1_AA", "J2_AB")
            value: Exchange coupling value (eV)
            shell: Neighbor shell (1, 2, 3, ...)
            sublattices: Tuple of sublattice names ("A", "B") - auto-generated if None
        """
        if sublattices is None:
            # Generate parameter name automatically
            param_name = f"J{shell}_{name.split('_')[-1] if '_' in name else 'AA'}"
        else:
            sub_i, sub_j = sublattices
            param_name = f"J{shell}_{sub_i}{sub_j}"
        
        self.J_params[param_name] = value
        print(f"Added exchange: {param_name} = {value:.4f} eV")
    
    def add_kitaev(
        self, 
        name: str, 
        value: float, 
        shell: int, 
        component: str, 
        sublattices: Optional[Tuple[str, str]] = None
    ):
        """
        Add Kitaev interaction parameter.
        
        Args:
            name: Parameter name (e.g., "K1_x_AB")
            value: Kitaev coupling value (eV)
            shell: Neighbor shell (1, 2, 3, ...)
            component: Spin component ("x", "y", "z")
            sublattices: Tuple of sublattice names ("A", "B") - auto-generated if None
        """
        if component not in ["x", "y", "z"]:
            raise ValueError("Kitaev component must be 'x', 'y', or 'z'")
        
        if sublattices is None:
            # Extract from name or use default
            if '_' in name and len(name.split('_')) >= 3:
                sub_pair = name.split('_')[-1]
            else:
                sub_pair = "AA"
            param_name = f"K{shell}_{component}_{sub_pair}"
        else:
            sub_i, sub_j = sublattices
            param_name = f"K{shell}_{component}_{sub_i}{sub_j}"
        
        self.K_params[param_name] = value
        print(f"Added Kitaev: {param_name} = {value:.4f} eV")
    
    def add_single_ion(self, name: str, value: float, sublattice: Optional[str] = None):
        """
        Add single-ion anisotropy parameter.
        
        Args:
            name: Parameter name (e.g., "A_A", "A_B")
            value: Anisotropy constant (eV)
            sublattice: Sublattice name ("A", "B", ...) - auto-generated if None
        """
        if sublattice is None:
            # Extract from name or use default
            if '_' in name:
                sub_name = name.split('_')[-1]
            else:
                sub_name = "A"
            param_name = f"A_{sub_name}"
        else:
            param_name = f"A_{sublattice}"
        
        self.A_params[param_name] = value
        print(f"Added single-ion: {param_name} = {value:.4f} eV")
    
    def add_dmi(self, value: float):
        """
        Add Dzyaloshinskii-Moriya interaction.
        
        Args:
            value: DMI constant D_z (eV)
        """
        self.D_z = value
        print(f"Added DMI: D_z = {value:.4f} eV")
    
    def set_bond_direction(self, site_i: int, site_j: int, direction: str):
        """
        Manually set bond direction for Kitaev interactions.
        
        Args:
            site_i: First site index
            site_j: Second site index  
            direction: Bond direction ("x", "y", "z")
        """
        if direction not in ["x", "y", "z"]:
            raise ValueError("Bond direction must be 'x', 'y', or 'z'")
        
        bond_key = (min(site_i, site_j), max(site_i, site_j))
        self.bond_directions[bond_key] = direction
        print(f"Set bond ({site_i}, {site_j}) direction: {direction}")
    
    def _auto_determine_bond_directions(self):
        """
        Automatically determine bond directions based on lattice geometry.
        
        This is a simple implementation that assigns directions based on 
        the primary coordinate differences. For more complex lattices,
        users should manually set bond directions.
        """
        positions = self.structure.get_positions()
        cell = self.structure.get_cell()
        
        # For now, use a simple heuristic based on coordinate differences
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                diff = positions[j] - positions[i]
                
                # Apply periodic boundary conditions
                for k in range(3):
                    if abs(diff[k]) > 0.5 * np.linalg.norm(cell[k]):
                        diff[k] -= np.sign(diff[k]) * np.linalg.norm(cell[k])
                
                # Determine primary direction
                abs_diff = np.abs(diff)
                primary_axis = np.argmax(abs_diff)
                direction = ["x", "y", "z"][primary_axis]
                
                bond_key = (i, j)
                self.bond_directions[bond_key] = direction
        
        print(f"Auto-determined {len(self.bond_directions)} bond directions")
    
    def add_all_exchange_combinations(self, shell: int, values: Dict[str, float]):
        """
        Add all exchange combinations for a given shell.
        
        Args:
            shell: Neighbor shell
            values: Dict mapping sublattice pairs to values {"AA": -1.0, "AB": 2.0, ...}
        """
        for pair, value in values.items():
            if len(pair) == 2:
                sub_i, sub_j = pair[0], pair[1]
                self.add_exchange(f"J{shell}_{pair}", value, shell, (sub_i, sub_j))
            else:
                raise ValueError(f"Invalid sublattice pair: {pair}")
    
    def add_all_kitaev_combinations(self, shell: int, values: Dict[str, float]):
        """
        Add all Kitaev combinations for a given shell.
        
        Args:
            shell: Neighbor shell
            values: Dict mapping component-sublattice combinations to values 
                   {"x_AA": 0.5, "y_AB": 0.3, ...}
        """
        for key, value in values.items():
            parts = key.split('_')
            if len(parts) == 2:
                component, sub_pair = parts
                if len(sub_pair) == 2:
                    sub_i, sub_j = sub_pair[0], sub_pair[1]
                    self.add_kitaev(f"K{shell}_{key}", value, shell, component, (sub_i, sub_j))
                else:
                    raise ValueError(f"Invalid sublattice pair in: {key}")
            else:
                raise ValueError(f"Invalid Kitaev key format: {key}")
    
    def get_parameter_summary(self) -> str:
        """Get a summary of all set parameters."""
        lines = ["Cluster Expansion Parameters:"]
        lines.append(f"  Shells: {self.shell_list}")
        lines.append(f"  Sublattices: {self.sublattice_names}")
        
        if self.J_params:
            lines.append("  Exchange parameters:")
            for name, value in self.J_params.items():
                lines.append(f"    {name}: {value:8.4f} eV")
        
        if self.K_params:
            lines.append("  Kitaev parameters:")
            for name, value in self.K_params.items():
                lines.append(f"    {name}: {value:8.4f} eV")
        
        if self.A_params:
            lines.append("  Single-ion parameters:")
            for name, value in self.A_params.items():
                lines.append(f"    {name}: {value:8.4f} eV")
        
        if self.D_z is not None:
            lines.append(f"  DMI: D_z = {self.D_z:8.4f} eV")
        
        return "\n".join(lines)
    
    def build(self) -> Hamiltonian:
        """
        Build and return the cluster expansion Hamiltonian.
        
        Returns:
            Hamiltonian object with cluster expansion term
        """
        hamiltonian = Hamiltonian()
        
        # Validate that we have at least some parameters
        has_params = bool(self.J_params or self.K_params or self.A_params or self.D_z)
        if not has_params:
            raise ValueError("No parameters set. Add at least one interaction before building.")
        
        # Add cluster expansion term
        hamiltonian.add_cluster_expansion(
            shell_list=self.shell_list,
            sublattice_indices=self.sublattice_indices,
            J_params=self.J_params if self.J_params else None,
            K_params=self.K_params if self.K_params else None,
            A_params=self.A_params if self.A_params else None,
            D_z=self.D_z,
            bond_directions=self.bond_directions if self.bond_directions else None,
            name="cluster_expansion"
        )
        
        print("✅ Built cluster expansion Hamiltonian")
        print(self.get_parameter_summary())
        
        return hamiltonian
    
    def save_parameters(self, filename: str):
        """Save parameters to file for later use."""
        import json
        
        data = {
            "shell_list": self.shell_list,
            "sublattice_indices": self.sublattice_indices.tolist() if self.sublattice_indices is not None else None,
            "sublattice_names": self.sublattice_names,
            "J_params": self.J_params,
            "K_params": self.K_params,
            "A_params": self.A_params,
            "D_z": self.D_z,
            "bond_directions": {f"{k[0]},{k[1]}": v for k, v in self.bond_directions.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved parameters to {filename}")
    
    def load_parameters(self, filename: str):
        """Load parameters from file."""
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.shell_list = data["shell_list"]
        self.sublattice_indices = np.array(data["sublattice_indices"]) if data["sublattice_indices"] else None
        self.sublattice_names = data["sublattice_names"]
        self.J_params = data["J_params"]
        self.K_params = data["K_params"]
        self.A_params = data["A_params"]
        self.D_z = data["D_z"]
        
        # Convert bond directions back
        self.bond_directions = {}
        for key_str, direction in data["bond_directions"].items():
            i, j = map(int, key_str.split(','))
            self.bond_directions[(i, j)] = direction
        
        print(f"Loaded parameters from {filename}")


# Convenience functions for common lattice types
def create_bipartite_hamiltonian(
    structure: Atoms,
    shell_list: List[int],
    J_AA: float = 0.0,
    J_AB: float = -1.0,
    J_BB: float = 0.0,
    include_kitaev: bool = False,
    K_values: Optional[Dict[str, float]] = None
) -> Hamiltonian:
    """
    Create a Hamiltonian for a bipartite lattice (e.g., honeycomb, square).
    
    Args:
        structure: ASE Atoms object
        shell_list: List of neighbor shells
        J_AA: Exchange coupling within sublattice A
        J_AB: Exchange coupling between sublattices A and B
        J_BB: Exchange coupling within sublattice B
        include_kitaev: Whether to include Kitaev interactions
        K_values: Kitaev coupling values {"x_AB": value, "y_AB": value, "z_AB": value}
    
    Returns:
        Hamiltonian object
    """
    builder = ClusterExpansionBuilder(structure, shell_list)
    
    # Auto-detect bipartite structure (simplified - assumes alternating pattern)
    n_sites = len(structure)
    sublattice_indices = [i % 2 for i in range(n_sites)]
    builder.set_sublattices(sublattice_indices, ["A", "B"])
    
    # Add exchange interactions for first shell
    if abs(J_AA) > 1e-12:
        builder.add_exchange("J1_AA", J_AA, shell=1, sublattices=("A", "A"))
    if abs(J_AB) > 1e-12:
        builder.add_exchange("J1_AB", J_AB, shell=1, sublattices=("A", "B"))
    if abs(J_BB) > 1e-12:
        builder.add_exchange("J1_BB", J_BB, shell=1, sublattices=("B", "B"))
    
    # Add Kitaev interactions if requested
    if include_kitaev and K_values:
        for key, value in K_values.items():
            if abs(value) > 1e-12:
                component = key.split('_')[0]
                sub_pair = key.split('_')[1] if len(key.split('_')) > 1 else "AB"
                sub_i, sub_j = sub_pair[0], sub_pair[1]
                builder.add_kitaev(f"K1_{key}", value, shell=1, component=component, sublattices=(sub_i, sub_j))
    
    return builder.build()


def create_triangular_hamiltonian(
    structure: Atoms,
    shell_list: List[int],
    J1: float = -1.0,
    J2: float = 0.0,
    J3: float = 0.0,
    include_dmi: bool = False,
    D_z: float = 0.0
) -> Hamiltonian:
    """
    Create a Hamiltonian for a triangular lattice.
    
    Args:
        structure: ASE Atoms object
        shell_list: List of neighbor shells
        J1: First neighbor exchange
        J2: Second neighbor exchange  
        J3: Third neighbor exchange
        include_dmi: Whether to include DMI
        D_z: DMI coupling value
    
    Returns:
        Hamiltonian object
    """
    builder = ClusterExpansionBuilder(structure, shell_list)
    
    # Single sublattice for triangular lattice
    builder.set_sublattices([0] * len(structure), ["A"])
    
    # Add exchange interactions
    J_values = [J1, J2, J3]
    for i, shell in enumerate(shell_list[:3]):
        if i < len(J_values) and abs(J_values[i]) > 1e-12:
            builder.add_exchange(f"J{shell}_AA", J_values[i], shell=shell, sublattices=("A", "A"))
    
    # Add DMI if requested
    if include_dmi and abs(D_z) > 1e-12:
        builder.add_dmi(D_z)
    
    return builder.build()