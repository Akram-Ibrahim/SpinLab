"""
Core SpinSystem class for managing spin configurations and structures.
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Dict, Any
from ase import Atoms
from ase.build import make_supercell

from .hamiltonian import Hamiltonian
from .neighbors import NeighborFinder
from .fast_ops import fast_calculate_magnetization, HAS_NUMBA


class SpinSystem:
    """
    Core class for managing spin systems, configurations, and structures.
    
    This class provides a unified interface for handling different types of
    magnetic systems (Ising, XY, 3D) with flexible Hamiltonian definitions.
    """
    
    def __init__(
        self,
        structure: Atoms,
        hamiltonian: Hamiltonian,
        spin_magnitude: float = 1.0,
        magnetic_model: str = "3d",
        use_fast: bool = True
    ):
        """
        Initialize a spin system.
        
        Args:
            structure: ASE Atoms object describing the atomic structure
            hamiltonian: Hamiltonian object defining interactions
            spin_magnitude: Magnitude of spins (default: 1.0)
            magnetic_model: Type of magnetic model ("ising", "xy", "3d")
            use_fast: Whether to use Numba acceleration
        """
        self.structure = structure.copy()
        self.hamiltonian = hamiltonian
        self.spin_magnitude = spin_magnitude
        self.magnetic_model = magnetic_model.lower()
        self.use_fast = use_fast and HAS_NUMBA
        
        self.n_spins = len(structure)
        self.positions = structure.get_positions()
        
        # Initialize spin configuration as None - will be set later
        self._spin_config = None
        
        # Setup neighbor lists
        self.neighbor_finder = NeighborFinder(structure)
        self._neighbors = {}
        
        # Validate magnetic model
        valid_models = ["ising", "xy", "3d"]
        if self.magnetic_model not in valid_models:
            raise ValueError(f"Invalid magnetic model: {self.magnetic_model}. "
                           f"Must be one of {valid_models}")
    
    @property
    def spin_config(self) -> Optional[np.ndarray]:
        """Get current spin configuration."""
        return self._spin_config
    
    @spin_config.setter
    def spin_config(self, config: np.ndarray):
        """Set spin configuration with validation."""
        if config.shape[0] != self.n_spins:
            raise ValueError(f"Spin config must have {self.n_spins} spins, "
                           f"got {config.shape[0]}")
        
        if self.magnetic_model == "ising":
            if config.shape[1] < 3:
                # Convert from angles to Cartesian if needed
                config = self._angles_to_cartesian(config)
            # Validate Ising spins (should be ±1 in z-direction)
            z_components = config[:, 2]
            if not np.allclose(np.abs(z_components), self.spin_magnitude):
                raise ValueError("Ising spins must be ±S in z-direction")
        
        self._spin_config = config
    
    def get_neighbors(
        self, 
        cutoffs: Union[float, List[float]], 
        max_neighbors: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get neighbor lists for different interaction shells.
        
        Args:
            cutoffs: Cutoff distance(s) for neighbor shells
            max_neighbors: Maximum number of neighbors per shell
            
        Returns:
            Dictionary with neighbor arrays for each shell
        """
        if isinstance(cutoffs, (int, float)):
            cutoffs = [cutoffs]
        
        neighbors = {}
        for i, cutoff in enumerate(cutoffs):
            shell_name = f"shell_{i+1}"
            max_nn = max_neighbors[i] if max_neighbors else None
            neighbors[shell_name] = self.neighbor_finder.find_neighbors(
                cutoff, max_neighbors=max_nn
            )
        
        self._neighbors = neighbors
        return neighbors
    
    def make_supercell(self, supercell_matrix: np.ndarray) -> 'SpinSystem':
        """
        Create a supercell of the current spin system.
        
        Args:
            supercell_matrix: 3x3 matrix defining the supercell
            
        Returns:
            New SpinSystem with supercell structure
        """
        super_structure = make_supercell(self.structure, supercell_matrix)
        
        # Create new spin system with supercell
        super_system = SpinSystem(
            structure=super_structure,
            hamiltonian=self.hamiltonian,
            spin_magnitude=self.spin_magnitude,
            magnetic_model=self.magnetic_model
        )
        
        # If we have a spin configuration, replicate it
        if self._spin_config is not None:
            super_system.spin_config = self._replicate_spin_config(
                supercell_matrix
            )
        
        return super_system
    
    def generate_spin_orientations(
        self, 
        angular_resolution: float = 1.0
    ) -> np.ndarray:
        """
        Generate allowed spin orientations based on magnetic model.
        
        Args:
            angular_resolution: Angular resolution in degrees
            
        Returns:
            Array of (theta, phi) orientations in degrees
        """
        if self.magnetic_model == "ising":
            # Only ±z orientations
            return np.array([[0.0, 0.0], [180.0, 0.0]])
        
        elif self.magnetic_model == "xy":
            # Only xy-plane orientations
            phi_values = np.arange(0, 360, angular_resolution)
            return np.array([[90.0, phi] for phi in phi_values])
        
        elif self.magnetic_model == "3d":
            # Full 3D orientations with uniform distribution on sphere
            n_divs = int(180 / angular_resolution)
            v_values = np.linspace(0, 1, n_divs, endpoint=False)
            theta_values = np.degrees(np.arccos(2 * v_values - 1))
            phi_values = np.arange(0, 360, angular_resolution)
            
            orientations = []
            for theta in theta_values:
                for phi in phi_values:
                    orientations.append([theta, phi])
            
            return np.array(orientations)
    
    def random_configuration(
        self, 
        orientations: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate a random spin configuration.
        
        Args:
            orientations: Allowed orientations (theta, phi) in degrees
            seed: Random seed for reproducibility
            
        Returns:
            Random spin configuration in Cartesian coordinates
        """
        if seed is not None:
            np.random.seed(seed)
        
        if orientations is None:
            orientations = self.generate_spin_orientations()
        
        # Randomly select orientations
        n_orientations = len(orientations)
        indices = np.random.choice(n_orientations, size=self.n_spins)
        selected_orientations = orientations[indices]
        
        # Convert to Cartesian
        config = self._angles_to_cartesian(selected_orientations)
        self.spin_config = config
        
        return config
    
    def ferromagnetic_configuration(
        self, 
        theta: float = 0.0, 
        phi: float = 0.0
    ) -> np.ndarray:
        """
        Generate a ferromagnetic configuration.
        
        Args:
            theta: Polar angle in degrees
            phi: Azimuthal angle in degrees
            
        Returns:
            Ferromagnetic spin configuration
        """
        orientations = np.full((self.n_spins, 2), [theta, phi])
        config = self._angles_to_cartesian(orientations)
        self.spin_config = config
        
        return config
    
    def antiferromagnetic_configuration(
        self, 
        pattern: str = "checkerboard"
    ) -> np.ndarray:
        """
        Generate antiferromagnetic configurations.
        
        Args:
            pattern: AF pattern ("checkerboard", "stripe", etc.)
            
        Returns:
            Antiferromagnetic spin configuration
        """
        if pattern == "checkerboard":
            # Simple checkerboard pattern based on coordinate sum
            coords = self.positions
            parity = ((coords[:, 0] + coords[:, 1] + coords[:, 2]) > 
                     np.median(coords[:, 0] + coords[:, 1] + coords[:, 2]))
            
            config = np.zeros((self.n_spins, 3))
            config[:, 2] = np.where(parity, self.spin_magnitude, -self.spin_magnitude)
        
        else:
            raise NotImplementedError(f"AF pattern '{pattern}' not implemented")
        
        self.spin_config = config
        return config
    
    def _angles_to_cartesian(self, angles: np.ndarray) -> np.ndarray:
        """Convert (theta, phi) angles to Cartesian coordinates."""
        theta = np.radians(angles[:, 0])
        phi = np.radians(angles[:, 1])
        
        x = self.spin_magnitude * np.sin(theta) * np.cos(phi)
        y = self.spin_magnitude * np.sin(theta) * np.sin(phi)
        z = self.spin_magnitude * np.cos(theta)
        
        return np.column_stack((x, y, z))
    
    def _cartesian_to_angles(self, cartesian: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to (theta, phi) angles."""
        x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.degrees(np.arccos(z / r))
        phi = np.degrees(np.arctan2(y, x))
        phi = np.where(phi < 0, phi + 360, phi)  # Ensure phi in [0, 360)
        
        return np.column_stack((theta, phi))
    
    def _replicate_spin_config(self, supercell_matrix: np.ndarray) -> np.ndarray:
        """Replicate spin configuration for supercell."""
        if self._spin_config is None:
            return None
        
        # This is a simplified replication - in practice you might want
        # more sophisticated handling of magnetic domains, etc.
        n_copies = int(np.abs(np.linalg.det(supercell_matrix)))
        return np.tile(self._spin_config, (n_copies, 1))
    
    def calculate_magnetization(self) -> np.ndarray:
        """Calculate total magnetization vector."""
        if self._spin_config is None:
            return np.zeros(3)
        
        if self.use_fast:
            # Use fast Numba implementation
            return fast_calculate_magnetization(self._spin_config)
        else:
            # Fallback NumPy implementation
            return np.mean(self._spin_config, axis=0)
    
    def calculate_energy(self) -> float:
        """Calculate total energy of current configuration."""
        if self._spin_config is None:
            raise ValueError("No spin configuration set")
        
        return self.hamiltonian.calculate_energy(
            self._spin_config, 
            self._neighbors,
            self.positions
        )
    
    def __repr__(self) -> str:
        return (f"SpinSystem(n_spins={self.n_spins}, "
                f"model={self.magnetic_model}, "
                f"spin_magnitude={self.spin_magnitude})")