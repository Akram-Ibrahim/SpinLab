"""
Spin optimization framework for finding ground states and metastable configurations.
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Tuple, Any, Union
import time
from tqdm import tqdm

from ..core.spin_system import SpinSystem
from .methods import ConjugateGradient, LBFGS, SimulatedAnnealing


class SpinOptimizer:
    """
    Framework for optimizing spin configurations to find ground states,
    metastable states, and transition paths.
    """
    
    def __init__(
        self,
        spin_system: SpinSystem,
        method: str = "lbfgs",
        random_seed: Optional[int] = None
    ):
        """
        Initialize spin optimizer.
        
        Args:
            spin_system: SpinSystem to optimize
            method: Optimization method ("lbfgs", "cg", "sa")
            random_seed: Random seed for reproducibility
        """
        self.spin_system = spin_system
        self.method_name = method
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create optimization method
        self.method = self._create_method(method)
        
        # Optimization state
        self.best_energy = np.inf
        self.best_configuration = None
        self.optimization_history = []
        
        # Constraints
        self.constraints = []
        
        # Performance tracking
        self.timing_info = {}
    
    def _create_method(self, method_name: str):
        """Create the specified optimization method."""
        method_map = {
            "lbfgs": LBFGS,
            "cg": ConjugateGradient,
            "sa": SimulatedAnnealing
        }
        
        if method_name not in method_map:
            raise ValueError(f"Unknown optimization method: {method_name}")
        
        return method_map[method_name](self)
    
    def add_constraint(self, constraint_func: Callable[[np.ndarray], bool]):
        """
        Add a constraint function for valid configurations.
        
        Args:
            constraint_func: Function that returns True if configuration is valid
        """
        self.constraints.append(constraint_func)
    
    def _check_constraints(self, spins: np.ndarray) -> bool:
        """Check if a configuration satisfies all constraints."""
        for constraint in self.constraints:
            if not constraint(spins):
                return False
        return True
    
    def objective_function(self, spins: np.ndarray) -> float:
        """
        Objective function to minimize (energy + constraint penalties).
        
        Args:
            spins: Spin configuration
            
        Returns:
            Energy value to minimize
        """
        # Set configuration and calculate energy
        original_config = self.spin_system.spin_config.copy() if self.spin_system.spin_config is not None else None
        self.spin_system.spin_config = spins
        
        try:
            energy = self.spin_system.calculate_energy()
            
            # Add constraint penalties
            if not self._check_constraints(spins):
                energy += 1e6  # Large penalty for constraint violation
            
            return energy
        
        finally:
            # Restore original configuration
            if original_config is not None:
                self.spin_system.spin_config = original_config
    
    def gradient(self, spins: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of energy with respect to spin orientations.
        
        Args:
            spins: Current spin configuration
            
        Returns:
            Gradient vector (torque on each spin)
        """
        # Get neighbors if not already calculated
        if not hasattr(self.spin_system, '_neighbors') or not self.spin_system._neighbors:
            self.spin_system.get_neighbors(4.0)
        
        n_spins = len(spins)
        gradient = np.zeros((n_spins, 3))
        
        # Calculate effective field (negative gradient)
        for i in range(n_spins):
            field = self.spin_system.hamiltonian.calculate_effective_field(
                spins,
                self.spin_system._neighbors,
                self.spin_system.positions,
                i
            )
            
            # The gradient is the torque: S × H_eff
            gradient[i] = np.cross(spins[i], field)
        
        return gradient
    
    def optimize(
        self,
        initial_config: Optional[np.ndarray] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        **method_kwargs
    ) -> Dict[str, Any]:
        """
        Optimize spin configuration to find minimum energy state.
        
        Args:
            initial_config: Initial spin configuration (random if None)
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance
            callback: Optional callback function
            verbose: Whether to show progress
            method_kwargs: Additional arguments for optimization method
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        # Set initial configuration
        if initial_config is None:
            if self.spin_system.spin_config is None:
                self.spin_system.random_configuration(seed=self.random_seed)
            initial_config = self.spin_system.spin_config.copy()
        else:
            self.spin_system.spin_config = initial_config.copy()
        
        # Initialize tracking
        self.optimization_history.clear()
        self.best_energy = self.objective_function(initial_config)
        self.best_configuration = initial_config.copy()
        
        # Run optimization
        result = self.method.optimize(
            initial_config=initial_config,
            max_iterations=max_iterations,
            tolerance=tolerance,
            callback=callback,
            verbose=verbose,
            **method_kwargs
        )
        
        # Update best result
        if result['final_energy'] < self.best_energy:
            self.best_energy = result['final_energy']
            self.best_configuration = result['final_configuration'].copy()
        
        # Calculate timing
        total_time = time.time() - start_time
        self.timing_info = {
            'total_time': total_time,
            'time_per_iteration': total_time / result.get('iterations', 1),
            'iterations_per_second': result.get('iterations', 1) / total_time
        }
        
        result['timing'] = self.timing_info
        result['best_energy'] = self.best_energy
        result['best_configuration'] = self.best_configuration
        
        return result
    
    def find_ground_state(
        self,
        n_attempts: int = 10,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Find ground state using multiple random starting points.
        
        Args:
            n_attempts: Number of optimization attempts with different starting points
            max_iterations: Maximum iterations per attempt
            tolerance: Convergence tolerance
            verbose: Whether to show progress
            
        Returns:
            Dictionary with ground state results
        """
        best_energy = np.inf
        best_config = None
        all_results = []
        
        if verbose:
            print(f"Searching for ground state with {n_attempts} attempts...")
        
        for attempt in range(n_attempts):
            # Generate random starting configuration
            random_config = self.spin_system.random_configuration(seed=self.random_seed + attempt if self.random_seed else None)
            
            # Optimize from this starting point
            result = self.optimize(
                initial_config=random_config,
                max_iterations=max_iterations,
                tolerance=tolerance,
                verbose=False
            )
            
            all_results.append(result)
            
            # Update best result
            if result['final_energy'] < best_energy:
                best_energy = result['final_energy']
                best_config = result['final_configuration'].copy()
            
            if verbose:
                print(f"Attempt {attempt + 1}: E = {result['final_energy']:.6f} eV")
        
        # Set best configuration
        self.best_energy = best_energy
        self.best_configuration = best_config
        self.spin_system.spin_config = best_config
        
        return {
            'ground_state_energy': best_energy,
            'ground_state_configuration': best_config,
            'all_attempts': all_results,
            'energy_spread': np.std([r['final_energy'] for r in all_results]),
            'success_rate': sum(1 for r in all_results if r.get('converged', False)) / n_attempts
        }
    
    def find_metastable_states(
        self,
        n_states: int = 5,
        energy_window: float = 0.1,
        min_separation: float = 0.1,
        max_attempts: int = 100,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find multiple metastable states within an energy window.
        
        Args:
            n_states: Target number of metastable states
            energy_window: Energy window above ground state (eV)
            min_separation: Minimum energy separation between states (eV)
            max_attempts: Maximum optimization attempts
            verbose: Whether to show progress
            
        Returns:
            List of metastable state dictionaries
        """
        # First find ground state
        if self.best_energy == np.inf:
            gs_result = self.find_ground_state(verbose=verbose)
            ground_energy = gs_result['ground_state_energy']
        else:
            ground_energy = self.best_energy
        
        metastable_states = []
        attempts = 0
        
        if verbose:
            print(f"Searching for {n_states} metastable states...")
        
        while len(metastable_states) < n_states and attempts < max_attempts:
            # Generate random starting point
            random_config = self.spin_system.random_configuration(
                seed=self.random_seed + attempts if self.random_seed else None
            )
            
            # Optimize
            result = self.optimize(
                initial_config=random_config,
                max_iterations=500,
                tolerance=1e-6,
                verbose=False
            )
            
            final_energy = result['final_energy']
            
            # Check if this is a valid metastable state
            if (ground_energy <= final_energy <= ground_energy + energy_window):
                # Check separation from existing states
                is_new_state = True
                for existing_state in metastable_states:
                    if abs(final_energy - existing_state['energy']) < min_separation:
                        is_new_state = False
                        break
                
                if is_new_state:
                    metastable_states.append({
                        'energy': final_energy,
                        'configuration': result['final_configuration'].copy(),
                        'relative_energy': final_energy - ground_energy,
                        'optimization_result': result
                    })
                    
                    if verbose:
                        print(f"Found state {len(metastable_states)}: "
                              f"E = {final_energy:.6f} eV "
                              f"(+{final_energy - ground_energy:.6f} eV)")
            
            attempts += 1
        
        # Sort by energy
        metastable_states.sort(key=lambda x: x['energy'])
        
        return metastable_states
    
    def calculate_transition_path(
        self,
        initial_config: np.ndarray,
        final_config: np.ndarray,
        n_images: int = 10,
        method: str = "neb"
    ) -> Dict[str, Any]:
        """
        Calculate transition path between two configurations.
        
        Args:
            initial_config: Starting configuration
            final_config: Target configuration
            n_images: Number of intermediate images
            method: Path calculation method ("neb", "string")
            
        Returns:
            Dictionary with transition path information
        """
        if method == "neb":
            return self._nudged_elastic_band(initial_config, final_config, n_images)
        elif method == "string":
            return self._string_method(initial_config, final_config, n_images)
        else:
            raise ValueError(f"Unknown path method: {method}")
    
    def _nudged_elastic_band(
        self,
        initial_config: np.ndarray,
        final_config: np.ndarray,
        n_images: int
    ) -> Dict[str, Any]:
        """
        Implement Nudged Elastic Band method for transition paths.
        
        This is a simplified implementation - a full NEB would require
        careful treatment of the elastic forces and proper optimization.
        """
        # Create initial path by linear interpolation
        path = []
        for i in range(n_images):
            alpha = i / (n_images - 1)
            config = (1 - alpha) * initial_config + alpha * final_config
            
            # Normalize spins
            magnitudes = np.linalg.norm(config, axis=1, keepdims=True)
            config = config / magnitudes * self.spin_system.spin_magnitude
            
            path.append(config)
        
        # Calculate energies along path
        energies = []
        for config in path:
            energy = self.objective_function(config)
            energies.append(energy)
        
        # Find highest energy point (transition state estimate)
        max_idx = np.argmax(energies)
        transition_state = path[max_idx]
        transition_energy = energies[max_idx]
        
        # Calculate energy barrier
        initial_energy = energies[0]
        final_energy = energies[-1]
        barrier = transition_energy - initial_energy
        
        return {
            'path_configurations': path,
            'path_energies': np.array(energies),
            'transition_state': transition_state,
            'transition_energy': transition_energy,
            'energy_barrier': barrier,
            'initial_energy': initial_energy,
            'final_energy': final_energy
        }
    
    def _string_method(
        self,
        initial_config: np.ndarray,
        final_config: np.ndarray,
        n_images: int
    ) -> Dict[str, Any]:
        """
        Implement string method for transition paths.
        
        This is a placeholder implementation.
        """
        # For now, use same linear interpolation as NEB
        return self._nudged_elastic_band(initial_config, final_config, n_images)
    
    def analyze_hessian(
        self,
        config: Optional[np.ndarray] = None,
        displacement: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Analyze Hessian matrix at a configuration to characterize critical points.
        
        Args:
            config: Configuration to analyze (uses current if None)
            displacement: Finite difference displacement
            
        Returns:
            Dictionary with Hessian analysis
        """
        if config is None:
            config = self.spin_system.spin_config
            if config is None:
                raise ValueError("No configuration to analyze")
        
        n_spins = len(config)
        n_dof = 3 * n_spins  # 3 components per spin
        
        # Calculate Hessian using finite differences
        hessian = np.zeros((n_dof, n_dof))
        base_energy = self.objective_function(config)
        
        flat_config = config.flatten()
        
        for i in range(n_dof):
            for j in range(i, n_dof):
                # Calculate second derivative H_ij
                config_pp = flat_config.copy()
                config_pp[i] += displacement
                config_pp[j] += displacement
                config_pp = config_pp.reshape(n_spins, 3)
                energy_pp = self.objective_function(config_pp)
                
                config_pm = flat_config.copy()
                config_pm[i] += displacement
                config_pm[j] -= displacement
                config_pm = config_pm.reshape(n_spins, 3)
                energy_pm = self.objective_function(config_pm)
                
                config_mp = flat_config.copy()
                config_mp[i] -= displacement
                config_mp[j] += displacement
                config_mp = config_mp.reshape(n_spins, 3)
                energy_mp = self.objective_function(config_mp)
                
                config_mm = flat_config.copy()
                config_mm[i] -= displacement
                config_mm[j] -= displacement
                config_mm = config_mm.reshape(n_spins, 3)
                energy_mm = self.objective_function(config_mm)
                
                hessian[i, j] = (energy_pp - energy_pm - energy_mp + energy_mm) / (4 * displacement**2)
                hessian[j, i] = hessian[i, j]  # Symmetric
        
        # Eigenvalue analysis
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        
        # Classify critical point
        n_negative = np.sum(eigenvalues < -1e-8)
        n_positive = np.sum(eigenvalues > 1e-8)
        n_zero = len(eigenvalues) - n_negative - n_positive
        
        if n_negative == 0:
            critical_point_type = "minimum"
        elif n_negative == 1:
            critical_point_type = "saddle_point_1"
        else:
            critical_point_type = f"saddle_point_{n_negative}"
        
        return {
            'hessian_matrix': hessian,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'critical_point_type': critical_point_type,
            'n_negative_modes': n_negative,
            'n_positive_modes': n_positive,
            'n_zero_modes': n_zero,
            'stability': n_negative == 0
        }
    
    def reset(self):
        """Reset optimizer state."""
        self.best_energy = np.inf
        self.best_configuration = None
        self.optimization_history.clear()
        self.timing_info.clear()
    
    def __repr__(self) -> str:
        return (f"SpinOptimizer(method={self.method_name}, "
                f"n_spins={self.spin_system.n_spins}, "
                f"best_energy={self.best_energy:.6f})")