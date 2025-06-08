"""
Optimization methods for spin systems.
"""

import numpy as np
from typing import Optional, Dict, Callable, Tuple, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from scipy.optimize import minimize
import time
from tqdm import tqdm

if TYPE_CHECKING:
    from .spin_optimizer import SpinOptimizer


class OptimizationMethod(ABC):
    """Abstract base class for optimization methods."""
    
    def __init__(self, optimizer: 'SpinOptimizer'):
        """Initialize with reference to main optimizer."""
        self.optimizer = optimizer
    
    @abstractmethod
    def optimize(
        self,
        initial_config: np.ndarray,
        max_iterations: int,
        tolerance: float,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform optimization."""
        pass


class LBFGS(OptimizationMethod):
    """
    Limited-memory BFGS optimization using scipy.optimize.
    
    Uses spherical coordinates to handle the constraint that spins
    have fixed magnitude.
    """
    
    def optimize(
        self,
        initial_config: np.ndarray,
        max_iterations: int,
        tolerance: float,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using scipy L-BFGS-B."""
        # Convert to spherical coordinates for unconstrained optimization
        initial_angles = self._cartesian_to_spherical(initial_config)
        
        # Setup optimization tracking
        self._optimization_data = {
            'energies': [],
            'configurations': [],
            'iteration_count': 0
        }
        
        def objective_and_gradient(angles_flat):
            """Combined objective and gradient function."""
            angles = angles_flat.reshape(-1, 2)
            config = self._spherical_to_cartesian(angles)
            
            # Calculate energy and gradient
            energy = self.optimizer.objective_function(config)
            cart_grad = self.optimizer.gradient(config)
            sph_grad = self._transform_gradient_to_spherical(config, cart_grad, angles)
            
            # Store data
            self._optimization_data['energies'].append(energy)
            self._optimization_data['configurations'].append(config.copy())
            self._optimization_data['iteration_count'] += 1
            
            if verbose and self._optimization_data['iteration_count'] % 10 == 0:
                print(f"Iteration {self._optimization_data['iteration_count']}: E = {energy:.6f} eV")
            
            # Call user callback
            if callback is not None:
                callback(self.optimizer, self._optimization_data['iteration_count'], config)
            
            return energy, sph_grad.flatten()
        
        # Run optimization
        start_time = time.time()
        
        result = minimize(
            objective_and_gradient,
            initial_angles.flatten(),
            method='L-BFGS-B',
            jac=True,  # Gradient provided by objective function
            options={
                'maxiter': max_iterations,
                'ftol': tolerance,
                'gtol': tolerance
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Convert final result back to Cartesian
        final_angles = result.x.reshape(-1, 2)
        final_config = self._spherical_to_cartesian(final_angles)
        
        return {
            'converged': result.success,
            'final_configuration': final_config,
            'final_energy': result.fun,
            'iterations': self._optimization_data['iteration_count'],
            'function_evaluations': result.nfev,
            'optimization_time': optimization_time,
            'message': result.message,
            'energy_history': np.array(self._optimization_data['energies']),
            'configuration_history': self._optimization_data['configurations']
        }
    
    def _cartesian_to_spherical(self, config: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to spherical angles."""
        x, y, z = config[:, 0], config[:, 1], config[:, 2]
        r = np.linalg.norm(config, axis=1)
        theta = np.arccos(np.clip(z / r, -1, 1))  # Polar angle [0, π]
        phi = np.arctan2(y, x)    # Azimuthal angle [-π, π]
        return np.column_stack((theta, phi))
    
    def _spherical_to_cartesian(self, angles: np.ndarray) -> np.ndarray:
        """Convert spherical angles to Cartesian coordinates."""
        theta, phi = angles[:, 0], angles[:, 1]
        S = self.optimizer.spin_system.spin_magnitude
        x = S * np.sin(theta) * np.cos(phi)
        y = S * np.sin(theta) * np.sin(phi)
        z = S * np.cos(theta)
        return np.column_stack((x, y, z))
    
    def _transform_gradient_to_spherical(self, config: np.ndarray, cart_grad: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Transform Cartesian gradient to spherical coordinate gradient."""
        theta, phi = angles[:, 0], angles[:, 1]
        S = self.optimizer.spin_system.spin_magnitude
        
        # Jacobian transformation
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        
        # Gradient transformation
        grad_theta = S * (cart_grad[:, 0] * cos_theta * cos_phi + 
                         cart_grad[:, 1] * cos_theta * sin_phi - 
                         cart_grad[:, 2] * sin_theta)
        
        grad_phi = S * sin_theta * (-cart_grad[:, 0] * sin_phi + 
                                   cart_grad[:, 1] * cos_phi)
        
        return np.column_stack((grad_theta, grad_phi))


class ConjugateGradient(OptimizationMethod):
    """
    Conjugate gradient optimization using scipy.optimize.
    
    Uses spherical coordinates to handle spin magnitude constraints.
    """
    
    def optimize(
        self,
        initial_config: np.ndarray,
        max_iterations: int,
        tolerance: float,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using scipy CG method."""
        # Convert to spherical coordinates for unconstrained optimization
        initial_angles = self._cartesian_to_spherical(initial_config)
        
        # Setup optimization tracking
        self._optimization_data = {
            'energies': [],
            'configurations': [],
            'iteration_count': 0
        }
        
        def objective_and_gradient(angles_flat):
            """Combined objective and gradient function."""
            angles = angles_flat.reshape(-1, 2)
            config = self._spherical_to_cartesian(angles)
            
            # Calculate energy and gradient
            energy = self.optimizer.objective_function(config)
            cart_grad = self.optimizer.gradient(config)
            sph_grad = self._transform_gradient_to_spherical(config, cart_grad, angles)
            
            # Store data
            self._optimization_data['energies'].append(energy)
            self._optimization_data['configurations'].append(config.copy())
            self._optimization_data['iteration_count'] += 1
            
            if verbose and self._optimization_data['iteration_count'] % 10 == 0:
                print(f"Iteration {self._optimization_data['iteration_count']}: E = {energy:.6f} eV")
            
            # Call user callback
            if callback is not None:
                callback(self.optimizer, self._optimization_data['iteration_count'], config)
            
            return energy, sph_grad.flatten()
        
        # Run optimization
        start_time = time.time()
        
        result = minimize(
            objective_and_gradient,
            initial_angles.flatten(),
            method='CG',
            jac=True,  # Gradient provided by objective function
            options={
                'maxiter': max_iterations,
                'gtol': tolerance
            }
        )
        
        optimization_time = time.time() - start_time
        
        # Convert final result back to Cartesian
        final_angles = result.x.reshape(-1, 2)
        final_config = self._spherical_to_cartesian(final_angles)
        
        return {
            'converged': result.success,
            'final_configuration': final_config,
            'final_energy': result.fun,
            'iterations': self._optimization_data['iteration_count'],
            'function_evaluations': result.nfev,
            'optimization_time': optimization_time,
            'message': result.message,
            'energy_history': np.array(self._optimization_data['energies']),
            'configuration_history': self._optimization_data['configurations']
        }
    
    def _cartesian_to_spherical(self, config: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to spherical angles."""
        x, y, z = config[:, 0], config[:, 1], config[:, 2]
        r = np.linalg.norm(config, axis=1)
        theta = np.arccos(np.clip(z / r, -1, 1))  # Polar angle [0, π]
        phi = np.arctan2(y, x)    # Azimuthal angle [-π, π]
        return np.column_stack((theta, phi))
    
    def _spherical_to_cartesian(self, angles: np.ndarray) -> np.ndarray:
        """Convert spherical angles to Cartesian coordinates."""
        theta, phi = angles[:, 0], angles[:, 1]
        S = self.optimizer.spin_system.spin_magnitude
        x = S * np.sin(theta) * np.cos(phi)
        y = S * np.sin(theta) * np.sin(phi)
        z = S * np.cos(theta)
        return np.column_stack((x, y, z))
    
    def _transform_gradient_to_spherical(self, config: np.ndarray, cart_grad: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Transform Cartesian gradient to spherical coordinate gradient."""
        theta, phi = angles[:, 0], angles[:, 1]
        S = self.optimizer.spin_system.spin_magnitude
        
        # Jacobian transformation
        sin_theta, cos_theta = np.sin(theta), np.cos(theta)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        
        # Gradient transformation
        grad_theta = S * (cart_grad[:, 0] * cos_theta * cos_phi + 
                         cart_grad[:, 1] * cos_theta * sin_phi - 
                         cart_grad[:, 2] * sin_theta)
        
        grad_phi = S * sin_theta * (-cart_grad[:, 0] * sin_phi + 
                                   cart_grad[:, 1] * cos_phi)
        
        return np.column_stack((grad_theta, grad_phi))


class SimulatedAnnealing(OptimizationMethod):
    """
    Simulated annealing optimization for spin systems.
    """
    
    def optimize(
        self,
        initial_config: np.ndarray,
        max_iterations: int,
        tolerance: float,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        initial_temperature: float = 100.0,
        final_temperature: float = 0.1,
        cooling_rate: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using simulated annealing."""
        
        config = initial_config.copy()
        energy = self.optimizer.objective_function(config)
        
        best_config = config.copy()
        best_energy = energy
        
        temperature = initial_temperature
        energies = [energy]
        temperatures = [temperature]
        accepted_moves = 0
        
        # Get allowed orientations
        orientations = self.optimizer.spin_system.generate_spin_orientations()
        
        kB = 8.617333e-5  # eV/K
        
        if verbose:
            pbar = tqdm(total=max_iterations, desc="SA Optimization")
        
        for iteration in range(max_iterations):
            # Generate new configuration by flipping random spins
            new_config = config.copy()
            n_flips = max(1, int(len(config) * 0.1))  # Flip 10% of spins
            
            for _ in range(n_flips):
                spin_idx = np.random.randint(len(config))
                orientation_idx = np.random.randint(len(orientations))
                
                theta, phi = np.radians(orientations[orientation_idx])
                new_spin = self.optimizer.spin_system.spin_magnitude * np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                new_config[spin_idx] = new_spin
            
            # Calculate new energy
            new_energy = self.optimizer.objective_function(new_config)
            
            # Acceptance criterion
            delta_energy = new_energy - energy
            
            if delta_energy < 0 or (temperature > 0 and 
                                   np.random.random() < np.exp(-delta_energy / (kB * temperature))):
                # Accept move
                config = new_config
                energy = new_energy
                accepted_moves += 1
                
                # Update best
                if energy < best_energy:
                    best_energy = energy
                    best_config = config.copy()
            
            # Cool down
            temperature *= cooling_rate
            temperature = max(temperature, final_temperature)
            
            energies.append(energy)
            temperatures.append(temperature)
            
            # Check convergence
            if temperature <= final_temperature and abs(delta_energy) < tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
            
            # Callback
            if callback is not None:
                callback(self.optimizer, iteration, config)
            
            if verbose:
                pbar.set_postfix({
                    'Energy': f'{energy:.6f}',
                    'Best': f'{best_energy:.6f}',
                    'T': f'{temperature:.2f}'
                })
                pbar.update(1)
        
        if verbose:
            pbar.close()
        
        acceptance_rate = accepted_moves / max_iterations
        
        return {
            'converged': temperature <= final_temperature,
            'final_configuration': best_config,
            'final_energy': best_energy,
            'iterations': iteration + 1,
            'acceptance_rate': acceptance_rate,
            'final_temperature': temperature,
            'energy_history': np.array(energies),
            'temperature_history': np.array(temperatures)
        }


