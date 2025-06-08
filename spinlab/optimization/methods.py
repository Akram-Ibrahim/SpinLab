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
    Limited-memory BFGS optimization for spin systems.
    
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
        """Optimize using L-BFGS."""
        # Convert to spherical coordinates for unconstrained optimization
        initial_angles = self._cartesian_to_spherical(initial_config)
        
        # Setup optimization
        result_dict = {
            'energies': [],
            'configurations': [],
            'gradients': []
        }
        
        def objective(angles_flat):
            """Objective function in spherical coordinates."""
            angles = angles_flat.reshape(-1, 2)
            config = self._spherical_to_cartesian(angles)
            energy = self.optimizer.objective_function(config)
            
            result_dict['energies'].append(energy)
            result_dict['configurations'].append(config.copy())
            
            return energy
        
        def gradient_func(angles_flat):
            """Gradient in spherical coordinates."""
            angles = angles_flat.reshape(-1, 2)
            config = self._spherical_to_cartesian(angles)
            
            # Get Cartesian gradient (torques)
            cart_grad = self.optimizer.gradient(config)
            
            # Transform to spherical coordinate gradient
            sph_grad = self._transform_gradient_to_spherical(config, cart_grad, angles)
            
            result_dict['gradients'].append(sph_grad.copy())
            
            return sph_grad.flatten()
        
        # Setup callback
        iteration_count = [0]
        
        def scipy_callback(xk):
            """Callback for scipy.optimize."""
            iteration_count[0] += 1
            if callback is not None:
                angles = xk.reshape(-1, 2)
                config = self._spherical_to_cartesian(angles)
                callback(self.optimizer, iteration_count[0], config)
            
            if verbose and iteration_count[0] % 10 == 0:
                current_energy = result_dict['energies'][-1] if result_dict['energies'] else 0
                print(f"Iteration {iteration_count[0]}: E = {current_energy:.6f} eV")
        
        # Run optimization
        start_time = time.time()
        
        result = minimize(
            objective,
            initial_angles.flatten(),
            method='L-BFGS-B',
            jac=gradient_func,
            callback=scipy_callback,
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
            'iterations': iteration_count[0],
            'function_evaluations': result.nfev,
            'gradient_evaluations': result.njev,
            'optimization_time': optimization_time,
            'message': result.message,
            'energy_history': np.array(result_dict['energies']),
            'configuration_history': result_dict['configurations']
        }
    
    def _cartesian_to_spherical(self, config: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates to spherical angles."""
        x, y, z = config[:, 0], config[:, 1], config[:, 2]
        
        # Calculate theta and phi
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Polar angle [0, π]
        phi = np.arctan2(y, x)    # Azimuthal angle [-π, π]
        
        return np.column_stack((theta, phi))
    
    def _spherical_to_cartesian(self, angles: np.ndarray) -> np.ndarray:
        """Convert spherical angles to Cartesian coordinates."""
        theta, phi = angles[:, 0], angles[:, 1]
        
        x = self.optimizer.spin_system.spin_magnitude * np.sin(theta) * np.cos(phi)
        y = self.optimizer.spin_system.spin_magnitude * np.sin(theta) * np.sin(phi)
        z = self.optimizer.spin_system.spin_magnitude * np.cos(theta)
        
        return np.column_stack((x, y, z))
    
    def _transform_gradient_to_spherical(
        self, 
        config: np.ndarray, 
        cart_grad: np.ndarray, 
        angles: np.ndarray
    ) -> np.ndarray:
        """Transform Cartesian gradient to spherical coordinate gradient."""
        theta, phi = angles[:, 0], angles[:, 1]
        
        # Transformation matrix elements
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # Partial derivatives of Cartesian coordinates w.r.t. spherical
        S = self.optimizer.spin_system.spin_magnitude
        
        # dx/dtheta, dx/dphi, etc.
        dx_dtheta = S * cos_theta * cos_phi
        dx_dphi = -S * sin_theta * sin_phi
        
        dy_dtheta = S * cos_theta * sin_phi
        dy_dphi = S * sin_theta * cos_phi
        
        dz_dtheta = -S * sin_theta
        dz_dphi = 0
        
        # Transform gradient
        grad_theta = (cart_grad[:, 0] * dx_dtheta + 
                     cart_grad[:, 1] * dy_dtheta + 
                     cart_grad[:, 2] * dz_dtheta)
        
        grad_phi = (cart_grad[:, 0] * dx_dphi + 
                   cart_grad[:, 1] * dy_dphi + 
                   cart_grad[:, 2] * dz_dphi)
        
        return np.column_stack((grad_theta, grad_phi))


class ConjugateGradient(OptimizationMethod):
    """
    Conjugate gradient optimization for spin systems.
    """
    
    def optimize(
        self,
        initial_config: np.ndarray,
        max_iterations: int,
        tolerance: float,
        callback: Optional[Callable] = None,
        verbose: bool = True,
        restart_interval: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize using conjugate gradient method."""
        
        config = initial_config.copy()
        energy = self.optimizer.objective_function(config)
        
        # Initial gradient (torque)
        gradient = self.optimizer.gradient(config)
        search_direction = -gradient.copy()  # Initial search direction
        
        energies = [energy]
        configurations = [config.copy()]
        
        if verbose:
            pbar = tqdm(total=max_iterations, desc="CG Optimization")
        
        for iteration in range(max_iterations):
            # Line search along search direction
            alpha = self._line_search(config, search_direction)
            
            # Update configuration
            new_config = config + alpha * search_direction
            
            # Normalize spins
            new_config = self._normalize_spins(new_config)
            
            # Calculate new energy and gradient
            new_energy = self.optimizer.objective_function(new_config)
            new_gradient = self.optimizer.gradient(new_config)
            
            # Check convergence
            if abs(new_energy - energy) < tolerance and np.linalg.norm(new_gradient) < tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
            
            # Calculate beta for conjugate direction (Polak-Ribière formula)
            if iteration % restart_interval == 0:
                # Restart CG
                beta = 0
            else:
                numerator = np.sum(new_gradient * (new_gradient - gradient))
                denominator = np.sum(gradient * gradient)
                beta = max(0, numerator / denominator) if denominator > 0 else 0
            
            # Update search direction
            search_direction = -new_gradient + beta * search_direction
            
            # Update state
            config = new_config
            energy = new_energy
            gradient = new_gradient
            
            energies.append(energy)
            configurations.append(config.copy())
            
            # Callback
            if callback is not None:
                callback(self.optimizer, iteration, config)
            
            if verbose:
                pbar.set_postfix({'Energy': f'{energy:.6f}'})
                pbar.update(1)
        
        if verbose:
            pbar.close()
        
        return {
            'converged': iteration < max_iterations - 1,
            'final_configuration': config,
            'final_energy': energy,
            'iterations': iteration + 1,
            'energy_history': np.array(energies),
            'configuration_history': configurations
        }
    
    def _line_search(self, config: np.ndarray, direction: np.ndarray) -> float:
        """Simple line search along given direction."""
        alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0]
        best_alpha = 0
        best_energy = self.optimizer.objective_function(config)
        
        for alpha in alpha_values:
            test_config = config + alpha * direction
            test_config = self._normalize_spins(test_config)
            test_energy = self.optimizer.objective_function(test_config)
            
            if test_energy < best_energy:
                best_energy = test_energy
                best_alpha = alpha
        
        return best_alpha
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.optimizer.spin_system.spin_magnitude


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


