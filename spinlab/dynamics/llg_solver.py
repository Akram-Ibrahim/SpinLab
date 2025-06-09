"""
Landau-Lifshitz-Gilbert (LLG) equation solver for spin dynamics.
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Tuple, Any, Union
import time
from tqdm import tqdm

from ..core.spin_system import SpinSystem
from .integrators import HeunIntegrator, RK4Integrator, SemiImplicitIntegrator
from ..core.fast_ops import (
    llg_rhs, normalize_spins, HAS_NUMBA
)


class LLGSolver:
    """
    Solver for the Landau-Lifshitz-Gilbert equation.
    
    The LLG equation describes the precession and damping of magnetic moments:
    dS/dt = -γ(S × H_eff) + (α/|S|)(S × dS/dt)
    
    where:
    - γ is the gyromagnetic ratio
    - α is the Gilbert damping parameter  
    - H_eff is the effective magnetic field
    """
    
    def __init__(
        self,
        spin_system: SpinSystem,
        damping: float = 0.01,
        gyromagnetic_ratio: float = 1.76e11,  # rad/(s·T)
        integrator: str = "heun",
        random_seed: Optional[int] = None,
        use_fast: bool = True
    ):
        """
        Initialize LLG solver.
        
        Args:
            spin_system: SpinSystem to evolve
            damping: Gilbert damping parameter α
            gyromagnetic_ratio: Gyromagnetic ratio γ in rad/(s·T)
            integrator: Integration method ("heun", "rk4", "semi_implicit")
            random_seed: Random seed for stochastic dynamics
            use_fast: Whether to use Numba acceleration
        """
        self.spin_system = spin_system
        self.damping = damping
        self.gamma = gyromagnetic_ratio
        self.random_seed = random_seed
        self.use_fast = use_fast and HAS_NUMBA
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize integrator
        self.integrator = self._create_integrator(integrator)
        
        # Current state
        self.time = 0.0
        self.step_count = 0
        
        # Data storage
        self.time_history: List[float] = []
        self.energy_history: List[float] = []
        self.magnetization_history: List[np.ndarray] = []
        self.torque_history: List[float] = []
        
        # Performance tracking
        self.timing_info = {}
        
        # Temperature for stochastic dynamics (if needed)
        self.temperature = 0.0
        self.kB = 8.617333e-5  # eV/K
        
        # Pre-computed data for fast operations
        self._neighbor_array = None
        self._J_effective = None
    
    def _create_integrator(self, integrator_name: str):
        """Create the specified integrator."""
        integrator_map = {
            "heun": HeunIntegrator,
            "rk4": RK4Integrator, 
            "semi_implicit": SemiImplicitIntegrator
        }
        
        if integrator_name not in integrator_map:
            raise ValueError(f"Unknown integrator: {integrator_name}")
        
        return integrator_map[integrator_name](self)
    
    def set_temperature(self, temperature: float):
        """Set temperature for stochastic LLG dynamics."""
        self.temperature = temperature
    
    def calculate_effective_field(self, spins: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate effective magnetic field for all spins.
        
        Args:
            spins: Spin configuration (uses current if None)
            
        Returns:
            Effective field for each spin (Tesla)
        """
        if spins is None:
            spins = self.spin_system.spin_config
        
        if spins is None:
            raise ValueError("No spin configuration available")
        
        # Get neighbors if not already calculated
        if not hasattr(self.spin_system, '_neighbors') or not self.spin_system._neighbors:
            # Use default neighbor finding
            self.spin_system.get_neighbors(4.0)  # Default cutoff
        
        n_spins = len(spins)
        effective_field = np.zeros((n_spins, 3))
        
        # Calculate field from Hamiltonian
        for i in range(n_spins):
            field = self.spin_system.hamiltonian.calculate_effective_field(
                spins, 
                self.spin_system._neighbors,
                self.spin_system.positions,
                i
            )
            
            # Convert from energy units to Tesla
            # Field in eV → Field in Tesla (approximate conversion)
            mu_B = 5.78838e-5  # eV/T
            effective_field[i] = field / mu_B
        
        return effective_field
    
    def calculate_llg_rhs(self, spins: np.ndarray) -> np.ndarray:
        """
        Calculate right-hand side of LLG equation.
        
        Args:
            spins: Current spin configuration
            
        Returns:
            Time derivatives dS/dt
        """
        # Calculate effective field
        H_eff = self.calculate_effective_field(spins)
        
        # Add thermal noise if temperature > 0
        if self.temperature > 0:
            H_eff += self._calculate_thermal_field()
        
        if self.use_fast:
            # Use fast Numba implementation
            return llg_rhs(spins, H_eff, self.gamma, self.damping)
        else:
            # Fallback NumPy implementation
            # LLG equation: dS/dt = -γ(S × H_eff) + α(S × (S × H_eff))/|S|²
            
            # Precession term: -γ(S × H_eff)
            precession = -self.gamma * np.cross(spins, H_eff)
            
            # Damping term: α(S × (S × H_eff))/|S|²
            cross_SH = np.cross(spins, H_eff)
            damping_term = self.damping * np.cross(spins, cross_SH)
            
            # Normalize by |S|² (spin magnitude squared)
            spin_magnitudes_sq = np.sum(spins**2, axis=1, keepdims=True)
            damping_term = damping_term / spin_magnitudes_sq
            
            return precession + damping_term
    
    def _calculate_thermal_field(self) -> np.ndarray:
        """Calculate stochastic thermal field for finite temperature dynamics."""
        if self.temperature <= 0:
            return np.zeros((self.spin_system.n_spins, 3))
        
        # Thermal field strength
        # From fluctuation-dissipation theorem
        mu_B = 5.78838e-5  # eV/T
        dt = getattr(self, '_current_dt', 1e-15)  # Default timestep in seconds
        
        field_strength = np.sqrt(
            2 * self.damping * self.kB * self.temperature / 
            (self.gamma * mu_B * dt)
        )
        
        # Generate random field
        thermal_field = np.random.normal(
            0, field_strength, 
            (self.spin_system.n_spins, 3)
        )
        
        return thermal_field
    
    def step(self, dt: float) -> np.ndarray:
        """
        Perform one integration step.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Updated spin configuration
        """
        self._current_dt = dt  # Store for thermal field calculation
        
        if self.spin_system.spin_config is None:
            raise ValueError("No spin configuration to evolve")
        
        # Perform integration step
        new_spins = self.integrator.step(self.spin_system.spin_config, dt)
        
        # Normalize spins to maintain magnitude
        new_spins = self._normalize_spins(new_spins)
        
        # Update system
        self.spin_system.spin_config = new_spins
        self.time += dt
        self.step_count += 1
        
        return new_spins
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain constant magnitude."""
        if self.use_fast:
            # Use fast Numba implementation
            return normalize_spins(spins, self.spin_system.spin_magnitude)
        else:
            # Fallback NumPy implementation
            magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
            magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)  # Avoid division by zero
            
            normalized = spins / magnitudes
            return normalized * self.spin_system.spin_magnitude
    
    def run(
        self,
        total_time: float,
        dt: float,
        sampling_interval: int = 1,
        callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run LLG dynamics simulation.
        
        Args:
            total_time: Total simulation time (seconds)
            dt: Time step (seconds)
            sampling_interval: Steps between data sampling
            callback: Optional callback function
            verbose: Whether to show progress
            
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        n_steps = int(total_time / dt)
        
        # Initialize configuration if needed
        if self.spin_system.spin_config is None:
            self.spin_system.random_configuration()
        
        # Setup progress bar
        pbar = tqdm(total=n_steps, desc="LLG Steps", disable=not verbose)
        
        # Reset data storage
        self.time_history.clear()
        self.energy_history.clear()
        self.magnetization_history.clear()
        self.torque_history.clear()
        
        # Initial data point
        self._record_data()
        
        # Main simulation loop
        for step in range(n_steps):
            # Perform integration step
            self.step(dt)
            
            # Record data
            if step % sampling_interval == 0:
                self._record_data()
            
            # Call callback if provided
            if callback is not None:
                callback(self, step)
            
            pbar.update(1)
        
        pbar.close()
        
        # Calculate timing
        total_sim_time = time.time() - start_time
        self.timing_info = {
            'total_time': total_sim_time,
            'time_per_step': total_sim_time / n_steps,
            'steps_per_second': n_steps / total_sim_time,
            'simulated_time': total_time
        }
        
        # Compile results
        results = {
            'times': np.array(self.time_history),
            'energies': np.array(self.energy_history),
            'magnetizations': np.array(self.magnetization_history),
            'torques': np.array(self.torque_history),
            'final_magnetization': self.spin_system.calculate_magnetization(),
            'final_energy': self.spin_system.calculate_energy(),
            'timing': self.timing_info
        }
        
        return results
    
    def _record_data(self):
        """Record current state data."""
        self.time_history.append(self.time)
        
        # Energy
        energy = self.spin_system.calculate_energy()
        self.energy_history.append(energy)
        
        # Magnetization
        magnetization = self.spin_system.calculate_magnetization()
        self.magnetization_history.append(magnetization.copy())
        
        # Average torque magnitude
        if self.spin_system.spin_config is not None:
            drdt = self.calculate_llg_rhs(self.spin_system.spin_config)
            avg_torque = np.mean(np.linalg.norm(drdt, axis=1))
            self.torque_history.append(avg_torque)
    
    def find_equilibrium(
        self,
        dt: float = 1e-15,
        max_time: float = 1e-9,
        torque_tolerance: float = 1e-6,
        energy_tolerance: float = 1e-12,
        check_interval: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evolve system to equilibrium using energy and torque criteria.
        
        Args:
            dt: Time step
            max_time: Maximum simulation time
            torque_tolerance: Convergence criterion for torque
            energy_tolerance: Convergence criterion for energy change
            check_interval: Steps between convergence checks
            verbose: Whether to show progress
            
        Returns:
            Dictionary with equilibrium results
        """
        if self.spin_system.spin_config is None:
            self.spin_system.random_configuration()
        
        max_steps = int(max_time / dt)
        pbar = tqdm(total=max_steps, desc="Finding equilibrium", disable=not verbose)
        
        energy_window = []
        torque_window = []
        window_size = 10
        
        converged = False
        step = 0
        
        while step < max_steps and not converged:
            # Perform step
            self.step(dt)
            step += 1
            
            # Check convergence periodically
            if step % check_interval == 0:
                # Calculate current torque
                drdt = self.calculate_llg_rhs(self.spin_system.spin_config)
                avg_torque = np.mean(np.linalg.norm(drdt, axis=1))
                
                # Calculate current energy
                current_energy = self.spin_system.calculate_energy()
                
                # Update windows
                torque_window.append(avg_torque)
                energy_window.append(current_energy)
                
                if len(torque_window) > window_size:
                    torque_window.pop(0)
                    energy_window.pop(0)
                
                # Check convergence
                if len(torque_window) == window_size:
                    torque_converged = avg_torque < torque_tolerance
                    energy_converged = (np.std(energy_window) < energy_tolerance)
                    
                    if torque_converged and energy_converged:
                        converged = True
                        if verbose:
                            print(f"\nConverged at step {step}")
                            print(f"Final torque: {avg_torque:.2e}")
                            print(f"Energy std: {np.std(energy_window):.2e}")
            
            pbar.update(1)
        
        pbar.close()
        
        final_energy = self.spin_system.calculate_energy()
        final_magnetization = self.spin_system.calculate_magnetization()
        
        return {
            'converged': converged,
            'steps': step,
            'final_time': self.time,
            'final_energy': final_energy,
            'final_magnetization': final_magnetization,
            'final_torque': avg_torque if 'avg_torque' in locals() else None
        }
    
    def calculate_spin_wave_spectrum(
        self,
        q_points: np.ndarray,
        temperature: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Calculate spin wave spectrum using linear response theory.
        
        Args:
            q_points: Array of q-points to calculate
            temperature: Temperature for thermal averaging
            
        Returns:
            Dictionary with frequencies and eigenvectors
        """
        # This is a simplified implementation
        # Full implementation would require detailed calculation of
        # the Hessian matrix of the energy with respect to spin orientations
        
        # Find equilibrium configuration
        equilibrium_result = self.find_equilibrium(verbose=False)
        
        if not equilibrium_result['converged']:
            print("Warning: System not fully converged for spin wave calculation")
        
        # For now, return placeholder results
        # Full implementation would involve:
        # 1. Calculate exchange matrix in reciprocal space
        # 2. Diagonalize to find eigenfrequencies and eigenvectors
        # 3. Include temperature effects via Bose-Einstein statistics
        
        n_q = len(q_points)
        n_modes = 2 * self.spin_system.n_spins  # 2 modes per spin (transverse)
        
        frequencies = np.zeros((n_q, n_modes))
        eigenvectors = np.zeros((n_q, n_modes, n_modes), dtype=complex)
        
        # Placeholder: uniform dispersion
        for i, q in enumerate(q_points):
            q_magnitude = np.linalg.norm(q)
            frequencies[i] = q_magnitude * np.ones(n_modes)  # Linear dispersion
            eigenvectors[i] = np.eye(n_modes, dtype=complex)  # Identity eigenvectors
        
        return {
            'q_points': q_points,
            'frequencies': frequencies,
            'eigenvectors': eigenvectors,
            'equilibrium_energy': equilibrium_result['final_energy']
        }
    
    def reset(self):
        """Reset solver state."""
        self.time = 0.0
        self.step_count = 0
        self.time_history.clear()
        self.energy_history.clear()
        self.magnetization_history.clear()
        self.torque_history.clear()
        self.timing_info.clear()
    
    def save_trajectory(self, filename: str):
        """Save time evolution trajectory."""
        data = {
            'times': np.array(self.time_history),
            'energies': np.array(self.energy_history),
            'magnetizations': np.array(self.magnetization_history),
            'torques': np.array(self.torque_history),
            'parameters': {
                'damping': self.damping,
                'gamma': self.gamma,
                'temperature': self.temperature
            }
        }
        np.savez(filename, **data)
    
    def __repr__(self) -> str:
        return (f"LLGSolver(damping={self.damping}, "
                f"gamma={self.gamma:.2e}, "
                f"steps={self.step_count})")