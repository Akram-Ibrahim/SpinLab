"""
Monte Carlo simulation engine for spin systems.
"""

import numpy as np
from typing import Optional, Dict, List, Callable, Tuple, Any
import time
from tqdm import tqdm

from .spin_system import SpinSystem
from ..utils.random import set_random_seed
from .fast_ops import (
    fast_mc_sweep, fast_metropolis_single_flip, 
    fast_calculate_magnetization, HAS_NUMBA
)


class MonteCarlo:
    """
    Monte Carlo simulation engine for spin systems.
    
    Implements Metropolis-Hastings algorithm with various optimization
    features and analysis capabilities.
    """
    
    def __init__(
        self,
        spin_system: SpinSystem,
        temperature: float,
        random_seed: Optional[int] = None,
        use_fast: bool = True
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            spin_system: SpinSystem to simulate
            temperature: Temperature in Kelvin
            random_seed: Random seed for reproducibility
            use_fast: Whether to use Numba acceleration
        """
        self.spin_system = spin_system
        self.temperature = temperature
        self.random_seed = random_seed
        self.use_fast = use_fast and HAS_NUMBA
        
        if random_seed is not None:
            set_random_seed(random_seed)
        
        # Boltzmann constant in eV/K
        self.kB = 8.617333e-5
        
        # Simulation state
        self.current_energy = None
        self.step_count = 0
        self.accepted_moves = 0
        
        # Data storage
        self.energy_history: List[float] = []
        self.magnetization_history: List[np.ndarray] = []
        self.acceptance_history: List[float] = []
        
        # Performance tracking
        self.timing_info = {}
        
        # Pre-compute neighbor array for fast operations
        self._neighbor_array = None
        self._J_effective = None
        self._orientations_rad = None
    
    def _prepare_fast_operations(self, orientations: np.ndarray):
        """Prepare data structures for fast Numba operations."""
        if not self.use_fast:
            return
        
        # Get neighbor array (assume first shell for now)
        if hasattr(self.spin_system, '_neighbors') and self.spin_system._neighbors:
            first_shell = list(self.spin_system._neighbors.keys())[0]
            self._neighbor_array = self.spin_system._neighbors[first_shell]
        else:
            # Use a default neighbors array if not available
            self._neighbor_array = np.full((self.spin_system.n_spins, 6), -1, dtype=np.int64)
        
        # Extract effective exchange coupling (simplified for now)
        # In a full implementation, this would handle multiple exchange terms
        self._J_effective = -0.01  # Default value
        
        # Get Hamiltonian terms and extract exchange
        for term in self.spin_system.hamiltonian.terms:
            if hasattr(term, 'J'):
                self._J_effective = term.J
                break
        
        # Convert orientations to radians
        self._orientations_rad = np.radians(orientations)
    
    def run(
        self,
        n_steps: int,
        equilibration_steps: int = 1000,
        sampling_interval: int = 1,
        orientations: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_steps: Total number of MC steps
            equilibration_steps: Steps for equilibration before sampling
            sampling_interval: Interval between samples
            orientations: Allowed spin orientations (auto-generated if None)
            callback: Optional callback function called each step
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        # Generate orientations if not provided
        if orientations is None:
            orientations = self.spin_system.generate_spin_orientations()
        
        # Initialize spin configuration if not set
        if self.spin_system.spin_config is None:
            self.spin_system.random_configuration(orientations, self.random_seed)
        
        # Prepare fast operations
        self._prepare_fast_operations(orientations)
        
        # Calculate initial energy
        self.current_energy = self.spin_system.calculate_energy()
        
        # Setup progress bar
        pbar = tqdm(total=n_steps, desc="MC Steps", disable=not verbose)
        
        # Reset counters
        self.step_count = 0
        self.accepted_moves = 0
        sampling_count = 0
        
        # Main simulation loop
        for step in range(n_steps):
            # Perform one MC sweep (one attempt per spin)
            sweep_accepted = self._perform_sweep(orientations)
            self.accepted_moves += sweep_accepted
            self.step_count += 1
            
            # Record data after equilibration
            if step >= equilibration_steps and step % sampling_interval == 0:
                self.energy_history.append(self.current_energy)
                magnetization = self.spin_system.calculate_magnetization()
                self.magnetization_history.append(magnetization.copy())
                
                acceptance_rate = self.accepted_moves / (self.step_count * self.spin_system.n_spins)
                self.acceptance_history.append(acceptance_rate)
                
                sampling_count += 1
            
            # Call callback if provided
            if callback is not None:
                callback(self, step)
            
            pbar.update(1)
        
        pbar.close()
        
        # Calculate timing
        total_time = time.time() - start_time
        self.timing_info = {
            'total_time': total_time,
            'time_per_step': total_time / n_steps,
            'steps_per_second': n_steps / total_time
        }
        
        # Compile results
        final_acceptance_rate = self.accepted_moves / (self.step_count * self.spin_system.n_spins)
        results = {
            'energies': np.array(self.energy_history),
            'magnetizations': np.array(self.magnetization_history),
            'acceptance_rates': np.array(self.acceptance_history),
            'final_energy': self.current_energy,
            'final_magnetization': self.spin_system.calculate_magnetization(),
            'acceptance_rate': final_acceptance_rate,  # For parallel MC compatibility
            'total_acceptance_rate': final_acceptance_rate,  # Keep backward compatibility
            'n_samples': sampling_count,
            'n_steps': n_steps,  # Add for parallel MC compatibility
            'timing': self.timing_info
        }
        
        return results
    
    def _perform_sweep(self, orientations: np.ndarray) -> int:
        """
        Perform one Monte Carlo sweep (attempt to flip each spin once).
        
        Args:
            orientations: Allowed spin orientations
            
        Returns:
            Number of accepted moves in this sweep
        """
        if self.use_fast and self._neighbor_array is not None and self._orientations_rad is not None:
            # Use fast Numba implementation
            n_accepted, delta_energy = fast_mc_sweep(
                self.spin_system.spin_config,
                self._neighbor_array,
                self._orientations_rad,
                self._J_effective,
                self.temperature,
                self.spin_system.spin_magnitude,
                random_order=True
            )
            
            # Update energy
            self.current_energy += delta_energy
            
            return n_accepted
        else:
            # Fallback to original implementation
            accepted_in_sweep = 0
            
            # Randomize order of spin updates
            spin_indices = np.random.permutation(self.spin_system.n_spins)
            
            for spin_idx in spin_indices:
                if self._attempt_spin_flip(spin_idx, orientations):
                    accepted_in_sweep += 1
            
            return accepted_in_sweep
    
    def _attempt_spin_flip(self, spin_idx: int, orientations: np.ndarray) -> bool:
        """
        Attempt to flip a single spin.
        
        Args:
            spin_idx: Index of spin to flip
            orientations: Allowed orientations
            
        Returns:
            True if move was accepted
        """
        # Store original spin
        original_spin = self.spin_system.spin_config[spin_idx].copy()
        
        # Propose new orientation
        new_orientation_idx = np.random.randint(len(orientations))
        new_orientation = orientations[new_orientation_idx]
        
        # Convert to Cartesian and update spin
        theta, phi = np.radians(new_orientation)
        new_spin = self.spin_system.spin_magnitude * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        self.spin_system.spin_config[spin_idx] = new_spin
        
        # Calculate energy change
        new_energy = self.spin_system.calculate_energy()
        delta_energy = new_energy - self.current_energy
        
        # Metropolis acceptance criterion
        if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / (self.kB * self.temperature)):
            # Accept move
            self.current_energy = new_energy
            return True
        else:
            # Reject move - restore original spin
            self.spin_system.spin_config[spin_idx] = original_spin
            return False
    
    def calculate_thermodynamic_properties(self) -> Dict[str, float]:
        """
        Calculate thermodynamic properties from simulation data.
        
        Returns:
            Dictionary with thermodynamic properties
        """
        if len(self.energy_history) == 0:
            raise ValueError("No simulation data available")
        
        energies = np.array(self.energy_history)
        magnetizations = np.array(self.magnetization_history)
        
        # Energy properties
        mean_energy = np.mean(energies)
        energy_variance = np.var(energies)
        
        # Heat capacity
        heat_capacity = energy_variance / (self.kB * self.temperature**2 * self.spin_system.n_spins)
        
        # Magnetization properties
        mag_magnitudes = np.linalg.norm(magnetizations, axis=1)
        mean_magnetization = np.mean(mag_magnitudes)
        magnetization_variance = np.var(mag_magnitudes)
        
        # Magnetic susceptibility
        susceptibility = magnetization_variance / (self.kB * self.temperature * self.spin_system.n_spins)
        
        # Binder cumulant (fourth-order cumulant)
        mag_squared = mag_magnitudes**2
        mag_fourth = mag_magnitudes**4
        mean_mag_squared = np.mean(mag_squared)
        mean_mag_fourth = np.mean(mag_fourth)
        
        if mean_mag_squared > 0:
            binder_cumulant = 1 - mean_mag_fourth / (3 * mean_mag_squared**2)
        else:
            binder_cumulant = 0
        
        return {
            'mean_energy': mean_energy,
            'energy_per_spin': mean_energy / self.spin_system.n_spins,
            'heat_capacity': heat_capacity,
            'mean_magnetization': mean_magnetization,
            'magnetization_variance': magnetization_variance,
            'susceptibility': susceptibility,
            'binder_cumulant': binder_cumulant,
            'temperature': self.temperature
        }
    
    def estimate_correlation_time(self, observable: str = 'energy') -> float:
        """
        Estimate autocorrelation time for an observable.
        
        Args:
            observable: Which observable to analyze ('energy' or 'magnetization')
            
        Returns:
            Estimated correlation time in MC steps
        """
        if observable == 'energy':
            data = np.array(self.energy_history)
        elif observable == 'magnetization':
            mag_data = np.array(self.magnetization_history)
            data = np.linalg.norm(mag_data, axis=1)
        else:
            raise ValueError("Observable must be 'energy' or 'magnetization'")
        
        if len(data) < 100:
            return np.nan
        
        # Calculate autocorrelation function
        n = len(data)
        data_centered = data - np.mean(data)
        
        # Use FFT for efficient autocorrelation
        f_data = np.fft.fft(data_centered, n=2*n)
        autocorr = np.fft.ifft(f_data * np.conj(f_data)).real
        autocorr = autocorr[:n] / autocorr[0]
        
        # Find where autocorrelation drops to 1/e
        try:
            tau_idx = np.where(autocorr < 1/np.e)[0][0]
            return float(tau_idx)
        except IndexError:
            # If correlation doesn't decay enough, return NaN
            return np.nan
    
    def get_effective_sample_size(self, observable: str = 'energy') -> int:
        """
        Estimate effective sample size accounting for autocorrelation.
        
        Args:
            observable: Which observable to analyze
            
        Returns:
            Effective number of independent samples
        """
        tau = self.estimate_correlation_time(observable)
        if np.isnan(tau):
            return len(self.energy_history)
        
        return max(1, int(len(self.energy_history) / (2 * tau + 1)))
    
    def save_configuration(self, filename: str):
        """Save current spin configuration to file."""
        if self.spin_system.spin_config is None:
            raise ValueError("No spin configuration to save")
        
        np.save(filename, self.spin_system.spin_config)
    
    def load_configuration(self, filename: str):
        """Load spin configuration from file."""
        config = np.load(filename)
        self.spin_system.spin_config = config
        self.current_energy = self.spin_system.calculate_energy()
    
    def reset(self):
        """Reset simulation state."""
        self.current_energy = None
        self.step_count = 0
        self.accepted_moves = 0
        self.energy_history.clear()
        self.magnetization_history.clear()
        self.acceptance_history.clear()
        self.timing_info.clear()
    
    def __repr__(self) -> str:
        return (f"MonteCarlo(T={self.temperature}, "
                f"n_spins={self.spin_system.n_spins}, "
                f"steps={self.step_count})")


class ParallelTempering:
    """
    Parallel tempering (replica exchange) Monte Carlo implementation.
    """
    
    def __init__(
        self,
        spin_system: SpinSystem,
        temperatures: List[float],
        exchange_interval: int = 100,
        random_seed: Optional[int] = None
    ):
        """
        Initialize parallel tempering simulation.
        
        Args:
            spin_system: SpinSystem to simulate
            temperatures: List of temperatures
            exchange_interval: Steps between exchange attempts
            random_seed: Random seed
        """
        self.spin_system = spin_system
        self.temperatures = sorted(temperatures)
        self.exchange_interval = exchange_interval
        self.n_replicas = len(temperatures)
        
        # Create MC simulators for each temperature
        self.mc_sims = []
        for i, temp in enumerate(temperatures):
            seed = random_seed + i if random_seed is not None else None
            mc = MonteCarlo(spin_system, temp, seed)
            self.mc_sims.append(mc)
        
        # Exchange statistics
        self.exchange_attempts = 0
        self.exchange_accepts = 0
    
    def run(
        self,
        n_steps: int,
        equilibration_steps: int = 1000,
        sampling_interval: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run parallel tempering simulation.
        
        Returns:
            Dictionary with results for each temperature
        """
        # Initialize all replicas
        orientations = self.spin_system.generate_spin_orientations()
        
        for mc in self.mc_sims:
            if mc.spin_system.spin_config is None:
                mc.spin_system.random_configuration(orientations)
            mc.current_energy = mc.spin_system.calculate_energy()
        
        pbar = tqdm(total=n_steps, desc="PT Steps", disable=not verbose)
        
        for step in range(n_steps):
            # Perform MC steps for all replicas
            for mc in self.mc_sims:
                mc._perform_sweep(orientations)
            
            # Attempt replica exchanges
            if step % self.exchange_interval == 0:
                self._attempt_exchanges()
            
            # Record data for all replicas
            if step >= equilibration_steps and step % sampling_interval == 0:
                for mc in self.mc_sims:
                    mc.energy_history.append(mc.current_energy)
                    magnetization = mc.spin_system.calculate_magnetization()
                    mc.magnetization_history.append(magnetization.copy())
            
            pbar.update(1)
        
        pbar.close()
        
        # Compile results
        results = {}
        for i, (temp, mc) in enumerate(zip(self.temperatures, self.mc_sims)):
            results[f'T_{temp}'] = {
                'temperature': temp,
                'energies': np.array(mc.energy_history),
                'magnetizations': np.array(mc.magnetization_history),
                'final_energy': mc.current_energy,
                'final_magnetization': mc.spin_system.calculate_magnetization()
            }
        
        results['exchange_rate'] = self.exchange_accepts / max(1, self.exchange_attempts)
        
        return results
    
    def _attempt_exchanges(self):
        """Attempt to exchange configurations between adjacent replicas."""
        for i in range(self.n_replicas - 1):
            self.exchange_attempts += 1
            
            # Calculate exchange probability
            T1, T2 = self.temperatures[i], self.temperatures[i + 1]
            E1 = self.mc_sims[i].current_energy
            E2 = self.mc_sims[i + 1].current_energy
            
            kB = self.mc_sims[0].kB
            delta_beta = 1/(kB * T1) - 1/(kB * T2)
            delta_E = E2 - E1
            
            # Metropolis criterion for exchange
            if delta_beta * delta_E <= 0 or np.random.random() < np.exp(-delta_beta * delta_E):
                # Accept exchange
                self.exchange_accepts += 1
                
                # Swap configurations
                config1 = self.mc_sims[i].spin_system.spin_config.copy()
                config2 = self.mc_sims[i + 1].spin_system.spin_config.copy()
                
                self.mc_sims[i].spin_system.spin_config = config2
                self.mc_sims[i + 1].spin_system.spin_config = config1
                
                # Update energies
                self.mc_sims[i].current_energy = E2
                self.mc_sims[i + 1].current_energy = E1