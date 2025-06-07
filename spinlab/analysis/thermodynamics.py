"""
Thermodynamic analysis tools for spin systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from ..utils.constants import PHYSICAL_CONSTANTS


class ThermodynamicsAnalyzer:
    """
    Comprehensive thermodynamic analysis for spin systems.
    
    Provides tools for calculating thermodynamic properties, phase transitions,
    critical phenomena, and finite-size scaling analysis.
    """
    
    def __init__(self):
        """Initialize thermodynamics analyzer."""
        self.kB = PHYSICAL_CONSTANTS['kB']
        
        # Data storage
        self.temperatures = []
        self.energies = []
        self.magnetizations = []
        self.specific_heats = []
        self.susceptibilities = []
        self.binder_cumulants = []
        
        # Analysis results
        self.critical_temperature = None
        self.critical_exponents = {}
        self.phase_boundaries = {}
    
    def add_data_point(
        self,
        temperature: float,
        energy_data: np.ndarray,
        magnetization_data: np.ndarray,
        n_spins: int
    ):
        """
        Add thermodynamic data for a specific temperature.
        
        Args:
            temperature: Temperature in Kelvin
            energy_data: Array of energy samples
            magnetization_data: Array of magnetization vectors
            n_spins: Number of spins in the system
        """
        self.temperatures.append(temperature)
        
        # Calculate mean energy per spin
        mean_energy = np.mean(energy_data) / n_spins
        self.energies.append(mean_energy)
        
        # Calculate magnetization magnitude
        if magnetization_data.ndim == 2:
            mag_magnitudes = np.linalg.norm(magnetization_data, axis=1)
        else:
            mag_magnitudes = np.abs(magnetization_data)
        
        mean_magnetization = np.mean(mag_magnitudes)
        self.magnetizations.append(mean_magnetization)
        
        # Calculate specific heat
        energy_variance = np.var(energy_data)
        specific_heat = energy_variance / (n_spins * (self.kB * temperature)**2)
        self.specific_heats.append(specific_heat)
        
        # Calculate magnetic susceptibility
        mag_variance = np.var(mag_magnitudes)
        susceptibility = mag_variance / (self.kB * temperature * n_spins)
        self.susceptibilities.append(susceptibility)
        
        # Calculate Binder cumulant
        mag_squared = mag_magnitudes**2
        mag_fourth = mag_magnitudes**4
        mean_mag_squared = np.mean(mag_squared)
        mean_mag_fourth = np.mean(mag_fourth)
        
        if mean_mag_squared > 0:
            binder_cumulant = 1 - mean_mag_fourth / (3 * mean_mag_squared**2)
        else:
            binder_cumulant = 0
        
        self.binder_cumulants.append(binder_cumulant)
    
    def calculate_thermodynamic_properties(
        self,
        temperatures: np.ndarray,
        simulation_results: List[Dict[str, Any]],
        n_spins: int
    ) -> Dict[str, np.ndarray]:
        """
        Calculate thermodynamic properties from simulation results.
        
        Args:
            temperatures: Array of temperatures
            simulation_results: List of simulation result dictionaries
            n_spins: Number of spins
            
        Returns:
            Dictionary with thermodynamic properties
        """
        # Clear existing data
        self.temperatures.clear()
        self.energies.clear()
        self.magnetizations.clear()
        self.specific_heats.clear()
        self.susceptibilities.clear()
        self.binder_cumulants.clear()
        
        # Process each temperature
        for temp, result in zip(temperatures, simulation_results):
            energy_data = result['energies']
            magnetization_data = result['magnetizations']
            
            self.add_data_point(temp, energy_data, magnetization_data, n_spins)
        
        # Convert to numpy arrays for easier handling
        properties = {
            'temperatures': np.array(self.temperatures),
            'energies': np.array(self.energies),
            'magnetizations': np.array(self.magnetizations),
            'specific_heats': np.array(self.specific_heats),
            'susceptibilities': np.array(self.susceptibilities),
            'binder_cumulants': np.array(self.binder_cumulants)
        }
        
        return properties
    
    def find_critical_temperature(
        self,
        method: str = "specific_heat",
        smoothing: float = 0.1
    ) -> Dict[str, Any]:
        """
        Find critical temperature using various methods.
        
        Args:
            method: Method to use ("specific_heat", "susceptibility", "binder")
            smoothing: Smoothing parameter for spline fitting
            
        Returns:
            Dictionary with critical temperature analysis
        """
        if len(self.temperatures) == 0:
            raise ValueError("No thermodynamic data available")
        
        temps = np.array(self.temperatures)
        
        if method == "specific_heat":
            data = np.array(self.specific_heats)
        elif method == "susceptibility":
            data = np.array(self.susceptibilities)
        elif method == "binder":
            data = np.array(self.binder_cumulants)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fit spline for smoothing
        spline = UnivariateSpline(temps, data, s=smoothing)
        
        # Find maximum
        if method in ["specific_heat", "susceptibility"]:
            # Find maximum
            result = minimize_scalar(
                lambda t: -spline(t),
                bounds=(temps.min(), temps.max()),
                method='bounded'
            )
            Tc = result.x
            max_value = -result.fun
        else:  # Binder cumulant
            # Find where Binder cumulant crosses universal value
            universal_value = 2/3  # 3D Ising universal value
            
            # Find crossing point
            temp_fine = np.linspace(temps.min(), temps.max(), 1000)
            binder_fine = spline(temp_fine)
            
            # Find closest point to universal value
            idx = np.argmin(np.abs(binder_fine - universal_value))
            Tc = temp_fine[idx]
            max_value = binder_fine[idx]
        
        self.critical_temperature = Tc
        
        return {
            'critical_temperature': Tc,
            'method': method,
            'max_value': max_value,
            'spline_fit': spline
        }
    
    def calculate_critical_exponents(
        self,
        Tc: Optional[float] = None,
        fit_range: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate critical exponents using power law fits.
        
        Args:
            Tc: Critical temperature (auto-detect if None)
            fit_range: Relative temperature range for fitting
            
        Returns:
            Dictionary with critical exponents
        """
        if Tc is None:
            if self.critical_temperature is None:
                self.find_critical_temperature()
            Tc = self.critical_temperature
        
        temps = np.array(self.temperatures)
        
        # Reduced temperature
        t = np.abs(temps - Tc) / Tc
        
        # Fit range
        mask = (t > 0.01) & (t < fit_range)
        
        if np.sum(mask) < 3:
            raise ValueError("Not enough data points for critical exponent fitting")
        
        t_fit = t[mask]
        
        exponents = {}
        
        # α (specific heat): C ~ t^(-α)
        C_fit = np.array(self.specific_heats)[mask]
        if len(C_fit) > 2:
            # Log-linear fit
            log_t = np.log(t_fit)
            log_C = np.log(C_fit)
            
            # Remove infinite values
            finite_mask = np.isfinite(log_t) & np.isfinite(log_C)
            if np.sum(finite_mask) > 2:
                coeffs = np.polyfit(log_t[finite_mask], log_C[finite_mask], 1)
                exponents['alpha'] = -coeffs[0]
        
        # β (magnetization): M ~ t^β (below Tc)
        below_Tc = (temps < Tc) & mask
        if np.sum(below_Tc) > 2:
            t_below = t[below_Tc]
            M_below = np.array(self.magnetizations)[below_Tc]
            
            log_t = np.log(t_below)
            log_M = np.log(M_below + 1e-10)  # Avoid log(0)
            
            finite_mask = np.isfinite(log_t) & np.isfinite(log_M)
            if np.sum(finite_mask) > 2:
                coeffs = np.polyfit(log_t[finite_mask], log_M[finite_mask], 1)
                exponents['beta'] = coeffs[0]
        
        # γ (susceptibility): χ ~ t^(-γ)
        chi_fit = np.array(self.susceptibilities)[mask]
        if len(chi_fit) > 2:
            log_t = np.log(t_fit)
            log_chi = np.log(chi_fit)
            
            finite_mask = np.isfinite(log_t) & np.isfinite(log_chi)
            if np.sum(finite_mask) > 2:
                coeffs = np.polyfit(log_t[finite_mask], log_chi[finite_mask], 1)
                exponents['gamma'] = -coeffs[0]
        
        self.critical_exponents = exponents
        return exponents
    
    def finite_size_scaling(
        self,
        system_sizes: List[int],
        properties_by_size: Dict[int, Dict[str, np.ndarray]],
        observable: str = "susceptibility"
    ) -> Dict[str, Any]:
        """
        Perform finite-size scaling analysis.
        
        Args:
            system_sizes: List of system sizes (number of spins)
            properties_by_size: Thermodynamic properties for each size
            observable: Observable to analyze
            
        Returns:
            Dictionary with finite-size scaling results
        """
        # Find critical temperatures for each size
        Tc_list = []
        max_values = []
        
        for size in system_sizes:
            props = properties_by_size[size]
            temps = props['temperatures']
            
            if observable == "susceptibility":
                data = props['susceptibilities']
            elif observable == "specific_heat":
                data = props['specific_heats']
            else:
                raise ValueError(f"Unknown observable: {observable}")
            
            # Find maximum
            max_idx = np.argmax(data)
            Tc_list.append(temps[max_idx])
            max_values.append(data[max_idx])
        
        Tc_list = np.array(Tc_list)
        max_values = np.array(max_values)
        sizes = np.array(system_sizes)
        
        # Finite-size scaling: Tc(L) = Tc(∞) + A*L^(-1/ν)
        # where ν is the correlation length exponent
        
        # Fit to extract Tc(∞) and ν
        def scaling_func(params):
            Tc_inf, A, nu = params
            Tc_predicted = Tc_inf + A * sizes**(-1/nu)
            return np.sum((Tc_list - Tc_predicted)**2)
        
        from scipy.optimize import minimize
        
        # Initial guess
        initial_guess = [np.mean(Tc_list), 1.0, 1.0]
        
        result = minimize(scaling_func, initial_guess, method='Nelder-Mead')
        
        if result.success:
            Tc_inf, A, nu = result.x
        else:
            Tc_inf, A, nu = np.nan, np.nan, np.nan
        
        # Maximum value scaling: χ_max ~ L^(γ/ν)
        log_L = np.log(sizes)
        log_max = np.log(max_values)
        
        if len(log_L) > 1:
            coeffs = np.polyfit(log_L, log_max, 1)
            gamma_over_nu = coeffs[0]
        else:
            gamma_over_nu = np.nan
        
        return {
            'Tc_infinite': Tc_inf,
            'nu': nu,
            'gamma_over_nu': gamma_over_nu,
            'system_sizes': sizes,
            'Tc_by_size': Tc_list,
            'max_values': max_values,
            'fit_quality': result.fun if result.success else np.inf
        }
    
    def plot_thermodynamic_properties(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create comprehensive thermodynamic property plots.
        
        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if len(self.temperatures) == 0:
            raise ValueError("No thermodynamic data to plot")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        temps = np.array(self.temperatures)
        
        # Energy vs Temperature
        axes[0, 0].plot(temps, self.energies, 'o-', color='blue')
        axes[0, 0].set_xlabel('Temperature (K)')
        axes[0, 0].set_ylabel('Energy per spin (eV)')
        axes[0, 0].set_title('Internal Energy')
        axes[0, 0].grid(True)
        
        # Magnetization vs Temperature
        axes[0, 1].plot(temps, self.magnetizations, 'o-', color='red')
        axes[0, 1].set_xlabel('Temperature (K)')
        axes[0, 1].set_ylabel('Magnetization')
        axes[0, 1].set_title('Magnetization')
        axes[0, 1].grid(True)
        
        # Specific Heat vs Temperature
        axes[1, 0].plot(temps, self.specific_heats, 'o-', color='green')
        axes[1, 0].set_xlabel('Temperature (K)')
        axes[1, 0].set_ylabel('Specific Heat / kB')
        axes[1, 0].set_title('Specific Heat')
        axes[1, 0].grid(True)
        
        # Mark critical temperature if available
        if self.critical_temperature is not None:
            axes[1, 0].axvline(self.critical_temperature, color='black', 
                              linestyle='--', alpha=0.7, label=f'Tc = {self.critical_temperature:.1f} K')
            axes[1, 0].legend()
        
        # Susceptibility vs Temperature
        axes[1, 1].plot(temps, self.susceptibilities, 'o-', color='purple')
        axes[1, 1].set_xlabel('Temperature (K)')
        axes[1, 1].set_ylabel('Susceptibility / kB')
        axes[1, 1].set_title('Magnetic Susceptibility')
        axes[1, 1].grid(True)
        
        # Mark critical temperature if available
        if self.critical_temperature is not None:
            axes[1, 1].axvline(self.critical_temperature, color='black', 
                              linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_data(self, filename: str):
        """Export thermodynamic data to file."""
        data = {
            'temperatures': np.array(self.temperatures),
            'energies': np.array(self.energies),
            'magnetizations': np.array(self.magnetizations),
            'specific_heats': np.array(self.specific_heats),
            'susceptibilities': np.array(self.susceptibilities),
            'binder_cumulants': np.array(self.binder_cumulants),
            'critical_temperature': self.critical_temperature,
            'critical_exponents': self.critical_exponents
        }
        
        np.savez(filename, **data)
    
    def import_data(self, filename: str):
        """Import thermodynamic data from file."""
        data = np.load(filename, allow_pickle=True)
        
        self.temperatures = data['temperatures'].tolist()
        self.energies = data['energies'].tolist()
        self.magnetizations = data['magnetizations'].tolist()
        self.specific_heats = data['specific_heats'].tolist()
        self.susceptibilities = data['susceptibilities'].tolist()
        self.binder_cumulants = data['binder_cumulants'].tolist()
        
        if 'critical_temperature' in data:
            self.critical_temperature = float(data['critical_temperature'])
        
        if 'critical_exponents' in data:
            self.critical_exponents = data['critical_exponents'].item()
    
    def analyze_phase_diagram(
        self,
        parameter_grid: Dict[str, np.ndarray],
        simulation_results: Dict[Tuple, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze phase diagram in parameter space.
        
        Args:
            parameter_grid: Dictionary with parameter arrays
            simulation_results: Results indexed by parameter tuples
            
        Returns:
            Dictionary with phase diagram analysis
        """
        # This is a framework for phase diagram analysis
        # Implementation would depend on specific parameters being varied
        
        phase_diagram = {}
        
        # Extract parameter names and values
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        if len(param_names) == 2:
            # 2D phase diagram
            X, Y = np.meshgrid(param_values[0], param_values[1])
            
            # Calculate order parameter across parameter space
            order_parameter = np.zeros_like(X)
            
            for i, x_val in enumerate(param_values[0]):
                for j, y_val in enumerate(param_values[1]):
                    key = (x_val, y_val)
                    if key in simulation_results:
                        result = simulation_results[key]
                        # Use magnetization as order parameter
                        mag_data = result['magnetizations']
                        if mag_data.ndim == 2:
                            order_parameter[j, i] = np.mean(np.linalg.norm(mag_data, axis=1))
                        else:
                            order_parameter[j, i] = np.mean(np.abs(mag_data))
            
            phase_diagram = {
                'parameter_names': param_names,
                'X': X,
                'Y': Y,
                'order_parameter': order_parameter
            }
        
        return phase_diagram
    
    def __repr__(self) -> str:
        n_temps = len(self.temperatures)
        Tc_str = f", Tc={self.critical_temperature:.2f}K" if self.critical_temperature else ""
        return f"ThermodynamicsAnalyzer(n_temperatures={n_temps}{Tc_str})"