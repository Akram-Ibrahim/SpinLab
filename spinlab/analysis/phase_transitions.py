"""
Phase transition analysis tools.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

from ..utils.constants import PHYSICAL_CONSTANTS


class PhaseTransitionAnalyzer:
    """
    Analyze phase transitions and critical phenomena in spin systems.
    """
    
    def __init__(self):
        """Initialize phase transition analyzer."""
        self.kB = PHYSICAL_CONSTANTS['kB']
    
    def critical_scaling_analysis(
        self,
        temperatures: np.ndarray,
        order_parameter: np.ndarray,
        susceptibility: np.ndarray,
        specific_heat: np.ndarray,
        Tc_guess: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform critical scaling analysis to extract critical exponents.
        
        Args:
            temperatures: Temperature array
            order_parameter: Order parameter (e.g., magnetization)
            susceptibility: Susceptibility values
            specific_heat: Specific heat values
            Tc_guess: Initial guess for critical temperature
            
        Returns:
            Dictionary with critical exponents and analysis results
        """
        if Tc_guess is None:
            # Find Tc from susceptibility maximum
            Tc_guess = temperatures[np.argmax(susceptibility)]
        
        results = {'Tc_guess': Tc_guess}
        
        # Define reduced temperature
        t = np.abs(temperatures - Tc_guess) / Tc_guess
        
        # Critical exponent β (order parameter)
        beta_result = self._fit_critical_exponent(
            t, order_parameter, temperatures < Tc_guess, 'beta'
        )
        results['beta'] = beta_result
        
        # Critical exponent γ (susceptibility)
        gamma_result = self._fit_critical_exponent(
            t, susceptibility, np.ones(len(t), dtype=bool), 'gamma', inverse=True
        )
        results['gamma'] = gamma_result
        
        # Critical exponent α (specific heat)
        alpha_result = self._fit_critical_exponent(
            t, specific_heat, np.ones(len(t), dtype=bool), 'alpha', inverse=True
        )
        results['alpha'] = alpha_result
        
        return results
    
    def _fit_critical_exponent(
        self,
        t: np.ndarray,
        observable: np.ndarray,
        mask: np.ndarray,
        exponent_name: str,
        inverse: bool = False
    ) -> Dict[str, Any]:
        """
        Fit critical exponent for an observable.
        
        Args:
            t: Reduced temperature array
            observable: Observable values
            mask: Mask for which data points to use
            exponent_name: Name of the exponent
            inverse: Whether observable ~ t^(-exponent)
            
        Returns:
            Dictionary with fit results
        """
        # Filter data
        t_fit = t[mask]
        obs_fit = observable[mask]
        
        # Remove zeros and negative values for log fit
        valid = (t_fit > 0) & (obs_fit > 0) & np.isfinite(obs_fit)
        t_fit = t_fit[valid]
        obs_fit = obs_fit[valid]
        
        if len(t_fit) < 3:
            return {'exponent': np.nan, 'amplitude': np.nan, 'fit_quality': 0}
        
        try:
            # Power law fit: observable = A * t^exponent
            log_t = np.log(t_fit)
            log_obs = np.log(obs_fit)
            
            # Linear fit in log space
            coeffs = np.polyfit(log_t, log_obs, 1)
            exponent = coeffs[0]
            log_amplitude = coeffs[1]
            amplitude = np.exp(log_amplitude)
            
            if inverse:
                exponent = -exponent
            
            # Calculate fit quality (R²)
            log_obs_pred = coeffs[0] * log_t + coeffs[1]
            ss_res = np.sum((log_obs - log_obs_pred) ** 2)
            ss_tot = np.sum((log_obs - np.mean(log_obs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'exponent': exponent,
                'amplitude': amplitude,
                'fit_quality': r_squared,
                't_range': (t_fit.min(), t_fit.max()),
                'n_points': len(t_fit)
            }
        
        except Exception as e:
            return {'exponent': np.nan, 'amplitude': np.nan, 'fit_quality': 0, 'error': str(e)}
    
    def finite_size_scaling(
        self,
        system_sizes: List[int],
        temperatures: np.ndarray,
        observables_by_size: Dict[int, Dict[str, np.ndarray]],
        observable_name: str = 'susceptibility'
    ) -> Dict[str, Any]:
        """
        Perform finite-size scaling analysis.
        
        Args:
            system_sizes: List of system sizes
            temperatures: Temperature array
            observables_by_size: Observables for each system size
            observable_name: Which observable to analyze
            
        Returns:
            Dictionary with finite-size scaling results
        """
        results = {
            'system_sizes': system_sizes,
            'observable_name': observable_name,
            'Tc_by_size': [],
            'max_values': [],
            'scaling_analysis': {}
        }
        
        # Find critical temperatures and maximum values for each size
        for size in system_sizes:
            observable = observables_by_size[size][observable_name]
            max_idx = np.argmax(observable)
            
            Tc_size = temperatures[max_idx]
            max_value = observable[max_idx]
            
            results['Tc_by_size'].append(Tc_size)
            results['max_values'].append(max_value)
        
        results['Tc_by_size'] = np.array(results['Tc_by_size'])
        results['max_values'] = np.array(results['max_values'])
        
        # Finite-size scaling fits
        sizes = np.array(system_sizes)
        
        # Tc(L) = Tc(∞) + A*L^(-1/ν)
        try:
            def tc_scaling(L, Tc_inf, A, nu):
                return Tc_inf + A * L**(-1/nu)
            
            popt_tc, _ = curve_fit(tc_scaling, sizes, results['Tc_by_size'])
            results['scaling_analysis']['Tc_infinite'] = popt_tc[0]
            results['scaling_analysis']['nu'] = popt_tc[2]
        except:
            results['scaling_analysis']['Tc_infinite'] = np.nan
            results['scaling_analysis']['nu'] = np.nan
        
        # Observable_max(L) ~ L^(γ/ν) or similar
        try:
            log_sizes = np.log(sizes)
            log_max_vals = np.log(results['max_values'])
            
            coeffs = np.polyfit(log_sizes, log_max_vals, 1)
            results['scaling_analysis']['max_value_exponent'] = coeffs[0]
        except:
            results['scaling_analysis']['max_value_exponent'] = np.nan
        
        return results
    
    def data_collapse(
        self,
        temperatures: np.ndarray,
        observables_by_size: Dict[int, Dict[str, np.ndarray]],
        Tc: float,
        nu: float,
        observable_name: str = 'susceptibility',
        scaling_dimension: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform data collapse analysis.
        
        Args:
            temperatures: Temperature array
            observables_by_size: Observables for each system size
            Tc: Critical temperature
            nu: Critical exponent ν
            observable_name: Observable to collapse
            scaling_dimension: Scaling dimension of observable
            
        Returns:
            Dictionary with collapsed data
        """
        system_sizes = list(observables_by_size.keys())
        
        collapsed_data = {
            'scaled_temperature': [],
            'scaled_observable': [],
            'system_sizes': []
        }
        
        for size in system_sizes:
            observable = observables_by_size[size][observable_name]
            
            # Scaling variable: (T - Tc) * L^(1/ν)
            scaled_t = (temperatures - Tc) * (size ** (1/nu))
            
            # Scale observable if scaling dimension provided
            if scaling_dimension is not None:
                scaled_obs = observable / (size ** scaling_dimension)
            else:
                scaled_obs = observable
            
            collapsed_data['scaled_temperature'].extend(scaled_t)
            collapsed_data['scaled_observable'].extend(scaled_obs)
            collapsed_data['system_sizes'].extend([size] * len(scaled_t))
        
        return collapsed_data
    
    def binder_cumulant_analysis(
        self,
        system_sizes: List[int],
        binder_cumulants_by_size: Dict[int, np.ndarray],
        temperatures: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze Binder cumulant crossing to find critical temperature.
        
        Args:
            system_sizes: List of system sizes
            binder_cumulants_by_size: Binder cumulants for each size
            temperatures: Temperature array
            
        Returns:
            Dictionary with Binder analysis results
        """
        results = {
            'system_sizes': system_sizes,
            'crossing_temperatures': [],
            'universal_value': None
        }
        
        # Find crossing points between consecutive sizes
        for i in range(len(system_sizes) - 1):
            size1, size2 = system_sizes[i], system_sizes[i + 1]
            
            U1 = binder_cumulants_by_size[size1]
            U2 = binder_cumulants_by_size[size2]
            
            # Find crossing point
            diff = U1 - U2
            
            # Look for sign change
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            
            if len(sign_changes) > 0:
                # Use first crossing (there might be multiple)
                crossing_idx = sign_changes[0]
                
                # Linear interpolation for more precise crossing
                t1, t2 = temperatures[crossing_idx], temperatures[crossing_idx + 1]
                d1, d2 = diff[crossing_idx], diff[crossing_idx + 1]
                
                if d2 != d1:
                    crossing_temp = t1 - d1 * (t2 - t1) / (d2 - d1)
                else:
                    crossing_temp = (t1 + t2) / 2
                
                results['crossing_temperatures'].append(crossing_temp)
        
        # Estimate universal value at crossing
        if results['crossing_temperatures']:
            avg_crossing_temp = np.mean(results['crossing_temperatures'])
            
            # Interpolate to find universal value
            universal_values = []
            for size in system_sizes:
                U = binder_cumulants_by_size[size]
                U_at_crossing = np.interp(avg_crossing_temp, temperatures, U)
                universal_values.append(U_at_crossing)
            
            results['universal_value'] = np.mean(universal_values)
            results['average_crossing_temperature'] = avg_crossing_temp
        
        return results
    
    def plot_critical_scaling(
        self,
        scaling_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot critical scaling analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # β exponent (order parameter)
        if 'beta' in scaling_results:
            beta_data = scaling_results['beta']
            if not np.isnan(beta_data['exponent']):
                axes[0].set_title(f"Order Parameter\nβ = {beta_data['exponent']:.3f}")
        
        # γ exponent (susceptibility)
        if 'gamma' in scaling_results:
            gamma_data = scaling_results['gamma']
            if not np.isnan(gamma_data['exponent']):
                axes[1].set_title(f"Susceptibility\nγ = {gamma_data['exponent']:.3f}")
        
        # α exponent (specific heat)
        if 'alpha' in scaling_results:
            alpha_data = scaling_results['alpha']
            if not np.isnan(alpha_data['exponent']):
                axes[2].set_title(f"Specific Heat\nα = {alpha_data['exponent']:.3f}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_finite_size_scaling(
        self,
        fss_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot finite-size scaling analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sizes = np.array(fss_results['system_sizes'])
        
        # Tc vs system size
        axes[0].plot(1/sizes, fss_results['Tc_by_size'], 'o-')
        axes[0].set_xlabel('1/L')
        axes[0].set_ylabel('Tc(L)')
        axes[0].set_title('Critical Temperature vs System Size')
        axes[0].grid(True, alpha=0.3)
        
        # Maximum values vs system size (log-log)
        axes[1].loglog(sizes, fss_results['max_values'], 's-')
        axes[1].set_xlabel('System Size L')
        axes[1].set_ylabel(f"Max {fss_results['observable_name']}")
        axes[1].set_title('Observable Maximum vs System Size')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_data_collapse(
        self,
        collapse_data: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Plot data collapse."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot data for each system size with different colors
        unique_sizes = list(set(collapse_data['system_sizes']))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sizes)))
        
        for size, color in zip(unique_sizes, colors):
            mask = np.array(collapse_data['system_sizes']) == size
            
            scaled_t = np.array(collapse_data['scaled_temperature'])[mask]
            scaled_obs = np.array(collapse_data['scaled_observable'])[mask]
            
            ax.plot(scaled_t, scaled_obs, 'o', color=color, 
                   label=f'L = {size}', alpha=0.7, markersize=4)
        
        ax.set_xlabel('Scaled Temperature (T - Tc)L^(1/ν)')
        ax.set_ylabel('Scaled Observable')
        ax.set_title('Data Collapse Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()