"""
Correlation analysis tools for spin systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.spatial.distance import pdist, squareform

from ..core.fast_ops import HAS_NUMBA
if HAS_NUMBA:
    from numba import njit, prange
else:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(n):
        return range(n)


class CorrelationAnalyzer:
    """
    Analyze spatial and temporal correlations in spin systems.
    """
    
    def __init__(self, use_fast: bool = True):
        """
        Initialize correlation analyzer.
        
        Args:
            use_fast: Whether to use Numba acceleration
        """
        self.use_fast = use_fast and HAS_NUMBA
    
    def calculate_spatial_correlation(
        self,
        spin_config: np.ndarray,
        positions: np.ndarray,
        max_distance: float = 10.0,
        n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate spatial spin correlation function.
        
        Args:
            spin_config: (n_spins, 3) spin configuration
            positions: (n_spins, 3) atomic positions
            max_distance: Maximum distance for correlation
            n_bins: Number of distance bins
            
        Returns:
            Tuple of (distances, correlations)
        """
        distances = pdist(positions)
        
        # Calculate all pairwise spin correlations
        n_spins = len(spin_config)
        correlations = []
        pair_distances = []
        
        for i in range(n_spins):
            for j in range(i + 1, n_spins):
                # Spin correlation
                correlation = np.dot(spin_config[i], spin_config[j])
                correlations.append(correlation)
                
                # Distance
                distance = np.linalg.norm(positions[i] - positions[j])
                pair_distances.append(distance)
        
        correlations = np.array(correlations)
        pair_distances = np.array(pair_distances)
        
        # Bin by distance
        bins = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        binned_correlations = []
        for i in range(n_bins):
            mask = (pair_distances >= bins[i]) & (pair_distances < bins[i + 1])
            if np.sum(mask) > 0:
                binned_correlations.append(np.mean(correlations[mask]))
            else:
                binned_correlations.append(0.0)
        
        return bin_centers, np.array(binned_correlations)
    
    def calculate_structure_factor(
        self,
        spin_config: np.ndarray,
        positions: np.ndarray,
        lattice_vectors: np.ndarray,
        grid_size: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate static structure factor S(q).
        
        Args:
            spin_config: (n_spins, 3) spin configuration
            positions: (n_spins, 3) atomic positions
            lattice_vectors: (3, 3) lattice vectors
            grid_size: Size of reciprocal space grid
            
        Returns:
            Tuple of (q_points, structure_factor)
        """
        # For 2D systems, project to xy plane
        if positions.shape[1] == 3:
            pos_2d = positions[:, :2]
        else:
            pos_2d = positions
        
        # Get magnetization components
        mx = spin_config[:, 0]
        my = spin_config[:, 1]
        mz = spin_config[:, 2]
        
        # Create real-space grid
        x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
        y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
        
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate spins onto grid (simplified nearest-neighbor)
        mx_grid = np.zeros((grid_size, grid_size))
        my_grid = np.zeros((grid_size, grid_size))
        mz_grid = np.zeros((grid_size, grid_size))
        
        for i, pos in enumerate(pos_2d):
            # Find nearest grid point
            ix = np.argmin(np.abs(x_grid - pos[0]))
            iy = np.argmin(np.abs(y_grid - pos[1]))
            
            mx_grid[iy, ix] += mx[i]
            my_grid[iy, ix] += my[i]
            mz_grid[iy, ix] += mz[i]
        
        # Calculate structure factors
        sx_q = fft2(mx_grid)
        sy_q = fft2(my_grid)
        sz_q = fft2(mz_grid)
        
        # Total structure factor
        S_q = np.abs(sx_q)**2 + np.abs(sy_q)**2 + np.abs(sz_q)**2
        
        # Create q-points
        qx = fftfreq(grid_size, d=(x_max - x_min) / grid_size)
        qy = fftfreq(grid_size, d=(y_max - y_min) / grid_size)
        Qx, Qy = np.meshgrid(qx, qy)
        
        q_magnitude = np.sqrt(Qx**2 + Qy**2)
        
        return q_magnitude, S_q
    
    def calculate_temporal_correlation(
        self,
        spin_trajectories: np.ndarray,
        max_lag: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate temporal autocorrelation function.
        
        Args:
            spin_trajectories: (n_steps, n_spins, 3) time series
            max_lag: Maximum time lag (default: n_steps//4)
            
        Returns:
            Tuple of (time_lags, autocorrelations)
        """
        n_steps = spin_trajectories.shape[0]
        if max_lag is None:
            max_lag = n_steps // 4
        
        # Calculate autocorrelation for each spin component
        autocorrs = []
        
        for component in range(3):  # x, y, z components
            component_data = spin_trajectories[:, :, component]
            
            # Average over all spins
            avg_trajectory = np.mean(component_data, axis=1)
            
            # Calculate autocorrelation
            autocorr = self._autocorrelation(avg_trajectory, max_lag)
            autocorrs.append(autocorr)
        
        # Average over components
        total_autocorr = np.mean(autocorrs, axis=0)
        
        time_lags = np.arange(max_lag + 1)
        
        return time_lags, total_autocorr
    
    def _autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function using FFT."""
        n = len(data)
        
        # Zero-pad for FFT
        padded_data = np.zeros(2 * n)
        padded_data[:n] = data - np.mean(data)
        
        # FFT-based autocorrelation
        fft_data = np.fft.fft(padded_data)
        autocorr_fft = np.fft.ifft(fft_data * np.conj(fft_data)).real
        
        # Extract relevant part and normalize
        autocorr = autocorr_fft[:max_lag + 1]
        autocorr = autocorr / autocorr[0]  # Normalize to 1 at lag 0
        
        return autocorr
    
    def calculate_spin_stiffness(
        self,
        spin_config: np.ndarray,
        positions: np.ndarray,
        neighbor_array: np.ndarray,
        J: float,
        temperature: float
    ) -> float:
        """
        Calculate spin stiffness (helicity modulus).
        
        Args:
            spin_config: (n_spins, 3) spin configuration
            positions: (n_spins, 3) atomic positions
            neighbor_array: (n_spins, max_neighbors) neighbor indices
            J: Exchange coupling
            temperature: Temperature in Kelvin
            
        Returns:
            Spin stiffness value
        """
        kB = 8.617333e-5  # eV/K
        
        # Calculate current energy
        energy_0 = self._calculate_exchange_energy(spin_config, neighbor_array, J)
        
        # Apply small twist in x-direction
        theta = 0.01  # Small twist angle
        twisted_spins = self._apply_twist(spin_config, positions, theta, axis=0)
        energy_twist = self._calculate_exchange_energy(twisted_spins, neighbor_array, J)
        
        # Spin stiffness calculation
        L = np.max(positions[:, 0]) - np.min(positions[:, 0])  # System size
        spin_stiffness = -(energy_twist - energy_0) / (theta**2 * L)
        
        return spin_stiffness
    
    def _calculate_exchange_energy(
        self,
        spins: np.ndarray,
        neighbor_array: np.ndarray,
        J: float
    ) -> float:
        """Calculate exchange energy."""
        energy = 0.0
        
        for i in range(len(spins)):
            for j_idx in range(neighbor_array.shape[1]):
                j = neighbor_array[i, j_idx]
                if j >= 0:
                    energy += -J * np.dot(spins[i], spins[j])
        
        return energy * 0.5  # Avoid double counting
    
    def _apply_twist(
        self,
        spins: np.ndarray,
        positions: np.ndarray,
        theta: float,
        axis: int = 0
    ) -> np.ndarray:
        """Apply a small twist to the spin configuration."""
        twisted_spins = spins.copy()
        
        # Get system size in twist direction
        L = np.max(positions[:, axis]) - np.min(positions[:, axis])
        
        for i in range(len(spins)):
            # Calculate twist angle for this position
            pos = positions[i, axis]
            twist_angle = theta * pos / L
            
            # Apply rotation around z-axis
            cos_twist = np.cos(twist_angle)
            sin_twist = np.sin(twist_angle)
            
            old_x = twisted_spins[i, 0]
            old_y = twisted_spins[i, 1]
            
            twisted_spins[i, 0] = cos_twist * old_x - sin_twist * old_y
            twisted_spins[i, 1] = sin_twist * old_x + cos_twist * old_y
        
        return twisted_spins
    
    def plot_correlations(
        self,
        distances: np.ndarray,
        correlations: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot spatial correlation function."""
        plt.figure(figsize=(8, 6))
        
        plt.plot(distances, correlations, 'o-', linewidth=2, markersize=4)
        plt.xlabel('Distance (Å)')
        plt.ylabel('Spin Correlation ⟨Si·Sj⟩')
        plt.title('Spatial Spin Correlation Function')
        plt.grid(True, alpha=0.3)
        
        # Add exponential fit if correlation decays
        if len(distances) > 5:
            # Simple exponential fit
            try:
                from scipy.optimize import curve_fit
                
                def exp_decay(x, a, xi):
                    return a * np.exp(-x / xi)
                
                # Fit only positive correlations
                mask = correlations > 0.1
                if np.sum(mask) > 3:
                    popt, _ = curve_fit(exp_decay, distances[mask], correlations[mask])
                    
                    x_fit = np.linspace(distances[0], distances[-1], 100)
                    y_fit = exp_decay(x_fit, *popt)
                    
                    plt.plot(x_fit, y_fit, '--', color='red', alpha=0.7,
                            label=f'Fit: ξ = {popt[1]:.2f} Å')
                    plt.legend()
            except:
                pass  # Fit failed
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_structure_factor(
        self,
        q_magnitude: np.ndarray,
        structure_factor: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot structure factor."""
        plt.figure(figsize=(10, 8))
        
        # 2D plot
        plt.imshow(structure_factor, origin='lower', cmap='hot',
                  extent=[q_magnitude.min(), q_magnitude.max(),
                         q_magnitude.min(), q_magnitude.max()])
        plt.colorbar(label='S(q)')
        plt.xlabel('qx (Å⁻¹)')
        plt.ylabel('qy (Å⁻¹)')
        plt.title('Static Structure Factor')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


@njit
def fast_correlation_calculation(spins, positions, max_distance, n_bins):
    """Fast calculation of spatial correlations using Numba."""
    n_spins = spins.shape[0]
    
    # Initialize bins
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_counts = np.zeros(n_bins)
    bin_correlations = np.zeros(n_bins)
    
    # Calculate correlations
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            # Distance
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance <= max_distance:
                # Find bin
                bin_idx = int(distance / max_distance * n_bins)
                if bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                
                # Correlation
                correlation = (spins[i, 0] * spins[j, 0] + 
                             spins[i, 1] * spins[j, 1] + 
                             spins[i, 2] * spins[j, 2])
                
                bin_correlations[bin_idx] += correlation
                bin_counts[bin_idx] += 1
    
    # Average
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_correlations[i] /= bin_counts[i]
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, bin_correlations