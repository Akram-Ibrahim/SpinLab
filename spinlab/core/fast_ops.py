"""
High-performance Numba-accelerated operations for spin systems.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from numba import njit, prange, types
    from numba.typed import Dict as NumbaDict
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(n):
        return range(n)

# Physical constants
KB_EV_K = 8.617333e-5  # Boltzmann constant in eV/K
MU_B_EV_T = 5.78838e-5  # Bohr magneton in eV/T


@njit(parallel=True, fastmath=True)
def fast_exchange_energy(spins, neighbor_array, J):
    """
    Fast exchange energy calculation using Numba.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        neighbor_array: (n_spins, max_neighbors) array of neighbor indices
        J: Exchange coupling constant
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    max_neighbors = neighbor_array.shape[1]
    energies = np.zeros(n_spins)
    
    for i in prange(n_spins):
        energy = 0.0
        for j_idx in range(max_neighbors):
            j = neighbor_array[i, j_idx]
            if j >= 0 and j < n_spins:  # Valid neighbor
                # Manual dot product for numba
                dot_product = (spins[i, 0] * spins[j, 0] + 
                             spins[i, 1] * spins[j, 1] + 
                             spins[i, 2] * spins[j, 2])
                energy += -J * dot_product
        energies[i] = energy * 0.5  # Avoid double counting
    
    return energies


@njit(parallel=True, fastmath=True)
def fast_anisotropic_exchange_energy(spins, neighbor_array, coupling_matrix):
    """
    Fast anisotropic exchange energy calculation.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        neighbor_array: (n_spins, max_neighbors) array of neighbor indices
        coupling_matrix: (3, 3) coupling matrix
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    max_neighbors = neighbor_array.shape[1]
    energies = np.zeros(n_spins)
    
    for i in prange(n_spins):
        energy = 0.0
        for j_idx in range(max_neighbors):
            j = neighbor_array[i, j_idx]
            if j >= 0 and j < n_spins:
                # Calculate Si · J · Sj
                temp = np.zeros(3)
                for k in range(3):
                    for l in range(3):
                        temp[k] += coupling_matrix[k, l] * spins[j, l]
                
                dot_product = (spins[i, 0] * temp[0] + 
                             spins[i, 1] * temp[1] + 
                             spins[i, 2] * temp[2])
                energy += -dot_product
        
        energies[i] = energy * 0.5
    
    return energies


@njit(parallel=True, fastmath=True)
def fast_single_ion_anisotropy_energy(spins, K, axis):
    """
    Fast single-ion anisotropy energy calculation.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        K: Anisotropy constant
        axis: (3,) easy axis vector
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    energies = np.zeros(n_spins)
    
    for i in prange(n_spins):
        dot_product = (spins[i, 0] * axis[0] + 
                      spins[i, 1] * axis[1] + 
                      spins[i, 2] * axis[2])
        energies[i] = -K * dot_product * dot_product
    
    return energies


@njit(parallel=True, fastmath=True)
def fast_zeeman_energy(spins, B_field, g_factor):
    """
    Fast Zeeman energy calculation.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        B_field: (3,) magnetic field vector in Tesla
        g_factor: Landé g-factor
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    energies = np.zeros(n_spins)
    
    factor = -g_factor * MU_B_EV_T
    
    for i in prange(n_spins):
        dot_product = (spins[i, 0] * B_field[0] + 
                      spins[i, 1] * B_field[1] + 
                      spins[i, 2] * B_field[2])
        energies[i] = factor * dot_product
    
    return energies


@njit(parallel=True, fastmath=True)
def fast_dmi_energy(spins, neighbor_array, D_vector):
    """
    Fast DMI energy calculation.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        neighbor_array: (n_spins, max_neighbors) array of neighbor indices
        D_vector: (3,) DM vector
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    max_neighbors = neighbor_array.shape[1]
    energies = np.zeros(n_spins)
    
    for i in prange(n_spins):
        energy = 0.0
        for j_idx in range(max_neighbors):
            j = neighbor_array[i, j_idx]
            if j >= 0 and j < n_spins:
                # Cross product Si × Sj
                cross_x = spins[i, 1] * spins[j, 2] - spins[i, 2] * spins[j, 1]
                cross_y = spins[i, 2] * spins[j, 0] - spins[i, 0] * spins[j, 2]
                cross_z = spins[i, 0] * spins[j, 1] - spins[i, 1] * spins[j, 0]
                
                # D · (Si × Sj)
                dot_product = (D_vector[0] * cross_x + 
                             D_vector[1] * cross_y + 
                             D_vector[2] * cross_z)
                energy += dot_product
        
        energies[i] = energy * 0.5
    
    return energies


@njit(parallel=True, fastmath=True)
def fast_effective_field(spins, neighbor_array, J):
    """
    Fast effective field calculation for exchange interaction.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        neighbor_array: (n_spins, max_neighbors) array of neighbor indices
        J: Exchange coupling constant
        
    Returns:
        (n_spins, 3) array of effective fields
    """
    n_spins = spins.shape[0]
    max_neighbors = neighbor_array.shape[1]
    fields = np.zeros((n_spins, 3))
    
    for i in prange(n_spins):
        field_x = 0.0
        field_y = 0.0
        field_z = 0.0
        
        for j_idx in range(max_neighbors):
            j = neighbor_array[i, j_idx]
            if j >= 0 and j < n_spins:
                field_x += J * spins[j, 0]
                field_y += J * spins[j, 1]
                field_z += J * spins[j, 2]
        
        fields[i, 0] = field_x
        fields[i, 1] = field_y
        fields[i, 2] = field_z
    
    return fields


@njit(fastmath=True)
def fast_spin_cross_product(spin, field):
    """
    Fast cross product for single spin and field.
    
    Args:
        spin: (3,) spin vector
        field: (3,) field vector
        
    Returns:
        (3,) cross product spin × field
    """
    result = np.zeros(3)
    result[0] = spin[1] * field[2] - spin[2] * field[1]
    result[1] = spin[2] * field[0] - spin[0] * field[2]
    result[2] = spin[0] * field[1] - spin[1] * field[0]
    return result


@njit(parallel=True, fastmath=True)
def fast_llg_rhs(spins, effective_fields, gamma, alpha):
    """
    Fast calculation of LLG equation right-hand side.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        effective_fields: (n_spins, 3) array of effective fields
        gamma: Gyromagnetic ratio
        alpha: Gilbert damping parameter
        
    Returns:
        (n_spins, 3) array of time derivatives
    """
    n_spins = spins.shape[0]
    derivatives = np.zeros((n_spins, 3))
    
    for i in prange(n_spins):
        # S × H_eff
        cross_SH_x = spins[i, 1] * effective_fields[i, 2] - spins[i, 2] * effective_fields[i, 1]
        cross_SH_y = spins[i, 2] * effective_fields[i, 0] - spins[i, 0] * effective_fields[i, 2]
        cross_SH_z = spins[i, 0] * effective_fields[i, 1] - spins[i, 1] * effective_fields[i, 0]
        
        # S × (S × H_eff)
        cross_S_cross_SH_x = (spins[i, 1] * cross_SH_z - spins[i, 2] * cross_SH_y)
        cross_S_cross_SH_y = (spins[i, 2] * cross_SH_x - spins[i, 0] * cross_SH_z)
        cross_S_cross_SH_z = (spins[i, 0] * cross_SH_y - spins[i, 1] * cross_SH_x)
        
        # |S|²
        spin_mag_sq = spins[i, 0]**2 + spins[i, 1]**2 + spins[i, 2]**2
        
        # LLG equation: dS/dt = -γ(S × H) + α(S × (S × H))/|S|²
        derivatives[i, 0] = -gamma * cross_SH_x + alpha * cross_S_cross_SH_x / spin_mag_sq
        derivatives[i, 1] = -gamma * cross_SH_y + alpha * cross_S_cross_SH_y / spin_mag_sq
        derivatives[i, 2] = -gamma * cross_SH_z + alpha * cross_S_cross_SH_z / spin_mag_sq
    
    return derivatives


@njit(parallel=True, fastmath=True)
def fast_normalize_spins(spins, target_magnitude):
    """
    Fast spin normalization to maintain constant magnitude.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        target_magnitude: Target magnitude for each spin
        
    Returns:
        Normalized spin array
    """
    n_spins = spins.shape[0]
    normalized = np.zeros_like(spins)
    
    for i in prange(n_spins):
        magnitude = np.sqrt(spins[i, 0]**2 + spins[i, 1]**2 + spins[i, 2]**2)
        if magnitude > 0:
            factor = target_magnitude / magnitude
            normalized[i, 0] = spins[i, 0] * factor
            normalized[i, 1] = spins[i, 1] * factor
            normalized[i, 2] = spins[i, 2] * factor
        else:
            # If magnitude is zero, set to z-direction
            normalized[i, 0] = 0.0
            normalized[i, 1] = 0.0
            normalized[i, 2] = target_magnitude
    
    return normalized


@njit(fastmath=True)
def fast_metropolis_single_flip(
    spins, 
    neighbor_array, 
    orientations, 
    site_idx, 
    J, 
    temperature,
    spin_magnitude
):
    """
    Fast single spin flip attempt using Metropolis criterion.
    
    Args:
        spins: (n_spins, 3) spin configuration
        neighbor_array: (n_spins, max_neighbors) neighbor indices
        orientations: (n_orientations, 2) allowed (theta, phi) orientations in radians
        site_idx: Index of spin to flip
        J: Exchange coupling
        temperature: Temperature in Kelvin
        spin_magnitude: Magnitude of spins
        
    Returns:
        (accepted, energy_change) tuple
    """
    # Store original spin
    orig_spin = np.array([spins[site_idx, 0], spins[site_idx, 1], spins[site_idx, 2]])
    
    # Calculate original energy contribution
    orig_energy = 0.0
    max_neighbors = neighbor_array.shape[1]
    
    for j_idx in range(max_neighbors):
        j = neighbor_array[site_idx, j_idx]
        if j >= 0 and j < spins.shape[0]:
            dot_product = (orig_spin[0] * spins[j, 0] + 
                         orig_spin[1] * spins[j, 1] + 
                         orig_spin[2] * spins[j, 2])
            orig_energy += -J * dot_product
    
    # Propose new orientation
    orientation_idx = np.random.randint(0, orientations.shape[0])
    theta = orientations[orientation_idx, 0]
    phi = orientations[orientation_idx, 1]
    
    # Convert to Cartesian
    new_spin = np.array([
        spin_magnitude * np.sin(theta) * np.cos(phi),
        spin_magnitude * np.sin(theta) * np.sin(phi),
        spin_magnitude * np.cos(theta)
    ])
    
    # Calculate new energy contribution
    new_energy = 0.0
    for j_idx in range(max_neighbors):
        j = neighbor_array[site_idx, j_idx]
        if j >= 0 and j < spins.shape[0]:
            dot_product = (new_spin[0] * spins[j, 0] + 
                         new_spin[1] * spins[j, 1] + 
                         new_spin[2] * spins[j, 2])
            new_energy += -J * dot_product
    
    # Energy change
    delta_energy = new_energy - orig_energy
    
    # Metropolis criterion
    if delta_energy <= 0 or np.random.random() < np.exp(-delta_energy / (KB_EV_K * temperature)):
        # Accept move
        spins[site_idx, 0] = new_spin[0]
        spins[site_idx, 1] = new_spin[1]
        spins[site_idx, 2] = new_spin[2]
        return True, delta_energy
    else:
        # Reject move - spins array unchanged
        return False, 0.0


@njit(parallel=True, fastmath=True)
def fast_mc_sweep(
    spins, 
    neighbor_array, 
    orientations, 
    J, 
    temperature,
    spin_magnitude,
    random_order=True
):
    """
    Fast Monte Carlo sweep over all spins.
    
    Args:
        spins: (n_spins, 3) spin configuration
        neighbor_array: (n_spins, max_neighbors) neighbor indices  
        orientations: (n_orientations, 2) allowed orientations
        J: Exchange coupling
        temperature: Temperature in Kelvin
        spin_magnitude: Magnitude of spins
        random_order: Whether to randomize spin update order
        
    Returns:
        (n_accepted, total_energy_change) tuple
    """
    n_spins = spins.shape[0]
    n_accepted = 0
    total_delta_energy = 0.0
    
    # Create update order
    if random_order:
        indices = np.random.permutation(n_spins)
    else:
        indices = np.arange(n_spins)
    
    for i in range(n_spins):
        site_idx = indices[i]
        accepted, delta_energy = fast_metropolis_single_flip(
            spins, neighbor_array, orientations, site_idx, 
            J, temperature, spin_magnitude
        )
        
        if accepted:
            n_accepted += 1
            total_delta_energy += delta_energy
    
    return n_accepted, total_delta_energy


@njit(parallel=True, fastmath=True)
def fast_calculate_magnetization(spins):
    """
    Fast calculation of total magnetization.
    
    Args:
        spins: (n_spins, 3) spin configuration
        
    Returns:
        (3,) total magnetization vector
    """
    n_spins = spins.shape[0]
    
    # Use reduction for parallel sum
    total_x = 0.0
    total_y = 0.0  
    total_z = 0.0
    
    for i in prange(n_spins):
        total_x += spins[i, 0]
        total_y += spins[i, 1]
        total_z += spins[i, 2]
    
    return np.array([total_x / n_spins, total_y / n_spins, total_z / n_spins])


@njit(parallel=True, fastmath=True)
def fast_local_solid_angles(spins, positions, triangles):
    """
    Fast calculation of local solid angles for topological charge.
    
    Args:
        spins: (n_spins, 3) spin configuration
        positions: (n_spins, 2) 2D positions
        triangles: (n_triangles, 3) triangle vertex indices
        
    Returns:
        (n_triangles,) array of local solid angles
    """
    n_triangles = triangles.shape[0]
    solid_angles = np.zeros(n_triangles)
    
    for t in prange(n_triangles):
        i1, i2, i3 = triangles[t, 0], triangles[t, 1], triangles[t, 2]
        
        # Get spins at triangle vertices
        s1 = np.array([spins[i1, 0], spins[i1, 1], spins[i1, 2]])
        s2 = np.array([spins[i2, 0], spins[i2, 1], spins[i2, 2]])
        s3 = np.array([spins[i3, 0], spins[i3, 1], spins[i3, 2]])
        
        # Calculate scalar triple product s1 · (s2 × s3)
        cross_s2_s3_x = s2[1] * s3[2] - s2[2] * s3[1]
        cross_s2_s3_y = s2[2] * s3[0] - s2[0] * s3[2]
        cross_s2_s3_z = s2[0] * s3[1] - s2[1] * s3[0]
        
        numerator = s1[0] * cross_s2_s3_x + s1[1] * cross_s2_s3_y + s1[2] * cross_s2_s3_z
        
        # Calculate denominator
        dot_s1_s2 = s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2]
        dot_s1_s3 = s1[0] * s3[0] + s1[1] * s3[1] + s1[2] * s3[2]
        dot_s2_s3 = s2[0] * s3[0] + s2[1] * s3[1] + s2[2] * s3[2]
        
        denominator = 1.0 + dot_s1_s2 + dot_s1_s3 + dot_s2_s3
        
        # Calculate winding direction from positions
        p1 = np.array([positions[i1, 0], positions[i1, 1]])
        p2 = np.array([positions[i2, 0], positions[i2, 1]])
        p3 = np.array([positions[i3, 0], positions[i3, 1]])
        
        # Cross product in 2D: (p2-p1) × (p3-p1)
        cross_2d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        sign = 1.0 if cross_2d > 0 else -1.0
        
        # Local solid angle
        if denominator > 1e-12:
            solid_angles[t] = sign * 2.0 * np.arctan2(abs(numerator), denominator)
        else:
            solid_angles[t] = 0.0
    
    return solid_angles


def get_fast_operations():
    """
    Get dictionary of available fast operations.
    
    Returns:
        Dictionary mapping operation names to functions
    """
    if not HAS_NUMBA:
        return {}
    
    return {
        'exchange_energy': fast_exchange_energy,
        'anisotropic_exchange_energy': fast_anisotropic_exchange_energy,
        'single_ion_anisotropy_energy': fast_single_ion_anisotropy_energy,
        'zeeman_energy': fast_zeeman_energy,
        'dmi_energy': fast_dmi_energy,
        'effective_field': fast_effective_field,
        'llg_rhs': fast_llg_rhs,
        'normalize_spins': fast_normalize_spins,
        'metropolis_single_flip': fast_metropolis_single_flip,
        'mc_sweep': fast_mc_sweep,
        'calculate_magnetization': fast_calculate_magnetization,
        'local_solid_angles': fast_local_solid_angles
    }


# Check if numba is available and working
def check_numba_availability():
    """Check if Numba is available and working correctly."""
    if not HAS_NUMBA:
        return False, "Numba not installed"
    
    try:
        # Test compilation with a simple function
        @njit
        def test_func(x):
            return x * 2
        
        result = test_func(5.0)
        return True, "Numba available and working"
    
    except Exception as e:
        return False, f"Numba installation issue: {e}"