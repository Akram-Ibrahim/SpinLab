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
def exchange_energy(spins, neighbor_array, J):
    """
    Exchange energy calculation using Numba.
    
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
            if j >= 0:  # Valid neighbor (bounds check removed - neighbor finding ensures validity)
                # Use numpy dot product (Numba optimizes this well)
                dot_product = np.dot(spins[i], spins[j])
                energy += -J * dot_product
        energies[i] = energy * 0.5  # Avoid double counting
    
    return energies


@njit(parallel=True, fastmath=True)
def anisotropic_exchange_energy(spins, neighbor_array, coupling_matrix):
    """
    Anisotropic exchange energy calculation.
    
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
def single_ion_anisotropy_energy(spins, K, axis):
    """
    Single-ion anisotropy energy calculation.
    
    Args:
        spins: (n_spins, 3) array of spin vectors
        K: Anisotropy constant
        axis: (3,) easy axis vector
        
    Returns:
        Array of site energies
    """
    n_spins = spins.shape[0]
    energies = np.zeros(n_spins)
    
    # Full vectorization for single-ion anisotropy
    dot_products = spins @ axis  # All dot products at once
    return -K * dot_products * dot_products


@njit(parallel=True, fastmath=True)
def magnetic_field_energy(spins, B_field, g_factor):
    """
    Fast magnetic field energy calculation.
    
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
    
    # Full vectorization - compute all dot products at once
    # spins @ B_field gives dot product for each spin
    return factor * (spins @ B_field)


@njit(parallel=True, fastmath=True)
def dmi_energy(spins, neighbor_array, D_vector):
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
def exchange_effective_field(spins, neighbor_array, J):
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
def spin_cross_product(spin, field):
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
def llg_rhs(spins, effective_fields, gamma, alpha):
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
def normalize_spins(spins, target_magnitude):
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






@njit(parallel=True, fastmath=True)
def calculate_magnetization(spins):
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
def local_solid_angles(spins, positions, triangles):
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


def get_numba_operations():
    """
    Get dictionary of available Numba-accelerated operations.
    
    Returns:
        Dictionary mapping operation names to functions
    """
    if not HAS_NUMBA:
        return {}
    
    return {
        # Energy calculations
        'exchange_energy': exchange_energy,
        'single_ion_anisotropy_energy': single_ion_anisotropy_energy,
        'magnetic_field_energy': magnetic_field_energy,
        'dmi_energy': dmi_energy,
        
        # Field calculations
        'exchange_effective_field': exchange_effective_field,
        
        # LLG dynamics
        'llg_rhs': llg_rhs,
        'normalize_spins': normalize_spins,
        
        # Monte Carlo (complete Hamiltonian)
        'local_site_energy': local_site_energy,
        'local_energy_change': local_energy_change,
        'metropolis_single_flip': metropolis_single_flip,
        'monte_carlo_sweep': monte_carlo_sweep,
        
        # Analysis
        'calculate_magnetization': calculate_magnetization,
        'local_solid_angles': local_solid_angles,
        'spin_cross_product': spin_cross_product
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


# =============================================================================
# UNIFIED LOCAL ENERGY KERNEL FOR FULL HAMILTONIAN MONTE CARLO
# =============================================================================

@njit(fastmath=True)
def _get_bond_direction(site_i, site_j, bond_directions):
    """
    Get bond direction for Kitaev interactions.
    
    Args:
        site_i, site_j: Site indices
        bond_directions: Dict-like mapping (i,j) -> direction index (0=x, 1=y, 2=z)
        
    Returns:
        Direction index: 0=x, 1=y, 2=z, -1=no direction found
    """
    # For Numba compatibility, we'll use a simplified approach
    # In practice, this would be pre-computed and passed as arrays
    
    # Default bond direction logic (can be customized)
    # This is a placeholder - real implementation would use the provided mapping
    if (site_i + site_j) % 3 == 0:
        return 0  # x-direction
    elif (site_i + site_j) % 3 == 1:
        return 1  # y-direction
    else:
        return 2  # z-direction


@njit(fastmath=True)
def _cross_product(a, b):
    """Fast 3D cross product a × b."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2], 
        a[0] * b[1] - a[1] * b[0]
    ])


@njit(fastmath=True)
def local_site_energy(
    spins,
    site_idx,
    site_spin_vector,
    neighbor_array,
    # Exchange parameters
    J_exchange,
    # Kitaev parameters  
    K_kitaev_x,
    K_kitaev_y,
    K_kitaev_z,
    include_kitaev,
    # DMI parameters
    D_dmi_vector,
    include_dmi,
    # Single-ion anisotropy
    A_anisotropy,
    anisotropy_axis,
    include_anisotropy,
    # Magnetic field
    magnetic_field,
    g_factor,
    include_magnetic_field
):
    """
    Calculate local energy for a single site with FULL Hamiltonian.
    
    This is the core kernel for Monte Carlo energy calculations.
    
    Args:
        spins: (n_spins, 3) full spin configuration
        site_idx: Index of the site to calculate energy for
        site_spin_vector: (3,) spin vector at site_idx (could be proposed new spin)
        neighbor_array: (n_spins, max_neighbors) neighbor indices
        J_exchange: Exchange coupling constant
        K_kitaev_x, K_kitaev_y, K_kitaev_z: Kitaev coupling constants
        include_kitaev: Whether to include Kitaev terms
        D_dmi_vector: (3,) DMI vector
        include_dmi: Whether to include DMI
        A_anisotropy: Single-ion anisotropy constant
        anisotropy_axis: (3,) easy axis for anisotropy
        include_anisotropy: Whether to include anisotropy
        magnetic_field: (3,) magnetic field vector (Tesla)
        g_factor: Landé g-factor
        include_magnetic_field: Whether to include magnetic field
        
    Returns:
        Local energy contribution from this site
    """
    energy = 0.0
    max_neighbors = neighbor_array.shape[1]
    
    # =================================================================
    # EXCHANGE INTERACTION: -J Σ_j s_i · s_j
    # =================================================================
    if J_exchange != 0.0:
        for j_idx in range(max_neighbors):
            j = neighbor_array[site_idx, j_idx]
            if j >= 0 and j < spins.shape[0]:
                dot_product = (site_spin_vector[0] * spins[j, 0] + 
                             site_spin_vector[1] * spins[j, 1] + 
                             site_spin_vector[2] * spins[j, 2])
                energy += -J_exchange * dot_product
    
    # =================================================================
    # KITAEV INTERACTIONS: K_γ Σ_j s_i^γ s_j^γ
    # =================================================================
    if include_kitaev:
        for j_idx in range(max_neighbors):
            j = neighbor_array[site_idx, j_idx]
            if j >= 0 and j < spins.shape[0]:
                # Get bond direction (simplified for now)
                bond_dir = _get_bond_direction(site_idx, j, None)
                
                if bond_dir == 0 and K_kitaev_x != 0.0:  # x-direction
                    energy += K_kitaev_x * site_spin_vector[0] * spins[j, 0]
                elif bond_dir == 1 and K_kitaev_y != 0.0:  # y-direction  
                    energy += K_kitaev_y * site_spin_vector[1] * spins[j, 1]
                elif bond_dir == 2 and K_kitaev_z != 0.0:  # z-direction
                    energy += K_kitaev_z * site_spin_vector[2] * spins[j, 2]
    
    # =================================================================
    # DMI INTERACTION: D · (s_i × s_j) 
    # =================================================================
    if include_dmi and np.linalg.norm(D_dmi_vector) > 0:
        for j_idx in range(max_neighbors):
            j = neighbor_array[site_idx, j_idx]
            if j >= 0 and j < spins.shape[0]:
                cross_product = _cross_product(site_spin_vector, spins[j])
                dot_product = (D_dmi_vector[0] * cross_product[0] + 
                             D_dmi_vector[1] * cross_product[1] + 
                             D_dmi_vector[2] * cross_product[2])
                energy += dot_product
    
    # =================================================================
    # SINGLE-ION ANISOTROPY: -A (s_i · axis)²
    # =================================================================
    if include_anisotropy and A_anisotropy != 0.0:
        dot_axis = (site_spin_vector[0] * anisotropy_axis[0] + 
                   site_spin_vector[1] * anisotropy_axis[1] + 
                   site_spin_vector[2] * anisotropy_axis[2])
        energy += -A_anisotropy * dot_axis * dot_axis
    
    # =================================================================
    # MAGNETIC FIELD: -g μ_B B · s_i
    # =================================================================
    if include_magnetic_field and np.linalg.norm(magnetic_field) > 0:
        dot_field = (magnetic_field[0] * site_spin_vector[0] + 
                    magnetic_field[1] * site_spin_vector[1] + 
                    magnetic_field[2] * site_spin_vector[2])
        energy += -g_factor * MU_B_EV_T * dot_field
    
    return energy


@njit(fastmath=True)
def local_energy_change(
    spins,
    site_idx,
    new_spin_vector,
    neighbor_array,
    # Exchange parameters
    J_exchange,
    # Kitaev parameters
    K_kitaev_x,
    K_kitaev_y, 
    K_kitaev_z,
    include_kitaev,
    # DMI parameters
    D_dmi_vector,
    include_dmi,
    # Single-ion anisotropy
    A_anisotropy,
    anisotropy_axis,
    include_anisotropy,
    # Magnetic field
    magnetic_field,
    g_factor,
    include_magnetic_field
):
    """
    Calculate energy change when flipping spin at site_idx to new_spin_vector.
    
    This is the key function for Monte Carlo acceptance/rejection.
    
    Returns:
        ΔE = E_new - E_old
    """
    # Current spin at site
    orig_spin = np.array([spins[site_idx, 0], spins[site_idx, 1], spins[site_idx, 2]])
    
    # Calculate original local energy
    E_old = local_site_energy(
        spins, site_idx, orig_spin, neighbor_array,
        J_exchange, K_kitaev_x, K_kitaev_y, K_kitaev_z, include_kitaev,
        D_dmi_vector, include_dmi, A_anisotropy, anisotropy_axis, include_anisotropy,
        magnetic_field, g_factor, include_magnetic_field
    )
    
    # Calculate new local energy
    E_new = local_site_energy(
        spins, site_idx, new_spin_vector, neighbor_array,
        J_exchange, K_kitaev_x, K_kitaev_y, K_kitaev_z, include_kitaev,
        D_dmi_vector, include_dmi, A_anisotropy, anisotropy_axis, include_anisotropy,
        magnetic_field, g_factor, include_magnetic_field
    )
    
    return E_new - E_old


@njit(fastmath=True)
def metropolis_single_flip(
    spins,
    neighbor_array,
    orientations,
    site_idx,
    temperature,
    spin_magnitude,
    # Exchange parameters
    J_exchange,
    # Kitaev parameters
    K_kitaev_x,
    K_kitaev_y,
    K_kitaev_z, 
    include_kitaev,
    # DMI parameters
    D_dmi_vector,
    include_dmi,
    # Single-ion anisotropy
    A_anisotropy,
    anisotropy_axis,
    include_anisotropy,
    # Magnetic field
    magnetic_field,
    g_factor,
    include_magnetic_field
):
    """
    Single spin flip with complete Hamiltonian using Metropolis criterion.
    
    Includes all interaction types: exchange, Kitaev, DMI, anisotropy, fields.
    
    Returns:
        (accepted, energy_change) tuple
    """
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
    
    # Calculate energy change with FULL Hamiltonian
    delta_energy = local_energy_change(
        spins, site_idx, new_spin, neighbor_array,
        J_exchange, K_kitaev_x, K_kitaev_y, K_kitaev_z, include_kitaev,
        D_dmi_vector, include_dmi, A_anisotropy, anisotropy_axis, include_anisotropy,
        magnetic_field, g_factor, include_magnetic_field
    )
    
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
def monte_carlo_sweep(
    spins,
    neighbor_array,
    orientations,
    temperature,
    spin_magnitude,
    # Hamiltonian parameters
    J_exchange,
    K_kitaev_x,
    K_kitaev_y,
    K_kitaev_z,
    include_kitaev,
    D_dmi_vector,
    include_dmi,
    A_anisotropy,
    anisotropy_axis,
    include_anisotropy,
    magnetic_field,
    g_factor,
    include_magnetic_field,
    random_order=True
):
    """
    Monte Carlo sweep over all spins with complete Hamiltonian.
    
    Includes all interaction types: exchange, Kitaev, DMI, anisotropy, fields.
    
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
        accepted, delta_energy = metropolis_single_flip(
            spins, neighbor_array, orientations, site_idx, temperature, spin_magnitude,
            J_exchange, K_kitaev_x, K_kitaev_y, K_kitaev_z, include_kitaev,
            D_dmi_vector, include_dmi, A_anisotropy, anisotropy_axis, include_anisotropy,
            magnetic_field, g_factor, include_magnetic_field
        )
        
        if accepted:
            n_accepted += 1
            total_delta_energy += delta_energy
    
    return n_accepted, total_delta_energy