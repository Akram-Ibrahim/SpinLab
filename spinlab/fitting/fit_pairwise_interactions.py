"""
Comprehensive cluster expansion fitting with sublattice resolution.

This module provides tools to fit exchange parameters, Kitaev interactions,
DM interactions, and single-ion anisotropy with full sublattice resolution.

Physics Model:
    H = Σ_{k=1..n} Σ_{L,M} J_k^{LM} Σ_{⟨ij⟩_k^{LM}} s_i · s_j                    (isotropic exchange)
        + Σ_{k=1..n} Σ_{L,M} Σ_{γ=x,y,z} K_{k,γ}^{LM} Σ_{⟨ij⟩_k^{LM}} s_i^γ s_j^γ  (Kitaev interactions)
        + D_z Σ_{⟨ij⟩_1} ẑ · (s_i × s_j)                                          (DMI, usually no sublattice)
        + Σ_L A^L Σ_{i∈L} (s_i^z)^2                                               (single-ion per sublattice)

Where L,M label sublattices and ⟨ij⟩_k^{LM} denotes bonds between sublattices L and M in shell k.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from itertools import combinations_with_replacement
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings


def get_sublattice_pairs(n_sublattices: int, include_diagonal: bool = True) -> List[Tuple[int, int]]:
    """
    Generate all unique sublattice pairs for a given number of sublattices.
    
    Args:
        n_sublattices: Number of sublattices
        include_diagonal: Whether to include same-sublattice pairs (AA, BB, ...)
        
    Returns:
        List of (sublattice_i, sublattice_j) pairs
    """
    if include_diagonal:
        # Include all pairs: AA, AB, BB for bipartite, etc.
        return list(combinations_with_replacement(range(n_sublattices), 2))
    else:
        # Only inter-sublattice pairs: AB, AC, BC, ... (no AA, BB, CC)
        pairs = []
        for i in range(n_sublattices):
            for j in range(i+1, n_sublattices):
                pairs.append((i, j))
        return pairs


def generate_feature_names(
    shell_list: List[int],
    sublattice_pairs: List[Tuple[int, int]],
    include_kitaev: bool = False,
    include_single_ion: bool = True,
    include_dmi: bool = True,
    n_sublattices: int = 1,
    sublattice_names: Optional[List[str]] = None
) -> List[str]:
    """
    Generate systematic feature names for the design matrix.
    
    Args:
        shell_list: List of shell indices [1, 2, 3, ...]
        sublattice_pairs: List of (sub_i, sub_j) pairs to include
        include_kitaev: Whether Kitaev terms are included
        include_single_ion: Whether single-ion terms are included
        include_dmi: Whether DMI terms are included
        n_sublattices: Number of sublattices
        sublattice_names: Names for sublattices (default: A, B, C, ...)
        
    Returns:
        List of feature names in order
    """
    if sublattice_names is None:
        sublattice_names = [chr(ord('A') + i) for i in range(n_sublattices)]
    
    feature_names = []
    
    # Isotropic exchange terms: J_k^{LM}
    for shell_k in shell_list:
        for (sub_i, sub_j) in sublattice_pairs:
            sub_name_i = sublattice_names[sub_i]
            sub_name_j = sublattice_names[sub_j]
            feature_names.append(f"J{shell_k}_{sub_name_i}{sub_name_j}")
    
    # Kitaev terms: K_{k,γ}^{LM}
    if include_kitaev:
        for shell_k in shell_list:
            for gamma_name in ['x', 'y', 'z']:
                for (sub_i, sub_j) in sublattice_pairs:
                    sub_name_i = sublattice_names[sub_i]
                    sub_name_j = sublattice_names[sub_j]
                    feature_names.append(f"K{shell_k}_{gamma_name}_{sub_name_i}{sub_name_j}")
    
    # Single-ion anisotropy: A^L
    if include_single_ion:
        for sub_i in range(n_sublattices):
            sub_name = sublattice_names[sub_i]
            feature_names.append(f"A_{sub_name}")
    
    # DMI: D_z (usually no sublattice resolution)
    if include_dmi:
        feature_names.append("D_z")
    
    return feature_names


def design_matrix(
    spins: np.ndarray,
    shell_list: List[int],
    pair_tables: Dict[str, np.ndarray],
    sublattice_indices: np.ndarray,
    sublattice_pairs: List[Tuple[int, int]],
    include_kitaev: bool = False,
    include_single_ion: bool = True,
    include_dmi: bool = True
) -> np.ndarray:
    """
    Construct design matrix row for a single spin configuration with sublattice resolution.
    
    Args:
        spins: Spin configuration [N, 3] with |s_i| = 1
        shell_list: List of shell indices to include [1, 2, 3, ...]
        pair_tables: Dict mapping "shell_k" -> neighbor indices [N, max_neighbors]
        sublattice_indices: Array [N] assigning each site to sublattice (0, 1, 2, ...)
        sublattice_pairs: List of (sub_i, sub_j) pairs to include
        include_kitaev: Whether to include Kitaev interaction terms
        include_single_ion: Whether to include single-ion anisotropy term
        include_dmi: Whether to include DM interaction term
        
    Returns:
        phi: Feature vector [n_features]
    """
    N = spins.shape[0]
    n_sublattices = int(np.max(sublattice_indices)) + 1
    features = []
    
    # Isotropic exchange contributions: J_k^{LM} Σ_{⟨ij⟩_k^{LM}} s_i · s_j
    for shell_k in shell_list:
        shell_name = f"shell_{shell_k}"
        
        if shell_name not in pair_tables:
            raise ValueError(f"Shell {shell_name} not found in pair_tables")
        
        neighbor_array = pair_tables[shell_name]  # [N, max_neighbors]
        
        for (sub_i, sub_j) in sublattice_pairs:
            phi_k_ij = 0.0
            
            if neighbor_array.size > 0:
                for i in range(N):
                    if sublattice_indices[i] != sub_i:
                        continue  # Site i not in sublattice sub_i
                    
                    site_spin = spins[i]  # [3]
                    neighbor_indices = neighbor_array[i]  # [max_neighbors]
                    
                    # Filter valid neighbors in sublattice sub_j
                    valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                    for j in valid_neighbors:
                        if sublattice_indices[j] == sub_j:
                            neighbor_spin = spins[j]  # [3]
                            dot_product = np.dot(site_spin, neighbor_spin)
                            phi_k_ij += dot_product
                
                # Divide by 2 to avoid double counting if sub_i == sub_j
                if sub_i == sub_j:
                    phi_k_ij /= 2.0
            
            features.append(phi_k_ij)
    
    # Kitaev contributions: K_{k,γ}^{LM} Σ_{⟨ij⟩_k^{LM}} s_i^γ s_j^γ
    if include_kitaev:
        for shell_k in shell_list:
            shell_name = f"shell_{shell_k}"
            neighbor_array = pair_tables[shell_name]  # [N, max_neighbors]
            
            for gamma in range(3):  # x, y, z components
                for (sub_i, sub_j) in sublattice_pairs:
                    phi_k_gamma_ij = 0.0
                    
                    if neighbor_array.size > 0:
                        for i in range(N):
                            if sublattice_indices[i] != sub_i:
                                continue
                            
                            neighbor_indices = neighbor_array[i]
                            valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                            
                            for j in valid_neighbors:
                                if sublattice_indices[j] == sub_j:
                                    # K_{k,γ} s_i^γ s_j^γ
                                    si_gamma = spins[i, gamma]
                                    sj_gamma = spins[j, gamma]
                                    phi_k_gamma_ij += si_gamma * sj_gamma
                        
                        # Divide by 2 to avoid double counting if sub_i == sub_j
                        if sub_i == sub_j:
                            phi_k_gamma_ij /= 2.0
                    
                    features.append(phi_k_gamma_ij)
    
    # Single-ion anisotropy: A^L Σ_{i∈L} (s_i^z)²
    if include_single_ion:
        for sub_l in range(n_sublattices):
            phi_A_l = 0.0
            for i in range(N):
                if sublattice_indices[i] == sub_l:
                    phi_A_l += spins[i, 2]**2  # z-component squared
            features.append(phi_A_l)
    
    # DM interaction: Φ_D = Σ_{⟨ij⟩_1} ẑ · (s_i × s_j)
    if include_dmi:
        shell_1_name = f"shell_1"
        
        if shell_1_name not in pair_tables:
            warnings.warn("DMI requested but shell_1 not in pair_tables. Setting Φ_D = 0.")
            phi_D = 0.0
        else:
            neighbor_array = pair_tables[shell_1_name]  # First neighbor pairs
            
            if neighbor_array.size == 0:
                phi_D = 0.0
            else:
                phi_D = 0.0
                
                for i in range(N):
                    site_spin = spins[i]  # [3]
                    neighbor_indices = neighbor_array[i]  # [max_neighbors]
                    
                    # Filter valid neighbors
                    valid_neighbors = neighbor_indices[neighbor_indices >= 0]
                    
                    for j in valid_neighbors:
                        neighbor_spin = spins[j]  # [3]
                        
                        # Cross product s_i × s_j
                        cross_product = np.cross(site_spin, neighbor_spin)
                        
                        # ẑ · (s_i × s_j) = (s_i × s_j)_z (z-component)
                        z_component = cross_product[2]
                        phi_D += z_component
                
                # Divide by 2 to avoid double counting
                phi_D /= 2.0
        
        features.append(phi_D)
    
    return np.array(features)


def design_matrix_batch(
    spin_configs: List[np.ndarray],
    shell_list: List[int],
    pair_tables: Dict[str, np.ndarray],
    include_single_ion: bool = True,
    include_dmi: bool = True
) -> np.ndarray:
    """
    Construct design matrix for multiple spin configurations.
    
    Args:
        spin_configs: List of spin configurations, each [N, 3]
        shell_list: List of shell indices to include
        pair_tables: Dict mapping "shell_k" -> neighbor indices
        include_single_ion: Whether to include single-ion anisotropy
        include_dmi: Whether to include DM interaction
        
    Returns:
        Phi_matrix: Design matrix [n_configs, n_features]
    """
    n_configs = len(spin_configs)
    
    # Get feature dimension from first configuration
    if n_configs == 0:
        return np.array([]).reshape(0, 0)
    
    phi_0 = design_matrix(
        spin_configs[0], shell_list, pair_tables, 
        include_single_ion, include_dmi
    )
    n_features = len(phi_0)
    
    # Assemble full matrix
    Phi_matrix = np.zeros((n_configs, n_features))
    Phi_matrix[0] = phi_0
    
    for i in range(1, n_configs):
        Phi_matrix[i] = design_matrix(
            spin_configs[i], shell_list, pair_tables,
            include_single_ion, include_dmi
        )
    
    return Phi_matrix


def fit_parameters(
    E_list: np.ndarray,
    Phi_matrix: np.ndarray,
    method: str = "ols",
    alpha: float = 1.0,
    normalize_features: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Fit Hamiltonian parameters θ = [J₁, J₂, ..., Jₙ, A, D_z] using linear regression.
    
    Solves: E = Φ @ θ + ε
    
    Args:
        E_list: Target energies [n_configs]
        Phi_matrix: Design matrix [n_configs, n_features]
        method: Regression method ("ols", "ridge", "lasso")
        alpha: Regularization strength (for ridge/lasso)
        normalize_features: Whether to standardize features before fitting
        
    Returns:
        theta: Fitted parameters [n_features]
        info: Dictionary with fitting information (R², residuals, etc.)
    """
    if Phi_matrix.shape[0] != len(E_list):
        raise ValueError(f"Inconsistent sizes: {Phi_matrix.shape[0]} configs vs {len(E_list)} energies")
    
    if Phi_matrix.shape[0] < Phi_matrix.shape[1]:
        warnings.warn(f"Underdetermined system: {Phi_matrix.shape[0]} configs < {Phi_matrix.shape[1]} parameters")
    
    # Handle feature normalization
    scaler = None
    if normalize_features:
        scaler = StandardScaler()
        Phi_scaled = scaler.fit_transform(Phi_matrix)
    else:
        Phi_scaled = Phi_matrix
    
    # Select regression method
    if method.lower() == "ols":
        regressor = LinearRegression(fit_intercept=False)
    elif method.lower() == "ridge":
        regressor = Ridge(alpha=alpha, fit_intercept=False)
    elif method.lower() == "lasso":
        regressor = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ols', 'ridge', or 'lasso'")
    
    # Fit the model
    regressor.fit(Phi_scaled, E_list)
    theta_scaled = regressor.coef_
    
    # Unscale parameters if normalization was used
    if normalize_features:
        theta = theta_scaled / scaler.scale_
    else:
        theta = theta_scaled
    
    # Compute fitting diagnostics
    E_pred = Phi_matrix @ theta
    residuals = E_list - E_pred
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((E_list - np.mean(E_list))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Root mean square error
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Mean absolute error
    mae = np.mean(np.abs(residuals))
    
    info = {
        'method': method,
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'residuals': residuals,
        'predictions': E_pred,
        'n_configs': len(E_list),
        'n_features': Phi_matrix.shape[1],
        'alpha': alpha if method in ['ridge', 'lasso'] else None,
        'scaler': scaler,
        'regressor': regressor
    }
    
    return theta, info


def fit_from_configs(
    spin_configs: List[np.ndarray],
    energies: np.ndarray,
    spin_system,  # SpinSystem object
    max_shell: int,
    method: str = "ols",
    alpha: float = 1.0,
    include_single_ion: bool = True,
    include_dmi: bool = True,
    normalize_features: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience wrapper to fit Hamiltonian parameters from spin configurations.
    
    Uses the existing SpinSystem neighbor finding infrastructure.
    
    Args:
        spin_configs: List of spin configurations [N, 3]
        energies: Target energies [n_configs]
        spin_system: SpinSystem object (used for neighbor finding)
        max_shell: Maximum neighbor shell to include
        method: Regression method ("ols", "ridge", "lasso")
        alpha: Regularization strength
        include_single_ion: Include single-ion anisotropy A
        include_dmi: Include DM interaction D_z
        normalize_features: Standardize features before fitting
        
    Returns:
        theta: Fitted parameters [J₁, ..., Jₙ, A?, D_z?]
        fit_info: Fitting diagnostics and metadata
    """
    print(f"🔧 Setting up neighbor tables up to shell {max_shell}...")
    
    # Use existing SpinSystem infrastructure to get neighbors
    # We'll use reasonable cutoffs based on the structure
    positions = spin_system.positions
    
    # Estimate cutoffs based on nearest neighbor distances
    distances = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            distances.append(dist)
    
    distances = np.array(sorted(distances))
    unique_distances = []
    tolerance = 0.1
    
    for dist in distances:
        if len(unique_distances) == 0 or abs(dist - unique_distances[-1]) > tolerance:
            unique_distances.append(dist)
    
    # Use the first max_shell unique distances as cutoffs
    cutoffs = unique_distances[:max_shell]
    if len(cutoffs) < max_shell:
        warnings.warn(f"Only {len(cutoffs)} unique distances found, but max_shell={max_shell} requested")
        max_shell = len(cutoffs)
    
    print(f"   Using cutoffs: {cutoffs}")
    
    # Get neighbor tables using SpinSystem infrastructure
    pair_tables = spin_system.get_neighbors(cutoffs)
    
    # Report neighbor statistics
    total_pairs = 0
    for shell_name, neighbor_array in pair_tables.items():
        # Count valid neighbors (exclude -1 entries)
        valid_count = np.sum(neighbor_array >= 0)
        total_pairs += valid_count
        print(f"   {shell_name}: {valid_count} total neighbor connections")
    
    print(f"   Found {total_pairs} total neighbor connections across {len(pair_tables)} shells")
    
    # Construct shell list
    shell_list = list(range(1, max_shell + 1))
    
    print(f"📊 Assembling design matrix for {len(spin_configs)} configurations...")
    
    # Build design matrix
    Phi_matrix = design_matrix_batch(
        spin_configs, shell_list, pair_tables,
        include_single_ion, include_dmi
    )
    
    n_features = Phi_matrix.shape[1]
    feature_names = [f"J{k}" for k in shell_list]
    if include_single_ion:
        feature_names.append("A")
    if include_dmi:
        feature_names.append("D_z")
    
    print(f"   Design matrix shape: {Phi_matrix.shape}")
    print(f"   Features: {feature_names}")
    
    print(f"🎯 Fitting parameters using {method.upper()}...")
    
    # Fit parameters
    theta, fit_info = fit_parameters(
        energies, Phi_matrix, method=method, alpha=alpha,
        normalize_features=normalize_features
    )
    
    # Add metadata to fit_info
    fit_info.update({
        'shell_list': shell_list,
        'feature_names': feature_names,
        'max_shell': max_shell,
        'include_single_ion': include_single_ion,
        'include_dmi': include_dmi,
        'pair_tables': pair_tables,
        'cutoffs': cutoffs,
        'n_positions': len(positions)
    })
    
    print(f"✅ Fitting completed!")
    print(f"   R² = {fit_info['r_squared']:.4f}")
    print(f"   RMSE = {fit_info['rmse']:.6f} eV")
    print(f"   MAE = {fit_info['mae']:.6f} eV")
    
    # Print fitted parameters
    print(f"📋 Fitted parameters:")
    for i, (name, value) in enumerate(zip(feature_names, theta)):
        print(f"   {name}: {value:10.6f} eV")
    
    return theta, fit_info


def predict_energy(
    spins: np.ndarray,
    theta: np.ndarray,
    shell_list: List[int],
    pair_tables: Dict[str, np.ndarray],
    include_single_ion: bool = True,
    include_dmi: bool = True
) -> float:
    """
    Predict energy for a spin configuration using fitted parameters.
    
    Args:
        spins: Spin configuration [N, 3]
        theta: Fitted parameters [n_features]
        shell_list: List of shells used in fitting
        pair_tables: Neighbor tables
        include_single_ion: Whether A term was included in fitting
        include_dmi: Whether D_z term was included in fitting
        
    Returns:
        energy: Predicted energy (scalar)
    """
    phi = design_matrix(
        spins, shell_list, pair_tables, 
        include_single_ion, include_dmi
    )
    
    return np.dot(phi, theta)