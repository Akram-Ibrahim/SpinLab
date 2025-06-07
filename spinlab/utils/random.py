"""Random number utilities."""

import numpy as np
from typing import Optional


def set_random_seed(seed: int):
    """
    Set random seed for reproducible results.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)


def generate_random_unit_vectors(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random unit vectors uniformly distributed on sphere.
    
    Args:
        n: Number of vectors to generate
        seed: Optional random seed
        
    Returns:
        Array of shape (n, 3) with unit vectors
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use Muller method for uniform distribution on sphere
    phi = np.random.uniform(0, 2*np.pi, n)
    cos_theta = np.random.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    
    return np.column_stack((x, y, z))