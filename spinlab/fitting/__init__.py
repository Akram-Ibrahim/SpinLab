"""
Parameter fitting and cluster expansion tools for SpinLab.

This module provides tools to fit magnetic interaction parameters from
spin configurations and energies using cluster expansion methods.
"""

from .fit_exchange import (
    design_matrix,
    design_matrix_batch, 
    fit_parameters,
    fit_from_configs,
    predict_energy
)

from .pair_hamiltonian import PairHamiltonian

__all__ = [
    "design_matrix",
    "design_matrix_batch",
    "fit_parameters", 
    "fit_from_configs",
    "predict_energy",
    "PairHamiltonian"
]