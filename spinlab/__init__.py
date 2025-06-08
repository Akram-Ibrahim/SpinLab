"""
SpinLab: Comprehensive spin simulation and analysis package.

A modern Python package for Monte Carlo simulations, LLG spin dynamics,
spin optimization, and thermodynamic analysis of magnetic systems.
"""

__version__ = "0.1.0"
__author__ = "Akram Ibrahim"
__email__ = "akram.ibrahim@example.com"

from . import core
from . import dynamics
from . import optimization
from . import analysis
from . import utils

# Note: IO functionality is available in utils.io module

from .core import SpinSystem, MonteCarlo, ParallelMonteCarlo
from .core.hamiltonian import Hamiltonian, KitaevTerm
from .dynamics import LLGSolver
from .optimization import SpinOptimizer
from .analysis import ThermodynamicsAnalyzer

# Cluster expansion utilities
from .utils import ClusterExpansionBuilder, create_bipartite_hamiltonian, create_triangular_hamiltonian

# Performance utilities
from .core.fast_ops import check_numba_availability, HAS_NUMBA
from .utils.performance import run_performance_test, benchmark_numba_speedup

__all__ = [
    "SpinSystem",
    "MonteCarlo",
    "ParallelMonteCarlo",
    "Hamiltonian",
 
    "KitaevTerm",
    "LLGSolver",
    "SpinOptimizer",
    "ThermodynamicsAnalyzer",
    "ClusterExpansionBuilder",
    "create_bipartite_hamiltonian",
    "create_triangular_hamiltonian",
    "check_numba_availability",
    "HAS_NUMBA",
    "run_performance_test",
    "benchmark_numba_speedup",
    "core",
    "dynamics",
    "optimization", 
    "analysis",
    "utils"
]