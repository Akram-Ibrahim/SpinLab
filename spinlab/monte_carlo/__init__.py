"""
Monte Carlo simulation methods for spin systems.

This module provides Monte Carlo simulation engines including:
- Single-core Monte Carlo with detailed analysis capabilities
- Multi-core parallel Monte Carlo for high-throughput simulations
"""

from .monte_carlo import MonteCarlo
from .parallel_monte_carlo import ParallelMonteCarlo

__all__ = ["MonteCarlo", "ParallelMonteCarlo"]