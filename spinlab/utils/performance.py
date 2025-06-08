"""
Performance testing and benchmarking utilities.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from ase.build import bulk

from ..core.spin_system import SpinSystem
from ..core.hamiltonian import Hamiltonian
from ..monte_carlo import MonteCarlo
from ..dynamics.llg_solver import LLGSolver
from ..core.fast_ops import check_numba_availability, HAS_NUMBA


def benchmark_numba_speedup(
    system_sizes: List[int] = [10, 20, 50, 100],
    n_iterations: int = 100
) -> Dict[str, Any]:
    """
    Benchmark Numba speedup for various operations.
    
    Args:
        system_sizes: List of system sizes (linear dimension)
        n_iterations: Number of iterations for timing
        
    Returns:
        Dictionary with benchmark results
    """
    # Check if Numba is available
    numba_available, message = check_numba_availability()
    
    results = {
        'numba_available': numba_available,
        'numba_message': message,
        'system_sizes': system_sizes,
        'benchmarks': {}
    }
    
    if not numba_available:
        print(f"Numba not available: {message}")
        return results
    
    print("Benchmarking Numba speedup...")
    print("=" * 50)
    
    for size in system_sizes:
        print(f"\nSystem size: {size}x{size}x{size} = {size**3} spins")
        
        # Create test system
        structure = bulk('Fe', 'bcc', a=2.87)
        structure = structure.repeat((size, size, size))
        
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(-0.01)
        
        # Test with and without fast operations
        spin_system_fast = SpinSystem(structure, hamiltonian, use_fast=True)
        spin_system_slow = SpinSystem(structure, hamiltonian, use_fast=False)
        
        # Get neighbors
        spin_system_fast.get_neighbors([3.0])
        spin_system_slow.get_neighbors([3.0])
        
        # Initialize random configurations
        spin_system_fast.random_configuration(seed=42)
        spin_system_slow.random_configuration(seed=42)
        
        size_results = {}
        
        # Benchmark energy calculation
        print("  Benchmarking energy calculation...")
        
        # Fast version
        start_time = time.time()
        for _ in range(n_iterations):
            energy_fast = spin_system_fast.calculate_energy()
        time_fast = time.time() - start_time
        
        # Slow version
        start_time = time.time()
        for _ in range(n_iterations):
            energy_slow = spin_system_slow.calculate_energy()
        time_slow = time.time() - start_time
        
        speedup_energy = time_slow / time_fast if time_fast > 0 else float('inf')
        
        size_results['energy_calculation'] = {
            'time_fast': time_fast,
            'time_slow': time_slow,
            'speedup': speedup_energy,
            'energy_fast': energy_fast,
            'energy_slow': energy_slow,
            'energy_match': np.abs(energy_fast - energy_slow) < 1e-10
        }
        
        print(f"    Energy calculation speedup: {speedup_energy:.1f}x")
        
        # Benchmark magnetization calculation
        print("  Benchmarking magnetization calculation...")
        
        # Fast version
        start_time = time.time()
        for _ in range(n_iterations):
            mag_fast = spin_system_fast.calculate_magnetization()
        time_fast = time.time() - start_time
        
        # Slow version  
        start_time = time.time()
        for _ in range(n_iterations):
            mag_slow = spin_system_slow.calculate_magnetization()
        time_slow = time.time() - start_time
        
        speedup_mag = time_slow / time_fast if time_fast > 0 else float('inf')
        
        size_results['magnetization_calculation'] = {
            'time_fast': time_fast,
            'time_slow': time_slow,
            'speedup': speedup_mag,
            'magnetization_match': np.allclose(mag_fast, mag_slow, atol=1e-10)
        }
        
        print(f"    Magnetization calculation speedup: {speedup_mag:.1f}x")
        
        # Benchmark Monte Carlo steps (smaller number of iterations)
        mc_iterations = min(10, n_iterations // 10)
        if mc_iterations > 0:
            print("  Benchmarking Monte Carlo steps...")
            
            # Fast MC
            mc_fast = MonteCarlo(spin_system_fast, temperature=300.0, use_fast=True)
            start_time = time.time()
            for _ in range(mc_iterations):
                orientations = spin_system_fast.generate_spin_orientations()
                mc_fast._prepare_fast_operations(orientations)
                mc_fast._perform_sweep(orientations)
            time_fast = time.time() - start_time
            
            # Slow MC
            mc_slow = MonteCarlo(spin_system_slow, temperature=300.0, use_fast=False)
            start_time = time.time()
            for _ in range(mc_iterations):
                orientations = spin_system_slow.generate_spin_orientations()
                mc_slow._perform_sweep(orientations)
            time_slow = time.time() - start_time
            
            speedup_mc = time_slow / time_fast if time_fast > 0 else float('inf')
            
            size_results['monte_carlo_sweep'] = {
                'time_fast': time_fast,
                'time_slow': time_slow,
                'speedup': speedup_mc,
                'iterations': mc_iterations
            }
            
            print(f"    Monte Carlo sweep speedup: {speedup_mc:.1f}x")
        
        # Benchmark LLG dynamics (even fewer iterations)
        llg_iterations = min(5, n_iterations // 20)
        if llg_iterations > 0:
            print("  Benchmarking LLG dynamics...")
            
            # Fast LLG
            llg_fast = LLGSolver(spin_system_fast, use_fast=True)
            start_time = time.time()
            for _ in range(llg_iterations):
                spins = spin_system_fast.spin_config.copy()
                llg_fast.calculate_llg_rhs(spins)
            time_fast = time.time() - start_time
            
            # Slow LLG
            llg_slow = LLGSolver(spin_system_slow, use_fast=False)
            start_time = time.time()
            for _ in range(llg_iterations):
                spins = spin_system_slow.spin_config.copy()
                llg_slow.calculate_llg_rhs(spins)
            time_slow = time.time() - start_time
            
            speedup_llg = time_slow / time_fast if time_fast > 0 else float('inf')
            
            size_results['llg_dynamics'] = {
                'time_fast': time_fast,
                'time_slow': time_slow,
                'speedup': speedup_llg,
                'iterations': llg_iterations
            }
            
            print(f"    LLG dynamics speedup: {speedup_llg:.1f}x")
        
        results['benchmarks'][size] = size_results
    
    return results


def print_benchmark_summary(results: Dict[str, Any]):
    """Print a summary of benchmark results."""
    
    print("\n" + "=" * 60)
    print("NUMBA SPEEDUP BENCHMARK SUMMARY")
    print("=" * 60)
    
    if not results['numba_available']:
        print(f"Numba not available: {results['numba_message']}")
        return
    
    print(f"Numba status: {results['numba_message']}")
    print()
    
    # Create summary table
    operations = ['energy_calculation', 'magnetization_calculation', 'monte_carlo_sweep', 'llg_dynamics']
    op_names = ['Energy Calc', 'Magnetization', 'MC Sweep', 'LLG Dynamics']
    
    # Print header
    print(f"{'System Size':<12}", end="")
    for name in op_names:
        print(f"{name:>15}", end="")
    print()
    
    print("-" * (12 + 15 * len(op_names)))
    
    # Print speedups for each system size
    for size in results['system_sizes']:
        if size in results['benchmarks']:
            print(f"{size**3:<12}", end="")
            
            for op in operations:
                if op in results['benchmarks'][size]:
                    speedup = results['benchmarks'][size][op]['speedup']
                    print(f"{speedup:>14.1f}x", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()
    
    print()
    
    # Print average speedups
    print("AVERAGE SPEEDUPS:")
    for i, op in enumerate(operations):
        speedups = []
        for size in results['system_sizes']:
            if (size in results['benchmarks'] and 
                op in results['benchmarks'][size]):
                speedup = results['benchmarks'][size][op]['speedup']
                if speedup != float('inf'):
                    speedups.append(speedup)
        
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"  {op_names[i]}: {avg_speedup:.1f}x")
    
    print()
    print("Note: Speedups are calculated as (NumPy time) / (Numba time)")
    print("Higher values indicate better performance with Numba acceleration.")


def run_performance_test():
    """Run a comprehensive performance test."""
    print("SpinLab Performance Test")
    print("=" * 40)
    
    # Run benchmarks
    results = benchmark_numba_speedup(
        system_sizes=[5, 10, 20],  # Smaller sizes for quick testing
        n_iterations=50
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    return results


def memory_usage_analysis(system_sizes: List[int] = [10, 20, 50, 100]) -> Dict[str, Any]:
    """
    Analyze memory usage for different system sizes.
    
    Args:
        system_sizes: List of system sizes to test
        
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    results = {}
    
    print("Memory Usage Analysis")
    print("=" * 30)
    
    for size in system_sizes:
        print(f"\nSystem size: {size}x{size}x{size} = {size**3} spins")
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create system
        structure = bulk('Fe', 'bcc', a=2.87)
        structure = structure.repeat((size, size, size))
        
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(-0.01)
        
        spin_system = SpinSystem(structure, hamiltonian)
        spin_system.get_neighbors([3.0])
        spin_system.random_configuration()
        
        # Measure memory after system creation
        system_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_used = system_memory - baseline_memory
        memory_per_spin = memory_used / (size**3) * 1024  # KB per spin
        
        results[size] = {
            'n_spins': size**3,
            'memory_used_mb': memory_used,
            'memory_per_spin_kb': memory_per_spin
        }
        
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Memory per spin: {memory_per_spin:.2f} KB")
        
        # Clean up
        del spin_system, structure, hamiltonian
    
    return results


if __name__ == "__main__":
    # Run performance test
    run_performance_test()