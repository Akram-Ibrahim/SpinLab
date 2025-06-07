#!/usr/bin/env python3
"""
Numba Performance Demonstration for SpinLab.

This example demonstrates the significant speedup achieved with Numba
acceleration compared to pure NumPy implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from ase.build import bulk

# Import SpinLab components
from spinlab import SpinSystem, MonteCarlo, LLGSolver
from spinlab.core.hamiltonian import Hamiltonian
from spinlab.utils.performance import benchmark_numba_speedup, print_benchmark_summary
from spinlab.core.fast_ops import check_numba_availability


def main():
    """Run Numba performance demonstration."""
    
    print("SpinLab: Numba Performance Demonstration")
    print("=" * 50)
    
    # Check Numba availability
    numba_available, message = check_numba_availability()
    print(f"Numba status: {message}")
    
    if not numba_available:
        print("Numba is not available. Installing Numba will provide significant speedups!")
        print("Install with: pip install numba")
        return
    
    print("\n1. BASIC SPEEDUP COMPARISON")
    print("-" * 30)
    
    # Create a test system
    print("Creating test system (20x20x20 spins)...")
    structure = bulk('Fe', 'bcc', a=2.87)
    structure = structure.repeat((20, 20, 20))
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(-0.01)  # Ferromagnetic exchange
    
    # Systems with and without Numba
    spin_system_fast = SpinSystem(structure, hamiltonian, use_fast=True)
    spin_system_slow = SpinSystem(structure, hamiltonian, use_fast=False)
    
    # Setup neighbors and configurations
    spin_system_fast.get_neighbors([3.0])
    spin_system_slow.get_neighbors([3.0])
    
    spin_system_fast.random_configuration(seed=42)
    spin_system_slow.random_configuration(seed=42)
    
    print(f"System size: {len(structure)} spins")
    
    # Energy calculation benchmark
    print("\nBenchmarking energy calculation (100 iterations)...")
    
    n_iterations = 100
    
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
    
    speedup = time_slow / time_fast
    
    print(f"  Numba time: {time_fast:.4f} seconds")
    print(f"  NumPy time: {time_slow:.4f} seconds")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Energy match: {np.abs(energy_fast - energy_slow) < 1e-10}")
    
    # Monte Carlo benchmark
    print("\nBenchmarking Monte Carlo simulation...")
    
    # Run short MC simulations
    mc_fast = MonteCarlo(spin_system_fast, temperature=300.0, use_fast=True)
    mc_slow = MonteCarlo(spin_system_slow, temperature=300.0, use_fast=False)
    
    # Fast MC
    start_time = time.time()
    result_fast = mc_fast.run(n_steps=500, verbose=False)
    time_mc_fast = time.time() - start_time
    
    # Reset system
    spin_system_slow.random_configuration(seed=42)
    
    # Slow MC
    start_time = time.time()
    result_slow = mc_slow.run(n_steps=500, verbose=False)
    time_mc_slow = time.time() - start_time
    
    mc_speedup = time_mc_slow / time_mc_fast
    
    print(f"  Numba MC time: {time_mc_fast:.4f} seconds")
    print(f"  NumPy MC time: {time_mc_slow:.4f} seconds") 
    print(f"  MC Speedup: {mc_speedup:.1f}x")
    
    # LLG dynamics benchmark
    print("\nBenchmarking LLG dynamics...")
    
    llg_fast = LLGSolver(spin_system_fast, use_fast=True)
    llg_slow = LLGSolver(spin_system_slow, use_fast=False)
    
    # Fast LLG
    start_time = time.time()
    result_llg_fast = llg_fast.run(total_time=1e-12, dt=1e-15, verbose=False)
    time_llg_fast = time.time() - start_time
    
    # Reset system
    spin_system_slow.random_configuration(seed=42)
    
    # Slow LLG
    start_time = time.time()
    result_llg_slow = llg_slow.run(total_time=1e-12, dt=1e-15, verbose=False)
    time_llg_slow = time.time() - start_time
    
    llg_speedup = time_llg_slow / time_llg_fast
    
    print(f"  Numba LLG time: {time_llg_fast:.4f} seconds")
    print(f"  NumPy LLG time: {time_llg_slow:.4f} seconds")
    print(f"  LLG Speedup: {llg_speedup:.1f}x")
    
    print("\n2. COMPREHENSIVE BENCHMARK")
    print("-" * 30)
    
    # Run comprehensive benchmark
    results = benchmark_numba_speedup(
        system_sizes=[5, 10, 15, 20],
        n_iterations=20
    )
    
    print_benchmark_summary(results)
    
    print("\n3. SCALING ANALYSIS")
    print("-" * 30)
    
    # Test scaling with system size
    sizes = [5, 10, 15, 20, 25]
    speedups_energy = []
    speedups_mc = []
    
    for size in sizes:
        print(f"Testing size {size}^3 = {size**3} spins...")
        
        # Create system
        struct = bulk('Fe', 'bcc', a=2.87)
        struct = struct.repeat((size, size, size))
        
        ham = Hamiltonian()
        ham.add_exchange(-0.01)
        
        sys_fast = SpinSystem(struct, ham, use_fast=True)
        sys_slow = SpinSystem(struct, ham, use_fast=False)
        
        sys_fast.get_neighbors([3.0])
        sys_slow.get_neighbors([3.0])
        
        sys_fast.random_configuration(seed=42)
        sys_slow.random_configuration(seed=42)
        
        # Energy benchmark
        start = time.time()
        for _ in range(10):
            sys_fast.calculate_energy()
        time_f = time.time() - start
        
        start = time.time()
        for _ in range(10):
            sys_slow.calculate_energy()
        time_s = time.time() - start
        
        speedup_energy = time_s / time_f
        speedups_energy.append(speedup_energy)
        
        # MC benchmark (fewer iterations for larger systems)
        mc_iters = max(1, 100 // size)
        
        mc_f = MonteCarlo(sys_fast, 300.0, use_fast=True)
        mc_s = MonteCarlo(sys_slow, 300.0, use_fast=False)
        
        start = time.time()
        mc_f.run(n_steps=mc_iters, verbose=False)
        time_mc_f = time.time() - start
        
        sys_slow.random_configuration(seed=42)
        
        start = time.time()
        mc_s.run(n_steps=mc_iters, verbose=False)
        time_mc_s = time.time() - start
        
        speedup_mc = time_mc_s / time_mc_f
        speedups_mc.append(speedup_mc)
        
        print(f"  Energy speedup: {speedup_energy:.1f}x, MC speedup: {speedup_mc:.1f}x")
    
    # Plot scaling results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    n_spins = [s**3 for s in sizes]
    plt.loglog(n_spins, speedups_energy, 'o-', label='Energy calculation', linewidth=2)
    plt.loglog(n_spins, speedups_mc, 's-', label='Monte Carlo', linewidth=2)
    plt.xlabel('Number of spins')
    plt.ylabel('Speedup factor')
    plt.title('Numba Speedup vs System Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    operations = ['Energy\nCalculation', 'Monte Carlo\nSimulation', 'LLG\nDynamics']
    speedups = [speedup, mc_speedup, llg_speedup]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = plt.bar(operations, speedups, color=colors, alpha=0.7)
    plt.ylabel('Speedup factor')
    plt.title('Numba Speedup for Different Operations\n(20³ = 8000 spins)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, speedup_val in zip(bars, speedups):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup_val:.1f}x',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('numba_speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSpeedup analysis plot saved as 'numba_speedup_analysis.png'")
    
    plt.show()
    
    print("\n4. PERFORMANCE RECOMMENDATIONS")
    print("-" * 35)
    
    print("Based on the benchmarks:")
    print(f"• Energy calculations are {speedup:.1f}x faster with Numba")
    print(f"• Monte Carlo simulations are {mc_speedup:.1f}x faster with Numba")
    print(f"• LLG dynamics are {llg_speedup:.1f}x faster with Numba")
    print()
    print("Recommendations:")
    print("• Always use use_fast=True (default) for production runs")
    print("• Numba provides larger speedups for bigger systems")
    print("• First-time compilation may add ~1-2 seconds startup time")
    print("• Memory usage is similar between NumPy and Numba versions")
    
    print("\nDemo completed successfully!")
    print("The Numba acceleration is now integrated into all SpinLab components.")


if __name__ == "__main__":
    main()