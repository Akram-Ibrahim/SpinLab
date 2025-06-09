#!/usr/bin/env python3
"""
Benchmark SpinLab performance for 24x24 system to compare with SpinMCPack.
"""

import numpy as np
import time
from ase.build import bulk
import spinlab
from spinlab import SpinSystem, MonteCarlo
from spinlab.core.hamiltonian import Hamiltonian
from spinlab.core.fast_ops import check_numba_availability

def create_24x24_system():
    """Create a 24x24 system similar to SpinMCPack setup."""
    print("Creating 24x24 spin system...")
    
    # Create a simple square lattice (similar to 2D system)
    # Using Fe BCC but making it effectively 2D by having thin z-dimension
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((24, 24, 1))  # 24x24x1 = 576 sites
    
    print(f"System size: {len(structure)} atoms")
    print(f"Lattice dimensions: {structure.get_cell().lengths()}")
    
    # Define Hamiltonian with similar parameters to typical SpinMCPack
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")  # Ferromagnetic coupling
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
    spin_system.get_neighbors([3.5])  # Find neighbors within reasonable cutoff
    
    print(f"Neighbors found: {len(spin_system._neighbors) if spin_system._neighbors else 0} shells")
    
    return spin_system

def benchmark_monte_carlo(spin_system, n_sweeps=1000, temperature=100.0):
    """Benchmark Monte Carlo performance."""
    print(f"\n🚀 Benchmarking Monte Carlo: {n_sweeps} sweeps at T={temperature}K")
    print("=" * 60)
    
    # Check Numba status
    numba_available, message = check_numba_availability()
    print(f"Numba status: {message}")
    
    # Initialize random configuration
    spin_system.random_configuration()
    
    # Create Monte Carlo instance
    mc = MonteCarlo(spin_system, temperature=temperature, use_fast=True)
    
    # Run benchmark with timing
    start_time = time.time()
    
    results = mc.run(
        n_steps=n_sweeps,
        equilibration_steps=100,  # Small equilibration
        sampling_interval=100,    # Sample every 100 steps
        verbose=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate performance metrics
    sweeps_per_second = n_sweeps / total_time
    time_per_sweep = total_time / n_sweeps
    
    print(f"\n📊 Performance Results:")
    print(f"{'='*40}")
    print(f"Total time:        {total_time:.2f} seconds")
    print(f"Time per sweep:    {time_per_sweep*1000:.2f} ms")
    print(f"Sweeps per second: {sweeps_per_second:.2f} sweep/s")
    print(f"Final energy:      {results['final_energy']:.4f} eV")
    print(f"Acceptance rate:   {results['acceptance_rate']:.3f}")
    
    # Compare with your SpinMCPack performance
    your_speed = 2.77  # sweep/s from your example
    speedup = sweeps_per_second / your_speed
    
    print(f"\n🏁 Comparison with SpinMCPack:")
    print(f"{'='*40}")
    print(f"Your SpinMCPack:   {your_speed:.2f} sweep/s")
    print(f"SpinLab:           {sweeps_per_second:.2f} sweep/s")
    print(f"Speedup factor:    {speedup:.1f}x {'🚀' if speedup > 1 else '🐌'}")
    
    return results, sweeps_per_second

def benchmark_different_sizes():
    """Benchmark different system sizes to show scaling."""
    print(f"\n📈 System Size Scaling Test")
    print("=" * 60)
    
    sizes = [(12, 12), (24, 24), (32, 32)]
    n_sweeps = 500  # Shorter test for scaling
    
    results = []
    
    for nx, ny in sizes:
        print(f"\nTesting {nx}x{ny} system ({nx*ny} sites)...")
        
        # Create system
        structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
        structure = structure.repeat((nx, ny, 1))
        
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        
        spin_system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
        spin_system.get_neighbors([3.5])
        spin_system.random_configuration()
        
        # Benchmark
        mc = MonteCarlo(spin_system, temperature=100.0, use_fast=True)
        
        start_time = time.time()
        mc_results = mc.run(n_steps=n_sweeps, equilibration_steps=50, verbose=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        sweeps_per_second = n_sweeps / total_time
        
        results.append({
            'size': f"{nx}x{ny}",
            'sites': nx * ny,
            'speed': sweeps_per_second,
            'time_per_sweep': total_time / n_sweeps * 1000  # ms
        })
        
        print(f"  Speed: {sweeps_per_second:.2f} sweep/s")
    
    print(f"\n📊 Scaling Summary:")
    print(f"{'Size':<8} {'Sites':<6} {'Speed (sweep/s)':<15} {'Time/sweep (ms)':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['size']:<8} {r['sites']:<6} {r['speed']:<15.2f} {r['time_per_sweep']:<15.2f}")

def main():
    """Run comprehensive speed benchmark."""
    print("🧪 SpinLab Performance Benchmark")
    print("Comparing with SpinMCPack performance on 24x24 system")
    print("=" * 60)
    
    # Create the 24x24 system
    spin_system = create_24x24_system()
    
    # Benchmark main performance (similar to your test)
    results, speed = benchmark_monte_carlo(
        spin_system, 
        n_sweeps=2000,  # Reasonable number for testing
        temperature=100.0
    )
    
    # Show scaling with different sizes
    benchmark_different_sizes()
    
    print(f"\n🎯 Key Takeaway:")
    print(f"{'='*40}")
    if speed > 2.77:
        improvement = speed / 2.77
        print(f"SpinLab is {improvement:.1f}x FASTER than your SpinMCPack!")
        print(f"For 250k sweeps, SpinLab would take: {250000/speed/3600:.1f} hours")
        print(f"vs SpinMCPack estimated: {250000/2.77/3600:.1f} hours")
    else:
        print(f"SpinLab performance: {speed:.2f} sweep/s")
        print(f"Room for optimization - check Numba installation")
    
    print(f"\n💡 Tips for maximum performance:")
    print("- Ensure Numba is properly installed")
    print("- Use larger systems to amortize overhead")
    print("- Consider parallel Monte Carlo for multiple replicas")

if __name__ == "__main__":
    main()