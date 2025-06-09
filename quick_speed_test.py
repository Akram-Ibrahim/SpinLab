#!/usr/bin/env python3
"""
Quick speed test to immediately see SpinLab performance vs SpinMCPack.
"""

import numpy as np
import time
from ase.build import bulk
import spinlab
from spinlab import SpinSystem, MonteCarlo
from spinlab.core.hamiltonian import Hamiltonian

def quick_24x24_test():
    """Quick test of 24x24 system performance."""
    print("⚡ Quick SpinLab Speed Test (24x24 system)")
    print("=" * 50)
    
    # Create 24x24 system
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((24, 24, 1))  # 576 sites
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
    spin_system.get_neighbors([3.5])
    spin_system.random_configuration()
    
    print(f"System: {len(structure)} sites (24x24)")
    
    # Quick benchmark: 100 sweeps with timing
    mc = MonteCarlo(spin_system, temperature=100.0, use_fast=True)
    
    print("Running 100 sweeps for speed measurement...")
    start_time = time.time()
    
    # Run just 100 sweeps to get quick estimate
    results = mc.run(n_steps=100, equilibration_steps=10, verbose=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    sweeps_per_second = 100 / total_time
    
    print(f"\n📊 Results:")
    print(f"Time for 100 sweeps: {total_time:.3f} seconds")
    print(f"Speed: {sweeps_per_second:.2f} sweep/s")
    
    # Compare with SpinMCPack
    spinmcpack_speed = 2.77
    speedup = sweeps_per_second / spinmcpack_speed
    
    print(f"\n🏁 Comparison:")
    print(f"SpinMCPack:  {spinmcpack_speed:.2f} sweep/s")
    print(f"SpinLab:     {sweeps_per_second:.2f} sweep/s")
    print(f"Speedup:     {speedup:.1f}x")
    
    if speedup > 1:
        print(f"🚀 SpinLab is {speedup:.1f}x FASTER!")
        time_for_250k = 250000 / sweeps_per_second / 3600
        print(f"250k sweeps would take: {time_for_250k:.1f} hours (vs {250000/2.77/3600:.1f} hours)")
    else:
        print(f"⚠️  SpinLab is slower - check Numba installation")
    
    return sweeps_per_second

if __name__ == "__main__":
    speed = quick_24x24_test()
    print(f"\nTo run comprehensive benchmark: python benchmark_speed.py")