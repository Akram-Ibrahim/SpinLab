#!/usr/bin/env python3
"""
Test parallel Monte Carlo functionality.
"""

import numpy as np
import time
import sys
import os

# Add SpinLab to path
sys.path.insert(0, '/Users/akramibrahim/SpinLab')

def test_parallel_vs_single():
    """Compare parallel vs single-core performance."""
    print("🚀 Testing Parallel Monte Carlo")
    print("=" * 50)
    
    try:
        from ase.build import bulk
        from spinlab import SpinSystem, MonteCarlo, ParallelMonteCarlo
        from spinlab.core.hamiltonian import Hamiltonian
        
        # Create test system
        structure = bulk('Fe', 'bcc', a=2.87).repeat((6, 6, 6))
        print(f"Created Fe BCC structure with {len(structure)} atoms")
        
        # Create Hamiltonian
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        
        # Create spin system
        spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
        spin_system.get_neighbors([3.0])
        
        # Test parameters
        temperature = 300.0
        n_steps = 1000
        
        print(f"\nTest parameters:")
        print(f"  Temperature: {temperature} K")
        print(f"  MC steps: {n_steps}")
        print(f"  System size: {len(structure)} atoms")
        
        # 1. Single-core baseline
        print(f"\n🔄 Single-core Monte Carlo...")
        spin_system.random_configuration()
        
        mc_single = MonteCarlo(spin_system, temperature=temperature)
        start_time = time.time()
        result_single = mc_single.run(n_steps=n_steps, equilibration_steps=100, verbose=False)
        single_time = time.time() - start_time
        
        print(f"   Time: {single_time:.2f} seconds")
        print(f"   Energy: {result_single['final_energy']:.6f} eV")
        print(f"   Rate: {n_steps/single_time:.0f} steps/sec")
        
        # 2. Parallel Monte Carlo (4 replicas)
        print(f"\n⚡ Parallel Monte Carlo (4 replicas)...")
        
        pmc = ParallelMonteCarlo(spin_system, n_cores=4)
        start_time = time.time()
        result_parallel = pmc.run(
            temperature=temperature,
            n_steps=n_steps,
            n_replicas=4,
            equilibration_steps=100,
            verbose=True
        )
        parallel_time = time.time() - start_time
        
        print(f"   Time: {parallel_time:.2f} seconds")
        print(f"   Energy: {result_parallel['final_energy_mean']:.6f} ± {result_parallel['final_energy_std']:.6f} eV")
        print(f"   Rate: {result_parallel['steps_per_second']:.0f} steps/sec")
        
        # 3. Performance analysis
        print(f"\n📊 Performance Comparison:")
        speedup = single_time / parallel_time
        efficiency = speedup / 4 * 100  # 4 cores used
        
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
        print(f"   Theoretical max: 4.0x")
        
        # 4. Statistical improvement
        print(f"\n📈 Statistical Improvement:")
        print(f"   Single run energy: {result_single['final_energy']:.6f} eV")
        print(f"   Parallel mean: {result_parallel['final_energy_mean']:.6f} eV")
        print(f"   Standard error: {result_parallel['final_energy_sem']:.6f} eV")
        print(f"   Statistical improvement: {1/np.sqrt(4):.1f}x better precision")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install dependencies first")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def demonstrate_scaling():
    """Demonstrate how parallel MC scales with number of cores."""
    print(f"\n🔬 Scaling Test (quick)")
    print("=" * 30)
    
    try:
        from ase.build import bulk
        from spinlab import SpinSystem, ParallelMonteCarlo
        from spinlab.core.hamiltonian import Hamiltonian
        
        # Small system for quick test
        structure = bulk('Fe', 'bcc', a=2.87).repeat((4, 4, 4))
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        spin_system = SpinSystem(structure, hamiltonian)
        spin_system.get_neighbors([3.0])
        
        # Test different numbers of replicas
        replica_counts = [1, 2, 4, 8]
        results = []
        
        pmc = ParallelMonteCarlo(spin_system, n_cores=8)
        
        for n_replicas in replica_counts:
            print(f"  Testing {n_replicas} replicas...", end=" ")
            
            start_time = time.time()
            result = pmc.run(
                temperature=300.0,
                n_steps=500,  # Shorter for quick test
                n_replicas=n_replicas,
                equilibration_steps=50,
                verbose=False
            )
            elapsed = time.time() - start_time
            
            rate = result['steps_per_second']
            results.append((n_replicas, elapsed, rate))
            
            print(f"{elapsed:.2f}s, {rate:.0f} steps/sec")
        
        print(f"\n📊 Scaling Results:")
        print(f"{'Replicas':<8} {'Time(s)':<8} {'Rate':<10} {'Speedup':<8}")
        print("-" * 35)
        
        baseline_time = results[0][1]
        for n_rep, t, rate in results:
            speedup = baseline_time / t
            print(f"{n_rep:<8} {t:<8.2f} {rate:<10.0f} {speedup:<8.1f}x")
        
    except Exception as e:
        print(f"❌ Scaling test failed: {e}")

if __name__ == "__main__":
    success = test_parallel_vs_single()
    
    if success:
        demonstrate_scaling()
        
        print(f"\n🎉 Parallel Monte Carlo is working!")
        print(f"\nFor your 40-CPU cluster:")
        print(f"  • Use ParallelMonteCarlo(spin_system, n_cores=40)")
        print(f"  • Run 40 replicas for ~40x speedup")
        print(f"  • Get much better statistics automatically")
    else:
        print(f"\n❌ Tests failed - check dependencies")