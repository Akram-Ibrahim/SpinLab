#!/usr/bin/env python3
"""
Test script to validate the parallel Monte Carlo resource allocation fix.
This tests the conservative core limiting to prevent "Resource temporarily unavailable" errors.
"""

import multiprocessing as mp
import time
import numpy as np
from ase.build import bulk

# Import SpinLab components
from spinlab import SpinSystem, ParallelMonteCarlo
from spinlab.core.hamiltonian import Hamiltonian

def test_conservative_core_limit():
    """Test that ParallelMonteCarlo uses conservative core limits."""
    
    print("🧪 Testing Parallel Monte Carlo conservative core limits...")
    print(f"📊 System has {mp.cpu_count()} total CPU cores")
    
    # Create small test system
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((5, 5, 5))  # Small system for quick test
    
    # Simple ferromagnetic Hamiltonian
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
    spin_system.get_neighbors([3.0])
    spin_system.random_configuration()
    
    print(f"🔬 Test system: {len(structure)} Fe atoms")
    
    # Test different core configurations
    test_configs = [
        {"n_cores": None, "description": "Auto-detect (conservative limit)"},
        {"n_cores": mp.cpu_count(), "description": "All available cores"},
        {"n_cores": min(mp.cpu_count(), 20), "description": "Limited to 20 cores max"},
        {"n_cores": 4, "description": "Fixed 4 cores (safe test)"}
    ]
    
    for config in test_configs:
        print(f"\n🔧 Testing: {config['description']}")
        
        try:
            # Create ParallelMonteCarlo with specified cores
            pmc = ParallelMonteCarlo(spin_system, n_cores=config['n_cores'])
            actual_cores = pmc.n_cores
            
            print(f"   ✅ Initialized successfully with {actual_cores} cores")
            
            # Run small test simulation
            start_time = time.time()
            results = pmc.run(
                temperature=300.0,
                n_steps=100,  # Very short test
                n_replicas=min(actual_cores, 8),  # Limit replicas for quick test
                verbose=False
            )
            test_time = time.time() - start_time
            
            print(f"   ✅ Simulation completed in {test_time:.2f}s")
            print(f"   📊 Energy: {results['final_energy_mean']:.4f} ± {results['final_energy_sem']:.4f} eV")
            print(f"   🎯 {results['n_successful']}/{results['n_replicas']} replicas successful")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if "Resource temporarily unavailable" in str(e):
                print(f"   💡 This confirms the resource allocation issue")

def test_batched_approach():
    """Test an alternative batched approach for very high core counts."""
    
    print(f"\n🔬 Testing batched approach for high core counts...")
    
    # Create test system
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((4, 4, 4))
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
    spin_system.get_neighbors([3.0])
    spin_system.random_configuration()
    
    # Test running in batches
    total_replicas = 40
    batch_size = 8  # Conservative batch size
    n_batches = (total_replicas + batch_size - 1) // batch_size
    
    print(f"   🎯 Target: {total_replicas} total replicas")
    print(f"   📦 Strategy: {n_batches} batches of {batch_size} replicas each")
    
    all_energies = []
    all_magnetizations = []
    total_time = 0
    
    for batch_idx in range(n_batches):
        current_batch_size = min(batch_size, total_replicas - batch_idx * batch_size)
        
        print(f"   🔄 Batch {batch_idx + 1}/{n_batches}: {current_batch_size} replicas")
        
        try:
            pmc = ParallelMonteCarlo(spin_system, n_cores=current_batch_size)
            
            start_time = time.time()
            results = pmc.run(
                temperature=300.0,
                n_steps=200,
                n_replicas=current_batch_size,
                verbose=False
            )
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Collect results
            all_energies.extend(results['all_final_energies'])
            all_magnetizations.extend(results['all_final_magnetizations'])
            
            print(f"     ✅ Completed in {batch_time:.2f}s")
            
        except Exception as e:
            print(f"     ❌ Batch {batch_idx + 1} failed: {e}")
            break
    
    if len(all_energies) > 0:
        # Calculate aggregated statistics
        energies = np.array(all_energies)
        magnetizations = np.array(all_magnetizations)
        
        energy_mean = np.mean(energies)
        energy_sem = np.std(energies) / np.sqrt(len(energies))
        mag_magnitudes = np.linalg.norm(magnetizations, axis=1)
        mag_mean = np.mean(mag_magnitudes)
        mag_sem = np.std(mag_magnitudes) / np.sqrt(len(mag_magnitudes))
        
        print(f"\n📊 Batched Results Summary:")
        print(f"   ✅ Completed {len(all_energies)} replicas in {total_time:.2f}s")
        print(f"   📈 Energy: {energy_mean:.6f} ± {energy_sem:.6f} eV")
        print(f"   🧲 |M|: {mag_mean:.3f} ± {mag_sem:.3f}")
        print(f"   ⚡ Effective rate: {len(all_energies) * 200 / total_time:.0f} steps/sec")

if __name__ == "__main__":
    print("🚀 SpinLab Parallel Monte Carlo Resource Fix Test")
    print("=" * 60)
    
    # Test conservative core limits
    test_conservative_core_limit()
    
    # Test batched approach for high replica counts
    test_batched_approach()
    
    print(f"\n✅ Testing complete!")
    print(f"💡 The conservative core limit should prevent resource allocation errors.")
    print(f"🎯 For 40+ replicas, consider using the batched approach shown above.")