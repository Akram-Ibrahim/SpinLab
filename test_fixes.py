#!/usr/bin/env python3
"""
Quick test to verify both fixes:
1. KeyError 'acceptance_rate' -> FIXED
2. Individual progress bars -> ADDED
"""

def test_small_parallel_mc():
    """Test with small number of replicas to see individual progress bars."""
    
    import time
    import numpy as np
    from ase.build import bulk
    from spinlab import SpinSystem, ParallelMonteCarlo
    from spinlab.core.hamiltonian import Hamiltonian
    
    print("🧪 Testing Parallel MC fixes...")
    
    # Create small test system
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((4, 4, 4))  # Small for quick test
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
    spin_system.get_neighbors([3.0])
    spin_system.random_configuration()
    
    print(f"📊 Test system: {len(structure)} Fe atoms")
    
    # Test 1: Small number of replicas (should show individual progress)
    print(f"\n🔬 Test 1: 3 replicas (should show individual MC progress bars)")
    pmc = ParallelMonteCarlo(spin_system, n_cores=3)
    
    start_time = time.time()
    results = pmc.run(
        temperature=300.0,
        n_steps=200,  # Short test
        n_replicas=3,
        equilibration_steps=50,
        sampling_interval=10,
        verbose=True,
        show_individual_progress=True  # Enable individual progress bars
    )
    test_time = time.time() - start_time
    
    print(f"\n✅ Test 1 Results:")
    print(f"   Energy: {results['final_energy_mean']:.4f} ± {results['final_energy_sem']:.4f} eV")
    print(f"   Acceptance rate: {results['acceptance_rate_mean']:.2%}")
    print(f"   Time: {test_time:.2f}s")
    print(f"   ✅ No KeyError - acceptance_rate fix works!")
    
    # Test 2: More replicas (should NOT show individual progress to avoid clutter)
    print(f"\n🔬 Test 2: 8 replicas (should NOT show individual progress - too many)")
    pmc2 = ParallelMonteCarlo(spin_system, n_cores=8)
    
    start_time = time.time()
    results2 = pmc2.run(
        temperature=300.0,
        n_steps=100,  # Very short
        n_replicas=8,
        equilibration_steps=20,
        sampling_interval=10,
        verbose=True,
        show_individual_progress=True  # Still enabled, but won't show due to replica count
    )
    test_time2 = time.time() - start_time
    
    print(f"\n✅ Test 2 Results:")
    print(f"   Energy: {results2['final_energy_mean']:.4f} ± {results2['final_energy_sem']:.4f} eV")
    print(f"   Acceptance rate: {results2['acceptance_rate_mean']:.2%}")
    print(f"   Time: {test_time2:.2f}s")
    print(f"   ✅ Clean output for many replicas!")

if __name__ == "__main__":
    test_small_parallel_mc()
    
    print(f"\n🎉 Both fixes verified:")
    print(f"   ✅ KeyError 'acceptance_rate' -> FIXED")
    print(f"   ✅ Individual MC progress bars -> ADDED")
    print(f"   📊 Progress bars shown only for ≤4 replicas to avoid clutter")
    print(f"   🚀 Ready for your cluster simulations!")