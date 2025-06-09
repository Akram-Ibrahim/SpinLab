#!/usr/bin/env python3
"""
Test the fix for monte_carlo_sweep parameter passing.
"""

import numpy as np
from ase.build import bulk
from spinlab import SpinSystem, MonteCarlo
from spinlab.core.hamiltonian import Hamiltonian

def test_basic_mc():
    """Test basic Monte Carlo functionality."""
    print("🧪 Testing Monte Carlo fix...")
    
    # Create small test system
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((4, 4, 1))  # Small system for testing
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
    spin_system.get_neighbors([3.5])
    spin_system.random_configuration()
    
    print(f"System: {len(structure)} sites")
    print("Running MC...")
    
    # Create MC and run a few steps
    mc = MonteCarlo(spin_system, temperature=100.0, use_fast=True)
    
    try:
        results = mc.run(n_steps=10, equilibration_steps=2, verbose=False)
        print("✅ Monte Carlo working correctly!")
        print(f"Final energy: {results['final_energy']:.4f} eV")
        print(f"Acceptance rate: {results['acceptance_rate']:.3f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_mc()
    if success:
        print("\n🎉 Fix successful! You can now run the speed test.")
        print("Run: python3 quick_speed_test.py")
    else:
        print("\n❌ Still having issues.")