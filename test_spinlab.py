#!/usr/bin/env python3
"""
Simple test script to demonstrate SpinLab functionality.
"""

import sys
import numpy as np

# Add SpinLab to path
sys.path.insert(0, '/Users/akramibrahim/SpinLab')

def test_basic_functionality():
    """Test basic SpinLab functionality."""
    print("Testing SpinLab basic functionality...")
    
    # Import SpinLab
    import spinlab
    print(f"✓ SpinLab {spinlab.__version__} imported successfully")
    
    # Check Numba status
    numba_available, message = spinlab.check_numba_availability()
    print(f"✓ {message}")
    
    # Create a simple test system
    try:
        from ase import Atoms
        from spinlab.core.spin_system import SpinSystem
        from spinlab.core.hamiltonian import Hamiltonian
        
        # Create simple structure (2x2x2 cubic lattice)
        positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    positions.append([i, j, k])
        positions = np.array(positions)
        
        # Create ASE Atoms object
        structure = Atoms('Fe8', positions=positions)
        
        # Create simple Hamiltonian
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        
        # Create spin system
        spin_system = SpinSystem(
            structure=structure,
            hamiltonian=hamiltonian,
            magnetic_model="heisenberg"
        )
        
        # Find neighbors
        neighbor_array = spin_system.get_neighbors([2.0])
        print(f"✓ Created spin system with {len(positions)} spins")
        
        # Initialize random configuration
        spin_system.random_configuration()
        print("✓ Initialized random spin configuration")
        
        # Calculate energy
        energy = spin_system.calculate_energy()
        print(f"✓ Total energy: {energy:.6f} eV")
        
        # Test Monte Carlo
        from spinlab.core.monte_carlo import MonteCarlo
        mc = MonteCarlo(spin_system, temperature=300.0)
        result = mc.run(n_steps=100, equilibration_steps=10)
        print(f"✓ Monte Carlo simulation completed")
        print(f"  Final energy: {result['final_energy']:.6f} eV")
        print(f"  Final magnetization: {np.linalg.norm(result['final_magnetization']):.3f}")
        
        print("\n🎉 All tests passed! SpinLab is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)