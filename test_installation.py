#!/usr/bin/env python3
"""
Test script to verify SpinLab installation and dependencies.
Run this after setting up your virtual environment.
"""

import sys
import importlib

def test_import(module_name, optional=False):
    """Test if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        status = "⚠️ " if optional else "❌"
        print(f"{status} {module_name} - {str(e)}")
        return False

def main():
    print("🧪 SpinLab Installation Test")
    print("=" * 40)
    
    # Test core SpinLab
    print("\n📦 Core SpinLab:")
    if test_import('spinlab'):
        import spinlab
        print(f"   Version: {spinlab.__version__}")
        
        # Test Numba acceleration
        available, msg = spinlab.check_numba_availability()
        print(f"   Numba: {msg}")
        
        # Test core modules
        test_import('spinlab.core.spin_system')
        test_import('spinlab.core.monte_carlo')
        test_import('spinlab.dynamics.llg_solver')
        test_import('spinlab.optimization.spin_optimizer')
        test_import('spinlab.analysis.thermodynamics')
    
    # Test scientific computing
    print("\n🔬 Scientific Computing:")
    test_import('numpy')
    test_import('scipy')
    test_import('matplotlib')
    test_import('pandas')
    test_import('h5py')
    test_import('ase')
    test_import('tqdm')
    
    # Test high performance
    print("\n⚡ High Performance:")
    test_import('numba', optional=True)
    
    # Test analysis tools
    print("\n📊 Analysis Tools:")
    test_import('jupyter', optional=True)
    test_import('plotly', optional=True)
    test_import('seaborn', optional=True)
    test_import('sklearn', optional=True)
    
    # Quick functionality test
    print("\n🎯 Quick Functionality Test:")
    try:
        from ase import Atoms
        from spinlab import SpinSystem
        from spinlab.core.hamiltonian import Hamiltonian
        
        # Create minimal test system
        positions = [[0, 0, 0], [1, 0, 0]]
        structure = Atoms('Fe2', positions=positions)
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        
        spin_system = SpinSystem(structure, hamiltonian)
        spin_system.random_configuration()
        energy = spin_system.calculate_energy()
        
        print(f"✅ Basic simulation works - Energy: {energy:.6f} eV")
        
    except Exception as e:
        print(f"❌ Basic simulation failed: {e}")
    
    print("\n🎉 Installation test complete!")
    print("\nTo start analyzing:")
    print("  jupyter notebook")
    print("  # Then open Fe_BCC_SpinLab_Test.ipynb")

if __name__ == "__main__":
    main()