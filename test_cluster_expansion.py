#!/usr/bin/env python3
"""
Quick test script for cluster expansion functionality.
"""

import numpy as np
from ase import Atoms

# Test imports
try:
    from spinlab import ClusterExpansionBuilder, create_bipartite_hamiltonian
    from spinlab.core.hamiltonian import ClusterExpansionTerm, KitaevTerm
    from spinlab import SpinSystem
    print("✅ Successfully imported cluster expansion components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)


def test_basic_functionality():
    """Test basic cluster expansion functionality."""
    print("\n🧪 Testing basic functionality...")
    
    # Create simple square lattice
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    structure = Atoms('Fe4', positions=positions, cell=[2, 2, 10], pbc=[True, True, False])
    
    # Test ClusterExpansionBuilder
    builder = ClusterExpansionBuilder(structure, shell_list=[1])
    builder.set_sublattices([0, 1, 0, 1], ["A", "B"])
    builder.add_exchange("J1_AB", -1.0, shell=1, sublattices=("A", "B"))
    builder.add_kitaev("K1_x_AB", 0.5, shell=1, component="x", sublattices=("A", "B"))
    builder.add_single_ion("A_A", 0.1, sublattice="A")
    builder.add_dmi(0.2)
    
    # Build Hamiltonian
    hamiltonian = builder.build()
    
    print(f"   Created Hamiltonian with {len(hamiltonian.terms)} terms")
    print(f"   Term names: {hamiltonian.term_names}")
    
    # Test energy calculation
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
    spin_system.ferromagnetic_configuration()
    neighbors = spin_system.get_neighbors([1.5])
    
    energy = spin_system.calculate_energy()
    print(f"   Calculated energy: {energy:.4f} eV")
    
    return True


def test_convenience_functions():
    """Test convenience functions."""
    print("\n🧪 Testing convenience functions...")
    
    # Create honeycomb-like structure
    positions = [
        [0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [1.5, 0.866, 0],
        [0, 1.732, 0], [1, 1.732, 0]
    ]
    structure = Atoms('C6', positions=positions, cell=[2, 3, 10], pbc=[True, True, False])
    
    # Test bipartite Hamiltonian
    hamiltonian = create_bipartite_hamiltonian(
        structure=structure,
        shell_list=[1],
        J_AB=-1.0,
        include_kitaev=True,
        K_values={"x_AB": 0.3, "y_AB": 0.3, "z_AB": 0.6}
    )
    
    print(f"   Created bipartite Hamiltonian with {len(hamiltonian.terms)} terms")
    
    return True


def test_parameter_management():
    """Test parameter saving/loading."""
    print("\n🧪 Testing parameter management...")
    
    # Create structure
    positions = [[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]]
    structure = Atoms('Ni3', positions=positions, cell=[2, 2, 10], pbc=[True, True, False])
    
    # Create builder and add parameters
    builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
    builder.add_exchange("J1_AA", -1.0, shell=1, sublattices=("A", "A"))
    builder.add_kitaev("K1_z_AA", 0.5, shell=1, component="z", sublattices=("A", "A"))
    
    # Test parameter summary
    summary = builder.get_parameter_summary()
    print(f"   Parameter summary generated (length: {len(summary)} chars)")
    
    # Test save/load
    builder.save_parameters("test_params.json")
    
    builder2 = ClusterExpansionBuilder(structure, shell_list=[1, 2])
    builder2.load_parameters("test_params.json")
    
    # Verify parameters match
    assert builder.J_params == builder2.J_params
    assert builder.K_params == builder2.K_params
    
    print("   ✅ Parameter save/load successful")
    
    return True


def main():
    """Run all tests."""
    print("🚀 Testing SpinLab Cluster Expansion Functionality")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_convenience_functions,
        test_parameter_management
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Cluster expansion functionality is working.")
    else:
        print("⚠️  Some tests failed. Check implementation.")


if __name__ == "__main__":
    main()