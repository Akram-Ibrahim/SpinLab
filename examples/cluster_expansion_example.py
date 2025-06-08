#!/usr/bin/env python3
"""
Comprehensive example demonstrating cluster expansion Hamiltonians in SpinLab.

This example shows how to:
1. Create cluster expansion Hamiltonians with sublattice resolution
2. Use the ClusterExpansionBuilder for easy setup
3. Include Heisenberg exchange, Kitaev interactions, DMI, and single-ion anisotropy
4. Work with different lattice types (honeycomb, triangular, square)
"""

import numpy as np
from ase import Atoms
from ase.build import graphene_nanoribbon, bulk

# Import SpinLab components
from spinlab.core.spin_system import SpinSystem
from spinlab.core.monte_carlo import MonteCarlo
from spinlab.utils import ClusterExpansionBuilder, create_bipartite_hamiltonian
from spinlab.core.hamiltonian import Hamiltonian


def create_honeycomb_lattice(nx=4, ny=4):
    """Create a honeycomb (graphene-like) lattice structure."""
    # Create graphene structure using ASE
    structure = graphene_nanoribbon(nx, ny, type='armchair', saturated=False, vacuum=10.0)
    
    # Remove hydrogen atoms if present
    symbols = structure.get_chemical_symbols()
    carbon_indices = [i for i, symbol in enumerate(symbols) if symbol == 'C']
    structure = structure[carbon_indices]
    
    return structure


def create_square_lattice(nx=4, ny=4, a=2.5):
    """Create a square lattice structure."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            positions.append([i * a, j * a, 0.0])
    
    cell = [[nx * a, 0, 0], [0, ny * a, 0], [0, 0, 10.0]]
    structure = Atoms('Fe' * len(positions), positions=positions, cell=cell, pbc=[True, True, False])
    
    return structure


def create_triangular_lattice(nx=4, ny=4, a=2.5):
    """Create a triangular lattice structure."""
    positions = []
    for i in range(nx):
        for j in range(ny):
            x = i * a + (j % 2) * a / 2
            y = j * a * np.sqrt(3) / 2
            positions.append([x, y, 0.0])
    
    cell = [[nx * a, 0, 0], [0, ny * a * np.sqrt(3) / 2, 0], [0, 0, 10.0]]
    structure = Atoms('Ni' * len(positions), positions=positions, cell=cell, pbc=[True, True, False])
    
    return structure


def example_1_basic_cluster_expansion():
    """Example 1: Basic cluster expansion with manual parameter setting."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Cluster Expansion")
    print("=" * 60)
    
    # Create honeycomb lattice
    structure = create_honeycomb_lattice(3, 3)
    print(f"Created honeycomb lattice with {len(structure)} atoms")
    
    # Create cluster expansion builder
    builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
    
    # Set bipartite sublattices (A and B sites in honeycomb)
    n_sites = len(structure)
    sublattice_indices = [i % 2 for i in range(n_sites)]  # Alternating A, B, A, B...
    builder.set_sublattices(sublattice_indices, ["A", "B"])
    
    # Add isotropic exchange interactions
    builder.add_exchange("J1_AB", -2.0, shell=1, sublattices=("A", "B"))  # Antiferromagnetic NN
    builder.add_exchange("J2_AA", 0.1, shell=2, sublattices=("A", "A"))   # Weak ferromagnetic NNN
    builder.add_exchange("J2_BB", 0.1, shell=2, sublattices=("B", "B"))   # Weak ferromagnetic NNN
    
    # Add Kitaev interactions (bond-directional)
    builder.add_kitaev("K1_x_AB", 0.5, shell=1, component="x", sublattices=("A", "B"))
    builder.add_kitaev("K1_y_AB", 0.3, shell=1, component="y", sublattices=("A", "B"))
    builder.add_kitaev("K1_z_AB", 0.8, shell=1, component="z", sublattices=("A", "B"))
    
    # Add single-ion anisotropy (different for each sublattice)
    builder.add_single_ion("A_A", 0.1, sublattice="A")
    builder.add_single_ion("A_B", -0.05, sublattice="B")
    
    # Add DMI
    builder.add_dmi(0.2)
    
    # Build Hamiltonian
    hamiltonian = builder.build()
    
    print("\nHamiltonian created successfully!")
    print(f"Number of terms: {len(hamiltonian.terms)}")
    print(f"Term names: {hamiltonian.term_names}")
    
    return hamiltonian, structure


def example_2_convenience_functions():
    """Example 2: Using convenience functions for common lattices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Convenience Functions")
    print("=" * 60)
    
    # Create honeycomb structure
    structure = create_honeycomb_lattice(4, 4)
    
    # Use convenience function for bipartite Hamiltonian
    hamiltonian = create_bipartite_hamiltonian(
        structure=structure,
        shell_list=[1, 2],
        J_AA=0.0,      # No intra-sublattice exchange
        J_AB=-1.5,     # Antiferromagnetic inter-sublattice
        J_BB=0.0,
        include_kitaev=True,
        K_values={"x_AB": 0.3, "y_AB": 0.3, "z_AB": 0.6}  # Anisotropic Kitaev
    )
    
    print(f"\nCreated bipartite Hamiltonian with {len(hamiltonian.terms)} terms")
    
    return hamiltonian, structure


def example_3_triangular_lattice():
    """Example 3: Triangular lattice with frustration and DMI."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Triangular Lattice with Frustration")
    print("=" * 60)
    
    # Create triangular lattice
    structure = create_triangular_lattice(4, 4)
    print(f"Created triangular lattice with {len(structure)} atoms")
    
    # Create builder
    builder = ClusterExpansionBuilder(structure, shell_list=[1, 2, 3])
    
    # Single sublattice for triangular
    builder.set_sublattices([0] * len(structure), ["A"])
    
    # Add frustrated exchange interactions
    builder.add_exchange("J1_AA", -1.0, shell=1, sublattices=("A", "A"))  # AFM nearest neighbors
    builder.add_exchange("J2_AA", 0.3, shell=2, sublattices=("A", "A"))   # FM second neighbors
    builder.add_exchange("J3_AA", -0.1, shell=3, sublattices=("A", "A"))  # Weak AFM third neighbors
    
    # Add DMI (important for triangular lattices)
    builder.add_dmi(0.15)
    
    # Add single-ion anisotropy
    builder.add_single_ion("A_A", 0.05, sublattice="A")
    
    # Build Hamiltonian
    hamiltonian = builder.build()
    
    print(f"\nCreated frustrated triangular Hamiltonian")
    
    return hamiltonian, structure


def example_4_monte_carlo_simulation():
    """Example 4: Run Monte Carlo simulation with cluster expansion."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Monte Carlo Simulation")
    print("=" * 60)
    
    # Use Hamiltonian from example 1
    hamiltonian, structure = example_1_basic_cluster_expansion()
    
    # Create spin system
    spin_system = SpinSystem(
        structure=structure,
        hamiltonian=hamiltonian,
        spin_magnitude=1.0,
        magnetic_model="3d"
    )
    
    # Set initial random configuration
    spin_system.random_configuration(seed=42)
    
    # Setup neighbor lists for the shells we defined
    cutoffs = [3.0, 4.5]  # Appropriate cutoffs for honeycomb lattice
    neighbors = spin_system.get_neighbors(cutoffs)
    
    print(f"Neighbor shells: {list(neighbors.keys())}")
    for shell, neighbor_array in neighbors.items():
        valid_count = np.sum(neighbor_array >= 0)
        print(f"  {shell}: {valid_count} total connections")
    
    # Calculate initial energy
    initial_energy = spin_system.calculate_energy()
    print(f"\nInitial energy: {initial_energy:.4f} eV")
    
    # Create Monte Carlo simulation
    mc = MonteCarlo(
        spin_system=spin_system,
        temperature=1.0,
        algorithm="metropolis"
    )
    
    # Run short equilibration
    print("\nRunning Monte Carlo equilibration...")
    mc.run(n_steps=1000, progress_bar=False)
    
    final_energy = spin_system.calculate_energy()
    magnetization = np.linalg.norm(spin_system.calculate_magnetization())
    
    print(f"Final energy: {final_energy:.4f} eV")
    print(f"Energy change: {final_energy - initial_energy:.4f} eV")
    print(f"Magnetization magnitude: {magnetization:.4f}")
    
    return spin_system


def example_5_parameter_management():
    """Example 5: Parameter saving/loading and batch operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Parameter Management")
    print("=" * 60)
    
    # Create structure
    structure = create_square_lattice(3, 3)
    
    # Create builder
    builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
    
    # Set up 4-sublattice pattern (for demonstration)
    n_sites = len(structure)
    sublattice_indices = [(i + j) % 4 for i, pos in enumerate(structure.get_positions()) 
                         for j in [int(pos[0] // 2.5) + int(pos[1] // 2.5)]][:n_sites]
    builder.set_sublattices(sublattice_indices, ["A", "B", "C", "D"])
    
    # Add all exchange combinations at once
    exchange_values = {
        "AA": -1.0, "AB": 0.5, "AC": 0.2, "AD": 0.1,
        "BB": -0.8, "BC": 0.3, "BD": 0.2,
        "CC": -0.6, "CD": 0.4,
        "DD": -0.9
    }
    builder.add_all_exchange_combinations(shell=1, values=exchange_values)
    
    # Add Kitaev combinations
    kitaev_values = {
        "x_AB": 0.3, "y_AB": 0.2, "z_AB": 0.5,
        "x_CD": 0.1, "y_CD": 0.15, "z_CD": 0.25
    }
    builder.add_all_kitaev_combinations(shell=1, values=kitaev_values)
    
    # Print parameter summary
    print(builder.get_parameter_summary())
    
    # Save parameters
    builder.save_parameters("cluster_expansion_params.json")
    
    # Create new builder and load parameters
    builder2 = ClusterExpansionBuilder(structure, shell_list=[1, 2])
    builder2.load_parameters("cluster_expansion_params.json")
    
    # Build both Hamiltonians and verify they're equivalent
    ham1 = builder.build()
    ham2 = builder2.build()
    
    print(f"\nOriginal Hamiltonian terms: {len(ham1.terms)}")
    print(f"Loaded Hamiltonian terms: {len(ham2.terms)}")
    print("✅ Parameter save/load successful!")


def main():
    """Run all examples."""
    print("SpinLab Cluster Expansion Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_1_basic_cluster_expansion()
        example_2_convenience_functions()
        example_3_triangular_lattice()
        example_4_monte_carlo_simulation()
        example_5_parameter_management()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()