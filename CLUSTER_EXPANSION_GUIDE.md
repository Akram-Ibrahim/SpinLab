# Cluster Expansion Hamiltonians in SpinLab

This guide shows how to use SpinLab's comprehensive cluster expansion functionality to create sophisticated magnetic Hamiltonians with sublattice resolution.

## Overview

SpinLab's cluster expansion system supports:

- **Isotropic Heisenberg exchange** J_k for shells k=1...n
- **Bond-directional Kitaev interactions** K_{k,γ} (γ = x,y,z)
- **Out-of-plane DMI** D_z
- **Single-ion anisotropy** A
- **Sublattice-resolved versions** of all interactions (e.g., J_1^AA, J_1^AB for bipartite lattices)

## Quick Start

### Basic Usage

```python
from ase import Atoms
from spinlab import ClusterExpansionBuilder, SpinSystem

# Create your structure (ASE Atoms object)
structure = Atoms('Fe4', positions=[[0,0,0], [1,0,0], [0,1,0], [1,1,0]], 
                  cell=[2, 2, 10], pbc=[True, True, False])

# Create cluster expansion builder
builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])

# Set sublattices (optional - defaults to single sublattice)
builder.set_sublattices([0, 1, 0, 1], ["A", "B"])  # Bipartite lattice

# Add interactions
builder.add_exchange("J1_AB", -2.0, shell=1, sublattices=("A", "B"))
builder.add_kitaev("K1_z_AB", 0.5, shell=1, component="z", sublattices=("A", "B"))
builder.add_single_ion("A_A", 0.1, sublattice="A")
builder.add_dmi(0.2)

# Build Hamiltonian
hamiltonian = builder.build()

# Use with SpinSystem
spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
```

### Convenience Functions

For common lattice types, use built-in convenience functions:

```python
from spinlab import create_bipartite_hamiltonian

# Honeycomb/square lattice with Kitaev interactions
hamiltonian = create_bipartite_hamiltonian(
    structure=structure,
    shell_list=[1, 2],
    J_AB=-1.5,     # Antiferromagnetic nearest neighbor
    include_kitaev=True,
    K_values={"x_AB": 0.3, "y_AB": 0.3, "z_AB": 0.6}
)
```

## Detailed Usage

### 1. Setting Up Sublattices

```python
# For bipartite lattice (honeycomb, square antiferromagnet)
n_sites = len(structure)
sublattice_indices = [i % 2 for i in range(n_sites)]
builder.set_sublattices(sublattice_indices, ["A", "B"])

# For more complex sublattices
sublattice_indices = [0, 1, 2, 0, 1, 2, ...]  # 3-sublattice system
builder.set_sublattices(sublattice_indices, ["A", "B", "C"])
```

### 2. Adding Exchange Interactions

```python
# Individual exchange parameters
builder.add_exchange("J1_AA", -1.0, shell=1, sublattices=("A", "A"))
builder.add_exchange("J1_AB", 2.0, shell=1, sublattices=("A", "B"))
builder.add_exchange("J2_AA", 0.1, shell=2, sublattices=("A", "A"))

# Batch addition of all combinations
exchange_values = {
    "AA": -1.0, "AB": 2.0, "BB": -0.8
}
builder.add_all_exchange_combinations(shell=1, values=exchange_values)
```

### 3. Adding Kitaev Interactions

Kitaev interactions are bond-directional (γ-bonds couple γ-components):

```python
# Individual Kitaev parameters
builder.add_kitaev("K1_x_AB", 0.5, shell=1, component="x", sublattices=("A", "B"))
builder.add_kitaev("K1_y_AB", 0.3, shell=1, component="y", sublattices=("A", "B"))
builder.add_kitaev("K1_z_AB", 0.8, shell=1, component="z", sublattices=("A", "B"))

# Batch addition
kitaev_values = {
    "x_AB": 0.5, "y_AB": 0.3, "z_AB": 0.8,
    "x_AA": 0.1, "y_AA": 0.1, "z_AA": 0.2
}
builder.add_all_kitaev_combinations(shell=1, values=kitaev_values)

# Manual bond direction assignment (for complex lattices)
builder.set_bond_direction(site_i=0, site_j=1, direction="x")  # x-type bond
builder.set_bond_direction(site_i=2, site_j=3, direction="y")  # y-type bond
```

### 4. Single-Ion Anisotropy

```python
# Different anisotropy for each sublattice
builder.add_single_ion("A_A", 0.1, sublattice="A")   # Easy-axis for A sites
builder.add_single_ion("A_B", -0.05, sublattice="B") # Easy-plane for B sites
```

### 5. Dzyaloshinskii-Moriya Interaction

```python
# Out-of-plane DMI (typically for first neighbors only)
builder.add_dmi(0.2)  # D_z = 0.2 eV
```

## Parameter Management

### Saving and Loading Parameters

```python
# Save parameters to JSON file
builder.save_parameters("my_hamiltonian.json")

# Load parameters into new builder
new_builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
new_builder.load_parameters("my_hamiltonian.json")
hamiltonian = new_builder.build()
```

### Parameter Summary

```python
# Get summary of all parameters
print(builder.get_parameter_summary())
```

## Physics Examples

### 1. Frustrated Triangular Lattice

```python
# J1-J2 model on triangular lattice with DMI
builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
builder.set_sublattices([0] * len(structure), ["A"])  # Single sublattice

builder.add_exchange("J1_AA", -1.0, shell=1, sublattices=("A", "A"))  # AFM NN
builder.add_exchange("J2_AA", 0.3, shell=2, sublattices=("A", "A"))   # FM NNN
builder.add_dmi(0.15)  # DMI favors spirals
builder.add_single_ion("A_A", 0.05, sublattice="A")  # Easy-axis anisotropy

hamiltonian = builder.build()
```

### 2. Kitaev Honeycomb Model

```python
# Pure Kitaev model on honeycomb lattice
builder = ClusterExpansionBuilder(structure, shell_list=[1])
builder.set_sublattices([i % 2 for i in range(len(structure))], ["A", "B"])

# Kitaev interactions (bond-directional)
builder.add_kitaev("K1_x_AB", 1.0, shell=1, component="x", sublattices=("A", "B"))
builder.add_kitaev("K1_y_AB", 1.0, shell=1, component="y", sublattices=("A", "B"))
builder.add_kitaev("K1_z_AB", 1.0, shell=1, component="z", sublattices=("A", "B"))

hamiltonian = builder.build()
```

### 3. Heisenberg-Kitaev Model

```python
# Combination of Heisenberg and Kitaev interactions
builder = ClusterExpansionBuilder(structure, shell_list=[1])
builder.set_sublattices([i % 2 for i in range(len(structure))], ["A", "B"])

# Heisenberg exchange
builder.add_exchange("J1_AB", -1.0, shell=1, sublattices=("A", "B"))

# Kitaev interactions
K = 0.5  # Kitaev coupling strength
builder.add_kitaev("K1_x_AB", K, shell=1, component="x", sublattices=("A", "B"))
builder.add_kitaev("K1_y_AB", K, shell=1, component="y", sublattices=("A", "B"))
builder.add_kitaev("K1_z_AB", K, shell=1, component="z", sublattices=("A", "B"))

hamiltonian = builder.build()
```

### 4. Complex Multi-Sublattice System

```python
# 4-sublattice system with full interaction matrix
builder = ClusterExpansionBuilder(structure, shell_list=[1, 2])
builder.set_sublattices(sublattice_indices, ["A", "B", "C", "D"])

# All exchange combinations
exchange_matrix = {
    "AA": -1.0, "AB": 0.5, "AC": 0.3, "AD": 0.1,
    "BB": -0.8, "BC": 0.4, "BD": 0.2,
    "CC": -0.9, "CD": 0.6,
    "DD": -0.7
}
builder.add_all_exchange_combinations(shell=1, values=exchange_matrix)

# Sublattice-specific single-ion anisotropy
anisotropies = {"A": 0.1, "B": -0.05, "C": 0.08, "D": -0.02}
for sublattice, value in anisotropies.items():
    builder.add_single_ion(f"A_{sublattice}", value, sublattice=sublattice)

hamiltonian = builder.build()
```

## Integration with SpinLab

The cluster expansion Hamiltonian integrates seamlessly with all SpinLab functionality:

```python
# Monte Carlo simulations
from spinlab import MonteCarlo

spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
spin_system.random_configuration(seed=42)
neighbors = spin_system.get_neighbors(cutoffs=[3.0, 4.5])  # For shells 1, 2

mc = MonteCarlo(spin_system, temperature=1.0)
mc.run(n_steps=10000)

# LLG dynamics
from spinlab import LLGSolver

llg = LLGSolver(spin_system, alpha=0.1, gamma=1.76e11)
llg.run(dt=1e-15, n_steps=1000)

# Thermodynamic analysis
from spinlab import ThermodynamicsAnalyzer

analyzer = ThermodynamicsAnalyzer(spin_system)
temps = np.linspace(0.1, 5.0, 50)
cv, chi = analyzer.heat_capacity_susceptibility(temps, n_samples=1000)
```

## Advanced Features

### Automatic Bond Direction Detection

For Kitaev interactions, SpinLab can automatically determine bond directions based on lattice geometry:

```python
builder = ClusterExpansionBuilder(structure, shell_list=[1], auto_bond_directions=True)
# Bond directions assigned based on primary coordinate differences
```

### Manual Bond Direction Control

For complex lattices, manually specify bond directions:

```python
builder = ClusterExpansionBuilder(structure, shell_list=[1], auto_bond_directions=False)

# Manually assign each bond type
for i, j in bond_pairs:
    # Determine bond type based on your lattice geometry
    if is_x_bond(i, j):
        builder.set_bond_direction(i, j, "x")
    elif is_y_bond(i, j):
        builder.set_bond_direction(i, j, "y")
    else:
        builder.set_bond_direction(i, j, "z")
```

### Parameter Validation

The builder automatically validates parameters and provides helpful error messages:

```python
try:
    builder.add_kitaev("K1_w_AB", 0.5, shell=1, component="w", sublattices=("A", "B"))
except ValueError as e:
    print(e)  # "Kitaev component must be 'x', 'y', or 'z'"
```

## Best Practices

1. **Always specify sublattices** for bipartite/multipartite systems
2. **Use meaningful parameter names** (e.g., "J1_AB" not "exchange1")  
3. **Save parameters** to JSON files for reproducibility
4. **Check parameter summaries** before building Hamiltonians
5. **Use appropriate cutoffs** when getting neighbors for your shell definitions
6. **Test with simple configurations** before running long simulations

## Troubleshooting

### Common Issues

1. **Missing neighbor shells**: Ensure cutoffs in `get_neighbors()` match your shell definitions
2. **Sublattice mismatches**: Check that sublattice indices match your structure
3. **Bond direction errors**: For Kitaev interactions, verify bond directions are set correctly
4. **Parameter name conflicts**: Use unique, descriptive parameter names

### Debugging Tips

```python
# Check parameter summary
print(builder.get_parameter_summary())

# Verify neighbor shells
neighbors = spin_system.get_neighbors(cutoffs)
for shell, neighbor_array in neighbors.items():
    print(f"{shell}: {np.sum(neighbor_array >= 0)} connections")

# Test energy calculation
energy = spin_system.calculate_energy()
print(f"Energy: {energy} eV")
```

This cluster expansion system provides the flexibility to model complex magnetic systems while maintaining ease of use through the builder pattern and convenience functions.