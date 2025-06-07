# SpinLab

A comprehensive Python package for spin simulations and magnetic analysis, supporting Monte Carlo simulations, Landau-Lifshitz-Gilbert (LLG) spin dynamics, ground state optimization, and thermodynamic analysis.

## Features

- **High Performance**: Numba JIT compilation provides 10-100x speedup over pure NumPy
- **Monte Carlo Simulations**: Metropolis-Hastings algorithm with parallel tempering support
- **LLG Spin Dynamics**: Multiple integration schemes for time evolution
- **Ground State Optimization**: Various algorithms (L-BFGS, conjugate gradient, simulated annealing, genetic algorithms)
- **Thermodynamic Analysis**: Phase transitions, critical phenomena, finite-size scaling
- **Flexible Hamiltonians**: Support for exchange, anisotropy, DMI, Zeeman, and custom terms
- **Multiple Magnetic Models**: Ising, XY, and Heisenberg models
- **ASE Integration**: Compatible with Atomic Simulation Environment
- **Modern Architecture**: Clean, modular design with comprehensive documentation

## Installation

### From PyPI (Recommended)

```bash
pip install spinlab-sim
```

### From GitHub

```bash
pip install git+https://github.com/Akram-Ibrahim/SpinLab.git
```

### From Source

```bash
git clone https://github.com/Akram-Ibrahim/SpinLab.git
cd SpinLab
pip install -e .
```

### Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- ASE >= 3.21.0
- matplotlib >= 3.3.0
- pandas >= 1.3.0
- tqdm >= 4.60.0
- h5py >= 3.1.0
- numba >= 0.56.0 (optional, for performance)

## Quick Start

### Basic Monte Carlo Simulation

```python
import numpy as np
from ase.build import bulk
import spinlab
from spinlab import SpinSystem, MonteCarlo, check_numba_availability
from spinlab.core.hamiltonian import Hamiltonian

# Check if Numba acceleration is available
numba_available, message = check_numba_availability()
print(f"Numba status: {message}")

# Create a simple cubic lattice
structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
structure = structure.repeat((10, 10, 10))

# Define Hamiltonian with nearest-neighbor exchange
hamiltonian = Hamiltonian()
hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")  # Ferromagnetic

# Create spin system (use_fast=True by default for automatic acceleration)
spin_system = SpinSystem(structure, hamiltonian, magnetic_model="heisenberg")
spin_system.get_neighbors([3.0])  # Find neighbors within 3 Å

# Initialize random configuration
spin_system.random_configuration()

# Run Monte Carlo simulation (automatically uses Numba if available)
mc = MonteCarlo(spin_system, temperature=300.0)
results = mc.run(n_steps=10000, equilibration_steps=1000)

print(f"Final energy: {results['final_energy']:.4f} eV")
print(f"Final magnetization: {results['final_magnetization']}")
```

### LLG Spin Dynamics

```python
from spinlab import LLGSolver

# Create LLG solver
llg = LLGSolver(spin_system, damping=0.01, gyromagnetic_ratio=1.76e11)

# Set initial configuration
spin_system.ferromagnetic_configuration(theta=10, phi=0)  # Slightly tilted

# Run dynamics
results = llg.run(total_time=1e-9, dt=1e-15)

print(f"Time evolution over {results['times'][-1]:.2e} seconds")
```

### Ground State Optimization

```python
from spinlab import SpinOptimizer

# Create optimizer
optimizer = SpinOptimizer(spin_system, method="lbfgs")

# Find ground state
results = optimizer.find_ground_state(n_attempts=10)

print(f"Ground state energy: {results['ground_state_energy']:.6f} eV")
```

### Thermodynamic Analysis

```python
from spinlab import ThermodynamicsAnalyzer

# Run simulations at multiple temperatures
temperatures = np.linspace(10, 500, 25)
simulation_results = []

for T in temperatures:
    mc = MonteCarlo(spin_system, T)
    result = mc.run(n_steps=5000)
    simulation_results.append(result)

# Analyze thermodynamics
analyzer = ThermodynamicsAnalyzer()
properties = analyzer.calculate_thermodynamic_properties(
    temperatures, simulation_results, len(structure)
)

# Find critical temperature
critical_info = analyzer.find_critical_temperature()
print(f"Critical temperature: {critical_info['critical_temperature']:.1f} K")

# Plot results
analyzer.plot_thermodynamic_properties()
```

## Command Line Interface

SpinLab provides a convenient CLI for common tasks:

```bash
# Monte Carlo simulation
spinlab mc structure.cif -T 300 -J -0.01 -n 10000

# LLG dynamics
spinlab llg structure.cif -t 1e-9 -dt 1e-15 --damping 0.01

# Ground state optimization
spinlab optimize structure.cif -m lbfgs -n 10

# Thermodynamic analysis
spinlab thermo structure.cif -T 10 500 20 -n 5000
```

## Advanced Usage

### Custom Hamiltonians

```python
from spinlab.core.hamiltonian import Hamiltonian

# Create complex Hamiltonian
hamiltonian = Hamiltonian()

# Isotropic exchange (multiple shells)
hamiltonian.add_exchange(J1=-0.01, neighbor_shell="shell_1")
hamiltonian.add_exchange(J2=0.002, neighbor_shell="shell_2")

# Anisotropic exchange
coupling_matrix = np.array([[-0.01, 0, 0],
                           [0, -0.01, 0],
                           [0, 0, -0.015]])
hamiltonian.add_anisotropic_exchange(coupling_matrix)

# Single-ion anisotropy
hamiltonian.add_single_ion_anisotropy(K=-0.001, axis=[0, 0, 1])

# Dzyaloshinskii-Moriya interaction
hamiltonian.add_dmi(D_vector=[0, 0, 0.001])

# External magnetic field
hamiltonian.add_zeeman(B_field=[0, 0, 1.0], g_factor=2.0)
```

### Parallel Tempering

```python
from spinlab.core.monte_carlo import ParallelTempering

# Setup temperature ladder
temperatures = np.logspace(1, 3, 10)  # 10 to 1000 K

# Run parallel tempering
pt = ParallelTempering(spin_system, temperatures)
results = pt.run(n_steps=10000)

# Analyze results for each temperature
for T in temperatures:
    result = results[f'T_{T}']
    print(f"T = {T:.1f} K: E = {result['final_energy']:.4f} eV")
```

### Phase Diagrams

```python
# Scan parameter space
J_values = np.linspace(-0.02, 0.01, 10)
B_values = np.linspace(0, 2.0, 10)

phase_data = {}
for J in J_values:
    for B in B_values:
        # Update Hamiltonian
        hamiltonian = Hamiltonian()
        hamiltonian.add_exchange(J)
        hamiltonian.add_zeeman([0, 0, B])
        
        # Run simulation
        spin_system = SpinSystem(structure, hamiltonian)
        mc = MonteCarlo(spin_system, temperature=100.0)
        result = mc.run(n_steps=5000)
        
        phase_data[(J, B)] = result

# Analyze phase diagram
analyzer = ThermodynamicsAnalyzer()
phase_diagram = analyzer.analyze_phase_diagram(
    {'J': J_values, 'B': B_values}, 
    phase_data
)
```

## Documentation

Comprehensive documentation is available at [https://spinlab.readthedocs.io](https://spinlab.readthedocs.io)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use SpinLab in your research, please cite:

```bibtex
@software{spinlab2024,
  title={SpinLab: A Comprehensive Python Package for Spin Simulations},
  author={Akram Ibrahim},
  year={2024},
  url={https://github.com/Akram-Ibrahim/SpinLab}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on methods from the SpinMCPack project
- Inspired by the Atomic Simulation Environment (ASE)
- Thanks to the scientific Python community

## Performance

SpinLab is designed for high performance with automatic Numba JIT compilation:

### Speedup Benchmarks

| Operation | System Size | NumPy Time | Numba Time | Speedup |
|-----------|-------------|------------|------------|---------|
| Energy Calculation | 8,000 spins | 0.45s | 0.012s | **37x** |
| Monte Carlo Step | 8,000 spins | 2.1s | 0.089s | **24x** |
| LLG Dynamics | 8,000 spins | 1.8s | 0.067s | **27x** |

### Performance Tips

```python
# Enable automatic acceleration (default)
spin_system = SpinSystem(structure, hamiltonian, use_fast=True)
mc = MonteCarlo(spin_system, temperature=300.0, use_fast=True)
llg = LLGSolver(spin_system, use_fast=True)

# Check performance
from spinlab.utils.performance import run_performance_test
results = run_performance_test()

# Benchmark your specific system
from spinlab.utils.performance import benchmark_numba_speedup
benchmark_results = benchmark_numba_speedup(system_sizes=[10, 20, 30])
```

## Examples

See the `examples/` directory for complete examples including:

- Magnetic phase transitions in 2D Ising model
- Skyrmion dynamics in chiral magnets
- Critical behavior in Heisenberg systems
- Finite-size scaling analysis
- Custom Hamiltonian implementations
- **Numba performance demonstrations**