#!/usr/bin/env python3
"""
Basic Monte Carlo simulation example using SpinLab.

This example demonstrates how to set up and run a simple Monte Carlo
simulation of a ferromagnetic Ising model on a 2D square lattice.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.build import make_supercell

# Import SpinLab components
from spinlab import SpinSystem, MonteCarlo, ThermodynamicsAnalyzer
from spinlab.core.hamiltonian import Hamiltonian


def main():
    """Run basic Monte Carlo simulation."""
    
    print("SpinLab: Basic Monte Carlo Example")
    print("=" * 40)
    
    # Create a 2D square lattice structure
    print("Creating structure...")
    structure = bulk('Fe', 'sc', a=3.0)  # Simple cubic
    
    # Make it 2D by creating a thin slab
    supercell_matrix = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 1]])
    structure = make_supercell(structure, supercell_matrix)
    
    print(f"Created structure with {len(structure)} atoms")
    
    # Define Hamiltonian with ferromagnetic exchange
    print("Setting up Hamiltonian...")
    hamiltonian = Hamiltonian()
    
    # Add nearest-neighbor ferromagnetic exchange
    J = -0.001  # Ferromagnetic coupling in eV
    hamiltonian.add_exchange(J, neighbor_shell="shell_1")
    
    print(f"Exchange coupling J = {J} eV")
    
    # Create spin system (2D Ising model)
    print("Creating spin system...")
    spin_system = SpinSystem(
        structure=structure,
        hamiltonian=hamiltonian,
        spin_magnitude=1.0,
        magnetic_model="ising"  # Ising spins (±1 along z)
    )
    
    # Find nearest neighbors
    cutoff = 3.5  # Slightly larger than lattice parameter
    neighbors = spin_system.get_neighbors([cutoff])
    print(f"Found neighbors with cutoff {cutoff} Å")
    
    # Initialize with random configuration
    spin_system.random_configuration(seed=42)
    initial_energy = spin_system.calculate_energy()
    initial_magnetization = spin_system.calculate_magnetization()
    
    print(f"Initial energy: {initial_energy:.4f} eV")
    print(f"Initial magnetization: {np.linalg.norm(initial_magnetization):.4f}")
    
    # Run Monte Carlo simulation at different temperatures
    temperatures = [50, 100, 200, 400, 800]
    results = {}
    
    print("\nRunning Monte Carlo simulations...")
    
    for temperature in temperatures:
        print(f"\nTemperature: {temperature} K")
        
        # Reset to random configuration for each temperature
        spin_system.random_configuration(seed=42)
        
        # Create Monte Carlo simulator
        mc = MonteCarlo(
            spin_system=spin_system,
            temperature=temperature,
            random_seed=42
        )
        
        # Run simulation
        result = mc.run(
            n_steps=5000,
            equilibration_steps=1000,
            sampling_interval=10,
            verbose=True
        )
        
        results[temperature] = result
        
        # Print results
        final_energy = result['final_energy']
        final_mag = np.linalg.norm(result['final_magnetization'])
        acceptance_rate = result['total_acceptance_rate']
        
        print(f"Final energy: {final_energy:.4f} eV")
        print(f"Final magnetization: {final_mag:.4f}")
        print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Analyze thermodynamics
    print("\nAnalyzing thermodynamics...")
    
    analyzer = ThermodynamicsAnalyzer()
    
    # Calculate properties for each temperature
    for temp in temperatures:
        result = results[temp]
        analyzer.add_data_point(
            temperature=temp,
            energy_data=result['energies'],
            magnetization_data=result['magnetizations'],
            n_spins=len(structure)
        )
    
    # Find critical temperature
    try:
        critical_info = analyzer.find_critical_temperature(method="specific_heat")
        Tc = critical_info['critical_temperature']
        print(f"Estimated critical temperature: {Tc:.1f} K")
        
        # Theoretical Tc for 2D Ising model: Tc = 2J/(kB * ln(1 + sqrt(2)))
        kB = 8.617333e-5  # eV/K
        J_theoretical = abs(J) * 4  # 4 nearest neighbors in 2D
        Tc_theoretical = J_theoretical / (kB * np.log(1 + np.sqrt(2)))
        print(f"Theoretical critical temperature: {Tc_theoretical:.1f} K")
        
    except Exception as e:
        print(f"Could not determine critical temperature: {e}")
    
    # Create plots
    print("\nCreating plots...")
    
    # Temperature dependence
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    temps = list(temperatures)
    energies = [results[T]['final_energy'] / len(structure) for T in temps]
    magnetizations = [np.linalg.norm(results[T]['final_magnetization']) for T in temps]
    specific_heats = analyzer.specific_heats
    
    # Energy vs Temperature
    axes[0].plot(temps, energies, 'o-', color='blue', linewidth=2)
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Energy per spin (eV)')
    axes[0].set_title('Internal Energy')
    axes[0].grid(True, alpha=0.3)
    
    # Magnetization vs Temperature
    axes[1].plot(temps, magnetizations, 'o-', color='red', linewidth=2)
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('Magnetization')
    axes[1].set_title('Magnetization')
    axes[1].grid(True, alpha=0.3)
    
    # Specific Heat vs Temperature
    axes[2].plot(temps, specific_heats, 'o-', color='green', linewidth=2)
    axes[2].set_xlabel('Temperature (K)')
    axes[2].set_ylabel('Specific Heat / kB')
    axes[2].set_title('Specific Heat')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
    print("Saved plots to 'monte_carlo_results.png'")
    
    # Show energy convergence for one temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    
    T_example = 200  # K
    energies_vs_step = results[T_example]['energies']
    steps = range(len(energies_vs_step))
    
    ax.plot(steps, energies_vs_step, color='blue', alpha=0.7)
    ax.set_xlabel('MC Step')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Energy Convergence at T = {T_example} K')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('energy_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved energy convergence plot to 'energy_convergence.png'")
    
    plt.show()
    
    print("\nSimulation completed successfully!")
    print("Check the generated plots for results visualization.")


if __name__ == "__main__":
    main()