"""
Command-line interface for SpinLab.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from ase.io import read

from . import SpinSystem, MonteCarlo, LLGSolver, SpinOptimizer, ThermodynamicsAnalyzer
from .core.hamiltonian import Hamiltonian


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SpinLab: Comprehensive spin simulation package",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monte Carlo simulation
    mc_parser = subparsers.add_parser('mc', help='Run Monte Carlo simulation')
    mc_parser.add_argument('structure', help='Structure file (POSCAR, CIF, etc.)')
    mc_parser.add_argument('-T', '--temperature', type=float, default=300.0,
                          help='Temperature in Kelvin (default: 300)')
    mc_parser.add_argument('-J', '--exchange', type=float, default=-0.01,
                          help='Exchange coupling in eV (default: -0.01)')
    mc_parser.add_argument('-n', '--steps', type=int, default=10000,
                          help='Number of MC steps (default: 10000)')
    mc_parser.add_argument('-o', '--output', default='mc_results',
                          help='Output prefix (default: mc_results)')
    
    # LLG dynamics
    llg_parser = subparsers.add_parser('llg', help='Run LLG dynamics simulation')
    llg_parser.add_argument('structure', help='Structure file')
    llg_parser.add_argument('-t', '--time', type=float, default=1e-9,
                           help='Simulation time in seconds (default: 1e-9)')
    llg_parser.add_argument('-dt', '--timestep', type=float, default=1e-15,
                           help='Time step in seconds (default: 1e-15)')
    llg_parser.add_argument('-α', '--damping', type=float, default=0.01,
                           help='Gilbert damping parameter (default: 0.01)')
    llg_parser.add_argument('-o', '--output', default='llg_results',
                           help='Output prefix (default: llg_results)')
    
    # Optimization
    opt_parser = subparsers.add_parser('optimize', help='Find ground state')
    opt_parser.add_argument('structure', help='Structure file')
    opt_parser.add_argument('-m', '--method', default='lbfgs',
                           choices=['lbfgs', 'cg', 'sa', 'genetic'],
                           help='Optimization method (default: lbfgs)')
    opt_parser.add_argument('-n', '--attempts', type=int, default=10,
                           help='Number of optimization attempts (default: 10)')
    opt_parser.add_argument('-o', '--output', default='optimization_results',
                           help='Output prefix (default: optimization_results)')
    
    # Thermodynamics
    thermo_parser = subparsers.add_parser('thermo', help='Thermodynamic analysis')
    thermo_parser.add_argument('structure', help='Structure file')
    thermo_parser.add_argument('-T', '--temperatures', nargs=3, type=float,
                              default=[10, 500, 20],
                              help='Temperature range: min max step (default: 10 500 20)')
    thermo_parser.add_argument('-n', '--steps', type=int, default=5000,
                              help='MC steps per temperature (default: 5000)')
    thermo_parser.add_argument('-o', '--output', default='thermo_results',
                              help='Output prefix (default: thermo_results)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'mc':
            run_monte_carlo(args)
        elif args.command == 'llg':
            run_llg_dynamics(args)
        elif args.command == 'optimize':
            run_optimization(args)
        elif args.command == 'thermo':
            run_thermodynamics(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_monte_carlo(args):
    """Run Monte Carlo simulation."""
    print(f"Running Monte Carlo simulation...")
    print(f"Structure: {args.structure}")
    print(f"Temperature: {args.temperature} K")
    print(f"Exchange: {args.exchange} eV")
    print(f"Steps: {args.steps}")
    
    # Load structure
    structure = read(args.structure)
    
    # Create Hamiltonian
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(args.exchange)
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian)
    spin_system.get_neighbors(4.0)  # 4 Å cutoff
    
    # Initialize random configuration
    spin_system.random_configuration()
    
    # Run MC simulation
    mc = MonteCarlo(spin_system, args.temperature)
    results = mc.run(
        n_steps=args.steps,
        equilibration_steps=args.steps // 10,
        verbose=True
    )
    
    # Save results
    np.savez(f"{args.output}.npz", **results)
    print(f"Results saved to {args.output}.npz")
    
    # Print summary
    print(f"\nFinal energy: {results['final_energy']:.6f} eV")
    print(f"Final magnetization: {np.linalg.norm(results['final_magnetization']):.4f}")
    print(f"Acceptance rate: {results['total_acceptance_rate']:.3f}")


def run_llg_dynamics(args):
    """Run LLG dynamics simulation."""
    print(f"Running LLG dynamics...")
    print(f"Structure: {args.structure}")
    print(f"Time: {args.time} s")
    print(f"Timestep: {args.timestep} s")
    print(f"Damping: {args.damping}")
    
    # Load structure
    structure = read(args.structure)
    
    # Create Hamiltonian (simple exchange)
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(-0.01)  # Default exchange
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian)
    spin_system.get_neighbors(4.0)
    
    # Initialize random configuration
    spin_system.random_configuration()
    
    # Run LLG dynamics
    llg = LLGSolver(spin_system, damping=args.damping)
    results = llg.run(
        total_time=args.time,
        dt=args.timestep,
        verbose=True
    )
    
    # Save results
    llg.save_trajectory(f"{args.output}.npz")
    print(f"Results saved to {args.output}.npz")
    
    # Print summary
    print(f"\nFinal energy: {results['final_energy']:.6f} eV")
    print(f"Final magnetization: {np.linalg.norm(results['final_magnetization']):.4f}")


def run_optimization(args):
    """Run ground state optimization."""
    print(f"Running optimization...")
    print(f"Structure: {args.structure}")
    print(f"Method: {args.method}")
    print(f"Attempts: {args.attempts}")
    
    # Load structure
    structure = read(args.structure)
    
    # Create Hamiltonian
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(-0.01)  # Default exchange
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian)
    spin_system.get_neighbors(4.0)
    
    # Run optimization
    optimizer = SpinOptimizer(spin_system, method=args.method)
    results = optimizer.find_ground_state(
        n_attempts=args.attempts,
        verbose=True
    )
    
    # Save results
    np.savez(f"{args.output}.npz", **results)
    print(f"Results saved to {args.output}.npz")
    
    # Print summary
    print(f"\nGround state energy: {results['ground_state_energy']:.6f} eV")
    print(f"Energy spread: {results['energy_spread']:.6f} eV")
    print(f"Success rate: {results['success_rate']:.3f}")


def run_thermodynamics(args):
    """Run thermodynamic analysis."""
    print(f"Running thermodynamic analysis...")
    print(f"Structure: {args.structure}")
    
    T_min, T_max, T_step = args.temperatures
    temperatures = np.arange(T_min, T_max + T_step, T_step)
    print(f"Temperature range: {T_min} - {T_max} K ({len(temperatures)} points)")
    
    # Load structure
    structure = read(args.structure)
    
    # Create Hamiltonian
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(-0.01)  # Default exchange
    
    # Create spin system
    spin_system = SpinSystem(structure, hamiltonian)
    spin_system.get_neighbors(4.0)
    
    # Run simulations at each temperature
    simulation_results = []
    
    for temp in temperatures:
        print(f"Temperature: {temp:.1f} K")
        
        # Initialize random configuration
        spin_system.random_configuration()
        
        # Run MC simulation
        mc = MonteCarlo(spin_system, temp)
        result = mc.run(
            n_steps=args.steps,
            equilibration_steps=args.steps // 10,
            verbose=False
        )
        
        simulation_results.append(result)
    
    # Analyze thermodynamics
    analyzer = ThermodynamicsAnalyzer()
    properties = analyzer.calculate_thermodynamic_properties(
        temperatures, simulation_results, len(structure)
    )
    
    # Find critical temperature
    try:
        critical_info = analyzer.find_critical_temperature()
        print(f"\nCritical temperature: {critical_info['critical_temperature']:.1f} K")
    except:
        print("\nCould not determine critical temperature")
    
    # Save results
    analyzer.export_data(f"{args.output}.npz")
    print(f"Results saved to {args.output}.npz")
    
    # Create plots
    try:
        analyzer.plot_thermodynamic_properties(save_path=f"{args.output}_plot.png")
        print(f"Plots saved to {args.output}_plot.png")
    except:
        print("Could not create plots (matplotlib might not be available)")


if __name__ == "__main__":
    main()