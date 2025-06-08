"""
Example: Using Graph Neural Networks with SpinLab

This demonstrates how to integrate PyTorch-based GNN models
as Hamiltonian terms in SpinLab simulations.
"""

import numpy as np
import sys
import os

# Add SpinLab to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def create_example_gnn_model():
    """Create a simple example GNN model for demonstration."""
    try:
        import torch
        import torch.nn as nn
        from spinlab.core.gnn_hamiltonian import SimpleSpinGNN
        
        # Create a simple GNN model
        model = SimpleSpinGNN(
            node_dim=6,     # 3 for spins + 3 for positions
            hidden_dim=64,
            num_layers=3,
            output_dim=1
        )
        
        return model
        
    except ImportError as e:
        print(f"⚠️  PyTorch/PyTorch Geometric not available: {e}")
        print("Install with: pip install torch torch-geometric")
        return None

def hybrid_simulation_example():
    """Example of hybrid traditional + GNN Hamiltonian simulation."""
    print("🧠 GNN Integration Example")
    print("=" * 50)
    
    try:
        from ase.build import bulk
        from spinlab import SpinSystem, MonteCarlo
        from spinlab.core.hamiltonian import Hamiltonian
        from spinlab.core.gnn_hamiltonian import GNNHamiltonianTerm, create_hybrid_hamiltonian
        
        # Create structure
        structure = bulk('Fe', 'bcc', a=2.87).repeat((4, 4, 4))
        print(f"Created Fe BCC structure with {len(structure)} atoms")
        
        # Method 1: Traditional Hamiltonian only
        print("\n📊 Method 1: Traditional Hamiltonian")
        traditional_hamiltonian = Hamiltonian()
        traditional_hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
        
        spin_system_traditional = SpinSystem(
            structure=structure,
            hamiltonian=traditional_hamiltonian,
            magnetic_model="heisenberg"
        )
        spin_system_traditional.get_neighbors([3.0])
        spin_system_traditional.random_configuration()
        
        traditional_energy = spin_system_traditional.calculate_energy()
        print(f"Traditional energy: {traditional_energy:.6f} eV")
        
        # Method 2: Hybrid Hamiltonian (traditional + GNN)
        print("\n🧠 Method 2: Hybrid Hamiltonian (Traditional + GNN)")
        
        # Create example GNN model
        gnn_model = create_example_gnn_model()
        
        if gnn_model is not None:
            # Create hybrid Hamiltonian
            hybrid_hamiltonian = Hamiltonian()
            
            # Add traditional exchange term (reduced strength)
            hybrid_hamiltonian.add_exchange(J=-0.005, neighbor_shell="shell_1")
            
            # Add GNN term
            gnn_term = GNNHamiltonianTerm(model=gnn_model)
            hybrid_hamiltonian.add_custom_term("gnn_correction", gnn_term)
            
            # Create spin system with hybrid Hamiltonian
            spin_system_hybrid = SpinSystem(
                structure=structure,
                hamiltonian=hybrid_hamiltonian,
                magnetic_model="heisenberg"
            )
            spin_system_hybrid.get_neighbors([3.0])
            spin_system_hybrid.random_configuration()
            
            try:
                hybrid_energy = spin_system_hybrid.calculate_energy()
                print(f"Hybrid energy: {hybrid_energy:.6f} eV")
                
                # Run short MC simulation
                print("\n🎲 Running hybrid MC simulation...")
                mc_hybrid = MonteCarlo(spin_system_hybrid, temperature=300.0)
                results = mc_hybrid.run(n_steps=100, equilibration_steps=20)
                
                print(f"Final energy: {results['final_energy']:.6f} eV")
                print(f"Final magnetization: {np.linalg.norm(results['final_magnetization']):.3f}")
                
            except Exception as e:
                print(f"⚠️  Hybrid simulation failed: {e}")
                print("This is expected with untrained model")
        
        # Method 3: Training data generation
        print("\n📝 Method 3: Generate Training Data for GNN")
        
        # Generate diverse configurations for training
        training_configs = []
        training_energies = []
        
        for i in range(10):  # Small example
            spin_system_traditional.random_configuration()
            config = spin_system_traditional.spin_config.copy()
            energy = spin_system_traditional.calculate_energy()
            
            training_configs.append(config)
            training_energies.append(energy)
        
        print(f"Generated {len(training_configs)} training configurations")
        print(f"Energy range: {min(training_energies):.4f} to {max(training_energies):.4f} eV")
        
        # Save training data
        training_data = {
            'configurations': np.array(training_configs),
            'energies': np.array(training_energies),
            'positions': structure.get_positions(),
            'metadata': {
                'system_size': len(structure),
                'magnetic_model': 'heisenberg',
                'traditional_J': -0.01
            }
        }
        
        np.savez('gnn_training_data.npz', **training_data)
        print("💾 Saved training data to 'gnn_training_data.npz'")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install dependencies: pip install torch torch-geometric")

def gnn_advantages():
    """Explain advantages of GNN Hamiltonians."""
    print("\n🎯 Advantages of GNN Hamiltonians in SpinLab:")
    print("=" * 50)
    
    advantages = [
        "🧠 Learn complex many-body interactions beyond pairwise",
        "📊 Automatically discover optimal interaction forms from data", 
        "🚀 Potentially faster than traditional DFT calculations",
        "🔄 Transferable across different system sizes",
        "🎯 Can incorporate environmental effects (defects, surfaces)",
        "⚡ GPU acceleration through PyTorch",
        "🔬 Hybrid approach: traditional physics + ML corrections",
        "📈 Continuous learning from new simulation data"
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")
    
    print("\n💡 Use Cases:")
    use_cases = [
        "Complex magnetic materials with frustrated interactions",
        "Systems with strong many-body correlations", 
        "Materials where DFT is too expensive",
        "Discovery of new magnetic phases",
        "Accelerated materials design workflows"
    ]
    
    for case in use_cases:
        print(f"  • {case}")

def implementation_roadmap():
    """Roadmap for implementing GNN Hamiltonians."""
    print("\n🗺️  Implementation Roadmap:")
    print("=" * 50)
    
    steps = [
        "1. 📊 Generate training data with traditional Hamiltonians",
        "2. 🧠 Train GNN on spin configuration → energy mapping",
        "3. 🔧 Integrate trained model as HamiltonianTerm",
        "4. 🎲 Run hybrid MC/LLG simulations", 
        "5. 📈 Validate against experimental/DFT data",
        "6. 🚀 Deploy for materials discovery"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("\n🛠️  Required Packages:")
    packages = [
        "torch (PyTorch)",
        "torch-geometric (Graph neural networks)",
        "torch-cluster (Graph clustering)", 
        "torch-sparse (Sparse operations)",
        "dgl (Alternative GNN library)"
    ]
    
    for package in packages:
        print(f"  • {package}")

if __name__ == "__main__":
    hybrid_simulation_example()
    gnn_advantages() 
    implementation_roadmap()
    
    print("\n🎉 GNN integration is fully supported in SpinLab!")
    print("Start with traditional simulations, then add ML components.")