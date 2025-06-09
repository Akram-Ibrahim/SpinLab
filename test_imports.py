#!/usr/bin/env python3
"""
Test script to verify all SpinLab imports work correctly.
"""

def test_basic_imports():
    """Test basic module imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        import scipy
        print("✅ NumPy and SciPy available")
    except ImportError as e:
        print(f"❌ Missing basic dependencies: {e}")
        return False
    
    return True

def test_fast_ops():
    """Test fast_ops imports specifically."""
    print("Testing fast_ops imports...")
    
    try:
        from spinlab.core.fast_ops import (
            calculate_magnetization, 
            llg_rhs, 
            normalize_spins,
            HAS_NUMBA,
            monte_carlo_sweep,
            metropolis_single_flip
        )
        print("✅ All fast_ops functions imported successfully")
        print(f"✅ Numba available: {HAS_NUMBA}")
        return True
        
    except ImportError as e:
        print(f"❌ fast_ops import error: {e}")
        return False

def test_spinlab_components():
    """Test main SpinLab components."""
    print("Testing SpinLab components...")
    
    try:
        from spinlab import SpinSystem, MonteCarlo, LLGSolver, SpinOptimizer
        from spinlab.core.hamiltonian import Hamiltonian
        print("✅ Main SpinLab components imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ SpinLab component import error: {e}")
        return False

def test_optimization():
    """Test optimization methods."""
    print("Testing optimization methods...")
    
    try:
        from spinlab.optimization import LBFGS, ConjugateGradient, SimulatedAnnealing
        print("✅ Optimization methods imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Optimization import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing SpinLab imports...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_fast_ops,
        test_spinlab_components,
        test_optimization
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All tests passed! SpinLab is working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting steps:")
        print("1. Run: python clear_cache.py")
        print("2. Restart Python kernel")
        print("3. If still failing, try: pip uninstall spinlab && pip install -e .")

if __name__ == "__main__":
    main()