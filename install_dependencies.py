#!/usr/bin/env python3
"""
Quick script to install SpinLab dependencies.
Run this if you get import errors.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 Installing SpinLab Dependencies")
    print("=" * 40)
    
    # Core dependencies
    dependencies = [
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.3.0",
        "ase>=3.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "h5py>=3.1.0"
    ]
    
    # Optional high-performance dependencies
    optional_deps = [
        "numba>=0.56.0"
    ]
    
    print("\n📦 Installing core dependencies...")
    failed = []
    
    for dep in dependencies:
        print(f"Installing {dep}...", end=" ")
        if install_package(dep):
            print("✅")
        else:
            print("❌")
            failed.append(dep)
    
    print("\n⚡ Installing optional dependencies...")
    for dep in optional_deps:
        print(f"Installing {dep}...", end=" ")
        if install_package(dep):
            print("✅")
        else:
            print("⚠️  (optional - will run slower without this)")
    
    if failed:
        print(f"\n❌ Failed to install: {', '.join(failed)}")
        print("Try installing manually:")
        for dep in failed:
            print(f"  pip install {dep}")
        return False
    else:
        print("\n🎉 All dependencies installed successfully!")
        print("\nNow you can run:")
        print("  python test_installation.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)