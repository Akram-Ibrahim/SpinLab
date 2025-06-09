#!/usr/bin/env python3
"""
Script to clear Python cache and fix import issues after function renaming.
"""

import os
import shutil
import sys
from pathlib import Path

def clear_python_cache(directory):
    """Clear all Python cache files in directory."""
    cache_dirs = []
    pyc_files = []
    
    for root, dirs, files in os.walk(directory):
        # Find __pycache__ directories
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))
        
        # Find .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(os.path.join(root, file))
    
    # Remove cache directories
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            print(f"Removed cache directory: {cache_dir}")
        except Exception as e:
            print(f"Error removing {cache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"Removed .pyc file: {pyc_file}")
        except Exception as e:
            print(f"Error removing {pyc_file}: {e}")

def main():
    """Clear cache and test import."""
    print("🧹 Clearing Python cache files...")
    
    # Clear cache in current directory
    clear_python_cache('.')
    
    # Also clear from site-packages if spinlab is installed
    try:
        import site
        for site_dir in site.getsitepackages():
            spinlab_dir = os.path.join(site_dir, 'spinlab')
            if os.path.exists(spinlab_dir):
                print(f"Clearing cache in installed package: {spinlab_dir}")
                clear_python_cache(spinlab_dir)
    except:
        pass
    
    print("\n✅ Cache cleared!")
    
    # Test import
    print("🧪 Testing SpinLab import...")
    try:
        import spinlab
        print("✅ SpinLab imported successfully!")
        
        # Test specific problematic import
        from spinlab.core.fast_ops import calculate_magnetization, HAS_NUMBA
        print("✅ fast_ops functions imported successfully!")
        
        # Test main components
        from spinlab import SpinSystem, MonteCarlo
        print("✅ Main components imported successfully!")
        
        print("\n🎉 All imports working correctly!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTry running:")
        print("1. pip uninstall spinlab")
        print("2. pip install -e .")
        print("3. Restart Python/Jupyter kernel")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()