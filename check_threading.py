#!/usr/bin/env python3
"""
Quick check of threading and CPU configuration.
"""

import os
import psutil
from spinlab.core.fast_ops import HAS_NUMBA, check_numba_availability

def check_threading_config():
    """Check current threading configuration."""
    print("🧵 Threading Configuration Check")
    print("=" * 40)
    
    # Basic system info
    print(f"Physical CPU cores:    {psutil.cpu_count(logical=False)}")
    print(f"Logical CPU cores:     {psutil.cpu_count(logical=True)}")
    
    # Check Numba
    numba_available, msg = check_numba_availability()
    print(f"Numba available:       {numba_available}")
    print(f"Numba status:          {msg}")
    
    # Check environment variables
    print(f"\nEnvironment Variables:")
    print(f"OMP_NUM_THREADS:       {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"MKL_NUM_THREADS:       {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"NUMBA_NUM_THREADS:     {os.environ.get('NUMBA_NUM_THREADS', 'not set')}")
    
    # Check Numba threading
    if HAS_NUMBA:
        try:
            from numba import config
            print(f"Numba parallel target: {getattr(config, 'THREADING_LAYER', 'unknown')}")
            print(f"Numba thread count:    {getattr(config, 'NUMBA_NUM_THREADS', 'default')}")
        except:
            print("Numba config:          Unable to get details")
    
    # Current process info
    try:
        process = psutil.Process()
        print(f"\nCurrent Process:")
        print(f"CPU affinity:          {len(process.cpu_affinity())} cores")
        print(f"Memory usage:          {process.memory_info().rss / 1024**2:.1f} MB")
    except:
        print("Process info:          Unable to get details")
    
    print(f"\n💡 Your previous result: 14.19 sweep/s (5.1x speedup)")
    print(f"   This suggests Numba is working well!")

if __name__ == "__main__":
    check_threading_config()