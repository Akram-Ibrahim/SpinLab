#!/usr/bin/env python3
"""
Enhanced speed test showing CPU usage and threading information.
"""

import numpy as np
import time
import os
import psutil
import threading
from ase.build import bulk
import spinlab
from spinlab import SpinSystem, MonteCarlo
from spinlab.core.hamiltonian import Hamiltonian
from spinlab.core.fast_ops import check_numba_availability, HAS_NUMBA

def get_system_info():
    """Get system and threading information."""
    info = {
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'active_threads': threading.active_count(),
        'process_cpu_count': len(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), 'cpu_affinity') else psutil.cpu_count(),
    }
    
    # Get Numba threading info
    if HAS_NUMBA:
        try:
            from numba import config
            info['numba_threads'] = config.NUMBA_NUM_THREADS if hasattr(config, 'NUMBA_NUM_THREADS') else "default"
        except:
            info['numba_threads'] = "unknown"
    else:
        info['numba_threads'] = "N/A (no Numba)"
    
    # Check environment variables
    info['omp_threads'] = os.environ.get('OMP_NUM_THREADS', 'not set')
    info['mkl_threads'] = os.environ.get('MKL_NUM_THREADS', 'not set')
    info['numba_env_threads'] = os.environ.get('NUMBA_NUM_THREADS', 'not set')
    
    return info

def monitor_cpu_during_run(duration, interval=0.1):
    """Monitor CPU usage during a time period."""
    cpu_percentages = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        cpu_percentages.append(psutil.cpu_percent(interval=interval))
    
    return {
        'avg_cpu': np.mean(cpu_percentages),
        'max_cpu': np.max(cpu_percentages),
        'samples': len(cpu_percentages)
    }

def detailed_24x24_test():
    """Enhanced speed test with CPU monitoring."""
    print("🖥️  Detailed SpinLab Speed Test (24x24 system)")
    print("=" * 60)
    
    # System information
    sys_info = get_system_info()
    numba_available, numba_msg = check_numba_availability()
    
    print("🔧 System Configuration:")
    print(f"   Physical CPUs:     {sys_info['cpu_count_physical']}")
    print(f"   Logical CPUs:      {sys_info['cpu_count_logical']}")
    print(f"   Process CPU count: {sys_info['process_cpu_count']}")
    print(f"   Active threads:    {sys_info['active_threads']}")
    print(f"   Numba status:      {numba_msg}")
    print(f"   Numba threads:     {sys_info['numba_threads']}")
    print(f"   OMP_NUM_THREADS:   {sys_info['omp_threads']}")
    print(f"   MKL_NUM_THREADS:   {sys_info['mkl_threads']}")
    print(f"   NUMBA_NUM_THREADS: {sys_info['numba_env_threads']}")
    
    # Create 24x24 system
    print(f"\n📦 Creating System:")
    structure = bulk('Fe', 'bcc', a=2.87, cubic=True)
    structure = structure.repeat((24, 24, 1))
    
    hamiltonian = Hamiltonian()
    hamiltonian.add_exchange(J=-0.01, neighbor_shell="shell_1")
    
    spin_system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
    spin_system.get_neighbors([3.5])
    spin_system.random_configuration()
    
    print(f"   System size:       {len(structure)} sites (24x24)")
    print(f"   Neighbor shells:   {len(spin_system._neighbors) if spin_system._neighbors else 0}")
    
    # Monitor CPU usage during benchmark
    print(f"\n🚀 Running Benchmark (100 sweeps):")
    print("   Monitoring CPU usage...")
    
    mc = MonteCarlo(spin_system, temperature=100.0, use_fast=True)
    
    # Start CPU monitoring in a separate thread
    cpu_monitor_active = True
    cpu_data = []
    
    def cpu_monitor():
        while cpu_monitor_active:
            cpu_data.append(psutil.cpu_percent(interval=0.1))
    
    # Start monitoring
    monitor_thread = threading.Thread(target=cpu_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Run benchmark
    start_time = time.time()
    start_cpu_times = psutil.cpu_times()
    
    results = mc.run(n_steps=100, equilibration_steps=10, verbose=True)
    
    end_time = time.time()
    end_cpu_times = psutil.cpu_times()
    
    # Stop monitoring
    cpu_monitor_active = False
    monitor_thread.join(timeout=1)
    
    total_time = end_time - start_time
    sweeps_per_second = 100 / total_time
    
    # CPU usage analysis
    if cpu_data:
        avg_cpu = np.mean(cpu_data)
        max_cpu = np.max(cpu_data)
        cpu_samples = len(cpu_data)
    else:
        avg_cpu = max_cpu = cpu_samples = 0
    
    # Calculate CPU time used
    cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)
    cpu_efficiency = (cpu_time_used / total_time) * 100 if total_time > 0 else 0
    
    print(f"\n📊 Performance Results:")
    print(f"{'='*50}")
    print(f"Total time:        {total_time:.3f} seconds")
    print(f"Time per sweep:    {total_time/100*1000:.2f} ms")
    print(f"Sweeps per second: {sweeps_per_second:.2f} sweep/s")
    print(f"Final energy:      {results['final_energy']:.4f} eV")
    print(f"Acceptance rate:   {results['acceptance_rate']:.3f}")
    
    print(f"\n🖥️  CPU Utilization:")
    print(f"{'='*50}")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Peak CPU usage:    {max_cpu:.1f}%")
    print(f"CPU efficiency:    {cpu_efficiency:.1f}% (single-threaded equivalent)")
    print(f"CPU samples:       {cpu_samples}")
    
    # Threading analysis
    effective_cores = cpu_efficiency / 100
    print(f"\n🧵 Threading Analysis:")
    print(f"{'='*50}")
    if effective_cores > 1.5:
        print(f"✅ Multi-threading detected (~{effective_cores:.1f} effective cores)")
    elif effective_cores > 0.8:
        print(f"✅ Single-threaded performance (~{effective_cores:.1f} cores)")
    else:
        print(f"⚠️  Low CPU utilization ({effective_cores:.1f} cores) - possible bottleneck")
    
    # Compare with SpinMCPack
    spinmcpack_speed = 2.77
    speedup = sweeps_per_second / spinmcpack_speed
    
    print(f"\n🏁 Comparison with SpinMCPack:")
    print(f"{'='*50}")
    print(f"SpinMCPack:        {spinmcpack_speed:.2f} sweep/s")
    print(f"SpinLab:           {sweeps_per_second:.2f} sweep/s")
    print(f"Speedup factor:    {speedup:.1f}x {'🚀' if speedup > 1 else '🐌'}")
    print(f"250k sweeps time:  {250000/sweeps_per_second/3600:.1f} hours (vs {250000/2.77/3600:.1f} hours)")
    
    # Performance recommendations
    print(f"\n💡 Performance Notes:")
    print(f"{'='*50}")
    if not HAS_NUMBA:
        print("⚠️  Numba not available - install for major speedup")
    elif effective_cores < 1.0:
        print("⚠️  Low CPU utilization - check system load")
    elif speedup > 10:
        print("🚀 Excellent performance! Numba optimization working well")
    elif speedup > 3:
        print("✅ Good performance with Numba acceleration")
    else:
        print("⚠️  Moderate speedup - room for optimization")
    
    return sweeps_per_second, sys_info

if __name__ == "__main__":
    try:
        speed, info = detailed_24x24_test()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install psutil")
        # Fallback to basic test
        import quick_speed_test
        quick_speed_test.quick_24x24_test()