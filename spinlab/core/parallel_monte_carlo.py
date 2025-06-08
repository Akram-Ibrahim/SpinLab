"""
Parallel Monte Carlo implementation using multiple CPU cores.
"""

import numpy as np
from typing import Optional, Dict, List, Any
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .spin_system import SpinSystem
from .monte_carlo import MonteCarlo
from ..utils.random import set_random_seed


class ParallelMonteCarlo:
    """
    Parallel Monte Carlo using independent replicas across CPU cores.
    
    This provides linear speedup by running multiple independent MC
    simulations simultaneously and aggregating the results.
    """
    
    def __init__(
        self,
        spin_system: SpinSystem,
        n_cores: Optional[int] = None
    ):
        """
        Initialize parallel Monte Carlo.
        
        Args:
            spin_system: SpinSystem to simulate
            n_cores: Number of CPU cores to use (None = auto-detect safe limit)
        """
        self.spin_system = spin_system
        
        # Conservative core detection to avoid overwhelming the system
        max_cores = mp.cpu_count()
        if n_cores is None:
            # Use conservative limit: max 80% of cores or 20 cores, whichever is smaller
            self.n_cores = min(max_cores, 20, max(1, int(max_cores * 0.8)))
        else:
            self.n_cores = min(n_cores, max_cores)
        
        print(f"🚀 Parallel MC initialized: {self.n_cores} cores (out of {max_cores} available)")
    
    def run(
        self,
        temperature: float,
        n_steps: int,
        n_replicas: Optional[int] = None,
        equilibration_steps: int = 1000,
        sampling_interval: int = 1,
        verbose: bool = True,
        show_individual_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Run multiple independent MC replicas in parallel.
        
        Args:
            temperature: Temperature in Kelvin
            n_steps: MC steps per replica
            n_replicas: Number of replicas (default: n_cores)
            equilibration_steps: Equilibration steps per replica
            sampling_interval: Sampling interval
            verbose: Show progress
            show_individual_progress: Show progress bars for individual replicas (only for ≤4 replicas by default)
            
        Returns:
            Aggregated results from all replicas with statistics
        """
        if n_replicas is None:
            n_replicas = self.n_cores
        
        if verbose:
            print(f"🔄 Running {n_replicas} parallel replicas at T={temperature}K")
            print(f"   {n_steps} steps per replica = {n_replicas * n_steps:,} total steps")
            if show_individual_progress and n_replicas <= 4:
                print(f"   📊 Individual MC progress bars will be shown for each replica")
        
        # Prepare arguments for each replica
        replica_args = []
        # Show individual progress only for small numbers of replicas to avoid clutter
        show_individual = show_individual_progress and n_replicas <= 4
        
        for replica_id in range(n_replicas):
            args = {
                'spin_system_data': self._serialize_spin_system(),
                'temperature': temperature,
                'n_steps': n_steps,
                'equilibration_steps': equilibration_steps,
                'sampling_interval': sampling_interval,
                'random_seed': replica_id * 12345,  # Unique seed per replica
                'replica_id': replica_id,
                'show_individual_progress': show_individual
            }
            replica_args.append(args)
        
        # Run replicas in parallel
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            if verbose:
                print(f"⚡ Launching {n_replicas} replicas on {self.n_cores} cores...")
            
            # Submit all jobs
            futures = [executor.submit(_run_single_replica, args) for args in replica_args]
            
            # Collect results with progress bar
            replica_results = []
            if verbose:
                pbar = tqdm(total=n_replicas, desc="Completed replicas", 
                          unit="replica", dynamic_ncols=True)
            
            for future in as_completed(futures):
                result = future.result()
                replica_results.append(result)
                if verbose:
                    pbar.update(1)
                    # Update description with current stats
                    if len(replica_results) > 1:
                        energies = [r['final_energy'] for r in replica_results]
                        pbar.set_postfix({
                            'E_avg': f"{np.mean(energies):.4f}",
                            'E_std': f"{np.std(energies):.4f}"
                        })
            
            if verbose:
                pbar.close()
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated = self._aggregate_results(replica_results, total_time, n_replicas)
        
        if verbose:
            print(f"✅ {n_replicas} replicas completed in {total_time:.2f}s")
            print(f"⚡ Effective rate: {n_replicas * n_steps / total_time / 1000:.1f}k steps/sec")
            print(f"🎯 Speedup vs single core: ~{n_replicas:.0f}x")
        
        return aggregated
    
    def run_temperature_series(
        self,
        temperatures: List[float],
        n_steps: int,
        n_replicas_per_temp: int = 4,
        equilibration_steps: int = 1000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run temperature series with parallel replicas at each temperature.
        
        Args:
            temperatures: List of temperatures to simulate
            n_steps: MC steps per replica
            n_replicas_per_temp: Replicas per temperature
            equilibration_steps: Equilibration steps
            verbose: Show progress
            
        Returns:
            Results for each temperature with error bars
        """
        if verbose:
            print(f"🌡️  Parallel temperature series:")
            print(f"   {len(temperatures)} temperatures × {n_replicas_per_temp} replicas")
            print(f"   Total: {len(temperatures) * n_replicas_per_temp} simulations")
        
        temp_results = {}
        total_start_time = time.time()
        
        for i, temp in enumerate(temperatures):
            if verbose:
                print(f"\n[{i+1:2d}/{len(temperatures)}] T = {temp:6.1f} K")
            
            # Run parallel replicas for this temperature
            result = self.run(
                temperature=temp,
                n_steps=n_steps,
                n_replicas=n_replicas_per_temp,
                equilibration_steps=equilibration_steps,
                verbose=False  # Don't show individual progress
            )
            
            temp_results[f'T_{temp:.1f}'] = result
            
            if verbose:
                print(f"   E = {result['final_energy_mean']:.6f} ± {result['final_energy_std']:.6f} eV")
                print(f"   |M| = {result['magnetization_magnitude_mean']:.3f} ± {result['magnetization_magnitude_std']:.3f}")
        
        total_time = time.time() - total_start_time
        
        if verbose:
            print(f"\n✅ Temperature series completed in {total_time/60:.1f} minutes")
        
        return {
            'temperatures': temperatures,
            'results_by_temperature': temp_results,
            'total_time': total_time,
            'n_replicas_per_temp': n_replicas_per_temp
        }
    
    def _serialize_spin_system(self) -> Dict[str, Any]:
        """Serialize spin system for multiprocessing."""
        return {
            'structure': self.spin_system.structure,
            'hamiltonian': self.spin_system.hamiltonian,
            'spin_magnitude': self.spin_system.spin_magnitude,
            'magnetic_model': self.spin_system.magnetic_model,
            'use_fast': self.spin_system.use_fast
        }
    
    def _aggregate_results(
        self, 
        replica_results: List[Dict], 
        total_time: float,
        n_replicas: int
    ) -> Dict[str, Any]:
        """Aggregate results from multiple replicas with statistics."""
        
        # Filter out failed replicas
        valid_results = [r for r in replica_results if 'error' not in r]
        failed_count = len(replica_results) - len(valid_results)
        
        if failed_count > 0:
            print(f"⚠️  {failed_count} replicas failed")
        
        if len(valid_results) == 0:
            raise RuntimeError("All replicas failed!")
        
        # Extract arrays
        final_energies = np.array([r['final_energy'] for r in valid_results])
        final_magnetizations = np.array([r['final_magnetization'] for r in valid_results])
        acceptance_rates = np.array([r['acceptance_rate'] for r in valid_results])
        
        # Calculate magnetization magnitudes
        mag_magnitudes = np.linalg.norm(final_magnetizations, axis=1)
        
        # Aggregate statistics
        aggregated = {
            # Metadata
            'n_replicas': n_replicas,
            'n_successful': len(valid_results),
            'n_failed': failed_count,
            'total_time': total_time,
            'strategy': 'parallel_replicas',
            
            # Energy statistics
            'final_energy_mean': np.mean(final_energies),
            'final_energy_std': np.std(final_energies),
            'final_energy_sem': np.std(final_energies) / np.sqrt(len(final_energies)),  # Standard error
            'final_energy_min': np.min(final_energies),
            'final_energy_max': np.max(final_energies),
            'final_energy': np.mean(final_energies),  # For compatibility
            
            # Magnetization statistics
            'final_magnetization_mean': np.mean(final_magnetizations, axis=0),
            'final_magnetization_std': np.std(final_magnetizations, axis=0),
            'final_magnetization': np.mean(final_magnetizations, axis=0),  # For compatibility
            
            # Magnetization magnitude statistics
            'magnetization_magnitude_mean': np.mean(mag_magnitudes),
            'magnetization_magnitude_std': np.std(mag_magnitudes),
            'magnetization_magnitude_sem': np.std(mag_magnitudes) / np.sqrt(len(mag_magnitudes)),
            
            # Acceptance rate statistics
            'acceptance_rate_mean': np.mean(acceptance_rates),
            'acceptance_rate_std': np.std(acceptance_rates),
            'acceptance_rate': np.mean(acceptance_rates),  # For compatibility
            
            # Best replica (lowest energy)
            'best_replica_id': np.argmin(final_energies),
            'best_energy': np.min(final_energies),
            'best_magnetization': final_magnetizations[np.argmin(final_energies)],
            
            # All raw data for further analysis
            'all_final_energies': final_energies,
            'all_final_magnetizations': final_magnetizations,
            'all_acceptance_rates': acceptance_rates,
            'all_magnetization_magnitudes': mag_magnitudes,
            
            # Performance metrics
            'steps_per_second': n_replicas * valid_results[0]['n_steps'] / total_time,
            'effective_speedup': n_replicas  # Theoretical speedup
        }
        
        return aggregated


def _run_single_replica(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single MC replica (for multiprocessing).
    
    This function runs in a separate process.
    """
    try:
        # Reconstruct spin system
        spin_system = SpinSystem(
            structure=args['spin_system_data']['structure'],
            hamiltonian=args['spin_system_data']['hamiltonian'],
            spin_magnitude=args['spin_system_data']['spin_magnitude'],
            magnetic_model=args['spin_system_data']['magnetic_model'],
            use_fast=args['spin_system_data']['use_fast']
        )
        
        # Setup neighbors and initial configuration
        spin_system.get_neighbors([3.0, 4.5])
        spin_system.random_configuration()
        
        # Set unique random seed for this replica
        set_random_seed(args['random_seed'])
        
        # Run MC simulation
        mc = MonteCarlo(
            spin_system=spin_system,
            temperature=args['temperature'],
            random_seed=args['random_seed'],
            use_fast=True
        )
        
        # Show individual progress bars when there are few replicas
        show_individual_progress = args.get('show_individual_progress', False)
        
        result = mc.run(
            n_steps=args['n_steps'],
            equilibration_steps=args['equilibration_steps'],
            sampling_interval=args['sampling_interval'],
            verbose=show_individual_progress
        )
        
        # Add replica metadata
        result['replica_id'] = args['replica_id']
        result['random_seed'] = args['random_seed']
        
        return result
        
    except Exception as e:
        # Return error result that won't crash aggregation
        return {
            'replica_id': args.get('replica_id', -1),
            'error': str(e),
            'final_energy': float('inf'),
            'final_magnetization': np.array([0, 0, 0]),
            'acceptance_rate': 0.0,
            'n_steps': args.get('n_steps', 0)
        }