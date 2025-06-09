"""
Visualization tools for spin systems - 2D plotting only.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.colors import Normalize
import time
from IPython.display import clear_output


class SpinVisualizer:
    """
    2D visualization tools for spin configurations and dynamics.
    """
    
    def __init__(self):
        """Initialize spin visualizer."""
        pass
    
    def plot_spin_configuration(
        self,
        positions: np.ndarray,
        spins: np.ndarray,
        title: str = "Spin Configuration",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 300,
        arrow_scale: float = 50,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D spin configuration with tricontourf background and black arrows.
        
        Args:
            positions: (n_spins, 2) or (n_spins, 3) positions
            spins: (n_spins, 3) spin vectors (Cartesian)
            title: Plot title
            figsize: Figure size
            dpi: Figure DPI
            arrow_scale: Scale for quiver arrows
            save_path: Path to save figure
        """
        # Extract 2D positions
        if positions.shape[1] == 3:
            x, y = positions[:, 0], positions[:, 1]
        else:
            x, y = positions[:, 0], positions[:, 1]
        
        # Convert Cartesian spins to spherical coordinates
        sx, sy, sz = spins[:, 0], spins[:, 1], spins[:, 2]
        
        # Calculate theta and phi from Cartesian coordinates
        r = np.sqrt(sx**2 + sy**2 + sz**2)
        theta = np.arccos(np.clip(sz / r, -1, 1))  # Polar angle [0, π]
        phi = np.arctan2(sy, sx)  # Azimuthal angle [-π, π]
        
        # Calculate arrow components (in-plane projection)
        arrow_length = 1.0
        u = arrow_length * np.sin(theta) * np.cos(phi)
        v = arrow_length * np.sin(theta) * np.sin(phi)
        
        # Background colors based on z-component (out-of-plane)
        background_colors = sz / r  # Normalized z-component
        
        # Create figure
        plt.figure(figsize=figsize, dpi=dpi)
        
        # Create triangulation for smooth background
        triang = tri.Triangulation(x, y)
        
        # Plot background with tricontourf
        norm = Normalize(vmin=-1, vmax=1)
        contour = plt.tricontourf(triang, background_colors, levels=100, 
                                 cmap=cm.jet, norm=norm)
        
        # Plot arrows (black for contrast)
        plt.quiver(x, y, u, v, color='black', pivot='mid', 
                  width=0.0035, scale=arrow_scale, headwidth=3, headlength=5)
        
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title(title, fontsize=12)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(contour, orientation='vertical', 
                           label='$m_z$', shrink=0.5)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1', '0', '1'])
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_energy_landscape(
        self,
        theta_range: np.ndarray,
        phi_range: np.ndarray,
        energy_surface: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot 2D energy landscape for single spin.
        
        Args:
            theta_range: Theta values (polar angle)
            phi_range: Phi values (azimuthal angle)
            energy_surface: Energy surface
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 2D contour plot only
        Theta, Phi = np.meshgrid(theta_range, phi_range)
        
        contour = ax.contourf(Theta, Phi, energy_surface, levels=20, cmap='viridis')
        ax.set_xlabel('θ (degrees)')
        ax.set_ylabel('φ (degrees)')
        ax.set_title('Energy Landscape')
        plt.colorbar(contour, ax=ax, label='Energy (eV)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_spin_dynamics(
        self,
        positions: np.ndarray,
        spin_trajectories: np.ndarray,
        save_path: Optional[str] = None,
        interval: int = 100,
        subsample_time: int = 1,
        subsample_space: Optional[int] = None
    ):
        """
        Create 2D animation of spin dynamics.
        
        Args:
            positions: (n_spins, 2/3) positions
            spin_trajectories: (n_steps, n_spins, 3) spin evolution
            save_path: Path to save animation
            interval: Time interval between frames (ms)
            subsample_time: Take every N time steps
            subsample_space: Take every N spins
        """
        from matplotlib.animation import FuncAnimation
        
        # Subsample
        if subsample_space is not None:
            pos = positions[::subsample_space]
            traj = spin_trajectories[:, ::subsample_space, :]
        else:
            pos = positions
            traj = spin_trajectories
        
        traj = traj[::subsample_time]
        
        # Setup 2D figure only
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            x, y = pos[:, 0], pos[:, 1] if pos.shape[1] >= 2 else (pos[:, 0], np.zeros_like(pos[:, 0]))
            sx, sy = traj[frame, :, 0], traj[frame, :, 1]
            sz = traj[frame, :, 2]
            
            # Use scatter for background color and quiver for arrows
            ax.scatter(x, y, c=sz, cmap='RdBu_r', s=30, vmin=-1, vmax=1)
            ax.quiver(x, y, sx, sy, color='black', scale=3, width=0.003)
            
            ax.set_xlabel('x (Å)')
            ax.set_ylabel('y (Å)')
            ax.set_title(f'Spin Dynamics - Frame {frame}')
            ax.set_aspect('equal')
            
            return ax,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(traj), 
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=1000//interval)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        
        plt.show()
        
        return anim
    
    def plot_energy_dynamics(
        self,
        times: np.ndarray,
        energies: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot energy vs time.
        
        Args:
            times: Time array
            energies: Energy values
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(times, energies, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Evolution')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_magnetization_dynamics(
        self,
        times: np.ndarray,
        magnetizations: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot magnetization vs time.
        
        Args:
            times: Time array
            magnetizations: (n_steps, 3) magnetization vectors
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Individual components
        axes[0, 0].plot(times, magnetizations[:, 0], label='Mx', color='red')
        axes[0, 0].set_ylabel('Mx')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(times, magnetizations[:, 1], label='My', color='green')
        axes[0, 1].set_ylabel('My')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(times, magnetizations[:, 2], label='Mz', color='blue')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Mz')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Magnitude
        mag_magnitude = np.linalg.norm(magnetizations, axis=1)
        axes[1, 1].plot(times, mag_magnitude, label='|M|', color='black')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('|M|')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_live_monitor(self, system, figsize=(15, 5)):
        """Create a real-time simulation monitor."""
        
        class LiveMonitor:
            def __init__(self, system, figsize):
                self.system = system
                self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=figsize)
                self.data = {'steps': [], 'energy': [], 'magnetization': []}
                
                # Setup axes
                self.ax1.set_title('Energy Evolution')
                self.ax1.set_xlabel('MC Steps')
                self.ax1.set_ylabel('Energy (eV)')
                self.ax1.grid(True, alpha=0.3)
                
                self.ax2.set_title('Magnetization')
                self.ax2.set_xlabel('MC Steps')
                self.ax2.set_ylabel('|M|')
                self.ax2.grid(True, alpha=0.3)
                
                self.ax3.set_title('Spin Configuration')
                self.ax3.set_aspect('equal')
                
                plt.tight_layout()
            
            def update(self, step, energy, magnetization):
                """Update all plots efficiently."""
                # Store data
                self.data['steps'].append(step)
                self.data['energy'].append(energy)
                self.data['magnetization'].append(np.linalg.norm(magnetization))
                
                # Update energy plot
                self.ax1.clear()
                self.ax1.plot(self.data['steps'], self.data['energy'], 'b-', alpha=0.8, linewidth=2)
                self.ax1.set_title(f'Energy: {energy:.4f} eV')
                self.ax1.set_xlabel('MC Steps')
                self.ax1.set_ylabel('Energy (eV)')
                self.ax1.grid(True, alpha=0.3)
                
                # Update magnetization plot
                self.ax2.clear()
                self.ax2.plot(self.data['steps'], self.data['magnetization'], 'r-', alpha=0.8, linewidth=2)
                self.ax2.set_title(f'|M|: {self.data["magnetization"][-1]:.3f}')
                self.ax2.set_xlabel('MC Steps')
                self.ax2.set_ylabel('|M|')
                self.ax2.grid(True, alpha=0.3)
                
                # Update spins
                self.ax3.clear()
                self._plot_spins_simple()
                
                plt.draw()
            
            def _plot_spins_simple(self):
                """Fast spin plotting for real-time updates."""
                spins = self.system.spin_config
                n_spins = len(spins)
                grid_size = int(np.sqrt(n_spins))
                
                if grid_size * grid_size == n_spins:
                    # Create 2D grid positions
                    x_1d = np.arange(grid_size)
                    y_1d = np.arange(grid_size)
                    X, Y = np.meshgrid(x_1d, y_1d)
                    x = X.flatten()
                    y = Y.flatten()
                    
                    # Convert spins to components
                    sx, sy, sz = spins[:, 0], spins[:, 1], spins[:, 2]
                    r = np.sqrt(sx**2 + sy**2 + sz**2)
                    
                    # Out-of-plane for background color
                    background_colors = sz / r
                    
                    # Simple scatter + quiver for speed
                    self.ax3.scatter(x, y, c=background_colors, cmap=cm.jet, 
                                   s=20, vmin=-1, vmax=1, alpha=0.7)
                    self.ax3.quiver(x, y, sx, sy, color='black', pivot='mid', 
                                  width=0.002, scale=15, headwidth=2, headlength=3)
                    
                    self.ax3.set_xlim(-1, grid_size)
                    self.ax3.set_ylim(-1, grid_size)
                    self.ax3.set_title(f'Spins ({grid_size}×{grid_size})')
                    self.ax3.set_aspect('equal')
        
        return LiveMonitor(system, figsize)
    
    def run_live_simulation(self, system, monitor, temperature=100, n_steps=1000, update_every=50):
        """Run simulation with visualization."""
        from ..monte_carlo import MonteCarlo
        
        mc = MonteCarlo(system, temperature=temperature)
        print(f"Starting simulation: T={temperature}K, {n_steps} steps")
        
        start_time = time.time()
        
        for step in range(0, n_steps, update_every):
            # Run batch
            batch_size = min(update_every, n_steps - step)
            result = mc.run(n_steps=batch_size, equilibration_steps=0, verbose=False)
            
            # Update visualization
            monitor.update(step + batch_size, result['final_energy'], result['final_magnetization'])
            
            # Progress
            elapsed = time.time() - start_time
            speed = (step + batch_size) / elapsed if elapsed > 0 else 0
            
            clear_output(wait=True)
            print(f"Step {step + batch_size}/{n_steps} | Speed: {speed:.1f} steps/s | "
                  f"E: {result['final_energy']:.4f} eV | |M|: {np.linalg.norm(result['final_magnetization']):.3f}")
            
            time.sleep(0.15)  # Visualization delay
        
        print(f"Complete! Time: {time.time() - start_time:.2f}s")
        return result
    
    def temperature_sweep_live(self, system, T_range, steps_per_T=200):
        """Temperature sweep with plotting."""
        from ..monte_carlo import MonteCarlo
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        results = {'T': [], 'E': [], 'M': [], 'M_components': []}
        
        for i, T in enumerate(T_range):
            print(f"T = {T:.1f}K ({i+1}/{len(T_range)})")
            
            # Simulate
            mc = MonteCarlo(system, temperature=T)
            result = mc.run(n_steps=steps_per_T, equilibration_steps=50, verbose=False)
            
            # Store results
            results['T'].append(T)
            results['E'].append(result['final_energy'] / len(system.positions))  # Per site
            results['M'].append(np.linalg.norm(result['final_magnetization']))
            results['M_components'].append(result['final_magnetization'])
            
            # Update plots
            ax1.clear()
            ax1.plot(results['T'], results['E'], 'bo-', markersize=6, linewidth=2)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Energy per site (eV)')
            ax1.set_title('Energy vs Temperature')
            ax1.grid(True, alpha=0.3)
            
            ax2.clear()
            ax2.plot(results['T'], results['M'], 'ro-', markersize=6, linewidth=2)
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('|Magnetization|')
            ax2.set_title('Magnetization vs Temperature')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.2)
        
        # Find critical temperature
        M_array = np.array(results['M'])
        if len(results['T']) > 3:
            # Simple criterion: steepest drop in magnetization
            dM_dT = np.gradient(M_array, results['T'])
            T_c_idx = np.argmin(dM_dT)
            T_c = results['T'][T_c_idx]
            
            ax2.axvline(T_c, color='green', linestyle='--', alpha=0.8, 
                       label=f'T_c ≈ {T_c:.1f}K')
            ax2.legend()
            
            print(f"Estimated T_c: {T_c:.1f}K")
        
        plt.show()
        return results