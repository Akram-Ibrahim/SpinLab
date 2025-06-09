"""
Visualization tools for spin systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any
import matplotlib.colors as colors
import matplotlib.tri as tri
from matplotlib import cm
from matplotlib.colors import Normalize
import time
from IPython.display import clear_output


class SpinVisualizer:
    """
    Visualization tools for spin configurations and dynamics.
    """
    
    def __init__(self):
        """Initialize spin visualizer."""
        pass
    
    def plot_spin_configuration_2d(
        self,
        positions: np.ndarray,
        spins: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        arrow_scale: float = 1.0,
        color_by: str = "z_component"
    ):
        """
        Plot 2D spin configuration with arrows.
        
        Args:
            positions: (n_spins, 2) or (n_spins, 3) positions
            spins: (n_spins, 3) spin vectors
            save_path: Path to save figure
            figsize: Figure size
            arrow_scale: Scale factor for arrows
            color_by: How to color spins ("z_component", "magnitude", "angle")
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract 2D positions
        if positions.shape[1] == 3:
            pos_2d = positions[:, :2]
        else:
            pos_2d = positions
        
        x, y = pos_2d[:, 0], pos_2d[:, 1]
        
        # Spin components for arrows
        sx, sy = spins[:, 0], spins[:, 1]
        
        # Color mapping
        if color_by == "z_component":
            colors_array = spins[:, 2]
            cmap = plt.cm.RdBu_r
            label = "Sz"
        elif color_by == "magnitude":
            colors_array = np.linalg.norm(spins, axis=1)
            cmap = plt.cm.viridis
            label = "|S|"
        elif color_by == "angle":
            angles = np.arctan2(spins[:, 1], spins[:, 0])
            colors_array = angles
            cmap = plt.cm.hsv
            label = "φ (rad)"
        else:
            colors_array = np.ones(len(spins))
            cmap = plt.cm.gray
            label = ""
        
        # Plot arrows with colors
        quiver = ax.quiver(
            x, y, sx, sy,
            colors_array,
            cmap=cmap,
            scale=1/arrow_scale,
            scale_units='xy',
            angles='xy',
            width=0.003
        )
        
        # Colorbar
        cbar = plt.colorbar(quiver, ax=ax)
        cbar.set_label(label)
        
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_title('2D Spin Configuration')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spin_configuration_3d(
        self,
        positions: np.ndarray,
        spins: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        arrow_scale: float = 1.0,
        subsample: Optional[int] = None
    ):
        """
        Plot 3D spin configuration.
        
        Args:
            positions: (n_spins, 3) positions
            spins: (n_spins, 3) spin vectors
            save_path: Path to save figure
            figsize: Figure size
            arrow_scale: Scale factor for arrows
            subsample: Subsample every N spins for large systems
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for large systems
        if subsample is not None and len(positions) > subsample:
            indices = np.arange(0, len(positions), len(positions) // subsample)
            pos = positions[indices]
            spin_vec = spins[indices]
        else:
            pos = positions
            spin_vec = spins
        
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        sx, sy, sz = spin_vec[:, 0], spin_vec[:, 1], spin_vec[:, 2]
        
        # Color by z-component
        colors_array = sz
        
        # Plot arrows
        quiver = ax.quiver(
            x, y, z, sx, sy, sz,
            colors_array,
            cmap=plt.cm.RdBu_r,
            length=arrow_scale,
            normalize=False
        )
        
        # Colorbar
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8)
        cbar.set_label('Sz')
        
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_zlabel('z (Å)')
        ax.set_title('3D Spin Configuration')
        
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
    
    def plot_energy_landscape(
        self,
        theta_range: np.ndarray,
        phi_range: np.ndarray,
        energy_surface: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot energy landscape for single spin.
        
        Args:
            theta_range: Theta values (polar angle)
            phi_range: Phi values (azimuthal angle)
            energy_surface: Energy surface
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 2D contour plot
        Theta, Phi = np.meshgrid(theta_range, phi_range)
        
        contour = axes[0].contourf(Theta, Phi, energy_surface, levels=20, cmap='viridis')
        axes[0].set_xlabel('θ (degrees)')
        axes[0].set_ylabel('φ (degrees)')
        axes[0].set_title('Energy Landscape')
        plt.colorbar(contour, ax=axes[0], label='Energy (eV)')
        
        # 3D surface plot
        ax_3d = fig.add_subplot(122, projection='3d')
        surface = ax_3d.plot_surface(Theta, Phi, energy_surface, 
                                   cmap='viridis', alpha=0.8)
        ax_3d.set_xlabel('θ (degrees)')
        ax_3d.set_ylabel('φ (degrees)')
        ax_3d.set_zlabel('Energy (eV)')
        ax_3d.set_title('3D Energy Surface')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_phase_diagram(
        self,
        parameter1: np.ndarray,
        parameter2: np.ndarray,
        order_parameter: np.ndarray,
        param1_name: str = "Parameter 1",
        param2_name: str = "Parameter 2",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot phase diagram.
        
        Args:
            parameter1: First parameter values
            parameter2: Second parameter values
            order_parameter: Order parameter values
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        im = ax.contourf(parameter1, parameter2, order_parameter, 
                        levels=20, cmap='RdYlBu_r')
        
        # Add contour lines
        contours = ax.contour(parameter1, parameter2, order_parameter, 
                             levels=10, colors='black', alpha=0.5, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Order Parameter')
        
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_title('Phase Diagram')
        
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
        Create animation of spin dynamics.
        
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
        
        # Setup figure
        if pos.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            def animate(frame):
                ax.clear()
                
                x, y = pos[:, 0], pos[:, 1]
                sx, sy = traj[frame, :, 0], traj[frame, :, 1]
                sz = traj[frame, :, 2]
                
                quiver = ax.quiver(x, y, sx, sy, sz, cmap='RdBu_r', 
                                 scale=1, scale_units='xy', angles='xy')
                
                ax.set_xlabel('x (Å)')
                ax.set_ylabel('y (Å)')
                ax.set_title(f'Spin Dynamics - Frame {frame}')
                ax.set_aspect('equal')
                
                return quiver,
        
        else:  # 3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            def animate(frame):
                ax.clear()
                
                x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
                sx, sy, sz = traj[frame, :, 0], traj[frame, :, 1], traj[frame, :, 2]
                
                quiver = ax.quiver(x, y, z, sx, sy, sz, sz, cmap='RdBu_r')
                
                ax.set_xlabel('x (Å)')
                ax.set_ylabel('y (Å)')
                ax.set_zlabel('z (Å)')
                ax.set_title(f'3D Spin Dynamics - Frame {frame}')
                
                return quiver,
        
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
    
    def plot_hysteresis_loop(
        self,
        magnetic_fields: np.ndarray,
        magnetizations: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ):
        """
        Plot magnetic hysteresis loop.
        
        Args:
            magnetic_fields: Applied magnetic field values
            magnetizations: Corresponding magnetization values
            save_path: Path to save figure
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(magnetic_fields, magnetizations, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Applied Field (T)')
        ax.set_ylabel('Magnetization')
        ax.set_title('Magnetic Hysteresis Loop')
        ax.grid(True, alpha=0.3)
        
        # Add zero lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spin_configuration_skyrmion_style(
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
        Plot spin configuration in skyrmion style with tricontourf background.
        
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
    
    def plot_spin_grid_skyrmion_style(
        self,
        spins: np.ndarray,
        grid_size: Optional[int] = None,
        title: str = "Spin Configuration",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 300,
        arrow_scale: float = 50
    ):
        """
        Plot spin grid in skyrmion style for regular lattices.
        
        Args:
            spins: (n_spins, 3) spin vectors
            grid_size: Grid size (auto-detected if None)
            title: Plot title
            figsize: Figure size
            dpi: Figure DPI
            arrow_scale: Arrow scale
        """
        n_spins = len(spins)
        
        if grid_size is None:
            grid_size = int(np.sqrt(n_spins))
        
        if grid_size * grid_size != n_spins:
            print(f"Warning: {n_spins} spins don't form {grid_size}x{grid_size} grid")
            return
        
        # Create regular grid positions
        x_1d = np.arange(grid_size)
        y_1d = np.arange(grid_size)
        X, Y = np.meshgrid(x_1d, y_1d)
        x = X.flatten()
        y = Y.flatten()
        
        # Use the skyrmion style plotting
        positions_2d = np.column_stack((x, y))
        self.plot_spin_configuration_skyrmion_style(
            positions_2d, spins, title, figsize, dpi, arrow_scale
        )
    
    def create_system_builder(self):
        """Create a simple system builder for quick setups."""
        def build_system(material='Fe', size=(16, 16, 1), J=-0.02, **kwargs):
            from ase.build import bulk
            from ..core.spin_system import SpinSystem
            from ..core.hamiltonian import Hamiltonian
            
            # Material database
            materials = {
                'Fe': {'crystal': 'bcc', 'a': 2.87, 'cutoff': 3.5},
                'Ni': {'crystal': 'fcc', 'a': 3.52, 'cutoff': 3.8},
                'Co': {'crystal': 'hcp', 'a': 2.51, 'cutoff': 3.2},
                'Mn': {'crystal': 'bcc', 'a': 3.08, 'cutoff': 3.8}
            }
            
            mat = materials[material]
            
            # Build structure & Hamiltonian
            structure = bulk(material, mat['crystal'], a=mat['a'], cubic=True).repeat(size)
            
            hamiltonian = Hamiltonian()
            hamiltonian.add_exchange(J=J, neighbor_shell="shell_1")
            
            # Optional additions
            if 'B_field' in kwargs:
                hamiltonian.add_magnetic_field(B_field=kwargs['B_field'], g_factor=2.0)
            if 'anisotropy' in kwargs:
                hamiltonian.add_single_ion_anisotropy(A=kwargs['anisotropy'], axis=[0,0,1])
            
            # Create system
            system = SpinSystem(structure, hamiltonian, magnetic_model="3d")
            system.get_neighbors([mat['cutoff']])
            system.random_configuration()
            
            return system
        
        return build_system
    
    def create_live_monitor(self, system, figsize=(15, 5)):
        """Create a real-time simulation monitor using skyrmion style."""
        
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
                
                self.ax3.set_title('Live Spin Configuration')
                self.ax3.set_aspect('equal')
                
                plt.tight_layout()
            
            def update(self, step, energy, magnetization):
                """Update all plots efficiently using skyrmion style."""
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
                
                # Update spins using skyrmion style
                self.ax3.clear()
                self._plot_spins_skyrmion_style()
                
                plt.draw()
            
            def _plot_spins_skyrmion_style(self):
                """Fast spin plotting using skyrmion style for real-time updates."""
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
                    
                    # In-plane components for arrows
                    u = sx
                    v = sy
                    
                    # Out-of-plane for background color
                    background_colors = sz / r
                    
                    # Simple quiver plot for real-time (tricontourf too slow)
                    self.ax3.scatter(x, y, c=background_colors, cmap=cm.jet, 
                                   s=20, vmin=-1, vmax=1, alpha=0.7)
                    self.ax3.quiver(x, y, u, v, color='black', pivot='mid', 
                                  width=0.002, scale=15, headwidth=2, headlength=3)
                    
                    self.ax3.set_xlim(-1, grid_size)
                    self.ax3.set_ylim(-1, grid_size)
                    self.ax3.set_title(f'Live Spins ({grid_size}×{grid_size})')
                    self.ax3.set_aspect('equal')
        
        return LiveMonitor(system, figsize)
    
    def run_live_simulation(self, system, monitor, temperature=100, n_steps=1000, update_every=50):
        """Run simulation with live visualization."""
        from ..monte_carlo import MonteCarlo
        
        mc = MonteCarlo(system, temperature=temperature)
        print(f"🚀 Live simulation: T={temperature}K, {n_steps} steps")
        
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
        
        print(f"✅ Complete! Time: {time.time() - start_time:.2f}s")
        return result
    
    def temperature_sweep_live(self, system, T_range, steps_per_T=200):
        """Live temperature sweep with instant plotting."""
        from ..monte_carlo import MonteCarlo
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        results = {'T': [], 'E': [], 'M': [], 'M_components': []}
        
        for i, T in enumerate(T_range):
            print(f"🌡️ T = {T:.1f}K ({i+1}/{len(T_range)})")
            
            # Simulate
            mc = MonteCarlo(system, temperature=T)
            result = mc.run(n_steps=steps_per_T, equilibration_steps=50, verbose=False)
            
            # Store results
            results['T'].append(T)
            results['E'].append(result['final_energy'] / len(system.positions))  # Per site
            results['M'].append(np.linalg.norm(result['final_magnetization']))
            results['M_components'].append(result['final_magnetization'])
            
            # Live update plots
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
            
            print(f"\n📊 Estimated T_c: {T_c:.1f}K")
        
        plt.show()
        return results