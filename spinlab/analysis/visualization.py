"""
Visualization tools for spin systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any
import matplotlib.colors as colors


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