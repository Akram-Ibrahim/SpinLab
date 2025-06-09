#!/usr/bin/env python3
"""
Efficient visualization utilities for magnetic systems.
Compact, high-performance plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import time

class SpinVisualizer:
    """Efficient spin configuration visualization."""
    
    @staticmethod
    def plot_spins_2d(spins, positions=None, figsize=(8, 8), title="Spin Configuration"):
        """Fast 2D spin visualization with arrows and colors."""
        
        n_spins = len(spins)
        grid_size = int(np.sqrt(n_spins))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if grid_size * grid_size == n_spins and positions is None:
            # Regular grid
            x = np.arange(grid_size)
            y = np.arange(grid_size)
            X, Y = np.meshgrid(x, y)
            
            spins_2d = spins.reshape(grid_size, grid_size, 3)
            
            # Color by z-component (out-of-plane)
            colors = spins_2d[:, :, 2]
            
            # Plot arrows (in-plane components)
            ax.quiver(X, Y, spins_2d[:, :, 0], spins_2d[:, :, 1], 
                     colors, cmap='RdBu', scale=3, alpha=0.8, width=0.003)
            
            ax.set_xlim(-0.5, grid_size-0.5)
            ax.set_ylim(-0.5, grid_size-0.5)
        
        else:
            # Irregular positions
            if positions is None:
                positions = np.random.rand(n_spins, 2) * 10
            
            # Use only 2D positions
            pos_2d = positions[:, :2] if positions.shape[1] > 2 else positions
            
            colors = spins[:, 2]  # z-component
            
            ax.quiver(pos_2d[:, 0], pos_2d[:, 1], spins[:, 0], spins[:, 1],
                     colors, cmap='RdBu', scale=3, alpha=0.8)
        
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='RdBu', norm=Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Sz component')
        
        return fig, ax

    @staticmethod
    def plot_thermodynamics(temperatures, energies, magnetizations, 
                          heat_capacities=None, figsize=(12, 4)):
        """Efficient thermodynamic property visualization."""
        
        n_plots = 3 if heat_capacities is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        if n_plots == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
        # Energy
        ax1.plot(temperatures, energies, 'bo-', markersize=4, linewidth=2)
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Energy per site (eV)')
        ax1.set_title('Energy vs Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Magnetization
        ax2.plot(temperatures, magnetizations, 'ro-', markersize=4, linewidth=2)
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('|Magnetization|')
        ax2.set_title('Magnetization vs Temperature')
        ax2.grid(True, alpha=0.3)
        
        # Heat capacity (if provided)
        if heat_capacities is not None:
            ax3.plot(temperatures, heat_capacities, 'go-', markersize=4, linewidth=2)
            ax3.set_xlabel('Temperature (K)')
            ax3.set_ylabel('Heat Capacity')
            ax3.set_title('Heat Capacity vs Temperature')
            ax3.grid(True, alpha=0.3)
            
            # Mark critical temperature
            T_c_idx = np.argmax(heat_capacities)
            T_c = temperatures[T_c_idx]
            ax3.axvline(T_c, color='red', linestyle='--', alpha=0.7, 
                       label=f'T_c ≈ {T_c:.1f}K')
            ax3.legend()
        
        plt.tight_layout()
        return fig, axes

class LivePlotter:
    """Real-time plotting for ongoing simulations."""
    
    def __init__(self, figsize=(12, 4)):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=figsize)
        self.data = {'steps': [], 'energy': [], 'magnetization': []}
        self.lines = {}
        
        # Initialize empty plots
        self.lines['energy'] = self.ax1.plot([], [], 'b-', alpha=0.7)[0]
        self.lines['magnetization'] = self.ax2.plot([], [], 'r-', alpha=0.7)[0]
        
        # Configure axes
        self.ax1.set_title('Energy Evolution')
        self.ax1.set_xlabel('MC Steps')
        self.ax1.set_ylabel('Energy (eV)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title('Magnetization Evolution')
        self.ax2.set_xlabel('MC Steps')
        self.ax2.set_ylabel('|M|')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title('Current Spin Configuration')
        self.ax3.set_aspect('equal')
        self.ax3.axis('off')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
    
    def update(self, step, energy, magnetization, spins=None):
        """Fast update of all plots."""
        
        # Add data
        self.data['steps'].append(step)
        self.data['energy'].append(energy)
        self.data['magnetization'].append(np.linalg.norm(magnetization))
        
        # Update line plots efficiently
        self.lines['energy'].set_data(self.data['steps'], self.data['energy'])
        self.lines['magnetization'].set_data(self.data['steps'], self.data['magnetization'])
        
        # Auto-scale axes
        if len(self.data['steps']) > 1:
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        # Update spin configuration
        if spins is not None:
            self.ax3.clear()
            self._plot_spins_fast(spins)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _plot_spins_fast(self, spins):
        """Fast spin plotting for real-time updates."""
        n_spins = len(spins)
        grid_size = int(np.sqrt(n_spins))
        
        if grid_size * grid_size == n_spins:
            # Regular grid - fastest plotting
            x = np.arange(grid_size)
            y = np.arange(grid_size)
            X, Y = np.meshgrid(x, y)
            
            spins_2d = spins.reshape(grid_size, grid_size, 3)
            colors = spins_2d[:, :, 2]
            
            # Simplified quiver for speed
            self.ax3.quiver(X, Y, spins_2d[:, :, 0], spins_2d[:, :, 1], 
                           colors, cmap='RdBu', scale=4, alpha=0.8, width=0.005)
            
            self.ax3.set_xlim(-1, grid_size)
            self.ax3.set_ylim(-1, grid_size)
            self.ax3.set_title(f'Spins ({grid_size}×{grid_size})')
        
        self.ax3.set_aspect('equal')

def create_animation(spin_configs, interval=100, figsize=(8, 8)):
    """Create spin configuration animation from saved configurations."""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    def animate(frame):
        ax.clear()
        spins = spin_configs[frame]
        
        n_spins = len(spins)
        grid_size = int(np.sqrt(n_spins))
        
        if grid_size * grid_size == n_spins:
            x = np.arange(grid_size)
            y = np.arange(grid_size)
            X, Y = np.meshgrid(x, y)
            
            spins_2d = spins.reshape(grid_size, grid_size, 3)
            colors = spins_2d[:, :, 2]
            
            ax.quiver(X, Y, spins_2d[:, :, 0], spins_2d[:, :, 1], 
                     colors, cmap='RdBu', scale=3, alpha=0.8)
            
            ax.set_xlim(-0.5, grid_size-0.5)
            ax.set_ylim(-0.5, grid_size-0.5)
            ax.set_title(f'Frame {frame+1}/{len(spin_configs)}')
            ax.set_aspect('equal')
    
    animation = FuncAnimation(fig, animate, frames=len(spin_configs), 
                            interval=interval, blit=False, repeat=True)
    
    return animation

# Utility functions for quick plotting
def quick_energy_plot(steps, energies, title="Energy Evolution"):
    """One-liner energy plot."""
    plt.figure(figsize=(8, 4))
    plt.plot(steps, energies, 'b-', linewidth=2)
    plt.xlabel('MC Steps')
    plt.ylabel('Energy (eV)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def quick_magnetization_plot(steps, magnetizations, title="Magnetization Evolution"):
    """One-liner magnetization plot."""
    plt.figure(figsize=(8, 4))
    mags = [np.linalg.norm(m) for m in magnetizations]
    plt.plot(steps, mags, 'r-', linewidth=2)
    plt.xlabel('MC Steps')
    plt.ylabel('|Magnetization|')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def quick_phase_diagram(temperatures, magnetizations, title="Phase Diagram"):
    """One-liner phase diagram."""
    plt.figure(figsize=(8, 5))
    mags = [np.linalg.norm(m) if hasattr(m, '__len__') else m for m in magnetizations]
    plt.plot(temperatures, mags, 'ro-', markersize=6, linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('|Magnetization|')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()