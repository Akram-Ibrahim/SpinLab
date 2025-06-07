"""
Numerical integrators for solving the LLG equation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llg_solver import LLGSolver


class Integrator(ABC):
    """Abstract base class for LLG integrators."""
    
    def __init__(self, solver: 'LLGSolver'):
        """Initialize integrator with reference to LLG solver."""
        self.solver = solver
    
    @abstractmethod
    def step(self, spins: np.ndarray, dt: float) -> np.ndarray:
        """
        Perform one integration step.
        
        Args:
            spins: Current spin configuration
            dt: Time step
            
        Returns:
            Updated spin configuration
        """
        pass


class HeunIntegrator(Integrator):
    """
    Heun's method (improved Euler) for LLG integration.
    
    This is a second-order predictor-corrector method that provides
    good stability and accuracy for the LLG equation.
    """
    
    def step(self, spins: np.ndarray, dt: float) -> np.ndarray:
        """Perform Heun integration step."""
        # Predictor step: Euler step
        k1 = self.solver.calculate_llg_rhs(spins)
        spins_pred = spins + dt * k1
        
        # Normalize predictor
        spins_pred = self._normalize_spins(spins_pred)
        
        # Corrector step: Average of slopes
        k2 = self.solver.calculate_llg_rhs(spins_pred)
        spins_new = spins + dt * 0.5 * (k1 + k2)
        
        return spins_new
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.solver.spin_system.spin_magnitude


class RK4Integrator(Integrator):
    """
    Fourth-order Runge-Kutta integrator for LLG equation.
    
    Provides higher accuracy than Heun's method but requires more
    function evaluations per step.
    """
    
    def step(self, spins: np.ndarray, dt: float) -> np.ndarray:
        """Perform RK4 integration step."""
        # Calculate k1
        k1 = self.solver.calculate_llg_rhs(spins)
        
        # Calculate k2
        spins_k2 = self._normalize_spins(spins + 0.5 * dt * k1)
        k2 = self.solver.calculate_llg_rhs(spins_k2)
        
        # Calculate k3
        spins_k3 = self._normalize_spins(spins + 0.5 * dt * k2)
        k3 = self.solver.calculate_llg_rhs(spins_k3)
        
        # Calculate k4
        spins_k4 = self._normalize_spins(spins + dt * k3)
        k4 = self.solver.calculate_llg_rhs(spins_k4)
        
        # Combine for final result
        spins_new = spins + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        return spins_new
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.solver.spin_system.spin_magnitude


class SemiImplicitIntegrator(Integrator):
    """
    Semi-implicit integrator for LLG equation.
    
    This method treats the damping term implicitly while keeping
    the precession term explicit, providing better stability
    for highly damped systems.
    """
    
    def step(self, spins: np.ndarray, dt: float) -> np.ndarray:
        """Perform semi-implicit integration step."""
        # Calculate effective field
        H_eff = self.solver.calculate_effective_field(spins)
        
        # Add thermal noise if temperature > 0
        if self.solver.temperature > 0:
            H_eff += self.solver._calculate_thermal_field()
        
        # Semi-implicit update
        # We solve: S_new = S + dt * [-γ(S_new × H) + α(S_new × (S_new × H))/|S|²]
        # This requires an iterative solution
        
        alpha = self.solver.damping
        gamma = self.solver.gamma
        
        spins_new = spins.copy()
        
        # Iterative solution (typically 2-3 iterations sufficient)
        for _ in range(3):
            # Calculate cross products
            cross_SH = np.cross(spins_new, H_eff)
            cross_S_cross_SH = np.cross(spins_new, cross_SH)
            
            # Spin magnitudes squared
            spin_mag_sq = np.sum(spins_new**2, axis=1, keepdims=True)
            
            # Update equation
            drdt = -gamma * cross_SH + alpha * cross_S_cross_SH / spin_mag_sq
            
            # Semi-implicit update with damping factor
            damping_factor = 1.0 / (1.0 + alpha * dt)
            spins_new = damping_factor * (spins + dt * drdt)
            
            # Normalize
            spins_new = self._normalize_spins(spins_new)
        
        return spins_new
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.solver.spin_system.spin_magnitude


class AdaptiveIntegrator(Integrator):
    """
    Adaptive time-stepping integrator using embedded RK methods.
    
    Automatically adjusts time step based on local error estimation
    to maintain accuracy while maximizing efficiency.
    """
    
    def __init__(
        self, 
        solver: 'LLGSolver', 
        tolerance: float = 1e-6,
        safety_factor: float = 0.9,
        max_factor: float = 2.0,
        min_factor: float = 0.1
    ):
        """
        Initialize adaptive integrator.
        
        Args:
            solver: LLG solver instance
            tolerance: Error tolerance for step size control
            safety_factor: Safety factor for step size adjustment
            max_factor: Maximum factor for step size increase
            min_factor: Minimum factor for step size decrease
        """
        super().__init__(solver)
        self.tolerance = tolerance
        self.safety_factor = safety_factor
        self.max_factor = max_factor
        self.min_factor = min_factor
        
        # Statistics
        self.accepted_steps = 0
        self.rejected_steps = 0
    
    def step(self, spins: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
        """
        Perform adaptive integration step.
        
        Args:
            spins: Current spin configuration
            dt: Initial time step
            
        Returns:
            Tuple of (updated spins, actual time step used)
        """
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            # Perform embedded RK step (RK4/RK5 pair)
            spins_4th, spins_5th = self._embedded_rk_step(spins, dt)
            
            # Estimate local error
            error = np.max(np.linalg.norm(spins_5th - spins_4th, axis=1))
            
            # Check if error is acceptable
            if error <= self.tolerance:
                # Accept step
                self.accepted_steps += 1
                
                # Calculate new step size for next step
                if error > 0:
                    factor = self.safety_factor * (self.tolerance / error) ** 0.2
                    factor = max(self.min_factor, min(self.max_factor, factor))
                    new_dt = dt * factor
                else:
                    new_dt = dt * self.max_factor
                
                return spins_5th, new_dt
            else:
                # Reject step and reduce time step
                self.rejected_steps += 1
                factor = max(self.min_factor, 
                           self.safety_factor * (self.tolerance / error) ** 0.25)
                dt *= factor
                attempt += 1
        
        # If we get here, use the last attempt regardless
        print(f"Warning: Adaptive integrator reached maximum attempts")
        return spins_4th, dt
    
    def _embedded_rk_step(self, spins: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform embedded RK4/RK5 step for error estimation.
        
        Uses the Dormand-Prince coefficients for the embedded pair.
        """
        # Dormand-Prince coefficients (simplified)
        # In practice, you'd use the full coefficient tables
        
        # Calculate stages
        k1 = self.solver.calculate_llg_rhs(spins)
        
        spins_2 = self._normalize_spins(spins + dt * 0.25 * k1)
        k2 = self.solver.calculate_llg_rhs(spins_2)
        
        spins_3 = self._normalize_spins(spins + dt * (3/32 * k1 + 9/32 * k2))
        k3 = self.solver.calculate_llg_rhs(spins_3)
        
        spins_4 = self._normalize_spins(spins + dt * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k4 = self.solver.calculate_llg_rhs(spins_4)
        
        # Fourth-order solution
        spins_4th = spins + dt * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4)
        
        # Fifth-order solution (simplified)
        spins_5th = spins + dt * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4)
        
        return spins_4th, spins_5th
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.solver.spin_system.spin_magnitude
    
    def get_statistics(self) -> dict:
        """Get integration statistics."""
        total_steps = self.accepted_steps + self.rejected_steps
        acceptance_rate = self.accepted_steps / total_steps if total_steps > 0 else 0
        
        return {
            'accepted_steps': self.accepted_steps,
            'rejected_steps': self.rejected_steps,
            'acceptance_rate': acceptance_rate,
            'total_attempts': total_steps
        }


class SymplecticIntegrator(Integrator):
    """
    Symplectic integrator for conservative LLG dynamics.
    
    Preserves certain geometric properties of the phase space,
    which can be important for long-time simulations.
    """
    
    def step(self, spins: np.ndarray, dt: float) -> np.ndarray:
        """Perform symplectic integration step."""
        # This is a simplified symplectic integrator
        # A full implementation would use proper symplectic methods
        # like the Störmer-Verlet scheme adapted for spins
        
        # For now, use a modified leapfrog scheme
        # In practice, this requires careful treatment of the constraint
        # that spins maintain constant magnitude
        
        # Calculate effective field
        H_eff = self.solver.calculate_effective_field(spins)
        
        # Half step for "momentum" (torque)
        torque = -self.solver.gamma * np.cross(spins, H_eff)
        
        # Update spins (treating as generalized coordinates)
        spins_new = spins + dt * torque
        
        # Re-normalize to constraint surface
        spins_new = self._normalize_spins(spins_new)
        
        # Add damping as a small perturbation
        if self.solver.damping > 0:
            H_eff_new = self.solver.calculate_effective_field(spins_new)
            damping_correction = self.solver.damping * np.cross(
                spins_new, np.cross(spins_new, H_eff_new)
            )
            spin_mag_sq = np.sum(spins_new**2, axis=1, keepdims=True)
            spins_new += dt * damping_correction / spin_mag_sq
            spins_new = self._normalize_spins(spins_new)
        
        return spins_new
    
    def _normalize_spins(self, spins: np.ndarray) -> np.ndarray:
        """Normalize spins to maintain magnitude."""
        magnitudes = np.linalg.norm(spins, axis=1, keepdims=True)
        magnitudes = np.where(magnitudes > 0, magnitudes, 1.0)
        normalized = spins / magnitudes
        return normalized * self.solver.spin_system.spin_magnitude