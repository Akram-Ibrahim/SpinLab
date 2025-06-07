"""Physical constants and unit conversions."""

import numpy as np

# Physical constants
PHYSICAL_CONSTANTS = {
    # Boltzmann constant
    'kB': 8.617333e-5,  # eV/K
    'kB_SI': 1.380649e-23,  # J/K
    
    # Bohr magneton
    'mu_B': 5.78838e-5,  # eV/T
    'mu_B_SI': 9.274010e-24,  # J/T
    
    # Gyromagnetic ratio for electron
    'gamma_e': 1.76085963e11,  # rad/(s·T)
    
    # Planck constant
    'hbar': 6.582119e-16,  # eV·s
    'hbar_SI': 1.054572e-34,  # J·s
    
    # Elementary charge
    'e': 1.602176e-19,  # C
    
    # Vacuum permeability
    'mu_0': 4*np.pi*1e-7,  # H/m
    
    # Speed of light
    'c': 2.99792458e8,  # m/s
}

# Unit conversion factors
UNIT_CONVERSIONS = {
    # Energy
    'eV_to_J': 1.602176e-19,
    'meV_to_eV': 1e-3,
    'K_to_eV': 8.617333e-5,
    
    # Magnetic field
    'T_to_Oe': 1e4,
    'Oe_to_T': 1e-4,
    
    # Length
    'Angstrom_to_m': 1e-10,
    'm_to_Angstrom': 1e10,
    
    # Time
    'fs_to_s': 1e-15,
    'ps_to_s': 1e-12,
    'ns_to_s': 1e-9,
}


def convert_units(value, from_unit: str, to_unit: str) -> float:
    """
    Convert between different units.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    conversion_key = f"{from_unit}_to_{to_unit}"
    
    if conversion_key in UNIT_CONVERSIONS:
        return value * UNIT_CONVERSIONS[conversion_key]
    else:
        # Try reverse conversion
        reverse_key = f"{to_unit}_to_{from_unit}"
        if reverse_key in UNIT_CONVERSIONS:
            return value / UNIT_CONVERSIONS[reverse_key]
        else:
            raise ValueError(f"Unknown unit conversion: {from_unit} to {to_unit}")


def temperature_to_energy(temperature: float, unit: str = "K") -> float:
    """
    Convert temperature to energy units.
    
    Args:
        temperature: Temperature value
        unit: Temperature unit ("K" or "eV")
        
    Returns:
        Energy in eV
    """
    if unit == "K":
        return temperature * PHYSICAL_CONSTANTS['kB']
    elif unit == "eV":
        return temperature
    else:
        raise ValueError(f"Unknown temperature unit: {unit}")


def energy_to_temperature(energy: float, unit: str = "K") -> float:
    """
    Convert energy to temperature units.
    
    Args:
        energy: Energy in eV
        unit: Target temperature unit ("K" or "eV")
        
    Returns:
        Temperature in specified unit
    """
    if unit == "K":
        return energy / PHYSICAL_CONSTANTS['kB']
    elif unit == "eV":
        return energy
    else:
        raise ValueError(f"Unknown temperature unit: {unit}")


def magnetic_field_to_energy(B_field: float, g_factor: float = 2.0) -> float:
    """
    Convert magnetic field to energy units.
    
    Args:
        B_field: Magnetic field in Tesla
        g_factor: Landé g-factor
        
    Returns:
        Energy in eV
    """
    return g_factor * PHYSICAL_CONSTANTS['mu_B'] * B_field