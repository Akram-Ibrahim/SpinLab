"""Input/output utilities for spin configurations and data."""

import numpy as np
import h5py
from typing import Dict, Any, Optional
import json
from pathlib import Path


def save_configuration(
    filename: str,
    spin_config: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "npy"
):
    """
    Save spin configuration to file.
    
    Args:
        filename: Output filename
        spin_config: Spin configuration array
        metadata: Optional metadata dictionary
        format: File format ("npy", "hdf5", "txt")
    """
    if format == "npy":
        np.save(filename, spin_config)
        if metadata is not None:
            metadata_file = filename.replace('.npy', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    elif format == "hdf5":
        with h5py.File(filename, 'w') as f:
            f.create_dataset('spin_config', data=spin_config)
            if metadata is not None:
                for key, value in metadata.items():
                    f.attrs[key] = value
    
    elif format == "txt":
        header = ""
        if metadata is not None:
            header = json.dumps(metadata)
        
        np.savetxt(filename, spin_config, header=header)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_configuration(
    filename: str,
    format: str = "auto"
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Load spin configuration from file.
    
    Args:
        filename: Input filename
        format: File format ("auto", "npy", "hdf5", "txt")
        
    Returns:
        Tuple of (spin_config, metadata)
    """
    filepath = Path(filename)
    
    if format == "auto":
        if filepath.suffix == ".npy":
            format = "npy"
        elif filepath.suffix in [".h5", ".hdf5"]:
            format = "hdf5"
        elif filepath.suffix == ".txt":
            format = "txt"
        else:
            raise ValueError(f"Cannot determine format from filename: {filename}")
    
    metadata = {}
    
    if format == "npy":
        spin_config = np.load(filename)
        metadata_file = filename.replace('.npy', '_metadata.json')
        if Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
    
    elif format == "hdf5":
        with h5py.File(filename, 'r') as f:
            spin_config = f['spin_config'][:]
            metadata = dict(f.attrs)
    
    elif format == "txt":
        with open(filename, 'r') as f:
            lines = f.readlines()
            if lines[0].startswith('#'):
                header = lines[0][1:].strip()
                try:
                    metadata = json.loads(header)
                except json.JSONDecodeError:
                    metadata = {'header': header}
        
        spin_config = np.loadtxt(filename)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return spin_config, metadata


def save_simulation_results(
    filename: str,
    results: Dict[str, Any],
    format: str = "hdf5"
):
    """
    Save simulation results to file.
    
    Args:
        filename: Output filename
        results: Results dictionary
        format: File format ("hdf5", "npz")
    """
    if format == "hdf5":
        with h5py.File(filename, 'w') as f:
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, dict):
                    group = f.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            group.create_dataset(subkey, data=subvalue)
                        else:
                            group.attrs[subkey] = subvalue
                else:
                    f.attrs[key] = value
    
    elif format == "npz":
        # Flatten nested dictionaries for npz format
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_results[f"{key}_{subkey}"] = subvalue
            else:
                flat_results[key] = value
        
        np.savez_compressed(filename, **flat_results)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_simulation_results(
    filename: str,
    format: str = "auto"
) -> Dict[str, Any]:
    """
    Load simulation results from file.
    
    Args:
        filename: Input filename
        format: File format ("auto", "hdf5", "npz")
        
    Returns:
        Results dictionary
    """
    filepath = Path(filename)
    
    if format == "auto":
        if filepath.suffix in [".h5", ".hdf5"]:
            format = "hdf5"
        elif filepath.suffix == ".npz":
            format = "npz"
        else:
            raise ValueError(f"Cannot determine format from filename: {filename}")
    
    if format == "hdf5":
        results = {}
        with h5py.File(filename, 'r') as f:
            # Load attributes
            for key, value in f.attrs.items():
                results[key] = value
            
            # Load datasets and groups
            def load_group(group, result_dict):
                for key, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        result_dict[key] = item[:]
                    elif isinstance(item, h5py.Group):
                        result_dict[key] = {}
                        # Load group attributes
                        for attr_key, attr_value in item.attrs.items():
                            result_dict[key][attr_key] = attr_value
                        # Load group datasets
                        load_group(item, result_dict[key])
            
            load_group(f, results)
    
    elif format == "npz":
        data = np.load(filename)
        results = dict(data)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return results


def export_to_ase(
    positions: np.ndarray,
    spin_config: np.ndarray,
    cell: np.ndarray,
    symbols: list = None
):
    """
    Export structure and spins to ASE Atoms object.
    
    Args:
        positions: Atomic positions
        spin_config: Spin configuration
        cell: Unit cell
        symbols: Chemical symbols
        
    Returns:
        ASE Atoms object with magnetic moments
    """
    from ase import Atoms
    
    if symbols is None:
        symbols = ['Fe'] * len(positions)
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=True
    )
    
    # Set magnetic moments
    atoms.set_initial_magnetic_moments(spin_config)
    
    return atoms


def import_from_ase(atoms):
    """
    Import structure and spins from ASE Atoms object.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Tuple of (positions, spin_config, cell)
    """
    positions = atoms.get_positions()
    cell = atoms.get_cell().array
    
    # Get magnetic moments if available
    try:
        spin_config = atoms.get_initial_magnetic_moments()
        if spin_config.ndim == 1:
            # Convert scalar moments to z-direction vectors
            spin_config = np.column_stack([
                np.zeros(len(positions)),
                np.zeros(len(positions)),
                spin_config
            ])
    except:
        # No magnetic moments - create default configuration
        spin_config = np.column_stack([
            np.zeros(len(positions)),
            np.zeros(len(positions)),
            np.ones(len(positions))
        ])
    
    return positions, spin_config, cell