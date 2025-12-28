"""
rut_18_itzi_flood_model.py - Itzi Simulation Module for Optimization
====================================================================

Module to run Itzi simulations from the optimization loop.
Calls run_itzi.py via GRASS Shell with clean environment.

Usage:
    from rut_18_itzi_flood_model import run_itzi_for_case
    
    # Run with default SWMM file
    result = run_itzi_for_case()
    
    # Run with custom SWMM file
    result = run_itzi_for_case(swmm_file="path/to/modified.inp")
    
    # Copy results to case directory
    result = run_itzi_for_case(
        swmm_file="path/to/modified.inp",
        output_dir="path/to/case_001"
    )
"""

import subprocess
import shutil
import os
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# PATHS
# ============================================================================
GRASS_BAT = r"C:\Program Files\GRASS GIS 8.4\grass84.bat"
CODIGOS_DIR = Path(__file__).parent.resolve()
RUN_ITZI_SCRIPT = CODIGOS_DIR / "run_itzi.py"

try:
    import config
    DEFAULT_SWMM_FILE = Path(config.SWMM_FILE)
    DEFAULT_OUTPUT_DIR = Path(config.ITZI_OUTPUT_DIR)
except ImportError:
    DEFAULT_SWMM_FILE = CODIGOS_DIR / "COLEGIO_TR25_v6.inp"
    DEFAULT_OUTPUT_DIR = CODIGOS_DIR / "itzi_results"


def _get_clean_env():
    """Get environment with conda paths removed to avoid GRASS conflicts."""
    env = os.environ.copy()
    
    # Remove conda paths and invalid entries from PATH
    path_parts = env.get('PATH', '').split(os.pathsep)
    clean_path = []
    for p in path_parts:
        # Skip conda paths
        if 'miniconda' in p.lower() or 'anaconda' in p.lower():
            continue
        # Skip empty or just dots (causes WinError 87)
        if not p or p == '.' or p == '..':
            continue
        clean_path.append(p)
    env['PATH'] = os.pathsep.join(clean_path)
    
    # Remove conda environment variables
    env.pop('CONDA_PREFIX', None)
    env.pop('CONDA_DEFAULT_ENV', None)
    
    # Force GRASS PROJ and GDAL to avoid version warnings
    grass_share = r"C:\Program Files\GRASS GIS 8.4\share"
    env['PROJ_LIB'] = os.path.join(grass_share, 'proj')
    env['GDAL_DATA'] = os.path.join(grass_share, 'gdal')
    
    return env


def run_itzi_for_case(
    swmm_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    vars_to_export: list = None,
    verbose: bool = True
) -> Dict:
    """
    Run Itzi simulation via GRASS Shell.
    
    Args:
        swmm_file: If provided, copies this SWMM file to the default location 
                   before running (for optimization cases)
        output_dir: If provided, copies results here after running
        vars_to_export: Variables to export ['depth', 'v', etc.]
        verbose: Print progress
        
    Returns:
        Dict with:
            - success: bool
            - output_dir: Path to results
            - max_depth_file: Path to max water depth raster
            - max_velocity_file: Path to max velocity raster
    """
    vars_to_export = vars_to_export or ['depth', 'v']
    
    # If custom SWMM file provided, copy it to the expected location
    if swmm_file and Path(swmm_file).exists():
        swmm_file = Path(swmm_file)
        if swmm_file != DEFAULT_SWMM_FILE:
            if verbose:
                print(f"  Copying SWMM file to: {DEFAULT_SWMM_FILE}")
            shutil.copy2(swmm_file, DEFAULT_SWMM_FILE)
    
    if verbose:
        print("="*60)
        print("RUNNING ITZI VIA GRASS SHELL")
        print("="*60)
        print(f"  GRASS: {GRASS_BAT}")
        print(f"  Script: {RUN_ITZI_SCRIPT}")
    
    # Build command
    vars_str = ','.join(vars_to_export)
    cmd = [GRASS_BAT, "--exec", "python", str(RUN_ITZI_SCRIPT), "--vars", vars_str]
    
    if verbose:
        print(f"  Command: {' '.join(cmd)}")
        print("="*60 + "\n")
    
    # Run with clean environment
    env = _get_clean_env()
    result = subprocess.run(cmd, cwd=str(CODIGOS_DIR), env=env)
    
    success = result.returncode == 0
    
    # Results
    output = {
        'success': success,
        'output_dir': str(DEFAULT_OUTPUT_DIR),
        'max_depth_file': str(DEFAULT_OUTPUT_DIR / 'max_water_depth.tif'),
        'max_velocity_file': str(DEFAULT_OUTPUT_DIR / 'max_velocity.tif'),
    }
    
    # Copy results to custom output_dir if specified
    if output_dir and Path(output_dir) != DEFAULT_OUTPUT_DIR:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in ['max_water_depth.tif', 'max_velocity.tif', 'itzi_statistics.csv', 'itzi_config.ini']:
            src = DEFAULT_OUTPUT_DIR / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)
        
        output['output_dir'] = str(output_dir)
        output['max_depth_file'] = str(output_dir / 'max_water_depth.tif')
        output['max_velocity_file'] = str(output_dir / 'max_velocity.tif')
    
    # Try to read max depth
    depth_file = Path(output['max_depth_file'])
    if depth_file.exists():
        try:
            import rasterio
            with rasterio.open(depth_file) as src:
                data = src.read(1)
                output['max_depth_m'] = float(data.max())
        except:
            pass
    
    if verbose:
        if success:
            print("\n" + "="*60)
            print("[OK] ITZI SIMULATION COMPLETE")
            print("="*60)
            if 'max_depth_m' in output:
                print(f"  Max depth: {output['max_depth_m']:.2f} m")
        else:
            print("\n" + "="*60)
            print("[ERROR] ITZI SIMULATION FAILED")
            print("="*60)
    
    return output


# ============================================================================
# MAIN - Example usage
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ITZI RUNNER MODULE TEST")
    print("="*60)
    
    # Example: run with defaults
    result = run_itzi_for_case(verbose=True)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Example for optimization loop:
    # result = run_itzi_for_case(
    #     swmm_file="cases/case_001/modified_swmm.inp",
    #     output_dir="cases/case_001/itzi_results"
    # )
