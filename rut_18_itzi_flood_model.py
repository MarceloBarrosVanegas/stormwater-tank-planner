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
import rasterio
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
    
    # Determine which SWMM file to use for simulation
    # IMPORTANT: Never overwrite the original project SWMM file!
    if swmm_file and Path(swmm_file).exists():
        # Use the custom SWMM file directly (it's already in the case directory)
        swmm_to_use = Path(swmm_file)
        if verbose:
            print(f"  Using custom SWMM file: {swmm_to_use}")
    else:
        # Use the default project file (baseline simulation)
        swmm_to_use = DEFAULT_SWMM_FILE
        if verbose:
            print(f"  Using default SWMM file: {swmm_to_use}")
    
    if verbose:
        print("="*60)
        print("RUNNING ITZI VIA GRASS SHELL")
        print("="*60)
        print(f"  GRASS: {GRASS_BAT}")
        print(f"  Script: {RUN_ITZI_SCRIPT}")
    
    # Build command with custom SWMM file path
    vars_str = ','.join(vars_to_export)
    cmd = [GRASS_BAT, "--exec", "python", str(RUN_ITZI_SCRIPT), 
           "--vars", vars_str, 
           "--swmm", str(swmm_to_use)]
    
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
        
        # Copy result files from default output
        for fname in ['max_water_depth.tif', 'max_velocity.tif', 'itzi_statistics.csv']:
            src = DEFAULT_OUTPUT_DIR / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)
        
        # Copy itzi_config.ini from codigos/ (where run_itzi.py creates it)
        config_src = CODIGOS_DIR / 'itzi_config.ini'
        if config_src.exists():
            shutil.copy2(config_src, output_dir / 'itzi_config.ini')
        
        output['output_dir'] = str(output_dir)
        output['max_depth_file'] = str(output_dir / 'max_water_depth.tif')
        output['max_velocity_file'] = str(output_dir / 'max_velocity.tif')
    
    # Try to read max depth
    depth_file = Path(output['max_depth_file'])
    if depth_file.exists():
        try:

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
    
    # Carpeta de salida para flood damage
    output_dir = r'C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\test_avoided_cost\avoided_cost\flood_damage'
    
    # INP file desde config
    swmm_file = config.SWMM_FILE
    print(f"  SWMM file: {swmm_file}")
    
    # Run ITZI with specified SWMM and copy results to flood_damage folder
    result = run_itzi_for_case(
        swmm_file=swmm_file,
        output_dir=output_dir,
        vars_to_export=['depth', 'v'],
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Verificar que los rasters estén en flood_damage
    from pathlib import Path
    print("\n" + "="*60)
    print("FILES IN OUTPUT DIR")
    print("="*60)
    for f in Path(output_dir).glob('*'):
        print(f"  {f.name}")

