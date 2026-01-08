"""
Run Itzi from Conda via GRASS Shell subprocess
===============================================

This script runs your existing run_itzi.py from Conda by calling
GRASS Shell as a subprocess. This avoids all DLL/NumPy conflicts.

Usage:
    python run_itzi_from_conda.py
    python run_itzi_from_conda.py --vars depth,v
"""

import subprocess
import argparse
import os
from pathlib import Path

# Paths
GRASS_BAT = r"C:\Program Files\GRASS GIS 8.4\grass84.bat"
SCRIPT_DIR = Path(__file__).parent
RUN_ITZI_SCRIPT = SCRIPT_DIR / "run_itzi.py"


def main():
    parser = argparse.ArgumentParser(description='Run Itzi via GRASS Shell')
    parser.add_argument('--vars', type=str, default='depth,v',
                       help='Variables to export (default: depth,v)')
    args = parser.parse_args()
    
    print("="*60)
    print("RUNNING ITZI VIA GRASS SHELL")
    print("="*60)
    print(f"  GRASS: {GRASS_BAT}")
    print(f"  Script: {RUN_ITZI_SCRIPT}")
    
    # Build command
    cmd = [GRASS_BAT, "--exec", "python", str(RUN_ITZI_SCRIPT)]
    
    if args.vars:
        cmd.extend(["--vars", args.vars])
    
    print(f"  Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    # Clean environment to avoid conda/GRASS conflicts
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
    
    # Run with clean environment
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("[OK] ITZI SIMULATION COMPLETE")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("[ERROR] ITZI SIMULATION FAILED")
        print("="*60)
    
    return result.returncode


if __name__ == "__main__":
    exit(main())
