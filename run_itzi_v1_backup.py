# run_itzi.py - Simple Itzi 2D Flood Simulation Runner
"""
Runs Itzi 2D flood simulation and exports selected variables.

Usage:
  python run_itzi.py                     # Export all variables
  python run_itzi.py --vars depth,v      # Export only depth and velocity
  python run_itzi.py --list              # Show available variables

Available variables: depth, v, wse, vdir, froude, qx, qy, drainage
"""
import os
import subprocess
import shutil
import argparse

# Available output variables
AVAILABLE_VARS = {
    'depth': ('water_depth', 'max_water_depth.tif', True),       # (itzi_name, filename, has_max)
    'v': ('v', 'max_velocity.tif', True),
    'wse': ('water_surface_elevation', 'last_wse.tif', False),
    'vdir': ('vdir', 'last_velocity_direction.tif', False),
    'froude': ('froude', 'last_froude.tif', False),
    'qx': ('qx', 'last_flow_x.tif', False),
    'qy': ('qy', 'last_flow_y.tif', False),
    'drainage': ('mean_drainage_flow', 'last_drainage_flow.tif', False),
}

# Parse arguments
parser = argparse.ArgumentParser(description='Run Itzi 2D flood simulation')
parser.add_argument('--vars', type=str, default='all',
                    help='Comma-separated list of variables to export (default: all)')
parser.add_argument('--swmm', type=str, default=None,
                    help='Path to SWMM .inp file (overrides config default)')
parser.add_argument('--output', type=str, default=None,
                    help='Output directory (overrides config default)')
parser.add_argument('--list', action='store_true', help='List available variables and exit')
args = parser.parse_args()

if args.list:
    print("\nAvailable variables:")
    print("-" * 50)
    for key, (itzi_name, filename, has_max) in AVAILABLE_VARS.items():
        max_str = "(MAX)" if has_max else "(last timestep)"
        print(f"  {key:10} -> {filename:30} {max_str}")
    print("\nUsage: python run_itzi.py --vars depth,v,froude")
    exit(0)

# Determine which variables to export
if args.vars == 'all':
    SELECTED_VARS = list(AVAILABLE_VARS.keys())  
else:
    SELECTED_VARS = [v.strip() for v in args.vars.split(',')]
    invalid = [v for v in SELECTED_VARS if v not in AVAILABLE_VARS]
    if invalid:
        print(f"Error: Unknown variables: {invalid}")
        print(f"Available: {list(AVAILABLE_VARS.keys())}")
        exit(1)

# Import from config
try:
    import config
    SIMULATION_DURATION_HOURS = config.ITZI_SIMULATION_DURATION_HOURS
    REPORT_STEP_MINUTES = config.REPORT_STEP_MINUTES
    DEM_FILE = str(config.ELEV_FILE)
    MANNING_VALUE = config.MANNING_VALUE
    MANNING_RASTER_FILE = str(config.MANNING_RASTER_FILE)
    SWMM_FILE = str(config.SWMM_FILE)
    GRASSDATA = str(config.GRASSDATA)
    CONFIG_FILE = str(config.ITZI_CONFIG_FILE)
    ITZI_EXE = str(config.ITZI_EXE)
    OUTPUT_DIR = str(config.ITZI_OUTPUT_DIR)
except ImportError:
    # Fallback defaults
    SIMULATION_DURATION_HOURS = 3.5
    REPORT_STEP_MINUTES = 5
    DEM_FILE = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\01_raster\elev_10_dmq_reprojected_clipped.tif"
    MANNING_VALUE = 0.035
    MANNING_RASTER_FILE = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\01_raster\manning_raster.tif"
    SWMM_FILE = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\COLEGIO_TR25_v6.inp"
    GRASSDATA = r"C:\Users\chelo\grassdata\tanques_tormenta"
    CONFIG_FILE = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\itzi_config.ini"
    ITZI_EXE = r"C:\Users\chelo\AppData\Roaming\Python\Python312\Scripts\itzi.exe"
    OUTPUT_DIR = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\itzi_results"

# Override SWMM file if provided via CLI
if args.swmm:
    SWMM_FILE = args.swmm
    print(f"[CLI] Using custom SWMM file: {SWMM_FILE}")

# Override OUTPUT_DIR if provided via CLI
if args.output:
    OUTPUT_DIR = args.output
    CONFIG_FILE = os.path.join(OUTPUT_DIR, 'itzi_config.ini')
    print(f"[CLI] Using custom output dir: {OUTPUT_DIR}")
    print(f"[CLI] Config file will be saved at: {CONFIG_FILE}")

# ============================================================================
# CREATE ITZI CONFIG
# ============================================================================
def crear_config_itzi():
    """Creates Itzi configuration file with SWMM drainage coupling."""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    hours = int(SIMULATION_DURATION_HOURS)
    minutes = int((SIMULATION_DURATION_HOURS - hours) * 60)
    duration_str = f"{hours:02d}:{minutes:02d}:00"
    record_step_str = f"00:{REPORT_STEP_MINUTES:02d}:00"
    
    config_content = f"""[time]
duration = {duration_str}
record_step = {record_step_str}

[grass]
grassdata = {os.path.dirname(GRASSDATA)}
location = {os.path.basename(GRASSDATA)}
mapset = PERMANENT

[input]
dem = dem
friction = friction

[output]
prefix = sim
values = water_depth, water_surface_elevation, v, vdir, mean_boundary_flow

[statistics]
stats = max
values = water_depth,v
stats_file = {OUTPUT_DIR}/itzi_statistics.csv

[drainage]
swmm_inp = {SWMM_FILE}
output = drainage_results

[options]
hmin = 0.01
cfl = 0.4
theta = 0.9
dtmax = 0.5
nprocs = 12
"""
    
    with open(CONFIG_FILE, 'w') as f:
        f.write(config_content)
    
    print(f"Config file created: {CONFIG_FILE}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("\n" + "="*60)
print("ITZI 2D FLOOD SIMULATION")
print("="*60)

# Create config
crear_config_itzi()

# DELETE GRASSDATA completely to ensure clean state (avoids temporal DB conflicts)
if os.path.exists(GRASSDATA):
    print("\nDeleting old GRASS location for fresh start...")
    shutil.rmtree(GRASSDATA)

# Create new GRASS location from DEM
print("\nCreating GRASS location...")
grass_exe = r"C:\Program Files\GRASS GIS 8.4\grass84.bat"
result = subprocess.run([grass_exe, '-c', DEM_FILE, GRASSDATA, '-e'], 
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f"  Warning: {result.stderr}")

# Initialize GRASS session
import grass.script as gscript
import grass.script.setup as gsetup
gsetup.init(os.path.dirname(GRASSDATA), os.path.basename(GRASSDATA), 'PERMANENT')

# Import DEM
print("\nImporting DEM...")
gscript.run_command('r.import', input=DEM_FILE, output='dem', overwrite=True)

# Set region
print("\nSetting computational region...")
gscript.run_command('g.region', raster='dem')

# Import friction map (from Manning raster)
print(f"\nImporting friction map from: {MANNING_RASTER_FILE}")
if os.path.exists(MANNING_RASTER_FILE):
    gscript.run_command('r.import', input=MANNING_RASTER_FILE, output='friction', overwrite=True)
    # Check friction stats
    friction_stats = gscript.parse_command('r.univar', map='friction', flags='g')
    print(f"  Friction Min: {float(friction_stats.get('min', 0)):.4f}")
    print(f"  Friction Max: {float(friction_stats.get('max', 0)):.4f}")
    print(f"  Friction Mean: {float(friction_stats.get('mean', 0)):.4f}")
else:
    print(f"  WARNING: Manning raster not found, using uniform value {MANNING_VALUE}")
    gscript.mapcalc(f'friction = {MANNING_VALUE}')

# DEM Statistics
print("\n--- DEM DIAGNOSTICS ---")
dem_stats = gscript.parse_command('r.univar', map='dem', flags='g')
print(f"  Min elevation: {float(dem_stats.get('min', 0)):.2f} m")
print(f"  Max elevation: {float(dem_stats.get('max', 0)):.2f} m")
print(f"  Range: {float(dem_stats.get('max', 0)) - float(dem_stats.get('min', 0)):.2f} m")

# No need to clean - we start with fresh GRASSDATA each time

# Run Itzi
print("\n" + "="*60)
print("STARTING SIMULATION")
print("="*60)
result = subprocess.run([ITZI_EXE, 'run', CONFIG_FILE])

if result.returncode == 0:
    print("\n" + "="*60)
    print("SIMULATION COMPLETED!")
    print("="*60)
    
    # Create output directory if it doesn't exist (don't delete existing)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # List all raster maps (strip @mapset suffix AND any whitespace/newlines)
    all_maps_raw = gscript.read_command('g.list', type='raster').strip().split('\n')
    all_maps = [m.split('@')[0].strip() for m in all_maps_raw if m.strip()]
    sim_maps = [m for m in all_maps if m.startswith('sim')]
    
    print(f"\nGenerated {len(sim_maps)} result maps")
    
    # Export max statistics maps
    print("\n" + "="*60)
    print("EXPORTING SELECTED VARIABLES TO GEOTIFF")
    print("="*60)
    print(f"Selected: {SELECTED_VARS}")
    
    exported = []
    
    for var_key in SELECTED_VARS:
        if var_key not in AVAILABLE_VARS:
            continue
            
        itzi_name, filename, has_max = AVAILABLE_VARS[var_key]
        
        # Find all timestep maps for this variable
        var_maps = sorted([m for m in sim_maps if f'_{itzi_name}_' in m and 'max' not in m])
        
        # Debug: print found maps
        print(f"  Found {len(var_maps)} maps for {itzi_name}")
        
        if not var_maps:
            print(f"  No maps found for: {itzi_name}")
            continue
        
        # Determine which map to export
        max_map = f"sim_{itzi_name}_max"
        if max_map in sim_maps:
            map_to_export = max_map
            print(f"  Using PRE-COMPUTED MAX: {max_map}")
        else:
            map_to_export = var_maps[-1]
            print(f"  Using LAST TIMESTEP: {map_to_export}")
        
        final_output = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # Try r.out.gdal first with explicit mapset
            full_map_name = f"{map_to_export}@PERMANENT"
            temp_tif = os.path.join(GRASSDATA, filename)
            
            gscript.run_command('r.out.gdal', input=full_map_name, output=temp_tif,
                               format='GTiff', overwrite=True, flags='c', createopt='COMPRESS=LZW')
            
            # Copy to final location
            shutil.copy2(temp_tif, final_output)
            os.remove(temp_tif)
            
            print(f"  ✓ Exported: {filename}")
            exported.append((map_to_export, filename))
            
        except Exception as e1:
            print(f"    r.out.gdal failed: {e1}")
            # Fallback: try using gdal_translate with GRASS driver
            try:
                grass_path = f"GRASS:{GRASSDATA}/PERMANENT/{map_to_export}"
                result = subprocess.run([
                    'gdal_translate', '-of', 'GTiff', 
                    grass_path, final_output
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  ✓ Exported (gdal_translate): {filename}")
                    exported.append((map_to_export, filename))
                else:
                    print(f"    gdal_translate failed: {result.stderr}")
            except Exception as e2:
                print(f"    All export methods failed: {e2}")
    
    # Print statistics for exported maps
    print("\n" + "="*60)
    print("RESULT STATISTICS")
    print("="*60)
    
    for map_name, filename in exported:
        try:
            stats = gscript.parse_command('r.univar', map=map_name, flags='g')
            print(f"\n  {filename.replace('.tif', '')}:")
            print(f"    Max: {float(stats.get('max', 0)):.3f}")
            print(f"    Mean: {float(stats.get('mean', 0)):.3f}")
            print(f"    Cells: {stats.get('n', 0)}")
        except:
            pass
    
    print("\n" + "="*60)
    print(f"RESULTS EXPORTED TO: {OUTPUT_DIR}")
    print("="*60)

else:
    print("\nERROR: Simulation failed")
    print("Check the error messages above")