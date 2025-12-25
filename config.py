import sys
import os
from pathlib import Path

# --- PROJECT ROOTS ---
# Automatically detect 'codigos' directory and Project Root
CODIGOS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODIGOS_DIR.parent

# --- EXTERNAL DEPENDENCIES ---
# PyPiper (ALCANTARILLADO_PyQt5)
# Assuming standard structure relative to User Home or OneDrive
# Try to find OneDrive for the current user
USER_HOME = Path.home()
ONEDRIVE_DIR = USER_HOME / "OneDrive"

# Fallback strategies for PyPiper
POSSIBLE_PYPIPER_PATHS = [
    ONEDRIVE_DIR / "ALCANTARILLADO_PyQt5/00_MODULOS/pypiper",
    USER_HOME / "ALCANTARILLADO_PyQt5/00_MODULOS/pypiper",
    
]

PYPIPER_DIR = None
for p in POSSIBLE_PYPIPER_PATHS:
    if p.exists():
        PYPIPER_DIR = p
        break

if PYPIPER_DIR:
    PYPIPER_SRC = PYPIPER_DIR / "src"
    PYPIPER_GUI = PYPIPER_DIR / "gui"
else:
    print("WARNING: PyPiper directory not found. Dependencies found later might fail.")
    PYPIPER_SRC = None
    PYPIPER_GUI = None

# --- ADD TO SYS.PATH (Helper) ---
def setup_sys_path():
    """Adds PyPiper paths to sys.path if not present."""
    if PYPIPER_SRC and str(PYPIPER_SRC) not in sys.path:
        sys.path.append(str(PYPIPER_SRC))
    if PYPIPER_GUI and str(PYPIPER_GUI) not in sys.path:
        sys.path.append(str(PYPIPER_GUI))

# --- DATA PATHS ---
GIS_DIR = PROJECT_ROOT / "gis"
RASTER_DIR = GIS_DIR / "01_raster"
FLOODING_STATS_DIR = CODIGOS_DIR / "00_flooding_stats"

# Default Files
DEFAULT_NODES_XLSX = FLOODING_STATS_DIR / "00_flooding_nodes.xlsx"
OSM_CACHE_PATH = PROJECT_ROOT / "osm_cache.graphml"

# --- CONSTANTS ---
FLOODING_COST_PER_M3 = 1250.0

print(f"[Config] Project Root: {PROJECT_ROOT}")
if PYPIPER_DIR:
    print(f"[Config] PyPiper found at: {PYPIPER_DIR}")
else:
    print(f"[Config] PyPiper NOT FOUND within candidates.")
