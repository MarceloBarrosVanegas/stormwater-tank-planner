import sys
import os
from pathlib import Path
from pyproj import CRS

# =============================================================================
# PROJECT ROOTS
# =============================================================================
CODIGOS_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODIGOS_DIR.parent

# =============================================================================
# PROJECT COORDINATE REFERENCE SYSTEM (CRS)
# =============================================================================
# SIRES-DMQ: Sistema de Referencia del DMQ (Quito)
PROJECT_CRS_WKT = """PROJCRS["SIRES-DMQ",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
            LENGTHUNIT["metre",1]],ID["EPSG",6326]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["unnamed",
        METHOD["Transverse Mercator",ID["EPSG",9807]],
        PARAMETER["Latitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",-78.5,
            ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8802]],
        PARAMETER["Scale factor at natural origin",1.0004584,
            SCALEUNIT["unity",1],ID["EPSG",8805]],
        PARAMETER["False easting",500000,
            LENGTHUNIT["metre",1],ID["EPSG",8806]],
        PARAMETER["False northing",10000000,
            LENGTHUNIT["metre",1],ID["EPSG",8807]]],
    CS[Cartesian,2],
    AXIS["(E)",east,ORDER[1],
        LENGTHUNIT["metre",1,ID["EPSG",9001]]],
    AXIS["(N)",north,ORDER[2],
        LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""

PROJECT_CRS = CRS(PROJECT_CRS_WKT)
# =============================================================================
# EXTERNAL DEPENDENCIES (PyPiper)
# =============================================================================
USER_HOME = Path.home()
ONEDRIVE_DIR = USER_HOME / "OneDrive"

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
    PYPIPER_UTIL_GEOMETRY = PYPIPER_DIR / "util" / "geometry"
else:
    PYPIPER_SRC = None
    PYPIPER_GUI = None
    PYPIPER_UTIL_GEOMETRY = None

def setup_sys_path():
    """Adds PyPiper paths to sys.path if not present (at beginning for priority)."""
    if PYPIPER_UTIL_GEOMETRY and str(PYPIPER_UTIL_GEOMETRY) not in sys.path:
        sys.path.insert(0, str(PYPIPER_UTIL_GEOMETRY))
    if PYPIPER_GUI and str(PYPIPER_GUI) not in sys.path:
        sys.path.insert(0, str(PYPIPER_GUI))
    if PYPIPER_SRC and str(PYPIPER_SRC) not in sys.path:
        sys.path.insert(0, str(PYPIPER_SRC))

# =============================================================================
# DATA DIRECTORIES
# =============================================================================
GIS_DIR = PROJECT_ROOT / "gis"
RASTER_DIR = GIS_DIR / "01_raster"
VECTOR_DIR = GIS_DIR / "00_vector"
FLOODING_STATS_DIR = CODIGOS_DIR / "00_flooding_stats"
FAILIURE_RISK_FILE = CODIGOS_DIR / "probabilistic_results/risk_estimation/04_spatial_analysis/pipe_failure_probability.gpkg"

# =============================================================================
# MAIN PROJECT FILES
# =============================================================================
# SWMM
SWMM_FILE = CODIGOS_DIR / "COLEGIO_TR25_v6.inp"
BASE_INP_TR = 25  # Return Period (years) of the base INP file

# Return Period List for analysis:
#   [] or [BASE_INP_TR] = Deterministic (single TR from config)
#   [tr1, tr2, ...]     = Probabilistic (EAD calculation)
TR_LIST = [BASE_INP_TR]  # Change to [1,2,5,10,25,50,100] for probabilistic EAD
VALIDATION_TR_LIST = [1,2,5,10,25,50,100]  # For NSGA final validation
N_GENERATIONS = 300  # NSGA generations
POP_SIZE = 100  # NSGA population
MIN_TANKS = 3   # Minimum number of active tanks (constraint for NSGA)


# Rasters
ELEV_FILE_ORIGINAL = RASTER_DIR / "elev_10_dmq_reprojected_clipped.tif"
ELEV_FILE_CARVED = RASTER_DIR / "dem_carved.tif"
ELEV_FILE = ELEV_FILE_CARVED  # Use carved DEM for Itzi simulation


# Vectors
PREDIOS_FILE = VECTOR_DIR / "07_predios_disponibles.shp"
NETWORK_FILE = VECTOR_DIR / "06_red_principal.gpkg"
FLOODING_NODES_FILE = FLOODING_STATS_DIR / "00_flooding_nodes.gpkg"
PREDIOS_DAMAGED_FILE = VECTOR_DIR / "predios_proyecto.gpkg"

# Other
DEFAULT_NODES_XLSX = FLOODING_STATS_DIR / "00_flooding_nodes.xlsx"
OSM_CACHE_PATH = PROJECT_ROOT / "osm_cache.graphml"
BASE_PRECIOS = CODIGOS_DIR / "base_precios.xlsx"


# =============================================================================
# ITZI 2D FLOOD SIMULATION SETTINGS
# =============================================================================
# GRASS GIS location for Itzi
GRASSDATA = USER_HOME / "grassdata" / "tanques_tormenta"

# Itzi executable
ITZI_EXE = USER_HOME / "AppData/Roaming/Python/Python312/Scripts/itzi.exe"

# Itzi output directory
ITZI_OUTPUT_DIR = CODIGOS_DIR / "itzi_results"

# Itzi config file
ITZI_CONFIG_FILE = CODIGOS_DIR / "itzi_config.ini"

# Manning friction coefficient (uniform)
MANNING_VALUE = 0.035

# Manning friction raster (variable by surface type)
MANNING_RASTER_FILE = RASTER_DIR / "manning_raster.tif"

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
ITZI_SIMULATION_DURATION_HOURS = 3.5  # Duration for Itzi/SWMM simulation (Hours)
REPORT_STEP_MINUTES = 5               # Report step for SWMM and Itzi

# Reporting Parameters
SHOW_PREDIOS_IN_REPORTS = False  # Enable/Disable predios background in comparison maps (slow)

# GREEN (NATURAL) SCENARIO PARAMETERS
GREEN_SCENARIO_CN = 76.0              # Default Curve Number for natural state (Forest/Pasture)
GREEN_SCENARIO_IMPERV = 15.0           # 3% residual imperviousness (rocks, clay, etc.)

# =============================================================================
# COST PARAMETERS
# =============================================================================


# Modular cost components for optimization
COST_COMPONENTS = {
    'deferred_investment': True,   # Pipe replacement cost (DeferredInvestmentCost)
    'flood_damage': False,          # Building damage from CLIMADA
    'river_damage': False,         # Future: downstream environmental impact
}

# =============================================================================
# TANK DESIGN PARAMETERS
# =============================================================================
CAPACITY_MAX_HD = 0.4 # Maximum h/D ratio in conduits
CAPACITY_MAX_FOR_AVOIDED_COST = 0.75

# 'flooding' for just avoid flooding nodes   or  'capacity' for avoiding h/D > DERIVATION_MAX_HD in all conduits
TANK_OPT_OBJECTIVE = 'flooding'
# TANK_OPT_OBJECTIVE = 'capacity'
MINIMUN_FLOODING_FLOW = 0.1  # Minimum flooding flow to consider a node flooded (m3/s)
MAX_ITERATIONS = 100  # Max iterations for tank sizing convergence
MAX_RESIZE_ITERATIONS = 10 # Max iterations for tank resizing convergence

TANK_DEPTH_M = 10.0              # Default tank depth in meters
TANK_MIN_VOLUME_M3 = 1000.0     # Minimum tank volume in cubic meters
TANK_MAX_VOLUME_M3 = 100000.0    # Maximum tank volume in cubic meters

TANK_MIN_UTILIZATION_PCT = 20.0 # Minimum tank utilization % (warn if below this)
TANK_VOLUME_ROUND_M3 = 100      # Round volumes to this increment for display
MAX_TANKS = 20
MAX_PRUNE_RETRIES =  2  # Max retries for pruning tanks
MAX_PREDIO_SLOPE = 30.0 # Maximum allowed predio slope in %. Predios steeper than this are discarded.
PREDIO_MAX_OCCUPANCY_RATIO = 0.85  # Exclude predios with >85% area occupied from path search

# Tank Volume Sizing
TANK_VOLUME_SAFETY_FACTOR = 1.05 # Safety factor applied to flooding volume
TANK_OCCUPATION_FACTOR = 150    # Extra space factor for access, pumps, maneuvering in tank area calculation


WEIR_CREST_MIN_M = 0.1         # Minimum weir crest height above tank bottom (m)
WEIR_DISCHARGE_COEFF = 1.84   # Weir discharge coefficient (Cd) for rectangular sharp-crested weir
DERIVATION_MIN_DISTANCE_M = 1  # Minimum distance (m) between derivation points on same pipe line
MIN_DETPH_FOR_DERIVATION_M = 6.0  # Minimum pipe depth tunnel

# Pipeline Design Defaults (used in rut_16)
MIN_PIPE_SLOPE = 0.004           # Minimum pipe slope (0.4%) for gravity flow
DEFAULT_PIPE_MATERIAL = 'HA'     # Default pipe material (Hormigón Armado)
DEFAULT_PIPE_SECTION = 'rectangular'  # Default pipe cross-section type
DEFAULT_PIPE_RUGOSITY = 'liso'   # Default pipe roughness category
DEFAULT_POZO_DEPTH = 6.0         # Default manhole/pozo depth (m)
MIN_PIPE_DIAMETER = 1.8          # Minimum pipe diameter (m)

TOLERANCE = 20  # min distance to simplify paths

# =============================================================================

# LAND COST PARAMETERS
LAND_COST_PER_M2 = 50.0         # Base cost per square meter of land
PREDIO_VIOLATION_THRESHOLD = 1000.0 # Threshold for penalizing oversized tanks occupying predio

# =============================================================================
# PROJECT METADATA (for reports and Excel output)
# =============================================================================
EXCEL_METADATA = {
    'proyecto_name': 'REDUCCION DE CAUDALES DE DESCARGA CON TANQUES DE TORMENTA',
    'sistema': 'SUBSISTEMA EL COLEGIO OCIDENTAL',
    'ubicacion': 'QUITO',
    'obra': 'COSTO DE REPOSICION DE TUBERIAS',
    'cliente': 'EPMAPS'
}

# Default if not provided
DEFAULT_PATH_WEIGHTS = {
    'length_weight': 1.0,     # 80% distancia - PRIORIDAD ALTA para rutas más cortas
    'elevation_weight': 0.0,  # 10% elevación - evitar subidas
    'road_weight': 0.0        # 10% tipo de calle - menor importancia
}

DEFAULT_ROAD_PREFERENCES = {
    'motorway': 5.0, 'trunk': 1.0, 'primary': 1.0,
    'secondary': 1.2, 'tertiary': 1.3, 'residential': 1.5,
    'service': 1.5, 'unclassified': 1.5,
    'footway': 5.0, 'path': 5.0, 'steps': 10.0,
    'default': 1.5
}


FLOODING_RANKING_WEIGHTS = {
    'total_volume': 0.5,
    'total_flow': 0.5,
    'failure_probability': 0.,
}

# =============================================================================
# INITIALIZATION MESSAGE
# =============================================================================
print(f"[Config] Project Root: {PROJECT_ROOT}")
if PYPIPER_DIR:
    print(f"[Config] PyPiper found at: {PYPIPER_DIR}")
else:
    print(f"[Config] PyPiper NOT FOUND within candidates.")
