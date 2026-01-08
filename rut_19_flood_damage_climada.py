"""
rut_19_flood_damage_climada.py
Flood damage assessment using CLIMADA with JRC depth-damage curves.

Uses:
- Itzi water depth raster (max_water_depth.tif)
- Predios GPKG with uso_vige (land use) for sector mapping
- CLIMADA JRC South America depth-damage curves by sector

Requirements:
    pip install climada climada-petals

Output:
- Total flood damage cost (USD)
- Damage per predio
"""
# =============================================================================
# IMPORTS
# =============================================================================
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
import config

# Paths
project_root = config.PROJECT_ROOT
vector_dir = config.VECTOR_DIR
itzi_output_dir = config.ITZI_OUTPUT_DIR

# Input files - Use predios_proyecto.gpkg (clipped to project area with real cadastral values)
predios_file = vector_dir / "predios_proyecto.gpkg"
depth_raster_file = itzi_output_dir / "max_water_depth.tif"

# Output files
damage_results_gpkg = itzi_output_dir / "flood_damage_results.gpkg"
damage_results_txt = itzi_output_dir / "flood_damage_report.txt"

# Columns for land use and property values
USE_COLUMN = "uso_vige"           # For 08_predios_curvas.gpkg
DESTINO_COLUMN = "desteconom"     # For predioGeo.shp
VALOR_TOTAL_COLUMN = "valtotal"   # Real cadastral value
VALOR_CONSTRUCCION_COLUMN = "valconstru"
VALOR_TERRENO_COLUMN = "valterreno"

# =============================================================================
# LAND USE -> CLIMADA SECTOR MAPPING
# Supports both uso_vige (08_predios_curvas) and desteconom (predioGeo.shp)
# =============================================================================
# CLIMADA JRC sectors available: 
# residential, commercial, industrial, transport, infrastructure, agriculture

# Mapping for uso_vige column (08_predios_curvas.gpkg)
LAND_USE_TO_SECTOR = {
    # Residential
    "Resid Urbano 1": "residential",
    "Resid Urbano 2": "residential",
    "Resid Urbano 3": "residential",
    "Resid Urbano 1QT": "residential",
    "Resid Rural 1": "residential",
    "Agricola Resid.": "residential",
    # Commercial / Mixed
    "Multiple": "commercial",
    "Area promocion": "commercial",
    # Industrial
    "Industrial 2": "industrial",
    "Industrial 3": "industrial",
    # Infrastructure
    "Equipamiento": "infrastructure",
    # Agriculture / Conservation
    "P. Ecol/Conser. Patri. N": "agriculture",
}

# Mapping for desteconom column (predioGeo.shp - real cadastral data)
# All destino values are mapped to CLIMADA sectors
DESTINO_ECONOMICO_TO_SECTOR = {
    # =========================================================================
    # RESIDENTIAL (curva CLIMADA JRC)
    # =========================================================================
    "RESIDENCIAL": "residential",
    "VIVIENDA": "residential",
    "HABITACIONAL": "residential",
    "HABITACIONAL / AGROPECUARIA": "residential",
    "LENOCINIO": "commercial",  # Commercial activity
    "DIPLOMÁTICO": "infrastructure",
    "DIPLOMÃTICO": "infrastructure",  # Encoding roto
    
    # =========================================================================
    # COMMERCIAL (curva CLIMADA JRC)
    # =========================================================================
    "COMERCIAL": "commercial",
    "COMERCIO": "commercial",
    "MIXTO": "commercial",
    "HOTEL": "commercial",
    "BANCO - FINANCIERA": "commercial",
    "GASOLINERA": "commercial",
    "SERVICIOS": "commercial",
    "TRANSPORTE TERRESTRE": "commercial",
    
    # =========================================================================
    # INDUSTRIAL (curva CLIMADA JRC)
    # =========================================================================
    "INDUSTRIAL": "industrial",
    "INDUSTRIA": "industrial",
    
    # =========================================================================
    # INFRASTRUCTURE (curva CUSTOM - equipamiento público)
    # =========================================================================
    "EDUCACIÓN": "infrastructure",
    "EDUCACION": "infrastructure",
    "EDUCACIÃN": "infrastructure",  # Encoding roto
    "SALUD": "infrastructure",
    "EQUIPAMIENTO": "infrastructure",
    "RECREACION": "infrastructure",
    "RECREACIÓN": "infrastructure",
    "RECREACIÃN Y DEPORTE": "infrastructure",  # Encoding roto
    "RECREACION Y DEPORTE": "infrastructure",
    "RECREACIÓN Y DEPORTE": "infrastructure",
    "PUBLICO": "infrastructure",
    "PÚBLICO": "infrastructure",
    "PÃBLICO": "infrastructure",  # Encoding roto
    "INSTITUCIONAL": "infrastructure",
    "INSTITUCIONAL PUBLICO": "infrastructure",
    "INSTITUCIONAL PÚBLICO": "infrastructure",
    "INSTITUCIONAL PÃBLICO": "infrastructure",  # Encoding roto
    "INSTITUCIONAL PRIVADO": "infrastructure",
    "CULTURA": "infrastructure",
    "RELIGIOSO": "infrastructure",
    "ASISTENCIA SOCIAL": "infrastructure",
    "ESPACIO PUBLICO": "infrastructure",
    "ESPACIO PÚBLICO": "infrastructure",
    "ESPACIO PÃBLICO": "infrastructure",  # Encoding roto
    
    # =========================================================================
    # AGRICULTURE (curva CUSTOM - cultivos/conservación)
    # =========================================================================
    "AGRICOLA": "agriculture",
    "AGRÍCOLA": "agriculture",
    "PROTECCION": "agriculture",
    "PROTECCIÓN": "agriculture",
    "PROTECCIÃN ECOLÃGICA": "agriculture",  # Encoding roto
    "PROTECCION ECOLOGICA": "agriculture",
    "CONSERVACION": "agriculture",
    "CONSERVACIÓN": "agriculture",
    "RECURSOS NATURALES": "agriculture",
}

DEFAULT_SECTOR = "residential"

# Fallback value when property has no cadastral value
DEFAULT_VALUE_PER_M2 = 400.0  # USD/m²

# =============================================================================
# CUSTOM DAMAGE CURVES FOR SECTORS NOT IN CLIMADA (JRC/FEMA/FAO based)
# =============================================================================
CUSTOM_DAMAGE_CURVES = {
    # Infrastructure/Roads - based on JRC/FEMA methodology
    # Roads are more resilient than buildings
    "infrastructure": {
        "depths": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        "mdr": [0.00, 0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90],
        "source": "Custom (JRC/FEMA methodology)"
    },
    # Agriculture - based on JRC/FAO methodology  
    # Crops are highly vulnerable at low depths
    "agriculture": {
        "depths": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        "mdr": [0.00, 0.35, 0.55, 0.70, 0.80, 0.90, 0.95, 1.00, 1.00],
        "source": "Custom (JRC/FAO methodology)"
    }
}


def get_max_depth_per_predio(predios_gdf, depth_raster_path):
    """
    Extract maximum water depth for each predio polygon from the raster.
    FAST VERSION: Filters first, then samples centroids + zonal stats only for affected.
    """
    import rasterio
    from shapely.geometry import box
    
    print("    Cargando raster de profundidad...")
    
    with rasterio.open(depth_raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        depth_data = src.read(1)  # Read full raster into memory
        transform = src.transform
        nodata = src.nodata if src.nodata else 0
        
        print(f"    Raster CRS: {raster_crs}")
        print(f"    Raster shape: {depth_data.shape}, bounds: {raster_bounds}")
        print(f"    Predios CRS: {predios_gdf.crs}")
        
        # Check where there's actual flood data (depth > 0)
        flooded_pixels = depth_data > 0
        n_flooded_pixels = flooded_pixels.sum()
        print(f"    Píxeles inundados: {n_flooded_pixels:,} ({100*n_flooded_pixels/depth_data.size:.2f}%)")
        
        if n_flooded_pixels == 0:
            print("    ⚠ No hay datos de inundación en el raster!")
            predios_gdf['max_depth_m'] = 0.0
            predios_gdf['mean_depth_m'] = 0.0
            return predios_gdf
    
    # Reproject predios to raster CRS if needed
    original_crs = predios_gdf.crs
    if predios_gdf.crs != raster_crs:
        print(f"    Reproyectando predios a CRS del raster...")
        predios_gdf = predios_gdf.to_crs(raster_crs)
    
    # Initialize columns
    predios_gdf['max_depth_m'] = 0.0
    predios_gdf['mean_depth_m'] = 0.0
    
    # Filter predios to raster extent
    raster_bbox = box(raster_bounds.left, raster_bounds.bottom, 
                      raster_bounds.right, raster_bounds.top)
    
    in_extent_mask = predios_gdf.geometry.intersects(raster_bbox)
    n_in_extent = in_extent_mask.sum()
    print(f"    Predios en extent del raster: {n_in_extent:,}")
    
    if n_in_extent == 0:
        print("    ⚠ Ningún predio en el extent del raster!")
        return predios_gdf
    
    # FAST APPROACH: Sample centroid first, then only do zonal stats for potentially flooded
    print("    Muestreando centroides (rápido)...")
    
    predios_subset = predios_gdf.loc[in_extent_mask].copy()
    centroids = predios_subset.geometry.centroid
    
    # Sample raster at centroids
    with rasterio.open(depth_raster_path) as src:
        # Get pixel coordinates for centroids
        coords = [(pt.x, pt.y) for pt in centroids]
        centroid_depths = list(src.sample(coords))
        centroid_depths = [d[0] if d[0] != nodata else 0.0 for d in centroid_depths]
    
    predios_subset['_centroid_depth'] = centroid_depths
    
    # Quick estimate: predios with flooded centroid are definitely affected
    n_centroid_flooded = (predios_subset['_centroid_depth'] > 0).sum()
    print(f"    Predios con centroide inundado: {n_centroid_flooded:,}")
    
    # For those with flooded centroids, use as initial estimate
    # For accurate max depth, would need full zonal stats (slow)
    predios_gdf.loc[in_extent_mask, 'max_depth_m'] = predios_subset['_centroid_depth'].values
    predios_gdf.loc[in_extent_mask, 'mean_depth_m'] = predios_subset['_centroid_depth'].values
    
    # For more accuracy on large flooded areas, do zonal stats on subset only
    if n_centroid_flooded > 0 and n_centroid_flooded < 5000:
        from rasterstats import zonal_stats
        print(f"    Calculando zonal stats para {n_centroid_flooded:,} predios afectados...")
        
        flooded_idx = predios_subset['_centroid_depth'] > 0
        flooded_predios = predios_subset.loc[flooded_idx]
        
        stats = zonal_stats(
            flooded_predios.geometry.tolist(),
            str(depth_raster_path),
            stats=['max', 'mean'],
            all_touched=True,
            nodata=nodata
        )
        
        max_depths = [s['max'] if s['max'] else 0.0 for s in stats]
        mean_depths = [s['mean'] if s['mean'] else 0.0 for s in stats]
        
        # Update values for flooded predios
        flooded_orig_idx = predios_gdf.index.isin(flooded_predios.index)
        predios_gdf.loc[flooded_orig_idx, 'max_depth_m'] = max_depths
        predios_gdf.loc[flooded_orig_idx, 'mean_depth_m'] = mean_depths
    
    n_flooded = (predios_gdf['max_depth_m'] > 0).sum()
    max_depth = predios_gdf['max_depth_m'].max()
    print(f"    Predios inundados: {n_flooded:,}, Profundidad máxima: {max_depth:.2f}m")
    
    return predios_gdf


def calculate_flood_damage_climada(
    predios_path: Path = None,
    depth_raster_path: Path = None,
    sector_mapping: dict = None,
    output_gpkg: Path = None,
    output_txt: Path = None,
    # Column names (optional - use defaults if not specified)
    destino_column: str = None,
    valconstru_column: str = None,
    valterreno_column: str = None,
    valtotal_column: str = None
) -> dict:
    """
    Calculate flood damage using CLIMADA JRC curves.
    
    Uses land use or destino economico to select appropriate damage curve per sector.
    
    CLIMADA JRC Available Sectors (South America):
        - residential: Viviendas (CLIMADA JRC)
        - commercial:  Comercios (CLIMADA JRC)
        - industrial:  Industrias (CLIMADA JRC)
        - transport:   Calles, carreteras (CLIMADA JRC)
        - infrastructure: Equipamiento público (CUSTOM curve)
        - agriculture: Cultivos (CUSTOM curve)
    
    Args:
        predios_path: Path to predios shapefile/gpkg with property data
        depth_raster_path: Path to max water depth raster from ITZI
        sector_mapping: Optional dict mapping destino economico -> sector name.
                       If None, uses default DESTINO_ECONOMICO_TO_SECTOR.
                       Example: {"EDUCACIÓN": "infrastructure", "VIVIENDA": "residential"}
        output_gpkg: Path to save results as geopackage
        output_txt: Path to save damage report text file
        destino_column: Column name for destino economico (default: 'desteconom')
        valconstru_column: Column name for construction value (default: 'valconstru')
        valterreno_column: Column name for land value (default: 'valterreno')
        valtotal_column: Column name for total value (default: 'valtotal')
    
    Returns:
        dict with total_damage_usd, flooded_properties, damage_by_sector, etc.
    """
    # CLIMADA imports
    try:
        from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood
    except ImportError as e:
        print("ERROR: CLIMADA not installed!")
        print("Install with: pip install climada climada-petals")
        return {"error": str(e)}
    
    # Defaults for file paths
    if predios_path is None:
        predios_path = predios_file
        
    if depth_raster_path is None:
        # No raster provided, use default from config
        depth_raster_path = depth_raster_file
        # Use default outputs from config if not specified
        if output_gpkg is None:
            output_gpkg = damage_results_gpkg
        if output_txt is None:
            output_txt = damage_results_txt
    else:
        # Custom raster provided
        # If outputs not specified, save in the same directory as the raster
        if output_gpkg is None:
            output_gpkg = Path(depth_raster_path).parent / "flood_damage_results.gpkg"
            print(f"  Note: Output GPKG not specified, saving to raster dir: {output_gpkg}")
        if output_txt is None:
            output_txt = Path(depth_raster_path).parent / "flood_damage_report.txt"
    
    # Defaults for column names (use argument or fall back to module constant)
    col_destino = destino_column if destino_column is not None else DESTINO_COLUMN
    col_valconstru = valconstru_column if valconstru_column is not None else VALOR_CONSTRUCCION_COLUMN
    col_valterreno = valterreno_column if valterreno_column is not None else VALOR_TERRENO_COLUMN
    col_valtotal = valtotal_column if valtotal_column is not None else VALOR_TOTAL_COLUMN
    
    # Use provided sector mapping or default
    active_sector_mapping = sector_mapping if sector_mapping is not None else DESTINO_ECONOMICO_TO_SECTOR
    
    print("="*60)
    print("FLOOD DAMAGE ASSESSMENT (CLIMADA JRC)")
    print("="*60)
    
    # =========================================================================
    # 1. LOAD PREDIOS AND DEPTH RASTER
    # =========================================================================
    print(f"\n[1] Loading predios: {predios_path}")
    
    if not Path(predios_path).exists():
        return {"error": f"Predios file not found: {predios_path}"}
    if not Path(depth_raster_path).exists():
        return {"error": f"Depth raster not found: {depth_raster_path}"}
    
    gdf = gpd.read_file(predios_path)
    n_original = len(gdf)
    print(f"  Loaded {n_original:,} properties")
    
    # =========================================================================
    # 1.5 REMOVE DUPLICATE GEOMETRIES (keep record with data)
    # =========================================================================
    print(f"\n[1.5] Removing duplicate geometries...")
    
    # Create geometry hash for duplicate detection
    gdf['_geom_wkt'] = gdf.geometry.to_wkt()
    
    # Sort to prioritize records WITH cadastral values (valtotal > 0 first)
    if col_valtotal in gdf.columns:
        gdf['_has_value'] = pd.to_numeric(gdf[col_valtotal], errors='coerce').fillna(0) > 0
        gdf = gdf.sort_values('_has_value', ascending=False)
    
    # Remove duplicates, keeping first (which has value due to sort)
    gdf = gdf.drop_duplicates(subset='_geom_wkt', keep='first')
    gdf = gdf.drop(columns=['_geom_wkt', '_has_value'], errors='ignore')
    
    n_after = len(gdf)
    n_removed = n_original - n_after
    print(f"  Removed {n_removed:,} duplicate geometries")
    print(f"  Unique properties: {n_after:,}")
    
    # =========================================================================
    # 2. MAP LAND USE TO SECTOR (for CLIMADA damage curves)
    # =========================================================================
    print(f"\n[2] Mapping land use to CLIMADA sectors...")
    print(f"  Using {'custom' if sector_mapping else 'default'} sector mapping")
    
    # Check which column is available and use appropriate mapping
    if col_destino in gdf.columns:
        # Using predioGeo.shp with desteconom column
        print(f"  Mapping from '{col_destino}' column")
        gdf["_sector"] = gdf[col_destino].map(
            lambda x: active_sector_mapping.get(str(x).upper().strip(), DEFAULT_SECTOR) if pd.notna(x) else DEFAULT_SECTOR
        )
    elif USE_COLUMN in gdf.columns:
        # Using 08_predios_curvas.gpkg with uso_vige column
        print(f"  Mapping from '{USE_COLUMN}' column")
        gdf["_sector"] = gdf[USE_COLUMN].map(
            lambda x: LAND_USE_TO_SECTOR.get(x, DEFAULT_SECTOR) if x else DEFAULT_SECTOR
        )
    else:
        print(f"  WARNING: No land use column found. Using default sector: {DEFAULT_SECTOR}")
        gdf["_sector"] = DEFAULT_SECTOR
    
    print("  Sector distribution:")
    for sector, count in gdf["_sector"].value_counts().items():
        print(f"    {sector}: {count:,}")
    
    # Show detailed mapping of destino economico to sector (damage curve)
    if col_destino in gdf.columns:
        print("\n  " + "="*55)
        print("  MAPEO DESTINO ECONÓMICO → CURVA DE DAÑO")
        print("  " + "="*55)
        
        # Group by sector and show all destinos
        destino_counts = gdf.groupby([col_destino, '_sector']).size().reset_index(name='count')
        
        for sector in ['residential', 'commercial', 'industrial', 'infrastructure', 'agriculture']:
            sector_data = destino_counts[destino_counts['_sector'] == sector]
            if len(sector_data) > 0:
                total = sector_data['count'].sum()
                curve_type = "CLIMADA JRC" if sector in ['residential', 'commercial', 'industrial', 'transport'] else "CUSTOM"
                print(f"\n  {sector.upper()} ({curve_type}) - {total:,} propiedades:")
                for _, row in sector_data.nlargest(10, 'count').iterrows():
                    dest = str(row[col_destino])[:30]
                    print(f"    • {dest}: {row['count']:,}")
        
        # Check for unmapped destinos (those that fell to default)
        gdf['_destino_norm_check'] = gdf[col_destino].astype(str).str.upper().str.strip()
        unmapped_mask = ~gdf['_destino_norm_check'].isin([k.upper() for k in active_sector_mapping.keys()])
        n_unmapped = unmapped_mask.sum()
        if n_unmapped > 0:
            print(f"\n  ⚠ DESTINOS NO MAPEADOS ({n_unmapped:,} props) - usando '{DEFAULT_SECTOR}':")
            unmapped_destinos = gdf.loc[unmapped_mask, '_destino_norm_check'].value_counts().head(10)
            for dest, count in unmapped_destinos.items():
                print(f"    • {dest[:40]}: {count:,}")
        
        print("  " + "="*55)
    
    # =========================================================================
    # 3. EXTRACT MAX DEPTH PER PREDIO
    # =========================================================================
    print(f"\n[3] Extracting max depth per predio...")
    
    gdf = get_max_depth_per_predio(gdf, depth_raster_path)
    
    flooded = gdf[gdf['max_depth_m'] > 0]
    print(f"  Flooded properties: {len(flooded):,} ({100*len(flooded)/len(gdf):.1f}%)")
    print(f"  Max depth recorded: {gdf['max_depth_m'].max():.2f} m")
    
    # =========================================================================
    # 4. LOAD JRC DAMAGE CURVES AND CALCULATE DAMAGE RATIO
    # =========================================================================
    print(f"\n[4] Loading JRC damage curves (South America)...")
    
    # Load impact functions for each sector
    impf_funcs = {}
    custom_curves_used = {}  # Track custom curves
    sectors = ["residential", "commercial", "industrial", "infrastructure", "agriculture"]
    
    for sector in sectors:
        try:
            impf = ImpfRiverFlood.from_jrc_region_sector("South America", sector)
            impf_funcs[sector] = impf
            print(f"    {sector}: CLIMADA loaded (max MDR at 6m: {impf.calc_mdr([6.0])[0]:.1%})")
        except Exception as e:
            # Use custom curve if available
            if sector in CUSTOM_DAMAGE_CURVES:
                custom_curves_used[sector] = CUSTOM_DAMAGE_CURVES[sector]
                print(f"    {sector}: Using CUSTOM curve ({CUSTOM_DAMAGE_CURVES[sector]['source']})")
            else:
                print(f"    {sector}: Not available - {e}")
    
    # =========================================================================
    # 5. CALCULATE DAMAGE RATIO PER PREDIO
    # =========================================================================
    print(f"\n[5] Calculating damage ratio per predio...")
    
    def interpolate_mdr(depth, depths, mdr_values):
        """Interpolate MDR from custom curve."""
        if depth <= 0:
            return 0.0
        if depth >= depths[-1]:
            return mdr_values[-1]
        # Linear interpolation
        for i in range(len(depths) - 1):
            if depths[i] <= depth <= depths[i + 1]:
                ratio = (depth - depths[i]) / (depths[i + 1] - depths[i])
                return mdr_values[i] + ratio * (mdr_values[i + 1] - mdr_values[i])
        return 0.0
    
    def get_damage_ratio(row):
        """Get Mean Damage Ratio (MDR) from JRC or custom curve based on depth and sector."""
        depth = row['max_depth_m']
        sector = row['_sector']
        
        if depth <= 0:
            return 0.0
        
        # Try CLIMADA curve first
        if sector in impf_funcs:
            mdr = impf_funcs[sector].calc_mdr([depth])[0]
            return float(mdr)
        
        # Use custom curve if available
        if sector in custom_curves_used:
            curve = custom_curves_used[sector]
            mdr = interpolate_mdr(depth, curve['depths'], curve['mdr'])
            return float(mdr)
        
        return 0.0
    
    gdf['damage_ratio'] = gdf.apply(get_damage_ratio, axis=1)
    gdf['damage_percent'] = gdf['damage_ratio'] * 100
    
    # =========================================================================
    # 6. ESTIMATE PROPERTY VALUE (using real cadastral data when available)
    # =========================================================================
    print(f"\n[6] Estimating property values...")
    
    gdf['_area_m2'] = gdf.geometry.area
    
    # Check which value columns are available
    has_valtotal = col_valtotal in gdf.columns
    has_valconstru = col_valconstru in gdf.columns
    has_valterreno = col_valterreno in gdf.columns
    has_destino = col_destino in gdf.columns
    
    # Initialize estimated_value column
    gdf['estimated_value_usd'] = 0.0
    gdf['_value_source'] = 'none'
    
    if has_valtotal or has_valconstru or has_valterreno:
        # Convert to numeric
        if has_valtotal:
            gdf['_valtotal'] = pd.to_numeric(gdf[col_valtotal], errors='coerce').fillna(0)
        else:
            gdf['_valtotal'] = 0.0
            
        if has_valconstru:
            gdf['_valconstru'] = pd.to_numeric(gdf[col_valconstru], errors='coerce').fillna(0)
        else:
            gdf['_valconstru'] = 0.0
            
        if has_valterreno:
            gdf['_valterreno'] = pd.to_numeric(gdf[col_valterreno], errors='coerce').fillna(0)
        else:
            gdf['_valterreno'] = 0.0
        
        # =====================================================================
        # ONLY USE CONSTRUCTION VALUE (valconstru) - NOT land value
        # According to JRC/CLIMADA methodology, flood damage applies to:
        # - Building structure and contents
        # - NOT the land itself (land doesn't get "damaged" by flooding)
        # =====================================================================
        gdf['_best_value'] = gdf['_valconstru']  # ONLY construction value
        
        # Mask for valid values
        has_valid_value = gdf['_best_value'] > 0
        n_with_value = has_valid_value.sum()
        print(f"  Properties with construction value: {n_with_value:,} of {len(gdf):,}")
        print(f"  NOTE: Using only CONSTRUCTION value (valconstru), not land value")
        
        # Step 1: Assign real values where available
        gdf.loc[has_valid_value, 'estimated_value_usd'] = gdf.loc[has_valid_value, '_best_value']
        gdf.loc[has_valid_value, '_value_source'] = 'real'
        
        # Step 2: Calculate average by destino economico (per m²)
        if has_destino and n_with_value > 0:
            # Normalize destino values (handle encoding issues, strip whitespace, uppercase)
            gdf['_destino_norm'] = gdf[col_destino].astype(str).str.strip().str.upper()
            
            # Calculate value per m²
            gdf['_value_per_m2'] = gdf['_best_value'] / gdf['_area_m2'].replace(0, 1)
            
            # Calculate averages per m² using normalized destino
            avg_per_m2_by_destino = gdf.loc[has_valid_value].groupby('_destino_norm')['_value_per_m2'].mean()
            overall_avg_per_m2 = gdf.loc[has_valid_value, '_value_per_m2'].mean()
            overall_avg = gdf.loc[has_valid_value, '_best_value'].mean()
            
            # For fallback, we'll use avg value (not per m2) to avoid needing area
            avg_by_destino = gdf.loc[has_valid_value].groupby('_destino_norm')['_best_value'].mean()
            
            print(f"  Average value by destino economico ($/m²):")
            for dest, avg in avg_per_m2_by_destino.sort_values(ascending=False).items():
                total_val = avg_by_destino.get(dest, 0)
                print(f"    {dest}: ${avg:,.0f}/m² (avg total: ${total_val:,.0f})")
            print(f"  Overall average: ${overall_avg_per_m2:,.0f}/m² (${overall_avg:,.0f} total)")
            print(f"  Total destino categories with data: {len(avg_by_destino)}")
            
            # =====================================================================
            # SAVE AVERAGES TO JSON (for use when no cadastral data is available)
            # =====================================================================
            import json
            avg_data = {
                "description": "Average construction value by destino economico ($/m²)",
                "source": str(predios_path),
                "overall_avg_per_m2": float(overall_avg_per_m2),
                "by_destino": {
                    dest: {
                        "value_per_m2": float(avg_per_m2_by_destino.get(dest, 0)),
                        "avg_total_value": float(avg_by_destino.get(dest, 0))
                    }
                    for dest in avg_per_m2_by_destino.index
                }
            }
            avg_json_path = output_gpkg.parent / "avg_construction_value_by_destino.json"
            with open(avg_json_path, 'w', encoding='utf-8') as f:
                json.dump(avg_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved averages to: {avg_json_path}")
            
            # Step 3: For predios without value, use area × ($/m² of their destino)
            no_value_mask = ~has_valid_value
            
            if no_value_mask.sum() > 0:
                # Map normalized destino to $/m² average
                gdf['_destino_per_m2'] = gdf['_destino_norm'].map(avg_per_m2_by_destino)
                
                # Count how many can use destino avg vs overall avg
                has_destino_match = gdf['_destino_norm'].isin(avg_per_m2_by_destino.index)
                
                # Apply fallback: area × $/m² of destino, or area × overall $/m²
                mask_destino = no_value_mask & has_destino_match
                mask_overall = no_value_mask & ~has_destino_match
                
                gdf.loc[mask_destino, 'estimated_value_usd'] = (
                    gdf.loc[mask_destino, '_area_m2'] * gdf.loc[mask_destino, '_destino_per_m2']
                )
                gdf.loc[mask_destino, '_value_source'] = 'avg_destino'
                
                gdf.loc[mask_overall, 'estimated_value_usd'] = (
                    gdf.loc[mask_overall, '_area_m2'] * overall_avg_per_m2
                )
                gdf.loc[mask_overall, '_value_source'] = 'avg_overall'
        else:
            overall_avg_per_m2 = DEFAULT_VALUE_PER_M2
            gdf.loc[~has_valid_value, 'estimated_value_usd'] = gdf.loc[~has_valid_value, '_area_m2'] * overall_avg_per_m2
            gdf.loc[~has_valid_value, '_value_source'] = 'avg_overall'
        
        # Summary
        n_real = (gdf['_value_source'] == 'real').sum()
        n_destino = (gdf['_value_source'] == 'avg_destino').sum()
        n_overall = (gdf['_value_source'] == 'avg_overall').sum()
        print(f"  Value sources: {n_real:,} real, {n_destino:,} avg by destino, {n_overall:,} overall avg")
        
    else:
        # No real values, use area-based estimate
        print(f"  No cadastral values found, using ${DEFAULT_VALUE_PER_M2}/m² estimate")
        gdf['estimated_value_usd'] = gdf['_area_m2'] * DEFAULT_VALUE_PER_M2
        gdf['_value_source'] = 'area_based'
    
    # Calculate damage
    gdf['damage_usd'] = gdf['damage_ratio'] * gdf['estimated_value_usd']
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    total_damage = gdf['damage_usd'].sum()
    total_value = gdf['estimated_value_usd'].sum()
    flooded_mask = gdf['damage_ratio'] > 0
    flooded_count = flooded_mask.sum()
    avg_damage_ratio = gdf.loc[flooded_mask, 'damage_ratio'].mean() if flooded_count > 0 else 0
    
    # By sector - ONLY for flooded properties
    flooded_gdf = gdf[flooded_mask]
    damage_by_sector = flooded_gdf.groupby('_sector')['damage_usd'].sum().to_dict()
    flooded_count_by_sector = flooded_gdf.groupby('_sector').size().to_dict()
    
    # Value of flooded properties only
    flooded_value = flooded_gdf['estimated_value_usd'].sum()
    
    print("\n" + "="*60)
    print("DAMAGE SUMMARY")
    print("="*60)
    print(f"  Total properties:       {len(gdf):,}")
    print(f"  Flooded properties:     {flooded_count:,} ({100*flooded_count/len(gdf):.1f}%)")
    print(f"  Avg damage ratio:       {avg_damage_ratio:.1%}")
    print(f"\n  Value of flooded props: ${flooded_value:,.0f}")
    print(f"  TOTAL FLOOD DAMAGE:     ${total_damage:,.0f}")
    print(f"  Damage as % of flooded: {100*total_damage/flooded_value:.2f}%")
    print("\n  Damage by sector (SOLO INUNDADOS):")
    for sector, amount in sorted(damage_by_sector.items(), key=lambda x: -x[1]):
        cnt = flooded_count_by_sector.get(sector, 0)
        print(f"    {sector}: ${amount:,.0f} ({cnt:,} flooded props)")
    print("="*60)
    
    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    print(f"\n[6] Saving results...")
    print(f"    GPKG: {output_gpkg}")
    print(f"    Report: {output_txt}")
    
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    
    # Save GPKG
    gdf.to_file(output_gpkg, driver="GPKG")
    
    # Save report
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FLOOD DAMAGE ASSESSMENT REPORT (CLIMADA JRC)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-"*70 + "\n")
        f.write("Damage curves: JRC Global Flood Depth-Damage Functions\n")
        f.write("Region: South America\n")
        f.write("Sectors: residential, commercial, industrial, infrastructure, agriculture\n\n")
        
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total properties:         {len(gdf):>12,}\n")
        f.write(f"Flooded properties:       {flooded_count:>12,} ({100*flooded_count/len(gdf):.1f}%)\n")
        f.write(f"Average damage ratio:     {avg_damage_ratio:>12.1%}\n")
        f.write(f"Total estimated value:    ${total_value:>11,.0f}\n")
        f.write(f"TOTAL FLOOD DAMAGE:       ${total_damage:>11,.0f}\n")
        f.write(f"Damage as % of value:     {100*total_damage/total_value:>11.2f}%\n\n")
        
        f.write("DAMAGE BY SECTOR\n")
        f.write("-"*70 + "\n")
        for sector, amount in sorted(damage_by_sector.items(), key=lambda x: -x[1]):
            cnt = flooded_count_by_sector.get(sector, 0)
            f.write(f"  {sector:<20}: ${amount:>12,.0f} ({cnt:,} properties)\n")
        f.write("="*70 + "\n")
    
    return {
        "total_damage_usd": float(total_damage),
        "total_properties": len(gdf),
        "flooded_properties": int(flooded_count),
        "damage_by_sector": damage_by_sector,
        "output_gpkg": str(output_gpkg),
        "output_report": str(output_txt),
        "gdf": gdf,  # Include processed GeoDataFrame for plotting
        "depth_raster_path": str(depth_raster_path)
    }


# =============================================================================
# FLOOD DAMAGE PLOTTER CLASS
# =============================================================================
class FloodDamagePlotter:
    """
    Class to generate visualizations of flood damage assessment results.
    
    Usage:
        result = calculate_flood_damage_climada(...)
        plotter = FloodDamagePlotter(result)
        plotter.plot_all()  # Generate all plots
    """
    
    # Color schemes
    SECTOR_COLORS = {
        'residential': '#4285F4',    # Blue
        'commercial': '#EA4335',     # Red
        'industrial': '#FBBC04',     # Yellow
        'infrastructure': '#34A853', # Green
        'agriculture': '#8B4513'     # Brown
    }
    
    DAMAGE_CMAP = 'YlOrRd'  # Yellow-Orange-Red for damage
    DEPTH_CMAP = 'Blues'    # Blues for water depth
    VALUE_CMAP = 'YlGn'     # Yellow-Green for property value
    
    def __init__(self, result: dict, output_dir: Path = None, network_path: Path = None):
        """
        Initialize plotter with result from calculate_flood_damage_climada.
        
        Args:
            result: Dict returned by calculate_flood_damage_climada()
            output_dir: Directory to save plots. If None, uses same as output_gpkg
            network_path: Path to sewer network GPKG (lines) to plot as background
        """
        import matplotlib.pyplot as plt
        self.plt = plt
        
        self.result = result
        self.gdf = result.get('gdf')
        self.depth_raster_path = result.get('depth_raster_path')
        
        # Load network if provided
        self.network_gdf = None
        if network_path and Path(network_path).exists():
            try:
                self.network_gdf = gpd.read_file(network_path)
                # Check CRS
                if self.gdf is not None and self.network_gdf.crs != self.gdf.crs:
                    self.network_gdf = self.network_gdf.to_crs(self.gdf.crs)
                print(f"  Loaded network from: {network_path}")
            except Exception as e:
                print(f"  Warning: Could not load network: {e}")
        
        if output_dir is None:
            # Use dynamic path from results, create 00_images subfolder relative to output GPKG
            output_gpkg_path = Path(result['output_gpkg'])
            output_dir = output_gpkg_path.parent / '00_images'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib
        self.plt.style.use('seaborn-v0_8-whitegrid')
        self.plt.rcParams['figure.dpi'] = 150
        self.plt.rcParams['savefig.dpi'] = 150
        self.plt.rcParams['font.size'] = 10
        
        print(f"FloodDamagePlotter initialized. Output dir: {self.output_dir}")
        
    def _plot_network(self, ax):
        """Helper to plot network background."""
        if self.network_gdf is not None:
            self.network_gdf.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5, zorder=1)

    # ... [rest of methods] ...

    
    def plot_all(self):
        """Generate all available plots."""
        print("\n" + "="*60)
        print("GENERATING FLOOD DAMAGE VISUALIZATIONS")
        print("="*60)
        
        self.plot_damage_by_sector()
        self.plot_damage_by_destino()
        self.plot_vulnerability_curves()
        self.plot_depth_vs_damage_scatter()
        self.plot_depth_histogram_stacked() # Improved version: Stacked by sector + Log scale
        # self.plot_damage_pie() # Removed: Redundant with bar chart percentages
        self.plot_value_source_pie()
        self.plot_damage_map()
        self.plot_exposure_map()
        self.plot_flood_depth_map()
        self.plot_value_distribution_boxplot()
        
        # New Risk Metrics
        self.plot_damage_concentration()
        self.plot_damage_profile_by_depth()
        
        # Advanced Statistics
        self.plot_relative_damage_violin()
        self.plot_correlation_heatmap()
        
        print(f"\n All plots saved to: {self.output_dir}")
        return self.output_dir
    
    def _format_currency_axis(self, ax, axis='y'):
        """Helper to format axis with readable currency (M/k)."""
        import matplotlib.ticker as ticker
        def currency_formatter(x, pos):
            if x >= 1e6:
                return f'${x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'${x*1e-3:.0f}k'
            else:
                return f'${x:,.0f}'
        
        formatter = ticker.FuncFormatter(currency_formatter)
        if 'x' in axis:
            ax.xaxis.set_major_formatter(formatter)
        if 'y' in axis:
            ax.yaxis.set_major_formatter(formatter)

    def plot_damage_by_sector(self):
        """Bar chart of damage by sector."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        damage_by_sector = self.result['damage_by_sector']
        sectors = list(damage_by_sector.keys())
        values = [damage_by_sector[s] for s in sectors]
        colors = [self.SECTOR_COLORS.get(s, '#888888') for s in sectors]
        
        bars = ax.barh(sectors, values, color=colors)
        ax.set_xlabel('Daño (USD)')
        ax.set_title('Daño por Sector (Curva de Daño)')
        
        # Add value labels
        total_damage = sum(values)
        for bar, val in zip(bars, values):
            pct = (val / total_damage * 100) if total_damage > 0 else 0
            ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'${val:,.0f} ({pct:.1f}%)', va='center', fontsize=9)
        
            ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'${val:,.0f} ({pct:.1f}%)', va='center', fontsize=9)
        
        ax.set_xlim(0, max(values) * 1.25) # More space for labels
        self._format_currency_axis(ax, axis='x')
        self.plt.tight_layout()
        
        path = self.output_dir / 'damage_by_sector.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_damage_by_destino(self, top_n=15):
        """Bar chart of damage by destino economico (top N)."""
        fig, ax = self.plt.subplots(figsize=(10, 8))
        
        flooded = self.gdf[self.gdf['damage_usd'] > 0]
        if '_destino_norm' not in flooded.columns:
            flooded['_destino_norm'] = flooded.get('desteconom', 'UNKNOWN').astype(str).str.upper()
        
        damage_by_destino = flooded.groupby('_destino_norm')['damage_usd'].sum().nlargest(top_n)
        
        bars = ax.barh(damage_by_destino.index[::-1], damage_by_destino.values[::-1], 
                      color='#4285F4')
        ax.set_xlabel('Daño (USD)')
        ax.set_title(f'Daño por Destino Económico (Top {top_n})')
        
        for bar, val in zip(bars, damage_by_destino.values[::-1]):
            ax.text(val + max(damage_by_destino)*0.01, bar.get_y() + bar.get_height()/2,
                   f'${val:,.0f}', va='center', fontsize=8)
        
        ax.set_xlim(0, max(damage_by_destino) * 1.25)
        self._format_currency_axis(ax, axis='x')
        self.plt.tight_layout()
        
        path = self.output_dir / 'damage_by_destino.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_vulnerability_curves(self):
        """Plot JRC depth-damage functions (Prioritizing CLIMADA with Hardcoded Fallback)."""
        fig, ax = self.plt.subplots(figsize=(10, 7))
        
        # Hardcoded JRC curves for South America (approximate) from rut_20
        # Used as BACKUP if CLIMADA fails or for custom sectors
        depth_points = [0, 1, 2, 3, 4, 5, 6]
        depths_interp = np.linspace(0, 6, 100)
        
        backup_curves = {
            'residential': [0.0, 0.25, 0.45, 0.60, 0.75, 0.85, 0.90],
            'commercial': [0.0, 0.30, 0.50, 0.65, 0.78, 0.88, 0.93],
            'industrial': [0.0, 0.20, 0.40, 0.55, 0.70, 0.82, 0.88],
            'infrastructure': [0.0, 0.05, 0.15, 0.25, 0.35, 0.50, 0.65],
            'agriculture': [0.0, 0.35, 0.55, 0.70, 0.80, 0.90, 0.95]
        }
        
        # Try importing CLIMADA
        climada_available = False
        try:
            from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood
            climada_available = True
        except ImportError:
            print("  Warning: CLIMADA not installed. Using backup curves for all sectors.")
        
        # Plot each sector
        sectors = ['residential', 'commercial', 'industrial', 'infrastructure', 'agriculture']
        
        for sector in sectors:
            color = self.SECTOR_COLORS.get(sector, '#333333')
            plotted = False
            
            # 1. Try CLIMADA JRC (only for supported sectors: res, com, ind)
            if climada_available and sector in ['residential', 'commercial', 'industrial']:
                try:
                    # Use full region name 'South America' instead of 'SA'
                    impf = ImpfRiverFlood.from_jrc_region_sector('South America', sector)
                    mdr = np.interp(depths_interp, impf.intensity, impf.mdd * impf.paa * 100)
                    # Clean label matching rut_20 style
                    ax.plot(depths_interp, mdr, label=sector, 
                           color=color, linewidth=2.5)
                    plotted = True
                except Exception as e:
                    # Silently fail or log sparingly to fall back to backup
                    print(f"  Warning: Could not load CLIMADA JRC curve for {sector}: {e}")
                    pass
            
            # 2. Use BACKUP/CUSTOM if not plotted yet
            if not plotted:
                mdr_values = backup_curves.get(sector)
                if mdr_values:
                    # Interpolate for smooth curve
                    mdr_smooth = np.interp(depths_interp, depth_points, mdr_values)
                    
                    # Clean label matching rut_20 style (no BACKUP suffix)
                    # Use dashed line for custom sectors (infra/agri) to distinguish slightly, 
                    # but keep main sectors solid if they are acting as JRC fallback
                    linestyle = '--' if sector in ['infrastructure', 'agriculture'] else '-'
                    
                    ax.plot(depths_interp, mdr_smooth * 100, label=sector, 
                           color=color, linewidth=2.5, linestyle=linestyle)
                    
                    # Add markers for backup points to show they are discrete
                    if sector in ['infrastructure', 'agriculture']:
                         ax.scatter(depth_points, np.array(mdr_values) * 100, color=color, s=30)
        
        ax.set_xlabel('Profundidad de Inundación (m)', fontsize=12)
        ax.set_ylabel('Ratio de Daño (%)', fontsize=12)
        ax.set_title('Curvas de Vulnerabilidad por Sector', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Sector', loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 105)
        
        self.plt.tight_layout()
        path = self.output_dir / 'vulnerability_curves.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_depth_vs_damage_scatter(self):
        """Scatter plot: depth vs damage, colored by sector."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        flooded = self.gdf[self.gdf['damage_usd'] > 0]
        
        for sector, color in self.SECTOR_COLORS.items():
            mask = flooded['_sector'] == sector
            if mask.sum() > 0:
                ax.scatter(flooded.loc[mask, 'max_depth_m'], 
                          flooded.loc[mask, 'damage_usd'],
                          c=color, label=sector, alpha=0.6, s=20)
        
        ax.set_xlabel('Profundidad Máxima (m)')
        ax.set_ylabel('Daño (USD)')
        ax.set_title('Relación Profundidad vs Daño por Sector')
        ax.legend()
        ax.set_yscale('log')
        
        self._format_currency_axis(ax, axis='y')
        self.plt.tight_layout()
        path = self.output_dir / 'depth_vs_damage_scatter.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_depth_histogram_stacked(self):
        """Stacked histogram of depth colored by sector (log scale)."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Prepare data: list of arrays (one per sector)
        # Filter for relevant depths (> 5cm) to reduce noise
        min_depth = 0.05
        
        sectors = sorted(self.gdf['_sector'].unique())
        data_list = []
        labels = []
        colors = []
        
        for sector in sectors:
            # Get depths for this sector
            subset = self.gdf[(self.gdf['_sector'] == sector) & (self.gdf['max_depth_m'] >= min_depth)]
            vals = subset['max_depth_m'].values
            if len(vals) > 0:
                data_list.append(vals)
                labels.append(sector)
                colors.append(self.SECTOR_COLORS.get(sector, '#888888'))
        
        if not data_list:
            return None

        # Create stacked histogram with log scale
        ax.hist(data_list, bins=25, stacked=True, label=labels, color=colors, 
               edgecolor='white', linewidth=0.5, alpha=0.9)
        
        ax.set_yscale('log')
        ax.set_xlabel('Profundidad Máxima (m)')
        ax.set_ylabel('N° Propiedades (Escala Log)')
        ax.set_title(f'Distribución de Profundidad por Sector (>{int(min_depth*100)}cm)')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        self.plt.tight_layout()
        path = self.output_dir / 'depth_histogram_stacked.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_depth_histogram(self):
        """Histogram of flood depths."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        flooded = self.gdf[self.gdf['max_depth_m'] > 0]
        
        ax.hist(flooded['max_depth_m'], bins=20, color='#4285F4', edgecolor='white', alpha=0.8)
        ax.axvline(flooded['max_depth_m'].mean(), color='red', linestyle='--', 
                  label=f"Promedio: {flooded['max_depth_m'].mean():.2f}m")
        ax.axvline(flooded['max_depth_m'].median(), color='orange', linestyle='--',
                  label=f"Mediana: {flooded['max_depth_m'].median():.2f}m")
        
        ax.set_xlabel('Profundidad Máxima (m)')
        ax.set_ylabel('Número de Propiedades')
        ax.set_title('Distribución de Profundidad de Inundación')
        ax.legend()
        
        self.plt.tight_layout()
        path = self.output_dir / 'depth_histogram.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_damage_pie(self):
        """Pie chart of damage by sector."""
        fig, ax = self.plt.subplots(figsize=(8, 8))
        
        damage_by_sector = self.result['damage_by_sector']
        sectors = list(damage_by_sector.keys())
        values = [damage_by_sector[s] for s in sectors]
        colors = [self.SECTOR_COLORS.get(s, '#888888') for s in sectors]
        
        # Filter out zeros
        non_zero = [(s, v, c) for s, v, c in zip(sectors, values, colors) if v > 0]
        if non_zero:
            sectors, values, colors = zip(*non_zero)
            
            def autopct(pct):
                val = pct * sum(values) / 100
                return f'{pct:.1f}%\n(${val:,.0f})'
            
            ax.pie(values, labels=None, colors=colors, autopct=autopct,
                  startangle=90, explode=[0.02]*len(values), pctdistance=0.85)
            
            # Add legend to avoid clutter
            ax.legend(sectors, title="Sectores", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            ax.set_title('Proporción de Daño por Sector')
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_pie.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_value_source_pie(self):
        """Pie chart of value estimation source."""
        fig, ax = self.plt.subplots(figsize=(8, 8))
        
        if '_value_source' in self.gdf.columns:
            source_counts = self.gdf['_value_source'].value_counts()
            colors = {'real': '#34A853', 'avg_destino': '#FBBC04', 'avg_overall': '#EA4335', 'none': '#888888'}
            
            ax.pie(source_counts.values, labels=source_counts.index, 
                  colors=[colors.get(s, '#888888') for s in source_counts.index],
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Fuente de Valor de Propiedad')
        
        self.plt.tight_layout()
        path = self.output_dir / 'value_source_pie.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_damage_map(self):
        """Map of properties colored by damage amount."""
        fig, ax = self.plt.subplots(figsize=(12, 10))
        
        # Plot network background
        self._plot_network(ax)
        
        # Plot non-flooded in gray
        non_flooded = self.gdf[self.gdf['damage_usd'] == 0]
        if len(non_flooded) > 0:
            non_flooded.plot(ax=ax, color='#E0E0E0', edgecolor='none', alpha=0.5)
        
        # Plot flooded colored by damage
        flooded = self.gdf[self.gdf['damage_usd'] > 0]
        if len(flooded) > 0:

            # Colorbar uses 'label'
            # Manual colorbar to fix formatting (1e6 -> M)
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            # Plot without automatic legend
            flooded.plot(ax=ax, column='damage_usd', cmap=self.DAMAGE_CMAP,
                        legend=False, edgecolor='black', linewidth=0.3)
            
            # Create manual colorbar
            vmin = flooded['damage_usd'].min()
            vmax = flooded['damage_usd'].max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=self.DAMAGE_CMAP)
            
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
            cbar.set_label('Daño (USD)')
            
            # Apply currency formatter
            self._format_currency_axis(cbar.ax, axis='y')
        
        ax.set_title(f"Mapa de Daño por Inundación\nTotal: ${self.result['total_damage_usd']:,.0f} USD")
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_aspect('equal')
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_map.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    
    def plot_damage_ratio_map(self):
        """Map of properties colored by damage ratio (%)."""
        fig, ax = self.plt.subplots(figsize=(12, 10))
        
        # Plot network background
        self._plot_network(ax)
        
        flooded = self.gdf[self.gdf['damage_percent'] > 0]
        
        if len(flooded) > 0:
            # Scheme='quantiles' creates a discrete Legend, which uses 'title'
            flooded.plot(ax=ax, column='damage_percent', cmap='Reds',
                        legend=True, legend_kwds={'title': 'Daño Relativo (%)', 'loc': 'upper right'},
                        edgecolor='black', linewidth=0.3, scheme='quantiles')
        
        ax.set_title('Mapa de Daño Relativo (% del Valor)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_aspect('equal')
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_ratio_map.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

        print(f"   {path.name}")
        return path

    def plot_flood_depth_map(self):
        """Map of flood depth raster with properties overlay."""
        import rasterio
        from rasterio.plot import show
        import numpy as np
        
        if not self.depth_raster_path or not Path(self.depth_raster_path).exists():
            print("  Warning: Depth raster not found for depth map.")
            return

        fig, ax = self.plt.subplots(figsize=(12, 10))
        
        # 1. Plot Raster
        try:
            with rasterio.open(self.depth_raster_path) as src:
                # Mask nodata (usually <= 0 for depth in this context)
                data = src.read(1)
                data = np.ma.masked_where(data <= 0, data)
                
                # Calculate sensible vmax (e.g. 98th percentile) to avoid outliers washing out the map
                # This fixes "white on white" issue for low depths
                valid_data = data.compressed()
                if len(valid_data) > 0:
                    vmax = np.nanpercentile(valid_data, 98) 
                else:
                    vmax = None
                
                # Plot raster with adjusted scale and higher opacity
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                # Use YlGnBu for better visibility against white background (starts with color, not white)
                im = ax.imshow(data, cmap='YlGnBu', extent=extent, alpha=0.9, zorder=2, vmax=vmax)
                
                # Add colorbar
                cbar = self.plt.colorbar(im, ax=ax, shrink=0.6, label='Profundidad (m)')
        except Exception as e:
            print(f"  Error plotting raster: {e}")
            
        # 2. Plot predios (background/context)
        # All properties in light gray outlines
        self.gdf.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.2, alpha=0.5, zorder=1)
        
        # 3. Plot network if avail
        self._plot_network(ax)

        ax.set_title('Mapa de Profundidad de Inundación')
        ax.set_xlabel('Este (m)')
        ax.set_ylabel('Norte (m)')
        
        self.plt.tight_layout()
        path = self.output_dir / 'flood_depth_map.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_value_distribution_boxplot(self):
        """Boxplot of construction value by detailed destino economico."""
        fig, ax = self.plt.subplots(figsize=(12, 8))
        
        # Filter data
        data = self.gdf[self.gdf['estimated_value_usd'] > 0].copy()
        
        if len(data) == 0:
            return

        # Get top 10 destinos specifically for readability
        if '_destino_norm' not in data.columns:
             data['_destino_norm'] = data.get('desteconom', 'UNKNOWN').astype(str).str.upper()
             
        top_destinos = data['_destino_norm'].value_counts().head(10).index
        data_filtered = data[data['_destino_norm'].isin(top_destinos)]
        
        if len(data_filtered) == 0:
            return

        # Prepare for boxplot
        # Use log scale and hide outliers for better readability
        data_filtered.boxplot(column='estimated_value_usd', by='_destino_norm', ax=ax, 
                             showfliers=False,  # Hide outliers to make box visible
                             flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, linestyle='none'))
        
        ax.set_title('Distribución de Valor de Construcción (Top 10 Destinos)')
        ax.set_xlabel('')
        ax.set_ylabel('Valor Estimado (USD)')
        # ax.set_yscale('log') # Optional: log scale if needed, but showfliers=False usually fixes readability
        self._format_currency_axis(ax, axis='y')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Fix title generated by pandas
        self.plt.suptitle('') 
        
        self.plt.tight_layout()
        path = self.output_dir / 'value_distribution_boxplot.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_affected_counts_by_sector(self):
        """Grouped bar chart: Total vs Flooded properties by sector."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Counts
        total_counts = self.gdf['_sector'].value_counts()
        flooded_counts = self.gdf[self.gdf['damage_usd'] > 0]['_sector'].value_counts()
        
        sectors = list(total_counts.index)
        
        # Align data
        totals = [total_counts.get(s, 0) for s in sectors]
        flooded = [flooded_counts.get(s, 0) for s in sectors]
        
        x = np.arange(len(sectors))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, totals, width, label='Total Propiedades', color='#CCCCCC')
        rects2 = ax.bar(x + width/2, flooded, width, label='Inundadas', color='#EA4335')
        
        ax.set_ylabel('Número de Propiedades')
        ax.set_title('Comparación: Total vs Afectadas por Sector')
        ax.set_xticks(x)
        ax.set_xticklabels(sectors)
        ax.legend()
        
        # Add labels with percentages of flooding for each sector
        # For Totals
        ax.bar_label(rects1, padding=3, fmt='{:,.0f}', fontsize=8, color='gray')
        
        # For Flooded: Show count AND % of sector flooded
        labels = []
        for i, count in enumerate(flooded):
            total = totals[i]
            pct = (count / total * 100) if total > 0 else 0
            labels.append(f"{count:,.0f}\n({pct:.1f}%)")
            
        ax.bar_label(rects2, labels=labels, padding=3, fontsize=9, fontweight='bold')
        
        self.plt.tight_layout()
        path = self.output_dir / 'affected_counts_by_sector.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_damage_log_histogram(self):
        """Histogram of damage (log scale)."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        damages = self.gdf.loc[self.gdf['damage_usd'] > 0, 'damage_usd']
        
        if len(damages) > 0:
            log_bins = np.logspace(np.log10(max(1, damages.min())), np.log10(damages.max()), 30)
            ax.hist(damages, bins=log_bins, color='#EA4335', edgecolor='white', alpha=0.7)
            
            ax.set_xscale('log')
            ax.set_xlabel('Daño (USD) - Escala Logarítmica')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Daños')
            
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:,.0f}'.format(y)))
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_distribution.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
        
    def plot_relative_damage_violin(self):
        """Violin plot of relative damage (%) by sector (Risk Distribution)."""
        fig, ax = self.plt.subplots(figsize=(12, 7))
        
        # Filter relevant data (damage > 0)
        flooded = self.gdf[self.gdf['damage_usd'] > 0]
        if len(flooded) == 0:
            return

        sectors = sorted(flooded['_sector'].unique())
        data_list = []
        labels = []
        colors = []
        
        for sector in sectors:
            vals = flooded.loc[flooded['_sector'] == sector, 'damage_percent'].values
            if len(vals) > 5: # Need minimal data for kernel density
                data_list.append(vals)
                labels.append(sector)
                colors.append(self.SECTOR_COLORS.get(sector, 'gray'))
        
        if not data_list:
            return

        # Plot violin
        parts = ax.violinplot(data_list, showmeans=False, showmedians=True, showextrema=False)
        
        # Customize colors
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            
        if 'cmedians' in parts:
             parts['cmedians'].set_edgecolor('black')
             parts['cmedians'].set_linewidth(1.5)

        # Set x-ticks
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        
        ax.set_title('Distribución de Daño Relativo por Sector (Vulnerabilidad Realizada)')
        ax.set_ylabel('Daño Relativo (%)')
        ax.set_xlabel('Sector')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        self.plt.tight_layout()
        path = self.output_dir / 'relative_damage_violin.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_correlation_heatmap(self):
        """Heatmap of correlations between key variables."""
        fig, ax = self.plt.subplots(figsize=(8, 7))
        
        # Select numeric columns
        cols = ['max_depth_m', 'estimated_value_usd', 'damage_usd', 'damage_percent']
        labels = ['Profundidad', 'Valor Propiedad', 'Daño ($)', 'Daño (%)']
        
        data = self.gdf[self.gdf['damage_usd'] > 0][cols]
        if len(data) < 10:
            return
            
        corr = data.corr()
        
        # Plot heatmap manually with imshow
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlación de Pearson')
        
        # Ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        self.plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
        
        # Annotate
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white")
                               
        ax.set_title("Matriz de Correlación (Factores de Riesgo)", y=-0.1)
        
        self.plt.tight_layout()
        path = self.output_dir / 'risk_correlation_heatmap.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
            
    def plot_dashboard(self):
        """Create a summary dashboard with key metrics and plots."""
        import matplotlib.gridspec as gridspec
        
        fig = self.plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3)
        
        # 1. Title and KPIs (Top Left)
        ax_kpi = fig.add_subplot(gs[0, 0])
        ax_kpi.axis('off')
        
        total_damage = self.result['total_damage_usd']
        flooded_props = self.result['flooded_properties']
        max_depth = self.gdf['max_depth_m'].max()
        
        kpi_text = (
            f"FLOOD DAMAGE DASHBOARD\n\n"
            f"Total Damage:\n${total_damage:,.0f}\n\n"
            f"Affected Properties:\n{flooded_props:,}\n\n"
            f"Max Water Depth:\n{max_depth:.2f} m\n\n"
            f"Avg Damage Ratio:\n{self.gdf.loc[self.gdf['damage_ratio']>0, 'damage_ratio'].mean():.1%}"
        )
        ax_kpi.text(0.1, 0.5, kpi_text, fontsize=14, va='center', family='sans-serif')
        
        # 2. Damage by Sector Bar Chart (Top Middle)
        ax_bar = fig.add_subplot(gs[0, 1])
        damage_by_sector = self.result['damage_by_sector']
        sectors = sorted(damage_by_sector.keys(), key=lambda x: damage_by_sector[x])
        vals = [damage_by_sector[s] for s in sectors]
        cols = [self.SECTOR_COLORS.get(s, '#888888') for s in sectors]
        ax_bar.barh(sectors, vals, color=cols)
        ax_bar.set_title('Damage by Sector')
        ax_bar.tick_params(axis='x', rotation=45)
        
        # 3. Depth Histogram (Top Right)
        ax_hist = fig.add_subplot(gs[0, 2])
        flooded = self.gdf[self.gdf['max_depth_m'] > 0]
        if len(flooded) > 0:
            ax_hist.hist(flooded['max_depth_m'], bins=15, color='#4285F4', alpha=0.7)
        ax_hist.set_title('Flood Depth Distribution')
        ax_hist.set_xlabel('Depth (m)')
        
        # 4. Damage Map (Bottom Left & Middle)
        ax_map = fig.add_subplot(gs[1, :2])
        flooded = self.gdf[self.gdf['damage_usd'] > 0]
        if len(flooded) > 0:
            # Use a simple plot for dashboard speed
            flooded.plot(ax=ax_map, column='damage_usd', cmap='YlOrRd', legend=True)
        ax_map.set_title('Damage Map')
        ax_map.set_aspect('equal')
        ax_map.axis('off')
        
        # 5. Vulnerability Curves (Bottom Right)
        ax_curve = fig.add_subplot(gs[1, 2])
        depths = np.linspace(0, 5, 50)
        # Simplified curves for dashboard
        try:
            from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood
            for s in ['residential', 'commercial']:
                impf = ImpfRiverFlood.from_jrc_region_sector('SA', s)
                mdr = np.interp(depths, impf.intensity, impf.mdd * impf.paa * 100)
                ax_curve.plot(depths, mdr, label=s, color=self.SECTOR_COLORS.get(s))
            ax_curve.legend(fontsize=8)
        except:
            ax_curve.text(0.5, 0.5, "Curves N/A")
        ax_curve.set_title('Vulnerability Curves')
        ax_curve.set_xlabel('Depth (m)')
        ax_curve.set_ylabel('MDR (%)')
        
        self.plt.tight_layout()
        path = self.output_dir / 'dashboard.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path
    

    
    def plot_damage_concentration(self):
        """Lorenz curve of damage concentration (Risk Inequality)."""
        fig, ax = self.plt.subplots(figsize=(10, 10))
        
        # Filter flooded
        flooded = self.gdf[self.gdf['damage_usd'] > 0].copy()
        if len(flooded) == 0:
            return
            
        # Sort by damage descending
        flooded = flooded.sort_values('damage_usd', ascending=False)
        
        # Calculate cumulative percentages
        total_damage = flooded['damage_usd'].sum()
        cum_damage_pct = flooded['damage_usd'].cumsum() / total_damage * 100
        cum_props_pct = np.arange(1, len(flooded) + 1) / len(flooded) * 100
        
        # Plot curve
        ax.plot(cum_props_pct, cum_damage_pct, color='#DB4437', linewidth=2.5, label='Curva de Concentración')
        
        # Reference line (equality)
        ax.plot([0, 100], [0, 100], color='gray', linestyle='--', alpha=0.5, label='Equilibrio Perfecto')
        
        # Annotate Pareto points (e.g., 20% props -> X% damage)
        p20_idx = int(len(flooded) * 0.2)
        if p20_idx < len(flooded):
            d20_val = cum_damage_pct.iloc[p20_idx]
            ax.plot(20, d20_val, 'o', color='black')
            ax.annotate(f'20% prop. = {d20_val:.0f}% daño', xy=(20, d20_val), 
                       xytext=(25, d20_val-5), arrowprops=dict(arrowstyle='->'))
            
        ax.set_title('Concentración del Daño (Curva de Lorenz)')
        ax.set_xlabel('% Acumulado de Propiedades Afectadas (Ordenadas por Daño)')
        ax.set_ylabel('% Acumulado del Daño Total')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_concentration_curve.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_damage_profile_by_depth(self):
        """Total damage aggregated by flood depth ranges."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        flooded = self.gdf[self.gdf['damage_usd'] > 0].copy()
        if len(flooded) == 0:
            return

        # Define bins
        bins = [0, 0.2, 0.5, 1.0, 2.0, 100]
        labels = ['0-20cm', '20-50cm', '50cm-1m', '1m-2m', '>2m']
        
        flooded['depth_cat'] = pd.cut(flooded['max_depth_m'], bins=bins, labels=labels)
        
        # Aggregate damage
        dmg_by_cat = flooded.groupby('depth_cat')['damage_usd'].sum()
        
        # Plot
        bars = ax.bar(dmg_by_cat.index, dmg_by_cat.values, color='#F4B400', edgecolor='white')
        
        ax.set_title('Perfil de Daño por Rango de Profundidad')
        ax.set_xlabel('Rango de Profundidad')
        ax.set_ylabel('Daño Total Acumulado (USD)')
        
        # Labels
        for bar, val in zip(bars, dmg_by_cat.values):
            pct = val / flooded['damage_usd'].sum() * 100
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val,
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self._format_currency_axis(ax, axis='y')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        self.plt.tight_layout()
        path = self.output_dir / 'damage_profile_by_depth.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

    def plot_exposure_map(self):
        """Map of properties colored by construction value (exposure)."""
        fig, ax = self.plt.subplots(figsize=(12, 10))
        
        # Plot network background
        self._plot_network(ax)
        
        # Filter to those with value
        has_value = self.gdf[self.gdf['estimated_value_usd'] > 0]
        
        if len(has_value) > 0:
            # Colorbar uses 'label'
            # Manual colorbar to fix formatting (1e8 -> M)
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            has_value.plot(ax=ax, column='estimated_value_usd', cmap=self.VALUE_CMAP,
                          legend=False, edgecolor='gray', linewidth=0.1)
                          
            vmin = has_value['estimated_value_usd'].min()
            vmax = has_value['estimated_value_usd'].max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=self.VALUE_CMAP)
            
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
            cbar.set_label('Valor (USD)')
            
            # Apply currency formatter
            self._format_currency_axis(cbar.ax, axis='y')
        
        ax.set_title('Mapa de Exposición (Valor de Construcción)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        ax.set_aspect('equal')
        
        self.plt.tight_layout()
        path = self.output_dir / 'exposure_map.png'
        self.plt.savefig(path)
        self.plt.close()
        print(f"   {path.name}")
        return path

if __name__ == "__main__":
    # =========================================================================
    # EJEMPLO DE USO
    # =========================================================================
    # Puedes ejecutar este módulo directamente para probar con datos de ejemplo
    # 
    # Uso básico (usa rutas por defecto definidas en config.py):
    #     python rut_19_flood_damage_climada.py
    #
    # Uso con ruta específica a raster de profundidad:
    #     Modifica depth_raster abajo
    # =========================================================================
    
    from pathlib import Path
    
    # Ruta al raster de max water depth de ITZI
    # Cambia esto por tu raster de prueba
    depth_raster = Path(r"C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\test_avoided_cost\avoided_cost\flood_damage\max_water_depth.tif")
    
    # Ruta a la red de alcantarillado (opcional, para visualización)
    network_gpkg = Path(r"C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\00_vector\06_red_principal.gpkg")
    
    print(f"Testing CLIMADA flood damage assessment")
    print(f"Depth raster: {depth_raster}")
    print(f"Network: {network_gpkg}")
    
    if not depth_raster.exists():
        print("ERROR: Depth raster not found. Using default paths...")
        result = calculate_flood_damage_climada()
    else:
        result = calculate_flood_damage_climada(depth_raster_path=depth_raster)
    
    # Mostrar resultados
    if "error" not in result:
        print(f"\n{'='*60}")
        print("RESULTADOS FINALES")
        print(f"{'='*60}")
        print(f"  Daño total:              ${result['total_damage_usd']:,.0f} USD")
        print(f"  Propiedades inundadas:   {result.get('flooded_properties', 'N/A'):,}")
        print(f"  Total propiedades:       {result.get('total_properties', 'N/A'):,}")
        print(f"\n  Archivos generados:")
        print(f"    - GPKG: {result.get('output_gpkg', 'N/A')}")
        print(f"    - Report: {result.get('output_report', 'N/A')}")
        
        # -------------------------------------------------------------
        # GENERAR VISUALIZACIONES
        # -------------------------------------------------------------
        try:
            plotter = FloodDamagePlotter(result, network_path=network_gpkg)
            output_dir = plotter.plot_all()
            print(f"\n  Visualizaciones generadas en: {output_dir}")
            
            print(f"\n All plots saved to: {output_dir}")
        except Exception as e:
            print(f"\n  ERROR generando visualizaciones: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"ERROR: {result['error']}")
