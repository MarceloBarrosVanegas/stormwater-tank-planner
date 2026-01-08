"""
clip_predios_to_raster.py
Recorta el shapefile de predios catastrales al extent del raster de elevación del proyecto.
Usa sqlite3 para leer el GPKG y manejar el encoding problemático.
"""
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path
import subprocess
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
import config

# Input files
PREDIO_SHP = config.VECTOR_DIR / "PREDIO" / "predioGeo.shp"
ELEV_RASTER = config.ELEV_FILE  # DEM del proyecto

# Output file
OUTPUT_GPKG = config.VECTOR_DIR / "predios_proyecto.gpkg"


def clip_predios_to_raster():
    """
    Recorta el shapefile de predios al extent del raster de elevación.
    """
    print("="*60)
    print("CLIP PREDIOS CATASTRALES AL ÁREA DEL PROYECTO")
    print("="*60)
    
    # =========================================================================
    # 1. Get raster extent and transform to WGS84
    # =========================================================================
    print(f"\n[1] Leyendo extent del raster: {ELEV_RASTER.name}")
    
    with rasterio.open(ELEV_RASTER) as src:
        bounds = src.bounds
        raster_crs = src.crs
        print(f"    Bounds (proyecto): {bounds}")
        
        bounds_wgs84 = transform_bounds(raster_crs, 'EPSG:4326', 
                                        bounds.left, bounds.bottom, 
                                        bounds.right, bounds.top)
        print(f"    Bounds (WGS84): {bounds_wgs84}")
    
    # =========================================================================
    # 2. Use ogr2ogr to clip
    # =========================================================================
    print(f"\n[2] Recortando con ogr2ogr...")
    
    minx, miny, maxx, maxy = bounds_wgs84
    
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
        print(f"    Borrado archivo existente")
    
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    
    # Use ogr2ogr with specific settings
    cmd = [
        'ogr2ogr',
        '-f', 'GPKG',
        str(OUTPUT_GPKG),
        str(PREDIO_SHP),
        '-spat', str(minx), str(miny), str(maxx), str(maxy),
        '-skipfailures',
        '-oo', 'ENCODING=LATIN1',  # Force input encoding
    ]
    print(f"    Ejecutando ogr2ogr...")
    
    # Run without capturing output to avoid encoding issues
    result = subprocess.run(cmd, capture_output=False)
    
    if not OUTPUT_GPKG.exists():
        print("    Error: No se creó el archivo de salida")
        return None
    
    print(f"    ✓ Archivo creado: {OUTPUT_GPKG.name}")
    
    # =========================================================================
    # 3. Read desteconom values using sqlite3 (avoids geopandas encoding issues)
    # =========================================================================
    print(f"\n[3] Leyendo valores de desteconom con SQLite...")
    
    conn = sqlite3.connect(str(OUTPUT_GPKG))
    
    # Get table name
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(f"    Tablas: {list(tables['name'])}")
    
    # Find the main table (not rtree or gpkg_* tables)
    main_table = None
    for t in tables['name']:
        if not t.startswith('gpkg_') and not t.startswith('rtree_'):
            main_table = t
            break
    
    if main_table:
        print(f"    Tabla principal: {main_table}")
        
        # Count records
        count_df = pd.read_sql(f"SELECT COUNT(*) as count FROM '{main_table}'", conn)
        print(f"    Total predios: {count_df.iloc[0]['count']:,}")
        
        # Get unique desteconom values
        try:
            dest_df = pd.read_sql(f"""
                SELECT desteconom, COUNT(*) as count 
                FROM '{main_table}' 
                GROUP BY desteconom 
                ORDER BY count DESC
            """, conn)
            
            print(f"\n{'='*60}")
            print("VALORES ÚNICOS DE 'desteconom':")
            print("="*60)
            for _, row in dest_df.head(25).iterrows():
                dest = str(row['desteconom']) if row['desteconom'] else "(vacío)"
                print(f"  {dest:<35}: {row['count']:>6,}")
            print(f"\n  TOTAL TIPOS: {len(dest_df)}")
            
        except Exception as e:
            print(f"    Error leyendo desteconom: {e}")
            # Try getting column list
            cols_df = pd.read_sql(f"PRAGMA table_info('{main_table}')", conn)
            print(f"    Columnas disponibles: {list(cols_df['name'])}")
    
    conn.close()
    
    # =========================================================================
    # 4. Now read with geopandas for reprojection (using fiona engine)
    # =========================================================================
    print(f"\n[4] Reproyectando a CRS del proyecto...")
    
    try:
        gdf = gpd.read_file(OUTPUT_GPKG, engine='fiona', encoding='latin-1')
        gdf = gdf.to_crs(config.PROJECT_CRS)
        gdf.to_file(OUTPUT_GPKG, driver="GPKG")
        print(f"    ✓ Reproyectado y guardado!")
        return gdf
    except Exception as e:
        print(f"    Advertencia al reproyectar: {e}")
        print(f"    El archivo GPKG existe pero puede necesitar reproyección manual en QGIS")
        return None


if __name__ == "__main__":
    gdf = clip_predios_to_raster()
    print(f"\n" + "="*60)
    print(f"✓ COMPLETADO!")
    print(f"  Archivo: {OUTPUT_GPKG}")
    print("="*60)
