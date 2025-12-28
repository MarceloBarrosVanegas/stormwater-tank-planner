"""
rut_19_flood_damage_climada.py
Flood damage assessment using CLIMADA with JRC depth-damage curves.

Uses:
- Itzi water depth raster (max_water_depth.tif)
- Predios GPKG with property values and land use
- CLIMADA JRC South America depth-damage curves

Requirements:
    pip install climada climada-petals

Output:
- Total flood damage cost (USD)
- Damage per property (added to predios GPKG)
"""
# =============================================================================
# IMPORTS
# =============================================================================
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - All paths from config.py
# =============================================================================
import config

# Project paths
project_root = config.PROJECT_ROOT
vector_dir = config.VECTOR_DIR
itzi_output_dir = config.ITZI_OUTPUT_DIR

# Input files
predios_file = vector_dir / "08_predios_curvas.gpkg"
depth_raster_file = itzi_output_dir / "max_water_depth.tif"

# Output files
damage_results_gpkg = itzi_output_dir / "flood_damage_results.gpkg"
damage_results_txt = itzi_output_dir / "flood_damage_results.txt"

# Column names in predios
VALUE_COLUMN = "valconstru"
USE_COLUMN = "uso_vige"

# =============================================================================
# CLIMADA JRC SECTORS (South America)
# Available sectors: residential, commercial, industrial, transport, infrastructure, agriculture
# =============================================================================

# Mapping from 'uso_vige' field to CLIMADA JRC sector
# Values: 'Multiple', 'Equipamiento', 'P. Ecol/Conser. Patri. N', 'Resid Urbano 2',
#         'Resid Urbano 3', 'Resid Urbano 1', 'Industrial 3', 'Agricola Resid.',
#         'Area promocion', 'Industrial 2', 'Resid Urbano 1QT', 'Resid Rural 1'

LAND_USE_MAPPING = {
    # Residential
    "Resid Urbano 1": "residential",
    "Resid Urbano 2": "residential",
    "Resid Urbano 3": "residential",
    "Resid Urbano 1QT": "residential",
    "Resid Rural 1": "residential",
    "Agricola Resid.": "residential",  # Mixed agricultural-residential
    
    # Commercial / Mixed
    "Multiple": "commercial",  # Mixed use typically has commercial
    "Area promocion": "commercial",  # Development areas
    
    # Industrial
    "Industrial 2": "industrial",
    "Industrial 3": "industrial",
    
    # Infrastructure
    "Equipamiento": "infrastructure",  # Public facilities
    
    # Agriculture / Natural
    "P. Ecol/Conser. Patri. N": "agriculture",  # Ecological protection
}

DEFAULT_SECTOR = "residential"


def calculate_flood_damage_climada(
    predios_path: Path = None,
    depth_raster_path: Path = None,
    value_column: str = None,
    use_column: str = None,
    output_gpkg: Path = None,
    output_txt: Path = None
) -> dict:
    """
    Calculate flood damage using CLIMADA with JRC curves.
    
    Parameters
    ----------
    predios_path : str
        Path to predios GPKG
    depth_raster_path : str
        Path to max water depth raster from Itzi
    value_column : str
        Column name with property value
    use_column : str
        Column name with land use type
    output_path : str
        Optional path to save results
        
    Returns
    -------
    dict : Damage summary
    """
    # Import CLIMADA
    try:
        from climada.entity import Exposures, ImpactFuncSet
        from climada.engine import ImpactCalc
        from climada.hazard import Hazard
        from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood
    except ImportError as e:
        print("ERROR: CLIMADA not installed!")
        print("Install with: pip install climada climada-petals")
        return {"error": str(e)}
    
    # Use config defaults if not specified
    if predios_path is None:
        predios_path = predios_file
    if depth_raster_path is None:
        depth_raster_path = depth_raster_file
    if value_column is None:
        value_column = VALUE_COLUMN
    if use_column is None:
        use_column = USE_COLUMN
    if output_gpkg is None:
        output_gpkg = damage_results_gpkg
    if output_txt is None:
        output_txt = damage_results_txt
    
    print("="*60)
    print("FLOOD DAMAGE ASSESSMENT (CLIMADA + JRC Curves)")
    print("="*60)
    
    # =========================================================================
    # 1. CREATE HAZARD FROM WATER DEPTH RASTER
    # =========================================================================
    print(f"\n[1] Loading hazard from: {depth_raster_path}")
    
    if not Path(depth_raster_path).exists():
        print(f"  ERROR: Depth raster not found!")
        return {"error": "Depth raster not found"}
    
    # Load as CLIMADA Hazard (RF = River Flood)
    haz = Hazard.from_raster([depth_raster_path], haz_type="RF")
    print(f"  Hazard created: {haz.size} events, {len(haz.centroids.lat)} centroids")
    
    # =========================================================================
    # 2. CREATE EXPOSURES FROM PREDIOS
    # =========================================================================
    print(f"\n[2] Loading exposures from: {predios_path}")
    
    gdf = gpd.read_file(predios_path)
    print(f"  Properties loaded: {len(gdf):,}")
    
    # Check required columns
    if value_column not in gdf.columns:
        print(f"  ERROR: Value column '{value_column}' not found")
        print(f"  Available: {list(gdf.columns)}")
        return {"error": f"Column {value_column} not found"}
    
    # Prepare exposures dataframe
    # CLIMADA needs: latitude, longitude, value, impf_RF
    gdf = gdf.copy()
    gdf["value"] = gdf[value_column].fillna(0)
    
    # Get centroids
    gdf["latitude"] = gdf.geometry.centroid.y
    gdf["longitude"] = gdf.geometry.centroid.x
    
    # Map land use to sector
    if use_column in gdf.columns:
        gdf["_sector"] = gdf[use_column].map(
            lambda x: LAND_USE_MAPPING.get(x, DEFAULT_SECTOR) if x else DEFAULT_SECTOR
        )
    else:
        gdf["_sector"] = DEFAULT_SECTOR
    
    # Print sector distribution
    print("\n  Sector distribution:")
    for sector, count in gdf["_sector"].value_counts().items():
        print(f"    {sector}: {count:,}")
    
    # =========================================================================
    # 3. CREATE IMPACT FUNCTIONS (JRC South America curves)
    # =========================================================================
    print("\n[3] Loading JRC impact functions (South America)...")
    
    impf_set = ImpactFuncSet()
    sectors = ["residential", "commercial", "industrial", "transport", "infrastructure", "agriculture"]
    sector_ids = {}
    
    for i, sector in enumerate(sectors):
        impf = ImpfRiverFlood.from_jrc_region_sector("South America", sector)
        impf.id = i + 1  # Assign unique ID
        impf_set.append(impf)
        sector_ids[sector] = impf.id
        print(f"  {sector}: id={impf.id}")
    
    # Assign impact function ID to each property based on sector
    gdf["impf_RF"] = gdf["_sector"].map(sector_ids)
    
    # =========================================================================
    # 4. CREATE CLIMADA EXPOSURES
    # =========================================================================
    print("\n[4] Creating CLIMADA Exposures...")
    
    exp = Exposures(gdf)
    exp.set_lat_lon()
    exp.check()
    
    print(f"  Total exposure value: ${exp.gdf['value'].sum():,.0f}")
    
    # =========================================================================
    # 5. CALCULATE IMPACT
    # =========================================================================
    print("\n[5] Calculating flood impact...")
    
    impact = ImpactCalc(exp, impf_set, haz).impact()
    
    # Get damage per property
    gdf["damage_usd"] = impact.eai_exp  # Expected annual impact per exposure
    
    # =========================================================================
    # 6. CALCULATE DETAILED COST BREAKDOWN
    # =========================================================================
    # Based on FEMA/JRC damage categories:
    # - Structural damage: 60% of total (building structure)
    # - Contents damage: 25% of total (furniture, equipment)
    # - Cleanup costs: 10% of total (debris removal, sanitization)
    # - Business interruption: 5% of total (lost revenue, relocation)
    
    gdf["damage_structural"] = gdf["damage_usd"] * 0.60
    gdf["damage_contents"] = gdf["damage_usd"] * 0.25
    gdf["damage_cleanup"] = gdf["damage_usd"] * 0.10
    gdf["damage_interruption"] = gdf["damage_usd"] * 0.05
    
    # =========================================================================
    # 7. AGGREGATE RESULTS
    # =========================================================================
    total_damage = gdf["damage_usd"].sum()
    total_value = gdf["value"].sum()
    flooded_count = (gdf["damage_usd"] > 0).sum()
    
    # Damage by sector
    damage_by_sector = gdf.groupby("_sector")["damage_usd"].sum().to_dict()
    count_by_sector = gdf.groupby("_sector").size().to_dict()
    
    # Cost breakdown totals
    total_structural = gdf["damage_structural"].sum()
    total_contents = gdf["damage_contents"].sum()
    total_cleanup = gdf["damage_cleanup"].sum()
    total_interruption = gdf["damage_interruption"].sum()
    
    print("\n" + "="*60)
    print("DAMAGE SUMMARY")
    print("="*60)
    print(f"  Total properties: {len(gdf):,}")
    print(f"  Properties with damage: {flooded_count:,} ({100*flooded_count/len(gdf):.1f}%)")
    print(f"  Total property value: ${total_value:,.0f}")
    print(f"\n  TOTAL FLOOD DAMAGE: ${total_damage:,.0f}")
    print(f"  Damage as % of value: {100*total_damage/total_value:.2f}%" if total_value > 0 else "")
    print("\n  Cost breakdown:")
    print(f"    Structural damage (60%): ${total_structural:,.0f}")
    print(f"    Contents damage (25%):   ${total_contents:,.0f}")
    print(f"    Cleanup costs (10%):     ${total_cleanup:,.0f}")
    print(f"    Business interruption:   ${total_interruption:,.0f}")
    print("\n  Damage by sector:")
    for sector, amount in sorted(damage_by_sector.items(), key=lambda x: -x[1]):
        print(f"    {sector}: ${amount:,.0f} ({count_by_sector.get(sector, 0):,} properties)")
    print("="*60)
    
    # =========================================================================
    # 8. SAVE RESULTS
    # =========================================================================
    print(f"\n[7] Saving results:")
    print(f"    GPKG: {output_gpkg}")
    print(f"    Report: {output_txt}")
    
    # Create output directory if needed
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    
    # Save GPKG
    gdf.to_file(output_gpkg, driver="GPKG")
    
    # Save text report
    from datetime import datetime
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FLOOD DAMAGE ASSESSMENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("INPUT DATA\n")
        f.write("-"*70 + "\n")
        f.write(f"Predios file: {predios_path}\n")
        f.write(f"Depth raster: {depth_raster_path}\n")
        f.write(f"Value column: {value_column}\n")
        f.write(f"Land use column: {use_column}\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-"*70 + "\n")
        f.write("Damage curves: JRC Global Flood Depth-Damage Functions\n")
        f.write("Region: South America\n")
        f.write("Tool: CLIMADA (Climate Adaptation Framework)\n")
        f.write("Sectors: residential, commercial, industrial, transport,\n")
        f.write("         infrastructure, agriculture\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total properties analyzed:     {len(gdf):>15,}\n")
        f.write(f"Properties with damage:        {flooded_count:>15,} ({100*flooded_count/len(gdf):.1f}%)\n")
        f.write(f"Total property value:          ${total_value:>14,.0f}\n")
        f.write(f"TOTAL FLOOD DAMAGE:            ${total_damage:>14,.0f}\n")
        f.write(f"Damage as % of value:          {100*total_damage/total_value:>14.2f}%\n" if total_value > 0 else "")
        f.write("\n")
        
        f.write("COST BREAKDOWN\n")
        f.write("-"*70 + "\n")
        f.write("Category                       Amount (USD)        % of Total\n")
        f.write("-"*70 + "\n")
        f.write(f"Structural damage (buildings)  ${total_structural:>14,.0f}         60%\n")
        f.write(f"Contents damage (inventory)    ${total_contents:>14,.0f}         25%\n")
        f.write(f"Cleanup & restoration          ${total_cleanup:>14,.0f}         10%\n")
        f.write(f"Business interruption          ${total_interruption:>14,.0f}          5%\n")
        f.write("-"*70 + "\n")
        f.write(f"TOTAL                          ${total_damage:>14,.0f}        100%\n")
        f.write("\n")
        
        f.write("DAMAGE BY SECTOR\n")
        f.write("-"*70 + "\n")
        f.write("Sector               Properties      Damage (USD)    % of Total\n")
        f.write("-"*70 + "\n")
        for sector, amount in sorted(damage_by_sector.items(), key=lambda x: -x[1]):
            pct = 100 * amount / total_damage if total_damage > 0 else 0
            cnt = count_by_sector.get(sector, 0)
            f.write(f"{sector:<20} {cnt:>10,}    ${amount:>14,.0f}     {pct:>5.1f}%\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    result = {
        "total_damage_usd": float(total_damage),
        "total_properties": len(gdf),
        "flooded_properties": int(flooded_count),
        "total_value_usd": float(total_value),
        "damage_percent": 100 * total_damage / total_value if total_value > 0 else 0,
        "damage_structural_usd": float(total_structural),
        "damage_contents_usd": float(total_contents),
        "damage_cleanup_usd": float(total_cleanup),
        "damage_interruption_usd": float(total_interruption),
        "damage_by_sector": damage_by_sector,
        "output_gpkg": output_path,
        "output_report": report_path
    }
    
    return result


if __name__ == "__main__":
    # Simple execution using config defaults
    result = calculate_flood_damage_climada()
    print(f"\nTotal Damage: ${result.get('total_damage_usd', 0):,.0f}")

