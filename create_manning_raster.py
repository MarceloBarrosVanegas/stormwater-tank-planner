"""
create_manning_raster.py
Creates a Manning friction raster from a classified surface raster.

Usage:
    python create_manning_raster.py input_classified.tif output_manning.tif

The classified raster should have integer values (0, 1, 2, ...) representing
different surface types. The script applies Manning's n values to each class.
"""
import rasterio
import numpy as np
from pathlib import Path
from pyproj import CRS

# =============================================================================
# MANNING VALUES BY SURFACE CLASS
# =============================================================================
# Classification scheme:
# 0 = Asfalto (calles, pavimento)
# 1 = Árboles altos y arbustos
# 2 = Pasto bajo
# 3 = Suelo degradado / tierra / pasto muy corto
# 4 = Urbano (edificios, casas)
# 5 = Urbano (edificios, casas)

MANNING_LOOKUP = {
    0: 0.013,   # Asfalto - muy liso, flujo rápido
    1: 0.120,   # Árboles altos y arbustos - alta resistencia
    2: 0.035,   # Pasto bajo - resistencia moderada
    3: 0.025,   # Suelo degradado/tierra - algo de resistencia
    4: 0.100,   # Urbano - alta resistencia (obstrucciones)
    5: 0.100,   # Urbano - alta resistencia (obstrucciones)
    6: 0.030,   # Páramo con follaje muy bajo
    255: 0.035, # NoData -> default
}

# Default value for unmapped classes
DEFAULT_MANNING = 0.035

# =============================================================================
# PROJECT CRS (SIRES-DMQ)
# =============================================================================
SIRES_DMQ_WKT = """PROJCRS["SIRES-DMQ",
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


def create_manning_raster(input_path: str, output_path: str, 
                          manning_lookup: dict = None) -> str:
    """
    Creates a Manning friction raster from a classified surface raster.
    
    Parameters
    ----------
    input_path : str
        Path to classified raster (integer values)
    output_path : str
        Path for output Manning raster
    manning_lookup : dict, optional
        Dictionary mapping class values to Manning's n
        
    Returns
    -------
    str : Path to created raster
    """
    if manning_lookup is None:
        manning_lookup = MANNING_LOOKUP
    
    print(f"[Manning] Reading classified raster: {input_path}")
    
    with rasterio.open(input_path) as src:
        classified = src.read(1)
        profile = src.profile.copy()
        
        # Get unique classes
        unique_classes = np.unique(classified[~np.isnan(classified.astype(float))])
        print(f"[Manning] Found {len(unique_classes)} unique classes: {unique_classes[:10]}...")
        
        # Create Manning raster
        manning = np.full(classified.shape, DEFAULT_MANNING, dtype=np.float32)
        
        for class_val, manning_n in manning_lookup.items():
            mask = (classified == class_val)
            count = np.sum(mask)
            if count > 0:
                manning[mask] = manning_n
                print(f"  Class {class_val}: Manning {manning_n:.3f} -> {count:,} cells")
        
        # Update profile - PRESERVE INPUT CRS
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=None,
            # CRS stays from source - don't override!
        )
        
        # Write output
        print(f"\n[Manning] Writing output: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(manning, 1)
    
    # Print statistics
    print(f"\n[Manning] Statistics:")
    print(f"  Min: {np.nanmin(manning):.4f}")
    print(f"  Max: {np.nanmax(manning):.4f}")
    print(f"  Mean: {np.nanmean(manning):.4f}")
    print(f"  CRS: {profile.get('crs', 'preserved from input')}")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Manning raster from classified surface')
    parser.add_argument('input', help='Input classified raster (integer values)')
    parser.add_argument('output', help='Output Manning raster path')
    parser.add_argument('--show-lookup', action='store_true', help='Show Manning lookup table')
    
    args = parser.parse_args()
    
    if args.show_lookup:
        print("\n[Manning] Lookup Table:")
        print("-" * 40)
        for class_val, manning_n in sorted(MANNING_LOOKUP.items()):
            print(f"  Class {class_val:3d} -> Manning {manning_n:.3f}")
        print("-" * 40)
        print(f"  Default: {DEFAULT_MANNING}")
        return
    
    create_manning_raster(args.input, args.output)
    print("\n[Manning] Done!")


if __name__ == "__main__":
    main()
