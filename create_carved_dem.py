"""
create_carved_dem.py
Creates a carved DEM by lowering elevation in street areas.

Usage:
    python create_carved_dem.py input_dem.tif streets_buffer.gpkg output_carved_dem.tif

This lowers the DEM elevation in street areas to channel water flow through streets.
"""
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from pathlib import Path
import argparse

# Default carve depth (meters to lower DEM in streets)
DEFAULT_CARVE_DEPTH = 0.30


def create_carved_dem(dem_path: str, streets_gpkg: str, output_path: str, 
                      carve_depth: float = DEFAULT_CARVE_DEPTH) -> str:
    """
    Creates a carved DEM by lowering elevation in street areas.
    
    Parameters
    ----------
    dem_path : str
        Path to input DEM raster
    streets_gpkg : str
        Path to streets buffer GPKG (polygons)
    output_path : str
        Path for output carved DEM
    carve_depth : float
        Depth in meters to lower streets (default 0.30m)
        
    Returns
    -------
    str : Path to created raster
    """
    print(f"[Carve] Reading DEM: {dem_path}")
    
    # Read streets
    print(f"[Carve] Reading streets buffer: {streets_gpkg}")
    streets = gpd.read_file(streets_gpkg)
    print(f"  Features: {len(streets)}")
    
    # Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        
        print(f"  DEM shape: {dem.shape}")
        print(f"  DEM CRS: {src.crs}")
        
        # Reproject streets to match DEM CRS if needed
        if streets.crs != src.crs:
            print(f"  Reprojecting streets from {streets.crs} to {src.crs}")
            streets = streets.to_crs(src.crs)
        
        # Create mask from street polygons
        print(f"\n[Carve] Rasterizing street buffer...")
        street_mask = features.rasterize(
            [(geom, 1) for geom in streets.geometry],
            out_shape=dem.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        street_cells = np.sum(street_mask == 1)
        total_cells = dem.size
        print(f"  Street cells: {street_cells:,} ({100*street_cells/total_cells:.2f}%)")
        
        # Apply carve - lower elevation where streets are
        print(f"\n[Carve] Applying carve depth: {carve_depth}m")
        dem_carved = dem.copy()
        dem_carved[street_mask == 1] -= carve_depth
        
        # Statistics
        original_mean = np.nanmean(dem[street_mask == 1])
        carved_mean = np.nanmean(dem_carved[street_mask == 1])
        print(f"  Street area original mean: {original_mean:.2f}m")
        print(f"  Street area carved mean: {carved_mean:.2f}m")
        
        # Write output
        print(f"\n[Carve] Writing carved DEM: {output_path}")
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(dem_carved, 1)
    
    print(f"\n[Carve] Done!")
    print(f"  Carved {street_cells:,} cells by {carve_depth}m")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create carved DEM from streets buffer')
    parser.add_argument('dem', help='Input DEM raster')
    parser.add_argument('streets', help='Streets buffer GPKG (polygons)')
    parser.add_argument('output', help='Output carved DEM path')
    parser.add_argument('--depth', type=float, default=DEFAULT_CARVE_DEPTH,
                        help=f'Carve depth in meters (default: {DEFAULT_CARVE_DEPTH})')
    
    args = parser.parse_args()
    
    create_carved_dem(args.dem, args.streets, args.output, args.depth)


if __name__ == "__main__":
    main()
