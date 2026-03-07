import geopandas as gpd
import pandas as pd

# Leer el archivo GPKG
gpkg_path = 'optimization_results/Seq_Iter_26/Seq_Iter_26.gpkg'

print("=" * 100)
print("ANALISIS DEL ARCHIVO GPKG")
print("=" * 100)

# Listar capas disponibles
layers = gpd.list_layers(gpkg_path)
print("\nCapas disponibles en el GPKG:")
print(layers)

# Leer cada capa
for layer_name in layers['name']:
    print(f"\n{'='*100}")
    print(f"CAPA: {layer_name}")
    print("=" * 100)
    
    gdf = gpd.read_file(gpkg_path, layer=layer_name)
    print(f"Columnas: {list(gdf.columns)}")
    print(f"Filas: {len(gdf)}")
    print("\nPrimeras filas:")
    print(gdf.head())
    
    # Si es la capa de tanques, mostrar volúmenes
    if 'volume' in gdf.columns or 'tank_volume' in gdf.columns or 'Vol' in str(gdf.columns):
        vol_col = None
        for col in gdf.columns:
            if 'volume' in col.lower() or 'vol' in col.lower():
                vol_col = col
                break
        if vol_col:
            print(f"\n--- Volumenes ({vol_col}) ---")
            print(gdf[vol_col].describe())
            print(f"Suma total: {gdf[vol_col].sum():.2f} m³")
