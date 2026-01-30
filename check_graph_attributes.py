import osmnx as ox
import pandas as pd
from pathlib import Path

# Load the cache file
cache_path = Path('../osm_cache.graphml')

if cache_path.exists():
    print(f"Loading graph from {cache_path}...")
    G = ox.load_graphml(cache_path)
    
    # Get edge attributes
    _, edges = ox.graph_to_gdfs(G)
    
    print("\nColumns available in edges GeoDataFrame:")
    print(edges.columns.tolist())
    
    if 'highway' in edges.columns:
        print("\nFound 'highway' attribute. Top values:")
        print(edges['highway'].value_counts().head(10))
    else:
        print("\n'highway' attribute NOT found!")
else:
    print(f"Cache file {cache_path} not found.")
