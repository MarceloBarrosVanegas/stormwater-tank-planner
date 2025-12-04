import os
import sys
from pyproj import CRS, Transformer
import pandas as pd

sys.path.append(r'C:\Users\chelo\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\src')
sys.path.append(r'C:\Users\chelo\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\gui')

sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\src')
sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\gui')


from pathlib import Path
import geopandas as gpd
import numpy  as np

from rut_00_path_finder import PathFinder
from rut_02_get_flodded_nodes import SWMMOverflowAnalyzer
from rut_02_elevation import ElevationGetter, ElevationSource
from rut_03_run_sewer_design import SewerPipeline
from rut_01_swmm_handel import SWMMSimplePlot


if __name__ == "__main__":

    inp_file = Path(r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\COLEGIO_TR25_v6.inp").resolve()
    path_proy = r'C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\modulos\tanques_tormenta'
    elev_files_list = [ r'C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\01_raster\elev.tif' ]
    image_path = r'C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\01_raster\imagen.tif'
    
    source_crs = CRS("""PROJCRS["SIRES-DMQ",
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
CS[Cartesian,3],
AXIS["(E)",east,ORDER[1],
    LENGTHUNIT["metre",1,ID["EPSG",9001]]],
AXIS["(N)",north,ORDER[2],
    LENGTHUNIT["metre",1,ID["EPSG",9001]]],
AXIS["ellipsoidal height (h)",up,ORDER[3],
LENGTHUNIT["metre",1,ID["EPSG",9001]]]]""")
    proj_from = source_crs.to_2d()
    proj_to      = "EPSG:32717"
    

    source = ElevationSource(path_proy, proj_to)
    tree = source.get_elev_source(
        elev_files_list,
        check_unique_values=False,
        ellipsoidal2orthometric=False,
        m_ramales=None,
        elevation_shift=0
    )
    getter = ElevationGetter(tree=tree, m_ramales=None, threshold_distance=0.7)


    swmm_solver = SWMMOverflowAnalyzer(
        inp_file,
        w_vol=0.5,
        w_peak=0.3,
        w_hours=0.2,
        top_timeline=10,
        source_crs= proj_from,
    )
    flooding_df, flooding_gdf = swmm_solver.get_all_peak_details()

    
    # get predios vector file
    predios_path = r'C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\00_vector\07_predios_disponibles.shp'
    predios_gdf = gpd.read_file(predios_path)
    predios_gdf['centroide'] = predios_gdf.geometry.centroid
    predios_gdf['z'] = getter.get_elevation_from_tree_coords(predios_gdf.geometry.centroid.get_coordinates().to_numpy())
    

    
    # flodding nodes
    path_proy = r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\00_flooding_stats\00_flooding_nodes.gpkg"
    flooding_gdf =  gpd.read_file(path_proy)
    flooding_gdf.to_crs(proj_to, inplace=True)
    flooding_gdf.index = flooding_gdf['NodeID']
    flooding_gdf['z'] = getter.get_elevation_from_tree_coords(flooding_gdf.geometry.centroid.get_coordinates().to_numpy())
    
    

    # Iterate over each flooded node using itertuples for better performance and readability.
    for node in flooding_gdf.itertuples():
        # Filter properties (predios) that are at a lower or equal elevation than the node's bed elevation
        filtro_elevacion = predios_gdf['z'] <= node.z - node.NodeDepth
    
        # For each suitable property, find and plot the path from the flooded node.
        for predio in predios_gdf[filtro_elevacion].itertuples():
            # Create a detailed title for the plot, including information about the node and the property.
            title = (f"Node {node.NodeID} - Node Elevation {node.z:.2f} - Node Depth {node.NodeDepth:.2f} - Node Flooding Flow {node.FloodingFlow:.3f} -Volume {node.FloodingVolume:.2f}\n"
                     f"Predio Elevation {predio.z:.2f} - Predio Area {predio.geometry.area:.2f} - Predio ID {predio.Index}\n")
    
            # Define the start coordinates as the centroid of the flooded node's geometry.
            coords = node.geometry.centroid
            start_coords = (coords.x, coords.y)
    
            # Define the end coordinates as the centroid of the target property.
            end_coords = (predio.centroide.x, predio.centroide.y)
            
    
            # Define pathfinding weights and road preferences
            path_weights = {
                'length_weight': 0.4,
                'elevation_weight': 0.4,
                'road_weight': 0.2
            }
            road_preferences = {
                'motorway': 5.0, 'trunk': 5.0, 'primary': 2.5,
                'secondary': 2.5, 'tertiary': 2.5, 'residential': 0.5,
                'service': 1.0, 'unclassified': 1.0,
                'footway': 10.0, 'path': 10.0, 'steps': 20.0,
                'default': 1.5
            }
    
            # Execute the PathFinder to find and plot the optimal route.
            pf = PathFinder(start_point=start_coords, end_point=end_coords,  proj_to=proj_to)
            
            pf.run(
                title=title,
                path_proy=path_proy,
                elev_files_list=elev_files_list,
                proj_to=proj_to,
                image_path=image_path,
                **path_weights,
                road_preferences=road_preferences
            )
        
            # net = SWMMSimplePlot(str(inp_file))
            # # Extract the NodeID from your pandas object
            # node_id = str(node.get('NodeID', node.name)) if hasattr(node, 'get') else str(node)
            # nodes_gdf, edges_gdf = net.get_upstream_network(node_id)
            # nodes_gdf, edges_gdf = net.get_upstream_network(node)
            # net.plot(selected_node=node)

        
            
            
            pipe = SewerPipeline(
                project_name=pf.vector_name,
                pozo_hmin_arg=max(pf.node_depth + 0.5, 8),
                q = pf.flooding_flow,
                proj_to=32717
            )
            pipe.run()
            
            sys.exit()
            

