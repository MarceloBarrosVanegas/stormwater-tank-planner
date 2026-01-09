import os
import osmnx as ox
import networkx as nx
import rasterio
from pyproj import CRS, Transformer
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from rasterio.warp import transform_bounds
from shapely.ops import linemerge
import sys
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import split, linemerge, unary_union


# Add local modules to path
sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\src')
sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\gui')
sys.path.append(r'C:\Users\chelo\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\src')
sys.path.append(r'C:\Users\chelo\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\gui')

from rut_02_elevation import ElevationGetter, ElevationSource
from utils_pypiper import DirTree





class PathFinder:
    """
    Finds the least-cost path between two points considering distance and elevation.
    """

    def __init__(self, start_point, end_point, proj_to):
        """
        Initializes the PathFinder with start and end points.

        Args:
            start_point (tuple): A tuple of (easting, northing) for the start point.
            end_point (tuple): A tuple of (easting, northing) for the end point.
            crs (str): The Coordinate Reference System of the input points (e.g., "EPSG:32717").
        """
        self.start_point = start_point
        self.end_point = end_point
        self.source_crs = CRS.from_user_input(proj_to)
        self.target_crs = CRS.from_user_input("EPSG:4326")  # WGS84 Lat/Lon

        # Convert input coordinates to Lat/Lon for internal use with osmnx
        self.start_point_latlon = self._transform_to_latlon(start_point)
        self.end_point_latlon = self._transform_to_latlon(end_point)


        self.graph = None
        self.used_edges = set()  # Track edges already used by existing pipelines
        self.junction_node = None  # Set if path ends at a junction with existing pipe
        print("PathFinder initialized with CRS.")

    def mark_edges_used(self, path_nodes: list):
        """
        Mark the edges of a path as used by an existing pipeline.
        Future paths that cross these edges will terminate at the junction.
        
        Args:
            path_nodes: List of node IDs from a completed path
        """
        if not path_nodes or len(path_nodes) < 2:
            return
        for i in range(len(path_nodes) - 1):
            edge = (path_nodes[i], path_nodes[i+1])
            edge_rev = (path_nodes[i+1], path_nodes[i])
            self.used_edges.add(edge)
            self.used_edges.add(edge_rev)  # Add both directions
    
    def clear_used_edges(self):
        """Clear all used edges (start fresh for a new optimization run)."""
        self.used_edges.clear()
        self.junction_node = None
    
    def get_junction_info(self):
        """Returns info about junction if path ended at one."""
        return self.junction_node

    def _transform_to_latlon(self, point):
        """Converts coordinates from the source CRS to latitude and longitude."""
        transformer = Transformer.from_crs(self.source_crs, self.target_crs, always_xy=True)
        lon, lat = transformer.transform(point[0], point[1])
        return (lat, lon)

    def set_start_end_points(self, start_point, end_point):
        """
        Updates the start and end points for the path finder.
        Useful for reusing the existing graph with new endpoints.
        
        Args:
            start_point (tuple): (easting, northing)
            end_point (tuple): (easting, northing)
        """
        self.start_point = start_point
        self.end_point = end_point
        self.start_point_latlon = self._transform_to_latlon(start_point)
        self.end_point_latlon = self._transform_to_latlon(end_point)

    def download_osm_data(self, safety_margin_km=0.5, network_type='all', cache_path=None):

        """
        Downloads street network data from OpenStreetMap or loads from cache.

        Args:
            safety_margin_km (float): The safety margin in kilometers.
            network_type (str): The type of network to download.
            cache_path (str, optional): Path to .graphml file for caching.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading OSM data from cache: {cache_path}...")
            try:
                self.graph = ox.load_graphml(cache_path)
                print("OSM graph loaded from cache successfully.")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Proceeding to download.")

        print(f"Downloading OSM data for network type '{network_type}'...")
        try:
            # Calculate Euclidean distance in meters
            dist_between_points_m = math.sqrt(
                (self.end_point[0] - self.start_point[0]) ** 2 +
                (self.end_point[1] - self.start_point[1]) ** 2
            )

            radius_m = (dist_between_points_m / 2) + (safety_margin_km * 1000)

            center_point = (
                (self.start_point_latlon[0] + self.end_point_latlon[0]) / 2,
                (self.start_point_latlon[1] + self.end_point_latlon[1]) / 2
            )

            self.graph = ox.graph_from_point(center_point, dist=radius_m, network_type=network_type)
            print("OSM data downloaded and graph created successfully.")
            
            # Save to cache if requested
            if cache_path and self.graph is not None:
                try:
                    ox.save_graphml(self.graph, cache_path)
                    print(f"OSM graph saved to cache: {cache_path}")
                except Exception as e:
                     print(f"Warning: Could not save OSM cache: {e}")

        except Exception as e:
            print(f"An error occurred during OSM data download: {e}")
            self.graph = None


    def get_graph_geodataframes(self):
        """
        Converts the graph to node and edge GeoDataFrames.

        Returns:
            tuple: A tuple containing two GeoDataFrames (nodes, edges).
                   Returns (None, None) if the graph is not available.
        """
        if self.graph:
            print("Converting graph to GeoDataFrames...")
            nodes, edges = ox.graph_to_gdfs(self.graph)
            print("GeoDataFrames created successfully.")
            return nodes, edges
        else:
            print("No graph data available to convert.")
            return None, None


    def set_node_elevations(self, nodes_gdf, elevations):
        """
        Updates the graph nodes with elevation data.

        Args:
            nodes_gdf (GeoDataFrame): GeoDataFrame of the graph nodes.
            elevations (array-like): An array or list of elevation values corresponding
                                     to the nodes in nodes_gdf.
        """
        if self.graph and nodes_gdf is not None and len(nodes_gdf) == len(elevations):
            print("Updating graph with elevation data...")
            # Create a dictionary mapping node ID to its elevation
            elevation_dict = dict(zip(nodes_gdf.index, elevations))
            # Set the 'elevation' attribute for each node in the graph
            nx.set_node_attributes(self.graph, elevation_dict, 'elevation')
            print("Graph nodes updated with elevation.")
            print('-'*60)
            print()

            # You can verify by checking a node:
            # first_node_id = list(self.graph.nodes)[0]
            # print(f"Elevation for node {first_node_id}: {self.graph.nodes[first_node_id].get('elevation')}")
        else:
            print("Could not update elevations. Check graph, nodes GeoDataFrame, or elevations array.")



    def find_shortest_path_with_elevation_old(self,
                                              length_weight=0.4,
                                              elevation_weight=0.4,
                                              road_weight=0.2,
                                              max_depth=7.0,
                                              road_preferences=None):
        """
        Encuentra la ruta óptima para alcantarillado por gravedad considerando:
        - Longitud del trayecto
        - Profundidad extra de excavación (cuando el terreno sube)
        - Preferencias de tipo de vía
        - Penalización por profundidad excesiva
        
        Args:
            length_weight (float): Peso para longitud de arista.
            elevation_weight (float): Peso para profundidad extra de excavación.
            road_weight (float): Peso para penalización por tipo de vía.
            max_depth (float): Profundidad máxima antes de penalización extrema.
            road_preferences (dict, opcional): Mapeo de tipos de vía a factores de penalización.
        
        Returns:
            list: IDs de nodos de la ruta más corta o None.
        """
        if not self.graph:
            print("Grafo no disponible. No se puede encontrar la ruta.")
            return None
    
        # Validar que los pesos sean válidos y sumen 1
        total_w = length_weight + elevation_weight + road_weight
        if any(w < 0 for w in (length_weight, elevation_weight, road_weight)) or not math.isclose(total_w, 1.0):
            raise ValueError("Todos los pesos deben ser >= 0 y sumar 1.")
    
        # Penalizaciones por defecto para tipos de vía (menor = más preferible)
        if road_preferences is None:
            road_preferences = {
                'motorway': 0.5, 'trunk': 0.6, 'primary': 0.7,      # Vías principales - más fácil construir
                'secondary': 0.8, 'tertiary': 0.9, 'residential': 1.0,  # Vías medias
                'service': 1.2, 'unclassified': 1.1,                # Vías menores
                'footway': 3.0, 'path': 3.0, 'steps': 5.0           # Evitar estas
            }
        default_penalty = road_preferences.get('default', 1.5)
    
        # Obtener nodos de inicio y fin
        start_node = ox.nearest_nodes(self.graph,
                                      self.start_point_latlon[1],
                                      self.start_point_latlon[0])
        end_node = ox.nearest_nodes(self.graph,
                                    self.end_point_latlon[1],
                                    self.end_point_latlon[0])
        
        print(f"Nodo inicial: {start_node}, Nodo final: {end_node}")
    
        # Calcular valores máximos para normalización
        max_length = 0
        max_elevation_penalty = 0

        for u, v, data in self.graph.edges(data=True):
            length = data.get('length', 0)
            max_length = max(max_length, length)
            
            # Calcular profundidad extra de excavación (solo cuando sube)
            u_elevation = self.graph.nodes[u].get('elevation', 0)
            v_elevation = self.graph.nodes[v].get('elevation', 0)
            elevation_penalty = max(0, v_elevation - u_elevation)  # Solo penalizar subidas
            
            # Penalización adicional si excede profundidad máxima
            if elevation_penalty > max_depth:
                elevation_penalty = elevation_penalty + (elevation_penalty - max_depth) * 5  # Penalización exponencial
            
            max_elevation_penalty = max(max_elevation_penalty, elevation_penalty)
        
        # Evitar división por cero
        max_length = max_length or 1.0
        max_elevation_penalty = max_elevation_penalty or 1.0
        
        # Calcular rangos para penalizaciones de vía
        penalties = list(road_preferences.values()) + [default_penalty]
        min_penalty, max_penalty = min(penalties), max(penalties)
        penalty_range = (max_penalty - min_penalty) or 1.0
    
        # print(f"Longitud máxima: {max_length:.2f}m")
        # print(f"Penalización máxima por elevación: {max_elevation_penalty:.2f}m")
        
        # Calcular costo para cada arista
        for u, v, data in self.graph.edges(data=True):
            # 1. Componente de longitud (normalizada)
            length = data.get('length', 0)
            length_norm = length / max_length
            
            # 2. Componente de elevación (profundidad extra de excavación)
            u_elevation = self.graph.nodes[u].get('elevation', 0)
            v_elevation = self.graph.nodes[v].get('elevation', 0)
            elevation_penalty = max(0, v_elevation - u_elevation)  # Solo penalizar subidas
            
            # Penalización exponencial si excede profundidad máxima
            if elevation_penalty > max_depth:
                elevation_penalty = elevation_penalty + (elevation_penalty - max_depth) * 5
            
            elevation_norm = elevation_penalty / max_elevation_penalty
            
            # 3. Componente de tipo de vía
            highway = data.get('highway', 'unclassified')
            if isinstance(highway, list):
                highway = highway[0]  # Tomar el primer tipo si hay múltiples
            
            road_penalty = road_preferences.get(highway, default_penalty)
            road_norm = (road_penalty - min_penalty) / penalty_range
            
            # Costo total combinado
            total_cost = (length_weight * length_norm +
                         elevation_weight * elevation_norm +
                         road_weight * road_norm)
            
            data['cost'] = total_cost
            
            # Debug para algunas aristas
            if elevation_penalty > 0:
                # print(f"Arista {u}->{v}: longitud={length:.1f}m, subida={v_elevation-u_elevation:.1f}m, "
                #       f"vía={highway}, costo={total_cost:.3f}")
                pass
    
        print(f"Buscando ruta óptima desde nodo {start_node} hasta {end_node}...")
        try:
            self.shortest_path = nx.shortest_path(self.graph,
                                                  source=start_node,
                                                  target=end_node,
                                                  weight='cost')
            
            # Calcular estadísticas de la ruta encontrada
            total_length = 0
            total_elevation_gain = 0
            total_elevation_loss = 0
            
            for i in range(len(self.shortest_path) - 1):
                u = self.shortest_path[i]
                v = self.shortest_path[i + 1]
                edge_data = self.graph[u][v][0]  # Primer arista si hay múltiples
                
                total_length += edge_data.get('length', 0)
                
                u_elev = self.graph.nodes[u].get('elevation', 0)
                v_elev = self.graph.nodes[v].get('elevation', 0)
                elev_diff = v_elev - u_elev
                
                if elev_diff > 0:
                    total_elevation_gain += elev_diff
                else:
                    total_elevation_loss += abs(elev_diff)
            
            print(f"Ruta encontrada:")
            print(f"  - Longitud total: {total_length:.1f}m")
            print(f"  - Elevación ganada (excavación extra): {total_elevation_gain:.1f}m")
            print(f"  - Elevación perdida (favorable): {total_elevation_loss:.1f}m")
            print(f"  - Número de segmentos: {len(self.shortest_path) - 1}")
            
            return self.shortest_path
            
        except nx.NetworkXNoPath:
            print("No se pudo encontrar una ruta entre los puntos de inicio y fin.")
            self.shortest_path = None
            return None

    def find_shortest_path_with_elevation(self,
                                          length_weight=0.4,
                                          elevation_weight=0.4,
                                          road_weight=0.2,
                                          max_depth=7.0,
                                          road_preferences=None):
        """
        Encuentra la ruta óptima para alcantarillado por gravedad considerando:
        - Longitud del trayecto
        - Profundidad extra de excavación (cuando el terreno sube)
        - Preferencias de tipo de vía
        - Penalización por profundidad excesiva
        
        Args:
            length_weight (float): Peso para longitud de arista.
            elevation_weight (float): Peso para profundidad extra de excavación.
            road_weight (float): Peso para penalización por tipo de vía.
            max_depth (float): Profundidad máxima antes de penalización extrema.
            road_preferences (dict, opcional): Mapeo de tipos de vía a factores de penalización.
        
        Returns:
            list: IDs de nodos de la ruta más corta o None.
        """
        if not self.graph:
            print("Grafo no disponible. No se puede encontrar la ruta.")
            return None
    
        # Validar pesos
        total_w = length_weight + elevation_weight + road_weight
        if any(w < 0 for w in (length_weight, elevation_weight, road_weight)) or not math.isclose(total_w, 1.0):
            raise ValueError("Todos los pesos deben ser >= 0 y sumar 1.")
    
        # Penalizaciones por defecto para tipos de vía
        if road_preferences is None:
            road_preferences = {
                'motorway': 0.5, 'trunk': 0.6, 'primary': 0.7,
                'secondary': 0.8, 'tertiary': 0.9, 'residential': 1.0,
                'service': 1.2, 'unclassified': 1.1,
                'footway': 3.0, 'path': 3.0, 'steps': 5.0
            }
        default_penalty = road_preferences.get('default', 1.5)
    
        # Obtener nodos de inicio y fin
        start_node = ox.nearest_nodes(self.graph,
                                      self.start_point_latlon[1],
                                      self.start_point_latlon[0])
        end_node = ox.nearest_nodes(self.graph,
                                    self.end_point_latlon[1],
                                    self.end_point_latlon[0])
    
        print(f"Nodo inicial: {start_node}, Nodo final: {end_node}")
    
        # Calcular valores máximos para normalización
        max_length = 0
        max_elevation_penalty = 0
    
        print("----" *10)
        for u, v, data in self.graph.edges(data=True):
            length = data.get('length', 0)
            max_length = max(max_length, length)
    
            u_elevation = self.graph.nodes[u].get('elevation', 0)
            v_elevation = self.graph.nodes[v].get('elevation', 0)
            elevation_diff = max(0, v_elevation - u_elevation)  # Solo penalizar subidas
    
            if elevation_diff > max_depth:
                elevation_diff = elevation_diff + (elevation_diff - max_depth) * 5
    
            max_elevation_penalty = max(max_elevation_penalty, elevation_diff)
    
        # Evitar división por cero
        max_length = max_length or 1.0
        max_elevation_penalty = max_elevation_penalty or 1.0
    
        # Calcular rangos para penalizaciones de vía
        penalties = list(road_preferences.values()) + [default_penalty]
        min_penalty, max_penalty = min(penalties), max(penalties)
        penalty_range = (max_penalty - min_penalty) or 1.0
            
    
        # Calcular costo para cada arista
        for u, v, data in self.graph.edges(data=True):
            length = data.get('length', 0)
            length_norm = length / max_length
    
            highway = data.get('highway', 'unclassified')
            if isinstance(highway, list):
                highway = highway[0]
    
            road_penalty = road_preferences.get(highway, default_penalty)
            road_norm = (road_penalty - min_penalty) / penalty_range
    
            elevation_diff = max(0, self.graph.nodes[v].get('elevation', 0) - self.graph.nodes[u].get('elevation', 0))
            elevation_norm = elevation_diff / max_elevation_penalty
    
            # Costo total combinado
            total_cost = (length_weight * length_norm +
                          elevation_weight * elevation_norm +
                          road_weight * road_norm)
            data['cost'] = total_cost
    
        try:
            self.shortest_path = nx.shortest_path(self.graph,
                                                  source=start_node,
                                                  target=end_node,
                                                  weight='cost')
            
            # === JUNCTION DETECTION ===
            # Check if path crosses any used edges (existing pipelines)
            # If so, terminate at the junction node (end of smaller flow pipe)
            self.junction_node = None
            if self.used_edges:
                for i in range(len(self.shortest_path) - 1):
                    u = self.shortest_path[i]
                    v = self.shortest_path[i + 1]
                    edge = (u, v)
                    edge_rev = (v, u)
                    
                    # Check if this edge is already used by another pipeline
                    if edge in self.used_edges or edge_rev in self.used_edges:
                        # Junction found! Truncate path here
                        # The path ends at node 'u' (before the used edge)
                        # Node 'u' becomes an intermediate node on the main pipe
                        self.junction_node = {
                            'node_id': u,
                            'connects_to_edge': (u, v),
                            'original_target': end_node,
                            'truncated_at_index': i
                        }
                        
                        # Only truncate if the resulting path has at least 3 nodes (min for viable pipeline)
                        MIN_PATH_NODES = 3
                        if i + 1 >= MIN_PATH_NODES:
                            # Truncate the path to end at the junction
                            self.shortest_path = self.shortest_path[:i+1]
                            print(f"  [PathFinder] Junction detected at node {u}! Path truncated (connects to existing pipeline)")
                        else:
                            # Path would be too short - don't truncate, let it continue
                            self.junction_node = None
                            print(f"  [PathFinder] Junction at node {u} skipped (path too short: {i+1} nodes < {MIN_PATH_NODES})")
                        break
    
            # Extraer coordenadas y elevaciones reales
            path_distances = []
            path_real_elevations = []
    
            current_distance = 0.0
            for i in range(len(self.shortest_path) - 1):
                u = self.shortest_path[i]
                v = self.shortest_path[i + 1]
    
                # Añadir posición actual
                u_elevation = self.graph.nodes[u].get('elevation', 0)
                path_distances.append(current_distance)
                path_real_elevations.append(u_elevation)
    
                # Añadir longitud del tramo
                edge_length = self.graph[u][v][0].get('length', 0)
                current_distance += edge_length
    
            # Añadir último nodo
            last_node = self.shortest_path[-1]
            path_distances.append(current_distance)
            path_real_elevations.append(self.graph.nodes[last_node].get('elevation', 0))
    
            # Calcular línea base con pendiente ≥ 0.4%
            elev_inicio_base = path_real_elevations[0]  # Elevación inicial real
            pendiente_minima = 0.4 / 100  # 0.4% como factor decimal
            elevaciones_base = [elev_inicio_base]
            last_base_elevation = elev_inicio_base  # Esta es la última elevación usada para calcular la nueva
    
            for i in range(len(path_real_elevations) - 1):
                u = self.shortest_path[i]
                v = self.shortest_path[i + 1]
                edge_length = self.graph[u][v][0].get('length', 0)
                elev_inicial = path_real_elevations[i]
                elev_final_real = path_real_elevations[i + 1]
    
                # Pendiente natural (en %)
                pendiente_natural = (elev_inicial - elev_final_real) / edge_length * 100
    
                # Aplicar la regla:
                if pendiente_natural < 0.4:
                    # Forzar pendiente del 0.4%
                    elev_final_ideal = last_base_elevation - edge_length * pendiente_minima
                else:
                    # Usar pendiente natural si es ≥ 0.4%
                    elev_final_ideal = elev_final_real
    
                elevaciones_base.append(elev_final_ideal)
                last_base_elevation = elev_final_ideal  # Actualizar la elevación base para el siguiente segmento
    
            # Dibujar el perfil de elevación
            plt.figure(figsize=(12, 6))
    
            # Perfil real de elevación
            plt.plot(path_distances, path_real_elevations, color='green', linewidth=2, label='Perfil Real')
    
            # Línea base con pendiente ajustada
            plt.plot(path_distances, elevaciones_base, color='red', linestyle='--', linewidth=2, label='Línea Base (≥0.4%)')
    
            # Marcar los nodos
            for i, (dist, elev_real, elev_base) in enumerate(zip(path_distances, path_real_elevations, elevaciones_base)):
                plt.scatter(dist, elev_real, color='blue', s=100, zorder=5)
                plt.text(dist, elev_real + 1, str(i), fontsize=10, ha='center')  # Etiquetas de nodos
    
                # Opcional: dibujar flechas hacia la línea base
                plt.plot([dist, dist], [elev_real, elev_base], color='gray', linestyle=':', linewidth=1, zorder=0)
    
            # # Configuración del gráfico
            # plt.title('Perfil de Elevación a lo largo de la Ruta Óptima')
            # plt.xlabel('Distancia Acumulada (m)')
            # plt.ylabel('Elevación (m)')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
    
            # # Mostrar el gráfico
            # plt.show()
    
            return self.shortest_path
    
        except nx.NetworkXNoPath:
            print("❌ No existe una ruta factible entre los puntos.")
            return None



    def plot_downloaded_area(self, image_path=None, title=None):
        if not self.graph:
            print("No graph data to plot. Please run download_osm_data() first.")
            return



        # Prepare edge colors and collect highway types
        road_colors = {
            'motorway': 'blue', 'trunk': 'blue', 'primary': 'c',
            'secondary': 'c', 'tertiary': 'c', 'residential': 'lightgray',
            'service': 'lightgray', 'unclassified': 'lightgray',
            'footway': 'lightgray', 'path': 'lightgray', 'steps': 'lightgray'
        }
        default_color = 'black'
        edge_colors, highway_types = [], set()

        for u, v, data in self.graph.edges(data=True):
            hwy = data.get('highway', 'unclassified')
            if isinstance(hwy, list):
                hwy = hwy[0]
            edge_colors.append(road_colors.get(hwy, default_color))
            highway_types.add(hwy)

        # Plot base graph without showing or closing
        fig, ax = ox.plot_graph(
            self.graph,
            show=False, close=False,
            edge_color=edge_colors,
            node_size=0,
            bgcolor='#FFFFFF'
        )


        # plot image
        if image_path:
            src = rasterio.open(image_path)
            # Read bands and stack
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
            rgb = np.dstack((r, g, b))

            # Transform raster bounds to the plot CRS (WGS84 lat/lon)
            left, bottom, right, top = transform_bounds(
                src.crs,
                self.target_crs.to_string(),
                *src.bounds
            )

            # Plot with the transformed extent
            ax.imshow(
                rgb,
                extent=(left, right, bottom, top),
                alpha=0.7
            )



        # Overlay start/end points
        ax.scatter(
            self.start_point_latlon[1],
            self.start_point_latlon[0],
            c='lime', s=100, zorder=5, label='Start'
        )
        ax.scatter(
            self.end_point_latlon[1],
            self.end_point_latlon[0],
            c='red', s=100, zorder=5, label='End'
        )

        # Legend for roads
        legend_handles = [
            Line2D([0], [0], color=color, lw=4, label=rt.capitalize())
            for rt, color in road_colors.items()
            if rt in highway_types
        ]
        scatter_h, scatter_l = ax.get_legend_handles_labels()
        ax.legend(handles=scatter_h + legend_handles, loc='upper left', bbox_to_anchor=(1, 1))

        # Plot the route on the same axes
        if hasattr(self, 'shortest_path') and self.shortest_path:
            ox.plot_graph_route(
                self.graph,
                self.shortest_path,
                route_color='r',
                route_linewidth=6,
                node_size=0,
                show=False, close=False,
                ax=ax
            )
            total_length = nx.path_weight(self.graph, self.shortest_path, weight="length")

        # Finalize plot
        ax.set_axis_on()
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.grid(True, linestyle='--', alpha=0.9)
        plt.title(title + ' Path Length: {:.2f} Km'.format(total_length/1000))
        plt.tight_layout()
        
        path_images_dir = r'01_images_paths'
        if not os.path.exists(path_images_dir):
            os.makedirs(path_images_dir, exist_ok=True)
        plt.show()
        plt.savefig(os.path.join(path_images_dir, f"{self.vector_name}_path.png"), dpi=300, bbox_inches='tight')

    def get_path_as_gdf(self):
            """
            Converts the calculated shortest path into a single-feature GeoDataFrame
            containing one merged polyline.
    
            Returns:
                geopandas.GeoDataFrame: A GeoDataFrame with a single polyline representing
                                        the shortest path. Returns None if no path is found.
            """
            if not hasattr(self, 'shortest_path') or not self.shortest_path:
                print("No shortest path available to convert to GeoDataFrame.")
                return None
    
            # Get the full GeoDataFrame of all edges in the graph
            _, edges_gdf = ox.graph_to_gdfs(self.graph)
    
            # Create a list of edge tuples (u, v) from the path's node list
            path_edges = list(zip(self.shortest_path[:-1], self.shortest_path[1:]))
    
            # The edges GDF is indexed by (u, v, key). We need to find the correct key for each edge.
            path_edge_indexes = []
            for u, v in path_edges:
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data:
                    # Add the first key found for the edge (u,v)
                    path_edge_indexes.append((u, v, list(edge_data.keys())[0]))
    
            # Select the edges from the main GeoDataFrame
            path_segments_gdf = edges_gdf.loc[path_edge_indexes]
    
            if path_segments_gdf.empty:
                return None
    
            # Merge the individual line segments into a single geometry object (MultiLineString)
            # and then merge that into a single polyline (LineString).
            merged_line = linemerge(path_segments_gdf.geometry.unary_union)
    
            # Create a new GeoDataFrame for the single polyline
            polyline_gdf = gpd.GeoDataFrame(
                geometry=[merged_line],
                crs=path_segments_gdf.crs
                # crs= self.source_crs
            )
    
            return polyline_gdf

    @classmethod
    def split_by_length(cls, line: LineString, max_len: float) -> list[LineString]:
        coords = list(line.coords)
        parts = []
        start = 0
        acc = 0.0
        for i in range(1, len(coords)):
            acc += LineString([coords[i-1], coords[i]]).length
            if acc >= max_len or i == len(coords) - 1:
                parts.append(LineString(coords[start : i+1]))
                start = i
                acc = 0.0
        return parts

    def get_simplified_path(
        self,
        tolerance: float = 15,
        must_keep: list[tuple[float, float]] = None
    ) -> gpd.GeoDataFrame | None:
        """
        Simplifica la ruta pero preserva:
          1) todos los vértices que queden tras un primer .simplify(preserve_topology=True)
          2) los nodos de self.shortest_path
          3) cualquier punto adicional en must_keep
        """
        from shapely.ops import linemerge
        poly_gdf = self.get_path_as_gdf()
        if poly_gdf is None:
            return None
    
        # nos aseguramos de usar el CRS correcto
        poly_gdf = poly_gdf.to_crs(self.source_crs)
        merged: LineString = poly_gdf.geometry.iloc[0]
    
        # 1) extraemos los vértices "seguros" del primer simplify con preserve_topology
        base_simpl = merged.simplify(15, preserve_topology=True)
        
        
        safe_pts = []
        # Check geometry type to avoid Shapely error when accessing .coords on MultiLineString
        if base_simpl.geom_type == 'LineString':
            # Es LineString simple
            safe_pts = list(base_simpl.coords)
        elif base_simpl.geom_type == 'MultiLineString':
            # Es MultiLineString - extraer coordenadas de cada segmento
            for line in base_simpl.geoms:
                safe_pts.extend(list(line.coords))
        else:
            # Fallback: intentar convertir a LineString con linemerge
            merged_line = linemerge(base_simpl)
            if merged_line.geom_type == 'LineString':
                safe_pts = list(merged_line.coords)
            else:
                # Si aún es multi-parte, extraer de cada geometría
                for line in merged_line.geoms:
                    safe_pts.extend(list(line.coords))
    
        # 2) construimos el conjunto completo de splitters
        node_pts = [
            Point(self.graph.nodes[n]['x'], self.graph.nodes[n]['y'])
            for n in self.shortest_path
        ]
        extra_pts = [Point(x, y) for x, y in (must_keep or [])]
        splitter = MultiPoint(node_pts + extra_pts + [Point(x, y) for x, y in safe_pts])
    
        # 3) cortamos en todos esos puntos
        raw = split(merged, splitter)
        segments = list(raw.geoms) if hasattr(raw, 'geoms') else [raw]
    
        # 4) dividimos cada segmento para que no supere tolerance de longitud
        pieces = []
        for seg in segments:
            pieces.extend(self.split_by_length(seg, tolerance))
    
        # 5) simplificamos cada pieza, preserve_topology evita borrar extremos
        simplified = [
            seg.simplify(tolerance, preserve_topology=True)
            for seg in pieces
        ]
    
        # 6) unimos de nuevo
        uni = unary_union(simplified)
        final = uni if isinstance(uni, LineString) else linemerge(uni)
    
        # 7) devolvemos GeoDataFrame en source_crs
        return gpd.GeoDataFrame(geometry=[final], crs=self.source_crs)
    
    


    def run(self,
            title: str,
            path_proy: str,
            elev_files_list: list[str],
            proj_to: str,
            image_path: str = None,
            length_weight: float = 0.3,
            elevation_weight: float = 0.4,
            road_weight: float = 0.3,
            road_preferences: dict = None
            ) -> None:
        """
        Class method that instantiates PathFinder and executes the full workflow:
         1. Download OSM network
         2. Convert to GeoDataFrames
         3. Project nodes and extract coordinates
         4. Fetch elevations
         5. Update graph with elevations
         6. Compute road preferences, find best path, and plot
        """


        # Download and prepare graph
        self.download_osm_data(safety_margin_km=0.2, network_type='all')
        nodes_gdf, edges_gdf = self.get_graph_geodataframes()
        if nodes_gdf is None:
            return

        # Project and extract coords
        nodes_proj = nodes_gdf.to_crs(proj_to)
        xy = nodes_proj.geometry.get_coordinates().to_numpy()

        # Fetch elevations
        source = ElevationSource(path_proy, proj_to)
        tree = source.get_elev_source(
            elev_files_list,
            check_unique_values=False,
            ellipsoidal2orthometric=False,
            m_ramales=None,
            elevation_shift=0
        )
        getter = ElevationGetter(tree=tree, m_ramales=None, threshold_distance=0.7)
        elevations = getter.get_elevation_from_tree_coords(xy)

        # Update graph
        self.set_node_elevations(nodes_gdf, elevations)

        # Find and plot best path using the provided weights and preferences
        best = self.find_shortest_path_with_elevation(
            length_weight=length_weight,
            elevation_weight=elevation_weight,
            road_weight=road_weight,
            road_preferences=road_preferences,
        )
        if best:
            print("\nNodes in the best path:", best)
            project_name = (
            title.split("-")[0].strip().replace(" ", "").replace("Node", "")
            + "_"
            + title.split("-")[-1].strip().replace(" ", "")
            )
            self.node_depth = float(
            title.split("-")[2].strip().replace(" ", "").replace("NodeDepth", "")
            )
            
            self.flooding_flow = float(
            title.split("-")[3].strip().replace(" ", "").replace("NodeFloodingFlow", "")
            ) * 1000  # Convert  m³/s to l/s

            self.vector_name = project_name

            path_gdf = self.get_simplified_path(tolerance=50)            
            path_gdf.to_file(path_proy)
        
            self.plot_downloaded_area(image_path, title)
            