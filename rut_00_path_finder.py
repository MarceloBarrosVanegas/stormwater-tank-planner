import os
import osmnx as ox
import networkx as nx
import rasterio
from pyproj import CRS, Transformer
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.warp import transform_bounds
import numpy as np
from shapely.geometry import LineString
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import split, linemerge, unary_union
from rdp import rdp
from simplification import cutil

import config
config.setup_sys_path()

from rut_02_elevation import ElevationGetter, ElevationSource

class PathFinder:
    """
    Finds the least-cost path between two points considering distance and elevation.
    """

    def __init__(self, start_point=None, end_point=None, proj_to=None):
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

        if start_point is not  None or end_point is not None:
            # Convert input coordinates to Lat/Lon for internal use with osmnx
            self.start_point_latlon = self._transform_to_latlon(start_point)
            self.end_point_latlon = self._transform_to_latlon(end_point)


        self.graph = None
        self.used_edges = set()  # Track edges already used by existing pipelines
        self.junction_node = None  # Set if path ends at a junction with existing pipe
        

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
                # Ensure it's projected even if loaded from cache
                if self.graph is not None:
                    self.graph = ox.project_graph(self.graph, to_crs=self.source_crs)
                return self.graph
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
            
            # Project graph to source CRS
            if self.graph is not None:
                self.graph = ox.project_graph(self.graph, to_crs=self.source_crs)
            
            # Save to cache if requested
            if cache_path and self.graph is not None:
                try:
                    ox.save_graphml(self.graph, cache_path)
                    print(f"OSM graph saved to cache: {cache_path}")
                except Exception as e:
                     print(f"Warning: Could not save OSM cache: {e}")
                    
            return self.graph

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
            # print("Converting graph to GeoDataFrames...")
            nodes, edges = ox.graph_to_gdfs(self.graph)
            # print("GeoDataFrames created successfully.")
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
            # Create a dictionary mapping node ID to its elevation
            elevation_dict = dict(zip(nodes_gdf.index, elevations))
            # Set the 'elevation' attribute for each node in the graph
            nx.set_node_attributes(self.graph, elevation_dict, 'elevation')
            print('-'*60)
            print()

            # You can verify by checking a node:
            # first_node_id = list(self.graph.nodes)[0]
            # print(f"Elevation for node {first_node_id}: {self.graph.nodes[first_node_id].get('elevation')}")
        else:
            print("Could not update elevations. Check graph, nodes GeoDataFrame, or elevations array.")

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
        # Since the graph is now PROJECTED to self.source_crs (meters), 
        # we pass the projected coordinates directly instead of latlon.
        start_node = ox.nearest_nodes(self.graph,
                                      self.start_point[0],
                                      self.start_point[1])
        end_node = ox.nearest_nodes(self.graph,
                                    self.end_point[0],
                                    self.end_point[1])
    
        print(f"Nodo inicial: {start_node} (UTM X: {self.start_point[0]:.1f}), Nodo final: {end_node} (UTM X: {self.end_point[0]:.1f})")
    
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
    
            # # Dibujar el perfil de elevación
            # plt.figure(figsize=(12, 6))
            #
            # # Perfil real de elevación
            # plt.plot(path_distances, path_real_elevations, color='green', linewidth=2, label='Perfil Real')
            #
            # # Línea base con pendiente ajustada
            # plt.plot(path_distances, elevaciones_base, color='red', linestyle='--', linewidth=2, label='Línea Base (≥0.4%)')
            #
            # # Marcar los nodos
            # for i, (dist, elev_real, elev_base) in enumerate(zip(path_distances, path_real_elevations, elevaciones_base)):
            #     plt.scatter(dist, elev_real, color='blue', s=100, zorder=5)
            #     plt.text(dist, elev_real + 1, str(i), fontsize=10, ha='center')  # Etiquetas de nodos
            #
            #     # Opcional: dibujar flechas hacia la línea base
            #     plt.plot([dist, dist], [elev_real, elev_base], color='gray', linestyle=':', linewidth=1, zorder=0)
            #
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
            union_geom = path_segments_gdf.geometry.unary_union
            
            if union_geom.geom_type == 'LineString':
                merged_line = union_geom
            else:
                merged_line = linemerge(union_geom)
    
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


    def simplify_gdf(self,
        gdf: gpd.GeoDataFrame,
        method: str = "rdp",       # "rdp" o "vw"
        tolerance: float = 5.0,    # epsilon para RDP, threshold para VW
        protected_points: list = None,  # [(x,y), ...] puntos a preservar
        min_distance: float = 0    # Distancia mínima entre puntos resultantes
    ) -> gpd.GeoDataFrame:
        """
        Simplifica las geometrías de un GeoDataFrame de LineStrings.
    
        Args:
            gdf: GeoDataFrame con geometrías LineString
            method: "rdp" (por distancia) o "vw" (por ángulo/área)
            tolerance: epsilon para RDP, threshold para VW
            protected_points: Lista de (x,y) de puntos que NO deben eliminarse
                             (ej: intersecciones de vías)
            min_distance: Distancia mínima entre puntos finales (metros).
                         Después de simplificar, elimina puntos más cercanos que esto.
    
        Returns:
            GeoDataFrame con las geometrías simplificadas
        """
        # Convertir protected_points a set de tuplas redondeadas para búsqueda rápida
        protected_set = set()
        if protected_points:
            for pt in protected_points:
                protected_set.add((round(pt[0], 1), round(pt[1], 1)))
        
        rows = []
    
        for idx, row in gdf.iterrows():
            coords = np.array(row.geometry.coords)
            points = coords[:, :2].copy()
            
            if not protected_set:
                # Sin puntos protegidos: simplificación normal
                if method == "rdp":
                    simplified = rdp(points, epsilon=tolerance)
                elif method == "vw":
                    simplified = cutil.simplify_coords_vw(points, tolerance)
                else:
                    raise ValueError(f"method debe ser 'rdp' o 'vw', recibió: '{method}'")
            else:
                # Con puntos protegidos: simplificar por segmentos
                # Encontrar índices de puntos protegidos
                protected_indices = [0]  # Siempre mantener inicio
                for i, pt in enumerate(points):
                    if (round(pt[0], 1), round(pt[1], 1)) in protected_set:
                        protected_indices.append(i)
                protected_indices.append(len(points) - 1)  # Siempre mantener fin
                protected_indices = sorted(set(protected_indices))
                
                # Simplificar cada segmento entre puntos protegidos
                simplified_list = []
                for j in range(len(protected_indices) - 1):
                    start_idx = protected_indices[j]
                    end_idx = protected_indices[j + 1]
                    segment = points[start_idx:end_idx + 1]
                    
                    if len(segment) <= 2:
                        seg_simplified = segment
                    elif method == "rdp":
                        seg_simplified = rdp(segment, epsilon=tolerance)
                    elif method == "vw":
                        seg_simplified = cutil.simplify_coords_vw(segment, tolerance)
                    else:
                        raise ValueError(f"method debe ser 'rdp' o 'vw', recibió: '{method}'")
                    
                    # Añadir evitando duplicar punto de unión
                    if simplified_list:
                        simplified_list.extend(seg_simplified[1:])
                    else:
                        simplified_list.extend(seg_simplified)
                
                simplified = np.array(simplified_list)
            
            # ================================================================
            # FILTRO FINAL: Eliminar puntos muy cercanos (min_distance)
            # ================================================================
            if min_distance and min_distance > 0 and len(simplified) > 2:
                filtered_coords = [simplified[0]]  # Siempre mantener inicio
                for pt in simplified[1:-1]:  # Recorrer puntos intermedios
                    last_pt = filtered_coords[-1]
                    dist = math.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2)
                    if dist >= min_distance:
                        filtered_coords.append(pt)
                filtered_coords.append(simplified[-1])  # Siempre mantener fin
                simplified = np.array(filtered_coords)
    
            row_new = row.copy()
            row_new.geometry = LineString(simplified)
            rows.append(row_new)
    
        return gpd.GeoDataFrame(rows, crs=gdf.crs)


    def get_simplified_path(
        self,
        tolerance: float = 2,
        min_distance: float = 30,  # Distancia mínima entre intersecciones a preservar
        must_keep: list[tuple[float, float]] = None
    ) -> gpd.GeoDataFrame | None:
        """
        Simplifica la ruta preservando intersecciones reales de vías.
        
        Una intersección real es un nodo del grafo con grado >= 3 
        (3 o más calles conectadas).
        
        Args:
            tolerance: Tolerancia para simplificación RDP (metros)
            min_distance: Distancia mínima entre puntos protegidos (metros)
                         Si dos intersecciones están más cerca, solo se mantiene una
            must_keep: Puntos adicionales a preservar (opcional)
        
        Returns:
            GeoDataFrame con la línea simplificada o None
        """
        poly_gdf = self.get_path_as_gdf()
        if poly_gdf is None:
            return None
        
        # ================================================================
        # IDENTIFICAR INTERSECCIONES REALES (grado >= 3)
        # ================================================================
        raw_intersection_points = []
        for node_id in self.shortest_path:
            if self.graph.degree(node_id) >= 3:
                x = self.graph.nodes[node_id]['x']
                y = self.graph.nodes[node_id]['y']
                raw_intersection_points.append((x, y))
        
        print(f"  [Simplify] Intersecciones detectadas: {len(raw_intersection_points)}")
        
        # ================================================================
        # FILTRAR POR DISTANCIA MÍNIMA
        # ================================================================
        intersection_points = []
        for pt in raw_intersection_points:
            if not intersection_points:
                # Primer punto siempre se agrega
                intersection_points.append(pt)
            else:
                # Calcular distancia al último punto agregado
                last_pt = intersection_points[-1]
                dist = math.sqrt((pt[0] - last_pt[0])**2 + (pt[1] - last_pt[1])**2)
                if dist >= min_distance:
                    intersection_points.append(pt)
        
        print(f"  [Simplify] Después de filtro min_distance={min_distance}m: {len(intersection_points)} puntos protegidos")
        
        # Agregar puntos adicionales si se especifican
        if must_keep:
            intersection_points.extend(must_keep)
        
        # Usar tu función simplify_gdf con puntos protegidos y filtro de distancia
        return self.simplify_gdf(
            poly_gdf, 
            method="rdp", 
            tolerance=tolerance,
            protected_points=intersection_points if intersection_points else None,
            min_distance=min_distance
        )
    
    
    


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
        self.download_osm_data(safety_margin_km=0.5, network_type='all')
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
            