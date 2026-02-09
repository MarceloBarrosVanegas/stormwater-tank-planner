"""
Dynamic Solution Evaluator for Tank Optimization
=================================================

Evaluates optimization solutions by executing the full workflow:
PathFinder → SewerPipeline → SWMMModifier → SWMM → flooding residual

Dependencies
------------
- rut_00_path_finder: PathFinder for route optimization
- rut_03_run_sewer_design: SewerPipeline for pipe design
- rut_14_swmm_modifier: SWMMModifier for adding tanks to SWMM
- utils_pypiper: DirTree for project structure
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Any, Dict
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import osmnx as ox
import functools
import numpy as np
import json

import config
config.setup_sys_path()

from rut_00_path_finder import PathFinder
from rut_02_elevation import ElevationGetter, ElevationSource
from rut_03_run_sewer_design import SewerPipeline
from rut_06_pipe_sizing import SeccionLlena, SeccionParcialmenteLlena, CapacidadMaximaTuberia
from rut_02_get_flodded_nodes import CandidatePair
from rut_13_cost_functions import CostCalculator
from rut_14_swmm_modifier import SWMMModifier
# from rut_17_comparison_reporter import ScenarioComparator
# from rut_18_itzi_flood_model import run_itzi_for_case
# from rut_19_flood_damage_climada import calculate_flood_damage_climada
from rut_20_avoided_costs import AvoidedCostRunner
from rut_21_construction_cost import SewerConstructionCost
from rut_27_model_metrics import MetricExtractor, SystemMetrics




def combined_weight(graph, path_weights, road_preferences, u, v, d):
    length_cost = d.get('length', 1) * path_weights['length_weight']
    
    # Calculate elevation difference between nodes u and v
    u_elev = graph.nodes[u].get('elevation', 0)
    v_elev = graph.nodes[v].get('elevation', 0)
    # Penalize uphill changes (v_elev > u_elev) for gravity flow paths
    elevation_diff = max(0, v_elev - u_elev)
        
    elevation_cost = elevation_diff * path_weights['elevation_weight']
    road_type = d.get('highway', 'default')
    if isinstance(road_type, list): road_type = road_type[0]
    road_cost = road_preferences.get(road_type, road_preferences['default']) * path_weights['road_weight']
    return length_cost + elevation_cost + road_cost


def calculate_path_for_candidate(candidate, graph, source_node, target_node, path_weights, road_preferences):
    """Calculate path length for a candidate - runs in parallel threads."""
    weight_func = functools.partial(combined_weight, graph, path_weights, road_preferences)
    try:
        # Calculate shortest path using combined weight
        try:
            path = nx.shortest_path(graph, source_node, target_node, weight=weight_func)
        except nx.NetworkXNoPath:
            return candidate, None, None, "No path"
        
        if path and len(path) >= 2:
            # Create LineString from path coordinates
            coords = [(graph.nodes[node]['x'], graph.nodes[node]['y']) for node in path]
            linestring = LineString(coords)
            # Use Euclidean distance (length of LineString in meters if projected)
            distance = linestring.length
            return candidate, distance, linestring, None
        else:
            return candidate, None, None, "Path too short"
    except Exception as e:
        return candidate, None, None, str(e)


class DynamicSolutionEvaluator:
    """
    Evaluates optimization solutions by running the complete pipeline:

    1. Generate PathFinder for each active node-predio pair
    2. Merge all paths into a single GPKG file
    3. Run SewerPipeline to design the pipes
    4. Add tanks to SWMM using SWMMModifier
    5. Run SWMM simulation
    6. Read remaining flooding
    """


    def __init__(self,
                 path_proy: Path,
                 elev_files_list: List[str],
                 proj_to: str,
                 work_dir: str = None,
                 ):


        
        self.path_proy = Path(path_proy)
        self.elev_files_list = elev_files_list or []
        self.proj_to = proj_to
        self.inp_file = config.SWMM_FILE
        self.predio_capacity = {}

        # 1. Setup Directories
        self._setup_work_dir(work_dir)
        self.metrics_extractor = MetricExtractor(
                                project_root=config.PROJECT_ROOT,
                                predios_path=config.PREDIOS_FILE)


        # 2. Configuration & Weights
        self.path_weights = config.DEFAULT_PATH_WEIGHTS
        self.road_preferences = config.DEFAULT_ROAD_PREFERENCES
        self._solution_counter = 0
        
        # Track designed pipelines across iterations for tree-based routing
        self.last_designed_gdf = None  # Will store design_gdf from previous iteration
        
        # Predio capacity tracking for cumulative volume validation
        self.predio_tracking = {}  # {predio_id: {'area_total', 'volumen_acumulado', 'ramales'}}
        self.ramal_to_predio = {}  # {ramal_id: predio_id} - maps pipeline branches to their final predio
        self.forbidden_pairs = set()  # {(node_id, target_id)} - combinations that exceeded capacity
        
        # INCREMENTAL PATH GENERATION: Cache paths from previous iterations
        self.cached_path_gdfs = []  # List of path GeoDataFrames from all previous iterations
        self.processed_node_ids = set()  # Node IDs that already have paths generated

        # 3. Initialize Sub-Systems
        self._init_baseline()
        self._init_predio_tracking()  # Initialize predio capacity tracking
        self._init_path_finder()


        self._base_precios_path = str(config.BASE_PRECIOS)
        self._build_topology_from_conduits(clear_existing=True)

        self.last_designed_gdf = None



    def _setup_work_dir(self, work_dir):
        if work_dir is None:
            self.work_dir = self.path_proy / "codigos" / "temp_optimizer"
        else:
            self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _init_baseline(self):
            print(f"  Running Baseline Simulation (Initial State)...")
            self.metrics_extractor.run(config.SWMM_FILE)
            self.baseline_metrics = self.metrics_extractor.metrics
            print(f"  Baseline Flooding: {self.baseline_metrics.total_flooding_volume:,.2f} m3")


            # Run Economic Baseline
            baseline_dir = self.work_dir / "00_Baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            TR_LIST = config.TR_LIST if config.TR_LIST else [config.BASE_INP_TR]
            if len(TR_LIST) == 0: TR_LIST = [config.BASE_INP_TR]

            self.is_probabilistic_mode = len(TR_LIST) > 1
            if self.is_probabilistic_mode:
                print(f"  [Baseline] Probabilistic mode - TRs: {TR_LIST}")
            else:
                print(f"  [Baseline] Deterministic mode - TR: {TR_LIST[0]}")

            runner = AvoidedCostRunner(
                output_base=str(baseline_dir),
                base_precios_path=str(config.BASE_PRECIOS),
                base_inp_path=str(config.SWMM_FILE),
                flood_metrics = self.baseline_metrics
            )

            damage_baseline = runner.run(tr_list=TR_LIST)
            results = damage_baseline.get('results', [])

            if results:
                if self.is_probabilistic_mode:
                    cost_results = self._calculate_probabilistic_baseline(results)
                else:
                    cost_results = self._calculate_deterministic_baseline(results[0])
                    
            self.baseline_metrics.cost = cost_results['total_cost']
            
    def _init_path_finder(self):
        # Calculate combined bounds
        combined_geoms = pd.concat([self.metrics_extractor.nodes_gdf, self.metrics_extractor.predios_gdf])
        min_x, min_y, max_x, max_y = combined_geoms.total_bounds

        # Initialize PathFinder
        northeast_point = (max_x, max_y)
        southwest_point = (min_x, min_y)
        self.shared_path_finder = PathFinder(proj_to=self.proj_to, start_point=southwest_point, end_point=northeast_point)

        # Download/Load OSM Cache
        osm_cache_path = config.OSM_CACHE_PATH
        self.shared_path_finder.download_osm_data(cache_path=str(osm_cache_path))

        # Map Elevations if graph exists
        if self.shared_path_finder.graph:
            nodes_osm, _ = self.shared_path_finder.get_graph_geodataframes()
            if nodes_osm is not None:
                print("Mapping Elevations to OSM Graph...")
                nodes_proj = nodes_osm.to_crs(self.proj_to)
                xy = nodes_proj.geometry.get_coordinates().to_numpy()

                source = ElevationSource(self.path_proy, self.proj_to)
                tree = source.get_elev_source(
                    self.elev_files_list,
                    check_unique_values=False,
                    ellipsoidal2orthometric=False,
                    m_ramales=None,
                    elevation_shift=0
                )
                getter = ElevationGetter(tree=tree, m_ramales=None, threshold_distance=0.7)
                elevations = getter.get_elevation_from_tree_coords(xy)
                self.shared_path_finder.set_node_elevations(nodes_osm, elevations)
        else:
            sys.exit("  Warning: Failed to initialize shared OSM map.")

    def _init_predio_tracking(self):
        """Initialize predio capacity tracking from predios GeoDataFrame."""
        self.predio_tracking = {}
        predios_gdf = self.metrics_extractor.predios_gdf
        for idx, row in predios_gdf.iterrows():
            self.predio_tracking[idx] = {
                'area_total': row.geometry.area,
                'volumen_acumulado': 0.0,
                'ramales': []
            }
        self.ramal_to_predio = {}
        print(f"  [Tracking] Initialized capacity tracking for {len(self.predio_tracking)} predios")



    def _build_topology_from_conduits(self, clear_existing: bool = False):
        """
        Build node topology and static link data directly from conduits DataFrame.
        Leverages pandas for efficiency.
        """
        if clear_existing:
            self.node_topology = {}
            self.static_link_data = {}

        # 1. Static Link Data (Fast Dictionary Conversion)
        # We rely on NetworkExporter to provide: MaxFlow, Length, Geom1, Geom2, Slope, Roughness, Shape, MaxFullFlow
        # We just dump the relevant columns to the dictionary.

        conduits = self.metrics_extractor.swmm_gdf.copy()
        cols_to_keep = ['MaxFlow', 'Length', 'Geom1', 'Geom2', 'Slope', 'Roughness', 'MaxFullFlow', 'Shape']

        # In case some columns are missing (though unlikely with NetworkExporter), strict subsetting
        existing_cols = [c for c in conduits.columns if c in cols_to_keep]
        self.static_link_data.update(conduits[existing_cols].to_dict(orient='index'))

        # 2. Topology (Upstream/Downstream)
        # Downstream: Inlet -> [Links]
        ds_groups = conduits.groupby('InletNode').apply(lambda x: x.index.astype(str).tolist()).to_dict()

        # Upstream: Outlet -> [Links]
        us_groups = conduits.groupby('OutletNode').apply(lambda x: x.index.astype(str).tolist()).to_dict()

        # Merge into node_topology
        all_nodes = set(ds_groups.keys()) | set(us_groups.keys())
        for n in all_nodes:
            self.node_topology.setdefault(n, {'upstream': [], 'downstream': []})
            if n in ds_groups:
                self.node_topology[n]['downstream'].extend(ds_groups[n])
            if n in us_groups:
                self.node_topology[n]['upstream'].extend(us_groups[n])

    def _calculate_probabilistic_baseline(self, results):
        trs = [r['tr'] for r in results]
        probs = [1.0 / tr for tr in trs]
        total_damages = [r['total_impact_usd'] for r in results]
        flood_damages = [r['flood_damage_usd'] for r in results]
        investment_costs = [r['investment_cost_usd'] for r in results]

        ead_total = 0
        ead_flood = 0
        ead_investment = 0
        for i in range(len(trs) - 1):
            dp = probs[i] - probs[i+1]
            ead_total += dp * (total_damages[i] + total_damages[i+1]) / 2
            ead_flood += dp * (flood_damages[i] + flood_damages[i+1]) / 2
            ead_investment += dp * (investment_costs[i] + investment_costs[i+1]) / 2

        self.baseline_ead_total = ead_total
        self.baseline_ead_flood = ead_flood
        self.baseline_ead_investment = ead_investment
        self.baseline_infra_cost = ead_investment
        self.baseline_flood_damage = ead_flood

        print(f"\n  {'='*60}")
        print(f"  BASELINE EAD (Expected Annual Damage)")
        print(f"  {'='*60}")
        print(f"  EAD Flood Damage:      ${ead_flood:,.2f}/año")
        print(f"  EAD Infrastructure:    ${ead_investment:,.2f}/año")
        print(f"  EAD TOTAL:             ${ead_total:,.2f}/año")
        print(f"  {'='*60}\n")
        
           # Retornar diccionario con resultados
        return {
            'total_cost': ead_total,
            'flood_damage_cost': ead_flood,
            'infrastructure_repair_cost': ead_investment,
        }
        
    def _calculate_deterministic_baseline(self, result):
        self.baseline_infra_cost = result['investment_cost_usd']
        self.baseline_flood_damage = result['flood_damage_usd']
        self.baseline_ead_investment = self.baseline_infra_cost
        

        print(f"\n  {'='*60}")
        print(f"  BASELINE SINGLE TR{result['tr']} COST")
        print(f"  {'='*60}")
        print(f"  Flood Damage:          ${self.baseline_flood_damage:,.2f}")
        print(f"  Infrastructure:        ${self.baseline_infra_cost:,.2f}")
        print(f"  {'='*60}\n")
        
            # Retornar diccionario con resultados
        return {
            'infrastructure_repair_cost': self.baseline_infra_cost,
            'flood_damage_cost': self.baseline_flood_damage,
            'total_cost': self.baseline_ead_investment,
        }


    def _get_predio_occupancy_ratio(self, predio_id) -> float:
        """Calculate current occupancy ratio for a predio (0.0 to 1.0+)."""
        if predio_id not in self.predio_tracking:
            return 0.0
        tracking = self.predio_tracking[predio_id]
        if tracking['area_total'] <= 0:
            return 1.0  # Treat zero-area as fully occupied
        # Calculate max volume that fits in area
        max_volume = (tracking['area_total'] * config.TANK_DEPTH_M)
        if max_volume <= 0:
            return 1.0
        return tracking['volumen_acumulado'] / max_volume

    def _check_predio_has_capacity(self, predio_id, additional_volume: float) -> tuple:
        """
        Check if a predio has capacity for additional volume.
        
        Args:
            predio_id: ID of the predio to check
            additional_volume: Volume to add (m³)
            
        Returns:
            (bool, str): (has_capacity, message)
        """
        if predio_id not in self.predio_tracking:
            return False, f"Predio {predio_id} not found in tracking"
        
        tracking = self.predio_tracking[predio_id]
        volumen_total = tracking['volumen_acumulado'] + additional_volume
        
        # Calculate required area for total volume
        area_requerida = (volumen_total / config.TANK_DEPTH_M) + config.TANK_OCCUPATION_FACTOR
        
        if area_requerida > tracking['area_total']:
            return False, (f"Insufficient area: needs {area_requerida:.0f} m², "
                          f"available {tracking['area_total']:.0f} m²")
        
        return True, "OK"

    def _register_volume_to_predio(self, predio_id, volume: float, ramal_id: str = None):
        """Register volume assigned to a predio and update ramal mapping."""
        if predio_id in self.predio_tracking:
            self.predio_tracking[predio_id]['volumen_acumulado'] += volume
            if ramal_id and ramal_id not in self.predio_tracking[predio_id]['ramales']:
                self.predio_tracking[predio_id]['ramales'].append(ramal_id)
            
            occupancy = self._get_predio_occupancy_ratio(predio_id)
            print(f"  [Tracking] Predio {predio_id}: +{volume:.0f} m³ → "
                  f"Total: {self.predio_tracking[predio_id]['volumen_acumulado']:.0f} m³ "
                  f"({occupancy*100:.1f}% occupancy)")
        
        # Update ramal → predio mapping
        if ramal_id:
            self.ramal_to_predio[ramal_id] = predio_id

    def _create_case_directory(self, solution_name: str) -> Path:
        """Create a simple directory for the specific case."""
        case_dir = self.work_dir / solution_name
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _save_input_gpkg_for_rut03(self,
                                  path_gdfs: List[gpd.GeoDataFrame],
                                  case_dir: Path,
                                  ) -> str:
        """
        Save the paths as a GPKG formatted strictly for rut_03 input.
        """
        if not path_gdfs:
            return None

        # Concatenate paths
        merged_gdf = pd.concat(path_gdfs, ignore_index=True)
        if merged_gdf.crs is None:
            merged_gdf = merged_gdf.set_crs(self.proj_to)

        rut03_rows = []
        for i, row in merged_gdf.iterrows():
            geom = row.geometry
            ramal = row.node_ramal

            swmm_gdf = self.metrics_extractor.swmm_gdf
            node_id = row.get('node_id')
            matching_conduits = swmm_gdf[swmm_gdf['InletNode'] == node_id]
            if matching_conduits.empty:
                matching_conduits = swmm_gdf[swmm_gdf['OutletNode'] == node_id]
            existing_height = np.mean(matching_conduits.Geom1)
            h_min = row.node_max_depth - round(config.CAPACITY_MAX_HD * existing_height, 2)
            
            rut03_rows.append({
                'geometry': geom,
                'Ramal': str(ramal),
                'Tipo': 'pluvial',
                'Material': getattr(config, 'DEFAULT_PIPE_MATERIAL', 'HA'),
                'Seccion': getattr(config, 'DEFAULT_PIPE_SECTION', 'rectangular'),
                'Rugosidad': getattr(config, 'DEFAULT_PIPE_RUGOSITY', 'liso'),
                'Estado': 'nuevo',
                'Fase': '1',
                'Obs': f"tunel|{json.dumps(row[1:].to_dict())}",
                'Pozo_hmin': h_min,
                'cobertura_min': getattr(config, 'DEFAULT_POZO_DEPTH', 8.0),
                'D_min': getattr(config, 'MIN_PIPE_DIAMETER', 1.2)
            })

        input_gdf = gpd.GeoDataFrame(rut03_rows, crs=merged_gdf.crs)

        # Save directly to case directory
        gpkg_path = case_dir / "input_routes.gpkg"
        input_gdf.to_file(str(gpkg_path), driver="GPKG", OVERWRITE='YES')

        return str(gpkg_path)

    @staticmethod
    def _save_case_summary(case_dir: Path, active_pairs: List[CandidatePair], solution_name: str):
        """Save a text file with case details."""
        summary_path = case_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Solution: {solution_name}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Active Tanks: {len(active_pairs)}\n")
            f.write("-" * 40 + "\n")
            f.write("Active Nodes:\n")
            for pair in active_pairs:
                f.write(f"  Node: {pair['node_id']} -> Predio: {pair['predio_id']}\n")
                f.write(f"    Capacidad Diseño: {pair['total_volume']:.2f} m3\n")
                f.write(f"    Caudal Pico: {pair['total_flow']:.4f} m3/s\n")
                f.write(f"    Coords (Derivación): ({pair['node_x']:.2f}, {pair['node_y']:.2f})\n")
                f.write(f"    Coords (Terreno): ({pair['predio_x_centroid']:.2f}, {pair['predio_y_centroid']:.2f})\n")
            f.write("-" * 40 + "\n")








    def is_predio_available(self, predio_id):
        """Check if predio has less than MAX_OCCUPANCY_RATIO occupied."""
        occupancy = self._get_predio_occupancy_ratio(predio_id)
        return occupancy < config.PREDIO_MAX_OCCUPANCY_RATIO

    def _get_ramal_from_node(self, node_id: str) -> str:
        """
        Finds the Ramal ID that a specific pipeline node belongs to.
        Looks up the node in last_designed_gdf.
        """
        if self.last_designed_gdf is None or self.last_designed_gdf.empty:
            return None
            
        # Assuming 'Pozo' column contains the Node IDs
        match = self.last_designed_gdf[self.last_designed_gdf['Pozo'].astype(str) == str(node_id)]
        if match.empty:
            return None
            
        return str(match.iloc[0]['Ramal'])

    def _get_target_info_from_ramal(self, ramal_id: str) -> dict:
        """
        Retrieves the target (destination) for a given Ramal ID.
        Searches in the cumulative_paths to find the defining path for this ramal.
        """
        if not hasattr(self, 'cumulative_paths'):
            return None

        # Iterate defined paths to find which one created this ramal
        # path_gdf['node_ramal'] stores the ramal ID assigned to that path
        for source_node, path_gdf in self.cumulative_paths.items():
            if not path_gdf.empty and str(path_gdf['node_ramal'].iloc[0]) == str(ramal_id):
                # Columns 'target_id' and 'target_type' are created in _generate_paths
                return {
                    'target_id': path_gdf['target_id'].iloc[0],
                    'target_type': path_gdf['target_type'].iloc[0]
                }
        return None

    def _trace_downstream_tank(self, start_node_id: str, visited=None) -> str:
        """
        Recursively traces a node downstream through Ramals until a Tank (Predio) is found.
        
        Returns:
            predio_id (str): If a tank is reached.
            None: If the chain is broken, undefined, or forms a cycle.
        """
        if visited is None:
            visited = set()
            
        # Cycle detection
        if start_node_id in visited:
            print(f"  [Cycle] Recursive trace detected loop at node {start_node_id}")
            return None
        visited.add(start_node_id)
        
        # 1. Identify which Ramal this node is on
        ramal_id = self._get_ramal_from_node(start_node_id)
        if ramal_id is None:
            return None
            
        # 2. Find where this Ramal goes (Target)
        target_info = self._get_target_info_from_ramal(ramal_id)
        if not target_info:
            return None
            
        target_type = target_info['target_type']
        target_id = target_info['target_id']
        
        # 3. Base Case: Found a Tank/Predio
        if target_type == 'tank':
            return target_id
            
        # 4. Recursive Case: Found another Node -> Continue tracing
        elif target_type == 'node':
            return self._trace_downstream_tank(target_id, visited)
            
        return None
    
    def is_node_ramal_available(self, node_id):
        """
        Check if the downstream tank available for this node has capacity.
        Uses recursive tracing to find the ultimate tank.
        """
        # Trace downstream to find the real predio/tank
        predio_final = self._trace_downstream_tank(node_id)
        
        if predio_final is None:
            # If we can't trace it to a tank (e.g., broken link), assume available
            # to avoid blocking potentially valid connections (or return False to be strict)
            return True
            
        return self.is_predio_available(predio_final)


    def _find_best_target_by_path_length(self,
                                          source_point: tuple,
                                          source_elevation: float,
                                          source_volume: float = 0.0,
                                          source_node_id: str = None) -> tuple:
        """
        Find the best target (predio or pipeline node) for a flooded node.
        
        Args:
            source_point: (x, y) of the source node (flooded node)
            source_elevation: Elevation of the source node
            source_volume: Volume to drain from this node (m³) - used for capacity validation
        Returns:
            Tuple of (target_x, target_y, target_type, metadata) or None if no valid target
        """
        print(f"  [Target] Source point: ({source_point[0]:.1f}, {source_point[1]:.1f})")
    
        # Initialize node_df as empty DataFrame with correct columns
        node_df = pd.DataFrame(columns=['id', 'type', 'invert_elevation', 'x', 'y'])
        
        # Extract coordinates from last designed pipeline only if it exists
        if self.last_designed_gdf is not None:
            coords_df = self.last_designed_gdf.geometry.get_coordinates()
            node_df = coords_df.iloc[1::2]  # Select every other row starting from index 1
            node_df.columns = ['x', 'y']
            node_df['type'] = 'node'
            node_df['invert_elevation'] = self.last_designed_gdf['ZFF']  # Invert elevation from pipeline end
            node_df['id'] = self.last_designed_gdf['Pozo']  # Node ID from Pozo column
            node_df = node_df[['id', 'type', 'invert_elevation', 'x', 'y']]  # Reorder columns
        
        # Extract centroid coordinates from predios GeoDataFrame
        predios_gdf = self.metrics_extractor.predios_gdf
        predios_df = predios_gdf.geometry.centroid.get_coordinates()
        predios_df.columns = ['x', 'y']
        predios_df['type'] = 'tank'
        predios_df['invert_elevation'] = predios_gdf['z'] - config.TANK_DEPTH_M  # Tank invert elevation
        predios_df['id'] = predios_gdf.index.to_list()  # Use index as ID
        predios_df = predios_df[['id', 'type', 'invert_elevation', 'x', 'y']]  # Reorder columns
        
        # Combine node and predio dataframes into possible targets
        posible_targets_df = pd.concat([node_df, predios_df], ignore_index=True)
        
        # Filter candidates where source elevation is sufficient for gravity flow
        elev_filter = source_elevation >= posible_targets_df['invert_elevation']
        candidates_df = posible_targets_df[elev_filter].copy()
        candidates_df.reset_index(drop=True, inplace=True)
        
        # =====================================================================
        # EARLY OCCUPANCY FILTER: Exclude predios/nodes with >80% occupancy
        # =====================================================================
        
        # Apply occupancy filter
        initial_count = len(candidates_df)
        
        # Filter predios (type == 'tank')
        predio_mask = candidates_df['type'] == 'tank'
        predio_available = candidates_df.loc[predio_mask, 'id'].apply(self.is_predio_available)
        
        # Filter nodes (type == 'node')
        node_mask = candidates_df['type'] == 'node'
        node_available = candidates_df.loc[node_mask, 'id'].apply(self.is_node_ramal_available)
        
        # Combine masks
        available_mask = pd.Series(True, index=candidates_df.index)
        available_mask.loc[predio_mask] = predio_available.values
        available_mask.loc[node_mask] = node_available.values
        
        candidates_df = candidates_df[available_mask]
        candidates_df.reset_index(drop=True, inplace=True)
        
        filtered_count = initial_count - len(candidates_df)
        if filtered_count > 0:
            print(f"  [Occupancy] Filtered {filtered_count} targets with >{config.PREDIO_MAX_OCCUPANCY_RATIO*100:.0f}% occupancy")
        
        # Convert filtered dataframe to list of dictionaries
        candidates = candidates_df.to_dict('records')
    
        
        
        # Use 50% of available cores (minimum 2)
        num_cores = max(2, os.cpu_count() // 2)
        print(f"  [Parallel] Using {num_cores} threads for path calculation (ThreadPoolExecutor)")
        
        # Get graph reference once
        graph = self.shared_path_finder.graph
        
        if len(candidates) == 0:
            print("  [Target] No candidates available after elevation filtering.")
            return None
        
        # Vectorized nearest node search (outside the parallel loop)
        print(f"  [Parallel] Nearest node search for {len(candidates)} candidates...")
        source_node = ox.nearest_nodes(graph, source_point[0], source_point[1])
        candidate_xs = [c['x'] for c in candidates]
        candidate_ys = [c['y'] for c in candidates]
        # osmnx can handle lists of coordinates for vectorized search
        target_nodes = ox.nearest_nodes(graph, candidate_xs, candidate_ys)
    
        results = []
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Create list of tasks with pre-computed source and target nodes
            tasks = [
                (c, graph, source_node, target_nodes[i], self.path_weights, self.road_preferences)
                for i, c in enumerate(candidates)
            ]
            
            # Submit all tasks
            futures = [executor.submit(calculate_path_for_candidate, *task) for task in tasks]
            
            # Collect results as they complete
            for future in as_completed(futures):
                candidate, path_length, linestring, error = future.result()
                results.append((candidate, path_length, linestring, error))
    
    
        valid_results = []
        filtered_gravity = 0
        for candidate, path_length, linestring, error in results:
            if path_length is not None:
                # Target elevation is already stored in 'invert_elevation' in the candidates DataFrame
                target_elev = candidate['invert_elevation']
                
                # Calculate required slope drop (0.4% min)
                slope_drop = path_length * config.MIN_PIPE_SLOPE
                min_source_required = target_elev + slope_drop
                
                if source_elevation >= min_source_required:
                    valid_results.append((candidate, path_length, error))
                else:
                    filtered_gravity += 1
            else:
                # No path found - include for logging
                valid_results.append((candidate, path_length, error))
        
        if filtered_gravity > 0:
            print(f"  [Target] Filtered {filtered_gravity} candidates by gravity (0.4% slope rule)")
        
        # =====================================================================
        # EXACT CAPACITY VALIDATION: Sort by path_length, check capacity
        # =====================================================================
        # Sort valid results by path_length (shortest first)
        sorted_results = sorted(
            [(c, pl, e) for c, pl, e in valid_results if pl is not None],
            key=lambda x: x[1]
        )
        
        if len(sorted_results) == 0:
            raise ValueError(
                f"No valid path found for source point {source_point}. "
                f"All candidates filtered by elevation or gravity constraints."
            )
        
        # Iterate through candidates by path_length, select first with capacity
        for candidate, path_length, error in sorted_results:
            target_type = candidate['type']
            
            # Check if this pair is forbidden (previous overflow)
            if source_node_id:
                target_check_id = str(candidate['id'])
                if (str(source_node_id), target_check_id) in self.forbidden_pairs:
                    print(f"  [Forbidden] Skipping target {target_check_id} for node {source_node_id} (previously failed capacity)")
                    continue

                # If target is a node, check if the ramal's final predio is forbidden
                if target_type == 'node':
                    ramal_id = self._get_ramal_from_node(candidate['id'])
                    # We can't easily check the predio here without looking it up,
                    # but the Forbidden logic stores (node_id, predio_id) or (node_id, node_id).
                    # If we blocked (node_source, predio_final), we should check that too.
                    # For now, relying on direct target checks.
            
            if target_type == 'tank':
                predio_id = candidate['id']
                has_capacity, msg = self._check_predio_has_capacity(predio_id, source_volume)
                
                if has_capacity:
                    print(f"  [Target] SELECTED: predio {predio_id} ({candidate['x'], candidate['y']}) (OSM: {path_length:.1f}m)")
                    return (candidate['x'], candidate['y'], target_type, candidate)
                else:
                    print(f"  [Capacity] Predio {predio_id} skipped: {msg}")
                    
            elif target_type == 'node':
                ramal_id = self._get_ramal_from_node(candidate['id'])
                predio_final = self.ramal_to_predio[ramal_id]  # Always exists for nodes in last_designed_gdf
                
                has_capacity, msg = self._check_predio_has_capacity(predio_final, source_volume)
                
                if has_capacity:
                    print(f"  [Target] *** SELECTED: pipeline node '{candidate['id']}' → Predio {predio_final} (OSM: {path_length:.1f}m) ***")
                    return (candidate['x'], candidate['y'], target_type, candidate)
                else:
                    print(f"  [Capacity] Node {candidate['id']} → Predio {predio_final} skipped: {msg}")
        
        # If we get here, NO candidate had sufficient capacity - this IS a real error
        raise ValueError(
            f"NO PREDIO WITH SUFFICIENT CAPACITY: Checked {len(sorted_results)} candidates "
            f"for source volume {source_volume:.0f} m³, none had enough space. "
            f"Consider adding more predios or reducing tank volumes."
        )


    def check_minimun_tank_size(self) -> list | bool:
        """
        Identifica tanques que no cumplen con el volumen mínimo requerido.
        
        Validación de Viabilidad Técnica:
        ==================================
        - Itera sobre todos los tanques en el sistema simulado
        - Compara el volumen almacenado máximo vs. el mínimo configurable
        - Identifica nodo de origen y predio de destino para cada tanque pequeño
        - Útil para filtrar soluciones inviables antes de análisis económico
        
        Criterio de Rechazo:
        ====================
        Si `max_stored_volume < TANK_MIN_VOLUME_M3`, el tanque es considerado
        demasiado pequeño para ser práctico (costos de construcción, mantenimiento, etc.)
        
        Returns:
            list: Lista de diccionarios con información de TODOS los tanques pequeños:
                  [{'node_id': str, 'predio_id': int, 'target_id': int, 'tank_id': str, 'stored_volume': float}, ...]
            bool: False si todos los tanques cumplen el tamaño mínimo
        
        Example:
            >>> result = evaluator.check_minimun_tank_size()
            >>> if result:  # Hay tanques pequeños (lista no vacía)
            >>>     for tank_info in result:
            >>>         print(f"Tank {tank_info['tank_id']} is undersized: {tank_info['stored_volume']} m³")
        """
        # Obtener datos de tanques desde las métricas de simulación SWMM
        tanks = self.current_metrics.tank_data
        small_tanks = []  # Lista para almacenar todos los tanques problemáticos
        
        # ========================================================================
        # ITERAR SOBRE CADA TANQUE Y VALIDAR VOLUMEN MÍNIMO
        # ========================================================================
        for tank, tank_value in tanks.items():
            # --- PASO 1: Extraer predio_id desde el nombre del tanque ---
            predio_id = int(tank.replace("tank_", ""))
            
            # --- PASO 2: Buscar el node_id que conecta a este predio ---
            node_id = [
                key
                for key, values in self.cumulative_paths.items()
                if (values['target_id'].item()) == predio_id
            ][0]
            
            # --- PASO 3: Obtener volumen almacenado máximo del tanque ---
            stored_volume = tank_value['max_stored_volume']
            
            # --- PASO 4: Validar contra volumen mínimo configurable ---
            if stored_volume < config.TANK_MIN_VOLUME_M3:
                print(f"  [Min Size] Tank {tank} at predio {predio_id} (node {node_id}) has insufficient volume: "
                      f"required {config.TANK_MIN_VOLUME_M3} m³, has {stored_volume:.2f} m³")
                
                # --- PASO 5: Agregar tanque problemático a la lista ---
                small_tanks.append({
                    'node_id': node_id,
                    'predio_id': predio_id,
                    'target_id': predio_id,
                    'tank_id': tank,
                    'stored_volume': stored_volume
                })
        
        # ========================================================================
        # RETORNAR RESULTADO DE VALIDACIÓN
        # ========================================================================
        if len(small_tanks) > 0:
            # Hay tanques pequeños → retornar lista completa
            return small_tanks
        else:
            # Todos los tanques cumplen el mínimo → retornar False
            return False  # ✓ CONSISTENTE CON resize_tanks_based_on_exceedance
    

    def resize_tanks_based_on_exceedance(self,
                                         current_inp_file: str,
                                         active_pairs: list = None,
                                         ) -> list | bool:
        """
        Ajusta iterativamente tanques y tuberías hasta eliminar desbordamientos en el modelo SWMM.
        
        Proceso Iterativo:
        ==================
        1. Ejecuta simulación SWMM y extrae métricas (caudales, volúmenes de tanques, flooding)
        2. Actualiza caudales de diseño con valores reales de la simulación
        3. Re-diseña tuberías con los nuevos caudales (si cambios >5%)
        4. Ajusta tamaño de tanques según volumen excedente (flooding)
        5. Valida capacidad física del predio (área disponible)
        6. Repite hasta que no haya flooding o se alcance el máximo de iteraciones
        
        Args:
            current_inp_file: Ruta al archivo .inp de SWMM a modificar
            active_pairs: Lista de pares nodo-predio activos (para rastrear capacidad)
        
        Returns:
            list: Si se excede capacidad de uno o más predios, retorna lista de overflows:
                  [{'node_id': str, 'predio_id': int, 'target_id': str, 'tank_id': str}, ...]
            bool: False si converge exitosamente (sin problemas)
        
        Raises:
            RuntimeError: Si se alcanzan MAX_RESIZE_ITERATIONS sin convergencia
        """
        # Inicializar herramientas de diseño y modificación
        slll = SeccionLlena()
        modifier = SWMMModifier(current_inp_file)
        
        # Lista para acumular overflows si ocurren múltiples
        overflow_list = []

        
        iteration = 0
        while iteration < config.MAX_RESIZE_ITERATIONS:
            iteration += 1
            run = False  # Flag para detectar si hubo cambios en esta iteración
            
            # ====================================================================
            # PASO 2: Actualizar caudales de diseño con valores de simulación
            # ====================================================================
            old_flow = {}
            
            for node_id, gdf_path in self.cumulative_paths.items():
                ramal = gdf_path['node_ramal'].item()
                filtro_ramal = self.last_designed_gdf['Ramal'] ==  ramal
                length_tramos = len(self.last_designed_gdf[filtro_ramal])
                end_tramo_name = self.last_designed_gdf[filtro_ramal]['Tramo'].to_list()[0].split('-')[1]

                if length_tramos == 1:
                    if gdf_path.target_type.item() == 'tank':
                        end_tramo_name = 'tank_' + str(gdf_path['target_id'].item())

                target_link = f'{node_id}-{end_tramo_name}'
                new_flow = float(self.current_metrics.link_data[target_link]['max_flow'])
                old_flow[ramal] = float(gdf_path['total_flow'])
                gdf_path['total_flow'] = new_flow
            
            # ====================================================================
            # PASO 3: Re-diseñar tuberías con caudales actualizados
            # ====================================================================
            new_path_gdfs = list(self.cumulative_paths.values())
            input_gpkg = self._save_input_gpkg_for_rut03(new_path_gdfs, self.case_dir)
            
            out_gpkg = self._run_sewer_pipeline(
                input_gpkg=input_gpkg,
                new_path_gdfs=new_path_gdfs,
                solution_name=self.solution_name,
                case_dir=self.case_dir,
            )
            
            self.last_designed_gdf = gpd.read_file(out_gpkg).copy()
            
            # ====================================================================
            # PASO 4: Detectar cambios significativos de caudal (>5%)
            # ====================================================================
            for node_id, gdf_path in self.cumulative_paths.items():
                ramal = gdf_path['node_ramal'].item()
                filtro_ramal = self.last_designed_gdf['Ramal'] == ramal
                length_tramos = len(self.last_designed_gdf[filtro_ramal])
                end_tramo_name = self.last_designed_gdf[filtro_ramal]['Tramo'].to_list()[0].split('-')[1]

                if length_tramos == 1:
                    if gdf_path.target_type.item() == 'tank':
                        end_tramo_name = 'tank_' + str(gdf_path['target_id'].item())

                target_link = f'{node_id}-{end_tramo_name}'
                new_flow = float(self.current_metrics.link_data[target_link]['max_flow'])
                
                if abs(new_flow - old_flow[ramal]) > 0.05 * old_flow[ramal]:  # Cambio >5%
                    print(f"  [Resize Iter {iteration}] Significant flow change detected on link {target_link}: "
                          f"old {old_flow[ramal]:.2f} m³/s -> new {new_flow:.2f} m³/s")
                    
                    links = self.last_designed_gdf['Tramo'].to_list()
                    links[0] = target_link
                    shapes = self.last_designed_gdf['Seccion'].map(modifier.seccion_type_map).to_list()
                    geoms = slll.section_str2float(self.last_designed_gdf['D_ext'], return_all=True)
                    geoms[np.isnan(geoms)] = 0.0
                    
                    modifier.modify_xsections(links, shapes, geoms)
                    run = True
            
            # ====================================================================
            # PASO 5: Ajustar tamaño de tanques según volumen excedente
            # ====================================================================
            tanks = self.current_metrics.tank_data
            for tk_id, tk_data in tanks.items():
                exceedance_volume = tk_data['exceedance_volume']
                
                # --- SUB-PASO 5A: EXPANSIÓN DE TANQUE (flooding > 0) ---
                if exceedance_volume > 0:
                    new_vol = tk_data['max_stored_volume'] + exceedance_volume
                    new_area = (new_vol / config.TANK_DEPTH_M) * config.TANK_VOLUME_SAFETY_FACTOR + config.TANK_OCCUPATION_FACTOR
                    
                    predio_id = int(tk_id.replace("tank_", ""))
                    if predio_id in self.predio_tracking:
                        tracking = self.predio_tracking[predio_id]
                        
                        # ¿Cabe el tanque ampliado en el terreno?
                        if new_area > tracking['area_total']:
                            # OVERFLOW: El tanque no cabe → agregar a lista
                            node_id = None
                            target_id = predio_id
                            
                            if active_pairs:
                                for pair in active_pairs:
                                    if pair.get('predio_id') == predio_id:
                                        node_id = pair.get('node_id')
                                        if pair.get('target_type') == 'node':
                                            target_id = pair.get('predio_id')
                                        break
                            
                            print(f"  [Overflow] {tk_id} needs {new_area:.0f} m², "
                                  f"predio {predio_id} only has {tracking['area_total']:.0f} m²")
                            
                            overflow_list.append({
                                'node_id': node_id,
                                'predio_id': predio_id,
                                'target_id': target_id,
                                'tank_id': tk_id
                            })
                            continue
                        
                        self.predio_tracking[predio_id]['volumen_acumulado'] = new_vol
                    
                    modifier.modify_storage_area(tk_id, new_area)
                    modifier.save(current_inp_file)
                    print(f"  [Resize Iter {iteration}] {tk_id}: Increased volume by {exceedance_volume:.1f} m³ to {new_vol:.1f} m³")
                    run = True
                
                # --- SUB-PASO 5B: REDUCCIÓN DE TANQUE ---
                depth_target = config.TANK_DEPTH_M - (config.TANK_VOLUME_SAFETY_FACTOR - 1) * config.TANK_DEPTH_M
                variacion_permitida = depth_target * 0.05
                max_depth = tk_data['max_depth']
                
                if max_depth < depth_target - variacion_permitida:
                    new_area = (tk_data['max_stored_volume'] / config.TANK_DEPTH_M) * config.TANK_VOLUME_SAFETY_FACTOR + config.TANK_OCCUPATION_FACTOR
                    modifier = SWMMModifier(current_inp_file)
                    modifier.modify_storage_area(tk_id, new_area)
                    modifier.save(current_inp_file)
                    print(f"  [Resize Iter {iteration}] {tk_id}: Reduced tank size to control max depth {max_depth:.2f} m")
                    run = True
            
            # ====================================================================
            # VERIFICAR SI HUBO OVERFLOWS EN ESTA ITERACIÓN
            # ====================================================================
            if len(overflow_list) > 0:
                print(f"  [Resize] Detected {len(overflow_list)} overflow(s) - returning for handling")
                return overflow_list
            
            # ====================================================================
            # PASO 6: Verificar convergencia
            # ====================================================================
            if not run:
                # No hubo cambios → sistema converge exitosamente
                print(f"  [Resize] No changes in iteration {iteration} - system converged")
                return False  # ✓ RETORNAR FALSE (consistente con check_minimun_tank_size)
            
            # ====================================================================
            # PASO 7: Re-ejecutar simulación con cambios aplicados
            # ====================================================================
            inp_dir = Path(current_inp_file).parent
            for out_file in inp_dir.glob('*.out'):
                try:
                    os.remove(out_file)
                except OSError:
                    pass
            
            self.metrics_extractor.run(current_inp_file)
            self.current_metrics = self.metrics_extractor.metrics
            
            # ====================================================================
            # PASO 8: Verificar si se eliminó  el flooding
            # ====================================================================
            total_exceedance = sum(tk['exceedance_volume'] for tk in self.current_metrics.tank_data.values())
            if total_exceedance <= 0:
                print(f"  [Resize] All tanks converged after {iteration} iteration(s) - no flooding")
                break
        
        # ========================================================================
        # FINAL: Advertencia si no converge dentro del límite de iteraciones
        # ========================================================================
        if iteration >= config.MAX_RESIZE_ITERATIONS:
            print(f"  [Warning] Max resize iterations ({config.MAX_RESIZE_ITERATIONS}) reached - some tanks may still have exceedance")
        
        # Convergencia exitosa (con o sin advertencia)
        return False  # sin problemas que requieran retry

    def _generate_paths(self,
                       active_pairs: List[CandidatePair],
                       ) -> List[gpd.GeoDataFrame]:
        """
        Generate PathFinder paths for each active pair REUSING the shared graph.
        
        NEW: Creates TREE STRUCTURE by allowing paths to connect to existing
        pipeline nodes instead of going directly to predio, avoiding parallel lines.
        """
        path_gdfs = []
        pairs = []
        for i, pair in enumerate(active_pairs):
            source_node_x, source_node_y, source_z = pair['node_x'], pair['node_y'], pair['invert_elevation']
            source_volume = pair.get('total_volume', 0.0)

            res = self._find_best_target_by_path_length(
                source_point=(source_node_x, source_node_y),
                source_elevation=source_z,
                source_volume=source_volume,
                source_node_id=str(pair.get('node_id', '')),
            )
            
            # _find_best_target_by_path_length now raises ValueError on failure
            # This should never return None, but defensive check per strict rules
            if res is None:
                raise RuntimeError(
                    f"Unexpected None from _find_best_target_by_path_length for pair {i+1}. "
                    f"This indicates a bug - function should raise, not return None."
                )
                
            target_x, target_y, target_type, target_node_metadata = res
            if target_type == 'node':
                print(f"  [Tree] Path {i+1}: Connecting to existing pipeline node instead of predio")
            else:
                print(f"  [Direct] Path {i+1}: Connecting directly to predio")

            target_node_metadata['total_volume'] = pair['total_volume']
            target_node_metadata['ramal'] = str(self._solution_counter)
            target_node_metadata['target_type'] = target_type  # Store for downstream use
            pair['predio_id'] = target_node_metadata['id']
            pair['predio_x_centroid'] = target_node_metadata['x']
            pair['predio_y_centroid'] = target_node_metadata['y']
            pair['predio_ground_z'] = target_node_metadata['invert_elevation']
            pair['is_tank'] = target_node_metadata['type'] == 'tank'
            pair['ramal'] = target_node_metadata['ramal']
            

            
            
            # Register volume to the predio (directly or via ramal)
            ramal_id =pair['ramal']
            if target_type == 'tank':
                pair['predio_area'] = round(self.metrics_extractor.predios_gdf.loc[target_node_metadata['id']].geometry.area, 1)
                # Register volume directly to this predio
                self._register_volume_to_predio(target_node_metadata['id'], source_volume, ramal_id)
            else:
                pair['predio_area'] = 0
                # Node connections always go to existing ramales that have predio mapped
                node_ramal_id = self._get_ramal_from_node(target_node_metadata['id'])
                predio_final = self.ramal_to_predio[node_ramal_id]
                self._register_volume_to_predio(predio_final, source_volume, ramal_id)
                


            # Set endpoints
            self.shared_path_finder.set_start_end_points(
                (source_node_x, source_node_y),
                (target_x, target_y)
            )

            # Find path on existing graph
            best_path = self.shared_path_finder.find_shortest_path_with_elevation(
                **self.path_weights,
                road_preferences=self.road_preferences
            )

            if best_path:
                path_gdf = self.shared_path_finder.get_simplified_path(tolerance=config.TOLERANCE, min_distance=config.MIN_NODE_DISTANCE_M)

                if path_gdf is not None:
                    # Ensure we operate on the geometry column safely
                    if not path_gdf.empty:
                        geom = path_gdf.geometry.iloc[0]

                        # Handle MultiLineString - merge into single LineString
                        if geom.geom_type == 'MultiLineString':

                            geom = linemerge(geom)
                            # If still multi after merge, take longest
                            if geom.geom_type == 'MultiLineString':
                                geom = max(geom.geoms, key=lambda g: g.length)
                            path_gdf.geometry.iloc[0] = geom

                        if geom.geom_type == 'LineString':
                            coords = list(geom.coords)
                            # Force first point to match exact derivation node
                            coords[0] = (source_node_x, source_node_y)
                            # # Force last point to match exact target coordinates
                            coords[-1] = (target_x, target_y)
                            path_gdf.geometry.iloc[0] = LineString(coords)

                    path_gdf['total_flow'] = pair['total_flow']
                    path_gdf['node_id'] = pair['node_id']
                    path_gdf['node_invert_elevation'] = pair['invert_elevation']
                    path_gdf['node_max_depth'] = pair['node_depth']
                    path_gdf['node_ramal'] = str(self._solution_counter)
                    
                    for key in target_node_metadata.keys():
                        path_gdf[f'target_{key}'] = target_node_metadata[key]
                    
                    path_gdfs.append(path_gdf)
                    pairs.append(pair)

            else:
                return [], None, pair  # Failed to find path

        return path_gdfs, target_node_metadata, pairs

    def _run_sewer_pipeline(
            self,
            input_gpkg: str,
            new_path_gdfs: list,
            solution_name: str,
            case_dir: Path,
    ) -> Path:
        """
        Execute SewerPipeline design and locate the output GPKG file.

        Args:
            input_gpkg: Path to input GPKG with routes
            new_path_gdfs: List of GeoDataFrames with path data
            solution_name: Name of the solution/project
            case_dir: Directory for case outputs

        Returns:
            Path: Path to the output GPKG file

        Raises:
            FileNotFoundError: If input GPKG doesn't exist
            RuntimeError: If pipeline fails or output not found
        """
        print("  Running pipeline design...")

        try:
            # Verify input exists
            if not Path(input_gpkg).exists():
                raise FileNotFoundError(f"Input GPKG not found: {input_gpkg}")

            # Build flows dictionary
            flows_dict = {}
            for gdf in new_path_gdfs:
                ramal_name = gdf['node_ramal'].item()
                flows_dict[ramal_name + ".0"] = gdf["total_flow"].iloc[0]

            # Initialize pipeline
            pipeline = SewerPipeline(
                elev_file_path=self.elev_files_list[0],
                vector_file_path=str(input_gpkg),
                project_name=solution_name,
                flows_dict=flows_dict,
                proj_to=str(self.proj_to),
                path_out=str(case_dir),
            )

            # Execute pipeline
            try:
                pipeline.run()
            except SystemExit:
                pass  # Pipeline exits via SystemExit on success
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"SewerPipeline crashed: {e}") from e

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize SewerPipeline: {e}") from e

        # Locate output GPKG
        out_gpkg = case_dir / f"{solution_name}.gpkg"

        if not out_gpkg.exists():
            # Search for alternative GPKG files
            possible = [
                f for f in case_dir.glob("*.gpkg")
                if "input" not in f.name.lower()
            ]
            if possible:
                out_gpkg = possible[0]
                print(f"  [Info] Found output GPKG with different name: {out_gpkg.name}")
            else:
                raise RuntimeError(
                    f"No output GPKG found in {case_dir}. "
                    f"SewerPipeline failed silently."
                )

        return out_gpkg


    def evaluate_solution(
            self,
            active_pairs: List[CandidatePair],
            solution_name: str = None,
            current_metrics: SystemMetrics = None,
            iteration: int = 0,
    ) -> Tuple[float, float]:
        """
        Evaluate a complete solution using SewerPipeline + SWMM.
        
        Pipeline Steps:
        ===============
        1. Initialize: Store metrics and setup case directory
        2. Path Generation: Identify and generate paths for new pairs (sticky paths)
        3. Sewer Design: Run SewerPipeline to design pipes
        4. SWMM Integration: Add derivation to model and run simulation
        5. Tank Sizing: Resize tanks based on exceedance volume
        6. Cost Calculation: Run final SWMM simulation and calculate costs
        
        Returns:
            Tuple[cost, current_inp_file, overflow, small_tanks]
        """
       
        # =====================================================================
        # STEP 1: INITIALIZE - Store metrics and setup
        # =====================================================================
        self.current_metrics = current_metrics
        self.nodes_gdf = current_metrics.nodes_gdf
        self.predios_gdf = current_metrics.predios_gdf
        self._solution_counter = iteration
        self.solution_name = solution_name

        case_dir = self.case_dir = self._create_case_directory(solution_name)

        # Initialize tracking structures (first iteration only)
        if not hasattr(self, 'cached_active_pairs'):
            self.cached_active_pairs = {}  # NodeID -> CandidatePair
        if not hasattr(self, 'cumulative_paths'):
            self.cumulative_paths = {}  # NodeID -> path_gdf

        # =====================================================================
        # STEP 2: PATH GENERATION - Identify new paths (sticky paths reuse)
        # =====================================================================
        new_active_pairs = [
            pair for pair in active_pairs
            if pair['node_id'] not in self.cumulative_paths
        ]
        print(f"  [Sticky] Reusing {len(active_pairs) - len(new_active_pairs)} paths, generating {len(new_active_pairs)} new.")

        # Generate paths for new pairs only
        new_path_gdfs, target_node_metadata, pairs = self._generate_paths(new_active_pairs)

        # # Update active_pairs with generated path data
        # pairs_map = {p['node_id']: p for p in pairs}
        # for active in active_pairs:
        #     n_id = active.get('node_id')
        #     if n_id in pairs_map:
        #         active.update(pairs_map[n_id])

        if not target_node_metadata:
            return None, None

        # Store new paths in cumulative collection
        for gdf in new_path_gdfs:
            if not gdf.empty:
                self.cumulative_paths[gdf['node_id'].iloc[0]] = gdf

        # =====================================================================
        # STEP 3: SEWER DESIGN - Run SewerPipeline
        # =====================================================================
        self._save_case_summary(case_dir, active_pairs, solution_name)

        new_path_gdfs = list(self.cumulative_paths.values())
        input_gpkg = self._save_input_gpkg_for_rut03(new_path_gdfs, case_dir)

        out_gpkg = self._run_sewer_pipeline(
            input_gpkg=input_gpkg,
            new_path_gdfs=new_path_gdfs,
            solution_name=solution_name,
            case_dir=case_dir,
        )

        # Load designed pipeline for tree-based routing in next iteration
        self.last_designed_gdf = gpd.read_file(out_gpkg).copy()

        # Update active_pairs with pipeline design data
        for i, pair in enumerate(active_pairs):
            ramal_str = pair['ramal']
            ramal_data = self.last_designed_gdf[self.last_designed_gdf['Ramal'] == ramal_str]
            if not ramal_data.empty:
                if 'D_ext' in ramal_data.columns:
                    pair['diameter'] = str(ramal_data['D_ext'].iloc[-1])
                if 'L' in ramal_data.columns:
                    pair['pipeline_length'] = float(ramal_data['L'].sum())

        # =====================================================================
        # STEP 4: SWMM INTEGRATION - Add derivation to model
        # =====================================================================
        swmm_modifier = SWMMModifier(inp_file=self.inp_file, crs=config.PROJECT_CRS)
        current_inp_file = swmm_modifier.add_derivation_to_model(self.last_designed_gdf, case_dir, solution_name)

        self.metrics_extractor.run(current_inp_file, self.last_designed_gdf)
        self.current_metrics = self.metrics_extractor.metrics

        # =====================================================================
        # STEP 5: TANK SIZING - Resize based on exceedance volume
        # =====================================================================
        overflow = self.resize_tanks_based_on_exceedance(current_inp_file, active_pairs)
        small_tanks = self.check_minimun_tank_size()

        if overflow or small_tanks:
            print(f"  [Evaluation] Detected overflow or undersized tanks - returning for handling")
            return 0, current_inp_file, overflow, small_tanks

        # =====================================================================
        # STEP 6: COST CALCULATION - Run final SWMM simulation
        # =====================================================================
        print("  Running SWMM simulation...")
        cost, current_inp_file = self._run_cost_simulation(
            vector_path=Path(out_gpkg),
            case_dir=case_dir,
            active_pairs=active_pairs,
            current_inp_file=current_inp_file
        )

        return cost, current_inp_file, overflow, small_tanks

    def _run_cost_simulation(self,
                            vector_path: Path,
                            case_dir: Path,
                            active_pairs: list = None,
                             current_inp_file: Path =  None) -> Tuple[dict, str]:
        """
        Integrate designed pipes into SWMM, run simulation, and calculate all costs.
        
        Cost Structure:
        ===============
        1. Investment Cost (CAPEX):
           - cost_links: Pipeline construction cost
           - cost_tanks: Tank construction cost  
           - cost_land: Land acquisition cost for tanks
           
        2. Residual Cost (Damages with solution):
           - flood_damage_cost: Economic damage from remaining floods
           - infrastructure_repair_cost: Pipe replacement cost for failing pipes
           
        Returns:
            Tuple[dict, str]: (cost_dict, current_inp_file_path)
        """
        active_pairs = active_pairs or []

        
        # =====================================================================
        # STEP 2: CALCULATE PIPELINE CONSTRUCTION COST
        # =====================================================================
        calculator = SewerConstructionCost(
            vector_path=vector_path,
            tipo='pluvial',
            fase=None,  # None = procesa TODAS las fases del GPKG
            domiciliarias_vector_path=None,
            base_precios=config.BASE_PRECIOS,
        )

        excel_path = str(vector_path.with_suffix('.xlsx'))
        cost_links = calculator.run(excel_output_path=excel_path, excel_metadata=config.EXCEL_METADATA)
        
        
        # =====================================================================
        # STEP 3: CALCULATE TANK AND LAND COSTS
        # =====================================================================
        cost_tanks = 0.0
        cost_land = 0.0
        n_tanks = 0
        
        for pair in active_pairs:
            if pair.get('is_tank', False):
                n_tanks += 1
                predio_id = pair['predio_id']
                tank_id = f"tank_{predio_id}"
                
                if tank_id in self.current_metrics.tank_data:
                    tank_data = self.current_metrics.tank_data[tank_id]
                    tank_volume = tank_data['total_volume']
                    tank_depth = tank_data['max_depth']
                    
                    # Store in pair for reporting
                    pair['tank_volume_simulation'] = tank_volume
                    pair['tank_max_depth'] = tank_depth
                    pair['cost_tank'] = CostCalculator.calculate_tank_cost(tank_volume)
                    pair['cost_land'] = (tank_volume / config.TANK_DEPTH_M) * config.LAND_COST_PER_M2
                    
                    # Accumulate totals
                    cost_tanks += pair['cost_tank']
                    cost_land += pair['cost_land']
        
        # =====================================================================
        # STEP 4: CALCULATE RESIDUAL COSTS (DAMAGES WITH SOLUTION IN PLACE)
        # =====================================================================
        runner = AvoidedCostRunner(
            output_base=str(case_dir / "00_avoided_costs"),
            base_precios_path=str(config.BASE_PRECIOS),
            base_inp_path=str(current_inp_file),
            flood_metrics=self.current_metrics
        )

        damage_results = runner.run(tr_list=[config.BASE_INP_TR])
        results = damage_results.get('results', [])
        
        if results:
            if self.is_probabilistic_mode:
                residual_cost = self._calculate_probabilistic_baseline(results)
            else:
                residual_cost = self._calculate_deterministic_baseline(results[0])
        
        # =====================================================================
        # STEP 5: BUILD COST SUMMARY DICTIONARY
        # =====================================================================
        investment_cost = cost_links + cost_tanks + cost_land
        total_residual = residual_cost.get('total_cost', 0.0)
        total_cost = investment_cost + total_residual
        
        cost = {
            # Investment costs (CAPEX)
            'investment': {
                'links': cost_links,
                'tanks': cost_tanks,
                'land': cost_land,
                'total': investment_cost,
            },
            # Residual costs (remaining damages)
            'residual': {
                'flood_damage': residual_cost.get('flood_damage_cost', 0.0),
                'infrastructure_repair': residual_cost.get('infrastructure_repair_cost', 0.0),
                'total': residual_cost.get('flood_damage_cost', 0.0) + residual_cost.get('infrastructure_repair_cost', 0.0),
            },

            # Metadata
            'n_tanks': n_tanks,
        }
        
        # Store in metrics for downstream access
        self.current_metrics.cost = cost
        
        print(f"  [Cost] Investment: ${investment_cost:,.0f} (Links: ${cost_links:,.0f}, Tanks: ${cost_tanks:,.0f}, Land: ${cost_land:,.0f})")
        print(f"  [Cost] Residual: ${total_residual:,.0f} (Flood: ${residual_cost.get('flood_damage_cost', 0.0):,.0f}, Infrastructure: ${residual_cost.get('infrastructure_repair_cost', 0.0):,.0f})")
        print(f"  [Cost] Total: ${total_cost:,.0f}")

        return cost, current_inp_file
    
        
        
        # # --- RUN ITZI FLOOD MODEL (if flood_damage enabled) ---
        # itzi_result = {}
        # if config.COST_COMPONENTS.get('flood_damage', False):
        #     print(f"\n  [Itzi] Running 2D surface flood simulation...")
        #     itzi_dir = case_dir / "02_itzi_2d"
        #     itzi_dir.mkdir(parents=True, exist_ok=True)
        #
        #     itzi_result = run_itzi_for_case(
        #         swmm_file=str(final_inp_path),
        #         output_dir=str(itzi_dir),
        #         verbose=False
        #     )
        #     if itzi_result.get('success'):
        #         print(f"  [Itzi] ✓ 2D simulation complete (Max Depth: {itzi_result.get('max_depth_m', 0):.2f}m)")
        #
        #         # Run CLIMADA on depth raster
        #         if 'max_depth_file' in itzi_result and itzi_result['max_depth_file']:
        #              damage_dir = case_dir / "03_flood_damage"
        #              damage_dir.mkdir(parents=True, exist_ok=True)
        #
        #              # Use CLIMADA flood damage calculation
        #              climada_res = calculate_flood_damage_climada(
        #                  depth_raster_path=itzi_result['max_depth_file'],
        #                  output_gpkg=damage_dir / "flood_damage_results.gpkg",
        #                  output_txt=damage_dir / "flood_damage_report.txt"
        #              )
        #              itzi_result['climada_result'] = climada_res
        #              print(f"  [CLIMADA] Total Damage: ${climada_res.get('total_damage_usd', 0):,.2f}")
        #     else:
        #         raise RuntimeError(f"ITZI simulation failed: {itzi_result}")
        # else:
        #     print(f"\n  [Itzi] SKIPPED (flood_damage=False in config)")
        #
        # # --- CALCULATE ECONOMIC IMPACT ---
        # print("\n  [Economics] Calculating Economic Impact...")
        #
        # # Use CLIMADA results from itzi_result (if available)
        # if itzi_result and 'climada_result' in itzi_result and itzi_result['climada_result']:
        #     climada_res = itzi_result['climada_result']
        #     flood_damage = climada_res.get('total_damage_usd', 0.0)
        #
        #     # Store for optimizer to calculate avoided cost
        #     self.last_flood_damage_usd = flood_damage
        #
        #     # If this is the first CLIMADA run, store as baseline proxy
        #     if not hasattr(self, 'baseline_flood_damage') or self.baseline_flood_damage == 0:
        #         # First run: estimate baseline from current damage + marginal improvement
        #         # This is an approximation - ideally baseline ITZI should run separately
        #         self.baseline_flood_damage = flood_damage
        #         print(f"  [Economics] Baseline CLIMADA Damage (estimated): ${self.baseline_flood_damage:,.2f}")
        #
        #     # Return flooding damage as economic impact
        #     economic_res = {
        #         'net_impact_usd': flood_damage,
        #         'flood_damage_usd': flood_damage,
        #         'details': climada_res
        #     }
        #     print(f"  [Economics] ITZI/CLIMADA Flood Damage: ${flood_damage:,.2f}")
        # else:
        #     # No CLIMADA result - use volume-based approximation or zero
        #     self.last_flood_damage_usd = 0.0
        #     economic_res = {
        #         'net_impact_usd': 0.0,
        #         'flood_damage_usd': 0.0,
        #         'details': {}
        #     }
        #     print(f"  [Economics] Flood Damage: $0.00 (ITZI/CLIMADA disabled)")
        #
        # # --- GENERATE AVOIDED COST BUDGET (if deferred_investment enabled) ---
        # infrastructure_benefit = 0.0
        # solution_infra_cost = 0.0
        #
        # if config.COST_COMPONENTS.get('deferred_investment', False):
        #     from rut_20_avoided_costs import DeferredInvestmentCost
        #     deferred_calc = DeferredInvestmentCost(
        #         base_precios_path=str(config.CODIGOS_DIR / "base_precios.xlsx"),
        #         capacity_threshold=0.9
        #     )
        #     solution_infra_cost = deferred_calc.run(
        #         inp_path=str(final_inp_path),
        #         output_dir=str(case_dir)
        #     )
        #
        #     # Store for NSGA optimizer to access
        #     self.last_solution_infra_cost = solution_infra_cost
        #
        #     # Calculate infrastructure benefit (Baseline - Solution)
        #     if hasattr(self, 'baseline_infra_cost') and self.baseline_infra_cost > 0:
        #         infrastructure_benefit = max(0, self.baseline_infra_cost - solution_infra_cost)
        #         print(f"  [Economics] Infrastructure Benefit: ${infrastructure_benefit:,.2f}")
        #         print(f"    (Baseline: ${self.baseline_infra_cost:,.2f} - Solution: ${solution_infra_cost:,.2f})")
        # else:
        #     print(f"  [Economics] Deferred Investment: SKIPPED (disabled in config)")
        #
        # # Update economic result with infrastructure benefit
        # flood_damage = economic_res.get('flood_damage_usd', 0.0)
        # total_benefit = infrastructure_benefit  # Add other benefits here if enabled
        # net_impact_usd = flood_damage - total_benefit  # Positive = bad, Negative = good
        #
        # economic_res['infrastructure_benefit'] = infrastructure_benefit
        # economic_res['total_benefits'] = total_benefit
        # economic_res['net_impact_usd'] = net_impact_usd
        #
        # # Net Impact (The "Badness" we want to minimize)
        # print(f"  [Economics] Net Economic Impact (Damage - Benefits): ${net_impact_usd:,.2f}")
        # if infrastructure_benefit > 0:
        #     print(f"    - Infrastructure Benefit: ${infrastructure_benefit:,.2f}")
        #
        #
        # print(f"  [Economics] Total Construction Cost: ${total_cost:,.2f}")
        # bc_ratio = total_benefit / total_cost if total_cost > 0 else 0
        # print(f"  [Economics] B/C Ratio:               {bc_ratio:.2f}")
        
        # Store economic result for optimizer to access
        # self.last_economic_result = economic_res
        
        # Store final INP path for next iteration's re-ranking
