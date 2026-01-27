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
from rut_17_comparison_reporter import ScenarioComparator
from rut_18_itzi_flood_model import run_itzi_for_case
from rut_19_flood_damage_climada import calculate_flood_damage_climada
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
        
        # INCREMENTAL PATH GENERATION: Cache paths from previous iterations
        self.cached_path_gdfs = []  # List of path GeoDataFrames from all previous iterations
        self.processed_node_ids = set()  # Node IDs that already have paths generated

        # 3. Initialize Sub-Systems
        self._init_baseline()
        self._init_path_finder()


        self._base_precios_path = str(config.BASE_PRECIOS)
        self._build_topology_from_conduits(clear_existing=True)



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



    def _find_best_target_by_path_length(self,
                                          source_point: tuple,
                                          source_elevation: float,) -> tuple:
        """
        
        Args:
            source_point: (x, y) of the source node (flooded node)
            source_elevation: Elevation of the source node
        Returns:
            Tuple of (target_x, target_y, is_intermediate, metadata)
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
        predios_df = self.predios_gdf.geometry.centroid.get_coordinates()
        predios_df.columns = ['x', 'y']
        predios_df['type'] = 'tank'
        predios_df['invert_elevation'] = self.predios_gdf['z'] - config.TANK_DEPTH_M  # Tank invert elevation
        predios_df['id'] = self.predios_gdf.index.to_list()  # Use index as ID
        predios_df = predios_df[['id', 'type', 'invert_elevation', 'x', 'y']]  # Reorder columns
        
        # Combine node and predio dataframes into possible targets
        posible_targets_df = pd.concat([node_df, predios_df], ignore_index=True)
        
        # Filter candidates where source elevation is sufficient for gravity flow
        elev_filter = source_elevation >= posible_targets_df['invert_elevation']
        candidates_df = posible_targets_df[elev_filter]
        candidates_df.reset_index(drop=True, inplace=True)
        
        #filter for available area in predios
        cap = self.predio_capacity.get(cand['predio_id'], 0)
        # available_area = cap['total_area'] - cap['used_area']
        # required_area = (cand['total_volume'] / config.TANK_DEPTH_M) * config.TANK_OCCUPATION_FACTOR
        #
        # if required_area * 0.9 >= available_area:
        #     # print(f"  [Skip] Node {cand['node_id']} lacks space in Predio {cand['predio_id']}")
        #     continue
        
        
        
        
        
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
        
        # Find best result (shortest path) from valid candidates
        best_path_length = float('inf')
        best_target = None
    
        for candidate, path_length, error in valid_results:
            if path_length is not None:
                is_better = path_length < best_path_length
                prefix = "  -> BEST!" if is_better else ""
                # print(f"    [OSM] {candidate['id']}: {path_length:.1f}m{prefix}")
                
                if is_better:
                    best_path_length = path_length
                    best_target = (
                        candidate['x'],
                        candidate['y'],
                        candidate['type'],
                        candidate # The candidate dictionary itself as metadata
                    )
            else:
                print(f"    [OSM] {candidate['id']}: {error or 'No path found'}")
        
        if best_target:
            if best_target[2] == 'node':
                print(f"  [Target] *** SELECTED: pipeline node '{best_target[3]['id']}' (OSM: {best_path_length:.1f}m) ***")
            else:
                print(f"  [Target] *** SELECTED: predio (OSM: {best_path_length:.1f}m) ***")
        
        return best_target
    
    def _generate_paths(self,
                       active_pairs: List[CandidatePair],
                       ) -> List[gpd.GeoDataFrame]:
        """
        Generate PathFinder paths for each active pair REUSING the shared graph.
        
        NEW: Creates TREE STRUCTURE by allowing paths to connect to existing 
        pipeline nodes instead of going directly to predio, avoiding parallel lines.
        """
        path_gdfs = []

        for i, pair in enumerate(active_pairs):
            source_node_x, source_node_y, source_z = pair['node_x'], pair['node_y'], pair['invert_elevation']

            res = self._find_best_target_by_path_length(
                source_point=(source_node_x, source_node_y),
                source_elevation=source_z,
            )
            
            if res:
                target_x, target_y, target_type, target_node_metadata = res
                if target_type == 'node':
                    print(f"  [Tree] Path {i+1}: Connecting to existing pipeline node instead of predio")
                else:
                    print(f"  [Direct] Path {i+1}: Connecting directly to predio")

            target_node_metadata['total_volume'] = pair['total_volume']
            target_node_metadata['ramal'] = self._solution_counter - 1
            pair['predio_id'] = target_node_metadata['id']
            pair['predio_x_centroid'] = target_node_metadata['x']
            pair['predio_y_centroid'] = target_node_metadata['y']
            pair['predio_ground_z'] = target_node_metadata['invert_elevation']
            
            if target_type == 'tank':
                pair['predio_area'] = round(self.predios_gdf.loc[target_node_metadata['id']].geometry.area, 1)
            else:
                pair['predio_area'] = 0
                
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
                path_gdf = self.shared_path_finder.get_simplified_path(tolerance=config.TOLERANCE)

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
                    path_gdf['node_ramal'] = self._solution_counter - 1
                    
                    for key in target_node_metadata.keys():
                        path_gdf[f'target_{key}'] = target_node_metadata[key]
                    
                    path_gdfs.append(path_gdf)

            else:
                return [], None, pair  # Failed to find path

        return path_gdfs, target_node_metadata, pair

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
        input_gdf.to_file(str(gpkg_path), driver="GPKG")

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

    def evaluate_solution(self,
                          active_pairs: List[CandidatePair],
                          solution_name: str = None,
                          current_metrics: SystemMetrics = None,
                          ) -> Tuple[float, float]:

        """
        Evaluate a complete solution using SewerPipeline + SWMM.
        """
        # 1. Store current metrics
        self.current_metrics = current_metrics
        self.nodes_gdf = current_metrics.nodes_gdf
        self.predios_gdf = current_metrics.predios_gdf
        

        # 2. Setup Case Directory
        case_dir = self.current_case_dir = self._create_case_directory(solution_name)

        # Initialize cached_active_pairs if not exists
        if not hasattr(self, 'cached_active_pairs'):
            self.cached_active_pairs = {}  # NodeID -> CandidatePair

        # # If this is NOT the first iteration, re-calculate flooding metrics
        # # using the INP from the previous iteration to get updated priorities.
        # if self._solution_counter > 0:
        #     print(f"  [Re-Rank] Iteration {self._solution_counter + 1}: Recalculating flooding with {Path(self.last_solution_inp_path).name}...")
        #     metrics_runner = FloodingMetrics(
        #         inp_file_path=str(self.last_solution_inp_path),
        #         risk_file_path=config.FAILIURE_RISK_FILE
        #     )
        #     # CRITICAL FIX: Do NOT replace self.nodes_gdf entirely, as its indices
        #     # must remain stable for the optimizer's 'assignments' mapping.
        #     updated_metrics_gdf = metrics_runner.run()
        #
        #     # Reset current metrics but PRESERVE order/length
        #     self.nodes_gdf['total_volume'] = 0.0
        #     self.nodes_gdf['total_flow'] = 0.0
        #     if 'NodeDepth' in self.nodes_gdf.columns: self.nodes_gdf['NodeDepth'] = 0.0
        #
        #     # Map updated metrics back to baseline nodes
        #     if not updated_metrics_gdf.empty:
        #         new_map = updated_metrics_gdf.set_index('NodeID')
        #         for node_id in new_map.index:
        #             if node_id in self.nodes_gdf['NodeID'].values:
        #                 stable_idx = self.nodes_gdf.index[self.nodes_gdf['NodeID'] == node_id][0]
        #                 self.nodes_gdf.at[stable_idx, 'total_volume'] = new_map.at[node_id, 'total_volume']
        #                 self.nodes_gdf.at[stable_idx, 'total_flow'] = new_map.at[node_id, 'total_flow']
        #                 if 'NodeDepth' in new_map.columns:
        #                     self.nodes_gdf.at[stable_idx, 'NodeDepth'] = new_map.at[node_id, 'NodeDepth']
        #
        #     # Re-initialize the selector with updated flooding data (for pruning/ranking)
        #     self.candidate_selector = WeightedCandidateSelector(self.nodes_gdf, self.predios_gdf)
        #     print(f"  [Re-Rank] Candidate selector updated with stable indices.")
        
        # 2. Increment counter and setup persistence
        self._solution_counter += 1
        if not hasattr(self, 'cumulative_paths'):
            self.cumulative_paths = {}  # NodeID -> path_gdf

        # 3. Sticky Paths: Only generate NEW paths, reuse existing ones
        new_active_pairs = []
        for pair in active_pairs:
            if pair['node_id'] not in self.cumulative_paths:
                new_active_pairs.append(pair)
        
        print(f"  [Sticky] Reusing {len(active_pairs) - len(new_active_pairs)} paths, generating {len(new_active_pairs)} new.")

        # 4. Generate Paths for NEW pairs only
        # They will try to connect to self.last_designed_gdf (previous iterations)
        new_path_gdfs, target_node_metadata, pair = self._generate_paths(new_active_pairs)
        if not target_node_metadata:
            return None, None
        
        for gdf in new_path_gdfs:
            if not gdf.empty:
                self.cumulative_paths[gdf['node_id'].iloc[0]] = gdf

        active_pairs[-1] = pair  # Update last pair with target metadata
        
        # cap = self.predio_capacity.get(cand['predio_id'], 0)
        # available_area = cap['total_area'] - cap['used_area']
        # required_area = (cand['total_volume'] / config.TANK_DEPTH_M) * config.TANK_OCCUPATION_FACTOR
        #
        # if required_area * 0.9 >= available_area:
        #     # print(f"  [Skip] Node {cand['node_id']} lacks space in Predio {cand['predio_id']}")
        #     continue
        #
        
        
        # Save summary immediately 
        self._save_case_summary(case_dir, active_pairs, solution_name)

        # 6. Save Input GPKG for rut_03 (Sewer Design)
        new_path_gdfs = list(self.cumulative_paths.values())
        input_gpkg = self._save_input_gpkg_for_rut03(new_path_gdfs, case_dir)

        # 5. Run SewerPipeline (rut_03)
        print(f"  Running pipeline design...")
        try:
            # Explicitly verify input exists
            if not Path(input_gpkg).exists():
                 raise FileNotFoundError(f"Input GPKG not found: {input_gpkg}")


            # Build flows_dict from path_gdfs (which reflects only valid paths)
            flows_dict = {}
            for i, gdf in enumerate(new_path_gdfs):
                ramal_name = str(i)
                flows_dict[ramal_name + '.0'] = gdf['total_flow'].iloc[0]


            pipeline = SewerPipeline(
                elev_file_path=self.elev_files_list[0],
                vector_file_path=str(input_gpkg),
                project_name=solution_name,
                flows_dict=flows_dict,
                proj_to=str(self.proj_to),
                path_out=str(case_dir)
            )

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

        # 6. Read Designed Output
       
        out_gpkg = case_dir / f"{solution_name}.gpkg"

        if not out_gpkg.exists():
            # Try finding ANY gpkg that isn't the input or 'input_routes'
            possible = [f for f in case_dir.glob("*.gpkg") if "input" not in f.name.lower()]
            if possible:
                out_gpkg = possible[0]
                print(f"  [Info] Found output GPKG with different name: {out_gpkg.name}")
            else:
                raise RuntimeError(f"No output GPKG found in {case_dir}. SewerPipeline failed silently.")


        # Store for next iteration's tree-based path generation
        self.last_designed_gdf = gpd.read_file(out_gpkg).copy()
        
        # === UPDATE active_pairs with pipeline design data ===
        for i, pair in enumerate(active_pairs):
            ramal_str = str(i)
            ramal_data = self.last_designed_gdf[self.last_designed_gdf['Ramal'] == ramal_str]
            if not ramal_data.empty:
                # Get diameter (D_ext column, first value - they should be consistent per ramal)
                if 'D_ext' in ramal_data.columns:
                    pair['diameter'] = str(ramal_data['D_ext'].iloc[0])
                # Get total pipeline length (sum of L column)
                if 'L' in ramal_data.columns:
                    pair['pipeline_length'] = float(ramal_data['L'].sum())
                
                if i == target_node_metadata['ramal']:
                    pair['predio_id'] = target_node_metadata['id']
                    pair['predio_x_centroid'] = target_node_metadata['x']
                    pair['predio_y_centroid'] = target_node_metadata['y']
                    pair['is_tank'] = target_node_metadata['type'] == 'tank'
                    

        # 7. Update SWMM Model & Simulate (rut_14)
        print("  Running SWMM simulation...")
        cost, current_inp_file = self._run_swmm_simulation(
            vector_path=Path(out_gpkg),
            solution_name=solution_name,
            case_dir=case_dir
        )

        return cost, current_inp_file

    def _run_swmm_simulation(self,
                            vector_path: Path,
                            solution_name: str,
                            case_dir: Path) -> Tuple[float, float]:
        """
        Integrate designed pipes into SWMM, run sim, calc cost & flooding.
        """

        # --- CONSTRUCTION COST CALCULATION ---
        calculator = SewerConstructionCost(
            vector_path=vector_path,
            tipo='pluvial',
            fase=None,  # None = procesa TODAS las fases del GPKG
            domiciliarias_vector_path=None,
            base_precios=config.BASE_PRECIOS,
        )

        # Excel
        excel_path = str(vector_path.with_suffix('.xlsx'))
        total_cost = calculator.run(excel_output_path=excel_path, excel_metadata=config.EXCEL_METADATA)

        # --- INTEGRATE SOLUTION INTO SWMM MODEL ---
        swmm_modifier = SWMMModifier(inp_file=self.inp_file, crs=config.PROJECT_CRS)
        current_inp_file = swmm_modifier.add_derivation_to_model(self.last_designed_gdf, case_dir, solution_name)

        



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
        
        
        return total_cost, current_inp_file
    
