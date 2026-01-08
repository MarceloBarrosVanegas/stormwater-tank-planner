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
import swmmio
import time
from pathlib import Path
from typing import List, Tuple, Any
from dataclasses import dataclass
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from pyswmm import Simulation

# Add paths for local modules
import config
config.setup_sys_path()


from rut_00_path_finder import PathFinder
from rut_02_elevation import ElevationGetter, ElevationSource
from rut_03_run_sewer_design import SewerPipeline
from rut_06_pipe_sizing import SeccionLlena
from rut_13_cost_functions import CostCalculator
from rut_14_swmm_modifier import SWMMModifier
from rut_17_comparison_reporter import MetricExtractor, ScenarioComparator, SystemMetrics
from rut_18_itzi_flood_model import run_itzi_for_case
from rut_19_flood_damage_climada import calculate_flood_damage_climada
from rut_20_avoided_costs import AvoidedCostRunner
from rut_25_from_inp_to_vector import NetworkExporter


@dataclass
class ActivePair:
    """Represents an active node-predio assignment in a solution."""
    node_idx: int
    predio_idx: int
    volume: float
    node_id: str
    predio_id: int
    node_x: float
    node_y: float
    predio_x: float
    predio_y: float
    predio_z: float
    flooding_flow: float
    flooding_flow: float
    node_invert_elev: float = 0.0
    node_depth: float = 0.0
    geometry: Any = None  # Stores the detailed shapely LineString of the route

# SimulationMetrics removed - using rut_17.SystemMetrics

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
                 nodes_gdf: gpd.GeoDataFrame,
                 predios_gdf: gpd.GeoDataFrame,
                 inp_file: str,
                 path_proy: Path,
                 elev_files_list: List[str],
                 proj_to: str,
                 work_dir: str = None,
                 path_weights: dict = None,
                 road_preferences: dict = None,
):
        
        self.nodes_gdf = nodes_gdf.copy()
        self.predios_gdf = predios_gdf.copy()
        self.inp_file = inp_file
        self.path_proy = Path(path_proy)
        self.elev_files_list = elev_files_list
        self.proj_to = proj_to

        
        # Working directory for temporary projects
        if work_dir is None:
            self.work_dir = self.path_proy / "codigos" / "temp_optimizer"
        else:
            self.work_dir = Path(work_dir)
        
        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Path finding weights
        self.path_weights = config.DEFAULT_PATH_WEIGHTS
        self.road_preferences = config.DEFAULT_ROAD_PREFERENCES
        self.elev_files_list = elev_files_list or []

        # Calculate combined bounds from both GeoDataFrames
        combined_geoms = pd.concat([self.nodes_gdf.geometry, self.predios_gdf.geometry])
        min_x, min_y, max_x, max_y = combined_geoms.total_bounds

        # Extract specific points requested
        northeast_point = (max_x, max_y)
        southwest_point = (min_x, min_y)
        self.shared_path_finder = PathFinder(proj_to=self.proj_to, start_point=southwest_point, end_point=northeast_point)

        # Use persistent cache for OSM data
        osm_cache_path = config.OSM_CACHE_PATH
        print(f"  [Init] Using OSM Cache Path: {osm_cache_path}")
        self.shared_path_finder.download_osm_data(cache_path=str(osm_cache_path))
        
        if self.shared_path_finder.graph:
            nodes_osm, _ = self.shared_path_finder.get_graph_geodataframes()
            if nodes_osm is not None:
                # Pre-calculate elevations ONCE
                print("  Mapping Elevations to OSM Graph...")

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
            print("  Warning: Failed to initialize shared OSM map.")
            self.shared_path_finder = None
            
        # Counter for unique solution IDs
        self._solution_counter = 0

        # --- ECONOMIC MODEL ---
        # ITZI is always used - no fallback to volume-based mode
        base_precios_path = str(config.CODIGOS_DIR / "base_precios.xlsx")
        print("  [Economics] Using Deferred Investment and Flood Damage avoided cost.")
        self.economic_evaluator = None  # Will use AvoidedCostRunner per-case
        self._base_precios_path = base_precios_path


        # --- METRICS & COMPARATOR ---
        self.metric_extractor = MetricExtractor(flooding_cost_per_m3=config.FLOODING_COST_PER_M3)
        # Load existing network topology for link analysis
        self.node_topology = {} # NodeID -> {'upstream_links': [], 'downstream_links': []}
        self.static_link_data = {} # LinkID -> {'q_full': float, 'length': float, 'geom1': float}

        inp_file = str(self.inp_file)
        output_file = str(config.CODIGOS_DIR / "base_network.gpkg")

        exporter = NetworkExporter(inp_file)
        conduits = exporter.run(output_file, crs=config.PROJECT_CRS)
        self._build_topology_from_conduits(conduits, clear_existing=True)

        # Initialize Metrics
        if self._solution_counter == 0 or not hasattr(self, 'baseline_metrics'):
            print(f"  Running Baseline Simulation (Initial State)...")
            self.baseline_metrics = self._run_baseline()

            if hasattr(self.baseline_metrics, 'link_capacity_data'):
                for lid, static_info in self.static_link_data.items():
                     if lid in self.baseline_metrics.link_capacity_data:
                          # Merge
                          self.baseline_metrics.link_capacity_data[lid].update(static_info)
                     else:
                          # Add if missing (though q_peak would be missing too)
                          pass

            print(f"  Baseline Flooding: {self.baseline_metrics.total_flooding_volume:,.2f} m3")
            self.baseline_damage = 0.0  # Not meaningful without Itzi
            print(f"  [Economics Mode: REAL] Baseline damage will be calculated per-solution using Itzi/CLIMADA.")
            print(f"  [Note] Baseline volume stored for reduction comparison.")

            
            # Setup Comparator
            self.comparator = ScenarioComparator(self.baseline_metrics)


            baseline_dir = self.work_dir / "00_Baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            TR_LIST = [1, 2, 5, 10, 25, 50, 100]

            # 3. Instantiate Runner

            runner = AvoidedCostRunner(
                output_base=str(baseline_dir),
                base_precios_path=str(config.BASE_PRECIOS),
                base_inp_path=str(config.SWMM_FILE)
            )

            # 4. Run Analysis
            damage_baseline = runner.run(tr_list=TR_LIST)
            
            # Extract and display baseline EAD
            results = damage_baseline.get('results', [])
            if results:
                # Calculate total EAD using trapezoidal integration
                trs = [r['tr'] for r in results]
                probs = [1.0 / tr for tr in trs]
                total_damages = [r['total_impact_usd'] for r in results]
                flood_damages = [r['flood_damage_usd'] for r in results]
                investment_costs = [r['investment_cost_usd'] for r in results]
                
                # EAD = Area under the risk curve
                ead_total = 0
                ead_flood = 0
                ead_investment = 0
                for i in range(len(trs) - 1):
                    dp = probs[i] - probs[i+1]
                    ead_total += dp * (total_damages[i] + total_damages[i+1]) / 2
                    ead_flood += dp * (flood_damages[i] + flood_damages[i+1]) / 2
                    ead_investment += dp * (investment_costs[i] + investment_costs[i+1]) / 2
                
                # Store for optimization comparison
                self.baseline_ead_total = ead_total
                self.baseline_ead_flood = ead_flood
                self.baseline_ead_investment = ead_investment
                self.baseline_flood_damage = ead_flood  # For optimizer compatibility
                
                print(f"\n  {'='*60}")
                print(f"  BASELINE EAD (Expected Annual Damage)")
                print(f"  {'='*60}")
                print(f"  EAD Flood Damage:      ${ead_flood:,.2f}/año")
                print(f"  EAD Infrastructure:    ${ead_investment:,.2f}/año")
                print(f"  EAD TOTAL:             ${ead_total:,.2f}/año")
                print(f"  {'='*60}\n")

    # python
    def _build_topology_from_conduits(self, conduits: pd.DataFrame, clear_existing: bool = False):
        """
        Build/merge node topology and static link data from a conduits DataFrame.

        Parameters
        ----------
        conduits : pd.DataFrame
            DataFrame with at least 'InletNode' and 'OutletNode' columns (index used as link id).
        clear_existing : bool
            If True, reset self.node_topology and self.static_link_data before merging.

        Returns
        -------
        tuple
            (node_topology, static_link_data) updated on self and also returned.
        """
        if clear_existing:
            self.node_topology = {}
            self.static_link_data = {}

        def _to_float(val, default=0.0):
            try:
                return float(val) if pd.notna(val) else default
            except Exception:
                return default

        for idx, row in conduits.iterrows():
            link_id = str(idx)
            inlet = str(row.get('InletNode', '')).strip()
            outlet = str(row.get('OutletNode', '')).strip()

            if inlet:
                self.node_topology.setdefault(inlet, {'upstream': [], 'downstream': []})
                self.node_topology[inlet]['downstream'].append(link_id)

            if outlet:
                self.node_topology.setdefault(outlet, {'upstream': [], 'downstream': []})
                self.node_topology[outlet]['upstream'].append(link_id)

            l_val = _to_float(row.get('Length', 0.0))
            g1_val = _to_float(row.get('Geom1', 0.0))
            flow_max = _to_float(row.get('MaxQ', 0.0))

            self.static_link_data[link_id] = {
                'q_full': flow_max,
                'length': l_val,
                'geom1': g1_val
            }

    def _run_baseline(self) -> SystemMetrics:
        """Runs the base INP file once to establish baseline metrics."""
        # FIX: Ensure baseline uses a distinct output file to avoid overlap with solution runs
        base_out = str(Path(self.inp_file).with_name(f"baseline_{Path(self.inp_file).stem}.out"))

        # Identify all potential interest links (upstream/downstream of ANY candidate)
        potential_links = []
        if hasattr(self, 'nodes_gdf') and not self.nodes_gdf.empty:
            for nid in self.nodes_gdf['NodeID'].astype(str):
                if nid in self.node_topology:
                    potential_links.extend(self.node_topology[nid].get('upstream', []))
                    potential_links.extend(self.node_topology[nid].get('downstream', []))

        potential_links = list(set(potential_links))
        if potential_links:
            print(f"  [Info] Baseline: Extracting hydrographs for {len(potential_links)} potential links.")

        return self.metric_extractor.extract(base_out, target_links=potential_links)

    def _parse_dimension_to_mm(self, d_val) -> float:
        """
        Parses a dimension value (float, string, or 'WxH' string) to millimeters.
        Returns float (mm).
        """
        if pd.isna(d_val):
            return 200.0
            
        # Try using SeccionLlena if available
        if SeccionLlena is not None:
            try:
                # section_str2float returns 1D array of relevant vertical dimension 
                # (Diameter for circular, Height for others) by default.
                val_arr = SeccionLlena.section_str2float([str(d_val)])
                if len(val_arr) > 0:
                    return float(val_arr[0]) * 1000.0
            except Exception as e:
                print(f"Error parsing dimension '{d_val}': {e}")
                sys.exit('-----error parsing dimension----')
                
        # Default safety if SeccionLlena is not loaded (should not happen based on user input)
        try:
             return float(d_val) * 1000.0
        except:
             return 200.0

    def _get_pipe_vertical_dim(self, link_id: str) -> float:
        """
        Get the vertical dimension (diameter or height) of a pipe in meters.
        Returns the vertical dimension, not the full geometry.
        """
        try:
            links = self.swmm_model.links()
            if link_id in links.index:
                row = links.loc[link_id]
                geom1 = row.get('Geom1', 0.0)
                shape = str(row.get('Shape', 'CIRCULAR')).upper()
                
                # For circular pipes, Geom1 is diameter
                # For rectangular, Geom1 is height
                if pd.notna(geom1) and geom1 > 0:
                    return float(geom1)
        except Exception as e:
            pass
        return 0.5  # Default 0.5m if not found
    
    def _get_node_min_pipe_depth(self, node_id: str) -> float:
        """
        Get the minimum vertical dimension of pipes connected to a node.
        Checks both inlet (upstream) and outlet (downstream) pipes.
        Returns the minimum 0.7 * vertical_dim of all connected pipes.
        
        This is used to set the weir crest height - the weir should activate
        when water exceeds 70% of the pipe capacity.
        """
        try:
            import swmmio
            model = swmmio.Model(str(self.inp_file))  # Convert Path to string
            links = model.links.dataframe
            xsections = model.inp.xsections
            
            min_vertical = float('inf')
            
            for link_id, row in links.iterrows():
                inlet = str(row.get('InletNode', ''))
                outlet = str(row.get('OutletNode', ''))
                
                # Check if this link is connected to our node
                if inlet == node_id or outlet == node_id:
                    # Get geometry from xsections
                    if link_id in xsections.index:
                        geom1 = xsections.loc[link_id, 'Geom1']
                        if pd.notna(geom1) and float(geom1) > 0:
                            vertical_dim = float(geom1)
                            if vertical_dim < min_vertical:
                                min_vertical = vertical_dim
            
            if min_vertical == float('inf'):
                return 0.35  # Default: 0.7 * 0.5m = 0.35m
            
            return 0.7 * min_vertical  # Return 0.7 * minimum vertical dimension
            
        except Exception as e:
            print(f"  [Warning] Could not get pipe dimensions for {node_id}: {e}")
            return 0.35  # Default
    
    def _design_weir_hydraulic(self, node_id: str, tank_volume: float) -> dict:
        """
        Design weir based on hydraulic principles.
        
        Crest height = min(0.7*D_inlet, 0.7*D_outlet) 
        Width calculated from weir equation: Q = Cd * L * H^1.5
        
        Returns dict with: crest_height, width, design_flow
        """
        # Get minimum 0.7D from connected pipes
        crest_height = self._get_node_min_pipe_depth(node_id)
        crest_height = max(0.10, min(crest_height, 0.50))  # Clamp 0.10m to 0.50m
        
        # Estimate storm duration to calculate required flow
        STORM_PEAK_DURATION_HOURS = 2.0
        target_flow = tank_volume / (STORM_PEAK_DURATION_HOURS * 3600.0)  # m³/s
        
        # Weir equation: Q = Cd * L * H^1.5
        Cd = 1.84  # Rectangular weir discharge coefficient
        H_design = 0.25  # Design head above crest (m)
        
        # Calculate required weir length
        weir_width = target_flow / (Cd * (H_design ** 1.5))
        weir_width = max(1.0, min(30.0, weir_width))  # Clamp 1m to 30m
        
        return {
            'crest_height': crest_height,
            'width': weir_width,
            'design_flow': target_flow,
            'Cd': Cd
        }




    def _get_active_pairs(self, assignments: List[Tuple[int, float]]) -> List[ActivePair]:
        """
        Extract active node-predio pairs from an assignment list.
        Now allows MULTIPLE nodes per predio based on available area.
        
        Parameters
        ----------
        assignments : list of (predio_1idx, volume)
            Assignment for each node.
            
        Returns
        -------
        list of ActivePair
        """
        # Constants for area calculation
        TANK_DEPTH = 5.0  # meters
        OCCUPATION_FACTOR = 1.5  # Extra space for access, pumps, maneuvering
        
        # Pre-calculate available area for each predio
        predio_capacity = {}
        for idx in range(len(self.predios_gdf)):
            predio = self.predios_gdf.iloc[idx]
            area_m2 = predio.geometry.area if hasattr(predio.geometry, 'area') else 0.0
            predio_capacity[idx] = {
                'total_area': area_m2,
                'used_area': 0.0
            }
        
        # Collect ALL valid assignments (no conflict dropping)
        all_claims = []
        for node_idx, (predio_1idx, volume) in enumerate(assignments):
            if predio_1idx == 0 or volume <= 0:
                continue  # No tank for this node
                
            predio_idx = predio_1idx - 1  # Convert to 0-indexed
            if predio_idx < 0 or predio_idx >= len(self.predios_gdf):
                continue
            
            node = self.nodes_gdf.iloc[node_idx]
            predio = self.predios_gdf.iloc[predio_idx]
            
            node_pt = node.geometry.centroid
            predio_pt = predio.geometry.centroid
            dist = np.sqrt((node_pt.x - predio_pt.x)**2 + (node_pt.y - predio_pt.y)**2)
            
            all_claims.append({
                'node_idx': node_idx,
                'predio_idx': predio_idx,
                'volume': volume,
                'distance': dist
            })
        
        # Sort by VOLUME (largest first) for prioritization
        # This ensures bigger/better tanks get area priority over smaller ones
        all_claims.sort(key=lambda x: x['volume'], reverse=True)

        
        # Accept claims based on area availability
        active_pairs = []
        for claim in all_claims:
            predio_idx = claim['predio_idx']
            volume = claim['volume']
            
            # Calculate required area for this tank
            tank_footprint = volume / TANK_DEPTH  # m²
            required_area = tank_footprint * OCCUPATION_FACTOR
            
            # Check area availability
            cap = predio_capacity[predio_idx]
            available_area = cap['total_area'] - cap['used_area']
            
            if required_area > available_area:
                continue  # Skip - not enough space
            
            # Update area usage
            predio_capacity[predio_idx]['used_area'] += required_area
            
            # Create active pair
            node_idx = claim['node_idx']
            node = self.nodes_gdf.iloc[node_idx]
            predio = self.predios_gdf.iloc[predio_idx]
            
            active_pairs.append(ActivePair(
                node_idx=node_idx,
                predio_idx=predio_idx,
                volume=volume,
                node_id=node.NodeID,
                predio_id=predio.name if hasattr(predio, 'name') else predio_idx,
                node_x=node.geometry.centroid.x,
                node_y=node.geometry.centroid.y,
                predio_x=predio.geometry.centroid.x,
                predio_y=predio.geometry.centroid.y,
                predio_z=float(predio.z) if hasattr(predio, 'z') else 0.0,
                flooding_flow=node.FloodingFlow,
                node_invert_elev=float(node.InvertElevation) if hasattr(node, 'InvertElevation') else 0.0,
                node_depth=float(node.NodeDepth) if hasattr(node, 'NodeDepth') else 0.0
            ))
            
        return active_pairs
        
    def _generate_paths(self, 
                       active_pairs: List[ActivePair],
                       solution_name: str,
                       case_dir: Path) -> List[gpd.GeoDataFrame]:
        """
        Generate PathFinder paths for each active pair REUSING the shared graph.
        Enforces NON-OVERLAPPING paths by penalizing edges used by previous pairs in this solution.
        """
        path_gdfs = []
        
        if not self.shared_path_finder or not self.shared_path_finder.graph:
            print("  [Error] Shared PathFinder not available.")
            return []
            
        if not self.shared_path_finder or not self.shared_path_finder.graph:
            print("  [Error] Shared PathFinder not available.")
            return []
            
        for i, pair in enumerate(active_pairs):
            # print(f"  Generating path {i+1}/{len(active_pairs)}: {pair.node_id} -> P{pair.predio_id}")
            try:
                # Reuse shared instance, just update endpoints
                self.shared_path_finder.set_start_end_points(
                    (pair.node_x, pair.node_y),
                    (pair.predio_x, pair.predio_y)
                )
                
                # Find path on existing graph
                best_path = self.shared_path_finder.find_shortest_path_with_elevation(
                    **self.path_weights,
                    road_preferences=self.road_preferences
                )
                
                if best_path:
                    path_gdf = self.shared_path_finder.get_simplified_path(tolerance=50)
                    
                    if path_gdf is not None:
                        # Ensure we operate on the geometry column safely
                        if not path_gdf.empty:
                            geom = path_gdf.geometry.iloc[0]
                            if geom.geom_type == 'LineString':
                                coords = list(geom.coords)
                                # Force first point key to match exact node
                                coords[0] = (pair.node_x, pair.node_y)
                                path_gdf.geometry.iloc[0] = LineString(coords)

                        path_gdf['NodeID'] = pair.node_id
                        path_gdf['PredioID'] = pair.predio_id
                        path_gdf['Volume'] = pair.volume
                        path_gdf['FloodingFlow'] = pair.flooding_flow
                        path_gdf['NodeInvertElev'] = pair.node_invert_elev
                        path_gdf['NodeDepth'] = pair.node_depth
                        path_gdf['ActivePairIdx'] = i # Store original index for consistent naming
                        path_gdfs.append(path_gdf)
                        
                        # Store detailed geometry in the ActivePair for later visualization
                        if not path_gdf.empty and 'geometry' in path_gdf.columns:
                            active_pairs[i].geometry = path_gdf.geometry.iloc[0]
                            
            except Exception as e:
                print(f"    Error generating path for pair {i+1}: {e}")
                continue
        
        return path_gdfs

    def _create_case_directory(self, solution_name: str) -> Path:
        """Create a simple directory for the specific case."""
        case_dir = self.work_dir / solution_name
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _save_input_gpkg_for_rut03(self, 
                                  active_pairs: List[ActivePair], 
                                  path_gdfs: List[gpd.GeoDataFrame], 
                                  case_dir: Path, 
                                  solution_name: str) -> str:
        """
        Save the paths as a GPKG formatted strictly for rut_03 input.
        """
        if not path_gdfs:
            return None
            
        # Concatenate paths
        merged_gdf = pd.concat(path_gdfs, ignore_index=True)
        if merged_gdf.crs is None:
            merged_gdf = merged_gdf.set_crs(self.proj_to)

        # Create map for NodeID -> NodeDepth (or bottom elevation proxy)
        # Using NodeDepth as Pozo_hmin
        node_depth_map = {}
        if self.nodes_gdf is not None:
             # Check available columns
            depth_col = 'NodeDepth' if 'NodeDepth' in self.nodes_gdf.columns else None
            
            for row in self.nodes_gdf.itertuples():
                 if depth_col:
                     # use getattr to be safe
                     val = getattr(row, depth_col, 1.5)
                     node_depth_map[row.NodeID] = float(val)

        rut03_rows = []
        
        for i, row in merged_gdf.iterrows():
            geom = row.geometry
            if geom.geom_type != 'LineString':
                continue
            
            node_id = row.get('NodeID', f"N{i}")


            # Use preserved index from path generation to handle skipped paths
            ramal_idx = row.get('ActivePairIdx', i)
            ramal_id = str(ramal_idx)
            
            # ramal_id = f"R_{node_id}_{predio_id}"
            
            pozo_start = f"{ramal_id}.0"
            pozo_end = f"{ramal_id}.1"
            tramo = f"{pozo_start}-{pozo_end}"
            
            flow = row.get('FloodingFlow', 0.0)
            
            # Get Pozo_hmin for this node
            h_min = node_depth_map.get(node_id, 0.0)
            
            rut03_rows.append({
                'geometry': geom,
                'Ramal': ramal_id,
                'Tramo': tramo,
                'Tipo': 'pluvial',
                'Material': 'PVC',
                'Seccion': 'circular',
                'Rugosidad': 'liso',
                'Estado': 'nuevo',
                'Fase': '1',
                'Derivacion': 'False',
                'Pozo': pozo_start,
                'Obs': f"Flow={flow}",
                'Pozo_hmin': h_min
            })
            
        input_gdf = gpd.GeoDataFrame(rut03_rows, crs=merged_gdf.crs)
        
        # Save directly to case directory
        gpkg_path = case_dir / "input_routes.gpkg"
        input_gdf.to_file(str(gpkg_path), driver="GPKG")
        
        return str(gpkg_path)

    @staticmethod
    def _save_case_summary(case_dir: Path, active_pairs: List[ActivePair], solution_name: str):
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
                f.write(f"  Node: {pair.node_id} -> Predio: {pair.predio_id} (P{pair.predio_id})\n")
                f.write(f"    Volume: {pair.volume:.2f} m3\n")
                f.write(f"    Flooding Flow: {pair.flooding_flow:.4f} m3/s\n")
                f.write(f"    Coordinates (Node): ({pair.node_x:.2f}, {pair.node_y:.2f})\n")
            f.write("-" * 40 + "\n")

    def evaluate_solution(self, 
                          assignments: List[Tuple[int, float]],
                          run_swmm: bool = True,
                          solution_name: str = None) -> Tuple[float, float]:
        """
        Evaluate a complete solution using SewerPipeline + SWMM.
        """
        # 1. Get Active Pairs
        active_pairs = self._get_active_pairs(assignments)
        if not active_pairs:
            return (0.0, self.nodes_gdf['FloodingVolume'].sum())
            
        # Increment counter only for valid attempts that we will evaluate
        self._solution_counter += 1
        
        # Generate simple sequential solution name if not provided
        if not solution_name:
             solution_name = f"Case_{self._solution_counter:04d}"
        
        # 2. Setup Case Directory
        case_dir = self._create_case_directory(solution_name)
        
        # Save summary immediately to identify the case content
        self._save_case_summary(case_dir, active_pairs, solution_name)
        
        # 3. Generate Paths (PathFinder)
        path_gdfs = self._generate_paths(active_pairs, solution_name, case_dir)
        if not path_gdfs:
            print(f"  [Warning] PathFinder failed. Using fallback.")
            return self._fallback_cost_estimation(active_pairs)
            
        # 4. Save Input GPKG for rut_03
        input_gpkg = self._save_input_gpkg_for_rut03(active_pairs, path_gdfs, case_dir, solution_name)
        
        # 5. Run SewerPipeline (rut_03)
        print(f"  Running pipeline design...")
        try:
            # Explicitly verify input exists
            if not Path(input_gpkg).exists():
                 raise FileNotFoundError(f"Input GPKG not found: {input_gpkg}")

            
            flows_dict = {}
            pozo_hmin_dict = {}
            for i, pair in enumerate(active_pairs):
                ramal_name = str(i)
                flows_dict[ramal_name + '.0'] = pair.flooding_flow
                pozo_hmin_dict[ramal_name +  '.0'] = pair.node_depth + 2.0


            pipeline = SewerPipeline(
                elev_file_path=self.elev_files_list[0],
                vector_file_path=str(input_gpkg),
                project_name=solution_name,
                pozo_hmin_dict=pozo_hmin_dict,
                flows_dict=flows_dict,
                proj_to=str(self.proj_to),
                path_out=str(case_dir)
            )
            
            try:
                pipeline.run()
            except SystemExit:
                pass  # Pipeline exits via SystemExit on success
            except Exception as e:
                print(f"    [Error] SewerPipeline crashed: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_cost_estimation(active_pairs)
                
        except Exception as e:
            print(f"  [Error] Failed to initialize SewerPipeline: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_cost_estimation(active_pairs)
                
        except Exception as e:
            print(f"  [Error] Failed to initialize SewerPipeline: {e}")
            return self._fallback_cost_estimation(active_pairs)

        # 6. Read Designed Output
        # rut_03 output check
        out_gpkg = case_dir / f"{solution_name}.gpkg"
        
        if not out_gpkg.exists():
            # Try finding ANY gpkg that isn't the input or 'input_routes'
            possible = [f for f in case_dir.glob("*.gpkg") if "input" not in f.name.lower()]
            if possible:
                out_gpkg = possible[0]
                print(f"  [Info] Found output GPKG with different name: {out_gpkg.name}")
            else:
                print(f"  [Error] No output GPKG found in {case_dir}. SewerPipeline failed silently?")
                return self._fallback_cost_estimation(active_pairs)
            
        designed_gdf = gpd.read_file(out_gpkg)
        
        # 7. Update SWMM Model & Simulate (rut_14)
        print("  Running SWMM simulation...")
        cost, flooding = self._run_swmm_simulation(
            designed_gdf, active_pairs, solution_name, case_dir
        )
        
        return (cost, flooding)

    def _run_swmm_simulation(self, 
                            designed_gdf: gpd.GeoDataFrame, 
                            active_pairs: List[ActivePair],
                            solution_name: str,
                            case_dir: Path) -> Tuple[float, float]:
        """
        Integrate designed pipes into SWMM, run sim, calc cost & flooding.
        """
        # Create modifier
        temp_inp = case_dir / "model.inp"
        modifier = SWMMModifier(self.inp_file)
        
        # FIX: Apply simulation timing from config
        report_step = f"00:{config.REPORT_STEP_MINUTES:02d}:00"
        modifier.set_report_step(report_step)
        
        # Set simulation duration from config
        hours = int(config.ITZI_SIMULATION_DURATION_HOURS)
        minutes = int((config.ITZI_SIMULATION_DURATION_HOURS - hours) * 60)
        end_time = f"{hours:02d}:{minutes:02d}:00"
        modifier.set_end_time(end_time)
        
        # Calculate Construction Cost based on DESIGNED gdf
        total_cost = 0.0
        
        # Pipes Cost
        for row in designed_gdf.itertuples():
            d_mm = self._parse_dimension_to_mm(getattr(row, 'D_int', 0.2))
            length = float(row.L) if hasattr(row, 'L') else row.geometry.length
            
            cost_per_m = 50.0 + (0.5 * d_mm)
            total_cost += cost_per_m * length
            
        # --- PRE-SCAN PIPES for End Elevations (ZFF) ---
        # We need to find the final ZFF of the last segment clearly to set the Tank Invert.
        
        ramal_end_data = {} # ramal_idx -> {'last_node_idx': int, 'zff': float}
        
        for row in designed_gdf.itertuples():
            tramo = str(row.Tramo)
            # Tramo format: {Ramal}.{i}-{Ramal}.{i+1}
            # e.g. 0.0-0.1
            
            parts = tramo.split('-')
            if len(parts) >= 2:
                end_node = parts[-1] # 0.1
                
                node_parts = end_node.split('.')
                if len(node_parts) == 2:
                    ramal_idx_str = node_parts[0]
                    node_idx_str = node_parts[1]
                    
                    if ramal_idx_str.isdigit() and node_idx_str.isdigit():
                        r_idx = int(ramal_idx_str)
                        n_idx = int(node_idx_str)
                        
                        current_data = ramal_end_data.get(r_idx, {'last_node_idx': -1, 'zff': 0.0})
                        
                        if n_idx > current_data['last_node_idx']:
                            # Update with this row's ZFF if available
                            zff = float(row.ZFF) if hasattr(row, 'ZFF') else 0.0
                            ramal_end_data[r_idx] = {'last_node_idx': n_idx, 'zff': zff}

        # --- ADD TANKS (Storage Units) ---
        for i, pair in enumerate(active_pairs):
            
            # Determine correct invert elevation
            # Default to GPKG elevation (predio_z)
            z_invert = pair.predio_z
            
            # If we found a pipe feeding this tank, use its discharge elevation (ZFF)
            # This ensures hydraulic continuity (Pipe Bottom -> Tank Bottom)
            if i in ramal_end_data:
                z_pipe = ramal_end_data[i]['zff']
                # If pipe ZFF is reasonable (not 0.0 unless intended), use it.
                # Or just trust it. Data from rut_03 should be correct.
                z_invert = z_pipe
                
            # Add storage unit
            tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
            modifier.add_storage_unit(
                name=tank_name,
                area=pair.volume / config.TANK_DEPTH_M, 
                max_depth=config.TANK_DEPTH_M,
                invert_elev=z_invert
            )
            # Add tank coordinates
            modifier.add_coordinate(tank_name, pair.predio_x, pair.predio_y)
            
            c_land = (pair.volume / config.TANK_DEPTH_M * config.TANK_OCCUPATION_FACTOR) * 50.0 
            c_tank = CostCalculator.calculate_tank_cost(pair.volume)
            pair.cost_tank = c_tank
            pair.cost_land = c_land
            total_cost += c_tank + c_land
            
        # --- CONNECT PIPES ---
        # Add Designed Pipes FIRST (creates junctions for i.0, i.1, etc.)
        conn_map = {}
        
        # END Nodes (Last Node -> Tank)
        for r_idx, data in ramal_end_data.items():
            if 0 <= r_idx < len(active_pairs):
                pair = active_pairs[r_idx]
                tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
                
                # The node index identified as "last" connects TO the tank
                last_node_name = f"{r_idx}.{data['last_node_idx']}"
                conn_map[last_node_name] = tank_name

        modifier.add_designed_pipeline(designed_gdf, conn_map)
        
        # --- ADD DIVERSION WEIRS (AFTER pipeline so junctions exist) ---
        # Create weirs to divert flow from flooding nodes to the derivation pipes
        # IMPORTANT: Weir crest height = min(0.7*D_inlet, 0.7*D_outlet) - hydraulic design
        for i, pair in enumerate(active_pairs):
            weir_name = f"WR_{pair.node_id}_{i}"
            diversion_node = f"{i}.0"  # First node of the derivation pipe (was just created)
            
            # --- HYDRAULIC WEIR DESIGN ---
            # Get tank volume (clamped)
            MIN_TANK_VOL = config.TANK_MIN_VOLUME_M3
            MAX_TANK_VOL = config.TANK_MAX_VOLUME_M3
            raw_volume = pair.volume if pair.volume > 0 else 3000.0
            tank_volume_m3 = max(MIN_TANK_VOL, min(raw_volume, MAX_TANK_VOL))
            
            # Calculate weir dimensions using hydraulic equations
            weir_design = self._design_weir_hydraulic(pair.node_id, tank_volume_m3)
            
            crest_height = weir_design['crest_height']
            weir_width = weir_design['width']
            target_q = weir_design['design_flow']
            Cd = weir_design['Cd']
            
            print(f"  [Weir Hydraulic] {weir_name}: 0.7D={crest_height:.2f}m, TankVol={tank_volume_m3:.0f}m³ -> Q={target_q:.2f} cms, W={weir_width:.2f}m")
            
            modifier.add_weir(
                name=weir_name,
                from_node=pair.node_id,  
                to_node=diversion_node,   
                weir_type="SIDEFLOW",
                crest_height=crest_height,
                discharge_coeff=Cd,
                width=weir_width,
                end_contractions=0,
                flap_gate=False 
            )
            
        # Save .inp
        # FIX: Explicitly name files with the solution name so PySWMM and Reporter don't get confused
        final_inp_path = case_dir / f"model_{solution_name}.inp"
        modifier.save(str(final_inp_path))
        
        # --- RUN REAL SWMM SIMULATION ---
        # Output file path (.rpt and .out are generated automatically by SWMM usually, 
        # but we need to ensure .out is created for binary reading)
        # swmmio/pyswmm usually expects .out
        
        out_file = final_inp_path.with_suffix('.out')
        rpt_file = final_inp_path.with_suffix('.rpt')
        
        # Clean previous
        if out_file.exists():
            try: os.remove(out_file)
            except: pass
            
        try:
            # Run Simulation using pyswmm
            with Simulation(str(final_inp_path), outputfile=str(out_file)) as sim:
                for _ in sim: pass
        except Exception as e:
            print(f"  [Error] SWMM Simulation failed: {e}")
            # Fallback to invalid high flooding
            return total_cost, 9e9
            
        # Extract Real Metrics
        # (Moved below to include target links calculation)
        
        # --- COMPARISON & LOGGING ---
        # Prepare derivation geometries for visualization
        from shapely.geometry import LineString, Point
        derivations_geom = []
        for pair in active_pairs:
            # We need the node coords and predio coords
            # pair has predio_x, predio_y. We need node coords.
            # We can find node coords from nodes_gdf or just assume we have them.
            # But wait, pair only stores node_id.
            # We need to look up node geometry.
            
            # Optimization: Pre-lookup or do it here. nodes_gdf is indexed by something? 
            # resets index usually.
            
            # Use the detailed geometry if available (calculated by PathFinder)
            if pair.geometry is not None:
                derivations_geom.append(pair.geometry)
            else:
                # Fallback to straight line if PathFinder didn't provide geometry
                node_rows = self.nodes_gdf[self.nodes_gdf['NodeID'] == pair.node_id]
                if not node_rows.empty:
                    nx = node_rows.iloc[0].geometry.x
                    ny = node_rows.iloc[0].geometry.y
                    px = pair.predio_x
                    py = pair.predio_y
                    derivations_geom.append(LineString([(nx, ny), (px, py)]))
                else:
                    # Should be unreachable if data is consistent, but just in case
                    derivations_geom.append(LineString([(pair.predio_x, pair.predio_y), (pair.predio_x+1, pair.predio_y+1)]))

        # Identify Upstream/Downstream links for hydrograph comparison
        # Identify Upstream/Downstream/Derivation links for detailed hydrograph comparison
        target_links = []
        detailed_links = {} # node_id -> {'upstream': [], 'downstream': [], 'derivation': []}
        
        for i, pair in enumerate(active_pairs):
            nid = str(pair.node_id)
            d_links = {'upstream': [], 'downstream': [], 'derivation': []}
            
            # 1. Existing Network Links
            if nid in self.node_topology:
                d_links['upstream'] = self.node_topology[nid].get('upstream', [])
                d_links['downstream'] = self.node_topology[nid].get('downstream', [])
                
            # 2. Derivation Link (New Pipe)
            ramal_id = str(i)
            # Search designed_gdf for all links starting with this ramal_id
            # Tramo format: {Ramal}.{i}-{Ramal}.{i+1}
            
            deriv_links = []
            if designed_gdf is not None and not designed_gdf.empty:
                for row in designed_gdf.itertuples():
                     tramo = str(row.Tramo)
                     if tramo.startswith(f"{ramal_id}."):
                         deriv_links.append(tramo)
            
            # Fallback if empty
            if not deriv_links:
                 deriv_links = [f"{ramal_id}.0-{ramal_id}.1"]
            
            # For hydrograph plots: use ONLY the first link per derivation (cleaner legend)
            derivation_id = deriv_links[0]
            d_links['derivation'] = [derivation_id]
            
            detailed_links[nid] = d_links
            
            # Aggregate for extractor (include ALL derivation links)
            target_links.extend(d_links['upstream'])
            target_links.extend(d_links['downstream'])
            target_links.extend(deriv_links)  # All links for extraction
        
        # ADD WEIR NAMES to target_links for extraction from .out file
        weir_names = [f"WR_{pair.node_id}_{i}" for i, pair in enumerate(active_pairs)]
        target_links.extend(weir_names)
        
        # Deduplicate
        target_links = list(set(target_links))
        
        if not target_links:
            print(f"  [Warning] No target links identified for hydrographs.")

        # Extract Real Metrics (Pass target links)
        target_nodes = list(detailed_links.keys())
        
        # FIX: Ensure we extract hydrographs for ALL nodes that were flooded in the baseline
        # This allows rut_17 to compare the exact same set of nodes (showing improvements as 0s)
        if self.baseline_metrics and self.baseline_metrics.flooding_volumes:
            base_flooded = [n for n, v in self.baseline_metrics.flooding_volumes.items() if v > 1.0]
            target_nodes.extend(base_flooded)
            
        # Also include tank nodes for debug  
        tank_names = [f"TK_{pair.node_id}_{pair.predio_id}" for pair in active_pairs]
        target_nodes.extend(tank_names)
            
        # Deduplicate
        target_nodes = list(set(target_nodes))
        
        current_metrics = self.metric_extractor.extract(str(out_file), target_links=target_links, target_nodes=target_nodes)
        
        # --- DEBUG: Print tank and source node data ---
        print(f"\n  [DEBUG] Tank and Source Node Analysis:")
        print(f"  {'Source Node':<12} {'Base Flood (m3)':<16} {'Sol Flood (m3)':<16} {'Tank Inflow':<12}")
        print(f"  {'-'*60}")
        
        for pair in active_pairs:
            nid = str(pair.node_id)
            tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
            
            # Baseline flooding for this node
            base_flood = self.baseline_metrics.flooding_volumes.get(nid, 0.0) if self.baseline_metrics else 0.0
            
            # Solution flooding for this node
            sol_flood = current_metrics.flooding_volumes.get(nid, 0.0)
            
            # Tank inflow (from hydrograph if extracted)
            tank_inflow = current_metrics.flooding_volumes.get(tank_name, 0.0)
            
            print(f"  {nid:<12} {base_flood:<16.1f} {sol_flood:<16.1f} {tank_inflow:<12.1f}")
        
        print()

        # Prepare Annotation Data using extracted metrics
        annotations_data = [] 
        for i, pair in enumerate(active_pairs):
            nid = str(pair.node_id)
            links = detailed_links.get(nid, {})
            deriv_id = links.get('derivation', [''])[0]
            
            # Default values
            q_val = 0.0
            d_val = 0.0
            vol_val = pair.volume
            
            # Get Max Flow from Hydrographs
            if deriv_id and deriv_id in current_metrics.link_hydrographs:
                flows = current_metrics.link_hydrographs[deriv_id]['flow']
                if flows:
                    q_val = max(flows)
            
             # Get Diameter/Section Text from Designed GDF
            # Mapping logic: 'Tramo' == deriv_id
            if designed_gdf is not None:
                 row = designed_gdf[designed_gdf['Tramo'] == deriv_id]
                 if not row.empty:
                      # User requested raw text. trying 'D_int' first, then 'Seccion'
                      # effectively bypassing parsing
                      if 'D_int' in row.columns:
                          d_val = str(row.iloc[0]['D_int'])
                      elif 'Seccion' in row.columns:
                           d_val = str(row.iloc[0]['Seccion'])
                      else:
                           d_val = "N/A"
            
            # ALWAYS PRINT for verification
            print(f"  [Info] Annotation Data: {nid} -> {deriv_id}: Q={q_val:.4f} cms, D={d_val}, Vol={vol_val:.1f} m3")
            
            annotations_data.append({
                'q_peak': q_val,
                'diameter': d_val, # Now a string!
                'tank_vol': vol_val,
                'derivation_id': deriv_id,
                'node_id': nid,
                'predio_id': pair.predio_id
            })

        # --- INJECT DESIGN VOL INTO TANK UTILIZATION (BEFORE PLOTTING) ---
        # This ensures generate_tank_hydrograph_plots has access to design_vol
        for pair in active_pairs:
            tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
            designed_depth = 5.0  # We use 5m depth in tank design
            designed_vol = pair.volume
            
            if tank_name in current_metrics.tank_utilization:
                max_depth = current_metrics.tank_utilization[tank_name]['max_depth']
                used_vol = (designed_vol / designed_depth) * max_depth if designed_depth > 0 else 0
                
                current_metrics.tank_utilization[tank_name]['design_vol'] = designed_vol
                current_metrics.tank_utilization[tank_name]['stored_volume'] = used_vol
                
                # Update annotations_data with stored_volume
                for ann in annotations_data:
                    if ann.get('node_id') == pair.node_id:
                        ann['stored_vol'] = used_vol
                        break

        # --- COMPARISON & LOGGING ---
        self.comparator.generate_comparison_plots(
            current_metrics, 
            solution_name, 
            case_dir, 
            nodes_gdf=self.nodes_gdf,
            inp_path=str(self.inp_file),
            derivations=derivations_geom,
            detailed_links=detailed_links,
            annotations_data=annotations_data,
            designed_gdf=designed_gdf
        )

        # Calculate Delta (Volume-Based Simple Metric for Comparison)
        delta_vol = self.baseline_metrics.total_flooding_volume - current_metrics.total_flooding_volume
        pct_improvement = 0.0
        if self.baseline_metrics.total_flooding_volume > 0:
            pct_improvement = (delta_vol / self.baseline_metrics.total_flooding_volume) * 100.0
            
        print(f"  [Result] Solution {solution_name}:")
        print(f"    - Construction Cost:  ${total_cost:,.2f}")
        print(f"    - Real Flooding Vol:  {current_metrics.total_flooding_volume:,.2f} m3 (Base: {self.baseline_metrics.total_flooding_volume:,.0f})")
        
        # --- WHITE ELEPHANT PENALTY ---
        penalty_cost = 0.0
        for pair in active_pairs:
            tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
            deriv_id = detailed_links[str(pair.node_id)]['derivation'][0]
            total_inflow_vol_est = 0.0
            
            if deriv_id in current_metrics.link_hydrographs:
                 flow_ts = current_metrics.link_hydrographs[deriv_id]['flow']
                 times_ts = current_metrics.link_hydrographs[deriv_id]['times'] 
                 if flow_ts and len(times_ts) > 1:
                     t_start = times_ts[0]
                     t_secs = [(t - t_start).total_seconds() for t in times_ts]
                     total_inflow_vol_est = np.trapz(flow_ts, x=t_secs)
            
            threshold_vol = pair.volume * 0.05
            
            if total_inflow_vol_est < threshold_vol:
                tank_specific_cost = getattr(pair, 'cost_tank', 0) + getattr(pair, 'cost_land', 0)
                penalty_cost += tank_specific_cost * 1.0 
                print(f"    [Penalty] Tank {tank_name} under-utilized (Vol={total_inflow_vol_est:.1f}/{pair.volume:.1f}). Penalty: ${tank_specific_cost:,.0f}")

        if penalty_cost > 0:
             total_cost += penalty_cost
             print(f"    - Utilization Penalty: +${penalty_cost:,.2f}")
        
        if delta_vol >= 0:
            print(f"    - Flooding REDUCED:   {delta_vol:,.2f} m3 ({pct_improvement:.1f}% Improvement)")
        else:
            print(f"    - Flooding INCREASED: +{abs(delta_vol):,.2f} m3 ({abs(pct_improvement):.1f}% Worsened)")

        # --- ECONOMIC REPORTING ---
        # REAL MODE: Use ComprehensiveEconomicEvaluator results directly
        # The real benefits were already printed in the [Economics] section earlier.
        print(f"    [Economics Mode: REAL] See [Economics] section above for damage/benefit analysis.")
            
        print(f"    - Flooded Nodes:      {current_metrics.flooded_nodes_count} (Base: {self.baseline_metrics.flooded_nodes_count})")
        
        # Store metrics for iterative re-ranking in GreedyTankOptimizer
        self.last_metrics = current_metrics
        
        # --- TANK UTILIZATION SUMMARY ---
        if current_metrics.tank_utilization:
            print(f"\n  [Tank Utilization Summary]")
            print(f"  {'Tank':<20} {'Design Vol':<12} {'Stored Vol':<12} {'Max Depth':<12} {'Util %':<8}")
            print(f"  {'-'*70}")
            for pair in active_pairs:
                tank_name = f"TK_{pair.node_id}_{pair.predio_id}"
                designed_depth = 5.0  # We use 5m depth in tank design
                designed_vol = pair.volume
                
                if tank_name in current_metrics.tank_utilization:
                    max_depth = current_metrics.tank_utilization[tank_name]['max_depth']
                    utilization_pct = (max_depth / designed_depth) * 100 if designed_depth > 0 else 0
                    used_vol = (designed_vol / designed_depth) * max_depth if designed_depth > 0 else 0
                    
                    # INJECT DESIGN VOL FOR PLOTTING
                    current_metrics.tank_utilization[tank_name]['design_vol'] = designed_vol
                    
                    print(f"  {tank_name:<20} {designed_vol:>10,.0f}  {used_vol:>10,.0f}  {max_depth:>10.2f} m  {utilization_pct:>6.1f}%")
                else:
                    print(f"  {tank_name:<20} {designed_vol:>10,.0f}  {'N/A':<10}  {'Not Found':<10}  {'---':<6}")

        
        # --- GENERATE COMPREHENSIVE REPORT FOR USER-FACING SOLUTIONS ---
        # Only for Sequential Steps or named solutions (to avoid massive I/O during GA)
        if solution_name.startswith("Seq_Step") or solution_name.startswith("Solution_"):
             print(f"  [Report] Generating visual comparison for {solution_name}...")
             try:
                 self.comparator.generate_unified_statistical_dashboard(
                     solution=current_metrics,
                     solution_name=solution_name,
                     save_dir=case_dir
                 )
                 # Generate tank hydrograph plots (derivation flow, inflow, depth)
                 self.comparator.generate_tank_hydrograph_plots(
                     solution=current_metrics,
                     solution_name=solution_name,
                     save_dir=case_dir,
                     detailed_links=detailed_links
                 )
             except Exception as e:
                 print(f"  [Warning] Failed to generate report: {e}")
        
        # --- RUN ITZI FLOOD MODEL (if flood_damage enabled) ---
        itzi_result = {}
        if config.COST_COMPONENTS.get('flood_damage', False):
            print(f"\n  [Itzi] Running 2D surface flood simulation...")
            itzi_dir = case_dir / "02_itzi_2d"
            itzi_dir.mkdir(parents=True, exist_ok=True)
            
            itzi_result = run_itzi_for_case(
                swmm_file=str(final_inp_path),
                output_dir=str(itzi_dir),
                verbose=False
            )
            if itzi_result.get('success'):
                print(f"  [Itzi] ✓ 2D simulation complete (Max Depth: {itzi_result.get('max_depth_m', 0):.2f}m)")
                
                # Run CLIMADA on depth raster
                if 'max_depth_file' in itzi_result and itzi_result['max_depth_file']:
                     damage_dir = case_dir / "03_flood_damage"
                     damage_dir.mkdir(parents=True, exist_ok=True)
                     
                     # Use CLIMADA flood damage calculation
                     climada_res = calculate_flood_damage_climada(
                         depth_raster_path=itzi_result['max_depth_file'],
                         output_gpkg=damage_dir / "flood_damage_results.gpkg",
                         output_txt=damage_dir / "flood_damage_report.txt"
                     )
                     itzi_result['climada_result'] = climada_res
                     print(f"  [CLIMADA] Total Damage: ${climada_res.get('total_damage_usd', 0):,.2f}")
            else:
                raise RuntimeError(f"ITZI simulation failed: {itzi_result}")
        else:
            print(f"\n  [Itzi] SKIPPED (flood_damage=False in config)")

        # --- CALCULATE ECONOMIC IMPACT ---
        print("\n  [Economics] Calculating Economic Impact...")
        
        # Use CLIMADA results from itzi_result (if available)
        if itzi_result and 'climada_result' in itzi_result and itzi_result['climada_result']:
            climada_res = itzi_result['climada_result']
            flood_damage = climada_res.get('total_damage_usd', 0.0)
            
            # Store for optimizer to calculate avoided cost
            self.last_flood_damage_usd = flood_damage
            
            # If this is the first CLIMADA run, store as baseline proxy
            if not hasattr(self, 'baseline_flood_damage') or self.baseline_flood_damage == 0:
                # First run: estimate baseline from current damage + marginal improvement
                # This is an approximation - ideally baseline ITZI should run separately
                self.baseline_flood_damage = flood_damage * 1.2  # Assume ~20% improvement from first tank
                print(f"  [Economics] Baseline CLIMADA Damage (estimated): ${self.baseline_flood_damage:,.2f}")
            
            # Return flooding damage as economic impact
            economic_res = {
                'net_impact_usd': flood_damage,
                'flood_damage_usd': flood_damage,
                'details': climada_res
            }
            print(f"  [Economics] ITZI/CLIMADA Flood Damage: ${flood_damage:,.2f}")
        else:
            # No CLIMADA result - use volume-based approximation or zero
            self.last_flood_damage_usd = 0.0
            economic_res = {
                'net_impact_usd': 0.0,
                'flood_damage_usd': 0.0,
                'details': {}
            }
            print(f"  [Economics] Flood Damage: $0.00 (ITZI/CLIMADA disabled)")
        
        # --- GENERATE AVOIDED COST BUDGET (if deferred_investment enabled) ---
        if config.COST_COMPONENTS.get('deferred_investment', False):
            from rut_20_avoided_costs import DeferredInvestmentCost
            deferred_calc = DeferredInvestmentCost(
                base_precios_path=str(config.CODIGOS_DIR / "base_precios.xlsx"),
                capacity_threshold=0.9
            )
            deferred_calc.run(
                inp_path=str(final_inp_path),
                output_dir=str(case_dir)
            )
        else:
            print(f"  [Economics] Deferred Investment: SKIPPED (disabled in config)")
        
        # Net Impact (The "Badness" we want to minimize)
        # Typically = Residual Damage - Benefits
        # Or if VolumeBased: Volume * Rate
        net_impact_usd = economic_res.get('net_impact_usd', 0.0)
        details = economic_res.get('details', {})
        
        print(f"  [Economics] Net Economic Impact (Damage - Benefits): ${net_impact_usd:,.2f}")
        if 'residual_damage' in details:
            print(f"    - Residual Damage:   ${details['residual_damage']:,.2f}")
        if 'total_benefits' in details:
            print(f"    - Total Benefits:    ${details['total_benefits']:,.2f}")
            for b_name, b_val in details.get('benefits_breakdown', {}).items():
                print(f"      * {b_name}: ${b_val:,.2f}")

        # --- FINAL RESULTS ---
        # The optimizer expects (cost, secondary_metric).
        # We assume total_cost (construction) is the primary cost, and we want to pass the simplified flooding volume 
        # as the secondary metric for the Pareto front, OR we pass the Full Economic Impact?
        # The current signature is (cost, remaining_flooding_volume).
        # But 'cost' is construction cost usually.
        # Wait, if we want to minimize Total Cost TO SOCIETY, it should be Construction + Impact.
        # For now, let's keep Construction as Index 0, and Flooding Volume as Index 1 (for compatibility with optimizer logic)
        # OR, we can update the Optimizer to use 'Net Economic Impact' as the second objective?
        # The user's request "Stopping Criterion: ECONOMIC BREAKEVEN" implies we need to compare Construction vs Benefits.
        # Breakeven = (Benefits >= Construction).
        # The optimizer check logic is in rut_10.
        
        # Let's verify what rut_10 does with the return value.
        # It expects (cost, remaining_flooding). cost is traditionally construction.
        # But for Breakeven analysis, it needs Benefits.
        # The optimizer currently doesn't see benefits directly returned here unless we change valid_pairs or something.
        
        # ACTUALLY: The return value is just for printing/plotting in rut_10 loop.
        # But wait, rut_10 uses `cost` and `remaining_flooding`.
        
        # CRITICAL: We need to expose the BENEFITS to the optimizer for the breakeven check.
        # We can attach it to the current_metrics or return it. But the signature is fixed.
        # A hack is to treat "remaining_flooding" as "Net Impact" if we change units? No, dangerous.
        
        # Let's print them vividly. For the Breakeven check in rut_10:
        # It compares `total_cost` (construction) with `reduction * cost_per_m3`.
        # We should update rut_10 later to read the real benefits if we want "Real" breakeven.
        # For now, we print detailed info here.
        
        print(f"  [Economics] Total Construction Cost: ${total_cost:,.2f}")
        print(f"  [Economics] B/C Ratio:               {(details.get('total_benefits', 0)/total_cost):.2f}" if total_cost > 0 else "Inf")
        
        return total_cost, current_metrics.total_flooding_volume
    

    def _read_designed_gpkg(self, output_gpkg: Path) -> gpd.GeoDataFrame:
        """Helper to read and validate designed GPKG."""
        if not output_gpkg.exists():
            return None
        return gpd.read_file(output_gpkg)

