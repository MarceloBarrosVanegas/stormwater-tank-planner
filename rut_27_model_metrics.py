"""
Stormwater Tank Optimization Metrics Extractor
===============================================

This module provides tools for extracting hydraulic metrics from SWMM (Storm Water Management Model) simulations,
loading spatial data for predios (properties), and ranking candidate pairs for stormwater tank placement.
It integrates with elevation data, risk assessments, and network analysis to support optimization workflows.

Key Components:
- SystemMetrics: Dataclass for storing system-wide and per-node hydraulic metrics.
- CandidatePair: Dataclass representing potential node-predio connections for tank placement.
- MetricExtractor: Main class for loading data, running simulations, extracting metrics, and ranking candidates.

Workflow Overview:
1. Initialize MetricExtractor with project paths and SWMM file.
2. Load predios GeoDataFrame with elevation and slope data.
3. Run SWMM simulation and extract output metrics (flooding, flows, depths).
4. Map spatial risk data to flooded nodes.
5. Generate and rank candidate pairs based on hydraulic criteria and weights.
6. Store results in entity attributes for further processing.


Usage:
    runner = MetricExtractor(project_root=config.PROJECT_ROOT, swmm_file=config.SWMM_FILE, predios_path=config.PREDIOS_FILE)
    runner.run()
    # Access results: runner.ranked_candidates, runner.metrics
    
    
[Start: Initialize MetricExtractor]
    |
    v
[Load Predios Data]
    - Read predios GeoDataFrame
    - Validate/reproject CRS
    - Calculate elevations and slopes
    - Filter by slope criteria
    |
    v
[Extract SWMM Metrics]
    - Load network from INP file
    - Run SWMM simulation (if needed)
    - Parse output: system hydrographs, link/node/tank data
    - Calculate volumes, flows, depths
    - Generate flooded nodes GeoDataFrame
    |
    v
[Map Spatial Risk]
    - Load risk data from GPKG
    - Perform nearest-neighbor matching
    - Assign failure probabilities to nodes
    |
    v
[Generate Candidate Pairs]
    - Compare nodes vs. predios (elevation filter)
    - Parallel instantiation of CandidatePair objects
    - Calculate distances, gaps, etc.
    |
    v
[Rank Candidates]
    - Use WeightedCandidateSelector for scoring
    - Sort by score (desc) and distance (asc)
    - Deduplicate by node ID
    - Store ranked list in entity
    |
    v
[End: Access Results]
    - runner.ranked_candidates (list of dicts)
    - runner.metrics (SystemMetrics object)
    
"""

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ThreadPoolExecutor
from pyproj import CRS
from pyswmm import Output, SystemSeries, Simulation
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point, Polygon
from swmm.toolkit.shared_enum import LinkAttribute, NodeAttribute
from tqdm import tqdm
from tabulate import tabulate
from line_profiler import profile


import config
config.setup_sys_path()

from rut_02_elevation import ElevationGetter, ElevationSource
from rut_02_get_flodded_nodes import CandidatePair, PredioSlopeCalculator, WeightedCandidateSelector
from rut_06_pipe_sizing import CapacidadMaximaTuberia
from rut_25_from_inp_to_vector import NetworkExporter






def _instantiate_candidate_pair(task):
    """
    Standalone worker function to instantiate a CandidatePair.
    Separated from the main class to avoid nested functions.
    """
    n = task['node']
    p = task['predio']

    # Cálculo de distancia integrado
    d = math.dist(task['start_xy'], task['end_xy'])

    return CandidatePair(
        node_id=n['NodeID'],
        predio_id=p['id'],
        node_volume_flood=n['total_volume'],
        node_max_flow=n['total_flow'],
        node_max_depth=n['NodeDepth'],
        node_z_invert=n['InvertElevation'],
        node_z_surface=float(task['node_surface_z']),
        node_x=float(task['start_xy'][0]),
        node_y=float(task['start_xy'][1]),
        node_geometry=n['geometry'],
        node_probability_failure=n['failure_prob'],
        node_flow_over_capacity=n['flow_over_capacity'],
        node_flooding_flow=n['flow_node_flooding'],
        node_volume_over_capacity=n['vol_over_capacity'],
        node_volume_flooding=n['vol_node_flooding'],
        node_distance_to_predio=d,
        node_elevation_gap_to_predio=float(task['gap']),
        derivation_link="",
        derivation_link_geometry=None,
        derivation_target_node_geometry=None,
        predio_area_m2=round(p['geometry'].area, 0),
        predio_geometry=p['geometry'],
        predio_x_centroid=float(task['end_xy'][0]),
        predio_y_centroid=float(task['end_xy'][1]),
        predio_ground_z=float(p['z']),
        tank_volume_simulation=0.0,
        tank_max_depth=0.0,
        
        cost_tank=0.0,
        cost_land=0.0,
        is_tank=True,
        target_id=""
    )


@dataclass
class SystemMetrics:
    """
    Stores system-wide hydraulic metrics and detailed per-node data.
    """

    # SYSTEM-WIDE
    total_flooding_volume: float = 0.0  # m3
    total_outfall_volume: float = 0.0  # m3
    total_max_outfall_flow: float = 0.0  # m3
    total_max_flooding_flow: float = 0.0  # m3
    system_flood_hydrograph: Dict[str, np.ndarray] = field(default_factory=dict)  # {'times': [], 'total_rate': []}
    system_outfall_flow_hydrograph: Dict[str, np.ndarray] = field(default_factory=dict)  # {'times': [], 'total_rate': []}

    # Per-Node Stats (NodeID -> Stats Dict)
    node_data: Dict[str, Dict] = field(default_factory=dict)
    link_data: Dict[str, Dict] = field(default_factory=dict)
    tank_data: Dict[str, Dict] = field(default_factory=dict)

    swmm_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    flooded_nodes_gdf: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)

    flooded_nodes_count: int = 0
    cost: float = 0.0

    # NETWORK HEALTH
    surcharged_links_count: int = 0
    overloaded_links_length: float = 0.0
    system_mean_utilization: float = 0.0
    

@dataclass
class CandidatePair:
    """
    Represents a candidate assignment of a node to a predio.
    Shared data structure for ranking and dynamic evaluation.
    """
    # --- Identificadores y Metadatos ---
    node_id: str  # Nombre/ID del nodo (ej. J-123)
    predio_id: str  # Nombre/ID del predio (ej. P-456)

    # --- Datos del Nodo de derivacion ---
    node_volume_flood: float = 0.0  # Volumen de inundación en el nodo (m3)
    node_max_flow: float = 0.0  # Caudal pico de inundación (m3/s)
    node_max_depth: float = 0.0  # H_max del nodo original
    node_z_invert: float = 0.0  # Elevación batea del nodo (m)
    node_z_surface: float = 0.0  # Elevación superficie del nodo (m)
    node_x: float = 0.0  # Coordenada X del nodo (m)
    node_y: float = 0.0  # Coordenada Y del nodo (m)
    node_probability_failure: float = 0.0
    node_flow_over_capacity: float = 0.0  # Caudal sobre capacidad de tubería (m3/s)
    node_flooding_flow: float = 0.0  # Caudal de inundación del nodo (m3/s)
    node_volume_over_capacity: float = 0.0  # Volumen sobre capacidad de tubería (m3)
    node_volume_flooding: float = 0.0  # Volumen de inundación del nodo (m3)
    node_geometry: Optional[Point] = None  # Point con las coordenadas del nodo
    node_distance_to_predio: float = 0.0  # Distancia euclidiana al predio (m)
    node_elevation_gap_to_predio: float = 0.0  # Desnivel entre nodo y predio (m)

    # --- Tuberia ---
    derivation_link: str = ""  # ID del tramo que más aporta a la inundación
    derivation_link_geometry: Optional[LineString] = None
    derivation_target_node_geometry: Optional[Point] = None

    # --- Datos del Predio  ---
    predio_area_m2: float = 0.0
    predio_geometry: Optional[Polygon] = None
    predio_x_centroid: float = 0.0
    predio_y_centroid: float = 0.0
    predio_ground_z: float = 0.0  # Elevación del terreno

    # ---  Tanque  ---
    tank_volume_simulation: float = 0.0  # Volumen capturado en simulación post-ejecución
    tank_max_depth: float = 0.0  # Profundidad de diseño (por defecto en config)

    # --- Datos de Tubería (de rut_03 diseño) ---
    diameter: str = 'N/A'  # Diámetro de tubería diseñada (ej. "1.2" o "1.5x1.2")
    pipeline_length: float = 0.0  # Longitud total de tubería (m)

    # --- Costos Estimados (Para rut_17 Dashboard) ---
    cost_tank: float = 0.0
    cost_land: float = 0.0

    # --- Contexto del Árbol (Tree Routing) ---
    is_tank: bool = True  # ¿Se conecta a un tanque o es solo una tubería?
    target_id: str = ""  # ID del nodo/tanque al que finalmente descarga


class MetricExtractor:
    """
    Extracts metrics from SWMM binary output (.out), loads predios, and ranks candidates.
    Functions as an entity class with data stored in attributes.
    """

    def __init__(self, project_root: Path = None, predios_path=None):
        self.project_root = Path(project_root) if project_root else Path(os.getcwd())
        print(f"Project Root: {self.project_root}")

        # Default configuration
        self.predios_gdf = self.get_predios_gdf(predios_path)
        self.nodes_gdf = None
        self.swmm_gdf = None
        self.ranked_candidates = []  # Entity attribute for ranked pairs
        self.metrics = None  # Entity attribute for metrics

        self.at_capacity_flow = CapacidadMaximaTuberia()
        self.parse_shape_from_swmm = {'RECT_CLOSED': 'rectangular', 'RECT_OPEN': 'rectangular', 'MODBASKETHANDLE': 'rectangular', 'CIRCULAR': 'circular'}
        self.risk_file_path = config.FAILIURE_RISK_FILE

    @staticmethod
    def parse_seccion_to_pypiper(conduit_row: pd.Series) -> str:
        """
        Extrae la sección de una tubería como string para PyPiper.

        Args:
            conduit_row: Serie de Pandas con datos de la tubería (fila del DataFrame)

        Returns:
            str: "geom1" si circular, "geom1xgeom2" si rectangular
        """
        shape = str(conduit_row['Shape']).strip().upper()
        geom1 = conduit_row['Geom1']

        # Circular: solo diámetro
        if shape == 'CIRCULAR':
            return str(geom1)

        # Rectangular: ancho x alto
        if shape in ['RECT_CLOSED', 'RECTANGULAR', 'MODBASKETHANDLE', 'RECT_OPEN']:
            geom2 = conduit_row['Geom2']
            return f"{geom1}x{geom2}"

        raise ValueError(f"Forma no soportada: {shape}")

    @staticmethod
    def _calculate_volume_from_series(series: pd.Series, threshold: float = 0.0) -> float:
        """
        Calcula volumen integrando una serie temporal usando método rectangular.

        Args:
            series: Serie con índice datetime y valores de caudal [m³/s]
            threshold: Umbral mínimo para considerar el flujo

        Returns:
            Volumen total [m³]
        """
        if len(series) == 0:
            return 0.0

        max_val = series.max()

        if max_val <= threshold or len(series) <= 1:
            return 0.0

        # Asegurar índice datetime
        idx = pd.to_datetime(series.index)

        # dt en segundos entre muestras consecutivas
        dt_secs = idx.to_series().diff().dt.total_seconds().fillna(0).values

        # Usar tasas de la muestra anterior para integración rectangular
        prev_rates = series.shift(1, fill_value=0).values

        # Volumen (m³) = sum(rate_prev (m³/s) * dt (s))
        volume = (prev_rates * dt_secs).sum()

        return float(volume)

    @staticmethod
    def run_swmm_simulation(inp_file_path: str) -> Path:
        """
        Runs SWMM simulation with progress bar.
        Returns path to .out file. Skips if .out already exists.
        """
        inp_path = Path(inp_file_path)
        out_path = inp_path.with_suffix('.out')

        if out_path.exists():
            # print(f"[SWMM] Output already exists: {out_path}")
            return out_path

        with Simulation(str(inp_path)) as sim:
            total_duration = (sim.end_time - sim.start_time).total_seconds()
            with tqdm(total=100, desc="Running SWMM Simulation", unit="%", disable=True) as pbar:
                last_pct = 0
                for _ in sim:  # Fixed: Renamed 'step' to '_' as it's unused
                    current_pct = int(((sim.current_time - sim.start_time).total_seconds() / total_duration) * 100)
                    if current_pct > last_pct:
                        pbar.update(current_pct - last_pct)
                        last_pct = current_pct
                pbar.update(100 - last_pct)

        return out_path

    def load_swmm_network(self, in_file_path: str) -> Tuple[gpd.GeoDataFrame, pd.Series]:
        """
        Carga la red física y calcula capacidades de tuberías.

        Returns:
            swmm_gdf: GeoDataFrame con la red y columnas Seccion, D_int
            q_at_capacity_series: Serie con capacidad máxima por link (m³/s)
        """
        exporter = NetworkExporter(str(in_file_path))
        swmm_gdf, nodes_df = exporter.run(None, run_hydraulics=False, crs=config.PROJECT_CRS)

        # Calcular secciones
        swmm_gdf['Seccion'] = swmm_gdf['Shape'].map(self.parse_shape_from_swmm)
        swmm_gdf['D_int'] = swmm_gdf.apply(self.parse_seccion_to_pypiper, axis=1)

        # Calcular capacidades
        q_at_capacity, _, _ = self.at_capacity_flow.calcular_capacidad_maxima(
            D_int=swmm_gdf['D_int'].to_numpy().astype(str),
            S=np.where(swmm_gdf['Slope'] < 0, 0.01, swmm_gdf['Slope']),
            Rug=swmm_gdf['Roughness'].to_numpy().astype(float),
            Seccion=swmm_gdf['Seccion'].to_numpy().astype(str),
            h_D_objetivo=config.CAPACITY_MAX_HD
        )

        q_at_capacity_series = pd.Series(np.round(q_at_capacity / 1000, 3), index=swmm_gdf.index)

        return swmm_gdf, nodes_df, q_at_capacity_series

    def extract(self, in_file_path, extrac_items) -> SystemMetrics:
        """
        Parses output file and returns SystemMetrics.
        Extracts hydrographs, link data, node data, tank data, and flooded_nodes_gdf.
        """
        metrics = SystemMetrics()
        in_file_path = str(in_file_path)

        # =========================================================================
        # 1. OBTENER RED FÍSICA
        # =========================================================================
        with tqdm(total=2, desc="Load INP file", unit="step") as pbar:
            pbar.set_description("Reading INP file")
            swmm_gdf, nodes_df, q_at_capacity_series = self.load_swmm_network(in_file_path)
            pbar.update(1)

            pbar.set_description("Running SWMM Simulation")
            out_file_path = self.run_swmm_simulation(in_file_path)
            pbar.update(1)

        # =========================================================================
        # 2. EXTRACCIÓN DE DATOS DE SWMM OUTPUT
        # =========================================================================
        flooded_candidates = []
        with Output(str(out_file_path)) as out:
            all_nodes = list(out.nodes)
            all_links = list(out.links)
            tank_nodes = [nid for nid in all_nodes if nid.startswith("tank_")]

            if len(extrac_items.get('include_nodes', {})) == 0:
                extrac_items['include_nodes'] = all_nodes

            if len(extrac_items.get('include_links', {})) == 0:
                extrac_items['include_links'] = all_links

            # =====================================================================
            # 3. HIDROGRAMA DE FLOODING DEL SISTEMA
            # =====================================================================
            system_flood_series = pd.Series(SystemSeries(out).flood_losses)
            system_outfall_flow_series = pd.Series(SystemSeries(out).outfall_flows)

            if len(system_flood_series) > 0:
                system_flood_series = system_flood_series.sort_index()
                metrics.system_flood_hydrograph = {
                    'times': system_flood_series.index.to_numpy(),
                    'total_rate': system_flood_series.values
                }
                metrics.total_flooding_volume = self._calculate_volume_from_series(system_flood_series)
                metrics.total_max_flooding_flow = system_flood_series.max()

                metrics.system_outfall_flow_hydrograph = {
                    'times': system_outfall_flow_series.index.to_numpy(),
                    'total_rate': system_outfall_flow_series.values
                }
                metrics.total_outfall_volume = self._calculate_volume_from_series(system_outfall_flow_series)
                metrics.total_max_outfall_flow = system_outfall_flow_series.max()
            else:
                metrics.total_flooding_volume = 0.0
                metrics.system_flood_hydrograph = {'times': np.array([]), 'total_rate': np.array([])}

            # =====================================================================
            # 4. PROCESAR LINKS
            # =====================================================================
            max_flow_dict = {}
            max_vel_dict = {}
            max_capacity_dict = {}

            for lid in tqdm(all_links, desc="Processing Links", unit="link"):
                lid_str = str(lid).strip()

                flow_series = pd.Series(out.link_series(lid, LinkAttribute.FLOW_RATE))
                depth_series = pd.Series(out.link_series(lid, LinkAttribute.FLOW_DEPTH))
                capacity_series = pd.Series(out.link_series(lid, LinkAttribute.CAPACITY))
                volume_series = pd.Series(out.link_series(lid, LinkAttribute.FLOW_VOLUME))
                velocity_series = pd.Series(out.link_series(lid, LinkAttribute.FLOW_VELOCITY))

                max_flow = float(flow_series.max()) if len(flow_series) > 0 else 0.0
                max_vel = float(velocity_series.max()) if len(velocity_series) > 0 else 0.0
                max_cap = float(capacity_series.max()) if len(capacity_series) > 0 else 0.0

                max_flow_dict[lid_str] = max_flow
                max_vel_dict[lid_str] = max_vel
                max_capacity_dict[lid_str] = max_cap

                if lid in extrac_items.get('include_links', []):
                    total_volume = self._calculate_volume_from_series(flow_series, threshold=0)
                    metrics.link_data[lid] = {
                        'total_volume': total_volume,
                        'max_flow': max_flow,
                        'max_depth': float(depth_series.max()) if len(depth_series) > 0 else 0.0,
                        'max_capacity': max_cap,
                        'max_velocity': max_vel,
                        'flow_series': flow_series,
                        'depth_series': depth_series,
                        'capacity_series': capacity_series,
                        'volume_series': volume_series,
                        'velocity_series': velocity_series,
                        'peak_time': flow_series.idxmax() if len(flow_series) > 0 else None
                    }

            # Actualizar swmm_gdf
            swmm_gdf.index = swmm_gdf.index.astype(str).str.strip()
            swmm_gdf['MaxFlow'] = swmm_gdf.index.map(max_flow_dict).fillna(0.0)
            swmm_gdf['MaxVel'] = swmm_gdf.index.map(max_vel_dict).fillna(0.0)
            swmm_gdf['Capacity'] = swmm_gdf.index.map(max_capacity_dict).fillna(0.0)
            swmm_gdf['flow_over_pipe_capacity'] = np.maximum(swmm_gdf['MaxFlow'] - q_at_capacity_series, 0)
            swmm_gdf['flow_pipe_capacity'] = q_at_capacity_series
            swmm_gdf['flow_pipe_capacity'] = q_at_capacity_series
            metrics.swmm_gdf = swmm_gdf

            # =====================================================================
            # 4.1 NETWORK HEALTH METRICS
            # =====================================================================
            # Filter valid pipes (capacity > 0) to avoid division by zero
            valid_pipes = swmm_gdf[swmm_gdf['Capacity'] > 0].copy()
            if not valid_pipes.empty:
                # Calculate utilization
                valid_pipes['utilization'] = valid_pipes['MaxFlow'] / valid_pipes['Capacity']
                
                # Surcharged status
                valid_pipes['Surcharged'] = valid_pipes['utilization'] >= 1.0
                
                # Update the main GDF with these new columns
                swmm_gdf = swmm_gdf.join(valid_pipes[['utilization', 'Surcharged']], rsuffix='_calc')
                # Fill NaNs for non-valid pipes (e.g. dummy links)
                swmm_gdf['utilization'] = swmm_gdf['utilization'].fillna(0.0)
                swmm_gdf['Surcharged'] = swmm_gdf['Surcharged'].fillna(False)

                # Surcharged links (over 100% capacity)
                surcharged_links = swmm_gdf[swmm_gdf['Surcharged'] == True]
                metrics.surcharged_links_count = len(surcharged_links)
                metrics.overloaded_links_length = surcharged_links['Length'].sum()
                
                # System mean utilization (using mean of ratios)
                metrics.system_mean_utilization = swmm_gdf[swmm_gdf['Capacity'] > 0]['utilization'].mean() * 100 # In percentage
            else:
                 metrics.surcharged_links_count = 0
                 metrics.overloaded_links_length = 0.0
                 metrics.system_mean_utilization = 0.0

            inletnodes_series = swmm_gdf['InletNode'].astype(str).str.strip().str.upper()

            # =====================================================================
            # 5. PROCESAR NODOS
            # =====================================================================
            flooded_nodes_count = 0
            for nid in tqdm(all_nodes, desc="Processing Nodes", unit="node"):
                nid_str = str(nid).strip().upper()

                if nid_str not in extrac_items.get('include_nodes', []):
                    continue

                incoming_links = inletnodes_series[inletnodes_series == nid_str].index.tolist()
                max_over_capacity_flow = 0.0
                flow_incoming_series = pd.Series([0.0])
                capacity_incoming = 0.0

                if len(incoming_links) > 0:
                    excesses = swmm_gdf['flow_over_pipe_capacity'].loc[incoming_links]
                    link_with_max_excess = excesses.idxmax()
                    max_over_capacity_flow = float(excesses.max())

                    link_name = str(link_with_max_excess)
                    if link_name in all_links:
                        try:
                            flow_incoming_series = pd.Series(out.link_series(link_name, LinkAttribute.FLOW_RATE))
                            capacity_incoming = float(q_at_capacity_series.loc[link_with_max_excess])
                        except (TypeError, KeyError):
                            flow_incoming_series = pd.Series([0.0])
                            capacity_incoming = 0.0

                flooding_series = pd.Series(out.node_series(nid, NodeAttribute.FLOODING_LOSSES))
                depth_series = pd.Series(out.node_series(nid, NodeAttribute.INVERT_DEPTH))
                max_flooding_flow = float(flooding_series.max())

                if config.TANK_OPT_OBJECTIVE == 'capacity':
                    max_flow = max_flooding_flow + max_over_capacity_flow
                else:
                    max_flow = max_flooding_flow

                flooding_volume = self._calculate_volume_from_series(flooding_series, threshold=config.MINIMUN_FLOODING_FLOW)

                if config.TANK_OPT_OBJECTIVE == 'capacity':
                    flow_excess_series = (flow_incoming_series - capacity_incoming).clip(lower=0)
                    over_capacity_volume = self._calculate_volume_from_series(flow_excess_series, threshold=0)
                    total_volume = flooding_volume + over_capacity_volume
                else:
                    flow_excess_series = (flow_incoming_series - capacity_incoming).clip(lower=0)
                    over_capacity_volume = self._calculate_volume_from_series(flow_excess_series, threshold=0)
                    total_volume = flooding_volume
                    

                if flooding_volume > 0:
                    flooded_nodes_count += 1

                x = float(nodes_df.loc[nid_str, 'X'])
                y = float(nodes_df.loc[nid_str, 'Y'])
                invert_elevation = float(nodes_df.loc[nid_str, 'InvertElev'])


                max_depth = float(depth_series.max()) if len(depth_series) > 0 else 0.0

                metrics.node_data[nid] = {
                    'total_volume': total_volume,
                    'max_flow': max_flow,
                    'max_depth': max_depth,
                    'flooding_volume': flooding_volume,
                    'over_capacity_volume': over_capacity_volume,
                    'flooding_series': flooding_series,
                    'depth_series': depth_series,
                    'max_flooding_flow': max_flooding_flow,
                    'max_over_capacity_flow': max_over_capacity_flow,
                    'peak_time': flooding_series.idxmax() if len(flooding_series) > 0 else None,
                    'x': x,
                    'y': y,
                    'invert_elevation': invert_elevation,
                }

                
                flooded_candidates.append({
                    'node_id': nid,
                    'node_x': x,
                    'node_y': y,
                    'flow_over_capacity': max_over_capacity_flow,
                    'flow_node_flooding': max_flooding_flow,
                    'total_flow':max_flow,
                    'vol_over_capacity': over_capacity_volume,
                    'vol_node_flooding': flooding_volume,
                    'total_volume': total_volume,
                    'node_depth': max_depth,
                    'invert_elevation': invert_elevation,
                    'geometry': Point(x, y),
                })

            metrics.flooded_nodes_count = flooded_nodes_count

            # =====================================================================
            # 6. PROCESAR TANQUES
            # =====================================================================
            for nid in tqdm(tank_nodes, desc="Processing Tanks", unit="tank"):
                depth_series = pd.Series(out.node_series(nid, NodeAttribute.INVERT_DEPTH))
                flooding_series = pd.Series(out.node_series(nid, NodeAttribute.FLOODING_LOSSES))
                flow_series = pd.Series(out.node_series(nid, NodeAttribute.TOTAL_INFLOW))
                volume_series = pd.Series(out.node_series(nid, NodeAttribute.PONDED_VOLUME))

                metrics.tank_data[nid] = {
                    'total_volume': self._calculate_volume_from_series(flow_series, threshold=0),
                    'max_flow': float(flow_series.max()) if len(flow_series) > 0 else 0.0,
                    'max_depth': float(depth_series.max()) if len(depth_series) > 0 else 0.0,
                    'max_flooding_flow': float(flooding_series.max()) if len(flooding_series) > 0 else 0.0,
                    'max_stored_volume': float(volume_series.max()) if len(volume_series) > 0 else 0.0,
                    'flow_series': flow_series,
                    'volume_series': volume_series,
                    'depth_series': depth_series,
                    'flooding_series': flooding_series,
                    'peak_time': flow_series.idxmax() if len(flow_series) > 0 else None,
                    'exceedance_volume': self._calculate_volume_from_series(flooding_series, threshold=0),
                }

        # =========================================================================
        # 7. GENERAR GEODATAFRAME
        # =========================================================================
        if len(flooded_candidates) > 0:
            metrics.flooded_nodes_gdf = gpd.GeoDataFrame(flooded_candidates, geometry='geometry', crs=config.PROJECT_CRS)
        else:
            metrics.flooded_nodes_gdf = gpd.GeoDataFrame()

        return metrics

    def get_spatial_risk(self, df):
        """Maps risk data to nodes using spatial proximity (nearest neighbor)."""
        if not self.risk_file_path or not Path(self.risk_file_path).exists() or df.empty:
            df['failure_prob'] = 0
            return df

        risk_gdf = gpd.read_file(self.risk_file_path)

        if not risk_gdf.empty and 'failure_prob' in risk_gdf.columns:
            # Ensure geometries are valid
            risk_gdf = risk_gdf[risk_gdf.geometry.is_valid & ~risk_gdf.geometry.is_empty]

            # get_coordinates returns a dataframe with columns x, y.
            # The index of this dataframe corresponds to the index in risk_gdf.
            coords_risk_df = risk_gdf.geometry.get_coordinates()
            coords_risk_df_index = coords_risk_df.index.to_numpy()

            # Coordinate arrays
            risk_coords = coords_risk_df[['x', 'y']].to_numpy()
            node_coords = df[['node_x', 'node_y']].to_numpy()

            if len(risk_coords) > 0 and len(node_coords) > 0:
                # Calculate distance matrix (RiskFeaturesCoordinates x Nodes)
                dists = cdist(risk_coords, node_coords, metric='euclidean')

                # Find the index in the flattened 'risk_coords' array that is closest
                nearest_coord_idx = np.argmin(dists, axis=0)
                min_dists = np.min(dists, axis=0)

                # Map the flattened vertex index back to the original risk_gdf index label
                matched_gdf_indices = coords_risk_df_index[nearest_coord_idx]

                # Retrieve probabilities using .loc (label-based lookup)
                nearest_probs = risk_gdf.loc[matched_gdf_indices, 'failure_prob'].values

                # Assign failure_prob only if distance < 0.1, otherwise 0
                df['failure_prob'] = np.where(min_dists < 0.1, nearest_probs, 0.0)
            else:
                print("Warning: Coordinate mismatch or empty geometries for risk/nodes.")
                df['failure_prob'] = 0.0
        else:
            print("Warning: Risk GPKG empty or missing 'failure_prob' column.")
            df['failure_prob'] = 0.0

        return df

    
    @staticmethod
    def ensure_crs(gdf, target_crs, name="layer"):
        if gdf.crs is None:
            print(f"⚠️ {name} no tiene CRS definido. No se puede reproyectar con seguridad.")
            return gdf

        try:
            # Convertir a objeto pyproj.CRS para comparación robusta (igual que en el raster)
            current_crs = CRS.from_user_input(gdf.crs)
            if current_crs == target_crs:
                print(f"CRS de {name} correcto. Omitiendo reproyección.")
                return gdf
        except Exception:
            pass  # Si falla la comparación, asumimos que son diferentes

        print(f"Reproyectando {name} al CRS del proyecto...")
        gdf.to_crs(target_crs, inplace=True)

        return gdf

    @staticmethod
    def validate_and_update_raster_crs(raster_path: Path, target_crs: CRS) -> Path:
        """
        Verifica el CRS del raster usando comparación lógica de objetos (pyproj).
        Si no coincide con target_crs, lo reproyecta.
        """
        with rasterio.open(raster_path) as src:
            # Verificar si el raster tiene CRS
            if src.crs is None:
                print(f"⚠️ ADVERTENCIA: El raster '{raster_path.name}' no tiene CRS definido.")
                print("   -> No se puede verificar ni reproyectar automáticamente. Se usará el archivo original.")
                return raster_path

            try:
                # rasterio.crs.CRS tiene to_wkt(), por lo que pyproj puede leerlo
                current_crs = CRS.from_user_input(src.crs)

                # La igualdad (==) en pyproj verifica si representan la misma proyección física
                if current_crs == target_crs:
                    print(f"El CRS del raster es correcto: {raster_path.name}")
                    return raster_path
            except Exception as e:
                print(f"No se pudo comparar CRS automáticamente ({e}). Se procederá a reproyectar.")

            print(f"CRS incorrecto o formato distinto en {raster_path.name}. Reproyectando a referencia del proyecto...")

            # Calcular la nueva transformación y dimensiones
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Definir nombre del nuevo archivo
            new_filename = f"{raster_path.stem}_reprojected{raster_path.suffix}"
            new_path = raster_path.parent / new_filename

            # Realizar la reproyección
            with rasterio.open(new_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )

            print(f"Raster reproyectado y guardado en: {new_path}")
            return new_path

    def get_predios_gdf(self, predios_path: Path) -> gpd.GeoDataFrame:

        elev_file = self.validate_and_update_raster_crs(config.ELEV_FILE, config.PROJECT_CRS)

        # 1. Elevation Setup
        elev_source = ElevationSource(str(self.project_root), config.PROJECT_CRS)
        elev_tree = elev_source.get_elev_source(
            [str(elev_file)],
            check_unique_values=False,
            ellipsoidal2orthometric=False,
            m_ramales=None,
            elevation_shift=0
        )
        getter = ElevationGetter(tree=elev_tree, m_ramales=None, threshold_distance=0.7)

        # 2. Load Predios
        self.predios_gdf = gpd.read_file(predios_path, engine='pyogrio')
        self.predios_gdf = self.ensure_crs(gdf=self.predios_gdf, target_crs=config.PROJECT_CRS)
        coords = self.predios_gdf.geometry.centroid.get_coordinates().to_numpy()
        self.predios_gdf['z'] = getter.get_elevation_from_tree_coords(coords)

        # 2.1 Calculate Slopes & Filter
        slope_calc = PredioSlopeCalculator(str(elev_file))
        self.predios_gdf = slope_calc.calculate_slopes(self.predios_gdf)

        # 1. Filter by Slope
        max_slope = config.MAX_PREDIO_SLOPE
        valid_mask = (self.predios_gdf['mean_slope_pct'] <= max_slope) | (self.predios_gdf['mean_slope_pct'].isna())
        self.predios_gdf = self.predios_gdf[valid_mask].copy().reset_index(drop=True)
        self.predios_gdf.reset_index(drop=True, inplace=True)
        self.predios_gdf['id'] = self.predios_gdf.index.astype(str)

        if self.predios_gdf.empty:
            sys.exit("  [Error] No predios meet the slope criteria.")

        return self.predios_gdf


    def calculate_scores(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculates weighted scores for flooding nodes based on available metrics.
        Normalizes all values to 0-1 range before weighting.
        
        Metrics used:
        - total_volume: Higher is better (flooding + over capacity)
        - total_flow: Higher is better (flooding + over capacity)
        - failure_prob: Higher is better (risk)
        """
        self.weights = config.FLOODING_RANKING_WEIGHTS
        if gdf.empty:
            gdf['score'] = 0.0
            return gdf
        
        df = gdf.copy()
        
        # --- Normalization (Min-Max) ---
        def normalize(series):
            if series.max() == series.min():
                return np.zeros(len(series))
            return (series - series.min()) / (series.max() - series.min())
        
        # Normalize available metrics
        df['norm_vol'] = normalize(df['total_volume'])
        df['norm_flow'] = normalize(df['total_flow'])
        df['norm_risk'] = normalize(df['failure_prob'])
        
        # Calculate weighted score (only with available metrics)
        score = (
            self.weights.get('total_volume', 0.4) * df['norm_vol'] +
            self.weights.get('total_flow', 0.3) * df['norm_flow'] +
            self.weights.get('failure_probability', 0.3) * df['norm_risk']
        )
        
        df['score'] = score
        
        # Clean up temporary columns
        df.drop(columns=['norm_vol', 'norm_flow', 'norm_risk'], inplace=True)
        
        return df.sort_values(by='score', ascending=False)
    

    @staticmethod
    def get_candidate_pairs(
                            nodes_gdf: gpd.GeoDataFrame,
                            predios_gdf: gpd.GeoDataFrame,
                            max_workers: int = None
                            ) -> List[CandidatePair]:
        """
        Versión Multinúcleo (Separada): Prepara los datos vectorialmente y
        reparte la creación de objetos en el pool de hilos usando una función externa.
        """
        print(f"[Task] Comparando {len(nodes_gdf)} nodos vs {len(predios_gdf)} predios...")

        # 1. Preparación Vectorizada (Filtrado instantáneo)
        node_invert = nodes_gdf['InvertElevation'].values
        node_surface_z = node_invert + nodes_gdf['NodeDepth'].values
        node_x = nodes_gdf.geometry.x.values
        node_y = nodes_gdf.geometry.y.values

        predio_z = predios_gdf['z'].values
        predio_x = predios_gdf.geometry.centroid.x.values
        predio_y = predios_gdf.geometry.centroid.y.values

        mask = predio_z[np.newaxis, :] <= node_invert[:, np.newaxis]
        i_indices, j_indices = np.where(mask)
        total_tasks = len(i_indices)

        if total_tasks == 0:
            return []

        # Convertimos a diccionarios para el pool
        nodes_list = nodes_gdf.iloc[i_indices].to_dict('records')
        predios_list = predios_gdf.iloc[j_indices].to_dict('records')

        # Preparar lista de tareas para el executor
        tasks = []
        for idx in range(total_tasks):
            tasks.append({
                'node': nodes_list[idx],
                'predio': predios_list[idx],
                'j': int(j_indices[idx]),
                'node_surface_z': node_surface_z[i_indices[idx]],
                'start_xy': (node_x[i_indices[idx]], node_y[i_indices[idx]]),
                'end_xy': (predio_x[j_indices[idx]], predio_y[j_indices[idx]]),
                'gap': node_invert[i_indices[idx]] - predio_z[j_indices[idx]]
            })

        # 2. Ejecución Paralela usando la función externa
        workers = max_workers or int(os.cpu_count() / 2.0)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Usamos map con la función standalone _instantiate_candidate_pair
            valid_pairs = list(tqdm(executor.map(_instantiate_candidate_pair, tasks),
                                    total=total_tasks, desc="Cargando candidatos"))

        return valid_pairs

    def run(self, inp_file_path: str = None):
        """
        Single run method: Extracts flooding metrics, maps spatial risk, and ranks candidates.
        Populates entity attributes instead of returning values.
        """
        # Extract flooding metrics
        metrics = self.extract(inp_file_path, {})
        self.swmm_gdf = metrics.swmm_gdf
        self.nodes_gdf = metrics.flooded_nodes_gdf

        print(f"[Flooding Metrics] Mapping spatial risk data from {self.risk_file_path}...")
        # Add Risk Data
        metrics.flooded_nodes_gdf = self.get_spatial_risk(metrics.flooded_nodes_gdf)

        if self.nodes_gdf is None or self.predios_gdf is None:
            sys.exit("Error: Data not loaded.")

        # # Identify Valid Candidate Pairs
        # n_nodes = len(self.nodes_gdf)
        # n_predios = len(self.predios_gdf)
        # print(f"Comparing {n_nodes} nodes against {n_predios} predios...")
        # valid_pairs = self.get_candidate_pairs(self.nodes_gdf, self.predios_gdf)
        #
        # n_valid = len(valid_pairs)
        # if n_valid == 0:
        #     sys.exit("  [Error] No valid candidate pairs found based on elevation criteria.")

        # Use WeightedCandidateSelector for consistent scoring
        # selector = WeightedCandidateSelector(self.nodes_gdf, self.predios_gdf)
        df = self.calculate_scores(self.nodes_gdf)

        # Sort by Score (Desc) and then Distance (Asc) as tie-breaker
        df = df.sort_values(by=['score'], ascending=[False])

        # Convert back to list of dicts and store in entity
        self.ranked_candidates = df.to_dict('records')
        self.metrics = metrics
        
        print("\n" + "="*100)
        print("CANDIDATOS PARA TANQUE DE TORMENTA")
        print("="*100)
        cols = ['node_id', 'total_flow', 'total_volume', 'failure_prob', 'score']
        headers = ['ID Nodo', 'Caudal (m³/s)', 'Vol', 'Prob. Falla', 'score']
        print(tabulate(
            df[cols].head(5),
            headers=headers,
            tablefmt='grid',
            floatfmt=('.0f', '.2f', '.1f', '.2f', '.3f'),
            showindex=False,
            numalign='right',
            stralign='left'
        ))
        print("="*100 + "\n")

  

if __name__ == "__main__":
    runner = MetricExtractor(
                            project_root=config.PROJECT_ROOT,
                            predios_path=config.PREDIOS_FILE)

    # Run everything in one call (no return values)
    runner.run(config.SWMM_FILE)

    # Access results as entity attributes
    print(f"Ranked candidates: {len(runner.ranked_candidates)}")
    print(f"Total flooding volume: {runner.metrics.total_flooding_volume}")
