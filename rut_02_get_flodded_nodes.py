"""
rut_02a_prioritize_urgency.py
-----------------------------
Systematic prioritization of stormwater network interventions.
Ranks nodes based on Volume, Flow, and Risk.
Adapts metric calculation based on config.TANK_OPT_OBJECTIVE ('capacity' or 'flooding').
"""

import os
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
from tqdm import tqdm
from pyswmm import Output, Simulation
from swmm.toolkit.shared_enum import NodeAttribute, LinkAttribute
from swmm.toolkit.shared_enum import NodeAttribute, LinkAttribute
from scipy.spatial.distance import cdist
import rasterio
from rasterio import features
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from shapely.geometry import Point, Polygon, LineString


# Project imports
import config
from rut_06_pipe_sizing import CapacidadMaximaTuberia
from rut_25_from_inp_to_vector import NetworkExporter

# Suppress warnings
warnings.filterwarnings("ignore")



from dataclasses import dataclass
from typing import Tuple, Optional




@dataclass
class CandidatePair:
    """
    Represents a candidate assignment of a node to a predio.
    Shared data structure for ranking and dynamic evaluation.
    """
    # --- Identificadores y Metadatos ---
    node_id: str           # Nombre/ID del nodo (ej. J-123)
    predio_id: str         # Nombre/ID del predio (ej. P-456)
    
    # --- Datos del Nodo de derivacion ---
    node_volume_flood: float = 0.0    # Volumen de inundación en el nodo (m3)
    node_max_flow: float = 0.0        # Caudal pico de inundación (m3/s)
    node_max_depth: float = 0.0       # H_max del nodo original
    node_z_invert: float = 0.0        # Elevación batea del nodo (m)
    node_z_surface: float = 0.0     # Elevación superficie del nodo (m)
    node_x: float = 0.0             # Coordenada X del nodo (m)
    node_y: float = 0.0             # Coordenada Y del nodo (m)
    node_probability_failure: float =0.0
    node_flow_over_capacity: float = 0.0 # Caudal sobre capacidad de tubería (m3/s)
    node_flooding_flow: float = 0.0       # Caudal de inundación del nodo (m3/s)
    node_volume_over_capacity: float = 0.0 # Volumen sobre capacidad de tubería (m3)
    node_volume_flooding: float = 0.0  # Volumen de inundación del nodo (m3)
    node_geometry: Optional[Point] = None # Point con las coordenadas del nodo
    node_distance_to_predio: float = 0.0 # Distancia euclidiana al predio (m)
    node_elevation_gap_to_predio: float = 0.0    # Desnivel entre nodo y predio (m)
    
    # --- Tuberia ---
    derivation_link: str = ""         # ID del tramo que más aporta a la inundación
    derivation_link_geometry: Optional[LineString] = None
    derivation_target_node_geometry: Optional[Point] = None
    
    # --- Datos del Predio  ---
    predio_area_m2: float = 0.0
    predio_geometry: Optional[Polygon] = None
    predio_x_centroid: float = 0.0
    predio_y_centroid: float = 0.0
    predio_ground_z: float = 0.0      # Elevación del terreno
    
    # ---  Tanque  ---
    tank_volume_simulation: float = 0.0 # Volumen capturado en simulación post-ejecución
    tank_max_depth: float = 0.0       # Profundidad de diseño (por defecto en config)
    
    # --- Datos de Tubería (de rut_03 diseño) ---
    diameter: str = 'N/A'              # Diámetro de tubería diseñada (ej. "1.2" o "1.5x1.2")
    pipeline_length: float = 0.0       # Longitud total de tubería (m)
    
    # --- Costos Estimados (Para rut_17 Dashboard) ---
    cost_link: float = 0.0
    cost_tank: float = 0.0
    cost_land: float = 0.0
    
    # --- Contexto del Árbol (Tree Routing) ---
    is_tank: bool = True              # ¿Se conecta a un tanque o es solo una tubería?
    target_id: str = ""               # ID del nodo/tanque al que finalmente descarga
    
    # --- Propiedades Calculadas ---
    @property
    def total_cost(self) -> float:
        return self.cost_pipeline + self.cost_tank + self.cost_land




class FloodingMetrics:
    """Calculates flooding metrics for nodes in a SWMM model."""

    def __init__(self, inp_file_path, risk_file_path):
        #swwm file path
        self.inp_file_path = inp_file_path
        #risk geopackage file path
        self.risk_file_path = risk_file_path

        self.at_capacity_flow = CapacidadMaximaTuberia()
        self.parse_shape_from_swmm = {'RECT_CLOSED': 'rectangular','RECT_OPEN': 'rectangular' , 'MODBASKETHANDLE': 'rectangular', 'CIRCULAR': 'circular'}

    @staticmethod
    def parse_seccion_to_pypiper( conduit_row: pd.Series) -> str:
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

    def load_swmm_model(self):
        in_file_path = str(self.inp_file_path)

        self.exporter = NetworkExporter(str(in_file_path))
        swmm_gdf = self.exporter.run(None, crs=config.PROJECT_CRS)

        swmm_gdf['Seccion'] = swmm_gdf['Shape'].map(self.parse_shape_from_swmm)
        swmm_gdf['D_int'] = swmm_gdf.apply(self.parse_seccion_to_pypiper, axis=1)

        q_at_capacity, v_at_capacity, h_at_capacity = self.at_capacity_flow.calcular_capacidad_maxima(
            D_int=swmm_gdf['D_int'].to_numpy().astype(str),
            S=np.where(swmm_gdf['Slope'] < 0, 0.01, swmm_gdf['Slope']),
            Rug=swmm_gdf['Roughness'].to_numpy().astype(float),
            Seccion=swmm_gdf['Seccion'].to_numpy().astype(str),
            h_D_objetivo=config.CAPACITY_MAX_HD
        )

        self.q_at_capacity_series = pd.Series(np.round(q_at_capacity / 1000, 3), index=swmm_gdf.index)

        swmm_gdf['flow_over_pipe_capacity'] = np.where(
            swmm_gdf['MaxFlow'] - self.q_at_capacity_series < 0,
            0,
            swmm_gdf['MaxFlow'] - self.q_at_capacity_series
        )
        swmm_gdf['flow_pipe_capacity'] = self.q_at_capacity_series


        return swmm_gdf

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
    def is_original_node(nid):
        """Check if node is part of original network (not tank or derivation)"""
        # Tanks start with TK_
        if nid.startswith('TK_'):
            return False
        # Derivation nodes are format: 0.0, 0.1, 1.0, etc.
        parts = nid.split('.')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return False
        return True

    def extract_metrics(self):
        out_file_path = str(self.exporter.out_path_file)

        # Normalizar inlet nodes una sola vez
        inletnodes_series = self.swmm_gdf['InletNode'].astype(str).str.strip().str.upper()
        results_list = []
        with Output(out_file_path) as out:
            # Precalcular listas de nodos/links para evitar múltiples iteraciones
            all_nodes = list(out.nodes)
            original_nodes = [nid for nid in all_nodes if self.is_original_node(nid)]

            # =====================================================================
            # 3. PROCESAR NODOS ORIGINALES (un solo loop)
            # =====================================================================
            for nid in tqdm(original_nodes, desc="Extracting Nodes Flooding Metrics"):
                nid_str = str(nid).strip().upper()

                incoming_links = inletnodes_series[inletnodes_series == nid_str].index.tolist()
                
                # --- Calcular flujo entrante si es necesario ---
                if config.TANK_OPT_OBJECTIVE == 'capacity':
                    if len(incoming_links) > 0:
                        # Encontrar el link con el máximo exceso
                        excesses = self.swmm_gdf['flow_over_pipe_capacity'].loc[incoming_links]
                        link_with_max_excess = excesses.idxmax()
                        flow_over_pipe_capacity_incoming = float(excesses.max())

                        # Oftener la serie de flujo y la capacidad del link con max exceso
                        flow_incoming_series = pd.Series(out.link_series(link_with_max_excess, LinkAttribute.FLOW_RATE))
                        capacity_incoming = float(self.q_at_capacity_series.loc[link_with_max_excess])

                    else:
                        flow_over_pipe_capacity_incoming = 0.0
                        flow_incoming_series = pd.Series(dtype=float)  # Serie vacía por defecto
                        capacity_incoming = 0.0
                else:
                        flow_over_pipe_capacity_incoming = 0.0
                        flow_incoming_series = pd.Series(dtype=float)  # Serie vacía por defecto
                        capacity_incoming = 0.0

                # --- Obtener series de flooding ---
                flooding_series = pd.Series(out.node_series(nid, NodeAttribute.FLOODING_LOSSES))
                flooding_peak = float(flooding_series.max()) if len(flooding_series) > 0 else 0.0

                # --- Calcular volumen de flooding del nodo ---
                vol_node_flooding = self._calculate_volume_from_series(
                    flooding_series,
                    threshold=config.MINIMUN_FLOODING_FLOW
                )

                # --- Calcular volumen sobre capacidad ---
                vol_over_pipe_capacity = 0.0
                if flow_over_pipe_capacity_incoming > config.MINIMUN_FLOODING_FLOW and len(flow_incoming_series) > 0:
                    # Corregido: restar la capacidad real en lugar del exceso
                    flow_excess_series = flow_incoming_series - capacity_incoming
                    flow_excess_series = flow_excess_series.clip(lower=0)
                    vol_over_pipe_capacity = self._calculate_volume_from_series(
                        flow_excess_series,
                        threshold=0
                    )

                # --- Almacenar resultados si hay flooding o exceso ---
                vol_total = vol_node_flooding + vol_over_pipe_capacity

                if vol_total > 0:
                    # Obtener profundidad máxima
                    depth_series = pd.Series(out.node_series(nid, NodeAttribute.INVERT_DEPTH))
                    node_depth = float(depth_series.max()) if len(depth_series) > 0 else 0.0

                    # Coordinates
                    x_coord =self.swmm_gdf.loc[incoming_links]['X2'].max()
                    y_coord = self.swmm_gdf.loc[incoming_links]['Y2'].max()
                    geom = Point(x_coord, y_coord)
                    invert_elevation = self.swmm_gdf.loc[incoming_links]['OutletNode_InvertElev'].max()
                    results_list.append({
                        'NodeID': nid,
                        'x': x_coord,
                        'y': y_coord,
                        'flow_over_capacity': flow_over_pipe_capacity_incoming,
                        'flow_node_flooding': flooding_peak,
                        'total_flow': flow_over_pipe_capacity_incoming + flooding_peak,
                        'vol_over_capacity': vol_over_pipe_capacity,
                        'vol_node_flooding': vol_node_flooding,
                        'total_volume': vol_total,
                        'NodeDepth': node_depth,
                        'InvertElevation':invert_elevation,
                        'geometry': geom,
                    })

        gdf_results = gpd.GeoDataFrame(results_list, geometry='geometry', crs=config.PROJECT_CRS)
        return gdf_results

    def get_spatial_risk(self, df):
        """Maps risk data to nodes using spatial proximity (nearest neighbor)."""
        if not self.risk_file_path or not Path(self.risk_file_path).exists() or df.empty:
            df['FailureProbability'] = 0
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
            node_coords = df[['x', 'y']].to_numpy()

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
                df['FailureProbability'] = np.where(min_dists < 0.1, nearest_probs, 0.0)
            else:
                print("Warning: Coordinate mismatch or empty geometries for risk/nodes.")
                df['FailureProbability'] = 0.0
        else:
            print("Warning: Risk GPKG empty or missing 'failure_prob' column.")
            df['FailureProbability'] = 0.0


        return df

    @staticmethod
    def save_results(gdf, output_dir):
        """
        Saves results to Excel (.xlsx) and GeoPackage (.gpkg) if output_dir is provided.
        """
        if output_dir is None or gdf.empty:
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        base_name = "prioritized_nodes"

        # 1. Save GeoPackage (preserves geometry)
        gpkg_file = out_path / f"{base_name}.gpkg"
        try:
            gdf.to_file(gpkg_file, driver="GPKG")
            print(f"[Flooding Metrics] Saved GPKG: {gpkg_file}")
        except Exception as e:
            print(f"[Flooding Metrics] Error saving GPKG: {e}")

        # 2. Save Excel (strips geometry for cleaner table)
        xlsx_file = out_path / f"{base_name}.xlsx"
        try:
            # Convert to standard pandas DataFrame and drop geometry column
            df_excel = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
            df_excel.to_excel(xlsx_file, index=False)
            print(f"[Flooding Metrics] Saved Excel: {xlsx_file}")
        except Exception as e:
            print(f"[Flooding Metrics] Error saving Excel: {e}")

    def run(self, output_dir=None):
        print(f"[Flooding Metrics] 1. Parsing SWMM model and calculating pipe capacities from {self.inp_file_path}...")
        # parse swmm model
        self.swmm_gdf = self.load_swmm_model()

        print("[Flooding Metrics] 2. Extracting flooding metrics (volume, flow, depth) from simulation results...")
        # extract flooding metrics
        gdf = self.extract_metrics()

        print(f"[Flooding Metrics] 3. Mapping spatial risk data from {self.risk_file_path}...")
        # --- Add Risk Data ---
        gdf = self.get_spatial_risk(gdf)

        # Sort by Volume -> Flow -> Risk
        # Prioritize where the most water is, then where it's flowing fastest, then risk.
        if not gdf.empty:
            gdf = gdf.sort_values(
                by=['total_volume', 'total_flow', 'FailureProbability'],
                ascending=[False, False, False]
            )

        print(f"[Flooding Metrics] Analysis complete. Resulting DataFrame shape: {gdf.shape}")

        # Save results if path is provided
        if output_dir:
            self.save_results(gdf, output_dir)

        return gdf

class TankValidator:
    """
    Centralizes all tank validation logic.
    Applies volume, area, and configuration constraints.
    Supports finding alternative predios if the primary one is full.
    """
    
    def __init__(self, predios_gdf, nodes_gdf):
        self.predios_gdf = predios_gdf
        self.nodes_gdf = nodes_gdf
        
        # Load constraints from config
        self.MIN_VOLUME = config.TANK_MIN_VOLUME_M3
        self.MAX_VOLUME = config.TANK_MAX_VOLUME_M3
        self.TANK_DEPTH = config.TANK_DEPTH_M
        self.OCCUPATION_FACTOR = config.TANK_OCCUPATION_FACTOR
        
        # Track used area per predio
        self._predio_used_area = {} # predio_idx -> used_area_m2
        self.rejection_log = []
    
    def reset(self):
        """Reset tracking for a new evaluation."""
        self._predio_used_area = {}
        self.rejection_log = []

    def _cabe_en_predio(self, predio_idx: int, volume: float) -> Tuple[bool, str, float]:
        """Checks if a tank of given volume fits in a predio."""
        predio = self.predios_gdf.iloc[predio_idx]
        predio_area = round(predio.geometry.area,2)
        
        # Cap volume to MAX
        adjusted_volume = min(volume, self.MAX_VOLUME)
        tank_footprint = adjusted_volume / self.TANK_DEPTH
        required_area = tank_footprint * self.OCCUPATION_FACTOR
        
        already_used_area = self._predio_used_area.get(predio_idx, 0.0)
        available = predio_area - already_used_area
        
        if required_area > available:
            return False, f"Insufficient area: needs {required_area:.1f}m2, avail {available:.1f}m2", 0
            
        return True, "OK", adjusted_volume


    def validate_and_reserve(self, original_predio_idx: int, volume: float) -> Tuple[bool, str, int, float]:
        """
        Validates an assignment and reserves the area if valid.
        
        NOTE: Re-assignment logic DISABLED to avoid conflicts with rut_15's
        predio capacity tracking. If primary predio is full, assignment is rejected.
        rut_15 handles predio selection and re-pairing.
        
        Returns: (success, reason, final_predio_idx, final_volume)
        """
        # BASIC VALIDATION
        if volume < self.MIN_VOLUME:
            return False, f"Volume {volume:.1f} < MIN {self.MIN_VOLUME}", -1, 0
            
        fits, reason, adj_vol = self._cabe_en_predio(original_predio_idx, volume)
        
        if fits:
            self._predio_used_area[original_predio_idx] = self._predio_used_area.get(original_predio_idx, 0.0) + (adj_vol / self.TANK_DEPTH * self.OCCUPATION_FACTOR)
            return True, "OK", original_predio_idx, adj_vol
        
        # REJECT - let rut_15 handle re-pairing
        return False, reason, -1, 0


class PredioSlopeCalculator:
    """
    Calculates the average slope for each predio using a vectorized approach.
    """

    def __init__(self, elev_raster_path):
        self.elev_raster_path = elev_raster_path

    def calculate_slopes(self, predios_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculates the mean slope (%) for each predio polygon.

        Args:
            predios_gdf: GeoDataFrame containing predio polygons.

        Returns:
            GeoDataFrame with a new 'mean_slope_pct' column.
        """
        if predios_gdf.empty:
            predios_gdf['mean_slope_pct'] = 0.0
            return predios_gdf

        try:
            with rasterio.open(self.elev_raster_path) as src:
                # 1. Read DEM data
                # We read the first band
                Z = src.read(1)
                transform = src.transform
                cell_size_x = transform[0]
                cell_size_y = -transform[4]  # Usually negative in standard geo-tiffs

                # 2. Compute Slope (Vectorized gradient)
                # np.gradient returns [gradient_y, gradient_x]
                grad_y, grad_x = np.gradient(Z, cell_size_y, cell_size_x)

                # Slope in %: sqrt(dx^2 + dy^2) * 100
                slope_pct = np.sqrt(grad_x ** 2 + grad_y ** 2) * 100

                # Handling NoData (if any exist in Z, they might affect slope)
                if src.nodata is not None:
                    mask_nodata = (Z == src.nodata)
                    slope_pct[mask_nodata] = 0

                # 3. Rasterize Predios (Vectorized Mask)
                # Create an ID grid effectively mapping pixels to predio indices
                # We use index + 1 because 0 is usually background
                shapes = ((geom, idx + 1) for idx, geom in enumerate(predios_gdf.geometry))

                # Ensure we match the raster shape and transform
                id_grid = features.rasterize(
                    shapes=shapes,
                    out_shape=Z.shape,
                    transform=transform,
                    fill=0,
                    dtype=np.int32
                )

                # 4. Aggregate using bincount (Super fast)
                # Flatten arrays
                ids_flat = id_grid.ravel()
                slopes_flat = slope_pct.ravel()

                # Determine max ID to set bin count size
                max_id = ids_flat.max()

                # Sum of slopes per ID
                slope_sums = np.bincount(ids_flat, weights=slopes_flat, minlength=max_id + 1)

                # Count of pixels per ID
                pixel_counts = np.bincount(ids_flat, minlength=max_id + 1)

                # 5. Calculate means
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    slope_means = slope_sums / pixel_counts

                # slope_means[0] is background, we ignore it.
                # Predio i corresponds to slope_means[i+1]

                # Map back to GDF
                # We initialized indices 0 to len-1.
                # Result array indices 1 to len.

                # Create a Series for mapping, fill missing with 0
                # We only care about indices 1 to max_id present in the grid
                # Note: Some predios might not overlap raster or be too small for a pixel
                # They will have pixel_count 0.

                # Safe access
                n_predios = len(predios_gdf)
                results = np.zeros(n_predios)

                for idx in range(n_predios):
                    bin_idx = idx + 1
                    if bin_idx <= max_id and pixel_counts[bin_idx] > 0:
                        results[idx] = slope_means[bin_idx]
                    else:
                        results[idx] = 0.0  # Or NaN if preferred

                predios_gdf['mean_slope_pct'] = results

                print(f"[PredioSlope] Calculated slopes for {len(predios_gdf)} predios .")

        except Exception as e:
            print(f"[PredioSlope] Error calculating slopes: {e}")
            import traceback
            traceback.print_exc()
            predios_gdf['mean_slope_pct'] = 0.0

        return predios_gdf

