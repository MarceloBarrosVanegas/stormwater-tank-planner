"""
rut_20_avoided_costs.py
========================
Economic Impact Assessment Module.

Defines two main evaluators:
1. VolumeBasedEvaluator: Simple proxy cost ($/m3).
2. ComprehensiveEconomicEvaluator: Aggregates real physical costs and benefits.
   - FloodDamageCost (Climada/Itzi)
   - DeferredInvestmentCost (Capacity Savings)
   - TrafficDisruptionCost
   - PavementDegradationCost
   - ErosionControlCost
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import config
config.setup_sys_path()




import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import subprocess
import os


# Import construction cost calculator from rut_21
from rut_21_construction_cost import SewerConstructionCost
from rut_25_from_inp_to_vector import NetworkExporter
from rut_22_scenario_generator import generate_inp_file
import rut_22_scenario_generator as rut_22
from rut_21_risk_analysis import RiskAnalyzer


# Import pipe sizing module from pypiper
from rut_06_pipe_sizing import SeccionParcialmenteLlena, PipeSizing
from rut_00_app_config import geometry_vg
from RUT_1 import varGlobals
vg = varGlobals()


# =============================================================================
# ABSTRACT INTERFACE
# =============================================================================

class EconomicModel(ABC):
    """Abstract base for economic assessment strategies."""
    
    @abstractmethod
    def calculate_economic_impact(self, 
                                  baseline_metrics: Any, 
                                  solution_metrics: Any, 
                                  **kwargs) -> Dict[str, Any]:
        """
        Returns a dict containing:
        - 'net_impact_usd': Total economic cost (Damage - Benefits)
        - 'details': Dictionary with breakdown
        """
        pass

# =============================================================================
# 1. VOLUME BASED
# =============================================================================

class VolumeBasedEvaluator(EconomicModel):
    """
    Calculates cost purely based on remaining flood volume.
    Cost = Volume * Rate.
    This does NOT consider avoided infrastructure costs.
    """
    
    def __init__(self, cost_per_m3: float = 1250.0):
        self.cost_per_m3 = cost_per_m3

    def calculate_economic_impact(self, baseline_metrics: Any, solution_metrics: Any, **kwargs) -> Dict[str, Any]:
        vol = getattr(solution_metrics, 'total_flooding_volume', 0.0)
        cost = vol * self.cost_per_m3
        
        return {
            "net_impact_usd": cost,
            "details": {
                "method": "VolumeBased",
                "volume_m3": vol,
                "rate": self.cost_per_m3,
                "damage_cost": cost,
                "benefits_usd": 0.0
            }
        }


# =============================================================================
# 2. COMPREHENSIVE / REAL EVALUATOR (Aggregator)
# =============================================================================

class BaseStrategy(ABC):
    @abstractmethod
    def calculate(self, base, sol, **kwargs) -> Dict[str, Any]:
        """Returns {'value': float, 'metadata': dict}"""
        pass


class FloodDamage:
    """
    Calcula daño por inundación usando CLIMADA.
    Requiere 'climada_result' en kwargs.
    Si no tienes CLIMADA, usa VolumeBasedEvaluator.
    """
    def __init__(self):
        pass  # No parameters needed

    def calculate(self, base, sol, **kwargs) -> Dict[str, Any]:
        climada_res = kwargs.get('climada_result')
        if not climada_res or 'total_damage_usd' not in climada_res:
            raise ValueError(
                "FloodDamageReal requiere 'climada_result' en kwargs. "
                "Ejecuta ITZI/CLIMADA primero, o usa VolumeBasedEvaluator."
            )
        
        return {
            "value": climada_res['total_damage_usd'],
            "metadata": {"source": "CLIMADA"}
        }
        
    def run_climada_standalone(self, depth_raster_path: str, output_dir: str) -> Dict:
        """Helper to invoke CLIMADA logic on demand."""
        from rut_19_flood_damage_climada import calculate_flood_damage_climada
        return calculate_flood_damage_climada(
            depth_raster_path=Path(depth_raster_path),
            output_gpkg=Path(output_dir) / "flood_damage.gpkg",
            output_txt=Path(output_dir) / "damage_report.txt"
        )


class FloodDamageFromItzi:
    """
    Calcula daño por inundación ejecutando ITZI (2D) + CLIMADA.
    
    Workflow (run method):
    1. Ejecuta ITZI desde GRASS Shell (run_itzi_from_conda.py)
    2. Lee el raster max_water_depth.tif resultante
    3. Calcula daño con CLIMADA (rut_19_flood_damage_climada.py)
    4. Retorna daño total en USD
    
    Similar a DeferredInvestmentCost pero para daños de inundación.
    Guarda resultados en: output_dir/avoided_cost/flood_damage/
    """
    
    def __init__(self, itzi_vars: str = 'depth,v', crs=config.PROJECT_CRS):
        self.itzi_vars = itzi_vars
        self.crs = crs
        
        # Resultados
        self.itzi_result = {}
        self.climada_result = {}
        self.total_damage = 0.0
    
    def run(self, 
            inp_path: str = None, 
            output_dir: str = None,
            predios_path: str = None) -> Dict[str, Any]:
        """
        Ejecuta ITZI + CLIMADA y devuelve el resultado completo.
        
        Args:
            inp_path: Ruta al archivo .inp de SWMM (opcional, usa config)
            output_dir: Directorio base de salida (se crea avoided_cost/flood_damage/)
            predios_path: Ruta al GPKG de predios (opcional)
        
        Returns:
            Dict con 'total_damage_usd', 'damage_by_sector', 'output_dir', etc.
        """
        # --- Crear estructura de carpetas ---
        if not output_dir:
            raise ValueError("output_dir es requerido para guardar los resultados de flood damage")
        
        base_dir = Path(output_dir) / "avoided_cost" / "flood_damage"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        case_name = Path(output_dir).name
        print(f"  [FloodDamageFromItzi] Output dir: {base_dir}")
        
        # --- 1. Ejecutar ITZI ---
        print(f"  [FloodDamageFromItzi] Ejecutando simulación ITZI 2D...")
        self.itzi_result = self._run_itzi(inp_path=inp_path, output_dir=str(base_dir))
        
        if not self.itzi_result.get('success'):
            error_msg = self.itzi_result.get('error', 'Unknown error')
            print(f"  [FloodDamageFromItzi] Error ITZI: {error_msg}")
            return {"error": f"ITZI failed: {error_msg}", "total_damage_usd": 0.0}
        
        print(f"  [FloodDamageFromItzi] ✓ ITZI completado")
        
        # --- 2. Obtener raster de profundidad ---
        depth_raster = self.itzi_result.get('max_water_depth_raster')
        if not depth_raster or not Path(depth_raster).exists():
            # Fallback to default ITZI output
            depth_raster = config.ITZI_OUTPUT_DIR / "max_water_depth.tif"
        
        if not Path(depth_raster).exists():
            return {"error": f"Depth raster not found: {depth_raster}", "total_damage_usd": 0.0}
        
        print(f"  [FloodDamageFromItzi] Raster de profundidad: {depth_raster}")
        
        # --- 3. Ejecutar CLIMADA ---
        print(f"  [FloodDamageFromItzi] Calculando daño con CLIMADA...")
        self.climada_result = self._run_climada(
            depth_raster_path=str(depth_raster),
            output_dir=str(base_dir),
            predios_path=predios_path,
            case_name=case_name
        )
        
        if 'error' in self.climada_result:
            print(f"  [FloodDamageFromItzi] Error CLIMADA: {self.climada_result['error']}")
            return self.climada_result
        
        self.total_damage = self.climada_result['total_damage_usd']
        
        print(f"  [FloodDamageFromItzi] Daño total calculado: ${self.total_damage:,.0f} USD")
        print(f"  [FloodDamageFromItzi] Resultados guardados en: {base_dir}")
        
        # Agregar output_dir al resultado
        self.climada_result['output_dir'] = str(base_dir)
        
        return self.climada_result
    
    def generate_visualizations(self, output_dir: str):
        """Generates plots using rut_19.FloodDamagePlotter."""
        if not self.climada_result:
            print("  [Warning] No CLIMADA results to plot.")
            return

        # Use standard network path from config
        network_gpkg = config.NETWORK_FILE

        # Mostrar resultados
        if "error" not in self.climada_result:
            print(f"\n{'=' * 60}")
            print("RESULTADOS FINALES")
            print(f"{'=' * 60}")
            print(f"  Daño total:              ${self.climada_result['total_damage_usd']:,.0f} USD")
            print(f"  Propiedades inundadas:   {self.climada_result.get('flooded_properties', 'N/A'):,}")
            print(f"  Total propiedades:       {self.climada_result.get('total_properties', 'N/A'):,}")
            print(f"\n  Archivos generados:")
            print(f"    - GPKG: {self.climada_result.get('output_gpkg', 'N/A')}")
            print(f"    - Report: {self.climada_result.get('output_report', 'N/A')}")

            # -------------------------------------------------------------
            # GENERAR VISUALIZACIONES
            # -------------------------------------------------------------
            try:
                from rut_19_flood_damage_climada import FloodDamagePlotter

                plotter = FloodDamagePlotter(self.climada_result, network_path=network_gpkg)
                output_dir = plotter.plot_all()
                print(f"\n  Visualizaciones generadas en: {output_dir}")

                print(f"\n All plots saved to: {output_dir}")
            except Exception as e:
                print(f"\n  ERROR generando visualizaciones: {e}")
                import traceback
                traceback.print_exc()

            


    def _run_itzi(self, inp_path: str = None, output_dir: str = None) -> Dict:
        """
        Ejecuta ITZI via GRASS Shell subprocess.
        
        Lógica integrada de run_itzi_from_conda.py:
        - Llama directamente a GRASS Shell (grass84.bat)
        - Limpia el entorno para evitar conflictos Conda/GRASS
        
        Args:
            inp_path: Ruta al archivo .inp de SWMM
            output_dir: Directorio de salida para ITZI
        """
        script_dir = Path(__file__).parent
        run_itzi_script = script_dir / "run_itzi.py"
        grass_bat = r"C:\Program Files\GRASS GIS 8.4\grass84.bat"
        
        if not Path(grass_bat).exists():
            return {"success": False, "error": f"GRASS not found: {grass_bat}"}
        
        if not run_itzi_script.exists():
            return {"success": False, "error": f"Script not found: {run_itzi_script}"}
        
        # Build command: grass84.bat --exec python run_itzi.py --vars depth,v --swmm X --output Y
        cmd = [grass_bat, "--exec", "python", str(run_itzi_script)]
        if self.itzi_vars:
            cmd.extend(["--vars", self.itzi_vars])
        if inp_path:
            cmd.extend(["--swmm", str(inp_path)])
        if output_dir:
            cmd.extend(["--output", str(output_dir)])
        
        print(f"  [FloodDamageFromItzi] GRASS: {grass_bat}")
        print(f"  [FloodDamageFromItzi] Script: {run_itzi_script}")
        print(f"  [FloodDamageFromItzi] Command: {' '.join(cmd)}")
        
        # Clean environment to avoid conda/GRASS conflicts
        env = os.environ.copy()
        
        # Remove conda paths and invalid entries from PATH
        path_parts = env.get('PATH', '').split(os.pathsep)
        clean_path = []
        for p in path_parts:
            # Skip conda paths
            if 'miniconda' in p.lower() or 'anaconda' in p.lower():
                continue
            # Skip empty or just dots (causes WinError 87)
            if not p or p == '.' or p == '..':
                continue
            clean_path.append(p)
        env['PATH'] = os.pathsep.join(clean_path)
        
        # Remove conda environment variables
        env.pop('CONDA_PREFIX', None)
        env.pop('CONDA_DEFAULT_ENV', None)
        
        # Force GRASS PROJ and GDAL paths
        grass_share = r"C:\Program Files\GRASS GIS 8.4\share"
        env['PROJ_LIB'] = os.path.join(grass_share, 'proj')
        env['GDAL_DATA'] = os.path.join(grass_share, 'gdal')
        
        try:
            # No capture_output para ver avance en tiempo real
            result = subprocess.run(
                cmd, 
                cwd=str(script_dir), 
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Usar output_dir si fue pasado, sino fallback a config
                raster_dir = Path(output_dir) if output_dir else config.ITZI_OUTPUT_DIR
                return {
                    "success": True,
                    "max_water_depth_raster": str(raster_dir / "max_water_depth.tif")
                }
            else:
                return {
                    "success": False, 
                    "error": f"ITZI exited with code {result.returncode}"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "ITZI timeout (>1 hour)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_climada(self, 
                     depth_raster_path: str, 
                     output_dir: str,
                     predios_path: str = None,
                     case_name: str = None) -> Dict:
        """Ejecuta CLIMADA en el raster de profundidad."""
        from rut_19_flood_damage_climada import calculate_flood_damage_climada
        
        out_dir = Path(output_dir)
        prefix = f"{case_name}_" if case_name else ""
        
        return calculate_flood_damage_climada(
            predios_path=Path(predios_path) if predios_path else None,
            depth_raster_path=Path(depth_raster_path),
            output_gpkg=out_dir / f"{prefix}flood_damage_results.gpkg",
            output_txt=out_dir / f"{prefix}flood_damage_report.txt"
        )
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    


class DeferredInvestmentCost:
    """
    Calcula beneficio por tuberías que ya no necesitan reemplazo.
    
    Workflow (run method):
    1. Extrae datos de SWMM (capacity, flow, coordinates, elevations)
    2. Identifica tuberías que fallan en baseline
    3. Identifica cuáles mejoraron en solution
    4. Calcula costo evitado con SewerConstructionCost
    """
    
    def __init__(self, base_precios_path: str, capacity_threshold: float = 0.85, crs=config.PROJECT_CRS):
        self.capacity_threshold = capacity_threshold
        self.base_precios_path = base_precios_path
        self.crs = crs
        
        # Resultados
        self.baseline_link_data = {}
        self.solution_link_data = {}
        self.saved_pipes = []
        self.total_benefit = 0.0

        self.spll = SeccionParcialmenteLlena()
        self.ps = PipeSizing()

    @staticmethod
    def extract_link_data(inp_path: str):
        """
        Extrae toda la información de conduits + nodos desde SWMM.
        """
        print(f"  [DeferredInvestmentCost] Leyendo INP: {inp_path}")
        
        # Use new NetworkExporter class
        exporter = NetworkExporter(inp_path)
        # We don't save a file here, just want the dataframe
        gdf = exporter.run()
        
        if gdf is None or gdf.empty:
            sys.exit("  [DeferredInvestmentCost] Error: Extractor returned empty data.")

        return gdf
    
    def _classify_pz(self, diameter: float, in_offset: float) -> str:
        """Clasifica el tipo de pozo según diámetro y offset de entrada (salto)."""
        # Clasificación base por diámetro
        if diameter < 0.6: pz = 'pz-b1'
        elif diameter < 0.8: pz = 'pz-b2'
        elif diameter < 1.0: pz = 'pz-b3'
        elif diameter < 1.3: pz = 'pz-b4'
        else: pz = 'pz-esp'
        
        # Clasificación por salto (InOffset)
        if in_offset > 2.5: pz = 'pz-s3'
        elif in_offset > 1.2: pz = 'pz-s2'
        elif in_offset > 0.7: pz = 'pz-s1'
            
        return pz

    def _dimensionar_seccion(self, q_accu, mannig, slope, seccion, grupo):
        """
        Dimensiona la seccion de las tuberias basandose en caudal, rugosidad y pendiente.
        Retorna un array de strings con el diametro interno (circular) o dimensiones (rectangular).
        """

        # Crear arrays de ceros para el resultado
        d_int = np.full(len(q_accu), dtype='U256', fill_value='0')

        # 1. Dimensionar asumiendo sección circular inicial
        d_init = self.spll.seccion_init_circular(q_accu, mannig, slope)

        # Obtener diámetros de PVC
        diametros_disponibles = self.ps.get_internal_diameter_by_material('PVC')
        diametro_maximo = np.max(diametros_disponibles)

        # 2. Condición circular vs rectangular
        is_forced_rect = (seccion == 'rectangular')
        cond_circular = (d_init < diametro_maximo * 0.95) & (~is_forced_rect)

        # 3. Secciones circulares
        indices = np.searchsorted(diametros_disponibles, d_init)
        indices = np.minimum(indices, len(diametros_disponibles) - 1)

        if np.any(cond_circular):
            circular_seccion = diametros_disponibles[indices[cond_circular]].astype('U256')
            circular_seccion_anterior = grupo['Geom1'][cond_circular].to_numpy()
            circular_seccion_anterior = self.ps.get_equivalent_diameter(
                circular_seccion_anterior,
                np.full(shape=len(circular_seccion_anterior), fill_value='PVC', dtype='U256')
            )
            circular_seccion_nuevo = np.maximum(
                circular_seccion.astype(float),
                circular_seccion_anterior.astype(float)
            )
            d_int[cond_circular] = circular_seccion_nuevo.astype('U256')

        # 4. Secciones rectangulares
        cond_rectangular = ~cond_circular
        if np.any(cond_rectangular):
            # Calcular dimensiones rectangulares sólo para las filas rectangulares
            base_all, calado_all = self.spll.seccion_init_rectangular('0', q_accu, mannig, slope)

            # Seleccionar sólo las posiciones rectangulares como arrays \[n\_rect\]
            base = base_all[cond_rectangular].astype(float)
            calado = calado_all[cond_rectangular].astype(float)

            base_anterior = grupo['Geom1'][cond_rectangular].to_numpy(dtype=float)
            calado_anterior = grupo['Geom2'][cond_rectangular].to_numpy(dtype=float)

            base_seccion_nuevo = np.maximum(base, base_anterior)
            calado_seccion_nuevo = np.maximum(calado, calado_anterior)

            # Formatear como "BxY"
            rectangular_seccion_nuevo = np.char.add(
                np.char.mod('%.2f', base_seccion_nuevo),
                np.char.add('x', np.char.mod('%.2f', calado_seccion_nuevo))
            )

            # Asignar sólo en las posiciones rectangulares
            d_int[cond_rectangular] = rectangular_seccion_nuevo.astype('U256')

        return d_int
    
    def _get_surface_type_from_osm(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Consulta OSM para detectar el tipo de superficie bajo cada tramo.
        
        Categorías válidas:
        - pavimento_rigido: highways principales (primary, secondary, tertiary)
        - pavimento_flexible: highways menores con asfalto
        - adoquin: zonas peatonales, plazas
        - vereda: footways, sidewalks
        - lastre: caminos sin pavimentar
        - pasto: parques, áreas verdes, default
        
        Returns:
            pd.Series con tipo de superficie para cada tramo
        """
        try:
            import osmnx as ox
            ox.settings.log_console = False
            ox.settings.use_cache = True
            ox.settings.timeout = 60  # Timeout de 60 segundos
        except ImportError:
            print("  [Warning] osmnx no instalado. Usando 'pasto' por defecto.")
            print("  Instalar con: pip install osmnx")
            return pd.Series(['pasto'] * len(gdf), index=gdf.index)
        
        # Asegurar CRS WGS84 para OSM
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        
        # Calcular centro y radio (como en PathFinder) - más eficiente que bbox
        bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Calcular radio en metros (distancia desde el centro a la esquina + margen)
        import math
        lat_diff = (bounds[3] - bounds[1]) / 2
        lon_diff = (bounds[2] - bounds[0]) / 2
        # Aproximación: 1 grado lat ≈ 111km, 1 grado lon ≈ 111km * cos(lat)
        radius_m = math.sqrt((lat_diff * 111000)**2 + (lon_diff * 111000 * math.cos(math.radians(center_lat)))**2)
        radius_m = min(radius_m + 100, 2000)  # Añadir 100m de margen, max 2km para evitar queries gigantes
        
        print(f"  [OSM] Descargando datos: centro=({center_lat:.4f}, {center_lon:.4f}), radio={radius_m:.0f}m")
        
        # Descargar TODO de OSM en una sola query con múltiples tags
        all_tags = {
            'highway': True,           # Carreteras, calles, senderos
            'landuse': True,           # Uso del suelo (residencial, comercial, industrial, verde)
            # 'leisure': True,           # Ocio (parques, canchas, plazas)
            # 'building': True,          # Edificaciones
            # 'amenity': True,           # Servicios (parking, escuelas, hospitales)
            # 'natural': True,           # Natural (bosques, praderas, agua)
            'surface': True,           # Tag de superficie directa
            'area:highway': True,      # Áreas de carretera (rotondas, plazas)
            # 'man_made': True,          # Construcciones (plazas, puentes)
            # 'place': True,             # Plazas, squares
        }
        
        try:
            # Usar features_from_point con distancia (como PathFinder usa graph_from_point)
            osm_data = ox.features_from_point((center_lat, center_lon), tags=all_tags, dist=radius_m)
            print(f"  [OSM] Descargados {len(osm_data)} elementos")
        except Exception as e:
            print(f"  [OSM] Error descargando datos: {e}")
            print(f"  [OSM] Usando clasificación por defecto (pavimento_flexible)")
            return pd.Series(['pavimento_flexible'] * len(gdf), index=gdf.index)
        
        # Separar por tipo de tag para clasificación jerárquica
        if not osm_data.empty:
            highways = osm_data[osm_data.get('highway', pd.Series()).notna()] if 'highway' in osm_data.columns else gpd.GeoDataFrame()
            amenity = osm_data[osm_data.get('amenity', pd.Series()).notna()] if 'amenity' in osm_data.columns else gpd.GeoDataFrame()
            landuse = osm_data[osm_data.get('landuse', pd.Series()).notna()] if 'landuse' in osm_data.columns else gpd.GeoDataFrame()
            leisure = osm_data[osm_data.get('leisure', pd.Series()).notna()] if 'leisure' in osm_data.columns else gpd.GeoDataFrame()
            natural = osm_data[osm_data.get('natural', pd.Series()).notna()] if 'natural' in osm_data.columns else gpd.GeoDataFrame()
            
            print(f"  [OSM] highways={len(highways)}, amenity={len(amenity)}, landuse={len(landuse)}, leisure={len(leisure)}, natural={len(natural)}")
        else:
            highways = amenity = landuse = leisure = natural = gpd.GeoDataFrame()
        
        # =====================================================================
        # CLASIFICACIÓN VECTORIZADA usando sjoin (mucho más rápido que iterrows)
        # =====================================================================
        
        # Crear GeoDataFrame de centroides con buffer para intersection
        centroids = gdf_wgs84.copy()
        centroids['geometry'] = centroids.geometry.centroid.buffer(0.00015)  # ~5m buffer
        centroids['_orig_idx'] = centroids.index
        centroids = centroids.reset_index(drop=True)
        
        # Inicializar con default
        centroids['sup_road'] = 'pavimento_flexible'
        
        # 1. HIGHWAYS - máxima prioridad
        if not highways.empty and 'highway' in highways.columns:
            highways_valid = highways[highways.geometry.notnull()].copy()
            if not highways_valid.empty:
                # Ensure 'surface' column exists (may not be in OSM data)
                if 'surface' not in highways_valid.columns:
                    highways_valid['surface'] = ''
                
                join_cols = ['geometry', 'highway', 'surface']
                joined = gpd.sjoin(centroids, highways_valid[join_cols], how='left', predicate='intersects')
                
                # Crear máscara de clasificación vectorizada
                hw = joined['highway'].fillna('')
                surf = joined['surface'].fillna('')
                
                # Clasificar según tipo de highway
                mask_rigido = hw.isin(['primary', 'secondary', 'trunk'])
                mask_flexible = hw.isin(['tertiary', 'residential', 'unclassified']) & ~surf.isin(['unpaved', 'gravel', 'dirt'])
                mask_lastre = (hw.isin(['tertiary', 'residential', 'unclassified']) & surf.isin(['unpaved', 'gravel', 'dirt'])) | hw.isin(['track', 'service'])
                mask_adoquin = hw.isin(['footway', 'path', 'pedestrian', 'sidewalk']) & surf.isin(['paving_stones', 'cobblestone', 'sett', 'unhewn_cobblestone'])
                mask_footway_rigido = hw.isin(['footway', 'path', 'pedestrian', 'sidewalk']) & surf.isin(['concrete', 'paved'])
                mask_footway_flex = hw.isin(['footway', 'path', 'pedestrian', 'sidewalk']) & ~mask_adoquin & ~mask_footway_rigido
                
                # Aplicar clasificación (prioridad: rigido > flexible > adoquin > lastre)
                joined.loc[mask_lastre, 'sup_road'] = 'lastre'
                joined.loc[mask_adoquin, 'sup_road'] = 'adoquin'
                joined.loc[mask_footway_flex, 'sup_road'] = 'pavimento_flexible'
                joined.loc[mask_footway_rigido, 'sup_road'] = 'pavimento_rigido'
                joined.loc[mask_flexible, 'sup_road'] = 'pavimento_flexible'
                joined.loc[mask_rigido, 'sup_road'] = 'pavimento_rigido'
                
                # Tomar primera clasificación por índice original (drop duplicates)
                result = joined.drop_duplicates(subset=['_orig_idx'], keep='first')
                centroids = centroids.merge(result[['_orig_idx', 'sup_road']], on='_orig_idx', how='left', suffixes=('_old', ''))
                centroids['sup_road'] = centroids['sup_road'].fillna(centroids['sup_road_old'])
                centroids = centroids.drop(columns=['sup_road_old'], errors='ignore')
        
        # 2. LANDUSE - para los que quedaron sin clasificar
        if not landuse.empty and 'landuse' in landuse.columns:
            landuse_valid = landuse[landuse.geometry.notnull()].copy()
            if not landuse_valid.empty:
                unclassified = centroids[centroids['sup_road'] == 'pavimento_flexible'].copy()
                if not unclassified.empty:
                    joined = gpd.sjoin(unclassified[['geometry', '_orig_idx']], landuse_valid[['geometry', 'landuse']], how='left', predicate='intersects')
                    
                    lu = joined['landuse'].fillna('')
                    joined['new_surface'] = 'pavimento_flexible'
                    joined.loc[lu.isin(['residential', 'commercial', 'retail']), 'new_surface'] = 'pavimento_flexible'
                    joined.loc[lu.isin(['industrial']), 'new_surface'] = 'pavimento_rigido'
                    joined.loc[lu.isin(['grass', 'meadow', 'forest', 'farmland', 'orchard']), 'new_surface'] = 'pasto'
                    
                    result = joined.drop_duplicates(subset=['_orig_idx'], keep='first')[['_orig_idx', 'new_surface']]
                    update_map = result.set_index('_orig_idx')['new_surface'].to_dict()
                    centroids.loc[centroids['_orig_idx'].isin(update_map.keys()), 'sup_road'] = centroids['_orig_idx'].map(update_map)
        
        # Restaurar índice original y devolver serie
        centroids = centroids.set_index('_orig_idx')
        result_series = centroids['sup_road'].reindex(gdf.index).fillna('pavimento_flexible')
        
        print(f"  [OSM] Clasificación completada: {result_series.value_counts().to_dict()}")
        
        return result_series

    def _calculate_real_cost(self, pipes_df, output_dir: Optional[str] = None):
        """
        Genera GPKG temporal con tuberías dimensionadas usando rut_06_pipe_sizing
        y ejecuta SewerConstructionCost para calcular el costo real.

        Similar a section_sizing_int pero sin verificar diámetros decrecientes.
        
        Args:
            pipes_df: DataFrame of failing pipes
            output_dir: Optional path to save intermediate files
        """

        # --- output_dir es requerido ---
        if not output_dir:
            raise ValueError("output_dir es requerido para guardar los archivos de costos evitados")

        grupos = []
        for _, grupo in pipes_df.groupby('Ramal'):


            #sort by elevation descending
            grupo.sort_values(by='InletNode_InvertElev', ascending=False, inplace=True)

            # 1. Extraer arrays para dimensionamiento
            q = grupo['MaxFlow'] * 1000
            q_accu = np.where(q <= 0, 0.001, q)  # Evitar caudales <= 0
            slope = np.where(grupo['Slope'].to_numpy() < 0.0001, 0.0001, grupo['Slope'].to_numpy())
            mannig = grupo['Roughness'].fillna(0.013).values
            raw_shapes = grupo['Shape'].fillna('circular').astype(str).values
            seccion = np.where(np.char.lower(raw_shapes.astype('U256')) == 'circular', 'circular', 'rectangular')

            # 2. Dimensionamiento Hidráulico (D_int)
            d_int = self._dimensionar_seccion(q_accu, mannig, slope, seccion, grupo)

            # 3. Clasificación de Sección (Circular vs Rectangular)
            is_rect = np.char.find(d_int, 'x') >= 0
            is_circular = ~is_rect
            seccion = np.where(is_rect, 'rectangular', 'circular')

            # 4. Cálculo de Diámetro Externo (D_ext)
            d_ext = np.empty_like(d_int)

            # Caso Circular: Obtener D_ext de catálogo PVC usando el valor numérico
            if np.any(is_circular):
                d_vals = d_int[is_circular].astype(float)
                d_ext[is_circular] = self.ps.get_external_diameter_from_internal(d_vals, 'PVC')

            # Caso Rectangular: D_ext se asume igual a la dimensión interna (texto)
            if np.any(is_rect):
                d_ext[is_rect] = d_int[is_rect]

            # 4. Asignación masiva al DataFrame
            grupo['D_int'] = d_int
            grupo['D_ext'] = d_ext
            grupo['Seccion'] = seccion
            grupo['Material'] = np.where(is_rect, 'HA', 'PVC')
            incrementos_start = pd.Series(range(len(d_int)), index=grupo['Ramal'].index).astype('U256')
            incrementos_end = pd.Series(range(1, len(d_int) + 1), index=grupo['Ramal'].index).astype('U256')
            pz_start = grupo['Ramal'].str.cat(incrementos_start, sep='.')
            pz_end = grupo['Ramal'].str.cat(incrementos_end, sep='.')
            grupo['Tramo'] = pz_start.str.cat(pz_end, sep='-')
            grupos.append(grupo)

        # =====================================================================
        # CONSTRUIR GEODATAFRAME
        # =====================================================================
        pipes_rehabilitation_gdf = gpd.GeoDataFrame(pd.concat(grupos, ignore_index=False))
        pipes_rehabilitation_gdf.set_crs(self.crs, inplace=True, allow_override=True)
        
        hi = pipes_rehabilitation_gdf['InletNode_MaxDepth']
        hf = pipes_rehabilitation_gdf['OutletNode_MaxDepth']
        
        zfi = pipes_rehabilitation_gdf['InletNode_InvertElev'] + hi
        zff = pipes_rehabilitation_gdf['OutletNode_InvertElev'] + hf
        
        zti = zfi + hi
        ztf = zff + hf
        
        avg_depth = (hi + hf) / 2.0
        metodo = np.where(avg_depth <= 6.0, 'zanja abierta', 'tunel')
        
        in_offset = pd.to_numeric(pipes_rehabilitation_gdf['InOffset'], errors='coerce').to_numpy()
        
        # clasificacion de Pozos
        d_str = pipes_rehabilitation_gdf['D_int'].astype('U256')
        dimension_vertical =self.spll.section_str2float(d_str)
        conds = [
            in_offset > 2.5,
            in_offset > 1.2,
            dimension_vertical < 0.6,
            dimension_vertical < 0.8,
            dimension_vertical < 1.0,
            dimension_vertical < 1.3
        ]
        choices = [
            'pz-s3', 'pz-s2', 'pz-b1', 'pz-b2', 'pz-b3', 'pz-b4'
        ]
        pz_class = np.select(conds, choices, default='pz-b4')

        pipes_rehabilitation_gdf['Ramal'] = pipes_df['Ramal'].astype('U256')
        pipes_rehabilitation_gdf['Tipo'] = 'pluvial'
        pipes_rehabilitation_gdf['L'] = pipes_rehabilitation_gdf.geometry.length
        pipes_rehabilitation_gdf['Rugosidad'] = 'liso'
        pipes_rehabilitation_gdf['Estado'] = 'nuevo'
        pipes_rehabilitation_gdf['Fase'] = 'reposicion'
        pipes_rehabilitation_gdf['metodo_constructivo'] = metodo
        pipes_rehabilitation_gdf['ZTI'] = zti
        pipes_rehabilitation_gdf['ZTF'] = ztf
        pipes_rehabilitation_gdf['ZFI'] = zfi
        pipes_rehabilitation_gdf['ZFF'] = zff
        pipes_rehabilitation_gdf['HI'] = hi
        pipes_rehabilitation_gdf['HF'] = hf
        pipes_rehabilitation_gdf['SALTO'] = in_offset
        pipes_rehabilitation_gdf['pz_class'] = pz_class
        pipes_rehabilitation_gdf['sup_road'] = self._get_surface_type_from_osm(pipes_rehabilitation_gdf)

        # Remove rows with null, empty or invalid geometries
        pipes_rehabilitation_gdf = pipes_rehabilitation_gdf[
            pipes_rehabilitation_gdf.geometry.notnull()
            & (~pipes_rehabilitation_gdf.geometry.is_empty)
            & pipes_rehabilitation_gdf.geometry.is_valid
            ].copy()

        # Reset index after geometry filtering
        pipes_rehabilitation_gdf.reset_index(drop=True, inplace=True)

        # Identify rows with any NaN in any column
        mask_nan = pipes_rehabilitation_gdf.isna().any(axis=1)

        # Indices (before dropping) of rows with NaN
        rows_to_drop = pipes_rehabilitation_gdf.index[mask_nan]

        # Optional: show how many and which ones
        n_dropped = len(rows_to_drop)
        print(f"Se van a eliminar {n_dropped} filas con al menos un NaN en `pipes_rehabilitation_gdf`.")
        if n_dropped > 0:
            print("Índices (antes de resetear índice) de filas eliminadas:", list(rows_to_drop))
            # Si también quieres ver el contenido:
            print(pipes_rehabilitation_gdf.loc[rows_to_drop])

        # Drop those rows
        pipes_rehabilitation_gdf = pipes_rehabilitation_gdf.drop(index=rows_to_drop).copy()

        # Reset index after NaN filtering
        pipes_rehabilitation_gdf.reset_index(drop=True, inplace=True)

        # Mensaje aclarando que es un valor aproximado / estimado
        print(
            "\n[AVISO] El costo calculado es una aproximación del valor de inversión "
            "y debe considerarse como un valor estimado de referencia, no definitivo."
        )

        # Estructura: output_dir / avoided_cost / deferred_investment
        base_dir = Path(output_dir) / "avoided_cost" / "deferred_investment"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract case name from output_dir (e.g. "Case_001" from ".../Case_001")
        case_name = Path(output_dir).name
        
        # Nombre con el caso: Case_001_pipes_rehabilitation.gpkg
        gpkg_path = base_dir / f"{case_name}_pipes_rehabilitation.gpkg"
        print(f"  [AvoidedCost] Saving GPKG to: {gpkg_path}")
        

        
        # Set CRS before saving
        if self.crs is not None:
            pipes_rehabilitation_gdf.set_crs(self.crs, inplace=True, allow_override=True)
            # print(f"  [AvoidedCost] CRS set to: {self.crs.name if hasattr(self.crs, 'name') else self.crs}")
        
        pipes_rehabilitation_gdf.to_file(gpkg_path, driver='GPKG', layer='tramos')
        
        calc = SewerConstructionCost(
            vector_path=str(gpkg_path),
            tipo='PLUVIAL',
            fase=None,
            base_precios=self.base_precios_path
        )
        
        # Excel
        excel_path = str(gpkg_path.with_suffix('.xlsx'))

        total_cost = calc.run(excel_output_path=excel_path, excel_metadata=config.EXCEL_METADATA)
        
        # --- GENERATE TXT COST SUMMARY BY COMPONENTS ---
        self._generate_cost_summary_txt(excel_path, base_dir, case_name, total_cost, pipes_rehabilitation_gdf)
        
        # --- GENERATE STATISTICS AND PLOTS ---
        self._generate_pipe_statistics(pipes_rehabilitation_gdf, base_dir, case_name, total_cost)
                
        return total_cost

    def _calculate_real_cost(self, pipes_df, output_dir: Optional[str] = None):
        """
        Genera GPKG temporal con tuberías dimensionadas usando rut_06_pipe_sizing
        y ejecuta SewerConstructionCost para calcular el costo real.

        Similar a section_sizing_int pero sin verificar diámetros decrecientes.
        
        Args:
            pipes_df: DataFrame of failing pipes
            output_dir: Optional path to save intermediate files
        """

        # --- output_dir es requerido ---
        if not output_dir:
            raise ValueError("output_dir es requerido para guardar los archivos de costos evitados")

        grupos = []
        for _, grupo in pipes_df.groupby('Ramal'):


            #sort by elevation descending
            grupo.sort_values(by='InletNode_InvertElev', ascending=False, inplace=True)

            # 1. Extraer arrays para dimensionamiento
            q = grupo['MaxFlow'] * 1000
            q_accu = np.where(q <= 0, 0.001, q)  # Evitar caudales <= 0
            slope = np.where(grupo['Slope'].to_numpy() < 0.0001, 0.0001, grupo['Slope'].to_numpy())
            mannig = grupo['Roughness'].fillna(0.013).values
            raw_shapes = grupo['Shape'].fillna('circular').astype(str).values
            seccion = np.where(np.char.lower(raw_shapes.astype('U256')) == 'circular', 'circular', 'rectangular')

            # 2. Dimensionamiento Hidráulico (D_int)
            d_int = self._dimensionar_seccion(q_accu, mannig, slope, seccion, grupo)

            # 3. Clasificación de Sección (Circular vs Rectangular)
            is_rect = np.char.find(d_int, 'x') >= 0
            is_circular = ~is_rect
            seccion = np.where(is_rect, 'rectangular', 'circular')

            # 4. Cálculo de Diámetro Externo (D_ext)
            d_ext = np.empty_like(d_int)

            # Caso Circular: Obtener D_ext de catálogo PVC usando el valor numérico
            if np.any(is_circular):
                d_vals = d_int[is_circular].astype(float)
                d_ext[is_circular] = self.ps.get_external_diameter_from_internal(d_vals, 'PVC')

            # Caso Rectangular: D_ext se asume igual a la dimensión interna (texto)
            if np.any(is_rect):
                d_ext[is_rect] = d_int[is_rect]

            # 4. Asignación masiva al DataFrame
            grupo['D_int'] = d_int
            grupo['D_ext'] = d_ext
            grupo['Seccion'] = seccion
            grupo['Material'] = np.where(is_rect, 'HA', 'PVC')
            incrementos_start = pd.Series(range(len(d_int)), index=grupo['Ramal'].index).astype('U256')
            incrementos_end = pd.Series(range(1, len(d_int) + 1), index=grupo['Ramal'].index).astype('U256')
            pz_start = grupo['Ramal'].str.cat(incrementos_start, sep='.')
            pz_end = grupo['Ramal'].str.cat(incrementos_end, sep='.')
            grupo['Tramo'] = pz_start.str.cat(pz_end, sep='-')
            grupos.append(grupo)

        # =====================================================================
        # CONSTRUIR GEODATAFRAME
        # =====================================================================
        pipes_rehabilitation_gdf = gpd.GeoDataFrame(pd.concat(grupos, ignore_index=False))
        pipes_rehabilitation_gdf.set_crs(self.crs, inplace=True, allow_override=True)
        
        hi = pipes_rehabilitation_gdf['InletNode_MaxDepth']
        hf = pipes_rehabilitation_gdf['OutletNode_MaxDepth']
        
        zfi = pipes_rehabilitation_gdf['InletNode_InvertElev'] + hi
        zff = pipes_rehabilitation_gdf['OutletNode_InvertElev'] + hf
        
        zti = zfi + hi
        ztf = zff + hf
        
        avg_depth = (hi + hf) / 2.0
        metodo = np.where(avg_depth <= 6.0, 'zanja abierta', 'tunel')
        
        in_offset = pd.to_numeric(pipes_rehabilitation_gdf['InOffset'], errors='coerce').to_numpy()
        
        # clasificacion de Pozos
        d_str = pipes_rehabilitation_gdf['D_int'].astype('U256')
        dimension_vertical =self.spll.section_str2float(d_str)
        conds = [
            in_offset > 2.5,
            in_offset > 1.2,
            dimension_vertical < 0.6,
            dimension_vertical < 0.8,
            dimension_vertical < 1.0,
            dimension_vertical < 1.3
        ]
        choices = [
            'pz-s3', 'pz-s2', 'pz-b1', 'pz-b2', 'pz-b3', 'pz-b4'
        ]
        pz_class = np.select(conds, choices, default='pz-b4')

        pipes_rehabilitation_gdf['Ramal'] = pipes_df['Ramal'].astype('U256')
        pipes_rehabilitation_gdf['Tipo'] = 'pluvial'
        pipes_rehabilitation_gdf['L'] = pipes_rehabilitation_gdf.geometry.length
        pipes_rehabilitation_gdf['Rugosidad'] = 'liso'
        pipes_rehabilitation_gdf['Estado'] = 'nuevo'
        pipes_rehabilitation_gdf['Fase'] = 'reposicion'
        pipes_rehabilitation_gdf['metodo_constructivo'] = metodo
        pipes_rehabilitation_gdf['ZTI'] = zti
        pipes_rehabilitation_gdf['ZTF'] = ztf
        pipes_rehabilitation_gdf['ZFI'] = zfi
        pipes_rehabilitation_gdf['ZFF'] = zff
        pipes_rehabilitation_gdf['HI'] = hi
        pipes_rehabilitation_gdf['HF'] = hf
        pipes_rehabilitation_gdf['SALTO'] = in_offset
        pipes_rehabilitation_gdf['pz_class'] = pz_class
        pipes_rehabilitation_gdf['sup_road'] = self._get_surface_type_from_osm(pipes_rehabilitation_gdf)

        # Remove rows with null, empty or invalid geometries
        pipes_rehabilitation_gdf = pipes_rehabilitation_gdf[
            pipes_rehabilitation_gdf.geometry.notnull()
            & (~pipes_rehabilitation_gdf.geometry.is_empty)
            & pipes_rehabilitation_gdf.geometry.is_valid
            ].copy()

        # Reset index after geometry filtering
        pipes_rehabilitation_gdf.reset_index(drop=True, inplace=True)

        # Identify rows with any NaN in any column
        mask_nan = pipes_rehabilitation_gdf.isna().any(axis=1)

        # Indices (before dropping) of rows with NaN
        rows_to_drop = pipes_rehabilitation_gdf.index[mask_nan]

        # Optional: show how many and which ones
        n_dropped = len(rows_to_drop)
        print(f"Se van a eliminar {n_dropped} filas con al menos un NaN en `pipes_rehabilitation_gdf`.")
        if n_dropped > 0:
            print("Índices (antes de resetear índice) de filas eliminadas:", list(rows_to_drop))
            # Si también quieres ver el contenido:
            print(pipes_rehabilitation_gdf.loc[rows_to_drop])

        # Drop those rows
        pipes_rehabilitation_gdf = pipes_rehabilitation_gdf.drop(index=rows_to_drop).copy()

        # Reset index after NaN filtering
        pipes_rehabilitation_gdf.reset_index(drop=True, inplace=True)

        # Mensaje aclarando que es un valor aproximado / estimado
        print(
            "\n[AVISO] El costo calculado es una aproximación del valor de inversión "
            "y debe considerarse como un valor estimado de referencia, no definitivo."
        )

        # Estructura: output_dir / avoided_cost / deferred_investment
        base_dir = Path(output_dir) / "avoided_cost" / "deferred_investment"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract case name from output_dir (e.g. "Case_001" from ".../Case_001")
        case_name = Path(output_dir).name
        
        # Nombre con el caso: Case_001_pipes_rehabilitation.gpkg
        gpkg_path = base_dir / f"{case_name}_pipes_rehabilitation.gpkg"
        print(f"  [AvoidedCost] Saving GPKG to: {gpkg_path}")
        

        
        # Set CRS before saving
        if self.crs is not None:
            pipes_rehabilitation_gdf.set_crs(self.crs, inplace=True, allow_override=True)
            print(f"  [AvoidedCost] CRS set to: {self.crs.name if hasattr(self.crs, 'name') else self.crs}")
        
        pipes_rehabilitation_gdf.to_file(gpkg_path, driver='GPKG', layer='tramos')
        
        calc = SewerConstructionCost(
            vector_path=str(gpkg_path),
            tipo='PLUVIAL',
            fase=None,
            base_precios=self.base_precios_path
        )
        
        # Excel
        excel_path = str(gpkg_path.with_suffix('.xlsx'))

        total_cost = calc.run(excel_output_path=excel_path, excel_metadata=config.EXCEL_METADATA)
        
        # --- GENERATE TXT COST SUMMARY BY COMPONENTS ---
        self._generate_cost_summary_txt(excel_path, base_dir, case_name, total_cost, pipes_rehabilitation_gdf)
        
        # --- GENERATE STATISTICS AND PLOTS ---
        self._generate_pipe_statistics(pipes_rehabilitation_gdf, base_dir, case_name, total_cost)
                
        return total_cost
    
    def _generate_cost_summary_txt(self, excel_path: str, output_dir: Path, case_name: str, total_cost: float, gdf: gpd.GeoDataFrame):
        """
        Genera un archivo TXT con resumen de costos por componentes.
        """
        try:
            # Leer Excel para obtener costos por componente
            df_costs = pd.read_excel(excel_path, sheet_name='Presupuesto', header=None)
            
            # Buscar filas que contienen subtotales o componentes
            txt_path = output_dir / f"{case_name}_resumen_costos.txt"
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"RESUMEN DE COSTOS - {case_name}\n")
                f.write("=" * 70 + "\n\n")
                
                # Metadatos del proyecto
                f.write("INFORMACIÓN DEL PROYECTO\n")
                f.write("-" * 40 + "\n")
                for key, val in config.EXCEL_METADATA.items():
                    f.write(f"  {key}: {val}\n")
                f.write("\n")
                
                # Estadísticas generales de tuberías
                f.write("ESTADÍSTICAS GENERALES DE TUBERÍAS\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Número de tramos: {len(gdf)}\n")
                
                if 'Length' in gdf.columns:
                    f.write(f"  Longitud total: {gdf['Length'].sum():,.2f} m\n")
                    f.write(f"  Longitud promedio por tramo: {gdf['Length'].mean():,.2f} m\n")
                    f.write(f"  Longitud mínima: {gdf['Length'].min():,.2f} m\n")
                    f.write(f"  Longitud máxima: {gdf['Length'].max():,.2f} m\n")
                
                if 'D_int' in gdf.columns:
                    f.write(f"  Secciones distintas: {gdf['D_int'].nunique()}\n")
                f.write("\n")
                
                # Estadísticas de profundidad
                f.write("PROFUNDIDADES DE EXCAVACIÓN\n")
                f.write("-" * 40 + "\n")
                if 'HI' in gdf.columns and 'HF' in gdf.columns:
                    avg_depth = (gdf['HI'].mean() + gdf['HF'].mean()) / 2
                    max_depth = max(gdf['HI'].max(), gdf['HF'].max())
                    min_depth = min(gdf['HI'].min(), gdf['HF'].min())
                    f.write(f"  Profundidad promedio: {avg_depth:,.2f} m\n")
                    f.write(f"  Profundidad mínima: {min_depth:,.2f} m\n")
                    f.write(f"  Profundidad máxima: {max_depth:,.2f} m\n")
                f.write("\n")
                
                # Desglose por Material
                f.write("DESGLOSE POR MATERIAL\n")
                f.write("-" * 40 + "\n")
                if 'Material' in gdf.columns and 'Length' in gdf.columns:
                    for mat, grp in gdf.groupby('Material'):
                        f.write(f"  {mat}:\n")
                        f.write(f"    - Tramos: {len(grp)}\n")
                        f.write(f"    - Longitud: {grp['Length'].sum():,.2f} m ({100*grp['Length'].sum()/gdf['Length'].sum():.1f}%)\n")
                f.write("\n")
                
                # Desglose por Tipo de Superficie
                f.write("DESGLOSE POR TIPO DE SUPERFICIE\n")
                f.write("-" * 40 + "\n")
                if 'sup_road' in gdf.columns and 'Length' in gdf.columns:
                    for surf, grp in gdf.groupby('sup_road'):
                        f.write(f"  {surf}:\n")
                        f.write(f"    - Tramos: {len(grp)}\n")
                        f.write(f"    - Longitud: {grp['Length'].sum():,.2f} m ({100*grp['Length'].sum()/gdf['Length'].sum():.1f}%)\n")
                        if 'HI' in grp.columns and 'HF' in grp.columns:
                            avg_d = (grp['HI'].mean() + grp['HF'].mean()) / 2
                            f.write(f"    - Profundidad promedio: {avg_d:,.2f} m\n")
                f.write("\n")
                
                # Desglose por Método Constructivo
                if 'metodo_constructivo' in gdf.columns:
                    f.write("DESGLOSE POR MÉTODO CONSTRUCTIVO\n")
                    f.write("-" * 40 + "\n")
                    for met, grp in gdf.groupby('metodo_constructivo'):
                        f.write(f"  {met}:\n")
                        f.write(f"    - Tramos: {len(grp)}\n")
                        if 'Length' in grp.columns:
                            f.write(f"    - Longitud: {grp['Length'].sum():,.2f} m\n")
                    f.write("\n")
                
                # Costo total
                f.write("COSTO TOTAL\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Total: ${total_cost:,.2f} USD\n")
                if 'Length' in gdf.columns and gdf['Length'].sum() > 0:
                    f.write(f"  Costo por metro lineal: ${total_cost/gdf['Length'].sum():,.2f} USD/m\n")
                f.write("\n")
                
                # Fecha de generación
                from datetime import datetime
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
            
            print(f"  [AvoidedCost] Resumen TXT guardado: {txt_path}")
            
        except Exception as e:
            print(f"  [Warning] No se pudo generar resumen TXT: {e}")
    
    def _generate_pipe_statistics(self, gdf: gpd.GeoDataFrame, output_dir: Path, case_name: str, total_cost: float):
        """
        Genera gráficos y estadísticas de las tuberías a rehabilitar.
        """

        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Estadísticas de Tuberías - {case_name}', fontsize=16, fontweight='bold')
            
            # 1. Mapa de tuberías
            ax1 = axes[0, 0]
            if 'D_int' in gdf.columns:
                gdf.plot(ax=ax1, column='D_int', cmap='RdYlGn_r', linewidth=2, legend=False)
            else:
                gdf.plot(ax=ax1, color='red', linewidth=2)
            ax1.set_title('Ubicación de Tuberías a Rehabilitar', fontsize=12)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Distribución de secciones (agrupado en rangos de 0.5m)
            ax2 = axes[0, 1]
            if 'D_int' in gdf.columns:
                # Extraer la primera dimensión numérica para agrupar
                def get_first_dimension(val):
                    try:
                        val_str = str(val).strip()
                        if 'x' in val_str.lower():
                            return float(val_str.lower().split('x')[0])
                        return float(val_str)
                    except:
                        return np.nan
                
                dims = gdf['D_int'].apply(get_first_dimension).dropna()
                
                if len(dims) > 0:
                    # Siempre 6 rangos fijos, valores se adaptan a los datos
                    bins = np.linspace(dims.min(), dims.max(), 7)  # 7 bordes = 6 rangos
                    labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(6)]
                    categorized = pd.cut(dims, bins=bins, labels=labels, include_lowest=True)
                    range_counts = categorized.value_counts().sort_index()
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(range_counts)))
                    bars = ax2.bar(range(len(range_counts)), range_counts.values, color=colors)
                    ax2.set_xticks(range(len(range_counts)))
                    ax2.set_xticklabels(range_counts.index, rotation=45, ha='right')
                    ax2.set_xlabel('Rango de Sección (m)')
                    ax2.set_ylabel('Cantidad de Tramos')
                    ax2.set_title('Distribución por Rango de Sección', fontsize=12)
                    
                    # Añadir totales en barras
                    for bar, val in zip(bars, range_counts.values):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                                 str(int(val)), ha='center', va='bottom', fontsize=9)
            
            # 3. Distribución de longitudes
            ax3 = axes[1, 0]
            if 'L' in gdf.columns:
                lengths = gdf['L']
                ax3.hist(lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax3.axvline(lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {lengths.mean():.1f}m')
                ax3.axvline(lengths.median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {lengths.median():.1f}m')
                ax3.set_xlabel('Longitud (m)')
                ax3.set_ylabel('Frecuencia')
                ax3.set_title('Distribución de Longitudes', fontsize=12)
                ax3.legend()
            
            # 4. Resumen estadístico
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Calcular estadísticas
            n_pipes = len(gdf)
            total_length = gdf['L'].sum() if 'L' in gdf.columns else 0
            n_secciones = gdf['D_int'].nunique() if 'D_int' in gdf.columns else 0
            
            stats_text = f"""
╔══════════════════════════════════════════╗
║     RESUMEN DE REHABILITACIÓN            ║
╠══════════════════════════════════════════╣
║  Número de Tramos:     {n_pipes:>10}        ║
║  Longitud Total:       {total_length:>10,.1f} m     ║
║  Secciones Distintas:  {n_secciones:>10}        ║
║                                          ║
║  COSTO ESTIMADO:                         ║
║    Total:              ${total_cost:>12,.2f}    ║
║    Por metro:          ${total_cost/max(total_length,1):>12,.2f}    ║
╚══════════════════════════════════════════╝
"""
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
            
            plt.tight_layout()
            
            # Guardar
            plot_path = output_dir / f"{case_name}_pipe_statistics.png"
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  [Stats] Saved: {plot_path}")
            
        except Exception as e:
            print(f"  [Warning] Could not generate statistics plot: {e}")
    
    def run(self, inp_path: str, output_dir: Optional[str] = None) -> float:
        """
        Calcula el costo de reemplazo de tuberías que fallan (h/D > threshold).

        Args:
            inp_path: Ruta al archivo .inp
            output_dir: Optional path to save intermediate files (e.g. avoided cost gpkg)

        Returns:
            float: Costo total de reemplazo de tuberías fallando
        """
        print("\n" + "=" * 60)
        print("DeferredInvestmentCost - Cálculo de Costo de Reemplazo")
        print("=" * 60)

        # Step 1: Extraer datos
        print("\n--- PASO 1: Extracción de datos SWMM ---")
        self.swmm_gdf = self.extract_link_data(inp_path)

        # Step 2: Identificar tuberías fallando
        print(f"\n--- PASO 2: Tuberías fallando (h/D >= {self.capacity_threshold}) ---")
        
        if 'Capacity' not in self.swmm_gdf.columns:
            sys.exit("CRITICAL ERROR: 'Capacity' column not found in SWMM results. "
                               "The hydraulic simulation likely failed (e.g., 'Series out of sequence'). "
                               "Cannot calculate pipe replacement costs without valid hydraulic results.")
                               
        filtro = self.swmm_gdf['Capacity'] > self.capacity_threshold
        self.failing_pipes = self.swmm_gdf[filtro].copy()

        print(f"  Total tuberías: {len(self.swmm_gdf)}")
        print(f"  Tuberías fallando: {len(self.failing_pipes)}")

        # Step 3: Calcular costo de reemplazo
        if not self.failing_pipes.empty:
            print("\n--- PASO 3: Cálculo de costo con RUT_21 ---")
            self.total_cost = self._calculate_real_cost(self.failing_pipes, output_dir=output_dir)
            print(f"  Costo total de reemplazo: ${self.total_cost:,.2f}")
        else:
            print("\n--- PASO 3: No hay tuberías que reemplazar ---")
            self.total_cost = 0.0

        print("\n" + "=" * 60)
        return self.total_cost



#---------------------------------------------------------------------------------
class PavementDegradationCost(BaseStrategy):
    """
    Calculates cost of pavement degradation due to flooding.
    
    Logic:
    - Use flood depth at street nodes
    - Apply damage function: f(depth) -> damage_factor
    - Cost = Area_Affected × Repair_Cost/m² × damage_factor
    
    This calculates the DAMAGE in baseline and solution, 
    the benefit is the reduction.
    """
    def __init__(self, repair_cost_m2: float = 50.0, influence_area_m2: float = 100.0):
        self.repair_cost_m2 = repair_cost_m2
        self.influence_area_m2 = influence_area_m2
        # Damage factor curve based on depth
        self.DAMAGE_CURVE = [
            (0.10, 0.00),  # < 10cm: no damage
            (0.30, 0.10),  # 10-30cm: 10% damage
            (0.50, 0.30),  # 30-50cm: 30% damage
            (1.00, 0.60),  # 50-100cm: 60% damage
            (float('inf'), 1.00)  # >100cm: 100% damage
        ]

    def calculate(self, base, sol, **kwargs) -> Dict[str, Any]:
        """Calculate avoided pavement damage cost."""
        base_damage = self._calculate_damage(base)
        sol_damage = self._calculate_damage(sol)
        
        benefit = base_damage - sol_damage  # Positive = improvement
        
        return {
            "value": max(0, benefit),  # Only count positive benefits
            "metadata": {
                "baseline_damage": base_damage,
                "solution_damage": sol_damage,
                "nodes_analyzed": len(getattr(base, 'node_depths', {}))
            }
        }
    
    def _calculate_damage(self, metrics) -> float:
        """Calculate total pavement damage for a scenario."""
        node_depths = getattr(metrics, 'node_depths', {})
        total_damage = 0.0
        
        for node_id, depth in node_depths.items():
            factor = self._get_damage_factor(depth)
            damage = self.influence_area_m2 * self.repair_cost_m2 * factor
            total_damage += damage
            
        return total_damage
    
    def _get_damage_factor(self, depth: float) -> float:
        """Get damage factor based on flood depth."""
        for threshold, factor in self.DAMAGE_CURVE:
            if depth <= threshold:
                return factor
        return 1.0


class TrafficDisruptionCost(BaseStrategy):
    """
    Calculates cost of traffic disruption due to flooding.
    
    Logic:
    - Identify nodes on major roads (if depth > 15cm, road is blocked)
    - Calculate hours of disruption
    - Cost = AADT × Cost_per_vehicle_hour × Duration
    """
    def __init__(self, 
                 depth_threshold: float = 0.15,
                 avg_aadt: float = 5000.0,  # Average Annual Daily Traffic
                 cost_per_vehicle_hour: float = 0.30):
        self.depth_threshold = depth_threshold
        self.avg_aadt = avg_aadt
        self.cost_per_vehicle_hour = cost_per_vehicle_hour

    def calculate(self, base, sol, **kwargs) -> Dict[str, Any]:
        """Calculate avoided traffic disruption cost."""
        base_cost = self._calculate_disruption_cost(base)
        sol_cost = self._calculate_disruption_cost(sol)
        
        benefit = base_cost - sol_cost
        
        return {
            "value": max(0, benefit),
            "metadata": {
                "baseline_disruption_cost": base_cost,
                "solution_disruption_cost": sol_cost
            }
        }
    
    def _calculate_disruption_cost(self, metrics) -> float:
        """Calculate traffic disruption cost for a scenario."""
        node_depths = getattr(metrics, 'node_depths', {})
        hydrographs = getattr(metrics, 'flood_hydrographs', {})
        
        total_cost = 0.0
        
        for node_id, depth in node_depths.items():
            if depth > self.depth_threshold:
                # Road is blocked - estimate duration
                duration_hours = 2.0  # Default: assume 2 hours
                
                if node_id in hydrographs:
                    # Try to calculate actual duration from hydrograph
                    times = hydrographs[node_id].get('times', [])
                    if len(times) >= 2:
                        duration_hours = (times[-1] - times[0]).total_seconds() / 3600.0
                
                # Calculate cost
                hourly_traffic = self.avg_aadt / 24.0
                cost = hourly_traffic * self.cost_per_vehicle_hour * duration_hours
                total_cost += cost
        
        return total_cost


class ErosionControlCost(BaseStrategy):
    """
    Calculates erosion damage cost in natural channels.
    
    Logic:
    - Identify natural channel sections (not pipes)
    - Calculate shear stress: τ = γ × R × S
    - If τ > τ_critical, erosion occurs
    - Cost = Length × Repair_Cost/m
    
    This is a placeholder - full implementation requires channel geometry data.
    """
    def __init__(self, 
                 critical_shear_stress: float = 20.0,  # Pa (for gravel)
                 repair_cost_per_m: float = 300.0):
        self.critical_shear_stress = critical_shear_stress
        self.repair_cost_per_m = repair_cost_per_m

    def calculate(self, base, sol, **kwargs) -> Dict[str, Any]:
        """Calculate avoided erosion damage."""
        # Placeholder - requires channel identification and geometry
        # For now, return 0 benefit
        return {
            "value": 0.0,
            "metadata": {
                "status": "Requires channel geometry data",
                "info": "Not yet implemented - needs link type classification"
            }
        }

#---------------------------------------------------------------------------------
class AvoidedCostRunner:
    """
    Orchestrates the full Avoided Cost Assessment:
    1. Deferred Investment Cost (Pipe Replacement)
    2. Flood Damage Cost (ITZI + CLIMADA)
    
    Features:
    - Generates scenarios dynamically using rut_22 if needed.
    - Runs probabilistic analysis for a list of Return Periods.
    """
    
    def _read_file_robust(self, path: Path) -> str:
        """Attempts to read a file with multiple encodings."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Could not decode {path} with encodings: {encodings}")

    def __init__(self, output_base: str, base_precios_path: str, base_inp_path: str = None, scenarios_dir: str = None):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.base_precios_path = base_precios_path
        
        # Scenarios folder (Flexible location)
        if scenarios_dir:
            self.scenarios_dir = Path(scenarios_dir)
        else:
            # Default: inside output base
            self.scenarios_dir = self.output_base / "scenarios"
            
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        # Base INP for generation
        self.base_inp_path = Path(base_inp_path)
        
        # Load base content once
        if self.base_inp_path.exists():
            self.base_inp_content = self._read_file_robust(self.base_inp_path)
        else:
            sys.exit(f"[Warning] Base INP not found at {self.base_inp_path}. Generation failed.")


    def _get_or_create_scenario(self, tr: int) -> Path:
        """
        Ensures the INP file for the given TR exists.
        If not, generates it using rut_22 logic.
        """
        filename = f"COLEGIO_TR{tr:03d}.inp"
        inp_path = self.scenarios_dir / filename
        
        # Check if exists (and is non-empty?)
        if inp_path.exists():
            # We could assume it's good, or regenerate to be safe. 
            # For now, use existing to save time, unless forced (not impl).
            return inp_path
            
        print(f"  [Gen] Generating scenario TR={tr} years...")
        if not self.base_inp_content:
            raise FileNotFoundError(f"Cannot generate scenario, base INP missing: {self.base_inp_path}")
            
        # 1. Generate Hyetograph
        # Using variables from rut_22 or defaults
        duration = 60
        dt = 5
        df_hyeto = rut_22.generate_alternating_block_hyetograph(tr, duration, dt)
        
        # 2. Generate INP
        rut_22.generate_inp_file(self.base_inp_content, tr, df_hyeto, inp_path)
        
        return inp_path

    def run(self, tr_list: List[int]):
        """
        Runs the assessment for the provided list of Return Periods.
        """
        if not isinstance(tr_list, list) or not tr_list:
            print("[Error] tr_list must be a non-empty list of integers.")
            return
            
        # Sort TRs
        tr_list = sorted(list(set(tr_list)))
        
        results = []
        print(f"\n[AvoidedCostRunner] Processing {len(tr_list)} Return Periods: {tr_list}")
        
        # Initialize Evaluators
        investment_eval = DeferredInvestmentCost(
            base_precios_path=self.base_precios_path, 
            capacity_threshold=0.99
        )
        flood_eval = FloodDamageFromItzi(itzi_vars='depth,v')
        
        for tr in tr_list:
            print(f"\n" + "="*50)
            print(f" PROCESSING TR {tr} years")
            print("="*50)
            
            # 1. Ensure Scenario Exists
            try:
                inp_path = self._get_or_create_scenario(tr)
            except Exception as e:
                print(f"[Error] Failed to generate/find scenario for TR {tr}: {e}")
                continue
            
            # 2. Output Folder
            out_folder = self.output_base / f"TR_{tr:03d}"
            out_folder.mkdir(exist_ok=True)
            
            scenario_res = {'tr': tr, 'inp': inp_path.name}
            
            # --- A. Pipe Investment Cost ---
            if config.COST_COMPONENTS.get('deferred_investment', True):
                try:
                    print(f"\n> Metric A: Deferred Investment (Pipes)")
                    cost_pipes = investment_eval.run(
                        inp_path=str(inp_path),
                        output_dir=str(out_folder)
                    )
                    scenario_res['investment_cost_usd'] = cost_pipes
                except Exception as e:
                    print(f"[Error] Pipe Cost TR {tr}: {e}")
                    scenario_res['investment_cost_usd'] = 0.0
                    import traceback
                    traceback.print_exc()
                    sys.exit()
            else:
                print(f"\n> Metric A: Deferred Investment - SKIPPED (config)")
                scenario_res['investment_cost_usd'] = 0.0

            # --- B. Flood Damage Cost ---
            if config.COST_COMPONENTS.get('flood_damage', True):
                try:
                    print(f"\n> Metric B: Flood Damage (ITZI+CLIMADA)")
                    f_res = flood_eval.run(
                        inp_path=str(inp_path),
                        output_dir=str(out_folder)
                    )
                    scenario_res['flood_damage_usd'] = f_res['total_damage_usd']
                    scenario_res['damage_gpkg'] = f_res.get('output_gpkg')
                    
                    # Generate Visualizations
                    print(f"  > Generating visualizations for TR {tr}...")
                    flood_eval.generate_visualizations(output_dir=str(out_folder))

                except Exception as e:
                    print(f"[Error] Flood Damage TR {tr}: {e}")
                    scenario_res['flood_damage_usd'] = 0.0
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n> Metric B: Flood Damage - SKIPPED (config)")
                scenario_res['flood_damage_usd'] = 0.0
                scenario_res['damage_gpkg'] = None
            
            # Total
            scenario_res['total_impact_usd'] = scenario_res['investment_cost_usd'] + scenario_res['flood_damage_usd']
            results.append(scenario_res)
            
        # 3. Probabilistic Analysis (EAD) via rut_21
        ead_flood = None
        ead_total = None
        
        if len(results) > 1:
            print("\n>>> CALCULATING PROBABILISTIC METRICS (EAD via RiskAnalyzer)...")
            
            # Save Summary first
            res_df = pd.DataFrame(results)
            res_df.to_csv(self.output_base / "avoided_cost_summary.csv", index=False)
            print(f"  Summary saved to: {self.output_base / 'avoided_cost_summary.csv'}")
            
            # Use RiskAnalyzer for EAD calculation
            analyzer = RiskAnalyzer(self.output_base / "risk_estimation")
            
            # Build dict {tr: gpkg_path}
            damage_paths = {
                r['tr']: r['damage_gpkg'] 
                for r in results 
                if r.get('damage_gpkg') and Path(r['damage_gpkg']).exists()
            }
            
            # Extract infrastructure costs per TR
            extra_costs = {r['tr']: r.get('investment_cost_usd', 0.0) for r in results}
            
            if damage_paths:
                # Analyze EAD uncertainty and get results
                ead_result = analyzer.analyze_ead_uncertainty(
                    res_df, damage_paths, extra_costs=extra_costs, n_boot=1000
                )
                analyzer.calculate_fragility_curves(res_df)
                
                # Extract EAD from result if available
                if ead_result and isinstance(ead_result, dict):
                    ead_total = ead_result.get('ead_total', 0)
                    ead_flood = ead_result.get('ead_flood', 0)
                    print(f"  [RiskAnalyzer] EAD Total: ${ead_total:,.2f}")
                    print(f"  [RiskAnalyzer] EAD Flood: ${ead_flood:,.2f}")
                
                print("  [RiskAnalyzer] Completed successfully.")
            else:
                print("  [RiskAnalyzer] Skipped: No valid damage GPKGs found.")

        return {
            'results': results, 
            'ead_flood': ead_flood, 
            'ead_total': ead_total
        }

if __name__ == '__main__':
    
    print("=" * 60)
    print("RUNNING AVOIDED COST ASSESSMENT (PROBABILISTIC)")
    print("=" * 60)
    
    # 1. Configuration using standard paths from config
    # Where results will go
    # Where results will go
    OUTPUT_DIR = config.CODIGOS_DIR / "probabilistic_results"
    # Prices database
    BASE_PRECIOS = config.BASE_PRECIOS
    # Base INP (Template) - Optional, defaults to config one
    BASE_INP = config.SWMM_FILE
    
    # 2. Define Scenarios (List of Return Periods)
    # TRs to run
    TR_LIST = [1,2,5,10,25,50, 100]
    
    # 3. Instantiate Runner
    runner = AvoidedCostRunner(
        output_base=str(OUTPUT_DIR),
        base_precios_path=str(BASE_PRECIOS),
        base_inp_path=str(BASE_INP)
    )
    
    # 4. Run Analysis
    runner.run(tr_list=TR_LIST)


