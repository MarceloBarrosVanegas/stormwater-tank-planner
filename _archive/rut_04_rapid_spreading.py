"""
volumen_inundacion.py

Módulo para calcular la lámina y el área de inundación a partir de:
- un DEM en GeoTIFF
- un nodo (x, y)
- un volumen objetivo (por ejemplo, overflow de SWMM en m³)

La clase VolumeInundationModel implementa un enfoque inverso al "fill lake":
encuentra la cota de agua H tal que el volumen de agua sobre el DEM, conectado
al nodo, sea igual al volumen objetivo. Además, permite generar un mapa sencillo
de la zona inundada.
"""

from collections import deque
from typing import Tuple, Dict, Any, Optional

import numpy as np
import rasterio
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
import pandas as pd


class VolumeInundationModel:
    """
    Modelo para calcular la lámina y el área de inundación a partir de:
    - un DEM
    - un nodo (x, y)
    - un volumen objetivo (p. ej. overflow de SWMM en m³)
    """

    def __init__(self, dem_path: str):
        """
        Parameters
        ----------
        dem_path : str
            Ruta al DEM en formato GeoTIFF.
        """
        self.src = rasterio.open(dem_path)
        self.dem = self.src.read(1).astype("float64")
        self.transform = self.src.transform
        self.nodata = self.src.nodata

        # Máscara de celdas válidas
        if self.nodata is not None:
            self.valid = self.dem != self.nodata
        else:
            self.valid = np.ones_like(self.dem, dtype=bool)

        # Área de celda (positivo)
        cell_size_x = self.transform.a
        cell_size_y = -self.transform.e  # suele venir negativo
        self.cell_area = cell_size_x * cell_size_y

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _coords_to_rowcol(self, x: float, y: float) -> Tuple[int, int]:
        """Convierte coordenadas (x,y) a índices (row, col) del raster."""
        row, col = rowcol(self.transform, x, y)
        nrows, ncols = self.dem.shape
        if not (0 <= row < nrows and 0 <= col < ncols):
            raise ValueError("El nodo está fuera de la extensión del DEM.")
        return int(row), int(col)

    @staticmethod
    def _flood_fill_connected(
        mask: np.ndarray,
        seed_row: int,
        seed_col: int,
        connectivity: int = 8,
    ) -> np.ndarray:
        """
        Devuelve una máscara booleana con la región conectada al píxel seed
        dentro de 'mask' (que ya es booleana).
        """
        nrows, ncols = mask.shape
        connected = np.zeros_like(mask, dtype=bool)

        # Si el píxel semilla no está inundado, no hay región conectada
        if not mask[seed_row, seed_col]:
            return connected

        if connectivity == 4:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8 conectividad
            neighbors = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]

        q = deque()
        q.append((seed_row, seed_col))
        connected[seed_row, seed_col] = True

        while q:
            r, c = q.popleft()
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    if mask[rr, cc] and not connected[rr, cc]:
                        connected[rr, cc] = True
                        q.append((rr, cc))

        return connected

    def _compute_volume_for_level(
        self,
        H: float,
        seed_row: int,
        seed_col: int,
        connectivity: int = 8,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calcula el volumen inundado para una cota H,
        restringido a la región conectada al nodo.
        """
        # celdas potencialmente inundadas y válidas
        inundated = (self.dem < H) & self.valid

        # restringir a región conectada al nodo
        connected = self._flood_fill_connected(
            inundated, seed_row, seed_col, connectivity=connectivity
        )

        depth = np.where(connected, H - self.dem, 0.0)
        volume = float(np.sum(depth) * self.cell_area)
        return volume, depth, connected

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def find_level_for_volume(
        self,
        node_x: float,
        node_y: float,
        V_target: float,
        H_min: Optional[float] = None,
        H_max: Optional[float] = None,
        tol_volume: float = 1.0,
        max_iter: int = 50,
        connectivity: int = 8,
        auto_expand_Hmax: bool = True,
        expand_step: float = 2.0,
        max_expansions: int = 10,
    ) -> Dict[str, Any]:
        """
        Encuentra la cota de agua H tal que el volumen inundado conectado al nodo
        sea igual al volumen objetivo (en m³).

        Parameters
        ----------
        node_x, node_y : float
            Coordenadas del nodo (en el mismo sistema del DEM).
        V_target : float
            Volumen objetivo (m³).
        H_min, H_max : float, opcional
            Rango inicial de búsqueda en metros. Si son None, se usa:
            H_min = cota del nodo; H_max = H_min + 5 m.
        tol_volume : float
            Tolerancia de volumen (m³) para detener la búsqueda.
        max_iter : int
            Máximo de iteraciones de la bisección.
        connectivity : int
            4 u 8 conectividad en el crecimiento de la mancha.
        auto_expand_Hmax : bool
            Si True, aumenta H_max en 'expand_step' mientras V(H_max) < V_target.
        expand_step : float
            Incremento de H_max cuando se expande.
        max_expansions : int
            Máximo de expansiones de H_max.

        Returns
        -------
        dict con:
            - "H"             : cota de agua estimada (m)
            - "V"             : volumen calculado (m³)
            - "depth"         : raster 2D de lámina (m)
            - "mask"          : máscara booleana de inundación
            - "area"          : área inundada (m²)
            - "mean_depth"    : lámina media (m)
            - "seed_row_col"  : (row, col) del nodo
        """

        # Si volumen objetivo es cero (o casi), devolvemos sin inundación
        seed_row, seed_col = self._coords_to_rowcol(node_x, node_y)
        node_elev = float(self.dem[seed_row, seed_col])

        if V_target <= 0:
            depth = np.zeros_like(self.dem, dtype="float64")
            mask = np.zeros_like(self.dem, dtype=bool)
            return {
                "H": node_elev,
                "V": 0.0,
                "depth": depth,
                "mask": mask,
                "area": 0.0,
                "mean_depth": 0.0,
                "seed_row_col": (seed_row, seed_col),
            }

        # Rango inicial de alturas
        if H_min is None:
            H_min = node_elev
        if H_max is None:
            H_max = H_min + 5.0

        # Volumen en el límite inferior
        V_min, _, _ = self._compute_volume_for_level(
            H_min, seed_row, seed_col, connectivity
        )

        if V_target < V_min:
            raise ValueError(
                "El volumen objetivo es menor que el volumen inundado en H_min. "
                "Posiblemente H_min es demasiado alto o el volumen es muy pequeño."
            )

        # Volumen en el límite superior (con posible expansión automática)
        V_max, _, _ = self._compute_volume_for_level(
            H_max, seed_row, seed_col, connectivity
        )

        expansions = 0
        while auto_expand_Hmax and V_max < V_target and expansions < max_expansions:
            H_max += expand_step
            V_max, _, _ = self._compute_volume_for_level(
                H_max, seed_row, seed_col, connectivity
            )
            expansions += 1

        if V_max < V_target:
            raise ValueError(
                "No se alcanzó el volumen objetivo incluso después de expandir H_max. "
                "Prueba con un H_max más alto o revisa el volumen objetivo."
            )

        # Búsqueda por bisección
        last_H: float = H_min
        last_V: float = V_min
        last_depth: np.ndarray = np.zeros_like(self.dem, dtype="float64")
        last_mask: np.ndarray = np.zeros_like(self.dem, dtype=bool)

        for _ in range(max_iter):
            H_mid = 0.5 * (H_min + H_max)
            V_mid, depth_mid, mask_mid = self._compute_volume_for_level(
                H_mid, seed_row, seed_col, connectivity
            )

            last_H, last_V = H_mid, V_mid
            last_depth, last_mask = depth_mid, mask_mid

            if abs(V_mid - V_target) <= tol_volume:
                break

            if V_mid < V_target:
                H_min = H_mid
            else:
                H_max = H_mid

        # Área y lámina media
        area_inund = float(np.sum(last_mask) * self.cell_area)
        mean_depth = (last_V / area_inund) if area_inund > 0 else 0.0

        return {
            "H": float(last_H),
            "V": float(last_V),
            "depth": last_depth,
            "mask": last_mask,
            "area": area_inund,
            "mean_depth": float(mean_depth),
            "seed_row_col": (seed_row, seed_col),
        }

    # ------------------------------------------------------------------
    # Helpers para guardar y dibujar
    # ------------------------------------------------------------------

    def save_depth_raster(
        self,
        out_path: str,
        depth_array: np.ndarray,
        nodata: float = 0.0,
    ) -> None:
        """
        Guarda la lámina de agua como GeoTIFF.
        """
        profile = self.src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=nodata,
            count=1,
        )

        depth_out = depth_array.astype("float32")

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(depth_out, 1)

    def plot_inundation(
        self,
        depth_array: np.ndarray,
        show_dem: bool = False,
        title: Optional[str] = None,
        cmap_inund: str = "Blues",
    ) -> None:
        """
        Dibuja un mapa sencillo de la zona inundada.

        Parameters
        ----------
        depth_array : np.ndarray
            Lámina de agua (m) en cada celda (mismo tamaño que el DEM).
        show_dem : bool
            Si True, dibuja el DEM en grises de fondo y la inundación encima.
        title : str, opcional
            Título del gráfico.
        cmap_inund : str
            Colormap para la lámina de agua.
        """
        depth = np.array(depth_array, copy=True)
        depth[depth < 0] = 0.0

        fig, ax = plt.subplots(figsize=(8, 6))

        if show_dem:
            dem_img = np.ma.masked_array(self.dem, ~self.valid)
            ax.imshow(dem_img, cmap="gray", origin="upper")
            # superponer lámina
            depth_masked = np.ma.masked_array(depth, depth <= 0)
            ax.imshow(depth_masked, cmap=cmap_inund, origin="upper", alpha=0.6)
        else:
            depth_masked = np.ma.masked_array(depth, depth <= 0)
            im = ax.imshow(depth_masked, cmap=cmap_inund, origin="upper")
            fig.colorbar(im, ax=ax, label="Lámina de agua (m)")

        ax.set_xlabel("Columnas (índice de píxel)")
        ax.set_ylabel("Filas (índice de píxel)")
        ax.set_title(title or "Zona inundada")

        plt.tight_layout()
        plt.show()

    def close(self) -> None:
        """Cierra el dataset del DEM."""
        self.src.close()


"""
main_inundacion.py

Script principal para:
- Leer la tabla de nodos con volumen de flooding (00_flooding_nodes.xlsx)
- Para uno o para todos los nodos, calcular la lámina y zona inundada
  usando el DEM y el volumen de overflow de SWMM.

Requiere:
- volumen_inundacion.py en la misma carpeta (con la clase VolumeInundationModel)
"""



# ==============================
# CONFIGURACIÓN – EDITA SOLO ESTO
# ==============================

# DEM (TU RUTA EXACTA)
DEM_PATH = r"C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\gis\01_raster\elev_DMQ.tif"

# Excel de nodos con flooding (usa el nombre que ya tienes; ajusta la ruta si quieres)
NODES_XLSX = r"C:\Users\Alienware\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\00_flooding_stats\00_flooding_nodes.xlsx"

# Nombres de columnas del Excel (son los que tiene tu archivo)
COL_NODE_ID = "NodeID"
COL_X = "NodeCoordsX"
COL_Y = "NodeCoordsY"
COL_VOLUME = "FloodingVolume"

# ¿Correr para UN nodo o para TODOS?
RUN_ALL_NODES = False     # True = procesa todos los nodos con FloodingVolume > 0

# Si RUN_ALL_NODES = False, se usa este ID de nodo (tú ya tienes P0061405, etc.)
NODE_ID_TO_ANALYZE = "P0061405"


# ==============================
# FUNCIÓN PRINCIPAL
# ==============================

def main():
    # 1. Leer nodos desde Excel
    print(f"Leyendo nodos desde: {NODES_XLSX}")
    nodes = pd.read_excel(NODES_XLSX)

    # Verificar columnas mínimas
    for col in [COL_NODE_ID, COL_X, COL_Y, COL_VOLUME]:
        if col not in nodes.columns:
            raise ValueError(f"ERROR: No se encontró la columna '{col}' en el Excel.")

    # 2. Crear el modelo con el DEM
    print(f"Abriendo DEM: {DEM_PATH}")
    model = VolumeInundationModel(DEM_PATH)

    try:
        if RUN_ALL_NODES:
            # ==============================
            # MODO: TODOS LOS NODOS
            # ==============================
            subset = nodes[nodes[COL_VOLUME] > 0].copy()
            print(f"Procesando {len(subset)} nodos con {COL_VOLUME} > 0 ...")

            for idx, row in subset.iterrows():
                node_id = row[COL_NODE_ID]
                node_x = row[COL_X]
                node_y = row[COL_Y]
                V_overflow = row[COL_VOLUME]

                print(f"\n--- Nodo {node_id} ---")
                print(f"X={node_x}, Y={node_y}, V={V_overflow} m³")

                try:
                    result = model.find_level_for_volume(
                        node_x=node_x,
                        node_y=node_y,
                        V_target=V_overflow,
                        H_min=None,      # por defecto: cota del nodo
                        H_max=None,      # cota nodo + 5 m (se expande si hace falta)
                        tol_volume=1.0,  # tolerancia de volumen (m³)
                        max_iter=40,
                        connectivity=8,
                    )

                    H_agua = result["H"]
                    V_calc = result["V"]
                    area_inund = result["area"]
                    h_media = result["mean_depth"]
                    lamina = result["depth"]

                    print(f"Cota de agua    : {H_agua:.3f} m")
                    print(f"Volumen calc.   : {V_calc:.2f} m³")
                    print(f"Área inundada   : {area_inund:.2f} m²")
                    print(f"Lámina media    : {h_media:.3f} m")

                    # Guardar raster de lámina
                    out_tif = f"lamina_{node_id}.tif"
                    model.save_depth_raster(out_tif, lamina)
                    print(f"Raster de lámina guardado en: {out_tif}")

                    # Si quieres ver el dibujo de TODOS, descomenta esto
                    model.plot_inundation(
                        depth_array=lamina,
                        show_dem=True,
                        title=f"Inundación nodo {node_id}",
                    )

                except Exception as e:
                    print(f"  ERROR procesando nodo {node_id}: {e}")

        else:
            # ==============================
            # MODO: UN SOLO NODO
            # ==============================
            print(f"Buscando nodo con ID = {NODE_ID_TO_ANALYZE!r}")
            sel = nodes[nodes[COL_NODE_ID] == NODE_ID_TO_ANALYZE]

            if sel.empty:
                raise ValueError(
                    f"No se encontró el nodo con ID '{NODE_ID_TO_ANALYZE}' "
                    f"en la columna {COL_NODE_ID}."
                )

            row = sel.iloc[0]
            node_id = row[COL_NODE_ID]
            node_x = row[COL_X]
            node_y = row[COL_Y]
            V_overflow = row[COL_VOLUME]

            print(f"Nodo: {node_id}")
            print(f"X={node_x}, Y={node_y}, Volumen={V_overflow} m³")

            result = model.find_level_for_volume(
                node_x=node_x,
                node_y=node_y,
                V_target=V_overflow,
                H_min=None,      # cota del nodo
                H_max=None,      # cota nodo + 5 m (auto-expansión)
                tol_volume=1.0,
                max_iter=40,
                connectivity=8,
            )

            H_agua = result["H"]
            V_calc = result["V"]
            area_inund = result["area"]
            h_media = result["mean_depth"]
            lamina = result["depth"]

            print("\nRESULTADOS:")
            print(f"Cota de agua    : {H_agua:.3f} m")
            print(f"Volumen calc.   : {V_calc:.2f} m³")
            print(f"Área inundada   : {area_inund:.2f} m²")
            print(f"Lámina media    : {h_media:.3f} m")

            # Guardar raster de lámina
            out_tif = f"lamina_{node_id}.tif"
            model.save_depth_raster(out_tif, lamina)
            print(f"Raster de lámina guardado en: {out_tif}")

            # DIBUJO DE LA ZONA INUNDADA SOBRE EL DEM
            model.plot_inundation(
                depth_array=lamina,
                show_dem=True,
                title=f"Inundación nodo {node_id}",
            )

    finally:
        model.close()
        print("DEM cerrado.")


# ==============================
# EJECUCIÓN
# ==============================
if __name__ == "__main__":
    main()
