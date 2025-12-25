"""
overflow_analyzer_run.py
Corre un modelo SWMM desde Python, analiza overflow de nodos
y genera un paquete completo de gráficos estadísticos.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import re
import geopandas as gpd
from shapely.geometry import Point
import warnings, sys

from tqdm import tqdm
import swmmio
from pyswmm import Output, Simulation
from swmm.toolkit.shared_enum import NodeAttribute


plt.rcParams["savefig.dpi"] = 300
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



def _sanitize(name: str) -> str:
    """
    Reemplaza caracteres ilegales en nombres de archivo por subrayado.
    Útil para Windows y multiplataforma.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


class SWMMOverflowAnalyzer:
    # ---------------------- 0.  Carga, simula y configura --------------- #
    def __init__(
        self,
        inp_path: Path,
        w_vol: float = 0.5,
        w_peak: float = 0.3,
        w_hours: float = 0.2,
        top_timeline: int = 10,
        source_crs: str ="EPSG:32717",
    ):
        self.inp_path = Path(inp_path)
        if not self.inp_path.is_file():
            raise FileNotFoundError(self.inp_path)

        self.rpt_path = self.inp_path.with_suffix(".rpt")
        self.top_timeline = top_timeline

        # pesos
        self.w_vol, self.w_peak, self.w_hours = w_vol, w_peak, w_hours

        # 0.1  Corre el modelo (si no existe .rpt o prefieres forzar)
        self._run_swmm()

        # 0.2  Instancia swmmio
        self.model = swmmio.Model(str(self.inp_path))

        self.df_metrics = self._extract_metrics()
        self._compute_scores()



        self.out_path = self.inp_path.with_suffix(".out")
        self.source_crs = source_crs

    def _run_swmm(self):
        print(f">> Ejecutando SWMM sobre {self.inp_path.name}")
        cwd = os.getcwd()  # guarda cwd actual
        os.chdir(self.inp_path.parent)  # ⇦ cambia a la carpeta del .inp
        try:
            with Simulation(str(self.inp_path)) as sim:
                with tqdm(total=100, desc="SWMM %", unit="%",
                          bar_format="{l_bar}{bar}| {n_fmt}%") as bar:
                    for _ in sim:
                        bar.n = round(sim.percent_complete * 100, 1)  # :contentReference[oaicite:2]{index=2}
                        bar.refresh()
        finally:
            os.chdir(cwd)  # devuelve el cwd
        print("✓ Simulación terminada")
        # comprueba que el .out existe
        self.out_path = self.inp_path.with_suffix(".out").resolve()
        if not self.out_path.exists():
            raise FileNotFoundError(f"No se generó {self.out_path}")


    # ---------------------- 1.  Extrae métricas ------------------------- #
    def _extract_metrics(self):
        with Output(str(self.out_path)) as out:  # fichero binario
            dt = out.report  # Δt en s
            filas = []
            for nid in out.nodes:
                serie = out.node_series(nid, NodeAttribute.FLOODING_LOSSES)
                vals = pd.Series(serie.values(), dtype=float)
                filas.append([
                    nid,
                    (vals * dt).sum(),  # Volumen m³
                    vals.max(),  # Pico m³/s
                    (vals > 0).sum() * dt / 3600  # Horas
                ])
        df = (pd.DataFrame(filas, columns=["NodeID", "Total Vol (m3)",
                                           "Peak Flow (m3/s)", "Hours Flooded"])
              .set_index("NodeID")
              .sort_values("Total Vol (m3)", ascending=False))
        return df

    # ---------------------- 2.  Score y ranking ------------------------- #
    def _compute_scores(self):
        df = self.df_metrics
        norm = (df - df.min()) / (df.max() - df.min())
        df["Score"] = (
            self.w_vol * norm["Total Vol (m3)"]
            + self.w_peak * norm["Peak Flow (m3/s)"]
            + self.w_hours * norm["Hours Flooded"]
        )
        q95, q80 = df["Score"].quantile([0.95, 0.80])
        df["Class"] = np.select(
            [df["Score"] >= q95, df["Score"] >= q80], ["A", "B"], default="C"
        )
        self.df_metrics = df.sort_values("Score", ascending=False)

    # ---------------------- 3.  Utilidades de guardado ------------------ #
    def _save_fig(self, fig, name):
        safe_name = _sanitize(name)
        out_path = self.out_dir / f"{safe_name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # ---------------------- 4.  Gráficos ------------------------------- #
    def plot_hist_kde(self):
        for col in self.df_metrics.columns[:3]:
            data = self.df_metrics[col]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(data, bins=30, density=True, alpha=0.4)
            kde = stats.gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 300)
            ax.plot(xs, kde(xs))
            ax.set_title(f"Histograma + KDE · {col}")
            ax.set_xlabel(col)
            self._save_fig(fig, f"hist_kde_{col.replace(' ', '_')}")

    def plot_box_violin(self):
        cols = self.df_metrics.columns[:3]
        data = [self.df_metrics[c] for c in cols]

        # box
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, vert=False, labels=cols)
        ax.set_title("Boxplot métricas overflow")
        self._save_fig(fig, "boxplot")

        # violin
        fig, ax = plt.subplots(figsize=(6, 4))
        parts = ax.violinplot(data, vert=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)
        ax.set_yticks(range(1, len(cols) + 1))
        ax.set_yticklabels(cols)
        ax.set_title("Violin plot métricas overflow")
        self._save_fig(fig, "violin")

    def plot_pareto(self):
        df = self.df_metrics.sort_values("Total Vol (m3)", ascending=False)
        cum = df["Total Vol (m3)"].cumsum() / df["Total Vol (m3)"].sum() * 100

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(df.index, df["Total Vol (m3)"])
        ax1.tick_params(axis="x", rotation=90, labelsize=6)
        ax1.set_ylabel("Volumen (m3)")

        ax2 = ax1.twinx()
        ax2.plot(df.index, cum, color="red", marker=".")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("% acumulado")
        ax1.set_title("Pareto Volumen Overflow")
        self._save_fig(fig, "pareto")

    def plot_scatter_matrix(self):
        pd.plotting.scatter_matrix(
            self.df_metrics[self.df_metrics.columns[:3]],
            figsize=(8, 8),
            diagonal="kde",
            alpha=0.5,
        )
        plt.suptitle("Scatter matrix métricas overflow")
        fig = plt.gcf()
        self._save_fig(fig, "scatter_matrix")

    def plot_ecdf(self):
        for col in self.df_metrics.columns[:3]:
            data = np.sort(self.df_metrics[col])
            y = np.arange(1, len(data) + 1) / len(data)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.step(data, y, where="post")
            ax.set_xlabel(col)
            ax.set_ylabel("F(x)")
            ax.set_title(f"ECDF – {col}")
            self._save_fig(fig, f"ecdf_{col.replace(' ', '_')}")

    def plot_corr_heatmap(self):
        corr = self.df_metrics[self.df_metrics.columns[:3]].corr()
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns)
        for (i, j), val in np.ndenumerate(corr):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
        ax.set_title("Matriz correlación")
        self._save_fig(fig, "heatmap_corr")

    def plot_qq(self):
        for col in self.df_metrics.columns[:3]:
            fig, ax = plt.subplots(figsize=(4, 4))
            stats.probplot(self.df_metrics[col], dist="norm", plot=ax)
            ax.set_title(f"QQ‑plot – {col}")
            self._save_fig(fig, f"qq_{col.replace(' ', '_')}")

    def plot_timeline(self):
        if not self.out_path.exists():  # self.out_path = inp.with_suffix(".out")
            print("No .out disponible → se omite timeline.")
            return

        # --- abre el binario --------------------------------------------------
        with Output(str(self.out_path)) as out:  # ✔ PySWMM Output
            sim_time = list(out.times)  # lista datetime

            top_nodes = self.df_metrics.head(self.top_timeline).index
            fig, ax = plt.subplots(figsize=(10, 5))

            for nd in top_nodes:
                q = out.node_series(nd, NodeAttribute.FLOODING_LOSSES)
                ax.plot(sim_time, q.values(), label=nd, alpha=0.7)

        ax.set_ylabel("Overflow (m³/s)")
        ax.set_title(f"Serie temporal – top {self.top_timeline} nodos")
        ax.legend(ncol=2, fontsize=8)
        self._save_fig(fig, "timeline")

    def plot_bubble_map(self):
        nodes_df = self.model.nodes()
        nodes_df["geometry"] = [Point(c[0]) for c in nodes_df["coords"]]
        gdf = gpd.GeoDataFrame(nodes_df, geometry="geometry")

        merged = gdf.merge(self.df_metrics[["Total Vol (m3)", "Class"]], left_index=True, right_index=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        merged.plot(
            ax=ax,
            markersize=merged["Total Vol (m3)"] / merged["Total Vol (m3)"].max() * 300,
            column="Class",
            cmap="Set1",
            legend=True,
            alpha=0.7,
        )
        ax.set_axis_off()
        ax.set_title("Mapa burbuja – tamaño ∝ Volumen, color = clase")
        self._save_fig(fig, "bubble_map")

    # ---------------------- 5.  Ejecuta todo --------------------------- #
    def run_all_plots(self):

        # 0.3  Carpeta de salida
        self.out_dir = Path('00_figs_flooding')
        self.out_dir.mkdir(exist_ok=True)

        self.plot_hist_kde()
        self.plot_box_violin()
        self.plot_pareto()
        self.plot_scatter_matrix()
        self.plot_ecdf()
        self.plot_corr_heatmap()
        self.plot_qq()
        self.plot_timeline()
        self.plot_bubble_map()
        print(f"✔ Figuras guardadas en: {self.out_dir.resolve()}")

    def get_all_peak_details(self) -> pd.DataFrame:
        from swmm.toolkit.shared_enum import LinkAttribute

        self.out_dir_stats = Path('00_flooding_stats')
        self.out_dir_stats.mkdir(exist_ok=True)

        # grab node types from swmmio
        nodes_df = self.model.nodes()
        links = self.model.links()

        if "OutfallType" not in nodes_df.columns:
            nodes_df["OutfallType"] = "node"
        else:
            nodes_df['OutfallType'] = nodes_df['OutfallType'].fillna('node')

        if "StorageCurve" not in nodes_df.columns:
            nodes_df["StorageCurve"] = "no_storage"
        else:
            nodes_df['StorageCurve'] = nodes_df['StorageCurve'].fillna('no_storage')




        print(f"▶ Procesando Nudos sobre {self.inp_path.name}")
        records = []
        with Output(str(self.out_path)) as out:
            for nid in tqdm(out.nodes, desc="Processing nodes"):
                # lookup node type
                node_type = nodes_df.at[nid, "OutfallType"]
                storage_type = nodes_df.at[nid, "StorageCurve"]

                if node_type not in ["node"] or storage_type not in ["no_storage"]:
                    continue

                # flooding losses series
                serie = out.node_series(nid, NodeAttribute.FLOODING_LOSSES)
                series = pd.Series(serie.values(), index=list(serie.keys()), dtype=float)
                if series.empty:
                    continue

                peak_time = series.idxmax()
                peak_val = series.max()

                time_in_seconds = series.index.to_series().diff().dt.total_seconds()
                volumen_perdido_total = (time_in_seconds * series).sum()


                if peak_val <= 0.1:
                    continue

                # first incoming and outgoing link IDs
                in_edges = list(self.model.network.in_edges(nid, keys=True))
                out_edges = list(self.model.network.out_edges(nid, keys=True))
                incoming = in_edges[0][2] if in_edges else None
                outgoing = out_edges[0][2] if out_edges else None


                if incoming:
                    diameter_inflow = links.loc[incoming, ['MaxQ', 'MaxV', 'MaxDPerc', "Shape", 'Geom1', 'Geom2', 'Geom3', 'Geom4']]
                    diameter_inflow.index = diameter_inflow.index.map(lambda x: f"InFlow_{x}")

                if outgoing:
                    diameter_outflow = links.loc[outgoing, ['MaxQ', 'MaxV', 'MaxDPerc', "Shape", 'Geom1', 'Geom2', 'Geom3', 'Geom4']]
                    diameter_outflow.index = diameter_outflow.index.map(lambda x: f"OutFlow_{x}")


                dict_out = {
                    "NodeID": nid,
                    "NodeCoordsX": nodes_df.at[nid, "coords"][0][0],
                    "NodeCoordsY": nodes_df.at[nid, "coords"][0][1],
                    
                    "InvertElevation": nodes_df.at[nid, "InvertElev"],
                    "NodeDepth": nodes_df.at[nid, "MaxDepth"],
                    "PeakTime": peak_time,
                    "FloodingFlow": peak_val,
                    "FloodingVolume": volumen_perdido_total,

                }

                dict_out.update(diameter_inflow.to_dict() if diameter_inflow is not None else {})
                dict_out.update(diameter_outflow.to_dict() if diameter_outflow is not None else {})

                records.append(dict_out)

        df = pd.DataFrame(records)
        df = df.sort_values("FloodingFlow", ascending=False).reset_index(drop=True)

        df.to_excel(
            self.out_dir_stats / "00_flooding_nodes.xlsx",
            index=False,
            float_format="%.2f",
        )

        

        gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['NodeCoordsX'], df['NodeCoordsY']), crs=self.source_crs
        )
        gdf.to_file(
            self.out_dir_stats / "00_flooding_nodes.gpkg",
        )

        return df, gdf




if __name__ == "__main__":

    inp_file = Path(r"C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\COLEGIO_TR25_v6.inp").resolve()

    swmm_solver = SWMMOverflowAnalyzer(
        inp_file,
        w_vol=0.5,
        w_peak=0.3,
        w_hours=0.2,
        top_timeline=10,
    )
    df = swmm_solver.get_all_peak_details()

    swmm_solver.run_all_plots()
