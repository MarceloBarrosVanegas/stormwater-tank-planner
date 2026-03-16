"""
Cross-TR Validator
==================
Valida robustez de soluciones optimizadas bajo múltiples periodos de retorno.

Paso 1: Simular baseline para cada TR
Paso 2: Simular solución optimizada con lluvia de cada TR
Paso 3: Comparar sol_trX vs base_trX (validación funcional por TR)
Paso 4: Comparar sol_tr25 vs base_trX (validación cruzada)
"""

import sys
import re
import math
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.stats import gaussian_kde
import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint

# Config
sys.path.insert(0, str(Path(__file__).parent))
import config

config.setup_sys_path()

print("[CrossTR] Iniciando...")
print(f"Config: {config.CODIGOS_DIR}")
print(f"TR Actual: {getattr(config, 'BASE_INP_TR', 25)}")

from rut_27_model_metrics import MetricExtractor, SystemMetrics

# =============================================================================
# PALETA Y ESTILO
# =============================================================================
COLORS = {
    'baseline': '#E05C5C',  # rojo suave
    'solution': '#4A90D9',  # azul
    'rain': '#A8C8E8',  # azul claro para lluvia
    'grid': '#E8E8E8',
    'text': '#2C2C2C',
}

TR_MARKERS = {
    1:  'o',   # círculo
    2:  's',   # cuadrado
    5:  '^',   # triángulo
    10: 'D',   # diamante
    25: '*',   # estrella
    50: 'P',   # cruz gruesa
    100:'X',
}

TR_COLORS = {
    1: '#D62728',
    2: '#FF7F0E',
    5: '#BCBD22',
    10: '#2CA02C',
    25: '#1F77B4',
    50: '#9467BD',
    100: '#8C564B',
}


def _style_ax(ax, title='', xlabel='', ylabel='', legend=True):
    ax.set_facecolor('#FAFAFA')
    ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['text'], pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=COLORS['text'])
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=COLORS['text'])
    ax.tick_params(labelsize=7, colors=COLORS['text'])
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    if legend:
        leg = ax.legend(fontsize=7, framealpha=0.9, edgecolor='#CCCCCC')
        if leg: leg.get_frame().set_linewidth(0.5)


# =============================================================================
# FUNCIONES IDF Y HIETOGRAMA (una sola definición)
# =============================================================================
def calc_intensity(t_min: float, T_yr: float) -> float:
    """Intensidad [mm/h] para duración t_min y periodo T_yr usando curva IDF local."""
    num = 13.9378 * math.log(T_yr) + 40.7176
    den = (35.5037 + t_min) ** 0.9997
    return num / den


def calc_depth(t_min: float, T_yr: float) -> float:
    return calc_intensity(t_min, T_yr) * t_min


def gen_hyetograph(tr: int, dur_min: float = 60, dt_min: float = 5) -> pd.DataFrame:
    """
    Genera hietograma por bloques alternos para periodo de retorno tr.

    Returns:
        DataFrame con columnas: Offset_Min, Time_Str, Block_Depth_mm, Intensity_mm_h
    """
    n = int(dur_min / dt_min)
    durs = np.arange(dt_min, dur_min + dt_min, dt_min)
    cds = np.array([calc_depth(float(d), tr) for d in durs])
    increments = np.diff(np.concatenate([[0.0], cds]))
    sorted_blocks = np.sort(increments)[::-1]

    h = np.zeros(n)
    ri, li = n // 2, n // 2 - 1
    for i, dep in enumerate(sorted_blocks):
        if i % 2 == 0:
            if ri < n:
                h[ri] = dep;
                ri += 1
        else:
            if li >= 0:
                h[li] = dep;
                li -= 1

    tm = np.arange(0, dur_min, dt_min)
    ts = [f"{int(t // 60)}:{int(t % 60):02d}" for t in tm]
    df = pd.DataFrame({'Offset_Min': tm, 'Time_Str': ts, 'Block_Depth_mm': h})
    df['Intensity_mm_h'] = df['Block_Depth_mm'] * (60.0 / dt_min)
    if not df.empty:
        df.loc[0, 'Intensity_mm_h'] = 0.0
    return df


def gen_inp_with_tr(base_content: str, tr: int, tr_actual: int, hyeto: pd.DataFrame, out_path: Path) -> Path:
    """
    Genera archivo INP reemplazando la serie de tiempo de lluvia por hietograma del TR dado.
    """
    old_name = f"TORMENTA_COLEGIO_TR{tr_actual}"
    new_name = f"TORMENTA_COLEGIO_TR{tr}"
    ts_new = f"COLEGIO_TR{tr}"
    ts_old = f"COLEGIO_TR{tr_actual}"

    cont = base_content.replace(old_name, new_name)
    if "[TITLE]" in cont:
        cont = re.sub(r'\[TITLE\]\n.*', f'[TITLE]\nAnalisis - TR {tr} Anios', cont, count=1)
    else:
        cont = f"[TITLE]\nAnalisis - TR {tr} Anios\n\n" + cont

    lines = cont.splitlines()
    new_lines = []
    in_rg = in_ts = False

    for line in lines:
        s = line.strip()
        if s.startswith("["):
            if s.startswith("[RAINGAGES]"):
                in_rg, in_ts = True, False
            elif s.startswith("[TIMESERIES]"):
                in_rg, in_ts = False, True
                new_lines.append(line)
                continue
            else:
                in_rg, in_ts = False, False

        if in_rg and new_name in line and ";" not in line:
            new_lines.append(f"{new_name} INTENSITY 0:05     1.0      TIMESERIES {ts_new}")
        elif in_ts:
            if (s.startswith(ts_old) or s.startswith(ts_new)) and ";" not in s:
                continue
            new_lines.append(line)
        else:
            new_lines.append(line)

    new_lines.append("\n[TIMESERIES]")
    new_lines.append(f";;TR={tr}")
    new_lines.append(";;Name           Date       Time       Value")
    for _, row in hyeto.iterrows():
        new_lines.append(f"{ts_new:<16}           {row['Time_Str']:<10} {row['Intensity_mm_h']:.4f}")

    out_path.write_text("\n".join(new_lines), encoding='latin-1', errors='replace')
    return out_path


# =============================================================================
# HELPERS DE SERIES DE TIEMPO
# =============================================================================
def _normalize_times(times_array) -> np.ndarray:
    """Convierte array de timestamps a minutos desde t=0."""
    if len(times_array) == 0:
        return np.array([])
    t0 = pd.Timestamp(times_array[0])
    return np.array([(pd.Timestamp(t) - t0).total_seconds() / 60 for t in times_array])


def _cumulative_volume(times_min: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Volumen acumulado [m³] a partir de caudales [m³/s] y tiempos [min]."""
    if len(times_min) < 2:
        return np.zeros(len(times_min))
    dt_sec = np.diff(times_min) * 60
    dv = rates[:-1] * dt_sec
    return np.concatenate([[0.0], np.cumsum(dv)])


def _get_capacity_distribution(metrics: SystemMetrics) -> np.ndarray:
    """Extrae valores máximos de h/D (Capacity) para todos los links."""
    vals = []
    for lid, data in metrics.link_data.items():
        cap_series = data.get('capacity_series', pd.Series())
        if len(cap_series) > 0:
            vals.append(float(cap_series.max()))
    return np.array(vals)


# =============================================================================
# CLASE PRINCIPAL
# =============================================================================
class CrossTRValidator:

    def __init__(self, work_dir=None):
        self.baseline_inp = Path(config.SWMM_FILE)
        self.tr_actual = getattr(config, 'BASE_INP_TR', 25)
        self.tr_list = getattr(config, 'CROSS_TR_VALIDATION_LIST', [1, 2, 5, 10, 25, 50])

        if work_dir is None:
            self.work_dir = Path(config.CODIGOS_DIR) / "optimization_results" / "00_Baseline"
        else:
            self.work_dir = Path(work_dir)

        self.scenarios_dir = self.work_dir / "scenarios"
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

        # Almacenar extractor completo (con series de tiempo)
        self.baseline_extractors: dict[int, MetricExtractor] = {}

        print(f"[CrossTR] TR List: {self.tr_list}")
        self._generate_baselines()

    # -------------------------------------------------------------------------
    # PASO 1: Simular baselines
    # -------------------------------------------------------------------------
    def _generate_baselines(self):
        with open(self.baseline_inp, 'r', encoding='latin-1', errors='replace') as f:
            base_content = f.read()

        print(f"\n[CrossTR] Generando {len(self.tr_list)} baselines...")

        for tr in self.tr_list:
            print(f"\n  [TR{tr}] Generando INP...")
            sp = self.scenarios_dir / f"baseline_TR{tr:03d}.inp"

            if tr == self.tr_actual:
                with open(sp, 'w', encoding='latin-1') as f:
                    f.write(base_content)
                print(f"    Copiado base TR{self.tr_actual}")
            else:
                h = gen_hyetograph(tr)
                gen_inp_with_tr(base_content, tr, self.tr_actual, h, sp)
                print(f"    Hietograma generado: total={h['Block_Depth_mm'].sum():.1f} mm")

            print(f"    Simulando...")
            ex = MetricExtractor(
                project_root=config.PROJECT_ROOT,
                predios_path=config.PREDIOS_FILE
            )
            ex.run(sp)
            self.baseline_extractors[tr] = ex
            print(f"    Flooding: {ex.metrics.total_flooding_volume:,.0f} m³  |  "
                  f"Peak: {ex.metrics.total_max_flooding_flow:.3f} m³/s")

        print(f"\n[CrossTR] {len(self.baseline_extractors)} baselines listos.")

    @property
    def baseline_metrics(self) -> dict:
        """Resumen escalar de métricas baseline (compatibilidad hacia atrás)."""
        return {
            tr: {
                'flood_volume': ex.metrics.total_flooding_volume,
                'flood_peak': ex.metrics.total_max_flooding_flow,
                'outfall_peak': ex.metrics.total_max_outfall_flow,
            }
            for tr, ex in self.baseline_extractors.items()
        }

    # -------------------------------------------------------------------------
    # PASO 2 & 3 & 4: Comparar solución
    # -------------------------------------------------------------------------
    def compare_solution(self, sol_inp: Path, name: str = "sol") -> dict:
        """
        Corre la solución una sola vez (con tr_actual) y compara contra todos
        los baseline_extractors ya en memoria. No re-simula los baselines.

        Args:
            sol_inp: INP de la solución (se corre solo con tr_actual).
            name:    Nombre identificador para archivos de salida.

        Returns:
            dict con métricas resumen por TR.
        """
        sol_inp = Path(sol_inp)

        # Directorio de salida numerado según carpetas existentes
        existing = [p for p in sol_inp.parent.iterdir() if p.is_dir()]
        next_idx = len(existing) + 1
        out_dir      = sol_inp.parent / f"{next_idx:02d}_compare_tr"
        sol_scen_dir = out_dir / "modelos"
        sol_fig_dir  = out_dir  # Guardar figuras directamente en compare_tr, sin subcarpeta 'figuras'
        sol_scen_dir.mkdir(parents=True, exist_ok=True)
        sol_fig_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[CrossTR] Comparando solución: {name}")
        print(f"          Output → {out_dir}")

        # -- PASO 2: Correr solución para cada TR --
        with open(sol_inp, 'r', encoding='latin-1', errors='replace') as f:
            sol_content = f.read()

        sol_extractors: dict[int, MetricExtractor] = {}
        for tr in self.tr_list:
            sp = sol_scen_dir / f"sol_{name}_TR{tr:03d}.inp"
            if tr == self.tr_actual:
                with open(sp, 'w', encoding='latin-1') as f:
                    f.write(sol_content)
            else:
                h = gen_hyetograph(tr)
                gen_inp_with_tr(sol_content, tr, self.tr_actual, h, sp)

            print(f"  [Sol TR{tr}] Simulando...")
            ex = MetricExtractor(
                project_root=config.PROJECT_ROOT,
                predios_path=config.PREDIOS_FILE
            )
            ex.run(sp)
            sol_extractors[tr] = ex
            print(f"    Flooding: {ex.metrics.total_flooding_volume:,.0f} m³  |  "
                  f"Peak: {ex.metrics.total_max_flooding_flow:.3f} m³/s")

        # -- PASO 3: Gráficos funcionales (base_trX vs sol_trX) --
        print("\n[CrossTR] Generando gráficos Paso 3 (funcional)...")
        self._plot_paso3_functional(sol_extractors, sol_fig_dir, name)

        # -- PASO 4: Gráficos cruzados (sol_tr25 vs base_trX) --
        print("[CrossTR] Generando gráficos Paso 4 (cross-TR)...")
        self._plot_paso4_cross_tr(sol_extractors[self.tr_actual], sol_fig_dir, name)

        # -- Extras: Risk Curve, Node Map, Radar --
        print("[CrossTR] Generando gráficos adicionales...")
        self._plot_risk_curve(sol_extractors, sol_fig_dir, name)
        self._plot_node_map(sol_extractors, sol_fig_dir, name)
        self._plot_radar(sol_extractors, sol_fig_dir, name)

        # -- CSV resumen --
        self._export_summary_csv(sol_extractors, sol_fig_dir, name)
        self._export_nodes_csv(sol_extractors, sol_fig_dir, name)

        print(f"\n[CrossTR] Guardado en: {out_dir}")
        self.out_dir = out_dir  # Guardar referencia para uso externo
        
        # Retornar resumen
        return {
            tr: {
                'base_flood_vol':  self.baseline_extractors[tr].metrics.total_flooding_volume,
                'sol_flood_vol':   sol_extractors[tr].metrics.total_flooding_volume,
                'base_flood_peak': self.baseline_extractors[tr].metrics.total_max_flooding_flow,
                'sol_flood_peak':  sol_extractors[tr].metrics.total_max_flooding_flow,
            }
            for tr in self.tr_list
        }

    # -------------------------------------------------------------------------
    # PASO 3: Gráfico funcional — base_trX vs sol_trX para cada TR
    # -------------------------------------------------------------------------
    def _plot_paso3_functional(self, sol_extractors: dict, out_dir: Path, name: str):
        """
        Una figura por TR con 4 paneles:
          [0,0] Flooding Flow (m³/s) vs tiempo
          [0,1] Outfall Flow (m³/s) vs tiempo
          [1,0] Volumen Acumulado de Flooding (m³)
          [1,1] Distribución de Capacidad h/D (CDF)
        """
        for tr in self.tr_list:
            base_m = self.baseline_extractors[tr].metrics
            sol_m = sol_extractors[tr].metrics

            fig = plt.figure(figsize=(14, 9))
            fig.patch.set_facecolor('white')

            # Supertítulo
            red_pct = 0.0
            if base_m.total_flooding_volume > 0:
                red_pct = (base_m.total_flooding_volume - sol_m.total_flooding_volume) / base_m.total_flooding_volume * 100
            fig.suptitle(
                f"Paso 3 · Validación Funcional  ·  TR{tr} años  ·  {name}\n"
                f"Reducción flooding: {red_pct:+.1f}%  |  "
                f"Base: {base_m.total_flooding_volume:,.0f} m³  →  "
                f"Sol: {sol_m.total_flooding_volume:,.0f} m³",
                fontsize=11, fontweight='bold', color=COLORS['text'], y=0.98
            )

            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                                   left=0.08, right=0.97, top=0.90, bottom=0.08)

            ax_ff = fig.add_subplot(gs[0, 0])  # Flooding Flow
            ax_of = fig.add_subplot(gs[0, 1])  # Outfall Flow
            ax_vol = fig.add_subplot(gs[1, 0])  # Volumen acumulado
            ax_cap = fig.add_subplot(gs[1, 1])  # Distribución Capacidad

            # ── Panel 1: Flooding Flow ──────────────────────────────────────
            self._plot_flow_series(
                ax_ff,
                base_m.system_flood_hydrograph,
                sol_m.system_flood_hydrograph,
                title=f'Caudal de Flooding  [TR{tr}]',
                ylabel='Caudal (m³/s)',
            )

            # ── Panel 2: Outfall Flow ───────────────────────────────────────
            self._plot_flow_series(
                ax_of,
                base_m.system_outfall_flow_hydrograph,
                sol_m.system_outfall_flow_hydrograph,
                title=f'Caudal Outfall  [TR{tr}]',
                ylabel='Caudal (m³/s)',
            )

            # ── Panel 3: Volumen acumulado de flooding ──────────────────────
            self._plot_cumulative_volume(ax_vol, base_m, sol_m, tr)

            # ── Panel 4: Distribución de capacidad (h/D) ───────────────────
            self._plot_capacity_distribution(ax_cap, base_m, sol_m, tr)

            fig.savefig(out_dir / f"paso3_TR{tr:03d}_{name}.png",
                        dpi=160, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"    Guardado: paso3_TR{tr:03d}_{name}.png")

    # -------------------------------------------------------------------------
    # PASO 4: Gráfico cross-TR — sol_tr25 vs base_trX
    # -------------------------------------------------------------------------
    def _plot_paso4_cross_tr(self, sol_tr_actual: MetricExtractor, out_dir: Path, name: str):
        """
        Una figura con 4 paneles mostrando sol_tr25 vs cada baseline_trX.
        Base TR_actual se incluye SIEMPRE en las series de tiempo como referencia
        de diseño (línea gruesa), aunque no aparezca en el bar chart cruzado.
        """
        tr_cross = [tr for tr in self.tr_list if tr != self.tr_actual]

        if not tr_cross:
            return

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        fig.suptitle(
            f"Paso 4 · Validación Cruzada  ·  Sol TR{self.tr_actual} vs Baselines  ·  {name}",
            fontsize=12, fontweight='bold', color=COLORS['text'], y=0.99
        )

        # gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
        #                        left=0.08, right=0.97, top=0.93, bottom=0.08)

        # DESPUÉS
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                               left=0.08, right=0.97, top=0.93, bottom=0.12)

        ax_ff = fig.add_subplot(gs[0, 0])
        ax_of = fig.add_subplot(gs[0, 1])
        ax_vol = fig.add_subplot(gs[1, 0])
        ax_bar = fig.add_subplot(gs[1, 1])

        sol_m = sol_tr_actual.metrics
        base_ref_m = self.baseline_extractors[self.tr_actual].metrics  # Base TR25 = referencia

        # ── Paneles 1 & 2: Flooding Flow y Outfall Flow ────────────────────
        for ax, hydro_key, title, ylabel in [
            (ax_ff, 'system_flood_hydrograph',
             f'Flooding Flow: Sol TR{self.tr_actual} vs Bases', 'Caudal Flooding (m³/s)'),
            (ax_of, 'system_outfall_flow_hydrograph',
             f'Outfall Flow: Sol TR{self.tr_actual} vs Bases', 'Caudal Outfall (m³/s)'),
        ]:
            ax.set_facecolor('#FAFAFA')
            ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)

            # 1) Baselines de otros TR primero (fondo, líneas finas y tenues)
            for tr in tr_cross:
                base_hydro = getattr(self.baseline_extractors[tr].metrics, hydro_key, {})
                if base_hydro.get('times', np.array([])).size > 0:
                    t_b = _normalize_times(base_hydro['times'])
                    r_b = base_hydro['total_rate']
                    mk  = TR_MARKERS.get(tr, 'o')
                    # markers cada ~15 puntos para no saturar
                    every = max(1, len(t_b) // 15)
                    ax.plot(t_b, r_b,
                            color=TR_COLORS.get(tr, '#888888'), linewidth=0.9,
                            alpha=0.6, label=f'Base TR{tr}',
                            marker=mk, markevery=every, markersize=5,
                            markeredgewidth=0.5, zorder=3)

            # 2) Base TR_actual como referencia de diseño
            base_ref_hydro = getattr(base_ref_m, hydro_key, {})
            if base_ref_hydro.get('times', np.array([])).size > 0:
                t_ref = _normalize_times(base_ref_hydro['times'])
                mk = TR_MARKERS.get(self.tr_actual, '*')
                every = max(1, len(t_ref) // 15)
                ax.plot(t_ref, base_ref_hydro['total_rate'],
                        color=TR_COLORS.get(self.tr_actual, '#1F77B4'),
                        linewidth=2.0, linestyle='-', alpha=0.85,
                        label=f'Base TR{self.tr_actual} (diseño)',
                        marker=mk, markevery=every, markersize=7,
                        markeredgewidth=0.5, zorder=5)

            # 3) Solución — línea negra gruesa, sin marker (protagonista)
            sol_hydro = getattr(sol_m, hydro_key, {})
            if sol_hydro.get('times', np.array([])).size > 0:
                t_sol = _normalize_times(sol_hydro['times'])
                ax.plot(t_sol, sol_hydro['total_rate'],
                        color='#111111', linewidth=3.0, linestyle='-',
                        label=f'Sol TR{self.tr_actual} ✓', zorder=8)

            _style_ax(ax, title=title, xlabel='Tiempo (min)', ylabel=ylabel)

        # ── Panel 3: Volumen acumulado ──────────────────────────────────────
        ax_vol.set_facecolor('#FAFAFA')
        ax_vol.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)

        for tr in tr_cross:
            base_hydro = self.baseline_extractors[tr].metrics.system_flood_hydrograph
            if base_hydro.get('times', np.array([])).size > 0:
                t_b = _normalize_times(base_hydro['times'])
                v_b = _cumulative_volume(t_b, base_hydro['total_rate'])
                mk = TR_MARKERS.get(tr, 'o')
                every = max(1, len(t_b) // 15)
                ax_vol.plot(t_b, v_b / 1000,
                            color=TR_COLORS.get(tr, '#888888'), linewidth=0.9,
                            alpha=0.6, label=f'Base TR{tr}',
                            marker=mk, markevery=every, markersize=5,
                            markeredgewidth=0.5, zorder=3)

        ref_hydro = base_ref_m.system_flood_hydrograph
        if ref_hydro.get('times', np.array([])).size > 0:
            t_ref = _normalize_times(ref_hydro['times'])
            v_ref = _cumulative_volume(t_ref, ref_hydro['total_rate'])
            mk = TR_MARKERS.get(self.tr_actual, '*')
            every = max(1, len(t_ref) // 15)
            ax_vol.plot(t_ref, v_ref / 1000,
                        color=TR_COLORS.get(self.tr_actual, '#1F77B4'),
                        linewidth=2.0, alpha=0.85,
                        label=f'Base TR{self.tr_actual} (diseño)',
                        marker=mk, markevery=every, markersize=7,
                        markeredgewidth=0.5, zorder=5)

        sol_hydro = sol_m.system_flood_hydrograph
        if sol_hydro.get('times', np.array([])).size > 0:
            t_sol = _normalize_times(sol_hydro['times'])
            v_sol = _cumulative_volume(t_sol, sol_hydro['total_rate'])
            ax_vol.plot(t_sol, v_sol / 1000,
                        color='#111111', linewidth=3.0,
                        label=f'Sol TR{self.tr_actual} ✓', zorder=8)

        _style_ax(ax_vol,
                  title=f'Volumen Acumulado Flooding: Sol TR{self.tr_actual} vs Bases',
                  xlabel='Tiempo (min)', ylabel='Volumen Acumulado (10³ m³)')

        # ── Panel 4: TR equivalente — ¿a qué baseline se parece la solución? ─
        ax_bar.set_facecolor('#FAFAFA')

        # Calcular las 3 métricas para la solución
        sol_ff_peak = max(sol_m.system_flood_hydrograph.get('total_rate', [0]))
        sol_vol = sol_m.total_flooding_volume
        sol_of_peak = max(sol_m.system_outfall_flow_hydrograph.get('total_rate', [0]))

        # Calcular diferencias CON SIGNO para cada TR y cada métrica
        # Positivo = baseline tiene más que solución (solución redujo)
        # Negativo = baseline tiene menos que solución (solución empeoró)
        tr_metrics = {}
        for tr in self.tr_list:
            base_m = self.baseline_extractors[tr].metrics
            
            base_ff = max(base_m.system_flood_hydrograph.get('total_rate', [0]))
            base_vol = base_m.total_flooding_volume
            base_of = max(base_m.system_outfall_flow_hydrograph.get('total_rate', [0]))

            tr_metrics[tr] = {
                'ff': {
                    'value': base_ff,
                    'sol_value': sol_ff_peak,
                    'diff': (sol_ff_peak - base_ff) / max(base_ff, 1e-10) * 100,
                },
                'vol': {
                    'value': base_vol,
                    'sol_value': sol_vol,
                    'diff': (sol_vol - base_vol) / max(base_vol, 1e-10) * 100,
                },
                'of': {
                    'value': base_of,
                    'sol_value': sol_of_peak,
                    'diff': (sol_of_peak - base_of) / max(base_of, 1e-10) * 100,
                },
            }

        # Colores por métrica
        metric_colors = {'ff': '#E74C3C', 'vol': '#3498DB', 'of': '#2ECC71'}
        metric_names = {'ff': 'Flooding Flow Peak', 'vol': 'Flooding Volume', 'of': 'Outfall Flow Peak'}

        # Función para dibujar grupo de barras con signo
        def _plot_metric_group(ax, y_start, metric_key, label):
            # Diferencias con signo para graficar
            signed_diffs = [tr_metrics[tr][metric_key]['diff'] for tr in self.tr_list]
            
            # Best match: el TR que cumple diff <= tolerancia (con fuzzy)
            # Si ninguno cumple, el que tenga menor diff (mejor reducción)
            # DESPUÉS
            TOLERANCE = getattr(config, 'CROSS_TR_TOLERANCE', 0.10)
            FUZZY = getattr(config, 'CROSS_TR_FUZZY_ATOL', 0.02)

            best_idx = None
            for i, tr in enumerate(self.tr_list):
                sol_val = tr_metrics[tr][metric_key]['sol_value']
                base_val = tr_metrics[tr][metric_key]['value']
                if sol_val <= base_val * (1 + TOLERANCE + FUZZY):
                    best_idx = i
                    break

            # Si ninguno cumple, el de menor diff (menos peor)
            if best_idx is None:
                best_idx = signed_diffs.index(min(signed_diffs))
            
            color = metric_colors[metric_key]
            
            # Título del grupo
            ax.text(-2, y_start + 0.6, label, va='bottom', ha='right',
                    fontsize=9, fontweight='bold', color='#2C2C2C')
            
            # Dibujar barras
            for i, tr in enumerate(self.tr_list):
                y = y_start - i * 0.7
                diff = signed_diffs[i]
                is_best = (i == best_idx)
                is_design = (tr == self.tr_actual)
                
                # Color más intenso si es el mejor, más tenue si no
                alpha = 1.0 if is_best else 0.35
                edge = '#2C2C2C' if is_best else 'white'
                lw = 2 if is_best else 0.5
                
                # Dibujar barra - puede ser positiva o negativa
                ax.barh(y, diff, height=0.55, color=color, alpha=alpha,
                        edgecolor=edge, linewidth=lw, zorder=3)
                
                # Etiqueta TR - siempre al lado opuesto de la barra
                tr_lbl = f'TR{tr}'
                if is_design:
                    tr_lbl += ' ★'
                if is_best:
                    tr_lbl += ' ✓'
                
                weight = 'bold' if (is_best or is_design) else 'normal'
                lbl_color = '#2E7D32' if is_best else ('#1F77B4' if is_design else '#666666')
                
                # Posición: lado opuesto a la dirección de la barra
                # Si barra positiva (derecha), label va a la izquierda
                # Si barra negativa (izquierda), label va a la derecha
                if diff >= 0:
                    lbl_x = -2  # Izquierda del 0
                    lbl_ha = 'right'
                else:
                    lbl_x = 2   # Derecha del 0
                    lbl_ha = 'left'
                
                ax.text(lbl_x, y, tr_lbl, va='center', ha=lbl_ha, fontsize=8,
                        fontweight=weight, color=lbl_color)
                
                # Valor con signo + o -
                sign = '+' if diff >= 0 else ''
                val_text = f'{sign}{diff:.1f}%'
                
                # Posición del texto: más alejado de la barra para evitar solapamiento
                offset = max(abs(diff) * 0.15, 5)  # Mínimo 5% de offset
                if diff >= 0:
                    text_x = diff + offset
                    ha = 'left'
                else:
                    text_x = diff - offset
                    ha = 'right'
                    
                ax.text(text_x, y, val_text, va='center', fontsize=8,
                        fontweight='bold' if is_best else 'normal', color='#2C2C2C',
                        ha=ha)
            
            return signed_diffs[best_idx], self.tr_list[best_idx], y_start - len(self.tr_list) * 0.7

        # Dibujar los 3 grupos
        y_pos = 13
        results = []
        
        min_ff, best_ff, y_pos = _plot_metric_group(ax_bar, y_pos, 'ff', 'Flooding Flow Peak')
        results.append(('Flooding Flow', best_ff, min_ff))
        
        y_pos -= 0.8
        min_vol, best_vol, y_pos = _plot_metric_group(ax_bar, y_pos, 'vol', 'Flooding Volume')
        results.append(('Flooding Volume', best_vol, min_vol))
        
        y_pos -= 0.8
        min_of, best_of, y_pos = _plot_metric_group(ax_bar, y_pos, 'of', 'Outfall Flow Peak')
        results.append(('Outfall Flow', best_of, min_of))

        # Configurar ejes - ahora permite valores negativos y positivos
        all_diffs = []
        for tr in self.tr_list:
            all_diffs.extend([
                tr_metrics[tr]['ff']['diff'],
                tr_metrics[tr]['vol']['diff'],
                tr_metrics[tr]['of']['diff']
            ])
        
        max_val = max(abs(min(all_diffs)), abs(max(all_diffs))) * 1.2
        ax_bar.set_xlim(-max_val, max_val)
        ax_bar.set_ylim(y_pos - 1, 15)
        ax_bar.set_xscale('symlog', linthresh=50)  # lineal entre -50% y +50%, log fuera
        ax_bar.set_xlabel('Diferencia vs Solución (%)  ← Menos flooding / Más flooding →', 
                          fontsize=8, color=COLORS['text'])
        ax_bar.set_yticks([])
        
        # Línea vertical en 0 (referencia)
        ax_bar.axvline(x=0, color='#333333', linewidth=1.5, linestyle='-', zorder=1)
        
        # Grid y spines
        ax_bar.grid(True, axis='x', color='#E0E0E0', linewidth=0.8, zorder=0)
        ax_bar.spines['left'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['bottom'].set_color('#CCCCCC')

        # Título
        ax_bar.set_title(f'¿A qué TR se parece la Solución TR{self.tr_actual}?\n(comparación por métrica)',
                         fontsize=10, fontweight='bold', color=COLORS['text'], pad=10)

        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=metric_colors['ff'], label='Flooding Flow', alpha=0.8),
            Patch(facecolor=metric_colors['vol'], label='Flooding Volume', alpha=0.8),
            Patch(facecolor=metric_colors['of'], label='Outfall Flow', alpha=0.8),
        ]
        ax_bar.legend(handles=legend_elements, loc='lower right', fontsize=8,
                      framealpha=0.9, edgecolor='#CCCCCC', title='Métricas')

        # Resumen
        note_lines = ['Más similar por métrica:']
        for metric, tr, diff in results:
            note_lines.append(f'✓ {metric}: TR{tr} ({diff:.1f}%)')
        note_lines.append('★ TR de diseño')
        
        # ax_bar.text(0.98, 0.98, '\n'.join(note_lines), transform=ax_bar.transAxes,
        #             fontsize=8, va='top', ha='right', color='#2C2C2C', linespacing=1.2,
        #             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
        #                       edgecolor='#4CAF50', linewidth=1.5))

        # DESPUÉS
        note_parts = ['Más similar por métrica:']
        for metric, tr, diff in results:
            sign = '+' if diff >= 0 else ''
            note_parts.append(f'✓ {metric}: TR{tr} ({sign}{diff:.1f}%)')
        note_parts.append('★ TR de diseño')

        fig.text(0.5, 0.02, '     '.join(note_parts),
                 ha='center', va='bottom', fontsize=9, color='#2C2C2C',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                           edgecolor='#4CAF50', linewidth=1.5))

        # Guardar datos de comparación a CSV para uso en rut_15
        import csv
        # Guardar tabla de comparacion a CSV
        csv_path = out_dir / f"cross_tr_comparison_{name}.csv"
        # Tolerancia de 0-1 (ej: 0.15 = 15%)
        SIMILARITY_TOLERANCE = getattr(config, 'CROSS_TR_TOLERANCE', 0.15) * 100
        
        # Encontrar el TR con menor diferencia (best match) para cada métrica
        ff_best_tr = min(self.tr_list, key=lambda tr: abs(tr_metrics[tr]['ff']['diff']))
        vol_best_tr = min(self.tr_list, key=lambda tr: abs(tr_metrics[tr]['vol']['diff']))
        of_best_tr = min(self.tr_list, key=lambda tr: abs(tr_metrics[tr]['of']['diff']))
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            cw = csv.writer(f)
            # Encabezados con valores base, solución, diferencia absoluta y diferencia %
            cw.writerow(['TR', 
                         'ff_base', 'ff_sol', 'ff_diff_abs', 'flooding_flow',
                         'vol_base', 'vol_sol', 'vol_diff_abs', 'flooding_volume', 
                         'of_base', 'of_sol', 'of_diff_abs', 'outfall_peak_flow',
                         'TR_Design', 'Best_Match',
                         'ff_best_tr', 'vol_best_tr', 'of_best_tr'])
            
            for tr in self.tr_list:
                ff_diff = tr_metrics[tr]['ff']['diff']
                vol_diff = tr_metrics[tr]['vol']['diff']
                of_diff = tr_metrics[tr]['of']['diff']
                ff_base = tr_metrics[tr]['ff']['value']
                vol_base = tr_metrics[tr]['vol']['value']
                of_base = tr_metrics[tr]['of']['value']
                is_design = tr == self.tr_actual
                
                # Calcular valores de solución a partir de la diferencia
                ff_sol = ff_base * (1 + ff_diff/100) if ff_base > 0 else 0
                vol_sol = vol_base * (1 + vol_diff/100) if vol_base > 0 else 0
                of_sol = of_base * (1 + of_diff/100) if of_base > 0 else 0
                
                # Diferencias absolutas (sol - base)
                ff_diff_abs = ff_sol - ff_base
                vol_diff_abs = vol_sol - vol_base
                of_diff_abs = of_sol - of_base
                
                # Verificar si este TR esta dentro de tolerancia para todas las metricas
                ff_ok = abs(ff_diff) <= SIMILARITY_TOLERANCE
                vol_ok = abs(vol_diff) <= SIMILARITY_TOLERANCE
                of_ok = abs(of_diff) <= SIMILARITY_TOLERANCE
                best_match = ff_ok and vol_ok and of_ok
                
                cw.writerow([tr, 
                            f"{ff_base:.2f}", f"{ff_sol:.2f}", f"{ff_diff_abs:.2f}", f"{ff_diff:.2f}",
                            f"{vol_base:.2f}", f"{vol_sol:.2f}", f"{vol_diff_abs:.2f}", f"{vol_diff:.2f}",
                            f"{of_base:.2f}", f"{of_sol:.2f}", f"{of_diff_abs:.2f}", f"{of_diff:.2f}",
                            is_design, best_match,
                            ff_best_tr, vol_best_tr, of_best_tr])
        print(f"    Guardado: cross_tr_comparison_{name}.csv")

        fig.savefig(out_dir / f"paso4_cross_tr_{name}.png",
                    dpi=160, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Guardado: paso4_cross_tr_{name}.png")



    # -------------------------------------------------------------------------
    # HELPERS DE PLOTS
    # -------------------------------------------------------------------------
    def _plot_flow_series(self, ax, base_hydro: dict, sol_hydro: dict,
                          title: str, ylabel: str):
        """Grafica dos hidrogramas (baseline vs solución) en el mismo eje."""
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)

        bt = base_hydro.get('times', np.array([]))
        br = base_hydro.get('total_rate', np.array([]))
        st = sol_hydro.get('times', np.array([]))
        sr = sol_hydro.get('total_rate', np.array([]))

        if bt.size > 0:
            t_b = _normalize_times(bt)
            ax.fill_between(t_b, br, alpha=0.15, color=COLORS['baseline'], zorder=2)
            ax.plot(t_b, br, color=COLORS['baseline'], linewidth=1.8,
                    label='Baseline', zorder=3)

        if st.size > 0:
            t_s = _normalize_times(st)
            ax.fill_between(t_s, sr, alpha=0.15, color=COLORS['solution'], zorder=2)
            ax.plot(t_s, sr, color=COLORS['solution'], linewidth=1.8,
                    linestyle='--', label='Solución', zorder=3)

        _style_ax(ax, title=title, xlabel='Tiempo (min)', ylabel=ylabel)

    def _plot_cumulative_volume(self, ax, base_m: SystemMetrics, sol_m: SystemMetrics, tr: int):
        """Volumen acumulado de flooding (m³)."""
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)

        for metrics, color, label, ls in [
            (base_m, COLORS['baseline'], f'Baseline  ({base_m.total_flooding_volume:,.0f} m³)', '-'),
            (sol_m, COLORS['solution'], f'Solución  ({sol_m.total_flooding_volume:,.0f} m³)', '--'),
        ]:
            hydro = metrics.system_flood_hydrograph
            times = hydro.get('times', np.array([]))
            rates = hydro.get('total_rate', np.array([]))
            if times.size > 1:
                t_min = _normalize_times(times)
                v_cum = _cumulative_volume(t_min, rates)
                ax.plot(t_min, v_cum, color=color, linewidth=1.8,
                        linestyle=ls, label=label, zorder=3)

        _style_ax(ax,
                  title=f'Volumen Acumulado Flooding  [TR{tr}]',
                  xlabel='Tiempo (min)', ylabel='Volumen Acumulado (m³)')

    def _plot_capacity_distribution(self, ax, base_m: SystemMetrics, sol_m: SystemMetrics, tr: int):
        """
        Distribución de frecuencia (KDE + histograma) de utilización h/D por tubería.
        Muestra cómo se distribuye la carga en la red, no la acumulada.
        """
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, color=COLORS['grid'], linewidth=0.8, zorder=0)

        x_max = 0.0
        for metrics, color, label, alpha_hist in [
            (base_m, COLORS['baseline'], 'Baseline', 0.25),
            (sol_m, COLORS['solution'], 'Solución', 0.20),
        ]:
            vals = _get_capacity_distribution(metrics)
            if len(vals) < 3:
                continue
            vals = vals[vals > 0]
            if len(vals) < 3:
                continue
            x_max = max(x_max, np.percentile(vals, 99))

            # Histograma de fondo
            ax.hist(vals, bins=25, density=True, color=color, alpha=alpha_hist,
                    zorder=2)
            # KDE encima
            try:
                kde = gaussian_kde(vals, bw_method='silverman')
                x_kde = np.linspace(0, max(vals) * 1.05, 300)
                ax.plot(x_kde, kde(x_kde), color=color, linewidth=1.8,
                        label=f'{label} (n={len(vals)})', zorder=3)
            except Exception:
                pass

        # Umbrales
        ax.axvline(x=1.0, color='#FF6B35', linewidth=1.2, linestyle=':',
                   label='Saturación (h/D=1.0)', zorder=4)
        ax.axvline(x=0.8, color='#FFC107', linewidth=1.0, linestyle=':',
                   label='Diseño (h/D=0.8)', zorder=4)

        ax.set_xlim(left=0, right=min(x_max * 1.1, 1.5) if x_max > 0 else 1.5)
        ax.set_ylim(bottom=0)
        _style_ax(ax,
                  title=f'Distribución Frecuencia h/D Tuberías  [TR{tr}]',
                  xlabel='Utilización máxima (h/D)',
                  ylabel='Densidad')

    # -------------------------------------------------------------------------
    # EXTRA 1: Curva de Riesgo — todas las variables vs TR (escala log)
    # -------------------------------------------------------------------------
    def _plot_risk_curve(self, sol_extractors: dict, out_dir: Path, name: str):
        """
        Curva de riesgo en escala log. 6 paneles (2 filas x 3 columnas).
        Labels de % en cajas coloreadas alternando arriba/abajo para evitar solapamiento.
        """
        trs = self.tr_list

        def _get(extractors_dict, tr, attr):
            return getattr(extractors_dict[tr].metrics, attr, 0.0) or 0.0

        panels = [
            ('total_flooding_volume', 'Volumen de Flooding', 'm³', '↓ mejor', True),
            ('total_max_flooding_flow', 'Caudal Pico Flooding', 'm³/s', '↓ mejor', True),
            ('total_outfall_volume', 'Volumen Outfall', 'm³', '↑ mejor', False),
            ('total_max_outfall_flow', 'Caudal Pico Outfall', 'm³/s', '↑ mejor', False),
            ('overloaded_links_length', 'Links Saturados', 'm', '↓ mejor', True),
            ('network_health_score', 'Network Health Score', '0–1', '↑ mejor', False),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Curvas de Riesgo — Todas las Variables  ·  {name}',
                     fontsize=15, fontweight='bold', color=COLORS['text'], y=0.99)

        axes_flat = axes.flatten()

        for idx, (attr, title, unit, direction, lower_better) in enumerate(panels):
            ax = axes_flat[idx]
            ax.set_facecolor('#FAFAFA')
            ax.grid(True, color=COLORS['grid'], linewidth=0.9, which='both', zorder=0)

            b_vals = [_get(self.baseline_extractors, tr, attr) for tr in trs]
            s_vals = [_get(sol_extractors, tr, attr) for tr in trs]

            # ── Curvas ────────────────────────────────────────────────────
            ax.plot(trs, b_vals, 'o-', color=COLORS['baseline'], linewidth=2.4,
                    markersize=8, label='Baseline', zorder=4, markerfacecolor='white',
                    markeredgewidth=2)
            ax.plot(trs, s_vals, 's-', color='#111111', linewidth=2.8,
                    markersize=8, label='Solución', zorder=5)

            # ── Área entre curvas ─────────────────────────────────────────
            if lower_better:
                improve = [b >= s for b, s in zip(b_vals, s_vals)]
            else:
                improve = [s >= b for b, s in zip(b_vals, s_vals)]

            ax.fill_between(trs, b_vals, s_vals, where=improve,
                            alpha=0.15, color='#4CAF50', zorder=2)
            ax.fill_between(trs, b_vals, s_vals, where=[not i for i in improve],
                            alpha=0.15, color='#E05C5C', zorder=2)

            # ── TR de diseño ───────────────────────────────────────────────
            ax.axvline(x=self.tr_actual, color='#9C27B0', linewidth=1.1,
                       linestyle=':', alpha=0.75, zorder=3)

            # ── Labels de % — alternando arriba/abajo, con caja coloreada ─
            y_min, y_max = min(min(b_vals), min(s_vals)), max(max(b_vals), max(s_vals))
            y_span = y_max - y_min if y_max > y_min else 1.0

            for i, (tr, bv, sv) in enumerate(zip(trs, b_vals, s_vals)):
                ref = bv if bv != 0 else (sv if sv != 0 else None)
                if ref is None:
                    continue
                if lower_better:
                    pct = (bv - sv) / abs(ref) * 100
                else:
                    pct = (sv - bv) / abs(ref) * 100

                clr = '#1B5E20' if pct >= 0 else '#B71C1C'
                bgclr = '#E8F5E9' if pct >= 0 else '#FFEBEE'

                # Alternar: puntos pares arriba, impares abajo
                # para los puntos de la solución
                base_y = sv
                if i % 2 == 0:
                    offset_pts = 22
                    va_box = 'bottom'
                else:
                    offset_pts = -22
                    va_box = 'top'

                ax.annotate(
                    f'{pct:+.0f}%',
                    xy=(tr, base_y),
                    xytext=(0, offset_pts),
                    textcoords='offset points',
                    ha='center', va=va_box,
                    fontsize=9, fontweight='bold', color=clr,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=bgclr,
                              edgecolor=clr, linewidth=0.8, alpha=0.92),
                    arrowprops=dict(arrowstyle='-', color=clr,
                                    lw=0.8, alpha=0.6),
                    zorder=8
                )

            # ── Ejes y estilo ──────────────────────────────────────────────
            ax.set_xscale('log')
            ax.set_xticks(trs)
            ax.set_xticklabels([f'TR{t}' for t in trs], fontsize=9)
            ax.set_ylim(bottom=0, top=y_max * 1.35)  # espacio arriba para labels
            ax.set_xlabel('Periodo de Retorno (años)', fontsize=9, color=COLORS['text'])
            ax.set_ylabel(f'{title} ({unit})', fontsize=9, color=COLORS['text'])
            ax.tick_params(labelsize=8.5, colors=COLORS['text'])
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')

            # Título con dirección como subtítulo coloreado
            dir_color = '#1B5E20' if '↓' in direction else '#1565C0'
            ax.set_title(f'{title}\n', fontsize=11, fontweight='bold',
                         color=COLORS['text'], pad=4)
            ax.text(0.5, 1.01, direction, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=9,
                    color=dir_color, fontstyle='italic')

            if idx == 0:
                leg = ax.legend(fontsize=9, framealpha=0.92, edgecolor='#CCCCCC',
                                loc='upper left')
                leg.get_frame().set_linewidth(0.8)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"extra1_risk_curve_{name}.png",
                    dpi=160, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Guardado: extra1_risk_curve_{name}.png")

    # -------------------------------------------------------------------------
    # EXTRA 2: Mapa de nodos — scatter coloreado por reducción de flooding
    # -------------------------------------------------------------------------
    def _plot_node_map(self, sol_extractors: dict, out_dir: Path, name: str):
        """
        Scatter X/Y de nodos sobre red de tuberías (GPKG).
        - Tamaño burbuja ∝ volumen baseline
        - Color ∝ reducción clampeada a ±100% (evita distorsión por nodos con vol. mínimo)
        - Nodos sin flooding: pequeños y grises
        - Labels para los N nodos con mayor volumen baseline
        """
        base_m = self.baseline_extractors[self.tr_actual].metrics
        sol_m = sol_extractors[self.tr_actual].metrics

        if not base_m.node_data or not sol_m.node_data:
            print("    [node_map] Sin node_data disponible, omitiendo.")
            return

        # ── Construir GeoDataFrame de nodos y reprojectar a PROJECT_CRS ────
        project_crs = getattr(config, 'PROJECT_CRS', 'EPSG:32717')

        records = []
        for nid, bdata in base_m.node_data.items():
            sdata = sol_m.node_data.get(nid, {})
            bv = bdata.get('flooding_volume', 0.0)
            sv = sdata.get('flooding_volume', 0.0) if sdata else bv
            if bv > 0:
                raw_red = (bv - sv) / bv * 100
                red = float(np.clip(raw_red, -100, 100))
            else:
                red = 0.0
            records.append({
                'node_id': str(nid),
                'x': bdata.get('x', 0),
                'y': bdata.get('y', 0),
                'base_vol': bv,
                'sol_vol': sv,
                'reduction': red,
                'flooded': bv > 0,
            })

        df = pd.DataFrame(records)

        from pyproj import CRS as ProjCRS
        target_crs = ProjCRS.from_user_input(project_crs)

        # Convertir nodos a GeoDataFrame y asegurar CRS correcto
        nodes_gdf = gpd.GeoDataFrame(
            df,
            geometry=[ShapelyPoint(row.x, row.y) for row in df.itertuples()],
            crs=project_crs
        )
        if nodes_gdf.crs and not nodes_gdf.crs.equals(target_crs):
            nodes_gdf = nodes_gdf.to_crs(target_crs)

        df['x'] = nodes_gdf.geometry.x
        df['y'] = nodes_gdf.geometry.y

        flooded = df[df['flooded']].copy()
        dry = df[~df['flooded']].copy()

        # ── Figura ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 11))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F7FA')

        # ── 1) Red de tuberías como fondo (reprojectada a PROJECT_CRS) ──────
        network_file = getattr(config, 'NETWORK_FILE', None)
        if network_file and Path(network_file).exists():
            try:
                pipes_gdf = gpd.read_file(network_file)
                pipes_gdf = pipes_gdf[pipes_gdf.geometry.notna()]
                pipes_gdf = pipes_gdf[pipes_gdf.geometry.geom_type.isin(
                    ['LineString', 'MultiLineString'])]

                # Reprojectar al CRS del proyecto si es necesario
                if pipes_gdf.crs is not None and not pipes_gdf.crs.equals(target_crs):
                    pipes_gdf = pipes_gdf.to_crs(target_crs)
                elif pipes_gdf.crs is None:
                    pipes_gdf = pipes_gdf.set_crs(target_crs)

                # print(f"    Red de tuberías cargada: {len(pipes_gdf)} segmentos (CRS: {pipes_gdf.crs})")

                # Colorear por capacidad si está disponible, si no gris uniforme
                if 'Capacity' in pipes_gdf.columns or 'utilization' in pipes_gdf.columns:
                    cap_col = 'utilization' if 'utilization' in pipes_gdf.columns else 'Capacity'
                    cap_vals = pipes_gdf[cap_col].fillna(0).clip(0, 1)
                    pipe_cmap = LinearSegmentedColormap.from_list(
                        'pipe_load', ['#C8D6E5', '#F0A500', '#C0392B'])
                    pipe_colors = pipe_cmap(cap_vals.values)
                    for geom, color in zip(pipes_gdf.geometry, pipe_colors):
                        if geom.geom_type == 'LineString':
                            xs, ys = geom.xy
                            ax.plot(xs, ys, color=color, linewidth=1.1,
                                    alpha=0.7, zorder=1)
                        elif geom.geom_type == 'MultiLineString':
                            for part in geom.geoms:
                                xs, ys = part.xy
                                ax.plot(xs, ys, color=color, linewidth=1.1,
                                        alpha=0.7, zorder=1)
                    sm_pipe = plt.cm.ScalarMappable(
                        cmap=pipe_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    sm_pipe.set_array([])
                    cb_pipe = fig.colorbar(sm_pipe, ax=ax, fraction=0.018,
                                           pad=0.12, shrink=0.45, aspect=20)
                    cb_pipe.set_label('Carga tubería (h/D)', fontsize=7.5,
                                      color=COLORS['text'])
                    cb_pipe.ax.tick_params(labelsize=7)
                else:
                    for geom in pipes_gdf.geometry:
                        if geom.geom_type == 'LineString':
                            xs, ys = geom.xy
                            ax.plot(xs, ys, color='#A0B4C8', linewidth=0.9,
                                    alpha=0.6, zorder=1)
                        elif geom.geom_type == 'MultiLineString':
                            for part in geom.geoms:
                                xs, ys = part.xy
                                ax.plot(xs, ys, color='#A0B4C8', linewidth=0.9,
                                        alpha=0.6, zorder=1)
            except Exception as e:
                print(f"    [node_map] No se pudo cargar NETWORK_FILE: {e}")
        else:
            print("    [node_map] NETWORK_FILE no definido o no existe, omitiendo red.")

        # ── 2) Nodos sin flooding (pequeños, grises con borde) ──────────────
        if not dry.empty:
            ax.scatter(dry['x'], dry['y'], s=18, color='#D5DCE8',
                       edgecolors='#99AABB', linewidths=0.4,
                       alpha=0.7, zorder=2, label='Sin flooding')

        # ── 3) Nodos con flooding ───────────────────────────────────────────
        if not flooded.empty:
            # Tamaño: escala más agresiva para que las burbujas sean visibles
            sizes = (np.sqrt(flooded['base_vol'].clip(lower=1)) * 1.2).clip(lower=35, upper=600)

            sc = ax.scatter(
                flooded['x'], flooded['y'],
                s=sizes,
                c=flooded['reduction'],
                cmap='RdYlGn',
                vmin=-100, vmax=100,  # clamp fijo ±100%
                alpha=0.88,
                edgecolors='white',
                linewidths=0.8,
                zorder=4
            )

            # Colorbar principal (reducción)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02,
                                shrink=0.65, aspect=25)
            cbar.set_label('Reducción Flooding (%)\n+100 = eliminado  |  −100 = duplicado',
                           fontsize=8.5, color=COLORS['text'])
            cbar.ax.tick_params(labelsize=8)
            # Marcas clave
            cbar.set_ticks([-100, -50, 0, 50, 100])
            cbar.set_ticklabels(['-100%', '-50%', '0%', '+50%', '+100%'])

        # ── Estilo general ───────────────────────────────────────────────────
        n_mejor = int((flooded['reduction'] > 5).sum()) if not flooded.empty else 0
        n_peor = int((flooded['reduction'] < -5).sum()) if not flooded.empty else 0
        n_neutro = len(flooded) - n_mejor - n_peor if not flooded.empty else 0
        n_total = len(flooded)

        ax.set_title(
            f'Mapa de Nodos — Reducción Flooding  [TR{self.tr_actual}]  ·  {name}\n'
            f'Nodos con flooding: {n_total}  |  '
            f'Mejorados: {n_mejor}  ·  Empeorados: {n_peor}  ·  Sin cambio: {n_neutro}'
            f'  (detalle → nodos_TR{self.tr_actual:03d}_{name}.csv)',
            fontsize=10.5, fontweight='bold', color=COLORS['text'], pad=10
        )
        ax.set_xlabel('X (m)', fontsize=9, color=COLORS['text'])
        ax.set_ylabel('Y (m)', fontsize=9, color=COLORS['text'])
        ax.tick_params(labelsize=8, colors=COLORS['text'])
        ax.set_aspect('equal', adjustable='datalim')
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')

        # Leyenda de tamaños de burbuja
        legend_sizes = [100, 1000, 5000, 20000]
        legend_labels = ['100 m³', '1,000 m³', '5,000 m³', '20,000 m³']
        legend_handles = []
        from matplotlib.lines import Line2D
        for sv_ref, lbl in zip(legend_sizes, legend_labels):
            s_ref = float(np.clip(np.sqrt(sv_ref) * 1.2, 35, 600))
            legend_handles.append(
                plt.scatter([], [], s=s_ref, color='#888888', alpha=0.7,
                            edgecolors='white', label=f'Base: {lbl}')
            )

        leg = ax.legend(handles=legend_handles,
                        title='Volumen baseline', title_fontsize=7.5,
                        fontsize=7.5, framealpha=0.93, loc='lower right',
                        edgecolor='#BBBBBB', markerscale=1.0)
        leg.get_frame().set_linewidth(0.8)

        # Nota al pie sobre el clamp
        fig.text(0.5, 0.005,
                 'Nota: reducción clampeada a ±100%. Nodos con volumen baseline < 1 m³ '
                 'pueden mostrar variaciones % grandes — revisar en CSV.',
                 ha='center', fontsize=7, color='#888888', style='italic')

        fig.savefig(out_dir / f"extra2_node_map_{name}.png",
                    dpi=160, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Guardado: extra2_node_map_{name}.png")

    # -------------------------------------------------------------------------
    # EXTRA 3: Radar / Spider multi-métrica
    # -------------------------------------------------------------------------
    def _plot_radar(self, sol_extractors: dict, out_dir: Path, name: str):
        """
        Spider chart comparando baseline vs solución en 5 métricas normalizadas,
        una figura por TR. Permite ver el perfil de desempeño completo.
        """
        metric_labels = [
            'Vol. Flooding\n(↓ mejor)',
            'Pico Flooding\n(↓ mejor)',
            'Pico Outfall\n(↑ mejor)',
            'Links\nSaturados\n(↓ mejor)',
            'Health\nScore\n(↑ mejor)',
        ]
        N = len(metric_labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # cerrar polígono

        fig, axes = plt.subplots(
            2, math.ceil(len(self.tr_list) / 2),
            figsize=(15, 8),
            subplot_kw=dict(polar=True)
        )
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Radar Multi-Métrica  ·  {name}',
                     fontsize=12, fontweight='bold', color=COLORS['text'])

        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Recolectar todos los valores para normalización global
        all_vals = {i: [] for i in range(N)}
        for tr in self.tr_list:
            bm = self.baseline_extractors[tr].metrics
            sm = sol_extractors[tr].metrics
            raw_b = [bm.total_flooding_volume, bm.total_max_flooding_flow,
                     bm.total_max_outfall_flow, float(bm.surcharged_links_count),
                     bm.network_health_score]
            raw_s = [sm.total_flooding_volume, sm.total_max_flooding_flow,
                     sm.total_max_outfall_flow, float(sm.surcharged_links_count),
                     sm.network_health_score]
            for i, (b, s) in enumerate(zip(raw_b, raw_s)):
                all_vals[i].extend([b, s])

        # Rangos para normalización (0-1) — guard contra todos-cero
        ranges = []
        for v in all_vals.values():
            lo, hi = min(v), max(v)
            # Si todos los valores son iguales (incluido todo-cero), usar rango [0,1]
            if hi == lo:
                lo, hi = 0.0, 1.0
            ranges.append((lo, hi))

        def normalize_radar(vals):
            out = []
            for i, (lo, hi) in enumerate(ranges):
                span = hi - lo  # ya garantizado > 0 por el guard de arriba
                v = (vals[i] - lo) / span
                out.append(float(np.clip(v, 0.0, 1.0)))
            return out

        # "Invertir" métricas donde menor = mejor: idx 0,1,3
        invert_idx = {0, 1, 3}

        for ax_idx, tr in enumerate(self.tr_list):
            ax = axes_flat[ax_idx]
            bm = self.baseline_extractors[tr].metrics
            sm = sol_extractors[tr].metrics

            raw_b = [bm.total_flooding_volume, bm.total_max_flooding_flow,
                     bm.total_max_outfall_flow, float(bm.surcharged_links_count),
                     bm.network_health_score]
            raw_s = [sm.total_flooding_volume, sm.total_max_flooding_flow,
                     sm.total_max_outfall_flow, float(sm.surcharged_links_count),
                     sm.network_health_score]

            norm_b = normalize_radar(raw_b)
            norm_s = normalize_radar(raw_s)

            # Invertir ejes "↓ mejor" para que hacia afuera = mejor
            norm_b = [1 - v if i in invert_idx else v for i, v in enumerate(norm_b)]
            norm_s = [1 - v if i in invert_idx else v for i, v in enumerate(norm_s)]

            vals_b = norm_b + norm_b[:1]
            vals_s = norm_s + norm_s[:1]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=9, color=COLORS['text'])
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=7, color='#999999')
            ax.set_ylim(0, 1)
            ax.grid(color=COLORS['grid'], linewidth=0.7)
            ax.spines['polar'].set_color('#CCCCCC')

            ax.plot(angles, vals_b, color=COLORS['baseline'], linewidth=1.5, zorder=3)
            ax.fill(angles, vals_b, color=COLORS['baseline'], alpha=0.18, zorder=2)

            ax.plot(angles, vals_s, color='#111111', linewidth=2.0, zorder=4)
            ax.fill(angles, vals_s, color=COLORS['solution'], alpha=0.22, zorder=3)

            is_design = tr == self.tr_actual
            ax.set_title(f'TR{tr}{"  ★" if is_design else ""}',
                         fontsize=9, fontweight='bold' if is_design else 'normal',
                         color='#7B1FA2' if is_design else COLORS['text'],
                         pad=12)

        # Ocultar ejes sobrantes
        for ax_idx in range(len(self.tr_list), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        # Leyenda global
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=COLORS['baseline'], linewidth=1.5, label='Baseline'),
            Line2D([0], [0], color='#111111', linewidth=2.0, label='Solución'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2,
                   fontsize=12, framealpha=0.95, edgecolor='#CCCCCC',
                   markerscale=1.5)

        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        fig.savefig(out_dir / f"extra3_radar_{name}.png",
                    dpi=160, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"    Guardado: extra3_radar_{name}.png")

    # -------------------------------------------------------------------------
    # EXPORT CSV
    # -------------------------------------------------------------------------
    def _export_summary_csv(self, sol_extractors: dict, out_dir: Path, name: str):
        """Exporta CSV con resumen de métricas escalares por TR."""
        import csv

        rows_paso3 = []
        rows_paso4 = []
        sol_tr25_vol = sol_extractors[self.tr_actual].metrics.total_flooding_volume

        for tr in self.tr_list:
            bv = self.baseline_extractors[tr].metrics.total_flooding_volume
            sv = sol_extractors[tr].metrics.total_flooding_volume
            bp = self.baseline_extractors[tr].metrics.total_max_flooding_flow
            sp_ = sol_extractors[tr].metrics.total_max_flooding_flow
            bo = self.baseline_extractors[tr].metrics.total_max_outfall_flow
            so = sol_extractors[tr].metrics.total_max_outfall_flow

            red_vol = (bv - sv) / bv * 100 if bv > 0 else 0.0
            red_peak = (bp - sp_) / bp * 100 if bp > 0 else 0.0

            rows_paso3.append([
                tr, f'{bv:,.1f}', f'{sv:,.1f}', f'{red_vol:.2f}',
                f'{bp:.4f}', f'{sp_:.4f}', f'{red_peak:.2f}',
                f'{bo:.4f}', f'{so:.4f}'
            ])

            # Paso 4: sol_tr25 vs base_trX
            if tr != self.tr_actual:
                red_cross = (bv - sol_tr25_vol) / bv * 100 if bv > 0 else 0.0
                rows_paso4.append([tr, f'{bv:,.1f}', f'{sol_tr25_vol:,.1f}', f'{red_cross:.2f}'])

        with open(out_dir / f"resumen_{name}.csv", 'w', newline='', encoding='utf-8') as f:
            cw = csv.writer(f)
            cw.writerow([f'=== PASO 3: Validación Funcional — {name} ==='])
            cw.writerow(['TR', 'Base_Vol_m3', 'Sol_Vol_m3', 'Red_Vol_%',
                         'Base_Peak_m3s', 'Sol_Peak_m3s', 'Red_Peak_%',
                         'Base_OutfallPeak', 'Sol_OutfallPeak'])
            cw.writerows(rows_paso3)
            cw.writerow([])
            cw.writerow([f'=== PASO 4: Validación Cruzada — Sol TR{self.tr_actual} vs Bases ==='])
            cw.writerow(['TR_Base', 'Base_Vol_m3', f'Sol_TR{self.tr_actual}_Vol_m3', 'Red_Cruzada_%'])
            cw.writerows(rows_paso4)

        print(f"    CSV guardado: resumen_{name}.csv")

    # -------------------------------------------------------------------------
    # EXPORT CSV — Detalle por nodo
    # -------------------------------------------------------------------------
    def _export_nodes_csv(self, sol_extractors: dict, out_dir: Path, name: str):
        """
        Exporta un CSV por TR con detalle de cada nodo que tuvo flooding en baseline:
          node_id, x, y,
          base_vol_m3, sol_vol_m3, delta_vol_m3, red_vol_pct,
          base_peak_m3s, sol_peak_m3s, delta_peak_m3s, red_peak_pct,
          estado  (Mejorado / Empeorado / Sin cambio / Eliminado)
        Ordenado por base_vol_m3 desc.
        """
        import csv

        def _peak_from_series(series) -> float:
            """Caudal pico [m³/s] desde flooding_series (pd.Series o dict)."""
            if series is None:
                return 0.0
            if isinstance(series, pd.Series):
                return float(series.max()) if len(series) > 0 else 0.0
            if isinstance(series, dict):
                vals = series.get('total_rate', series.get('values', []))
                return float(np.max(vals)) if len(vals) > 0 else 0.0
            return 0.0

        for tr in self.tr_list:
            base_m = self.baseline_extractors[tr].metrics
            sol_m = sol_extractors[tr].metrics

            rows = []
            for nid, bdata in base_m.node_data.items():
                bv = bdata.get('flooding_volume', 0.0)
                if bv <= 0:
                    continue  # solo nodos que inundan en baseline

                sdata = sol_m.node_data.get(nid, {})
                sv = sdata.get('flooding_volume', 0.0) if sdata else bv

                # Pico de caudal por nodo
                bp = _peak_from_series(bdata.get('flooding_series'))
                sp_ = _peak_from_series(sdata.get('flooding_series')) if sdata else bp

                delta_vol = sv - bv
                red_vol = (bv - sv) / bv * 100
                delta_peak = sp_ - bp
                red_peak = (bp - sp_) / bp * 100 if bp > 0 else 0.0

                if sv <= 0:
                    estado = 'Eliminado'
                elif red_vol >= 5:
                    estado = 'Mejorado'
                elif red_vol <= -5:
                    estado = 'Empeorado'
                else:
                    estado = 'Sin cambio'

                rows.append({
                    'node_id': str(nid),
                    'x': f"{bdata.get('x', 0):.1f}",
                    'y': f"{bdata.get('y', 0):.1f}",
                    'base_vol_m3': f"{bv:.2f}",
                    'sol_vol_m3': f"{sv:.2f}",
                    'delta_vol_m3': f"{delta_vol:+.2f}",
                    'red_vol_pct': f"{red_vol:+.1f}",
                    'base_peak_m3s': f"{bp:.4f}",
                    'sol_peak_m3s': f"{sp_:.4f}",
                    'delta_peak_m3s': f"{delta_peak:+.4f}",
                    'red_peak_pct': f"{red_peak:+.1f}",
                    'estado': estado,
                })

            # Ordenar por volumen baseline desc
            rows.sort(key=lambda r: float(r['base_vol_m3']), reverse=True)

            out_path = out_dir / f"nodos_TR{tr:03d}_{name}.csv"
            with open(out_path, 'w', newline='', encoding='utf-8') as f:
                cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
                cw.writeheader()
                cw.writerows(rows)

            n_mejorados = sum(1 for r in rows if r['estado'] in ('Mejorado', 'Eliminado'))
            n_empeorados = sum(1 for r in rows if r['estado'] == 'Empeorado')
            print(f"    CSV nodos TR{tr}: {len(rows)} nodos  |  "
                  f"Mejorados/Eliminados: {n_mejorados}  |  Empeorados: {n_empeorados}  "
                  f"→ nodos_TR{tr:03d}_{name}.csv")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("CrossTRValidator  —  Validación Multi-TR")
    print("=" * 65)

    v = CrossTRValidator()

    print("\n=== Baselines ===")
    for tr, m in v.baseline_metrics.items():
        print(f"  TR{tr:>3}:  Flooding {m['flood_volume']:>12,.0f} m³  |  "
              f"Peak {m['flood_peak']:.3f} m³/s  |  "
              f"Outfall {m['outfall_peak']:.3f} m³/s")

    # Buscar solución optimizada
    sp = Path(config.CODIGOS_DIR) / "optimization_results" / "Seq_Iter_04" / "model_Seq_Iter_04.inp"
    if sp.exists():
        print(f"\nComparando solución: {sp}")
        resumen = v.compare_solution(sp, "Seq_Iter_04")
        print("\n=== Resumen Paso 3 ===")
        for tr, r in resumen.items():
            red = (r['base_flood_vol'] - r['sol_flood_vol']) / r['base_flood_vol'] * 100 if r['base_flood_vol'] > 0 else 0
            print(f"  TR{tr:>3}:  Base {r['base_flood_vol']:>10,.0f} m³  →  Sol {r['sol_flood_vol']:>10,.0f} m³  "
                  f"({red:+.1f}%)")
    else:
        print(f"\n[Aviso] Solución no encontrada: {sp}")

    print("\n✓ Listo.")