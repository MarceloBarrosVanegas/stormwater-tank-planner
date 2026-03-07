"""
StormwaterCharts
================
Clase para generar visualizaciones de optimizacion de tanques de retencion.

GRAFICAS:
  mariposa()  → Impacto global Baseline vs Final (barras horizontales)
  pareto()    → Curvas Pareto para cualquier lista de variables del CSV.
                Max 4 paneles por figura; si pasas mas, genera varias figuras.

USO RAPIDO:
    from stormwater_charts import StormwaterCharts

    ch = StormwaterCharts('sequence_tracking.csv', outdir='outputs/')

    # 4 variables → 1 figura
    ch.pareto(['flooding_reduction', 'outfall_peak_flow',
               'flooded_nodes_count', 'system_mean_utilization'])

    # 5 variables → figura 1 (4 paneles) + figura 2 (1 panel)
    ch.pareto(['flooding_reduction', 'outfall_peak_flow',
               'flooded_nodes_count', 'system_mean_utilization',
               'surcharged_links_count'])

    ch.mariposa()
    ch.save_all()   # mariposa + pareto default de una vez

VARIABLES DISPONIBLES EN EL CATALOGO (nombres exactos del CSV):
    flooding_reduction       Volumen reducido acumulado
    flooding_volume          Volumen residual
    flooded_nodes_count      Nodos inundados
    outfall_peak_flow        Caudal pico en outfall
    system_mean_utilization  Utilizacion media h/D
    surcharged_links_count   Links sobrecargados
    total_tank_volume        Volumen total tanques
    cost_per_m3_reduced      Costo por m3 reducido

Si una variable NO esta en el catalogo se dibuja igual con
configuracion generica. Tambien puedes pasar tu propio catalogo
via custom_meta={} (ver docstring de pareto()).
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ── Paleta ────────────────────────────────────────────────────────────────────
GREEN  = '#15803D';  LGREEN = '#DCFCE7'
ORANGE = '#EA580C';  LORANG = '#FFEDD5'
RED    = '#B91C1C';  LRED   = '#FEE2E2'
BLUE   = '#1D4ED8';  LBLUE  = '#DBEAFE'
SLATE  = '#1E293B'
GRAY   = '#9CA3AF'
LGRAY  = '#E5E7EB'

_STYLE = {
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.13,
    'grid.linewidth':    0.6,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#FAFAFA',
    'axes.titlesize':    11.5,
    'axes.titleweight':  'bold',
}

# ── Catalogo de variables conocidas ──────────────────────────────────────────
# invert=True  → curva BAJA (menor es mejor)
# invert=False → curva SUBE (mayor es mejor)
VAR_CATALOG = {
    'flooding_reduction': dict(
        title   = 'Volumen de Inundacion Reducido',
        ylabel  = 'Reduccion acumulada (m3)',
        invert  = False,
        fmt_y   = lambda v: f'{v/1000:.1f}K m3',
        fmt_eff = lambda v: f'{v:,.0f} m3/M$',
        c_good  = GREEN, c_bad = ORANGE,
    ),
    'flooding_volume': dict(
        title   = 'Inundacion Residual',
        ylabel  = 'Volumen residual (m3)',
        invert  = True,
        fmt_y   = lambda v: f'{v/1000:.1f}K m3',
        fmt_eff = lambda v: f'{v:,.0f} m3/M$',
        c_good  = GREEN, c_bad = RED,
    ),
    'flooded_nodes_count': dict(
        title   = 'Nodos Inundados en la Red',
        ylabel  = 'Nodos inundados',
        invert  = True,
        fmt_y   = lambda v: f'{v:.0f} nodos',
        fmt_eff = lambda v: f'{v:.2f} nodos/M$',
        c_good  = GREEN, c_bad = RED,
    ),
    'outfall_peak_flow': dict(
        title   = 'Caudal Pico en Outfall',
        ylabel  = 'Caudal pico (m3/s)',
        invert  = True,
        fmt_y   = lambda v: f'{v:.1f} m3/s',
        fmt_eff = lambda v: f'{v:.3f} m3/s/M$',
        c_good  = GREEN, c_bad = RED,
    ),
    'system_mean_utilization': dict(
        title   = 'Utilizacion Media del Sistema',
        ylabel  = 'Utilizacion media h/D (%)',
        invert  = False,
        fmt_y   = lambda v: f'{v:.0f}%',
        fmt_eff = lambda v: f'{v:.2f} %/M$',
        c_good  = GREEN, c_bad = ORANGE,
    ),
    'surcharged_links_count': dict(
        title   = 'Links Sobrecargados (h/D > 1)',
        ylabel  = 'Links sobrecargados',
        invert  = False,
        fmt_y   = lambda v: f'{v:.0f}',
        fmt_eff = lambda v: f'{v:.2f} links/M$',
        c_good  = GREEN, c_bad = ORANGE,
    ),
    'total_tank_volume': dict(
        title   = 'Volumen Total de Tanques Instalados',
        ylabel  = 'Volumen acumulado (m3)',
        invert  = False,
        fmt_y   = lambda v: f'{v/1000:.1f}K m3',
        fmt_eff = lambda v: f'{v:,.0f} m3/M$',
        c_good  = GREEN, c_bad = ORANGE,
    ),
    'cost_per_m3_reduced': dict(
        title   = 'Costo Marginal por m3 Reducido',
        ylabel  = '$/m3 reducido',
        invert  = True,
        fmt_y   = lambda v: f'${v:,.0f}',
        fmt_eff = lambda v: f'${v:,.0f}/M$',
        c_good  = GREEN, c_bad = RED,
    ),
}

# Layout de subplots segun numero de paneles en la figura
_LAYOUTS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}


# ═════════════════════════════════════════════════════════════════════════════
class StormwaterCharts:
    """
    Genera graficas de optimizacion de tanques de retencion.

    Parameters
    ----------
    csv_path : str   Ruta al sequence_tracking.csv
    outdir   : str   Carpeta de salida (se crea si no existe)
    dpi      : int   Resolucion de imagenes (default 180)
    """

    MAX_PANELS = 4   # paneles maximos por figura

    def __init__(self, csv_path: str, outdir: str = '.', dpi: int = 180):
        self.outdir = outdir
        self.dpi    = dpi
        os.makedirs(outdir, exist_ok=True)
        self.df        = pd.read_csv(csv_path)
        self.df['inv_M']    = self.df['cost_investment_total'] / 1e6
        self.df['new_tank'] = self.df['current_tank_volume'] > 0
        self.baseline  = self.df.iloc[0]
        self.final     = self.df.iloc[-1]

    # ── Utilidades ────────────────────────────────────────────────────────────
    @staticmethod
    def _marginal_slope(inv, y):
        """Eficiencia marginal |Δy/ΔM$| por paso. Paso 0 = NaN."""
        s = [np.nan]
        for i in range(1, len(inv)):
            di = inv[i] - inv[i-1]
            dy = y[i]   - y[i-1]
            s.append(abs(dy / di) if di > 0 else np.nan)
        return np.array(s)

    @staticmethod
    def _place_labels(ax, xs, ys, labels, dy_up=9, dy_down=-14):
        """Etiquetas de texto sin solapamiento vertical."""
        yr     = ax.get_ylim()[1] - ax.get_ylim()[0]
        gap    = yr * 0.06
        placed = []
        for xi, yi, lbl in zip(xs, ys, labels):
            dy = dy_up
            for px, py in placed:
                if abs(xi - px) < 3 and abs(yi - py) < gap:
                    dy = dy_down
                    break
            ax.annotate(lbl, xy=(xi, yi), xytext=(4, dy),
                        textcoords='offset points',
                        fontsize=7.5, color=SLATE, zorder=7,
                        annotation_clip=True)
            placed.append((xi, yi + dy * yr / 300))

    def _savefig(self, fig, name):
        path = os.path.join(self.outdir, name)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight')
        print(f'  Guardado: {path}')

    # ── Panel individual Pareto ───────────────────────────────────────────────
    def _draw_panel(self, ax, col, meta):
        df     = self.df
        df_p   = df[df['step'] > 0].copy()
        x      = df_p['inv_M'].values
        y      = df_p[col].values
        invert = meta.get('invert', False)
        c_good = meta.get('c_good', GREEN)
        c_bad  = meta.get('c_bad',  ORANGE)

        # Eficiencia marginal
        eff     = self._marginal_slope(df['inv_M'].values, df[col].values)[1:]
        med_eff = np.nanmedian(eff)
        colors  = [c_good if (not np.isnan(e) and e >= med_eff) else c_bad
                   for e in eff]

        # Fit logaritmico
        try:
            popt, _ = curve_fit(lambda t, a, b: a * np.log(t) + b,
                                x, y, p0=[-1e4, y[0]])
            xf = np.linspace(x.min(), x.max(), 400)
            ax.plot(xf, popt[0]*np.log(xf)+popt[1],
                    '--', color=GRAY, lw=1.3, alpha=0.4, zorder=3)
        except Exception:
            pass

        # Limites Y con margen para etiquetas
        yr   = y.max() - y.min()
        y_lo = y.min() - yr * (0.18 if invert else 0.06)
        y_hi = y.max() + yr * (0.10 if invert else 0.22)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(x.min() - 0.5, x.max() + 4)

        # Mediana horizontal del valor
        med_y = np.median(y)
        ax.axhline(med_y, color=BLUE, lw=1.1, ls=':', alpha=0.5, zorder=2)
        ax.text(0.99, med_y, f' Mediana\n {meta["fmt_y"](med_y)}',
                transform=ax.get_yaxis_transform(),
                fontsize=7.5, color=BLUE, va='center', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          alpha=0.75, ec='none'))

        # Zona de quiebre (primer paso bajo la mediana de eficiencia)
        first_below = next(
            (x[i] for i, e in enumerate(eff)
             if not np.isnan(e) and e < med_eff), None)
        if first_below is not None:
            ax.axvspan(x.min()-0.5, first_below, alpha=0.05, color=GREEN, zorder=0)
            ax.axvspan(first_below, x.max()+4,   alpha=0.03, color=ORANGE, zorder=0)
            ax.axvline(first_below, color=GRAY, lw=0.9, ls=':', alpha=0.45, zorder=2)
            q_y  = (y_lo + yr*0.03) if invert else (y_hi*0.993)
            q_va = 'bottom'         if invert else 'top'
            ax.text(first_below + 0.4, q_y,
                    f'Quiebre ${first_below:.0f}M',
                    fontsize=7.5, color=GRAY, va=q_va, ha='left', style='italic')

        # Trayectoria y scatter
        ax.plot(x, y, '-', color=LGRAY, lw=0.9, zorder=3)
        ax.scatter(x, y, c=colors, s=72, zorder=6,
                   edgecolors='white', linewidths=1.0)

        # Etiquetas sin solapamiento (solo pasos con tanque nuevo)
        new_steps = set(df_p.loc[df_p['new_tank'], 'step'].values)
        xs_l = [xi for xi, s in zip(x, df_p['step']) if s in new_steps]
        ys_l = [yi for yi, s in zip(y, df_p['step']) if s in new_steps]
        lb_l = [f'p{s}'  for s in df_p['step']         if s in new_steps]
        self._place_labels(ax, xs_l, ys_l, lb_l,
                           dy_up=9, dy_down=-14 if not invert else -17)

        # Caja eficiencia marginal
        ax.text(0.98, 0.97,
                f'Eff. marginal mediana:\n{meta["fmt_eff"](med_eff)}',
                transform=ax.transAxes, va='top', ha='right',
                fontsize=8, color=SLATE,
                bbox=dict(boxstyle='round,pad=0.35',
                          fc='white', ec=LGRAY, alpha=0.92))

        # Leyenda en el espacio libre segun direccion de la curva
        leg_loc = 'upper left' if invert else 'lower left'
        leg_ba  = (0.02, 0.98) if invert else (0.02, 0.02)
        ax.legend(handles=[
            mpatches.Patch(color=c_good, alpha=0.85, label='Eff. >= mediana'),
            mpatches.Patch(color=c_bad,  alpha=0.85, label='Eff. <  mediana'),
            plt.Line2D([0],[0], ls=':', color=BLUE, lw=1.4, label='Mediana del valor'),
        ], fontsize=7.5, loc=leg_loc, bbox_to_anchor=leg_ba,
           framealpha=0.92, edgecolor=LGRAY, borderpad=0.5)

        ax.set_xlabel('Inversion Acumulada (M$)', fontsize=9, labelpad=4)
        ax.set_ylabel(meta.get('ylabel', col), fontsize=9)
        ax.set_title(meta.get('title', col), pad=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:.0f}M'))

    # ═════════════════════════════════════════════════════════════════════════
    # MARIPOSA
    # ═════════════════════════════════════════════════════════════════════════
    def mariposa(self, filename='mariposa_impacto.png', show=False):
        """
        Barras horizontales con impacto global. Escala log simetrica (symlog).
        """
        plt.rcParams.update(_STYLE)
        b, f = self.baseline, self.final

        metrics = [
            ('Inversion total requerida',
             -f['inv_M'],
             f"${f['inv_M']:.0f}M", 'cost'),
            ('Links sobrecargados  (+presion)',
             -((f['surcharged_links_count']-b['surcharged_links_count'])
               / b['surcharged_links_count']*100),
             f"+{(f['surcharged_links_count']-b['surcharged_links_count'])/b['surcharged_links_count']*100:.1f}%",
             'cost'),
            ('Volumen tanques instalados',
             f['total_tank_volume']/1000,
             f"{f['total_tank_volume']/1000:.0f}K m3", 'info'),
            ('Utilizacion media red',
             (f['system_mean_utilization']-b['system_mean_utilization'])
             / b['system_mean_utilization']*100,
             f"+{(f['system_mean_utilization']-b['system_mean_utilization'])/b['system_mean_utilization']*100:.1f}%",
             'benefit'),
            ('Caudal pico outfall  (reduccion)',
             (b['outfall_peak_flow']-f['outfall_peak_flow'])
             / b['outfall_peak_flow']*100,
             f"-{(b['outfall_peak_flow']-f['outfall_peak_flow'])/b['outfall_peak_flow']*100:.1f}%",
             'benefit'),
            ('Nodos inundados  (reduccion)',
             (b['flooded_nodes_count']-f['flooded_nodes_count'])
             / b['flooded_nodes_count']*100,
             f"-{(b['flooded_nodes_count']-f['flooded_nodes_count'])/b['flooded_nodes_count']*100:.1f}%",
             'benefit'),
            ('Reduccion de inundacion  (vol.)',
             f['flooding_reduction']/b['flooding_volume']*100,
             f"-{f['flooding_reduction']/b['flooding_volume']*100:.1f}%",
             'benefit'),
        ]

        labels  = [m[0] for m in metrics]
        values  = [m[1] for m in metrics]
        display = [m[2] for m in metrics]
        kinds   = [m[3] for m in metrics]
        bar_clr = {'benefit': GREEN,  'cost': ORANGE,  'info': BLUE}
        bg_clr  = {'benefit': LGREEN, 'cost': LORANG,  'info': LBLUE}

        fig = plt.figure(figsize=(13, 7.5))
        ax  = fig.add_axes([0.30, 0.10, 0.62, 0.76])
        yp  = np.arange(len(labels))

        for i, kind in enumerate(kinds):
            ax.axhspan(i-0.48, i+0.48, color=bg_clr[kind], alpha=0.28, zorder=0)
        for i, (val, kind) in enumerate(zip(values, kinds)):
            ax.barh(i, val, color=bar_clr[kind], alpha=0.88,
                    edgecolor='white', height=0.62, zorder=3)

        ax.set_xscale('symlog', linthresh=2, base=10)
        ax.axvline(0, color=SLATE, lw=1.2, alpha=0.5, zorder=4)
        ax.axhline(1.5, color=GRAY, lw=1.0, ls='--', alpha=0.4, zorder=2)
        ax.axhline(2.5, color=GRAY, lw=1.0, ls='--', alpha=0.4, zorder=2)

        vol_str = f"{f['total_tank_volume']/1000:.0f}K m3"
        for i, (val, disp) in enumerate(zip(values, display)):
            abs_w = abs(val)
            if disp == vol_str:
                ax.text(val*0.55, i, f"  {disp}", va='center', ha='left',
                        fontsize=10.5, fontweight='bold', color='white', zorder=6)
            elif abs_w >= 12:
                ax.text(val*0.58, i, disp, va='center', ha='center',
                        fontsize=10.5, fontweight='bold', color='white', zorder=6)
            else:
                pad  = 1.5
                xpos = val+pad if val >= 0 else val-pad
                ha   = 'left' if val >= 0 else 'right'
                ax.text(xpos, i, disp, va='center', ha=ha,
                        fontsize=10.5, fontweight='bold', color=SLATE, zorder=6)

        tick_c = {'benefit': GREEN, 'cost': ORANGE, 'info': BLUE}
        prefix = {'benefit': '(+)', 'cost': '(-)', 'info': '(i)'}
        ax.set_yticks(yp)
        ax.set_yticklabels(
            [f"{prefix[k]}  {l}" for l, k in zip(labels, kinds)], fontsize=10.5)
        for tick, kind in zip(ax.get_yticklabels(), kinds):
            tick.set_color(tick_c[kind])

        for txt, y_fig, col in [
            ('MEJORAS\nHIDRAULICAS', 0.67, GREEN),
            ('INFRAESTR.',           0.36, BLUE),
            ('COSTOS\nY CARGAS',     0.17, ORANGE),
        ]:
            fig.text(0.025, y_fig, txt, ha='center', va='center',
                     fontsize=8.5, fontweight='bold', color=col, alpha=0.8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec=col, alpha=0.5, lw=1))

        ax.set_xticks([-100,-25,-5,0,5,25,100,300])
        ax.set_xticklabels(['-100','-25','-5','0','+5','+25','+100','+300'],
                           fontsize=9, color=GRAY)
        ax.set_xlim(-165, 480)
        ax.set_xlabel('Cambio respecto al Baseline  (escala log simetrica)',
                      fontsize=10, color=GRAY, labelpad=8)
        ax.legend(handles=[
            mpatches.Patch(color=GREEN,  alpha=0.88, label='Mejora hidraulica'),
            mpatches.Patch(color=ORANGE, alpha=0.88, label='Carga / costo'),
            mpatches.Patch(color=BLUE,   alpha=0.88, label='Infraestructura instalada'),
        ], loc='lower right', fontsize=9.5, framealpha=0.9, edgecolor=GRAY)
        ax.set_title('Impacto Global: Baseline  →  Solucion Optimizada\n'
                     'Todos los indicadores clave en una sola vista',
                     fontsize=13, fontweight='bold', pad=14)

        if show: plt.show()
        self._savefig(fig, filename)
        plt.close(fig)
        return fig

    # ═════════════════════════════════════════════════════════════════════════
    # PARETO FLEXIBLE
    # ═════════════════════════════════════════════════════════════════════════
    def pareto(self, variables, custom_meta=None,
               filename_base='pareto', show=False):
        """
        Genera figuras de curvas Pareto para la lista de variables indicada.

        Maximo 4 paneles por figura. Si hay mas variables, se generan
        multiples figuras automaticamente:
            pareto_01.png  → primeras 4 variables
            pareto_02.png  → siguientes 4
            ...
        Si hay exactamente 1 figura, el nombre es simplemente pareto.png
        (o el filename_base que elijas).

        Parameters
        ----------
        variables : list[str]
            Nombres de columnas del CSV. Ejemplos:
                ['flooding_reduction', 'outfall_peak_flow']
                ['flooding_reduction', 'outfall_peak_flow',
                 'flooded_nodes_count', 'system_mean_utilization',
                 'surcharged_links_count']   ← genera 2 figuras

        custom_meta : dict, optional
            Para agregar o sobreescribir entradas del catalogo.
            Estructura por variable:
            {
                'mi_columna': {
                    'title':   'Titulo del panel',
                    'ylabel':  'Etiqueta eje Y',
                    'invert':  True/False,   # True = curva baja (menor=mejor)
                    'fmt_y':   lambda v: f'{v:.2f}',
                    'fmt_eff': lambda v: f'{v:.4f}/M$',
                    'c_good':  '#15803D',    # color eficiente
                    'c_bad':   '#EA580C',    # color ineficiente
                }
            }

        filename_base : str
            Prefijo del archivo de salida. Default 'pareto'.

        show : bool
            Si True muestra en pantalla ademas de guardar.

        Returns
        -------
        list[Figure]  Lista de figuras generadas.
        """
        plt.rcParams.update(_STYLE)

        # Catalogo efectivo para esta llamada
        catalog = dict(VAR_CATALOG)
        if custom_meta:
            catalog.update(custom_meta)

        # Resolver metadata de cada variable
        resolved = []
        for var in variables:
            if var not in self.df.columns:
                print(f'  [ERROR] "{var}" no existe en el CSV. Se omite.')
                continue
            if var in catalog:
                meta = catalog[var]
            else:
                print(f'  [aviso] "{var}" no esta en el catalogo. '
                      f'Usando configuracion generica.')
                meta = dict(
                    title   = var.replace('_', ' ').title(),
                    ylabel  = var,
                    invert  = False,
                    fmt_y   = lambda v: f'{v:.4g}',
                    fmt_eff = lambda v: f'{v:.4g}/M$',
                    c_good  = GREEN, c_bad = ORANGE,
                )
            resolved.append((var, meta))

        if not resolved:
            print('  [ERROR] No hay variables validas para graficar.')
            return []

        # Dividir en chunks de MAX_PANELS
        n_total = len(resolved)
        chunks  = [resolved[i:i+self.MAX_PANELS]
                   for i in range(0, n_total, self.MAX_PANELS)]
        n_figs  = len(chunks)

        figs = []
        for idx, chunk in enumerate(chunks):
            n             = len(chunk)
            nrows, ncols  = _LAYOUTS[n]
            fw = 7.5 * ncols
            fh = 5.5 * nrows + 1.2

            fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                                     squeeze=False)
            fig.subplots_adjust(
                hspace=0.42 if nrows > 1 else 0.25,
                wspace=0.30 if ncols > 1 else 0.20,
            )

            for ax_i, (var, meta) in enumerate(chunk):
                self._draw_panel(axes.flatten()[ax_i], var, meta)

            # Ocultar ejes sobrantes (p.ej. si chunk tiene 3 en layout 2x2)
            for ax_i in range(n, nrows * ncols):
                axes.flatten()[ax_i].set_visible(False)

            # Suptitle con info de figura
            if n_figs > 1:
                extra = f'Figura {idx+1} de {n_figs}'
            else:
                extra = f'{n_total} indicador{"es" if n_total > 1 else ""}'

            fig.suptitle(
                'Curvas Pareto: Inversion vs Indicadores Hidraulicos\n'
                'Color = eficiencia marginal real  |  '
                'Verde = sobre la mediana  |  Naranja/Rojo = bajo la mediana\n'
                f'({extra})',
                fontsize=11, fontweight='bold', y=1.02,
            )

            # Nombre de archivo
            fname = (f'{filename_base}.png' if n_figs == 1
                     else f'{filename_base}_{idx+1:02d}.png')

            if show: plt.show()
            self._savefig(fig, fname)
            plt.close(fig)
            figs.append(fig)

        return figs

    # ── Guardar todo ──────────────────────────────────────────────────────────
    def save_all(self, variables=None):
        """
        Genera mariposa + pareto de una vez.

        Parameters
        ----------
        variables : list[str] or None
            Variables para pareto(). Si None usa las 4 por defecto.
        """
        print('Generando graficas...')
        self.mariposa()
        self.pareto(variables or [
            'flooding_reduction',
            'outfall_peak_flow',
            'flooded_nodes_count',
            'system_mean_utilization',
        ])
        print('Listo.')


# ── Script directo ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'sequence_tracking.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'outputs'

    ch = StormwaterCharts(csv, outdir=out)

    # Prueba con 5 variables → genera pareto_01.png y pareto_02.png
    ch.pareto([
        'flooding_reduction',
        'outfall_peak_flow',
        'flooded_nodes_count',
        'system_mean_utilization',
        'surcharged_links_count',
    ])
    ch.mariposa()
