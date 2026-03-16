
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from datetime import datetime
from matplotlib.ticker import MaxNLocator, FuncFormatter
import json
import contextily as ctx

class EvolutionDashboardGenerator:
    """
    Generates evolution plots for the sequential optimization process.
    """
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        self.df = results_df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Counter for automatic figure numbering
        self._figure_counter = 0
        
        # Set modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
    
    def _get_next_figure_path(self, name: str, extension: str = "png") -> Path:
        """
        Generates a numbered filename for figures.
        Automatically increments the counter for each call.
        
        Args:
            name: Descriptive name for the figure (e.g., 'efficiency_by_tank')
            extension: File extension without dot (default: 'png')
        
        Returns:
            Path object with numbered filename: e.g., '00_efficiency_by_tank.png'
        """
        filename = f"{self._figure_counter:02d}_{name}.{extension}"
        self._figure_counter += 1
        return self.output_dir / filename
        
    def generate_all(self):
        """Generates all dashboard plots."""
        if self.df.empty:
            print("  [Dashboard] No data to plot.")
            return

        print("  [Dashboard] Generating evolution plots...")
        self.plot_efficiency_by_tank()
        self.plot_outfall_reduction_by_tank()
        self.plot_system_utilization_by_tank()
        self.plot_flooding_flow_by_tank()
        self.plot_cost_reduction_evolution()
        self.plot_individual_roi_curves()
        self.plot_pareto_curves()
        self.plot_system_evolution()

    def format_currency_smart(self, x, pos=None):
        """Dynamic formatting for large currency values."""
        if x >= 1e9:
            return f'${x*1e-9:.1f}B'
        elif x >= 1e6:
            return f'${x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'${x*1e-3:.0f}K'
        else:
            return f'${x:.0f}'

    def plot_efficiency_by_tank(self):
            """
            Scatter plot: Costo Total vs Reducción de Inundación por TANQUE FÍSICO.
            Agrupa todos los costos por tanque.
            """
            if len(self.df) < 1:
                return
                
            # Identify tank groups: when n_tanks increases, it's a new tank
            n_tanks = self.df['n_tanks'] if 'n_tanks' in self.df.columns else self.df['step']
            is_new_tank = n_tanks.diff().fillna(1) > 0
            tank_id = is_new_tank.cumsum()
            
            # Group by physical tank - sum all costs
            # Calculate marginal cost from consecutive investment totals
            self.df['_marginal_cost'] = self.df['cost_investment_total'].diff().fillna(self.df['cost_investment_total'].iloc[0])
            
            agg_dict = {
                '_marginal_cost': 'sum',
                'marginal_reduction': 'sum',
                'current_tank_volume': 'first',
                'added_predio': 'first',
                'step': 'first',
            }
            if 'final_tank_volume' in self.df.columns:
                agg_dict['final_tank_volume'] = 'first'
                
            tank_groups = self.df.groupby(tank_id).agg(agg_dict).reset_index(drop=True)
            
            # Filtrar solo tanques válidos (con costo > 0 y reducción >= 0)
            # Esto elimina el paso 0 (baseline) que no es un tanque real
            valid_tank_mask = (tank_groups['_marginal_cost'] > 0) & (tank_groups['marginal_reduction'] >= 0)
            tank_groups = tank_groups[valid_tank_mask].reset_index(drop=True)
            
            tank_cost = tank_groups['_marginal_cost']
            tank_reduction = tank_groups['marginal_reduction']
            
            vol_col = 'final_tank_volume' if 'final_tank_volume' in tank_groups.columns else 'current_tank_volume'
            tank_volume = tank_groups[vol_col].fillna(1000)
            
            tank_predio = tank_groups['added_predio']
            tank_step = tank_groups['step']
            
            # Calculate efficiency: $/m³ reduced
            # Mark as invalid (inf) if cost <= 0 OR reduction <= 0
            cost_per_m3 = tank_cost / tank_reduction.replace(0, np.nan)
            
            # Identify invalid entries (negative cost, negative reduction, or zero)
            is_invalid = (tank_cost <= 0) | (tank_reduction <= 0) | (cost_per_m3 < 0)
            
            # For valid entries only, get min/max for colormap
            valid_cpm = cost_per_m3[~is_invalid]
            if len(valid_cpm) > 0:
                cpm_valid_max = valid_cpm.max()
            else:
                cpm_valid_max = 1000  # fallback
            
            # Replace invalid with a very high value for sorting (will go to bottom)
            cost_per_m3_for_sort = cost_per_m3.copy()
            cost_per_m3_for_sort[is_invalid] = np.inf
            
            # For display, keep original but mark as invalid
            cost_per_m3 = cost_per_m3.fillna(cpm_valid_max)
            
            # Normalize for colormap (inverted: lower = better = green)
            cpm_min, cpm_max = cost_per_m3.min(), cost_per_m3.max()
            if cpm_max > cpm_min:
                cpm_normalized = 1 - ((cost_per_m3 - cpm_min) / (cpm_max - cpm_min))
            else:
                cpm_normalized = pd.Series([0.5] * len(cost_per_m3))


            # Create figure with table left, two charts stacked on right - FORMATO A4
            fig = plt.figure(figsize=(24, 14))  # Más ancho para mejor legibilidad
            
            # Create GridSpec: left column for table, right column for 2 stacked charts
            gs = fig.add_gridspec(2, 2, width_ratios=[0.30, 0.70], height_ratios=[1, 1], 
                                  wspace=0.15, hspace=0.30)
            ax_table = fig.add_subplot(gs[:, 0])  # Table spans both rows
            ax_bar = fig.add_subplot(gs[0, 1])    # Bar chart on top
            ax_evo = fig.add_subplot(gs[1, 1])    # Evolution chart on bottom
            
            # === LEFT PANEL: TABLE ===
            ax_table.axis('off')
            
            # Prepare table data - KEEP ORIGINAL ORDER FIRST
            table_rows = []
            for i in range(len(tank_groups)):
                # Check if this entry is invalid
                is_inv = is_invalid.iloc[i] if hasattr(is_invalid, 'iloc') else is_invalid[i]
                eff_display = "N/A" if is_inv else f"${cost_per_m3.iloc[i]:,.0f}"
                eff_sort = cost_per_m3_for_sort.iloc[i] if hasattr(cost_per_m3_for_sort, 'iloc') else cost_per_m3_for_sort[i]
                
                table_rows.append({
                    'original_idx': i,  # Track original index
                    'tank_label': f"T{i+1}",
                    'predio': f"P{tank_predio.iloc[i]}",
                    'costo': self.format_currency_smart(tank_cost.iloc[i]),
                    'reduccion': f"{tank_reduction.iloc[i]:,.0f}",
                    'volumen': f"{tank_volume.iloc[i]:,.0f}",
                    'eficiencia': eff_display,
                    'eficiencia_num': eff_sort,  # For sorting (inf for invalid)
                    'is_invalid': is_inv
                })
            
            # Sort by efficiency (lowest first = best, inf goes to bottom)
            table_rows_sorted = sorted(table_rows, key=lambda x: x['eficiencia_num'])

            
            # Create table data for display - INCLUDE TANK LABEL
            table_data = []
            for rank, row in enumerate(table_rows_sorted, 1):
                table_data.append([
                    str(rank),
                    row['tank_label'],  # ADD TANK LABEL
                    row['predio'],
                    row['costo'],
                    row['reduccion'],
                    row['volumen'],
                    row['eficiencia']
                ])
            
            col_labels = ['Rank', 'Tank', 'Predio', 'Costo', 'Reduc.', 'Vol.', '$/m³']
            
            # Create table
            table = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc='center',
                loc='center',
                colColours=['#2e5cb8'] * 7,
                bbox=[0.0, 0.05, 1.0, 0.85]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(15)  # Font más grande para tabla
            table.scale(1.2, 2.2)   # Escala más grande
            
            # Style table - mark invalid rows with light red
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(fontweight='bold', color='white', fontsize=13)
                    cell.set_facecolor('#2e5cb8')
                    cell.set_height(0.08)
                else:  # Data rows
                    row_data = table_rows_sorted[i-1] if i-1 < len(table_rows_sorted) else None
                    if row_data and row_data.get('is_invalid', False):
                        # Invalid entries: light red background
                        cell.set_facecolor('#ffcdd2')
                        cell.set_text_props(color='#b71c1c')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_height(0.08)
                cell.set_edgecolor('#aaaaaa')
                cell.set_linewidth(1.5)
            
            ax_table.set_title('Ranking de Eficiencia\n(Mejor → Peor)',
                              fontsize=14, fontweight='bold', pad=10)
            
            # === TOP RIGHT: BAR CHART - Eficiencia por Tanque ===
            # Usar orden de instalación (1°, 2°, 3°...) en lugar de nombres de tanque
            tank_labels = [f"{i+1}°" for i in range(len(tank_groups))]
            
            # Get efficiency values (use a high value for invalid ones for display)
            eff_values = cost_per_m3.copy()
            eff_display = eff_values.copy()
            
            # For invalid entries, set to 0 for bar height (will show as N/A)
            for i in range(len(eff_display)):
                if is_invalid.iloc[i] if hasattr(is_invalid, 'iloc') else is_invalid[i]:
                    eff_display.iloc[i] = 0
            
            # Colors: normalize valid entries only (green = low/good, red = high/bad)
            valid_eff = eff_values[~is_invalid]
            if len(valid_eff) > 0:
                eff_mean ,eff_min, eff_max = valid_eff.mean(), valid_eff.min(), valid_eff.max()
            else:
                eff_min, eff_max = 0, 1000
                
            # Create color array
            colors = []
            for i in range(len(eff_values)):
                is_inv = is_invalid.iloc[i] if hasattr(is_invalid, 'iloc') else is_invalid[i]
                if is_inv:
                    colors.append('#cccccc')  # Gray for invalid
                else:
                    # Normalize: 0 = best (green), 1 = worst (red)
                    if eff_max > eff_min:
                        norm = (eff_values.iloc[i] - eff_min) / (eff_max - eff_min)
                    else:
                        norm = 0.5
                    colors.append(plt.cm.RdYlGn(1 - norm))  # Invert so green=low
            
            # Create bar chart
            x_pos = np.arange(1, len(tank_labels) + 1)  # Alineado con tank_numbers (1-12)
            bars = ax_bar.bar(x_pos, eff_display, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, eff_values)):
                is_inv = is_invalid.iloc[i] if hasattr(is_invalid, 'iloc') else is_invalid[i]
                if is_inv:
                    label = "N/A"
                    y_pos = eff_max * 0.1 if eff_max > 0 else 100
                    color = '#888888'
                else:
                    label = f"${val:,.0f}"
                    y_pos = bar.get_height()
                    color = 'black'
                
                ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos + eff_max*0.03,
                       label, ha='center', va='bottom', fontsize=15, fontweight='bold',
                       color=color, rotation=25)
            
            # Styling
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels([])  # Sin etiquetas - comparte eje con evo
            ax_bar.set_xlabel('')  # Sin label - comparte eje con evo
            ax_bar.set_xlim(0.5, len(tank_labels) + 0.5)  # Alineado con evo
            ax_bar.set_ylabel('Eficiencia ($/m³)', fontsize=16, fontweight='bold')
            # Título sin N/A si es 0 - en una sola línea con offset hacia abajo
            n_na = is_invalid.sum()
            title_suffix = f'({len(tank_groups)} Tanques, {n_na} N/A)' if n_na > 0 else f'({len(tank_groups)} Tanques)'
            ax_bar.set_title(f'Eficiencia por Tanque: Costo por m³ Reducido {title_suffix}',
                         fontsize=16, fontweight='bold', pad=35)
            
            # Grid
            ax_bar.grid(True, linestyle='--', alpha=0.4, color='#cccccc', axis='y', linewidth=1)
            ax_bar.set_axisbelow(True)
            
            # Y axis formatting
            ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax_bar.tick_params(axis='y', labelsize=15)
            
            # # Add average line
            # if len(valid_eff) > 0:
            #
            #     ax_bar.axhline(y=eff_mean, color='#1976d2', linestyle='--', linewidth=2.5,
            #               label=f'Promedio: ${eff_mean:,.0f}/m³')
            #     ax_bar.legend(loc='upper right', fontsize=11, framealpha=0.95)
            
            # === BOTTOM RIGHT: EVOLUTION CHART - Eficiencia Promedio Acumulada ===
            # Calculate running average efficiency: total_cost / total_reduction at each tank
            tank_numbers = np.arange(1, len(tank_groups) + 1)  # 1 to N tanks
            
            # Calculate cumulative cost and reduction for each tank
            cumulative_cost = np.cumsum(tank_cost.values)
            cumulative_reduction = np.cumsum(tank_reduction.values)
            
            # Running average efficiency
            running_avg_eff = np.where(cumulative_reduction > 0, 
                                       cumulative_cost / cumulative_reduction, 
                                       np.nan)
            
            # Plot line
            ax_evo.plot(tank_numbers, running_avg_eff, 'o-', color='#1976d2', linewidth=3,
                       markersize=12, markeredgecolor='#0d47a1', markeredgewidth=2,
                       label='Eficiencia Promedio Acumulada')
            
            # Fill area under curve
            ax_evo.fill_between(tank_numbers, 0, running_avg_eff, alpha=0.2, color='#64b5f6')
            
            # Add value labels
            for i, (tank_num, eff) in enumerate(zip(tank_numbers, running_avg_eff)):
                if not np.isnan(eff):
                    ax_evo.annotate(f'${eff:,.0f}', xy=(tank_num, eff), xytext=(0, 12),
                                   textcoords='offset points', ha='center', va='bottom',
                                   fontsize=15, fontweight='bold', color='#0d47a1',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            alpha=0.9, edgecolor='none'))
            
            # Styling - usar orden de instalación
            ax_evo.set_xticks(tank_numbers)
            ax_evo.set_xticklabels(tank_labels, fontsize=14, fontweight='bold')
            ax_evo.set_xlabel('Tanque (Orden de Instalación)', fontsize=15, fontweight='bold')
            ax_evo.set_ylabel('Eficiencia Promedio ($/m³)', fontsize=14, fontweight='bold')
            ax_evo.set_title('Evolución de la Eficiencia Promedio: $/m³ al Agregar Tanques',
                            fontsize=16, fontweight='bold', pad=10)
            
            ax_evo.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=1)
            ax_evo.set_axisbelow(True)
            ax_evo.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax_evo.tick_params(axis='y', labelsize=15)
            ax_evo.legend(loc='upper left', fontsize=15, framealpha=0.95)
            
            # Summary box removido a pedido del usuario
            
            # === EXPORT TABLE TO CSV ===
            csv_data = []
            for rank, row in enumerate(table_rows_sorted, 1):
                orig_idx = row['original_idx']
                csv_data.append({
                    'Rank': rank,
                    'Tanque': row['tank_label'],
                    'Predio': tank_predio.iloc[orig_idx],
                    'Costo_Total': tank_cost.iloc[orig_idx],
                    'Reduccion_m3': tank_reduction.iloc[orig_idx],
                    'Volumen_m3': tank_volume.iloc[orig_idx],
                    'Eficiencia_USD_m3': cost_per_m3.iloc[orig_idx]
                })
            
            df_table = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "efficiency_ranking.csv"
            df_table.to_csv(csv_path, index=False)
            print(f"  [Dashboard] Saved: {csv_path}")
            
            plt.tight_layout()
            
            save_path = self._get_next_figure_path("efficiency_by_tank")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  [Dashboard] Saved: {save_path}")

    def plot_outfall_reduction_by_tank(self):
        """
        Gráfico de REDUCCIÓN DE CAUDAL EN OUTFALL por tanque (m³/s).
        Muestra cuánto reduce cada tanque en caudal absoluto.
        """
        if len(self.df) < 1 or 'outfall_peak_flow' not in self.df.columns:
            return
        
        # Lógica de agrupación por tanque
        n_tanks = self.df['n_tanks'] if 'n_tanks' in self.df.columns else self.df['step']
        is_new_tank = n_tanks.diff().fillna(1) > 0
        tank_id = is_new_tank.cumsum()
        
        agg_dict = {
            'outfall_peak_flow': ['first', 'last'],
            'cost_investment_total': ['first', 'last'],
            'marginal_reduction': 'sum',
            'current_tank_volume': 'first',
            'added_predio': 'first',
            'step': 'first',
        }
        if 'final_tank_volume' in self.df.columns:
            agg_dict['final_tank_volume'] = 'first'
        
        tank_groups = self.df.groupby(tank_id).agg(agg_dict).reset_index(drop=True)
        
        # Aplanar MultiIndex
        tank_groups.columns = ['_'.join(col).strip() if col[1] else col[0] for col in tank_groups.columns.values]
        
        # Calcular reducción neta
        # En el caso de outfall y cost, es el final del grupo actual vs el final del grupo anterior
        # Pero podemos usar last - first_of_next, sin embargo, el last de T1 es el first de T2 (en terms de estado base)
        # Vamos a usar la misma logica simple pero a nivel de los datos diarios
        
        # Filtrar tanques válidos (costo > 0)
        # tank_groups['cost_investment_total_last'] - tank_groups['cost_investment_total_first'] no sirve bien si hay gaps
        # Usamos los marginales calculados antes de agrupar para evitar issues con el first/last
        self.df['_marginal_outfall_reduction'] = self.df['outfall_peak_flow'].diff().fillna(0) * -1
        self.df['_marginal_cost'] = self.df['cost_investment_total'].diff().fillna(self.df['cost_investment_total'].iloc[0])
        
        agg_dict_marg = {
            '_marginal_cost': 'sum',
            'marginal_reduction': 'sum',
            '_marginal_outfall_reduction': 'sum',
            'current_tank_volume': 'first',
            'added_predio': 'first',
            'step': 'first',
        }
        if 'final_tank_volume' in self.df.columns:
            agg_dict_marg['final_tank_volume'] = 'first'
            
        tank_groups = self.df.groupby(tank_id).agg(agg_dict_marg).reset_index(drop=True)
        
        valid_tank_mask = tank_groups['_marginal_cost'] > 0
        tank_groups = tank_groups[valid_tank_mask].reset_index(drop=True)
        
        tank_cost = tank_groups['_marginal_cost']
        outfall_reduction = tank_groups['_marginal_outfall_reduction']
        flooding_reduction = tank_groups['marginal_reduction']
        
        vol_col = 'final_tank_volume' if 'final_tank_volume' in tank_groups.columns else 'current_tank_volume'
        tank_volume = tank_groups[vol_col].fillna(1000)
        tank_predio = tank_groups['added_predio']
        
        # No hay inválidos - mostrar todos los valores incluyendo negativos
        is_invalid = pd.Series([False] * len(outfall_reduction))
        
        # Figura
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.30, 0.70], height_ratios=[1, 1], 
                              wspace=0.15, hspace=0.30)
        ax_table = fig.add_subplot(gs[:, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_evo = fig.add_subplot(gs[1, 1])
        
        # === TABLA: Ranking por reducción de caudal (mayor primero) ===
        ax_table.axis('off')
        
        table_rows = []
        for i in range(len(tank_groups)):
            table_rows.append({
                'original_idx': i,
                'tank_label': f"T{i+1}",
                'predio': f"P{tank_predio.iloc[i]}",
                'reduccion_outfall': outfall_reduction.iloc[i],
                'reduccion_flood': flooding_reduction.iloc[i],
                'volumen': tank_volume.iloc[i],
            })
        
        # Ordenar por reducción de caudal (mayor primero)
        table_rows_sorted = sorted(table_rows, key=lambda x: x['reduccion_outfall'], reverse=True)
        
        # Calcular reducción acumulada y porcentaje
        cumulative_reduction = 0
        # Obtener caudal inicial (baseline) para calcular porcentaje
        initial_outfall = self.df['outfall_peak_flow'].iloc[0] if len(self.df) > 0 else 1
        table_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            # Mostrar valor marginal con signo (+/-)
            reduc_val = row['reduccion_outfall']
            reduc_str = f"{reduc_val:+.2f}" if reduc_val != 0 else "0.00"
            # Calcular porcentaje de reducción
            pct_val = (reduc_val / initial_outfall) * 100 if initial_outfall != 0 else 0
            pct_str = f"{pct_val:+.1f}%"
            table_data.append([
                str(rank),
                row['tank_label'],
                row['predio'],
                reduc_str,
                pct_str,  # Porcentaje de reducción
                f"{row['volumen']:,.0f}",
            ])
        
        col_labels = ['Rank', 'Tank', 'Predio', 'Caudal\n(m³/s)', 'Reduc.\n(%)', 'Vol.\n(m³)']
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colColours=['#2e7d32'] * 6,
            bbox=[0.0, 0.05, 1.0, 0.85]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(15)  # Mismo tamaño que eficiencia
        table.scale(1.2, 2.2)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', color='white', fontsize=12)
                cell.set_facecolor('#2e7d32')
                cell.set_height(0.08)
            else:
                row_data = table_rows_sorted[i-1] if i-1 < len(table_rows_sorted) else None
                if row_data and row_data.get('is_invalid', False):
                    cell.set_facecolor('#ffcdd2')
                    cell.set_text_props(color='#b71c1c')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_height(0.08)
            cell.set_edgecolor('#aaaaaa')
            cell.set_linewidth(1.5)
        
        ax_table.set_title('Ranking por Reducción\nde Caudal Outfall\n(Mejor → Peor)',
                          fontsize=14, fontweight='bold', pad=10)
        
        # === BARRAS: Reducción de caudal por tanque ===
        tank_labels = [f"{i+1}°" for i in range(len(tank_groups))]
        
        # Colores: verde para reducción, rojo para aumento
        colors = []
        for val in outfall_reduction:
            if val > 0:
                colors.append('#2e7d32')  # Verde oscuro - reducción
            elif val < 0:
                colors.append('#c62828')  # Rojo - aumento
            else:
                colors.append('#9e9e9e')  # Gris - sin cambio
        
        x_pos = np.arange(1, len(tank_labels) + 1)  # Alineado con tank_numbers (1-12)
        bars = ax_bar.bar(x_pos, outfall_reduction, color=colors, edgecolor='black', linewidth=1.5)
        
        # Labels en barras
        out_max = outfall_reduction.max()
        out_min = outfall_reduction.min()
        y_range = out_max - out_min if out_max != out_min else 1
        
        for bar, val in zip(bars, outfall_reduction):
            label = f"{val:+.2f}" if val != 0 else "0.00"
            # Posicionar label arriba si es positivo, abajo si es negativo
            if val >= 0:
                y_pos = bar.get_height() + y_range * 0.02
                va = 'bottom'
            else:
                y_pos = bar.get_height() - y_range * 0.02
                va = 'top'
            ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos,
                   label, ha='center', va=va, fontsize=14, fontweight='bold',
                   color='black', rotation=0)
        
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([])  # Sin etiquetas en X (comparte con evo)
        ax_bar.set_xlabel('')
        ax_bar.set_xlim(0.5, len(tank_groups) + 0.5)  # Alineado con evo
        ax_bar.set_ylabel('Reducción Caudal Outfall (m³/s)', fontsize=15, fontweight='bold')
        ax_bar.set_title(f'Efecto en Caudal Outfall por Tanque ({len(tank_groups)} Tanques)',
                     fontsize=16, fontweight='bold', pad=35)
        
        ax_bar.grid(True, linestyle='--', alpha=0.4, color='#cccccc', axis='y', linewidth=1)
        ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis='y', labelsize=14)
        
        # === EVOLUCIÓN: Reducción acumulada ===
        tank_numbers = np.arange(1, len(tank_groups) + 1)
        cumulative_outfall = np.cumsum(outfall_reduction.values)
        
        ax_evo.plot(tank_numbers, cumulative_outfall, 'o-', color='#2e7d32', linewidth=3,
                   markersize=12, markeredgecolor='#1b5e20', markeredgewidth=2,
                   label='Reducción Acumulada Outfall')
        
        ax_evo.fill_between(tank_numbers, 0, cumulative_outfall, alpha=0.2, color='#81c784')
        
        for i, (tank_num, val) in enumerate(zip(tank_numbers, cumulative_outfall)):
            ax_evo.annotate(f'{val:.1f}', xy=(tank_num, val), xytext=(0, 12),
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=14, fontweight='bold', color='#1b5e20',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.9, edgecolor='none'))
        
        ax_evo.set_xticks(tank_numbers)
        ax_evo.set_xticklabels(tank_labels, fontsize=14, fontweight='bold')
        ax_evo.set_xlabel('Tanque (Orden de Instalación)', fontsize=15, fontweight='bold')
        ax_evo.set_xlim(0.5, len(tank_groups) + 0.5)  # Alineado con bar
        ax_evo.set_ylabel('Reducción Acumulada (m³/s)', fontsize=15, fontweight='bold')
        ax_evo.set_title('Evolución de la Reducción Acumulada de Caudal en Outfall',
                        fontsize=16, fontweight='bold', pad=10)
        
        ax_evo.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=1)
        ax_evo.set_axisbelow(True)
        ax_evo.tick_params(axis='y', labelsize=14)
        ax_evo.legend(loc='upper left', fontsize=14, framealpha=0.95)
        
        # Export CSV
        csv_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            orig_idx = row['original_idx']
            csv_data.append({
                'Rank': rank,
                'Tanque': row['tank_label'],
                'Predio': tank_predio.iloc[orig_idx],
                'Efecto_Outfall_m3s': outfall_reduction.iloc[orig_idx],
                'Reduccion_Flooding_m3': flooding_reduction.iloc[orig_idx],
                'Volumen_Tanque_m3': tank_volume.iloc[orig_idx],
            })
        
        df_table = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "outfall_effect_ranking.csv"
        df_table.to_csv(csv_path, index=False)
        print(f"  [Dashboard] Saved: {csv_path}")
        
        plt.tight_layout()
        
        save_path = self._get_next_figure_path("outfall_reduction_by_tank")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_cost_reduction_evolution(self):
        """
        Evolución de Inversión vs Reducción — multiplot resumen 3x2
        + gráficos individuales por variable.
        """
        if len(self.df) < 1:
            return

        steps = self.df["step"].values
        cost_total = self.df["cost_investment_total"].values
        colors_bars = plt.cm.Blues(np.linspace(0.3, 0.55, len(steps)))

        # Variable definitions with units
        var_defs = [
            ('flooding_volume', 'Vol. Inundación', '#c62828', 'o', 'm³'),
            ('flooding_flow', 'Caudal Inundación', '#e65100', 's', 'm³/s'),
            ('outfall_peak_flow', 'Pico Outfall', '#1565c0', 'D', 'm³/s'),
            ('flooded_nodes_count', 'Nodos Inundados', '#00838f', 'p', 'nodos'),
            ('surcharged_links_count', 'Links Sobrecargados', '#2e7d32', 'v', 'links'),
            ('system_mean_utilization', 'Utilización Red', '#6a1b9a', '^', '%'),
        ]

        # Filter valid variables and pre-compute
        valid_vars = []
        for col, label, color, marker, unit_label in var_defs:
            if col in self.df.columns:
                vals = self.df[col].values.astype(float)
                if vals[0] > 0:
                    pct = ((vals[0] - vals) / vals[0]) * 100
                    valid_vars.append((col, label, color, marker, unit_label, pct, vals))

        if not valid_vars:
            return

        # ── Helper: draw one variable on a given axes pair ──
        def _draw_variable(ax_bar, ax_line, col, label, color, marker,
                          unit_label, pct, raw_vals, show_all_values=False):
            """Draw bars + line + annotations for one variable."""
            # Investment bars
            ax_bar.bar(steps, cost_total, color=colors_bars,
                      edgecolor="#7bafd4", linewidth=0.5, alpha=0.25)
            ax_bar.set_ylabel("Inversión Acumulada (USD)", fontsize=13 if show_all_values else 11,
                            color="#1a4d8f", fontweight="bold")
            ax_bar.tick_params(axis="y", labelcolor="#1a4d8f", labelsize=12 if show_all_values else 10)
            ax_bar.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))

            # % reduction line
            ax_line.plot(steps, pct, f'{marker}-', color=color,
                        linewidth=2.5, markersize=10 if show_all_values else 8,
                        markeredgecolor='white', markeredgewidth=1.0, alpha=0.95)
            ax_line.set_ylim(-5, 105)
            ax_line.set_ylabel("Reducción (%)", fontsize=13 if show_all_values else 11,
                             color=color, fontweight="bold")
            ax_line.tick_params(axis="y", labelcolor=color, labelsize=12 if show_all_values else 10)
            ax_line.grid(False)  # disable twinx default grid

            # (no horizontal reference lines)

            # Format raw value (number only — units are in the title)
            if 'volume' in col:
                fmt_v = lambda v: f"{v:,.0f}"
            elif 'flow' in col:
                fmt_v = lambda v: f"{v:,.2f}"
            elif 'count' in col or 'nodes' in col or 'links' in col:
                fmt_v = lambda v: f"{int(v)}"
            elif 'utilization' in col:
                fmt_v = lambda v: f"{v:.1f}"
            else:
                fmt_v = lambda v: f"{v:,.1f}"

            # ALL values shown in both modes
            val_indices = list(range(len(steps)))
            pct_step = 2 if show_all_values else 3

            # Raw values at top of plot — rotated 45°, black, single row
            for i in val_indices:
                ax_line.annotate(
                    fmt_v(raw_vals[i]),
                    xy=(steps[i], 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 8), textcoords="offset points",
                    ha="left", va="bottom",
                    fontsize=10 if show_all_values else 8.5,
                    fontweight="bold", color="#333333",
                    rotation=45,
                    clip_on=False, annotation_clip=False,
                )

            # % reduction near data points
            pct_indices = [i for i in range(len(steps))
                          if i == 0 or i == len(steps)-1 or i % pct_step == 0]
            for i in pct_indices:
                ax_line.annotate(
                    f"{pct[i]:.1f}%",
                    xy=(steps[i], pct[i]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11 if show_all_values else 9,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             alpha=0.85, edgecolor=color, linewidth=0.4),
                )

            ax_bar.grid(True, linestyle=":", color="#dddddd", alpha=0.5, axis="y", linewidth=0.5)
            ax_bar.grid(True, linestyle=":", color="#dddddd", alpha=0.4, axis="x", linewidth=0.5)
            ax_bar.set_axisbelow(True)

        # ══════════════════════════════════════════════════════════
        # PART 1: Summary multiplot 3×2
        # ══════════════════════════════════════════════════════════
        n_vars = len(valid_vars)
        ncols = 2
        nrows = (n_vars + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True)
        if nrows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Evolución de la Inversión vs Reducción de Indicadores Hidráulicos",
                    fontsize=18, fontweight="bold", y=0.98)

        for vi, (col, label, color, marker, unit_label, pct, raw_vals) in enumerate(valid_vars):
            row, col_idx = divmod(vi, ncols)
            ax_bar = axes[row, col_idx]
            ax_line = ax_bar.twinx()

            # Title: includes unit so individual values don't need it
            ax_bar.set_title(f"{label} [{unit_label}]  →  {pct[-1]:.1f}%",
                           fontsize=14, fontweight="bold", color=color, pad=40)

            _draw_variable(ax_bar, ax_line, col, label, color, marker,
                          unit_label, pct, raw_vals, show_all_values=False)

        # X-axis labels on bottom row
        for col_idx in range(ncols):
            ax_bottom = axes[-1, col_idx]
            ax_bottom.set_xticks(steps)
            ax_bottom.set_xticklabels(steps, fontsize=13, fontweight="bold")
            ax_bottom.set_xlabel("Paso de Optimización", fontsize=13, fontweight="bold")

        if n_vars % 2 == 1:
            axes[-1, -1].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3, w_pad=2)

        save_path = self._get_next_figure_path("cost_reduction_evolution")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

        # Removed 'PART 2: Individual full-page graphs' block as per user request to drop graphs 05 to 11.

    def plot_roi_curve(self):
        """
        ROI curve — formato original con múltiples variables.
        Top: ROI acumulado (% reducción / $M) por variable.
        Bottom: ROI marginal (barras agrupadas) por variable por paso.
        Usa % reducción para normalizar entre variables con unidades distintas.
        """
        if len(self.df) < 1:
            return

        steps = self.df['step'].values
        cost_total = self.df['cost_investment_total'].values
        cost_in_millions = cost_total / 1_000_000

        # Marginal cost per step
        marginal_cost_M = np.diff(cost_total, prepend=0) / 1_000_000

        # Variables to plot — use % reduction so all are comparable
        var_defs = [
            ('flooding_volume', 'Vol. Inundación', '#c62828', 'o'),
            ('flooding_flow', 'Caudal Inundación', '#e65100', 's'),
            ('outfall_peak_flow', 'Pico Outfall', '#1565c0', 'D'),
            ('flooded_nodes_count', 'Nodos Inundados', '#00838f', 'p'),
            ('surcharged_links_count', 'Links Sobrecargados', '#2e7d32', 'v'),
            ('system_mean_utilization', 'Utilización Red', '#6a1b9a', '^'),
        ]

        variables = []
        for col, label, color, marker in var_defs:
            if col in self.df.columns:
                vals = self.df[col].values.astype(float)
                baseline = vals[0]
                if baseline > 0:
                    # % reduction from baseline
                    pct_reduction = ((baseline - vals) / baseline) * 100
                    # Cumulative ROI: % reduced / $M invested
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cum_roi = np.where(cost_in_millions > 0, pct_reduction / cost_in_millions, 0)
                    # Marginal ROI: marginal % / marginal $M
                    marginal_pct = np.diff(pct_reduction, prepend=0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        marg_roi = np.where(marginal_cost_M > 0, marginal_pct / marginal_cost_M, 0)
                    variables.append({
                        'col': col, 'label': label, 'color': color, 'marker': marker,
                        'cum_roi': cum_roi, 'marg_roi': marg_roi,
                    })

        if not variables:
            return

        # Create figure — same as original: 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12),
                                         gridspec_kw={'height_ratios': [1, 1]})

        # ===== TOP: Cumulative ROI (lines) =====
        for vi, v in enumerate(variables):
            ax1.plot(steps, v['cum_roi'], f"{v['marker']}-", color=v['color'],
                    linewidth=2.5 if vi == 0 else 2.0,
                    markersize=10 if vi == 0 else 7,
                    markeredgecolor='white', markeredgewidth=1,
                    label=v['label'], alpha=0.9)

        # Fill under main variable only
        if variables:
            ax1.fill_between(steps, 0, variables[0]['cum_roi'],
                           alpha=0.15, color=variables[0]['color'])

        # Labels for main variable — key points only (first, best, last)
        main_roi = variables[0]['cum_roi']
        best_roi_idx = np.argmax(main_roi)
        key_points = list(set([0, best_roi_idx, len(steps)-1]))
        for idx in key_points:
            if main_roi[idx] > 0:
                ax1.annotate(f'{main_roi[idx]:.2f}', xy=(steps[idx], main_roi[idx]),
                            xytext=(0, 20), textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color=variables[0]['color'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     alpha=0.95, edgecolor=variables[0]['color'], linewidth=1.5))

        # Optimal point marker for main variable
        ax1.axvline(x=steps[best_roi_idx], color='#2e7d32', linestyle=':',
                   linewidth=2.5, alpha=0.6, label=f'Mejor ROI: Paso {steps[best_roi_idx]}')
        ax1.scatter(steps[best_roi_idx], main_roi[best_roi_idx], s=300, color='#4caf50',
                   edgecolor='#1b5e20', linewidth=3, zorder=10, marker='*')

        ax1.set_ylabel('% Reducción / $1M invertido', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_title('Retorno de Inversión (ROI) Acumulado — Multi-Variable',
                     fontsize=13, fontweight='bold', pad=15)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax1.tick_params(axis='both', labelsize=10)
        ax1.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, linewidth=1)
        ax1.set_axisbelow(True)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([])  # Hide x labels on top plot
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.95,
                  edgecolor='black', fancybox=True, ncol=2)
        y_max = max(v['cum_roi'].max() for v in variables) * 1.15
        ax1.set_ylim(0, y_max)

        # ===== BOTTOM: Marginal ROI (grouped bars) =====
        n_vars = len(variables)
        bar_width = 0.7 / n_vars
        x = np.arange(len(steps))

        for vi, v in enumerate(variables):
            offset = (vi - n_vars/2 + 0.5) * bar_width
            ax2.bar(x + offset, v['marg_roi'], bar_width * 0.9,
                   color=v['color'], alpha=0.75, label=v['label'],
                   edgecolor='white', linewidth=0.5)

        # Average line for main variable
        main_marg = variables[0]['marg_roi']
        avg_main = np.mean(main_marg[main_marg > 0]) if np.any(main_marg > 0) else 0
        ax2.axhline(y=avg_main, color=variables[0]['color'], linestyle='--',
                   linewidth=2.5, label=f'Promedio {variables[0]["label"]}: {avg_main:.1f}',
                   alpha=0.7)

        ax2.set_xlabel('Paso de Optimización (Tanque)', fontsize=11, fontweight='bold', labelpad=10)
        ax2.set_ylabel('ROI Marginal\n(% red./$1M)', fontsize=11, fontweight='bold', labelpad=10)
        ax2.set_title('ROI Marginal por Paso — Multi-Variable',
                     fontsize=13, fontweight='bold', pad=15)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, axis='y', linewidth=1)
        ax2.set_axisbelow(True)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'p{s}' for s in steps], fontsize=9, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.95,
                  edgecolor='black', fancybox=True, ncol=2)

        plt.tight_layout()

        save_path = self._get_next_figure_path("roi_curve")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_individual_roi_curves(self):
        """
        Generates individual ROI curves for each hydraulic indicator.
        Format: Top = Cumulative ROI, Bottom = Marginal ROI.
        """
        if len(self.df) < 1:
            return

        steps = self.df['step'].values
        cost_total = self.df['cost_investment_total'].values
        cost_in_millions = cost_total / 1_000_000
        marginal_cost_M = np.diff(cost_total, prepend=0) / 1_000_000

        var_defs = [
            ('flooding_volume', 'Volumen de Inundacion', 'm³'),
            ('flooding_flow', 'Caudal de Inundacion', 'm³/s'),
            ('outfall_peak_flow', 'Caudal Pico Outfall', 'm³/s'),
            ('flooded_nodes_count', 'Nodos Inundados', 'nodos'),
            ('surcharged_links_count', 'Links Sobrecargados', 'links'),
            ('system_mean_utilization', 'Utilizacion de Red', '%'),
        ]

        for col, label, unit in var_defs:
            if col not in self.df.columns:
                continue
                
            vals = self.df[col].values.astype(float)
            baseline = vals[0]
            if baseline <= 0:
                continue
                
            # Reduction in native units
            reduction = baseline - vals
            
            # Cumulative ROI: native units reduced / $1M invested
            with np.errstate(divide='ignore', invalid='ignore'):
                cum_roi = np.where(cost_in_millions > 0, reduction / cost_in_millions, 0)
                
            # Marginal ROI: marginal reduction / marginal $1M
            marginal_reduction = np.diff(reduction, prepend=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                marg_roi = np.where(marginal_cost_M > 0, marginal_reduction / marginal_cost_M, 0)
            
            # Formatter logic depending on magnitude
            if np.max(cum_roi) >= 1000:
                fmt_fn = lambda x: f'{x:,.0f}'
            elif np.max(cum_roi) >= 10:
                fmt_fn = lambda x: f'{x:,.1f}'
            else:
                fmt_fn = lambda x: f'{x:,.2f}'

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [1, 1]})
            
            # 1. TOP PLOT
            best_roi_idx = np.argmax(cum_roi)
            
            ax1.plot(steps, cum_roi, 'o-', color='#1976d2', linewidth=4,
                            markersize=14, markeredgecolor='#0d47a1', markeredgewidth=2,
                            label='ROI Acumulado')
            ax1.fill_between(steps, 0, cum_roi, alpha=0.25, color='#64b5f6')
            
            # Value labels on top plot
            for i, (step, r) in enumerate(zip(steps, cum_roi)):
                if r > 0.001:
                    ax1.annotate(fmt_fn(r), xy=(step, r), xytext=(0, 16),
                                textcoords='offset points', ha='center', va='bottom',
                                fontsize=14, fontweight='bold', color='#0d47a1',
                                rotation=90,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                         alpha=0.9, edgecolor='#1976d2', linewidth=1))

            ax1.axvline(x=steps[best_roi_idx], color='#2e7d32', linestyle=':',
                        linewidth=3, alpha=0.8, label=f'Mejor ROI: Paso {steps[best_roi_idx]}')
            ax1.scatter(steps[best_roi_idx], cum_roi[best_roi_idx], s=400, color='#4caf50',
                        edgecolor='#1b5e20', linewidth=3, zorder=10, marker='*')
                        
            ax1.set_ylabel(f'Reduccion [{unit}/$1M]', fontsize=16, fontweight='bold', labelpad=10)
            ax1.set_title(f'Retorno de Inversion (ROI) ACUMULADO : {label}',
                          fontsize=18, fontweight='bold', pad=15)
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: fmt_fn(x)))
            ax1.tick_params(axis='both', labelsize=14)
            ax1.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, linewidth=1.5)
            ax1.set_axisbelow(True)
            ax1.set_xticks(steps)
            ax1.set_xticklabels([])
            
            ax1.legend(loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black', fancybox=True)
            ax1.set_ylim(0, np.max(cum_roi) * 1.25)
            
            # 2. BOTTOM PLOT
            valid_marg = marg_roi[marg_roi > 0]
            avg_marg = np.mean(valid_marg) if len(valid_marg) > 0 else 0
            median_marg = np.median(valid_marg) if len(valid_marg) > 0 else 0
            
            colors = ['#2e7d32' if mr >= median_marg else '#d32f2f' for mr in marg_roi]
            bars = ax2.bar(steps, marg_roi, color=colors, alpha=0.85,
                           edgecolor='black', linewidth=1.5, width=0.7)
                           
            ax2.axhline(y=avg_marg, color='#ff9800', linestyle='--', linewidth=3,
                        label=f'Promedio: {fmt_fn(avg_marg)}', alpha=0.9)
                        
            for bar, mr in zip(bars, marg_roi):
                if mr > 0.001:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + np.max(marg_roi)*0.02,
                            fmt_fn(mr), ha='center', va='bottom',
                            fontsize=12, fontweight='bold', color='black',
                            rotation=90,
                            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

            ax2.set_xlabel('Numero de Paso (Tanque)', fontsize=16, fontweight='bold', labelpad=10)
            ax2.set_ylabel(f'ROI de Este Paso\n[{unit}/$1M]', fontsize=16, fontweight='bold', labelpad=10)
            ax2.set_title(f'ROI Marginal : {label}', fontsize=18, fontweight='bold', pad=15)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: fmt_fn(x)))
            ax2.tick_params(axis='both', labelsize=14)
            ax2.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, axis='y', linewidth=1)
            ax2.set_axisbelow(True)
            ax2.set_xticks(steps)
            ax2.set_xticklabels([f'p{s}' for s in steps], fontsize=13, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=14, framealpha=0.95, edgecolor='black', fancybox=True)
            
            max_marg = np.max(marg_roi) if np.max(marg_roi) > 0 else 1
            ax2.set_ylim(0, max_marg * 1.25)
            
            plt.tight_layout()
            
            safe_name = col.replace('_', '-')
            save_path = self._get_next_figure_path(f"roi_curve_{safe_name}")
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  [Dashboard] Saved: {save_path}")

    def plot_pareto_curves(self):
        """
        Curvas Pareto: Inversion vs Indicadores Hidraulicos.
        Muestra la evolucion de variables hidraulicas vs inversion acumulada.
        """
        if len(self.df) < 2:
            return
        
        # Variables a graficar: (columna_csv, titulo, unidad, es_inverso)
        # es_inverso=True significa que menor es mejor (para colorear eficiencia)
        variables = [
            ('flooding_volume', 'Volumen de Inundacion', 'm³', True),
            ('flooded_nodes_count', 'Nodos Inundados', 'nodos', True),
            ('outfall_peak_flow', 'Caudal Pico en Outfall', 'm³/s', True),
            ('surcharged_links_count', 'Links Sobre-cargados', 'links', True),
            ('flooding_flow', 'Caudal de Inundacion', 'm³/s', True),
            ('system_mean_utilization', 'Utilizacion Red', '%', True),
        ]
        
        # Calcular eficiencia marginal para colorear puntos
        self.df['_marginal_eff'] = self.df['marginal_reduction'] / (self.df['_marginal_cost'] / 1e6 + 1e-10)
        median_eff = self.df['_marginal_eff'].median()
        
        # Crear figura con subplots 3x2
        n_vars = len(variables)
        n_rows = 3
        n_cols = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))  # Vertical (14 de ancho, 24 de alto)
        axes = axes.flatten()
        
        # Titulo general
        fig.suptitle('Curvas Pareto: Inversion vs Indicadores Hidraulicos\n' + 
                    'Color = eficiencia marginal (Verde = sobre mediana | Rojo = bajo)',
                    fontsize=16, fontweight='bold', y=0.98)
        
        for idx, (col, title, unit, inverse) in enumerate(variables):
            if col not in self.df.columns:
                continue
                
            ax = axes[idx]
            
            # Datos
            x = self.df['cost_investment_total'].values / 1e6  # En millones
            y = self.df[col].values
            steps = self.df['step'].values
            eff = self.df['_marginal_eff'].values
            
            # Colores por eficiencia
            colors = ['#2e7d32' if e >= median_eff else '#c62828' for e in eff]
            
            # Scatter plot
            scatter = ax.scatter(x, y, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=3)
            
            # Linea de tendencia suave
            if len(x) > 2:
                try:
                    from scipy.interpolate import make_interp_spline
                    x_smooth = np.linspace(x.min(), x.max(), 100)
                    spl = make_interp_spline(x, y, k=min(3, len(x)-1))
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, 'k--', alpha=0.3, linewidth=1.5, label='Tendencia')
                except:
                    ax.plot(x, y, 'k--', alpha=0.3, linewidth=1.5)
            
            # Mostrar puntos clave: 0, 6, 12, 18, 24 (o el último disponible) con sus VALORES NUMÉRICOS
            def find_nearest_idx(target_step):
                distances = np.abs(steps - target_step)
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] <= 2:
                    return min_dist_idx
                return None
            
            # Función para formatear valores según su magnitud
            def format_val(val, unit):
                if abs(val) >= 1000:
                    return f'{val/1000:.1f}k'
                elif abs(val) >= 1:
                    return f'{val:.0f}' if val == int(val) else f'{val:.1f}'
                else:
                    return f'{val:.2f}'
            
            # Punto 0 (inicio) - grande con valor numérico
            ax.scatter(x[0], y[0], s=250, c='darkblue', edgecolors='white', linewidth=2.5, zorder=5, marker='o')
            val_0 = format_val(y[0], unit)
            ax.annotate(f'p0: {val_0}', (x[0], y[0]), textcoords="offset points", 
                       xytext=(-30, 25), ha='center', fontsize=14, color='darkblue', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7, lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.95, edgecolor='darkblue', linewidth=1.5))
            
            # Puntos intermedios con valores numéricos
            intermediate_steps = [6, 12, 18, 24]
            colors_intermediate = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2']  # Azul, verde, naranja, morado
            for i, step_num in enumerate(intermediate_steps):
                idx = find_nearest_idx(step_num)
                if idx is not None and idx != 0 and idx != len(steps)-1:
                    color = colors_intermediate[i % len(colors_intermediate)]
                    ax.scatter(x[idx], y[idx], s=150, c=color, edgecolors='white', linewidth=2, zorder=4)
                    val = format_val(y[idx], unit)
                    # Offset alternado para evitar solapamiento
                    offset_x = 25 if i % 2 == 0 else -25
                    offset_y = 22 if i < 2 else -22
                    ax.annotate(f'p{int(steps[idx])}:\n{val}', (x[idx], y[idx]), textcoords="offset points", 
                               xytext=(offset_x, offset_y), ha='center', fontsize=13, color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))
            
            # Punto final (último) - grande con valor numérico
            ax.scatter(x[-1], y[-1], s=250, c='darkgreen', edgecolors='white', linewidth=2.5, zorder=5, marker='o')
            val_f = format_val(y[-1], unit)
            ax.annotate(f'p{int(steps[-1])}: {val_f}', (x[-1], y[-1]), textcoords="offset points", 
                       xytext=(35, -25), ha='center', fontsize=13, color='darkgreen', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7, lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.95, edgecolor='darkgreen', linewidth=1.5))
            
            # Linea de mediana del valor
            median_y = np.median(y)
            ax.axhline(y=median_y, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
            ax.text(x.max()*0.98, median_y, f'Med: {median_y:.0f}', 
                   ha='right', va='bottom', fontsize=13, color='blue', alpha=0.7)
            
            # Eje Y secundario con porcentaje de cambio
            ax2 = ax.twinx()
            baseline_val = y[0] if len(y) > 0 and y[0] != 0 else 1
            # Calcular limites de porcentaje basados en los valores actuales
            y_min, y_max = ax.get_ylim()
            pct_min = ((y_min - baseline_val) / baseline_val) * 100
            pct_max = ((y_max - baseline_val) / baseline_val) * 100
            ax2.set_ylim(pct_min, pct_max)
            ax2.set_ylabel('% Cambio', fontsize=13, color='gray', alpha=0.8)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=10)
            
            # Configuracion del subplot principal
            ax.set_xlabel('Inversion (M$)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{title}\n({unit})', fontsize=12, fontweight='bold')
            ax.set_title(f'{title}', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=11)
            ax.grid(True, alpha=0.3)
            
            # Leyenda de eficiencia
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2e7d32', label=f'Eff. >= mediana'),
                Patch(facecolor='#c62828', label=f'Eff. < mediana'),
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        # Ocultar subplot vacio si hay
        if n_vars < len(axes):
            for idx in range(n_vars, len(axes)):
                axes[idx].axis('off')
        
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
        
        save_path = self._get_next_figure_path("pareto_curves")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_system_utilization_by_tank(self):
        """
        Gráfico de utilización del sistema por tanque.
        Similar a outfall_reduction pero para system_mean_utilization.
        """
        if len(self.df) < 1 or 'system_mean_utilization' not in self.df.columns:
            return
        
        # Lógica de agrupación por tanque
        n_tanks = self.df['n_tanks'] if 'n_tanks' in self.df.columns else self.df['step']
        is_new_tank = n_tanks.diff().fillna(1) > 0
        tank_id = is_new_tank.cumsum()
        
        # Calcular cambio marginal en utilización (anterior - actual, positivo = mejora)
        # Esto invierte el signo: reducción de utilización = valor positivo
        util_diff = self.df['system_mean_utilization'].diff().fillna(0)
        self.df['_marginal_utilization'] = -util_diff  # Negamos para que mejora sea positiva
        self.df['_marginal_cost'] = self.df['cost_investment_total'].diff().fillna(self.df['cost_investment_total'].iloc[0])
        
        agg_dict = {
            '_marginal_cost': 'sum',
            '_marginal_utilization': 'sum',
            'marginal_reduction': 'sum',
            'current_tank_volume': 'first',
            'added_predio': 'first',
            'step': 'first',
        }
        if 'final_tank_volume' in self.df.columns:
            agg_dict['final_tank_volume'] = 'first'
        
        tank_groups = self.df.groupby(tank_id).agg(agg_dict).reset_index(drop=True)
        
        # Filtrar tanques válidos (costo > 0)
        valid_tank_mask = tank_groups['_marginal_cost'] > 0
        tank_groups = tank_groups[valid_tank_mask].reset_index(drop=True)
        
        tank_cost = tank_groups['_marginal_cost']
        utilization_change = tank_groups['_marginal_utilization']
        
        vol_col = 'final_tank_volume' if 'final_tank_volume' in tank_groups.columns else 'current_tank_volume'
        tank_volume = tank_groups[vol_col].fillna(1000)
        tank_predio = tank_groups['added_predio']
        
        # Obtener utilización inicial para calcular porcentaje
        initial_util = self.df['system_mean_utilization'].iloc[0] if len(self.df) > 0 else 1
        
        # Figura
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.30, 0.70], height_ratios=[1, 1], 
                              wspace=0.15, hspace=0.30)
        ax_table = fig.add_subplot(gs[:, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_evo = fig.add_subplot(gs[1, 1])
        
        # === TABLA ===
        ax_table.axis('off')
        
        table_rows = []
        for i in range(len(tank_groups)):
            table_rows.append({
                'original_idx': i,
                'tank_label': f"T{i+1}",
                'predio': f"P{tank_predio.iloc[i]}",
                'util_change': utilization_change.iloc[i],
                'volumen': tank_volume.iloc[i],
            })
        
        # Ordenar por cambio en utilización (mayor mejora primero)
        table_rows_sorted = sorted(table_rows, key=lambda x: x['util_change'], reverse=True)
        
        # Calcular valores acumulados en el orden original (por orden de instalación)
        cumulative_by_original_idx = {}
        cumsum = 0
        for i in range(len(tank_groups)):
            cumsum += utilization_change.iloc[i]
            cumulative_by_original_idx[i] = cumsum
        
        table_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            orig_idx = row['original_idx']
            util_val = row['util_change']  # Ya está en pp (0-100)
            util_str = f"{util_val:+.2f}%" if util_val != 0 else "0.00%"
            # Valor acumulado en %
            cumul_val = cumulative_by_original_idx[orig_idx]
            cumul_str = f"{cumul_val:.2f}%"
            table_data.append([
                str(rank),
                row['tank_label'],
                row['predio'],
                util_str,
                cumul_str,
                f"{row['volumen']:,.0f}",
            ])
        
        col_labels = ['Rank', 'Tank', 'Predio', 'Marginal\n(%)', 'Acumulado\n(%)', 'Vol.\n(m³)']
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colColours=['#6a1b9a'] * 6,  # Púrpura para diferenciar
            bbox=[0.0, 0.05, 1.0, 0.85]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(15)
        table.scale(1.2, 2.2)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', color='white', fontsize=12)
                cell.set_facecolor('#6a1b9a')
                cell.set_height(0.08)
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_height(0.08)
            cell.set_edgecolor('#aaaaaa')
            cell.set_linewidth(1.5)
        
        ax_table.set_title('Ranking por Cambio en\nUtilización del Sistema\n(Mejor → Peor)',
                          fontsize=14, fontweight='bold', pad=10)
        
        # === BARRAS ===
        tank_labels = [f"{i+1}°" for i in range(len(tank_groups))]
        
        colors = []
        for val in utilization_change:
            if val > 0:
                colors.append('#2e7d32')  # Verde - mejora
            elif val < 0:
                colors.append('#c62828')  # Rojo - empeora
            else:
                colors.append('#9e9e9e')  # Gris - sin cambio
        
        x_pos = np.arange(1, len(tank_labels) + 1)  # Alineado con tank_numbers (1-12)
        bars = ax_bar.bar(x_pos, utilization_change, color=colors, edgecolor='black', linewidth=1.5)
        
        # Labels en barras
        util_max = utilization_change.max()
        util_min = utilization_change.min()
        y_range = util_max - util_min if util_max != util_min else 1
        
        for bar, val in zip(bars, utilization_change):  # Ya está en pp
            label = f"{val:+.2f}%" if val != 0 else "0.00%"
            if val >= 0:
                y_pos = bar.get_height() + y_range * 0.02
                va = 'bottom'
            else:
                y_pos = bar.get_height() - y_range * 0.02
                va = 'top'
            ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos,
                   label, ha='center', va=va, fontsize=14, fontweight='bold',
                   color='black', rotation=0)
        
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([])  # Sin etiquetas - comparte eje con evo
        ax_bar.set_xlabel('')  # Sin label - comparte eje con evo
        ax_bar.set_xlim(0.5, len(tank_labels) + 0.5)  # Alineado con evo
        ax_bar.set_ylabel('Reducción de Utilización (%)', fontsize=15, fontweight='bold')
        ax_bar.set_title(f'Reducción de Utilización del Sistema por Tanque ({len(tank_groups)} Tanques)',
                     fontsize=16, fontweight='bold', pad=35)
        
        ax_bar.grid(True, linestyle='--', alpha=0.4, color='#cccccc', axis='y', linewidth=1)
        ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis='y', labelsize=14)
        
        # === EVOLUCIÓN ===
        tank_numbers = np.arange(1, len(tank_groups) + 1)
        cumulative_util = np.cumsum(utilization_change.values)
        
        ax_evo.plot(tank_numbers, cumulative_util, 'o-', color='#6a1b9a', linewidth=3,
                   markersize=12, markeredgecolor='#4a148c', markeredgewidth=2,
                   label='Cambio Acumulado Utilización')
        
        ax_evo.fill_between(tank_numbers, 0, cumulative_util, alpha=0.2, color='#ce93d8')
        
        for i, (tank_num, val) in enumerate(zip(tank_numbers, cumulative_util)):
            ax_evo.annotate(f'{val:.2f}%', xy=(tank_num, val), xytext=(0, 12),
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=14, fontweight='bold', color='#4a148c',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.9, edgecolor='none'))
        
        ax_evo.set_xticks(tank_numbers)
        ax_evo.set_xticklabels(tank_labels, fontsize=14, fontweight='bold')
        ax_evo.set_xlabel('Tanque (Orden de Instalación)', fontsize=15, fontweight='bold')
        ax_evo.set_xlim(0.5, len(tank_groups) + 0.5)  # Alineado con bar
        ax_evo.set_ylabel('Cambio Acumulado (%)', fontsize=15, fontweight='bold')
        ax_evo.set_title('Evolución del Cambio Acumulado en Utilización del Sistema',
                        fontsize=16, fontweight='bold', pad=10)
        
        ax_evo.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=1)
        ax_evo.set_axisbelow(True)
        ax_evo.tick_params(axis='y', labelsize=14)
        ax_evo.legend(loc='upper left', fontsize=14, framealpha=0.95)
        
        # Export CSV
        csv_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            orig_idx = row['original_idx']
            csv_data.append({
                'Rank': rank,
                'Tanque': row['tank_label'],
                'Predio': tank_predio.iloc[orig_idx],
                'Utilizacion_Change': utilization_change.iloc[orig_idx],
                'Volumen_Tanque_m3': tank_volume.iloc[orig_idx],
            })
        
        df_table = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "system_utilization_ranking.csv"
        df_table.to_csv(csv_path, index=False)
        print(f"  [Dashboard] Saved: {csv_path}")
        
        plt.tight_layout()
        
        save_path = self._get_next_figure_path("system_utilization_by_tank")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_flooding_flow_by_tank(self):
        """
        Gráfico de flooding flow por tanque.
        Similar a outfall_reduction pero para flooding_flow.
        """
        if len(self.df) < 1 or 'flooding_flow' not in self.df.columns:
            return
        
        # Lógica de agrupación por tanque
        n_tanks = self.df['n_tanks'] if 'n_tanks' in self.df.columns else self.df['step']
        is_new_tank = n_tanks.diff().fillna(1) > 0
        tank_id = is_new_tank.cumsum()
        
        # Calcular reducción marginal en flooding flow (anterior - actual)
        # diff() hace actual - anterior, así que multiplicamos por -1 para que la reducción sea positiva
        self.df['_marginal_flooding_flow'] = self.df['flooding_flow'].diff().fillna(0) * -1
        self.df['_marginal_cost'] = self.df['cost_investment_total'].diff().fillna(self.df['cost_investment_total'].iloc[0])
        
        agg_dict = {
            '_marginal_cost': 'sum',
            '_marginal_flooding_flow': 'sum',
            'marginal_reduction': 'sum',
            'current_tank_volume': 'first',
            'added_predio': 'first',
            'step': 'first',
        }
        if 'final_tank_volume' in self.df.columns:
            agg_dict['final_tank_volume'] = 'first'
        
        tank_groups = self.df.groupby(tank_id).agg(agg_dict).reset_index(drop=True)
        
        # Filtrar tanques válidos (costo > 0)
        valid_tank_mask = tank_groups['_marginal_cost'] > 0
        tank_groups = tank_groups[valid_tank_mask].reset_index(drop=True)
        
        tank_cost = tank_groups['_marginal_cost']
        flooding_flow_change = tank_groups['_marginal_flooding_flow']
        
        vol_col = 'final_tank_volume' if 'final_tank_volume' in tank_groups.columns else 'current_tank_volume'
        tank_volume = tank_groups[vol_col].fillna(1000)
        tank_predio = tank_groups['added_predio']
        
        # Obtener flooding flow inicial para calcular porcentaje
        initial_flood = self.df['flooding_flow'].iloc[0] if len(self.df) > 0 else 1
        
        # Figura
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(2, 2, width_ratios=[0.30, 0.70], height_ratios=[1, 1], 
                              wspace=0.15, hspace=0.30)
        ax_table = fig.add_subplot(gs[:, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_evo = fig.add_subplot(gs[1, 1])
        
        # === TABLA ===
        ax_table.axis('off')
        
        table_rows = []
        for i in range(len(tank_groups)):
            table_rows.append({
                'original_idx': i,
                'tank_label': f"T{i+1}",
                'predio': f"P{tank_predio.iloc[i]}",
                'flood_change': flooding_flow_change.iloc[i],
                'volumen': tank_volume.iloc[i],
            })
        
        # Ordenar por reducción de flooding flow (mayor reducción primero)
        table_rows_sorted = sorted(table_rows, key=lambda x: x['flood_change'], reverse=True)
        
        table_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            flood_val = row['flood_change']
            flood_str = f"{flood_val:+.2f}" if flood_val != 0 else "0.00"
            # Porcentaje de cambio respecto a flooding inicial
            pct_val = (flood_val / initial_flood) * 100 if initial_flood != 0 else 0
            pct_str = f"{pct_val:+.1f}%"
            table_data.append([
                str(rank),
                row['tank_label'],
                row['predio'],
                flood_str,
                pct_str,
                f"{row['volumen']:,.0f}",
            ])
        
        col_labels = ['Rank', 'Tank', 'Predio', 'Flood Flow\n(m³/s)', 'Reduc.\n(%)', 'Vol.\n(m³)']
        
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colColours=['#d84315'] * 6,  # Naranja/rojo para diferenciar
            bbox=[0.0, 0.05, 1.0, 0.85]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(15)
        table.scale(1.2, 2.2)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(fontweight='bold', color='white', fontsize=12)
                cell.set_facecolor('#d84315')
                cell.set_height(0.08)
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_height(0.08)
            cell.set_edgecolor('#aaaaaa')
            cell.set_linewidth(1.5)
        
        ax_table.set_title('Ranking por Reducción de\nFlooding Flow\n(Mejor → Peor)',
                          fontsize=14, fontweight='bold', pad=10)
        
        # === BARRAS ===
        tank_labels = [f"{i+1}°" for i in range(len(tank_groups))]
        
        colors = []
        for val in flooding_flow_change:
            if val > 0:
                colors.append('#2e7d32')  # Verde - reduce flooding
            elif val < 0:
                colors.append('#c62828')  # Rojo - aumenta flooding
            else:
                colors.append('#9e9e9e')  # Gris - sin cambio
        
        x_pos = np.arange(1, len(tank_labels) + 1)  # Alineado con tank_numbers (1-12)
        bars = ax_bar.bar(x_pos, flooding_flow_change, color=colors, edgecolor='black', linewidth=1.5)
        
        # Labels en barras
        flood_max = flooding_flow_change.max()
        flood_min = flooding_flow_change.min()
        y_range = flood_max - flood_min if flood_max != flood_min else 1
        
        for bar, val in zip(bars, flooding_flow_change):
            label = f"{val:+.2f}" if val != 0 else "0.00"
            if val >= 0:
                y_pos = bar.get_height() + y_range * 0.02
                va = 'bottom'
            else:
                y_pos = bar.get_height() - y_range * 0.02
                va = 'top'
            ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos,
                   label, ha='center', va=va, fontsize=14, fontweight='bold',
                   color='black', rotation=0)
        
        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([])  # Sin etiquetas en X (comparte con evo)
        ax_bar.set_xlabel('')
        ax_bar.set_xlim(0.5, len(tank_groups) + 0.5)  # Alineado con evo
        ax_bar.set_ylabel('Reducción Flooding Flow (m³/s)', fontsize=15, fontweight='bold')
        ax_bar.set_title(f'Efecto en Flooding Flow por Tanque ({len(tank_groups)} Tanques)',
                     fontsize=16, fontweight='bold', pad=35)
        
        ax_bar.grid(True, linestyle='--', alpha=0.4, color='#cccccc', axis='y', linewidth=1)
        ax_bar.set_axisbelow(True)
        ax_bar.tick_params(axis='y', labelsize=14)
        
        # === EVOLUCIÓN ===
        tank_numbers = np.arange(1, len(tank_groups) + 1)
        cumulative_flood = np.cumsum(flooding_flow_change.values)
        
        ax_evo.plot(tank_numbers, cumulative_flood, 'o-', color='#d84315', linewidth=3,
                   markersize=12, markeredgecolor='#bf360c', markeredgewidth=2,
                   label='Reducción Acumulada Flooding Flow')
        
        ax_evo.fill_between(tank_numbers, 0, cumulative_flood, alpha=0.2, color='#ffab91')
        
        for i, (tank_num, val) in enumerate(zip(tank_numbers, cumulative_flood)):
            ax_evo.annotate(f'{val:.1f}', xy=(tank_num, val), xytext=(0, 12),
                           textcoords='offset points', ha='center', va='bottom',
                           fontsize=14, fontweight='bold', color='#bf360c',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    alpha=0.9, edgecolor='none'))
        
        ax_evo.set_xticks(tank_numbers)
        ax_evo.set_xticklabels(tank_labels, fontsize=14, fontweight='bold')
        ax_evo.set_xlabel('Tanque (Orden de Instalación)', fontsize=15, fontweight='bold')
        ax_evo.set_xlim(0.5, len(tank_groups) + 0.5)  # Alineado con bar
        ax_evo.set_ylabel('Reducción Acumulada (m³/s)', fontsize=15, fontweight='bold')
        ax_evo.set_title('Evolución de la Reducción Acumulada de Flooding Flow',
                        fontsize=16, fontweight='bold', pad=10)
        
        ax_evo.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=1)
        ax_evo.set_axisbelow(True)
        ax_evo.tick_params(axis='y', labelsize=14)
        ax_evo.legend(loc='upper left', fontsize=14, framealpha=0.95)
        
        # Export CSV
        csv_data = []
        for rank, row in enumerate(table_rows_sorted, 1):
            orig_idx = row['original_idx']
            csv_data.append({
                'Rank': rank,
                'Tanque': row['tank_label'],
                'Predio': tank_predio.iloc[orig_idx],
                'Flooding_Flow_Change_m3s': flooding_flow_change.iloc[orig_idx],
                'Volumen_Tanque_m3': tank_volume.iloc[orig_idx],
            })
        
        df_table = pd.DataFrame(csv_data)
        csv_path = self.output_dir / "flooding_flow_ranking.csv"
        df_table.to_csv(csv_path, index=False)
        print(f"  [Dashboard] Saved: {csv_path}")
        
        plt.tight_layout()
        
        save_path = self._get_next_figure_path("flooding_flow_by_tank")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_system_evolution(self, variables_config=None):
        """
        Gráfica MARIPOSA con estilo profesional symlog.
        
        Args:
            variables_config: Lista de dicts con keys:
                - col: nombre de columna en CSV
                - label: texto a mostrar
                - grupo: 'infraestructura' | 'costos' | 'mejoras'
                - valor_fn: callable(b, f) -> float
                - display_fn: callable(v) -> str
                - positivo: bool (True = mayor es mejor)
        """
        if len(self.df) < 1:
            return
        
        baseline_row = self.df[self.df['step'] == 0]
        if len(baseline_row) == 0:
            return
        
        b = baseline_row.iloc[0]
        f = self.df.iloc[-1]
        
        # Paleta de colores
        C  = {'mejoras': '#15803D', 'costos': '#C2410C', 'infraestructura': '#1D4ED8'}
        BG = {'mejoras': '#F0FDF4', 'costos': '#FFF7ED', 'infraestructura': '#EFF6FF'}
        GRUPO_ORDER = {'infraestructura': 0, 'costos': 1, 'mejoras': 2}
        
        # Configuración por defecto - nombres EXACTOS del CSV
        if variables_config is None:
            variables_config = [
                # INFRAESTRUCTURA
                dict(col='n_tanks', label='n_tanks', grupo='infraestructura',
                     valor_fn=lambda b, f: float(f['n_tanks']),
                     display_fn=lambda v: f'{v:.0f}', positivo=True),
                dict(col='total_tank_volume', label='total_tank_volume', grupo='infraestructura',
                     valor_fn=lambda b, f: float(f['total_tank_volume']) / 1000,
                     display_fn=lambda v: f'{v:.0f}K m3', positivo=True),
                dict(col='derivation_links_length', label='derivation_links_length', grupo='infraestructura',
                     valor_fn=lambda b, f: float(f['derivation_links_length']),
                     display_fn=lambda v: f'{v:.0f} m', positivo=True),
                # COSTOS (negativo = barra a la izquierda)
                dict(col='cost_investment_total', label='cost_investment_total', grupo='costos',
                     valor_fn=lambda b, f: -float(f['cost_investment_total']) / 1e6,
                     display_fn=lambda v: f'${abs(v):.1f}M', positivo=False),
                dict(col='cost_tanks', label='cost_tanks', grupo='costos',
                     valor_fn=lambda b, f: -float(f['cost_tanks']) / 1e6,
                     display_fn=lambda v: f'${abs(v):.1f}M', positivo=False),
                dict(col='cost_links', label='cost_links', grupo='costos',
                     valor_fn=lambda b, f: -float(f['cost_links']) / 1e6,
                     display_fn=lambda v: f'${abs(v):.1f}M', positivo=False),
                dict(col='cost_land', label='cost_land', grupo='costos',
                     valor_fn=lambda b, f: -float(f['cost_land']) / 1e6,
                     display_fn=lambda v: f'${abs(v):.1f}M', positivo=False),
                # MEJORAS (cambio % respecto al baseline)
                dict(col='flooding_volume', label='flooding_volume', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['flooding_volume']) - float(f['flooding_volume'])) / float(b['flooding_volume']) * 100 if float(b['flooding_volume']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
                dict(col='flooded_nodes_count', label='flooded_nodes_count', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['flooded_nodes_count']) - float(f['flooded_nodes_count'])) / float(b['flooded_nodes_count']) * 100 if float(b['flooded_nodes_count']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
                dict(col='outfall_peak_flow', label='outfall_peak_flow', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['outfall_peak_flow']) - float(f['outfall_peak_flow'])) / float(b['outfall_peak_flow']) * 100 if float(b['outfall_peak_flow']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
                dict(col='surcharged_links_count', label='surcharged_links_count', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['surcharged_links_count']) - float(f['surcharged_links_count'])) / float(b['surcharged_links_count']) * 100 if float(b['surcharged_links_count']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
                dict(col='system_mean_utilization', label='system_mean_utilization', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['system_mean_utilization']) - float(f['system_mean_utilization'])) / float(b['system_mean_utilization']) * 100 if float(b['system_mean_utilization']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
                dict(col='flooding_flow', label='flooding_flow', grupo='mejoras',
                     valor_fn=lambda b, f: (float(b['flooding_flow']) - float(f['flooding_flow'])) / float(b['flooding_flow']) * 100 if float(b['flooding_flow']) > 0 else 0,
                     display_fn=lambda v: f'-{abs(v):.1f}%', positivo=True),
            ]
        
        # Ordenar por grupo
        variables_config = sorted(variables_config, key=lambda d: GRUPO_ORDER.get(d['grupo'], 9))
        
        # Resolver items validos
        items = []
        for cfg in variables_config:
            col = cfg['col']
            if col not in f.index or col not in b.index:
                continue
            try:
                val = cfg['valor_fn'](b, f)
                if not np.isfinite(val):
                    continue
            except Exception:
                continue
            items.append((cfg['label'], val, cfg['grupo'], cfg['display_fn'], cfg['positivo']))
        
        if not items:
            return
        
        # Orden visual: primero en la lista = arriba en la grafica
        items_rev = list(reversed(items))
        n = len(items_rev)
        
        # Figura - TAMAÑO A4 VERTICAL (210mm x 297mm = 8.27 x 11.69 pulgadas)
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=150)
        ax.set_facecolor('white')
        
        # Bandas de grupo
        group_spans = {}
        for yi, (label, val, grupo, dfn, pos) in enumerate(items_rev):
            if grupo not in group_spans:
                group_spans[grupo] = [yi, yi]
            else:
                group_spans[grupo][0] = min(group_spans[grupo][0], yi)
                group_spans[grupo][1] = max(group_spans[grupo][1], yi)
        
        for grupo, (y0, y1) in group_spans.items():
            ax.axhspan(y0 - 0.48, y1 + 0.48, color=BG[grupo], alpha=1.0, zorder=0)
        
        # Escala symlog
        ax.set_xscale('symlog', linthresh=1, base=10)
        ax.axvline(0, color='#94A3B8', lw=1.5, alpha=0.7, zorder=2)
        
        # Barras y etiquetas
        for yi, (label, val, grupo, display_fn, positivo) in enumerate(items_rev):
            color = C[grupo]
            ax.barh(yi, val, color=color, height=0.56, alpha=0.85,
                    edgecolor='white', linewidth=0.8, zorder=3)
            
            disp = display_fn(val)
            abs_val = abs(val)
            
            # Texto centrado visualmente en la barra (considerando escala symlog)
            # Calcular centro visual transformando a coordenadas de pantalla y volviendo
            try:
                # Convertir extremos de la barra a coordenadas de pantalla
                x_ends = ax.transData.transform([[0, yi], [val, yi]])[:, 0]
                x_mid_screen = (x_ends[0] + x_ends[1]) / 2
                # Convertir punto medio de vuelta a coordenadas de datos
                tx = ax.transData.inverted().transform([x_mid_screen, 0])[0]
            except:
                tx = val / 2  # Fallback
            
            ha = 'center'
            col_txt = '#1E293B'
            
            # Fondo blanco solido
            bbox_props = dict(boxstyle='round,pad=0.25', facecolor='white', 
                             alpha=1.0, edgecolor='none')
            
            ax.text(tx, yi, disp, va='center', ha=ha, fontsize=12,
                   fontweight='bold', color=col_txt, zorder=5,
                   bbox=bbox_props)
        
        # Etiquetas del eje Y
        tick_col = {g: C[g] for g in C}
        group_prefix = {'infraestructura': '(i)', 'costos': '(-)', 'mejoras': '(+)'}
        
        ax.set_yticks(range(n))
        ylabels = [f"{group_prefix[g]}  {lbl}" for lbl, _, g, _, _ in items_rev]
        ax.set_yticklabels(ylabels, fontsize=13)
        for tick, (_, _, grupo, _, _) in zip(ax.get_yticklabels(), items_rev):
            tick.set_color(tick_col[grupo])
        
        # Eje X - ajustar límite dinámicamente según valores máximos
        max_val = max([abs(v) for _, v, _, _, _ in items_rev] + [100])
        max_infra = max([v for _, v, g, _, _ in items_rev if g == 'infraestructura'] + [100])
        max_costo = max([abs(v) for _, v, g, _, _ in items_rev if g == 'costos'] + [100])
        
        # Calcular límites con margen
        lim_der = max(max_val * 1.3, max_infra * 1.2, 300)
        lim_izq = -max(max_costo * 1.3, 130)
        
        # Ajustar ticks según el rango
        if lim_der > 1000:
            xticks = [-100, -25, -5, 0, 5, 25, 100, 300, 1000]
            xticklabels = ['-100', '-25', '-5', '0', '+5', '+25', '+100', '+300', '+1000']
        elif lim_der > 500:
            xticks = [-100, -25, -5, 0, 5, 25, 100, 300, 500]
            xticklabels = ['-100', '-25', '-5', '0', '+5', '+25', '+100', '+300', '+500']
        else:
            xticks = [-100, -25, -5, 0, 5, 25, 100, 300]
            xticklabels = ['-100', '-25', '-5', '0', '+5', '+25', '+100', '+300']
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=11, color='#6B7280')
        ax.set_xlim(lim_izq, lim_der)
        ax.set_xlabel('Cambio respecto al Baseline (escala log simetrica)',
                     fontsize=12, color='#6B7280', labelpad=10)
        
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color('#E2E8F0')
        
        # Leyenda
        legend_handles = [
            mpatches.Patch(color=C['infraestructura'], alpha=0.85, label='(i) Infraestructura instalada'),
            mpatches.Patch(color=C['costos'], alpha=0.85, label='(-) Costo / carga'),
            mpatches.Patch(color=C['mejoras'], alpha=0.85, label='(+) Mejora hidraulica'),
        ]
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99),
                 fontsize=11, framealpha=0.95, edgecolor='#CBD5E1', borderpad=0.6)
        
        # Titulo
        ax.set_title('Impacto Global de la Optimizacion\nBaseline  →  Solucion Final',
                    fontsize=16, fontweight='bold', pad=18, color='#1E293B')
        
        save_path = self._get_next_figure_path("system_evolution")
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")

    def plot_group_cards_map_from_networks_and_swmm(
        self,
        networks_gpkg_path: str,
        swmm_inp_path: str,
        sequence_csv_path: str,
        basemap_alpha: float = 0.5,
        output_name: str = "group_cards_network_map"
    ):
        """
        Genera figura con panel izquierdo (fichas de tanques) y panel derecho (mapa de red).
        
        Panel izquierdo: fichas apiladas T1, T2, T3... ordenadas por costo descendente
        Panel derecho: mapa con red en gris, grupos coloreados, labels T*, globos azules con ramal
        
        Args:
            networks_gpkg_path: Ruta al GPKG de redes (con columna Obs en JSON)
            swmm_inp_path: Ruta al modelo SWMM .inp
            sequence_csv_path: Ruta al CSV con costos y secuencia
            basemap_alpha: Opacidad del basemap satelital (default 0.5)
            output_name: Nombre base para la figura de salida
        """
        import swmmio
        
        # ============================================================
        # 1. CARGAR DATOS
        # ============================================================
        
        # 1.1 Cargar GPKG de redes
        networks_gdf = gpd.read_file(networks_gpkg_path)
        
        # Verificar columnas requeridas
        required_cols = ['Obs', 'Ramal']
        for col in required_cols:
            if col not in networks_gdf.columns:
                raise KeyError(f"Columna requerida '{col}' no encontrada en GPKG de redes")
        
        # 1.2 Parsear columna Obs (JSON) y extraer datos del tanque
        def parse_obs_json(obs_str):
            """Parsea el JSON de la columna Obs."""
            if pd.isna(obs_str):
                return {}
            try:
                return json.loads(obs_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON inválido en columna Obs: {obs_str[:100]}...") from e
        
        networks_gdf['obs_parsed'] = networks_gdf['Obs'].apply(parse_obs_json)
        
        # Extraer campos del JSON
        networks_gdf['target_id'] = networks_gdf['obs_parsed'].apply(lambda x: x.get('target_id'))
        networks_gdf['target_x'] = networks_gdf['obs_parsed'].apply(lambda x: x.get('target_x'))
        networks_gdf['target_y'] = networks_gdf['obs_parsed'].apply(lambda x: x.get('target_y'))
        networks_gdf['target_total_volume'] = networks_gdf['obs_parsed'].apply(lambda x: x.get('target_total_volume', 0))
        networks_gdf['target_ramal'] = networks_gdf['obs_parsed'].apply(lambda x: x.get('target_ramal'))
        
        # Filtrar solo las tuberías que tienen target_id (pertenecen a un grupo/tanque)
        networks_with_target = networks_gdf[networks_gdf['target_id'].notna()].copy()
        
        if networks_with_target.empty:
            raise ValueError("No se encontraron tuberías con target_id válido en el GPKG")
        
        # 1.3 Cargar CSV de secuencia
        sequence_df = pd.read_csv(sequence_csv_path)
        
        # Limpiar columnas de costo (quitar $ y comas)
        currency_cols = ['current_tank_cost', 'current_tank_land', 'cost_links']
        for col in currency_cols:
            if col in sequence_df.columns:
                sequence_df[col] = sequence_df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
                sequence_df[col] = pd.to_numeric(sequence_df[col], errors='coerce').fillna(0)
        
        # 1.4 Cargar modelo SWMM para obtener volúmenes por ramal
        swmm_model = swmmio.Model(swmm_inp_path)
        
        # ============================================================
        # 2. CALCULAR MÉTRICAS POR GRUPO/TANQUE
        # ============================================================
        
        # Agrupar por target_id
        group_data = []
        
        for target_id, group_gdf in networks_with_target.groupby('target_id'):
            # Datos del tanque del JSON
            tank_x = group_gdf['target_x'].iloc[0]
            tank_y = group_gdf['target_y'].iloc[0]
            tank_volume = group_gdf['target_total_volume'].iloc[0]
            
            # Buscar datos del tanque en el CSV (por added_node o similar)
            # El target_id suele ser el node_id (ej: P0071343)
            tank_row = sequence_df[sequence_df['added_node'] == target_id]
            
            if tank_row.empty:
                # Intentar buscar por predio si no encuentra por nodo
                tank_row = sequence_df[sequence_df['added_predio'].astype(str) == str(target_id)]
            
            if tank_row.empty:
                # Si no se encuentra, usar valores por defecto (0)
                tank_cost_value = 0
                tank_land_cost = 0
            else:
                # Tomar el último valor (iteración final)
                tank_row = tank_row.iloc[-1]
                tank_cost_value = tank_row.get('current_tank_cost', 0)
                tank_land_cost = tank_row.get('current_tank_land', 0)
            
            # Calcular métricas por ramal dentro de este grupo
            ramal_data = []
            for ramal_name, ramal_gdf in group_gdf.groupby('Ramal'):
                # Longitud total del ramal
                total_length = ramal_gdf.geometry.length.sum()
                
                # Diámetro máximo
                if 'Diameter' in ramal_gdf.columns:
                    dmax = ramal_gdf['Diameter'].max()
                elif 'Geom1' in ramal_gdf.columns:
                    dmax = ramal_gdf['Geom1'].max()
                else:
                    dmax = 0
                
                # Costo de derivación para este ramal (del CSV)
                # Buscar en el CSV el costo_links correspondiente
                deriv_cost = 0  # Placeholder - se mejorará con SWMM
                
                # Volumen aportado (placeholder - bloque separado para mejorar después)
                volume_aportado = 0
                
                ramal_data.append({
                    'ramal': ramal_name,
                    'costo': deriv_cost,
                    'volumen': volume_aportado,
                    'longitud': total_length,
                    'dmax': dmax,
                    'geometry': ramal_gdf.geometry.unary_union
                })
            
            # Calcular totales del grupo
            total_cost = tank_cost_value + tank_land_cost + sum([r['costo'] for r in ramal_data])
            total_volume = tank_volume + sum([r['volumen'] for r in ramal_data])
            total_length = sum([r['longitud'] for r in ramal_data])
            
            group_data.append({
                'target_id': target_id,
                'tank_x': tank_x,
                'tank_y': tank_y,
                'tank_cost': tank_cost_value,
                'tank_land_cost': tank_land_cost,
                'tank_volume': tank_volume,
                'total_cost': total_cost,
                'total_volume': total_volume,
                'total_length': total_length,
                'ramales': ramal_data,
                'geometry': group_gdf.geometry.unary_union
            })
        
        # ============================================================
        # 3. ORDENAR GRUPOS POR COSTO TOTAL (DESCENDENTE)
        # ============================================================
        
        group_data.sort(key=lambda x: x['total_cost'], reverse=True)
        
        # Asignar nombres T1, T2, T3...
        colors = plt.cm.Set3(np.linspace(0, 1, len(group_data)))
        for i, group in enumerate(group_data):
            group['tank_label'] = f"T{i+1}"
            group['color'] = colors[i]
        
        # ============================================================
        # 4. CREAR FIGURA CON GridSpec
        # ============================================================
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.35, 0.65], wspace=0.1)
        
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        
        # ============================================================
        # 5. PANEL IZQUIERDO: FICHAS DE TANQUES
        # ============================================================
        
        ax_left.set_facecolor('white')
        ax_left.axis('off')
        
        # Calcular posiciones verticales para las fichas
        n_groups = len(group_data)
        card_height = 0.85 / n_groups
        card_spacing = 0.02
        
        for i, group in enumerate(group_data):
            y_top = 0.95 - i * (card_height + card_spacing)
            y_bottom = y_top - card_height
            
            # Dibujar rectángulo de la ficha
            rect = mpatches.FancyBboxPatch(
                (0.02, y_bottom), 0.96, card_height,
                boxstyle="round,pad=0.01",
                facecolor='#F8F9FA',
                edgecolor=group['color'],
                linewidth=2,
                transform=ax_left.transAxes
            )
            ax_left.add_patch(rect)
            
            # Título del tanque
            title_text = f"Tanque {group['tank_label']}"
            ax_left.text(0.05, y_top - 0.02, title_text,
                        transform=ax_left.transAxes,
                        fontsize=12, fontweight='bold',
                        color=group['color'])
            
            # Línea resumen: X | Y | Valor tanque | Volumen total
            summary_y = y_top - 0.06
            summary_text = f"X: {group['tank_x']:.2f} | Y: {group['tank_y']:.2f} | Valor tanque: ${group['tank_cost']/1e6:.2f}M | Volumen total: {group['tank_volume']/1000:.1f}k m³"
            ax_left.text(0.05, summary_y, summary_text,
                        transform=ax_left.transAxes,
                        fontsize=8, color='#495057')
            
            # Tabla de ramales
            table_y_start = summary_y - 0.05
            row_height = 0.035
            
            # Headers de tabla
            headers = ['Ramal', 'Costo', 'Volumen', 'Longitud', 'Dmáx']
            x_positions = [0.05, 0.25, 0.45, 0.65, 0.85]
            
            for j, header in enumerate(headers):
                ax_left.text(x_positions[j], table_y_start, header,
                            transform=ax_left.transAxes,
                            fontsize=8, fontweight='bold',
                            color='#212529')
            
            # Filas de ramales
            for j, ramal in enumerate(group['ramales']):
                row_y = table_y_start - (j + 1) * row_height
                
                ax_left.text(x_positions[0], row_y, str(ramal['ramal']),
                            transform=ax_left.transAxes, fontsize=7, color='#495057')
                ax_left.text(x_positions[1], row_y, f"${ramal['costo']/1e6:.2f}M" if ramal['costo'] > 0 else "$-",
                            transform=ax_left.transAxes, fontsize=7, color='#495057')
                ax_left.text(x_positions[2], row_y, f"{ramal['volumen']/1000:.1f}k" if ramal['volumen'] > 0 else "-",
                            transform=ax_left.transAxes, fontsize=7, color='#495057')
                ax_left.text(x_positions[3], row_y, f"{ramal['longitud']:.0f}",
                            transform=ax_left.transAxes, fontsize=7, color='#495057')
                ax_left.text(x_positions[4], row_y, f"{ramal['dmax']:.2f}" if ramal['dmax'] > 0 else "-",
                            transform=ax_left.transAxes, fontsize=7, color='#495057')
            
            # Fila TOTAL GRUPO
            total_row_y = table_y_start - (len(group['ramales']) + 1.5) * row_height
            
            ax_left.text(x_positions[0], total_row_y, 'TOTAL GRUPO',
                        transform=ax_left.transAxes, fontsize=8, fontweight='bold',
                        color='#212529')
            ax_left.text(x_positions[1], total_row_y, f"${group['total_cost']/1e6:.2f}M",
                        transform=ax_left.transAxes, fontsize=8, fontweight='bold',
                        color='#212529')
            ax_left.text(x_positions[2], total_row_y, f"{group['total_volume']/1000:.1f}k",
                        transform=ax_left.transAxes, fontsize=8, fontweight='bold',
                        color='#212529')
            ax_left.text(x_positions[3], total_row_y, f"{group['total_length']:.0f}",
                        transform=ax_left.transAxes, fontsize=8, fontweight='bold',
                        color='#212529')
            ax_left.text(x_positions[4], total_row_y, '—',
                        transform=ax_left.transAxes, fontsize=8, fontweight='bold',
                        color='#6C757D')
        
        # ============================================================
        # 6. PANEL DERECHO: MAPA DE RED
        # ============================================================
        
        # Convertir a GeoDataFrame para facilitar plotting
        all_network = networks_gdf.copy()
        
        # Convertir a CRS Web Mercator para contextily
        if all_network.crs is None:
            all_network.set_crs(epsg=4326, inplace=True)
        all_network_3857 = all_network.to_crs(epsg=3857)
        
        # Crear GeoDataFrame de los grupos
        groups_gdf = gpd.GeoDataFrame(
            group_data,
            geometry=[g['geometry'] for g in group_data],
            crs=all_network.crs
        )
        groups_gdf_3857 = groups_gdf.to_crs(epsg=3857)
        
        # 6.1 Dibujar toda la red en gris suave
        all_network_3857.plot(
            ax=ax_right,
            color='#ADB5BD',
            alpha=0.4,
            linewidth=0.8,
            label='Red existente'
        )
        
        # 6.2 Dibujar cada grupo con su color
        for i, group in enumerate(group_data):
            group_geom = groups_gdf_3857.iloc[i]
            
            # Dibujar geometría del grupo
            if hasattr(group_geom.geometry, 'geoms'):
                for geom in group_geom.geometry.geoms:
                    ax_right.plot(*geom.xy, color=group['color'], linewidth=3, alpha=0.8)
            else:
                ax_right.plot(*group_geom.geometry.xy, color=group['color'], linewidth=3, alpha=0.8)
            
            # Calcular centroide para label T*
            centroid = group_geom.geometry.centroid
            
            # Label T1, T2, etc.
            ax_right.annotate(
                group['tank_label'],
                xy=(centroid.x, centroid.y),
                fontsize=14, fontweight='bold',
                color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=group['color'], 
                         edgecolor='white', linewidth=2)
            )
        
        # 6.3 Globos azules con nombre del ramal en el punto de salida
        for group in group_data:
            for ramal in group['ramales']:
                # Calcular punto de salida del ramal (punto más cercano al tanque)
                ramal_geom = ramal['geometry']
                
                if hasattr(ramal_geom, 'coords'):
                    # Es una LineString - tomar el primer punto
                    if len(list(ramal_geom.coords)) > 0:
                        punto_salida = ramal_geom.coords[0]
                    else:
                        continue
                elif hasattr(ramal_geom, 'geoms') and len(list(ramal_geom.geoms)) > 0:
                    # Es una GeometryCollection o similar - tomar el primer punto de la primera geometría
                    first_geom = list(ramal_geom.geoms)[0]
                    if hasattr(first_geom, 'coords') and len(list(first_geom.coords)) > 0:
                        punto_salida = first_geom.coords[0]
                    else:
                        continue
                else:
                    continue
                
                # Convertir punto a Web Mercator si es necesario
                if all_network.crs.to_epsg() == 4326:
                    from shapely.geometry import Point
                    punto = Point(punto_salida)
                    punto_gdf = gpd.GeoDataFrame(geometry=[punto], crs=all_network.crs)
                    punto_3857 = punto_gdf.to_crs(epsg=3857).geometry.iloc[0]
                    px, py = punto_3857.x, punto_3857.y
                else:
                    px, py = punto_salida[0], punto_salida[1]
                
                # Globo azul con nombre del ramal
                ax_right.annotate(
                    str(ramal['ramal']),
                    xy=(px, py),
                    xytext=(15, 15),
                    textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#0D6EFD',
                             edgecolor='white', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', color='#0D6EFD', lw=1.5)
                )
        
        # 6.4 Agregar basemap satelital con opacidad
        ctx.add_basemap(
            ax_right,
            source=ctx.providers.Esri.WorldImagery,
            alpha=basemap_alpha
        )
        
        # Configurar ejes del mapa
        ax_right.set_title('Distribución Espacial de Tanques y Derivaciones',
                          fontsize=14, fontweight='bold', pad=10)
        ax_right.set_axis_off()
        
        # ============================================================
        # 7. GUARDAR FIGURA
        # ============================================================
        
        save_path = self._get_next_figure_path(output_name)
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")


if __name__ == "__main__":
    print("Running Dashboard Generator with REAL data...")
    
    # ============================================
    # CONFIGURABLE: Carpeta de resultados
    # ============================================
    RESULTS_FOLDER = "optimization_results_t5_min_vol"  # <-- Cambiar según el caso
    # ============================================
    
    # Read real results from optimization
    csv_path = Path(RESULTS_FOLDER) / "sequence_tracking.csv"
    
    if csv_path.exists():
        df_real = pd.read_csv(csv_path)
        print(f"  Loaded {len(df_real)} rows from {csv_path}")
        print(f"  Columns: {list(df_real.columns)}")
        
        # ============================================================
        # LIMPIAR COLUMNAS DE MONEDA (quitar $ y comas)
        # ============================================================
        currency_cols = ['cost_social_total', 'cost_investment_total', 'cost_links', 
                        'cost_tanks', 'cost_land', 'cost_residual_total', 'cost_residual_flood',
                        'cost_residual_infra', 'current_tank_cost', 'current_tank_land',
                        '_marginal_cost', 'cost_display']
        
        for col in currency_cols:
            if col in df_real.columns:
                df_real[col] = df_real[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
                df_real[col] = pd.to_numeric(df_real[col], errors='coerce').fillna(0)
        
        # ============================================================
        
        # Generate in the same output folder
        output_path = Path(RESULTS_FOLDER)
        
        gen = EvolutionDashboardGenerator(df_real, output_path)
        gen.generate_all()
        
        # ============================================
        # NUEVA FUNCIÓN: Mapa de grupos/tanques con red
        # ============================================
        print("\n  [Dashboard] Generating group cards with network map...")
        try:
            gen.plot_group_cards_map_from_networks_and_swmm(
                networks_gpkg_path="base_network.gpkg",
                swmm_inp_path="base_swmm.inp",
                sequence_csv_path=str(csv_path),
                basemap_alpha=0.5,
                output_name="group_cards_network_map"
            )
        except Exception as e:
            # Re-lanzar el error para ver el traceback completo
            import traceback
            traceback.print_exc()
            raise
        
        print(f"\nDone. Check {output_path.absolute()}")
    else:
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run the optimization first (rut_10_run_tanque_tormenta.py)")


