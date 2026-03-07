
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
        self.plot_roi_curve()
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
        if len(self.df) < 1:
            return
    
        fig, ax1 = plt.subplots(figsize=(14, 10))  # Más grande para legibilidad
    
        steps = self.df["step"].values
        cost_total = self.df["cost_investment_total"].values
        reduction_total = self.df["flooding_reduction"].values
        flooding_remaining = self.df["flooding_volume"].values if "flooding_volume" in self.df.columns else None
    
        # Bars (cost)
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(steps)))
        ax1.bar(
            steps,
            cost_total,
            color=colors,
            edgecolor="#2c5aa0",
            linewidth=2,
            label="Inversion Acumulada",
            alpha=0.6,
        )
    
        ax1.set_ylabel("Inversion Acumulada ($)", fontsize=11, fontweight="bold", color="#1a4d8f", labelpad=5)
        ax1.tick_params(axis="y", labelcolor="#1a4d8f")
        ax1.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
    
        # Right axis line (reduction)
        ax2 = ax1.twinx()
        ax2.plot(
            steps,
            reduction_total,
            "o-",
            color="#2e7d32",
            linewidth=2.0,
            markersize=10,
            markeredgecolor="#1b5e20",
            markeredgewidth=1,
            label="Reduccion Acumulada",
            alpha=1.0,
        )
    
        ax2.set_ylabel(
            "Reduccion de Inundacion (m3)",
            fontsize=11,
            fontweight="bold",
            color="#2e7d32",
            labelpad=20,
        )
        ax2.tick_params(axis="y", labelcolor="#2e7d32", pad=10)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
        # Optional third axis (residual) with more outward spacing
        if flooding_remaining is not None:
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 95))
            ax3.plot(
                steps,
                flooding_remaining,
                "s--",
                color="#c62828",
                linewidth=1.0,
                markersize=5,
                markeredgecolor="#8e0000",
                markeredgewidth=1.0,
                label="Inundacion Residual",
                alpha=0.9,
            )
            ax3.set_ylabel("Inundacion Residual (m3)", fontsize=11, color="#c62828", fontweight="bold", labelpad=26)
            ax3.tick_params(axis="y", labelcolor="#c62828", pad=10)
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
        # --- reduction labels at TOP (outside) - solo cada 3 pasos ---
        for i, (step, red) in enumerate(zip(steps, reduction_total)):
            if i % 3 == 0 or i == len(steps) - 1:  # Mostrar cada 3 pasos + último
                y_offset_pts = 45
                ax2.annotate(
                    f"{red:,.0f}",
                    xy=(step, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, y_offset_pts),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#2e7d32",
                    rotation=0,  # Sin rotación para mejor lectura
                    clip_on=False,
                    annotation_clip=False,
                    zorder=20,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')
                )
    
        # --- custom 2-line x labels (step + $M in blue) ---
        ax1.set_xlabel("Paso de Optimizacion (Tanque Agregado)", fontsize=12, fontweight="bold", labelpad=100)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([])
        ax1.tick_params(axis="x", length=0)
    
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        for s, c in zip(steps, cost_total):
            ax1.annotate(
                f"{s}",
                xy=(s, 0.03),
                xycoords=trans,
                xytext=(0, -26),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="black",
                clip_on=False,
            )
            ax1.annotate(
                f"{c/1_000_000:,.1f} M",
                xy=(s, 0.0),
                xycoords=trans,
                xytext=(0, -48),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#1976d2",
                clip_on=False,
                rotation=30,  # Menos rotación
                rotation_mode="anchor",
            )
    
        # Grid
        ax1.grid(True, linestyle="--", color="#cccccc", alpha=0.4, axis="y", linewidth=1)
        ax1.set_axisbelow(True)
    
        # Figure titles (top)
        main_title = "Evolucion de Inversion vs Reduccion de Inundacion"
        sub_title = "Analisis Costo-Beneficio por Paso"
    
        # More room top/bottom so outside labels do not get cut
        plt.subplots_adjust(bottom=0.10, top=0.85)
        # fig.text(0.5, 0.965, main_title, ha="center", va="top", fontsize=18, fontweight="bold")
        # fig.text(0.5, 0.935, sub_title, ha="center", va="top", fontsize=14, fontweight="bold")
    
        # Legend (kept, but move a bit lower due to bigger bottom area)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if flooding_remaining is not None:
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2 + lines3,
                labels1 + labels2 + labels3,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=3,
                fontsize=10,
                framealpha=0.95,
                edgecolor="black",
                fancybox=True,
            )
        else:
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
                ncol=2,
                fontsize=14,
                framealpha=0.9,
            )
    
        save_path = self._get_next_figure_path("cost_reduction_evolution")
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [Dashboard] Saved: {save_path}")
    
    
    def plot_roi_curve(self):
        """
        ROI curve showing flood reduction obtained per $1M invested.
        Shows when investment returns start to diminish.
        """
        if len(self.df) < 1:
            return
        
        # Get cumulative data
        steps = self.df['step'].values
        cost_total = self.df['cost_investment_total'].values
        reduction_total = self.df['flooding_reduction'].values
        
        # Calculate ROI: m³ reduced per $1M invested
        cost_in_millions = cost_total / 1_000_000
        roi = reduction_total / cost_in_millions
        roi = np.where(cost_in_millions > 0, roi, 0)
        
        # Calculate marginal ROI (change between steps)
        marginal_cost = np.diff(cost_total, prepend=0) / 1_000_000
        marginal_reduction = np.diff(reduction_total, prepend=0)
        marginal_roi = np.where(marginal_cost > 0, marginal_reduction / marginal_cost, 0)
        
        # Create figure - FORMATO A4
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 11.69),
                                         gridspec_kw={'height_ratios': [1, 1]})
        
        # ===== TOP PLOT: Cumulative ROI =====
        # Optimal point (definir antes de usar)
        best_roi_idx = np.argmax(roi)
        
        line = ax1.plot(steps, roi, 'o-', color='#1976d2', linewidth=5,
                        markersize=18, markeredgecolor='#0d47a1', markeredgewidth=3,
                        label='ROI Acumulado')
        
        # Fill under curve
        ax1.fill_between(steps, 0, roi, alpha=0.25, color='#64b5f6')
        
        # Add value labels solo para inicio, mejor, y fin
        key_points = [0, best_roi_idx, len(steps)-1]
        for idx in key_points:
            step, r = steps[idx], roi[idx]
            ax1.annotate(f'{r:,.0f}', xy=(step, r), xytext=(0, 20),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='#0d47a1',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.95, edgecolor='#1976d2', linewidth=1.5))
        
        # Styling
        ax1.set_ylabel('Reduccion [m³/$1M] ', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_title('Retorno de Inversion (ROI) ACUMULADO',
                      fontsize=13, fontweight='bold', pad=15)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax1.tick_params(axis='both', labelsize=10)
        ax1.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, linewidth=1.5)
        ax1.set_axisbelow(True)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([])  # Hide x labels on top plot
        
        # Optimal point marker
        ax1.axvline(x=steps[best_roi_idx], color='#2e7d32', linestyle=':',
                    linewidth=3.5, alpha=0.8, label=f'Mejor ROI: Paso {steps[best_roi_idx]}')
        ax1.scatter(steps[best_roi_idx], roi[best_roi_idx], s=500, color='#4caf50',
                    edgecolor='#1b5e20', linewidth=4, zorder=10, marker='*')
        
        # # Large summary box
        # summary1 = (
        #     f"ROI INICIAL:  {roi[0]:,.0f} m³/$1M\n"
        #     f"ROI FINAL:    {roi[-1]:,.0f} m³/$1M\n"
        #     f"ROI MÁXIMO:   {roi[best_roi_idx]:,.0f} m³/$1M\n"
        #     f"DEGRADACIÓN:  {((roi[0] - roi[-1]) / roi[0] * 100):.0f}%"
        # )
        #
        # ax1.text(0.02, 0.97, summary1, transform=ax1.transAxes, fontsize=15,
        #         verticalalignment='top', horizontalalignment='left',
        #         bbox=dict(boxstyle='round,pad=0.8', facecolor='#e3f2fd',
        #                  alpha=0.95, edgecolor='#1976d2', linewidth=3),
        #         fontfamily='monospace', fontweight='bold')
        
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.95,
                   edgecolor='black', fancybox=True)
        
        # Set y-axis limits with padding
        y_max = roi.max() * 1.15
        ax1.set_ylim(0, y_max)
        
        # ===== BOTTOM PLOT: Marginal ROI =====
        colors = ['#2e7d32' if mr >= np.median(marginal_roi[marginal_roi > 0]) else '#d32f2f'
                  for mr in marginal_roi]
        
        bars = ax2.bar(steps, marginal_roi, color=colors, alpha=0.85,
                       edgecolor='black', linewidth=2.5, width=0.7)
        
        # Value labels on bars - solo los más importantes (>5000)
        for bar, mr in zip(bars, marginal_roi):
            if mr > 5000:  # Solo mostrar valores grandes
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + marginal_roi.max()*0.02,
                        f'{mr:,.0f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Average line
        avg_marginal_roi = np.mean(marginal_roi[marginal_roi > 0])
        ax2.axhline(y=avg_marginal_roi, color='#ff9800', linestyle='--', linewidth=3.5,
                    label=f'Promedio: {avg_marginal_roi:,.0f} m³/$1M', alpha=0.9)
        
        # Styling
        ax2.set_xlabel('Número de Paso (Tanque)', fontsize=11, fontweight='bold', labelpad=10)
        ax2.set_ylabel('ROI de Este Paso\n(m³/$1M)', fontsize=11, fontweight='bold', labelpad=10)
        ax2.set_title('ROI Marginal',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax2.tick_params(axis='both', labelsize=10)
        ax2.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, axis='y', linewidth=1)
        ax2.set_axisbelow(True)
        ax2.set_xticks(steps)
        ax2.set_xticklabels([f'p{s}' for s in steps], fontsize=9, fontweight='bold')
        
        # Find best and worst
        best_step_idx = np.argmax(marginal_roi)
        worst_step_idx = np.argmin(marginal_roi[marginal_roi > 0]) if np.any(marginal_roi > 0) else 0
        
        # summary2 = (
        #     f"MEJOR:  Paso {steps[best_step_idx]}\n"
        #     f"  → {marginal_roi[best_step_idx]:,.0f} m³/$1M\n\n"
        #     f"PEOR:   Paso {steps[worst_step_idx]}\n"
        #     f"  → {marginal_roi[worst_step_idx]:,.0f} m³/$1M"
        # )
        #
        # ax2.text(0.98, 0.97, summary2, transform=ax2.transAxes, fontsize=15,
        #         verticalalignment='top', horizontalalignment='right',
        #         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fffacd',
        #                  alpha=0.95, edgecolor='#daa520', linewidth=3),
        #         fontfamily='monospace', fontweight='bold')
        
        ax2.legend(loc='upper left', fontsize=10, framealpha=0.95,
                   edgecolor='black', fancybox=True)
        
        # Set y-axis limits with padding
        y_max2 = marginal_roi.max() * 1.15
        ax2.set_ylim(0, y_max2)
        
        plt.tight_layout()
        
        save_path = self._get_next_figure_path("roi_curve")
        fig.savefig(save_path, bbox_inches='tight', dpi=120)
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
        ]
        
        # Calcular eficiencia marginal para colorear puntos
        self.df['_marginal_eff'] = self.df['marginal_reduction'] / (self.df['_marginal_cost'] / 1e6 + 1e-10)
        median_eff = self.df['_marginal_eff'].median()
        
        # Crear figura con subplots 2x3
        n_vars = len(variables)
        n_rows = 2
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14))  # Más grande para mejor legibilidad
        axes = axes.flatten()
        
        # Titulo general
        fig.suptitle('Curvas Pareto: Inversion vs Indicadores Hidraulicos\n' + 
                    'Color = eficiencia marginal (Verde = sobre mediana | Rojo = bajo)',
                    fontsize=12, fontweight='bold', y=0.98)
        
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
            ax.scatter(x[0], y[0], s=200, c='darkblue', edgecolors='white', linewidth=2.5, zorder=5, marker='o')
            val_0 = format_val(y[0], unit)
            ax.annotate(f'p0: {val_0}', (x[0], y[0]), textcoords="offset points", 
                       xytext=(-25, 20), ha='center', fontsize=10, color='darkblue', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkblue', alpha=0.7, lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.95, edgecolor='darkblue', linewidth=1.5))
            
            # Puntos intermedios con valores numéricos
            intermediate_steps = [6, 12, 18, 24]
            colors_intermediate = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2']  # Azul, verde, naranja, morado
            for i, step_num in enumerate(intermediate_steps):
                idx = find_nearest_idx(step_num)
                if idx is not None and idx != 0 and idx != len(steps)-1:
                    color = colors_intermediate[i % len(colors_intermediate)]
                    ax.scatter(x[idx], y[idx], s=120, c=color, edgecolors='white', linewidth=2, zorder=4)
                    val = format_val(y[idx], unit)
                    # Offset alternado para evitar solapamiento
                    offset_x = 22 if i % 2 == 0 else -22
                    offset_y = 18 if i < 2 else -18
                    ax.annotate(f'p{int(steps[idx])}:\n{val}', (x[idx], y[idx]), textcoords="offset points", 
                               xytext=(offset_x, offset_y), ha='center', fontsize=9, color=color, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color, linewidth=1.5))
            
            # Punto final (último) - grande con valor numérico
            ax.scatter(x[-1], y[-1], s=200, c='darkgreen', edgecolors='white', linewidth=2.5, zorder=5, marker='o')
            val_f = format_val(y[-1], unit)
            ax.annotate(f'p{int(steps[-1])}: {val_f}', (x[-1], y[-1]), textcoords="offset points", 
                       xytext=(30, -20), ha='center', fontsize=10, color='darkgreen', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7, lw=1.5),
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.95, edgecolor='darkgreen', linewidth=1.5))
            
            # Linea de mediana del valor
            median_y = np.median(y)
            ax.axhline(y=median_y, color='blue', linestyle=':', alpha=0.5, linewidth=1)
            ax.text(x.max()*0.98, median_y, f'Med: {median_y:.0f}', 
                   ha='right', va='bottom', fontsize=7, color='blue', alpha=0.7)
            
            # Eje Y secundario con porcentaje de cambio
            ax2 = ax.twinx()
            baseline_val = y[0] if len(y) > 0 and y[0] != 0 else 1
            # Calcular limites de porcentaje basados en los valores actuales
            y_min, y_max = ax.get_ylim()
            pct_min = ((y_min - baseline_val) / baseline_val) * 100
            pct_max = ((y_max - baseline_val) / baseline_val) * 100
            ax2.set_ylim(pct_min, pct_max)
            ax2.set_ylabel('% Cambio', fontsize=8, color='gray', alpha=0.7)
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
            
            # Configuracion del subplot principal
            ax.set_xlabel('Inversion (M$)', fontsize=9)
            ax.set_ylabel(f'{title}\n({unit})', fontsize=9)
            ax.set_title(f'{title}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Leyenda de eficiencia
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2e7d32', label=f'Eff. >= mediana'),
                Patch(facecolor='#c62828', label=f'Eff. < mediana'),
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=7)
        
        # Ocultar subplot vacio si hay
        if n_vars < len(axes):
            for idx in range(n_vars, len(axes)):
                axes[idx].axis('off')
        
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
        
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


if __name__ == "__main__":
    print("Running Dashboard Generator with REAL data...")
    
    # Read real results from optimization
    csv_path = Path("optimization_results/sequence_tracking.csv")
    
    if csv_path.exists():
        df_real = pd.read_csv(csv_path)
        print(f"  Loaded {len(df_real)} rows from {csv_path}")
        print(f"  Columns: {list(df_real.columns)}")
        
        # Generate in the same output folder
        output_path = Path("optimization_results")
        
        gen = EvolutionDashboardGenerator(df_real, output_path)
        gen.generate_all()
        
        print(f"Done. Check {output_path.absolute()}")
    else:
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run the optimization first (rut_10_run_tanque_tormenta.py)")


