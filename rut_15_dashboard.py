
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.transforms as mtransforms
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
        
        # Set modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
        
    def generate_all(self):
        """Generates all dashboard plots."""
        if self.df.empty:
            print("  [Dashboard] No data to plot.")
            return

        print("  [Dashboard] Generating evolution plots...")
        self.plot_efficiency_by_tank()
        self.plot_cost_reduction_evolution()
        self.plot_roi_curve()

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
            
            tank_groups = self.df.groupby(tank_id).agg({
                '_marginal_cost': 'sum',
                'marginal_reduction': 'sum',
                'current_tank_volume': 'first',
                'added_predio': 'first',
                'step': 'first',
            }).reset_index(drop=True)
            
            tank_cost = tank_groups['_marginal_cost']
            tank_reduction = tank_groups['marginal_reduction']
            tank_volume = tank_groups['current_tank_volume'].fillna(1000)
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

            
            # Create figure with table left, two charts stacked on right
            fig = plt.figure(figsize=(24, 14))
            
            # Create GridSpec: left column for table, right column for 2 stacked charts
            gs = fig.add_gridspec(2, 2, width_ratios=[0.25, 0.75], height_ratios=[1, 1], 
                                  wspace=0.12, hspace=0.25)
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
            table.set_fontsize(10)
            table.scale(1.0, 2.0)
            
            # Style table - mark invalid rows with light red
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(fontweight='bold', color='white', fontsize=11)
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
            
            ax_table.set_title('Ranking de Eficiencia\n(Ordenado: Mejor → Peor)',
                              fontsize=14, fontweight='bold', pad=15)
            
            # === TOP RIGHT: BAR CHART - Eficiencia por Tanque ===
            tank_labels = [f"T{i+1}" for i in range(len(tank_groups))]
            
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
            x_pos = np.arange(len(tank_labels))
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
                
                ax_bar.text(bar.get_x() + bar.get_width()/2, y_pos + eff_max*0.02,
                       label, ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=color, rotation=45)
            
            # Styling
            ax_bar.set_xticks(x_pos)
            ax_bar.set_xticklabels(tank_labels, fontsize=10, fontweight='bold')
            ax_bar.set_xlabel('Tanque', fontsize=12, fontweight='bold')
            ax_bar.set_ylabel('Eficiencia ($/m³)', fontsize=12, fontweight='bold')
            ax_bar.set_title(f'Eficiencia por Tanque: Costo por m³ Reducido\n({len(tank_groups)} Tanques, {is_invalid.sum()} N/A)',
                         fontsize=14, fontweight='bold', pad=10)
            
            # Grid
            ax_bar.grid(True, linestyle='--', alpha=0.4, color='#cccccc', axis='y', linewidth=1)
            ax_bar.set_axisbelow(True)
            
            # Y axis formatting
            ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
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
            ax_evo.plot(tank_numbers, running_avg_eff, 'o-', color='#1976d2', linewidth=2.5,
                       markersize=10, markeredgecolor='#0d47a1', markeredgewidth=1.5,
                       label='Eficiencia Promedio Acumulada')
            
            # Fill area under curve
            ax_evo.fill_between(tank_numbers, 0, running_avg_eff, alpha=0.2, color='#64b5f6')
            
            # Add value labels
            for i, (tank_num, eff) in enumerate(zip(tank_numbers, running_avg_eff)):
                if not np.isnan(eff):
                    ax_evo.annotate(f'${eff:,.0f}', xy=(tank_num, eff), xytext=(0, 10),
                                   textcoords='offset points', ha='center', va='bottom',
                                   fontsize=9, fontweight='bold', color='#0d47a1',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                            alpha=0.8, edgecolor='none'))
            
            # Styling
            ax_evo.set_xticks(tank_numbers)
            ax_evo.set_xticklabels([f'T{i}' for i in tank_numbers], fontsize=9, fontweight='bold')
            ax_evo.set_xlabel('Tanque (Acumulado)', fontsize=12, fontweight='bold')
            ax_evo.set_ylabel('Eficiencia Promedio ($/m³)', fontsize=12, fontweight='bold')
            ax_evo.set_title('Evolución de la Eficiencia Promedio: $/m³ al Agregar Tanques',
                            fontsize=14, fontweight='bold', pad=10)
            
            ax_evo.grid(True, linestyle='--', alpha=0.4, color='#cccccc', linewidth=1)
            ax_evo.set_axisbelow(True)
            ax_evo.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax_evo.legend(loc='upper left', fontsize=10, framealpha=0.95)
            
            # Summary box (on bar chart)
            total_cost_sum = self.df['cost_investment_total'].iloc[-1]
            total_reduction_sum = self.df['flooding_reduction'].iloc[-1]
            avg_cpm = total_cost_sum / total_reduction_sum if total_reduction_sum > 0 else 0
            
            # Find best/worst from VALID entries only
            valid_indices = cost_per_m3_for_sort[cost_per_m3_for_sort != np.inf].index
            if len(valid_indices) > 0:
                valid_cpm_series = cost_per_m3_for_sort[valid_indices]
                best_idx = valid_cpm_series.idxmin()
                worst_idx = valid_cpm_series.idxmax()
                best_text = f"Mejor: T{best_idx+1} (${cost_per_m3.iloc[best_idx]:,.0f}/m3)"
                worst_text = f"Peor:  T{worst_idx+1} (${cost_per_m3.iloc[worst_idx]:,.0f}/m3)"
            else:
                best_text = "Mejor: N/A"
                worst_text = "Peor:  N/A"
            
            n_invalid = is_invalid.sum()
            
            stats = (
                f"=== RESUMEN ===\n"
                f"Tanques: {len(tank_groups)} ({n_invalid} N/A)\n"
                f"Inversion: {self.format_currency_smart(total_cost_sum)}\n"
                f"Reduccion: {total_reduction_sum:,.0f} m3\n"
                f"Costo Prom: ${avg_cpm:,.0f}/m3\n"
                f"---------------\n"
                f"{best_text}\n"
                f"{worst_text}"
            )
            
            ax_bar.text(0.02, 0.98, stats, transform=ax_bar.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffacd',
                             alpha=0.95, edgecolor='#daa520', linewidth=2),
                    fontfamily='monospace')
            
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
            
            save_path = self.output_dir / "00_efficiency_by_tank.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  [Dashboard] Saved: {save_path}")

    def plot_cost_reduction_evolution(self):
        if len(self.df) < 1:
            return
    
        fig, ax1 = plt.subplots(figsize=(22, 10))
    
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
    
        ax1.set_ylabel("Inversion Acumulada ($)", fontsize=13, fontweight="bold", color="#1a4d8f", labelpad=5)
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
            "Reduccion de Inundacion Acumulada (m3)",
            fontsize=14,
            fontweight="bold",
            color="#2e7d32",
            labelpad=32,  # more separation from ticks/figure edge
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
                linewidth=2.0,
                markersize=7,
                markeredgecolor="#8e0000",
                markeredgewidth=1.5,
                label="Inundacion Residual",
                alpha=1.0,
            )
            ax3.set_ylabel("Inundacion Residual (m3)", fontsize=11, color="#c62828", fontweight="bold", labelpad=26)
            ax3.tick_params(axis="y", labelcolor="#c62828", pad=10)
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    
        # --- reduction labels at TOP (outside) ---
        for i, (step, red) in enumerate(zip(steps, reduction_total)):
            y_offset_pts = 35
            ax2.annotate(
                f"{red:,.0f} m3",
                xy=(step, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, y_offset_pts),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#2e7d32",
                rotation=45,              # angle in degrees (e.g., 0, 45, 90, -30)
                rotation_mode="anchor",   # keeps the text anchored to (x, y)
                # bbox=dict(
                #     boxstyle="round,pad=0.3",
                #     facecolor="white",
                #     alpha=0.9,
                #     edgecolor="#2e7d32",
                # ),
                clip_on=False,
                annotation_clip=False,  # important: do not clip outside axes
                zorder=20,
            )
    
        # --- custom 2-line x labels (step + $M in blue) ---
        ax1.set_xlabel("Paso de Optimizacion (Tanque Agregado)", fontsize=13, fontweight="bold", labelpad=120)
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
                fontsize=14,
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
                fontsize=14,
                fontweight="bold",
                color="#1976d2",
                clip_on=False,
                rotation=45            ,  # angle in degrees (e.g., 0, 45, 90, -30)
                rotation_mode="anchor",   # keeps the text anchored to (x, y)
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
                fontsize=14,
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
    
        save_path = self.output_dir / "01_cost_reduction_evolution.png"
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
        
        # Create larger figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14),
                                         gridspec_kw={'height_ratios': [1, 1]})
        
        # ===== TOP PLOT: Cumulative ROI =====
        line = ax1.plot(steps, roi, 'o-', color='#1976d2', linewidth=5,
                        markersize=18, markeredgecolor='#0d47a1', markeredgewidth=3,
                        label='ROI Acumulado')
        
        # Fill under curve
        ax1.fill_between(steps, 0, roi, alpha=0.25, color='#64b5f6')
        
        # Add large value labels
        for step, r in zip(steps, roi):
            ax1.annotate(f'{r:,.0f}', xy=(step, r), xytext=(0, 15),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='#0d47a1',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                 alpha=0.95, edgecolor='#1976d2', linewidth=1))
        
        # Styling
        ax1.set_ylabel('Reduccion [m³/$1M] ', fontsize=18, fontweight='bold', labelpad=15)
        ax1.set_title('Retorno  de Inversion (ROI) ACUMULADO',
                      fontsize=22, fontweight='bold', pad=25)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, linewidth=1.5)
        ax1.set_axisbelow(True)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([])  # Hide x labels on top plot
        
        # Optimal point
        best_roi_idx = np.argmax(roi)
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
        
        ax1.legend(loc='upper right', fontsize=15, framealpha=0.95,
                   edgecolor='black', fancybox=True)
        
        # Set y-axis limits with padding
        y_max = roi.max() * 1.15
        ax1.set_ylim(0, y_max)
        
        # ===== BOTTOM PLOT: Marginal ROI =====
        colors = ['#2e7d32' if mr >= np.median(marginal_roi[marginal_roi > 0]) else '#d32f2f'
                  for mr in marginal_roi]
        
        bars = ax2.bar(steps, marginal_roi, color=colors, alpha=0.85,
                       edgecolor='black', linewidth=2.5, width=0.7)
        
        # Large value labels on bars
        for bar, mr in zip(bars, marginal_roi):
            if mr > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + marginal_roi.max()*0.03,
                        f'{mr:,.0f}', ha='center', va='bottom',
                        fontsize=13, fontweight='bold', color='black')
        
        # Average line
        avg_marginal_roi = np.mean(marginal_roi[marginal_roi > 0])
        ax2.axhline(y=avg_marginal_roi, color='#ff9800', linestyle='--', linewidth=3.5,
                    label=f'Promedio: {avg_marginal_roi:,.0f} m³/$1M', alpha=0.9)
        
        # Styling
        ax2.set_xlabel('Número de Paso (Tanque)', fontsize=18, fontweight='bold', labelpad=15)
        ax2.set_ylabel('ROI de Este Paso\n(m³/$1M)', fontsize=18, fontweight='bold', labelpad=15)
        ax2.set_title('ROI Marginal',
                      fontsize=22, fontweight='bold', pad=25)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax2.tick_params(axis='both', labelsize=14)
        ax2.grid(True, linestyle='--', color='#aaaaaa', alpha=0.5, axis='y', linewidth=1.5)
        ax2.set_axisbelow(True)
        ax2.set_xticks(steps)
        ax2.set_xticklabels([f'p{s}' for s in steps], fontsize=13, fontweight='bold')
        
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
        
        ax2.legend(loc='upper left', fontsize=15, framealpha=0.95,
                   edgecolor='black', fancybox=True)
        
        # Set y-axis limits with padding
        y_max2 = marginal_roi.max() * 1.15
        ax2.set_ylim(0, y_max2)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "02_roi_curve.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=120)
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


