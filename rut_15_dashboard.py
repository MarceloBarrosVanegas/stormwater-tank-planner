
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
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
            tank_groups = self.df.groupby(tank_id).agg({
                'marginal_cost': 'sum',
                'marginal_reduction': 'sum',
                'current_tank_volume': 'first',
                'added_predio': 'first',
                'step': 'first',
            }).reset_index(drop=True)
            
            tank_cost = tank_groups['marginal_cost']
            tank_reduction = tank_groups['marginal_reduction']
            tank_volume = tank_groups['current_tank_volume'].fillna(1000)
            tank_predio = tank_groups['added_predio']
            tank_step = tank_groups['step']
            
            # Calculate efficiency: $/m³ reduced
            cost_per_m3 = tank_cost / tank_reduction.replace(0, np.nan)
            cost_per_m3 = cost_per_m3.fillna(cost_per_m3.max())
            
            # Normalize for colormap (inverted: lower = better = green)
            cpm_min, cpm_max = cost_per_m3.min(), cost_per_m3.max()
            if cpm_max > cpm_min:
                cpm_normalized = 1 - ((cost_per_m3 - cpm_min) / (cpm_max - cpm_min))
            else:
                cpm_normalized = pd.Series([0.5] * len(cost_per_m3))
            
            # Size based on tank volume
            if tank_volume.max() > 0:
                sizes = 100 + (tank_volume / tank_volume.max()) * 200
            else:
                sizes = 400
            
            # Create figure with two panels: table left, scatter right
            fig = plt.figure(figsize=(22, 10))
            
            # Create GridSpec
            gs = fig.add_gridspec(1, 2, width_ratios=[0.28, 0.72], wspace=0.15)
            ax_table = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[0, 1])
            
            # === LEFT PANEL: TABLE ===
            ax_table.axis('off')
            
            # Prepare table data - KEEP ORIGINAL ORDER FIRST
            table_rows = []
            for i in range(len(tank_groups)):
                table_rows.append({
                    'original_idx': i,  # Track original index
                    'tank_label': f"T{i+1}",
                    'predio': f"P{tank_predio.iloc[i]}",
                    'costo': self.format_currency_smart(tank_cost.iloc[i]),
                    'reduccion': f"{tank_reduction.iloc[i]:,.0f}",
                    'volumen': f"{tank_volume.iloc[i]:,.0f}",
                    'eficiencia': f"${cost_per_m3.iloc[i]:,.0f}",
                    'eficiencia_num': cost_per_m3.iloc[i]  # For sorting
                })
            
            # Sort by efficiency (lowest first = best)
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
            
            # Style table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header
                    cell.set_text_props(fontweight='bold', color='white', fontsize=11)
                    cell.set_facecolor('#2e5cb8')
                    cell.set_height(0.08)
                else:  # Data rows
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_height(0.08)
                cell.set_edgecolor('#aaaaaa')
                cell.set_linewidth(1.5)
            
            ax_table.set_title('Ranking de Eficiencia\n(Ordenado: Mejor → Peor)',
                              fontsize=14, fontweight='bold', pad=15)
            
            # === RIGHT PANEL: SCATTER PLOT ===
            scatter = ax.scatter(
                tank_cost,
                tank_reduction,
                c=cpm_normalized,
                cmap='RdYlGn',
                s=sizes,
                alpha=1.0,
                edgecolors='black',
                linewidths=2.5,
                zorder=5,
                vmin=0,
                vmax=1
            )
            
            # Labels - USE ORIGINAL ORDER (T1, T2, ..., T8)
            for i, (cost, reduction, predio) in enumerate(zip(tank_cost, tank_reduction, tank_predio)):
                # Color code the label based on rank
                eff_rank = [r['original_idx'] for r in table_rows_sorted].index(i) + 1
                
                # Color label based on efficiency ranking
                if eff_rank <= 3:
                    label_color = '#1b5e20'  # Dark green for best
                elif eff_rank >= 6:
                    label_color = '#b71c1c'  # Dark red for worst
                else:
                    label_color = '#f57f17'  # Amber for middle
                
                ax.annotate(
                    f"T{i+1}",
                    xy=(cost, reduction),
                    xytext=(12, 0),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    ha='left',
                    va='center',
                    color=label_color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             alpha=0.95, edgecolor=label_color, linewidth=2),
                )
            
            # Grid
            ax.grid(True, linestyle='--', alpha=0.3, color='#cccccc', linewidth=1)
            ax.set_axisbelow(True)
            
            # Trend line
            if len(tank_cost) > 1:
                z = np.polyfit(tank_cost, tank_reduction, 1)
                p = np.poly1d(z)
                y_pred = p(tank_cost)
                ss_res = np.sum((tank_reduction - y_pred) ** 2)
                ss_tot = np.sum((tank_reduction - np.mean(tank_reduction)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                x_trend = np.linspace(tank_cost.min() * 0.8, tank_cost.max() * 1.1, 100)
                ax.plot(x_trend, p(x_trend), '--', color='#1a237e', alpha=0.8, linewidth=3)
                
                eq_text = f"y = {z[0]:.4f}x + {z[1]:,.0f}\nR² = {r_squared:.3f}"
                ax.text(0.98, 0.02, eq_text, transform=ax.transAxes, fontsize=11,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#bbdefb',
                                 alpha=0.95, edgecolor='#1a237e', linewidth=2),
                        fontfamily='monospace')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('Eficiencia ($/m³)\nVERDE=Mejor | ROJO=Peor', fontsize=12, fontweight='bold')
            cbar.set_ticks([0, 0.5, 1])
            cbar.set_ticklabels([f'${cpm_max:,.0f}', f'${(cpm_max+cpm_min)/2:,.0f}', f'${cpm_min:,.0f}'])
            
            # Axis formatting
            ax.set_xlabel('Costo Total del Tanque ($)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Reducción de Inundación (m³)', fontsize=14, fontweight='bold')
            ax.set_title(f'Eficiencia por Tanque: Costo vs Reducción\n({len(tank_groups)} Tanques)',
                         fontsize=16, fontweight='bold', pad=15)
            
            ax.xaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Axis limits
            x_margin = (tank_cost.max() - tank_cost.min()) * 0.25
            y_margin = (tank_reduction.max() - tank_reduction.min()) * 0.15
            ax.set_xlim(max(0, tank_cost.min() - x_margin), tank_cost.max() + x_margin)
            ax.set_ylim(max(0, tank_reduction.min() - y_margin), tank_reduction.max() + y_margin)
            
            # Summary box
            total_cost_sum = self.df['cost_total'].iloc[-1]
            total_reduction_sum = self.df['flooding_reduction'].iloc[-1]
            avg_cpm = total_cost_sum / total_reduction_sum if total_reduction_sum > 0 else 0
            best_idx = cost_per_m3.idxmin()
            worst_idx = cost_per_m3.idxmax()
            
            stats = (
                f"=== RESUMEN ===\n"
                f"Tanques: {len(tank_groups)}\n"
                f"Inversion: {self.format_currency_smart(total_cost_sum)}\n"
                f"Reduccion: {total_reduction_sum:,.0f} m3\n"
                f"Costo Prom: ${avg_cpm:,.0f}/m3\n"
                f"---------------\n"
                f"Mejor: T{best_idx+1} (${cost_per_m3.iloc[best_idx]:,.0f}/m3)\n"
                f"Peor:  T{worst_idx+1} (${cost_per_m3.iloc[worst_idx]:,.0f}/m3)"
            )
            
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffacd',
                             alpha=0.95, edgecolor='#daa520', linewidth=2),
                    fontfamily='monospace')
            
            plt.grid(True, linestyle='--', color='#cccccc', alpha=0.4, axis='y', linewidth=1)
            
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
        """
        Dual-axis line chart showing:
        - Left axis: Cumulative cost (bars)
        - Right axis: Cumulative flood reduction (line)
        Shows how investment grows with flood reduction step by step.
        """
        if len(self.df) < 1:
            return
        
        fig, ax1 = plt.subplots(figsize=(14, 9))
        
        steps = self.df['step'].values
        cost_total = self.df['cost_total'].values
        reduction_total = self.df['flooding_reduction'].values
        flooding_remaining = self.df['flooding_volume'].values if 'flooding_volume' in self.df.columns else None
        
        # Bar chart for cumulative cost (left axis) - FIXED: darker, more saturated colors
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(steps)))  # Darker range
        bars = ax1.bar(steps, cost_total, color=colors, edgecolor='#2c5aa0',
                      linewidth=2, label='Inversion Acumulada', alpha=1.0)  # FIXED: alpha=1.0
        
        ax1.set_xlabel('Paso de Optimizacion (Tanque Agregado)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Inversion Acumulada ($)', fontsize=13, fontweight='bold', color='#1a4d8f')
        ax1.tick_params(axis='y', labelcolor='#1a4d8f')
        ax1.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
        ax1.set_xticks(steps)
        
        # Add cost labels on bars - alternate positions to avoid overlap
        for i, (bar, cost) in enumerate(zip(bars, cost_total)):
            y_offset = cost_total.max() * 0.02 if i % 2 == 0 else cost_total.max() * 0.06
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                    self.format_currency_smart(cost), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#1a4d8f')
        
        # Line chart for cumulative reduction (right axis) - FIXED: darker green
        ax2 = ax1.twinx()
        line = ax2.plot(steps, reduction_total, 'o-', color='#2e7d32',  # FIXED: darker green
                       linewidth=3.5, markersize=12, markeredgecolor='#1b5e20',
                       markeredgewidth=2, label='Reduccion Acumulada', alpha=1.0)  # FIXED: alpha=1.0
        
        ax2.set_ylabel('Reduccion de Inundacion Acumulada (m3)', fontsize=13, fontweight='bold', color='#2e7d32')
        ax2.tick_params(axis='y', labelcolor='#2e7d32')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add reduction labels on line - alternate positions
        for i, (step, red) in enumerate(zip(steps, reduction_total)):
            y_offset = 20 if i % 2 == 0 else -25
            va = 'bottom' if i % 2 == 0 else 'top'
            ax2.annotate(f'{red:,.0f} m3', xy=(step, red), xytext=(0, y_offset),
                        textcoords='offset points', ha='center', va=va, fontsize=9,
                        fontweight='bold', color='#2e7d32',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 alpha=0.9, edgecolor='#2e7d32'))
        
        # If we have flooding remaining, show it as secondary line - FIXED: darker red
        if flooding_remaining is not None:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(steps, flooding_remaining, 's--', color='#c62828',  # FIXED: darker red
                    linewidth=2.5, markersize=7, markeredgecolor='#8e0000',
                    markeredgewidth=1.5, label='Inundacion Residual', alpha=1.0)  # FIXED: alpha=1.0
            ax3.set_ylabel('Inundacion Residual (m3)', fontsize=11, color='#c62828', fontweight='bold')
            ax3.tick_params(axis='y', labelcolor='#c62828')
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Title and grid
        ax1.set_title('Evolucion de Inversion vs Reduccion de Inundacion\nAnalisis Costo-Beneficio por Paso',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.grid(True, linestyle='--', color='#cccccc', alpha=0.4, axis='y', linewidth=1)
        ax1.set_axisbelow(True)
        
        # Summary box
        initial_flooding = self.df['flooding_volume'].iloc[0] + self.df['flooding_reduction'].iloc[0] if 'flooding_volume' in self.df.columns else reduction_total[-1]
        final_reduction_pct = (reduction_total[-1] / initial_flooding * 100) if initial_flooding > 0 else 0
        
        summary = (
            f"=== RESUMEN ===\n"
            f"Pasos: {len(steps)}\n"
            f"Inversion Final: {self.format_currency_smart(cost_total[-1])}\n"
            f"Reduccion Total: {reduction_total[-1]:,.0f} m3\n"
            f"Reduccion: {final_reduction_pct:.1f}%\n"
            f"Eficiencia Prom: ${cost_total[-1]/reduction_total[-1]:,.0f}/m3"
        )
        
        ax1.text(0.02, 0.98, summary, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffacd',
                         alpha=0.95, edgecolor='#daa520', linewidth=2),
                fontfamily='monospace')
        
        # Combined legend - center top
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if flooding_remaining is not None:
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                      loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3,
                      fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
        else:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                      bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        save_path = self.output_dir / "01_cost_reduction_evolution.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
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
        cost_total = self.df['cost_total'].values
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14),
                                         gridspec_kw={'height_ratios': [1, 1]})
        
        # ===== TOP PLOT: Cumulative ROI =====
        line = ax1.plot(steps, roi, 'o-', color='#1976d2', linewidth=5,
                        markersize=18, markeredgecolor='#0d47a1', markeredgewidth=3,
                        label='ROI Acumulado')
        
        # Fill under curve
        ax1.fill_between(steps, 0, roi, alpha=0.25, color='#64b5f6')
        
        # Add large value labels
        for step, r in zip(steps, roi):
            ax1.annotate(f'{r:,.0f}\nm³/$1M', xy=(step, r), xytext=(0, 15),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='#0d47a1',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 alpha=0.95, edgecolor='#1976d2', linewidth=2))
        
        # Styling
        ax1.set_ylabel('m³ Reducidos por cada $1 Millón', fontsize=18, fontweight='bold', labelpad=15)
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
        ax2.set_xticklabels([f'Paso {s}' for s in steps], fontsize=13, fontweight='bold')
        
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


