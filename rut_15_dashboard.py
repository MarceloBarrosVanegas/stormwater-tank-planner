
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
    Generates evolution plots (07_1 to 07_4) and spatial reports 
    for the sequential optimization process.
    """
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        self.df = results_df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.max_open_warning': 0})
        
    def generate_all(self):
        """Generates all dashboard plots."""
        if self.df.empty:
            print("  [Dashboard] No data to plot.")
            return

        print("  [Dashboard] Generating evolution plots...")
        self.plot_07_1_cost_evolution()
        self.plot_07_2_network_health()
        self.plot_07_3_risk_ledger()
        self.plot_07_4_marginal_efficiency()
        
    def format_currency_smart(self, x, pos):
        """Dynamic formatting for large currency values."""
        if x >= 1e6:
            return f'${x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'${x*1e-3:.0f}K'
        else:
            return f'${x:.0f}'

    def plot_07_1_cost_evolution(self):
        """07_1: Cost Composition Evolution (Stacked Area)."""
        if self.df['step'].nunique() <= 1: return # Skip if only 1 point

        fig, ax = plt.subplots(figsize=(12, 7))
        
        iterations = self.df['step']
        
        # Ensure new column exists
        if 'cost_replacements' not in self.df.columns:
            self.df['cost_replacements'] = 0.0

        costs = {
            'cost_land': self.df['cost_land'],
            'cost_replacements': self.df['cost_replacements'],
            'cost_links': self.df['cost_links'],
            'cost_tanks': self.df['cost_tanks']
        }
        
        pal = sns.color_palette("viridis", 4)
        # Plot Stacked Area
        stacks = ax.stackplot(iterations, costs.values(), labels=costs.keys(), colors=pal, alpha=0.9)
        
        ax.set_title('07.1 Evolución de Inversión Acumulada', fontsize=14, fontweight='bold')
        ax.set_xlabel('Secuencia de Tanques', fontsize=12)
        ax.set_ylabel('Costo Total ($)', fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
        ax.legend(loc='upper left', title="Inversión", frameon=True, framealpha=0.9)
        ax.set_xlim(left=iterations.min(), right=iterations.max())
        
        # --- Add Labels at Every Step (Tank) ---
        # User requested: "Con cada tanque poner el valor porcentual en cada vértice"
        
        for i, step in enumerate(iterations):
            total_at_step = self.df['cost_total'].iloc[i]
            y_bottom = 0
            
            # Add Total Label at the top
            ax.text(step, total_at_step, self.format_currency_smart(total_at_step, 0), 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
            
            # Iterate components to place % labels
            for label, series in costs.items():
                val = series.iloc[i]
                if val > 0:
                    pct = (val / total_at_step) * 100
                    y_center = y_bottom + (val / 2)
                    
                    # Only label if segment is large enough (>2% of plot height approx) to be readable
                    if pct > 2: # Threshold lowered to show smaller components like Rehab
                        # Simplified label: just % inside the bar
                        # User requested BLACK text
                        # Adding a light white halo/box to ensure it's readable on dark segments
                        ax.text(step, y_center, f'{pct:.0f}%', 
                                ha='center', va='center', color='black', fontweight='bold', fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
                    
                    y_bottom += val
            
            # Vertical line for clarity
            ax.axvline(x=step, color='white', linestyle='-', alpha=0.3)

        # Custom integer ticker
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Vertical grid lines (Dark Gray, Dashed)
        ax.grid(axis='x', color='#555555', linestyle='--', alpha=0.4)

        save_path = self.output_dir / "07_1_cost_evolution.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def plot_07_2_network_health(self):
        """07_2: Cost Efficiency Curve (Cost vs Reduction)."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Data Preparation
        initial_flood = self.df['flooding_volume'].max()
        # Cumulative reduction = Initial - Current. (Assuming first step is T1)
        # However, usually there's a base scenario (step 0).
        # We'll assume the dataframe contains the optimization steps.
        # Reduction relative to the *start of this dataframe* or max found?
        # Let's use max as proxy for "No Tank" if not present.
        
        cumulative_reduction = initial_flood - self.df['flooding_volume']
        total_costs = self.df['cost_total']
        steps = self.df['step']
        
        # Plot Line
        ax.plot(cumulative_reduction, total_costs, marker='o', linestyle='-', color='navy', linewidth=2, alpha=0.7)
        
        # Plot Scatter Points
        ax.scatter(cumulative_reduction, total_costs, color='#2E86C1', s=100, zorder=5)
        
        # Labels for each step (T1, T2...)
        for i, txt in enumerate(steps):
            ax.annotate(f'T{int(txt)}', 
                        (cumulative_reduction.iloc[i], total_costs.iloc[i]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333333')
            
        # Axis Labels & Formatting
        ax.set_title('07.2 Curva de Eficiencia: Costo vs Reducción de Inundación', fontsize=14, fontweight='bold')
        ax.set_xlabel('Volumen de Inundación Eliminado (m³)', fontsize=12)
        ax.set_ylabel('Costo Total de Inversión ($)', fontsize=12)
        
        ax.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Grid
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Limits (optional, just to ensure breathing room)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)

        save_path = self.output_dir / "07_2_network_health.png" # Keeping filename for consistency with pipeline
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def plot_07_3_risk_ledger(self):
        """07_3: Total Social Cost (Zoomed)."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        steps = self.df['step']
        tsc = self.df['total_social_cost']
        
        # Main Line
        ax.plot(steps, tsc, color='navy', linewidth=3, label='Costo Social Total (Inversión + Riesgo)')
        
        min_idx = tsc.idxmin()
        min_step = steps.iloc[min_idx]
        min_val = tsc.iloc[min_idx]
        
        # Highlight Optimum
        ax.plot(min_step, min_val, marker='*', markersize=20, color='gold', markeredgecolor='black', label='Óptimo Económico')
        
        ax.annotate(f'Óptimo: Iter {min_step}\n{self.format_currency_smart(min_val,0)}', 
                   xy=(min_step, min_val), 
                   xytext=(0, 40), textcoords='offset points',
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.9))

        ax.set_title('07.3 The Risk Ledger: Búsqueda del Óptimo Económico', fontsize=14, fontweight='bold')
        ax.set_ylabel('Costo Social ($)', fontsize=12)
        ax.set_xlabel('Iteración', fontsize=12)
        ax.yaxis.set_major_formatter(FuncFormatter(self.format_currency_smart))
        
        # Force integer x-axis & grid
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='x', color='#555555', linestyle='--', alpha=0.4)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend()
        
        y_min, y_max = tsc.min(), tsc.max()
        margin = (y_max - y_min) * 0.3
        if margin == 0: margin = y_max * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

        save_path = self.output_dir / "07_3_risk_ledger.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def plot_07_4_marginal_efficiency(self):
        """07_4: Marginal Benefit (Bars) vs B/C Ratio (Line)."""
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        steps = self.df['step']
        reductions = self.df['flooding_reduction'].diff().fillna(self.df['flooding_reduction'].iloc[0])
        costs_marginal = self.df['cost_total'].diff().fillna(self.df['cost_total'].iloc[0])
        
        FLOOD_VALUE = 1250 
        bc_ratios = (reductions * FLOOD_VALUE) / costs_marginal
        bc_ratios = bc_ratios.clip(lower=0, upper=20) 
        
        # Bar Chart
        color_bar = '#85c1e9' 
        bars = ax1.bar(steps, reductions, color=color_bar, alpha=0.6, label='Reducción Marginal (m³)', edgecolor='white')
        ax1.set_xlabel('Secuencia de Tanques', fontsize=12)
        ax1.set_ylabel('Inundación Eliminada por Paso (m³)', color='#2e86c1', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#2e86c1')
        
        # Top Performers
        top_indices = reductions.nlargest(3).index
        for idx in top_indices:
            if idx < len(bars):
                bars[idx].set_color('#2e86c1') 
                bars[idx].set_edgecolor('navy')
                height = reductions.iloc[idx]
                ax1.text(steps.iloc[idx], height, f'{int(height)}', ha='center', va='bottom', fontsize=9, color='navy', fontweight='bold')
        
        # Line Chart
        ax2 = ax1.twinx()
        color_line = '#e74c3c'
        ax2.plot(steps, bc_ratios, color=color_line, marker='o', linewidth=2.5, label='Ratio Beneficio/Costo Marginal')
        ax2.set_ylabel('Ratio B/C Marginal', color=color_line, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_line)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Umbral B/C=1')
        
        # Force integer x-axis & grid
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.grid(axis='x', color='#555555', linestyle='--', alpha=0.4)
        
        ax1.set_title('07.4 Eficiencia Marginal: "First Mover Advantage"', fontsize=14, fontweight='bold')
        
        lines2, labels2 = ax2.get_legend_handles_labels()
        import matplotlib.patches as mpatches
        patch = mpatches.Patch(color='#2e86c1', label='Volumen Eliminado')
        ax2.legend([patch] + lines2, ['Volumen Eliminado'] + labels2, loc='upper right')
        
        save_path = self.output_dir / "07_4_marginal_efficiency.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    @staticmethod
    def export_spatial_state(iteration, metrics, output_dir):
        """
        Exports the current network state (Nodes/Links) to GPKG.
        Includes health metrics: utilization, surcharge status.
        """
        pass # To be implemented or used inline in optimizer

if __name__ == "__main__":
    print("Running Dashboard Generator in Standalone Mode...")
    
    # Create mock data mimicking the optimization results
    mock_data = {
        'step': [1, 2, 3, 4, 5],
        
        'cost_land':      [1000000, 2000000, 3000000, 4000000, 5000000],
        'cost_tanks':     [1000000, 1200000, 1500000, 2000000, 2300000],
        'cost_links':     [3000000, 6000000, 7500000, 8500000, 9500000],
        'cost_replacements': [500000, 750000, 900000, 1100000, 1300000],
        
        
        
        'flooding_volume': [160000, 120000, 80000, 50000, 30000],
        'flooding_reduction': [0, 40000, 80000, 110000, 130000],
        
        
        'overloaded_links_length': [5000, 4500, 3000, 2500, 2000],
        'residual_damage_usd': [20000000, 15000000, 10000000, 6250000, 3750000]
    }
    
    df_mock = pd.DataFrame(mock_data)
    df_mock['cost_total'] = df_mock['cost_land'] + df_mock['cost_links'] + df_mock['cost_tanks'] + df_mock['cost_replacements']
    df_mock['total_social_cost'] = df_mock['cost_total'] + df_mock['residual_damage_usd']
    
    # Init Generator with test output folder
    output_path = Path("test_dashboard_output")
    output_path.mkdir(exist_ok=True)
    
    gen = EvolutionDashboardGenerator(df_mock, output_path)
    
    # Test specific plot as requested
    print("Generating 07.1 Cost Evolution...")
    gen.plot_07_1_cost_evolution()
    
    print("Generating 07.2 Efficiency Curve...")
    gen.plot_07_2_network_health()
    
    # gen.plot_07_3_risk_ledger()
    # gen.plot_07_4_marginal_efficiency()
    
    print(f"Done. Check {output_path.absolute()}")
