import os
import sys
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.ops import unary_union
from matplotlib.lines import Line2D
from typing import List, Dict
from pathlib import Path
from line_profiler import profile
import json
import config
PLOT_TIME_LIMIT_HOURS = config.ITZI_SIMULATION_DURATION_HOURS
from rut_27_model_metrics import SystemMetrics

plt.style.use('ggplot')


class ScenarioComparator:
    """
    Compares two SystemMetrics (Baseline vs Solution) and generates reports/plots.
    """
    def __init__(self, baseline_metrics: SystemMetrics, baseline_inp_path: str = None):
        self.baseline = baseline_metrics
        self.baseline_inp_path = baseline_inp_path
        self._predios_gdf_cache = None  # Cache for lazy loading
        
    @profile
    def generate_comparison_plots(self,
                                  solution: SystemMetrics,
                                  solution_name: str,
                                  save_dir: Path,
                                  nodes_gdf: gpd.GeoDataFrame,
                                  derivations: gpd.GeoDataFrame = None,
                                  detailed_links: Dict=None,
                                  tank_details: List[Dict] = None,
                                  show_predios: bool = False,
                                  ):

        """Generates all comparison plots for a given solution."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Combined Map (Spatial + Scatter + Hydro + Hists + Tank Table)
        self.generate_combined_map(solution=solution,
                                   save_dir=save_dir,
                                   nodes_gdf=nodes_gdf,
                                   derivations=derivations,
                                   tank_details=tank_details,
                                   show_predios=show_predios)
        


        # 1.1 Capacity Comparison Maps (Baseline, Solution, Delta) - Single 2x2 figure
        self.generate_capacity_comparison_maps(solution,
                                               solution_name,
                                               save_dir,
                                               nodes_gdf=nodes_gdf,
                                               derivations=derivations,
                                               tank_details=tank_details,
                                               show_predios=show_predios)


        # 1.2 Velocity Comparison Maps (Baseline, Solution, Delta) - Single 2x2 figure
        self.generate_velocity_comparison_maps(solution,
                                               solution_name,
                                               save_dir,
                                               nodes_gdf=nodes_gdf,
                                               show_predios=show_predios,
                                               derivations=derivations,
                                               tank_details=tank_details)


        # 2. Hydrographs (3x3 grid)
        if len(detailed_links) > 0:
            self.generate_hydrograph_pages(solution, solution_name, save_dir, detailed_links)
            # 2.1 Tank Hydrographs (Specific Inflow/Weir plots)
            self.generate_tank_hydrograph_plots(solution, solution_name, save_dir, detailed_links)

        # # # 3. Longitudinal Profiles (3 per page)
        # if designed_gdf is not None:
        #      self.generate_profile_plots(solution, solution_name, save_dir, designed_gdf)

        # 4. Unified Statistical Dashboard (Depth, Flooding, Capacity)
        self.generate_unified_statistical_dashboard(solution, solution_name, save_dir)
        


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    @profile
    def generate_combined_map(self,
                              solution: SystemMetrics,
                              save_dir: Path,
                              nodes_gdf: gpd.GeoDataFrame,
                              derivations: gpd.GeoDataFrame,
                              tank_details: List[Dict] = None,
                              show_predios: bool = False):

        """
        Dashboard Layout:
        ┌─────────┬──────────────────────┬─────────────────────────┐
        │  Table  │        Map           │  Scatter  │ Cumulative  │
        │         │                      ├───────────┼─────────────┤
        │         │                      │  Hydro    │  Histogram  │
        └─────────┴──────────────────────┴─────────────────────────┘
        """
        fig = plt.figure(figsize=(22, 9))
        # Grid: 2 rows, 4 cols - Table | Map | 4 plots in 2x2 (last 2 columns)
        gs = fig.add_gridspec(2, 4, width_ratios=[0.6, 2.0, 0.8, 0.8])

        # === LEFT: TANK TABLE (col 0, rows 0-1) ===
        ax_table = fig.add_subplot(gs[:, 0])
        ax_table.axis('off')
        ax_table.set_title('Tank Details', fontsize=12, fontweight='bold', pad=5)

        # Build table data - compact format
        # L in km, Vol = actual volume rounded, Util% with warning color if low
        table_headers = ['#', 'Predio', 'Q\n[m³/s]', 'L\n[km]', 'D\n[m]', 'Volume\n[m³]', 'Área\n[m²]', '%Uso\nPredio']
        table_data = []
        tank_positions = {}  # To store positions for map labels
        low_util_rows = []  # Track rows with low utilization

        # Get config values
        if tank_details:
            for i, tank in enumerate(tank_details):
                # Support both dataclass (CandidatePair) and dict access
                def get_val(obj, key, default):
                    if hasattr(obj, key):
                        return getattr(obj, key, default)
                    elif hasattr(obj, 'get'):
                        return obj.get(key, default)
                    return default
                
                node_id = get_val(tank, 'node_id', '?')
                predio_id = get_val(tank, 'predio_id', '')
                longitud_km = get_val(tank, 'pipeline_length', 0) / 1000
                diametro = get_val(tank, 'diameter', 'N/A')
                predio_area = get_val(tank, 'predio_area', 0)
                
                # Get Q and Volume from solution.tank_data (simulation results)
                tank_id = f"tank_{predio_id}"
                if tank_id in solution.tank_data:
                    q_derivacion = solution.tank_data[tank_id]['max_flow']
                    vol_stored = solution.tank_data[tank_id]['max_stored_volume']
                else:
                    q_derivacion = get_val(tank, 'total_flow', 0)
                    vol_stored = get_val(tank, 'tank_volume_simulation', 0)
                
                # Calculate tank area from stored volume (same formula as rut_15_optimizer)
                tank_area = (vol_stored / config.TANK_DEPTH_M) + config.TANK_OCCUPATION_FACTOR
                percentage_used_area = (tank_area / predio_area * 100) if predio_area > 0 else 0.0

                # Track low utilization (if tank uses > 80% of predio)
                if percentage_used_area > 80:
                    low_util_rows.append(i)

                table_data.append([
                    str(i),
                    predio_id,
                    f"{q_derivacion:.2f}",
                    f"{longitud_km:.2f}",
                    f"{float(diametro):.2f}" if str(diametro).replace('.', '', 1).isdigit() else (
                        f"{float(str(diametro).split('x')[0]):.2f}x{float(str(diametro).split('x')[1]):.2f}"
                        if 'x' in str(diametro).lower() else str(diametro)[:9]
                    ),
                    f"{vol_stored:,.0f}",
                    f"{tank_area:,.0f}",
                    f"{percentage_used_area:.1f}%",
                ])
                tank_positions[node_id] = i  # Map node to number

        if table_data:
            table = ax_table.table(cellText=table_data,
                                   colLabels=table_headers,
                                   loc='center',
                                   cellLoc='center',
                                   colWidths=[0.15, 0.20, 0.20, 0.20, 0.25, 0.20, 0.20, 0.20])

            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.6)

            # Style header row
            for j in range(len(table_headers)):
                table[(0, j)].set_facecolor('#3498db')
                table[(0, j)].set_text_props(color='white', fontweight='bold')

        else:
            ax_table.text(0.5, 0.5, 'No tank data\navailable', ha='center', va='center', fontsize=11, color='gray', transform=ax_table.transAxes)

        # === spatial map (col 1, rows 0-1) ===
        ax_map = fig.add_subplot(gs[:, 1])
        self._plot_spatial_diff(ax_map, nodes_gdf, self.baseline.node_data, solution.node_data, derivations, show_predios=show_predios)

        # Add numbered labels and derivation lines on map (Dashboard style)
        self._plot_derivations_and_labels(ax_map, derivations, tank_details)

        # === RIGHT: 2x2 PLOTS (cols 2-3) ===
        # Row 0, Col 2: Scatter Plot
        ax_scatter = fig.add_subplot(gs[0, 2])
        self._plot_volume_scatter(ax_scatter, self.baseline.node_data, solution.node_data)

        # Row 0, Col 3: Cumulative Volume
        ax_cumulative = fig.add_subplot(gs[0, 3])
        self._plot_cumulative_volume(ax_cumulative, self.baseline, solution)

        # Row 1, Col 2: System Flooding Hydrograph
        ax_flood = fig.add_subplot(gs[1, 2])
        self._plot_system_flood_hydrograph(ax_flood, self.baseline, solution)

        # Row 1, Col 3: Outfall Flow Hydrograph
        ax_outfall = fig.add_subplot(gs[1, 3])
        self._plot_outfall_hydrograph(ax_outfall, self.baseline, solution)

        plt.tight_layout()
        plt.savefig(save_dir / "00_dashboard_map.png", dpi=100)
        plt.close(fig)

    @profile
    def _plot_spatial_diff(self, ax, nodes_gdf: gpd.GeoDataFrame, base_vols: Dict, sol_vols: Dict, derivations: gpd.GeoDataFrame, show_predios: bool = False):
        """
        Maps delta flooding + Background Network + New Derivations - Vectorized.
        """

        # 0. Optional Predios Background (Lazy loaded)
        if show_predios:
            predios_gdf = self._get_predios_gdf()
            if predios_gdf is not None and not predios_gdf.empty:
                predios_gdf.plot(ax=ax, color='#f5f5f5', edgecolor='#e0e0e0', linewidth=0.1, zorder=0)

        # 1. Background Network
        net_gdf = self.baseline.swmm_gdf.copy()
        net_gdf.plot(ax=ax, color='#222222', linewidth=1.2, alpha=0.6, zorder=1)

        if not derivations.empty:
            derivations.plot(ax=ax, color='blue', linewidth=2.5, alpha=0.8, zorder=5)

        # 3. Nodes (Colored by Delta) - Vectorized
        base_df = pd.DataFrame.from_dict(base_vols, orient='index')[['flooding_volume']]
        sol_df = pd.DataFrame.from_dict(sol_vols, orient='index')[['flooding_volume']]

        gdf = nodes_gdf.copy()
        gdf['NodeID_s'] = gdf['node_id'].astype(str)

        # Merge metrics
        gdf = gdf.merge(base_df, left_on='NodeID_s', right_index=True, how='left')
        gdf = gdf.merge(sol_df, left_on='NodeID_s', right_index=True, how='left')

        gdf['v_base'] = gdf['flooding_volume_x'].fillna(0)
        gdf['v_sol'] = gdf['flooding_volume_y'].fillna(0)
        gdf['delta'] = gdf['v_base'] - gdf['v_sol']

        # Vectorized color logic
        gdf['color'] = '#444444'  # Dark gray for insignficant change
        gdf.loc[gdf['delta'] > 1.0, 'color'] = 'green'
        gdf.loc[gdf['delta'] < -1.0, 'color'] = 'red'

        # Vectorized size logic
        gdf['markersize'] = 5
        gdf.loc[gdf['delta'] > 1.0, 'markersize'] = 15 + gdf['delta'].clip(0, 100) * 0.5
        # Note: delta < -1.0 means sol > base (worsened)
        gdf.loc[gdf['delta'] < -1.0, 'markersize'] = 30 + gdf['delta'].abs().clip(0, 100) * 0.5

        # Plot nodes layer
        gdf.plot(ax=ax, color=gdf['color'], markersize=gdf['markersize'], alpha=0.7, zorder=2)

        # Set limits based on Network + Derivations (ignoring huge predios background)
        bounds_geoms = []
        if net_gdf is not None and not net_gdf.empty:
            bounds_geoms.append(net_gdf.unary_union)
        elif nodes_gdf is not None and not nodes_gdf.empty:  # Fallback to filtered nodes if no net
            bounds_geoms.append(nodes_gdf.unary_union)

        geoms = list(derivations.geometry) if hasattr(derivations, "geometry") else list(derivations)
        bounds_geoms.extend(geoms)

        # Calculate the total bounding box of the network + new derivations
        unified = unary_union(bounds_geoms)
        minx, miny, maxx, maxy = unified.bounds

        margin = 50  # meters margin
        ax.set_xlim(minx - margin, maxx + margin)  # Now properly setting X limits
        ax.set_ylim(miny - margin, maxy + margin)
        ax.set_aspect('equal')

        ax.set_title("Spatial Flooding Delta (Green=Improved, Red=Worsened)")
        ax.set_axis_off()

    @staticmethod
    def get_cumulative(ax, metrics_obj, color, label):
        data = getattr(metrics_obj, "system_flood_hydrograph", None)
        if not data or "total_rate" not in data or "times" not in data:
            return None, None

        rates = np.asarray(data["total_rate"], dtype=float)  # m³/s
        times = np.asarray(data["times"])

        if len(times) < 2 or len(rates) < 2:
            return None, None

        # Convert times -> hours since start, and dt seconds between samples
        t0 = times[0]

        if isinstance(t0, pd.Timestamp):
            t_idx = pd.to_datetime(times)
            hrs = ((t_idx - t_idx[0]).total_seconds().to_numpy()) / 3600.0
            dt_secs = t_idx.to_series().diff().dt.total_seconds().fillna(0.0).to_numpy()
        else:
            # numpy datetime64 (or anything convertible to it)
            t64 = times.astype("datetime64[ns]")
            diffs_sec = (t64 - t64[0]).astype("timedelta64[s]").astype(float)
            hrs = diffs_sec / 3600.0
            dt_secs = np.empty_like(diffs_sec)
            dt_secs[0] = 0.0
            dt_secs[1:] = np.diff(diffs_sec)

        # Rectangular integration using previous sample rate:
        # volume[i] = volume[i-1] + rates[i-1] * dt[i]
        cumulative = np.zeros(len(rates), dtype=float)
        cumulative[1:] = np.cumsum(rates[:-1] * dt_secs[1:])

        ax.plot(
            hrs,
            cumulative,
            linestyle="-" if "Solution" in label else "--",
            color=color,
            alpha=0.8,
            linewidth=2,
            label=label,
        )

        # Final value annotation
        final_vol = cumulative[-1]
        ax.annotate(
            f"{final_vol:,.0f} m³",
            xy=(hrs[-1], final_vol),
            xytext=(-50, 5),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

        return hrs, cumulative

    def _plot_cumulative_volume(self, ax, baseline, solution):
        """Plots cumulative flooding volume over time (integral of rate)."""

        hrs_b, cum_b = self.get_cumulative(ax, baseline, "red", "Baseline")
        hrs_s, cum_s = self.get_cumulative(ax, solution, "blue", "Solution")

        ax.set_title("Cumulative Flooding Volume Over Time")
        ax.set_ylabel("Cumulative Volume (m³)")
        ax.set_xlabel("Time (hours)")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        if cum_b is not None and cum_s is not None and cum_b[-1] > 0:
            reduction = cum_b[-1] - cum_s[-1]
            pct = (reduction / cum_b[-1]) * 100.0
            ax.text(
                0.98,
                0.35,
                f"Reduction: {reduction:,.0f} m³ ({pct:.1f}%)",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
            )

    @staticmethod
    def plot_series(ax, metrics_obj, color, label):
            data = metrics_obj.system_flood_hydrograph
                
            y = np.array(data['total_rate'])
            times = np.array(data['times'])
            
            # Vectorized time conversion
            hrs = np.array([])
            if len(times) > 0:
                t0 = times[0]
                if isinstance(t0, (pd.Timestamp, pd.DatetimeIndex)) or hasattr(t0, 'total_seconds'):
                    # Handle both pandas/datetime objects
                    hrs = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
                elif isinstance(t0, np.datetime64):
                    # Fast numpy datetime64 conversion
                    hrs = (times - t0).astype('timedelta64[s]').astype(float) / 3600.0
                else:
                    # Fallback if already numeric
                    hrs = np.array(times)

            ax.plot(hrs, y, linestyle='-' if label=='Solution' else '--', color=color, alpha=0.8, label=label)
            
            # Peak Label
            if len(y) > 0:
                ymax = np.max(y)
                if ymax > 0:
                    idx = np.argmax(y)
                    ax.annotate(f'{ymax:.2f}', xy=(hrs[idx], ymax), xytext=(0, 5),
                                textcoords="offset points", color=color, fontsize=9, fontweight='bold')
            return hrs, y
    
    def _plot_system_flood_hydrograph(self, ax, baseline, solution):
        """Sums up flooding rates (cms) for extracted nodes to show system stress."""
        


        # Plot
        t_b, y_b = self.plot_series(ax, baseline, 'red', 'Baseline')
        t_s, y_s = self.plot_series(ax, solution, 'blue', 'Solution')

        ax.set_title("Total System Flooding Rate (All Nodes)")
        ax.set_ylabel("Flooding Rate (cms)")
        ax.set_xlabel("Time (hours)")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
        ax.legend()
        ax.grid(True, alpha=0.3)
        

        # Explicit Difference Verification for User
        if y_b is not None and y_s is not None:
             # Use PRE-CALCULATED volumes from metrics to ensure consistency with report
             vol_b_reported = baseline.total_flooding_volume
             vol_s_reported = solution.total_flooding_volume
             
             # Also calculate INTEGRATED volume from the plot data to check for consistency
             # Assuming x_b is in hours, convert to seconds
             # Simple trapz integration
             try:
                 # Convert hours to seconds
                 t_sec = np.array(t_b) * 3600.0
                 vol_b_integ = np.trapz(y_b, t_sec)
                 
                 t_sec_s = np.array(t_s) * 3600.0
                 vol_s_integ = np.trapz(y_s, t_sec_s)
             except:
                 vol_b_integ = 0
                 vol_s_integ = 0
             
             flow_diff = round(baseline.total_max_flooding_flow - solution.total_max_flooding_flow, 2)
             
             # Print STATS on the plot (Showing match/mismatch)
             stats_text = (f"Reduction: {flow_diff:,.2f} m3/s ({(1 - solution.total_max_flooding_flow / baseline.total_max_flooding_flow) * 100:,.2f}%)")
             
             ax.text(0.98, 0.35, stats_text, transform=ax.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='black', alpha=0.9))

    @staticmethod
    def _plot_hist(ax, data_base, data_sol, xlabel):
        if not data_base and not data_sol: return
        # Use KDE lines with transparent fill
        if data_base:
            sns.kdeplot(data_base, ax=ax, color='red', linewidth=2, label='Baseline', fill=True, alpha=0.15)
        if data_sol:
            sns.kdeplot(data_sol, ax=ax, color='blue', linewidth=2, label='Solution', fill=True, alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.set_title(f"Distribution: {xlabel}")

    def _plot_outfall_hydrograph(self, ax, baseline, solution):
        """Plots outfall flow comparison showing lamination effect of tanks."""
        
        def plot_series(metrics_obj, color, label, linestyle):
            data = getattr(metrics_obj, "system_outfall_flow_hydrograph", None)
            if not data or "total_rate" not in data or "times" not in data:
                return None, None
            
            rates = np.asarray(data["total_rate"], dtype=float)
            times = np.asarray(data["times"])
            
            if len(times) < 2 or len(rates) < 2:
                return None, None
            
            t0 = times[0]
            if isinstance(t0, pd.Timestamp):
                t_idx = pd.to_datetime(times)
                hrs = ((t_idx - t_idx[0]).total_seconds().to_numpy()) / 3600.0
            else:
                t64 = times.astype("datetime64[ns]")
                diffs_sec = (t64 - t64[0]).astype("timedelta64[s]").astype(float)
                hrs = diffs_sec / 3600.0
            
            ax.plot(hrs, rates, linestyle=linestyle, color=color, alpha=0.8, linewidth=2, label=label)
            
            # Peak annotation
            if len(rates) > 0:
                peak_val = np.max(rates)
                if peak_val > 0:
                    peak_idx = np.argmax(rates)
                    ax.annotate(f'{peak_val:.2f}', xy=(hrs[peak_idx], peak_val), 
                               xytext=(0, 5), textcoords="offset points",
                               color=color, fontsize=9, fontweight='bold')
            
            return hrs, rates
        
        hrs_b, rates_b = plot_series(baseline, 'red', 'Baseline', '--')
        hrs_s, rates_s = plot_series(solution, 'blue', 'Solution', '-')
        
        ax.set_title("Outfall Flow ")
        ax.set_ylabel("Flow Rate (m³/s)")
        ax.set_xlabel("Time (hours)")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Show peak reduction
        if rates_b is not None and rates_s is not None:
            peak_b = np.max(rates_b)
            peak_s = np.max(rates_s)
            reduction = peak_b - peak_s
            pct = (reduction / peak_b) * 100 if peak_b > 0 else 0
            
            ax.text(0.98, 0.35, f"Reduction: {reduction:.2f} m³/s ({pct:.1f}%)",
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='black', alpha=0.9))

    @staticmethod
    def _plot_derivations_and_labels(ax, derivations, tank_details=None):
        """Helper to plot blue derivation lines and numbered labels (Dashboard style)."""
        # 1. Blue Lines (Vectorized)
        derivations.plot(ax=ax, color='blue', linewidth=2.5, alpha=0.9, zorder=5)

        # 2. Numbered Labels & Predio Labels
        for r_id, df in derivations.groupby('Ramal'):
            row = df.iloc[0]
            target_node_metadata = json.loads(row.get('Obs', '').split('|')[1])
            target_type =  target_node_metadata['target_type']

            # First point (Diversion Node)
            start_pt = np.array(df.geometry.iloc[0].xy).T[0]

            # Plot Ramal ID (Blue Circle)
            ax.annotate(str(r_id), xy=start_pt,
                        fontsize=10, fontweight='bold', color='white',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.2',
                                  facecolor='blue',
                                  edgecolor='white',
                                  linewidth=0.5,
                                  alpha=1.0),
                        zorder=10)
            
            # Plot Predio '0' (Black Circle) - only if it's a tank
            if target_type == 'tank':
                # Last point of the last segment in this ramal
                end_pt = np.array(df.geometry.iloc[-1].xy).T[-1]
                ax.annotate('t', xy=end_pt,
                            fontsize=8, fontweight='bold', color='white',
                            ha='center', va='center',
                            bbox=dict(boxstyle='circle,pad=0.2',
                                      facecolor='black',
                                      edgecolor='black',
                                      linewidth=0.25,
                                      alpha=1.0),
                            zorder=10)
            else:
                # print(f"  [Plot-Debug] Ramal {r_id}: Skipping '0' label (is a node connection)")
                pass




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    def generate_profile_plots(self, metrics: SystemMetrics, sol_name: str, save_dir: Path, designed_gdf: gpd.GeoDataFrame):
        """Generates SWMM-style hydraulic longitudinal profiles for the new derivations."""
        print(f"[Reporter] Generating SWMM-Style Profile Plots for {sol_name}...")
        
        if 'Ramal' not in designed_gdf.columns:
            print("  [Error] designed_gdf missing 'Ramal' column.")
            return

        ramals = sorted(designed_gdf['Ramal'].unique())
        plots_per_page = 3
        
        for i in range(0, len(ramals), plots_per_page):
            chunk = ramals[i:i+plots_per_page]
            
            fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 4*len(chunk)))
            if len(chunk) == 1: axes = [axes]
            
            for j, r_id in enumerate(chunk):
                ax = axes[j]
                
                # Get and sort pipes
                pipes = designed_gdf[designed_gdf['Ramal'] == r_id].copy()
                
                def get_sort_key(tramo):
                    try:
                        return float(str(tramo).split('-')[0])
                    except:
                        return 0.0
                
                pipes['sort_key'] = pipes['Tramo'].apply(get_sort_key)
                pipes = pipes.sort_values('sort_key')
                
                # Build profile arrays
                x_positions = [0.0]  # Cumulative distance
                z_invert_start = []
                z_invert_end = []
                diameters = []
                node_labels = []
                
                # Get first node label
                if len(pipes) > 0:
                    first_parts = str(pipes.iloc[0]['Tramo']).split('-')
                    node_labels.append(first_parts[0] if len(first_parts) > 0 else "?")
                
                dist_accum = 0.0
                curr_z = None
                
                for idx, row in pipes.iterrows():
                    l = float(row.L) if hasattr(row, 'L') and not pd.isna(row.L) else 10.0
                    s = float(row.S) if hasattr(row, 'S') and not pd.isna(row.S) else 0.01
                    
                    # Get diameter (or section height)
                    d = 0.3  # Default
                    if hasattr(row, 'D_int') and not pd.isna(row.D_int):
                        try:
                            d = row.D_ext
                        except:
                            d = 0.3
                    
                    # Initial Z from data or integrate
                    if curr_z is None:
                        if hasattr(row, 'ZFI') and not pd.isna(row.ZFI):
                            curr_z = float(row.ZFI)
                        elif hasattr(row, 'Z') and not pd.isna(row.Z):
                            curr_z = float(row.Z)
                        else:
                            curr_z = 2800.0  # Placeholder
                    
                    z_start = curr_z
                    z_end = z_start - (l * s)  # Slope integration
                    
                    z_invert_start.append(z_start)
                    z_invert_end.append(z_end)
                    diameters.append(d)
                    
                    dist_accum += l
                    x_positions.append(dist_accum)
                    
                    # Node label at end of pipe
                    parts = str(row.Tramo).split('-')
                    node_labels.append(parts[1] if len(parts) > 1 else "?")
                    
                    curr_z = z_end
                
                def is_original_node(nid):
                    """Check if node is part of original network (not tank or derivation)"""
                    # Tanks start with TK_
                    if nid.startswith('TK_'):
                        return False
                    # Derivation nodes are format: 0.0, 0.1, 1.0, etc.
                    parts = nid.split('.')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return False
                    return True
                # --- DRAW SWMM-STYLE PROFILE ---
                n_pipes = len(z_invert_start)
                
                for p_idx in range(n_pipes):
                    x_start = x_positions[p_idx]
                    x_end = x_positions[p_idx + 1]
                    z_bot_start = z_invert_start[p_idx]
                    z_bot_end = z_invert_end[p_idx]
                    d = diameters[p_idx]
                    
                    z_top_start = z_bot_start + d
                    z_top_end = z_bot_end + d
                    
                    # Draw pipe outline (bottom and top lines)
                    ax.plot([x_start, x_end], [z_bot_start, z_bot_end], 'k-', linewidth=3)
                    ax.plot([x_start, x_end], [z_top_start, z_top_end], 'k-', linewidth=3)
                    
                    # Draw manholes (vertical lines at nodes)
                    if p_idx == 0:
                        ax.plot([x_start, x_start], [z_bot_start - 0.5, z_top_start + 0.5], 'k-', linewidth=2)
                    ax.plot([x_end, x_end], [z_bot_end - 0.5, z_top_end + 0.5], 'k-', linewidth=2)
                
                # Draw HGL (Water Surface) as green line
                # Use max depth from metrics if available, otherwise assume 70% filling
                hgl_y = []
                # Additional: Add annotations for invert levels
                # We'll collect them to annotate later
                invert_annotations = [] # (x, y, text, type)
                
                for p_idx in range(n_pipes):
                    z_bot = z_invert_start[p_idx]
                    d = diameters[p_idx]
                    
                    n_start = node_labels[p_idx]
                    depth = metrics.node_data.get(n_start, {}).get('max_depth', d * 0.7)  # Default 70% fill
                    hgl_y.append(z_bot + min(depth, d))  # Cap at crown
                    
                    # Annotate Start Invert
                    # Invert labeling: vertically below the pipe start
                    invert_annotations.append((x_positions[p_idx], z_bot, f"HI:{z_bot:.2f}", 'HI'))
                    
                
                # Add final point
                if n_pipes > 0:
                    z_bot = z_invert_end[-1]
                    d = diameters[-1]
                    n_end = node_labels[-1]
                    depth = metrics.node_data.get(n_end, {}).get('max_depth', d * 0.7)
                    hgl_y.append(z_bot + min(depth, d))
                    # Annotate End Invert
                    invert_annotations.append((x_positions[-1], z_bot, f"HF:{z_bot:.2f}", 'HF'))
                
                ax.plot(x_positions[:len(hgl_y)], hgl_y, 'g-', linewidth=2, label='Water Surface (Max)')
                
                # Node labels (Vertical text above)
                y_max_plot = ax.get_ylim()[1]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                
                for idx, x_pos in enumerate(x_positions[:len(node_labels)]):
                     lbl = node_labels[idx]
                     # Position above the top of the plot area mostly, or just above the pipe crown
                     # Find crown at this x
                     if idx < len(z_invert_start):
                         z_crow = z_invert_start[idx] + diameters[idx]
                     else:
                         z_crow = z_invert_end[-1] + diameters[-1]
                         
                     ax.text(x_pos, z_crow + y_range*0.05, lbl, rotation=90, 
                             ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkblue')

                # Hydraulic Annotations (HI/HF)
                for x, z, txt, type_ in invert_annotations:
                    offset = -y_range*0.05
                    ax.text(x, z + offset, txt, rotation=90, 
                            ha='center', va='top', fontsize=7, color='black')

                # Title with flow if available
                total_length = x_positions[-1] if x_positions else 0
                title = f"Water Elevation Profile: Ramal {r_id} (L={total_length:.1f}m)"
                
                # Try to get flow for title
                if 'q_pluvial' in pipes.columns:
                    q_val = pipes['q_pluvial'].iloc[0]
                    if q_val and q_val > 0:
                        title += f" - Q: {q_val*1000:.1f} L/s"
                
                ax.set_title(title, fontsize=12, color='red', fontweight='bold')
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Elevation (m)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
                # Invert X-axis to match SWMM style (flow direction)
                ax.invert_xaxis()
                
            plt.tight_layout()
            out_path = save_dir / f"{sol_name}_profiles_page_{i//plots_per_page + 1}.png"
            plt.savefig(out_path, dpi=150, facecolor='white')
            plt.close(fig)
            print(f"  Saved: {out_path}")
    
    def generate_unified_statistical_dashboard(self, solution: SystemMetrics, solution_name: str, save_dir: Path):
        """Generates dashboard_statistical_combined.png: 4x2 layout with all stats."""
        fig = plt.figure(figsize=(16, 20)) # Taller figure for 4 rows
        gs = fig.add_gridspec(4, 2)
        
        # Prepare Data
        base_depths = self.base_depths = [__['max_depth'] for _, __ in self.baseline.node_data.items()]
        sol_depths = self.sol_depths = [__['max_depth'] for _, __ in solution.node_data.items()]
        
        base_vols = self.base_vols = [__['flooding_volume'] for _, __ in self.baseline.node_data.items()]
        sol_vols = self.sol_vols =  [__['flooding_volume'] for _, __ in solution.node_data.items()]
        
        base_caps = self.base_caps = [__['max_capacity'] for _, __ in self.baseline.link_data.items()]
        sol_caps = self.sol_caps = [__['max_capacity'] for _, __ in solution.link_data.items()]
        
        base_velocity = self.base_velocity = [__['max_velocity'] for _, __ in self.baseline.link_data.items()]
        sol_velocity = self.sol_velocity = [__['max_velocity'] for _, __ in solution.link_data.items()]
        
        
        # Row 1: Scalars (Left) | Absolute Comparison (Right)
        ax_scalar = fig.add_subplot(gs[0, 0])
        self._plot_scalar_comparison(ax_scalar, solution, solution_name) #ok
        
        ax_abs = fig.add_subplot(gs[0, 1])
        self._plot_absolute_summary(ax_abs, solution, solution_name) 

        
        # Row 2: Node Depths (Hist | ECDF)
        ax_depth_hist = fig.add_subplot(gs[1, 0])
        self._plot_hist(ax_depth_hist, base_depths, sol_depths, "Node Max Depths (m)")#ok
        
        ax_depth_ecdf = fig.add_subplot(gs[1, 1])
        self._plot_ecdf(ax_depth_ecdf, base_depths, sol_depths, "Node Max Depths (m)", "CDF")
        
        # Row 3: Flooding Vols (Hist | ECDF)
        ax_flood_hist = fig.add_subplot(gs[2, 0])
        self._plot_hist(ax_flood_hist, base_vols, sol_vols, "Flooding Vol (m3)")
        
        ax_flood_ecdf = fig.add_subplot(gs[2, 1])
        self._plot_ecdf(ax_flood_ecdf, base_vols, sol_vols, "Flooding Vol (m3)", "CDF")
        
        # Row 4: Pipe Capacity (Hist | ECDF)
        ax_cap_hist = fig.add_subplot(gs[3, 0])
        self._plot_hist(ax_cap_hist, base_caps, sol_caps, "Max Capacity (d/D)")
        
        ax_cap_ecdf = fig.add_subplot(gs[3, 1])
        self._plot_ecdf(ax_cap_ecdf, base_caps, sol_caps, "Max Capacity (h/D)", "CDF")
        
        fig.suptitle(f"Unified Statistical Analysis: {solution_name}", fontsize=18, y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(save_dir / "05_dashboard_statistical_combined.png", dpi=100)
        plt.close(fig)
        print("  [Stats] Saved dashboard_statistical_combined.png")






    def generate_metric_maps(self, solution: SystemMetrics,
                             solution_name: str,
                             save_dir: Path,
                             nodes_gdf: gpd.GeoDataFrame, ):
        """
        Generates spatial metric map files showing Baseline vs Solution comparison.
        Currently generates:
        1. dashboard_map_flooding.png - Node flooding volume comparison (spatial)
        
        Note: Capacity and Velocity maps require per-link data that isn't available
        from the current extraction. The existing dashboard_map.png and 
        dashboard_statistical.png already show these metrics in other formats.
        """
        
        # Use the nodes_gdf directly (same as main dashboard)
        if nodes_gdf is None or nodes_gdf.empty:
            print("  [MetricMaps] Warning: No nodes_gdf available, skipping metric maps.")
            return
        
        # 1. FLOODING MAP - Spatial comparison of node flooding volumes
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Plot background nodes
        nodes_gdf.plot(ax=ax, color='lightgray', markersize=5, alpha=0.3)
        
        # Plot flooding difference (Improved vs Worsened)
        base_vols = {nid: d.get('flooding_volume', 0) for nid, d in self.baseline.node_data.items()}
        sol_vols = {nid: d.get('flooding_volume', 0) for nid, d in solution.node_data.items()}
        
        # Debug: Verify data is actually different
        base_total = sum(base_vols.values())
        sol_total = sum(sol_vols.values())
        print(f"  [MetricMaps-DATA] Baseline total flood: {base_total:.1f} m3, Solution total flood: {sol_total:.1f} m3")
        
        improved_geoms = []
        worsened_geoms = []
        improved_sizes = []
        worsened_sizes = []
        
        # Iterate over all known node IDs from flooding volumes
        all_nids = set(base_vols.keys()) | set(sol_vols.keys())
        
        # Create a lookup from node ID to geometry using 'node_id' column (like _plot_spatial_diff)
        name_to_geom = {}
        for idx in nodes_gdf.index:
            row = nodes_gdf.loc[idx]
            geom = row['geometry']
            # Use 'node_id' column which contains names like P0070286
            if 'node_id' in nodes_gdf.columns:
                name_to_geom[str(row['node_id'])] = geom
            # Fallback to index
            name_to_geom[str(idx)] = geom
        
        for nid_str in all_nids:
            base_v = base_vols.get(nid_str, 0)
            sol_v = sol_vols.get(nid_str, 0)
            diff = sol_v - base_v
            
            if abs(diff) > 1:  # Threshold for visibility
                geom = name_to_geom.get(nid_str)
                if geom is None:
                    continue
                size = min(10 + abs(diff) * 0.05, 80)
                if diff < 0:  # Improved (reduced flooding)
                    improved_geoms.append(geom)
                    improved_sizes.append(size)
                else:  # Worsened (increased flooding)
                    worsened_geoms.append(geom)
                    worsened_sizes.append(size)
        
        # Debug: print sample keys to identify format mismatch
        print(f"  [MetricMaps-DEBUG] flooding_volumes keys (first 5): {list(all_nids)[:5]}")
        print(f"  [MetricMaps-DEBUG] name_to_geom keys (first 5): {list(name_to_geom.keys())[:5]}")
        
        # Plot improved nodes (green)
        if improved_geoms:
            improved_gdf = gpd.GeoDataFrame(geometry=improved_geoms, crs=nodes_gdf.crs)
            improved_gdf.plot(ax=ax, color='green', markersize=improved_sizes, alpha=0.7)
        
        # Plot worsened nodes (red)
        if worsened_geoms:
            worsened_gdf = gpd.GeoDataFrame(geometry=worsened_geoms, crs=nodes_gdf.crs)
            worsened_gdf.plot(ax=ax, color='red', markersize=worsened_sizes, alpha=0.7)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label=f'Reduced Flooding ({len(improved_geoms)} nodes)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label=f'Increased Flooding ({len(worsened_geoms)} nodes)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Unchanged')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_title(f"Node Flooding Volume Change: Baseline vs {solution_name}", fontsize=14)
        ax.set_axis_off()
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(save_dir / "dashboard_map_flooding.png", dpi=100)
        plt.close(fig)
        
        print(f"  [MetricMaps] Saved dashboard_map_flooding.png (Improved: {len(improved_geoms)}, Worsened: {len(worsened_geoms)})")

    @profile
    def generate_capacity_comparison_maps(self, solution: SystemMetrics, solution_name: str, save_dir: Path, nodes_gdf: gpd.GeoDataFrame, show_predios: bool = False, derivations: List = None, tank_details: List[Dict] = None):
        """Generates a single 2x2 figure with capacity maps (h/D)."""
        save_dir = Path(save_dir)
        
        # Use swmm_gdf from metrics
        links_sol_gdf = solution.swmm_gdf
        links_base_gdf = self.baseline.swmm_gdf
        
        links_sol_gdf['Tramo'] = links_sol_gdf['InletNode'] + '-' + links_sol_gdf['OutletNode']
        links_sol_gdf =  links_sol_gdf[~links_sol_gdf['Tramo'].isin(derivations['Tramo'])].copy()

        
        
        # Load predios using lazy cache
        predios_gdf = self._get_predios_gdf() if show_predios else None
        if show_predios and (predios_gdf is None or predios_gdf.empty):
            show_predios = False

        # Compute bounds
        bounds = None
        ref_gdf = links_sol_gdf if links_sol_gdf is not None else links_base_gdf
        if ref_gdf is not None and not ref_gdf.empty:
            bounds = ref_gdf.total_bounds
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # 1. BASELINE CAPACITY MAP (No derivations)
        self._plot_absolute_capacity_map(axes[0], links_base_gdf, self.baseline,
                                            "Baseline (h/D)", predios_gdf, bounds, show_predios)

        # 2. SOLUTION CAPACITY MAP (With derivations)
        self._plot_absolute_capacity_map(axes[1], links_sol_gdf, solution,
                                            f"{solution_name} (h/D)", predios_gdf, bounds, show_predios)
        # # Overlay derivations on solution plot
        # self._plot_derivations_and_labels(axes[1], derivations, tank_details)

        # 3. DELTA MAP (With derivations, enhanced colors)
        self._plot_capacity_map(axes[2], links_sol_gdf, self.baseline, solution, predios_gdf, bounds, show_predios, derivations, tank_details, nodes_gdf=nodes_gdf)
        axes[2].set_title(f"Delta Categories: Baseline vs {solution_name}", fontsize=12)

        # 4. QUANTITATIVE MAGNITUDE MAP
        base_caps = self.baseline.swmm_gdf['Capacity']
        sol_caps = solution.swmm_gdf['Capacity']
        self._plot_delta_magnitude_map(axes[3], links_sol_gdf, base_caps, sol_caps,
                                       f"Quantitative Delta (h/D change)", predios_gdf, bounds, show_predios, derivations, tank_details, nodes_gdf=nodes_gdf)

        fig.suptitle("Pipe Capacity Usage Comparison (h/D)", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(save_dir / "01_dashboard_map_capacity.png", dpi=150)
        plt.close(fig)
    
    @profile
    def generate_velocity_comparison_maps(self, solution: SystemMetrics, solution_name: str, save_dir: Path, nodes_gdf: gpd.GeoDataFrame, show_predios: bool = False, derivations: List = None, tank_details: List[Dict] = None):
        """Generates a single 2x2 figure with velocity maps (m/s)."""
        save_dir = Path(save_dir)
        
        # Use swmm_gdf from metrics
        links_sol_gdf = solution.swmm_gdf if solution.swmm_gdf is not None and not solution.swmm_gdf.empty else None
        links_base_gdf = self.baseline.swmm_gdf if self.baseline.swmm_gdf is not None and not self.baseline.swmm_gdf.empty else None
        
        links_sol_gdf['Tramo'] = links_sol_gdf['InletNode'] + '-' + links_sol_gdf['OutletNode']
        links_sol_gdf =  links_sol_gdf[~links_sol_gdf['Tramo'].isin(derivations['Tramo'])].copy()
        
        # Load predios using lazy cache
        predios_gdf = self._get_predios_gdf() if show_predios else None
        if show_predios and (predios_gdf is None or predios_gdf.empty):
            show_predios = False
        
        # Compute bounds
        bounds = None
        ref_gdf = links_sol_gdf if links_sol_gdf is not None else links_base_gdf
        if ref_gdf is not None and not ref_gdf.empty:
            bounds = ref_gdf.total_bounds
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # 1. BASELINE VELOCITY MAP (No derivations)
        if links_base_gdf is not None:
            self._plot_absolute_velocity_map(axes[0], links_base_gdf, self.baseline, 
                                            "Baseline (m/s)", predios_gdf, bounds, show_predios)
        else:
            axes[0].text(0.5, 0.5, "No Baseline Data", ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("Baseline (m/s)")

        # 2. SOLUTION VELOCITY MAP (With derivations)
        if links_sol_gdf is not None:
            self._plot_absolute_velocity_map(axes[1], links_sol_gdf, solution, 
                                            f"{solution_name} (m/s)", predios_gdf, bounds, show_predios)
            # self._plot_derivations_and_labels(axes[1], derivations, tank_details)
        else:
            axes[1].text(0.5, 0.5, "No Solution Data", ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f"{solution_name} (m/s)")

        # 3. DELTA MAP (With derivations, enhanced colors)
        if links_sol_gdf is not None:
            self._plot_velocity_delta_map(axes[2], links_sol_gdf, self.baseline, solution, predios_gdf, bounds, show_predios, derivations, tank_details, nodes_gdf=nodes_gdf)
            axes[2].set_title(f"Delta Categories: Baseline vs {solution_name}", fontsize=12)

        # 4. QUANTITATIVE MAGNITUDE MAP
        if links_sol_gdf is not None:
            base_vels = self.baseline.swmm_gdf['MaxVel']
            sol_vels = solution.swmm_gdf['MaxVel']
            self._plot_delta_magnitude_map(axes[3], links_sol_gdf, base_vels, sol_vels,
                                           f"Quantitative Delta (Velocity change)", predios_gdf, bounds, show_predios, derivations, tank_details, nodes_gdf=nodes_gdf)

        fig.suptitle("Pipe Velocity Comparison (m/s)", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_dir / "02_dashboard_map_velocity.png", dpi=150)
        plt.close(fig)
        print("  [MetricMaps] Saved dashboard_map_velocity_comparison.png (2x2)")

    @profile
    def _plot_absolute_capacity_map(self, ax, links_gdf, metrics, title, predios_gdf=None, bounds=None, show_predios: bool = False):
        """Plots absolute pipe capacity usage (d/D) using colormap."""
        if links_gdf is None or links_gdf.empty:
            raise ValueError(f"_plot_absolute_capacity_map: links_gdf is None or empty for '{title}'")
        
        # Plot predios background first (very light gray) if requested
        if show_predios and predios_gdf is not None and not predios_gdf.empty:
            predios_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.2, zorder=0)
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Use existing Capacity column from swmm_gdf (already calculated)
        plot_gdf = links_gdf.copy()
        plot_gdf['cap_val'] = plot_gdf['Capacity']
        
        # 3 discrete color levels based on capacity usage (d/D)
        # Level 1: ≤ 0.65 (Green - Good)
        # Level 2: 0.65 - 0.80 (Orange - Caution)
        # Level 3: > 0.80 (Red - Critical)
        level_1 = plot_gdf[plot_gdf['cap_val'] <= 0.65]
        level_2 = plot_gdf[(plot_gdf['cap_val'] > 0.65) & (plot_gdf['cap_val'] <= 0.80)]
        level_3 = plot_gdf[plot_gdf['cap_val'] > 0.80]
        
        # Plot each level with its color
        if not level_1.empty:
            level_1.plot(ax=ax, color='#2ecc71', linewidth=2.0, alpha=0.9, zorder=2)  # Green
        if not level_2.empty:
            level_2.plot(ax=ax, color='#f39c12', linewidth=2.5, alpha=0.9, zorder=3)  # Orange
        if not level_3.empty:
            level_3.plot(ax=ax, color='#e74c3c', linewidth=3.0, alpha=0.9, zorder=4)  # Red
        
        # Custom legend for the 3 levels
        
        legend_elements = [
            Line2D([0], [0], color='#2ecc71', lw=3, label='≤ 0.65 (Bueno)'),
            Line2D([0], [0], color='#f39c12', lw=3, label='0.65-0.80 (Precaución)'),
            Line2D([0], [0], color='#e74c3c', lw=3, label='> 0.80 (Crítico)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)
        
        # Set extent to pipeline bounds with small margin
        if bounds is not None:
            margin = 50  # meters
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
        ax.set_aspect('equal')
    
    def _get_predios_gdf(self):
        """Lazy loads and caches the predios GeoDataFrame."""
        if self._predios_gdf_cache is not None:
            return self._predios_gdf_cache
            
        import config
        if hasattr(config, 'PREDIOS_DAMAGED_FILE') and config.PREDIOS_DAMAGED_FILE.exists():
            print(f"  [Reporter] Loading predios from {config.PREDIOS_DAMAGED_FILE.name}...")
            start_t = time.time()
            self._predios_gdf_cache = gpd.read_file(config.PREDIOS_DAMAGED_FILE)
            self._predios_gdf_cache = self._predios_gdf_cache.to_crs(config.PROJECT_CRS)
            print(f"  [Reporter] Loaded {len(self._predios_gdf_cache)} predios in {time.time() - start_t:.1f}s")
        else:
            raise FileNotFoundError(f"PREDIOS_DAMAGED_FILE not found in config: {getattr(config, 'PREDIOS_DAMAGED_FILE', 'N/A')}")
            
        return self._predios_gdf_cache

    def _plot_delta_magnitude_map(self, ax, links_gdf, baseline_vals, solution_vals, title, predios_gdf=None, bounds=None, show_predios: bool = False, derivations=None, tank_details=None, nodes_gdf=None):
        """Plots quantitative delta (Solution - Baseline) using colormap."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return

        # 1. Fondo de predios (opcional)
        if show_predios and predios_gdf is not None and not predios_gdf.empty:
            predios_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.2, zorder=0)

        # 2. Red de fondo (Gris muy tenue para que resalten los colores del Delta)
        links_gdf.plot(ax=ax, color='black', linewidth=1, alpha=0.3, zorder=1)

        # Preparación de datos
        base_s = pd.Series(baseline_vals)
        sol_s = pd.Series(solution_vals)
        plot_gdf = links_gdf.copy()
        plot_gdf.index = plot_gdf.index.astype(str).str.strip().str.upper()

        # Cálculo del Delta (Solución - Base)
        plot_gdf['delta'] = plot_gdf.index.map(sol_s).fillna(0) - plot_gdf.index.map(base_s).fillna(0)

        # Usamos el percentil 98 en lugar del máximo absoluto. Esto evita que un solo
        # valor extremo haga que todos los demás parezcan amarillos pálidos.
        vmax = plot_gdf['delta'].abs().quantile(0.98)
        if vmax < 0.05: vmax = 0.1  # Asegura un rango mínimo para la leyenda

        # Filtrar solo tramos con cambios significativos para dibujar encima
        significant_changes = plot_gdf[plot_gdf['delta'].abs() > 1e-4]

        if not significant_changes.empty:
            significant_changes.plot(
                ax=ax,
                column='delta',
                cmap='rocket',  # Rojo (Peor) - Amarillo (Igual) - Verde (Mejor)
                linewidth=2,  # Línea mucho más gruesa
                alpha=1.0,  # Sin transparencia para máximo contraste
                vmin=-vmax,
                vmax=vmax,
                zorder=3,
                legend=True,
                legend_kwds={
                    'label': "Delta h/D ((-) Mejora | (+) Peor)",
                    'orientation': "horizontal",
                    'pad': 0.08,
                    'shrink': 0.6,
                    'aspect': 40  # La hace más bajita
                }
            )

        # # 3. Dibujar derivaciones (si existen)
        # if not derivations.empty:
        #     self._plot_derivations_and_labels(ax, derivations, tank_details)

        # 4. Ajuste de zoom/extensión
        if bounds is not None:
            margin = 50
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        elif not links_gdf.empty:
            minx, miny, maxx, maxy = links_gdf.total_bounds
            margin = 50
            ax.set_xlim(minx - margin, maxx + margin)
            ax.set_ylim(miny - margin, maxy + margin)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_axis_off()
        ax.set_aspect('equal')


    def _plot_capacity_map(self, ax, links_gdf, baseline, solution, predios_gdf=None, bounds=None, show_predios: bool = False, derivations=None, tank_details=None, nodes_gdf=None):
        """Plots pipe capacity usage on map. Green=improved, Red=worsened, DarkGray=Unchanged."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return
        
        # Plot predios background
        if show_predios and predios_gdf is not None and not predios_gdf.empty:
            predios_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.2, zorder=0)
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Vectorized calculation
        base_caps = pd.Series(baseline.swmm_gdf['Capacity'])
        sol_caps = pd.Series(solution.swmm_gdf['Capacity'])
        
        # Use a copy and normalize index
        plot_gdf = links_gdf.copy()
        plot_gdf.index = plot_gdf.index.astype(str).str.strip().str.upper()
        
        # Map values
        plot_gdf['delta'] = plot_gdf.index.map(sol_caps).fillna(0) - plot_gdf.index.map(base_caps).fillna(0)
        
        # Unchanged
        unchanged = plot_gdf[abs(plot_gdf['delta']) <= 0.01]
        if not unchanged.empty:
            unchanged.plot(ax=ax, color='#444444', linewidth=0.8, alpha=0.4, zorder=2)

        # Improved (Green)
        # Improved (Green)
        improved = plot_gdf[plot_gdf['delta'] < -0.01]
        if not improved.empty:
            improved.plot(ax=ax, color='green', linewidth=3.5, alpha=0.9, zorder=3)
 
        # Worsened (Red)
        worsened = plot_gdf[plot_gdf['delta'] > 0.01]
        if not worsened.empty:
            worsened.plot(ax=ax, color='red', linewidth=3.5, alpha=0.9, zorder=3)
        
        # # Plot derivations if requested
        # if not derivations.empty:
        #     self._plot_derivations_and_labels(ax, derivations, tank_details)

        # Set extent
        if bounds is not None:
            margin = 50
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Improved'),
            Line2D([0], [0], color='red', lw=3, label='Worsened'),
            Line2D([0], [0], color='#444444', lw=2, label='Unchanged'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_axis_off()
        ax.set_aspect('equal')
    
    def _plot_flooding_map(self, ax, nodes_gdf, base_vols, sol_vols):
        """Plots node flooding volume difference on map. Green=reduced, Red=increased."""
        if nodes_gdf is None or nodes_gdf.empty:
            ax.text(0.5, 0.5, "No Node Data", ha='center')
            return
        
        # Background nodes
        nodes_gdf.plot(ax=ax, color='lightgray', markersize=5, alpha=0.3)
        
        # Vectorized delta calculation
        base_s = pd.Series(base_vols)
        sol_s = pd.Series(sol_vols)
        
        gdf = nodes_gdf.copy()
        gdf['NodeID_s'] = gdf.index.astype(str)
        
        gdf['v_base'] = gdf['NodeID_s'].map(base_s).fillna(0)
        gdf['v_sol'] = gdf['NodeID_s'].map(sol_s).fillna(0)
        gdf['delta'] = gdf['v_sol'] - gdf['v_base']
        
        # Filter for visibility
        sig_gdf = gdf[gdf['delta'].abs() > 1].copy()
        
        if not sig_gdf.empty:
            sig_gdf['color'] = sig_gdf['delta'].apply(lambda x: 'green' if x < 0 else 'red')
            sig_gdf['markersize'] = (10 + sig_gdf['delta'].abs() * 0.1).clip(0, 50)
            sig_gdf.plot(ax=ax, color=sig_gdf['color'], markersize=sig_gdf['markersize'], alpha=0.7)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reduced Flooding'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Increased Flooding')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    @profile
    def _plot_velocity_map(self, ax, links_gdf, baseline, solution):
        """Plots conduit velocity change on map. Green=decreased, Red=increased."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # Get velocities from metrics (assuming they're stored or can be computed)
        # For now, we use conduit_velocities which is an array, not per-link
        # This is a limitation - we'd need link-level velocity data
        # For demonstration, we'll show a placeholder message
        ax.text(0.5, 0.5, "Velocity comparison requires link-level velocity data\\n(Currently only aggregate velocities available)",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    @profile
    def _plot_absolute_velocity_map(self, ax, links_gdf, metrics, title, predios_gdf=None, bounds=None, show_predios: bool = False):
        """Plots absolute pipe velocity (m/s) using colormap."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return
        
        # Plot predios background first (very light gray) if requested
        if show_predios and predios_gdf is not None and not predios_gdf.empty:
            predios_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.2, zorder=0)
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Get velocity mapping from link_data
        velocities = metrics.swmm_gdf['MaxVel']
        
        # Create a Copy of GDF for plotting values
        plot_gdf = links_gdf.copy()
        # Normalizar índices para mapeo robusto
        plot_gdf.index = plot_gdf.index.astype(str).str.strip().str.upper()
        plot_gdf['vel_val'] = plot_gdf.index.map(velocities).fillna(0.0)
        
        # 3 discrete color levels based on velocity (m/s)
        # Level 1: ≤ 1.5 (Green - Good)
        # Level 2: 1.5 - 3.0 (Orange - Caution)
        # Level 3: > 3.0 (Red - Critical/Erosion)
        level_1 = plot_gdf[plot_gdf['vel_val'] <= 3.5]
        level_2 = plot_gdf[(plot_gdf['vel_val'] > 3.5) & (plot_gdf['vel_val'] <= 6.0)]
        level_3 = plot_gdf[plot_gdf['vel_val'] > 6]
        
        # Plot each level with its color
        if not level_1.empty:
            level_1.plot(ax=ax, color='#2ecc71', linewidth=2.0, alpha=0.9, zorder=2)  # Green
        if not level_2.empty:
            level_2.plot(ax=ax, color='#f39c12', linewidth=2.5, alpha=0.9, zorder=3)  # Orange
        if not level_3.empty:
            level_3.plot(ax=ax, color='#e74c3c', linewidth=3.0, alpha=0.9, zorder=4)  # Red
        
        # Custom legend for the 3 levels
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#2ecc71', lw=3, label='≤ 3.5 m/s (Baja)'),
            Line2D([0], [0], color='#f39c12', lw=3, label='3.5-6.0 m/s (Media)'),
            Line2D([0], [0], color='#e74c3c', lw=3, label='> 6.0 m/s (Alta)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)
        
        # Set extent to pipeline bounds with small margin
        if bounds is not None:
            margin = 50  # meters
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
        ax.set_aspect('equal')
    
    def _plot_velocity_delta_map(self, ax, links_gdf, baseline, solution, predios_gdf=None, bounds=None, show_predios: bool = False, derivations=None, tank_details=None, nodes_gdf=None):
        """Plots pipe velocity change on map. Green=decreased (better), Red=increased, DarkGray=Unchanged."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return
        
        # Plot predios background
        if show_predios and predios_gdf is not None and not predios_gdf.empty:
            predios_gdf.plot(ax=ax, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.2, zorder=0)
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
        
        # Vectorized calculation
        base_vels = pd.Series(baseline.swmm_gdf['MaxVel'])
        sol_vels = pd.Series(solution.swmm_gdf['MaxVel'])
        
        # Normalize index
        plot_gdf = links_gdf.copy()
        plot_gdf.index = plot_gdf.index.astype(str).str.strip().str.upper()
        
        # Map values
        plot_gdf['delta'] = plot_gdf.index.map(sol_vels).fillna(0) - plot_gdf.index.map(base_vels).fillna(0)
        
        # Unchanged
        unchanged = plot_gdf[abs(plot_gdf['delta']) <= 0.05]
        if not unchanged.empty:
            unchanged.plot(ax=ax, color='#444444', linewidth=0.8, alpha=0.4, zorder=2)

        # Improved (Decreased velocity, Green)
        improved = plot_gdf[plot_gdf['delta'] < -0.05]
        if not improved.empty:
            improved.plot(ax=ax, color='green', linewidth=3.5, alpha=0.9, zorder=3)

        # Worsened (Increased velocity, Red)
        worsened = plot_gdf[plot_gdf['delta'] > 0.05]
        if not worsened.empty:
            worsened.plot(ax=ax, color='red', linewidth=3.0, alpha=0.8, zorder=3)
        
        # # Plot derivations if requested
        # if not derivations.empty:
        #     self._plot_derivations_and_labels(ax, derivations, tank_details)

        # Set extent
        if bounds is not None:
            margin = 50
            ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
            ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
        
        # Legend

        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Decreased'),
            Line2D([0], [0], color='red', lw=3, label='Increased'),
            Line2D([0], [0], color='#444444', lw=2, label='Unchanged'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax.set_axis_off()
        ax.set_aspect('equal')

    def generate_hydrograph_pages(self, solution: SystemMetrics, solution_name: str, save_dir: Path, detailed_links: Dict):
        """Generates multiple pages of 3x3 hydrograph grids."""
        
        node_ids = list(detailed_links.keys())
        batch_size = 3
        
        for batch_idx, i in enumerate(range(0, len(node_ids), batch_size)):
            batch_nodes = node_ids[i : i + batch_size]
            
            # Create Figure (3 rows x 3 cols)
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3)
            
            for row_idx, nid in enumerate(batch_nodes):
                ax_flow = fig.add_subplot(gs[row_idx, 0])
                ax_cap = fig.add_subplot(gs[row_idx, 1])
                ax_flood = fig.add_subplot(gs[row_idx, 2])
                
                links = detailed_links[nid]
                self._plot_detailed_row_nx3(
                    ax_flow, ax_cap, ax_flood,
                    nid, links,
                    self.baseline, solution
                )
                
            fig.suptitle(f"Hydrograph Analysis ({i+1}-{i+len(batch_nodes)}): {solution_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_dir / f"03_dashboard_comparison_batch_{batch_idx+1}.png", dpi=100)
            plt.close(fig)

    def _overlay_annotations(self, ax, derivations, annotations_data):
        """Helper to draw simple numbered labels on derivation lines."""
        from shapely.geometry import LineString
        for i, item in enumerate(annotations_data):
            if i >= len(derivations): break
            
            row_num = i + 1  # Table row number (1-indexed)
            geom = derivations.geometry.iloc[i]
            if isinstance(geom, LineString):
                # Midpoint for label
                mid_pt = geom.interpolate(0.5, normalized=True)
                mx, my = mid_pt.x, mid_pt.y
                
                # Simple label: just the row number
                label = str(row_num)
                
                # Add compact text with box
                ax.text(mx, my, label, fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', 
                                 edgecolor='blue', linewidth=2),
                        zorder=20, ha='center', va='center', color='blue')



    def _plot_detailed_hydrographs_side_by_side(self, ax_sol, ax_base, base_hydros: Dict, sol_hydros: Dict, detailed_links: Dict):
        """
        Plots hydrographs side-by-side:
        Left (ax_sol): Solution State (Upstream [Pre], Downstream [Post], Derivation)
        Right (ax_base): Baseline State (Upstream [Base Pre], Downstream [Base Post])
        """
        # Limit to the first intervention node found to keep charts readable
        # The user asked "solo haz los tramos que hemos modificado".
        # Currently we handle 1 intervention clearly. If multiple, we might overlap.
        # Let's pick the first one in detailed_links.
        
        if not detailed_links: 
            return

        target_node = list(detailed_links.keys())[0] 
        links = detailed_links[target_node]
        
        # Define Links
        up_ids = links.get('upstream', [])
        down_ids = links.get('downstream', [])
        deriv_ids = links.get('derivation', [])
        
        # Helper to get time axis
        def get_hrs(data):
            if not data or not data['times']: return []
            t0 = data['times'][0]
            return [(t - t0).total_seconds()/3600.0 for t in data['times']]
            
        # --- LEFT: SOLUTION ---
        ax_sol.set_title("SOLUCIÓN (Con Derivación)", fontsize=12, fontweight='bold')
        
        # 1. Upstream (Antes Derivacion)
        for lid in up_ids:
            if lid in sol_hydros:
                d = sol_hydros[lid]
                ax_sol.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='darkorange', linewidth=2, label=f'Antes: {lid}')
        
        # 2. Derivation (La Derivacion) - Make it standout
        for lid in deriv_ids:
            if lid in sol_hydros:
                d = sol_hydros[lid]
                ax_sol.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='blue', linewidth=3, label=f'Derivación: {lid}')
                
        # 3. Downstream (Despues)
        for lid in down_ids:
            if lid in sol_hydros:
                d = sol_hydros[lid]
                # Use dashed or lighter for downstream in solution
                ax_sol.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='green', linestyle='-', linewidth=2, label=f'Después: {lid}')
                
        ax_sol.set_xlabel("Time (h)")
        ax_sol.set_ylabel("Flow (CMS)")
        ax_sol.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
        ax_sol.legend(loc='upper right', fontsize='small')
        ax_sol.grid(True, alpha=0.3)
        
        # --- RIGHT: BASELINE ---
        ax_base.set_title("LINEA BASE (Sin Derivación)", fontsize=12, fontweight='bold')
        
        # 1. Upstream Base
        for lid in up_ids:
            if lid in base_hydros:
                d = base_hydros[lid]
                ax_base.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='darkorange', linestyle='--', alpha=0.7, label=f'Base Antes: {lid}')

        # 2. Downstream Base
        for lid in down_ids:
            if lid in base_hydros:
                d = base_hydros[lid]
                ax_base.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='green', linestyle='--', alpha=0.7, label=f'Base Después: {lid}')
                
        ax_base.set_xlabel("Time (h)")
        ax_base.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
        # ax_base.set_ylabel("Flow (CMS)") # Shared axis concept
        ax_base.legend(loc='upper right', fontsize='small')
        ax_base.grid(True, alpha=0.3)
        
        # Match Y-limits for comparison
        y_max = max(ax_sol.get_ylim()[1], ax_base.get_ylim()[1])
        ax_sol.set_ylim(0, y_max)
        ax_base.set_ylim(0, y_max)

    def _plot_link_hydrographs(self, ax, base_hydros: Dict, sol_hydros: Dict):
        """Plots Flow Hydrographs for interest links (Baseline vs Solution)."""
        if not base_hydros:
            ax.text(0.5, 0.5, "No Link Hydrographs collected", ha='center')
            return
            
        # Plot all interest links
        # ... (rest of old code kept as fallback) ...
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(base_hydros))))
        
        has_plotted = False
        
        for i, (lid, data_base) in enumerate(base_hydros.items()):
            color = colors[i % len(colors)]
            
            # Time handling
            times = data_base['times']
            # Convert to hours relative to start
            if times:
                 t0 = times[0]
                 hrs = [(t - t0).total_seconds()/3600.0 for t in times]
                 
                 # Plot Baseline (Dashed)
                 ax.plot(hrs, data_base['flow'], linestyle='--', color=color, alpha=0.5, label=f'{lid} Base')
                 
                 # Plot Solution (Solid)
                 if lid in sol_hydros:
                     data_sol = sol_hydros[lid]
                     # Assume mostly aligned time steps for visualization
                     # If lengths differ, we should re-calc time axis, but usually valid.
                     hrs_sol = hrs
                     if len(data_sol['flow']) != len(hrs):
                          if data_sol['times']:
                               t0_s = data_sol['times'][0]
                               hrs_sol = [(t - t0_s).total_seconds()/3600.0 for t in data_sol['times']]
                     
                     ax.plot(hrs_sol, data_sol['flow'], linestyle='-', linewidth=2, color=color, label=f'{lid} Sol')
                 
                 has_plotted = True
                 
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flow (CMS)")
        ax.set_title("Flow Hydrographs @ Intervention Links")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis to configured hours
        if has_plotted:
             ax.legend(fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)



    def _plot_scalar_comparison(self, ax, sol: SystemMetrics, name: str):
        """Bar chart of % reduction for specific system metrics."""
        
        # 1. Total Flooding Volume (m3)
        b_flood = self.baseline.total_flooding_volume
        s_flood = sol.total_flooding_volume
        
        # 2. Peak Flooding Flow (cms)
        b_peak = self.baseline.total_max_flooding_flow
        s_peak = sol.total_max_flooding_flow

        # 3. Total Outfall Volume (m3)
        b_outfall = self.baseline.total_outfall_volume
        s_outfall = sol.total_outfall_volume
        
        # 4. System Pipe Capacity (Avg h/D)
        b_caps = [v.get('max_capacity', 0) for v in self.baseline.link_data.values()]
        s_caps = [v.get('max_capacity', 0) for v in sol.link_data.values()]
        b_cap = np.mean(b_caps) if b_caps else 0
        s_cap = np.mean(s_caps) if s_caps else 0
        
        # 5. Total Flooded Nodes (Count)
        b_nodes = self.baseline.flooded_nodes_count
        s_nodes = sol.flooded_nodes_count
        
        # 6. System Water Level (Avg max depth m)
        b_depths = [v.get('max_depth', 0) for v in self.baseline.node_data.values()]
        s_depths = [v.get('max_depth', 0) for v in sol.node_data.values()]
        b_depth = np.mean(b_depths) if b_depths else 0
        s_depth = np.mean(s_depths) if s_depths else 0

        # Mappings (Clearer names for the user)
        metrics = [
            "Total Flood Vol", 
            "Peak Flood Flow", 
            "Total Outfall", 
            "System Cap.", 
            "Flood Nodes", 
            "Water Depth"
        ]
        
        def get_stats(base, curr, higher_is_better=False):
            delta = curr - base
            if base <= 0:
                pct = 100.0 if curr > 0 else 0.0
            else:
                pct = (delta / base) * 100.0
            is_improvement = delta < 0
            if higher_is_better: is_improvement = delta > 0
            imp_pct = -pct 
            if higher_is_better: imp_pct = pct
            return imp_pct, is_improvement

        results = [
            get_stats(b_flood, s_flood),
            get_stats(b_peak, s_peak),
            get_stats(b_outfall, s_outfall),
            get_stats(b_cap, s_cap),
            get_stats(b_nodes, s_nodes),
            get_stats(b_depth, s_depth)
        ]
        
        reductions = [r[0] for r in results]
        improvements = [r[1] for r in results]
        
        colors = ['#2ecc71' if ok else '#e74c3c' for ok in improvements]
        bars = ax.bar(metrics, reductions, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=1.0)
        
        ax.set_ylabel("% Improvement")
        ax.set_title("System Improvements (%)", fontsize=14, fontweight='bold')
        ax.set_ylim(bottom=min(min(reductions, default=0), -15) * 1.4, top=115)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for bar, pct, ok in zip(bars, reductions, improvements):
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            offset = 3 if h >= 0 else -3
            status = "Mejor" if ok else "Peor"
            label = f"{pct:.1f}%\n({status})"
            ax.text(bar.get_x() + bar.get_width()/2., h + offset, label, 
                    ha='center', va=va, fontweight='bold', fontsize=9)

    def _plot_absolute_summary(self, ax, sol: SystemMetrics, name: str):
        """Grouped bar chart for absolute value comparison (Baseline vs Solution)."""
        metrics = ["Flooding Vol\n(m3)", "Peak Flood\n(cms)", "Outfall Vol\n(m3)", "Total Storage\n(m3)"]
        
        # Collect Data
        b_flood = self.baseline.total_flooding_volume
        s_flood = sol.total_flooding_volume
        
        b_peak = self.baseline.total_max_flooding_flow
        s_peak = sol.total_max_flooding_flow
        
        b_out = self.baseline.total_outfall_volume
        s_out = sol.total_outfall_volume
        
        # Storage is only in Solution
        s_store =  np.sum([__['max_stored_volume'] for _, __ in sol.tank_data.items()])
        b_store = np.sum([__['max_stored_volume'] for _, __ in self.baseline.tank_data.items()])
        
        base_vals = [b_flood, b_peak, b_out, b_store]
        sol_vals = [s_flood, s_peak, s_out, s_store]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # Scaling: Since volumes are HUGE and peak flow is SMALL, we use a trick or log scale.
        # But for visibility, let's use a dual axis or just normalize for plotting but label with real values.
        # Or just two subplots inside? Let's try twinx for Flow? No, that's messy.
        
        # Better: Group them but use log scale if necessary?
        
        ax.bar(x - width/2, base_vals, width, label='Baseline', color='#e74c3c', alpha=0.9)
        ax.bar(x + width/2, sol_vals, width, label='Solution', color='#3498db', alpha=0.9)
        
        ax.set_yscale('symlog', linthresh=10) # Log scale for large ranges
        ax.set_title("Absolute Totals: Baseline vs Solution", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.2)
        
        def format_val(val):
            if abs(val) >= 1_000_000:
                return f"{val/1_000_000:.1f}M"
            if abs(val) >= 1_000:
                return f"{val/1_000:.1f}k"
            return f"{val:.1f}"

        # Labels
        for i, (b_v, s_v) in enumerate(zip(base_vals, sol_vals)):
            # Base Label
            ax.text(i - width/2, b_v, format_val(b_v), 
                    ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
            # Sol Label
            ax.text(i + width/2, s_v, format_val(s_v), 
                    ha='center', va='bottom', fontsize=8, color='darkblue', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))







    def _plot_volume_scatter(self, ax, base_vols: Dict, sol_vols: Dict):
        """Scatter plot of Baseline vs Solution Volume (Log-Log) - Vectorized."""
        # Convert dicts to DataFrames for fast merging

        base_df = pd.DataFrame.from_dict(base_vols, orient='index')
        sol_df = pd.DataFrame.from_dict(sol_vols, orient='index')
        
        if base_df.empty or sol_df.empty:
            ax.text(0.5, 0.5, "No Volume Data", ha='center')
            return
            
        # Merge on NodeID (index)
        df = pd.merge(base_df[['flooding_volume']], 
                      sol_df[['flooding_volume']], 
                      left_index=True, right_index=True, suffixes=('_base', '_sol'))
        
        # Filter significant flooding (> 0.1 m3)
        df = df[(df['flooding_volume_base'] >= 0.1) | (df['flooding_volume_sol'] >= 0.1)]
        
        if df.empty:
            ax.text(0.5, 0.5, "No significant flooding nodes", ha='center')
            return
            
        # Calculate colors in bulk
        df['color'] = 'gray'
        df.loc[df['flooding_volume_sol'] < df['flooding_volume_base'] * 0.99, 'color'] = 'green'
        df.loc[df['flooding_volume_sol'] > df['flooding_volume_base'] * 1.01, 'color'] = 'red'
        
        # Plot in a single call
        ax.scatter(df['flooding_volume_base'], df['flooding_volume_sol'], 
                   c=df['color'], alpha=0.6, edgecolors='k', s=20)
        max_val = max(df['flooding_volume_base'].max(), df['flooding_volume_sol'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('Baseline Flooding (m3)')
        ax.set_ylabel('Solution Flooding (m3)')
        ax.set_title('Volume Change (Log-Log)')
        ax.loglog()
        ax.grid(True, which="both", alpha=0.3)


    @staticmethod
    def _plot_detailed_row_nx3(ax_flow, ax_cap, ax_flood, nid, links, baseline, solution):
        """Plots one row (3 cols) for a specific intervention node."""
        
        up_ids = links.get('upstream', [])
        down_ids = links.get('downstream', [])
        deriv_ids = links.get('derivation', [])

        # --- COL 1: FLOW (Caudal) ---
        ax_flow.set_title(f"Node {nid}: Caudal (Flow)", fontsize=10)
        ax_flow.set_ylabel("Caudal (m3/s)")
        
        # All lines solid for Solution to see them clearly
        for lid in up_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_flow.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='orange', label='Antes Derivacion')
        for lid in down_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_flow.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='green', label='Después Derivacion')
        for lid in deriv_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_flow.plot(d['flow_series'].index, d['flow_series'].to_numpy(), color='blue', linewidth=3, label='Derivación')

        ax_flow.legend(fontsize=8)
        ax_flow.grid(True, alpha=0.3)

        ax_flow.set_title(f"Node {nid}: Caudal [m3/s]", fontsize=10)
        ax_flow.legend(fontsize=8)
        ax_flow.grid(True, alpha=0.3)
        
        # --- COL 2: CAPACITY (Capacidad / Usage) ---
        ax_cap.set_title(f"Capacidad (Uso [0-1])", fontsize=10)
        
        # Plot Link Capacities (Usage fraction)
        for lid in up_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_cap.plot(d['capacity_series'].index, d['capacity_series'].to_numpy(), color='orange', linestyle='--', label='Antes')
        for lid in down_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_cap.plot(d['capacity_series'].index, d['capacity_series'].to_numpy(), color='green', linestyle='--', label='Después')
        for lid in deriv_ids:
            if lid in solution.link_data:
                d = solution.link_data[lid]
                ax_cap.plot(d['capacity_series'].index, d['capacity_series'].to_numpy(), color='blue', linewidth=2, linestyle='-', label='Derivación')
                
        ax_cap.set_ylim(0, 1.1)
        ax_cap.grid(True, alpha=0.3)
        
        # --- COL 3: FLOOD HYDROGRAPH (Node) ---
        ax_flood.set_title(f"Inundación Nodo {nid}", fontsize=10)
        ax_flood.set_ylabel("Profundidad (m)")
        ax_flood.legend(fontsize=8)
        ax_flood.grid(True, alpha=0.3)
        
        # Baseline Flood
        if nid in baseline.node_data:
            d = baseline.node_data[nid]
            ax_flood.plot(d['depth_series'].index, d['depth_series'].to_numpy(), color='red', linestyle='--', label='Base Depth')
            
        # Solution Flood
        if nid in solution.node_data:
            d = solution.node_data[nid]
            ax_flood.plot(d['depth_series'].index, d['depth_series'].to_numpy(), color='blue', label='Sol Depth')
            
    def _plot_system_capacity_hist(self, ax, baseline, solution):
        """Plots overlaid histograms of max pipe capacity usage for entire system."""
        
        def get_max_capacities(metrics_obj):
            caps = []
            for lid, data in metrics_obj.link_data.items():
                if data.get('capacity_series'):
                    caps.append(max(data['capacity_series']))
            return caps
        
        base_caps = get_max_capacities(baseline)
        sol_caps = get_max_capacities(solution)
        
        if not base_caps and not sol_caps:
            ax.text(0.5, 0.5, "No Capacity Data", ha='center', va='center')
            return
        
        # Plot overlapping histograms
        if base_caps:
            sns.histplot(base_caps, ax=ax, kde=True, color='gray', alpha=0.4, label='Baseline', bins=15)
        if sol_caps:
            sns.histplot(sol_caps, ax=ax, kde=True, color='blue', alpha=0.4, label='Solution', bins=15)
        
        ax.set_xlabel("Max Capacity Usage (0-1)")
        ax.set_title("System Pipe Capacity")
        ax.set_xlim(0, 1.5)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=1, label='Full (100%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_annotations_figure(self, save_dir, nodes_gdf, inp_path, derivations, annotations_data):
        """
        Generates a separate figure with clear map and annotations for Q, D, Vol.
        Labels placed carefully to avoid overlap.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Background: Nodes
        if nodes_gdf is not None:
             nodes_gdf.plot(ax=ax, color='lightgrey', markersize=20, alpha=0.5, zorder=1)
        
        # Plot Derivations
        from shapely.geometry import LineString
        for geom in derivations:
            if isinstance(geom, LineString):
                x, y = geom.xy
                ax.plot(x, y, color='blue', linewidth=2, alpha=0.7, zorder=2)
                # Plot End Point as Black Point (Predio)
                ax.scatter([x[-1]], [y[-1]], color='black', s=150, marker='o', zorder=3, label='Terreno')

        # Annotations
        for i, item in enumerate(annotations_data):
            if i >= len(derivations): break
            
            q = item['q_peak']
            d = item['diameter']
            vol = item['tank_vol']
            
            geom = derivations[i]
            if isinstance(geom, LineString):
                 # Midpoint for label
                 mid_pt = geom.interpolate(0.5, normalized=True)
                 mx, my = mid_pt.x, mid_pt.y
                 
                 label = f"Q: {q:.2f} cms\nD: {d:.2f}m\nVol: {vol:.0f} m3"
                 
                 # Add text with box
                 ax.text(mx, my, label, fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'),
                             zorder=10, ha='center')
        
        ax.set_title("Hydraulic Annotations: Derivations", fontsize=20)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(save_dir / "dashboard_annotations.png", dpi=100)
        plt.close(fig)


        
    def _plot_hydrographs(self, ax, base_hydros: Dict, sol_hydros: Dict):
        """Plots hydrographs for the most critical nodes found in baseline."""
        if not base_hydros:
            ax.text(0.5, 0.5, "No Hydrographs collected", ha='center')
            return
            
        # Plot up to 3
        for i, (nid, data_base) in enumerate(base_hydros.items()):
            if i >= 3: break
            
            times = [t.hour + t.minute/60.0 for t in data_base['times']] # Convert to hours if datetime
            # Fallback if times are not datetime
            if hasattr(data_base['times'][0], 'hour'):
                 time_axis = [(t - data_base['times'][0]).total_seconds()/3600.0 for t in data_base['times']]
            else:
                 time_axis = range(len(data_base['depths'])) # Simple index
                 
            # Baseline
            ax.plot(time_axis, data_base['depths'], linestyle='--', alpha=0.6, label=f'{nid} Base')
            
            # Solution (if available)
            # Find matching node in solution hydros? 
            # Solution might not have hydro if it wasn't top 3 there, but we need it.
            # Limitation: MetricExtractor only extracted top 3 of RESULT.
            # Ideally we extract specific nodes.
            # As a fallback, we grab from 'sol_hydros' if present, but it might not be there.
            
            # CHECK: MetricExtractor logic was "extract top N".
            # If critical node improved a lot, it might not be in sol_hydros top N.
            # This is a limitation. For now, plot what we have.
            
            if nid in sol_hydros:
                data_sol = sol_hydros[nid]
                # Align times? Assumed same simulation time steps
                ax.plot(time_axis, data_sol['depths'], linewidth=2, label=f'{nid} Sol')
                
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Hydrographs @ Critical Nodes")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
        ax.legend()
        ax.grid(True)

    def _plot_ecdf(self, ax, data_base, data_sol, xlabel, ylabel):
        """Plots Empirical CDF."""
        def ecdf(data):
            x = np.sort(data)
            n = x.size
            y = np.arange(1, n+1) / n
            return x, y
            
        if len(data_base) > 0:
            x_b, y_b = ecdf(data_base)
            ax.step(x_b, y_b, label='Baseline', color='gray', linestyle='--')
            
        if len(data_sol) > 0:
            x_s, y_s = ecdf(data_sol)
            ax.step(x_s, y_s, label='Solution', color='blue')
            
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='lower right')
        ax.set_title(f"ECDF: {xlabel}")

    @staticmethod
    def generate_tank_hydrograph_plots_old(solution: SystemMetrics, solution_name: str, save_dir: Path, detailed_links: Dict = None):
        """Generates plots for detailed tank hydrographs (Inflow, Volume, Depth). Max 4 per page."""
        if not solution.tank_data:
            return

        tank_ids = list(solution.tank_data.keys())
        tank_ids.sort()
        
        batch_size = 4
        
        for batch_idx, i in enumerate(range(0, len(tank_ids), batch_size)):
            batch_tanks = tank_ids[i : i + batch_size]
            
            n_rows = len(batch_tanks)
            # 3 Columns: Inflow (calc), Volume (calc), Depth (measured)
            fig = plt.figure(figsize=(24, 5 * n_rows))
            gs = fig.add_gridspec(n_rows, 3)
            
            for row_idx, tk_id in enumerate(batch_tanks):
                util_data = solution.tank_data[tk_id]
                design_vol = util_data.get('total_volume', 0.0)
                
                # Axes
                ax_flow = fig.add_subplot(gs[row_idx, 0])
                ax_vol = fig.add_subplot(gs[row_idx, 1])
                ax_depth = fig.add_subplot(gs[row_idx, 2])

                # --- PLOT 1: INFLOW (Calculado) ---
                variable = 'flow_series'
                # CONVERSIÓN A HORAS RELATIVAS (X-axis fix)
                t0 = util_data[variable].index[0]
                hrs = [(t - t0).total_seconds() / 3600.0 for t in util_data[variable].index]
                
                ax_flow.plot(hrs, util_data[variable].to_numpy(), color='blue', linewidth=1.5, label='Est. Inflow')
                ax_flow.set_title(f"{tk_id}\nInflow", fontweight='bold')
                ax_flow.set_ylabel("Flow (m³/s)")
                ax_flow.fill_between(hrs, util_data[variable].to_numpy(), color='blue', alpha=0.1)
                ax_flow.grid(True, alpha=0.3)
                ax_flow.text(0.95, 0.95, f"Peak: {util_data['max_flow']:.2f} m³/s", transform=ax_flow.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
                
                # --- PLOT 2: VOLUME ---
                variable = 'volume_series'
                ax_vol.plot(hrs, util_data[variable].to_numpy(), color='green', linewidth=2, label='Volume')
                ax_vol.set_title("Cumulative Volume")
                ax_vol.set_ylabel("Volume (m³)")
                ax_vol.fill_between(hrs, util_data[variable].to_numpy(), color='green', alpha=0.2)
                ax_vol.axhline(design_vol, color='darkgreen', linestyle='--', label='Capacity')
                ax_vol.grid(True, alpha=0.3)
                ax_vol.text(0.95, 0.95, f"Stored: {util_data['max_stored_volume']:,.0f} m³", transform=ax_vol.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
                
                # --- PLOT 3: DEPTH ---
                variable = 'depth_series'
                ax_depth.plot(hrs, util_data[variable].to_numpy(), color='tab:red', linewidth=2, label='Depth')
                ax_depth.axhline(config.TANK_DEPTH_M, color='darkred', linestyle='--', label=f'Design ({config.TANK_DEPTH_M}m)')
                ax_depth.fill_between(hrs, util_data[variable].to_numpy(), color='tab:red', alpha=0.2)
                
                # Etiquetas de texto
                vol_str = f"{util_data['total_volume']:,.0f} m³"
                label_text = (f"Design Vol: {vol_str}\n"
                           f"Design Depth: {config.TANK_DEPTH_M} m\n"
                           f"Max Depth: {util_data['max_depth']:.2f} m\n"
                           f"Util: {(util_data['max_depth']/config.TANK_DEPTH_M)*100:.0f}%")
                ax_depth.text(0.98, 0.95, label_text, transform=ax_depth.transAxes,
                           ha='right', va='top', fontsize=10,
                           bbox=dict(facecolor='lemonchiffon', alpha=0.7, edgecolor='orange'))
                ax_depth.set_title("Tank Depth (Filling Curve)")
                ax_depth.set_ylabel("Depth (m)")
                
                # Etiquetas de ejes comunes
                ax_flow.set_xlabel("Time (h)")
                ax_vol.set_xlabel("Time (h)")
                ax_depth.set_xlabel("Time (h)")
                
                # Límites del eje X (Ahora coinciden perfectamente)
                ax_flow.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                ax_vol.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                ax_depth.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                
            fig.suptitle(f"{solution_name}: Tank Hydrographs (Page {batch_idx+1})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_dir / f"04_{solution_name}_tank_hydrographs_page_{batch_idx+1}.png", dpi=100)
            plt.close(fig)
            print(f"  [Tanks] Saved page {batch_idx+1}")

    @staticmethod
    def generate_tank_hydrograph_plots(solution: SystemMetrics, solution_name: str, save_dir: Path, detailed_links: Dict = None):
        """Generates plots for detailed tank hydrographs (Inflow, Volume, Depth). Max 4 per page."""
        if not solution.tank_data:
            return

        tank_ids = list(solution.tank_data.keys())
        tank_ids.sort()
        
        batch_size = 4
        
        for batch_idx, i in enumerate(range(0, len(tank_ids), batch_size)):
            batch_tanks = tank_ids[i : i + batch_size]
            
            n_rows = len(batch_tanks)
            # 3 Columns: Inflow (calc), Volume (calc), Depth (measured)
            fig = plt.figure(figsize=(24, 5 * n_rows))
            gs = fig.add_gridspec(n_rows, 3)
            
            for row_idx, tk_id in enumerate(batch_tanks):
                util_data = solution.tank_data[tk_id]
                design_vol = util_data.get('total_volume', 0.0)
                
                # Axes
                ax_flow = fig.add_subplot(gs[row_idx, 0])
                ax_vol = fig.add_subplot(gs[row_idx, 1])
                ax_depth = fig.add_subplot(gs[row_idx, 2])

                # --- PLOT 1: INFLOW (Calculado) ---
                variable = 'flow_series'
                # CONVERSIÓN A HORAS RELATIVAS (X-axis fix)
                t0 = util_data[variable].index[0]
                hrs = [(t - t0).total_seconds() / 3600.0 for t in util_data[variable].index]
                
                ax_flow.plot(hrs, util_data[variable].to_numpy(), color='blue', linewidth=1.5, label='Est. Inflow')
                ax_flow.set_title(f"{tk_id}\nInflow", fontweight='bold', fontsize=18)
                ax_flow.set_ylabel("Flow (m³/s)", fontsize=14)
                ax_flow.fill_between(hrs, util_data[variable].to_numpy(), color='blue', alpha=0.1)
                ax_flow.grid(True, alpha=0.3)
                ax_flow.text(0.95, 0.95, f"Peak: {util_data['max_flow']:.2f} m³/s", transform=ax_flow.transAxes, ha='right', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
                
                # --- PLOT 2: VOLUME ---
                variable = 'volume_series'
                ax_vol.plot(hrs, util_data[variable].to_numpy(), color='green', linewidth=2, label='Volume')
                ax_vol.set_title("Cumulative Volume", fontsize=18)
                ax_vol.set_ylabel("Volume (m³)", fontsize=14)
                ax_vol.fill_between(hrs, util_data[variable].to_numpy(), color='green', alpha=0.2)
                # ax_vol.axhline(design_vol, color='darkgreen', linestyle='--', label='Capacity')
                ax_vol.grid(True, alpha=0.3)
                ax_vol.text(0.95, 0.95, f"Stored: {util_data['max_stored_volume']:,.0f} m³", transform=ax_vol.transAxes, ha='right', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
                
                # --- PLOT 3: DEPTH ---
                variable = 'depth_series'
                ax_depth.plot(hrs, util_data[variable].to_numpy(), color='tab:red', linewidth=2, label='Depth')
                ax_depth.axhline(config.TANK_DEPTH_M, color='darkred', linestyle='--', label=f'Design ({config.TANK_DEPTH_M}m)')
                ax_depth.fill_between(hrs, util_data[variable].to_numpy(), color='tab:red', alpha=0.2)
                
                # Etiquetas de texto mejoradas
                vol_str = f"{util_data['total_volume']:,.0f} m³"
                label_text = (f"Design Vol: {vol_str}\n"
                           f"Design Depth: {config.TANK_DEPTH_M} m\n"
                           f"Max Depth: {util_data['max_depth']:.2f} m\n"
                           f"Util: {(util_data['max_depth']/config.TANK_DEPTH_M)*100:.0f}%")
                ax_depth.text(0.98, 0.95, label_text, transform=ax_depth.transAxes,
                           ha='right', va='top', fontsize=15,
                           bbox=dict(facecolor='lemonchiffon', alpha=0.7, edgecolor='orange'))
                ax_depth.set_title("Tank Depth (Filling Curve)", fontsize=18)
                ax_depth.set_ylabel("Depth (m)", fontsize=14)
                
                # Configuración común de ejes
                for ax in [ax_flow, ax_vol, ax_depth]:
                    ax.set_xlabel("Time (h)", fontsize=14)
                    ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                
            fig.suptitle(f"{solution_name}: Tank Hydrographs (Page {batch_idx+1})", fontsize=24)
            plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Ajustado para que el título grande no choque
            fig.savefig(save_dir / f"04_{solution_name}_tank_hydrographs_page_{batch_idx+1}.png", dpi=100)
            plt.close(fig)
            print(f"  [Tanks] Saved page {batch_idx+1}")


if __name__ == "__main__":
    # Test logic
    pass
