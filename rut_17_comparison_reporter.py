
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

# swmm / pyswmm imports
from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute, LinkAttribute, SystemAttribute

# Import time limit from config
try:
    import config
    PLOT_TIME_LIMIT_HOURS = config.PLOT_TIME_LIMIT_HOURS
except:
    PLOT_TIME_LIMIT_HOURS = 3.5  # Default: 3.5 hours

plt.style.use('ggplot')

@dataclass
class SystemMetrics:
    """
    Stores system-wide hydraulic metrics and detailed per-node data.
    """
    # Scalars (Totals / Averages)
    total_flooding_volume: float = 0.0          # m3
    flooding_cost: float = 0.0                  # $
    flooded_nodes_count: int = 0
    avg_node_depth: float = 0.0                 # m
    avg_conduit_velocity: float = 0.0           # m/s
    
    # Detailed Data (NodeID -> Value)
    node_depths: Dict[str, float] = field(default_factory=dict)
    flooding_volumes: Dict[str, float] = field(default_factory=dict)
    
    # Time Series for Critical Nodes (NodeID -> {time: [], value: []})
    flood_hydrographs: Dict[str, Dict[str, list]] = field(default_factory=dict)
    
    # SYSTEM-WIDE aggregated series (Time, Value)
    system_flood_hydrograph: Dict[str, list] = field(default_factory=dict) # {'times': [], 'total_rate': []}
    
    # Time Series for Interest Links (LinkID -> {time: [], flow: []})
    link_hydrographs: Dict[str, Dict[str, list]] = field(default_factory=dict)
    
    # Time Series for Link Capacity (LinkID -> {time: [], capacity: []})
    link_capacities: Dict[str, Dict[str, list]] = field(default_factory=dict)
    
    # Arrays for distribution plots (Derived or stored for convenience)
    conduit_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    conduit_capacities: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Tank Utilization (TankID -> {max_depth, max_volume, designed_depth, designed_volume, utilization_pct})
    tank_utilization: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MetricExtractor:
    """
    Extracts metrics from SWMM binary output (.out).
    """
    def __init__(self, flooding_cost_per_m3: float = 1250.0):
        self.flooding_cost_per_m3 = flooding_cost_per_m3
        

    def extract(self, out_file_path: str, top_n_hydrographs: int = 3, target_links: List[str] = None, target_nodes: List[str] = None) -> SystemMetrics:
        """
        Parses output file and returns SystemMetrics.
        Extracts hydrographs for the top N flooded nodes AND specific target links.
        """
        if not os.path.exists(out_file_path):
            print(f"[MetricExtractor] Error: File not found {out_file_path}")
            return SystemMetrics()
            
        # DEBUG: unexpected identity check
        import time
        stats = os.stat(out_file_path)
        print(f"[MetricExtractor] Reading: {out_file_path} (Size: {stats.st_size} bytes, Mod: {time.ctime(stats.st_mtime)})")
        
        metrics = SystemMetrics()
        
        try:
            # Force close any potential lingering handles? Not needed with 'with' context usually.
            # ensure string
            out_file_path = str(out_file_path)
            
            with Output(out_file_path) as out:
                # 1. System Scalars
                # ... (rest of extraction logic remains the same, assuming Output(path) works correctly)
                
                # Verify we are reading the correct file by checking something unique if possible?
                # For now just proceed.
                
                # ... existing logic ...
                # Note: PySWMM direct access to system stats might be limited, 
                # so we aggregate from nodes/links or assume we parse .rpt file elsewhere.
                # Here we aggregate from nodes:
                
                # Node Depths & Flooding
                node_depths = {}
                flood_vols = {}
                
                # System Aggregation Arrays (ONLY ORIGINAL NETWORK NODES)
                system_flood_ts = None
                system_times = None
                
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
                
                # Calculate flooding and system aggregate for ORIGINAL NODES ONLY
                for nid in out.nodes:
                    # Skip tank and derivation nodes
                    if not is_original_node(nid):
                        continue
                        
                    # Get Flooding Series (This is the key metric)
                    f_series = out.node_series(nid, NodeAttribute.FLOODING_LOSSES)
                    
                    if f_series:
                        rates = list(f_series.values()) # cms
                        times = list(f_series.keys())
                        
                        # --- SYSTEM AGGREGATION (only for original network nodes) ---
                        if f_series:
                            # Create pandas series for this node
                            # Series index is datetime, values are rates
                            ts_node = pd.Series(data=list(f_series.values()), index=list(f_series.keys()))
                            
                            # Calculate max rate for volume check
                            max_rate = ts_node.max() if not ts_node.empty else 0.0
                            
                            if system_flood_ts is None:
                                system_flood_ts = ts_node
                            else:
                                # Sum with alignment (fill_value=0 ensures we don't lose data if times differ)
                                system_flood_ts = system_flood_ts.add(ts_node, fill_value=0)
                        
                        # Only compute volume if there's actual flooding
                        if max_rate > 0.001: # Significant flooding threshold
                            # Vol Calc
                            vol = 0.0
                            if len(times) > 1:
                                for i in range(1, len(times)):
                                    dt = (times[i] - times[i-1]).total_seconds()
                                    vol += rates[i-1] * dt
                                    
                            if vol > 0.1:
                                flood_vols[nid] = vol
                                
                                # Only get depth for flooded nodes (OPTIMIZATION)
                                d_vals = list(out.node_series(nid, NodeAttribute.INVERT_DEPTH).values())
                                if d_vals:
                                    node_depths[nid] = max(d_vals)
                
                # Store System Hydrograph (already filtered - only original network nodes)
                if system_flood_ts is not None:
                    # Sort index just in case
                    system_flood_ts = system_flood_ts.sort_index()
                    
                    system_times = system_flood_ts.index.tolist()
                    system_rates = system_flood_ts.values.tolist()
                    
                    metrics.system_flood_hydrograph = {
                        'times': system_times,
                        'total_rate': system_rates
                    }
                    
                    # Calculate total volume from summed filtered hydrograph
                    # This ensures consistency between the plot and the reported number
                    total_vol = 0.0
                    if len(system_times) > 1:
                        # Use trapezoidal integration or simply step integration strictly consistent with rates
                        # Using simple retangular (rate * dt) as typically SWMM reports average rate for step or similar
                        for i in range(1, len(system_times)):
                            dt = (system_times[i] - system_times[i-1]).total_seconds()
                            # Use rate at t-1
                            total_vol += system_rates[i-1] * dt
                    
                    metrics.total_flooding_volume = total_vol
                else:
                     metrics.total_flooding_volume = 0.0
                
                # Node-level data for detailed analysis 
                metrics.node_depths = node_depths
                metrics.flooding_volumes = flood_vols
                metrics.flooded_nodes_count = len(flood_vols)
                
                # Cost calc
                metrics.flooding_cost = metrics.total_flooding_volume * self.flooding_cost_per_m3
                
                if node_depths:
                    metrics.avg_node_depth = np.mean(list(node_depths.values()))
                
                # 2. Extract Hydrographs for Critical Nodes + Target Nodes
                # Sort flooding volumes desc
                sorted_flood = sorted(flood_vols.items(), key=lambda x: x[1], reverse=True)
                top_nodes = [x[0] for x in sorted_flood[:top_n_hydrographs]]
                
                nodes_to_extract = set(top_nodes)
                if target_nodes:
                    nodes_to_extract.update(target_nodes)
                
                for nid in nodes_to_extract:
                    try:
                        # Get depth hydrograph (Depth vs Time)
                        # We could also get Flowing Losses vs Time
                        depth_series = out.node_series(nid, NodeAttribute.INVERT_DEPTH)
                        # Convert datetimes to relative hours or similar? Keep as is for now.
                        times = list(depth_series.keys())
                        vals = list(depth_series.values())
                        
                        # Store as simple list of tuples or separate arrays
                        # Let's simple lists of timestamps and values
                        metrics.flood_hydrographs[nid] = {
                            'times': times,
                            'depths': vals,
                            'flood_vol': flood_vols.get(nid, 0.0),
                            'flood_rate': list(out.node_series(nid, NodeAttribute.FLOODING_LOSSES).values())
                        }
                    except Exception as e:
                         print(f"[MetricExtractor] Warning: Could not extract hydrograph for node {nid}: {e}")

                # 3. Links Analysis (Velocity and Capacity)
                velocities = []
                capacities = []
                
                for lid in out.links:
                     # Velocity data
                     v_vals = list(out.link_series(lid, LinkAttribute.FLOW_VELOCITY).values())
                     if v_vals:
                         velocities.append(max(v_vals))
                     
                     # Capacity data (d/D)
                     c_vals = list(out.link_series(lid, LinkAttribute.CAPACITY).values())
                     if c_vals:
                         capacities.append(max(c_vals))
                         
                metrics.conduit_velocities = np.array(velocities)
                metrics.conduit_capacities = np.array(capacities)
                
                if velocities:
                    metrics.avg_conduit_velocity = np.mean(velocities)
                    
                # 4. Extract Hydrographs for Target Links
                if target_links:
                     for lid in target_links:
                         try:
                             # Get Flow and Velocity
                             link_flow = list(out.link_series(lid, LinkAttribute.FLOW_RATE).values())
                             link_depth = list(out.link_series(lid, LinkAttribute.FLOW_DEPTH).values())
                             times = list(out.link_series(lid, LinkAttribute.FLOW_RATE).keys())
                             
                             metrics.link_hydrographs[lid] = {
                                 'times': times,
                                 'flow': link_flow,
                                 'depth': link_depth
                             }
                             
                             # Capacity (Fraction Full or similar)
                             cap_series = list(out.link_series(lid, LinkAttribute.CAPACITY).values())
                             metrics.link_capacities[lid] = {
                                 'times': times,
                                 'capacity': cap_series
                             }
                         except Exception as e:
                             print(f"[MetricExtractor] Warning: Could not extract hydrograph for link {lid}: {e}")

                # 5. EXTRACT TANK UTILIZATION (New Logic)
                # Look for all nodes starting with TK_
                for nid in out.nodes:
                    if nid.startswith("TK_"):
                        try:
                            # Get full depth series for tank
                            depth_series_raw = out.node_series(nid, NodeAttribute.INVERT_DEPTH)
                            times = list(depth_series_raw.keys())
                            depths = list(depth_series_raw.values())
                            max_depth = max(depths) if depths else 0.0
                            
                            metrics.tank_utilization[nid] = {
                                'max_depth': max_depth,
                                'max_volume': 0.0,
                                'depth_series': depths,
                                'times': times
                            }
                        except Exception as e:
                            print(f"[MetricExtractor] Warning: Error processing tank {nid}: {e}")


                return metrics
        except Exception as e:
            print(f"[MetricExtractor] Extraction Error: {e}")
            import traceback
            traceback.print_exc()
            
        return metrics

class ScenarioComparator:
    """
    Compares two SystemMetrics (Baseline vs Solution) and generates reports/plots.
    """
    def __init__(self, baseline_metrics: SystemMetrics):
        self.baseline = baseline_metrics
        
    def generate_comparison_plots(self, solution: SystemMetrics, solution_name: str, save_dir: Path, 
                                  nodes_gdf: gpd.GeoDataFrame, inp_path: str, derivations: List,
                                  annotations_data: List[Dict], detailed_links: Dict, designed_gdf: gpd.GeoDataFrame = None):
        """Generates all comparison plots for a given solution."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Combined Map (Spatial + Scatter + Hydro + Hists)
        self.generate_combined_map(solution, solution_name, save_dir, nodes_gdf, inp_path, derivations, annotations_data, detailed_links)
        
        # 2. Hydrographs (3x3 grid)
        if detailed_links:
            self.generate_hydrograph_pages(solution, solution_name, save_dir, detailed_links)
            # 2.1 Tank Hydrographs (Specific Inflow/Weir plots)
            self.generate_tank_hydrograph_plots(solution, solution_name, save_dir, detailed_links)
            
        # 3. Longitudinal Profiles (3 per page)
        if designed_gdf is not None:
             self.generate_profile_plots(solution, solution_name, save_dir, designed_gdf)
        
        # 4. Unified Statistical Dashboard (Depth, Flooding, Capacity)
        self.generate_unified_statistical_dashboard(solution, solution_name, save_dir)

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
                            d = float(row.D_int)
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
                    depth = metrics.node_depths.get(n_start, d * 0.7)  # Default 70% fill
                    hgl_y.append(z_bot + min(depth, d))  # Cap at crown
                    
                    # Annotate Start Invert
                    # Invert labeling: vertically below the pipe start
                    invert_annotations.append((x_positions[p_idx], z_bot, f"HI:{z_bot:.2f}", 'HI'))
                    
                
                # Add final point
                if n_pipes > 0:
                    z_bot = z_invert_end[-1]
                    d = diameters[-1]
                    n_end = node_labels[-1]
                    depth = metrics.node_depths.get(n_end, d * 0.7)
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
        base_depths = list(self.baseline.node_depths.values())
        sol_depths = list(solution.node_depths.values())
        
        base_vols = list(self.baseline.flooding_volumes.values())
        sol_vols = list(solution.flooding_volumes.values())
        
        base_caps = list(self.baseline.conduit_capacities) if len(self.baseline.conduit_capacities) > 0 else []
        sol_caps = list(solution.conduit_capacities) if len(solution.conduit_capacities) > 0 else []
        
        # Row 1: Scalars (Left) | Velocity ECDF (Right)
        ax_scalar = fig.add_subplot(gs[0, 0])
        self._plot_scalar_comparison(ax_scalar, solution, solution_name)
        
        ax_vel = fig.add_subplot(gs[0, 1])
        self._plot_ecdf(ax_vel, self.baseline.conduit_velocities, solution.conduit_velocities, "Max Pipe Vel (m/s)", "CDF")
        
        # Row 2: Node Depths (Hist | ECDF)
        ax_depth_hist = fig.add_subplot(gs[1, 0])
        self._plot_hist(ax_depth_hist, base_depths, sol_depths, "Node Max Depths (m)")
        
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
        self._plot_ecdf(ax_cap_ecdf, base_caps, sol_caps, "Max Capacity (d/D)", "CDF")
        
        fig.suptitle(f"Unified Statistical Analysis: {solution_name}", fontsize=18, y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(save_dir / "dashboard_statistical_combined.png", dpi=100)
        plt.close(fig)
        print("  [Stats] Saved dashboard_statistical_combined.png")


    
    def generate_combined_map(self, solution: SystemMetrics, solution_name: str, save_dir: Path, 
                              nodes_gdf: gpd.GeoDataFrame, inp_path: str, derivations: List,
                              annotations_data: List[Dict] = None,
                              detailed_links: Dict = None):
        """
        Dashboard Layout:
        Left (Col 0, Rows 0-2): Map
        Right (Col 1):
           Row 0: Scatter Plot
           Row 1: System Flooding Hydrograph
           Row 2: Cumulative Flooding Volume
           Row 3: Histograms (Capacity | Flood Vol)
        """
        fig = plt.figure(figsize=(24, 22))
        gs = fig.add_gridspec(4, 2, width_ratios=[2, 1])
        
        # --- LEFT: MAP ---
        ax_map = fig.add_subplot(gs[:, 0])
        self._plot_spatial_diff(ax_map, nodes_gdf, self.baseline.flooding_volumes, solution.flooding_volumes, inp_path, derivations)
        if annotations_data and derivations:
             self._overlay_annotations(ax_map, derivations, annotations_data)

        # --- RIGHT COLUMN ---
        
        # 1. Scatter (Top)
        ax_scatter = fig.add_subplot(gs[0, 1])
        self._plot_volume_scatter(ax_scatter, self.baseline.flooding_volumes, solution.flooding_volumes)
        
        # 2. System Flooding Hydrograph (Middle-Top)
        ax_flood = fig.add_subplot(gs[1, 1])
        self._plot_system_flood_hydrograph(ax_flood, self.baseline, solution)
        
        # 3. Cumulative Flooding Volume (Middle-Bottom) - NEW!
        ax_cumulative = fig.add_subplot(gs[2, 1])
        self._plot_cumulative_volume(ax_cumulative, self.baseline, solution)
        
        # 4. Flood Volume Histogram (Bottom)
        ax_hist_vol = fig.add_subplot(gs[3, 1])
        base_vols = list(self.baseline.flooding_volumes.values())
        sol_vols = list(solution.flooding_volumes.values())
        self._plot_hist(ax_hist_vol, base_vols, sol_vols, "Flood Vol (m³)")
        
        plt.tight_layout()
        plt.savefig(save_dir / "dashboard_map.png", dpi=100)
        plt.close(fig)

    def _plot_system_flood_hydrograph(self, ax, baseline, solution):
        """Sums up flooding rates (cms) for extracted nodes to show system stress."""
        
        def plot_series(metrics_obj, color, label):
            data = metrics_obj.system_flood_hydrograph
            if not data or 'total_rate' not in data:
                print(f"  [HYDRO-DEBUG] {label}: No system hydrograph found.")
                return None, None
                
            y = np.array(data['total_rate'])
            times = data['times']
            
            # Convert time to hours
            hrs = []
            if times:
                t0 = times[0]
                if hasattr(t0, 'total_seconds'):
                    hrs = [(t - t0).total_seconds()/3600.0 for t in times]
                elif hasattr(t0, 'hour'):
                    hrs = [(t - t0).total_seconds()/3600.0 for t in times]
                else:
                    hrs = times

            ax.plot(hrs, y, linestyle='-' if label=='Solution' else '--', color=color, alpha=0.8, label=label)
            
            # Peak Label
            if len(y) > 0:
                ymax = np.max(y)
                if ymax > 0:
                    idx = np.argmax(y)
                    ax.annotate(f'{ymax:.2f}', xy=(hrs[idx], ymax), xytext=(0, 5), 
                                textcoords="offset points", color=color, fontsize=9, fontweight='bold')
            return hrs, y

        # Plot
        t_b, y_b = plot_series(baseline, 'red', 'Baseline Total Flood Rate')
        t_s, y_s = plot_series(solution, 'blue', 'Solution Total Flood Rate')

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
             
             vol_diff = vol_b_reported - vol_s_reported # Reduction
             
             # Print STATS on the plot (Showing match/mismatch)
             stats_text = (f"SWMM Vol Base: {vol_b_reported:,.0f} m3\n"
                           f"Plot Vol Base: {vol_b_integ:,.0f} m3\n"
                           f"----------------\n"
                           f"SWMM Vol Sol:  {vol_s_reported:,.0f} m3\n"
                           f"Plot Vol Sol:  {vol_s_integ:,.0f} m3\n"
                           f"----------------\n"
                           f"Reduc (SWMM): {vol_diff:,.0f} m3")
             
             ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, 
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='black', alpha=0.9))

        # Sanity Check for User Reassurance
        if y_b is not None and y_s is not None and len(y_b) > 0 and len(y_s) > 0:
            if len(y_b) == len(y_s):
                if np.allclose(y_b, y_s, atol=1e-3):
                    ax.text(0.5, 0.5, "!! IDENTICAL SERIES DETECTED !!", 
                            transform=ax.transAxes, color='red', fontsize=14, 
                            ha='center', va='center', fontweight='bold',
                            bbox=dict(facecolor='yellow', alpha=0.8))
                    print("  [HYDRO-WARNING] Baseline and Solution system hydrographs are IDENTICAL!")
                else:
                    # Calculate difference stats
                    diff = np.sum(y_s) - np.sum(y_b)
                    pct = (diff / np.sum(y_b) * 100) if np.sum(y_b) > 0 else 0
                    print(f"  [HYDRO-CHECK] Hydrographs differ. Diff Sum: {diff:.2f} ({pct:.1f}%)")
            else:
                 print(f"  [HYDRO-CHECK] Hydrographs have different lengths ({len(y_b)} vs {len(y_s)}). They are different.")

    def _plot_cumulative_volume(self, ax, baseline, solution):
        """Plots cumulative flooding volume over time (integral of rate)."""
        
        def get_cumulative(metrics_obj, color, label):
            data = metrics_obj.system_flood_hydrograph
            if not data or 'total_rate' not in data:
                return None, None
                
            rates = np.array(data['total_rate'])  # m³/s
            times = data['times']
            
            if not times or len(times) < 2:
                return None, None
                
            # Convert times to hours and calculate dt in seconds
            t0 = times[0]
            hrs = []
            dt_secs = []
            for i, t in enumerate(times):
                hrs.append((t - t0).total_seconds() / 3600.0)
                if i > 0:
                    dt_secs.append((times[i] - times[i-1]).total_seconds())
            
            # Calculate cumulative volume
            cumulative = np.zeros(len(rates))
            for i in range(1, len(rates)):
                # Volume = rate * dt
                cumulative[i] = cumulative[i-1] + rates[i-1] * dt_secs[i-1]
            
            ax.plot(hrs, cumulative, linestyle='-' if 'Solution' in label else '--', 
                    color=color, alpha=0.8, linewidth=2, label=label)
            
            # Final value annotation
            final_vol = cumulative[-1]
            ax.annotate(f'{final_vol:,.0f} m³', xy=(hrs[-1], final_vol), 
                       xytext=(-50, 5), textcoords="offset points", 
                       color=color, fontsize=9, fontweight='bold')
            
            return hrs, cumulative
        
        hrs_b, cum_b = get_cumulative(baseline, 'red', 'Baseline Cumulative')
        hrs_s, cum_s = get_cumulative(solution, 'blue', 'Solution Cumulative')
        
        ax.set_title("Cumulative Flooding Volume Over Time")
        ax.set_ylabel("Cumulative Volume (m³)")
        ax.set_xlabel("Time (hours)")
        ax.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add reduction annotation
        if cum_b is not None and cum_s is not None:
            reduction = cum_b[-1] - cum_s[-1]
            pct = (reduction / cum_b[-1] * 100) if cum_b[-1] > 0 else 0
            ax.text(0.98, 0.05, f"Reduction: {reduction:,.0f} m³ ({pct:.1f}%)", 
                   transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                   bbox=dict(facecolor='lightgreen', alpha=0.8, edgecolor='green'))

    def generate_tank_hydrograph_plots(self, solution: SystemMetrics, solution_name: str, 
                                        save_dir: Path, detailed_links: Dict = None):
        """
        Generate tank hydrograph plots with 3 columns per tank:
        - Col 1: Weir (derivation) flow hydrograph
        - Col 2: Cumulative volume in tank (integral of inflow)
        - Col 3: Tank depth/level with % utilization
        
        One row per tank, saved as {solution_name}_tank_hydrographs.png
        """
        tank_data = solution.tank_utilization
        if not tank_data:
            print("  [Tanks] No tank utilization data available.")
            return
            
        n_tanks = len(tank_data)
        if n_tanks == 0:
            return
            
        fig, axes = plt.subplots(n_tanks, 3, figsize=(15, 4 * n_tanks))
        if n_tanks == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle(f'{solution_name}: Tank Hydrographs', fontsize=14, fontweight='bold', y=1.02)
        
        designed_depth = 5.0  # meters (tank design depth)
        
        for i, (tank_id, tank_info) in enumerate(tank_data.items()):
            # Get time series data
            times = tank_info.get('times', [])
            depths = tank_info.get('depth_series', [])
            
            if not times or not depths:
                for j in range(3):
                    axes[i, j].text(0.5, 0.5, 'No data', transform=axes[i, j].transAxes,
                                   ha='center', va='center', fontsize=12, color='gray')
                continue
                
            # Convert times to hours
            t0 = times[0]
            hrs = [(t - t0).total_seconds() / 3600.0 for t in times]
            
            # Extract node ID from tank ID (TK_P0061405_4 -> P0061405)
            parts = tank_id.split('_')
            source_node = parts[1] if len(parts) > 1 else tank_id
            
            # ===== COL 1: WEIR FLOW HYDROGRAPH =====
            ax1 = axes[i, 0]
            weir_flow = None
            weir_hrs = None
            
            # DEBUG: Print to console AND write to file
            debug_path = save_dir / "tank_hydro_debug.txt"
            print(f"\n  [TANK-DEBUG] === Tank: {tank_id}, source_node: {source_node} ===")
            print(f"  [TANK-DEBUG] detailed_links keys: {list(detailed_links.keys()) if detailed_links else 'None'}")
            print(f"  [TANK-DEBUG] source_node in detailed_links: {source_node in detailed_links if detailed_links else 'N/A'}")
            if detailed_links and source_node in detailed_links:
                print(f"  [TANK-DEBUG] derivation for this node: {detailed_links[source_node].get('derivation', [])}")
            print(f"  [TANK-DEBUG] link_hydrographs keys (first 30): {list(solution.link_hydrographs.keys())[:30]}")
            with open(debug_path, 'a') as f:
                f.write(f"\n=== Tank: {tank_id}, source_node: {source_node} ===\n")
                f.write(f"detailed_links keys: {list(detailed_links.keys()) if detailed_links else 'None'}\n")
                f.write(f"source_node in detailed_links: {source_node in detailed_links if detailed_links else 'N/A'}\n")
                if detailed_links and source_node in detailed_links:
                    f.write(f"derivation for this node: {detailed_links[source_node].get('derivation', [])}\n")
                f.write(f"link_hydrographs keys (first 30): {list(solution.link_hydrographs.keys())[:30]}\n")
            
            # CRITICAL FIX: We need to find the derivation pipe flow (like "0.0-0.1")
            # NOT the weir flow (like "WR_P0061405_0") which has zero values
            
            # Strategy 1: Use derivation link from detailed_links (most reliable)
            if detailed_links and source_node in detailed_links:
                derivation_links = detailed_links[source_node].get('derivation', [])
                for deriv_link in derivation_links:
                    if deriv_link in solution.link_hydrographs:
                        link_data = solution.link_hydrographs[deriv_link]
                        link_times = link_data.get('times', [])
                        link_flow = link_data.get('flow', [])
                        if link_times and link_flow and len(link_flow) > 0:
                            weir_hrs = [(t - link_times[0]).total_seconds() / 3600.0 for t in link_times]
                            weir_flow = link_flow
                            with open(debug_path, 'a') as f:
                                f.write(f"FOUND via Strategy 1: {deriv_link}, max_flow={max(link_flow)}\n")
                            break
            
            # Strategy 2: Find ramal links (format "X.Y-X.Z") with actual flow
            # These are the derivation pipes that carry water to tanks
            if weir_flow is None:
                best_flow = None
                best_max = 0
                best_lid = None
                for lid, link_data in solution.link_hydrographs.items():
                    # Match ramal pattern: "0.0-0.1", "0.1-0.2", "1.0-1.1", etc
                    if '-' in lid and lid.count('.') >= 2:
                        parts = lid.split('-')
                        if len(parts) == 2:
                            # Verify both parts have the ramal.node format
                            try:
                                p0 = parts[0].split('.')
                                p1 = parts[1].split('.')
                                if len(p0) == 2 and len(p1) == 2:
                                    link_times = link_data.get('times', [])
                                    link_flow = link_data.get('flow', [])
                                    if link_times and link_flow and len(link_flow) > 0:
                                        max_flow = max(link_flow)
                                        # Keep track of the link with highest flow
                                        if max_flow > best_max:
                                            best_max = max_flow
                                            best_flow = link_flow
                                            best_lid = lid
                                            best_times = link_times
                            except:
                                pass
                
                if best_flow is not None and best_max > 0.01:
                    weir_hrs = [(t - best_times[0]).total_seconds() / 3600.0 for t in best_times]
                    weir_flow = best_flow
            # Strategy 3: Search for WEIR flow (WR_{source_node}_*) - now captured during simulation
            if weir_flow is None:
                for lid, link_data in solution.link_hydrographs.items():
                    if lid.startswith(f'WR_{source_node}_'):
                        link_times = link_data.get('times', [])
                        link_flow = link_data.get('flow', [])
                        if link_times and link_flow and len(link_flow) > 0:
                            weir_hrs = [(t - link_times[0]).total_seconds() / 3600.0 for t in link_times]
                            weir_flow = link_flow
                            print(f"  [TANK-DEBUG] FOUND via Strategy 3 (Weir): {lid}, max_flow={max(link_flow)}")
                            with open(debug_path, 'a') as f:
                                f.write(f"FOUND via Strategy 3 (Weir): {lid}, max_flow={max(link_flow)}\n")
                            break
            
            # Final debug output
            if weir_flow is not None:
                print(f"  [TANK-DEBUG] SUCCESS: Found flow data with {len(weir_flow)} points, max={max(weir_flow):.4f}")
            else:
                print(f"  [TANK-DEBUG] FAILED: All 3 strategies failed to find flow data for tank {tank_id}")
            
            if weir_flow is not None:
                ax1.plot(weir_hrs, weir_flow, 'b-', linewidth=1.5)
                ax1.fill_between(weir_hrs, weir_flow, alpha=0.3, color='blue')
                peak_q = max(weir_flow)
                ax1.axhline(y=peak_q, color='darkblue', linestyle='--', alpha=0.5)
                ax1.text(0.98, 0.95, f'Peak: {peak_q:.2f} m³/s', transform=ax1.transAxes,
                        ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax1.text(0.5, 0.5, 'Weir data\nnot available', transform=ax1.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
            
            ax1.set_title(f'{tank_id}\nWeir Flow (Inflow)')
            ax1.set_xlabel('Time (hours)')
            ax1.set_ylabel('Flow (m³/s)')
            ax1.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
            ax1.grid(True, alpha=0.3)
            
            # ===== COL 2: CUMULATIVE VOLUME =====
            ax2 = axes[i, 1]
            if weir_flow is not None and len(weir_hrs) > 1:
                # Calculate cumulative volume (integral of flow)
                t_sec = np.array(weir_hrs) * 3600.0
                cumulative_vol = np.zeros(len(weir_flow))
                for k in range(1, len(weir_flow)):
                    dt = t_sec[k] - t_sec[k-1]
                    cumulative_vol[k] = cumulative_vol[k-1] + weir_flow[k] * dt
                
                ax2.plot(weir_hrs, cumulative_vol, 'g-', linewidth=2)
                ax2.fill_between(weir_hrs, cumulative_vol, alpha=0.3, color='green')
                final_vol = cumulative_vol[-1]
                ax2.text(0.98, 0.95, f'Total: {final_vol:,.0f} m³', transform=ax2.transAxes,
                        ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'Volume data\nnot available', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
            
            ax2.set_title('Cumulative Volume')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Volume (m³)')
            ax2.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
            ax2.grid(True, alpha=0.3)
            
            # ===== COL 3: TANK DEPTH (FILLING CURVE) =====
            ax3 = axes[i, 2]
            ax3.plot(hrs, depths, 'r-', linewidth=2)
            ax3.fill_between(hrs, depths, alpha=0.3, color='red')
            
            # Add designed depth line
            ax3.axhline(y=designed_depth, color='darkred', linestyle='--', 
                       linewidth=1.5, label=f'Design: {designed_depth}m')
            
            max_depth = max(depths) if depths else 0
            utilization_pct = (max_depth / designed_depth * 100) if designed_depth > 0 else 0
            
            # Annotation box with key metrics
            ax3.text(0.98, 0.95, 
                    f'Max: {max_depth:.2f} m\nDesign: {designed_depth:.1f} m\nUtil: {utilization_pct:.0f}%', 
                    transform=ax3.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
            
            ax3.set_title('Tank Depth (Filling Curve)')
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Depth (m)')
            ax3.set_xlim(0, PLOT_TIME_LIMIT_HOURS)  # Limit X-axis
            ax3.set_ylim(0, max(designed_depth * 1.2, max_depth * 1.1))
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(save_dir) / f'{solution_name}_tank_hydrographs.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  [Tanks] Saved: {save_path}")

    def generate_metric_maps(self, solution: SystemMetrics, solution_name: str, save_dir: Path,
                             nodes_gdf: gpd.GeoDataFrame, inp_path: str):
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
        base_vols = self.baseline.flooding_volumes
        sol_vols = solution.flooding_volumes
        
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
        
        # Create a lookup from node ID to geometry using 'NodeID' column (like _plot_spatial_diff)
        name_to_geom = {}
        for idx in nodes_gdf.index:
            row = nodes_gdf.loc[idx]
            geom = row['geometry']
            # Use 'NodeID' column which contains names like P0070286
            if 'NodeID' in nodes_gdf.columns:
                name_to_geom[str(row['NodeID'])] = geom
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
    
    def _plot_capacity_map(self, ax, links_gdf, baseline, solution):
        """Plots pipe capacity usage on map. Green=improved, Red=worsened."""
        if links_gdf is None or links_gdf.empty:
            ax.text(0.5, 0.5, "No Link Data", ha='center')
            return
        
        # Background network
        links_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # Compute max capacity for each link
        base_caps = {}
        for lid, data in baseline.link_capacities.items():
            if data.get('capacity'):
                base_caps[lid] = max(data['capacity'])
        
        sol_caps = {}
        for lid, data in solution.link_capacities.items():
            if data.get('capacity'):
                sol_caps[lid] = max(data['capacity'])
        
        # Common links
        common_lids = set(base_caps.keys()) & set(sol_caps.keys())
        
        for lid in common_lids:
            if lid in links_gdf.index:
                geom = links_gdf.loc[lid, 'geometry']
                diff = sol_caps[lid] - base_caps[lid]
                color = 'green' if diff < 0 else 'red' if diff > 0 else 'gray'
                lw = 1 + abs(diff) * 3
                gpd.GeoSeries([geom]).plot(ax=ax, color=color, linewidth=min(lw, 4), alpha=0.7)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Improved (Lower Usage)'),
            Line2D([0], [0], color='red', lw=3, label='Worsened (Higher Usage)'),
            Line2D([0], [0], color='gray', lw=1, label='Unchanged')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_flooding_map(self, ax, nodes_gdf, base_vols, sol_vols):
        """Plots node flooding volume difference on map. Green=reduced, Red=increased."""
        if nodes_gdf is None or nodes_gdf.empty:
            ax.text(0.5, 0.5, "No Node Data", ha='center')
            return
        
        # Background nodes
        nodes_gdf.plot(ax=ax, color='lightgray', markersize=5, alpha=0.3)
        
        for nid in nodes_gdf.index:
            nid_str = str(nid)
            base_v = base_vols.get(nid_str, 0)
            sol_v = sol_vols.get(nid_str, 0)
            diff = sol_v - base_v
            
            if abs(diff) > 1:  # Threshold for visibility
                geom = nodes_gdf.loc[nid, 'geometry']
                color = 'green' if diff < 0 else 'red'
                size = min(10 + abs(diff) * 0.1, 50)
                gpd.GeoSeries([geom]).plot(ax=ax, color=color, markersize=size, alpha=0.7)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reduced Flooding'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Increased Flooding')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
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
            plt.savefig(save_dir / f"dashboard_comparison_batch_{batch_idx+1}.png", dpi=100)
            plt.close(fig)

    def _overlay_annotations(self, ax, derivations, annotations_data):
        """Helper to draw annotations on existing map ax."""
        from shapely.geometry import LineString
        for i, item in enumerate(annotations_data):
            if i >= len(derivations): break
            
            q = item['q_peak']
            d = item['diameter']
            vol = item['tank_vol']
            stored = item.get('stored_vol', 0)  # Get stored volume if available
            
            geom = derivations[i]
            if isinstance(geom, LineString):
                 # Midpoint for label
                 mid_pt = geom.interpolate(0.5, normalized=True)
                 mx, my = mid_pt.x, mid_pt.y
                 
                 # Calculate Length
                 length_m = geom.length
                 
                 # Show both design and stored volume
                 if stored > 0:
                     pct = (stored / vol * 100) if vol > 0 else 0
                     label = f"Q: {q:.2f} cms\nD: {d}\nL: {length_m:.0f} m\nDis: {vol:.0f} m³\nCap: {stored:.0f} m³ ({pct:.0f}%)"
                 else:
                     label = f"Q: {q:.2f} cms\nD: {d}\nL: {length_m:.0f} m\nVol: {vol:.0f} m³"
                 
                 # Add text with box
                 ax.text(mx, my, label, fontsize=8, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'),
                             zorder=20, ha='center')



    # ... (other methods) ...

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
                ax_sol.plot(get_hrs(d), d['flow'], color='darkorange', linewidth=2, label=f'Antes: {lid}')
        
        # 2. Derivation (La Derivacion) - Make it standout
        for lid in deriv_ids:
            if lid in sol_hydros:
                d = sol_hydros[lid]
                ax_sol.plot(get_hrs(d), d['flow'], color='blue', linewidth=3, label=f'Derivación: {lid}')
                
        # 3. Downstream (Despues)
        for lid in down_ids:
            if lid in sol_hydros:
                d = sol_hydros[lid]
                # Use dashed or lighter for downstream in solution
                ax_sol.plot(get_hrs(d), d['flow'], color='green', linestyle='-', linewidth=2, label=f'Después: {lid}')
                
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
                ax_base.plot(get_hrs(d), d['flow'], color='darkorange', linestyle='--', alpha=0.7, label=f'Base Antes: {lid}')

        # 2. Downstream Base
        for lid in down_ids:
            if lid in base_hydros:
                d = base_hydros[lid]
                ax_base.plot(get_hrs(d), d['flow'], color='green', linestyle='--', alpha=0.7, label=f'Base Después: {lid}')
                
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
        """Bar chart of % reduction."""
        metrics = ["Volume", "Cost", "Nodes", "Avg Depth"]
        def get_pct(base, curr):
            if base <= 0: return 0.0
            return (base - curr) / base * 100.0
        
        reductions = [
            get_pct(self.baseline.total_flooding_volume, sol.total_flooding_volume),
            get_pct(self.baseline.flooding_cost, sol.flooding_cost),
            get_pct(self.baseline.flooded_nodes_count, sol.flooded_nodes_count),
            get_pct(self.baseline.avg_node_depth, sol.avg_node_depth)
        ]
        
        colors = ['green' if r >= 0 else 'red' for r in reductions]
        bars = ax.bar(metrics, reductions, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("% Improvement")
        ax.set_title("System Improvements")
        ax.set_ylim(bottom=min(min(reductions, default=0), -12) * 1.3, top=105) # More space at bottom
        for bar, val in zip(bars, reductions):
            height = bar.get_height()
            # If negative (Red), put text inside/above bottom if possible or below with enough space?
            # User complained about overlap with axis labels (likely at bottom spine).
            # Putting text just below zero line (top of negative bar) is cleaner if bar is long enough.
            # But let's stick to bar end, just ensure limits are wide enough.
            
            if height >= 0:
                 offset = 3
                 va = 'bottom'
            else:
                 offset = -2 # Slightly below bar end
                 va = 'top' # Hang below
            
            ax.text(bar.get_x() + bar.get_width()/2., height + offset, f'{val:.1f}%', ha='center', va=va, fontweight='bold', fontsize=9)

    def _plot_volume_scatter(self, ax, base_vols: Dict, sol_vols: Dict):
        """Scatter plot of Baseline vs Solution Volume (Log-Log)."""
        all_nodes = set(base_vols.keys()) | set(sol_vols.keys())
        x_vals, y_vals, colors = [], [], []
        
        for nid in all_nodes:
            v_base = base_vols.get(nid, 0.0)
            v_sol = sol_vols.get(nid, 0.0)
            if v_base < 0.1 and v_sol < 0.1: continue
            
            x_vals.append(v_base)
            y_vals.append(v_sol)
            if v_sol < v_base * 0.99: colors.append('green')
            elif v_sol > v_base * 1.01: colors.append('red')
            else: colors.append('gray')
            
        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, edgecolors='k', s=40)
        max_val = max(max(x_vals, default=1), max(y_vals, default=1))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('Baseline Flooding (m3)')
        ax.set_ylabel('Solution Flooding (m3)')
        ax.set_title('Volume Change (Log-Log)')
        ax.loglog()
        ax.grid(True, which="both", alpha=0.3)

    def _plot_spatial_diff(self, ax, nodes_gdf: gpd.GeoDataFrame, base_vols: Dict, sol_vols: Dict, inp_path: str, derivations: List):
        """
        Maps delta flooding + Background Network + New Derivations.
        """
        # 1. Background Network (Gray)
        if inp_path and os.path.exists(inp_path):
            try:
                import swmmio
                from shapely.geometry import LineString
                model = swmmio.Model(inp_path)
                conduits = model.conduits()
                lines = []
                for _, row in conduits.iterrows():
                    if 'coords' in row and isinstance(row['coords'], list) and len(row['coords']) >= 2:
                        lines.append(LineString(row['coords']))
                if lines:
                     gpd.GeoSeries(lines).plot(ax=ax, color='#555555', linewidth=1.5, alpha=0.8, zorder=0)
            except Exception as e:
                print(f"Background map error: {e}")
                
        # 2. New Derivations (Thick Blue)
        if derivations:
            from shapely.geometry import LineString
            for geom in derivations:
                if isinstance(geom, LineString):
                    x, y = geom.xy
                    ax.plot(x, y, color='blue', linewidth=2, zorder=5)
                    # END POINT BLACK (Predio) - User request
                    ax.scatter([x[-1]], [y[-1]], color='black', s=80, marker='o', zorder=6, label='Predio (Terreno)')

        # 3. Nodes (Colored by Delta)
        gdf = nodes_gdf.copy()
        deltas = []
        colors = []
        sizes = []
        
        for idx, row in gdf.iterrows():
            nid = str(row['NodeID'])
            v_base = base_vols.get(nid, 0.0)
            v_sol = sol_vols.get(nid, 0.0)
            delta = v_base - v_sol
            
            if abs(delta) < 1.0: 
                c = 'gray'; s = 5; alpha=0.2
            elif delta > 0: # Improved
                c = 'green'; s = 30 + min(delta, 100)*0.5; alpha=0.8
            else: # Worsened
                c = 'red'; s = 30 + min(abs(delta), 100)*0.5; alpha=0.8
            colors.append(c); sizes.append(s)
        
        gdf['color'] = colors
        gdf['size'] = sizes
        
        gdf.plot(ax=ax, color=gdf['color'], markersize=gdf['size'], alpha=0.7, zorder=2)
        ax.set_title("Spatial Flooding Delta (Green=Improved, Red=Worsened)")
        ax.axis('equal') # Equal scaling
        ax.set_axis_off()

    def _plot_detailed_row_nx3(self, ax_flow, ax_cap, ax_flood, nid, links, baseline, solution):
        """Plots one row (3 cols) for a specific intervention node."""
        
        up_ids = links.get('upstream', [])
        down_ids = links.get('downstream', [])
        deriv_ids = links.get('derivation', [])
        
        def get_hrs(data):
            if not data or not data['times']: return []
            t0 = data['times'][0]
            return [(t - t0).total_seconds()/3600.0 for t in data['times']]

        # --- COL 1: FLOW (Caudal) ---
        ax_flow.set_title(f"Node {nid}: Caudal (Flow)", fontsize=10)
        ax_flow.set_ylabel("Caudal (cms)")
        
        # All lines solid for Solution to see them clearly
        for lid in up_ids:
            if lid in solution.link_hydrographs:
                d = solution.link_hydrographs[lid]
                ax_flow.plot(get_hrs(d), d['flow'], color='orange', label='Antes (Sol)')
        for lid in down_ids:
            if lid in solution.link_hydrographs:
                d = solution.link_hydrographs[lid]
                ax_flow.plot(get_hrs(d), d['flow'], color='green', label='Después (Sol)')
        for lid in deriv_ids:
            if lid in solution.link_hydrographs:
                d = solution.link_hydrographs[lid]
                ax_flow.plot(get_hrs(d), d['flow'], color='blue', linewidth=3, label='Derivación')
        
        # Add Baseline flows (Dashed) for comparison
        for lid in up_ids:
            if lid in baseline.link_hydrographs:
                d = baseline.link_hydrographs[lid]
                ax_flow.plot(get_hrs(d), d['flow'], color='grey', linestyle='--', alpha=0.9, label='Antes (Base)') # Darker grey
        for lid in down_ids:
            if lid in baseline.link_hydrographs:
                d = baseline.link_hydrographs[lid]
                ax_flow.plot(get_hrs(d), d['flow'], color='lightgreen', linestyle='--', alpha=0.9, label='Después (Base)')
                
        ax_flow.legend(fontsize=8)
        ax_flow.grid(True, alpha=0.3)
        
        # --- IDENTITY CHECK ---
        # Check if Upstream/Downstream flow is identical to baseline
        is_identical = False
        diff_msg = ""
        
        # Check first upstream link as proxy
        if up_ids and up_ids[0] in solution.link_hydrographs and up_ids[0] in baseline.link_hydrographs:
            sol_d = solution.link_hydrographs[up_ids[0]]['flow']
            base_d = baseline.link_hydrographs[up_ids[0]]['flow']
            
            # Simple length check first
            if len(sol_d) == len(base_d):
                if np.allclose(sol_d, base_d, atol=1e-4):
                    is_identical = True
                    diff_msg = " [IDENTICAL TO BASELINE]"
                else:
                    diff_sum = np.sum(np.abs(np.array(sol_d) - np.array(base_d)))
                    diff_msg = f" [Diff Sum: {diff_sum:.2f}]"
            else:
                diff_msg = f" [Len Diff: {len(sol_d)}vs{len(base_d)}]"

        if is_identical:
             ax_flow.set_title(f"Node {nid}: Caudal (Flow) - WARNING: IDENTICAL DATA!", fontsize=10, color='red', fontweight='bold')
             ax_flow.text(0.5, 0.5, "IDENTICAL DATA\n(Solution = Baseline)", 
                          transform=ax_flow.transAxes, ha='center', va='center', 
                          color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.5))
        else:
             ax_flow.set_title(f"Node {nid}: Caudal (Flow){diff_msg}", fontsize=10)
        
        # --- COL 2: CAPACITY (Capacidad / Usage) ---
        ax_cap.set_title(f"Capacidad (Uso [0-1])", fontsize=10)
        
        # Plot Link Capacities (Usage fraction)
        for lid in up_ids:
            if lid in solution.link_capacities:
                d = solution.link_capacities[lid]
                ax_cap.plot(get_hrs(d), d['capacity'], color='orange', linestyle='--', label='Antes')
        for lid in down_ids:
            if lid in solution.link_capacities:
                d = solution.link_capacities[lid]
                ax_cap.plot(get_hrs(d), d['capacity'], color='green', linestyle='--', label='Después')
        for lid in deriv_ids:
            if lid in solution.link_capacities:
                d = solution.link_capacities[lid]
                ax_cap.plot(get_hrs(d), d['capacity'], color='blue', linewidth=2, linestyle='-', label='Derivación')
                
        ax_cap.set_ylim(0, 1.1)
        ax_cap.grid(True, alpha=0.3)
        
        # --- COL 3: FLOOD HYDROGRAPH (Node) ---
        ax_flood.set_title(f"Inundación Nodo {nid}", fontsize=10)
        ax_flood.set_ylabel("Profundidad (m)")
        
        # Baseline Flood
        if nid in baseline.flood_hydrographs:
            d = baseline.flood_hydrographs[nid]
            ax_flood.plot(get_hrs(d), d['depths'], color='red', linestyle='--', label='Base Depth')
            
        # Solution Flood
        if nid in solution.flood_hydrographs:
            d = solution.flood_hydrographs[nid]
            ax_flood.plot(get_hrs(d), d['depths'], color='blue', label='Sol Depth')
            
    def _plot_system_capacity_hist(self, ax, baseline, solution):
        """Plots overlaid histograms of max pipe capacity usage for entire system."""
        
        def get_max_capacities(metrics_obj):
            caps = []
            for lid, data in metrics_obj.link_capacities.items():
                if data.get('capacity'):
                    caps.append(max(data['capacity']))
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

    def _plot_hist(self, ax, data_base, data_sol, xlabel):
        if not data_base and not data_sol: return
        # Use KDE lines with transparent fill
        if data_base:
            sns.kdeplot(data_base, ax=ax, color='red', linewidth=2, label='Baseline', fill=True, alpha=0.15)
        if data_sol:
            sns.kdeplot(data_sol, ax=ax, color='blue', linewidth=2, label='Solution', fill=True, alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.set_title(f"Distribution: {xlabel}")
        
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

    def generate_tank_hydrograph_plots(self, solution: SystemMetrics, solution_name: str, save_dir: Path, detailed_links: Dict = None):
        """Generates plots for detailed tank hydrographs (Inflow, Volume, Depth). Max 4 per page."""
        if not solution.tank_utilization:
            return
            
        tank_ids = list(solution.tank_utilization.keys())
        tank_ids.sort()
        
        batch_size = 4
        
        for batch_idx, i in enumerate(range(0, len(tank_ids), batch_size)):
            batch_tanks = tank_ids[i : i + batch_size]
            
            n_rows = len(batch_tanks)
            # 3 Columns: Inflow (calc), Volume (calc), Depth (measured)
            fig = plt.figure(figsize=(24, 5 * n_rows))
            gs = fig.add_gridspec(n_rows, 3)
            
            for row_idx, tk_id in enumerate(batch_tanks):
                util_data = solution.tank_utilization[tk_id]
                max_depth = util_data.get('max_depth', 0)
                design_depth = 5.0
                
                # Extract source_node from tank_id (TK_P0061405_4 -> P0061405)
                parts = tk_id.split('_')
                source_node = parts[1] if len(parts) > 1 else tk_id
                
                # Try to get design_vol from detailed_links (passed from evaluator)
                design_vol = util_data.get('design_vol', 0.0)
                
                # FIX: If design_vol not available, estimate from stored_volume or max_volume
                if design_vol <= 0:
                    stored_vol = util_data.get('stored_volume', 0.0)
                    max_vol_data = util_data.get('max_volume', 0.0)
                    if stored_vol > 0:
                        # Estimate design capacity assuming 80% utilization typically
                        design_vol = stored_vol / 0.8
                    elif max_depth > 0:
                        # Fallback: assume a reasonable tank (default 20000 m³)
                        # This will at least show SOMETHING instead of nothing
                        design_vol = 20000.0
                        print(f"  [TANK-HYDRO] Warning: Using default design_vol={design_vol} for {tk_id}")
                
                times = util_data.get('times', [])
                depths = util_data.get('depth_series', [])
                
                # Axes
                ax_flow = fig.add_subplot(gs[row_idx, 0])
                ax_vol = fig.add_subplot(gs[row_idx, 1])
                ax_depth = fig.add_subplot(gs[row_idx, 2])
                
                if times and depths:
                     # Time axis
                     if hasattr(times[0], 'hour'):
                          hrs = [(t - times[0]).total_seconds()/3600.0 for t in times]
                     else:
                          hrs = times 
                     
                     # 1. Calculate Volume Series
                     # Area = DesignVol / DesignDepth
                     area = (design_vol / design_depth) if design_depth > 0 else 0
                     vols = [d * area for d in depths]
                     max_vol_reached = max(vols) if vols else 0
                     
                     # 2. Calculate Inflow (Approx dV/dt)
                     inflow = [0.0] * len(vols)
                     if len(vols) > 1:
                         for k in range(1, len(vols)):
                             dt_sec = (hrs[k] - hrs[k-1]) * 3600.0
                             if dt_sec > 0:
                                 # Q = dV / dt
                                 val = (vols[k] - vols[k-1]) / dt_sec
                                 # Smooth out noise/negative flows (sloshing)
                                 inflow[k] = max(0, val) 
                     
                     # --- PLOT 1: INFLOW (Derived) ---
                     ax_flow.plot(hrs, inflow, color='blue', linewidth=1.5, label='Est. Inflow')
                     ax_flow.set_title(f"{tk_id}\nWeir Flow (Inflow)", fontweight='bold')
                     ax_flow.set_ylabel("Flow (m³/s)")
                     ax_flow.fill_between(hrs, inflow, color='blue', alpha=0.1)
                     ax_flow.grid(True, alpha=0.3)
                     ax_flow.text(0.95, 0.95, f"Peak: {max(inflow):.2f} m³/s", transform=ax_flow.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

                     # --- PLOT 2: VOLUME ---
                     ax_vol.plot(hrs, vols, color='green', linewidth=2, label='Volume')
                     ax_vol.set_title("Cumulative Volume")
                     ax_vol.set_ylabel("Volume (m³)")
                     ax_vol.fill_between(hrs, vols, color='green', alpha=0.2)
                     ax_vol.axhline(design_vol, color='darkgreen', linestyle='--', label='Capacity')
                     ax_vol.grid(True, alpha=0.3)
                     ax_vol.text(0.95, 0.95, f"Stored: {max_vol_reached:,.0f} m³", transform=ax_vol.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

                     # --- PLOT 3: DEPTH ---
                     ax_depth.plot(hrs, depths, color='tab:red', linewidth=2, label='Depth')
                     ax_depth.axhline(design_depth, color='darkred', linestyle='--', label='Design (5m)')
                     ax_depth.fill_between(hrs, depths, color='tab:red', alpha=0.2)
                     
                     # Label
                     vol_str = f"{design_vol:,.0f} m³" if design_vol > 0 else "N/A"
                     label_text = (f"Design Vol: {vol_str}\n"
                                   f"Design Depth: {design_depth} m\n"
                                   f"Max Depth: {max_depth:.2f} m\n"
                                   f"Util: {(max_depth/design_depth)*100:.0f}%")
                     ax_depth.text(0.98, 0.95, label_text, transform=ax_depth.transAxes, 
                                   ha='right', va='top', fontsize=10,
                                   bbox=dict(facecolor='lemonchiffon', alpha=0.7, edgecolor='orange'))
                     ax_depth.set_title("Tank Depth (Filling Curve)")
                     ax_depth.set_ylabel("Depth (m)")
                
                # Common X-Axis Label
                ax_flow.set_xlabel("Time (h)")
                ax_vol.set_xlabel("Time (h)")
                ax_depth.set_xlabel("Time (h)")
                
                # Limit X-axis to configured duration
                ax_flow.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                ax_vol.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                ax_depth.set_xlim(0, PLOT_TIME_LIMIT_HOURS)
                
            fig.suptitle(f"{solution_name}: Tank Hydrographs (Page {batch_idx+1})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_dir / f"{solution_name}_tank_hydrographs_page_{batch_idx+1}.png", dpi=100)
            plt.close(fig)
            print(f"  [Tanks] Saved page {batch_idx+1}")

if __name__ == "__main__":
    # Test logic
    pass
