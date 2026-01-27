import os
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any
from pyswmm import Simulation
from swmmio import Model
from swmmio.utils.modify_model import replace_inp_section

import config
config.setup_sys_path()

class HydrologicalImpactAssessment:
    """
    Calculates hydrological impact by comparing peak discharges and volumes 
    across outfalls under three scenarios:
    1. GREEN (Natural Baseline): All subcatchments forced to forest/natural parameters.
    2. GREY BASELINE: The current urbanized state without tanks.
    3. GREY SOLUTION: The current state with optimized tanks.
    
    Both Green and Grey baselines are cached at class level to avoid redundant 
    simulation on subsequent iterations. Only Grey+Tanks runs each iteration.
    """
    
    # Class-level caches to avoid regenerating baselines each iteration
    _green_baseline_cache: Dict[str, str] = {}  # baseline_inp -> green_inp_path
    _grey_baseline_cache: Dict[str, Dict] = {}  # baseline_inp -> {'df': DataFrame, 'hydro': dict}
    _green_results_cache: Dict[str, Dict] = {}  # green_inp -> {'df': DataFrame, 'hydro': dict}
    
    def __init__(self, output_dir: str):
        self.cas_dir = Path(output_dir)
        self.output_dir = Path(output_dir) / "hydrological_impact"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.green_inp = self.output_dir / "model_green_baseline.inp"
        self.results = {} # scenario -> DataFrame of outfall results
        self.hydrographs = {} # scenario -> {time: [], total_q: []}
        
    def generate_natural_model(self, base_inp: str):
        """
        Creates a 'Natural' version of the INP using config.GREEN_SCENARIO settings.
        Sets %Imperv and CN for all subcatchments to reflect a natural/green state.
        """
        print(f"  [HydroImpact] Generating Green (Natural) Baseline from {Path(base_inp).name}...")
        
        # 1. Copy the baseline to green location FIRST (to avoid overwriting original)
        import shutil
        shutil.copy(str(base_inp), str(self.green_inp))
        
        # 2. Work on the COPIED file
        model = Model(str(self.green_inp))
        
        # 3. Modify [SUBCATCHMENTS] - Set %Imperv to natural residual value
        subcats = model.inp.subcatchments
        if subcats is not None and not subcats.empty:
            subcats['PercImperv'] = config.GREEN_SCENARIO_IMPERV
            replace_inp_section(str(self.green_inp), '[SUBCATCHMENTS]', subcats)
            
        # 4. Modify [INFILTRATION] - Adjust CN to Green baseline
        infil = model.inp.infiltration
        if infil is not None and not infil.empty:
            if 'CurveNum' in infil.columns:
                infil['CurveNum'] = config.GREEN_SCENARIO_CN
            elif 'CN' in infil.columns:
                infil['CN'] = config.GREEN_SCENARIO_CN
            else:
                infil.iloc[:, 0] = config.GREEN_SCENARIO_CN
            
            replace_inp_section(str(self.green_inp), '[INFILTRATION]', infil)
            
        print(f"  [HydroImpact] Green Model finalized at: {self.green_inp}")
        return str(self.green_inp)

    def run_assessment(self, baseline_inp: str, solution_inp: str):
        """Runs the three scenarios and compares all outfalls.
        
        Green and Grey baselines are cached to avoid redundant simulations.
        Only Grey+Tanks (solution) runs on each iteration.
        """
        baseline_key = str(baseline_inp)
        
        # === GREEN BASELINE (Cached) ===
        if baseline_key in HydrologicalImpactAssessment._green_results_cache:
            print(f"  [HydroImpact] Using cached Green Baseline results")
            green_results = HydrologicalImpactAssessment._green_results_cache[baseline_key]
            self.results['Natural (Green)'] = green_results['df']
            self.hydrographs['Natural (Green)'] = green_results['hydro']
        else:
            # Generate and simulate Green model
            if baseline_key in HydrologicalImpactAssessment._green_baseline_cache:
                green_path = HydrologicalImpactAssessment._green_baseline_cache[baseline_key]
            else:
                green_path = self.generate_natural_model(baseline_inp)
                HydrologicalImpactAssessment._green_baseline_cache[baseline_key] = green_path
            
            print(f"  [HydroImpact] Simulating Natural (Green)...")
            df, sys_q = self._extract_outfall_metrics(green_path)
            self.results['Natural (Green)'] = df
            self.hydrographs['Natural (Green)'] = sys_q
            HydrologicalImpactAssessment._green_results_cache[baseline_key] = {'df': df, 'hydro': sys_q}
        
        # === GREY BASELINE (Cached) ===
        if baseline_key in HydrologicalImpactAssessment._grey_baseline_cache:
            print(f"  [HydroImpact] Using cached Grey Baseline results")
            grey_results = HydrologicalImpactAssessment._grey_baseline_cache[baseline_key]
            self.results['Baseline (Grey)'] = grey_results['df']
            self.hydrographs['Baseline (Grey)'] = grey_results['hydro']
        else:
            print(f"  [HydroImpact] Simulating Baseline (Grey)...")
            df, sys_q = self._extract_outfall_metrics(baseline_inp)
            self.results['Baseline (Grey)'] = df
            self.hydrographs['Baseline (Grey)'] = sys_q
            HydrologicalImpactAssessment._grey_baseline_cache[baseline_key] = {'df': df, 'hydro': sys_q}
        
        # === GREY + TANKS (Solution - Always runs) ===
        print(f"  [HydroImpact] Simulating Solution (Grey+Tanks)...")
        df, sys_q = self._extract_outfall_metrics(solution_inp)
        self.results['Solution (Grey+Tanks)'] = df
        self.hydrographs['Solution (Grey+Tanks)'] = sys_q
            
        # Combine results into a comparison report
        self._generate_comparison_plots()
        # self._generate_summary_table()
        
    def _extract_outfall_metrics(self, inp_path: str) -> (pd.DataFrame, Dict):
        """Runs SWMM and extracts peak Q, total volume using SystemSeries.outfall_flows."""
        out_path = Path(inp_path).with_suffix('.out')
        
        # Run simulation
        with Simulation(inp_path, outputfile=str(out_path)) as sim:
            for _ in sim: pass
            
        from pyswmm import Output, SystemSeries
        from swmm.toolkit.shared_enum import NodeAttribute
        
        with Output(str(out_path)) as out:
            # Use SystemSeries.outfall_flows for system-wide outfall discharge
            system_outfall_series = pd.Series(SystemSeries(out).outfall_flows)
            
            # Calculate peak Q and volume from system series
            peak_q_total = system_outfall_series.max()
            
            # Integrate for total volume
            if len(system_outfall_series) > 1:
                times = system_outfall_series.index
                t_start = times[0]
                t_secs = [(t - t_start).total_seconds() for t in times]
                vol_m3_total = np.trapz(system_outfall_series.values, x=t_secs)
            else:
                vol_m3_total = 0.0
            
            # Create summary DataFrame (single row for system total)
            metrics = [{
                'OutfallID': 'SYSTEM_TOTAL',
                'Peak_Q_cms': peak_q_total,
                'Volume_m3': vol_m3_total
            }]
            
            # Also get per-outfall data for detailed analysis
            model = Model(inp_path)
            official_outfalls = model.inp.outfalls.index.tolist()
            
            for oid in official_outfalls:
                flow_series = pd.Series(out.node_series(oid, NodeAttribute.TOTAL_INFLOW))
                if flow_series.empty:
                    continue
                
                peak_q = flow_series.max()
                
                if len(flow_series) > 1:
                    times = flow_series.index
                    t_start = times[0]
                    t_secs = [(t - t_start).total_seconds() for t in times]
                    vol_m3 = np.trapz(flow_series.values, x=t_secs)
                else:
                    vol_m3 = 0.0
                    
                metrics.append({
                    'OutfallID': oid,
                    'Peak_Q_cms': peak_q,
                    'Volume_m3': vol_m3
                })
                
        sys_dict = {
            'times': system_outfall_series.index.tolist() if len(system_outfall_series) > 0 else [],
            'values': system_outfall_series.values.tolist() if len(system_outfall_series) > 0 else []
        }
        return pd.DataFrame(metrics), sys_dict

    def _generate_comparison_plots(self):
        """Generates summary bar charts and the system-wide hydrograph comparison."""
        if not self.results: return
        
        # 1. Total Summary Figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2)
        
        ax_q = fig.add_subplot(gs[0, 0])
        ax_vol = fig.add_subplot(gs[0, 1])
        ax_hydro = fig.add_subplot(gs[1, :])
        
        summary = []
        for name, df in self.results.items():
            summary.append({
                'Scenario': name,
                'Total_Peak_Q': df['Peak_Q_cms'].sum(),
                'Total_Volume': df['Volume_m3'].sum()
            })
        summary_df = pd.DataFrame(summary)
        
        colors = ['#2ecc71', '#e74c3c', '#3498db'] # Green, Red (Grey), Blue (Solution)
        
        # Peak Q Bar
        summary_df.plot(kind='bar', x='Scenario', y='Total_Peak_Q', ax=ax_q, color=colors, legend=False)
        ax_q.set_title("Total System Peak Discharge (m³/s)", fontweight='bold')
        for i, v in enumerate(summary_df['Total_Peak_Q']):
            ax_q.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
        
        # Volume Bar
        summary_df.plot(kind='bar', x='Scenario', y='Total_Volume', ax=ax_vol, color=colors, legend=False)
        ax_vol.set_title("Total System Runoff Volume (m³)", fontweight='bold')
        for i, v in enumerate(summary_df['Total_Volume']):
            ax_vol.text(i, v, f"{v:,.0f}", ha='center', va='bottom', fontweight='bold')
            
        # 2. System Hydrograph Comparison
        for i, (name, hg) in enumerate(self.hydrographs.items()):
            if len(hg['times']) > 0:
                ax_hydro.plot(hg['times'], hg['values'], label=name, color=colors[i], linewidth=2.5)
        
        ax_hydro.set_title("Total System Discharge Hydrograph (Sum of all outfalls)", fontweight='bold')
        ax_hydro.set_ylabel("Flow Rate (m³/s)")
        ax_hydro.set_xlabel("Simulation Time")
        ax_hydro.legend()
        ax_hydro.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.cas_dir/ "06_hydrological_comparison.png", dpi=150)
        plt.close()
        print(f"  [HydroImpact] Summary plots saved to: {self.output_dir / 'hydrological_comparison_summary.png'}")

    def _generate_spatial_impact_map(self):
        """Plots outfalls on a map with marker size proportional to peak flow."""
        try:
            # Get coordinates for outfalls
            import swmmio
            model = swmmio.Model(str(self.green_inp))
            nodes = model.inp.coordinates
            
            data_to_plot = []
            for name, df in self.results.items():
                temp_df = df.copy()
                temp_df['Scenario'] = name
                # Join with coordinates (swmmio coordinates are indexed by ID)
                temp_df = temp_df.merge(nodes, left_on='OutfallID', right_index=True)
                data_to_plot.append(temp_df)
            
            if not data_to_plot: return
            
            full_df = pd.concat(data_to_plot)
            gdf = gpd.GeoDataFrame(full_df, geometry=gpd.points_from_xy(full_df.X, full_df.Y))
            
            # Create side-by-side plots for each scenario
            scen_names = list(self.results.keys())
            fig, axes = plt.subplots(1, len(scen_names), figsize=(22, 10), sharex=True, sharey=True)
            
            max_q_overall = gdf['Peak_Q_cms'].max()
            scale = 2500 / max_q_overall if max_q_overall > 0 else 1
            
            for i, name in enumerate(scen_names):
                ax = axes[i]
                subset = gdf[gdf['Scenario'] == name]
                
                # Plot markers (impact circles)
                subset.plot(ax=ax, markersize=subset['Peak_Q_cms'] * scale, 
                            alpha=0.6, color='#3498db', edgecolor='black', linewidth=0.5)
                
                # Plot anchor dots for all outfalls
                subset.plot(ax=ax, markersize=8, color='black', alpha=0.4)
                
                ax.set_title(f"Peak Discharge (cms): {name}", fontweight='bold', fontsize=14)
                ax.set_aspect('equal')
                ax.set_axis_off()
                
                # Add outfall labels for significant ones (> 5% of max peak)
                for x, y, label, q in zip(subset.geometry.x, subset.geometry.y, subset.OutfallID, subset.Peak_Q_cms):
                    if q > max_q_overall * 0.05:
                        ax.text(x, y, f"{label}\n({q:.1f})", fontsize=8, ha='center', va='center', 
                                fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

            plt.tight_layout()
            plt.savefig(self.output_dir / "hydrological_spatial_impact.png", dpi=150)
            plt.close()
            print(f"  [HydroImpact] Spatial impact map saved to: {self.output_dir / 'hydrological_spatial_impact.png'}")
            
        except Exception as e:
            print(f"  [Warning] Failed to generate spatial impact map: {e}")
            import traceback
            traceback.print_exc()
        
    def _generate_summary_table(self):
        """Saves an Excel with outfall-by-outfall comparison."""
        # Merge all result dataframes
        merged = None
        for name, df in self.results.items():
            df_renamed = df.rename(columns={
                'Peak_Q_cms': f'PeakQ_{name}',
                'Volume_m3': f'Vol_{name}'
            })
            if merged is None:
                merged = df_renamed
            else:
                merged = pd.merge(merged, df_renamed, on='OutfallID', how='outer')
        
        # Calculate reductions % Baseline vs Solution
        if 'PeakQ_Baseline (Grey)' in merged.columns and 'PeakQ_Solution (Grey+Tanks)' in merged.columns:
            merged['Q_Reduction_Pct'] = (1 - merged['PeakQ_Solution (Grey+Tanks)'] / merged['PeakQ_Baseline (Grey)']) * 100
        
        merged.to_excel(self.output_dir / "hydrological_impact_per_outfall.xlsx", index=False)
        print(f"  [HydroImpact] Summary Excel saved to: {self.output_dir / 'hydrological_impact_per_outfall.xlsx'}")

if __name__ == "__main__":
    # Test execution dummy (requires valid paths setup in a real environment)
    pass
