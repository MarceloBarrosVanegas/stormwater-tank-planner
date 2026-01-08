import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
import scipy.stats as stats
import config

class RiskAnalyzer:
    """
    Advanced probabilistic risk analysis for ITZI flood damage results.
    Performs Bootstrap analysis on EAD and generates spatial risk maps.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        
        # Create organized subdirectories
        self.dirs = {
            'risk': self.output_dir / "03_risk_results",
            'maps': self.output_dir / "01_scenario_maps",
            'fragility': self.output_dir / "02_fragility",
            'spatial': self.output_dir / "04_spatial_analysis"
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # System Baselines (for normalization)
        self.total_property_value = 0.0
        self.total_construction_value = 0.0
        self.total_property_count = 0.0 # New
        self.total_network_length = 0.0
        self.total_pipe_count = 0.0 # New
        self.total_network_value = 0.0 
        self._calculate_system_baselines()

    def _calculate_system_baselines(self):
        """Calculates total exposed values for normalization."""
        print("  [RiskAnalyzer] Calculating system baselines...")
        
        try:
            # Search in PARENT directory as output_dir is a subdir
            sample_gpkg = list(self.output_dir.parent.glob("**/TR_010_flood_damage_results.gpkg"))
            if sample_gpkg:
                gdf = gpd.read_file(sample_gpkg[0])
                if 'estimated_value_usd' in gdf.columns:
                    self.total_property_value = gdf['estimated_value_usd'].sum()
                    self.total_property_count = len(gdf) # Count total exposed
                    print(f"    -> Total Property Value: ${self.total_property_value:,.0f}")
                    print(f"    -> Total Property Count: {self.total_property_count:,.0f}")
        except Exception as e:
            print(f"    [Warn] Could not calc total prop value: {e}")

        # 1b. Total Construction Value
        # Sum 'valconstru' or '_valconstru' from the sample GPKG
        try:
            if sample_gpkg:
                gdf = gpd.read_file(sample_gpkg[0])
                if '_valconstru' in gdf.columns:
                    self.total_construction_value = gdf['_valconstru'].sum()
                elif 'valconstru' in gdf.columns:
                    self.total_construction_value = gdf['valconstru'].sum()
                else:
                    print("    [Warn] '_valconstru' column missing. Using fallback (60% of prop value).")
                    self.total_construction_value = self.total_property_value * 0.6 
                
                print(f"    -> Total Construction Value: ${self.total_construction_value:,.0f}")
        except Exception as e:
             print(f"    [Error] Critical failure calculating Total Construction Value: {e}")
             self.total_construction_value = 0.0




        # 2. Total Network Length & Value
        # We use config.NETWORK_FILE
        try:
            net_path = config.NETWORK_FILE
            if net_path.exists():
                net_gdf = gpd.read_file(net_path)
                self.total_network_length = net_gdf.geometry.length.sum()
                self.total_pipe_count = len(net_gdf) # Count total network elements
                print(f"    -> Total Network Length: {self.total_network_length:,.0f} m")
                print(f"    -> Total Pipe Count: {self.total_pipe_count:,.0f}")
                
                # Estimate Total Network Value using SewerConstructionCost
                # We need to import it inside to avoid circular imports if any
                # Estimate Total Network Value using DeferredInvestmentCost (rut_20)
                # to ensure consistency with the user's methodology (dimensioning + replacement cost).
                from rut_20_avoided_costs import DeferredInvestmentCost
                
                print(f"    [RiskAnalyzer] Calculating FULL Network Replacement Cost (via rut_20)...")
                try:
                    # 1. Initialize Calculator
                    dic = DeferredInvestmentCost(base_precios_path=config.BASE_PRECIOS)
                    
                    # 2. Extract Full Network stats from SWMM INP (Dimensions, Depth, etc.)
                    # We use the Base INP defined in config. MUST be string for rut_25.
                    full_gdf = dic.extract_link_data(str(config.SWMM_FILE))
                    
                    # 3. Optimize/Dimension & Price the entire network
                    # Output to a cache dir
                    cache_dir = self.output_dir / "cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    estimated_val = dic._calculate_real_cost(full_gdf, output_dir=str(cache_dir))
                    
                    self.total_network_value = estimated_val
                    print(f"    -> Total Network Value (Calculated): ${self.total_network_value:,.0f}")
                    
                except Exception as ex_calc:
                    print(f"    [Warn] Failed to calc network value: {ex_calc}")
                    print("    -> Using Proxy: 250 USD/m")
                    self.total_network_value = self.total_network_length * 250.0
        except Exception as e:
            print(f"    [Warn] Could not calc network baseline: {e}")
        
    def analyze_ead_uncertainty(self, metrics_df, damage_gpkgs, extra_costs=None, 
                                n_boot=1000, fragility_curve=None):
        """
        Performs Bootstrap simulation to estimate EAD uncertainty.
        
        Args:
            metrics_df: DataFrame with columns ['tr', 'flood_damage_usd'] (used for reference)
            damage_gpkgs: Dict {tr: gpkg_path} to access property-level data
            extra_costs: Dict {tr: cost_usd} of non-spatial costs (e.g. pipe replacement)
            fragility_curve (dict): {TR: probability_of_failure} (0.0 to 1.0).
        """
        print(f"\n[RiskAnalyzer] Running Probabilistic EAD Analysis (n={n_boot})...")
        
        # 1. LOAD PROPERTY DATA
        tr_list = sorted(damage_gpkgs.keys())
        print(f"  Loading property-level data for {len(tr_list)} return periods...")
        
        # Initialize with the first GPKG to get geometries
        first_gpkg = damage_gpkgs[tr_list[0]]
        base_gdf = gpd.read_file(first_gpkg)
        prop_meta = base_gdf[['_sector', 'geometry']].copy()
        
        # Build Matrix: Rows=Properties, Cols=TRs
        damage_matrix = pd.DataFrame(index=base_gdf.index)
        
        for tr in tr_list:
            gdf = gpd.read_file(damage_gpkgs[tr])
            if len(gdf) != len(base_gdf):
               print(f"  [Warning] Mismatch in property count for TR {tr}. Alignment may be incorrect.")
            damage_matrix[tr] = gdf['damage_usd']
            
        # Filter unaffected for analysis speed, BUT keep full metadata for plotting
        full_prop_meta = prop_meta.copy() 
        affected_mask = damage_matrix.sum(axis=1) > 0
        damage_matrix = damage_matrix[affected_mask]
        prop_meta = prop_meta[affected_mask]
        
        print(f"  Analyzing {len(damage_matrix)} affected properties (out of {len(full_prop_meta)} total).")
        
        # Prepare Inputs for Analysis
        probs = 1.0 / np.array(tr_list)
        
        # Prepare Extra Costs Vector
        extra_costs_vector = np.zeros(len(tr_list))
        if extra_costs:
            for i, tr in enumerate(tr_list):
                extra_costs_vector[i] = extra_costs.get(tr, 0.0)
        
        # Ensure fragility curve defaults if None (for usage in Bootstrap below)
        if fragility_curve is None:
            fragility_curve = {tr: 1.0 for tr in tr_list}
            
        # ---------------------------------------------------------
        # GENERATE NEW REAL FRAGILITY CURVES (PLOT)
        # ---------------------------------------------------------
        # Build summary DF for plotting
        records_frag = []
        for i, tr in enumerate(tr_list):
            flood_mean = damage_matrix[tr].sum()
            infra_mean = extra_costs_vector[i] if extra_costs_vector is not None else 0.0
            records_frag.append({
                'tr': tr,
                'flood_damage': flood_mean,
                'infrastructure_cost': infra_mean
            })
        metrics_df_frag = pd.DataFrame(records_frag)
        
        # Call the new calculation and plotting
        real_fragility_df = self.calculate_fragility_curves(metrics_df_frag)
        # ---------------------------------------------------------

        boot_eads_total = []
        boot_eads_flood = []
        
        # Matrix operations for speed
        n_props = len(damage_matrix)
        
        for i in range(n_boot):
            # 1. FLOOD DAMAGE: Property-Level Bootstrapping (True Statistical Uncertainty)
            # Resample indices with replacement
            sample_idx = np.random.randint(0, n_props, size=n_props)
            
            # Sum damages for the sample across all TRs
            # Result: vector of length n_TRs
            sample_building_damage = damage_matrix.iloc[sample_idx].sum().values
            
            # 2. INFRASTRUCTURE COSTS
            # User requirement: The cost vector represents "Needed Investment" (Determininstic Consequence).
            # Do NOT apply Bernoulli reduction (Fragility) here, as it artificially lowers the risk.
            # We assume if the event occurs (TR), the cost is incurred.
            if extra_costs_vector is not None:
                # Apply small Gaussian noise for valuation uncertainty (+/- 5%)
                noise = np.random.normal(1.0, 0.05) 
                sampled_extra = extra_costs_vector * noise
            else:
                sampled_extra = np.zeros_like(sample_building_damage)
            
            # 3. CALCULATE EAD FOR THIS ITERATION
            # Sort by probability for integration (descending probs)
            sort_ord = np.argsort(probs)
            sorted_probs = probs[sort_ord]
            
            sorted_flood = sample_building_damage[sort_ord]
            sorted_total = (sample_building_damage + sampled_extra)[sort_ord]
            
            # Integrate EAD (Trapezoidal)
            ead_flood = np.trapz(sorted_flood, sorted_probs)
            ead_total = np.trapz(sorted_total, sorted_probs)
            
            boot_eads_total.append(abs(ead_total))
            boot_eads_flood.append(abs(ead_flood))
            
        boot_eads_total = np.array(boot_eads_total)
        boot_eads_flood = np.array(boot_eads_flood)
        
        # 3. Statistics & Plots (TOTAL)
        mean_total = np.mean(boot_eads_total)
        self._plot_bootstrap_hist(boot_eads_total, mean_total, 
                                  np.percentile(boot_eads_total, 2.5), 
                                  np.percentile(boot_eads_total, 97.5),
                                  filename=self.dirs['risk'] / "ead_dist_total.png",
                                  title="Distribution of Total Expected Annual Damage (EAD)"
            )
        print(f"    -> Bootstrap plot saved to {self.dirs['risk']}") # Green

        # 3b. Statistics & Plots (FLOOD ONLY)
        mean_flood = np.mean(boot_eads_flood)
        self._plot_bootstrap_hist(boot_eads_flood, mean_flood, 
                                  np.percentile(boot_eads_flood, 2.5), 
                                  np.percentile(boot_eads_flood, 97.5),
                                  filename=self.dirs['risk'] / "ead_dist_flood.png",
                                  title_suffix="(Flood Damage Only)",
                                  color='#1f78b4') # Blue

        # 3c. Statistics & Plots (INVESTMENT ONLY)
        # Investment EAD = Total EAD - Flood EAD (per iteration)
        boot_eads_invest = boot_eads_total - boot_eads_flood
        mean_invest = np.mean(boot_eads_invest)
        self._plot_bootstrap_hist(boot_eads_invest, mean_invest,
                                  np.percentile(boot_eads_invest, 2.5),
                                  np.percentile(boot_eads_invest, 97.5),
                                  filename=self.dirs['risk'] / "ead_dist_investment.png",
                                  title_suffix="(Infrastructure Repair Only)",
                                  color='#ff7f00') # Orange
                                  
        # 3d. Combined Histogram (Overlay)
        self._plot_combined_bootstrap_hist(boot_eads_flood, boot_eads_invest, boot_eads_total,
                                           filename=self.dirs['risk'] / "ead_dist_combined.png")

        print(f"\n  [Results Summary]")
        print(f"  Mean EAD Flood:      ${mean_flood:,.0f}")
        print(f"  Mean EAD Investment: ${mean_invest:,.0f}")
        print(f"  Mean EAD Total:      ${mean_total:,.0f}")
        
        # 4. Integrate Extra Costs into Spatial Map Title?
        extra_ead = mean_invest
        
        # 5. Spatial EAD Analysis (Map)
        # 5. Spatial EAD Analysis (Map)

        # 5b. Side-by-Side EAD Map (Requested Format)
        self._generate_side_by_side_ead_map(damage_matrix, probs, full_prop_meta, tr_list)
        
        # 5c. Failure Probability Maps (Spatial Analysis)
        self._generate_failure_probability_maps(damage_matrix, probs, full_prop_meta, tr_list)
        
        # 5d. Generate Individual Scenario Maps
        print("DEBUG: Calling _generate_scenario_maps")
        self._generate_scenario_maps(damage_gpkgs, full_prop_meta)
        
        # 6. Sector Analysis (Both)
        self._analyze_sectors_probabilistic(damage_matrix, prop_meta, probs, extra_costs_vector, filename=self.dirs['risk'] / "sector_risk_total.png")
        self._analyze_sectors_probabilistic(damage_matrix, prop_meta, probs, extra_costs_vector=None, filename=self.dirs['risk'] / "sector_risk_flood.png")

        # 7. Risk Curves (Prob vs Cost) - "The Curve"
        raw_flood_per_tr = damage_matrix.sum().values 
        
        # Apply sorting
        sort_ord = np.argsort(probs)
        sorted_probs = probs[sort_ord]
        sorted_raw_flood = raw_flood_per_tr[sort_ord]
        sorted_extra = extra_costs_vector[sort_ord]
        
        # Apply Fragility to expected curve (for visualization)
        # Expected Extra = Nominal * PoF
        if fragility_curve:
            expected_extra = []
            for tr_val, cost in zip(damage_matrix.columns[sort_ord], sorted_extra):
                pof = fragility_curve.get(tr_val, 1.0)
                # expected cost = nominal * pof
                expected_extra.append(cost * pof)
            sorted_extra = np.array(expected_extra)

        sorted_raw_total = sorted_raw_flood + sorted_extra
        
        # A. Flood Only Curve
        self._plot_risk_curve(sorted_probs, sorted_raw_flood, 
                              title="Risk Curve: Flood Damage (Expected)", 
                              filename=self.dirs['risk'] / "risk_curve_flood.png",
                              color='#1f78b4')
                              
        # B. Investment Only Curve
        self._plot_risk_curve(sorted_probs, sorted_extra,
                              title="Risk Curve: Infrastructure Repair (Expected)",
                              filename=self.dirs['risk'] / "risk_curve_investment.png",
                              color='#ff7f00')

        # C. Stacked Curve (Total Breakdown)
        self._plot_stacked_risk_curve(sorted_probs, sorted_raw_flood, sorted_extra,
                                      filename=self.dirs['risk'] / "risk_curve_stacked.png")
                                      
        # 8. Dashboards (Subplots)
        self._generate_dashboards(boot_eads_flood, boot_eads_invest, boot_eads_total,
                                  mean_flood, mean_invest, mean_total,
                                  sorted_probs, sorted_raw_flood, sorted_extra)


    def _generate_dashboards(self, boot_flood, boot_invest, boot_total, 
                             mean_flood, mean_invest, mean_total,
                             probs, flood_costs, extra_costs):
        """Generates unified dashboards (2x2 subplots)."""
        print("  Generating Summary Dashboards...")
        
        # --- Dashboard 1: Uncertainty Distributions ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("EAD Risk Uncertainty Dashboard", fontsize=16)
        
        # 1. Flood Dist
        self._plot_bootstrap_hist(boot_flood, mean_flood, 
                                  np.percentile(boot_flood, 2.5), np.percentile(boot_flood, 97.5),
                                  title_suffix="(Flood Damage)", color='#1f78b4', ax=axes[0,0])
        
        # 2. Invest Dist
        self._plot_bootstrap_hist(boot_invest, mean_invest, 
                                  np.percentile(boot_invest, 2.5), np.percentile(boot_invest, 97.5),
                                  title_suffix="(Infrastructure Repair)", color='#ff7f00', ax=axes[0,1])
                                  
        # 3. Total Dist
        self._plot_bootstrap_hist(boot_total, mean_total, 
                                  np.percentile(boot_total, 2.5), np.percentile(boot_total, 97.5),
                                  title_suffix="(Total Risk)", color='#33a02c', ax=axes[1,0])
                                  
        # 4. Combined
        self._plot_combined_bootstrap_hist(boot_flood, boot_invest, boot_total, filename=None, ax=axes[1,1])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_d1 = self.dirs['risk'] / "dashboard_ead_distributions.png"
        plt.savefig(out_d1, dpi=150)
        plt.close()
        print(f"  Generated: {out_d1.name}")

        # --- Dashboard 2: Risk Curves ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Risk Curves Dashboard (Expected Values)", fontsize=16)
        
        # 1. Flood Curve
        self._plot_risk_curve(probs, flood_costs, "Risk Curve: Flood", 
                              filename=None, color='#1f78b4', ax=axes[0,0])
                              
        # 2. Invest Curve
        self._plot_risk_curve(probs, extra_costs, "Risk Curve: Investment", 
                              filename=None, color='#ff7f00', ax=axes[0,1])
        
        # 3. Stacked Curve
        self._plot_stacked_risk_curve(probs, flood_costs, extra_costs, filename=None, ax=axes[1,0])
        
        # 4. Text Summary
        axes[1,1].axis('off')
        axes[1,1].text(0.1, 0.5, "EAD Summary (Probabilistic Mean):\n\n"
                                 f"Flood:      ${mean_flood:,.0f}\n"
                                 f"Investment: ${mean_invest:,.0f}\n"
                                 f"----------------\n"
                                 f"TOTAL:      ${mean_total:,.0f}", 
                       fontsize=14, family='monospace')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_d2 = self.dirs['risk'] / "dashboard_risk_curves.png"
        plt.savefig(out_d2, dpi=150)
        plt.close()
        print(f"  Generated: {out_d2.name}")

    def _plot_stacked_risk_curve(self, probs, flood_costs, extra_costs, filename=None, ax=None):
        """Plots a Stacked Area Risk Curve (Flood + Infrastructure)."""
        standalone = False
        if ax is None:
            standalone = True
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Stacked Area
        # X: probs
        # Y1: flood_costs
        # Y2: flood_costs + extra_costs
        total_costs = flood_costs + extra_costs
        
        # Calculate EAD components (Areas)
        ead_flood = abs(np.trapz(flood_costs, probs))
        ead_extra = abs(np.trapz(total_costs, probs)) - ead_flood
        ead_total = ead_flood + ead_extra

        # Plot Areas
        ax.fill_between(probs, 0, flood_costs, color='#4e79a7', alpha=0.6, 
                        label=f'Flood Damage (EAD=${ead_flood/1e6:.1f}M)')
        
        ax.fill_between(probs, flood_costs, total_costs, color='#f28e2b', alpha=0.6, 
                        label=f'Infrastructure Repair (EAD=${ead_extra/1e6:.1f}M)')
        
        # Plot Lines and Points
        ax.plot(probs, flood_costs, 'o-', color='#4e79a7', lw=1.5)
        ax.plot(probs, total_costs, 'o-', color='#f28e2b', lw=1.5)
        
        # Annotate ROI/Total
        for p, c_flood, c_total in zip(probs, flood_costs, total_costs):
            tr = 1/p
            ax.annotate(f"TR {tr:.0f}", (p, c_total), xytext=(0, 5), 
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

        title = f"Total Risk Curve (EAD)\nFlood Damage: ${ead_flood:,.0f} + Infrastructure Repair: ${ead_extra:,.0f}\nTOTAL: ${ead_total:,.0f}"
        ax.set_title(title, fontsize=10)
        
        ax.set_xlabel("Exceedance Probability (1/T)")
        ax.set_ylabel("Event Cost (USD)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y/1e6:.1f}M'))
        
        ax.legend(loc='upper right', fontsize=9)
        
        if standalone:
            plt.tight_layout()
            out_path = self.output_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Generated: {out_path.name}")

    def _plot_bootstrap_hist(self, boot_eads, mean, lower, upper, filename="ead_uncertainty.png", title_suffix="", title=None, color='#4285F4', ax=None):
        """Plots the distribution of the bootstrapped EAD."""
        standalone = False
        if ax is None:
            standalone = True
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check for constant data (zero variance)
        if np.std(boot_eads) < 1e-6:
             ax.axvline(mean, color=color, linewidth=4, label=f'Constant: ${mean:,.0f}')
             ax.set_title(f'Deterministic Distribution\n{title_suffix}', fontsize=10)
             # Set reasonable x limits
             ax.set_xlim(mean * 0.9, mean * 1.1)
             ax.set_yticks([])
             ax.set_ylabel("Density (Dirac)")
        else:
            ax.hist(boot_eads, bins=30, color=color, alpha=0.6, edgecolor='white', density=True)
            
            # KDE
            try:
                kde = stats.gaussian_kde(boot_eads)
                x_range = np.linspace(boot_eads.min(), boot_eads.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2, label='Density')
            except: pass
        
        # Lines
        ax.axvline(mean, color='k', linestyle='--', linewidth=2, label=f'Mean: ${mean:,.0f}')
        ax.axvline(lower, color='k', linestyle=':', label=f'95% CI Lower')
        ax.axvline(upper, color='k', linestyle=':', label=f'95% CI Upper')
        
        # Only set default title if not already set (i.e. if not constant)
        if not (np.std(boot_eads) < 1e-6):
            if title:
                ax.set_title(title, fontsize=10)
            else:
                ax.set_title(f'EAD Uncertainty Distribution\n{title_suffix}', fontsize=10)
        ax.set_xlabel('Expected Annual Damage (USD)')
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=9)
        
        # Format X axis currency
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        if standalone:
            plt.tight_layout()
            out_path = self.output_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Generated: {filename}")

    def _plot_combined_bootstrap_hist(self, boot_flood, boot_invest, boot_total, filename=None, ax=None):
        """Plots all 3 EAD distributions on one chart for comparison."""
        standalone = False
        if ax is None:
            standalone = True
            fig, ax = plt.subplots(figsize=(12, 7))
        
        # KDE Plotting helper
        def plot_kde(data, color, label):
            try:
                kde = stats.gaussian_kde(data)
                x = np.linspace(data.min(), data.max(), 200)
                ax.plot(x, kde(x), color=color, lw=2, label=f"{label} (Media: ${data.mean():,.0f})")
                ax.fill_between(x, 0, kde(x), color=color, alpha=0.2)
            except Exception as e:
                # Fallback for constant data (spike)
                ax.hist(data, bins=30, density=True, color=color, alpha=0.3, label=f"{label} (Media: ${data.mean():,.0f})")

        plot_kde(boot_flood, '#1f78b4', 'Flood Damage')
        plot_kde(boot_invest, '#ff7f00', 'Infrastructure Repair')
        plot_kde(boot_total, '#33a02c', 'Total Risk')
        
        ax.set_title("EAD Risk Distributions Comparison (Uncertainty)", fontsize=10)
        ax.set_xlabel("EAD (USD)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        if standalone:
            plt.tight_layout()
            out_path = self.output_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Generated: {filename}")

    def _plot_risk_curve(self, probs, costs, title, filename=None, color='#E15759', ax=None):
        """Plots the Probability vs Cost curve (EAD integration visualization)."""
        standalone = False
        if ax is None:
            standalone = True
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Line
        ax.plot(probs, costs, 'o-', color=color, linewidth=2, label='Event Cost')
        
        # Fill Area (EAD)
        # Note: trapz integration
        ead_val = abs(np.trapz(costs, probs))
        ax.fill_between(probs, costs, alpha=0.2, color=color, label=f'EAD = ${ead_val:,.0f}')
        
        # Annotate Points (Return Periods)
        for p, c in zip(probs, costs):
            tr = 1/p
            ax.annotate(f"TR {tr:.0f}", (p, c), xytext=(0, 5), 
                        textcoords='offset points', ha='center', fontsize=9)
        
        ax.set_title(f"{title}\nEstimated EAD: ${ead_val:,.0f}", fontsize=10)
        ax.set_xlabel("Exceedance Probability (1/T)")
        ax.set_ylabel("Event Cost (USD)")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y/1e6:.1f}M'))
        
        ax.legend()
        
        if standalone:
            plt.tight_layout()
            out_path = self.output_dir / filename
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  Generated: {out_path.name}")

    def _generate_side_by_side_ead_map(self, damage_matrix, probs, prop_meta, tr_list):
        """Generates a side-by-side map of Flood EAD vs Infrastructure EAD."""
        print("    [Map] Generating Side-by-Side EAD Map...")
        
        # Output Directory
        dir_ead = self.dirs['maps'] / "04_ead_maps"
        dir_ead.mkdir(parents=True, exist_ok=True)
        
        # 1. FLOOD EAD (Left)
        print("      -> Calculating Property EAD...")
        sort_ord = np.argsort(probs)
        sorted_probs = probs[sort_ord]
        
        # Calculate Property EADs
        # Use .loc to avoid Index Error (as fixed previously)
        prop_eads = damage_matrix.apply(lambda row: abs(np.trapz(row.values[sort_ord], sorted_probs)), axis=1)
        
        gdf_flood = prop_meta.copy()
        gdf_flood['ead'] = 0.0
        gdf_flood.loc[prop_eads.index, 'ead'] = prop_eads.values
        
        # Filter for visualization
        gdf_flood_plot = gdf_flood[gdf_flood['ead'] > 10] 
        
        # 2. INFRASTRUCTURE EAD (Right)
        print("      -> Calculating Pipe EAD...")
        pipe_dfs = []
        
        base_net_path = config.NETWORK_FILE
        full_pipes = None
        if base_net_path.exists():
            full_pipes = gpd.read_file(base_net_path)
            if 'Name' in full_pipes.columns:
                full_pipes = full_pipes.set_index('Name')

        for tr in tr_list:
            # Paths
            # Geometry: GPKG
            tr_code = f"TR_{int(tr):03d}"
            base_path = self.output_dir.parent / tr_code / "avoided_cost" / "deferred_investment"
            gpkg = base_path / f"{tr_code}_pipes_rehabilitation.gpkg"
            txt_report = base_path / f"{tr_code}_resumen_costos.txt"

            if gpkg.exists() and txt_report.exists():
                # 1. Get Cost Rate from TXT
                cost_per_meter = None
                try:
                    with open(txt_report, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Look for "Costo por metro lineal: $1,685.55 USD/m"
                        # Regex for robust extraction
                        import re
                        match = re.search(r"Costo por metro lineal:\s*\$([\d,]+\.?\d*)", content)
                        if match:
                            val_str = match.group(1).replace(',', '')
                            cost_per_meter = float(val_str)
                            print(f"        [TR {tr}] Parsed Rate: ${cost_per_meter:.2f}/m")
                        else:
                             # Fallback check for different formatting?
                             pass
                except Exception as e:
                    import sys
                    sys.exit(f"CRITICAL: Error reading TXT {txt_report.name}: {e}")

                if cost_per_meter is None:
                     sys.exit(f"CRITICAL: Could not find 'Costo por metro lineal' in {txt_report.name}")

                # 2. Read Geometry
                idf = gpd.read_file(gpkg)
                if idf.empty: continue
                
                # Identify Key (Tramo/Name) just for indexing stability
                geo_key = None
                for col in ['Tramo', 'Name', 'tramo', 'id']:
                    if col in idf.columns:
                        geo_key = col
                        break
                if not geo_key:
                     # If just geometry, generate an ID? No, strict mode requested.
                     # But actually, for EAD we just need geometry and cost. 
                     # Pivot needs an ID.
                     sys.exit(f"CRITICAL: Key column missing in {gpkg.name}")
                
                idf = idf.set_index(geo_key)

                # 3. Calculate Cost
                # We trust the geometry projection is in meters (SIRES-DMQ used in logs)
                idf['infrastructure_cost'] = idf.geometry.length * cost_per_meter
                
                # Store
                idf = idf.reset_index()
                idf['tr'] = float(tr)
                # Ensure Name
                if 'Name' not in idf.columns:
                    idf['Name'] = idf[geo_key]
                
                pipe_dfs.append(idf[['Name', 'infrastructure_cost', 'geometry', 'tr']])

            else:
                 print(f"        [Warn] Missing GPKG or TXT for TR {tr}: {gpkg.name}")
        
        gdf_infra_plot = gpd.GeoDataFrame()
        infra_total = 0.0
        
        if pipe_dfs:
            all_pipes_risk = pd.concat(pipe_dfs)
            # Pivot: Rows=Name, Cols=TR
            # Use 'infrastructure_cost' as value
            risk_matrix = all_pipes_risk.pivot_table(index='Name', columns='tr', values='infrastructure_cost', fill_value=0.0)
            
            # Now integrate per pipe
            def calc_pipe_ead(row):
                # Ensure we have columns for all TRs in sorted_probs
                row_sorted = row.sort_index() 
                costs = row_sorted.values
                trs = row_sorted.index.values
                
                curr_probs = 1.0 / trs
                
                p_sort_idx = np.argsort(curr_probs)
                p_sorted = curr_probs[p_sort_idx]
                c_sorted = costs[p_sort_idx]
                
                return abs(np.trapz(c_sorted, p_sorted))

            pipe_ead_series = risk_matrix.apply(calc_pipe_ead, axis=1)
            
            # Join with geometry
            # Use one of the read GPKGs or base net for geometry
            # We need the geometry of the pipes. We can take it from all_pipes_risk first entry per group
            unique_geoms = all_pipes_risk.groupby('Name')['geometry'].first()
            
            gdf_infra_plot = gpd.GeoDataFrame({'ead': pipe_ead_series, 'geometry': unique_geoms}, geometry='geometry')
            
            # Filter > 10
            gdf_infra_plot = gdf_infra_plot[gdf_infra_plot['ead'] > 10]
            
            if not gdf_infra_plot.empty:
                infra_total = gdf_infra_plot['ead'].sum()
                print(f"      -> Total Pipe EAD: ${infra_total:,.0f} (from {len(gdf_infra_plot)} pipes)")

        # 3. PLOT
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        flood_total = gdf_flood['ead'].sum() # Full sum
        
        fig.suptitle(f"Spatial Distribution of Risk (Expected Annual Damage)\nTotal System Risk: ${(flood_total + infra_total):,.0f}", fontsize=20)
        
        # --- LEFT: FLOOD EAD (Properties) ---
        ax_flood = axes[0]
        # Background: All Properties
        prop_meta.plot(ax=ax_flood, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)
        
        if not gdf_flood_plot.empty:
            gdf_flood_plot.plot(column='ead', ax=ax_flood, cmap='RdYlGn_r', 
                           legend=True, zorder=2,
                           legend_kwds={'label': 'Expected Annual Damage (USD)', 'shrink': 0.5, 'format': '${x:,.0f}'})
        else:
            print("      [Warn] Flood EAD plot is empty!")

        ax_flood.set_title(f"Flood Risk (Property EAD)\nTotal: ${flood_total:,.0f}", fontsize=14)
        ax_flood.set_axis_off()

        # --- RIGHT: INFRA EAD (Pipes) ---
        ax_infra = axes[1]
        
        # Background: Use SAME prop_meta for consistent extent, then overlay pipes
        prop_meta.plot(ax=ax_infra, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.1, zorder=1)
        
        # Plot ALL collected pipes as grey background lines
        if pipe_dfs:
            all_pipes_bg = pd.concat(pipe_dfs)
            # Ensure CRS matches prop_meta
            if hasattr(prop_meta, 'crs') and prop_meta.crs is not None:
                all_pipes_bg = gpd.GeoDataFrame(all_pipes_bg, geometry='geometry', crs=prop_meta.crs)
            all_pipes_bg.plot(ax=ax_infra, color='#c0c0c0', linewidth=1.0, zorder=2)
            print(f"      -> Plotted {len(all_pipes_bg)} background pipes")
        
        # Plot EAD pipes in orange/red
        if not gdf_infra_plot.empty:
            # Ensure CRS matches
            if hasattr(prop_meta, 'crs') and prop_meta.crs is not None:
                gdf_infra_plot = gdf_infra_plot.set_crs(prop_meta.crs, allow_override=True)
            
            gdf_infra_plot.plot(column='ead', ax=ax_infra, cmap='RdYlGn_r', linewidth=3.0,
                          legend=True, zorder=4,
                          legend_kwds={'label': 'Infra Repair EAD (USD)', 'shrink': 0.5, 'format': '${x:,.0f}'})
            print(f"      -> Plotted {len(gdf_infra_plot)} EAD pipes with total ${infra_total:,.0f}")
        else:
            print("      [Warn] Infrastructure EAD plot is empty!")

        ax_infra.set_title(f"Infrastructure Risk (Pipe EAD)\nTotal: ${infra_total:,.0f}", fontsize=14)
        ax_infra.set_axis_off()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = dir_ead / "spatial_ead_side_by_side.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"    -> Generated: {out_path.name}")

    def _generate_failure_probability_maps(self, damage_matrix, probs, prop_meta, tr_list):
        """
        Generates spatial maps showing the probability of failure for each property and pipe.
        P(failure) = Sum of probabilities (1/TR) for each TR where the entity is affected.
        """
        print("    [Map] Generating Spatial Failure Probability Maps...")
        
        # Output Directory
        dir_spatial = self.dirs['maps'].parent / "04_spatial_analysis"
        dir_spatial.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # 1. PROPERTY FAILURE PROBABILITY
        # =====================================================================
        print("      -> Calculating Property Failure Probability...")
        
        # For each property, calculate P(failure) = sum of 1/TR for TRs where damage > 0
        # damage_matrix: rows=properties, cols=TRs
        
        prop_failure_prob = pd.Series(0.0, index=damage_matrix.index)
        
        for tr_col in damage_matrix.columns:
            prob_tr = 1.0 / float(tr_col)  # Annual probability of this TR
            # Properties with damage > 0 at this TR
            affected_mask = damage_matrix[tr_col] > 0
            prop_failure_prob[affected_mask] += prob_tr
        
        # Cap at 1.0 (though mathematically shouldn't exceed for typical TR ranges)
        prop_failure_prob = prop_failure_prob.clip(upper=1.0)
        
        # Create GeoDataFrame for plotting
        gdf_prop_prob = prop_meta.copy()
        gdf_prop_prob['failure_prob'] = 0.0
        gdf_prop_prob.loc[prop_failure_prob.index, 'failure_prob'] = prop_failure_prob.values
        
        # Filter for visualization (only those with prob > 0)
        gdf_prop_plot = gdf_prop_prob[gdf_prop_prob['failure_prob'] > 0.001]
        
        n_props_at_risk = len(gdf_prop_plot)
        max_prob = gdf_prop_plot['failure_prob'].max() if not gdf_prop_plot.empty else 0
        print(f"         {n_props_at_risk} properties at risk (max P={max_prob:.1%})")
        
        # =====================================================================
        # 2. PIPE FAILURE PROBABILITY  
        # =====================================================================
        print("      -> Calculating Pipe Failure Probability...")
        
        pipe_failure_data = []
        
        for tr in tr_list:
            tr_code = f"TR_{int(tr):03d}"
            base_path = self.output_dir.parent / tr_code / "avoided_cost" / "deferred_investment"
            gpkg = base_path / f"{tr_code}_pipes_rehabilitation.gpkg"
            
            if gpkg.exists():
                try:
                    idf = gpd.read_file(gpkg)
                    if idf.empty: 
                        continue
                    
                    # Identify key column
                    geo_key = None
                    for col in ['Tramo', 'Name', 'tramo', 'id']:
                        if col in idf.columns:
                            geo_key = col
                            break
                    
                    if geo_key:
                        idf['pipe_id'] = idf[geo_key]
                    else:
                        idf['pipe_id'] = idf.index.astype(str)
                    
                    idf['tr'] = float(tr)
                    idf['prob'] = 1.0 / float(tr)
                    
                    pipe_failure_data.append(idf[['pipe_id', 'geometry', 'tr', 'prob']])
                except Exception as e:
                    print(f"         [Warn] TR {tr} pipe read failed: {e}")
        
        gdf_pipe_plot = gpd.GeoDataFrame()
        n_pipes_at_risk = 0
        max_pipe_prob = 0
        
        if pipe_failure_data:
            all_pipes = pd.concat(pipe_failure_data)
            
            # Sum probabilities per pipe
            pipe_probs = all_pipes.groupby('pipe_id')['prob'].sum().clip(upper=1.0)
            
            # Get unique geometries
            unique_geoms = all_pipes.groupby('pipe_id')['geometry'].first()
            
            gdf_pipe_plot = gpd.GeoDataFrame({
                'pipe_id': pipe_probs.index,
                'failure_prob': pipe_probs.values,
                'geometry': unique_geoms.values
            }, geometry='geometry')
            
            # Set CRS to match prop_meta
            if hasattr(prop_meta, 'crs') and prop_meta.crs is not None:
                gdf_pipe_plot = gdf_pipe_plot.set_crs(prop_meta.crs, allow_override=True)
            
            n_pipes_at_risk = len(gdf_pipe_plot)
            max_pipe_prob = gdf_pipe_plot['failure_prob'].max()
            print(f"         {n_pipes_at_risk} pipes at risk (max P={max_pipe_prob:.1%})")
        
        # =====================================================================
        # 3. SIDE-BY-SIDE PLOT
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        
        fig.suptitle("Spatial Failure Probability (Annual)\nProbability of Experiencing Damage/Failure", fontsize=20)
        
        # --- LEFT: PROPERTY FAILURE PROBABILITY ---
        ax_prop = axes[0]
        
        # Background
        prop_meta.plot(ax=ax_prop, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)
        
        if not gdf_prop_plot.empty:
            gdf_prop_plot.plot(column='failure_prob', ax=ax_prop, cmap='RdYlGn_r', 
                              vmin=0, vmax=1.0,
                              legend=True, zorder=2,
                              legend_kwds={'label': 'P(Flood Damage)', 'shrink': 0.5, 'format': '{x:.0%}'})
        
        ax_prop.set_title(f"Property Flood Risk\n{n_props_at_risk:,} properties at risk (max: {max_prob:.0%})", fontsize=14)
        ax_prop.set_axis_off()
        
        # --- RIGHT: PIPE FAILURE PROBABILITY ---
        ax_pipe = axes[1]
        
        # Background
        prop_meta.plot(ax=ax_pipe, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)
        
        if not gdf_pipe_plot.empty:
            gdf_pipe_plot.plot(column='failure_prob', ax=ax_pipe, cmap='RdYlGn_r',
                              vmin=0, vmax=1.0,
                              linewidth=3.0, legend=True, zorder=3,
                              legend_kwds={'label': 'P(Pipe Failure)', 'shrink': 0.5, 'format': '{x:.0%}'})
        
        ax_pipe.set_title(f"Infrastructure Failure Risk\n{n_pipes_at_risk:,} pipes at risk (max: {max_pipe_prob:.0%})", fontsize=14)
        ax_pipe.set_axis_off()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = dir_spatial / "failure_probability_map.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"    -> Generated: {out_path.name}")
        
        # =====================================================================
        # 4. SAVE GEOPACKAGE WITH RESULTS
        # =====================================================================
        # Property probabilities
        gdf_prop_prob.to_file(dir_spatial / "property_failure_probability.gpkg", driver="GPKG")
        print(f"    -> Saved: property_failure_probability.gpkg")
        
        if not gdf_pipe_plot.empty:
            gdf_pipe_plot.to_file(dir_spatial / "pipe_failure_probability.gpkg", driver="GPKG")
            print(f"    -> Saved: pipe_failure_probability.gpkg")
        
        return gdf_prop_prob, gdf_pipe_plot

    def _generate_scenario_maps(self, damage_gpkgs, prop_meta):
        """Generates separate damage maps for Flood and Infrastructure."""
        print("  Generating Separate Scenario Maps (Flood vs Infrastructure)...")
        from matplotlib.ticker import FuncFormatter
        
        # 1. Prepare Subdirectories
        dir_flood = self.dirs['maps'] / "01_flood_damage"
        dir_infra = self.dirs['maps'] / "02_deferred_investment"
        dir_flood.mkdir(parents=True, exist_ok=True)
        dir_infra.mkdir(parents=True, exist_ok=True)
        
        def currency_fmt(x, pos):
            if x >= 1e6: return f'${x/1e6:.1f}M'
            elif x >= 1e3: return f'${x/1e3:.0f}k'
            return f'${x:.0f}'

        # Pre-load network if available
        net_gdf = None
        if hasattr(config, 'NETWORK_FILE') and config.NETWORK_FILE.exists():
            try:
                 net_gdf = gpd.read_file(config.NETWORK_FILE)
                 if net_gdf.crs != prop_meta.crs:
                     net_gdf = net_gdf.to_crs(prop_meta.crs)
            except: pass

        for tr, gpkg_path in damage_gpkgs.items():
            try:
                gpkg_path = Path(gpkg_path) # Ensure it is a Path object
                
                # --- A. FLOOD DAMAGE MAP ---
                gdf_flood = gpd.read_file(gpkg_path)
                gdf_flood = gdf_flood[gdf_flood['damage_usd'] > 0]
                flood_loss = gdf_flood['damage_usd'].sum()
                
                if not gdf_flood.empty:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Background: Properties (predios)
                    prop_meta.plot(ax=ax, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)
                    
                    # Foreground: Flood Damage
                    gdf_flood.plot(column='damage_usd', ax=ax, cmap='RdYlGn_r', 
                                   legend=True, zorder=2,
                                   legend_kwds={'label': 'Flood Damage (USD)', 'format': FuncFormatter(currency_fmt), 'shrink': 0.6})
                    
                    ax.set_title(f"Flood Scenario: TR {tr} years ({1/tr:.1%} Prob.)\nTotal Damage: ${flood_loss:,.0f}", fontsize=12)
                    ax.set_axis_off()
                    
                    plt.tight_layout()
                    out_name = dir_flood / f"map_flood_TR_{tr:03d}.png"
                    plt.savefig(out_name, dpi=150)
                    plt.close()
                    print(f"    -> Flood: {out_name.name}")


                # --- B. INFRASTRUCTURE MAP ---
                # Search for Infra GPKG
                infra_gdf = None
                infra_loss = 0
                
                parent_dir = gpkg_path.parent.parent # avoided_cost or TR_XX
                
                # Check deferred_investment locations
                deferred_dir = None
                possible_paths = [
                    parent_dir / "deferred_investment",
                    parent_dir / "avoided_cost" / "deferred_investment",
                    parent_dir.parent / "deferred_investment"
                ]
                
                for p in possible_paths:
                    if p.exists():
                        deferred_dir = p
                        break

                if deferred_dir and deferred_dir.exists():
                    try:
                        # Match *TR_XXX* and *pipes* or *rehabilitation*
                        # Prioritize 'pipes_rehabilitation'
                        candidates = list(deferred_dir.glob(f"*{tr:03d}*pipes*.gpkg"))
                        if not candidates:
                            candidates = list(deferred_dir.glob(f"*{tr:03d}*rehabilitation*.gpkg"))
                        
                        if candidates:
                            infra_gdf = gpd.read_file(candidates[0])
                            # Sum cost
                            for col in ['cost_usd', 'rehabilitation_cost_usd', 'investment_cost_usd']:
                                if col in infra_gdf.columns:
                                    infra_loss = infra_gdf[col].sum()
                                    break
                    except Exception as e:
                        print(f"    [Warn] Failed reading infra GPKG: {e}")
                
                if infra_gdf is not None and not infra_gdf.empty:
                    # Try to get cost from TXT if columns are missing
                    if infra_loss == 0:
                         try:
                             txt_candidates = list(deferred_dir.glob(f"*{tr:03d}*resumen_costos.txt"))
                             if txt_candidates:
                                 with open(txt_candidates[0], 'r', encoding='utf-8', errors='ignore') as f:
                                     content = f.read()
                                     import re
                                     match = re.search(r"Total:\s*\$([\d,]+(?:\.\d+)?)", content)
                                     if match:
                                         infra_loss = float(match.group(1).replace(',', ''))
                         except: pass

                    # ---------------------------
                    # 1. INFRASTRUCTURE MAP (Solo)
                    # ---------------------------
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Background: Properties (predios)
                    prop_meta.plot(ax=ax, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)

                        
                    # Foreground: Pipes
                    if 'Capacity' in infra_gdf.columns:
                        infra_gdf.plot(column='Capacity', ax=ax, cmap='RdYlGn_r', linewidth=3,
                                     legend=True, zorder=3,
                                     legend_kwds={'label': 'Pipe Capacity (m³/s)', 'shrink': 0.6})
                        lbl = "Capacity"
                    else:
                        infra_gdf.plot(ax=ax, color='#ff7f00', linewidth=2.5, zorder=3, label='Rehabilitation Required')
                        lbl = "Rehab"

                    ax.set_title(f"Infrastructure Needs: TR {tr} years ({1/tr:.1%} Prob.)\nTotal Rehab Cost: ${infra_loss:,.0f}", fontsize=12)
                    ax.set_axis_off()
                    
                    if lbl == "Rehab":
                        import matplotlib.patches as mpatches
                        patch = mpatches.Patch(color='#ff7f00', label='Rehabilitation Target')
                        ax.legend(handles=[patch], loc='upper right')
                        
                    plt.tight_layout()
                    out_name = dir_infra / f"map_infra_TR_{tr:03d}.png"
                    plt.savefig(out_name, dpi=150)
                    plt.close()
                    print(f"    -> Infra: {out_name.name}")

                    # ---------------------------
                    # 2. COMBINED MAP (Side-by-Side)
                    # ---------------------------
                    dir_combined = self.dirs['maps'] / "03_total_damage"
                    dir_combined.mkdir(parents=True, exist_ok=True)
                    
                    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
                    fig.suptitle(f"Total Damage Scenario: TR {tr} years ({1/tr:.1%} Prob.)\nCombined Cost: ${(flood_loss + infra_loss):,.0f}", fontsize=16)
                    
                    # --- Left: Flood ---
                    ax_flood = axes[0]
                    # Background: Properties (predios)
                    prop_meta.plot(ax=ax_flood, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)
                    
                    if not gdf_flood.empty:
                        gdf_flood.plot(column='damage_usd', ax=ax_flood, cmap='RdYlGn_r', 
                                       legend=True, zorder=2,
                                       legend_kwds={'label': 'Flood Damage (USD)', 'format': FuncFormatter(currency_fmt), 'shrink': 0.4})
                    
                    ax_flood.set_title(f"Flood Damage\nEst. Loss: ${flood_loss:,.0f}", fontsize=14)
                    ax_flood.set_axis_off()
                    
                    # --- Right: Infrastructure ---
                    ax_infra = axes[1]
                    # Background: Properties (predios)
                    prop_meta.plot(ax=ax_infra, color='#e8e8e8', edgecolor='#d0d0d0', linewidth=0.15, zorder=1)

                    if 'Capacity' in infra_gdf.columns:
                        infra_gdf.plot(column='Capacity', ax=ax_infra, cmap='RdYlGn_r', linewidth=3,
                                     legend=True, zorder=3,
                                     legend_kwds={'label': 'Pipe Capacity (m³/s)', 'shrink': 0.4})
                    else:
                        infra_gdf.plot(ax=ax_infra, color='#ff7f00', linewidth=2.5, zorder=3) # Simple

                    ax_infra.set_title(f"Deferred Investment (Infrastructure)\nEst. Cost: ${infra_loss:,.0f}", fontsize=14)
                    ax_infra.set_axis_off()
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_combined = dir_combined / f"map_total_TR_{tr:03d}.png"
                    plt.savefig(out_combined, dpi=150)
                    plt.close()
                    print(f"    -> Combined: {out_combined.name}")
                
            except Exception as e:
                print(f"    [Error] TR {tr}: {e}")

    def _calculate_real_fragility(self, damage_df):
        """
        Calculates normalized fragility curves using system baselines.
        Returns DF with columns: [TR, Flood_Ratio, Infra_Ratio, Total_Ratio]
        """
        records = []
        
        # Baselines (avoid div by zero)
        base_prop = self.total_property_value if self.total_property_value > 0 else 1e9
        base_net_len = self.total_network_length if self.total_network_length > 0 else 1e5
        
        # Estimate total network value approx if not calculated
        # Assuming avg rehab cost $300/m for whole network as rough proxy if needed,
        # or simplified: Total Value = (Flood + Infra) / (Prop + NetValue)
        # Estimate total network value approx if not calculated
        # Use calculated value if available, else proxy
        print(f"    [Fragility DEBUG] self.total_network_value: {self.total_network_value}")
        
        base_net_val = self.total_network_value if self.total_network_value > 0 else (base_net_len * 250.0)
        print(f"    [Fragility DEBUG] Using base_net_val: {base_net_val}") 
        base_total = base_prop + base_net_val

        for idx, row in damage_df.iterrows():
            tr = row['tr']
            flood_usd = row['flood_damage']
            
            # Infra: We need Length of Critical Pipes (Cap > 0.9)
            # This info needs to be passed or extracted. 
            # Currently damage_df has 'infrastructure_cost' (USD).
            # We need to look up the specific GPKG to get the length of pipes > 0.9?
            # Or simplified: Infra_Ratio = Cost / Base_Net_Val (Financial Proxy)
            # The User specifically asked for "count/length of values with capacity > 0.9"
            
            # Let's try to find the infra gpkg for this TR to extract stats
            infra_ratio = 0.0
            crit_len = 0.0
            try:
                # Construct explicit path (TR folders are in the PARENT of risk_estimation dir)
                gpkg_path = self.output_dir.parent / f"TR_{int(tr):03d}" / "avoided_cost" / "deferred_investment" / f"TR_{int(tr):03d}_pipes_rehabilitation.gpkg"
                
                print(f"    [Fragility] Checking: {gpkg_path}")
                if gpkg_path.exists():
                    # import geopandas as gpd  # Already imported or handled?
                    gdf = gpd.read_file(gpkg_path)
                    if 'Capacity' in gdf.columns:
                        crit_pipes = gdf[gdf['Capacity'] > 0.9]
                        crit_len = crit_pipes.geometry.length.sum()
                        infra_ratio = crit_len / base_net_len
                        print(f"      -> Found {len(crit_pipes)} crit pipes. Ratio: {infra_ratio:.2%}")
                    else:
                        print(f"    [Error] 'Capacity' column missing in {gpkg_path.name}")
                        infra_ratio = 0.0 # NO FALLBACK
                else:
                    print(f"    [Error] GPKG NOT FOUND: {gpkg_path}")
                    infra_ratio = 0.0

            except Exception as e:
                print(f"    [Error] Fragility Calc Failed for TR {tr}: {e}")
                infra_ratio = 0.0

            # --- Physical Fragility (Count Ratios) ---
            # 1. Flood Count Ratio (Affected Props / Total Props)
            flood_count_ratio = 0.0
            try:
                flood_gpkg = self.output_dir.parent / f"TR_{int(tr):03d}" / "avoided_cost" / "flood_damage" / f"TR_{int(tr):03d}_flood_damage_results.gpkg"
                if flood_gpkg.exists():
                    f_gdf = gpd.read_file(flood_gpkg)
                    # Count where damage > 0 or depth > 0
                    if 'damage_usd' in f_gdf.columns:
                        affected_count = len(f_gdf[f_gdf['damage_usd'] > 0])
                        base_count = self.total_property_count if self.total_property_count > 0 else 1
                        flood_count_ratio = affected_count / base_count
            except Exception as ex_f:
                print(f"    [Warn] Flood Count Ratio failed: {ex_f}")

            # 2. Infra Count Ratio (Critical Pipes / Total Pipes)
            # We already found crit_pipes above (if available)
            infra_count_ratio = 0.0
            try:
                if 'crit_pipes' in locals() and isinstance(crit_pipes, (pd.DataFrame, gpd.GeoDataFrame)):
                     crit_count = len(crit_pipes)
                     base_pipe_count = self.total_pipe_count if self.total_pipe_count > 0 else 1
                     infra_count_ratio = crit_count / base_pipe_count
            except Exception as ex_i:
                 pass

            # Flood Ratio (Damage / Total Construction Value of Exposed)
            # User Request: "Flood Damage divided by Total Construction Value of Exposed Properties"
            base_prop_const = self.total_construction_value if self.total_construction_value > 0 else 1e9
            flood_ratio = flood_usd / base_prop_const
            
            # Total Ratio (Financial)
            # This remains Total USD / Total Assets (Construction + Network) or (Prop + Network)?
            # Let's keep it consistent: Total Damage / (Total Const Value + Total Network Value)
            total_usd = flood_usd + row['infrastructure_cost']
            total_ratio = total_usd / (base_prop_const + base_net_val)

            # Infra Money Ratio (New Request)
            infra_money_fragility = row['infrastructure_cost'] / base_net_val
            
            records.append({
                'tr': tr,
                'probability': 1/tr,
                'flood_fragility': min(flood_ratio, 1.0),
                'flood_count_fragility': min(flood_count_ratio, 1.0),
                'infra_fragility': min(infra_ratio, 1.0),
                'infra_count_fragility': min(infra_count_ratio, 1.0),
                'infra_money_fragility': min(infra_money_fragility, 1.0),
                'total_fragility': min(total_ratio, 1.0)
            })
            
        return pd.DataFrame(records).sort_values('tr')

    def calculate_fragility_curves(self, metrics_df, uncertainty_results=None):
        """
        Generates probability-damage curves (Fragility Curves).
        NOW uses Real Normalized Ratios.
        """
        print("    [Risk] Calculating Real Fragility Curves...")
        
        # Determine column names (support both old and new naming conventions)
        flood_col = 'flood_damage_usd' if 'flood_damage_usd' in metrics_df.columns else 'flood_damage'
        infra_col = 'investment_cost_usd' if 'investment_cost_usd' in metrics_df.columns else 'infrastructure_cost'
        
        # Check if columns exist
        available_cols = [c for c in [flood_col, infra_col] if c in metrics_df.columns]
        if not available_cols:
            print(f"    [Warning] Required columns not found. Available: {list(metrics_df.columns)}")
            return None
        
        # Use mean values for the main curve
        mean_df = metrics_df.groupby('tr')[available_cols].mean().reset_index()
        
        # Rename to standard names for downstream processing
        mean_df = mean_df.rename(columns={
            'flood_damage_usd': 'flood_damage',
            'investment_cost_usd': 'infrastructure_cost'
        })
        
        curve_df = self._calculate_real_fragility(mean_df)
        
        # Plot Economic
        self._plot_economic_fragility(curve_df)
        # Plot Physical
        self._plot_physical_fragility(curve_df)
        
        return curve_df

    def _plot_economic_fragility(self, curve_df):
        """Plots the Economic fragility curves ($/$)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Use categorical X axis (equidistant)
        trs = curve_df['tr'].values
        x_indices = np.arange(len(trs))
        x_labels = [f"TR {int(tr)}" for tr in trs]
        
        # --- LEFT: Flood Fragility (Economic) ---
        ax1.plot(x_indices, curve_df['flood_fragility'], 'o-', label='Economic Loss Ratio ($/$)\n(Damage / Total Const. Value)', color='blue', linewidth=2)
        
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.set_xlabel('Return Period (Years)')
        ax1.set_ylabel('Economic Damage Ratio ($/$)')
        ax1.set_title(f'Flood Vulnerability\n(Value of Damaged Properties / Total Exposed Value)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        # ax1.set_ylim(0, 1.1) # Auto-scale per user request 
        
        # Annotate Economic
        for x, y in zip(x_indices, curve_df['flood_fragility']):
            ax1.annotate(f"{y:.1%}", (x, y), xytext=(0, 5), 
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold', color='blue')


        # --- RIGHT: Infrastructure Economic Fragility ---
        ax2.plot(x_indices, curve_df['infra_money_fragility'], '^-', label='Economic Damage Ratio ($/$)\n(Repair Cost / Total Network Value)', color='red', linewidth=2)
        
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.set_xlabel('Return Period (Years)')
        ax2.set_ylabel('Economic Damage Ratio ($/$)')
        ax2.set_title(f'Infrastructure Vulnerability\n(Repair Cost / Total Assets Value)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        # ax2.set_ylim(0, 1.1) # Auto-scale per user request

        # Annotate
        for x, y in zip(x_indices, curve_df['infra_money_fragility']):
            ax2.annotate(f"{y:.1%}", (x, y), xytext=(0, 8), 
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
            
        fig.suptitle('System Economic Fragility (Risk = Probability x Consequence [$])', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = self.dirs['fragility'] / "fragility_curve_economic.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"    [Risk] Generated: {out_path.name}")

    def _plot_physical_fragility(self, curve_df):
        """Plots the Physical fragility curves (Count/Count and Length/Length)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        trs = curve_df['tr'].values
        x_indices = np.arange(len(trs))
        x_labels = [f"TR {int(tr)}" for tr in trs]
        
        # --- LEFT: Flood Physical (Count) ---
        ax1.plot(x_indices, curve_df['flood_count_fragility'], 's--', label='Physical Affected Ratio (#/#)\n(Affected Properties / Total Count)', color='cyan', linewidth=2)
        
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_labels, rotation=45)
        ax1.set_xlabel('Return Period (Years)')
        ax1.set_ylabel('Physical Ratio (#/#)')
        ax1.set_title(f'Flood Physical Vulnerability\n(Affected Count / Total Count)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        # ax1.set_ylim(0, 1.1) # Auto-scale
        
        for x, y in zip(x_indices, curve_df['flood_count_fragility']):
            ax1.annotate(f"{y:.1%}", (x, y), xytext=(0, 5), 
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

        # --- RIGHT: Infrastructure Physical (Length/Length) ---
        ax2.plot(x_indices, curve_df['infra_fragility'], 's-', label='Hydraulic Failure Ratio (L/L)\n(Critical Length / Total Length)', color='orange', linewidth=2)
        
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(x_labels, rotation=45)
        ax2.set_xlabel('Return Period (Years)')
        ax2.set_ylabel('Physical Ratio (L/L)')
        ax2.set_title(f'Infrastructure Physical Vulnerability\n(Critical Length / Total Length)', fontsize=12)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        # ax2.set_ylim(0, 1.1) # Auto-scale

        for x, y in zip(x_indices, curve_df['infra_fragility']):
            ax2.annotate(f"{y:.1%}", (x, y), xytext=(0, 8), 
                        textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
            
        fig.suptitle('System Physical Fragility (Performance Failure)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = self.dirs['fragility'] / "fragility_curve_physical.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"    [Risk] Generated: {out_path.name}")

    def _analyze_sectors_probabilistic(self, damage_matrix, prop_meta, probs, extra_costs_vector=None, filename="sector_risk.png"):
        """Break down risk by sector using probabilistic EAD, including infrastructure costs."""
        # NOTE: I need to actually change the function signature and save usage.
        
        print(f"  Analyzing Sector Risk -> {filename}...")
        
        # Merge EADs back
        sort_ord = np.argsort(probs)
        sorted_probs = probs[sort_ord]
        
        # 1. Spatial/Property EAD
        prop_eads = damage_matrix.apply(lambda row: abs(np.trapz(row.values[sort_ord], sorted_probs)), axis=1)
        
        df_risk = pd.DataFrame({
            'ead': prop_eads,
            'sector': prop_meta['_sector']
        })
        
        sector_risk = df_risk.groupby('sector')['ead'].sum()
        
        # 2. Add Extra Costs EAD (Infrastructure/Pipes)
        if extra_costs_vector is not None and np.sum(extra_costs_vector) > 0:
            # Integrate the scalar cost vector same as properties
            sorted_costs = extra_costs_vector[sort_ord]
            infra_ead = abs(np.trapz(sorted_costs, sorted_probs))
            
            # Add to series
            sector_risk['Infrastructure Repair'] = infra_ead
            print(f"    [Risk Audit] Infra EAD: ${infra_ead:,.0f} (Cost Vector Range: ${min(sorted_costs):,.0f} - ${max(sorted_costs):,.0f})")
            
        sector_risk = sector_risk.sort_values(ascending=False)
        print(f"    [Risk Audit] Sector Risk Breakdown:\n{sector_risk.to_string()}")
        
        # Rename sectors for display if needed
        # Mapping: sector -> Display Name
        sector_map = {
            'residential': 'Residential',
            'commercial': 'Commercial',
            'infrastructure': 'Public Infrastructure',
            'agriculture': 'Agriculture',
            'industrial': 'Industrial'
        }
        sector_risk.index = [sector_map.get(idx, idx) for idx in sector_risk.index]

        # Plot Bar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        sector_risk.plot(kind='bar', color=colors[:len(sector_risk)], ax=ax)
        
        ax.set_title("Total Risk Contribution (EAD) by Sector")
        ax.set_ylabel("EAD (USD)")
        ax.set_xlabel("Sector")
        
        # Labels
        total = sector_risk.sum()
        for i, v in enumerate(sector_risk):
            pct = v / total * 100
            ax.text(i, v, f"${v:,.0f}\n({pct:.1f}%)", ha='center', va='bottom')
            
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Generated: {filename}")

    # =========================================================================
    # RISK PROFILE ANALYSIS (Risk Appetite / Aversion)
    # =========================================================================
    
    def generate_risk_profile_report(self, 
                                      damage_df: pd.DataFrame,
                                      baseline_ead: float = None,
                                      project_ead: float = None,
                                      tank_investment: float = 0.0,
                                      design_tr: int = 25,
                                      project_life_years: int = 30,
                                      discount_rate: float = 0.08):
        """
        Generates a comprehensive Risk Profile Analysis report.
        
        Helps decision-makers understand:
        1. What TR they're designing for
        2. Probability of failure for different risk appetites
        3. Trade-off between investment and residual risk
        
        Args:
            damage_df: DataFrame with columns ['tr', 'flood_damage_usd', 'investment_cost_usd', 'total_impact_usd']
            baseline_ead: EAD without project (optional, calculated if not provided)
            project_ead: EAD with tanks (optional, calculated if not provided)
            tank_investment: Total investment in tanks ($)
            design_tr: Design return period (years)
            project_life_years: Project lifetime for risk calculations
            discount_rate: Social discount rate for NPV
            
        Returns:
            dict with risk metrics and generated file paths
        """
        print("\n" + "="*70)
        print("    RISK PROFILE ANALYSIS")
        print("="*70)
        
        # Create output directory
        risk_dir = self.output_dir / "risk_profile"
        risk_dir.mkdir(parents=True, exist_ok=True)
        
        # Define risk profiles
        RISK_PROFILES = {
            'Agresivo': {'tr': 10, 'color': '#e74c3c', 'emoji': '🔴'},
            'Moderado': {'tr': 25, 'color': '#f39c12', 'emoji': '🟡'},
            'Conservador': {'tr': 50, 'color': '#27ae60', 'emoji': '🟢'},
            'Muy Conservador': {'tr': 100, 'color': '#3498db', 'emoji': '🔵'}
        }
        
        # Extract TRs and damages from DataFrame
        trs = sorted(damage_df['tr'].unique())
        probs = [1.0 / tr for tr in trs]
        
        # Get damage columns
        flood_damages = damage_df.set_index('tr')['flood_damage_usd'].to_dict()
        if 'investment_cost_usd' in damage_df.columns:
            infra_damages = damage_df.set_index('tr')['investment_cost_usd'].to_dict()
        else:
            infra_damages = {tr: 0 for tr in trs}
        
        # Calculate EAD if not provided
        if baseline_ead is None or project_ead is None:
            # Approximate EAD from damage curve using trapezoidal integration
            total_damages = [flood_damages.get(tr, 0) + infra_damages.get(tr, 0) for tr in trs]
            ead_approx = 0
            for i in range(len(trs) - 1):
                dp = probs[i] - probs[i+1]
                avg_damage = (total_damages[i] + total_damages[i+1]) / 2
                ead_approx += dp * avg_damage
            
            if baseline_ead is None:
                baseline_ead = ead_approx
            if project_ead is None:
                project_ead = ead_approx  # Assumed same if not specified
        
        avoided_ead = baseline_ead - project_ead if baseline_ead and project_ead else 0
        
        # Calculate metrics for each profile
        profile_results = {}
        
        for profile_name, profile_data in RISK_PROFILES.items():
            tr = profile_data['tr']
            prob_annual = 1.0 / tr
            
            # Probability of at least 1 failure in project lifetime
            # P(at least 1) = 1 - P(none) = 1 - (1 - 1/TR)^N
            prob_at_least_one = 1 - (1 - prob_annual) ** project_life_years
            
            # Get damage at this TR (interpolate if needed)
            if tr in flood_damages:
                damage_at_tr = flood_damages[tr] + infra_damages.get(tr, 0)
            else:
                # Linear interpolation
                lower_tr = max([t for t in trs if t <= tr], default=trs[0])
                upper_tr = min([t for t in trs if t >= tr], default=trs[-1])
                if lower_tr == upper_tr:
                    damage_at_tr = flood_damages.get(lower_tr, 0) + infra_damages.get(lower_tr, 0)
                else:
                    ratio = (tr - lower_tr) / (upper_tr - lower_tr)
                    d_lower = flood_damages.get(lower_tr, 0) + infra_damages.get(lower_tr, 0)
                    d_upper = flood_damages.get(upper_tr, 0) + infra_damages.get(upper_tr, 0)
                    damage_at_tr = d_lower + ratio * (d_upper - d_lower)
            
            # Residual risk: EAD from events > design TR
            residual_ead = 0
            for i, t in enumerate(trs):
                if t > tr and i > 0:
                    dp = probs[i-1] - probs[i]
                    damage = flood_damages.get(t, 0) + infra_damages.get(t, 0)
                    residual_ead += dp * damage
            
            # Economic metrics
            if tank_investment > 0:
                payback = tank_investment / avoided_ead if avoided_ead > 0 else float('inf')
                bc_ratio = avoided_ead / tank_investment * project_life_years
            else:
                payback = 0
                bc_ratio = float('inf')
            
            profile_results[profile_name] = {
                'tr': tr,
                'prob_annual': prob_annual,
                'prob_annual_pct': prob_annual * 100,
                'prob_lifetime': prob_at_least_one,
                'prob_lifetime_pct': prob_at_least_one * 100,
                'damage_at_tr': damage_at_tr,
                'residual_ead': residual_ead,
                'payback_years': payback,
                'bc_ratio': bc_ratio,
                'color': profile_data['color'],
                'emoji': profile_data['emoji']
            }
        
        # =====================================================================
        # GENERATE TEXT REPORT
        # =====================================================================
        report_path = risk_dir / "risk_profile_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("          ANÁLISIS DE PERFIL DE RIESGO\n")
            f.write("="*70 + "\n\n")
            
            f.write("PARÁMETROS DEL ANÁLISIS\n")
            f.write("-"*40 + "\n")
            f.write(f"  TR de Diseño:          {design_tr} años\n")
            f.write(f"  Vida Útil del Proyecto:{project_life_years} años\n")
            f.write(f"  Tasa de Descuento:     {discount_rate*100:.1f}%\n")
            f.write(f"  Inversión en Tanques:  ${tank_investment:,.0f}\n\n")
            
            f.write("MÉTRICAS GLOBALES\n")
            f.write("-"*40 + "\n")
            f.write(f"  EAD Baseline (Sin Proyecto): ${baseline_ead:,.0f}/año\n")
            f.write(f"  EAD Con Proyecto:            ${project_ead:,.0f}/año\n")
            f.write(f"  EAD Evitado:                 ${avoided_ead:,.0f}/año\n\n")
            
            f.write("="*70 + "\n")
            f.write("          PERFILES DE RIESGO\n")
            f.write("="*70 + "\n\n")
            
            for profile_name, data in profile_results.items():
                f.write(f"{data['emoji']} {profile_name.upper()} (TR = {data['tr']} años)\n")
                f.write("-"*50 + "\n")
                f.write(f"  Probabilidad de Falla:\n")
                f.write(f"    • Anual:                   {data['prob_annual_pct']:.1f}%\n")
                f.write(f"    • En {project_life_years} años (al menos 1): {data['prob_lifetime_pct']:.1f}%\n")
                f.write(f"  Daño Esperado (TR={data['tr']}): ${data['damage_at_tr']:,.0f}\n")
                f.write(f"  Riesgo Residual (EAD):       ${data['residual_ead']:,.0f}/año\n")
                if tank_investment > 0:
                    f.write(f"  Período de Recuperación:     {data['payback_years']:.1f} años\n")
                    f.write(f"  Ratio B/C ({project_life_years} años):       {data['bc_ratio']:.2f}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("          INTERPRETACIÓN\n")
            f.write("="*70 + "\n\n")
            
            f.write("• AGRESIVO: Alta tolerancia al riesgo. Apropiado para áreas con bajo\n")
            f.write("  valor expuesto o donde las fallas tienen impacto limitado.\n\n")
            
            f.write("• MODERADO: Balance típico entre costo e inversión. Estándar para\n")
            f.write("  infraestructura urbana de mediana importancia.\n\n")
            
            f.write("• CONSERVADOR: Baja tolerancia a fallas. Recomendado para áreas con\n")
            f.write("  alto valor expuesto o infraestructura crítica.\n\n")
            
            f.write("• MUY CONSERVADOR: Mínimo riesgo aceptable. Apropiado para hospitales,\n")
            f.write("  instalaciones de emergencia, o zonas de alto valor económico.\n\n")
            
            f.write("="*70 + "\n")
            f.write("Fin del Reporte\n")
        
        print(f"  [Report] Saved: {report_path}")
        
        # =====================================================================
        # GENERATE PLOTS
        # =====================================================================
        
        # 1. PROBABILITY OF FAILURE BAR CHART
        fig, ax = plt.subplots(figsize=(12, 6))
        
        profiles = list(profile_results.keys())
        annual_probs = [profile_results[p]['prob_annual_pct'] for p in profiles]
        lifetime_probs = [profile_results[p]['prob_lifetime_pct'] for p in profiles]
        colors = [profile_results[p]['color'] for p in profiles]
        
        x = np.arange(len(profiles))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, annual_probs, width, label='Prob. Anual', color=colors, alpha=0.7)
        bars2 = ax.bar(x + width/2, lifetime_probs, width, label=f'Prob. en {project_life_years} años', 
                       color=colors, alpha=1.0, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Probabilidad de Falla (%)', fontsize=12)
        ax.set_xlabel('Perfil de Riesgo', fontsize=12)
        ax.set_title('Probabilidad de Falla por Perfil de Riesgo', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n(TR={profile_results[p]['tr']})" for p in profiles])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        prob_plot_path = risk_dir / "failure_probability_by_profile.png"
        fig.savefig(prob_plot_path, dpi=150)
        plt.close(fig)
        print(f"  [Plot] Saved: {prob_plot_path.name}")
        
        # 2. RESIDUAL RISK VS TR
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        trs_plot = [profile_results[p]['tr'] for p in profiles]
        residual_risks = [profile_results[p]['residual_ead'] / 1e6 for p in profiles]  # In millions
        damages = [profile_results[p]['damage_at_tr'] / 1e6 for p in profiles]
        
        ax1.bar(x, residual_risks, color=colors, alpha=0.8, label='Riesgo Residual (EAD)')
        ax1.set_xlabel('Perfil de Riesgo', fontsize=12)
        ax1.set_ylabel('Riesgo Residual EAD ($M/año)', fontsize=12, color='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{p}\n(TR={profile_results[p]['tr']})" for p in profiles])
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Secondary axis for damage at TR
        ax2 = ax1.twinx()
        ax2.plot(x, damages, 'ko-', linewidth=2, markersize=10, label='Daño Max (TR)')
        ax2.set_ylabel('Daño Máximo Esperado ($M)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.set_title('Riesgo Residual por Perfil de Aversión al Riesgo', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        residual_plot_path = risk_dir / "residual_risk_by_profile.png"
        fig.savefig(residual_plot_path, dpi=150)
        plt.close(fig)
        print(f"  [Plot] Saved: {residual_plot_path.name}")
        
        # 3. INVESTMENT vs RISK TRADE-OFF (if investment provided)
        if tank_investment > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate equivalent annual cost of investment
            annual_investment = tank_investment / project_life_years
            
            # Data for stacked bar
            for i, profile in enumerate(profiles):
                data = profile_results[profile]
                
                # Investment bar
                ax.barh(i, annual_investment / 1e6, color='#e74c3c', alpha=0.8, label='Inversión Anual' if i == 0 else '')
                
                # Residual risk bar (stacked)
                ax.barh(i, data['residual_ead'] / 1e6, left=annual_investment / 1e6, 
                       color=data['color'], alpha=0.6, label='Riesgo Residual' if i == 0 else '')
            
            ax.set_yticks(range(len(profiles)))
            ax.set_yticklabels([f"{p} (TR={profile_results[p]['tr']})" for p in profiles])
            ax.set_xlabel('Costo Anualizado ($M)', fontsize=12)
            ax.set_title('Composición del Costo Total por Perfil de Riesgo', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add total labels
            for i, profile in enumerate(profiles):
                data = profile_results[profile]
                total = annual_investment + data['residual_ead']
                ax.text(total / 1e6 + 0.1, i, f"${total/1e6:.2f}M", va='center', fontsize=10)
            
            plt.tight_layout()
            tradeoff_plot_path = risk_dir / "investment_risk_tradeoff.png"
            fig.savefig(tradeoff_plot_path, dpi=150)
            plt.close(fig)
            print(f"  [Plot] Saved: {tradeoff_plot_path.name}")
        
        print(f"\n  [RiskProfile] All outputs saved to: {risk_dir}")
        
        return {
            'report_path': str(report_path),
            'profiles': profile_results,
            'baseline_ead': baseline_ead,
            'project_ead': project_ead,
            'avoided_ead': avoided_ead,
            'tank_investment': tank_investment
        }
