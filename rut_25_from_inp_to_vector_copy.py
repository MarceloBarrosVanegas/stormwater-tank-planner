
import swmmio
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import os
import shutil
from pathlib import Path
import unicodedata
from pyswmm import Simulation, Output, LinkSeries

import config
config.setup_sys_path()

class CleanINP:
    """Handles the cleaning of INP files to ensure encoding compatibility."""
    
    def __init__(self, inp_path):
        self.original_path = inp_path
        self.cleaned_path = inp_path.replace(".inp", "_clean.inp")
        self._clean_file()

    def _clean_file(self):
        # print('-' *100 )
        # print(f"Cleaning INP file: {self.original_path}")
        try:
            # Read as latin-1 to capture most characters
            with open(self.original_path, 'r', encoding='latin-1', errors='replace') as f_in:
                content = f_in.read()
            
            # Force to ASCII to prevent 'swmmio' crashing on Windows cp1252
            content = unicodedata.normalize('NFKD', content).encode('ascii', 'ignore').decode('ascii')
            
            with open(self.cleaned_path, 'w', encoding='ascii') as f_out:
                f_out.write(content)
                
            # print(f"File cleaned and saved to: {self.cleaned_path}")
        except Exception as e:
            print(f"Critical error cleaning file: {e}")
            raise


class PhysicalNetwork:
    """Extracts the physical and topological properties of the network."""
    
    def __init__(self, model):
        self.model = model
        self.links_df = None
        self.gdf = None
        
    def build(self):
        # print("Building Physical Network...")
        # Load links and ensure we don't carry over bad results from swmmio auto-parsing
        raw_links = self.model.links.dataframe.copy()
        # Drop potential result columns if they exist to avoid confusion
        bad_cols = []
        self.links_df = raw_links.drop(columns=[c for c in bad_cols if c in raw_links.columns], errors='ignore')
        
        nodes_df = self.model.nodes.dataframe.copy()
        coords_df = self.model.inp.coordinates.copy()
        
        # 1. Enrich Nodes
        node_cols = ['InvertElev', 'MaxDepth']
        # Inlet
        self.links_df = self.links_df.merge(nodes_df[node_cols], left_on='InletNode', right_index=True, how='left')
        self.links_df.rename(columns={'InvertElev': 'InletNode_InvertElev', 'MaxDepth': 'InletNode_MaxDepth'}, inplace=True)
        # Outlet
        self.links_df = self.links_df.merge(nodes_df[node_cols], left_on='OutletNode', right_index=True, how='left')
        self.links_df.rename(columns={'InvertElev': 'OutletNode_InvertElev', 'MaxDepth': 'OutletNode_MaxDepth'}, inplace=True)
        
        # 2. Add Geometry
        nodes_in_xy = coords_df[['X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
        self.links_df = self.links_df.merge(nodes_in_xy, left_on='InletNode', right_index=True, how='left')
        
        nodes_out_xy = coords_df[['X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
        self.links_df = self.links_df.merge(nodes_out_xy, left_on='OutletNode', right_index=True, how='left')
        
        # Drop missing
        self.links_df = self.links_df.dropna(subset=['X1', 'Y1', 'X2', 'Y2'])
        
        # LineStrings
        self.links_df['geometry'] = self.links_df.apply(lambda row: LineString([(row.X1, row.Y1), (row.X2, row.Y2)]), axis=1)
        

        # Calculate Slope
        # Fill offsets with 0 if missing
        self.links_df['InOffset'] = pd.to_numeric(self.links_df.get('InOffset', 0), errors='coerce').fillna(0)
        self.links_df['OutOffset'] = pd.to_numeric(self.links_df.get('OutOffset', 0), errors='coerce').fillna(0)
        
        z_in = self.links_df['InletNode_InvertElev'] + self.links_df['InOffset']
        z_out = self.links_df['OutletNode_InvertElev'] + self.links_df['OutOffset']
        
        # Calculate geometric length if explicit 'Length' is missing
        geom_length = self.links_df['geometry'].apply(lambda x: x.length)
        length = self.links_df.get('Length', geom_length)
        
        # Avoid division by zero
        length = length.replace(0, 0.01)
        
        self.links_df['Slope'] = (z_in - z_out) / length
        
        # 3. Identify Collectors
        self._identify_ramales()
        
        self.gdf = gpd.GeoDataFrame(self.links_df, geometry='geometry')
        # print("Physical Network built.")
        return self.gdf





    def _identify_ramales(self):
        # print("Identifying Ramales (Collectors)...")
        G = nx.DiGraph()
        for idx, row in self.links_df.iterrows():
            G.add_edge(row['InletNode'], row['OutletNode'], length=row.get('Length', row.geometry.length))
            
        memo_dist = {}
        def get_upstream_dist(node):
            if node in memo_dist: return memo_dist[node]
            preds = list(G.predecessors(node))
            if not preds:
                memo_dist[node] = 0
                return 0
            max_d = 0
            for p in preds:
                d = get_upstream_dist(p) + G[p][node]['length']
                if d > max_d: max_d = d
            memo_dist[node] = max_d
            return max_d

        for n in G.nodes():
            try: get_upstream_dist(n)
            except: pass
            
        # Topo sort
        try: nodes_sorted = list(nx.topological_sort(G))
        except: nodes_sorted = sorted(G.nodes(), key=lambda n: memo_dist.get(n, 0))
        
        edge_ids = {}
        next_id = 1
        
        for node in nodes_sorted:
            succs = list(G.successors(node))
            if not succs: continue
            
            # Find main parent
            preds = list(G.predecessors(node))
            main_pred = None
            if preds:
                main_pred = max(preds, key=lambda p: memo_dist.get(p, 0) + G[p][node]['length'])
                
            current_id = edge_ids.get((main_pred, node)) if main_pred else None
            
            for succ in succs:
                if current_id:
                    edge_ids[(node, succ)] = current_id
                else:
                    edge_ids[(node, succ)] = next_id
                    next_id += 1
                    
        # print(f"Identified {next_id - 1} Ramales.")
        
        self.links_df['Ramal'] = self.links_df.apply(
            lambda row: edge_ids.get((row['InletNode'], row['OutletNode']), 0), axis=1
        ).astype('U256')


class HydraulicNetwork:
    """Handles simulation and result extraction using pyswmm."""

    def __init__(self, model):
        self.model = model

    def run_and_get_results(self):
        # print("Running Hydraulic Simulation with pyswmm...")
        sim_path = self.model.inp.path

        # Output file usually has same base as input but .out extension
        out_path = sim_path.replace(".inp", ".out")

        if not os.path.exists(out_path):
             # Try RPT path rename
             out_path = sim_path.replace(".inp", ".rpt").replace(".rpt", ".out")

        # Check if output file already exists; if so, skip simulation
        if os.path.exists(out_path):
            # print(f"Output file already exists at {out_path}. Skipping simulation.")
            pass
        else:
            # Run Simulation
            try:
                sim = Simulation(sim_path)
                sim.execute()
                print("Simulation completed.")
            except Exception as e:
                print(f"Simulation failed: {e}")
                return None

        # print("Extracting results from output file...")
        if not os.path.exists(out_path):
            print(f"Output file not found at {out_path}")
            return None

        results = {}
        try:
            with Output(out_path) as out:
                # print(f"Reading output file: {out_path}")
                # print(f"Extracting results for {len(out.links)} links...")

                # Iterate over all links in the output
                for link_id in out.links:
                    try:
                        # Get Time Series for this link
                        ls = LinkSeries(out)[link_id]

                        # Calculate Max Values
                        # Note: This loads the entire series into memory.
                        # pyswmm LinkSeries acts as a dict {datetime: value}, so max() on it gets the max DATE.
                        # We must take max(list(series.values()))
                        max_flow = max(list(ls.flow_rate.values()))
                        max_vel = max(list(ls.flow_velocity.values()))
                        max_cap = max(list(ls.capacity.values())) # Capacity in pyswmm is usually h/D (fraction full)

                        results[link_id] = {
                            'MaxFlow': max_flow,
                            'MaxVel': max_vel,
                            'Capacity': max_cap # User requested Capacity = h/D (pyswmm .capacity is usage fraction)
                        }
                    except Exception as e_link:
                        # Maybe link doesn't exist in results or has no data
                        continue

            df_res = pd.DataFrame.from_dict(results, orient='index')
            # print(f"Extracted results for {len(df_res)} links.")
            # print('-' * 100)
            return df_res

        except Exception as e:
            print(f"Failed to read binary output: {e}")
            return None


class NetworkExporter:
    """
    Main class (Facade) to export enriched network from SWMM INP files.
    
    The output GeoDataFrame contains the following columns:
    
    1. Physical Properties (from INP):
       - geometry: LineString representing the pipe.
       - Length: Conduit length [m].
       - Slope: Calculated slope (m/m).
       - Roughness: Manning's n.
       - InletNode, OutletNode: ID of upstream/downstream nodes.
       - Shape: Cross-section shape (e.g., RECT_CLOSED, CIRCULAR).
       - Geom1, Geom2, Geom3, Geom4: Cross-section dimensions (e.g., Depth/Diameter).
       - Barrels: Number of barrels.
       - InOffset, OutOffset: Invert offsets from node invert [m].
       - X1, Y1, X2, Y2: Coordinates of start/end nodes.
       
    2. Enriched Topology:
       - InletNode_InvertElev: Invert elevation of start node [m].
       - OutletNode_InvertElev: Invert elevation of end node [m].
       - InletNode_MaxDepth, OutletNode_MaxDepth: Max depth of nodes [m].
       - Ramal: Integer ID identifying continuous collectors (Graph-based).
       
    3. Hydraulic Results (if run_hydraulics=True):
       - MaxFlow: Maximum flow rate [m3/s] (or model units).
       - MaxVel: Maximum flow velocity [m/s].
       - Capacity: Maximum relative depth (h/D), range 0.0-1.0+.
                  (Note: Values >1.0 indicate pressurized flow/surcharge).

    Usage:
        exporter = NetworkExporter(inp_file)
        gdf = exporter.run(output_gpkg="output.gpkg")
    """
    def __init__(self, inp_file):
        self.inp_file = inp_file

    def run(self, output_gpkg=None, run_hydraulics=True, crs=config.PROJECT_CRS):
        """
        Runs the extraction pipeline.
        
        Args:
            output_gpkg (str, optional): Path to save GeoPackage. If None, does not save.
            run_hydraulics (bool): If True, runs pyswmm simulation and enriches results. 
                                   If False, exports only physical network.
            crs: Coordinate Reference System. Default: config.PROJECT_CRS (SIRES-DMQ).
            
        Returns:
            gpd.GeoDataFrame: Enriched network.
        """
        try:
            # 1. Clean File
            cleaner = CleanINP(self.inp_file)
            model = swmmio.Model(cleaner.cleaned_path)
            self.out_path_file = Path(cleaner.cleaned_path).with_suffix('.out')
            
            # 2. Build Physical
            physical = PhysicalNetwork(model)
            gdf = physical.build()
            
            # 3. Hydraulics (Optional)
            if run_hydraulics:
                hydraulic = HydraulicNetwork(model)
                results_df = hydraulic.run_and_get_results()
                
                if results_df is not None and not results_df.empty:
                    # print(f"Merging Hydraulic Results ({len(results_df)} rows) into Physical Network ({len(gdf)} rows)...")
                    # Ensure indices match format (string) and strip whitespace
                    gdf.index = gdf.index.astype(str).str.strip()
                    results_df.index = results_df.index.astype(str).str.strip()
                    
                    # Check intersection
                    intersection = gdf.index.intersection(results_df.index)
                    # print(f"Index Intersection Count: {len(intersection)}")
                    
                    if len(intersection) == 0:
                         print("WARNING: No overlap between Physical and Hydraulic indices! Check IDs.")
                    
                    gdf = gdf.merge(results_df, left_index=True, right_index=True, how='left')
                    # print("Merge complete.")
                    
                    # Validation
                    if 'MaxFlow' in gdf.columns:
                        non_na = gdf['MaxFlow'].notna().sum()
                        # print(f"Rows with valid MaxFlow: {non_na}")
                else:
                    # print("Skipping Hydraulic enrichment (no results).")
                    pass
            else:
                # print("Skipping Hydraulic Simulation (run_hydraulics=False).")
                pass
            
            # 4. Clean formatting
            for col in gdf.columns:
                if gdf[col].dtype == 'object' and col != 'geometry':
                    gdf[col] = gdf[col].astype(str)
            
            # 5. Set CRS if provided
            if crs:
                gdf.set_crs(crs, inplace=True, allow_override=True)
                # print(f"CRS set to: {crs}")
                    
            if output_gpkg:
                print(f"Saving to {output_gpkg}...")
                gdf.to_file(output_gpkg, driver='GPKG')
                # print("Done.")
                
            return gdf

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"CRITICAL FAILURE: {e}")
            return None

if __name__ == "__main__":
    inp_file = str(config.SWMM_FILE)
    output_file = str(config.CODIGOS_DIR / "full_network_enriched.gpkg")
    
    exporter = NetworkExporter(inp_file)
    exporter.run(output_file, crs=config.PROJECT_CRS)
