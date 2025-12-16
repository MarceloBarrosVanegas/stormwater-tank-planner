import os
import shutil
import math
import sys
import swmmio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from pyproj import CRS

# SYS PATH APPEND for rut_06 (User provided paths)
sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\src')
sys.path.append(r'C:\Users\Alienware\OneDrive\ALCANTARILLADO_PyQt5\00_MODULOS\pypiper\gui')
try:
    from rut_06_pipe_sizing import SeccionLlena
    print("Successfully imported SeccionLlena from rut_06_pipe_sizing")
except ImportError as e:
    print(f"Warning: Could not import SeccionLlena: {e}")
    SeccionLlena = None

class SWMMModifier:
    def __init__(self, inp_file, crs=None):
        self.inp_file = inp_file
        # Read file with compatible encoding
        with open(self.inp_file, 'r', encoding='latin-1') as f:
            self.lines = f.readlines()
            
        # Initialize swmmio model (optional usage)
        self.swmm_model = swmmio.Model(self.inp_file)
        self.flooding_gdf = None
        self.swmm_crs = crs
        
    def load_flooding_data(self, gpkg_path):
        """Loads the flooding nodes GPKG for fast data retrieval."""
        try:
            self.flooding_gdf = gpd.read_file(gpkg_path)
            # Only override CRS if not manually provided
            if self.swmm_crs is None:
                self.swmm_crs = self.flooding_gdf.crs
            
            # Ensure NodeID is the index for faster lookup
            if "NodeID" in self.flooding_gdf.columns:
                self.flooding_gdf.set_index("NodeID", inplace=True)
            print(f"Loaded flooding data from {gpkg_path} (CRS: {self.swmm_crs})")
        except Exception as e:
            print(f"Error loading GPKG: {e}")
            self.flooding_gdf = None

    def _find_last_line_of_section(self, section_name):
        """
        Finds the insertion index for a new item in a given section.
        It looks for the section header, then finds the end of the data block.
        """
        header = f"[{section_name}]" # Case specific, usually uppercase in SWMM
        
        # 1. Find the section header
        header_idx = -1
        for i, line in enumerate(self.lines):
            if line.strip().upper() == header:
                header_idx = i
                break
        
        if header_idx == -1:
            return -1 # Section not found
            
        # 2. Find the start of the NEXT section
        current_idx = header_idx + 1
        
        while current_idx < len(self.lines):
            line = self.lines[current_idx].strip()
            # If we hit a new section (start with [), stop
            if line.startswith("[") and line.endswith("]"):
                return current_idx # Insert *before* the next section header
            current_idx += 1
            
        # If we reached EOF without finding a new section, append at the end
        return len(self.lines)

    def _create_section(self, section_name, columns_header):
        """Creates a new section if it doesn't exist."""
        # Try to find a good place to insert (Default: Before [CONDUITS] or at end)
        insert_idx = -1
        target_successor = "[CONDUITS]" 
        
        for i, line in enumerate(self.lines):
            if line.strip().upper() == target_successor:
                insert_idx = i
                break
        
        if insert_idx == -1:
            insert_idx = len(self.lines) 
            
        # Prepare header text
        header_block = [
            "\n",
            f"[{section_name}]\n",
            f";;{columns_header}\n",
            f";;{'-'*len(columns_header)}\n"
        ]
        
        # Insert (reversed to keep order)
        for line in reversed(header_block):
            self.lines.insert(insert_idx, line)
            
        return self._find_last_line_of_section(section_name)

    def add_storage_unit(self, name, area, max_depth, terrain_elev=None, node_invert=None):
        """
        Adds a FUNCTIONAL storage unit (Constant Area).
        Logic for Elevation:
          - If terrain_elev is provided: Invert = terrain_elev - max_depth
          - Else if node_invert is provided: Invert = node_invert - max_depth
          - Else: Raises ValueError
        """
        if terrain_elev is not None:
            elev = terrain_elev - max_depth
            print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = Terrain {terrain_elev:.2f} - Depth {max_depth:.2f}")
        elif node_invert is not None:
            elev = node_invert - max_depth
            print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = NodeInvert {node_invert:.2f} - Depth {max_depth:.2f}")
        else:
            raise ValueError(f"Tank {name}: Must provide either terrain_elev or node_invert to calculate tank elevation.")

        # Params: A B C -> Area + 0*Depth^0
        params_str = f"{area:.2f} 0 0"
        
        new_line = (
            f"{name:<16} "
            f"{elev:<10.2f} "
            f"{max_depth:<10.2f} "
            f"0.0        "
            f"FUNCTIONAL "
            f"{params_str:<20} "
            f"0           0\n"
        )
        
        section = "STORAGE"
        idx = self._find_last_line_of_section(section)
        
        if idx == -1:
            headers = "Name           Elev.      MaxDepth   InitDepth  Shape      Curve Name/Params            PondedArea  EvapFrac"
            idx = self._create_section(section, headers)
            
        self.lines.insert(idx, new_line)

    def add_conduit(self, name, from_node, to_node, length, roughness=0.015, inlet_offset=0.0, outlet_offset=0.0):
        """
        Adds a conduit definition.
        [CONDUITS]
        ;;Name     Inlet      Outlet     Length  ManningN  InletOff  OutletOff  InitFlow  MaxFlow
        C1        Node1      TK1        10      0.013     0.0       0.0        0         0
        """
        new_line = (
            f"{name:<16} "
            f"{from_node:<16} "
            f"{to_node:<16} "
            f"{length:<10.2f} "
            f"{roughness:<10.4f} "
            f"{inlet_offset:<10.2f} "
            f"{outlet_offset:<10.2f} "
            f"0.0        0.0\n"
        )
        
        section = "CONDUITS"
        idx = self._find_last_line_of_section(section)
        if idx == -1:
             headers = "Name           Inlet           Outlet          Length     ManningN   InletOff   OutletOff  InitFlow   MaxFlow"
             idx = self._create_section(section, headers)
             
        self.lines.insert(idx, new_line)

    def add_rect_closed_xsection(self, link_name, height):
        """
        Adds RECT_CLOSED XSECTION.
        User Req: Base = 2 * Height.
        [XSECTIONS]
        ;;Link     Shape        Geom1(H)   Geom2(W)   Geom3  Geom4  Barrels
        C1        RECT_CLOSED  1.0        2.0        0      0      1
        """
        width = 2.0 * height
        
        new_line = (
            f"{link_name:<16} "
            f"RECT_CLOSED "
            f"{height:<10.2f} "
            f"{width:<10.2f} "
            f"0.0        0.0        1\n"
        )
        
        section = "XSECTIONS"
        idx = self._find_last_line_of_section(section)
        if idx != -1:
            self.lines.insert(idx, new_line)

    def add_coordinate(self, name, x, y):
        """
        Adds coordinate.
        [COORDINATES]
        ;;Node          X-Coord           Y-Coord
        """
        new_line = f"{name:<16} {x:<16.3f} {y:<16.3f}\n"
        
        section = "COORDINATES"
        idx = self._find_last_line_of_section(section)
        if idx != -1:
             self.lines.insert(idx, new_line)

    def get_node_coords(self, node_id):
        """
        Retrieves (x, y) coordinates for a given node.
        Prioritizes loaded GPKG, falls back to swmmio.
        """
        # 1. Try GPKG
        if self.flooding_gdf is not None:
            if node_id in self.flooding_gdf.index:
                try:
                    row = self.flooding_gdf.loc[node_id]
                    # Handle case where index might not be unique (returns DataFrame)
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    return float(row['NodeCoordsX']), float(row['NodeCoordsY'])
                except Exception as e:
                    print(f"GPKG lookup failed for coords: {e}")

        # 2. Fallback to swmmio
        try:
            nodes = self.swmm_model.nodes()
            if node_id in nodes.index:
                row = nodes.loc[node_id]
                coords = row.get('coords')
                if coords and len(coords) > 0:
                    pt = coords[0] 
                    return pt[0], pt[1]
        except Exception as e:
            print(f"swmmio lookup failed for coords: {e}")
            
        return 0.0, 0.0

    def get_node_invert(self, node_id):
        """
        Retrieves Invert Elevation for a given node.
        Prioritizes loaded GPKG, falls back to swmmio.
        """
        # 1. Try GPKG
        if self.flooding_gdf is not None:
            if node_id in self.flooding_gdf.index:
                try:
                    row = self.flooding_gdf.loc[node_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    return float(row['InvertElevation'])
                except Exception as e:
                    print(f"GPKG lookup failed for invert: {e}")

        # 2. Fallback to swmmio
        try:
            nodes = self.swmm_model.nodes()
            if node_id in nodes.index:
                return float(nodes.loc[node_id, 'InvertElev'])
        except Exception as e:
            print(f"swmmio lookup failed for invert: {e}")
            
        print(f"Warning: Node {node_id} invert not found. Defaulting to 0.0")
        return 0.0

    def add_pipeline_from_gpkg(self, gpkg_path, connection_id=None):
        """
        Parses a GPKG file to generate a SWMM pipeline (Junctions + Conduits).
        Replaces the FINAL Node with a Storage Unit using default sizing.
        Connects the START Node to an existing 'connection_id' if provided.
        """
        if SeccionLlena is None:
            print("Cannot generate pipeline: SeccionLlena dependency missing.")
            return

        print(f"Loading pipeline from {gpkg_path}...")
        gdf = gpd.read_file(gpkg_path)
        
        # CRS Reprojection
        if self.swmm_crs is not None:
             if gdf.crs != self.swmm_crs:
                 print(f"Reprojecting pipeline from {gdf.crs} to {self.swmm_crs}")
                 gdf = gdf.to_crs(self.swmm_crs)
        else:
             print("Warning: SWMM CRS not defined. Pipeline coords assume match.")
        
        # Track nodes to identify start/end
        from_nodes = set()
        to_nodes = set()
        
        node_data = {} 
        conduits_to_add = []
        xsections_to_add = []

        # SINGLE PASS: Build data structures
        for idx, row in gdf.iterrows():
            tramo = str(row['Tramo'])
            parts = tramo.split('-')
            if len(parts) < 2:
                print(f"Skipping invalid Tramo: {tramo}")
                continue
                
            node_from_id = parts[0]
            node_to_id = parts[1]
            
            from_nodes.add(node_from_id)
            to_nodes.add(node_to_id)

            # Geometry & Elevs
            geom = row.geometry
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                node_data[node_from_id] = {'x': coords[0][0], 'y': coords[0][1], 'invert': float(row['ZFI'])}
                node_data[node_to_id] = {'x': coords[-1][0], 'y': coords[-1][1], 'invert': float(row['ZFF'])}
            
            # Conduit Data
            # Name: Use Tramo 
            conduit_id = tramo 
            length = float(row['L'])
            
            # Manning
            manning_n = float(row['Rug'])

            conduits_to_add.append({
                'name': conduit_id,
                'from': node_from_id,
                'to': node_to_id,
                'len': length,
                'n': manning_n
            })

            # XSection Data
            d_int = str(row['D_int'])
            seccion_type = str(row['Seccion']).lower().strip()
            
            try:
                # geom_arr shape (1, 4) -> [geom0, geom1, geom2, geom3]
                geom_arr = SeccionLlena.section_str2float([d_int], return_all=True, sep='x')
                g = geom_arr[0] # The row
                
                shape_swmm = None
                geoms = [0.0, 0.0, 0.0, 0.0]
                
                if seccion_type == 'rectangular':
                    # Rectangular Closed: geom0=Base, geom1=Height
                    shape_swmm = 'RECT_CLOSED'
                    geoms[0] = g[1] # Height 
                    geoms[1] = g[0] # Width
                    
                elif seccion_type == 'rectangular_abierta':
                    # Rectangular Open
                    shape_swmm = 'RECT_OPEN'
                    geoms[0] = g[1] # Height
                    geoms[1] = g[0] # Width
                    
                elif seccion_type == 'trapezoidal':
                    # Trapezoidal Open
                    # SWMM TRAPEZOIDAL: Geom1=Height, Geom2=BottomWidth, Geom3=LeftSlope(H/V), Geom4=RightSlope(H/V)
                    # User doc: geom2/3 are "pendiente (vertical/horizontal)" -> V/H.
                    # SWMM usually uses H/V for side slopes in xsections? 
                    # Checking SWMM 5.1 Manual: "Side slopes are dimensionless (horizontal run per unit of vertical rise)." -> H/V.
                    # So if input is V/H, we invert it.
                    shape_swmm = 'TRAPEZOIDAL'
                    geoms[0] = g[1] # Height
                    geoms[1] = g[0] # Bottom Width
                    geoms[2] = 1.0 / g[2] if g[2] != 0 else 0.0 # Left Slope H/V
                    geoms[3] = 1.0 / g[3] if g[3] != 0 else 0.0 # Right Slope H/V
                    
                elif seccion_type == 'triangular':
                    # Triangular: geom0=?, geom1=H, geom2=SlopeL, geom3=SlopeR (V/H)
                    # SWMM TRIANGULAR: Geom1=MaxDepth, Geom2=TopWidth
                    # Calc TopWidth T = H * (RunL + RunR) = H * ( (1/SL) + (1/SR) )
                    shape_swmm = 'TRIANGULAR'
                    h_val = g[1]
                    s_l = g[2] # V/H
                    s_r = g[3] # V/H
                    
                    # Run/Rise = 1/Slope
                    run_l = 1.0 / s_l if s_l != 0 else 0.0
                    run_r = 1.0 / s_r if s_r != 0 else 0.0
                    
                    top_width = h_val * (run_l + run_r)
                    
                    geoms[0] = h_val
                    geoms[1] = top_width
                    
                elif seccion_type == 'circular':
                     shape_swmm = 'CIRCULAR'
                     geoms[0] = g[0] # Diameter
                
                if shape_swmm:
                    xsections_to_add.append({
                        'link': conduit_id,
                        'shape': shape_swmm,
                        'geoms': geoms
                    })
                else:
                    print(f"Warning: Unknown section type '{seccion_type}' for {conduit_id}")
                    
            except Exception as e:
                print(f"Error parsing section {d_int} ({seccion_type}): {e}")

        # --- TOPOLOGY ANALYSIS ---
        # Start Node: From but not To
        start_nodes = from_nodes - to_nodes
        start_node = list(start_nodes)[0] if start_nodes else None
        
        # End Node: To but not From
        end_nodes = to_nodes - from_nodes
        end_node = list(end_nodes)[0] if end_nodes else None
        
        print(f"Topology: StartNode={start_node}, EndNode={end_node}")

        # --- WRITE TO SWMM ---
        
        # 1. NODES
        for nid, data in node_data.items():
            # Skip End Node (Tank)
            if nid == end_node:
                continue
                
            # Skip Start Node IF connection_id provided (Using existing node)
            if connection_id and nid == start_node:
                # Optional: Update the coordinates of the existing node in the modifier if needed?
                # User said: "modificar la priemra cordeanda a la cordenada del nodo derivadi"
                # Actually, SWMM defines coordinates in [COORDINATES]. If we don't add it here, 
                # we assume the existing file has correct coords for connection_id.
                # If we WANT to snap the existing node to the pipeline start, we would act here.
                # Assuming simple connectivity for now.
                print(f"  Skipping definition of StartNode {nid} (Mapping to {connection_id})")
                continue
                
            self.add_junction(nid, data['invert'])
            self.add_coordinate(nid, data['x'], data['y'])
            
        # 2. TANK (End Node)
        if end_node:
            data = node_data[end_node]
            self.add_storage_unit(
                name=f"TK_{end_node}",
                area=100.0, 
                max_depth=5.0,
                node_invert=data['invert']
            )
            self.add_coordinate(f"TK_{end_node}", data['x'], data['y'])
            
        # 3. CONDUITS & XSECTIONS
        # Remap nodes in conduits
        for c in conduits_to_add:
            # Map Start Node
            if connection_id and c['from'] == start_node:
                c['from'] = connection_id
            
            # Map End Node
            if c['to'] == end_node:
                c['to'] = f"TK_{end_node}"
            
            self.add_conduit(c['name'], c['from'], c['to'], c['len'], c['n'])
            
        for x in xsections_to_add:
            self.add_xsection(x['link'], x['shape'], x['geoms'])

    def add_junction(self, name, elev, max_depth=0):
        """Adds a junction."""
        new_line = f"{name:<16} {elev:<10.2f} {max_depth:<10.2f} 0.0        0.0        0.0\n"
        section = "JUNCTIONS"
        idx = self._find_last_line_of_section(section)
        if idx == -1:
            self._create_section(section, "Name           Elev.      MaxDepth   InitDepth  SurDepth   Aponded")
            idx = self._find_last_line_of_section(section)
        self.lines.insert(idx, new_line)

    def add_xsection(self, link, shape, geoms):
        """Generic XSection adder. geoms is a list of [geom1, geom2, geom3, geom4]."""
        # SWMM format: Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert
        # Format string with fixed width
        geom_str = f"{geoms[0]:<10.2f} {geoms[1]:<10.2f} {geoms[2]:<10.2f} {geoms[3]:<10.2f}"
        new_line = f"{link:<16} {shape:<12} {geom_str} 1          0\n"
        
        section = "XSECTIONS"
        idx = self._find_last_line_of_section(section)
        if idx != -1: # Section usually exists
             self.lines.insert(idx, new_line)

    def save(self, output_path):
        with open(output_path, 'w', encoding='latin-1') as f:
            f.writelines(self.lines)

# ------------------------------------------------------------------------------
# TEST RUN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    base_file = "COLEGIO_TR25_v6.inp"
    gpkg_file = r"00_flooding_stats/00_flooding_nodes.gpkg" 

    source_crs = CRS("""PROJCRS["SIRES-DMQ",
BASEGEOGCRS["WGS 84",
    DATUM["World Geodetic System 1984",
        ELLIPSOID["WGS 84",6378137,298.257223563,
        LENGTHUNIT["metre",1]],ID["EPSG",6326]],
    PRIMEM["Greenwich",0,
        ANGLEUNIT["Degree",0.0174532925199433]]],
CONVERSION["unnamed",
    METHOD["Transverse Mercator",ID["EPSG",9807]],
    PARAMETER["Latitude of natural origin",0,
        ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8801]],
    PARAMETER["Longitude of natural origin",-78.5,
        ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8802]],
    PARAMETER["Scale factor at natural origin",1.0004584,
        SCALEUNIT["unity",1],ID["EPSG",8805]],
    PARAMETER["False easting",500000,
        LENGTHUNIT["metre",1],ID["EPSG",8806]],
    PARAMETER["False northing",10000000,
        LENGTHUNIT["metre",1],ID["EPSG",8807]]],
CS[Cartesian,3],
AXIS["(E)",east,ORDER[1],
    LENGTHUNIT["metre",1,ID["EPSG",9001]]],
AXIS["(N)",north,ORDER[2],
    LENGTHUNIT["metre",1,ID["EPSG",9001]]],
AXIS["ellipsoidal height (h)",up,ORDER[3],
LENGTHUNIT["metre",1,ID["EPSG",9001]]]]""")
    
    
    if os.path.exists(base_file):
        print(f"Reading {base_file}...")
        mod = SWMMModifier(base_file)
        
        # Load GPKG
        if os.path.exists(gpkg_file):
            mod.load_flooding_data(gpkg_file)
        else:
            print(f"Warning: GPKG file {gpkg_file} not found.")

        # --- TEST SETUP ---
        target_node = "P0061405" 
        print(f"Targeting Node: {target_node}")

        # 1. Get Node Data
        node_x, node_y = mod.get_node_coords(target_node)
        node_invert = mod.get_node_invert(target_node)
        
        # Case A: Terrain Provided (e.g., 2800 m)
        print("\n--- Test Case A: Terrain Provided (2800m) ---")
        mod.add_storage_unit(
            name="TK_TERRAIN",
            area=20000.0,
            max_depth=5.0,
            terrain_elev=2780.0
        )
        
        # # Case B: No Terrain, use Node Invert
        # print("\n--- Test Case B: No Terrain (Use Node Invert) ---")
        # mod.add_storage_unit(
        #     name="TK_NODE",
        #     area=200.0,
        #     max_depth=5.0,
        #     node_invert=node_invert
        # )
        
        # Add supporting elements for TK_TERRAIN for completeness
        # Add supporting elements for TK_TERRAIN for completeness
        mod.add_coordinate("TK_TERRAIN", node_x + 5.0, node_y)
        
        # 5. Conduit
        # CONNECT TO TOP: set outlet_offset = tank depth (5.0)
        # CONNECT TO BOTTOM: set outlet_offset = 0.0
        mod.add_conduit(
            name="C_TERRAIN", 
            from_node=target_node, 
            to_node="TK_TERRAIN", 
            length=15.0, 
            outlet_offset=5.0  # <--- CONNECTING TO TOP (Matches MaxDepth)
        )

        mod.add_rect_closed_xsection("C_TERRAIN", 0.8)

        # Save OLD test
        # out_file = "COLEGIO_TR25_v6_MODIFIED.inp"
        # mod.save(out_file)
        
        # --- TEST PIPELINE GPKG ---
        print("\n--- Test Case C: Pipeline from GPKG ---")
        gpkg_pipeline = r"P0061405_PredioID3.gpkg"
        
        if os.path.exists(gpkg_pipeline):
            # Reload fresh model to avoid duplicates from previous tests
            # Pass the manual CRS (source_crs)
            mod_pipe = SWMMModifier(base_file, crs=source_crs)
            
            # Load flooding data (CRS already set manually, so it won't be overwritten)
            if os.path.exists(gpkg_file):
                 mod_pipe.load_flooding_data(gpkg_file)
            
            # P0061405 is the derivation node
            mod_pipe.add_pipeline_from_gpkg(gpkg_pipeline, connection_id="P0061405")
            
            out_pipe = "COLEGIO_TR25_v6_PIPELINE.inp"
            mod_pipe.save(out_pipe)
            print(f"Pipeline File saved: {out_pipe}")
        else:
            print(f"Pipeline GPKG {gpkg_pipeline} not found.")

    else:
        print(f"Base file {base_file} not found.")
