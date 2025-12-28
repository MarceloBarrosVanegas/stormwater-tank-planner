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

"""
SWMM Modifier Module
====================

This module provides the `SWMMModifier` class, which allows for programmatic manipulation
of SWMM (Storm Water Management Model) input files (.inp). It supports adding nodes,
conduits, storage units, and coordinates, as well as integrating data from GeoPackage
(GPKG) files for pipeline generation.

Classes
-------
SWMMModifier
    A class to parse, modify, and save SWMM .inp files.

Dependencies
------------
- os, shutil, math, sys
- swmmio
- pandas, geopandas
- numpy
- pyproj
- rut_06_pipe_sizing.SeccionLlena (optional, for section parsing)
"""

# SYS PATH APPEND for rut_06 (User provided paths)
import config
config.setup_sys_path()
try:
    from rut_06_pipe_sizing import SeccionLlena
    print("Successfully imported SeccionLlena from rut_06_pipe_sizing")
except ImportError as e:
    print(f"Warning: Could not import SeccionLlena: {e}")
    SeccionLlena = None

class SWMMModifier:
    """
    A class to parse, modify, and save SWMM .inp files.

    This class reads an existing .inp file into memory as a list of lines, allowing
    for insertion of new sections and elements (junctions, conduits, storage units, etc.).
    It also provides functionality to integrate geospatial data from GPKG files.

    Parameters
    ----------
    inp_file : str
        The path to the input SWMM .inp file.
    crs : pyproj.CRS or str, optional
        The Coordinate Reference System (CRS) to be used for spatial operations.
        If None, it may be inferred from loaded GPKG data.

    Attributes
    ----------
    inp_file : str
        Path to the source .inp file.
    lines : list of str
        The content of the .inp file stored as a list of strings.
    swmm_model : swmmio.Model
        An initialized swmmio Model object for read operations (optional).
    flooding_gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing flooding node data loaded from a GPKG.
    swmm_crs : pyproj.CRS
        The active CRS for the model.
    """
    def __init__(self, inp_file, crs=None):
        self.inp_file = inp_file
        # Read file with compatible encoding
        with open(self.inp_file, 'r', encoding='latin-1') as f:
            self.lines = f.readlines()
            
        # Initialize swmmio model (optional usage)
        self.swmm_model = swmmio.Model(str(self.inp_file))
        self.flooding_gdf = None
        self.swmm_crs = crs
        
    def load_flooding_data(self, gpkg_path):
        """
        Loads the flooding nodes GPKG for fast data retrieval.

        Parameters
        ----------
        gpkg_path : str
            Path to the GeoPackage file containing flooding node information.

        Returns
        -------
        None

        Examples
        --------
        >>> modifier = SWMMModifier("model.inp")
        >>> modifier.load_flooding_data("00_flooding_nodes.gpkg")
        """
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

    def set_report_step(self, time_step="00:00:30"):
        """
        Updates the REPORT_STEP in the [OPTIONS] section.
        Essential for capturing flash flood peaks in the hydrograph.
        """
        section = "[OPTIONS]"
        header_idx = -1
        
        # Find section
        for i, line in enumerate(self.lines):
            if line.strip().upper() == section:
                header_idx = i
                break
        
        if header_idx == -1:
            print("Warning: [OPTIONS] section not found.")
            return

        # Scan for REPORT_STEP
        found = False
        current_idx = header_idx + 1
        while current_idx < len(self.lines):
            line = self.lines[current_idx]
            if line.strip().startswith("["):
                break # New section
                
            parts = line.split()
            if len(parts) > 1 and parts[0].upper() == "REPORT_STEP":
                # Replace line
                # Preserve whitespace structure roughly
                self.lines[current_idx] = f"REPORT_STEP          {time_step}\n"
                print(f"  [Info] Updated REPORT_STEP to {time_step}")
                found = True
                break
            current_idx += 1
            
        if not found:
            # Insert if not found
            self.lines.insert(header_idx + 1, f"REPORT_STEP          {time_step}\n")

    def set_end_time(self, end_time="03:30:00"):
        """
        Updates the END_TIME in the [OPTIONS] section.
        Controls simulation duration.
        
        Parameters
        ----------
        end_time : str
            The end time in HH:MM:SS format.
        """
        section = "[OPTIONS]"
        header_idx = -1
        
        # Find section
        for i, line in enumerate(self.lines):
            if line.strip().upper() == section:
                header_idx = i
                break
        
        if header_idx == -1:
            print("Warning: [OPTIONS] section not found.")
            return

        # Scan for END_TIME
        found = False
        current_idx = header_idx + 1
        while current_idx < len(self.lines):
            line = self.lines[current_idx]
            if line.strip().startswith("["):
                break # New section
                
            parts = line.split()
            if len(parts) > 1 and parts[0].upper() == "END_TIME":
                # Replace line
                self.lines[current_idx] = f"END_TIME             {end_time}\n"
                print(f"  [Info] Updated END_TIME to {end_time}")
                found = True
                break
            current_idx += 1
            
        if not found:
            # Insert if not found
            self.lines.insert(header_idx + 1, f"END_TIME             {end_time}\n")

    def add_storage_unit(self, name, area, max_depth, terrain_elev=None, node_invert=None, invert_elev=None):
        """
        Adds a FUNCTIONAL storage unit (Constant Area) to the SWMM model.

        Parameters
        ----------
        name : str
            The name/ID of the storage unit.
        area : float
            The constant surface area of the storage unit.
        max_depth : float
            The maximum depth of the storage unit.
        terrain_elev : float, optional
            The terrain elevation at the storage unit location. If provided,
            Invert Elevation = terrain_elev - max_depth.
        node_invert : float, optional
            The invert elevation of the connecting node. Used if `terrain_elev`
            is not provided. Invert Elevation = node_invert - max_depth.
        invert_elev : float, optional
            Directly sets the Invert Elevation (Bottom) of the tank. 
            Overrides other elevation parameters if provided.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no elevation parameter is provided.
        """
        if invert_elev is not None:
             elev = float(invert_elev)
             print(f"  [Info] Tank {name}: Using direct Invert Elevation {elev:.2f}")
        elif terrain_elev is not None:
            elev = terrain_elev - max_depth
            print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = Terrain {terrain_elev:.2f} - Depth {max_depth:.2f}")
        elif node_invert is not None:
            elev = node_invert - max_depth
            print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = NodeInvert {node_invert:.2f} - Depth {max_depth:.2f}")
        else:
            raise ValueError(f"Tank {name}: Must provide either invert_elev, terrain_elev or node_invert to calculate tank elevation.")

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
        Adds a conduit definition to the model.

        Parameters
        ----------
        name : str
            The name/ID of the conduit.
        from_node : str
            The ID of the inlet node.
        to_node : str
            The ID of the outlet node.
        length : float
            The length of the conduit.
        roughness : float, optional
            Manning's roughness coefficient (default is 0.015).
        inlet_offset : float, optional
            Depth of the conduit inlet above the node invert (default is 0.0).
        outlet_offset : float, optional
            Depth of the conduit outlet above the node invert (default is 0.0).

        Returns
        -------
        None

        Examples
        --------
        >>> modifier.add_conduit(
        ...     name="C1",
        ...     from_node="NodeA",
        ...     to_node="NodeB",
        ...     length=100.0,
        ...     roughness=0.013
        ... )
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

    def add_weir(self, name, from_node, to_node, weir_type="SIDEFLOW", crest_height=0.5, 
                 discharge_coeff=1.84, width=2.0, end_contractions=0, flap_gate=False):
        """
        Adds a weir to the model for flow diversion.

        Parameters
        ----------
        name : str
            The ID of the weir.
        from_node : str
            The ID of the inlet node (where water comes from).
        to_node : str  
            The ID of the outlet node (where water goes to, e.g., a tank or derivation).
        weir_type : str, optional
            Type of weir: 'TRANSVERSE', 'SIDEFLOW', 'V-NOTCH', 'TRAPEZOIDAL' (default 'SIDEFLOW').
        crest_height : float, optional
            Height of the weir crest above the inlet node invert (default 0.5m).
        discharge_coeff : float, optional
            Weir discharge coefficient (default 1.84 for rectangular).
        width : float, optional
            Crest width/length of the weir (default 2.0m).
        end_contractions : int, optional
            Number of end contractions (0, 1, or 2) (default 0).
        flap_gate : bool, optional
            Whether the weir has a flap gate (default False).

        Returns
        -------
        None
        """
        flap_str = "YES" if flap_gate else "NO"
        
        # [WEIRS] format for SWMM 5.2:
        # Name  FromNode  ToNode  Type  CrestHt  Cd  Gated  EC  Cd2  Surcharge
        # For SIDEFLOW: EC (end contractions) and Cd2 are optional, use empty fields
        new_line = (
            f"{name:<16} "
            f"{from_node:<16} "
            f"{to_node:<16} "
            f"{weir_type:<12} "
            f"{crest_height:<10.2f} "
            f"{discharge_coeff:<10.3f} "
            f"{flap_str:<8}\n"
        )
        
        section = "WEIRS"
        idx = self._find_last_line_of_section(section)
        if idx == -1:
            headers = ";;Name           From Node        To Node          Type         CrestHt    Qcoeff     Gated"
            idx = self._create_section(section, headers)
        
        self.lines.insert(idx, new_line)
        print(f"  [Weir] Added: {name} from {from_node} to {to_node} (Crest: {crest_height:.2f}m, Width: {width:.2f}m)")
        
        # Also add the XSECTION for the weir (required)
        # Weir XSection: Link Shape Height TopWidth SideSlope1 SideSlope2 Barrels Culvert
        xsect_line = f"{name:<16} RECT_OPEN    {1.0:<10.2f} {width:<10.2f} 0          0          1          \n"
        
        xsect_section = "XSECTIONS"
        xsect_idx = self._find_last_line_of_section(xsect_section)
        if xsect_idx != -1:
            self.lines.insert(xsect_idx, xsect_line)

    def add_rect_closed_xsection(self, link_name, height):
        """
        Adds a RECT_CLOSED cross-section to a link.

        The width is automatically set to 2.0 * height based on user requirements.

        Parameters
        ----------
        link_name : str
            The ID of the link (conduit).
        height : float
            The height of the rectangular section.

        Returns
        -------
        None

        Examples
        --------
        >>> modifier.add_rect_closed_xsection(link_name="C1", height=1.5)
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
        Adds a coordinate entry for a node.

        Parameters
        ----------
        name : str
            The ID of the node.
        x : float
            The X-coordinate (Easting).
        y : float
            The Y-coordinate (Northing).

        Returns
        -------
        None

        Examples
        --------
        >>> modifier.add_coordinate("NodeX", 500000.0, 9800000.0)
        """
        new_line = f"{name:<16} {x:<16.3f} {y:<16.3f}\n"
        
        section = "COORDINATES"
        idx = self._find_last_line_of_section(section)
        if idx != -1:
             self.lines.insert(idx, new_line)

    def get_node_coords(self, node_id):
        """
        Retrieves the (x, y) coordinates for a given node.

        It prioritizes data from the loaded flooding GPKG. If not found, it falls
        back to the loaded swmmio model.

        Parameters
        ----------
        node_id : str
            The ID of the node to lookup.

        Returns
        -------
        tuple of (float, float)
            The (x, y) coordinates of the node. Returns (0.0, 0.0) if not found.

        Examples
        --------
        >>> x, y = modifier.get_node_coords("Node123")
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

    def place_node_at_distance(self, new_node_id, ref_node_id, dist_x=0.0, dist_y=0.0):
        """
        Places a new node at a specific distance offset from a reference node.

        Calculates the new coordinates based on the reference node's position and the
        provided offsets, then adds the coordinate entry to the SWMM model.

        Parameters
        ----------
        new_node_id : str
            The ID of the new node to place.
        ref_node_id : str
            The ID of the existing reference node.
        dist_x : float, optional
            The distance to move in the X direction (default is 0.0).
        dist_y : float, optional
            The distance to move in the Y direction (default is 0.0).

        Returns
        -------
        tuple of (float, float)
            The calculated (x, y) coordinates of the new node.

        Examples
        --------
        >>> # Place 'NewNode' 5 meters East and 2 meters North of 'RefNode'
        >>> modifier.place_node_at_distance("NewNode", "RefNode", dist_x=5.0, dist_y=2.0)
        """
        x, y = self.get_node_coords(ref_node_id)
        
        new_x = x + dist_x
        new_y = y + dist_y
        
        print(f"Placing {new_node_id} at ({new_x:.3f}, {new_y:.3f}) relative to move {ref_node_id} ({x:.3f}, {y:.3f})")
        self.add_coordinate(new_node_id, new_x, new_y)
        return new_x, new_y

    def get_node_invert(self, node_id):
        """
        Retrieves the Invert Elevation for a given node.

        Prioritizes data from the loaded flooding GPKG. If not found, it falls
        back to the loaded swmmio model.

        Parameters
        ----------
        node_id : str
            The ID of the node.

        Returns
        -------
        float
            The invert elevation of the node. Returns 0.0 if not found.

        Examples
        --------
        >>> invert = modifier.get_node_invert("Node123")
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

        This method reads a pipeline defined in a GeoPackage, adds the corresponding
        junctions, conduits, and cross-sections to the SWMM model.
        - It replaces the FINAL Node of the pipeline with a Storage Unit (Tank).
        - It creates a connection from the START Node to an existing `connection_id` if provided.

        Parameters
        ----------
        gpkg_path : str
            Path to the GPKG file defining the pipeline.
        connection_id : str, optional
            The ID of an existing node in the model to connect the start of the pipeline to.
            If provided, the start node of the pipeline is mapped to this ID.

        Returns
        -------
        None

        Examples
        --------
        >>> modifier.add_pipeline_from_gpkg("pipeline.gpkg", connection_id="ExistingNodeID")
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
        """
        Adds a junction to the model.

        Parameters
        ----------
        name : str
            The ID of the junction.
        elev : float
            The invert elevation of the junction.
        max_depth : float, optional
            The maximum depth of the junction (default is 0).

        Returns
        -------
        None
        """
        new_line = f"{name:<16} {elev:<10.2f} {max_depth:<10.2f} 0.0        0.0        0.0\n"
        section = "JUNCTIONS"
        idx = self._find_last_line_of_section(section)
        if idx == -1:
            self._create_section(section, "Name           Elev.      MaxDepth   InitDepth  SurDepth   Aponded")
            idx = self._find_last_line_of_section(section)
        self.lines.insert(idx, new_line)

    def add_xsection(self, link, shape, geoms):
        """
        Generic method to add a cross-section to a link.

        Parameters
        ----------
        link : str
            The ID of the link.
        shape : str
            The SWMM shape type (e.g., 'RECT_CLOSED', 'CIRCULAR').
        geoms : list of float
            A list of geometry parameters [Geom1, Geom2, Geom3, Geom4].
            Specific meanings depend on the shape.

        Returns
        -------
        None
        """
        # SWMM format: Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert
        # Format string with fixed width
        geom_str = f"{geoms[0]:<10.2f} {geoms[1]:<10.2f} {geoms[2]:<10.2f} {geoms[3]:<10.2f}"
        new_line = f"{link:<16} {shape:<12} {geom_str} 1          0\n"
        
        section = "XSECTIONS"
        idx = self._find_last_line_of_section(section)
        if idx != -1: # Section usually exists
             self.lines.insert(idx, new_line)

    def save(self, output_path):
        """
        Saves the modified SWMM model to a file.

        Parameters
        ----------
        output_path : str
            The path where the .inp file will be saved.

        Returns
        -------
        None
        """
        with open(output_path, 'w', encoding='latin-1') as f:
            f.writelines(self.lines)

    def add_designed_pipeline(self, designed_gdf, connection_map=None):
        """
        Integrates a designed pipeline from rut_03 into the SWMM model.
        
        Parameters
        ----------
        designed_gdf : geopandas.GeoDataFrame
            The output GDF from SewerPipeline.
        connection_map : dict, optional
            A mapping of {OriginalNodeID: FinalNodeID} to connect the new pipes
            to existing model nodes (e.g. { 'R_1.0': 'NodeX', 'R_1.1': 'TK_NodeX' }).
            Nodes in the values of this map are assumed to ALREADY EXIST (or be added separately),
            so no new junction entries will be created for them.
        """
        if SeccionLlena is None:
            print("Cannot generate pipeline: SeccionLlena dependency missing.")
            return

        print(f"Adding designed pipeline ({len(designed_gdf)} links)...")
        connection_map = connection_map or {}
        
        # Identify nodes that should NOT be created (they effectively exist)
        existing_targets = set(connection_map.values())
        
        # Track IDs we have added in this batch to avoid duplicates
        added_nodes = set()
        
        xsections_to_add = []
        
        for idx, row in designed_gdf.iterrows():
            tramo = str(row['Tramo'])
            parts = tramo.split('-')
            
            # Robust split handling
            if len(parts) >= 2:
                u_original = parts[0]
                v_original = parts[-1] # Handle multi-dash names if any? usually simple.
            else:
                print(f"Skipping malformed Tramo: {tramo}")
                continue
                
            # Remap IDs
            u_final = connection_map.get(u_original, u_original)
            v_final = connection_map.get(v_original, v_original)
            
            # --- NODES ---
            # Process U (Start)
            if u_final not in existing_targets and u_final not in added_nodes:
                # Get coords from start of geometry
                if row.geometry.geom_type == 'LineString':
                    pt = row.geometry.coords[0]
                    # Add Junction
                    # ZFI is invert at start of pipe. Node invert should be at least this.
                    z = float(row.ZFI) if hasattr(row, 'ZFI') else 0.0
                    self.add_junction(u_final, z)
                    self.add_coordinate(u_final, pt[0], pt[1])
                    added_nodes.add(u_final)
            
            # Process V (End)
            if v_final not in existing_targets and v_final not in added_nodes:
                if row.geometry.geom_type == 'LineString':
                    pt = row.geometry.coords[-1]
                    z = float(row.ZFF) if hasattr(row, 'ZFF') else 0.0
                    self.add_junction(v_final, z)
                    self.add_coordinate(v_final, pt[0], pt[1])
                    added_nodes.add(v_final)
            
            # --- CONDUIT ---
            # Use Tramo as Name, but maybe prefix to ensure uniqueness?
            # rut_03 names are usually unique per project.
            
            length = float(row.L) if hasattr(row, 'L') else row.geometry.length
            n = float(row.Rug) if hasattr(row, 'Rug') else 0.010
            
            self.add_conduit(tramo, u_final, v_final, length, n)
            
            # --- XSECTION ---
            # Parse 'Seccion' and 'D_int' similar to add_pipeline_from_gpkg
            d_int = str(row.D_int) if hasattr(row, 'D_int') else "0.3"
            seccion_type = str(row.Seccion).lower().strip() if hasattr(row, 'Seccion') else "circular"
            
            # (Reuse logic from add_pipeline_from_gpkg or refactor. duplicating for safety/speed now)
            try:
                geom_arr = SeccionLlena.section_str2float([d_int], return_all=True, sep='x')
                g = geom_arr[0]
                
                shape_swmm = 'CIRCULAR'
                geoms = [0.0]*4
                
                if 'circular' in seccion_type:
                    shape_swmm = 'CIRCULAR'
                    geoms[0] = g[0]
                elif 'rect' in seccion_type and 'closed' in seccion_type: # check rut_03 naming
                     shape_swmm = 'RECT_CLOSED'
                     geoms[0] = g[1] # H
                     geoms[1] = g[0] # W
                else:
                     # Fallback to circular if unknown or add other types
                     # rut_03 usually produces circular for sewers
                     shape_swmm = 'CIRCULAR'
                     geoms[0] = g[0]
                
                xsections_to_add.append({
                    'link': tramo,
                    'shape': shape_swmm,
                    'geoms': geoms
                })
                
            except Exception as e:
                print(f"Error parsing section for {tramo}: {e}")
                # Fallback
                self.add_xsection(tramo, 'CIRCULAR', [0.3, 0, 0, 0])

        # Add all xsections
        for x in xsections_to_add:
            self.add_xsection(x['link'], x['shape'], x['geoms'])

if __name__ == "__main__":
    base_file = "COLEGIO_TR25_v6.inp"
    out_pipe = "COLEGIO_TR25_v6_PIPELINE.inp"

    gpkg_file = r"00_flooding_stats/00_flooding_nodes.gpkg" 
    gpkg_pipeline = r"P0061405_PredioID3.gpkg"
    target_node = "P0061405" 
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
        LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""
    )
    
    
    if os.path.exists(base_file):
        print(f"Reading {base_file}...")
        mod = SWMMModifier(base_file)
        
        # Load GPKG
        if os.path.exists(gpkg_file):
            mod.load_flooding_data(gpkg_file)
        else:
            print(f"Warning: GPKG file {gpkg_file} not found.")


        print(f"Targeting Node: {target_node}")
        # 1. Get Node Data
        node_x, node_y = mod.get_node_coords(target_node)
        node_invert = mod.get_node_invert(target_node)
        
        if os.path.exists(gpkg_pipeline):
            mod_pipe = SWMMModifier(base_file, crs=source_crs)

            if os.path.exists(gpkg_file):
                 mod_pipe.load_flooding_data(gpkg_file)
            
            mod_pipe.add_pipeline_from_gpkg(gpkg_pipeline, connection_id="P0061405")
            

            mod_pipe.save(out_pipe)
            print(f"Pipeline File saved: {out_pipe}")
        else:
            print(f"Pipeline GPKG {gpkg_pipeline} not found.")

    else:
        print(f"Base file {base_file} not found.")
