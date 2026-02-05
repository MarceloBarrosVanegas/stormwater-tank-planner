import os
import geopandas as gpd
from pathlib import Path
import json
import numpy as np

import config
config.setup_sys_path()

from rut_06_pipe_sizing import SeccionLlena


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

        self.swmm_crs = crs
        
        self.seccion_type_map = {'circular': 'CIRCULAR', 'rectangular': 'RECT_CLOSED'}
        


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
                # print(f"  [Info] Updated REPORT_STEP to {time_step}")
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
                # print(f"  [Info] Updated END_TIME to {end_time}")
                found = True
                break
            current_idx += 1
            
        if not found:
            # Insert if not found
            self.lines.insert(header_idx + 1, f"END_TIME             {end_time}\n")

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
             # print(f"  [Info] Tank {name}: Using direct Invert Elevation {elev:.2f}")
        elif terrain_elev is not None:
            elev = terrain_elev - max_depth
            # print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = Terrain {terrain_elev:.2f} - Depth {max_depth:.2f}")
        elif node_invert is not None:
            elev = node_invert - max_depth
            # print(f"  [Info] Tank {name}: Calc Invert {elev:.2f} = NodeInvert {node_invert:.2f} - Depth {max_depth:.2f}")
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


    def modify_xsections(self, links, shapes, geoms):
        """
        Batch modifies cross-sections for multiple links using parallel lists.
        Ideal for working with GeoDataFrames columns.
        
        Parameters
        ----------
        links : list of str
            List of Link IDs.
        shapes : list of str
            List of SWMM shape types (e.g., 'CIRCULAR', 'RECT_CLOSED').
        geoms : list of list of float or list of numpy arrays
            List of geometry parameter lists.
            Each element is a list or array [Geom1, Geom2, Geom3, Geom4].
            
        Returns
        -------
        int
            Number of cross-sections modified.
            
        Examples
        --------
        >>> links = ['T-01', 'T-02']
        >>> shapes = ['CIRCULAR', 'RECT_CLOSED']
        >>> geoms = [[1.2, 0, 0, 0], [1.5, 2.0, 0, 0]]
        >>> modifier.modify_xsections(links, shapes, geoms)
        """
        # Create a lookup dictionary: link_id -> (shape, geom_list)
        # Zip to ensure we handle them in sync
        updates = {}
        if len(links) != len(shapes) or len(links) != len(geoms):
            raise ValueError("Input lists (links, shapes, geoms) must have the same length.")
    
        for lk, sh, gm in zip(links, shapes, geoms):
            updates[lk] = {'shape': sh, 'geoms': gm}
    
        modified_count = 0
        
        section = "XSECTIONS"
        header = f"[{section}]"
        
        # 1. Find section start
        header_idx = -1
        for i, line in enumerate(self.lines):
            if line.strip().upper() == header:
                header_idx = i
                break
                
        if header_idx == -1:
            print(f"Warning: [{section}] section not found.")
            return 0
            
        # 2. Iterate through the section
        current_idx = header_idx + 1
        while current_idx < len(self.lines):
            line = self.lines[current_idx]
            stripped = line.strip()
            
            # Stop if next section starts
            if stripped.startswith("["):
                break
                
            # Skip comments/empty
            if not stripped or stripped.startswith(";"):
                current_idx += 1
                continue
                
            parts = stripped.split()
            if len(parts) > 0:
                link_id = parts[0]
                
                if link_id in updates:
                    # Retrieve new data
                    new_data = updates[link_id]
                    shape = new_data['shape']
                    geoms = new_data['geoms']
                    
                    # Convert to list if it's a numpy array
                    if hasattr(geoms, 'tolist'):
                        geoms = geoms.tolist()
                    else:
                        geoms = list(geoms)
                    
                    # Format geometry string (Geom1 Geom2 Geom3 Geom4)
                    # SWMM expects 4 geoms usually, fill with 0 if fewer are provided
                    g = geoms + [0.0] * (4 - len(geoms))
                    geom_str = f"{g[0]:<10.2f} {g[1]:<10.2f} {g[2]:<10.2f} {g[3]:<10.2f}"
                    
                    # Preserving Barrels (usually parts[6] if existing)
                    # Standard (parts): Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert
                    barrels = "1"
                    if len(parts) >= 7:
                        barrels = parts[6]
                    
                    new_line = f"{link_id:<16} {shape:<12} {geom_str} {barrels:<10} 0\n"
                    
                    self.lines[current_idx] = new_line
                    modified_count += 1
            
            current_idx += 1
            
        return modified_count
    
    def modify_storage_area(self, storage_id, new_area):
        """
        Modifies the area of an existing storage unit.
        
        This method iterates through the file line-by-line (consistent with the class design)
        to find the [STORAGE] section and the specific storage unit, then updates its
        area parameter while preserving other properties if possible.
        
        Parameters
        ----------
        storage_id : str
            The ID of the storage unit to modify.
        new_area : float
            The new constant area for the storage unit.
            
        Returns
        -------
        bool
            True if the storage unit was found and modified, False otherwise.
        """
        section = "STORAGE"
        header = f"[{section}]"
        
        # Find section start
        header_idx = -1
        for i, line in enumerate(self.lines):
            if line.strip().upper() == header:
                header_idx = i
                break
                
        if header_idx == -1:
            print(f"Warning: [{section}] section not found.")
            return False
            
        # Scan for the storage unit
        current_idx = header_idx + 1
        found = False
        
        while current_idx < len(self.lines):
            line = self.lines[current_idx]
            stripped = line.strip()
            
            # Stop if next section starts
            if stripped.startswith("["):
                break
                
            # Skip comments or empty lines
            if not stripped or stripped.startswith(";"):
                current_idx += 1
                continue
                
            parts = stripped.split()
            if len(parts) > 0 and parts[0] == storage_id:
                # Found the tank
                # Standard format: Name Elev MaxDepth InitDepth Shape Params...
                
                # Check for enough parts to at least identify shape
                if len(parts) >= 5:
                    shape = parts[4].upper()
                    
                    if shape == "FUNCTIONAL":
                        # Params are A B C. We replace A with new_area.
                        
                        # Existing values
                        name = parts[0]
                        elev = parts[1]
                        max_d = parts[2]
                        init_d = parts[3]
                        # shape is FUNCTIONAL
                        
                        # Use default 0 0 for B and C if missing
                        b_val = parts[6] if len(parts) > 6 else "0"
                        c_val = parts[7] if len(parts) > 7 else "0"
                        
                        # Remainder (e.g. PondedArea, EvapFrac)
                        remainder = " ".join(parts[8:]) if len(parts) > 8 else "0 0"
                        
                        # Construct new line
                        params_str = f"{new_area:.2f} {b_val} {c_val}"
                        
                        new_line = (
                            f"{name:<16} "
                            f"{elev:<10} "
                            f"{max_d:<10} "
                            f"{init_d:<10} "
                            f"FUNCTIONAL "
                            f"{params_str:<20} "
                            f"{remainder}\n"
                        )
                        
                        self.lines[current_idx] = new_line
                        print(f"  [Info] Modified Storage {storage_id} Area to {new_area:.2f}")
                        return True
                    else:
                        print(f"Warning: Storage {storage_id} is {shape}, not FUNCTIONAL. Area modification not supported yet.")
                        return False
                else:
                    print(f"Warning: Storage line for {storage_id} is malformed or too short.")
                    return False
            
            current_idx += 1
            
        if not found:
            print(f"Warning: Storage unit {storage_id} not found.")
            return False

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


    def get_node_invert(self, node_id):
        """
        Retrieves the Invert Elevation for a given node.
        Prioritizes GPKG/designed_gdf -> self.lines parsing (for newly added nodes).
        """
        # 1. Try internal dataframes first
        if self.existing_nodes_df is not None:
            if node_id in self.existing_nodes_df.index:
                row = self.existing_nodes_df.loc[node_id]
                return float(row.InvertElev)
            
        if self.designed_gdf is not None:
             # Check if node_id matches Pozo (Inlet) or end of a Tramo (Outlet)
             if 'Pozo' in self.designed_gdf.columns:
                 matches = self.designed_gdf[self.designed_gdf['Pozo'] == node_id]
                 if not matches.empty:
                     return float(matches.iloc[0]['ZFI'])
             
             if 'Tramo' in self.designed_gdf.columns:
                matches = self.designed_gdf[self.designed_gdf['Tramo'].astype(str).str.endswith(f"-{node_id}")]
                if not matches.empty:
                    return float(matches.iloc[0]['ZFF'])

        # 2. Search in self.lines (Storage and Junctions added THIS session)
        for section in ["STORAGE", "JUNCTIONS"]:
            header = f"[{section}]"
            in_section = False
            for line in self.lines:
                stripped = line.strip()
                if stripped.upper() == header:
                    in_section = True
                    continue
                if in_section:
                    if stripped.startswith("["): break
                    parts = stripped.split()
                    if len(parts) > 1 and parts[0] == node_id:
                        try:
                            return float(parts[1])
                        except ValueError:
                            continue
        
        return 0.0

    def get_node_depth(self, node_id):
        """
        Retrieves the 'node_depth' (Pozo Depth) for a given node.
        Strictly uses flooding_gdf (Input GPKG) or session edits.
        """
        # 1. Try GPKG / dataframes
        if self.existing_nodes_df is not None:
             if node_id in self.existing_nodes_df.index:
                row = self.existing_nodes_df.loc[node_id]
                return float(row.MaxDepth)
                
        if self.designed_gdf is not None:
             if 'Pozo' in self.designed_gdf.columns:
                 matches = self.designed_gdf[self.designed_gdf['Pozo'] == node_id]
                 if not matches.empty:
                     return float(matches.iloc[0].get('HI', 0.0))
             
             if 'Tramo' in self.designed_gdf.columns:
                matches = self.designed_gdf[self.designed_gdf['Tramo'].astype(str).str.endswith(f"-{node_id}")]
                if not matches.empty:
                    return float(matches.iloc[0].get('HF', 0.0))

        # 2. Search in self.lines
        for section in ["STORAGE", "JUNCTIONS"]:
            header = f"[{section}]"
            in_section = False
            for line in self.lines:
                stripped = line.strip()
                if stripped.upper() == header:
                    in_section = True
                    continue
                if in_section:
                    if stripped.startswith("["): break
                    parts = stripped.split()
                    if len(parts) > 2 and parts[0] == node_id:
                        try:
                            # Search format matches add_storage_unit and add_junction
                            return float(parts[2])
                        except ValueError:
                            continue

        return 0.0

    def get_existing_conduit_geom1(self, node_id):
        """
        Finds an existing conduit connected to node_id and returns its Geom1 (Height).
        Searches in [CONDUITS] and [XSECTIONS].
        """
        # 1. Find a conduit name connected to this node
        conduit_name = None
        section = "[CONDUITS]"
        in_section = False
        for line in self.lines:
            stripped = line.strip()
            if stripped.upper() == section:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("["): break
                parts = stripped.split()
                if len(parts) > 2:
                    # parts[0]=name, parts[1]=inlet, parts[2]=outlet
                    if parts[1] == node_id or parts[2] == node_id:
                        conduit_name = parts[0]
                        break
        
        if not conduit_name:
            return 0.0

        # 2. Find its Geom1 in [XSECTIONS]
        section = "[XSECTIONS]"
        in_section = False
        for line in self.lines:
            stripped = line.strip()
            if stripped.upper() == section:
                in_section = True
                continue
            if in_section:
                if stripped.startswith("["): break
                parts = stripped.split()
                if len(parts) > 2 and parts[0] == conduit_name:
                    try:
                        # parts[0]=link, parts[1]=shape, parts[2]=geom1
                        return float(parts[2])
                    except ValueError:
                        continue
        
        return 0.0

    def modify_node_invert(self, node_id, new_invert):
        """
        Modifies the invert elevation of an existing junction.
        """
        section = "JUNCTIONS"
        header_idx = -1
        for i, line in enumerate(self.lines):
            if line.strip().upper() == f"[{section}]":
                header_idx = i
                break
        
        if header_idx == -1:
            print(f"Warning: [{section}] section not found.")
            return

        current_idx = header_idx + 1
        found = False
        while current_idx < len(self.lines):
            line = self.lines[current_idx]
            if line.strip().startswith("["):
                break
            
            parts = line.split()
            if len(parts) > 1 and parts[0] == node_id:
                # Reconstruct line with new invert
                try:
                    name = parts[0]
                    max_d = parts[2] if len(parts) > 2 else "0"
                    init_d = parts[3] if len(parts) > 3 else "0"
                    sur_d = parts[4] if len(parts) > 4 else "0"
                    apond = parts[5] if len(parts) > 5 else "0"
                    
                    new_line = f"{name:<16} {new_invert:<10.2f} {max_d:<10} {init_d:<10} {sur_d:<10} {apond:<10}\n"
                    self.lines[current_idx] = new_line
                    # print(f"  [Info] Modified Node {node_id} Invert to {new_invert:.2f}")
                    found = True
                except Exception as e:
                    print(f"  [Error] Could not parse Junction line: {line} -> {e}")
                break
            current_idx += 1
            
        if not found:
            print(f"Warning: Node {node_id} not found in [JUNCTIONS] to modify.")

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

    def add_derivation_to_model(self, last_design_gdf, case_dir, solution_name):
        """
        Adds derivation pipelines from the last design GeoDataFrame to the SWMM model.
        The last node of each ramal is created as a storage tank or junction based on metadata.
        """
        # --- Set Simulation Times ---
        report_step = f"00:{config.REPORT_STEP_MINUTES:02d}:00"
        self.set_report_step(report_step)
        hours = int(config.ITZI_SIMULATION_DURATION_HOURS)
        minutes = int((config.ITZI_SIMULATION_DURATION_HOURS - hours) * 60)
        self.set_end_time(f"{hours:02d}:{minutes:02d}:00")
        
        added_nodes = set()
        tank_volume_dict = {}
        
        # --- Process each ramal (branch) ---
        for ramal, ramal_df in last_design_gdf.groupby('Ramal'):
            ramal_geom = SeccionLlena.section_str2float(
                ramal_df['D_ext'],
                return_all=True,
                sep='x'
            )
            ramal_seccion = ramal_df['Seccion'].map(self.seccion_type_map)
    
            # Extract metadata from first row
            obs_parts = ramal_df['Obs'].iloc[0].split('|')
            node_metadata = json.loads(obs_parts[1])
            
            # Extract target info from metadata
            target_type = node_metadata.get('target_type', 'node')
            target_id = node_metadata.get('target_id')
            target_x = node_metadata.get('target_x')
            target_y = node_metadata.get('target_y')
    
            previous_node = None
            total_rows = len(ramal_df)
            # print(f"  [SWMM] Processing ramal {ramal}: {total_rows} rows, target_type={target_type}, node_id={node_metadata.get('node_id')}")
    
            # --- Process each segment in the ramal ---
            for row_number, row in enumerate(ramal_df.itertuples()):
                # Determine if this is the last node
                is_last_node = (row_number == total_rows - 1)
                node_end = row.Pozo
                node_end_elev = row.ZFF
                node_end_depth = row.HF
                manning_roughness = row.Rug
                length = row.L
                
                if target_type == 'tank':
                    outlet_offset = row.SALTO if not is_last_node else config.TANK_DEPTH_M
                else:
                    outlet_offset = row.SALTO

                # Determine start node and inlet offset
                if row_number == 0:
                    node_start = node_metadata['node_id']
                    existing_height = self.get_existing_conduit_geom1(node_start)
                    inlet_offset = round(config.CAPACITY_MAX_HD * existing_height, 2)
                    tramo = f"{node_start}-{node_end}"
                else:
                    node_start = previous_node
                    inlet_offset = 0
                    tramo = row.Tramo
    
                # Get cross-section data
                shape_swmm = ramal_seccion.iloc[row_number]
                geoms = ramal_geom[row_number]
                geoms = np.nan_to_num(geoms, nan=0.0)
    
                # Add node (Tank if last and target_type='tank', Junction otherwise)
                if is_last_node:
                    if target_type == 'tank':
                        # Create tank with target_id
                        tank_name = f"tank_{target_id}"
                        tank_volume = node_metadata['target_total_volume']
                        
                        if tank_name not in added_nodes:
                            self.add_storage_unit(
                                name=tank_name,
                                area=(tank_volume / config.TANK_DEPTH_M) * config.TANK_VOLUME_SAFETY_FACTOR + config.TANK_OCCUPATION_FACTOR,
                                max_depth=config.TANK_DEPTH_M,
                                invert_elev=node_end_elev - config.TANK_DEPTH_M
                            )
                            self.add_coordinate(
                                name=tank_name,
                                x=target_x,
                                y=target_y
                            )
                            added_nodes.add(tank_name)
                            tank_volume_dict[tank_name] = tank_volume
                        else:
                            tank_volume_dict[tank_name] = tank_volume + tank_volume_dict[tank_name]
                            self.modify_storage_area( tank_name, tank_volume_dict[tank_name])
                            
                        final_node = tank_name
                        # print(f"  [Derivation] Created Tank {tank_name} at target {target_id}")
                    else:
                        # target_type == 'node': Connect to existing pipeline node
                        final_node = str(target_id)
                        
                        # Check if this node already exists in the model or added_nodes
                        if final_node not in added_nodes:
                            # Node doesn't exist yet - we need to create it
                            # Look up the node info in last_design_gdf
                            node_match = last_design_gdf[last_design_gdf['Pozo'].astype(str) == final_node]
                            
                            if not node_match.empty:
                                # Get elevation and depth from the matched row
                                node_elev = float(node_match.iloc[0]['ZFF'])
                                node_depth = float(node_match.iloc[0]['HF'])
                                node_x_coord = float(node_match.iloc[0].geometry.coords[-1][0])
                                node_y_coord = float(node_match.iloc[0].geometry.coords[-1][1])
                                
                                # Create the junction
                                self.add_junction(
                                    name=final_node,
                                    elev=node_elev,
                                    max_depth=node_depth
                                )
                                self.add_coordinate(
                                    name=final_node,
                                    x=node_x_coord,
                                    y=node_y_coord
                                )
                                added_nodes.add(final_node)
                                print(f"  [Derivation] Created Junction {final_node} for pipeline connection (elev: {node_elev:.2f})")
                            else:
                                # Fallback: use metadata from this ramal if node not found
                                # This shouldn't happen normally, but prevents crash
                                print(f"  [Warning] Node {final_node} not found in last_design_gdf, using target coords")
                                self.add_junction(
                                    name=final_node,
                                    elev=node_end_elev,  # Use current row's ZFF as approximation
                                    max_depth=node_end_depth
                                )
                                self.add_coordinate(
                                    name=final_node,
                                    x=target_x,
                                    y=target_y
                                )
                                added_nodes.add(final_node)
                    
                    # Update tramo to connect to final node
                    tramo = f"{node_start}-{final_node}"
                else:
                    # Intermediate node - assign final_node regardless
                    final_node = node_end
                    
                    if node_end not in added_nodes:
                        # Node not yet added - create junction
                        self.add_junction(
                            name=node_end,
                            elev=node_end_elev,
                            max_depth=node_end_depth
                        )
                        self.add_coordinate(
                            name=node_end,
                            x=row.X,
                            y=row.Y
                        )
                        added_nodes.add(node_end)  # Mark as added to prevent duplicates
        
                # Add conduit
                self.add_conduit(
                    name=tramo,
                    from_node=node_start,
                    to_node=final_node,
                    length=length,
                    roughness=manning_roughness,
                    inlet_offset=inlet_offset,
                    outlet_offset=outlet_offset
                )
    
                # Add cross-section
                self.add_xsection(
                    link=tramo,
                    shape=shape_swmm,
                    geoms=geoms
                )
    
                # Update for next iteration
                previous_node = final_node if is_last_node else node_end

        final_inp_path = case_dir / f"model_{solution_name}.inp"
        self.save(str(final_inp_path))
        

        # Clean all previous .out files in the directory
        inp_dir = Path(final_inp_path).parent
        for out_file in inp_dir.glob('*.out'):
            try:
                os.remove(out_file)
            except OSError:
                pass
        
        return final_inp_path
        


if __name__ == "__main__":
    # =========================================================================
    # DEBUG: Iteración 20 - Analizar por qué no existe ramal 20
    # =========================================================================
    inp_file = "COLEGIO_TR25_v6.inp"
    
    # CAMBIAR ESTE PATH al GPKG de la iteración que falló
    gpkg_file = r'C:\Users\chelo\OneDrive\SANTA_ISABEL\00_tanque_tormenta\codigos\optimization_results\Seq_Iter_20\Seq_Iter_20.gpkg'
    
    if not os.path.exists(gpkg_file):
        print(f"[Error] No existe el archivo: {gpkg_file}")
        print("Cambia gpkg_file al path correcto del GPKG de iteración 20")
        exit(1)
    
    last_design_gdf = gpd.read_file(gpkg_file)
    
    # =========================================================================
    # ANÁLISIS: Ver qué ramales hay y cuáles son sus propiedades
    # =========================================================================
    print("\n" + "="*80)
    print("ANÁLISIS DE RAMALES EN EL GPKG")
    print("="*80)
    
    ramales_unicos = last_design_gdf['Ramal'].unique()
    print(f"Ramales únicos encontrados: {sorted(ramales_unicos, key=lambda x: (x is None, str(x)))}")
    print(f"Total ramales: {len(ramales_unicos)}")
    
    # Buscar ramal 20 específicamente
    ramal_20 = last_design_gdf[last_design_gdf['Ramal'] == '20']
    print(f"\n¿Existe ramal '20'? {len(ramal_20) > 0}")
    if len(ramal_20) > 0:
        print(f"Filas de ramal 20: {len(ramal_20)}")
        print(ramal_20[['Ramal', 'Tramo', 'Pozo', 'Obs']].head(10).to_string())
    else:
        print("  -> Ramal 20 NO existe en el GPKG!")
    
    # Análisis de Obs para ver node_id
    print("\n" + "-"*80)
    print("ANÁLISIS: node_id por ramal")
    print("-"*80)
    for ramal in sorted(ramales_unicos, key=lambda x: (x is None, str(x))):
        ramal_df = last_design_gdf[last_design_gdf['Ramal'] == ramal]
        try:
            obs_parts = ramal_df['Obs'].iloc[0].split('|')
            node_metadata = json.loads(obs_parts[1])
            node_id = node_metadata.get('node_id', 'N/A')
            target_type = node_metadata.get('target_type', 'N/A')
            print(f"  Ramal {ramal}: node_id={node_id}, target_type={target_type}, filas={len(ramal_df)}")
        except Exception as e:
            print(f"  Ramal {ramal}: Error parseando Obs - {e}")
    
    print("\n" + "="*80)
    print("Ejecutando add_derivation_to_model...")
    print("="*80 + "\n")
    
    case_dir = Path(os.getcwd())
    solution_name = "debug_iter_20"
    
    swmm_modifier = SWMMModifier(inp_file=inp_file, crs=config.PROJECT_CRS)
    swmm_modifier.add_derivation_to_model(last_design_gdf, case_dir, solution_name)
    
    print("\n[Done] Revisar el archivo model_debug_iter_20.inp generado")
