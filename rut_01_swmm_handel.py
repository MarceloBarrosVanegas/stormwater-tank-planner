
import swmmio
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

class SWMMSimplePlot:
    def __init__(self, inp):
        self.model = swmmio.Model(inp)
        self._build_geodataframes()
        self._build_graph()

    # ------------------------------------------------------------------ #
    # 1. Construir GeoDataFrames
    # ------------------------------------------------------------------ #
    def _build_geodataframes(self):
        nodes_df = self.model.nodes()
        if "coords" not in nodes_df.columns:
            raise ValueError("El .inp no trae la columna 'coords'")

        # puntos
        node_geom = [Point(c[0]) for c in nodes_df["coords"]]
        self.gdf_nodes = gpd.GeoDataFrame(nodes_df, geometry=node_geom)
        
        # Ensure the index is used as node identifier
        self.gdf_nodes.reset_index(inplace=True)
        if 'index' in self.gdf_nodes.columns:
            self.gdf_nodes.rename(columns={'index': 'NodeID'}, inplace=True)

        # líneas
        conduits_df = self.model.conduits()
        line_geom = [LineString(c) for c in conduits_df["coords"]]
        self.gdf_edges = gpd.GeoDataFrame(conduits_df, geometry=line_geom)

    # ------------------------------------------------------------------ #
    # 2. Grafo (fixed to use proper node identifiers)
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        # Make sure we're using string identifiers for nodes
        edges_for_graph = self.gdf_edges.copy()
        
        # Convert node identifiers to strings if they aren't already
        edges_for_graph['InletNode'] = edges_for_graph['InletNode'].astype(str)
        edges_for_graph['OutletNode'] = edges_for_graph['OutletNode'].astype(str)
        
        self.G = nx.from_pandas_edgelist(
            edges_for_graph,
            source="InletNode",
            target="OutletNode",
            create_using=nx.DiGraph,
        )
        
        # Also ensure node GDF has string identifiers
        if 'NodeID' in self.gdf_nodes.columns:
            self.gdf_nodes['NodeID'] = self.gdf_nodes['NodeID'].astype(str)
        else:
            # Use index as NodeID if not present
            self.gdf_nodes['NodeID'] = self.gdf_nodes.index.astype(str)

    # ------------------------------------------------------------------ #
    # 3. Helper method to get node identifier
    # ------------------------------------------------------------------ #
    def _get_node_id(self, node):
        """
        Extract the proper node identifier from various input types
        """
        if isinstance(node, str):
            return node
        elif isinstance(node, (int, float)):
            return str(node)
        elif isinstance(node, pd.Series):
            # If it's a pandas Series, try to get the NodeID or index
            if 'NodeID' in node.index:
                return str(node['NodeID'])
            elif hasattr(node, 'name') and node.name is not None:
                return str(node.name)
            else:
                # Try to extract from the string representation
                node_str = str(node)
                if "NodeID=" in node_str:
                    # Extract NodeID from the string representation
                    import re
                    match = re.search(r"NodeID='([^']+)'", node_str)
                    if match:
                        return match.group(1)
                # Fallback to using the first value if it looks like an ID
                return str(node.iloc[0]) if len(node) > 0 else str(node)
        else:
            # For other types, try to convert to string
            return str(node)

    # ------------------------------------------------------------------ #
    # 4. Fixed plot method
    # ------------------------------------------------------------------ #
    def plot(
        self,
        selected_node=None,
        edge_color="lightgrey",
        edge_lw=0.4,
        node_color="black",
        node_size=5,
        upstream_color="red",
        upstream_lw=2.0,
        figsize=(10, 10),
    ):
        ax = plt.figure(figsize=figsize).gca()

        # toda la red
        self.gdf_edges.plot(ax=ax, color=edge_color, linewidth=edge_lw)

        # upstream (fixed)
        if selected_node is not None:
            node_id = self._get_node_id(selected_node)
            
            # Check if node exists in graph
            if node_id not in self.G:
                print(f"Warning: Node {node_id} not found in graph.")
                print(f"Available nodes: {list(self.G.nodes())[:10]}...")  # Show first 10
                return
            
            ups = nx.ancestors(self.G, node_id) | {node_id}
            up_edges = self.gdf_edges[
                self.gdf_edges.InletNode.astype(str).isin(ups) &
                self.gdf_edges.OutletNode.astype(str).isin(ups)
            ]
            up_edges.plot(ax=ax, color=upstream_color, linewidth=upstream_lw)

        # nodos
        self.gdf_nodes.plot(ax=ax, color=node_color, markersize=node_size)

        ax.set_axis_off()
        ax.set_aspect("equal", "box")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # 5. Fixed upstream network method
    # ------------------------------------------------------------------ #
    def get_upstream_network(self, selected_node):
        node_id = self._get_node_id(selected_node)
        
        # Check if node exists in graph
        if node_id not in self.G:
            available_nodes = list(self.G.nodes())
            raise ValueError(f"Node {node_id} not found in graph. "
                           f"Available nodes include: {available_nodes[:10]}")
        
        ups = nx.ancestors(self.G, node_id) | {node_id}

        # filter edges
        up_edges = self.gdf_edges[
            self.gdf_edges.InletNode.astype(str).isin(ups) &
            self.gdf_edges.OutletNode.astype(str).isin(ups)
        ].copy()

        # filter nodes
        up_nodes = self.gdf_nodes[
            self.gdf_nodes['NodeID'].isin(ups)
        ].copy()

        return up_nodes, up_edges

    # ------------------------------------------------------------------ #
    # 6. Utility methods for debugging
    # ------------------------------------------------------------------ #
    def list_nodes(self, n=10):
        """List first n nodes in the graph"""
        return list(self.G.nodes())[:n]
    
    def find_node(self, partial_id):
        """Find nodes containing partial_id in their identifier"""
        return [node for node in self.G.nodes() if partial_id in str(node)]
    
    def get_node_info(self, node_id):
        """Get information about a specific node"""
        node_id = self._get_node_id(node_id)
        if node_id in self.G:
            node_data = self.gdf_nodes[self.gdf_nodes['NodeID'] == node_id]
            return node_data
        else:
            return f"Node {node_id} not found"


# ----------------------------- USO ----------------------------------- #
if __name__ == "__main__":
    inp_file = r"E:\TANQUE MONJAS\MODELO EL COLEGIO Y EL BATAN 2012\ANEXO 4.  SWMM 2012\SWMM 2012\COLEGIO_ACTUAL_TR25_100%.inp"

    net = SWMMSimplePlot(inp_file)

    nodes_gdf, edges_gdf = net.get_upstream_network('P0061408')

    # # now you can save or further manipulate them with geopandas
    # nodes_gdf.to_file('upstream_nodes.shp')
    # edges_gdf.to_file('upstream_edges.shp')

    # 1) solo la red
    # net.plot()

    # 2) con upstream resaltado
    net.plot(selected_node="P0061408")
