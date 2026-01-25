import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv(override=False) 

class GraphVisualizer:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    @traceable(name="Generate 2D Graph", run_type="tool", save_result=True, use_cache=True)
    def generate_2d_html(self, output_path: str = "graph_2d.html"):
        """
        Creates an interactive 2D physics graph using PyVis.
        Best for reading entities and relations.
        """
        print(f"Generating 2D Graph: {output_path}")
        
        # 1. Convert to PyVis (needs a specific format)
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        
        # 2. Add Nodes with Metadata colors
        for node, attrs in self.graph.nodes(data=True):
            # Color code by domain if available
            color = "#00ff41" # Matrix Green default
            if attrs.get("domain") == "Mathematics":
                color = "#ff9900" # Orange
            elif attrs.get("domain") == "Computer Science":
                color = "#00ccff" # Blue
                
            net.add_node(node, label=node, title=f"Type: {attrs.get('type')}", color=color)

        # 3. Add Edges with Labels
        for u, v, data in self.graph.edges(data=True):
            # The label is the 'predicate' (e.g., "wrote", "studied")
            label = data.get("relation", "")
            # Tooltip shows the source document
            hover_text = f"Source: {data.get('source')} (ID: {data.get('chunk_id')})"
            
            net.add_edge(u, v, title=hover_text, label=label, color="#aaaaaa")

        # 4. Save
        net.force_atlas_2based()
        net.save_graph(output_path)
        return output_path

    @traceable(name="Generate 3D Graph", run_type="tool", save_result=True, use_cache=True)
    def generate_3d_html(self, output_path: str = "graph_3d.html"):
        """
        Creates a 3D Scatter plot using Plotly.
        Good for 'Wow Factor' but harder to read.
        """
        print(f"Generating 3D Graph: {output_path}")
        
        # 1. Calculate 3D Layout (Positions x, y, z)
        pos = nx.spring_layout(self.graph, dim=3, seed=42)

        # 2. Extract Coordinates for Nodes
        x_nodes, y_nodes, z_nodes = [], [], []
        node_labels = []
        
        for node in self.graph.nodes():
            x_nodes.append(pos[node][0])
            y_nodes.append(pos[node][1])
            z_nodes.append(pos[node][2])
            node_labels.append(node)

        # 3. Create Node Trace (The Dots)
        node_trace = go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(size=5, color='#00ccff'),
            text=node_labels,
            textposition="top center",
            hoverinfo='text'
        )

        # 4. Create Edge Traces (The Lines)
        edge_x, edge_y, edge_z = [], [], []
        
        for u, v in self.graph.edges():
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            
            edge_x.extend([x0, x1, None]) # None breaks the line segment
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='#888', width=1),
            hoverinfo='none'
        )

        # 5. Render
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=dict(text="DataForge Knowledge Graph (3D)", font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            template="plotly_dark"
                        ))
        
        fig.write_html(output_path)
        return output_path