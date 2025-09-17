import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math


class LargeTransformerGraphVisualizer:
    """
    Visualizer for large transformer computational graphs with interactive features.
    Designed for models like Gemma-2-2B-IT with many layers and attention heads.
    """

    def __init__(self, graph, num_layers: int, num_attention_heads: int,
                 model_name: str = "Large Transformer"):
        self.graph = graph
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.model_name = model_name

        # Color scheme
        self.colors = {
            'embedding': '#E8F4FD',
            'residual_pre': '#B8E6B8',
            'attention': '#FFD93D',
            'residual_mid': '#FFAAA5',
            'mlp': '#FF8C94',
            'residual_post': '#A8E6CF',
            'output': '#DDA0DD'
        }

        # Edge colors
        self.edge_colors = {
            'residual': 'rgba(0,0,0,0.6)',
            'attention': 'rgba(0,0,255,0.5)',
            'mlp': 'rgba(255,0,0,0.5)',
            'other': 'rgba(128,128,128,0.4)'
        }

    def create_hierarchical_layout(self):
        """Create a hierarchical layout optimized for large graphs."""
        positions = {}
        node_info = {}

        # Vertical spacing between layers
        layer_height = 100

        for node in self.graph.nodes:
            node_info[node] = {
                'type': node.component_type,
                'layer': getattr(node, 'layer', 0),
                'head_idx': getattr(node, 'head_idx', 0),
                'name': node.name
            }

            layer = node_info[node]['layer']

            if node.component_type == 'embedding':
                positions[node] = (0, -50)

            elif node.component_type == 'residual':
                if 'pre' in node.name:
                    positions[node] = (-200, layer * layer_height)
                elif 'mid' in node.name:
                    positions[node] = (0, layer * layer_height)
                else:  # post
                    positions[node] = (200, layer * layer_height)

            elif node.component_type == 'attention':
                # Arrange attention heads in a grid for large numbers
                heads_per_row = min(8, self.num_attention_heads)
                row = node.head_idx // heads_per_row
                col = node.head_idx % heads_per_row

                x_offset = (col - heads_per_row / 2) * 40
                y_offset = row * 30
                positions[node] = (x_offset - 100, layer * layer_height + y_offset)

            elif node.component_type == 'mlp':
                positions[node] = (100, layer * layer_height)

        return positions, node_info

    def create_interactive_graph(self, show_circuit_only: bool = False,
                                 circuit_edges: Optional[List] = None) -> go.Figure:
        """Create an interactive Plotly graph visualization."""

        positions, node_info = self.create_hierarchical_layout()

        # Filter for circuit if specified
        if show_circuit_only and circuit_edges:
            circuit_nodes = set()
            for edge in circuit_edges:
                circuit_nodes.add(edge.sender)
                circuit_nodes.add(edge.receiver)
            nodes_to_show = [n for n in self.graph.nodes if n in circuit_nodes]
            edges_to_show = circuit_edges
        else:
            nodes_to_show = list(self.graph.nodes)
            edges_to_show = list(self.graph.edges)

        # Create traces for different node types
        traces = []

        for node_type, color in self.colors.items():
            nodes_of_type = [n for n in nodes_to_show
                             if node_info.get(n, {}).get('type') == node_type]

            if not nodes_of_type:
                continue

            x_pos = [positions[n][0] for n in nodes_of_type]
            y_pos = [positions[n][1] for n in nodes_of_type]

            # Create hover text
            hover_text = []
            for n in nodes_of_type:
                info = node_info[n]
                if info['type'] == 'attention':
                    text = f"Layer {info['layer']}<br>Head {info['head_idx']}<br>{info['name']}"
                else:
                    text = f"Layer {info['layer']}<br>{info['name']}"
                hover_text.append(text)

            traces.append(go.Scatter(
                x=x_pos, y=y_pos,
                mode='markers',
                marker=dict(
                    size=12 if node_type == 'attention' else 15,
                    color=color,
                    line=dict(width=2, color='black')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=node_type.replace('_', ' ').title(),
                showlegend=True
            ))

        # Add edges
        edge_traces = self._create_edge_traces(edges_to_show, positions, node_info)
        traces.extend(edge_traces)

        # Create figure
        fig = go.Figure(data=traces)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{self.model_name} - {'Circuit' if show_circuit_only else 'Full Graph'}"
                     f"<br><sub>{self.num_layers} layers, {self.num_attention_heads} heads per layer</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=80),
            annotations=self._create_layer_annotations(),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=max(800, self.num_layers * 80)
        )

        return fig

    def _create_edge_traces(self, edges, positions, node_info) -> List[go.Scatter]:
        """Create edge traces with different colors for different connection types."""
        edge_traces = []

        # Group edges by type
        edge_groups = {'residual': [], 'attention': [], 'mlp': [], 'other': []}

        for edge in edges:
            sender_type = node_info.get(edge.sender, {}).get('type', 'other')
            receiver_type = node_info.get(edge.receiver, {}).get('type', 'other')

            if sender_type == 'attention' or receiver_type == 'attention':
                edge_groups['attention'].append(edge)
            elif sender_type == 'mlp' or receiver_type == 'mlp':
                edge_groups['mlp'].append(edge)
            elif 'residual' in sender_type or 'residual' in receiver_type:
                edge_groups['residual'].append(edge)
            else:
                edge_groups['other'].append(edge)

        # Create traces for each edge type
        for edge_type, edge_list in edge_groups.items():
            if not edge_list:
                continue

            x_coords = []
            y_coords = []

            for edge in edge_list:
                x0, y0 = positions[edge.sender]
                x1, y1 = positions[edge.receiver]

                x_coords.extend([x0, x1, None])
                y_coords.extend([y0, y1, None])

            edge_traces.append(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(width=1.5, color=self.edge_colors[edge_type]),
                hoverinfo='none',
                name=f"{edge_type.title()} Connections",
                showlegend=True
            ))

        return edge_traces

    def _create_layer_annotations(self) -> List[dict]:
        """Create layer number annotations."""
        annotations = []

        for layer in range(self.num_layers):
            annotations.append(dict(
                x=-350,
                y=layer * 100,
                text=f"Layer {layer}",
                showarrow=False,
                font=dict(size=14, color="black"),
                xanchor="center"
            ))

        return annotations

    def create_circuit_comparison(self, circuit_edges: List,
                                  circuit_nodes: Optional[List] = None) -> go.Figure:
        """Create a comparison showing full model vs discovered circuit."""

        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Full Model', 'Discovered Circuit'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Full model stats
        total_nodes = len(self.graph.nodes)
        total_edges = len(self.graph.edges)

        # Circuit stats
        if circuit_nodes is None:
            circuit_nodes = set()
            for edge in circuit_edges:
                circuit_nodes.add(edge.sender)
                circuit_nodes.add(edge.receiver)

        circuit_node_count = len(circuit_nodes)
        circuit_edge_count = len(circuit_edges)

        # Create bar charts
        categories = ['Nodes', 'Edges']
        full_values = [total_nodes, total_edges]
        circuit_values = [circuit_node_count, circuit_edge_count]

        fig.add_trace(
            go.Bar(x=categories, y=full_values, name='Full Model',
                   marker_color='lightblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=categories, y=circuit_values, name='Circuit',
                   marker_color='orange'),
            row=1, col=2
        )

        # Add reduction percentages
        node_reduction = (1 - circuit_node_count / total_nodes) * 100
        edge_reduction = (1 - circuit_edge_count / total_edges) * 100

        fig.update_layout(
            title_text=f"Circuit Discovery Results<br>"
                       f"<sub>Node Reduction: {node_reduction:.1f}% | "
                       f"Edge Reduction: {edge_reduction:.1f}%</sub>",
            showlegend=False
        )

        return fig


# Usage example and helper functions
def visualize_gemma_graph(graph, num_layers: int, num_attention_heads: int,
                          circuit_edges: Optional[List] = None):
    """
    Main function to visualize Gemma-2-2B-IT computational graph.

    Args:
        graph: Your computational graph object
        num_layers: Number of transformer layers (28 for Gemma-2-2B)
        num_attention_heads: Number of attention heads per layer
        circuit_edges: Optional list of discovered circuit edges
        show_summary: Whether to show summary view for very large models
    """

    visualizer = LargeTransformerGraphVisualizer(
        graph, num_layers, num_attention_heads, "Gemma-2-2B-IT"
    )

    if circuit_edges:
        # Show circuit discovery results
        comparison_fig = visualizer.create_circuit_comparison(circuit_edges)
        comparison_fig.show()

        # Show the discovered circuit
        circuit_fig = visualizer.create_interactive_graph(
            show_circuit_only=True,
            circuit_edges=circuit_edges
        )
        circuit_fig.show()
    else:
        # Show full graph (might be very large for Gemma)
        full_fig = visualizer.create_interactive_graph()
        full_fig.show()


# Alternative: Simplified layer-wise visualization for very large models
def create_layer_wise_visualization(graph, num_layers: int,
                                    num_attention_heads: int) -> go.Figure:
    """Create a simplified layer-wise view for models with many components."""

    fig = go.Figure()

    # Create boxes for each layer
    for layer in range(num_layers):
        # Layer background
        fig.add_shape(
            type="rect",
            x0=-0.5, y0=layer - 0.4,
            x1=num_attention_heads + 0.5, y1=layer + 0.4,
            line=dict(color="black", width=1),
            fillcolor="rgba(200,200,200,0.3)"
        )

        # Attention heads
        for head in range(num_attention_heads):
            fig.add_trace(go.Scatter(
                x=[head], y=[layer],
                mode='markers',
                marker=dict(size=15, color='#FFD93D',
                            line=dict(width=1, color='black')),
                showlegend=False,
                hovertemplate=f'Layer {layer}, Head {head}<extra></extra>'
            ))

        # MLP (shown as larger marker)
        fig.add_trace(go.Scatter(
            x=[num_attention_heads + 1], y=[layer],
            mode='markers',
            marker=dict(size=20, color='#FF8C94',
                        line=dict(width=1, color='black')),
            showlegend=False,
            hovertemplate=f'Layer {layer}, MLP<extra></extra>'
        ))

    fig.update_layout(
        title="Layer-wise Model Architecture",
        xaxis_title="Component Index",
        yaxis_title="Layer",
        showlegend=False,
        height=max(600, num_layers * 40)
    )

    return fig