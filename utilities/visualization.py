from collections import defaultdict
from typing import List
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from typing import Dict, Set, Tuple, List
from collections import defaultdict

def create_pythia_graph(graph, num_layers, num_attention_heads):
    """Create a NetworkX graph representing the Pythia-70m computational graph."""

    G = nx.DiGraph()
    pos = {}
    node_colors = {}
    node_types = {}

    # Define colors for different node types
    colors = {
        'embedding': '#E8F4FD',
        'residual_pre': '#B8E6B8',
        'attention': '#FFD93D',
        'mlp': '#FF8C94',
        'residual_post': '#A8E6CF'
    }

    for node in graph.nodes:
        G.add_node(node, layer=node.layer, component_type=node.component_type, name=node.name)
        if node.component_type == 'embedding':
            pos[node] = (0, 0)
            node_colors[node] = colors['embedding']
            node_types[node] = 'embedding'
        elif node.component_type == 'residual':
            if 'pre' in node.name:
                pos[node] = (0, node.layer * 15 - 4)
                node_colors[node] = colors['residual_pre']
                node_types[node] = 'residual_pre'
            else:
                pos[node] = (0, node.layer * 15 + 6)
                node_colors[node] = colors['residual_post']
                node_types[node] = 'residual_post'
        elif node.component_type == 'attention':
            # spread heads horizontally between -span and +span
            span = 2.0  # controls how wide the attention heads spread
            x_start = -span
            x_end = span
            step = (x_end - x_start) / max(1, num_attention_heads)
            x_pos = x_start + node.head_idx * step
            y_pos = node.layer * 15 + 1
            pos[node] = (x_pos, y_pos)
            node_colors[node] = colors['attention']
            node_types[node] = 'attention'
        elif node.component_type == 'mlp':
            pos[node] = (2, node.layer * 15 + 1)
            node_colors[node] = colors['mlp']
            node_types[node] = 'mlp'
        else:
            pos[node] = (np.random.rand() * 10 - 5, np.random.rand() * 10 - 5)
            node_colors[node] = '#CCCCCC'
            node_types[node] = 'other'

    for edge in graph.edges:
        G.add_edge(edge.sender, edge.receiver)

    return G, pos, node_colors, node_types


def visualize_pythia_graph(graph, num_layers, num_attention_heads,
                           figsize=(16, 12), node_size=3000, font_size=8):
    """Visualize the Pythia-70m computational graph using NetworkX."""

    # Create the graph
    G, pos, node_colors, node_types = create_pythia_graph(graph, num_layers, num_attention_heads)

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw nodes by type for better control
    for node_type, color in {
        'embedding': '#E8F4FD',
        'residual_pre': '#B8E6B8',
        'attention': '#FFD93D',
        'mlp': '#FF8C94',
        'residual_post': '#A8E6CF'
    }.items():
        nodes_of_type = [node for node, ntype in node_types.items() if ntype == node_type]
        if nodes_of_type:
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes_of_type,
                node_color=color,
                node_size=node_size,
                node_shape='o',
                edgecolors='black',
                linewidths=2,
                ax=ax
            )

    # Draw edges with different colors based on connection type
    attention_edges = []
    mlp_edges = []
    residual_edges = []

    for edge in G.edges():
        source, target = edge
        source_type = node_types[source]
        target_type = node_types[target]

        if source_type == 'attention' or target_type == 'attention':
            attention_edges.append(edge)
        elif source_type == 'mlp' or target_type == 'mlp':
            mlp_edges.append(edge)
        else:
            residual_edges.append(edge)

    # Draw different edge types with different colors
    if residual_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=residual_edges,
            edge_color='black', width=2, alpha=0.7,
            arrowsize=20, arrowstyle='->', ax=ax
        )

    if attention_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=attention_edges,
            edge_color='blue', width=2, alpha=0.6,
            arrowsize=20, arrowstyle='->', ax=ax
        )

    if mlp_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=mlp_edges,
            edge_color='red', width=2, alpha=0.6,
            arrowsize=20, arrowstyle='->', ax=ax
        )

    # Create custom labels
    labels = {}
    for node in G.nodes():
        node_type = node_types[node]
        if node_type == 'embedding':
            labels[node] = 'Embedding'
        elif node_type == 'residual_pre':
            labels[node] = f'Res Pre\nL{node.layer}'
        elif node_type == 'attention':
            labels[node] = f'Attn\nL{node.layer}H{node.head_idx}'
        elif node_type == 'mlp':
            labels[node] = f'MLP\nL{node.layer}'
        elif node_type == 'residual_post':
            labels[node] = f'Res Post\nL{node.layer}'

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, labels, font_size=font_size, font_weight='bold', ax=ax
    )

    # Create legend
    legend_elements = [
        mpatches.Patch(color='#E8F4FD', label='Embedding'),
        mpatches.Patch(color='#B8E6B8', label='Residual Pre'),
        mpatches.Patch(color='#FFD93D', label='Attention Heads'),
        mpatches.Patch(color='#FF8C94', label='MLP'),
        mpatches.Patch(color='#A8E6CF', label='Residual Post'),
        mpatches.Patch(color='white', label=''),  # Spacer
        plt.Line2D([0], [0], color='black', lw=2, label='Residual Connections'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Attention Connections'),
        plt.Line2D([0], [0], color='red', lw=2, label='MLP Connections')
    ]

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0))

    # Set title
    ax.set_title(f'Pythia-70m Computational Graph\n({num_layers} Layers, {num_attention_heads} Attention Heads)',
                 fontsize=16, fontweight='bold', pad=20)

    # Remove axes
    ax.set_axis_off()

    # Adjust layout
    plt.tight_layout()

    plt.show()

    return G, pos, node_colors, node_types

def visualize_computational_graph(
        graph,
        title: str = "Circuit Discovery: Computational Graph",
        width: int = 2000,
        height: int = 1400,
        show_edge_labels: bool = False
):
    """ Visualize a computational graph using Plotly to get an interactive visualization. """

    # Define color scheme for different component types
    color_map = {
        'embedding': '#FF6B6B',  # Red
        'residual_pre': '#4ECDC4',  # Teal
        'residual_mid': '#45B7D1',  # Light Blue
        'residual_post': '#96CEB4',  # Green
        'attention': '#FFEAA7',  # Yellow
        'mlp': '#DDA0DD'  # Plum
    }

    # Get number of layers and heads from the model configuration
    n_layers = graph.model.cfg.n_layers
    layers = list(range(n_layers + 1))
    n_heads = graph.model.cfg.n_heads

    # Calculate positions for nodes
    node_positions = {}

    # Define vertical ordering of component types within each layer
    component_order = {
        'embedding': 0,
        'residual_pre': 1,
        'attention': 2,
        'residual_mid': 3,
        'mlp': 4,
        'residual_post': 5
    }

    # Assign positions
    for node in graph.nodes:
        layer_idx = layers.index(node.layer)
        x_pos = layer_idx * 2  # Horizontal spacing between layers

        # Vertical position based on component type
        if node.component_type == 'residual':
            if 'pre' in node.name:
                base_y = component_order['residual_pre'] * 3
            elif 'mid' in node.name:
                base_y = component_order['residual_mid'] * 3
            else:
                base_y = component_order['residual_post'] * 3
        else:
            base_y = component_order.get(node.component_type, 3) * 3

        # For attention heads, spread them horizontally within their slot
        if node.component_type == 'attention' and node.head_idx is not None:
            # Offset x position for each head
            head_offset = (node.head_idx - (n_heads - 1) / 2) * 0.15
            x_pos += head_offset

        node_positions[node] = (x_pos, base_y)

    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_hover_text = []

    for edge in graph.edges:
        x0, y0 = node_positions[edge.sender]
        x1, y1 = node_positions[edge.receiver]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover_text.append(f"{edge.sender.name} → {edge.receiver.name}")

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.8, color='#888'),
        hoverinfo='skip',
        showlegend=False,
        opacity=0.5
    )

    # Prepare node traces (one per component type for legend)
    node_traces = {}

    for comp_type in component_order.keys():
        node_traces[comp_type] = {
            'x': [],
            'y': [],
            'text': [],
            'hover_text': [],
            'marker_size': []
        }

    for node in graph.nodes:
        x, y = node_positions[node]
        if node.component_type == 'residual':
            if 'pre' in node.name:
                comp_type = 'residual_pre'
            elif 'mid' in node.name:
                comp_type = 'residual_mid'
            else:
                comp_type = 'residual_post'
        else:
            comp_type = node.component_type

        node_traces[comp_type]['x'].append(x)
        node_traces[comp_type]['y'].append(y)
        node_traces[comp_type]['text'].append(node.name)

        # Create hover text
        in_degree = len(graph.reverse_adjacency[node]) if node in graph.reverse_adjacency else 0
        out_degree = len(graph.adjacency[node]) if node in graph.adjacency else 0
        hover_text = (
            f"<b>{node.name}</b><br>"
            f"Type: {node.component_type}<br>"
            f"Layer: {node.layer}<br>"
            f"In-degree: {in_degree}<br>"
            f"Out-degree: {out_degree}"
        )
        if node.head_idx is not None:
            hover_text += f"<br>Head: {node.head_idx}"

        node_traces[comp_type]['hover_text'].append(hover_text)

        # Size based on connectivity
        size = 10 + (in_degree + out_degree) * 2
        node_traces[comp_type]['marker_size'].append(min(size, 30))

    # Create figure
    fig = go.Figure()

    # Add edge trace
    fig.add_trace(edge_trace)

    # Add node traces
    for comp_type, trace_data in node_traces.items():
        if len(trace_data['x']) > 0:  # Only add if there are nodes of this type
            fig.add_trace(go.Scatter(
                x=trace_data['x'],
                y=trace_data['y'],
                mode='markers+text',
                name=comp_type.replace('_', ' ').title(),
                text=trace_data['text'],
                hovertext=trace_data['hover_text'],
                hoverinfo='text',
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                marker=dict(
                    size=trace_data['marker_size'],
                    color=color_map.get(comp_type, '#95a5a6'),
                    line=dict(width=2, color='white'),
                    symbol='circle'
                )
            ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Model: {graph.model_name} | "
                 f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}</sub>",
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        width=width,
        height=height,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            title="Layer",
            tickmode='array',
            tickvals=list(range(0, n_layers * 2, 2)),
            ticktext=[f"L{i}" for i in layers]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title=""
        ),
        margin=dict(l=20, r=20, t=100, b=40)
    )

    # Add annotations for component types
    for comp_type, order_idx in component_order.items():
        if any(n.component_type == comp_type for n in graph.nodes):
            fig.add_annotation(
                x=-0.5,
                y=order_idx * 3,
                text=comp_type.replace('_', ' ').title(),
                showarrow=False,
                xanchor='right',
                font=dict(size=10, color=color_map.get(comp_type, '#95a5a6')),
                bgcolor='white',
                bordercolor=color_map.get(comp_type, '#95a5a6'),
                borderwidth=1,
                borderpad=4
            )
    fig.show()
    fig.write_html("computational_graph.html")

def visualize_computational_graph_hierarchical(
        graph,
        title: str = "Circuit Discovery: Hierarchical View",
        width: int = 1600,
        height: int = 1200
):
    """ Alternative visualization function to get a more hierarchical, top-down visualization."""
    # Use Plotly's built-in tree layout approximation
    color_map = {
        'embedding': '#FF6B6B',
        'residual': '#4ECDC4',
        'residual_mid': '#45B7D1',
        'residual_post': '#96CEB4',
        'attention': '#FFEAA7',
        'mlp': '#DDA0DD'
    }
    # Build position using layer-based vertical layout
    n_layers = graph.model.cfg.n_layers
    layers = list(range(n_layers + 1))
    node_positions = {}
    # Group nodes by layer and type
    layer_groups = defaultdict(lambda: defaultdict(list))
    for node in graph.nodes:
        layer_groups[node.layer][node.component_type].append(node)
    # Assign positions
    for layer_idx, layer in enumerate(layers):
        y_base = -layer_idx * 4  # Vertical spacing
        groups = layer_groups[layer]
        all_nodes_in_layer = []
        for comp_type in ['embedding', 'residual', 'attention' 'mlp']:
            if comp_type == 'residual':
                for sub_type in ['residual_pre', 'residual_mid', 'residual_post']:
                    all_nodes_in_layer.extend(groups[sub_type])
            else:
                all_nodes_in_layer.extend(groups[comp_type])
        n_nodes = len(all_nodes_in_layer)
        for i, node in enumerate(all_nodes_in_layer):
            x_pos = (i - (n_nodes - 1) / 2) * 1.5
            node_positions[node] = (x_pos, y_base)
    # Create visualization similar to first function but with vertical layout
    edge_x, edge_y = [], []
    for edge in graph.edges:
        x0, y0 = node_positions[edge.sender]
        x1, y1 = node_positions[edge.receiver]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#888'),
        hoverinfo='skip',
        showlegend=False,
        opacity=0.4
    ))
    # Add nodes
    for comp_type in color_map.keys():
        if comp_type in ['residual_pre', 'residual_mid', 'residual_post']:
            nodes_of_type = [n for n in graph.nodes if n.component_type == 'residual' and comp_type.split('_')[1] in n.name]
        else:
            nodes_of_type = [n for n in graph.nodes if n.component_type == comp_type]
        if not nodes_of_type:
            continue
        xs = [node_positions[n][0] for n in nodes_of_type]
        ys = [node_positions[n][1] for n in nodes_of_type]
        texts = [n.name for n in nodes_of_type]
        hovers = []
        for n in nodes_of_type:
            in_deg = len(graph.reverse_adjacency[n])
            out_deg = len(graph.adjacency[n])
            hover = f"<b>{n.name}</b><br>Layer: {n.layer}<br>In: {in_deg} | Out: {out_deg}"
            hovers.append(hover)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            name=comp_type.replace('_', ' ').title(),
            text=texts,
            hovertext=hovers,
            hoverinfo='text',
            textposition="middle center",
            textfont=dict(size=8),
            marker=dict(
                size=15,
                color=color_map[comp_type],
                line=dict(width=2, color='white')
            )
        ))
    fig.update_layout(
        title=f"{title}<br><sub>{graph.model_name}</sub>",
        showlegend=True,
        width=width,
        height=height,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig
