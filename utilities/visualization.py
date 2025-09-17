from collections import defaultdict
from typing import List

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

def plot_activation_importance(num_layers, num_positions, patching_results):
    """Plots the activation importance heatmap based on patching results."""
    importance_matrix = np.zeros((num_layers, num_positions))
    for key, scores in patching_results.items():
        layer = int(key.split("_")[0][1:])
        position = int(key.split("pos")[-1])
        importance_matrix[layer, position] = np.mean(scores)
    plt.figure(figsize=(10, 6))
    plt.imshow(importance_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Normalized Score")
    plt.xlabel("Position")
    plt.ylabel("Layer")
    plt.title(f"Activation Importance")
    plt.show()


def plot_circuit_graph(circuit_nodes, iteration=None):
    """Visualize the circuit as a directed graph with edges inferred from transformer block order."""
    G = nx.DiGraph()

    # Define a predefined order of activation types for consistent layout
    activation_types_order = [
        "resid_pre", "hook_q", "hook_k", "v", "z",
        "resid_mid", "mlp_out", "resid_post"
    ]

    # Parse nodes into structured form
    parsed_nodes = []
    for node in circuit_nodes:
        parts = node.split("_")
        layer = int(parts[0][1:])
        act_type = "_".join(parts[1:-1])
        position = int(parts[-1].replace("pos", "")) if "pos" in parts[-1] else int(parts[-1])
        parsed_nodes.append((node, layer, act_type, position))
        G.add_node(node)

    # Build edges based on transformer forward pass order
    layer_groups = defaultdict(list)
    for node, layer, act_type, position in parsed_nodes:
        layer_groups[layer].append((node, act_type, position))

    for layer in sorted(layer_groups.keys()):
        # Sort activations in each layer according to activation_types_order
        layer_groups[layer].sort(key=lambda x: activation_types_order.index(x[1]))
        # Connect nodes in order inside the layer
        for i in range(len(layer_groups[layer]) - 1):
            G.add_edge(layer_groups[layer][i][0], layer_groups[layer][i+1][0])
        # Connect final activation in layer to first activation in next layer
        if layer + 1 in layer_groups:
            G.add_edge(layer_groups[layer][-1][0], layer_groups[layer+1][0][0])

    # Node positioning: parallel lines for different activation types
    pos = {}
    type_offset_map = {act: i for i, act in enumerate(activation_types_order)}
    for node, layer, act_type, position in parsed_nodes:
        y_coord = -layer - type_offset_map[act_type] * 0.15  # offset by type to avoid overlap
        pos[node] = (position, y_coord)

    # Color mapping for activation types
    color_map = plt.cm.get_cmap("tab10", len(activation_types_order))
    node_colors = [
        color_map(activation_types_order.index(act_type)) for _, _, act_type, _ in parsed_nodes
    ]

    # Draw graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Legend for activation types
    for idx, act_type in enumerate(activation_types_order):
        plt.scatter([], [], color=color_map(idx), label=act_type)
    plt.legend(title="Activation Types", loc="upper left", bbox_to_anchor=(1, 1))

    plt.axis("off")
    title = "Circuit Graph"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_head_importance(model, dataset, heads: List):
    """Create visualization of head importance scores"""
    # Prepare tasks for heatmap
    ablation_matrix = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for head in heads:
        layer, head, importance = head.layer, head.head, head.importance_score
        ablation_matrix[layer, head] = importance
    # Create subplots
    fig, ax = plt.subplots(figsize=(15, 6))
    # Head ablation heatmap
    sns.heatmap(ablation_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=ax, xticklabels=range(model.cfg.n_heads),
                yticklabels=range(model.cfg.n_layers))
    ax.set_title('Head Ablation Importance Scores')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    plt.tight_layout()
    plt.show()
    # Top heads bar plot
    head_names = [f"L{head.layer}_H{head.head}" for head in heads]
    head_scores = [head.importance_score for head in heads]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(head_names))
    width = 0.35
    ax.bar(x - width / 2, head_scores, width, label='Head Ablation', alpha=0.8)
    ax.set_xlabel('Attention Heads')
    ax.set_ylabel('Importance Score')
    ax.set_title('Top 10 Most Important Attention Heads')
    ax.set_xticks(x)
    ax.set_xticklabels(head_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_circuit_graph(model, heads: List):
    """Visualize the discovered circuit as a network graph"""
    G = nx.DiGraph()
    # Add nodes for each layer
    for layer in range(model.cfg.n_layers):
        G.add_node(f"Layer_{layer}", layer=layer, node_type="layer")
    # Add circuit heads as nodes
    circuit_heads = [f"L{head.layer}_H{head.head}" for head in heads]
    head_importance = {f"L{head.layer}_H{head.head}": head.importance_score for head in heads}
    for head_name in circuit_heads:
        layer = int(head_name[1])
        head_idx = int(head_name[4])
        importance = head_importance[head_name]
        G.add_node(head_name,
                   layer=layer,
                   head=head_idx,
                   importance=importance,
                   node_type="head")
        # Add edge from layer to head
        G.add_edge(f"Layer_{layer}", head_name)
        # Add edges between layers (information flow)
        if layer < model.cfg.n_layers - 1:
            G.add_edge(head_name, f"Layer_{layer + 1}")
    # Create layout
    pos = {}
    layer_width = max(len([n for n in G.nodes() if G.nodes[n].get('layer') == layer])
                      for layer in range(model.cfg.n_layers))
    for node in G.nodes():
        if G.nodes[node]['node_type'] == 'layer':
            layer = G.nodes[node]['layer']
            pos[node] = (layer * 2, 0)
        else:  # head
            layer = G.nodes[node]['layer']
            head_idx = G.nodes[node]['head']
            pos[node] = (layer * 2, (head_idx - model.cfg.n_heads / 2) * 0.5)
    # Plot
    plt.figure(figsize=(15, 10))
    # Draw layer nodes
    layer_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'layer']
    nx.draw_networkx_nodes(G, pos, nodelist=layer_nodes,
                           node_color='lightblue', node_size=800,
                           node_shape='s', alpha=0.7)
    # Draw head nodes with size proportional to importance
    head_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'head']
    head_sizes = [max(200, G.nodes[n]['importance'] * 5000) for n in head_nodes]
    head_colors = [G.nodes[n]['importance'] for n in head_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=head_nodes,
                           node_color=head_colors, node_size=head_sizes,
                           cmap='Reds', alpha=0.8)
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20)
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('IOI Circuit Graph\n(Node size proportional to importance)', size=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


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