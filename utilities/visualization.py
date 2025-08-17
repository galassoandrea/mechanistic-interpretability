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


def visualize_computational_graph(graph, figsize=(16, 12), layout_type='hierarchical',
                                  save_path=None, show_labels=True, node_size_scale=1.0,
                                  highlight_parallel=True):
    """
    Visualize the computational graph optimized for Pythia/GPT-NeoX parallel architecture.

    Args:
        graph: ComputationalGraph object
        figsize: Figure size (width, height)
        layout_type: 'hierarchical', 'spring', 'circular', or 'shell'
        save_path: Path to save the figure (optional)
        show_labels: Whether to show node labels
        node_size_scale: Scale factor for node sizes
        highlight_parallel: Whether to highlight parallel connections with different edge styles
    """

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes to NetworkX graph
    for node in graph.nodes:
        G.add_node(node,
                   layer=node.layer,
                   component_type=node.component_type,
                   name=node.name)

    # Add edges to NetworkX graph with edge type classification
    edge_types = classify_edge_types(graph)
    for edge in graph.edges:
        edge_type = edge_types.get((edge.sender, edge.receiver), 'other')
        G.add_edge(edge.sender, edge.receiver, weight=edge.weight, edge_type=edge_type)

    # Define color scheme optimized for Pythia architecture
    color_map = {
        'embedding': '#FF6B6B',  # Red - Input
        'attention': '#4ECDC4',  # Teal - Parallel component 1
        'mlp': '#45B7D1',  # Blue - Parallel component 2
        'residual': '#96CEB4',  # Green - Residual stream
        'resid': '#96CEB4',  # Green (alternative naming)
    }

    # Define node sizes (make residual nodes slightly larger as they're central)
    size_map = {
        'embedding': 900,
        'attention': 600,
        'mlp': 600,
        'residual': 700,  # Slightly larger for the main stream
        'resid': 700,
    }

    # Get node colors and sizes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        color = color_map.get(node.component_type, '#CCCCCC')  # Default gray
        size = size_map.get(node.component_type, 500) * node_size_scale
        node_colors.append(color)
        node_sizes.append(size)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout based on layout_type
    if layout_type == 'hierarchical':
        pos = create_pythia_hierarchical_layout(G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    elif layout_type == 'shell':
        # Group nodes by layer for shell layout
        shells = []
        layers = defaultdict(list)
        for node in G.nodes():
            layers[node.layer].append(node)
        for layer in sorted(layers.keys()):
            shells.append(layers[layer])
        pos = nx.shell_layout(G, nlist=shells)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw edges with different styles for different types
    if highlight_parallel:
        draw_parallel_architecture_edges(G, pos, ax)
    else:
        nx.draw_networkx_edges(G, pos,
                               edge_color='#888888',
                               arrows=True,
                               arrowsize=20,
                               arrowstyle='-|>',
                               alpha=0.6,
                               width=1.5,
                               ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           linewidths=2,
                           edgecolors='black',
                           ax=ax)

    # Draw labels if requested
    if show_labels:
        labels = {}
        for node in G.nodes():
            label = create_pythia_label(node)
            labels[node] = label

        nx.draw_networkx_labels(G, pos, labels,
                                font_size=8,
                                font_weight='bold',
                                ax=ax)

    # Create legend
    legend_elements = create_pythia_legend(color_map, highlight_parallel)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Set title and remove axes
    title = "Pythia-70M Computational Graph\n(Parallel Attention + MLP Architecture)"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add architecture info as text
    arch_info = "Architecture: resid_pre → {attention, MLP} → resid_post"
    ax.text(0.02, 0.02, arch_info, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax


def classify_edge_types(graph):
    """
    Classify edges based on their role in the Pythia architecture.
    """
    edge_types = {}

    for edge in graph.edges:
        sender_type = edge.sender.component_type
        receiver_type = edge.receiver.component_type
        sender_layer = edge.sender.layer
        receiver_layer = edge.receiver.layer

        # Classify edge types for Pythia architecture
        if sender_layer != receiver_layer:
            edge_types[(edge.sender, edge.receiver)] = 'layer_connection'
        elif sender_type == 'residual' and receiver_type == 'residual':
            edge_types[(edge.sender, edge.receiver)] = 'residual_connection'
        elif sender_type == 'residual' and receiver_type in ['attention', 'mlp']:
            edge_types[(edge.sender, edge.receiver)] = 'parallel_input'
        elif sender_type in ['attention', 'mlp'] and receiver_type == 'residual':
            edge_types[(edge.sender, edge.receiver)] = 'parallel_output'
        else:
            edge_types[(edge.sender, edge.receiver)] = 'other'

    return edge_types


def draw_parallel_architecture_edges(G, pos, ax):
    """
    Draw edges with different styles to highlight the parallel architecture.
    """
    # Define edge styles for different connection types
    edge_styles = {
        'layer_connection': {'color': '#333333', 'style': '-', 'width': 2.5, 'alpha': 0.8},
        'residual_connection': {'color': '#2E8B57', 'style': '-', 'width': 3.0, 'alpha': 0.9},
        'parallel_input': {'color': '#FF7F50', 'style': '--', 'width': 2.0, 'alpha': 0.7},
        'parallel_output': {'color': '#4169E1', 'style': '-.', 'width': 2.0, 'alpha': 0.7},
        'other': {'color': '#888888', 'style': '-', 'width': 1.0, 'alpha': 0.5},
    }

    # Group edges by type
    edges_by_type = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_type = data.get('edge_type', 'other')
        edges_by_type[edge_type].append((u, v))

    # Draw each edge type with its specific style
    for edge_type, edges in edges_by_type.items():
        style = edge_styles[edge_type]

        nx.draw_networkx_edges(G, pos,
                               edgelist=edges,
                               edge_color=style['color'],
                               style=style['style'],
                               width=style['width'],
                               alpha=style['alpha'],
                               arrows=True,
                               arrowsize=20,
                               arrowstyle='-|>',
                               ax=ax)


def create_pythia_hierarchical_layout(G):
    """
    Create a hierarchical layout optimized for Pythia's parallel architecture.
    """
    pos = {}
    layers = defaultdict(list)

    # Group nodes by layer
    for node in G.nodes():
        layers[node.layer].append(node)

    # Calculate positions with emphasis on parallel structure
    layer_height = 4.0  # Increased spacing for better visibility

    for layer_idx, nodes in layers.items():
        y = -layer_idx * layer_height

        # Group nodes by component type within each layer
        component_groups = defaultdict(list)
        for node in nodes:
            component_groups[node.component_type].append(node)

        # Special layout for Pythia parallel architecture
        if layer_idx == 0:
            # Embedding at the top center
            if 'embedding' in component_groups:
                pos[component_groups['embedding'][0]] = (0, y)
        else:
            # Arrange components to show parallel structure clearly
            x_positions = {
                'residual': {'resid_pre': -3, 'resid_post': 3},  # Spread residual nodes
                'attention': -1,  # Left side of parallel block
                'mlp': 1,  # Right side of parallel block
            }

            # Position residual nodes
            if 'residual' in component_groups or 'resid' in component_groups:
                resid_nodes = component_groups.get('residual', []) + component_groups.get('resid', [])
                resid_nodes_sorted = sorted(resid_nodes, key=lambda n: get_resid_order(n.full_activation))

                for node in resid_nodes_sorted:
                    if 'resid_pre' in node.full_activation:
                        pos[node] = (x_positions['residual']['resid_pre'], y)
                    elif 'resid_post' in node.full_activation:
                        pos[node] = (x_positions['residual']['resid_post'], y)
                    else:
                        pos[node] = (0, y)  # Default center position

            # Position attention nodes (left side of parallel block)
            if 'attention' in component_groups:
                attn_nodes = component_groups['attention']
                base_x = x_positions['attention']

                if len(attn_nodes) == 1:
                    pos[attn_nodes[0]] = (base_x, y)
                else:
                    # Spread multiple attention nodes vertically
                    for i, node in enumerate(attn_nodes):
                        y_offset = (i - len(attn_nodes) / 2 + 0.5) * 0.3
                        pos[node] = (base_x, y + y_offset)

            # Position MLP nodes (right side of parallel block)
            if 'mlp' in component_groups:
                mlp_nodes = component_groups['mlp']
                base_x = x_positions['mlp']

                if len(mlp_nodes) == 1:
                    pos[mlp_nodes[0]] = (base_x, y)
                else:
                    # Spread multiple MLP nodes vertically
                    for i, node in enumerate(mlp_nodes):
                        y_offset = (i - len(mlp_nodes) / 2 + 0.5) * 0.3
                        pos[node] = (base_x, y + y_offset)

    return pos


def get_resid_order(activation_name):
    """Helper function to order residual connections for Pythia."""
    if 'resid_pre' in activation_name:
        return 0
    elif 'resid_post' in activation_name:
        return 1
    else:
        return 2


def create_pythia_label(node):
    """Create labels optimized for Pythia architecture."""
    if node.component_type == 'embedding':
        return 'Embed'

    # Extract key parts from the name
    name_parts = node.name.split('.')
    if len(name_parts) >= 2:
        layer_part = name_parts[0]  # e.g., "L0"
        activation_part = name_parts[1]  # e.g., "resid_pre"

        # More descriptive shortening for Pythia
        if 'resid_pre' in activation_part:
            return f"{layer_part}.Pre"
        elif 'resid_post' in activation_part:
            return f"{layer_part}.Post"
        elif 'attn' in activation_part:
            if 'result' in activation_part or 'out' in activation_part:
                return f"{layer_part}.Att"
            else:
                return f"{layer_part}.A_in"
        elif 'mlp' in activation_part:
            if 'post' in activation_part or 'out' in activation_part:
                return f"{layer_part}.MLP"
            else:
                return f"{layer_part}.M_in"

        # Fallback
        activation_short = activation_part.replace('resid_', 'R_').replace('attn', 'A').replace('mlp', 'M')
        return f"{layer_part}.{activation_short}"

    return node.name[:10]


def create_pythia_legend(color_map, highlight_parallel):
    """Create legend elements for Pythia visualization."""
    legend_elements = []

    # Add node type legend
    for component_type, color in color_map.items():
        legend_elements.append(mpatches.Patch(color=color, label=component_type.capitalize()))

    # Add edge type legend if parallel highlighting is enabled
    if highlight_parallel:
        legend_elements.append(mpatches.Patch(color='white', label=''))  # Spacer
        legend_elements.append(mpatches.Patch(color='#333333', label='Layer connections'))
        legend_elements.append(mpatches.Patch(color='#2E8B57', label='Residual stream'))
        legend_elements.append(mpatches.Patch(color='#FF7F50', label='Parallel inputs'))
        legend_elements.append(mpatches.Patch(color='#4169E1', label='Parallel outputs'))

    return legend_elements


def analyze_pythia_graph_structure(graph):
    """
    Analyze the computational graph specifically for Pythia architecture patterns.
    """
    print("=== Pythia Computational Graph Analysis ===")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")

    # Count nodes by type and layer
    type_counts = defaultdict(int)
    layer_counts = defaultdict(lambda: defaultdict(int))

    for node in graph.nodes:
        type_counts[node.component_type] += 1
        layer_counts[node.layer][node.component_type] += 1

    print("\nNodes by type:")
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type}: {count}")

    print("\nNodes by layer:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer}:")
        for component_type, count in layer_counts[layer].items():
            print(f"    {component_type}: {count}")

    # Analyze parallel structure
    print("\n=== Parallel Architecture Analysis ===")
    edge_types = classify_edge_types(graph)
    edge_type_counts = defaultdict(int)

    for edge_type in edge_types.values():
        edge_type_counts[edge_type] += 1

    print("Edge types:")
    for edge_type, count in edge_type_counts.items():
        print(f"  {edge_type}: {count}")

    # Check for proper parallel structure
    layers_with_parallel = 0
    for layer in range(1, max(layer_counts.keys()) + 1):
        has_attention = 'attention' in layer_counts[layer]
        has_mlp = 'mlp' in layer_counts[layer]
        has_resid_pre = any('resid' in comp for comp in layer_counts[layer])
        has_resid_post = any('resid' in comp for comp in layer_counts[layer])

        if has_attention and has_mlp and has_resid_pre and has_resid_post:
            layers_with_parallel += 1

    print(f"\nLayers with complete parallel structure: {layers_with_parallel}")

    return edge_type_counts


def visualize_pythia_transformer_graph(computational_graph, analysis=True):
    """
    Complete visualization pipeline optimized for Pythia architecture.
    """
    if analysis:
        analyze_pythia_graph_structure(computational_graph)
        print("\n" + "=" * 60 + "\n")

    # Create visualizations optimized for Pythia
    print("Creating Pythia-optimized hierarchical layout...")
    fig, ax = visualize_computational_graph(
        computational_graph,
        figsize=(16, 12),
        layout_type='hierarchical',
        show_labels=True,
        node_size_scale=1.3,
        highlight_parallel=True
    )
    plt.show()

    # Optional: Also create a spring layout for comparison
    print("Creating spring layout for comparison...")
    fig, ax = visualize_computational_graph(
        computational_graph,
        figsize=(14, 10),
        layout_type='spring',
        show_labels=True,
        node_size_scale=1.0,
        highlight_parallel=True
    )
    plt.show()