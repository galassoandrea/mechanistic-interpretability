import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict

def plot_activation_importance(num_layers, num_positions, patching_results):
    """Plots the activation importance heatmap based on patching results."""
    importance_matrix = np.zeros((num_layers, num_positions))
    for key, scores in patching_results.items():
        layer = int(key.split("_")[0][1:])  # Extract layer number from 'L0_resid_pre_pos3'
        position = int(key.split("pos")[-1])
        importance_matrix[layer, position] = np.mean(scores)
    plt.figure(figsize=(10, 6))
    plt.imshow(importance_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Normalized Score")
    plt.xlabel("Position")
    plt.ylabel("Layer")
    plt.title(f"Activation Importance")
    plt.show()


def plot_circuit_graph(circuit_nodes, circuit_edges=None, iteration=None):
    """Visualize the circuit as a directed graph"""
    G = nx.DiGraph()
    # Add nodes
    for node in circuit_nodes:
        G.add_node(node)
    # Add edges if provided
    if circuit_edges:
        for edge in circuit_edges:
            if edge[0] in circuit_nodes and edge[1] in circuit_nodes:
                G.add_edge(*edge)
    # Create positions for nodes (layer vertically, position horizontally)
    pos = {}
    for node in circuit_nodes:
        # Example: "L0_resid_pre_pos3"
        parts = node.split("_")
        layer = int(parts[0][1:])
        position = int(parts[-1].replace("pos", "")) if "pos" in parts[-1] else int(parts[-1])
        pos[node] = (position, -layer)  # Flip layer so top is layer 0
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=500)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.axis("off")
    title = "Circuit Graph"
    if iteration is not None:
        title += f" (Iteration {iteration})"
    plt.title(title)
    plt.show()
