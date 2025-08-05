import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict
from evaluation import evaluate_circuit_performance

class Visualization:

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def visualize_head_importance(self, results: Dict):
        """Create visualization of head importance scores"""
        # Prepare data for heatmap
        ablation_matrix = np.zeros((self.model.cfg.n_layers, self.model.cfg.n_heads))

        for head_name, data in results['head_ablation_results'].items():
            layer, head = data['layer'], data['head']
            ablation_matrix[layer, head] = data['importance']

        # Create subplots
        fig, ax = plt.subplots(figsize=(15, 6))

        # Head ablation heatmap
        sns.heatmap(ablation_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                    ax=ax, xticklabels=range(self.model.cfg.n_heads),
                    yticklabels=range(self.model.cfg.n_layers))
        ax.set_title('Head Ablation Importance Scores')
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')

        plt.tight_layout()
        plt.show()

        # Top heads bar plot
        top_heads = results['head_importance_ranking'][:10]
        head_names = [h['head'] for h in top_heads]
        ablation_scores = [h['ablation_importance'] for h in top_heads]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(head_names))
        width = 0.35

        ax.bar(x - width / 2, ablation_scores, width, label='Head Ablation', alpha=0.8)

        ax.set_xlabel('Attention Heads')
        ax.set_ylabel('Importance Score')
        ax.set_title('Top 10 Most Important Attention Heads')
        ax.set_xticks(x)
        ax.set_xticklabels(head_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    def visualize_circuit_graph(self, results: Dict):
        """Visualize the discovered circuit as a network graph"""
        G = nx.DiGraph()

        # Add nodes for each layer
        for layer in range(self.model.cfg.n_layers):
            G.add_node(f"Layer_{layer}", layer=layer, node_type="layer")

        # Add circuit heads as nodes
        circuit_heads = results['circuit_components']
        head_importance = {h['head']: h['ablation_importance']
                           for h in results['head_importance_ranking']}

        for head_name in circuit_heads:
            layer = int(head_name[1:].split('H')[0])
            head_idx = int(head_name.split('H')[1])
            importance = head_importance[head_name]

            G.add_node(head_name,
                       layer=layer,
                       head=head_idx,
                       importance=importance,
                       node_type="head")

            # Add edge from layer to head
            G.add_edge(f"Layer_{layer}", head_name)

            # Add edges between layers (information flow)
            if layer < self.model.cfg.n_layers - 1:
                G.add_edge(head_name, f"Layer_{layer + 1}")

        # Create layout
        pos = {}
        layer_width = max(len([n for n in G.nodes() if G.nodes[n].get('layer') == layer])
                          for layer in range(self.model.cfg.n_layers))

        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'layer':
                layer = G.nodes[node]['layer']
                pos[node] = (layer * 2, 0)
            else:  # head
                layer = G.nodes[node]['layer']
                head_idx = G.nodes[node]['head']
                pos[node] = (layer * 2, (head_idx - self.model.cfg.n_heads / 2) * 0.5)

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

        # Print circuit summary
        print(f"\nDiscovered IOI Circuit Summary:")
        print(f"Total circuit components: {len(circuit_heads)}")
        print(f"Circuit heads: {sorted(circuit_heads)}")
        print(f"Baseline performance: {results['baseline_score']:.4f}")

        # Calculate circuit performance (ablate all non-circuit heads)
        circuit_performance = evaluate_circuit_performance(self.dataset, self.model, circuit_heads)
        print(f"Circuit-only performance: {circuit_performance:.4f}")
        print(f"Performance retention: {circuit_performance / results['baseline_score'] * 100:.1f}%")