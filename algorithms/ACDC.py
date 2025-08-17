import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from collections import defaultdict, deque
from utilities.evaluation import kl_divergence
from transformer_lens import HookedTransformer
import numpy as np
from tqdm import tqdm
import copy


@dataclass
class Node:
    """Represents a node in the computational graph."""
    name: str  # e.g., "mlp.0", "attn.1.head.2", "resid_pre.3"
    layer: int  # Layer index in the model
    component_type: str  # "attention", "mlp", "embedding", "resid", etc.
    full_activation: str
    head_idx: Optional[int] = None  # For attention heads
    position: Optional[int] = None  # For position-specific nodes

    def __hash__(self):
        return hash((self.name, self.layer, self.component_type, self.head_idx, self.position))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.name == other.name and
                self.layer == other.layer and
                self.component_type == other.component_type and
                self.head_idx == other.head_idx and
                self.position == other.position)

    def __repr__(self):
        return f"Node({self.name})"


@dataclass
class Edge:
    """Represents an edge in the computational graph."""
    sender: Node  # Source node
    receiver: Node  # Destination node
    weight: float = 1.0  # Optional edge weight/importance

    def __hash__(self):
        return hash((self.sender, self.receiver))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.sender == other.sender and self.receiver == other.receiver

    def __repr__(self):
        return f"Edge({self.sender.name} -> {self.receiver.name})"


class ComputationalGraph:
    """Represents the full computational graph of a transformer model."""

    def __init__(self):
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.adjacency: Dict[Node, Set[Node]] = defaultdict(set)  # sender -> receivers
        self.reverse_adjacency: Dict[Node, Set[Node]] = defaultdict(set)  # receiver -> senders

    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        self.edges.add(edge)
        self.adjacency[edge.sender].add(edge.receiver)
        self.reverse_adjacency[edge.receiver].add(edge.sender)

    def remove_edge(self, edge: Edge):
        """Remove an edge from the graph."""
        if edge in self.edges:
            self.edges.remove(edge)
            self.adjacency[edge.sender].discard(edge.receiver)
            self.reverse_adjacency[edge.receiver].discard(edge.sender)

    def get_senders(self, node: Node) -> Set[Node]:
        """Get all nodes that send to the given node."""
        return self.reverse_adjacency.get(node, set())

    def get_receivers(self, node: Node) -> Set[Node]:
        """Get all nodes that receive from the given node."""
        return self.adjacency.get(node, set())

    def topological_sort(self) -> List[Node]:
        """
        Return nodes in topological order (from output to input).
        For ACDC, we want reverse topological order.
        """
        in_degree = {node: len(self.get_senders(node)) for node in self.nodes}
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for receiver in self.get_receivers(node):
                in_degree[receiver] -= 1
                if in_degree[receiver] == 0:
                    queue.append(receiver)

        # Reverse for output-to-input ordering
        return list(reversed(result))

    def copy(self) -> 'ComputationalGraph':
        """Create a deep copy of the graph."""
        new_graph = ComputationalGraph()
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = self.edges.copy()
        new_graph.adjacency = {k: v.copy() for k, v in self.adjacency.items()}
        new_graph.reverse_adjacency = {k: v.copy() for k, v in self.reverse_adjacency.items()}
        return new_graph


class ACDC:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    """

    def __init__(self, model, dataset, task_name: str = "IOI", threshold: float = 0.01):

        self.model = model
        self.dataset = dataset
        self.task_name = task_name
        self.threshold = threshold
        self.device = model.cfg.device

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations
        self.activation_cache = {}

        # Hooks for intervention
        self.hooks = []

    def build_computational_graph(self):
        """
        Build the full computational graph of the model.
        The model used for the experiments is the EleutherAI/Pythia model.
        Its architecture is different from the original transformer architecture.
        Key difference: Parallel attention and MLP, not sequential:
        resid_pre → attention ↘
                 ↘     → resid_post
                 → MLP ↗
        """
        graph = ComputationalGraph()
        model_components = self.model.hook_dict.keys()
        # Store nodes by layer and type for easier edge creation
        nodes_by_layer = defaultdict(dict)

        # Add embedding node
        embed_node = Node(name="embed", layer=0, component_type="embedding", full_activation="hook_embed")
        graph.add_node(embed_node)
        nodes_by_layer[0]["embedding"] = embed_node

        # Add all other nodes (attention, MLP, residual connections)
        for full_activation in model_components:
            if full_activation.startswith("blocks."):
                if "attn" in full_activation:
                    # Attention nodes
                    layer = int(full_activation.split('.')[1]) + 1
                    act_name = full_activation.split('_', 1)[1]
                    node = Node(
                        name=f"L{layer}.{act_name}",
                        layer=layer,
                        component_type="attention",
                        full_activation=full_activation
                    )
                    graph.add_node(node)
                    if "attention" not in nodes_by_layer[layer]:
                        nodes_by_layer[layer]["attention"] = []
                    nodes_by_layer[layer]["attention"].append(node)
                elif "mlp" in full_activation:
                    # MLP nodes
                    layer = int(full_activation.split('.')[1]) + 1
                    act_name = full_activation.split('_', 1)[1]
                    node = Node(
                        name=f"L{layer}.{act_name}",
                        layer=layer,
                        component_type="mlp",
                        full_activation=full_activation
                    )
                    graph.add_node(node)
                    if "mlp" not in nodes_by_layer[layer]:
                        nodes_by_layer[layer]["mlp"] = []
                    nodes_by_layer[layer]["mlp"].append(node)
                elif "resid" in full_activation:
                    # Residual nodes
                    layer = int(full_activation.split('.')[1]) + 1
                    act_name = full_activation.split('_', 1)[1]
                    node = Node(
                        name=f"L{layer}.{act_name}",
                        layer=layer,
                        component_type="residual",
                        full_activation=full_activation
                    )
                    graph.add_node(node)
                    nodes_by_layer[layer][act_name] = node

        # Create edges following transformer architecture
        max_layer = 6 #max(nodes_by_layer.keys())
        # Handle embedding to layer 1 connection
        if 0 in nodes_by_layer and 1 in nodes_by_layer:
            embed_node = nodes_by_layer[0]["embedding"]
            layer_1_resid_pre = nodes_by_layer[1].get("resid_pre")
            if layer_1_resid_pre:
                edge = Edge(sender=embed_node, receiver=layer_1_resid_pre)
                graph.add_edge(edge)

        for layer in range(1, max_layer + 1):
            current_layer_nodes = nodes_by_layer[layer]

            # Previous layer's resid_post -> current layer's resid_pre
            if layer > 1:  # Skip for layer 1 (handled by embedding)
                prev_resid_post = nodes_by_layer[layer - 1].get("resid_post")
                curr_resid_pre = current_layer_nodes.get("resid_pre")
                if prev_resid_post and curr_resid_pre:
                    edge = Edge(sender=prev_resid_post, receiver=curr_resid_pre)
                    graph.add_edge(edge)

            # resid_pre -> attention (parallel path)
            resid_pre = current_layer_nodes.get("resid_pre")
            if resid_pre and "attention" in current_layer_nodes:
                for attn_node in current_layer_nodes["attention"]:
                    edge = Edge(sender=resid_pre, receiver=attn_node)
                    graph.add_edge(edge)

            # resid_pre -> MLP (parallel path)
            if resid_pre and "mlp" in current_layer_nodes:
                for mlp_node in current_layer_nodes["mlp"]:
                    edge = Edge(sender=resid_pre, receiver=mlp_node)
                    graph.add_edge(edge)

            # attention and MLP outputs -> resid_post (parallel combination)
            resid_post = current_layer_nodes.get("resid_post")
            if resid_post:
                # resid_pre -> resid_post
                if resid_pre:
                    edge = Edge(sender=resid_pre, receiver=resid_post)
                    graph.add_edge(edge)

                # Attention output -> resid_post
                if "attention" in current_layer_nodes:
                    for attn_node in current_layer_nodes["attention"]:
                        # Connect attention output nodes to resid_post
                        if self.is_output_activation(attn_node.full_activation, "attention"):
                            edge = Edge(sender=attn_node, receiver=resid_post)
                            graph.add_edge(edge)

                # MLP output -> resid_post
                if "mlp" in current_layer_nodes:
                    for mlp_node in current_layer_nodes["mlp"]:
                        # Connect MLP output nodes to resid_post
                        if self.is_output_activation(mlp_node.full_activation, "mlp"):
                            edge = Edge(sender=mlp_node, receiver=resid_post)
                            graph.add_edge(edge)

        return graph

    def is_output_activation(self, full_activation, component_type):
        """ Determine if an activation is an output activation for a given component type."""
        if component_type == "attention":
            output_patterns = [
                "attn.hook_result",  # Final attention output
                "attn.hook_z",  # Pre-output projection
            ]
            return any(pattern in full_activation for pattern in output_patterns)

        elif component_type == "mlp":
            output_patterns = [
                "mlp.hook_post",  # Final MLP output
                "mlp.hook_out",  # Alternative naming
            ]
            return any(pattern in full_activation for pattern in output_patterns)

        return False


    #def discover_circuit(self):
    #    print(f"Building computational graph for {self.task_name} task...")
    #    self.full_graph = self.build_computational_graph(n_layers, n_heads)
    #    self.circuit = self.full_graph.copy()
    #    ordered_nodes = self.circuit.topological_sort()
#
    #    print(f"Total nodes: {len(ordered_nodes)}")
    #    print(f"Total edges: {len(self.circuit.edges)}")
    #    print(f"Starting edge pruning with threshold: {self.threshold}")
#
    #    # Collect reference outputs
    #    clean_logits = []
    #    for example in tqdm(self.dataset, desc="Collecting reference outputs"):
    #        inputs = example.clean_prompt
    #        logits = self.model(inputs, return_type="logits")
    #        clean_logits.append(logits)
#
    #    # Iterate through nodes and prune edges
    #    edges_removed = 0
    #    total_edges_evaluated = 0
#
    #    for receiver in tqdm(ordered_nodes, desc="Pruning edges"):
    #        senders = self.circuit.get_senders(receiver)
    #        for sender in senders:
    #            edge = Edge(sender, receiver)
#
    #            # Temporarily remove the edge
    #            kl_divs = []
    #            for i, example in enumerate(self.dataset):
    #                clean_prompt = example.clean_prompt
#
    #                ablated_logits = self.model.run_with_hooks(
    #                    clean_prompt,
    #                )
    #                kl_divs.append(kl_divergence(clean_logits[i], ablated_logits))
#
    #            avg_kl_div = np.mean(kl_divs)
    #            total_edges_evaluated += 1
    #            if avg_kl_div < self.threshold:
    #                self.circuit.remove_edge(edge)
    #                edges_removed += 1
#
    #    print(f"\nCircuit discovery complete!")
    #    print(f"Edges evaluated: {total_edges_evaluated}")
    #    print(f"Edges removed: {edges_removed}")
    #    print(f"Final circuit edges: {len(self.circuit.edges)}")
#
    #    return self.circuit