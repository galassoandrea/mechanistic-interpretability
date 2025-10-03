from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque

@dataclass
class Node:
    """Represents a node in the computational graph."""
    name: str
    layer: int
    component_type: str
    full_activation: str
    head_idx: Optional[int] = None  # For attention heads

    def __hash__(self):
        return hash((self.full_activation, self.head_idx))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.full_activation == other.full_activation and
                self.head_idx == other.head_idx)

    def __repr__(self):
        if self.head_idx is not None:
            return f"Node({self.full_activation}, head={self.head_idx})"
        return f"Node({self.full_activation})"


@dataclass
class Edge:
    """Represents an edge in the computational graph."""
    sender: Node
    receiver: Node

    def __hash__(self):
        return hash((
            self.sender.full_activation, self.sender.head_idx,
            self.receiver.full_activation, self.receiver.head_idx
        ))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.sender.full_activation == other.sender.full_activation
            and self.sender.head_idx == other.sender.head_idx
            and self.receiver.full_activation == other.receiver.full_activation
            and self.receiver.head_idx == other.receiver.head_idx
        )

    def __repr__(self):
        s_head = f"_H{self.sender.head_idx}" if self.sender.head_idx is not None else ""
        r_head = f"_H{self.receiver.head_idx}" if self.receiver.head_idx is not None else ""
        return f"Edge({self.sender.full_activation}{s_head} -> {self.receiver.full_activation}{r_head})"

class ComputationalGraph:
    """Represents the full computational graph of a transformer model."""

    def __init__(self, model, model_name):
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.adjacency: Dict[Node, Set[Node]] = defaultdict(set)  # sender -> receivers
        self.reverse_adjacency: Dict[Node, Set[Node]] = defaultdict(set)  # receiver -> senders
        self.model = model
        self.model_name = model_name

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

    def remove_node(self, node: Node):
        """Remove a node from the graph."""
        if node in self.nodes:
            self.nodes.remove(node)
            # Remove all edges associated with the node
            for receiver in self.adjacency[node].copy():
                self.remove_edge(Edge(sender=node, receiver=receiver))
            for sender in self.reverse_adjacency[node].copy():
                self.remove_edge(Edge(sender=sender, receiver=node))
            # Clean up adjacency lists
            if node in self.adjacency:
                del self.adjacency[node]
            if node in self.reverse_adjacency:
                del self.reverse_adjacency[node]

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
        new_graph = ComputationalGraph(self.model, self.model_name)
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = self.edges.copy()
        new_graph.adjacency = {k: v.copy() for k, v in self.adjacency.items()}
        new_graph.reverse_adjacency = {k: v.copy() for k, v in self.reverse_adjacency.items()}
        return new_graph

def build_computational_graph(model, model_name):
    """Build the full computational graph of the model."""
    graph = ComputationalGraph(model, model_name)
    model_components = graph.model.hook_dict.keys()
    nodes_by_layer = defaultdict(dict)
    # Add embedding node
    embed_node = Node(name="embed", layer=0, component_type="embedding", full_activation="hook_embed")
    graph.add_node(embed_node)
    nodes_by_layer[0]["embedding"] = embed_node
    n_heads = graph.model.cfg.n_heads
    # Add all other nodes (attention, MLP, residual connections)
    for full_activation in model_components:
        if full_activation.startswith("blocks."):
            if "attn" in full_activation:
                # Attention nodes
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                if act_name == "hook_z":
                    for head_idx in range(n_heads):
                        node = Node(
                            name=act_name,
                            layer=layer,
                            component_type="attention",
                            head_idx=head_idx,
                            full_activation=full_activation
                        )
                        graph.add_node(node)
                        if "attention" not in nodes_by_layer[layer]:
                            nodes_by_layer[layer]["attention"] = []
                        nodes_by_layer[layer]["attention"].append(node)
            elif "mlp_out" in full_activation:
                # MLP nodes
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
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
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="residual",
                    full_activation=full_activation
                )
                graph.add_node(node)
                nodes_by_layer[layer][act_name] = node
    # Create edges following transformer architecture
    if "pythia" in graph.model_name:
        graph = create_parallel_edges(graph, nodes_by_layer)
    else:
        graph = create_sequential_edges(graph, nodes_by_layer)
    return graph

def create_parallel_edges(graph, nodes_by_layer):
    """Create edges following parallel transformer architecture."""
    max_layer = graph.model.cfg.n_layers
    # Handle embedding to layer 1 connection
    if 0 in nodes_by_layer and 1 in nodes_by_layer:
        embed_node = nodes_by_layer[0]["embedding"]
        layer_1_resid_pre = nodes_by_layer[1].get("hook_resid_pre")
        if layer_1_resid_pre:
            edge = Edge(sender=embed_node, receiver=layer_1_resid_pre)
            graph.add_edge(edge)
    for layer in range(1, max_layer + 1):
        current_layer_nodes = nodes_by_layer[layer]
        # Previous layer's resid_post -> current layer's resid_pre
        if layer > 1:
            prev_resid_post = nodes_by_layer[layer - 1].get("hook_resid_post")
            curr_resid_pre = current_layer_nodes.get("hook_resid_pre")
            if prev_resid_post and curr_resid_pre:
                edge = Edge(sender=prev_resid_post, receiver=curr_resid_pre)
                graph.add_edge(edge)
        # resid_pre -> attention (parallel path)
        resid_pre = current_layer_nodes.get("hook_resid_pre")
        if resid_pre and "attention" in current_layer_nodes:
            for attn_head_node in current_layer_nodes["attention"]:
                edge = Edge(sender=resid_pre, receiver=attn_head_node)
                graph.add_edge(edge)
        # resid_pre -> MLP (parallel path)
        if resid_pre and "mlp" in current_layer_nodes:
            for mlp_node in current_layer_nodes["mlp"]:
                edge = Edge(sender=resid_pre, receiver=mlp_node)
                graph.add_edge(edge)
        # attention and MLP outputs -> resid_post
        resid_post = current_layer_nodes.get("hook_resid_post")
        if resid_post:
            # resid_pre -> resid_post
            if resid_pre:
                edge = Edge(sender=resid_pre, receiver=resid_post)
                graph.add_edge(edge)
            # Attention heads output -> resid_post
            if "attention" in current_layer_nodes:
                for attn_head_node in current_layer_nodes["attention"]:
                    if "hook_z" in attn_head_node.full_activation:
                        edge = Edge(sender=attn_head_node, receiver=resid_post)
                        graph.add_edge(edge)
            # MLP output -> resid_post
            if "mlp" in current_layer_nodes:
                for mlp_node in current_layer_nodes["mlp"]:
                    if "mlp_out" in mlp_node.full_activation:
                        edge = Edge(sender=mlp_node, receiver=resid_post)
                        graph.add_edge(edge)
    return graph

def create_sequential_edges(graph, nodes_by_layer):
    """Create edges following sequential transformer architecture."""
    max_layer = graph.model.cfg.n_layers
    if 0 in nodes_by_layer and 1 in nodes_by_layer:
        embed_node = nodes_by_layer[0]["embedding"]
        layer_1_resid_pre = nodes_by_layer[1].get("hook_resid_pre")
        if layer_1_resid_pre:
            edge = Edge(sender=embed_node, receiver=layer_1_resid_pre)
            graph.add_edge(edge)
    for layer in range(1, max_layer + 1):
        current_layer_nodes = nodes_by_layer[layer]
        # Previous layer's resid_post -> current layer's resid_pre
        if layer > 1:
            prev_resid_post = nodes_by_layer[layer - 1].get("hook_resid_post")
            curr_resid_pre = current_layer_nodes.get("hook_resid_pre")
            if prev_resid_post and curr_resid_pre:
                edge = Edge(sender=prev_resid_post, receiver=curr_resid_pre)
                graph.add_edge(edge)
        # resid_pre -> attention
        resid_pre = current_layer_nodes.get("hook_resid_pre")
        if resid_pre and "attention" in current_layer_nodes:
            for attn_head_node in current_layer_nodes["attention"]:
                edge = Edge(sender=resid_pre, receiver=attn_head_node)
                graph.add_edge(edge)
        # attention -> resid_mid
        resid_mid = current_layer_nodes.get("hook_resid_mid")
        if resid_mid and "attention" in current_layer_nodes:
            for attn_head_node in current_layer_nodes["attention"]:
                if "hook_z" in attn_head_node.full_activation:
                    edge = Edge(sender=attn_head_node, receiver=resid_mid)
                    graph.add_edge(edge)
        # resid_pre -> resid_mid (skip connection)
        if resid_pre and resid_mid:
            edge = Edge(sender=resid_pre, receiver=resid_mid)
            graph.add_edge(edge)
        # resid_mid -> mlp
        if resid_mid and "mlp" in current_layer_nodes:
            for mlp_node in current_layer_nodes["mlp"]:
                edge = Edge(sender=resid_mid, receiver=mlp_node)
                graph.add_edge(edge)
        # mlp output -> resid_post
        resid_post = current_layer_nodes.get("hook_resid_post")
        if resid_post and "mlp" in current_layer_nodes:
            for mlp_node in current_layer_nodes["mlp"]:
                if "mlp_out" in mlp_node.full_activation:
                    edge = Edge(sender=mlp_node, receiver=resid_post)
                    graph.add_edge(edge)
        # resid_mid -> resid_post (skip connection)
        if resid_mid and resid_post:
            edge = Edge(sender=resid_mid, receiver=resid_post)
            graph.add_edge(edge)
    return graph