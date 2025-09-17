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
        return hash((self.name, self.layer, self.component_type, self.head_idx))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.name == other.name and
                self.layer == other.layer and
                self.component_type == other.component_type and
                self.head_idx == other.head_idx)

    def __repr__(self):
        return f"Node({self.name})"


@dataclass
class Edge:
    """Represents an edge in the computational graph."""
    sender: Node
    receiver: Node

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
        new_graph = ComputationalGraph()
        new_graph.nodes = self.nodes.copy()
        new_graph.edges = self.edges.copy()
        new_graph.adjacency = {k: v.copy() for k, v in self.adjacency.items()}
        new_graph.reverse_adjacency = {k: v.copy() for k, v in self.reverse_adjacency.items()}
        return new_graph