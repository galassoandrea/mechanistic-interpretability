import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict, deque
from utilities.evaluation import kl_divergence
import numpy as np
from tqdm import tqdm
from tasks import IOI, Induction

@dataclass
class Node:
    """Represents a node in the computational graph."""
    name: str  # e.g., "mlp.0", "attn.1.head.2", "resid_pre.3"
    layer: int  # Layer index in the model
    component_type: str  # "attention", "mlp", "embedding", "resid", etc.
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
    sender: Node  # Source node
    receiver: Node  # Destination node

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


class ACDC:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    """

    def __init__(self, model, task: str = "IOI", mode: str = "edge", method: str = "greedy", threshold: float = 0.1):

        self.model = model
        self.task_name = task
        self.threshold = threshold
        self.device = model.cfg.device
        self.mode = mode
        self.method = method

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations
        self.activation_cache = {}

        # Hooks for intervention
        self.hooks = []

        # Create dataset based on the task
        if task == "IOI":
            print("Building IOI dataset...")
            dataset_builder = IOI.IOIDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=50)
        elif task == "induction":
            print("Building Induction dataset...")
            dataset_builder = Induction.InductionDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=50)


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
        n_heads = self.model.cfg.n_heads

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
                elif "mlp" in full_activation:
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
        max_layer = 6 #max(nodes_by_layer.keys())
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
            if layer > 1:  # Skip for layer 1 (handled by embedding)
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

            # attention and MLP outputs -> resid_post (parallel combination)
            resid_post = current_layer_nodes.get("hook_resid_post")
            if resid_post:
                # resid_pre -> resid_post
                if resid_pre:
                    edge = Edge(sender=resid_pre, receiver=resid_post)
                    graph.add_edge(edge)

                # Attention heads output -> resid_post
                if "attention" in current_layer_nodes:
                    for attn_head_node in current_layer_nodes["attention"]:
                        if self.is_output_activation(attn_head_node.full_activation, "attention"):
                            edge = Edge(sender=attn_head_node, receiver=resid_post)
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
        if component_type == "attention" and "attn.hook_z" in full_activation:
            return True

        elif component_type == "mlp" and "hook_mlp_out" in full_activation:
            return True

        return False


    def discover_circuit(self):
        """ Main method to perform circuit discovery using edge pruning. """

        print(f"Building computational graph for {self.task_name} task...")
        self.full_graph = self.build_computational_graph()
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        # Collect reference outputs
        clean_logits = []
        caches = []
        for example in tqdm(self.dataset, desc="Collecting reference outputs"):
            inputs = example.clean_tokens
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(inputs, return_type="logits")
            clean_logits.append(logits)
            caches.append(cache)

        if self.mode == "edge":
            print(f"Starting edge pruning with threshold: {self.threshold}")
            # Iterate through nodes and prune edges
            edges_removed = 0
            total_edges_evaluated = 0
            kl_score = 0.0
            ablated_edges = []
            for receiver in tqdm(ordered_nodes, desc="Pruning edges"):
                if receiver.name == "hook_resid_post":
                    senders = self.full_graph.get_senders(receiver)
                    for sender in senders:
                        if sender.name in ["hook_resid_pre", "hook_mlp_out"]:
                            edge = Edge(sender, receiver)
                            print(f"Evaluating edge: L{sender.layer}-{sender.name.split('_', 1)[1]} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                            # Temporarily remove the edge
                            kl_divs = []
                            for i, example in enumerate(self.dataset):
                                clean_tokens = example.clean_tokens
                                # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                                ablated_logits = self.run_with_edge_ablation(clean_tokens, edge, caches[i], ablated_edges)
                                kl_div = kl_divergence(clean_logits[i], ablated_logits)
                                kl_divs.append(kl_div.item())
                            avg_kl_div = np.mean(kl_divs)
                            total_edges_evaluated += 1
                            print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                            if avg_kl_div < self.threshold:
                                self.circuit.remove_edge(edge)
                                edges_removed += 1
                                ablated_edges.append(edge)
                                print(f"Edge removed.")
                        elif sender.name == "hook_z":
                            edge = Edge(sender, receiver)
                            print(f"Evaluating edge: L{sender.layer}-Head{sender.head_idx} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                            kl_divs = []
                            for i, example in enumerate(self.dataset):
                                clean_tokens = example.clean_tokens
                                ablated_logits = self.run_with_edge_ablation(clean_tokens, edge, caches[i], ablated_edges)
                                kl_div = kl_divergence(clean_logits[i], ablated_logits)
                                kl_divs.append(kl_div.item())
                            avg_kl_div = np.mean(kl_divs)
                            total_edges_evaluated += 1
                            print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                            if avg_kl_div < self.threshold:
                                self.circuit.remove_edge(edge)
                                edges_removed += 1
                                ablated_edges.append(edge)
                                kl_score = avg_kl_div
                                print(f"Edge removed.")

            # Compute final KL divergence for independent evaluation (add all hooks at the same time)
            if self.method == "independent":
                kl_score = self.get_final_performance(clean_logits=clean_logits, clean_caches=caches, ablated_edges=ablated_edges)

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Edges evaluated: {total_edges_evaluated}")
            print(f"Edges removed: {edges_removed}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")
            print(f"Final KL divergence: {kl_score:.6f}")

        elif self.mode == "node":
            print(f"Starting node ablation with threshold: {self.threshold}")
            # Iterate through nodes and ablate them
            nodes_removed = 0
            total_nodes_evaluated = 0
            kl_score = 0.0
            ablated_nodes = []
            for node in tqdm(ordered_nodes, desc="Ablating nodes"):
                if node.name in ['hook_resid_pre', 'hook_resid_post', 'hook_mlp_out', 'hook_z']:
                    if node.name == "hook_z":
                        print(f"Evaluating node: L{node.layer}-Head{node.head_idx}")
                    else:
                        print(f"Evaluating node: L{node.layer}-{node.name.split('_', 1)[1]}")
                    # Temporarily remove the node
                    kl_divs = []
                    for i, example in enumerate(self.dataset):
                        clean_tokens = example.clean_tokens
                        # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                        ablated_logits = self.run_with_node_ablation(clean_tokens, node, caches[i],
                                                                     ablated_nodes)
                        kl_div = kl_divergence(clean_logits[i], ablated_logits)
                        kl_divs.append(kl_div.item())
                    avg_kl_div = np.mean(kl_divs)
                    total_nodes_evaluated += 1
                    print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                    if avg_kl_div < self.threshold:
                        self.circuit.remove_node(node)
                        nodes_removed += 1
                        ablated_nodes.append(node)
                        kl_score = avg_kl_div
                        print(f"Node removed.")
            # Compute final KL divergence for independent evaluation (add all hooks at the same time)
            print(ablated_nodes)
            if self.method == "independent":
                kl_score = self.get_final_performance(clean_logits=clean_logits, clean_caches=caches, ablated_nodes=ablated_nodes)

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Nodes evaluated: {total_nodes_evaluated}")
            print(f"Nodes removed: {nodes_removed}")
            print(f"Final circuit nodes: {len(self.circuit.nodes)}")
            print(f"Final KL divergence: {kl_score:.6f}")

        return self.circuit

    def create_edge_ablation_hook(
            self,
            sender: Node,
            receiver: Node,
            clean_cache: Dict[str, torch.Tensor]
    ) -> Callable:
        """
        Create a hook function for edge ablation.
        For Pythia with parallel attention/MLP:
        - resid_post = resid_pre + attn_out + mlp_out
        - To ablate attention head output -> resid_post: resid_post = resid_post - specific head output
        """

        def ablation_hook(activation, hook):
            # Get the clean sender contribution
            if sender.full_activation not in clean_cache:
                return activation

            if sender.component_type == "attention":
                sender_act = clean_cache[sender.full_activation]
                sender_act = sender_act[:, :, sender.head_idx, :]
                W_O = self.model.blocks[sender.layer - 1].attn.W_O
                W_O_h = W_O[sender.head_idx]
                sender_contribution = sender_act @ W_O_h
            else:
                sender_contribution = clean_cache[sender.full_activation]

            # Subtract sender's contribution from the output
            # This effectively removes the edge from sender to receiver
            ablated_activation = activation - sender_contribution

            return ablated_activation

        return ablation_hook


    def run_with_edge_ablation(
            self,
            inputs: torch.Tensor,
            edge_to_ablate: Edge,
            clean_cache: Dict[str, torch.Tensor],
            ablated_edges: Optional[List[Edge]] = None
    ) -> torch.Tensor:
        """Run model with edge ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are edges previously ablated and, if so, re-add the hooks for them
        if ablated_edges and self.mode == "greedy":
            for edge in ablated_edges:
                hook = self.create_edge_ablation_hook(
                    edge.sender,
                    edge.receiver,
                    clean_cache
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(edge.receiver.full_activation, hook)

        # Create ablation hook
        ablation_hook = self.create_edge_ablation_hook(
            edge_to_ablate.sender,
            edge_to_ablate.receiver,
            clean_cache
        )

        # Register hook on the receiver
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(edge_to_ablate.receiver.full_activation, ablation_hook)

        # Run forward pass
        with torch.no_grad():
            ablated_logits = self.model(inputs)

        return ablated_logits


    def create_node_ablation_hook(self, node: Node) -> Callable:
        """Create a hook function for node ablation."""

        def ablation_hook(activation, hook):
            # For attention heads, zero out only that head's output
            if node.component_type == "attention":
                ablated_activation = activation
                ablated_activation[:, :, node.head_idx, :] = 0
            else:
                # Zero out the entire activation to ablate the node
                ablated_activation = torch.zeros_like(activation)
            return ablated_activation

        return ablation_hook


    def run_with_node_ablation(
            self,
            inputs: torch.Tensor,
            node_to_ablate: Node,
            clean_cache: Dict[str, torch.Tensor],
            ablated_nodes: Optional[List[Node]] = None
            ) -> torch.Tensor:
        """Run model with node ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are nodes previously ablated and, if so, re-add the hooks for them
        if ablated_nodes and self.mode == "greedy":
            for node in ablated_nodes:
                hook = self.create_node_ablation_hook(node)
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(node.full_activation, hook)

        # Create ablation hook
        ablation_hook = self.create_node_ablation_hook(node_to_ablate)

        # Register hook on the node
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(node_to_ablate.full_activation, ablation_hook)

        # Run forward pass
        with torch.no_grad():
            ablated_logits = self.model(inputs)

        return ablated_logits


    def get_final_performance(self, clean_logits, clean_caches, ablated_nodes: Optional[List[Node]] = None, ablated_edges: Optional[List[Edge]] = None):
        """Evaluate final performance after all edges have been ablated."""
        kl_divs = []
        for i, example in enumerate(self.dataset):

            # Clear previous hooks
            self.model.reset_hooks()

            if ablated_edges:
                for edge in ablated_edges:
                    hook = self.create_edge_ablation_hook(
                        edge.sender,
                        edge.receiver,
                        clean_caches[i]
                    )
                    if hasattr(self.model, 'add_hook'):
                        self.model.add_hook(edge.receiver.full_activation, hook)

                with torch.no_grad():
                    ablated_logits = self.model(example.clean_tokens)
                kl_div = kl_divergence(clean_logits[i], ablated_logits)
                kl_divs.append(kl_div.item())

            if ablated_nodes:
                for node in ablated_nodes:
                    hook = self.create_node_ablation_hook(node)
                    if hasattr(self.model, 'add_hook'):
                        self.model.add_hook(node.full_activation, hook)

                with torch.no_grad():
                    ablated_logits = self.model(example.clean_tokens)
                kl_div = kl_divergence(clean_logits[i], ablated_logits)
                kl_divs.append(kl_div.item())
        avg_kl_div = np.mean(kl_divs)
        return avg_kl_div



