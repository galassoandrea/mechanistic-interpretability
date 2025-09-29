from collections import defaultdict
from algorithms.ACDC.utils import *
from utilities.evaluation import kl_divergence
import numpy as np
from tqdm import tqdm
from tasks import IOI, Induction, Factuality
from utilities.ComputationalGraph import ComputationalGraph, Node, Edge

class ACDC:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    """

    def __init__(self, model, model_name, task: str = "IOI", target: str = "edge", mode: str = "greedy",
                 method: str = "pruning", threshold: float = 0.1):

        self.model = model
        self.model_name = model_name
        self.task_name = task
        self.threshold = threshold
        self.device = model.cfg.device
        self.target = target
        self.mode = mode
        self.method = method

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations and logits
        self.clean_caches = []
        self.clean_logits = []
        self.corrupted_caches = []

        # Create dataset based on the task
        if task == "IOI":
            print("Building IOI dataset...")
            dataset_builder = IOI.IOIDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=10)
        elif task == "induction":
            print("Building Induction dataset...")
            dataset_builder = Induction.InductionDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=10)
        elif task == "Factuality":
            print("Building Factuality dataset...")
            dataset_builder = Factuality.FactualityDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset()
            # Keep only the first 10 examples for faster testing
            self.dataset = self.dataset[:10]

    def build_computational_graph(self):
        """Build the full computational graph of the model."""
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
        if "pythia" in self.model_name:
            graph = self.create_pythia_edges(graph, nodes_by_layer)
        elif "gemma" in self.model_name:
            graph = self.create_gemma_edges(graph, nodes_by_layer)
        return graph

    def create_pythia_edges(self, graph, nodes_by_layer):
        # Create edges following pythia model architecture
        max_layer = self.model.cfg.n_layers
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
                        if is_output_activation(attn_head_node.full_activation, "attention"):
                            edge = Edge(sender=attn_head_node, receiver=resid_post)
                            graph.add_edge(edge)
                # MLP output -> resid_post
                if "mlp" in current_layer_nodes:
                    for mlp_node in current_layer_nodes["mlp"]:
                        # Connect MLP output nodes to resid_post
                        if is_output_activation(mlp_node.full_activation, "mlp"):
                            edge = Edge(sender=mlp_node, receiver=resid_post)
                            graph.add_edge(edge)
        return graph

    def create_gemma_edges(self, graph, nodes_by_layer):
        # Create edges following gemma model architecture
        max_layer = self.model.cfg.n_layers
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
                    if is_output_activation(attn_head_node.full_activation, "attention"):
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
                    if is_output_activation(mlp_node.full_activation, "mlp"):
                        edge = Edge(sender=mlp_node, receiver=resid_post)
                        graph.add_edge(edge)
            # resid_mid -> resid_post (skip connection)
            if resid_mid and resid_post:
                edge = Edge(sender=resid_mid, receiver=resid_post)
                graph.add_edge(edge)
        return graph

    def discover_circuit(self):
        """ Main mode to perform circuit discovery using edge pruning. """

        print(f"Building computational graph for {self.model_name}...")
        self.full_graph = self.build_computational_graph()
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        # Collect clean and corrupted reference outputs and caches
        for example in tqdm(self.dataset, desc="Collecting reference outputs"):
            with torch.no_grad():
                clean_inputs = example.clean_tokens
                # Cache activations and keep only needed ones
                l_clean, c_clean = self.model.run_with_cache(clean_inputs, return_type="logits")
                c_clean = filter_hooks(c_clean, self.model_name, self.model.cfg.n_layers)
                l_clean = l_clean.cpu()
                self.clean_logits.append(l_clean)
                self.clean_caches.append(c_clean)
                if self.method == "patching":
                    # Also collect corrupted outputs for activation patching
                    corrupted_inputs = example.corrupted_tokens
                    _, c_corr = self.model.run_with_cache(corrupted_inputs, return_type="logits")
                    c_corr = filter_hooks(c_corr, self.model_name, self.model.cfg.n_layers)
                    self.corrupted_caches.append(c_corr)
        # Clear gpu
        torch.cuda.empty_cache()

        if "pythia" in self.model_name:
            self.circuit = self.discover_circuit_pythia(ordered_nodes)
        elif "gemma" in self.model_name:
            self.circuit = self.discover_circuit_gemma(ordered_nodes)
        return self.circuit

    def discover_circuit_pythia(self, ordered_nodes):
        ablated_edges = []
        ablated_nodes = []
        if self.target == "edge":
            print(f"Starting edge evaluation with threshold: {self.threshold}")
            # Iterate through nodes and prune edges
            edges_removed = 0
            total_edges_evaluated = 0
            for receiver in tqdm(ordered_nodes, desc="Evaluating edges"):
                if receiver.name == "hook_resid_post":
                    senders = self.full_graph.get_senders(receiver)
                    for sender in senders:
                        if sender.name == "hook_z":
                            print(
                                f"Evaluating edge: L{sender.layer}-Head{sender.head_idx} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                        else:
                            print(
                                f"Evaluating edge: L{sender.layer}-{sender.name.split('_', 1)[1]} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                        edge = Edge(sender, receiver)
                        # Temporarily remove the edge
                        kl_divs = []
                        for i, example in enumerate(self.dataset):
                            clean_tokens = example.clean_tokens
                            # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                            if self.method == "patching":
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    corrupted_cache=self.corrupted_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            else:
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            kl_div = kl_divergence(self.clean_logits[i].to(self.device), ablated_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        total_edges_evaluated += 1
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_edge(edge)
                            edges_removed += 1
                            ablated_edges.append(edge)
                            print(f"Edge removed.")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Edges evaluated: {total_edges_evaluated}")
            print(f"Edges removed: {edges_removed}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        elif self.target == "node":
            print(f"Starting node evaluation with threshold: {self.threshold}")
            # Iterate through nodes and ablate them
            nodes_removed = 0
            total_nodes_evaluated = 0
            for node in tqdm(ordered_nodes, desc="Evaluating nodes"):
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
                        if self.method == "patching":
                            ablated_logits = self.run_with_node_patching(
                                inputs=clean_tokens,
                                node_to_ablate=node,
                                corrupted_cache=self.corrupted_caches[i],
                                ablated_nodes=ablated_nodes
                            )
                        else:
                            ablated_logits = self.run_with_node_patching(
                                inputs=clean_tokens,
                                node_to_ablate=node,
                                ablated_nodes=ablated_nodes
                            )
                        kl_div = kl_divergence(self.clean_logits[i].to(self.device), ablated_logits)
                        kl_divs.append(kl_div.item())
                    avg_kl_div = np.mean(kl_divs)
                    total_nodes_evaluated += 1
                    print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                    if avg_kl_div < self.threshold:
                        self.circuit.remove_node(node)
                        nodes_removed += 1
                        ablated_nodes.append(node)
                        print(f"Node removed.")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Nodes evaluated: {total_nodes_evaluated}")
            print(f"Nodes removed: {nodes_removed}")
            print(f"Final circuit nodes: {len(self.circuit.nodes)}")

        # Clear gpu
        torch.cuda.empty_cache()

        # Compute final KL divergence (add all hooks at the same time)
        kl_score = get_final_performance(
            model=self.model,
            method=self.method,
            dataset=self.dataset,
            task_name=self.task_name,
            clean_logits=self.clean_logits,
            clean_caches=self.clean_caches,
            corrupted_caches=self.corrupted_caches,
            ablated_nodes=ablated_nodes,
            ablated_edges=ablated_edges
        )
        print(f"Final KL divergence: {kl_score:.6f}")

        return self.circuit

    def discover_circuit_gemma(self, ordered_nodes):
        ablated_edges = []
        ablated_nodes = []
        if self.target == "edge":
            print(f"Starting edge evaluation with threshold: {self.threshold}")
            # Iterate through nodes and prune edges
            edges_removed = 0
            total_edges_evaluated = 0
            for receiver in tqdm(ordered_nodes, desc="Evaluating edges"):
                if receiver.name == "hook_resid_post":
                    senders = self.full_graph.get_senders(receiver)
                    for sender in senders:
                        print(
                            f"Evaluating edge: L{sender.layer}-{sender.name.split('_', 1)[1]} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                        edge = Edge(sender, receiver)
                        # Temporarily remove the edge
                        kl_divs = []
                        for i, example in enumerate(self.dataset):
                            clean_tokens = example.clean_tokens
                            # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                            if self.method == "patching":
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    corrupted_cache=self.corrupted_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            else:
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            kl_div = kl_divergence(self.clean_logits[i].to(self.device), ablated_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        total_edges_evaluated += 1
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_edge(edge)
                            edges_removed += 1
                            ablated_edges.append(edge)
                            print(f"Edge removed.")
                elif receiver.name == "hook_resid_mid":
                    senders = self.full_graph.get_senders(receiver)
                    for sender in senders:
                        if sender.name == "hook_z":
                            print(
                                f"Evaluating edge: L{sender.layer}-Head{sender.head_idx} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                        else:
                            print(
                                f"Evaluating edge: L{sender.layer}-{sender.name.split('_', 1)[1]} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                        edge = Edge(sender, receiver)
                        # Temporarily remove the edge
                        kl_divs = []
                        for i, example in enumerate(self.dataset):
                            clean_tokens = example.clean_tokens
                            # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                            if self.method == "patching":
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    corrupted_cache=self.corrupted_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            else:
                                ablated_logits = self.run_with_edge_patching(
                                    inputs=clean_tokens,
                                    edge_to_ablate=edge,
                                    clean_cache=self.clean_caches[i],
                                    ablated_edges=ablated_edges
                                )
                            kl_div = kl_divergence(self.clean_logits[i].to(self.device), ablated_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        total_edges_evaluated += 1
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_edge(edge)
                            edges_removed += 1
                            ablated_edges.append(edge)
                            print(f"Edge removed.")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Edges evaluated: {total_edges_evaluated}")
            print(f"Edges removed: {edges_removed}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        elif self.target == "node":
            print(f"Starting node evaluation with threshold: {self.threshold}")
            # Iterate through nodes and ablate them
            nodes_removed = 0
            total_nodes_evaluated = 0
            for node in tqdm(ordered_nodes, desc="Evaluating nodes"):
                if node.name in ['hook_resid_pre', 'hook_resid_mid', 'hook_resid_post', 'hook_mlp_out', 'hook_z']:
                    if node.name == "hook_z":
                        print(f"Evaluating node: L{node.layer}-Head{node.head_idx}")
                    else:
                        print(f"Evaluating node: L{node.layer}-{node.name.split('_', 1)[1]}")
                    # Temporarily remove the node
                    kl_divs = []
                    for i, example in enumerate(self.dataset):
                        clean_tokens = example.clean_tokens
                        # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                        if self.method == "patching":
                            ablated_logits = self.run_with_node_patching(
                                inputs=clean_tokens,
                                node_to_ablate=node,
                                corrupted_cache=self.corrupted_caches[i],
                                ablated_nodes=ablated_nodes
                            )
                        else:
                            ablated_logits = self.run_with_node_patching(
                                inputs=clean_tokens,
                                node_to_ablate=node,
                                ablated_nodes=ablated_nodes
                            )
                        kl_div = kl_divergence(self.clean_logits[i].to(self.device), ablated_logits)
                        kl_divs.append(kl_div.item())
                    avg_kl_div = np.mean(kl_divs)
                    total_nodes_evaluated += 1
                    print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                    if avg_kl_div < self.threshold:
                        self.circuit.remove_node(node)
                        nodes_removed += 1
                        ablated_nodes.append(node)
                        print(f"Node removed.")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Nodes evaluated: {total_nodes_evaluated}")
            print(f"Nodes removed: {nodes_removed}")
            print(f"Final circuit nodes: {len(self.circuit.nodes)}")

        # Clear gpu
        torch.cuda.empty_cache()

        # Compute final KL divergence (add all hooks at the same time)
        kl_score = get_final_performance(
            model=self.model,
            method=self.method,
            dataset=self.dataset,
            task_name=self.task_name,
            clean_logits=self.clean_logits,
            clean_caches=self.clean_caches,
            corrupted_caches=self.corrupted_caches,
            ablated_nodes=ablated_nodes,
            ablated_edges=ablated_edges
        )
        print(f"Final KL divergence: {kl_score:.6f}")

        return self.circuit

    def run_with_edge_patching(
            self,
            inputs: torch.Tensor,
            edge_to_ablate: Edge,
            clean_cache: Dict[str, torch.Tensor],
            corrupted_cache: Optional[Dict[str, torch.Tensor]] = None,
            ablated_edges: Optional[List[Edge]] = None
    ) -> torch.Tensor:
        """Run model with edge ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are edges previously ablated and, if so, re-add the hooks for them
        if ablated_edges and self.mode == "greedy":
            for edge in ablated_edges:
                hook = create_edge_patching_hook(
                    self.model,
                    self.method,
                    edge.sender,
                    edge.receiver,
                    clean_cache,
                    corrupted_cache
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(edge.receiver.full_activation, hook)

        # Create ablation hook
        patching_hook = create_edge_patching_hook(
            self.model,
            self.method,
            edge_to_ablate.sender,
            edge_to_ablate.receiver,
            clean_cache,
            corrupted_cache
        )

        # Register hook on the receiver
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(edge_to_ablate.receiver.full_activation, patching_hook)

        # Run forward pass
        with torch.no_grad():
            ablated_logits = self.model(inputs)

        return ablated_logits

    def run_with_node_patching(
            self,
            inputs: torch.Tensor,
            node_to_ablate: Node,
            corrupted_cache: Optional[Dict[str, torch.Tensor]] = None,
            ablated_nodes: Optional[List[Node]] = None
    ) -> torch.Tensor:
        """Run model with node ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are nodes previously ablated and, if so, re-add the hooks for them
        if ablated_nodes and self.mode == "greedy":
            for node in ablated_nodes:
                hook = create_node_patching_hook(
                    self.model,
                    self.method,
                    node,
                    corrupted_cache
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(node.full_activation, hook)

        # Create ablation hook
        patching_hook = create_node_patching_hook(
                    self.model,
                    self.method,
                    node_to_ablate,
                    corrupted_cache
                )

        # Register hook on the node
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(node_to_ablate.full_activation, patching_hook)

        # Run forward pass
        with torch.no_grad():
            ablated_logits = self.model(inputs)

        return ablated_logits

