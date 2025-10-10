import json
import os
from algorithms.ACDC.utils import *
from utilities.evaluation import kl_divergence
import numpy as np
from tqdm import tqdm
from tasks import IOI, Induction, Factuality
from utilities.ComputationalGraph import Node, Edge, build_computational_graph

class ACDC:
    """
    Automatic Circuit Discovery (ACDC) Algorithm
    for finding minimal circuits responsible for specific tasks.
    """

    def __init__(self, model, model_name, task: str = "IOI", topic: Optional = None, target: str = "edge", mode: str = "greedy",
                 method: str = "pruning", threshold: float = 0.1):

        self.model = model
        self.model_name = model_name
        self.task_name = task
        self.topic = topic
        self.threshold = threshold
        self.device = model.cfg.device
        self.target = target
        self.mode = mode
        self.method = method

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations and logits
        self.clean_logits = []
        self.clean_sender_contributions = {}
        self.corrupted_sender_contributions = {}
        self.ablated_edges = []
        self.ablated_nodes = []

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
            if self.topic is None:
                raise ValueError("Topic must be specified for Factuality task.")
            print("Building Factuality dataset...")
            dataset_builder = Factuality.FactualityDatasetBuilder(model, topic=self.topic)
            self.dataset = dataset_builder.build_dataset()
            # Keep only the first 10 examples for faster testing
            self.dataset = self.dataset[:10]

    def discover_circuit(self):
        """ Main mode to perform circuit discovery using edge pruning. """

        print(f"Building computational graph for {self.model_name}...")
        self.full_graph = build_computational_graph(self.model, self.model_name)
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        clean_caches = []
        corrupted_caches = []

        # Collect clean and corrupted reference outputs and caches
        for example in tqdm(self.dataset, desc="Collecting reference outputs"):
            with torch.no_grad():
                clean_inputs = example.clean_tokens
                # Cache activations and keep only needed ones
                l_clean, c_clean = self.model.run_with_cache(clean_inputs, return_type="logits")
                c_clean = filter_hooks(c_clean, self.model_name, self.model.cfg.n_layers)
                l_clean = l_clean.cpu()
                self.clean_logits.append(l_clean)
                clean_caches.append(c_clean)
                if self.method == "patching":
                    # Also collect corrupted outputs for activation patching
                    corrupted_inputs = example.corrupted_tokens
                    _, c_corr = self.model.run_with_cache(corrupted_inputs, return_type="logits")
                    c_corr = filter_hooks(c_corr, self.model_name, self.model.cfg.n_layers)
                    corrupted_caches.append(c_corr)
                    
        # Precompute node contributions for all examples
        self.precompute_node_contributions(clean_caches, corrupted_caches)
        # Clear memory from caches since we don't need them anymore
        del clean_caches
        del corrupted_caches
        # Clear gpu
        torch.cuda.empty_cache()

        # Run circuit discovery based on the model
        if "pythia" in self.model_name:
            self.discover_circuit_parallel(ordered_nodes)
        else:
            self.discover_circuit_sequential(ordered_nodes)

        # Clear gpu
        torch.cuda.empty_cache()

        # Compute final KL divergence (add all hooks at the same time)
        kl_score = get_final_performance(
            model=self.model,
            dataset=self.dataset,
            task_name=self.task_name,
            clean_logits=self.clean_logits,
            clean_sender_contributions=self.clean_sender_contributions,
            ablated_nodes=self.ablated_nodes,
            ablated_edges=self.ablated_edges
        )
        print(f"Final KL divergence: {kl_score:.6f}")

        save_circuit(self.model_name, self.topic, self.target, self.ablated_nodes, self.ablated_edges)

        return self.circuit

    def discover_circuit_parallel(self, ordered_nodes):

        if self.target == "edge":
            print(f"Starting edge evaluation with threshold: {self.threshold}")
            # Iterate through nodes and prune edges
            edges_removed_this_iter = 1
            total_edges_removed = 0
            iteration = 0
            while edges_removed_this_iter > 0:
                edges_removed_this_iter = 0
                iteration += 1
                print(f"--- Starting iteration {iteration} ---")
                for receiver in tqdm(ordered_nodes, desc="Evaluating edges"):
                    if receiver.name == "hook_resid_post":
                        senders = self.circuit.get_senders(receiver).copy()
                        for sender in senders:
                            sender_id = get_node_id(sender)
                            print(f"Evaluating edge: {sender_id} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                            edge = Edge(sender, receiver)
                            # Temporarily remove the edge
                            kl_divs = []
                            for i, example in enumerate(self.dataset):
                                clean_tokens = example.clean_tokens
                                # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                                if self.method == "patching":
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        corrupted_sender_contributions=self.corrupted_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                else:
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                kl_div = kl_divergence(self.clean_logits[i].to(self.device), patched_logits)
                                kl_divs.append(kl_div.item())
                            avg_kl_div = np.mean(kl_divs)
                            print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                            if avg_kl_div < self.threshold:
                                self.circuit.remove_edge(edge)
                                edges_removed_this_iter += 1
                                self.ablated_edges.append(edge)
                                # Only remove sender from ordered_nodes if it has no more receivers
                                if len(self.circuit.get_receivers(edge.sender)) == 0:
                                    ordered_nodes.remove(edge.sender)
                                print(f"Edge removed.")
                total_edges_removed += edges_removed_this_iter
                print(f"Edges removed this iteration: {edges_removed_this_iter}")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Edges removed: {total_edges_removed}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        elif self.target == "node":
            print(f"Starting node evaluation with threshold: {self.threshold}")
            # Iterate through nodes and ablate them
            nodes_removed_this_iter = 1
            total_nodes_removed = 0
            while nodes_removed_this_iter > 0:
                nodes_removed_this_iter = 0
                for node in tqdm(ordered_nodes, desc="Evaluating nodes"):
                    if node.name in ['hook_resid_pre', 'hook_resid_post', 'hook_mlp_out', 'hook_z']:
                        node_id = get_node_id(node)
                        print(f"Evaluating node: {node_id}")
                        # Temporarily remove the node
                        kl_divs = []
                        for i, example in enumerate(self.dataset):
                            clean_tokens = example.clean_tokens
                            if self.method == "patching":
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens,
                                    i=i,
                                    node_to_patch=node,
                                    corrupted_sender_contributions=self.corrupted_sender_contributions,
                                    ablated_nodes=self.ablated_nodes
                                )
                            else:
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens,
                                    i=i,
                                    node_to_patch=node,
                                    ablated_nodes=self.ablated_nodes
                                )
                            kl_div = kl_divergence(self.clean_logits[i].to(self.device), patched_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_node(node)
                            ordered_nodes.remove(node)
                            nodes_removed_this_iter += 1
                            self.ablated_nodes.append(node)
                            print(f"Node removed.")
                total_nodes_removed += nodes_removed_this_iter
                print(f"Nodes removed this iteration: {nodes_removed_this_iter}")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Nodes removed: {total_nodes_removed}")
            print(f"Final circuit nodes: {len(self.circuit.nodes)}")

    def discover_circuit_sequential(self, ordered_nodes):
        if self.target == "edge":
            print(f"Starting edge evaluation with threshold: {self.threshold}")
            # Iterate through nodes and prune edges
            edges_removed_this_iter = 1
            total_edges_removed = 0
            while edges_removed_this_iter > 0:
                edges_removed_this_iter = 0
                for receiver in tqdm(ordered_nodes, desc="Evaluating edges"):
                    if receiver.name == "hook_resid_post":
                        senders = self.circuit.get_senders(receiver).copy()
                        for sender in senders:
                            sender_id = get_node_id(sender)
                            print(
                                f"Evaluating edge: {sender_id} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                            edge = Edge(sender, receiver)
                            # Temporarily remove the edge
                            kl_divs = []
                            for i, example in enumerate(self.dataset):
                                clean_tokens = example.clean_tokens
                                # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                                if self.method == "patching":
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        corrupted_sender_contributions=self.corrupted_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                else:
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                kl_div = kl_divergence(self.clean_logits[i].to(self.device), patched_logits)
                                kl_divs.append(kl_div.item())
                            avg_kl_div = np.mean(kl_divs)
                            print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                            if avg_kl_div < self.threshold:
                                self.circuit.remove_edge(edge)
                                total_edges_removed += 1
                                self.ablated_edges.append(edge)
                                # Only remove sender from ordered_nodes if it has no more receivers
                                if len(self.circuit.get_receivers(edge.sender)) == 0:
                                    ordered_nodes.remove(edge.sender)
                                print(f"Edge removed.")
                    elif receiver.name == "hook_resid_mid":
                        senders = self.circuit.get_senders(receiver).copy()
                        for sender in senders:
                            sender_id = get_node_id(sender)
                            print(f"Evaluating edge: {sender_id} -> L{receiver.layer}-{receiver.name.split('_', 1)[1]}")
                            edge = Edge(sender, receiver)
                            # Temporarily remove the edge
                            kl_divs = []
                            for i, example in enumerate(self.dataset):
                                clean_tokens = example.clean_tokens
                                # Ablate the edge by zeroing out the sender's contribution (not the whole activation) only on receiver
                                if self.method == "patching":
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        corrupted_sender_contributions=self.corrupted_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                else:
                                    patched_logits = self.run_with_edge_patching(
                                        inputs=clean_tokens,
                                        i=i,
                                        edge_to_patch=edge,
                                        clean_sender_contributions=self.clean_sender_contributions,
                                        ablated_edges=self.ablated_edges
                                    )
                                kl_div = kl_divergence(self.clean_logits[i].to(self.device), patched_logits)
                                kl_divs.append(kl_div.item())
                            avg_kl_div = np.mean(kl_divs)
                            print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                            if avg_kl_div < self.threshold:
                                self.circuit.remove_edge(edge)
                                edges_removed_this_iter += 1
                                self.ablated_edges.append(edge)
                                # Only remove sender from ordered_nodes if it has no more receivers
                                if len(self.circuit.get_receivers(edge.sender)) == 0:
                                    ordered_nodes.remove(edge.sender)
                                print(f"Edge removed.")
                total_edges_removed += edges_removed_this_iter
                print(f"Edges removed this iteration: {edges_removed_this_iter}")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Edges removed: {total_edges_removed}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        elif self.target == "node":
            print(f"Starting node evaluation with threshold: {self.threshold}")
            # Iterate through nodes and ablate them
            nodes_removed_this_iter = 1
            total_nodes_removed = 0
            while nodes_removed_this_iter > 0:
                nodes_removed_this_iter = 0
                for node in tqdm(ordered_nodes, desc="Evaluating nodes"):
                    if node.name in ['hook_resid_pre', 'hook_resid_mid', 'hook_resid_post', 'hook_mlp_out', 'hook_z']:
                        node_id = get_node_id(node)
                        print(f"Evaluating node: {node_id}")
                        # Temporarily remove the node
                        kl_divs = []
                        for i, example in enumerate(self.dataset):
                            clean_tokens = example.clean_tokens
                            if self.method == "patching":
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens,
                                    i=i,
                                    node_to_patch=node,
                                    corrupted_sender_contributions=self.corrupted_sender_contributions,
                                    ablated_nodes=self.ablated_nodes
                                )
                            else:
                                patched_logits = self.run_with_node_patching(
                                    inputs=clean_tokens,
                                    i=i,
                                    node_to_patch=node,
                                    ablated_nodes=self.ablated_nodes
                                )
                            kl_div = kl_divergence(self.clean_logits[i].to(self.device), patched_logits)
                            kl_divs.append(kl_div.item())
                        avg_kl_div = np.mean(kl_divs)
                        print(f"Avg KL Divergence = {avg_kl_div:.6f}")
                        if avg_kl_div < self.threshold:
                            self.circuit.remove_node(node)
                            ordered_nodes.remove(node)
                            nodes_removed_this_iter += 1
                            self.ablated_nodes.append(node)
                            print(f"Node removed.")
                total_nodes_removed += nodes_removed_this_iter
                print(f"Nodes removed this iteration: {nodes_removed_this_iter}")

            # Print summary of results
            print(f"\nCircuit discovery complete!")
            print(f"Nodes removed: {total_nodes_removed}")
            print(f"Final circuit nodes: {len(self.circuit.nodes)}")

    def run_with_edge_patching(
            self,
            inputs: torch.Tensor,
            i,
            edge_to_patch: Edge,
            clean_sender_contributions,
            corrupted_sender_contributions: Optional = None,
            ablated_edges: Optional[List[Edge]] = None
    ) -> torch.Tensor:
        """Run model with edge ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are edges previously pruned and, if so, re-add the hooks for them
        if ablated_edges and self.mode == "greedy":
            for edge in ablated_edges:
                sender_id = get_node_id(edge.sender)
                hook = create_edge_patching_hook(
                    method="pruning",
                    clean_sender_contribution=clean_sender_contributions[(sender_id,i)],
                    corrupted_sender_contribution=corrupted_sender_contributions[(sender_id,i)] if corrupted_sender_contributions else None
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(edge.receiver.full_activation, hook)

        # Create ablation hook
        sender_id = get_node_id(edge_to_patch.sender)
        patching_hook = create_edge_patching_hook(
            self.method,
            clean_sender_contributions[(sender_id, i)],
            corrupted_sender_contributions[(sender_id, i)] if corrupted_sender_contributions else None
        )

        # Register hook on the receiver
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(edge_to_patch.receiver.full_activation, patching_hook)

        # Run forward pass
        with torch.no_grad():
            patched_logits = self.model(inputs)

        return patched_logits

    def run_with_node_patching(
            self,
            inputs: torch.Tensor,
            i,
            node_to_patch: Node,
            corrupted_sender_contributions: Optional = None,
            ablated_nodes: Optional[List[Node]] = None
    ) -> torch.Tensor:
        """Run model with node ablation."""

        # Clear previous hooks
        self.model.reset_hooks()

        # In case of greedy evaluation, check if there are nodes previously ablated and, if so, re-add the hooks for them
        if ablated_nodes and self.mode == "greedy":
            for node in ablated_nodes:
                hook = create_node_patching_hook(
                    method="pruning",
                    node=node
                )
                if hasattr(self.model, 'add_hook'):
                    self.model.add_hook(node.full_activation, hook)

        # Create ablation hook
        node_id = get_node_id(node_to_patch)
        patching_hook = create_node_patching_hook(
                    self.method,
                    node_to_patch,
                    corrupted_sender_contributions[(node_id, i)] if corrupted_sender_contributions else None
                )

        # Register hook on the node
        if hasattr(self.model, 'add_hook'):
            self.model.add_hook(node_to_patch.full_activation, patching_hook)

        # Run forward pass
        with torch.no_grad():
            patched_logits = self.model(inputs)

        return patched_logits

    def precompute_node_contributions(self, clean_caches, corrupted_caches):
        """Precompute all node contributions for all examples."""

        for i, example in enumerate(self.dataset):
            for node in self.full_graph.nodes:
                # Skip embedding node
                if "hook_embed" in node.full_activation:
                    continue
                if node.component_type == "attention":
                    sender_act = clean_caches[i][node.full_activation]
                    clean_contribution = sender_act[:, :, node.head_idx, :].to(self.device)
                    if self.target == "edge":
                        W_O_h = self.model.blocks[node.layer - 1].attn.W_O[node.head_idx]
                        clean_contribution = clean_contribution @ W_O_h
                    node_id = f"L{node.layer}-Head{node.head_idx}"
                    self.clean_sender_contributions[(node_id, i)] = clean_contribution.to(self.device)
                    if self.method == "patching":
                        corrupted_sender_act = corrupted_caches[i][node.full_activation]
                        corrupted_contribution = corrupted_sender_act[:, :, node.head_idx, :].to(self.device)
                        if self.target == "edge":
                            W_O_h = self.model.blocks[node.layer - 1].attn.W_O[node.head_idx]
                            corrupted_contribution = corrupted_contribution @ W_O_h
                        self.corrupted_sender_contributions[(node_id, i)] = corrupted_contribution.to(self.device)
                else:
                    node_id = f"L{node.layer}-{node.name.split('_', 1)[1]}"
                    contribution = clean_caches[i][node.full_activation]
                    self.clean_sender_contributions[(node_id, i)] = contribution.to(self.device)
                    if self.method == "patching":
                        corrupted_contribution = corrupted_caches[i][node.full_activation]
                        self.corrupted_sender_contributions[(node_id, i)] = corrupted_contribution.to(self.device)


