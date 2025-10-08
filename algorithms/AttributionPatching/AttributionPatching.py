from collections import defaultdict
from algorithms.ACDC.utils import *
from utilities.evaluation import logits_to_logit_diff, factuality_nll_metric
import numpy as np
from tqdm import tqdm
from tasks import IOI, Induction, Factuality
from utilities.ComputationalGraph import ComputationalGraph, Node, Edge, build_computational_graph
import torch


class AttributionPatching:
    """
    Attribution Patching Algorithm
    Gradient-based approximation for finding important components in circuits.
    """

    def __init__(self, model, model_name, task: str = "IOI", topic: Optional = None, target: str = "edge",
                 threshold: float = 0.05):

        self.model = model
        self.model_name = model_name
        self.task_name = task
        self.threshold = threshold
        self.device = model.cfg.device
        self.target = target

        # Initialize graphs
        self.full_graph = None
        self.circuit = None

        # Cache for model activations and logits
        self.clean_caches = []
        self.clean_logits = []
        self.corrupted_caches = []

        # Attribution scores storage
        self.edge_attributions = {}
        self.node_attributions = {}

        # Create dataset based on the task
        if task == "IOI":
            print("Building IOI dataset...")
            dataset_builder = IOI.IOIDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=10)
            self.metric = logits_to_logit_diff
        elif task == "induction":
            print("Building Induction dataset...")
            dataset_builder = Induction.InductionDatasetBuilder(model)
            self.dataset = dataset_builder.build_dataset(num_samples=10)
        elif task == "Factuality":
            if topic is None:
                raise ValueError("Topic must be specified for Factuality task.")
            print("Building Factuality dataset...")
            dataset_builder = Factuality.FactualityDatasetBuilder(model, topic=topic)
            self.dataset = dataset_builder.build_dataset()
            self.metric = factuality_nll_metric
            # Keep only the first 10 examples for faster testing
            self.dataset = self.dataset[:10]


    def discover_circuit(self, threshold_percentile=10):
        """ Main mode to perform circuit discovery using edge attribution patching. """

        print(f"Building computational graph for {self.model_name}...")
        self.full_graph = build_computational_graph(self.model, self.model_name)
        self.circuit = self.full_graph.copy()
        ordered_nodes = self.circuit.topological_sort()

        print(f"Total nodes: {len(ordered_nodes)}")
        print(f"Total edges: {len(self.circuit.edges)}")

        # Store attribution scores
        edge_attributions = {edge: [] for edge in self.full_graph.edges}

        for example in tqdm(self.dataset, desc="Computing edge attributions..."):
            example_attributions = self.compute_example_attributions(example)
            for k, v in example_attributions.items():
                edge_attributions[k].append(v)

        edge_attributions = {k: np.mean(v) for k,v in edge_attributions.items()}
        self.edge_attributions = edge_attributions

        threshold = np.percentile(list(edge_attributions.values()), threshold_percentile)

        if self.target == "edge":
            # Sort edges by attribution score
            sorted_edges = sorted(
                self.edge_attributions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Keep edges above threshold
            removed_edges = 0
            total_edges = len(sorted_edges)

            for edge, attribution in sorted_edges:
                if attribution < self.threshold:
                    self.circuit.remove_edge(edge)
                    if edge.sender.name == "hook_z":
                        print(
                            f"Removed edge: L{edge.sender.layer}-Head{edge.sender.head_idx} -> L{edge.receiver.layer}-{edge.receiver.name.split('_', 1)[1]}")
                    else:
                        print(
                            f"Removed edge: L{edge.sender.layer}-{edge.sender.name.split('_', 1)[1]} -> L{edge.receiver.layer}-{edge.receiver.name.split('_', 1)[1]}")
                    removed_edges += 1

            print(f"\nAttribution Patching complete!")
            print(f"Total edges evaluated: {total_edges}")
            print(f"Edges removed (below threshold {self.threshold}): {removed_edges}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        elif self.target == "node":
            # Sort nodes by attribution score
            sorted_nodes = sorted(
                self.node_attributions.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Remove nodes above threshold and their edges
            removed_nodes = 0
            for node, attribution in sorted_nodes:
                if attribution < self.threshold:
                    self.circuit.remove_node(node)
                    print(f"Removed node {node.full_activation}")
                    removed_nodes += 1

            print(f"\nAttribution Patching complete!")
            print(f"Total nodes evaluated: {len(sorted_nodes)}")
            print(f"Nodes removed (below threshold {self.threshold}): {removed_nodes}")
            print(f"Final circuit edges: {len(self.circuit.edges)}")

        # Print attribution scores
        for k, v in self.edge_attributions.items():
            print(f"{k}: {v}")

        return self.circuit
    
    def compute_example_attributions(self, example):
        """
        Compute per-edge attributions for a single example.
        Uses backward hooks to capture gradients.
        """

        # Get clean forward activations and backward gradients (w.r.t. activations)

        # Get clean forward activations and backward gradients
        clean_value, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(example)

        # Get corrupted forward activations only (no grads required)
        corrupted_cache = self.get_corrupted_cache(example)

        # Compute per-edge attributions
        attributions = {}
        for edge in self.full_graph.edges:
            sender_name = edge.sender.full_activation
            receiver_name = edge.receiver.full_activation
            if "embed" in sender_name:
                continue
            # get forward activation (clean) and backward grad (clean)
            if sender_name not in clean_cache or sender_name not in clean_grad_cache or sender_name not in corrupted_cache:
                # no activation or no grad captured => attribution 0
                attributions[edge] = 0.0
                print(f"missing hook for {sender_name}")
                continue

            # get tensors
            e_clean = clean_cache[sender_name]
            e_corr = corrupted_cache[sender_name]
            grad_e = clean_grad_cache[sender_name]

            # Handle single attention heads activations - shape (batch, seq, n_heads, head_dim) for hook_z
            if "hook_z" in sender_name and hasattr(edge.sender, "head_idx"):
                h = edge.sender.head_idx
                e_clean_h = e_clean[..., h, :]
                e_corr_h = e_corr[..., h, :]
                grad_h = grad_e[..., h, :]
            else:
                # node-level activation (no head dimension)
                e_clean_h = e_clean
                e_corr_h = e_corr
                grad_h = grad_e

            # Ensure shapes align - flatten and compute dot product
            if e_clean_h.shape != e_corr_h.shape or e_clean_h.shape != grad_h.shape:
                # try broadcasting subtraction, then flatten
                try:
                    delta = (e_clean_h - e_corr_h).reshape(-1)
                    grad_flat = grad_h.reshape(-1)
                except Exception:
                    # as fallback convert to float zero attribution
                    attributions[edge] = 0.0
                    continue
            else:
                delta = (e_clean_h - e_corr_h).reshape(-1)
                grad_flat = grad_h.reshape(-1)

            # compute dot and absolute value
            try:
                dot = float(torch.dot(delta.to(self.device), grad_flat.to(self.device)).item())
                attr_score = abs(dot)
            except Exception:
                # numerical fallback
                attr_score = float(torch.sum((delta * grad_flat)).item())
                attr_score = abs(attr_score)

            attributions[edge] = attr_score

        return attributions

    def get_cache_fwd_and_bwd(self, example):
        """
        Registers forward and backward hooks, runs forward and backward on tokens,
        and returns (scalar_value, forward_cache_dict, backward_grad_cache_dict).
        """
        self.model.reset_hooks()
        needed_hooks = ["hook_z", "resid_pre", "resid_mid", "resid_post", "mlp_out"]

        forward_cache = {}
        backward_cache = {}

        def forward_cache_hook(act, hook):
            # store a detached copy of the forward activation
            if any(sub in hook.name for sub in needed_hooks):
                forward_cache[hook.name] = act.detach()

        def backward_cache_hook(grad, hook):
            # backward hook gets the gradient wrt the activation
            if any(sub in hook.name for sub in needed_hooks):
                backward_cache[hook.name] = grad.detach()

        for node in self.full_graph.nodes:
            self.model.add_hook(node.full_activation, forward_cache_hook, "fwd")
            self.model.add_hook(node.full_activation, backward_cache_hook, "bwd")

        # forward -> metric -> backward
        out = self.model(example.clean_tokens.to(self.device))
        value = self.metric(out, example, self.model)
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=example.clean_tokens.device, dtype=torch.float32)
        value.backward()

        self.model.reset_hooks()
        return value.item(), forward_cache, backward_cache

    def get_corrupted_cache(self, example):
        self.model.reset_hooks()
        needed_hooks = ["hook_z", "resid_pre", "resid_mid", "resid_post", "mlp_out"]
        corrupted_cache = {}

        def forward_cache_hook(act, hook):
            if any(sub in hook.name for sub in needed_hooks):
                corrupted_cache[hook.name] = act.detach()

        for node in self.full_graph.nodes:
            self.model.add_hook(node.full_activation, forward_cache_hook, "fwd")

        with torch.no_grad():
            _ = self.model(example.corrupted_tokens.to(self.device))

        self.model.reset_hooks()
        return corrupted_cache


