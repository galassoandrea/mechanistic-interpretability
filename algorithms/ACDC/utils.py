import json
import os

import torch
from typing import Dict, List, Optional, Callable
from utilities.evaluation import kl_divergence, evaluate_factuality
import numpy as np
from utilities.ComputationalGraph import Node, Edge

def create_edge_patching_hook(
        method,
        clean_sender_contribution: torch.Tensor,
        corrupted_sender_contribution: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for edge pruning/patching."""

    def patching_hook(activation, hook):
        # Subtract sender's contribution from the output
        if method == "patching":
            patched_activation = activation - clean_sender_contribution + corrupted_sender_contribution
        else:
            patched_activation = activation - clean_sender_contribution

        return patched_activation

    return patching_hook

def create_node_patching_hook(
        method,
        node: Node,
        clean_sender_contribution: Optional[torch.Tensor] = None,
        corrupted_sender_contribution: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for node ablation/patching."""

    def patching_hook(activation, hook):
        # Check the number of dimensions of the activation
        if activation.dim() == 4:
            # Attention layer (hook_z)
            patched_activation = activation
            if method == "patching":
                patched_activation[:,:,node.head_idx,:] = activation[:,:,node.head_idx,:] - clean_sender_contribution + corrupted_sender_contribution
            else:
                patched_activation[:,:,node.head_idx,:] = 0
        else:
            if method == "patching":
                patched_activation = activation - clean_sender_contribution + corrupted_sender_contribution
            else:
                patched_activation = 0
        return patched_activation

    return patching_hook

def add_all_hooks(
        model,
        i,
        clean_sender_contributions,
        ablated_nodes: Optional[List[Node]] = None,
        ablated_edges: Optional[List[Edge]] = None):
    """Add hooks for all activations in the circuit."""
    if ablated_edges != [] and ablated_edges is not None:
        for edge in ablated_edges:
            sender_id = get_node_id(edge.sender)
            hook = create_edge_patching_hook(
                method="pruning",
                clean_sender_contribution=clean_sender_contributions[(sender_id,i)]
            )
            if hasattr(model, 'add_hook'):
                model.add_hook(edge.receiver.full_activation, hook)
    if ablated_nodes != [] and ablated_nodes is not None:
        for node in ablated_nodes:
            sender_id = get_node_id(node)
            hook = create_node_patching_hook(
                method="pruning",
                node=node,
                clean_sender_contribution=clean_sender_contributions[(sender_id, i)]
            )
            if hasattr(model, 'add_hook'):
                model.add_hook(node.full_activation, hook)

def get_final_performance(
        model,
        dataset,
        task_name,
        clean_logits,
        clean_sender_contributions,
        ablated_nodes: Optional[List[Node]] = None,
        ablated_edges: Optional[List[Edge]] = None
):
    """Get final performance after all edges/nodes have been evaluated."""
    kl_divs = []
    logits = []
    labels = []

    for i, example in enumerate(dataset):
        # Clear previous hooks
        model.reset_hooks()

        # Add all hooks for patched edges/nodes
        add_all_hooks(model, i, clean_sender_contributions,
                       ablated_nodes, ablated_edges)

        with torch.no_grad():
            ablated_logits = model(example.clean_tokens)
        kl_div = kl_divergence(clean_logits[i].to(model.cfg.device), ablated_logits)
        kl_divs.append(kl_div.item())
        if task_name == "Factuality":
            logits.append(ablated_logits)
            labels.append(example.label)
    avg_kl_div = np.mean(kl_divs)
    # Free some memory
    model.reset_hooks()
    del clean_sender_contributions
    del clean_logits
    logits = [t.cpu() for t in logits]
    torch.cuda.empty_cache()
    if task_name == "Factuality":
        # Compute metrics for factuality evaluation
        evaluate_factuality(logits, labels, model)
    return avg_kl_div

def filter_hooks(cache, model_name, layers):
    all_hooks = []
    if "pythia" in model_name:
        hook_list = ["attn.hook_z", "hook_resid_pre", "hook_resid_post", "hook_mlp_out"]
    else:
        hook_list = ["attn.hook_z", "hook_resid_pre", "hook_resid_mid", "hook_resid_post", "hook_mlp_out"]
    for hook_name in hook_list:
        hooks = [f"blocks.{l}.{hook_name}" for l in range(layers)]
        all_hooks.extend(hooks)
    return {k: v.detach().cpu() for k, v in cache.items() if k in all_hooks}

def get_node_id(node: Node) -> str:
    """Get the ID string for a given node."""
    if node.component_type == "attention":
        node_id = f"L{node.layer}-Head{node.head_idx}"
    else:
        node_id = f"L{node.layer}-{node.name.split('_', 1)[1]}"
    return node_id