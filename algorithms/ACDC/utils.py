import torch
from typing import Dict, List, Optional, Callable
from utilities.evaluation import kl_divergence, evaluate_factuality
import numpy as np
from utilities.ComputationalGraph import Node, Edge

def create_edge_patching_hook(
        model,
        method,
        sender: Node,
        receiver: Node,
        clean_cache: Dict[str, torch.Tensor],
        corrupted_cache: Optional[Dict[str, torch.Tensor]] = None
) -> Callable:
    """Create a hook function for edge pruning/patching."""

    def patching_hook(activation, hook):
        # Get the clean sender contribution
        if sender.full_activation not in clean_cache:
            return activation

        if method == "patching" and corrupted_cache:
            if sender.component_type == "attention":
                clean_sender_act = clean_cache[sender.full_activation]
                clean_sender_act = clean_sender_act[:, :, sender.head_idx, :].to(model.cfg.device)
                corrupted_sender_act = corrupted_cache[sender.full_activation]
                corrupted_sender_act = corrupted_sender_act[:, :, sender.head_idx, :].to(model.cfg.device)
                W_O = model.blocks[sender.layer - 1].attn.W_O
                W_O_h = W_O[sender.head_idx]
                clean_sender_contribution = clean_sender_act @ W_O_h
                corrupted_sender_contribution = corrupted_sender_act @ W_O_h
            else:
                clean_sender_contribution = clean_cache[sender.full_activation].to(model.cfg.device)
                corrupted_sender_contribution = corrupted_cache[sender.full_activation].to(model.cfg.device)
            # Subtract sender's contribution from the output
            patched_activation = activation - clean_sender_contribution + corrupted_sender_contribution
        else:
            if sender.component_type == "attention":
                sender_act = clean_cache[sender.full_activation]
                sender_act = sender_act[:, :, sender.head_idx, :].to(model.cfg.device)
                W_O = model.blocks[sender.layer - 1].attn.W_O
                W_O_h = W_O[sender.head_idx]
                sender_contribution = sender_act @ W_O_h
            else:
                sender_contribution = clean_cache[sender.full_activation].to(model.cfg.device)
            # Subtract sender's contribution from the output
            patched_activation = activation - sender_contribution

        return patched_activation

    return patching_hook

def create_node_patching_hook(
        model,
        method,
        node: Node,
        corrupted_cache: Optional[Dict[str, torch.Tensor]] = None
) -> Callable:
    """Create a hook function for node ablation/patching."""

    def patching_hook(activation, hook):
        if method == "patching" and corrupted_cache:
            # For attention heads, patch only that head's output
            if node.component_type == "attention":
                corr_act = corrupted_cache[node.full_activation]
                corr_act = corr_act[:, :, node.head_idx, :]
                patched_activation = activation
                patched_activation[:, :, node.head_idx, :] = corr_act
            else:
                patched_activation = corrupted_cache[node.full_activation]
            return patched_activation.to(model.cfg.device)
        else:
            # For attention heads, patch only that head's output
            if node.component_type == "attention":
                patched_activation = activation
                patched_activation[:, :, node.head_idx, :] = 0
            else:
                patched_activation = torch.zeros_like(activation)
            return patched_activation.to(model.cfg.device)

    return patching_hook

def add_all_hooks(
        model,
        method,
        clean_cache,
        corrupted_cache,
        ablated_nodes: Optional[List[Node]] = None,
        ablated_edges: Optional[List[Edge]] = None):
    """Add hooks for all activations in the circuit."""
    if ablated_edges != [] and ablated_edges is not None:
        if method == "patching" and corrupted_cache:
            for edge in ablated_edges:
                hook = create_edge_patching_hook(
                    model,
                    method,
                    edge.sender,
                    edge.receiver,
                    clean_cache,
                    corrupted_cache
                )
                if hasattr(model, 'add_hook'):
                    model.add_hook(edge.receiver.full_activation, hook)
        else:
            for edge in ablated_edges:
                hook = create_edge_patching_hook(
                    model,
                    method,
                    edge.sender,
                    edge.receiver,
                    clean_cache
                )
                if hasattr(model, 'add_hook'):
                    model.add_hook(edge.receiver.full_activation, hook)
    if ablated_nodes != [] and ablated_nodes is not None:
        if method == "patching" and corrupted_cache:
            for node in ablated_nodes:
                hook = create_node_patching_hook(model, method, node, corrupted_cache)
                if hasattr(model, 'add_hook'):
                    model.add_hook(node.full_activation, hook)
        else:
            for node in ablated_nodes:
                hook = create_node_patching_hook(model, method, node)
                if hasattr(model, 'add_hook'):
                    model.add_hook(node.full_activation, hook)
                    
def get_final_performance(
        model,
        method,
        dataset,
        task_name,
        clean_logits,
        clean_caches,
        corrupted_caches,
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
        if method == "patching":
            add_all_hooks(model, method, clean_caches[i], corrupted_caches[i],
                           ablated_nodes, ablated_edges)
        else:
            add_all_hooks(model, method, clean_caches[i], None,
                           ablated_nodes, ablated_edges)
            
        with torch.no_grad():
            ablated_logits = model(example.clean_tokens)
        kl_div = kl_divergence(clean_logits[i].to(model.cfg.device), ablated_logits)
        kl_divs.append(kl_div.item())
        if task_name == "Factuality":
            logits.append(ablated_logits)
            labels.append(example.label)
    avg_kl_div = np.mean(kl_divs)
    if task_name == "Factuality":
        # Compute metrics for factuality evaluation
        evaluate_factuality(logits, labels, model)
    return avg_kl_div

def is_output_activation(full_activation, component_type):
    """ Determine if an activation is an output activation for a given component type."""
    if component_type == "attention" and "attn.hook_z" in full_activation:
        return True
    elif component_type == "mlp" and "hook_mlp_out" in full_activation:
        return True
    return False