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
        corrupted_sender_contribution: Optional[torch.Tensor] = None
) -> Callable:
    """Create a hook function for node ablation/patching."""

    def patching_hook(activation, hook):
        # Check the number of dimensions of the activation
        if activation.dim() == 4:
            # Attention layer (hook_z)
            patched_activation = activation
            if method == "patching":
                patched_activation[:,:,node.head_idx,:] = corrupted_sender_contribution
            else:
                patched_activation[:,:,node.head_idx,:] = 0
        else:
            if method == "patching":
                patched_activation = corrupted_sender_contribution
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
            hook = create_node_patching_hook(
                method="pruning",
                node=node
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

def save_circuit(model_name, topic, target, ablated_nodes: Optional[List[Node]] = None, ablated_edges: Optional[List[Edge]] = None):
    # Store removed edges/nodes metadata in a json file
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if target == "node" and ablated_nodes is not None:
        params_to_save = {
            "ablated_nodes": [
                {node.full_activation: node.head_idx}
                for node in ablated_nodes
            ]
        }
        save_dir = os.path.join(SCRIPT_DIR, "removed_nodes")
    else:
        params_to_save = {
            "ablated_edges": [
                {"sender": get_node_id(edge.sender), "receiver": edge.receiver.full_activation}
                for edge in ablated_edges
            ]
        }
        save_dir = os.path.join(SCRIPT_DIR, "removed_edges")

    os.makedirs(save_dir, exist_ok=True)

    # Build full file path
    save_path = os.path.join(save_dir, f"{model_name.replace('/', '-')}-{topic}.json")
    with open(save_path, "w") as f:
        json.dump(params_to_save, f, indent=2)

def add_circuit_hooks(model, model_name, target, topic):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(SCRIPT_DIR, f"removed_{target}s", f"{model_name.replace('/', '-')}-{topic}.json")
    with open(path, "r") as f:
        params = json.load(f)
        print(f"Loaded removed {target}s: {params}")
    for node in params["ablated_nodes"]:
        for full_activation, head_idx in node.items():
            if "attn" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                head_idx = head_idx
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="attention",
                    head_idx=head_idx,
                    full_activation=full_activation
                )
            elif "mlp_out" in full_activation:
                layer = int(full_activation.split('.')[1]) + 1
                act_name = full_activation.rsplit(".", 1)[1]
                node = Node(
                    name=act_name,
                    layer=layer,
                    component_type="mlp",
                    full_activation=full_activation
                )
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
            model.add_hook(node.full_activation, create_node_patching_hook(
                method="pruning",
                node=node
            ))