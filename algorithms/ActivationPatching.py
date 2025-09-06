from collections import defaultdict
import torch
from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from utilities.evaluation import logits_to_logit_diff
from tasks.IOI import IOIExample
from functools import partial
from jaxtyping import Float
from utilities.visualization import plot_circuit_graph, plot_activation_importance


@dataclass
class CircuitNode:
    """Represents a node in the computational circuit"""
    layer: int
    position: int
    activation_type: str
    importance_score: float = 0.0

    def __hash__(self):
        return hash((self.layer, self.position, self.activation_type))

    def __str__(self):
        return f"L{self.layer}_{self.activation_type}_pos{self.position}"

class ActivationPatching:
    """Main class for activation patching and circuit discovery"""
    def __init__(self, model: HookedTransformer, dataset: List[IOIExample]):
        self.model = model
        self.dataset = dataset
        self.device = model.cfg.device
        self.threshold: float = 0.3
        self.activation_types = ['resid_pre', 'z', 'mlp_out']
        self.circuit_nodes = set()
        self.patching_results = defaultdict(list)
        self.dropped_activations = set()

    def permanent_ablate_hook(self, position):
        def hook_fn(activation, hook):
            activation[:, position, :] = 0.0
            return activation

        return hook_fn

    def patch_single_activation(self, node: CircuitNode, example, clean_cache) -> float:
        """Patch a single activation and compute the resulting score"""
        # Define hook function
        def patching_hook(
                activation: Float[torch.Tensor, "batch pos d_model"],
                hook: HookPoint,
                position: int
        ) -> Float[torch.Tensor, "batch pos d_model"]:
            # Each HookPoint has a name attribute giving the name of the hook.
            clean_activation = clean_cache[hook.name]
            activation[:, position, :] = clean_activation[:, position, :]
            return activation

        temp_hook_fn = partial(patching_hook, position=node.position)
        # Run the model with the patching hook
        patched_logits = self.model.run_with_hooks(example.corrupted_tokens, fwd_hooks=[
            (get_act_name(node.activation_type, node.layer), temp_hook_fn)
        ])

        # Compute the IOI metric for the patched logits
        patched_score = logits_to_logit_diff(patched_logits, example.correct_token, example.incorrect_token)
        return patched_score

    def run_circuit_discovery(self, max_iter = 5):
        """Run circuit discovery by patching activations on the whole dataset"""
        num_tokens = self.dataset[0].clean_tokens.shape[0]
        print("Starting circuit discovery...")
        iteration = 0
        total_nodes_removed = 0
        while iteration < max_iter:
            iteration += 1
            nodes_removed_this_iter = 0
            print(f"Starting iteration {iteration}/{max_iter}")
            # Iterate over each layer, position and activation type
            for layer in range(self.model.cfg.n_layers):
                for position in range(num_tokens):
                    for activation_type in self.activation_types:
                        if (layer, position, activation_type) in self.dropped_activations:
                            continue
                        node = CircuitNode(layer, position, activation_type)
                        # Iterate over the dataset
                        for example in tqdm(self.dataset):
                            self.model.reset_hooks()

                            clean_tokens = example.clean_tokens
                            corrupted_tokens = example.corrupted_tokens

                            with torch.no_grad():
                                # Run the model with clean tokens to get the clean logits and activation cache
                                clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)

                                # Run the model with corrupted tokens to get the corrupted logits
                                corrupted_logits = self.model(corrupted_tokens)

                                # Compute the baseline performance
                                clean_score = logits_to_logit_diff(clean_logits, example.correct_token, example.incorrect_token)
                                corrupted_score = logits_to_logit_diff(corrupted_logits, example.correct_token, example.incorrect_token)

                                # skip example if denominator is zero (clean == corrupted)
                                d = (clean_score - corrupted_score)
                                if d == 0:
                                    continue

                                # Patch the activation and compute the score
                                patched_score = self.patch_single_activation(node, example, clean_cache)

                                # Normalize the patched score
                                normalized_score = (patched_score - corrupted_score) / d

                                # Store the results
                                self.patching_results[str(node)].append(float(normalized_score))

                        avg_score = sum(self.patching_results[str(node)]) / len(self.patching_results[str(node)])
                        if avg_score > self.threshold:
                            self.circuit_nodes.add(str(node))
                            print(f"Keeping important Node {node}\n")
                            print(f"Average patched score: {avg_score:.4f}")
                        else:
                            # Remove the node if its average score is below the threshold
                            print(f"Removing unimportant Node {node}\n")
                            print(f"Average patched score: {avg_score:.4f}")
                            self.model.add_hook(
                                get_act_name(node.activation_type, node.layer),
                                self.permanent_ablate_hook(position),
                                level=1
                            )
                            self.dropped_activations.add((layer, position, activation_type))
                            # Remove previously important node from circuit if its importance changed
                            if str(node) in self.circuit_nodes:
                                self.circuit_nodes.remove(str(node))
                            nodes_removed_this_iter += 1
            if nodes_removed_this_iter == 0:
                print("Circuit discovery complete!")
                break
            total_nodes_removed += nodes_removed_this_iter
            print(f"Iteration {iteration} complete. Nodes removed: {total_nodes_removed}")

            # Plot circuit graph for this iteration
            plot_circuit_graph(self.circuit_nodes, iteration=iteration)

        print(f"Total ablated activations: {len(self.dropped_activations)}")

        # Plot activation importance heatmap
        plot_activation_importance(self.model.cfg.n_layers, num_tokens, self.patching_results)

        return self.patching_results
