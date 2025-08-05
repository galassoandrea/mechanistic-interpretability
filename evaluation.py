from typing import List, Set
import torch
from transformer_lens.utils import get_act_name

from IOI_Dataset import IOIExample


def compute_ioi_metric(logits: torch.Tensor, examples: List[IOIExample]) -> float:
    """Compute the logit difference metric"""
    total_logit_diff = 0.0

    for i, example in enumerate(examples):
        # Get logits for the answer position
        answer_logits = logits[i, example.answer_token_pos, :]

        # Compute logit difference (correct - incorrect)
        correct_logit = answer_logits[example.correct_token].item()
        incorrect_logit = answer_logits[example.incorrect_token].item()

        total_logit_diff += (correct_logit - incorrect_logit)

    return total_logit_diff / len(examples)

def get_baseline_performance(model, dataset) -> float:
    """Get baseline performance on clean prompts"""
    clean_tokens = torch.cat([ex.clean_tokens for ex in dataset], dim=0)

    with torch.no_grad():
        logits = model(clean_tokens)
        baseline_score = compute_ioi_metric(logits, dataset)

    return baseline_score

def evaluate_circuit_performance(dataset, model, circuit_heads: Set[str]) -> float:
    """Evaluate performance for the circuit found"""
    clean_tokens = torch.cat([ex.clean_tokens for ex in dataset], dim=0)
    def circuit_ablation_hook(activations, hook):
        layer = hook.layer()
        for head in range(model.cfg.n_heads):
            head_name = f"L{layer}H{head}"
            if head_name not in circuit_heads:
                activations[:, :, head] = 0
        return activations
    hooks = []
    for layer in range(model.cfg.n_layers):
        hooks.append((get_act_name("z", layer), circuit_ablation_hook))
    with torch.no_grad():
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
        circuit_score = compute_ioi_metric(logits, dataset)
    return circuit_score