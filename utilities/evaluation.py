import torch
import torch.nn.functional as F

"""Evaluation functions for model outputs."""

def logits_to_logit_diff(logits, correct_token, incorrect_token):
    """Compute logit difference for correct and incorrect token answer."""
    return logits[0, -1, correct_token] - logits[0, -1, incorrect_token]


def kl_divergence(clean_logits, corrupted_logits, dim: int = -1):
    """Compute KL divergence between two logit distributions: KL(P || Q)."""
    # Convert logits to probability distributions
    log_probs_a = F.log_softmax(clean_logits, dim=dim)  # log P
    log_probs_b = F.log_softmax(corrupted_logits, dim=dim)  # log Q
    probs_a = log_probs_a.exp()  # P

    # KL(P || Q) = sum P * (logP - logQ)
    kl = torch.sum(probs_a * (log_probs_a - log_probs_b), dim=dim)

    return kl.mean()