import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

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


def evaluate_factuality(logits: torch.Tensor, labels: torch.Tensor):
    """Compute Accuracy, ROC-AUC, and Negative Log-Likelihood (cross-entropy loss)."""
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    labels_np = labels.detach().cpu().numpy()

    # Accuracy
    acc = accuracy_score(labels_np, preds)

    # ROC-AUC (only valid if both classes are present)
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float('nan')

    # NLL / cross-entropy log loss
    nll = log_loss(labels_np, probs, labels=[0, 1])

    return {"accuracy": acc, "roc_auc": auc, "nll": nll}
