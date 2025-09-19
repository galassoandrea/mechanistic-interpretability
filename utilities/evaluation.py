from typing import Dict, List

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


def evaluate_factuality(all_logits: List[torch.Tensor], all_labels, model):
    """
    Simplified version closer to your original code with key improvements.
    """

    predictions = []
    probs_positive = []

    # Get token IDs for '0' and '1' - moved outside loop for efficiency
    token_0_id = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    token_1_id = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    # Iterate over list of logits and Extract logits for the next token (last position)
    for logits in all_logits:
        next_token_logits = logits[0, -1, :]

        # Extract logits for these specific tokens
        binary_logits = torch.stack([
            next_token_logits[token_0_id],  # Logit for '0'
            next_token_logits[token_1_id]  # Logit for '1'
        ])

        # Convert to probabilities
        probs = torch.softmax(binary_logits, dim=0).cpu().numpy()

        # Get prediction (0 or 1)
        prediction = np.argmax(probs)
        predictions.append(prediction)
        probs_positive.append(probs[1])  # Probability of '1'

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truths = np.array(all_labels)
    probs_positive = np.array(probs_positive)

    # Create probability matrix for log_loss (sklearn expects 2D array)
    probs_matrix = np.column_stack([1 - probs_positive, probs_positive])

    # Calculate metrics
    accuracy = accuracy_score(ground_truths, predictions)

    # ROC-AUC (only if both classes are present)
    try:
        if len(np.unique(ground_truths)) > 1:
            roc_auc = roc_auc_score(ground_truths, probs_positive)
        else:
            roc_auc = float('nan')
            print("Warning: Only one class present in labels, cannot compute ROC-AUC")
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC: {e}")
        roc_auc = float('nan')

    # Negative Log-Likelihood
    try:
        nll = log_loss(ground_truths, probs_matrix, labels=[0, 1])
    except Exception as e:
        print(f"Warning: Could not compute NLL: {e}")
        nll = float('nan')

    print(f"Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, NLL: {nll:.4f}")

    return {'accuracy': accuracy, 'roc_auc': roc_auc, 'nll': nll}
