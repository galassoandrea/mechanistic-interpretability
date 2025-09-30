from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

"""Evaluation functions for model outputs."""


def logits_to_logit_diff(logits, example, model=None):
    """Compute logit difference for correct and incorrect token answer."""
    return logits[0, -1, example.correct_token] - logits[0, -1, example.incorrect_token]


def kl_divergence(clean_logits, corrupted_logits, dim: int = -1):
    """Compute KL divergence between two logit distributions: KL(P || Q)."""
    # Convert logits to probability distributions
    log_probs_a = F.log_softmax(clean_logits, dim=dim)  # log P
    log_probs_b = F.log_softmax(corrupted_logits, dim=dim)  # log Q
    probs_a = log_probs_a.exp()  # P

    # KL(P || Q) = sum P * (logP - logQ)
    kl = torch.sum(probs_a * (log_probs_a - log_probs_b), dim=dim)

    return kl.mean()

def factuality_nll_metric(logits, example, model):
    """
    Compute negative log-likelihood for binary classification.
    Model should predict '0' or '1' at the last position.
    """
    # Get logits at the last position (where model generates 0 or 1)
    final_logits = logits[:, -1, :]

    # Get token IDs for '0' and '1' - moved outside loop for efficiency
    token_0_id = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    token_1_id = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    # Extract logits for 0 and 1 tokens
    logit_0 = final_logits[:, token_0_id]
    logit_1 = final_logits[:, token_1_id]

    # Compute log probabilities over [0, 1]
    log_probs = torch.log_softmax(torch.stack([logit_0, logit_1], dim=-1), dim=-1)

    # NLL: -log_prob of correct class
    # log_probs[:, 0] is log P(0), log_probs[:, 1] is log P(1)
    nll = -log_probs[:, example.label].mean()

    return nll


def evaluate_factuality(all_logits: List[torch.Tensor], all_labels, model):
    """
    Optimized version that works directly with batches.
    """

    all_predictions = []
    all_probs_positive = []

    # Get token IDs for '0' and '1' - moved outside loop for efficiency
    token_0_id = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    token_1_id = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    # Process each batch
    for logits_batch in all_logits:
        # logits_batch shape: [batch_size, seq_len, vocab_size]
        batch_size = logits_batch.shape[0]

        # Extract logits for the next token (last position) for all samples in batch
        next_token_logits = logits_batch[:, -1, :]  # Shape: [batch_size, vocab_size]

        # Extract logits for tokens '0' and '1' for all samples
        binary_logits = torch.stack([
            next_token_logits[:, token_0_id],  # Logits for '0' across batch
            next_token_logits[:, token_1_id]  # Logits for '1' across batch
        ], dim=1)  # Shape: [batch_size, 2]

        # Convert to probabilities for the entire batch
        probs = torch.softmax(binary_logits, dim=1).cpu().numpy()  # Shape: [batch_size, 2]

        # Get predictions for the entire batch
        predictions = np.argmax(probs, axis=1)  # Shape: [batch_size]
        probs_positive = probs[:, 1]  # Probabilities of '1' for entire batch

        all_predictions.extend(predictions.tolist())
        all_probs_positive.extend(probs_positive.tolist())

    # Flatten labels if they're still in batch format
    flattened_labels = []
    for label_batch in all_labels:
        if isinstance(label_batch, torch.Tensor):
            if label_batch.dim() > 0:  # If it's a batch
                flattened_labels.extend(label_batch.cpu().numpy().tolist())
            else:  # If it's a single value
                flattened_labels.append(label_batch.item())
        elif isinstance(label_batch, (list, np.ndarray)):
            flattened_labels.extend(label_batch)
        else:
            flattened_labels.append(label_batch)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    ground_truths = np.array(flattened_labels)
    probs_positive = np.array(all_probs_positive)

    # Create probability matrix for log_loss
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

