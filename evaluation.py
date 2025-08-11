"""Evaluation functions for model outputs."""

def logits_to_logit_diff(logits, correct_token, incorrect_token):
    return logits[0, -1, correct_token] - logits[0, -1, incorrect_token]
