import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from algorithms.ACDC import ACDC
from utilities.visualization import visualize_pythia_graph
from algorithms.ActivationPatching import ActivationPatching
from tasks.Factuality import FactualityDatasetBuilder
import pandas as pd
import numpy as np
from utilities.evaluation import evaluate_factuality

def run_circuit_discovery():

    print("Loading model...")

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cuda" if torch.cuda.is_available() else "cpu")

    print("Model loaded successfully")
    print(f"Model device: {next(model.parameters()).device}")

    df = pd.read_csv("tasks/factuality_data/animals_true_false.csv")
    dataset_builder = FactualityDatasetBuilder(model)
    dataset = dataset_builder.build_dataset()

    example = dataset[0]
    print(f"Prompt length: {len(example.prompt_tokens)} tokens")

    # Get token IDs for "0" and "1"
    zero_token = model.to_tokens("0", prepend_bos=False)[0, 0].item()
    one_token = model.to_tokens("1", prepend_bos=False)[0, 0].item()

    y_true, y_pred, y_prob = [], [], []

    for example in tqdm(dataset, desc="Evaluating examples"):
        label = example.label

        # Run model
        with torch.no_grad():
            logits = model(example.prompt_tokens.unsqueeze(0).to(model.cfg.device), return_type="logits")

            # Get logits for the NEXT token (last position)
            next_token_logits = logits[0, -1, [zero_token, one_token]]
            probs = torch.softmax(next_token_logits, dim=-1).detach().cpu().numpy()

        # Prediction and storage
        pred = int(np.argmax(probs))
        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(probs[1])  # probability assigned to "1"

    # Compute metrics
    metrics = evaluate_factuality(
        logits=torch.tensor(y_prob).unsqueeze(-1),
        labels=torch.tensor(y_true)
    )
    acc = metrics["accuracy"]
    roc = metrics["roc_auc"]
    nll = metrics["nll"]

    print(f"Baseline Accuracy: {acc:.3f}")
    print(f"Baseline ROC-AUC: {roc:.3f}")
    print(f"Baseline NLL: {nll:.3f}")


if __name__ == "__main__":
    run_circuit_discovery()
