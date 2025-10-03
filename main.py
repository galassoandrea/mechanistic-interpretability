import torch
from transformer_lens import HookedTransformer
from algorithms.ACDC.ACDC import ACDC
from algorithms.AttributionPatching.AttributionPatching import AttributionPatching
from utilities.evaluation import evaluate_factuality
from utilities.GraphVisualizer import visualize_gemma_graph
from tqdm import tqdm
from utilities.ComputationalGraph import build_computational_graph

from utilities.visualization import visualize_pythia_graph


def run_circuit_discovery():
    print("Loading model...")
    # model_name = "EleutherAI/pythia-70m-deduped"
    # model_name = "meta-llama/Llama-2-7b-hf
    # model_name = "google/gemma-2-2b-it"
    model_name = "Qwen/Qwen3-0.6B"

    model = HookedTransformer.from_pretrained(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ACDC
    #algorithm = AttributionPatching(model, model_name, task="IOI")
    algorithm = ACDC(model, model_name, task="Factuality", topic="animals", target="edge", mode="greedy", method="patching", threshold=0.05)

    #all_logits = []
    #all_labels = []
#
    #for example in tqdm(algorithm.dataset, desc="Evaluating examples"):
    #    prompt_tokens = torch.tensor(example.clean_tokens).unsqueeze(0).to(model.cfg.device)
    #    with torch.no_grad():
    #        output = model(prompt_tokens)
    #        if hasattr(output, 'logits'):
    #            logit = output.logits
    #        else:
    #            logit = output
    #    all_logits.append(logit.cpu())
    #    all_labels.append(example.label)
#
    #evaluate_factuality(all_logits, all_labels, model)
#
    ## Clean memory
    #del all_logits
    #del all_labels
#
    # Run ACDC on factuality task
    initial_graph = build_computational_graph(model, model_name)
    visualize_pythia_graph(graph=initial_graph, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads, figsize=(20,24))
    circuit = algorithm.discover_circuit()
    visualize_pythia_graph(graph=circuit, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads, figsize=(20,24))



if __name__ == "__main__":
    run_circuit_discovery()



