import torch
from transformer_lens import HookedTransformer
from algorithms.ACDC import ACDC
from utilities.visualization import visualize_pythia_graph
from algorithms.ActivationPatching import ActivationPatching
from tasks.Factuality import FactualityDatasetBuilder
import pandas as pd
from utilities.GraphVisualizer import visualize_gemma_graph


def run_circuit_discovery():

    print("Loading model...")
    model_name = "EleutherAI/pythia-70m-deduped"
    #model_name = "meta-llama/Llama-2-7b-hf
    #model_name = "google/gemma-2-2b-it"

    model = HookedTransformer.from_pretrained(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    # Activation Patching
    #algorithm = ActivationPatching(model)
    #algorithm.run_circuit_discovery()

    # ACDC
    algorithm = ACDC(model, model_name, task="IOI", target="edge", mode="greedy", method="patching", threshold=0.05)
    full_graph = algorithm.build_computational_graph()
    #visualize_gemma_graph(graph=full_graph, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads)
    circuit = algorithm.discover_circuit()
    #visualize_gemma_graph(graph=circuit, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads)
    visualize_pythia_graph(graph=full_graph, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads, figsize=(20,24))
    visualize_pythia_graph(graph=circuit, num_layers=model.cfg.n_layers, num_attention_heads=model.cfg.n_heads, figsize=(20,24))


if __name__ == "__main__":
    run_circuit_discovery()

