import torch
from transformer_lens import HookedTransformer
from IOI_Dataset import IOIDatasetBuilder
from CircuitDiscovery import CircuitDiscovery
from visualization import Visualization


def run_circuit_discovery():

    print("Loading model...")
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cuda" if torch.cuda.is_available() else "cpu")

    print("Building IOI dataset...")
    dataset_builder = IOIDatasetBuilder(model)
    dataset = dataset_builder.build_dataset(num_samples=100)

    print(f"Generated {len(dataset)} IOI examples")
    print("Sample examples:")
    for i in range(3):
        ex = dataset[i]
        print(f"  Clean: {ex.clean_prompt} -> {ex.correct_answer}")
        print(f"  Corrupted: {ex.corrupted_prompt} -> {ex.incorrect_answer}")
        print()

    print("Starting circuit discovery...")
    circuit_discovery = CircuitDiscovery(model, dataset)
    results = circuit_discovery.run_full_analysis()

    print("Creating visualizations...")
    visualization = Visualization(model, dataset)
    visualization.visualize_head_importance(results)
    visualization.visualize_circuit_graph(results)

    print("Circuit discovery complete!")
#
    #return model, dataset, circuit_discovery, results



if __name__ == "__main__":
    run_circuit_discovery()
