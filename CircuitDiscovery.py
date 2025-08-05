import torch
from typing import List, Dict
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from IOI_Dataset import IOIExample
from evaluation import compute_ioi_metric, get_baseline_performance

class CircuitDiscovery:
    """Main class for performing circuit discovery"""

    def __init__(self, model: HookedTransformer, dataset: List[IOIExample]):
        self.model = model
        self.dataset = dataset
        self.device = next(model.parameters()).device

        # Initialize results storage
        self.head_importance_scores = {}
        self.circuit_evolution = []

        # Initialize results
        self.results = {
            'baseline_score': 0.0,
            'head_ablation_results': {},
            'head_importance_ranking': [],
            'circuit_components': set()
        }

    def head_ablation_experiment(self, layer: int, head: int) -> float:
        """Perform head ablation for a specific attention head"""
        clean_tokens = torch.cat([ex.clean_tokens for ex in self.dataset], dim=0)

        def ablation_hook(activations, hook):
            # Zero out the specific head
            activations[:, :, head] = 0
            return activations

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(get_act_name("z", layer), ablation_hook)]
            )
            ablated_score = compute_ioi_metric(logits, self.dataset)

        return ablated_score

    def run_full_analysis(self) -> Dict:
        """Run complete circuit discovery analysis using head ablation"""
        print("Computing baseline performance...")
        baseline_score = get_baseline_performance(self.model, self.dataset)
        print(f"Baseline IOI score (logit difference): {baseline_score:.4f}")

        # Initialize results
        self.results['baseline_score'] = baseline_score

        print("\nRunning head ablation experiments...")
        for layer in tqdm(range(self.model.cfg.n_layers), desc="Layers"):
            for head in range(self.model.cfg.n_heads):
                # Ablate head
                ablated_score = self.head_ablation_experiment(layer, head)
                importance_score = baseline_score - ablated_score

                head_name = f"L{layer}H{head}"
                self.results['head_ablation_results'][head_name] = {
                    'ablated_score': ablated_score,
                    'importance': importance_score,
                    'layer': layer,
                    'head': head
                }

        # Rank heads by importance
        head_rankings = []
        for head_name, data in self.results['head_ablation_results'].items():
            head_rankings.append({
                'head': head_name,
                'layer': data['layer'],
                'head_idx': data['head'],
                'ablation_importance': data['importance'],
            })

        # Sort by ablation importance (descending)
        head_rankings.sort(key=lambda x: x['ablation_importance'], reverse=True)
        self.results['head_importance_ranking'] = head_rankings

        # Identify circuit components (top 20% most important heads)
        top_k = max(5, len(head_rankings) // 5)  # At least 5 heads, or top 20%
        circuit_components = set()
        for ranking in head_rankings[:top_k]:
            if ranking['ablation_importance'] > 0.01:  # Threshold for significance
                circuit_components.add(ranking['head'])

        self.results['circuit_components'] = circuit_components

        return self.results