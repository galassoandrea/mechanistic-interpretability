import torch
from transformer_lens import HookedTransformer
from algorithms.ACDC import ACDC
from utilities.visualization import visualize_pythia_graph
from algorithms.ActivationPatching import ActivationPatching
from tasks.Induction import InductionDatasetBuilder

def run_circuit_discovery():

    print("Loading model...")
    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device="cuda" if torch.cuda.is_available() else "cpu")

    ## ACDC
    acdc = ACDC(model, task="IOI", mode="independent", threshold=0.01)
    full_graph = acdc.build_computational_graph()
    visualize_pythia_graph(graph=full_graph, figsize=(20,24))
    circuit = acdc.discover_circuit()
    visualize_pythia_graph(graph=circuit, figsize=(20,24))

if __name__ == "__main__":
    run_circuit_discovery()


"""Almost all activations in the model:
['blocks.0.attn.hook_k', 'blocks.0.attn.hook_q', 'blocks.0.attn.hook_v', 'blocks.0.attn.hook_z', 'blocks.0.attn.hook_attn_scores',
 'blocks.0.attn.hook_pattern', 'blocks.0.attn.hook_result', 'blocks.0.attn.hook_rot_k', 'blocks.0.attn.hook_rot_q',
 'blocks.0.mlp.hook_pre', 'blocks.0.mlp.hook_post', 'blocks.0.hook_attn_in', 'blocks.0.hook_mlp_in', 'blocks.0.hook_attn_out',
 'blocks.0.hook_mlp_out', 'blocks.0.hook_resid_pre', 'blocks.0.hook_resid_post', 'blocks.1.attn.hook_k', 'blocks.1.attn.hook_q',
 'blocks.1.attn.hook_v', 'blocks.1.attn.hook_z', 'blocks.1.attn.hook_attn_scores', 'blocks.1.attn.hook_pattern',
 'blocks.1.attn.hook_result', 'blocks.1.attn.hook_rot_k', 'blocks.1.attn.hook_rot_q', 'blocks.1.mlp.hook_pre',
 'blocks.1.mlp.hook_post', 'blocks.1.hook_attn_in', 'blocks.1.hook_mlp_in', 'blocks.1.hook_attn_out', 'blocks.1.hook_mlp_out',
 'blocks.1.hook_resid_pre', 'blocks.1.hook_resid_post', 'blocks.2.attn.hook_k', 'blocks.2.attn.hook_q', 'blocks.2.attn.hook_v',
 'blocks.2.attn.hook_z', 'blocks.2.attn.hook_attn_scores', 'blocks.2.attn.hook_pattern', 'blocks.2.attn.hook_result',
 'blocks.2.attn.hook_rot_k', 'blocks.2.attn.hook_rot_q', 'blocks.2.mlp.hook_pre', 'blocks.2.mlp.hook_post',
 'blocks.2.hook_attn_in', 'blocks.2.hook_mlp_in', 'blocks.2.hook_attn_out', 'blocks.2.hook_mlp_out',
 'blocks.2.hook_resid_pre', 'blocks.2.hook_resid_post', 'blocks.3.attn.hook_k', 'blocks.3.attn.hook_q',
 'blocks.3.attn.hook_v', 'blocks.3.attn.hook_z', 'blocks.3.attn.hook_attn_scores', 'blocks.3.attn.hook_pattern',
 'blocks.3.attn.hook_result', 'blocks.3.attn.hook_rot_k', 'blocks.3.attn.hook_rot_q', 'blocks.3.mlp.hook_pre',
 'blocks.3.mlp.hook_post', 'blocks.3.hook_attn_in', 'blocks.3.hook_mlp_in', 'blocks.3.hook_attn_out',
 'blocks.3.hook_mlp_out', 'blocks.3.hook_resid_pre', 'blocks.3.hook_resid_post', 'blocks.4.attn.hook_k',
 'blocks.4.attn.hook_q', 'blocks.4.attn.hook_v', 'blocks.4.attn.hook_z', 'blocks.4.attn.hook_attn_scores',
 'blocks.4.attn.hook_pattern', 'blocks.4.attn.hook_result', 'blocks.4.attn.hook_rot_k', 'blocks.4.attn.hook_rot_q',
 'blocks.4.mlp.hook_pre', 'blocks.4.mlp.hook_post', 'blocks.4.hook_attn_in', 'blocks.4.hook_mlp_in',
 'blocks.4.hook_attn_out', 'blocks.4.hook_mlp_out', 'blocks.4.hook_resid_pre', 'blocks.4.hook_resid_post',
 'blocks.5.attn.hook_k', 'blocks.5.attn.hook_q', 'blocks.5.attn.hook_v', 'blocks.5.attn.hook_z',
 'blocks.5.attn.hook_attn_scores', 'blocks.5.attn.hook_pattern', 'blocks.5.attn.hook_result',
 'blocks.5.attn.hook_rot_k', 'blocks.5.attn.hook_rot_q', 'blocks.5.mlp.hook_pre', 'blocks.5.mlp.hook_post',
 'blocks.5.hook_attn_in', 'blocks.5.hook_mlp_in', 'blocks.5.hook_attn_out', 'blocks.5.hook_mlp_out',
 'blocks.5.hook_resid_pre', 'blocks.5.hook_resid_post']
"""