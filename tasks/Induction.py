import torch
import numpy as np
from typing import List
import random
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def pad_sequences(tokens: List[torch.Tensor]) -> torch.Tensor:
    """Pad each token sequence to the maximum length in the batch"""
    max_length = max(token.shape[0] for token in tokens)
    padded_tokens = [
        F.pad(t, (0, max_length - t.shape[0]))
        for t in tokens
    ]
    return torch.stack(padded_tokens, dim=0)

@dataclass
class InductionExample:
    """Represents a single Induction example with both clean and corrupted versions"""
    clean_prompt: str
    corrupted_prompt: str
    pattern_a: str
    pattern_b: str
    corrupted_b: str
    clean_tokens: torch.Tensor
    corrupted_tokens: torch.Tensor
    first_a_pos: int
    first_b_pos: int
    second_a_pos: int
    answer_token_pos: int
    correct_token: torch.Tensor
    incorrect_token: torch.Tensor

class InductionDatasetBuilder:
    """Build a toy Induction dataset with clean and corrupted prompts"""

    def __init__(self, model):

        self.model = model
        self.token_pool = [
            #' A', ' B', ' C', ' D', ' E', ' F', ' G', ' H', ' I', ' J', ' K', ' L', ' M', ' N', ' O',
            #' P', ' Q', ' R', ' S', ' T', ' U', ' V', ' W', ' X', ' Y', ' Z',
            ' cat', ' dog', ' bird', ' fish', ' mouse', ' horse',
            ' red', ' blue', ' green', ' yellow', ' purple', ' orange',
            ' apple', ' banana', ' orange', ' grape', ' pear',
            #' car', ' bike', ' bus', ' train', ' plane', ' boat',
            #' sun', ' moon', ' star', ' sky', ' cloud', ' rain',
            #' tree', ' flower', ' grass', ' leaf', ' root', ' branch',
            #' house', ' building', ' room', ' door', ' window', ' roof',
            #' one', ' two', ' three', ' four', ' five', ' six', ' seven', ' eight', ' nine', ' ten',
            #' I', ' you', ' he', ' she', ' it', ' we', ' they',
            #' hello', ' world', ' foo', ' bar',
            #' 100', ' 200', ' 300',
            #' alpha', ' beta', ' gamma'
        ]

    def sample_token(self, exclude: List[str] = None) -> str:
        """Sample a token from the pool, excluding some"""
        exclude = exclude or []
        choice = random.choice([tok for tok in self.token_pool if tok not in exclude])
        return choice

    def generate_induction_example(self) -> InductionExample:
        """Generate a simple induction example with a clear pattern"""
        pattern_a = self.sample_token()
        pattern_b = self.sample_token(exclude=[pattern_a])
        corrupted_b = self.sample_token(exclude=[pattern_a, pattern_b])

        # Create filler content between the patterns
        filler_length = random.randint(3, 8)
        filler = " ".join([self.sample_token(exclude=[pattern_a, pattern_b, corrupted_b])
                          for _ in range(filler_length)])

        # Construct clean and corrupted prompts
        clean_prompt = f"{pattern_a} {pattern_b} {filler} {pattern_a}"
        corrupted_prompt = f"{pattern_a} {corrupted_b} {filler} {pattern_a}"

        # Tokenize
        clean_tokens = self.model.to_tokens(clean_prompt).squeeze(0)
        corrupted_tokens = self.model.to_tokens(corrupted_prompt).squeeze(0)
        pattern_a_token = self.model.to_single_token(pattern_a)
        correct_token = self.model.to_single_token(pattern_b)
        incorrect_token = self.model.to_single_token(pattern_a)

        a_positions = (clean_tokens == pattern_a_token).nonzero(as_tuple=True)[0]

        # Find first A position
        first_a_pos = a_positions[0].item()

        # Find first B position (right after first A)
        first_b_pos = first_a_pos + 1

        # Find second A position (last occurrence)
        second_a_pos = a_positions[-1].item()

        # Answer position is right after second A
        answer_token_pos = second_a_pos + 1

        return InductionExample(
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            pattern_a=pattern_a,
            pattern_b=pattern_b,
            corrupted_b=corrupted_b,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            first_a_pos=first_a_pos,
            first_b_pos=first_b_pos,
            second_a_pos=second_a_pos,
            answer_token_pos=answer_token_pos,
            correct_token=correct_token,
            incorrect_token=incorrect_token
        )

    def build_dataset(self, num_samples: int) -> List[InductionExample]:
        """Build a dataset with the specified number of examples"""
        dataset = []
        clean_tokens = []
        corrupted_tokens = []
        for _ in tqdm(range(num_samples), desc="Generating Induction examples"):
            example = self.generate_induction_example()
            clean_tokens.append(example.clean_tokens)
            corrupted_tokens.append(example.corrupted_tokens)
            dataset.append(example)

        # Pad sequences to the maximum length
        clean_tokens = pad_sequences(clean_tokens)
        corrupted_tokens = pad_sequences(corrupted_tokens)
        # Update dataset with padded tokens
        for i, example in enumerate(dataset):
            example.clean_tokens = clean_tokens[i]
            example.corrupted_tokens = corrupted_tokens[i]
        # Convert dataset to a list of IOIExample objects
        dataset = [InductionExample(
            clean_prompt=ex.clean_prompt,
            corrupted_prompt=ex.corrupted_prompt,
            pattern_a=ex.pattern_a,
            pattern_b=ex.pattern_b,
            corrupted_b=ex.corrupted_b,
            clean_tokens=ex.clean_tokens,
            corrupted_tokens=ex.corrupted_tokens,
            first_a_pos=ex.first_a_pos,
            first_b_pos=ex.first_b_pos,
            second_a_pos=ex.second_a_pos,
            answer_token_pos=ex.answer_token_pos,
            correct_token=ex.correct_token,
            incorrect_token=ex.incorrect_token
        ) for ex in dataset]
        return dataset