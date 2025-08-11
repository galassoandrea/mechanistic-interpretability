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

@dataclass
class IOIExample:
    """Represents a single IOI example with both clean and corrupted versions"""
    clean_prompt: str
    corrupted_prompt: str
    subject: str
    indirect_object: str
    verb: str
    clean_tokens: torch.Tensor
    corrupted_tokens: torch.Tensor
    answer_token_pos: int
    correct_answer: str
    incorrect_answer: str
    correct_token: torch.Tensor
    incorrect_token: torch.Tensor


class IOIDatasetBuilder:
    """Builds a toy IOI dataset with clean and corrupted prompts"""

    def __init__(self, model):
        self.names = [
            ' Mary', ' John', ' Tom', ' James', ' Dan', ' Sid', ' Martin', ' Amy'
        ]

        self.verbs = [
            "gave", "handed", "passed", "sent", "showed", "offered", "brought",
            "delivered", "presented", "provided", "sold", "lent", "threw"
        ]

        self.objects = [
            "the book", "a letter", "the key", "a gift", "the document", "a flower",
            "the package", "a note", "the phone", "a message", "the ticket", "a card"
        ]

        self.model = model

    def generate_ioi_example(self) -> IOIExample:
        """Generate a single IOI example with clean and corrupted versions"""
        # Sample names ensuring they're different
        subject, indirect_object = random.sample(self.names, 2)
        verb = random.choice(self.verbs)
        obj = random.choice(self.objects)

        # Template: "When [A] and [B] went to the store, [A] gave [obj] to"
        # The model should predict [B] (indirect object)
        clean_prompt = f"When {subject} and {indirect_object} went to the store, {subject} {verb} {obj} to"

        # Corrupted version: swap the indirect object with another random name
        other_names = [n for n in self.names if n not in [subject, indirect_object]]
        corrupted_indirect = random.choice(other_names)
        corrupted_prompt = f"When {subject} and {corrupted_indirect} went to the store, {subject} {verb} {obj} to"

        # Tokenize the prompts and the correct/incorrect answers
        clean_tokens = self.model.to_tokens(clean_prompt).squeeze(0)  # Remove batch dimension
        corrupted_tokens = self.model.to_tokens(corrupted_prompt).squeeze(0)  # Remove batch dimension
        correct_token = self.model.to_single_token(indirect_object)
        incorrect_token = self.model.to_single_token(corrupted_indirect)

        # Find answer position (last token position)
        answer_token_pos = clean_tokens.shape[0] - 1

        return IOIExample(
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            subject=subject,
            indirect_object=indirect_object,
            verb=verb,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
            answer_token_pos=answer_token_pos,
            correct_answer=indirect_object,
            incorrect_answer=corrupted_indirect,
            correct_token=correct_token,
            incorrect_token=incorrect_token
        )

    def pad_sequences(self, tokens: List) -> torch.Tensor:
        """Pad each token to the maximum token length in the dataset"""
        max_length = max(token.shape[0] for token in tokens)
        padded_tokens = [
            F.pad(t, (0, max_length - t.shape[0]))
            for t in tokens
        ]
        # Convert to tensor
        tokens = torch.stack(padded_tokens, dim=0)
        return tokens

    def build_dataset(self, num_samples: int = 100) -> List[IOIExample]:
        """Build a full IOI dataset"""
        dataset = []
        clean_tokens = []
        corrupted_tokens = []
        for _ in tqdm(range(num_samples), desc="Generating IOI examples"):
            example = self.generate_ioi_example()
            clean_tokens.append(example.clean_tokens)
            corrupted_tokens.append(example.corrupted_tokens)
            dataset.append(example)
        # Pad sequences to the maximum length
        clean_tokens = self.pad_sequences(clean_tokens)
        corrupted_tokens = self.pad_sequences(corrupted_tokens)
        # Update dataset with padded tokens
        for i, example in enumerate(dataset):
            example.clean_tokens = clean_tokens[i]
            example.corrupted_tokens = corrupted_tokens[i]
        # Convert dataset to a list of IOIExample objects
        dataset = [IOIExample(
            clean_prompt=ex.clean_prompt,
            corrupted_prompt=ex.corrupted_prompt,
            subject=ex.subject,
            indirect_object=ex.indirect_object,
            verb=ex.verb,
            clean_tokens=ex.clean_tokens,
            corrupted_tokens=ex.corrupted_tokens,
            answer_token_pos=ex.answer_token_pos,
            correct_answer=ex.correct_answer,
            incorrect_answer=ex.incorrect_answer,
            correct_token=ex.correct_token,
            incorrect_token=ex.incorrect_token
        ) for ex in dataset]
        return dataset


