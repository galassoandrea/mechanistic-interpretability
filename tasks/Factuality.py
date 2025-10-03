from dataclasses import dataclass
import random
from typing import List
import pandas as pd
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

def pad_sequences(tokens: List[torch.Tensor], max_length) -> torch.Tensor:
    """Pad each token sequence to the maximum length in the batch"""
    padded_tokens = [
        F.pad(t, (0, max_length - t.shape[0]))
        for t in tokens
    ]
    return torch.stack(padded_tokens, dim=0)


class FactualityTorchDataset(Dataset):
    def __init__(self, factuality_examples):
        self.examples = factuality_examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "statement": ex.statement,
            "label": torch.tensor(ex.label, dtype=torch.long),
            "prompt": ex.prompt,
            "clean_tokens": torch.tensor(ex.clean_tokens, dtype=torch.long),
        }


@dataclass
class FactualityExample:
    """Represents a single Factuality example"""
    clean_statement: str
    corrupted_statement: str
    label: int
    clean_prompt: str
    corrupted_prompt: str
    clean_tokens: List[str]
    corrupted_tokens: List[str]


class FactualityDatasetBuilder:
    """Builds a dataset for the Factuality task"""

    def __init__(self, model, topic):
        if topic == "animals":
            self.df = pd.read_csv("tasks/factuality_data/animals_true_false.csv")
            self.df["topic"] = "animals"
        elif topic == "capitals":
            self.df = pd.read_csv("tasks/factuality_data/capitals_true_false.csv")
            self.df["topic"] = "capitals"
        elif topic == "elements":
            self.df = pd.read_csv("tasks/factuality_data/elements_true_false.csv")
            self.df["topic"] = "elements"
        self.model = model
        self.system_role = ("You are a judge and your role is to judge whether the provided statement is true or false,"
                            " based on your knowledge. Answer with a 1 if the statement is true"
                            " and 0 if the statement is false."
                            " Here there are a few examples: ")

        # Pick 3 random elements from the dataset
        self.examples = self.df.sample(n=3, random_state=42)

        # Remove the 3 examples from the main dataframe
        self.df = self.df.drop(self.examples.index).reset_index(drop=True)

        # Convert examples to list of dicts
        self.examples = self.examples.to_dict(orient="records")

        # Pools of elements for corruption
        self.cities = ['Zimbabwe', 'Uganda', 'Argentina', 'North Korea', 'South Africa', 'Congo', 'Algeria', 'United States', 'Russia', 'Tanzania', 'Saudi Arabia', 'Papua New Guinea', 'Pakistan', 'Japan', 'Ukraine', 'Montenegro', 'Germany', 'Brazil', 'Nigeria', 'India', 'Philippines', 'United Arab Emirates', 'Greece', 'Uzbekistan', 'Czechia', 'South Korea', 'Guatemala', 'Macau', 'Djibouti', 'Mexico', 'Switzerland', 'Mauritania', 'Senegal', 'Cayman Islands', 'Malaysia', 'United Kingdom', 'China', 'Jamaica', 'Haiti']
        self.animals = ['beaver', 'leopard', 'swan', 'polar bear', 'wolverine', 'salmon', 'rhinoceros', 'manta', 'gecko', 'giant anteater', 'snake', 'skunk', 'hippopotamus', 'cow', 'vulture', 'deer', 'sparrow', 'seagull', 'mongoose', 'rat', 'crocodile', 'flamingo', 'tapir', 'jellyfish', 'walrus', 'hedgehog', 'hamster', 'giraffe', 'ostrich', 'dog', 'slug', 'tortoise', 'hummingbird', 'tiger', 'camel', 'zebra', 'lobster', 'kangaroo', 'aardvark', 'dolphin', 'manta ray', 'tuna', 'elephant', 'peacock', 'goldfish', 'raccoon', 'alpaca', 'axolotl', 'armadillo']
        self.elements = ['Tantalum', 'Calcium', 'Gadolinium', 'Samarium', 'Cerium', 'Iridium', 'Rhenium', 'Scandium', 'Nickel', 'Thallium', 'Silver', 'Oxygen', 'Actinium', 'Promethium', 'Astatine', 'Osmium', 'Platinum', 'Tin', 'Nitrogen', 'Beryllium', 'Arsenic', 'Lead', 'Mercury', 'Fluorine', 'Lanthanum', 'Radium', 'Iron', 'Zirconium', 'Praseodymium', 'Lithium', 'Francium', 'Ruthenium', 'Iodine', 'Neon', 'Copper', 'Erbium', 'Krypton', 'Rubidium', 'Thorium', 'Protactinium', 'Rhodium', 'Antimony', 'Boron', 'Bismuth', 'Tellurium', 'Titanium', 'Cadmium', 'Sulfur', 'Holmium', 'Gallium', 'Technetium', 'Tungsten']

    def corrupt_sentence(self, sentence, topic):
        words = sentence.split()
        if topic == "animals":
            clean_animal = words[1]
            other_animals = [a for a in self.animals if a != clean_animal]
            corr_animal = random.choice(other_animals)
            words[1] = corr_animal
        elif topic == "elements":
            clean_element = words[0]
            other_elements = [a for a in self.elements if a != clean_element]
            corr_element = random.choice(other_elements)
            words[0] = corr_element
        elif topic == "capitals":
            clean_city = words[0]
            other_cities = [a for a in self.elements if a != clean_city]
            corr_city = random.choice(other_cities)
            words[0] = corr_city
        sentence = " ".join(words)
        return sentence

    def build_base_prompt(self):
        """Builds the base prompt with system role and examples"""
        prompt = self.system_role
        for example in self.examples:
            prompt += f'\nStatement: {example["statement"]}\nEvaluation: {example["label"]}\nNow judge the following statement.\n'
        return prompt

    def build_single_prompt(self, example):
        """Builds a single prompt for a given example"""
        initial_prompt = self.build_base_prompt()
        clean_prompt = f'{initial_prompt}\nStatement: {example["statement"]}\nEvaluation: '
        corrupted_statement = self.corrupt_sentence(example["statement"], example["topic"])
        corrupted_prompt = f'{initial_prompt}\nStatement: {corrupted_statement}\nEvaluation: '
        clean_tokens = self.model.to_tokens(clean_prompt, prepend_bos=True).squeeze(0)
        corrupted_tokens = self.model.to_tokens(corrupted_prompt, prepend_bos=True).squeeze(0)
        return FactualityExample(
            clean_statement=example['statement'],
            corrupted_statement=corrupted_statement,
            label=example['label'],
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
            clean_tokens=clean_tokens,
            corrupted_tokens=corrupted_tokens,
        )

    def build_dataset(self):
        """Builds the complete dataset"""
        dataset = []
        clean_tokens = []
        corrupted_tokens = []
        for _, row in self.df.iterrows():
            example = {
                "statement": row['statement'],
                "label": row['label'],
                "topic": row['topic']
            }
            factuality_example = self.build_single_prompt(example)
            clean_tokens.append(factuality_example.clean_tokens)
            corrupted_tokens.append(factuality_example.corrupted_tokens)
            dataset.append(factuality_example)
        # Pad sequences to the maximum length
        max_length = max(token.shape[0] for token in clean_tokens)
        padded_clean_tokens = pad_sequences(clean_tokens, max_length)
        padded_corrupted_tokens = pad_sequences(corrupted_tokens, max_length)
        for i, example in enumerate(dataset):
            example.clean_tokens = padded_clean_tokens[i]
            example.corrupted_tokens = padded_corrupted_tokens[i]
        return dataset


