from dataclasses import dataclass
from typing import List
import pandas as pd

@dataclass
class FactualityExample:
    """Represents a single Factuality example"""
    statement: str
    label: int
    prompt: str
    clean_tokens: List[str]


class FactualityDatasetBuilder:
    """Builds a dataset for the Factuality task"""

    def __init__(self, model):
        self.df = pd.read_csv("tasks/factuality_data/animals_true_false.csv")
        self.model = model
        self.system_role = ("You are a judge and your role is to judge whether the provided statement is true or false,"
                            " based on your knowledge. Answer with a 1 if the statement is true"
                            " and 0 if the statement is false."
                            " Here there are a few examples: ")
        self.examples = [
            {
                "statement": "A dog is a type of animal.",
                "label": 1
            },
            {
                "statement": "A cat is a type of vehicle.",
                "label": 0
            },
            {
                "statement": "Elephants are the largest land animals.",
                "label": 1
            },
            {
                "statement": "The sun revolves around the earth.",
                "label": 0
            }
        ]

        self.complex_examples = [
            {
                "statement": "Who was the next British Prime Minister after Arthur Balfour?",
                "label": 0
            },
            {
                "statement": "The band Exile had a 70s No 1 hit with Kiss You All Over.",
                "label": 1
            },
            {
                "statement": "The common mineral used to make casts, moulds, blackboard chalk and plaster of Paris is calcium carbonate.",
                "label": 0
            }
        ]

    def build_base_prompt(self):
        """Builds the base prompt with system role and examples"""
        prompt = self.system_role
        for example in self.examples:
            prompt += f'\nStatement: {example["statement"]}\nEvaluation: {example["label"]}\nNow judge the following statement.\n'
        return prompt

    def build_single_prompt(self, example):
        """Builds a single prompt for a given example"""
        initial_prompt = self.build_base_prompt()
        prompt = f'{initial_prompt}\nStatement: {example["statement"]}\nEvaluation: '
        prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True)
        return FactualityExample(
            statement=example['statement'],
            label=example['label'],
            prompt=prompt,
            clean_tokens=prompt_tokens[0]
        )

    def build_dataset(self):
        """Builds the complete dataset"""
        dataset = []
        for _, row in self.df.iterrows():
            example = {
                "statement": row['statement'],
                "label": row['label']
            }
            factuality_example = self.build_single_prompt(example)
            dataset.append(factuality_example)
        for example in self.complex_examples:
            factuality_example = self.build_single_prompt(example)
            dataset.append(factuality_example)
        return dataset