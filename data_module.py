# data_module.py

import torch
import numpy as np
import datasets
import lightning as L
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class GLUEDataModule(L.LightningDataModule):
    """Data module for the GLUE dataset."""

    TASK_TEXT_FIELD_MAP = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    GLUE_TASK_NUM_LABELS = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    LOADER_COLUMNS = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.TASK_TEXT_FIELD_MAP[task_name]
        self.num_labels = self.GLUE_TASK_NUM_LABELS[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str = None):
        """Set up datasets and tokenization."""
        if not hasattr(self, 'dataset') or self.dataset is None:
            self.dataset = datasets.load_dataset("glue", self.task_name, cache_dir='./cache_dir')

            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    num_proc=8,  # Adjust this number based on your CPU cores
                    remove_columns=["label"],
                )
                columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
                self.dataset[split].set_format(type="torch", columns=columns)

            self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]


    def compute_class_weights(self):
        """Compute class weights based on the label distribution of the training data."""
        labels = np.array([example['labels'] for example in self.dataset["train"]])
        class_weights = torch.tensor(np.bincount(labels), dtype=torch.float)
        class_weights = class_weights.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return class_weights 
        
    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=8,  # Adjust based on CPU cores
            pin_memory=True,
        )

    def val_dataloader(self):
        """Validation dataloader."""
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                num_workers=8,  # Adjust based on CPU cores
                pin_memory=True,
            )
        else:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=16,  # Adjust based on CPU cores
                    pin_memory=True,
                )
            for x in self.eval_splits
        ]

    def test_dataloader(self):
        """Test dataloader."""
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        else:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        """Tokenize the input texts."""
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(
                zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]])
            )
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        features["labels"] = example_batch["label"]
        return features