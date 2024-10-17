import torch
import datasets
import evaluate
import pytorch_lightning as L
import wandb
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
import optuna
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

TASK_NAME = "mrpc"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
EPOCHS = 3


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

    def __init__(self, model_name_or_path: str = MODEL_NAME, task_name: str = TASK_NAME, max_seq_length: int = MAX_SEQ_LENGTH, train_batch_size: int = BATCH_SIZE, eval_batch_size: int = BATCH_SIZE, **kwargs):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.TASK_TEXT_FIELD_MAP[task_name]
        self.num_labels = self.GLUE_TASK_NUM_LABELS[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def compute_class_weights(self):
        """Compute class weights based on the label distribution of the training data."""
        labels = np.array([example['labels'] for example in self.dataset['train']])
        class_sample_count = np.bincount(labels)
        weight = 1.0 / class_sample_count
        return torch.tensor(weight * 1000, dtype=torch.float)

    def setup(self, stage: str):
        """Set up datasets and tokenization."""
        # Only load the dataset if it's not already loaded
        if not hasattr(self, "dataset"):
            self.dataset = datasets.load_dataset("glue", self.task_name, cache_dir='./cache_dir')

            for split in self.dataset.keys():
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    num_proc=8,
                    remove_columns=["label"],
                )
                columns = [c for c in self.dataset[split].column_names if c in self.LOADER_COLUMNS]
                self.dataset[split].set_format(type="torch", columns=columns)

            self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]


    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)

    def convert_to_features(self, example_batch, indices=None):
        """Tokenize the input texts."""
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
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


class GLUETransformer(L.LightningModule):
    def __init__(self, model_name_or_path: str, num_labels: int, task_name: str, learning_rate: float = 2e-5, warmup_steps: int = 0, weight_decay: float = 0.0, train_batch_size: int = 32, gradient_clip_val: float = 1.0, class_weights: Optional[torch.Tensor] = None, scheduler_type: str = "linear", scheduler_power: float = 1.0, num_cycles: int = 1, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = evaluate.load("glue", task_name)

        self.class_weights = class_weights if class_weights is not None else torch.ones(num_labels)

        self.val_preds = []
        self.val_labels = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, batch["labels"])
        preds = torch.argmax(outputs.logits, dim=-1)
        acc = (preds == batch["labels"]).float().mean()

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        self.val_preds.clear()
        self.val_labels.clear()

        self.metric.add_batch(predictions=preds, references=labels)
        result = self.metric.compute()
        for key, value in result.items():
            self.log(f"val_{key}", value, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = self.get_scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def get_scheduler(self, optimizer):
        if self.hparams.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.scheduler_type == "cosine_with_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_cycles=self.hparams.num_cycles,
            )
        elif self.hparams.scheduler_type == "polynomial":
            return get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
                power=self.hparams.scheduler_power,
            )
        elif self.hparams.scheduler_type == "constant":
            return get_constant_schedule(optimizer)
        elif self.hparams.scheduler_type == "constant_with_warmup":
            return get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.hparams.scheduler_type}")


def initialize_data_module(model_name_or_path="distilbert-base-uncased", task_name="mrpc", train_batch_size=32, eval_batch_size=32, max_seq_length=128):
    dm = GLUEDataModule(model_name_or_path=model_name_or_path, task_name=task_name, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, max_seq_length=max_seq_length)
    dm.setup("fit")
    class_weights = dm.compute_class_weights()
    return dm, class_weights


def objective(trial, dm, class_weights):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.5, 2.0)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)

    scheduler_type = trial.suggest_categorical("scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    num_cycles = trial.suggest_int("num_cycles", 1, 5) if scheduler_type == "cosine_with_restarts" else 1
    scheduler_power = trial.suggest_float("scheduler_power", 0.5, 2.0) if scheduler_type == "polynomial" else 1.0

    run_name = f"trial_lr_{learning_rate}_wd_{weight_decay}_clip_{gradient_clip_val}_scheduler_{scheduler_type}"

    model = GLUETransformer(
        model_name_or_path=MODEL_NAME,
        num_labels=NUM_LABELS,
        task_name=TASK_NAME,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        warmup_steps=warmup_steps,
        class_weights=class_weights,
        scheduler_type=scheduler_type,
        scheduler_power=scheduler_power,
        num_cycles=num_cycles,
    )

    wandb_logger = WandbLogger(project="week1", entity="machine-learning-ops", name=run_name)
    trainer = L.Trainer(max_epochs=EPOCHS, logger=wandb_logger, gradient_clip_val=gradient_clip_val)

    trainer.fit(model, datamodule=dm)

    best_score = trainer.callback_metrics.get("val_accuracy")
    wandb.finish()
    return best_score.item()


def main():
    dm, class_weights = initialize_data_module()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, dm, class_weights), n_trials=20)

    best_trial = study.best_trial
    print(f"Best trial value: {best_trial.value}")
    print(f"Best trial parameters: {best_trial.params}")


if __name__ == "__main__":
    main()
