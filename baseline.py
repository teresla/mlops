import torch
import datasets
import evaluate
import lightning as L
import wandb
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from typing import Optional, List
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*promote has been superseded by promote_options='default'.*"
)

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
        model_name_or_path: str = MODEL_NAME,
        task_name: str = TASK_NAME,
        max_seq_length: int = MAX_SEQ_LENGTH,
        train_batch_size: int = BATCH_SIZE,
        eval_batch_size: int = BATCH_SIZE,
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

    def compute_class_weights(self):
        """Compute class weights based on the label distribution of the training data."""
        labels = np.array([example['labels'] for example in self.dataset['train']])
        class_sample_count = np.bincount(labels)
        weight = 1.0 / class_sample_count
    
        return torch.tensor(weight * 1000, dtype=torch.float)

    def setup(self, stage: str):
        """Set up datasets and tokenization."""
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


class GLUETransformer(L.LightningModule):
    """LightningModule for training Transformer models on GLUE tasks."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        gradient_clip_val: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        eval_splits: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.metric = evaluate.load("glue", task_name)

        # Set class weights to equal weights if not provided
        if class_weights is None:
            self.class_weights = torch.ones(num_labels)  # Equal weighting
        else:
            self.class_weights = class_weights  # Use provided class weights

        # Initialize lists to store predictions and labels
        self.val_preds = []
        self.val_labels = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        
        # Move class_weights to the same device as outputs.logits
        class_weights = self.class_weights.to(outputs.logits.device)
        
        # Use the class_weights in the loss function
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(outputs.logits, batch["labels"])
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == batch["labels"]).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        logits = outputs.logits

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, dim=-1)
        else:
            preds = logits.squeeze()

        labels = batch["labels"]

       # Store preds and labels for metric computation
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of the epoch."""
        # Concatenate all predictions and labels
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)

        # Clear lists for the next epoch
        self.val_preds.clear()
        self.val_labels.clear()

        # Compute metrics
        self.metric.add_batch(predictions=preds, references=labels)
        result = self.metric.compute()

        # Log metrics
        for key, value in result.items():
            self.log(f"val_{key}", value, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def on_after_backward(self):
        """Log gradient norms after each backward pass."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.experiment.log({"total_grad_norm": total_norm})


# Define a function to create the data module and class weights once
def initialize_data_module(
    model_name_or_path: str = "distilbert-base-uncased", 
    task_name: str = "mrpc", 
    train_batch_size: int = 16, 
    eval_batch_size: int = 16, 
    max_seq_length: int = 128
):
    """Initialize the data module and compute class weights."""
    dm = GLUEDataModule(
        model_name_or_path=model_name_or_path,
        task_name=task_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        max_seq_length=max_seq_length,
    )
    
    # Set up the datamodule once
    dm.setup("fit")

    # Compute class weights once
    class_weights = dm.compute_class_weights()

    return dm, class_weights


# Define the experiment function
def run_experiment(
    run_comment: str,
    dm: GLUEDataModule = None,  # Default to None, should be passed once initialized
    class_weights: torch.Tensor = None,  # Default to None, should be passed once calculated
    task_name: str = TASK_NAME,
    model_name_or_path: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
    train_batch_size: int = BATCH_SIZE,
    max_seq_length: int = MAX_SEQ_LENGTH,
    learning_rate: float = 2e-5,
    warmup_steps = 0,
    weight_decay: float = 0.0,
    gradient_clip_val: float = 1.0,
    epochs: int = EPOCHS,
    accumulate_grad_batches: int = 1,
):
    """Run a single experiment with the specified hyperparameters."""
    run_name = (
        f"{run_comment}_lr_{learning_rate}_bs_{train_batch_size}_"
        f"maxlen_{max_seq_length}_warmup_{warmup_steps}_wd_{weight_decay}_clip_{gradient_clip_val}"
    )

    wandb_logger = WandbLogger(project="week2", entity="machine-learning-ops", name=run_name)

    # Reuse the class weights and datamodule (dm)
    model = GLUETransformer(
        model_name_or_path=model_name_or_path,
        num_labels=num_labels,
        eval_splits=dm.eval_splits,
        task_name=task_name,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        train_batch_size=train_batch_size,
        gradient_clip_val=gradient_clip_val,
        class_weights=class_weights,  # Pass class weights to the model
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=dm)
    wandb.finish()


def main():
    """Main function to execute runs."""
    L.seed_everything(42)
    TASK_NAME = "mrpc"
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 2
    BATCH_SIZE = 32
    MAX_SEQ_LENGTH = 64
    EPOCHS = 3
    # Set the hyperparameters here directly
    run_comment = "manual_run"
    task_name = "mrpc"
    model_name_or_path = "distilbert-base-uncased"
    num_labels = 2
    train_batch_size = 16
    eval_batch_size = 16
    max_seq_length = 128
    learning_rate = 2e-5
    warmup_steps = None
    weight_decay = 0.0
    gradient_clip_val = 1.0
    epochs = 3
    accumulate_grad_batches = 1


    # Initialize data module and class weights once
    dm, class_weights = initialize_data_module()

    # Run the experiment using the same datamodule and class weights
    run_experiment(
    run_comment="w2_incr lr more",
    train_batch_size=16,
    dm=dm,  # Pass the initialized datamodule
    class_weights=class_weights,  # Pass the precomputed class weights
    learning_rate=6e-5,  # Only specify what's different
    weight_decay=0.3,  # Adjust as needed
    warmup_steps=0, # 10% of steps per epoch
    gradient_clip_val=1.5
    )




if __name__ == "__main__":
    main()