# glue_model.py

import torch
import evaluate
import lightning as L
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from typing import Optional, List

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

        # Set class weights
        if class_weights is None:
            self.class_weights = torch.ones(num_labels)
        else:
            self.class_weights = class_weights

        # Initialize prediction storage
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

        # Store predictions and labels
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of the epoch."""
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)

        # Clear lists for next epoch
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