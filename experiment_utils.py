# experiment_utils.py

import os
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from typing import Tuple
from config import ExperimentConfig
from data_module import GLUEDataModule
from glue_model import GLUETransformer
import wandb

def create_run_name(config: ExperimentConfig) -> str:
    """Create a descriptive run name from configuration."""
    return (
        f"{config.run_comment}_lr_{config.training_config.learning_rate}_"
        f"warmup_{config.training_config.warmup_steps}_"
        f"wd_{config.training_config.weight_decay}_"
    )

def setup_wandb_logger(config: ExperimentConfig) -> WandbLogger:
    """Configure and initialize WandB logging."""
    run_name = create_run_name(config)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    
    # Ensure API key is set
    if not wandb_api_key:
        raise EnvironmentError("WANDB_API_KEY environment variable is not set.")
    
    # Login to wandb
    wandb.login(key=wandb_api_key)
    
    return WandbLogger(
        project="paraphrase-detection-distilbert-mrpc",
        entity="machine-learning-ops",
        name=run_name
    )

def create_trainer(config: ExperimentConfig, logger: WandbLogger) -> L.Trainer:
    """Create and configure the trainer."""
    return L.Trainer(
        max_epochs=config.training_config.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        gradient_clip_val=config.training_config.gradient_clip_val,
        log_every_n_steps=10,
        accumulate_grad_batches=config.training_config.accumulate_grad_batches,
    )

def initialize_experiment(config: ExperimentConfig) -> Tuple[GLUEDataModule, torch.Tensor]:
    """Initialize data module and compute class weights."""
    dm = GLUEDataModule(
        model_name_or_path=config.model_config.model_name,
        task_name=config.model_config.task_name,
        train_batch_size=config.model_config.train_batch_size,
        eval_batch_size=config.model_config.eval_batch_size,
        max_seq_length=config.model_config.max_seq_length,
    )
    dm.setup("fit")
    class_weights = dm.compute_class_weights()
    return dm, class_weights
