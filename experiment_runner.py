# experiment.runner.py

import wandb
import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from typing import Tuple
import sys
import os

from config import ExperimentConfig
from glue_model import GLUETransformer
from data_module import GLUEDataModule
from experiment_utils import (
    initialize_experiment,
    setup_wandb_logger,
    create_trainer,
)


def run_experiment(config: ExperimentConfig) -> None:
    """Run a single experiment with the specified configuration."""
    # Initialize experiment components
    dm, class_weights = initialize_experiment(config)
    logger = setup_wandb_logger(config)
    
    # Create model
    model = GLUETransformer(
        model_name_or_path=config.model_config.model_name,
        num_labels=config.model_config.num_labels,
        eval_splits=dm.eval_splits,
        task_name=config.model_config.task_name,
        learning_rate=config.training_config.learning_rate,
        warmup_steps=config.training_config.warmup_steps,
        weight_decay=config.training_config.weight_decay,
        train_batch_size=config.model_config.train_batch_size,
        gradient_clip_val=config.training_config.gradient_clip_val,
        class_weights=class_weights,
    )

    # Create and run trainer
    trainer = create_trainer(config, logger)
    trainer.fit(model, datamodule=dm)
    wandb.finish() 