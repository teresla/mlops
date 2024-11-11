# config.py

from dataclasses import dataclass

@dataclass
class ModelConfig:
    task_name: str = "mrpc"
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_seq_length: int = 64
    train_batch_size: int = 32
    eval_batch_size: int = 32

@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_clip_val: float = 1.0
    epochs: int = 1
    accumulate_grad_batches: int = 1

@dataclass
class ExperimentConfig:
    run_comment: str
    model_config: ModelConfig
    training_config: TrainingConfig 