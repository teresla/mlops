# main.py

import lightning as L
import sys
import argparse
from config import ExperimentConfig, ModelConfig, TrainingConfig
from experiment_runner import run_experiment

def main():
    """Main function to execute experiments."""
    # Set random seed
    L.seed_everything(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Experiment configuration")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--run_comment", type=str, default="command_line_run", help="Comment for this run")
    args = parser.parse_args()

    # Create configurations with command-line arguments
    model_config = ModelConfig(
        train_batch_size=16,
        max_seq_length=128
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip_val=1.5
    )

    experiment_config = ExperimentConfig(
        run_comment=args.run_comment,
        model_config=model_config,
        training_config=training_config
    )

    # Run experiment
    run_experiment(experiment_config)

if __name__ == "__main__":
    main()