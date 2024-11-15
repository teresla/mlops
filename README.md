# MLOps (Machine Learning Operations) Project - HSLU

## Overview

This repository is part of the Machine Learning Operations (MLOps) course at Hochschule Luzern (HSLU). The MLOps course explores the intersection of Machine Learning, DevOps, and Data Engineering, aiming to equip students with skills to build ML systems that are reliable, scalable, and ready for production with minimal manual overhead. Through these projects, we demonstrate real-world applications of MLOps techniques, from hyperparameter tuning in Project 1 to containerization in Project 2, emphasizing scalable and reproducible machine learning workflows.

## Project 1: Hyperparameter Tuning

### Problem Description

The objective of Project 1 was to fine-tune the `DistilBERT` model on the MRPC dataset for paraphrase detection. The focus was on improving validation accuracy by systematically tuning hyperparameters. Key parameters I tuned was learning rate, batch size, and gradient clipping were experimented with to understand their impact on model performance. All experiments were tracked with Weights & Biases (wandb) for easy comparison and reproducibility.

### Key Takeaways

- **Batch Size and Gradient Accumulation**: Initial tuning revealed a need for adjusting batch size along with gradient accumulation to avoid NaN issues.
- **Truncation Strategy**: Despite typically setting max sequence length during preprocessing, we explored shorter sequence lengths to test impact. This strategy required justification due to the potential of increased truncation.
- **Important vs. Unimportant Hyperparameters**: Early experimentation showed that tuning less influential hyperparameters could lead to redundant tuning rounds. Future tuning was optimized by prioritizing high-impact hyperparameters first.
- **Learning Rate Tuning**: Following feedback, exponential step tuning was applied to the learning rate for efficient search within defined ranges.

### Results and Reflections

The best-performing model configurations were achieved using Optuna for automated hyperparameter tuning, yielding a final validation accuracy of **85.3%**. Feedback underscored the importance of focusing on key parameters and minimizing unnecessary tuning of lower-impact hyperparameters. See wandb for full experiment tracking details and performance comparisons.

---

## Project 2: Containerization with Docker

### Problem Description

Project 2 builds on the hyperparameter tuning task by containerizing the training environment using Docker. This setup ensures that the local development environment aligns with cloud-based production, achieving reproducibility across setups. The task involves converting the Project 1 model training process into a script that can be executed as a single call within a Docker container.

### Steps for Running the Docker Container

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mlops-course-project.git
   cd mlops-course-project
   ```

2. **Build the Docker Image**:
   Ensure Docker is installed and running on your machine. Build the image using:
   ```bash
   docker build -t mlops_project .
   ```

3. **Run a Training Job**:
   Run the containerized training job with the best hyperparameters from Project 1:
   ```bash
   docker run mlops_project --checkpoint_dir models --lr 3e-5 --batch_size 16
   ```
   The job logs results to wandb for detailed tracking, accessible through the project’s wandb dashboard.

4. **Run on GitHub Codespaces**:
   For testing in GitHub Codespaces, clone the repository and follow the above steps. Any adjustments or insights are documented in the Project 2 Report.

### Key Challenges and Insights

- **Local vs. Cloud Execution**: Running Docker on local machines without GPU support required adjustments for longer run times, whereas Codespaces required optimized image size to fit memory constraints.
- **wandb Integration**: The containerized setup logs to wandb, ensuring experiment results are consistent and comparable across environments.

---

## Experiment Tracking

Both projects utilize wandb for experiment tracking, enabling visualization of training metrics and insights on parameter importance. Check the wandb project dashboard [here](wandb-link) for a detailed overview of experiments.

---

## Conclusion

Through these projects, we implemented MLOps best practices for hyperparameter tuning and environment reproducibility. Project 1 allowed for an in-depth understanding of hyperparameter impact, while Project 2 expanded the training pipeline's portability and consistency across different environments.
