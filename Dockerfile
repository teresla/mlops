# Use a Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment file
COPY environment.yml .

# Create the conda environment based on environment.yml
RUN conda env create -f environment.yml

# Activate the environment in every shell session
SHELL ["conda", "run", "-n", "mlops", "/bin/bash", "-c"]

# Ensure the environment is activated and set up PATH
ENV PATH /opt/conda/envs/mlops/bin:$PATH

# Copy the rest of your code into the container
COPY . .

# Command to run your code
CMD ["python", "main.py", "--learning_rate", "3e-5", "--weight_decay", "0.01", "--warmup_steps", "100"]
