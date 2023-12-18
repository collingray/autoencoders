import torch
import wandb
from torch import optim
from autoencoder import AutoEncoder
from configs import *

# Hyperparameters
n = 10  # Dimension of the input
m = 5  # Dimension of the hidden layer
lambda_reg = 0.01  # Regularization strength
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 64  # Batch size for training
epochs = 50  # Number of training epochs
dataset_size = 1000  # Assuming we have 1000 samples

# Sample synthetic dataset (just for demonstration purposes)
dataset = torch.randn(dataset_size, n)

# DataLoader can be used to split the dataset into batches
# For simplicity, we are skipping DataLoader here and processing the entire dataset at once

# Model instantiation
autoencoder = AutoEncoder(autoencoder_cfg)

# Optimizer setup
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

checkpoint = 0

try:
    # Initialize W&B project
    wandb.init(project=training_cfg["wandb_project"], entity=training_cfg["wandb_entity"])

    # Training loop
    for epoch in range(epochs):
        for i in range(0, dataset_size, batch_size):
            # Get the mini-batch
            x_batch = dataset[i:i + batch_size]

            # Forward pass
            x_out, f = autoencoder(x_batch)

            # Loss computation
            loss = autoencoder.loss(x_batch, x_out, f, lambda_reg)

            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss

            # Gradient step
            optimizer.step()  # Perform updates using calculated gradients

        # Print loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Log loss to W&B
        wandb.log({"loss": loss.item()})

        # Save model checkpoint
        if (epoch + 1) % training_cfg["epochs_per_checkpoint"] == 0:
            autoencoder.save(checkpoint)
            checkpoint += 1
finally:
    # Save the model
    autoencoder.save(checkpoint)