import torch
import wandb
from autoencoder import *
from buffer import *

lr = 1e-4
num_tokens = int(2e10)  # total number of tokens to train on, the dataset will wrap around as needed
batch_size = 8
beta1 = 0.9
beta2 = 0.99

wandb_project = "autoencoder"
wandb_entity = "collingray"
wandb.init(project=wandb_project, entity=wandb_entity)

model = "mistralai/Mistral-7B-Instruct-v0.1"
dataset = "roneneldan/TinyStories"
buffer_cfg = ActivationsBufferConfig(model_name=model, layers=[0], dataset_name=dataset, dataset_split="train")
buffer = ActivationsBuffer(buffer_cfg)

encoder_cfg = AutoEncoderConfig(n_dim=14336, m_dim=14336 * 2)  # 14336*5 = 71680
encoder = AutoEncoder(encoder_cfg)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), foreach=False)

try:
    for i in range(num_tokens // batch_size):
        enc, l1, l2, loss = encoder.forward(buffer.next(batch=batch_size))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i > 0 and i % 1000 == 0:
            print(f"Step {i}, l1_loss: {l1.item()}, l2_loss: {l2.item()}, total_loss: {loss.item()}")
            wandb.log({"l1_loss": l1.item(), "l2_loss": l2.item(), "total_loss": loss.item()})
            encoder.save(i // 1000)
finally:
    # Save the model
    encoder.save("final")
