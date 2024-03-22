import torch
import wandb
from autoencoder import *
from buffer import *

lr = 1e-4
num_tokens = int(2e10)  # total number of tokens to train on, the dataset will wrap around as needed
batch_size = 32
beta1 = 0.9
beta2 = 0.99

wandb_project = "autoencoder"
wandb_entity = "collingray"
wandb.init(project=wandb_project, entity=wandb_entity)

buffer_cfg = ActivationsBufferConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    layers=[0],
    dataset_name="roneneldan/TinyStories",
    dataset_split="train",
    buffer_size=65536,
    buffer_device="cpu"
)
buffer = ActivationsBuffer(buffer_cfg)

encoder_cfg = AutoEncoderConfig(n_dim=14336, m_dim=14336 * 2)  # 14336*5 = 71680
encoder = AutoEncoder(encoder_cfg)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), foreach=False)

try:
    for i in range(num_tokens // batch_size):
        enc, l1, l2, loss = encoder.forward(buffer.next(batch=batch_size).to(encoder_cfg.device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(f"Step {i}, l1_loss: {l1.item()}, l2_loss: {l2.item()}, total_loss: {loss.item()}")
            wandb.log({"l1_loss": l1.item(), "l2_loss": l2.item(), "total_loss": loss.item()})
            if i % 10000 == 0 and i > 0:
                encoder.save(i // 1000)
finally:
    # Save the model
    encoder.save("final")
