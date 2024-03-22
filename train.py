import torch
import wandb
from autoencoder import *
from buffer import *
import time
from tqdm import tqdm

lr = 1e-4
num_activations = int(2e10)  # total number of tokens to train on, the dataset will wrap around as needed
batch_size = 32
beta1 = 0.9
beta2 = 0.99
steps_per_report = 100
steps_per_save = 10000

wandb_project = "autoencoder"
wandb_entity = "collingray"
wandb.init(project=wandb_project, entity=wandb_entity)

buffer_cfg = ActivationsBufferConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    layers=[0],
    dataset_name="roneneldan/TinyStories",
    dataset_split="train",
    buffer_size=2**20,
    buffer_device="cpu",
    offload_device="cpu",
    circular_buffer=True,
)
buffer = ActivationsBuffer(buffer_cfg)

encoder_cfg = AutoEncoderConfig(n_dim=14336, m_dim=14336 * 2)  # 14336*5 = 71680
encoder = AutoEncoder(encoder_cfg)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), foreach=False)

try:
    prev_time = time.time()
    for i in tqdm(range(num_activations // batch_size)):
        enc, l1, l2, loss = encoder.forward(buffer.next(batch=batch_size).to(encoder_cfg.device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % steps_per_report == 0 and i > 0:
            wandb.log({
                "l1_loss": l1.item(),
                "l2_loss": l2.item(),
                "total_loss": loss.item(),
                "ms_per_act": 1000 * (time.time() - prev_time) / (batch_size * steps_per_report)
            })

            if i % steps_per_save == 0:
                encoder.save(i // steps_per_save)

            torch.cuda.empty_cache()
            prev_time = time.time()
finally:
    # Save the model
    encoder.save("final")
