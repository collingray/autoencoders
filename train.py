from typing import Optional

import torch
import wandb
from autoencoder import *
from buffer import *
import time
from tqdm import tqdm
from utils import *
import argparse
import gc

lr = 1e-4
num_activations = int(2e10)  # total number of tokens to train on, the dataset will wrap around as needed
batch_size = 128
beta1 = 0.9
beta2 = 0.99
steps_per_report = 100
steps_per_save = 10000

primary_device = "cuda:0"
offload_device = "cpu"

wandb_project = "autoencoder"
wandb_entity = "collingray"

argparser = argparse.ArgumentParser()
argparser.add_argument("--wb_name", type=Optional[str], default=None)
argparser.add_argument("--wb_notes", type=Optional[str], default=None)
args = argparser.parse_args()
wb_name = args.wb_name
wb_notes = args.wb_notes

wandb.init(project=wandb_project, entity=wandb_entity, name=wb_name, notes=wb_notes)

buffer_cfg = ActivationsBufferConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    layers=[0],
    dataset_name="roneneldan/TinyStories",
    dataset_split="train",
    buffer_size=2**19,
    device=primary_device,
    buffer_device=offload_device,
    offload_device=offload_device,
    model_batch_size=8,
    samples_per_seq=64,
)
buffer = ActivationsBuffer(buffer_cfg)

encoder_cfg = AutoEncoderConfig(
    n_dim=14336,
    m_dim=14336 * 4,
    device=primary_device,
)
encoder = AutoEncoder(encoder_cfg)

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), foreach=False)

try:
    prev_time = time.time()
    for i in tqdm(range(num_activations // batch_size)):
        # If the buffer needs to be refreshed, offload the encoder and its optimizer to the offload device to free up
        # memory on the primary device, which is needed by the buffer to load the next batch of activations.
        if buffer.will_refresh(batch=batch_size):
            encoder = encoder.to(offload_device)
            optimizer_to(optimizer, offload_device)
            torch.cuda.empty_cache()
            acts = buffer.next(batch=batch_size).to(encoder_cfg.device, non_blocking=True)
            encoder = encoder.to(primary_device)
            optimizer_to(optimizer, primary_device)
            gc.collect()
        else:
            acts = buffer.next(batch=batch_size).to(encoder_cfg.device, non_blocking=True)

        enc, l1, l2, loss = encoder(acts)
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
