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
num_activations = int(2e9)  # total number of tokens to train on, the dataset will wrap around as needed
batch_size = 8192
beta1 = 0.9
beta2 = 0.99
steps_per_report = 100
steps_per_save = 10000
expansion = 8
n_dim = 4096
m_dim = n_dim * expansion
base_frequency = 1 / m_dim
# To be used when gpu memory is tight, shuffles the encoder and buffer model back and forth from gpu to cpu to limit
# peak gpu memory usage
perform_offloading = False

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
    act_site="hook_mlp_out",
    dataset_name="roneneldan/TinyStories",
    dataset_split="train",
    buffer_size=2 ** 20,
    device=primary_device,
    buffer_device=offload_device,
    offload_device=offload_device if perform_offloading else None,
    shuffle_buffer=True,
    model_batch_size=8,
    samples_per_seq=None,
    max_seq_length=2048,
)
buffer = ActivationsBuffer(buffer_cfg)

encoder_cfg = AutoEncoderConfig(
    n_dim=n_dim,
    m_dim=m_dim,
    device=primary_device,
    lambda_reg=1e-4,
    tied=False,
    record_neuron_freqs=True,
)
encoder = AutoEncoder(encoder_cfg)

wandb.config.update({
    "max_lr": lr,
    "num_activations": num_activations,
    "batch_size": batch_size,
    "beta1": beta1,
    "beta2": beta2,
    "perform_offloading": perform_offloading,
    "encoder": encoder_cfg.__dict__,
    "buffer": buffer_cfg.__dict__,
})

optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2), foreach=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    total_steps=num_activations // batch_size,
    pct_start=0.1
)

try:
    prev_time = time.time()
    for i in tqdm(range(num_activations // batch_size)):
        # If offloading is enabled and the buffer needs to be refreshed, offload the encoder and its optimizer to the
        # offload device to free up memory on the primary device, which is needed by the buffer to load the next batch
        # of activations.
        if perform_offloading and buffer.will_refresh(batch=batch_size):
            encoder = encoder.to(offload_device)
            optimizer_to(optimizer, offload_device)
            torch.cuda.empty_cache()
            acts = buffer.next(batch=batch_size).to(encoder_cfg.device, non_blocking=True)
            encoder = encoder.to(primary_device)
            optimizer_to(optimizer, primary_device)
            gc.collect()
        else:
            acts = buffer.next(batch=batch_size).to(encoder_cfg.device, non_blocking=True)

        # 0 in the second dimension since we are only using one layer
        enc, loss, l1, mse = encoder(acts[:, 0, :])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if i % steps_per_report == 0 and i > 0:
            freqs, avg_fired = encoder.get_firing_data()

            wandb.log({
                "l1": l1.item(),
                "mse": mse.item(),
                "total_loss": loss.item(),
                "ms_per_act": 1000 * (time.time() - prev_time) / (batch_size * steps_per_report),
                "% <bf (10M rol. avg.)": (freqs < base_frequency).sum().item() / m_dim,
                "% <bf/10 (10M rol. avg.)": (freqs < (base_frequency / 10)).sum().item() / m_dim,
                "% <bf/100 (10M rol. avg.)": (freqs < (base_frequency / 100)).sum().item() / m_dim,
                "% <bf/1000 (10M rol. avg.)": (freqs < (base_frequency / 1000)).sum().item() / m_dim,
                "avg_neurons_fired": avg_fired,
                "lr": scheduler.get_last_lr()[0],
            })

            if i % steps_per_save == 0:
                encoder.save("chk")

            torch.cuda.empty_cache()
            prev_time = time.time()
finally:
    # Save the model
    encoder.save("final")
