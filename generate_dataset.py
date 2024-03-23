from buffer import *
from tqdm import tqdm
import torch

dataset_name = "roneneldan/TinyStories"
total_acts = 2e10
save_batch = 2**19
save_dir = "./activations"
save_name = "activations"

buffer_cfg = ActivationsBufferConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    layers=[0],
    dataset_name=dataset_name,
    dataset_split="train",
    buffer_size=save_batch,
    min_capacity=0,
    buffer_device="cpu",
    model_batch_size=16,
    samples_per_seq=64,
)
buffer = ActivationsBuffer(buffer_cfg)

for i in tqdm(range(int(total_acts) // save_batch)):
    acts = buffer.next(batch=save_batch)
    torch.save(acts, f"{save_dir}/{save_name}_{i}.pt")
