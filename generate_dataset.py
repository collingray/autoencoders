from buffer import *
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

dataset_name = "roneneldan/TinyStories"

buffer_cfg = ActivationsBufferConfig(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    layers=[0],
    dataset_name=dataset_name,
    dataset_split="train",
    buffer_size=2**19,
    buffer_device="cpu",
    model_batch_size=8,
    samples_per_seq=64,
)
buffer = ActivationsBuffer(buffer_cfg)

total_acts = 2e10
save_batch = 2**19

with ArrowWriter(path=f"{dataset_name}_acts.arrow") as writer:
    for i in tqdm(range(total_acts // save_batch)):
        acts = buffer.next(batch=save_batch)
        writer.write(acts.numpy())

writer.finalize()
