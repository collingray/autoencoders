import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sae_vis import SaeVisData, SaeVisConfig
from autoencoder import AutoEncoder

torch.set_grad_enabled(False)

# Load in the data
data = load_dataset("roneneldan/TinyStories", split="train").shuffle(42)["text"][:100000]

# Load in the model
model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", dtype="bfloat16")

# Load in the AutoEncoder
ae = AutoEncoder.load(name="autoencoder", checkpoint="final")

# Tokenize the data
tokens = model.to_tokens(data)[:, :128]

print("Tokens are of shape:", tokens.shape)

sae_vis_config = SaeVisConfig(
    hook_point="blocks.0.hook_mlp_out",
    batch_size=32,
    verbose=True,
)

# Generate the visualization data
sae_vis_data = SaeVisData.create(
    model=model,
    encoder=ae,
    tokens=tokens,
    cfg=sae_vis_config,
)

# Save the visualization
sae_vis_data.save_feature_centric_vis("feature_visualization.html")
