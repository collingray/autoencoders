import torch
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
from sae_vis import SaeVisData, SaeVisConfig
from autoencoder import AutoEncoder

torch.set_grad_enabled(False)

# Load in the data
data = load_dataset("roneneldan/TinyStories", split="train")

# Load in the model
model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", hook_layers=[0])

# Load in the AutoEncoder
ae = AutoEncoder.load(name="autoencoder", checkpoint="final")

# Tokenize the data
tokens = model.to_tokens(data["text"])

print("Tokens are of shape:", tokens.shape)

sae_vis_config = SaeVisConfig(
    hook_point="blocks.0.hook_mlp_out",
    features=range(64),
    batch_size=2048,
    verbose=True,
)

# Generate the visualization data
sae_vis_data = SaeVisData(
    model=model,
    encoder=ae,
    tokens=tokens,
    cfg=sae_vis_config,
)

# Save the visualization
sae_vis_data.save_feature_centric_vis("feature_visualization.html")