import json
import torch
import torch.nn as nn
from utils import DTYPES


class AutoEncoder(nn.Module):
    """
    Autoencoder model with a single hidden layer
    m >= n (overcomplete)
    this is done to help disentangle features (i think)
    """

    def __init__(self, cfg):
        super().__init__()

        n = cfg["n_dim"]
        m = cfg["m_dim"]
        torch.manual_seed(cfg["seed"])
        device = cfg["device"]
        dtype = DTYPES[cfg["dtype"]]

        self.lambda_reg = cfg["lambda_reg"]
        self.name = cfg["name"]
        self.version = cfg["version"]
        self.save_dir = cfg["save_dir"]

        # todo: nn's impl. using he initialization here, might be worth looking into that/other initialization dists.
        # encoder linear layer, goes from the models embedding space to the hidden layer
        self.encoder = nn.Linear(n, m, bias=True, device=device, dtype=dtype)
        # decoder linear layer, goes from the hidden layer back to models embeddings
        self.decoder = nn.Linear(m, n)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_centered = x - self.decoder.bias
        f = self.relu(self.encoder(x_centered))
        x_reconstructed = self.decoder(f)
        loss = self.__loss(x, x_reconstructed, f, self.lambda_reg)
        return f, loss

    @staticmethod
    def __loss(x, x_out, f, lambda_reg):
        l1_loss = lambda_reg * f.abs().sum() # L1 loss, promotes sparsity
        l2_loss = torch.mean((x_out - x) ** 2) # L2 loss, reconstruction loss
        return l1_loss + l2_loss

    @staticmethod
    def filename(name, version, checkpoint):
        return f"{name}_v{version}_c{checkpoint}"

    def save(self, checkpoint):
        # save the model
        filename = self.filename(self.name, self.version, checkpoint)
        torch.save(self.state_dict(), f"{self.save_dir}/{filename}.pt")
        with open(f"{self.save_dir}/{filename}_cfg.json", "w") as f:
            json.dump(self.cfg, f)
        print(f"Saved model to {self.save_dir}/{filename}.pt")

    @classmethod
    def load(cls, name, version, checkpoint, save_dir):
        filename = cls.filename(name, version, checkpoint)
        with open(f"{save_dir}/{filename}_cfg.json", "r") as f:
            cfg = json.load(f)
        model = cls(cfg)
        model.load_state_dict(torch.load(f"{save_dir}/{filename}.pt"))
        print(f"Loaded model from {save_dir}/{filename}.pt")
        return model