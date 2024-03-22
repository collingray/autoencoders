import json
import torch
import torch.nn as nn


class AutoEncoderConfig:
    def __init__(
            self,
            n_dim,
            m_dim,
            seed=None,
            device="cuda",
            lambda_reg=0.01,
            dtype=torch.bfloat16,
            name="autoencoder",
            save_dir="./weights",
    ):
        """
        :param n_dim: the dimension of the input
        :param m_dim: the dimension of the hidden layer
        :param seed: the seed to use for pytorch rng
        :param device: the device to use for the model
        :param lambda_reg: the regularization strength
        :param dtype: the dtype to use for the model
        :param name: the name to use when saving the model
        :param save_dir: the directory to save the model to
        """

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.seed = seed
        self.device = device
        self.lambda_reg = lambda_reg
        self.dtype = dtype
        self.name = name
        self.save_dir = save_dir


class AutoEncoderConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AutoEncoderConfig):
            o_dict = o.__dict__
            o_dict["dtype"] = o_dict["dtype"].__str__()[6:]
            return o_dict

        return json.JSONEncoder.default(self, o)


class AutoEncoderConfigDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.dict_to_object)

    @staticmethod
    def dict_to_object(d):
        if "dtype" in d:
            d["dtype"] = getattr(torch, d["dtype"])
        return AutoEncoderConfig(**d)


class AutoEncoder(nn.Module):
    """
    Autoencoder model with a single hidden layer
    """

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()

        self.cfg = cfg

        if cfg.seed:
            torch.manual_seed(cfg.seed)

        self.pre_encoder_bias = nn.Parameter(torch.zeros(cfg.n_dim, device=cfg.device, dtype=cfg.dtype))
        # encoder linear layer, goes from the models embedding space to the hidden layer
        self.encoder = nn.Linear(cfg.n_dim, cfg.m_dim, bias=False, device=cfg.device, dtype=cfg.dtype)
        self.pre_activation_bias = nn.Parameter(torch.zeros(cfg.m_dim, device=cfg.device, dtype=cfg.dtype))
        self.relu = nn.ReLU()
        # decoder linear layer, goes from the hidden layer back to models embeddings
        self.decoder = nn.Linear(cfg.m_dim, cfg.n_dim, bias=False, device=cfg.device, dtype=cfg.dtype)

    def forward(self, x):
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        loss = self.__loss(x, reconstructed, encoded, self.cfg.lambda_reg)
        return encoded, loss

    def encode(self, x):
        x = x - self.pre_encoder_bias
        return self.relu(self.encoder(x) + self.pre_activation_bias)

    def decode(self, x):
        return self.decoder(x) + self.pre_encoder_bias

    @staticmethod
    def __loss(x, x_out, latent, lambda_reg):
        l1_loss = lambda_reg * latent.abs().sum()  # L1 loss, promotes sparsity
        l2_loss = torch.mean((x_out - x) ** 2)  # L2 loss, reconstruction loss
        return l1_loss + l2_loss

    def save(self, checkpoint):
        # save the model
        filename = f"{self.cfg.name}_{checkpoint}"
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/{filename}.pt")
        with open(f"{self.cfg.save_dir}/{self.cfg.name}_cfg.json", "w") as f:
            json.dump(self.cfg, f, cls=AutoEncoderConfigEncoder)
        print(f"Saved model to {self.cfg.save_dir}/{filename}.pt")

    @classmethod
    def load(cls, name, checkpoint, save_dir="./weights"):
        filename = f"{save_dir}/{name}_{checkpoint}.pt"
        with open(f"{save_dir}/{name}_cfg.json", "r") as f:
            cfg = json.load(f, cls=AutoEncoderConfigDecoder)
        model = cls(cfg)
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
        return model
