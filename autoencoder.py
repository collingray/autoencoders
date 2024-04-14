import json
import torch
import torch.nn as nn
from utils import TiedLinear


class AutoEncoderConfig:
    def __init__(
            self,
            n_dim,
            m_dim,
            tied=False,
            seed=None,
            device="cuda",
            dtype=torch.bfloat16,
            lambda_reg=0.001,
            record_data=False,
            num_firing_buckets=10,
            firing_bucket_size=1000000,
            name="autoencoder",
            save_dir="./weights",
            **kwargs
    ):
        """
        :param n_dim: the dimension of the input
        :param m_dim: the dimension of the hidden layer
        :param tied: if True, the decoder weights are tied to the encoder weights
        :param seed: the seed to use for pytorch rng
        :param device: the device to use for the model
        :param dtype: the dtype to use for the model
        :param lambda_reg: the regularization strength
        :param record_data: if True, a variety of data will be recorded during on forward passes, including neuron
        firing frequencies and the average FVU
        :param num_firing_buckets: the number of buckets to use for recording neuron firing
        :param firing_bucket_size: the size of each bucket for recording neuron firing
        :param name: the name to use when saving the model
        :param save_dir: the directory to save the model to
        """

        self.n_dim = n_dim
        self.m_dim = m_dim
        self.tied = tied
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.lambda_reg = lambda_reg
        self.record_data = record_data
        self.num_firing_buckets = num_firing_buckets
        self.firing_bucket_size = firing_bucket_size
        self.name = name
        self.save_dir = save_dir

        # mapping for using sae-vis
        self.d_in = n_dim
        self.d_hidden = m_dim


# Custom JSON encoder and decoder for AutoEncoderConfig, as torch.dtype is not serializable by default
class AutoEncoderConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AutoEncoderConfig):
            o_dict = o.__dict__.copy()
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
        if cfg.tied:
            self.decoder = TiedLinear(self.encoder)  # tied weights, uses same dtype/device as encoder
        else:
            self.decoder = nn.Linear(cfg.m_dim, cfg.n_dim, bias=False, device=cfg.device, dtype=cfg.dtype)

        if cfg.record_data:
            self.register_data_buffers(cfg)

    def forward(self, x):
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        loss, l1, mse = self.loss(x, reconstructed, encoded, self.cfg.lambda_reg)

        if self.cfg.record_data:
            self.record_fvu_data(x, mse)

        return encoded, loss, l1, mse

    def encode(self, x, record=True):
        x = x - self.pre_encoder_bias
        x = self.relu(self.encoder(x) + self.pre_activation_bias)

        if self.cfg.record_data and record:
            self.record_firing_data(x)

        return x

    def decode(self, x):
        return self.decoder(x) + self.pre_encoder_bias

    def loss(self, x, x_out, latent, lambda_reg):
        l1 = self.normalized_l1(x, latent)
        mse = self.normalized_reconstruction_mse(x, x_out)
        total = (lambda_reg * l1) + mse

        return total, l1, mse

    # Mappings for using SAE-Vis
    @property
    def W_enc(self):
        return self.encoder.weight.T

    @property
    def W_dec(self):
        return self.decoder.weight.T

    @property
    def b_enc(self):
        return self.pre_activation_bias

    @property
    def b_dec(self):
        return self.pre_encoder_bias

    @staticmethod
    def normalized_reconstruction_mse(x, recons):
        """
        The MSE between the input and its reconstruction, normalized by the mean square of the input, averaged over the
        batch.

        [n_dim]*2 -> []
        Or
        [batch, n_dim]*2 -> []
        Or
        [batch, layer, n_dim]*2 -> [layer]
        """
        return (((x - recons) ** 2).mean(dim=-1) / (x ** 2).mean(dim=-1)).mean(dim=0)

    @staticmethod
    def normalized_l1(x, latent):
        """
        The L1 norm of the latent representation, normalized by the L2 norm of the input, averaged over the batch.

        [n_dim], [m_dim] -> []
        Or
        [batch, n_dim], [batch, m_dim] -> []
        Or
        [batch, layer, n_dim], [batch, layer, m_dim] -> [layer]
        """
        return (latent.norm(dim=-1, p=1) / x.norm(dim=-1, p=2)).mean(dim=0)

    def register_data_buffers(self, cfg):
        # Bucketed rolling avg. for memory efficiency
        self.register_buffer("num_encodes", torch.zeros(cfg.num_firing_buckets, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("neuron_firings",
                             torch.zeros(cfg.num_firing_buckets, cfg.m_dim, device=cfg.device, dtype=torch.int32),
                             persistent=False)

        self.register_buffer("num_forward_passes", torch.tensor(0, device=cfg.device, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("mse_ema", torch.tensor(0.0, device=cfg.device, dtype=cfg.dtype), persistent=False)
        self.register_buffer("input_avg", torch.zeros(cfg.n_dim, device=cfg.device, dtype=cfg.dtype), persistent=False)
        self.register_buffer("input_var", torch.zeros(cfg.n_dim, device=cfg.device, dtype=cfg.dtype), persistent=False)

    @torch.no_grad()
    def record_firing_data(self, x: torch.Tensor):
        if self.num_encodes[0] >= self.cfg.firing_bucket_size:
            # If we've exceeded the bucket size, roll the data and reset the first bucket
            self.num_encodes = torch.roll(self.num_encodes, 1, 0)
            self.neuron_firings = torch.roll(self.neuron_firings, 1, 0)
            self.num_encodes[0] = 0
            self.neuron_firings[0] = 0

        self.num_encodes[0] += x.shape[0]
        self.neuron_firings[0] += (x > 0).sum(dim=0)

    @torch.no_grad()
    def record_fvu_data(self, x: torch.Tensor, mse: torch.Tensor):
        batch_size = x.shape[0]

        n = self.num_forward_passes

        # Exponential moving average of the MSE, effectively updated `batch_size` times
        alpha = 0.05
        mse = mse.sum().item()
        self.mse_ema = ((1 - alpha) ** batch_size) * (self.mse_ema - mse) + mse

        # Update variance to include new inputs
        self.input_avg = ((n * self.input_avg) + x.sum(dim=0)) / (n + batch_size)
        self.input_var = ((n * self.input_var) + ((x - self.input_avg) ** 2).sum(dim=0)) / (n + batch_size)

        self.num_forward_passes += batch_size

    def get_data(self):
        """
        Get data on the firing of different neurons in the hidden layer
        :return: A tuple containing a tensor of the frequency with which each neuron fires and the average number of
        neurons that fired per pass
        """

        firings = self.neuron_firings.sum(dim=0).float()
        passes = self.num_encodes.sum().item()

        freqs = firings / passes
        avg_fired = firings.sum().item() / passes

        # Calculate the average FVU
        avg_fvu = self.mse_ema / self.input_var.mean()

        return freqs, avg_fired, avg_fvu

    def save(self, checkpoint):
        filename = f"{self.cfg.name}_{checkpoint}"
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/{filename}.pt")
        with open(f"{self.cfg.save_dir}/{self.cfg.name}_cfg.json", "w") as f:
            json.dump(self.cfg, f, cls=AutoEncoderConfigEncoder)

    @classmethod
    def load(cls, name, checkpoint, save_dir="./weights"):
        filename = f"{save_dir}/{name}_{checkpoint}.pt"
        with open(f"{save_dir}/{name}_cfg.json", "r") as f:
            cfg = json.load(f, cls=AutoEncoderConfigDecoder)
        model = cls(cfg)
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
        return model
