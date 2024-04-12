from typing import List, Literal, Optional

import torch
from overrides import overrides

from autoencoder import *


class AutoEncoderMultiLayerConfig(AutoEncoderConfig):
    def __init__(
            self,
            n_dim,
            m_dim,
            act_norms: List[float],
            act_renorm_type: Literal["linear", "sqrt", "log", "none"],
            act_renorm_scale: float,
            tied=False,
            seed=None,
            device="cuda",
            dtype=torch.bfloat16,
            lambda_reg=0.001,
            record_data=False,
            num_firing_buckets=10,
            firing_bucket_size=1000000,
            name="multilayer_autoencoder",
            save_dir="./weights",
            **kwargs
    ):
        """
        :param n_dim: the dimension of the input
        :param m_dim: the dimension of the hidden layer
        :param act_norms: the norms to use for layer activation renormalization
        :param act_renorm_type: the type of renormalization to use for layer activations, one of "linear", "sqrt",
            "log", "none". Activations are scaled by act_renorm_scale*(avg(norms)/norms[layer]), where norms are the
            result of the act_renorm_type applied to act_norms
        :param act_renorm_scale: a global scale to apply to all activations after renormalization
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

        super().__init__(n_dim, m_dim, tied, seed, device, dtype, lambda_reg, record_data, num_firing_buckets,
                         firing_bucket_size, name, save_dir, **kwargs)

        self.act_norms = act_norms
        self.act_renorm_type = act_renorm_type
        self.act_renorm_scale = act_renorm_scale


# Custom JSON encoder and decoder for AutoEncoderMultiLayerConfig, as torch.dtype is not serializable by default
class AutoEncoderMultiLayerConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, AutoEncoderMultiLayerConfig):
            o_dict = o.__dict__.copy()
            o_dict["dtype"] = o_dict["dtype"].__str__()[6:]
            return o_dict

        return json.JSONEncoder.default(self, o)


class AutoEncoderMultiLayerConfigDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.dict_to_object)

    @staticmethod
    def dict_to_object(d):
        if "dtype" in d:
            d["dtype"] = getattr(torch, d["dtype"])
        return AutoEncoderMultiLayerConfig(**d)


class AutoEncoderMultiLayer(AutoEncoder):
    """
    Autoencoder model with a single hidden layer that can be trained on activations from multiple layers of a model
    """

    def __init__(self, cfg: AutoEncoderMultiLayerConfig):
        super().__init__(cfg)

        if cfg.act_renorm_type == "none":
            # no renormalization
            norms = torch.ones(len(cfg.act_norms))
        elif cfg.act_renorm_type == "linear":
            norms = torch.Tensor(cfg.act_norms)
        elif cfg.act_renorm_type == "sqrt":
            norms = torch.Tensor(cfg.act_norms).sqrt()
        elif cfg.act_renorm_type == "log":
            norms = torch.Tensor(cfg.act_norms).log()
        else:
            raise ValueError(f"Invalid act_renorm_type {cfg.act_renorm_type}")

        self.register_buffer("act_scales", cfg.act_renorm_scale * norms.mean() / norms)

    @overrides
    def encode(self, x, layer: Optional[int] = None):
        # A layer should only be specified if x is a single layer
        if layer is not None:  # x: [batch_size, n_dim]
            x = x * self.act_scales[layer]
        else:  # x: [batch_size, num_layers, n_dim]
            x = torch.einsum("bln,l->bln", x, self.act_scales)

        return super().encode(x)

    @overrides
    def decode(self, x, layer: Optional[int] = None):
        # A layer should only be specified if x is a single layer
        if layer is not None:  # x: [batch_size, n_dim]
            x = x / self.act_scales[layer]
        else:  # x: [batch_size, num_layers, n_dim]
            x = torch.einsum("bln,l->bln", x, 1/self.act_scales)

        return super().decode(x)

    def save(self, checkpoint):
        filename = f"{self.cfg.name}_{checkpoint}"
        torch.save(self.state_dict(), f"{self.cfg.save_dir}/{filename}.pt")
        with open(f"{self.cfg.save_dir}/{self.cfg.name}_cfg.json", "w") as f:
            json.dump(self.cfg, f, cls=AutoEncoderMultiLayerConfigEncoder)

    @classmethod
    def load(cls, name, checkpoint, save_dir="./weights"):
        filename = f"{save_dir}/{name}_{checkpoint}.pt"
        with open(f"{save_dir}/{name}_cfg.json", "r") as f:
            cfg = json.load(f, cls=AutoEncoderMultiLayerConfigDecoder)
        model = cls(cfg)
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
        return model

