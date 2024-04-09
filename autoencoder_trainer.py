from dataclasses import dataclass
from typing import Optional, List

import torch
import wandb
from autoencoder import *
from buffer import *
import time
from tqdm import tqdm
from utils import *
import argparse
import gc


@dataclass
class AutoEncoderTrainerConfig:
    """
    The configuration for the `AutoEncoderTrainer` class

    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    total_steps: the total number of steps that will be taken by the lr scheduler, should be equal to the number of times train_on is called
    warmup_percent: the percentage of steps to use for warmup
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    """
    lr: float
    beta1: float
    beta2: float
    total_steps: int
    warmup_percent: float
    wb_project: str
    wb_entity: str
    wb_group: Optional[str] = None
    wb_config: Optional[dict] = None


class AutoEncoderTrainer:
    """
    The class for training an `AutoEncoder` model, which contains the model and optimizer, and trains on
    activations passed to it through the `train_on` method, but does not contain the data or training loop
    """

    def __init__(self, encoder_cfg: AutoEncoderConfig, trainer_cfg: AutoEncoderTrainerConfig):
        self.encoder = AutoEncoder(encoder_cfg)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=trainer_cfg.lr,
            betas=(trainer_cfg.beta1, trainer_cfg.beta2),
            foreach=False
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=trainer_cfg.lr,
            total_steps=trainer_cfg.total_steps,
            pct_start=trainer_cfg.warmup_percent
        )

        wandb.init(
            project=trainer_cfg.wb_project,
            entity=trainer_cfg.wb_entity,
            group=trainer_cfg.wb_group,
            config=trainer_cfg.wb_config
        )

    def train_on(self, acts, report=False):
        enc, loss, l1, mse = self.encoder(acts)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        if report:
            if self.encoder.cfg.record_neuron_freqs:
                freqs, avg_fired = self.encoder.get_firing_data()
                freq_data = {
                    "avg_neurons_fired": avg_fired,
                    "feature_density": wandb.Histogram(freqs.log10().nan_to_num(neginf=-10).cpu())
                }
            else:
                freq_data = {}

            wandb.log({
                "l1": l1.item(),
                "mse": mse.item(),
                "total_loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0],
                **freq_data,
            })
