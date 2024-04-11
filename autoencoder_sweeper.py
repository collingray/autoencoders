from time import sleep
from typing import List, Optional

from autoencoder_trainer import *
from buffer import *
from tqdm.autonotebook import tqdm
from utils import *
import gc
from torch.multiprocessing import JoinableQueue as Queue, spawn
import sys


@dataclass
class AutoEncoderSweeperConfig:
    """
    The configuration for the `AutoEncoderSweeper` class

    n_dim: the dimension of the input
    m_dim: the dimension of the hidden layer
    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    lambda_reg: the regularization strength to use for the autoencoder
    warmup_percent: the percentage of steps to use for warmup
    dtype: the dtype to use for the model and autoencoder
    device: the device to use for the model and autoencoder
    layer: the layer of the model to train the autoencoder on (0-indexed)
    total_activations: the total number of activations to train each autoencoder on
    batch_size: the batch size to use for training each autoencoder
    parallelism: the number of autoencoders to train in parallel
    steps_per_report: the number of steps to take before reporting to wandb
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    """
    n_dim: int
    m_dim: int
    lr: List[float]
    beta1: List[float]
    beta2: List[float]
    lambda_reg: List[float]
    warmup_percent: List[float]
    wb_project: str
    wb_entity: str
    wb_group: Optional[str] = None
    wb_config: Optional[dict] = None
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    layer: int = 0
    total_activations: int = int(2e7)
    batch_size: int = 1024
    parallelism: int = 8
    steps_per_report: int = 100


def create_trainer_worker(pidx: int, offset: int, sweep_cfgs: list[dict], act_queues: list[Queue], cfg: AutoEncoderSweeperConfig):

    sweep_cfg = sweep_cfgs[pidx]
    act_queue = act_queues[pidx]

    encoder_cfg = AutoEncoderConfig(
        n_dim=cfg.n_dim,
        m_dim=cfg.m_dim,
        device=cfg.device,
        dtype=cfg.dtype,
        lambda_reg=sweep_cfg["lambda_reg"],
        record_data=True,
    )

    trainer_cfg = AutoEncoderTrainerConfig(
        lr=sweep_cfg["lr"],
        beta1=sweep_cfg["beta1"],
        beta2=sweep_cfg["beta2"],
        total_steps=cfg.total_activations // cfg.batch_size,
        warmup_percent=sweep_cfg["warmup_percent"],
        wb_project=cfg.wb_project,
        wb_entity=cfg.wb_entity,
        wb_name="{}: reg={:.1e}_lr={:.1e}_b1={:g}_b2={:g}_wu={:g}".format(
            offset+pidx,
            sweep_cfg["lambda_reg"],
            sweep_cfg["lr"],
            sweep_cfg["beta1"],
            sweep_cfg["beta2"],
            sweep_cfg["warmup_percent"],
        ),
        wb_group=cfg.wb_group,
        wb_config={
            **(cfg.wb_config or {}),
            **sweep_cfg,
        },
    )

    trainer = AutoEncoderTrainer(encoder_cfg, trainer_cfg)

    try:
        while True:
            acts = act_queue.get(block=True, timeout=None)

            if acts is None:
                break

            trainer.train_on(acts)
            del acts
            act_queue.task_done()
    finally:
        act_queue.task_done()
        trainer.finish()


class AutoEncoderSweeper:
    def __init__(self, cfg: AutoEncoderSweeperConfig, buffer: ActivationsBuffer):
        self.cfg = cfg
        self.buffer = buffer

        self.sweep_cfgs = [
            {
                "lr": lr,
                "beta1": beta1,
                "beta2": beta2,
                "lambda_reg": lambda_reg,
                "warmup_percent": warmup_percent,
            }
            for lr in cfg.lr
            for beta1 in cfg.beta1
            for beta2 in cfg.beta2
            for lambda_reg in cfg.lambda_reg
            for warmup_percent in cfg.warmup_percent
        ]

    def run(self):
        print(f"Running sweep with {len(self.sweep_cfgs)} configurations")

        # required for torch.multiprocessing to work with CUDA tensors
        torch.multiprocessing.set_start_method('spawn', force=True)

        for i in range(0, len(self.sweep_cfgs), self.cfg.parallelism):
            num_trainers = min(self.cfg.parallelism, len(self.sweep_cfgs) - i)

            queues = [Queue(maxsize=1) for _ in range(num_trainers)]

            print(f"Running configs {i+1} to {i+num_trainers} of {len(self.sweep_cfgs)}")

            # reset buffer
            self.buffer.reset_dataset()

            trainer_workers = spawn(
                create_trainer_worker,
                nprocs=num_trainers,
                args=(i+1, self.sweep_cfgs[i:i + self.cfg.parallelism], queues, self.cfg),
                join=False
            )

            for _ in tqdm(range(self.cfg.total_activations // self.cfg.batch_size)):
                acts = self.buffer.next(batch=self.cfg.batch_size)
                # [batch_size, layers, n_dim] -> [batch_size, n_dim]
                acts = acts[:, self.cfg.layer, :].to(self.cfg.device, dtype=self.cfg.dtype)

                # block until all previous activations have been processed
                # this comes before so that buffer.next() can be called in parallel, since it can be slow to refresh
                for q in queues:
                    q.join()

                for q in queues:
                    q.put(acts)

            for q in queues:
                q.put(None)

            trainer_workers.join()

            for q in queues:
                q.close()

            gc.collect()
            torch.cuda.empty_cache()
