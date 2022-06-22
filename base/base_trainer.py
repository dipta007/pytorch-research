import logging
from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm

from utils.utils import ensure_dir, save_model, to_device, wandb_log

log = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, model, metrics=[], metrics_name=[]):
        self.config = config
        self.model = to_device(model, self.config.device)
        self.optimizer = self.get_optimizer()

        self.metrics = metrics
        self.metrics_name = metrics_name
        if len(self.metrics_name) == 0:
            self.metric_names = [m.__name__ for m in metrics]

        assert len(self.metrics_name) == len(
            self.metrics
        ), "metrics_name and metrics must have the same length"

        self.grad_accum = max(
            1, self.config.train.desired_batch_size // self.config.data.batch_size
        )
        log.info(f"Gradient accumulation: {self.grad_accum}")

        # will update later
        self.epoch = 0
        self.total_iterations = 0
        self.wandb_dict = defaultdict(lambda: 0.0)

        self.es_patience_counter = 0
        self.es_best_metric = float("inf")
        if self.config.train.early_stopping.goal == "max":
            self.es_best_metric = float("-inf")

    def train(self):
        with tqdm(
            self.train_loader, unit="batch", total=len(self.train_loader)
        ) as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {self.epoch}")
                self.model.train()
                self.wandb_dict = defaultdict(lambda: 0.0)

                data = to_device(data, self.config.device)
                y_pred = self.get_prediction(data)
                loss = self.get_loss(y_pred, data, "train")

                loss.backward()
                self.total_iterations += 1

                # ? Gradient accumulation & Update
                if self.total_iterations % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # ? Logging to wandb
                self.wandb_dict[f"train/loss"] = loss.item()
                for metric, metric_name in zip(self.metrics, self.metrics_name):
                    mv = metric(y_pred, data)
                    self.wandb_dict[f"train/{metric_name}"] = mv
                self.log()

                # ? Validation after an interval of steps
                if self.total_iterations % self.config.train.validate_every == 0:
                    self.eval()

                tepoch.set_postfix(loss=loss.item())

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        self.wandb_dict = defaultdict(lambda: 0.0)
        with tqdm(
            enumerate(self.valid_loader), unit="batch", total=len(self.valid_loader)
        ) as tepoch:
            for _, data in tepoch:
                tepoch.set_description(f"Validation")

                data = to_device(data, self.config.device)
                y_pred = self.get_prediction(data)
                loss = self.get_loss(y_pred, data, "val")

                # ? Logging to wandb
                self.wandb_dict[f"val/loss"] += loss.item()
                for metric, metric_name in zip(self.metrics, self.metrics_name):
                    mv = metric(y_pred, data)
                    self.wandb_dict[f"val/{metric_name}"] = mv

                tepoch.set_postfix(loss=loss.item())

        for metric_name in self.metrics_name:
            self.wandb_dict[f"val/{metric_name}"] /= len(self.valid_loader)
        self.log()

        self.check_for_improvement()
        self.check_early_stopping()

    def fit(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        for epoch in range(self.config.train.epochs):
            self.epoch = epoch
            self.train()
            self.eval()

    def log(self):
        self.wandb_dict["iteration"] = self.total_iterations
        self.wandb_dict["epoch"] = self.epoch
        wandb_log(self.wandb_dict)

    def get_prediction(self, data):
        y_pred = self.model(data)
        return y_pred

    def check_for_improvement(self):
        goal = self.config.train.early_stopping.goal
        metric = self.config.train.early_stopping.metric
        if goal == "max":
            if self.wandb_dict[metric] > self.es_best_metric:
                self.es_best_metric = self.wandb_dict[metric]
                self.es_patience_counter = 0
            else:
                self.es_patience_counter += 1
        elif goal == "min":
            if self.wandb_dict[metric] < self.es_best_metric:
                self.es_best_metric = self.wandb_dict[metric]
                self.es_patience_counter = 0
            else:
                self.es_patience_counter += 1

        if self.es_patience_counter == 0:
            save_model(
                self.model, f"{self.config.train.model_dir}/{self.config.exp_name}.pt"
            )

    def check_early_stopping(self):
        metric = self.config.train.early_stopping.metric
        if self.config.train.early_stopping.enabled:
            if self.es_patience_counter >= self.config.train.early_stopping.patience:
                log.info(f"Early stopping at epoch {self.epoch}")
                msg = f"Early stopping at epoch {self.epoch} iteration {self.total_iterations}, Best {metric}: {self.es_best_metric}"
                raise RuntimeError(msg)

    def save_checkpoint(self):
        ensure_dir(self.config.train.checkpoint_dir)
        path = f"{self.config.train.checkpoint_dir}/checkpoint_epoch_{self.epoch}_iter_{self.total_iterations}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "total_iterations": self.total_iterations,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, epoch, total_iterations):
        path = f"{self.config.train.checkpoint_dir}/checkpoint_epoch_{epoch}_iter_{total_iterations}.pt"
        checkpoint = torch.load(path)
        self.epoch = checkpoint["epoch"]
        self.total_iterations = checkpoint["total_iterations"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model = to_device(self.model)

    def get_loss(self, y_pred, data, mode="train"):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError
