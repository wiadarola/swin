import logging
from typing import Any, Literal

import hydra
import torch
import torchmetrics
import torchvision.transforms.v2 as T
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from src.loader import DeviceDataLoader
from src.model import SwinTransformer


def load_loaders(
    train: dict[str, Any], eval: dict[str, Any], device: torch.device
) -> tuple[DeviceDataLoader, DeviceDataLoader]:
    toTensor = T.Compose((T.ToImage(), T.ToDtype(torch.float, scale=True)))

    train_set = CIFAR10("data/", transform=toTensor, download=True)
    val_set = CIFAR10("data/", train=False, transform=toTensor, download=True)

    train_loader = DeviceDataLoader(DataLoader(train_set, **train), device)
    val_loader = DeviceDataLoader(DataLoader(val_set, **eval), device)

    return train_loader, val_loader


def reset_metrics(metrics: dict[str, torchmetrics.Metric]):
    """Reset metric state variables to their default values"""
    for metric in metrics.values():
        metric.reset()


def update_metrics(
    metrics: dict[str, torchmetrics.Metric], y: torch.Tensor, y_hat: torch.Tensor
):
    """Update the state variables of the metrics"""
    for metric in metrics.values():
        metric.update(y_hat, y)


def write_metrics(
    writer: SummaryWriter,
    metrics: dict[str, torchmetrics.Metric],
    epoch: int,
    stage: Literal["train", "val"],
):
    """Compute the final metric values and add to the writer summary"""
    for metric_name, metric in metrics.items():
        writer.add_scalar(f"{metric_name}/{stage}", metric.compute(), epoch)


def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DeviceDataLoader,
    avg_loss: torchmetrics.MeanMetric,
    metrics: dict[str, torchmetrics.Metric],
):
    avg_loss.reset()
    reset_metrics(metrics)

    model.train()
    for x, y in tqdm(dataloader, "Training", leave=False):
        y_hat = model(x)
        loss: torch.Tensor = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss.update(loss)
        update_metrics(metrics, y, y_hat)


def eval(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DeviceDataLoader,
    avg_loss: torchmetrics.MeanMetric,
    metrics: dict[str, torchmetrics.Metric],
):
    avg_loss.reset()
    reset_metrics(metrics)

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, "Validating", leave=False):
            y_hat = model(x)
            loss = criterion(y_hat, y)

            avg_loss.update(loss)
            update_metrics(metrics, y, y_hat)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log_dir = HydraConfig.get().run.dir
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    best_val_loss = torch.inf
    avg_loss = torchmetrics.MeanMetric().to(device)
    metrics = {
        "F1 Score": torchmetrics.F1Score(task="multiclass", num_classes=10).to(device),
        "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device),
    }
    logging.info(f"Computing metrics: {", ".join(metrics.keys())}")

    model = SwinTransformer(**cfg.model, n_classes=10).to(device)
    logging.info(f"Using model: {HydraConfig.get().runtime.choices.get("model")}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)

    num_epochs = cfg.trainer.num_epochs
    warmup_steps = cfg.trainer.warmup_steps
    lr_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs - warmup_steps
    )

    train_loader, val_loader = load_loaders(**cfg.data, device=device)

    logging.info(f"Training begun. Running for {num_epochs} epochs.")
    for epoch in tqdm(range(num_epochs), "Epoch"):
        lr_warmup.step()
        train(model, criterion, optimizer, train_loader, avg_loss, metrics)
        writer.add_scalar(f"loss/train", avg_loss.compute(), epoch)
        write_metrics(writer, metrics, epoch, stage="train")
        if epoch > warmup_steps:
            scheduler.step()

        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        eval(model, criterion, val_loader, avg_loss, metrics)
        val_loss = avg_loss.compute()
        writer.add_scalar("loss/val", val_loss, epoch)
        write_metrics(writer, metrics, epoch, stage="val")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/model_state_best_val.pt")

    torch.save(model.state_dict(), f"{log_dir}/model_state_last.pt")
    logging.info(f"Training ended. Best validation loss: {best_val_loss}")
    writer.close()


if __name__ == "__main__":
    main()
