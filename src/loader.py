from typing import Iterator

import torch
from torch.utils.data import DataLoader


class DeviceDataLoader:
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for x, y in self.dataloader:
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def __len__(self) -> int:
        return len(self.dataloader)
