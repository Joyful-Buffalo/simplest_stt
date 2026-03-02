import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class NoamLR(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, int(warmup_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        s = max(1, self.last_epoch + 1)
        w = self.warmup_steps
        scale = (w ** 0.5) * min(s ** -0.5, s * (w ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]