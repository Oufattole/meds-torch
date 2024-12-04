import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithLinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, steps_per_epoch, min_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): The optimizer instance.
            warmup_iters (int): Number of iterations for the linear warmup phase.
            max_iters (int): Total number of iterations for the training.
            min_lr (float, optional): The minimum learning rate for the cosine annealing phase.
            last_epoch (int, optional): The index of the last epoch. Default is -1, which means it's the first epoch.
        """
        self.warmup_iters = warmup_iters
        self.max_iters = int(max_iters) * steps_per_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate based on warmup and cosine annealing schedule.
        """
        # Current iteration (epoch index)
        current_iter = self.last_epoch

        # If in warmup phase, apply linear warmup
        if current_iter < self.warmup_iters:
            lr_scale = float(current_iter) / float(max(1, self.warmup_iters))
        else:
            # After warmup, apply cosine annealing
            progress = (current_iter - self.warmup_iters) / float(max(1, self.max_iters - self.warmup_iters))
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))


        # Compute the learning rate for each parameter group
        lrs = [base_lr * lr_scale for base_lr in self.base_lrs]

        return lrs