import math

import torch


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, peak_lr, decay_lr, warmup_steps, decay_steps, last_epoch=-1):
        self.peak_lr = peak_lr
        self.decay_lr = decay_lr
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self._warmup_steps:
            lr = self.peak_lr * step / max(1, self._warmup_steps)
        elif step < self._decay_steps:
            progress = (step - self._warmup_steps) / max(1, self._decay_steps - self._warmup_steps)
            lr = self.decay_lr + (self.peak_lr - self.decay_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.decay_lr
        return [lr for _ in self.base_lrs]
