import math


class CosineScheduler():
    def __init__(self, max_lr, warmup_steps, min_lr, total_steps):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, steps):

        if steps < self.warmup_steps:
            return self.max_lr * (steps+1) / self.warmup_steps
        if steps > self.total_steps:
            return self.min_lr
        decay_ratio = (steps - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return (0.5 * (1.0 + math.cos(math.pi*decay_ratio))) * (self.max_lr - self.min_lr) + self.min_lr
