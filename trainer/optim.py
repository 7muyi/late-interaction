import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from common.registry import registry


@registry.register_lr_scheduler("constant_with_warmup")
def constant_with_warmup(optimizer: torch.optim.Optimizer,
                         num_warmup_steps: int,
                         **kwargs) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


@registry.register_lr_scheduler("linear_with_warmup")
def linear_with_warmup(optimizer: torch.optim.Optimizer,
                       num_warmup_steps: int,
                       num_training_steps: int,
                       min_lr_ratio: float = 0.0,
                       **kwargs) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

    return LambdaLR(optimizer, lr_lambda)


@registry.register_lr_scheduler("cosine_with_warmup")
def cosine_with_warmup(optimizer: torch.optim.Optimizer,
                       num_warmup_steps: int,
                       num_training_steps: int,
                       min_lr_ratio: float = 0.0,
                       **kwargs) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
