from typing import Any, Optional

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

TEST = True

class Config:
    LR = 1e-4
    WD = 5e-5
    EPOCHS = 10
    BATCH_SIZE = 96
    OPTIMIZER = AdamW
    SCHEDULER = CosineAnnealingLR
    MAX_PARAMS = 20000
    MAX_COMPLETION_TOKENS = 32768


def build_optimizer(params, lr: float = Config.LR, weight_decay: float = Config.WD, **kwargs: Any) -> Optimizer:
    """Optimizer over a model's parameters: build_optimizer(model.parameters())."""
    return Config.OPTIMIZER(params, lr=lr, weight_decay=weight_decay, **kwargs)


def build_scheduler(optimizer: Optimizer, total_steps: Optional[int] = None, **kwargs: Any) -> LRScheduler:
    """LR schedule over an optimizer.

    CosineAnnealingLR's T_max is measured in scheduler steps, so pass
    total_steps = epochs * len(train_loader) when stepping once per batch,
    or leave it as None to anneal over Config.EPOCHS when stepping per epoch.
    """
    t_max = Config.EPOCHS if total_steps is None else total_steps

    # custom: not let lr go all the way to zero; only until middle
    # best for short epochs

    t_max *= 2
    return Config.SCHEDULER(optimizer, T_max=t_max, **kwargs)
