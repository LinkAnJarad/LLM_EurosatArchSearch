from torch import nn


def num_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Total number of parameters in a model (trainable ones only by default)."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
