"""Example architecture spec for the agent to copy and modify.

Everything here is plain data: no generated Python is executed. The agent can
swap component types, add/remove stages, or use composite components such as
``residual``, ``parallel``, and ``repeat``.

Run ``python -m models.skeleton`` to build the example model and print its
parameter count.
"""

from __future__ import annotations

from typing import Any

from models.BaseModel import BaseModel

NUM_CLASSES = 10
IMAGE_SIZE = 64
IN_CHANNELS = 3

# Skeleton: small CNN stem -> patch embedding -> transformer -> classifier.
# Equivalent to putting all three sections into a single "layers" list.
SKELETON_SPEC: dict[str, Any] = {
    "name": "conv_transformer_skeleton",
    "in_channels": IN_CHANNELS,
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "stem": [
        {
            "type": "conv_block",
            "in_channels": IN_CHANNELS,
            "out_channels": 32,
            "kernel_size": 3,
            "padding": 1,
            "norm": "batch",
            "activation": "relu",
            "pool": "max",
        },
        {
            "type": "conv_block",
            "in_channels": 32,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "norm": "batch",
            "activation": "relu",
            "pool": "max",
        },
    ],
    "body": [
        {
            "type": "patch_embed",
            "in_channels": 64,
            "embed_dim": 128,
            "patch_size": 4,
            "norm": "layer",
            "dropout": 0.0,
        },
        {
            "type": "transformer_encoder",
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "activation": "gelu",
            "positional_encoding": "learnable",
            "max_tokens": 256,
        },
    ],
    "head": [
        # Accepts (B, N, C) tokens, averages them, then classifies.
        {
            "type": "classifier",
            "num_classes": NUM_CLASSES,
            "dropout": 0.0,
        },
    ],
}


def build_skeleton(spec: dict[str, Any] | None = None) -> BaseModel:
    """Build a model from ``SKELETON_SPEC`` or a supplied spec."""
    return BaseModel.from_spec(spec or SKELETON_SPEC)


if __name__ == "__main__":
    model = build_skeleton()
    total = sum(param.numel() for param in model.parameters())
    print(model)
    print(f"{model.name}: {total:,} parameters")
