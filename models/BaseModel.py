"""Composable base model built from declarative, JSON-friendly component specs.

The agent should only submit data like::

    {
        "type": "conv_block",
        "in_channels": 3,
        "out_channels": 32,
        "kernel_size": 3,
        "padding": 1,
        "pool": "max",
    }

instead of arbitrary Python code. ``BaseModel`` turns those specs into real
``nn.Module`` objects using a fixed registry of known component types.

Unconventional layouts are supported through composite components such as
``sequential``, ``residual``, ``parallel`` and ``repeat``. You can add new
components with ``@register_component("name")``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

ComponentSpec = Mapping[str, Any]

_COMPONENTS: dict[str, Callable[[dict[str, Any]], nn.Module]] = {}


def register_component(name: str) -> Callable[[Callable[[dict[str, Any]], nn.Module]], Callable[[dict[str, Any]], nn.Module]]:
    """Register a component factory under a case-insensitive name."""

    def decorator(factory: Callable[[dict[str, Any]], nn.Module]) -> Callable[[dict[str, Any]], nn.Module]:
        key = name.lower()
        if key in _COMPONENTS:
            raise ValueError(f"Component {name!r} is already registered")
        _COMPONENTS[key] = factory
        return factory

    return decorator


def available_components() -> tuple[str, ...]:
    """Names accepted in a component spec's ``type`` field."""
    return tuple(sorted(_COMPONENTS))


def _reject_extra(kind: str, params: dict[str, Any]) -> None:
    if params:
        keys = ", ".join(sorted(params))
        raise TypeError(f"Unsupported option(s) for component {kind!r}: {keys}")


def _build_component(spec: ComponentSpec | nn.Module) -> nn.Module:
    if isinstance(spec, nn.Module):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError(f"Expected a component spec or nn.Module, got {type(spec).__name__}")

    params = dict(spec)
    kind = params.pop("type", None)
    if kind is None:
        raise ValueError("Each component spec needs a 'type' field")

    key = str(kind).lower()
    factory = _COMPONENTS.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown component {kind!r}. Available components: {', '.join(available_components())}"
        )

    module = factory(params)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Component factory {key!r} did not return an nn.Module")
    return module


def _build_layers(value: Any) -> list[nn.Module]:
    """Build one layer, a list of layers, or an empty list."""
    if value is None:
        return []
    if isinstance(value, (nn.Module, Mapping)):
        return [_build_component(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_build_component(layer) for layer in value]
    raise TypeError("layers must be a component spec or a sequence of component specs")


def _concat_layer_specs(*values: Any) -> list[Any]:
    layers: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (nn.Module, Mapping)):
            layers.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            layers.extend(value)
        else:
            raise TypeError("Layer sections must be a component spec or a sequence of them")
    return layers


def _is_no_norm(spec: Any) -> bool:
    if spec is None:
        return True
    if isinstance(spec, str):
        return spec.lower() in {"none", "identity"}
    return False


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "mish": nn.Mish,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
    "none": nn.Identity,
}


def _activation_module(spec: Any) -> nn.Module:
    if spec is None:
        return nn.Identity()
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        cls = _ACTIVATIONS.get(key)
        if cls is None:
            raise ValueError(
                f"Unknown activation {spec!r}. Available: {', '.join(sorted(_ACTIVATIONS))}"
            )
        return cls()
    if isinstance(spec, Mapping):
        params = dict(spec)
        kind = params.pop("type", params.pop("name", None))
        if kind is None:
            raise ValueError("Activation spec needs a 'type' or 'name'")
        key = str(kind).lower()
        cls = _ACTIVATIONS.get(key)
        if cls is None:
            raise ValueError(f"Unknown activation {kind!r}")
        return cls(**params)
    raise TypeError("activation must be a string, mapping, nn.Module, or None")


def _norm_module(kind: Any, num_features: int, *, spatial: bool, **kwargs: Any) -> nn.Module:
    if kind is None or (isinstance(kind, str) and kind.lower() in {"none", "identity"}):
        return nn.Identity()
    if isinstance(kind, nn.Module):
        return kind
    if isinstance(kind, Mapping):
        params = dict(kind)
        kind = params.pop("type", params.pop("name", None))
        kwargs = {**params, **kwargs}
    if kind is None:
        return nn.Identity()

    key = str(kind).lower().replace("-", "_")
    if key in {"batch", "batchnorm", "batch_norm", "batch2d"}:
        return nn.BatchNorm2d(num_features, **kwargs) if spatial else nn.BatchNorm1d(num_features, **kwargs)
    if key == "batch1d":
        return nn.BatchNorm1d(num_features, **kwargs)
    if key in {"layer", "layernorm", "layer_norm"}:
        return nn.LayerNorm(num_features, **kwargs)
    if key in {"group", "groupnorm", "group_norm"}:
        num_groups = int(kwargs.pop("num_groups", 8))
        return nn.GroupNorm(num_groups, num_features, **kwargs)
    if key in {"instance", "instancenorm", "instance_norm"}:
        return nn.InstanceNorm2d(num_features, **kwargs) if spatial else nn.InstanceNorm1d(num_features, **kwargs)
    raise ValueError(f"Unknown normalization {kind!r}")


def _linear_module(in_features: Any, out_features: int, **kwargs: Any) -> nn.Module:
    if in_features in (None, "auto", "lazy"):
        return nn.LazyLinear(int(out_features), **kwargs)
    return nn.Linear(int(in_features), int(out_features), **kwargs)


def _conv_module(in_channels: Any, out_channels: int, **kwargs: Any) -> nn.Module:
    if in_channels in (None, "auto", "lazy"):
        return nn.LazyConv2d(int(out_channels), **kwargs)
    return nn.Conv2d(int(in_channels), int(out_channels), **kwargs)


def _upsample_module(spec: Any) -> nn.Module:
    if spec is None:
        return nn.Identity()
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, str):
        key = spec.lower()
        if key in {"none", "identity"}:
            return nn.Identity()
        if key in {"max", "maxpool", "max_pool"}:
            return nn.MaxPool2d(kernel_size=2)
        if key in {"avg", "avgpool", "avg_pool"}:
            return nn.AvgPool2d(kernel_size=2)
        if key in {"adaptive_avg", "global_avg", "gap"}:
            return nn.AdaptiveAvgPool2d(1)
        if key in {"adaptive_max", "global_max"}:
            return nn.AdaptiveMaxPool2d(1)
        raise ValueError(f"Unknown pooling {spec!r}")
    if isinstance(spec, Mapping):
        params = dict(spec)
        kind = params.pop("type", params.pop("name", None))
        if kind is None:
            raise ValueError("Pooling spec needs a 'type' or 'name'")
        key = str(kind).lower()
        if key in {"max", "maxpool", "max_pool"}:
            return nn.MaxPool2d(**params)
        if key in {"avg", "avgpool", "avg_pool"}:
            return nn.AvgPool2d(**params)
        if key in {"adaptive_avg", "global_avg", "gap"}:
            return nn.AdaptiveAvgPool2d(**params)
        if key in {"adaptive_max", "global_max"}:
            return nn.AdaptiveMaxPool2d(**params)
        raise ValueError(f"Unknown pooling {kind!r}")
    raise TypeError("pool must be a string, mapping, nn.Module, or None")


# ---------------------------------------------------------------------------
# Primitive and block components
# ---------------------------------------------------------------------------


@register_component("identity")
def _identity(params: dict[str, Any]) -> nn.Module:
    _reject_extra("identity", params)
    return nn.Identity()


@register_component("sequential")
def _sequential(params: dict[str, Any]) -> nn.Module:
    layers = params.pop("layers", None)
    _reject_extra("sequential", params)
    return nn.Sequential(*_build_layers(layers))


@register_component("conv2d")
def _conv2d(params: dict[str, Any]) -> nn.Module:
    in_channels = params.pop("in_channels", None)
    out_channels = params.pop("out_channels", None)
    if out_channels is None:
        raise ValueError("conv2d requires 'out_channels'")
    return _conv_module(in_channels, int(out_channels), **params)


@register_component("conv_block")
def _conv_block(params: dict[str, Any]) -> nn.Module:
    in_channels = params.pop("in_channels", None)
    out_channels = params.pop("out_channels", None)
    if out_channels is None:
        raise ValueError("conv_block requires 'out_channels'")

    norm_spec = params.pop("norm", "batch")
    activation_spec = params.pop("activation", "relu")
    pool_spec = params.pop("pool", None)
    dropout = float(params.pop("dropout", 0.0))
    bias = params.pop("bias", None)
    if bias is None:
        bias = _is_no_norm(norm_spec)

    layers: list[nn.Module] = [
        _conv_module(in_channels, int(out_channels), bias=bias, **params),
        _norm_module(norm_spec, int(out_channels), spatial=True),
        _activation_module(activation_spec),
        _upsample_module(pool_spec),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    # Drop identities so the module list stays readable.
    return nn.Sequential(*(layer for layer in layers if not isinstance(layer, nn.Identity)))


@register_component("linear")
def _linear(params: dict[str, Any]) -> nn.Module:
    in_features = params.pop("in_features", None)
    out_features = params.pop("out_features", None)
    if out_features is None:
        raise ValueError("linear requires 'out_features'")
    return _linear_module(in_features, int(out_features), **params)


@register_component("linear_block")
def _linear_block(params: dict[str, Any]) -> nn.Module:
    in_features = params.pop("in_features", None)
    out_features = params.pop("out_features", None)
    if out_features is None:
        raise ValueError("linear_block requires 'out_features'")

    norm_spec = params.pop("norm", None)
    activation_spec = params.pop("activation", "relu")
    dropout = float(params.pop("dropout", 0.0))
    bias = params.pop("bias", None)
    if bias is None:
        bias = _is_no_norm(norm_spec)

    layers: list[nn.Module] = [
        _linear_module(in_features, int(out_features), bias=bias, **params),
        _norm_module(norm_spec, int(out_features), spatial=False),
        _activation_module(activation_spec),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*(layer for layer in layers if not isinstance(layer, nn.Identity)))


@register_component("mlp")
def _mlp(params: dict[str, Any]) -> nn.Module:
    dims = [int(dim) for dim in params.pop("dims", [])]
    if not dims:
        raise ValueError("mlp requires a non-empty 'dims' list")

    in_features = params.pop("in_features", None)
    norm_spec = params.pop("norm", None)
    activation_spec = params.pop("activation", "relu")
    dropout = float(params.pop("dropout", 0.0))
    final_activation = bool(params.pop("final_activation", False))
    bias = params.pop("bias", None)
    _reject_extra("mlp", params)

    if bias is None:
        bias = _is_no_norm(norm_spec)

    layers: list[nn.Module] = []
    current_in = in_features
    for index, out_features in enumerate(dims):
        is_last = index == len(dims) - 1
        layers.append(_linear_module(current_in, out_features, bias=bias))
        if not is_last or final_activation:
            layers.append(_norm_module(norm_spec, out_features, spatial=False))
            layers.append(_activation_module(activation_spec))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        current_in = out_features

    return nn.Sequential(*(layer for layer in layers if not isinstance(layer, nn.Identity)))


@register_component("patch_embed")
def _patch_embed(params: dict[str, Any]) -> nn.Module:
    in_channels = params.pop("in_channels", None)
    embed_dim = params.pop("embed_dim", params.pop("d_model", None))
    if embed_dim is None:
        raise ValueError("patch_embed requires 'embed_dim'")
    patch_size = int(params.pop("patch_size", 4))
    if "kernel_size" in params or "stride" in params:
        raise ValueError("patch_embed uses 'patch_size'; do not pass 'kernel_size' or 'stride'")
    norm_spec = params.pop("norm", None)
    dropout = float(params.pop("dropout", 0.0))
    bias = params.pop("bias", None)
    if bias is None:
        bias = _is_no_norm(norm_spec)

    projection = _conv_module(
        in_channels,
        int(embed_dim),
        kernel_size=patch_size,
        stride=patch_size,
        bias=bias,
        **params,
    )
    return _PatchEmbed(
        projection,
        _norm_module(norm_spec, int(embed_dim), spatial=False),
        dropout,
    )


@register_component("transformer_encoder")
def _transformer_encoder(params: dict[str, Any]) -> nn.Module:
    d_model = params.pop("d_model", params.pop("embed_dim", None))
    if d_model is None:
        raise ValueError("transformer_encoder requires 'd_model'")
    d_model = int(d_model)

    nhead = int(params.pop("nhead", 8))
    num_layers = int(params.pop("num_layers", 1))
    dim_feedforward = int(params.pop("dim_feedforward", d_model * 4))
    dropout = float(params.pop("dropout", 0.0))
    activation = params.pop("activation", "gelu")
    norm_first = bool(params.pop("norm_first", False))
    layer_norm_eps = float(params.pop("layer_norm_eps", 1e-5))
    positional_encoding = str(params.pop("positional_encoding", "none")).lower()
    max_tokens = int(params.pop("max_tokens", 1024))
    output = str(params.pop("output", "tokens")).lower()
    final_norm = params.pop("final_norm", None)
    _reject_extra("transformer_encoder", params)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        batch_first=True,
    )
    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers,
        norm=_norm_module(final_norm, d_model, spatial=False) if final_norm is not None else None,
    )
    return _TransformerEncoderBlock(
        encoder,
        d_model=d_model,
        positional_encoding=positional_encoding,
        max_tokens=max_tokens,
        output=output,
    )


@register_component("residual")
def _residual(params: dict[str, Any]) -> nn.Module:
    layers = params.pop("layers", None)
    scale = float(params.pop("scale", 1.0))
    shortcut = params.pop("shortcut", None)
    _reject_extra("residual", params)
    return _Residual(
        nn.Sequential(*_build_layers(layers)),
        nn.Sequential(*_build_layers(shortcut)) if shortcut is not None else None,
        scale=scale,
    )


@register_component("parallel")
def _parallel(params: dict[str, Any]) -> nn.Module:
    branches = params.pop("branches", None)
    if not branches:
        raise ValueError("parallel requires a non-empty 'branches' list")
    merge = str(params.pop("merge", "sum")).lower()
    concat_dim = int(params.pop("concat_dim", 1))
    _reject_extra("parallel", params)
    return _Parallel(
        [nn.Sequential(*_build_layers(branch)) for branch in branches],
        merge=merge,
        concat_dim=concat_dim,
    )


@register_component("repeat")
def _repeat(params: dict[str, Any]) -> nn.Module:
    layers = params.pop("layers", None)
    times = int(params.pop("times", 1))
    if times < 1:
        raise ValueError("repeat 'times' must be >= 1")
    share_weights = bool(params.pop("share_weights", False))
    _reject_extra("repeat", params)

    if share_weights:
        return _RepeatBlock(nn.Sequential(*_build_layers(layers)), times)

    blocks: list[nn.Module] = []
    for _ in range(times):
        blocks.extend(_build_layers(layers))
    return nn.Sequential(*blocks)


@register_component("flatten")
def _flatten(params: dict[str, Any]) -> nn.Module:
    start_dim = int(params.pop("start_dim", 1))
    end_dim = int(params.pop("end_dim", -1))
    _reject_extra("flatten", params)
    return nn.Flatten(start_dim, end_dim)


@register_component("reshape")
def _reshape(params: dict[str, Any]) -> nn.Module:
    shape = params.pop("shape", None)
    if shape is None:
        raise ValueError("reshape requires 'shape'")
    _reject_extra("reshape", params)
    return _Reshape(tuple(int(dim) for dim in shape))


@register_component("transpose")
def _transpose(params: dict[str, Any]) -> nn.Module:
    dim0 = int(params.pop("dim0", 1))
    dim1 = int(params.pop("dim1", 2))
    _reject_extra("transpose", params)
    return _Transpose(dim0, dim1)


@register_component("global_pool")
def _global_pool(params: dict[str, Any]) -> nn.Module:
    mode = str(params.pop("mode", "avg")).lower()
    _reject_extra("global_pool", params)
    if mode in {"avg", "average", "mean"}:
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
    if mode == "max":
        return nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(1))
    raise ValueError(f"Unknown global_pool mode {mode!r}")


@register_component("tokens_mean")
def _tokens_mean(params: dict[str, Any]) -> nn.Module:
    _reject_extra("tokens_mean", params)
    return _TokensMean()


@register_component("tokens_cls")
def _tokens_cls(params: dict[str, Any]) -> nn.Module:
    index = int(params.pop("index", 0))
    _reject_extra("tokens_cls", params)
    return _TokensSelect(index)


@register_component("tokens_flatten")
def _tokens_flatten(params: dict[str, Any]) -> nn.Module:
    _reject_extra("tokens_flatten", params)
    return _TokensFlatten()


@register_component("dropout")
def _dropout(params: dict[str, Any]) -> nn.Module:
    rate = float(params.pop("p", params.pop("rate", 0.5)))
    _reject_extra("dropout", params)
    return nn.Dropout(rate)


@register_component("activation")
def _activation(params: dict[str, Any]) -> nn.Module:
    return _activation_module(params)


@register_component("norm")
def _norm(params: dict[str, Any]) -> nn.Module:
    kind = params.pop("kind", params.pop("name", None))
    num_features = params.pop("num_features", None)
    if kind is None or num_features is None:
        raise ValueError("norm requires 'kind' and 'num_features'")
    spatial = bool(params.pop("spatial", False))
    return _norm_module(kind, int(num_features), spatial=spatial, **params)


@register_component("classifier")
def _classifier(params: dict[str, Any]) -> nn.Module:
    in_features = params.pop("in_features", None)
    num_classes = params.pop("num_classes", None)
    if num_classes is None:
        raise ValueError("classifier requires 'num_classes'")
    hidden_dim = params.pop("hidden_dim", None)
    activation = params.pop("activation", "relu")
    dropout = float(params.pop("dropout", 0.0))
    bias = bool(params.pop("bias", True))
    _reject_extra("classifier", params)
    return _Classifier(
        in_features=in_features,
        num_classes=int(num_classes),
        hidden_dim=int(hidden_dim) if hidden_dim is not None else None,
        activation=activation,
        dropout=dropout,
        bias=bias,
    )


# Convenience alias: {"type": "head", ...} behaves like {"type": "classifier", ...}
_COMPONENTS["head"] = _COMPONENTS["classifier"]


# ---------------------------------------------------------------------------
# Internal shapes and adapters
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Conv patch projection: (B, C, H, W) -> (B, N, embed_dim)."""

    def __init__(self, projection: nn.Module, norm: nn.Module, dropout: float) -> None:
        super().__init__()
        self.projection = projection
        self.norm = norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x).flatten(2).transpose(1, 2)
        return self.dropout(self.norm(x))


class _TransformerEncoderBlock(nn.Module):
    """Transformer encoder that accepts image feature maps or token sequences."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        d_model: int,
        positional_encoding: str,
        max_tokens: int,
        output: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.output = output

        if positional_encoding not in {"none", "learnable", "sinusoidal"}:
            raise ValueError(
                "positional_encoding must be 'none', 'learnable', or 'sinusoidal'"
            )
        self.positional_encoding = positional_encoding

        if positional_encoding == "learnable":
            self.position = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.trunc_normal_(self.position, std=0.02)
        elif positional_encoding == "sinusoidal":
            self.register_buffer(
                "position",
                self._sinusoidal(max_tokens, d_model),
                persistent=False,
            )
        else:
            self.position = None

        if output not in {"tokens", "image"}:
            raise ValueError("transformer_encoder output must be 'tokens' or 'image'")

    @staticmethod
    def _sinusoidal(max_tokens: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_tokens, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_shape: tuple[int, int] | None = None

        if x.dim() == 4:
            batch, channels, height, width = x.shape
            if channels != self.d_model:
                raise ValueError(
                    f"Expected {self.d_model} image channels for the transformer, got {channels}"
                )
            tokens = x.flatten(2).transpose(1, 2)
            image_shape = (height, width)
        elif x.dim() == 3:
            tokens = x
        else:
            raise ValueError("transformer_encoder expects (B, C, H, W) or (B, N, C) input")

        if tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected token width {self.d_model}, got {tokens.shape[-1]}"
            )

        if self.position is not None:
            if tokens.shape[1] > self.max_tokens:
                raise ValueError(
                    f"Sequence length {tokens.shape[1]} exceeds max_tokens={self.max_tokens}"
                )
            tokens = tokens + self.position[:, : tokens.shape[1]]

        tokens = self.encoder(tokens)

        if self.output == "tokens":
            return tokens
        if image_shape is None:
            raise ValueError(
                "transformer_encoder output='image' requires (B, C, H, W) input"
            )

        height, width = image_shape
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.d_model, height, width)


class _Residual(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module | None, scale: float = 1.0) -> None:
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        return identity + self.scale * self.block(x)


class _Parallel(nn.Module):
    def __init__(self, branches: Sequence[nn.Module], *, merge: str, concat_dim: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge
        self.concat_dim = concat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        if self.merge in {"sum", "add"}:
            return torch.stack(outputs, dim=0).sum(dim=0)
        if self.merge in {"mean", "avg"}:
            return torch.stack(outputs, dim=0).mean(dim=0)
        if self.merge == "max":
            return torch.stack(outputs, dim=0).amax(dim=0)
        if self.merge == "mul":
            result = outputs[0]
            for output in outputs[1:]:
                result = result * output
            return result
        if self.merge == "concat":
            return torch.cat(outputs, dim=self.concat_dim)
        raise ValueError(f"Unknown parallel merge mode {self.merge!r}")


class _RepeatBlock(nn.Module):
    def __init__(self, block: nn.Module, times: int) -> None:
        super().__init__()
        self.block = block
        self.times = times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.times):
            x = self.block(x)
        return x


class _Reshape(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], *self.shape)


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class _TokensMean(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class _TokensSelect(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.index]


class _TokensFlatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class _Classifier(nn.Module):
    """Final classifier that accepts image maps, token sequences, or features."""

    def __init__(
        self,
        *,
        in_features: Any,
        num_classes: int,
        hidden_dim: int | None,
        activation: Any,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            layers: list[nn.Module] = [_linear_module(in_features, num_classes, bias=bias)]
        else:
            layers = [
                _linear_module(in_features, hidden_dim, bias=bias),
                _activation_module(activation),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, num_classes, bias=bias))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        elif x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() != 2:
            raise ValueError("classifier expects (B, C), (B, N, C), or (B, C, H, W) input")
        return self.head(x)


class BaseModel(nn.Module):
    """A 64x64 RGB EuroSAT classifier built from a declarative layer list.

    Parameters
    ----------
    layers:
        Ordered component specs. Each spec must have a ``type``. Nested composite
        components (``sequential``, ``residual``, ``parallel``, ``repeat``) accept
        their own ``layers``/``branches``.
    in_channels, image_size:
        Shape of the expected input, used for the optional warm-up pass.
    num_classes:
        Stored for convenience; include a ``classifier``/``head`` component if
        you want the model to actually emit class logits.
    warmup:
        Run a dummy batch once during construction. This initialises lazy layers
        (``LazyConv2d``/``LazyLinear``) so ``model.parameters()`` is complete
        before the optimizer is built.
    warmup_shape:
        Optional override for the dummy batch shape, e.g. ``(2, 3, 64, 64)``.
    """

    def __init__(
        self,
        layers: Sequence[ComponentSpec | nn.Module],
        *,
        in_channels: int = 3,
        image_size: int = 64,
        num_classes: int = 10,
        name: str | None = None,
        warmup: bool = True,
        warmup_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.name = name or self.__class__.__name__
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.layers = nn.Sequential(*_build_layers(layers))
        self.warmup_shape = (
            tuple(int(dim) for dim in warmup_shape)
            if warmup_shape is not None
            else (2, self.in_channels, self.image_size, self.image_size)
        )

        if warmup:
            self.initialize()

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "BaseModel":
        """Build from a single spec dict.

        Accepts either ``layers`` or the sections ``stem`` + ``body`` + ``head``.
        """
        params = dict(spec)
        layers = params.pop("layers", None)
        has_sections = any(key in params for key in ("stem", "body", "head"))

        if layers is not None and has_sections:
            raise ValueError("Use either 'layers' or 'stem'/'body'/'head', not both")
        if layers is None:
            layers = _concat_layer_specs(
                params.pop("stem", None),
                params.pop("body", None),
                params.pop("head", None),
            )
        if not layers:
            raise ValueError("Spec must define 'layers' or at least one of 'stem'/'body'/'head'")

        return cls(layers=layers, **params)

    def initialize(self) -> "BaseModel":
        """Materialize lazy modules by running one no-grad dummy batch."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self(torch.zeros(*self.warmup_shape))
        self.train(was_training)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
