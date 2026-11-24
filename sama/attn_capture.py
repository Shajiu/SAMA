"""Capture attention weights from the last transformer layer only.

Why
---
SAMA's structural objective needs the *last* layer's self-attention map to know where
the model looks inside the answer region. Asking the model for ``output_attentions``
makes every one of the 28 layers keep a ``[batch, heads, seq, seq]`` tensor alive.
With ``max_length=4096`` and three views that is ~26 GiB per view, which is what forced
the earlier run to halve ``max_pixels``.

Silencing layers 0..N-2 turns those tensors into garbage as soon as the layer returns,
while ``attentions[-1]`` keeps exactly the same value as before. That is the whole
point of this module: the losses and the trainer need no change, and the numbers stay
bit-identical, so no re-baselining is needed.

Memory saved: about (num_layers - 1) x [B, H, S, S]; time saved: only the allocation
and copy overhead (roughly a few percent).
"""

from __future__ import annotations

import functools

import torch.nn as nn

_MARKER = "_sama_attn_capture_installed"


def _silence(layer_forward):
    """Wrap a decoder layer so it never returns its attention weights."""

    @functools.wraps(layer_forward)
    def wrapper(*args, **kwargs):
        kwargs["output_attentions"] = False
        return layer_forward(*args, **kwargs)

    return wrapper


def _find_transformer_layers(model) -> nn.ModuleList:
    """Locate the LLM decoder layers of a Qwen2.5-VL model loaded through ms-swift."""
    candidates = (
        ("model.language_model", lambda m: getattr(getattr(m, "model", None), "language_model", None)),
        ("language_model", lambda m: getattr(m, "language_model", None)),
        ("model.model", lambda m: getattr(getattr(m, "model", None), "model", None)),
        ("model", lambda m: getattr(m, "model", None)),
    )
    for path, getter in candidates:
        node = getter(model)
        layers = getattr(node, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    raise RuntimeError(
        "cannot locate transformer layers: tried " + ", ".join(name for name, _ in candidates)
    )


def enable_last_layer_attention(model) -> int:
    """Patch the decoder stack so only the final layer emits attention weights.

    Returns the number of layers that were silenced.
    """
    layers = _find_transformer_layers(model)
    if getattr(layers, _MARKER, False):
        return 0
    for layer in list(layers)[:-1]:
        if getattr(layer, _MARKER, False):
            continue
        layer.forward = _silence(layer.forward)
        setattr(layer, _MARKER, True)
    setattr(layers, _MARKER, True)
    return max(0, len(layers) - 1)
