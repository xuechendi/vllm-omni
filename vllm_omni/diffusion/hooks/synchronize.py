# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synchronize hook for profiling accuracy on asynchronous backends (XPU, etc).

When profiling with torch.profiler, asynchronous dispatch (common on XPU/SYCL)
means kernel timings overlap across transformer blocks, making per-block
attribution inaccurate. This hook inserts a device synchronize barrier after
each transformer block's forward pass so that profiler traces show clean,
non-overlapping regions per block.

Each block is also wrapped with a ``torch.profiler.record_function`` event
so that per-block regions are labelled in the trace (e.g.
``WanTransformerBlock_0``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook

logger = init_logger(__name__)

_HOOK_NAME = "profiler_synchronize"


class SynchronizeHook(ModelHook):
    """Wraps a block forward with a profiler event and synchronizes after it."""

    def __init__(self, synchronize_fn, event_name: str):
        self._synchronize_fn = synchronize_fn
        self._event_name = event_name
        self._record_fn: torch.profiler.record_function | None = None

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        self._record_fn = torch.profiler.record_function(self._event_name)
        self._record_fn.__enter__()
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        self._synchronize_fn()
        if self._record_fn is not None:
            self._record_fn.__exit__(None, None, None)
            self._record_fn = None
        return output


def apply_profiler_synchronize_hooks(pipeline: nn.Module) -> int:
    """Apply synchronize hooks to all transformer blocks in a pipeline.

    Uses ``_repeated_blocks`` (declared on every transformer model) to
    auto-discover block class names, then attaches a
    :class:`SynchronizeHook` to each matching sub-module.  Each block
    gets a named ``torch.profiler.record_function`` event (e.g.
    ``WanTransformerBlock_0``) so traces show labelled per-block regions.

    Args:
        pipeline: The top-level pipeline model (e.g. Wan22I2VPipeline).

    Returns:
        Number of hooks applied.
    """
    from vllm_omni.platforms import current_omni_platform

    synchronize_fn = current_omni_platform.synchronize

    transformer_attrs = ["transformer", "transformer_2", "dit", "unet"]
    count = 0

    for attr in transformer_attrs:
        transformer = getattr(pipeline, attr, None)
        if transformer is None:
            continue

        repeated_blocks = getattr(transformer, "_repeated_blocks", None)
        if not repeated_blocks:
            continue

        block_index = 0
        for submod in transformer.modules():
            if submod.__class__.__name__ in repeated_blocks:
                event_name = f"{submod.__class__.__name__}_{block_index}"
                registry = HookRegistry.get_or_create(submod)
                registry.register_hook(_HOOK_NAME, SynchronizeHook(synchronize_fn, event_name))
                block_index += 1
                count += 1

    return count


def remove_profiler_synchronize_hooks(pipeline: nn.Module) -> int:
    """Remove synchronize hooks from all transformer blocks.

    Returns:
        Number of hooks removed.
    """
    transformer_attrs = ["transformer", "transformer_2", "dit", "unet"]
    count = 0

    for attr in transformer_attrs:
        transformer = getattr(pipeline, attr, None)
        if transformer is None:
            continue

        repeated_blocks = getattr(transformer, "_repeated_blocks", None)
        if not repeated_blocks:
            continue

        for submod in transformer.modules():
            if submod.__class__.__name__ in repeated_blocks:
                registry: HookRegistry | None = getattr(submod, "_hook_registry", None)
                if registry is not None and registry.get_hook(_HOOK_NAME) is not None:
                    registry.remove_hook(_HOOK_NAME)
                    count += 1

    return count
