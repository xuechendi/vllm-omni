# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-graph replay of fixed-shape TaoMate-H3 DiT forwards.

TaoMate's audio teacher runs nine short forwards per five-second request over
a few hundred packed rows. At that size the DiT is launch-bound: the GPU
finishes each block long before the Python launch path of the next one, so
the nine forwards cost about 85 ms each in eager mode although the device
work is a fraction of that. Replaying a captured graph removes the launch
path entirely.

The capture/replay machinery is the shared ``vllm_omni.diffusion.graph.GraphedForward``.
TaoMate adds one thing: during capture it bypasses the exact AdaLN projection
cache, whose key is a host-side digest of the timestep embedding; the graph
then contains the projections themselves, which are cheap for the three
teacher timestep rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch.nn as nn

from vllm_omni.diffusion.graph import GraphedForward as _GraphedForward
from vllm_omni.diffusion.graph import clone_static, copy_into, kwargs_signature


@contextmanager
def _adaln_cache_bypassed(module: nn.Module) -> Iterator[None]:
    """Turn the host-keyed AdaLN projection cache off for the duration."""
    cache = getattr(module, "adaln_cache", None)
    if cache is None or not hasattr(cache, "max_bytes"):
        yield
        return
    previous = cache.max_bytes
    cache.max_bytes = 0
    try:
        yield
    finally:
        cache.max_bytes = previous


class GraphedForward(_GraphedForward):
    """The shared graph wrapper with TaoMate's AdaLN cache bypassed during capture."""

    def __init__(self, module: nn.Module, **kwargs: Any) -> None:
        kwargs.setdefault("name", "taomate_h3")
        super().__init__(module, capture_context=lambda: _adaln_cache_bypassed(module), **kwargs)


__all__ = ["GraphedForward", "clone_static", "copy_into", "kwargs_signature"]
