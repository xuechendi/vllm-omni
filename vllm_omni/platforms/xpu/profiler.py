# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing_extensions import override
from vllm.logger import init_logger

from vllm_omni.profiler.omni_torch_profiler import (
    OmniTorchProfilerWrapper,
    TorchProfilerActivity,
)

logger = init_logger(__name__)


class XPUTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """XPU-specific profiler wrapper.

    Uses torch.profiler with XPU ProfilerActivity to capture Intel GPU events.
    XPU events are exposed via torch.profiler.ProfilerActivity.XPU and work
    with the standard torch.profiler interface.
    """

    @override
    def _get_default_activities(self) -> list[TorchProfilerActivity]:
        """Default to CPU + XPU profiling for Intel GPUs."""
        return ["CPU", "XPU"]
