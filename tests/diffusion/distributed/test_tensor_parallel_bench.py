# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark tests for tensor parallel communication primitives.

Measures latency and effective bandwidth of all_reduce and reduce_scatter
operations that form the backbone of tensor parallelism, across 2 and 4 GPUs.

Usage
-----
Run all TP benchmarks (performance + correctness):

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s

Run only TP-2 (2-card) benchmarks:

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "tp2"

Run only TP-4 (4-card) benchmarks:

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "tp4"

Run only all_reduce benchmarks (both TP-2 and TP-4):

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "all_reduce_bench"

Run only reduce_scatter benchmarks:

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "reduce_scatter_bench"

Run only correctness tests (no timing):

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "correctness"

Filter by dtype (fp16 or bf16):

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "fp16"
    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "bf16"

Combine filters (e.g. TP-4 all_reduce fp16 only):

    pytest tests/diffusion/distributed/test_tensor_parallel_bench.py -v -s -k "tp4 and all_reduce_bench and fp16"

Output
------
Each benchmark prints a JSON line for CI monitoring, e.g.:

    TP_ALL_REDUCE_BENCH_JSON={'op': 'all_reduce', 'world_size': 2, 'dtype': 'torch.float16',
        'shape': '(1024, 4096)', 'msg_bytes': 8388608, 'avg_ms': 0.31, 'bus_bw_gbps': 27.26, ...}

Key metrics:
- avg_ms     : average per-iteration latency (lower is better)
- bus_bw_gbps: effective bus bandwidth in GB/s (higher is better)
              all_reduce formula  : 2*(N-1)/N * msg_bytes / time
              reduce_scatter formula: (N-1)/N * msg_bytes / time

Notes
-----
- Requires >= 2 GPUs for TP-2 tests, >= 4 GPUs for TP-4 tests.
  Tests auto-skip when insufficient GPUs are detected.
- Platform-agnostic: works on CUDA, XPU, and other torch device backends.
- A loose 50 ms sanity bound catches gross regressions without flakiness.
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _set_dist_env(*, rank: int, world_size: int, master_port: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _bench_elapsed_ms(fn, *, iters: int) -> float:
    """Platform-agnostic GPU timing: tries device Event, falls back to synchronize + wall clock."""
    device_type = current_omni_platform.device_type
    device_mod = getattr(torch, device_type, None)

    if device_mod is not None and hasattr(device_mod, "Event"):
        t0 = device_mod.Event(enable_timing=True)
        t1 = device_mod.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            fn()
        t1.record()
        current_omni_platform.synchronize()
        return float(t0.elapsed_time(t1))

    current_omni_platform.synchronize()
    wall_start = time.perf_counter()
    for _ in range(iters):
        fn()
    current_omni_platform.synchronize()
    return (time.perf_counter() - wall_start) * 1000.0


@dataclass(frozen=True, slots=True)
class _TPBenchCase:
    world_size: int
    dtype: torch.dtype
    hidden_dim: int
    seq_len: int

    @property
    def label(self) -> str:
        dt = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
        return f"tp{self.world_size}_{dt.get(self.dtype, str(self.dtype))}_{self.hidden_dim}h_{self.seq_len}s"


TP_BENCH_CASES: list[_TPBenchCase] = [
    _TPBenchCase(world_size=2, dtype=torch.float16, hidden_dim=4096, seq_len=1024),
    _TPBenchCase(world_size=2, dtype=torch.bfloat16, hidden_dim=4096, seq_len=1024),
    _TPBenchCase(world_size=4, dtype=torch.float16, hidden_dim=4096, seq_len=1024),
    _TPBenchCase(world_size=4, dtype=torch.bfloat16, hidden_dim=4096, seq_len=1024),
    _TPBenchCase(world_size=2, dtype=torch.float16, hidden_dim=8192, seq_len=2048),
    _TPBenchCase(world_size=4, dtype=torch.float16, hidden_dim=8192, seq_len=2048),
]


@pytest.mark.parametrize("case", TP_BENCH_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_tp_all_reduce_bench(case: _TPBenchCase) -> None:
    """Benchmark all_reduce latency and bandwidth under tensor parallelism."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _tp_all_reduce_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _tp_all_reduce_worker(local_rank: int, case: _TPBenchCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=case.world_size)

        tp_group = get_world_group().device_group

        tensor = torch.randn(
            case.seq_len, case.hidden_dim,
            dtype=case.dtype, device=device,
        )

        warmup_iters = 20
        bench_iters = 200

        with torch.no_grad():
            for _ in range(warmup_iters):
                t = tensor.clone()
                dist.all_reduce(t, group=tp_group)
            current_omni_platform.synchronize()

            def step():
                t = tensor.clone()
                dist.all_reduce(t, group=tp_group)

            total_ms = _bench_elapsed_ms(step, iters=bench_iters)
            avg_ms = total_ms / bench_iters

        elem_bytes = tensor.element_size()
        msg_bytes = tensor.numel() * elem_bytes
        bus_factor = 2.0 * (case.world_size - 1) / case.world_size
        bus_bw_gbps = (bus_factor * msg_bytes) / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "all_reduce",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "shape": f"({case.seq_len}, {case.hidden_dim})",
                "msg_bytes": msg_bytes,
                "avg_ms": round(avg_ms, 4),
                "bus_bw_gbps": round(bus_bw_gbps, 3),
                "iters": bench_iters,
            }
            print(f"TP_ALL_REDUCE_BENCH_JSON={result}")

        assert avg_ms < 50.0, (
            f"all_reduce too slow: {avg_ms:.3f}ms (world_size={case.world_size})"
        )
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("case", TP_BENCH_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_tp_reduce_scatter_bench(case: _TPBenchCase) -> None:
    """Benchmark reduce_scatter latency and bandwidth under tensor parallelism."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _tp_reduce_scatter_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _tp_reduce_scatter_worker(local_rank: int, case: _TPBenchCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=case.world_size)

        tp_group = get_world_group().device_group

        shard_seq = case.seq_len // case.world_size
        input_tensor = torch.randn(
            case.seq_len, case.hidden_dim,
            dtype=case.dtype, device=device,
        )
        output_tensor = torch.empty(
            shard_seq, case.hidden_dim,
            dtype=case.dtype, device=device,
        )

        input_list = list(input_tensor.chunk(case.world_size, dim=0))

        warmup_iters = 20
        bench_iters = 200

        with torch.no_grad():
            for _ in range(warmup_iters):
                dist.reduce_scatter(output_tensor, input_list, group=tp_group)
            current_omni_platform.synchronize()

            def step():
                dist.reduce_scatter(output_tensor, input_list, group=tp_group)

            total_ms = _bench_elapsed_ms(step, iters=bench_iters)
            avg_ms = total_ms / bench_iters

        elem_bytes = input_tensor.element_size()
        msg_bytes = input_tensor.numel() * elem_bytes
        bus_factor = (case.world_size - 1) / case.world_size
        bus_bw_gbps = (bus_factor * msg_bytes) / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "reduce_scatter",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "shape": f"({case.seq_len}, {case.hidden_dim})",
                "msg_bytes": msg_bytes,
                "avg_ms": round(avg_ms, 4),
                "bus_bw_gbps": round(bus_bw_gbps, 3),
                "iters": bench_iters,
            }
            print(f"TP_REDUCE_SCATTER_BENCH_JSON={result}")

        assert avg_ms < 50.0, (
            f"reduce_scatter too slow: {avg_ms:.3f}ms (world_size={case.world_size})"
        )
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.core_model
def test_tp_all_reduce_correctness(world_size: int, dtype: torch.dtype) -> None:
    """Verify all_reduce correctness under tensor parallelism across ranks."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Requires {world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    result_queue = torch.multiprocessing.get_context("spawn").Manager().Queue()
    torch.multiprocessing.spawn(
        _tp_all_reduce_correctness_worker,
        args=(world_size, dtype, master_port, result_queue),
        nprocs=world_size,
    )

    results = [result_queue.get() for _ in range(world_size)]
    results.sort(key=lambda x: x[0])

    ref = results[0][1]
    for rank, tensor in results[1:]:
        torch.testing.assert_close(
            tensor, ref,
            rtol=0, atol=0,
            msg=f"all_reduce results differ between rank 0 and rank {rank}",
        )


def _tp_all_reduce_correctness_worker(
    local_rank: int, world_size: int, dtype: torch.dtype,
    master_port: int, result_queue: torch.multiprocessing.Queue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=world_size)

        tp_group = get_world_group().device_group

        torch.manual_seed(42 + local_rank)
        tensor = torch.randn(128, 256, dtype=dtype, device=device)

        with torch.no_grad():
            dist.all_reduce(tensor, group=tp_group)

        result_queue.put((local_rank, tensor.cpu()))
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.core_model
def test_tp_reduce_scatter_correctness(world_size: int, dtype: torch.dtype) -> None:
    """Verify reduce_scatter correctness: scattered output matches expected sum."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Requires {world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    result_queue = torch.multiprocessing.get_context("spawn").Manager().Queue()
    torch.multiprocessing.spawn(
        _tp_reduce_scatter_correctness_worker,
        args=(world_size, dtype, master_port, result_queue),
        nprocs=world_size,
    )

    results = [result_queue.get() for _ in range(world_size)]
    results.sort(key=lambda x: x[0])

    for rank, output, expected in results:
        torch.testing.assert_close(
            output, expected,
            rtol=1e-3 if dtype == torch.bfloat16 else 1e-5,
            atol=1e-3 if dtype == torch.bfloat16 else 1e-5,
            msg=f"reduce_scatter output mismatch on rank {rank}",
        )


def _tp_reduce_scatter_correctness_worker(
    local_rank: int, world_size: int, dtype: torch.dtype,
    master_port: int, result_queue: torch.multiprocessing.Queue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=world_size)

        tp_group = get_world_group().device_group
        total_rows = 256
        hidden = 128
        shard_rows = total_rows // world_size

        all_inputs = []
        for r in range(world_size):
            torch.manual_seed(42 + r)
            all_inputs.append(torch.randn(total_rows, hidden, dtype=dtype, device=device))

        torch.manual_seed(42 + local_rank)
        my_input = torch.randn(total_rows, hidden, dtype=dtype, device=device)
        input_list = list(my_input.chunk(world_size, dim=0))
        output = torch.empty(shard_rows, hidden, dtype=dtype, device=device)

        with torch.no_grad():
            dist.reduce_scatter(output, input_list, group=tp_group)

        expected = sum(
            inp[local_rank * shard_rows : (local_rank + 1) * shard_rows]
            for inp in all_inputs
        )

        result_queue.put((local_rank, output.cpu(), expected.cpu()))
    finally:
        destroy_distributed_env()
