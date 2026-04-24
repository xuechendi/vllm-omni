# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark tests for various distributed communication primitives.

Measures latency and effective bandwidth of all_gather, all_to_all,
broadcast, and point-to-point (P2P) operations across 2 and 4 GPUs.
These operations are used across different parallelism strategies
(sequence parallel, CFG parallel, pipeline parallel, etc.).

Usage
-----
Run all var-distributed benchmarks (performance + correctness):

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s

Run only 2-card benchmarks:

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "w2"

Run only 4-card benchmarks:

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "w4"

Run a single operation (all_gather / broadcast / all_to_all / ring_p2p):

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "all_gather_bench"
    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "broadcast_bench"
    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "all_to_all_bench"
    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "ring_p2p_bench"

Run only correctness tests (no timing):

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "correctness"

Run the summary table (all ops at a fixed 8M-element fp16 tensor):

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "summary"

Filter by dtype:

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "fp16"
    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "bf16"

Combine filters (e.g. 4-card all_gather fp16 only):

    pytest tests/diffusion/distributed/test_var_distributed_bench.py -v -s -k "w4 and all_gather_bench and fp16"

Output
------
Each benchmark prints a JSON line for CI monitoring, e.g.:

    VAR_DIST_BENCH_JSON={'op': 'all_gather', 'world_size': 2, 'dtype': 'torch.float16',
        'msg_bytes': 2097152, 'avg_ms': 0.12, 'bus_bw_gbps': 8.74}

The summary test prints a comparison table:

    ======================================================================
     Distributed Ops Summary (world_size=2, 8M fp16 elements)
    ======================================================================
     Operation            Avg (ms)     BW (GB/s)
     -------------------- ------------ ------------
     all_reduce           0.3100       25.800
     all_gather           0.1500       10.500
     broadcast            0.0800       19.200
     reduce_scatter       0.1700       9.400
    ======================================================================

Key metrics:
- avg_ms      : average per-iteration latency (lower is better)
- bus_bw_gbps : effective bus bandwidth in GB/s (higher is better)
- algo_bw_gbps: algorithmic bandwidth (for broadcast)
- eff_gbps    : effective throughput (for all_to_all roundtrip)
- p2p_bw_gbps : unidirectional P2P bandwidth (for ring P2P)

Notes
-----
- Requires >= 2 GPUs for w2 tests, >= 4 GPUs for w4 tests.
  Tests auto-skip when insufficient GPUs are detected.
- Platform-agnostic: works on CUDA, XPU, and other torch device backends.
- Message sizes span 2 KB to 32 MB to cover both latency-bound and
  bandwidth-bound regimes.
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

from vllm_omni.diffusion.distributed.comm import RingComm, SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
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


def _max_all_reduce(pg: dist.ProcessGroup, value: float, *, device: torch.device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pg)
    return float(t.item())


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


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _BenchCase:
    world_size: int
    dtype: torch.dtype
    numel: int

    @property
    def msg_bytes(self) -> int:
        return self.numel * torch.tensor([], dtype=self.dtype).element_size()

    @property
    def label(self) -> str:
        dt = {torch.float16: "fp16", torch.bfloat16: "bf16"}
        mb = self.msg_bytes / (1024 * 1024)
        return f"w{self.world_size}_{dt.get(self.dtype, str(self.dtype))}_{mb:.0f}MB"


BENCH_CASES: list[_BenchCase] = [
    # Small messages (latency-bound)
    _BenchCase(world_size=2, dtype=torch.float16, numel=1024),
    _BenchCase(world_size=4, dtype=torch.float16, numel=1024),
    # Medium messages
    _BenchCase(world_size=2, dtype=torch.float16, numel=1024 * 1024),
    _BenchCase(world_size=4, dtype=torch.float16, numel=1024 * 1024),
    _BenchCase(world_size=2, dtype=torch.bfloat16, numel=1024 * 1024),
    _BenchCase(world_size=4, dtype=torch.bfloat16, numel=1024 * 1024),
    # Large messages (bandwidth-bound)
    _BenchCase(world_size=2, dtype=torch.float16, numel=16 * 1024 * 1024),
    _BenchCase(world_size=4, dtype=torch.float16, numel=16 * 1024 * 1024),
    _BenchCase(world_size=2, dtype=torch.bfloat16, numel=16 * 1024 * 1024),
    _BenchCase(world_size=4, dtype=torch.bfloat16, numel=16 * 1024 * 1024),
]

WARMUP_ITERS = 20
BENCH_ITERS = 200


# ---------------------------------------------------------------------------
# 1. all_gather benchmark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", BENCH_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_all_gather_bench(case: _BenchCase) -> None:
    """Benchmark all_gather latency and bandwidth."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _all_gather_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _all_gather_worker(local_rank: int, case: _BenchCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=case.world_size)

        pg = get_world_group().device_group
        shard_numel = case.numel // case.world_size
        local_tensor = torch.randn(shard_numel, dtype=case.dtype, device=device)
        gather_list = [torch.empty_like(local_tensor) for _ in range(case.world_size)]

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                dist.all_gather(gather_list, local_tensor, group=pg)
            current_omni_platform.synchronize()

            def step():
                dist.all_gather(gather_list, local_tensor, group=pg)

            total_ms = _bench_elapsed_ms(step, iters=BENCH_ITERS)
            avg_ms = total_ms / BENCH_ITERS

        bus_factor = (case.world_size - 1) / case.world_size
        bus_bw_gbps = (bus_factor * case.msg_bytes) / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "all_gather",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "msg_bytes": case.msg_bytes,
                "avg_ms": round(avg_ms, 4),
                "bus_bw_gbps": round(bus_bw_gbps, 3),
            }
            print(f"VAR_DIST_BENCH_JSON={result}")

        assert avg_ms < 50.0, f"all_gather too slow: {avg_ms:.3f}ms"
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# 2. broadcast benchmark
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", BENCH_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_broadcast_bench(case: _BenchCase) -> None:
    """Benchmark broadcast latency and bandwidth."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _broadcast_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _broadcast_worker(local_rank: int, case: _BenchCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=case.world_size)

        pg = get_world_group().device_group
        tensor = torch.randn(case.numel, dtype=case.dtype, device=device)

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                dist.broadcast(tensor, src=0, group=pg)
            current_omni_platform.synchronize()

            def step():
                dist.broadcast(tensor, src=0, group=pg)

            total_ms = _bench_elapsed_ms(step, iters=BENCH_ITERS)
            avg_ms = total_ms / BENCH_ITERS

        algo_bw_gbps = case.msg_bytes / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "broadcast",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "msg_bytes": case.msg_bytes,
                "avg_ms": round(avg_ms, 4),
                "algo_bw_gbps": round(algo_bw_gbps, 3),
            }
            print(f"VAR_DIST_BENCH_JSON={result}")

        assert avg_ms < 50.0, f"broadcast too slow: {avg_ms:.3f}ms"
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# 3. all_to_all (sequence parallel) benchmark
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _AllToAllCase:
    world_size: int
    dtype: torch.dtype
    batch_size: int
    seq_len_per_rank: int
    num_heads: int
    head_dim: int

    @property
    def label(self) -> str:
        dt = {torch.float16: "fp16", torch.bfloat16: "bf16"}
        return (
            f"w{self.world_size}_{dt.get(self.dtype, str(self.dtype))}"
            f"_b{self.batch_size}_s{self.seq_len_per_rank}_h{self.num_heads}"
        )


ALL_TO_ALL_CASES: list[_AllToAllCase] = [
    _AllToAllCase(world_size=2, dtype=torch.float16, batch_size=1, seq_len_per_rank=256, num_heads=32, head_dim=128),
    _AllToAllCase(world_size=4, dtype=torch.float16, batch_size=1, seq_len_per_rank=256, num_heads=32, head_dim=128),
    _AllToAllCase(world_size=2, dtype=torch.bfloat16, batch_size=2, seq_len_per_rank=512, num_heads=32, head_dim=128),
    _AllToAllCase(world_size=4, dtype=torch.bfloat16, batch_size=2, seq_len_per_rank=512, num_heads=32, head_dim=128),
    _AllToAllCase(world_size=2, dtype=torch.float16, batch_size=1, seq_len_per_rank=1024, num_heads=64, head_dim=128),
    _AllToAllCase(world_size=4, dtype=torch.float16, batch_size=1, seq_len_per_rank=1024, num_heads=64, head_dim=128),
]


@pytest.mark.parametrize("case", ALL_TO_ALL_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_all_to_all_bench(case: _AllToAllCase) -> None:
    """Benchmark SeqAllToAll4D (Ulysses SP) latency for forward + backward all-to-all."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    if case.num_heads % case.world_size != 0:
        pytest.skip(f"num_heads ({case.num_heads}) not divisible by world_size ({case.world_size})")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _all_to_all_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _all_to_all_worker(local_rank: int, case: _AllToAllCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(ulysses_degree=case.world_size)

        sp_group = get_sp_group().ulysses_group

        tensor = torch.randn(
            case.batch_size, case.seq_len_per_rank, case.num_heads, case.head_dim,
            dtype=case.dtype, device=device,
        )

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                out = SeqAllToAll4D.apply(sp_group, tensor, 2, 1, False)
                _ = SeqAllToAll4D.apply(sp_group, out, 1, 2, False)
            current_omni_platform.synchronize()

            def step():
                out = SeqAllToAll4D.apply(sp_group, tensor, 2, 1, False)
                _ = SeqAllToAll4D.apply(sp_group, out, 1, 2, False)

            total_ms = _bench_elapsed_ms(step, iters=BENCH_ITERS)
            avg_ms = total_ms / BENCH_ITERS

        elem_bytes = tensor.element_size()
        per_a2a_bytes = tensor.numel() * elem_bytes * 2
        total_comm_bytes = 2 * per_a2a_bytes
        eff_gbps = total_comm_bytes / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "seq_all_to_all_4d_roundtrip",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "shape": f"({case.batch_size}, {case.seq_len_per_rank}, {case.num_heads}, {case.head_dim})",
                "avg_ms": round(avg_ms, 4),
                "eff_gbps": round(eff_gbps, 3),
            }
            print(f"VAR_DIST_BENCH_JSON={result}")

        assert avg_ms < 50.0, f"all_to_all roundtrip too slow: {avg_ms:.3f}ms"
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# 4. Ring P2P (point-to-point) benchmark
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _P2PCase:
    world_size: int
    dtype: torch.dtype
    numel: int

    @property
    def msg_bytes(self) -> int:
        return self.numel * torch.tensor([], dtype=self.dtype).element_size()

    @property
    def label(self) -> str:
        dt = {torch.float16: "fp16", torch.bfloat16: "bf16"}
        mb = self.msg_bytes / (1024 * 1024)
        return f"w{self.world_size}_{dt.get(self.dtype, str(self.dtype))}_{mb:.1f}MB"


P2P_CASES: list[_P2PCase] = [
    _P2PCase(world_size=2, dtype=torch.float16, numel=1024 * 1024),
    _P2PCase(world_size=4, dtype=torch.float16, numel=1024 * 1024),
    _P2PCase(world_size=2, dtype=torch.float16, numel=8 * 1024 * 1024),
    _P2PCase(world_size=4, dtype=torch.float16, numel=8 * 1024 * 1024),
    _P2PCase(world_size=2, dtype=torch.bfloat16, numel=8 * 1024 * 1024),
    _P2PCase(world_size=4, dtype=torch.bfloat16, numel=8 * 1024 * 1024),
]


@pytest.mark.parametrize("case", P2P_CASES, ids=lambda c: c.label)
@pytest.mark.core_model
def test_ring_p2p_bench(case: _P2PCase) -> None:
    """Benchmark RingComm point-to-point send/recv latency."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < case.world_size:
        pytest.skip(f"Requires {case.world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    torch.multiprocessing.spawn(
        _ring_p2p_worker,
        args=(case, master_port),
        nprocs=case.world_size,
    )


def _ring_p2p_worker(local_rank: int, case: _P2PCase, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=case.world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=case.world_size, rank=local_rank)
        initialize_model_parallel(ring_degree=case.world_size)

        sp_group = get_sp_group()
        ring_pg = sp_group.ring_group

        tensor = torch.randn(case.numel, dtype=case.dtype, device=device)

        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                comm = RingComm(ring_pg)
                _ = comm.send_recv(tensor)
                comm.commit()
                comm.wait()
            current_omni_platform.synchronize()

            def step():
                comm = RingComm(ring_pg)
                _ = comm.send_recv(tensor)
                comm.commit()
                comm.wait()

            total_ms = _bench_elapsed_ms(step, iters=BENCH_ITERS)
            avg_ms = total_ms / BENCH_ITERS

        p2p_bw_gbps = case.msg_bytes / (avg_ms / 1000.0) / 1e9

        if local_rank == 0:
            result = {
                "op": "ring_p2p",
                "world_size": case.world_size,
                "dtype": str(case.dtype),
                "msg_bytes": case.msg_bytes,
                "avg_ms": round(avg_ms, 4),
                "p2p_bw_gbps": round(p2p_bw_gbps, 3),
            }
            print(f"VAR_DIST_BENCH_JSON={result}")

        assert avg_ms < 50.0, f"ring P2P too slow: {avg_ms:.3f}ms"
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# 5. Correctness tests for all_gather and broadcast
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.core_model
def test_all_gather_correctness(world_size: int, dtype: torch.dtype) -> None:
    """Verify all_gather assembles per-rank shards correctly."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Requires {world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    result_queue = torch.multiprocessing.get_context("spawn").Manager().Queue()
    torch.multiprocessing.spawn(
        _all_gather_correctness_worker,
        args=(world_size, dtype, master_port, result_queue),
        nprocs=world_size,
    )

    results = [result_queue.get() for _ in range(world_size)]
    results.sort(key=lambda x: x[0])

    ref_gathered = results[0][1]
    for rank, gathered in results[1:]:
        torch.testing.assert_close(
            gathered, ref_gathered,
            rtol=0, atol=0,
            msg=f"all_gather results differ between rank 0 and rank {rank}",
        )

    for rank, gathered in results:
        shard = gathered[rank * 128 : (rank + 1) * 128]
        torch.manual_seed(42 + rank)
        expected = torch.randn(128, 64, dtype=dtype)
        torch.testing.assert_close(
            shard, expected,
            rtol=0, atol=0,
            msg=f"all_gather shard from rank {rank} doesn't match original",
        )


def _all_gather_correctness_worker(
    local_rank: int, world_size: int, dtype: torch.dtype,
    master_port: int, result_queue: torch.multiprocessing.Queue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=world_size)

        pg = get_world_group().device_group

        torch.manual_seed(42 + local_rank)
        local_tensor = torch.randn(128, 64, dtype=dtype, device=device)
        gather_list = [torch.empty_like(local_tensor) for _ in range(world_size)]

        with torch.no_grad():
            dist.all_gather(gather_list, local_tensor, group=pg)

        gathered = torch.cat(gather_list, dim=0).cpu()
        result_queue.put((local_rank, gathered))
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.core_model
def test_broadcast_correctness(world_size: int, dtype: torch.dtype) -> None:
    """Verify broadcast delivers identical data from root to all ranks."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Requires {world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    result_queue = torch.multiprocessing.get_context("spawn").Manager().Queue()
    torch.multiprocessing.spawn(
        _broadcast_correctness_worker,
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
            msg=f"broadcast result differs between rank 0 and rank {rank}",
        )


def _broadcast_correctness_worker(
    local_rank: int, world_size: int, dtype: torch.dtype,
    master_port: int, result_queue: torch.multiprocessing.Queue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(tensor_parallel_size=world_size)

        pg = get_world_group().device_group

        if local_rank == 0:
            torch.manual_seed(42)
            tensor = torch.randn(256, 128, dtype=dtype, device=device)
        else:
            tensor = torch.zeros(256, 128, dtype=dtype, device=device)

        with torch.no_grad():
            dist.broadcast(tensor, src=0, group=pg)

        result_queue.put((local_rank, tensor.cpu()))
    finally:
        destroy_distributed_env()


# ---------------------------------------------------------------------------
# 6. Combined summary benchmark (runs all ops and prints comparison table)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.core_model
def test_distributed_ops_summary(world_size: int) -> None:
    """Run all distributed ops at a fixed tensor size and print a comparison summary."""
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Requires {world_size} GPUs, got {available_gpus}")

    master_port = _find_free_port()
    result_queue = torch.multiprocessing.get_context("spawn").Manager().Queue()
    torch.multiprocessing.spawn(
        _summary_worker,
        args=(world_size, master_port, result_queue),
        nprocs=world_size,
    )

    if not result_queue.empty():
        summary = result_queue.get()
        print(f"\n{'='*70}")
        print(f" Distributed Ops Summary (world_size={world_size}, 8M fp16 elements)")
        print(f"{'='*70}")
        print(f" {'Operation':<20} {'Avg (ms)':<12} {'BW (GB/s)':<12}")
        print(f" {'-'*20} {'-'*12} {'-'*12}")
        for entry in summary:
            print(f" {entry['op']:<20} {entry['avg_ms']:<12.4f} {entry['bw_gbps']:<12.3f}")
        print(f"{'='*70}\n")


def _summary_worker(
    local_rank: int, world_size: int, master_port: int,
    result_queue: torch.multiprocessing.Queue,
) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    _set_dist_env(rank=local_rank, world_size=world_size, master_port=master_port)

    try:
        init_distributed_environment(world_size=world_size, rank=local_rank)
        initialize_model_parallel(
            tensor_parallel_size=world_size,
            ulysses_degree=1,
            ring_degree=1,
        )

        pg = get_world_group().device_group
        dtype = torch.float16
        numel = 8 * 1024 * 1024
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        msg_bytes = numel * elem_bytes

        iters = 100
        warmup = 10
        results = []

        def bench(name: str, fn) -> dict:
            with torch.no_grad():
                for _ in range(warmup):
                    fn()
                current_omni_platform.synchronize()
                total_ms = _bench_elapsed_ms(fn, iters=iters)
                avg_ms = total_ms / iters
            avg_ms = _max_all_reduce(pg, avg_ms, device=device)
            bw_gbps = msg_bytes / (avg_ms / 1000.0) / 1e9
            return {"op": name, "avg_ms": avg_ms, "bw_gbps": bw_gbps}

        # all_reduce
        ar_tensor = torch.randn(numel, dtype=dtype, device=device)
        results.append(bench("all_reduce", lambda: dist.all_reduce(ar_tensor, group=pg)))

        # all_gather
        shard = torch.randn(numel // world_size, dtype=dtype, device=device)
        gather_list = [torch.empty_like(shard) for _ in range(world_size)]
        results.append(bench("all_gather", lambda: dist.all_gather(gather_list, shard, group=pg)))

        # broadcast
        bc_tensor = torch.randn(numel, dtype=dtype, device=device)
        results.append(bench("broadcast", lambda: dist.broadcast(bc_tensor, src=0, group=pg)))

        # reduce_scatter
        rs_input = torch.randn(numel, dtype=dtype, device=device)
        rs_output = torch.empty(numel // world_size, dtype=dtype, device=device)
        rs_list = list(rs_input.chunk(world_size))
        results.append(bench("reduce_scatter", lambda: dist.reduce_scatter(rs_output, rs_list, group=pg)))

        if local_rank == 0:
            result_queue.put(results)
    finally:
        destroy_distributed_env()
