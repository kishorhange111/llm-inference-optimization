"""Tests for dynamic and continuous batching scheduler with priority queues."""

import pytest

from infopt.kv_paging import KVBlockAllocator
from infopt.scheduler import (
    BatchPolicy,
    ContinuousBatchingScheduler,
    InferenceRequest,
    Priority,
    SchedulerConfig,
)


def _mixed_workload() -> list[InferenceRequest]:
    """Mix of short (8 tok) and long (120 tok) outputs arriving in waves."""
    reqs = []
    for i in range(12):
        reqs.append(
            InferenceRequest(
                request_id=f"r{i}",
                prompt_tokens=64,
                max_output_tokens=120 if i % 4 == 0 else 8,
                arrival_time=0.0 if i < 6 else 0.05,
            )
        )
    return reqs


def test_continuous_beats_static_batching_on_mixed_lengths():
    static_sched = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4, policy=BatchPolicy.STATIC))
    cont_sched = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=4, policy=BatchPolicy.CONTINUOUS))
    for req in _mixed_workload():
        static_sched.submit(req)
    for req in _mixed_workload():
        cont_sched.submit(req)

    res_static = static_sched.run_until_complete()
    res_cont = cont_sched.run_until_complete()

    assert len(res_static.completed_requests) == 12
    assert len(res_cont.completed_requests) == 12
    assert res_cont.report.output_tok_s > res_static.report.output_tok_s
    assert res_cont.report.e2e_latency_ms["p50"] < res_static.report.e2e_latency_ms["p50"]


def test_priority_queue_ordering_and_starvation_aging():
    sched = ContinuousBatchingScheduler(SchedulerConfig(max_batch_size=1, policy=BatchPolicy.CONTINUOUS, aging_window_s=0.1))
    sched.submit(InferenceRequest("batch-old", 32, 5, arrival_time=0.0, priority=Priority.BATCH))
    sched.submit(InferenceRequest("rt-1", 32, 5, arrival_time=0.0, priority=Priority.REALTIME))
    sched.submit(InferenceRequest("inter-1", 32, 5, arrival_time=0.0, priority=Priority.INTERACTIVE))

    res = sched.run_until_complete()
    order = [r.request_id for r in res.completed_requests]
    assert order[0] == "rt-1"
    assert order[1] == "inter-1"
    assert order[2] == "batch-old"


def test_kv_preemption_under_tight_memory():
    # 10 blocks x 16 tokens = 160 token slots total; 4 requests growing up to 48 tokens each
    alloc = KVBlockAllocator(num_blocks=10, block_size=16)
    sched = ContinuousBatchingScheduler(
        SchedulerConfig(max_batch_size=4, policy=BatchPolicy.CONTINUOUS),
        kv_allocator=alloc,
    )
    for i in range(4):
        sched.submit(
            InferenceRequest(
                request_id=f"mem-{i}",
                prompt_tokens=32,
                max_output_tokens=20,
                arrival_time=0.0,
                priority=Priority.REALTIME if i == 0 else Priority.BATCH,
            )
        )
    res = sched.run_until_complete()
    assert len(res.completed_requests) == 4
    assert res.preemptions >= 1
    assert alloc.stats().used_blocks == 0


def test_invalid_request_rejected():
    sched = ContinuousBatchingScheduler()
    with pytest.raises(ValueError):
        sched.submit(InferenceRequest("bad", prompt_tokens=0, max_output_tokens=5))
