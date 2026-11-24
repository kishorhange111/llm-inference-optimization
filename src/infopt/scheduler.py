"""Dynamic and continuous (iteration-level) batching scheduler with priority queues.

Models the core scheduling difference between static batching (all requests in a
batch wait for the longest sequence to finish) and continuous batching (finished
requests exit immediately on every decode iteration and waiting requests join),
with priority-queue admission, aging to prevent starvation, token-budget caps,
and optional KV-cache preemption via :class:`infopt.kv_paging.KVBlockAllocator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

from infopt.kv_paging import KVBlockAllocator, OutOfBlocksError
from infopt.profiler import LatencyThroughputProfiler, ProfileReport


class BatchPolicy(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    CONTINUOUS = "continuous"


class Priority(IntEnum):
    REALTIME = 0
    INTERACTIVE = 1
    BATCH = 2


@dataclass
class InferenceRequest:
    """Lifecycle state of one scheduled generation request."""

    request_id: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_time: float = 0.0
    priority: Priority = Priority.INTERACTIVE
    tokens_generated: int = 0
    prefilled: bool = False
    start_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_output_tokens


@dataclass(frozen=True)
class SchedulerConfig:
    """Parameters governing batching, step latency model, and starvation aging."""

    max_batch_size: int = 16
    max_tokens_per_step: int = 2048
    policy: BatchPolicy = BatchPolicy.CONTINUOUS
    prefill_ms_per_token: float = 0.08
    decode_base_ms: float = 8.0
    decode_per_seq_ms: float = 0.6
    aging_window_s: float = 2.0


@dataclass
class ScheduleSimulationResult:
    """Outcome of running the scheduler over a workload."""

    policy: BatchPolicy
    report: ProfileReport
    total_steps: int
    avg_batch_size: float
    preemptions: int
    completed_requests: list[InferenceRequest] = field(default_factory=list)


class ContinuousBatchingScheduler:
    """Discrete-event simulator for static, dynamic, and continuous LLM batching."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        kv_allocator: KVBlockAllocator | None = None,
    ) -> None:
        self.config = config or SchedulerConfig()
        self.kv_allocator = kv_allocator
        self.now: float = 0.0
        self._pending: list[InferenceRequest] = []
        self._waiting: list[InferenceRequest] = []
        self._running: list[InferenceRequest] = []
        self._swapped: list[InferenceRequest] = []
        self._completed: list[InferenceRequest] = []
        self._preemptions: int = 0
        self._steps: int = 0
        self._batch_size_sum: int = 0

    def submit(self, request: InferenceRequest) -> None:
        """Enqueue a request for admission at ``request.arrival_time``."""
        if request.prompt_tokens <= 0 or request.max_output_tokens <= 0:
            raise ValueError("prompt_tokens and max_output_tokens must be > 0")
        self._pending.append(request)

    def _effective_priority(self, req: InferenceRequest) -> tuple[float, float, str]:
        """Lower tuple sorts first; waiting time promotes lower-priority items (aging)."""
        wait_s = max(0.0, self.now - req.arrival_time)
        promotion = wait_s / self.config.aging_window_s if self.config.aging_window_s > 0 else 0.0
        return (float(req.priority) - promotion, req.arrival_time, req.request_id)

    def _move_arrivals(self) -> None:
        still_pending: list[InferenceRequest] = []
        for req in self._pending:
            if req.arrival_time <= self.now + 1e-12:
                self._waiting.append(req)
            else:
                still_pending.append(req)
        self._pending = still_pending

    def _try_admit(self) -> None:
        """Admit swapped or waiting requests into the active batch respecting budgets."""
        # In STATIC mode, do not admit new requests until the current batch finishes completely.
        if self.config.policy == BatchPolicy.STATIC and self._running:
            return

        step_tokens = len(self._running)  # 1 decode token per running seq

        # Prefer resuming swapped-out sequences before admitting fresh waiting ones.
        if self._swapped and self.kv_allocator is not None:
            resumed: list[InferenceRequest] = []
            for req in self._swapped:
                if len(self._running) >= self.config.max_batch_size:
                    break
                try:
                    self.kv_allocator.swap_in(req.request_id)
                except OutOfBlocksError:
                    break
                self._running.append(req)
                resumed.append(req)
            for req in resumed:
                self._swapped.remove(req)

        self._waiting.sort(key=self._effective_priority)
        admitted: list[InferenceRequest] = []
        for req in self._waiting:
            if len(self._running) >= self.config.max_batch_size:
                break
            if self._running and step_tokens + req.prompt_tokens > self.config.max_tokens_per_step:
                continue
            if self.kv_allocator is not None:
                try:
                    self.kv_allocator.allocate_sequence(req.request_id, req.prompt_tokens)
                except OutOfBlocksError:
                    break
            req.start_time = self.now
            self._running.append(req)
            step_tokens += req.prompt_tokens
            admitted.append(req)
        for req in admitted:
            self._waiting.remove(req)

    def _preempt_lowest_priority(self) -> bool:
        """Swap out the lowest-priority running sequence to free KV blocks."""
        if not self._running or self.kv_allocator is None:
            return False
        victim = max(self._running, key=lambda r: (int(r.priority), -r.tokens_generated))
        self._running.remove(victim)
        self.kv_allocator.swap_out(victim.request_id)
        self._swapped.append(victim)
        self._preemptions += 1
        return True

    def step(self) -> bool:
        """Advance the scheduler by one iteration. Returns False when no work remains."""
        self._move_arrivals()
        if not self._running and not self._waiting and not self._swapped:
            if not self._pending:
                return False
            self.now = min(r.arrival_time for r in self._pending)
            self._move_arrivals()

        self._try_admit()
        if not self._running:
            # Deadlock guard if KV pool is tiny and everything is swapped
            if self._swapped:
                raise RuntimeError("No running requests and swapped sequences cannot fit in KV allocator")
            return False

        self._steps += 1
        self._batch_size_sum += len(self._running)

        prefill_tokens = sum(r.prompt_tokens for r in self._running if not r.prefilled)
        prefill_s = (prefill_tokens * self.config.prefill_ms_per_token) / 1000.0
        decode_s = (self.config.decode_base_ms + len(self._running) * self.config.decode_per_seq_ms) / 1000.0
        self.now += prefill_s + decode_s

        finished_now: list[InferenceRequest] = []
        for req in list(self._running):
            if req not in self._running or req.is_finished:
                continue  # may have been preempted or already finished (padding in static batch)
            if not req.prefilled:
                req.prefilled = True
                req.first_token_time = self.now
                req.tokens_generated += 1
            else:
                if self.kv_allocator is not None:
                    while True:
                        try:
                            self.kv_allocator.append_tokens(req.request_id, 1)
                            break
                        except OutOfBlocksError:
                            if not self._preempt_lowest_priority() or req not in self._running:
                                break
                if req not in self._running:
                    continue
                req.tokens_generated += 1

            if req.is_finished:
                finished_now.append(req)

        if self.config.policy == BatchPolicy.STATIC:
            # Static batching holds all sequences until every sequence in the batch completes.
            if all(r.is_finished for r in self._running):
                for req in self._running:
                    req.finish_time = self.now
                    if self.kv_allocator is not None:
                        self.kv_allocator.free_sequence(req.request_id)
                    self._completed.append(req)
                self._running.clear()
        else:
            # Dynamic / Continuous batching frees finished requests immediately.
            for req in finished_now:
                req.finish_time = self.now
                self._running.remove(req)
                if self.kv_allocator is not None:
                    self.kv_allocator.free_sequence(req.request_id)
                self._completed.append(req)

        return True

    def run_until_complete(self, gpu: str | None = "A100-40GB") -> ScheduleSimulationResult:
        """Run until all submitted requests complete and return latency/throughput metrics."""
        while self.step():
            pass

        profiler = LatencyThroughputProfiler(f"scheduler-{self.config.policy.value}", gpu=gpu)
        for req in self._completed:
            assert req.first_token_time is not None and req.finish_time is not None
            profiler.record(
                request_id=req.request_id,
                start=req.arrival_time,
                first_token=req.first_token_time,
                end=req.finish_time,
                prompt_tokens=req.prompt_tokens,
                output_tokens=req.tokens_generated,
            )
        report = profiler.summarize()
        avg_bs = self._batch_size_sum / self._steps if self._steps > 0 else 0.0
        return ScheduleSimulationResult(
            policy=self.config.policy,
            report=report,
            total_steps=self._steps,
            avg_batch_size=avg_bs,
            preemptions=self._preemptions,
            completed_requests=list(self._completed),
        )
