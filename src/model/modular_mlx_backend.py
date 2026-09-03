from __future__ import annotations

import concurrent.futures
from collections import deque
import datetime
import gzip
import math
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
from typing import Any, Mapping


SCHEMA_VERSION = "todorov.modular-mlx-backend.1"
CPU_SCHEMA_VERSION = "todorov.cpu-witness.1"
IPC_SCHEMA_VERSION = "todorov.modular-mlx-ipc.1"
MLX_VERSION = "0.29.3"
MLX_PYTHON = Path("/Users/dttdrv/Projects/Transformerov/.venv/bin/python")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "modular_sequence_role_mlx.py"
QUALIFIER_PATH = PROJECT_ROOT / "scripts" / "qualify_modular_mlx.py"
BACKEND_PATH = PROJECT_ROOT / "src" / "model" / "modular_mlx_backend.py"
MODEL_PATH = PROJECT_ROOT / "src" / "model" / "modular_neural_machine.py"
SOURCES_PATH = PROJECT_ROOT / "src" / "model" / "modular_sources.py"
CPU_EVALUATOR_PATH = PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "modular_sequence_role_cpu.py"
RUNG_ONE_SEEDS = (11, 23, 37, 53, 71)
RUNG_TWO_SEED = 83
POSITIONS = 45_613_056
TARGET_SECONDS = 600
HARD_LIMIT_SECONDS = 1200
ATTEMPT_EVENT_ROWS = 41_472
ROUTING_EVIDENCE_ROWS = 588_240
PARAMETER_UPDATE_RELATIVE_TOLERANCE = 1e-5
PARAMETER_UPDATE_ABSOLUTE_TOLERANCE = 1e-8
TRAINING_ROUTE_ROWS_PER_SEED = {
    "donor": 34_816,
    "router_only": 26_112,
    "joint": 17_408,
    "dense_base": 17_408,
    "dense_continuation": 8_704,
}
ROUTING_ROW_KEYS = (
    "addresses_probed",
    "batch_index",
    "block",
    "block_load_histogram",
    "candidate_blocks",
    "canonical_bypass_ids",
    "checkpoint_sha256",
    "claim_seed",
    "condition",
    "construction_seed",
    "effective_remote_ids",
    "example_index",
    "foreign_source",
    "foreign_source_hit",
    "forward_sequence",
    "intervention",
    "local_block_ids",
    "logical_update",
    "max_bucket_load",
    "model",
    "original_source_hit",
    "overflow_count",
    "phase",
    "posting_reads",
    "query_position",
    "query_underfill_count",
    "raw_remote_ids",
    "required_source",
    "route_workspace_bytes",
    "row_kind",
    "run_id",
    "rung",
    "schema_version",
    "stage",
    "valid_posting_histogram",
)
STAGE_CONTRACTS = {
    "donor": {"updates": 1024, "warmup_updates": 64, "batch_size": 16, "seed_count": 1},
    "router_only": {"updates": 768, "warmup_updates": 48, "batch_size": 16, "seed_count": 5},
    "joint": {"updates": 512, "warmup_updates": 32, "batch_size": 16, "seed_count": 5},
    "dense_base": {"updates": 1024, "warmup_updates": 64, "batch_size": 16, "seed_count": 5},
    "dense_continuation": {"updates": 512, "warmup_updates": 32, "batch_size": 16, "seed_count": 5},
    "rung_two": {"updates": 1536, "warmup_updates": 96, "batch_size": 8, "seed_count": 1},
}


class MlxBackendRefusal(RuntimeError):
    pass


class MlxProtocolError(RuntimeError):
    pass


class MlxBatchLedgerError(RuntimeError):
    pass


class MlxBackgroundWriterError(RuntimeError):
    pass


class MlxResourceSamplerError(RuntimeError):
    pass


class MlxQualificationError(RuntimeError):
    pass


class MlxDeadlineExceeded(MlxQualificationError):
    pass


def enforce_deadline(deadline_ns: int, clock: Any = time.monotonic_ns) -> int:
    if type(deadline_ns) is not int or deadline_ns <= 0 or not callable(clock):
        raise MlxDeadlineExceeded("qualification deadline differs")
    remaining = deadline_ns - int(clock())
    if remaining <= 0:
        raise MlxDeadlineExceeded("qualification exceeded the 1,200 second hard deadline")
    return remaining


class OrderedBackgroundWriter:
    def __init__(self, consumer: Any, max_pending: int):
        if not callable(consumer) or type(max_pending) is not int or max_pending < 1:
            raise MlxBackgroundWriterError("routing writer configuration differs")
        self.consumer = consumer
        self.max_pending = max_pending
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="modular-routing")
        self.pending: deque[Any] = deque()
        self.closed = False

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def _resolve_one(self) -> None:
        future = self.pending.popleft()
        try:
            future.result()
        except BaseException as error:
            for pending in self.pending:
                pending.cancel()
            self.pending.clear()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            raise MlxBackgroundWriterError("routing writer failed") from error

    def submit(self, value: Any) -> None:
        if self.closed:
            raise MlxBackgroundWriterError("routing writer is closed")
        self.pending.append(self.executor.submit(self.consumer, value))
        if len(self.pending) >= self.max_pending:
            self._resolve_one()

    def close(self) -> None:
        if self.closed:
            raise MlxBackgroundWriterError("routing writer is closed")
        while self.pending:
            self._resolve_one()
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        for future in self.pending:
            future.cancel()
        while self.pending:
            future = self.pending.popleft()
            try:
                future.result()
            except BaseException:
                pass
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.closed = True


class QualificationResourceSampler:
    def __init__(self, run_id: str, parent_pid: int, child_pid: int, process_sampler: Any, swap_sampler: Any, interval_seconds: float, writer: Any = None, prior_rows: list[Mapping[str, Any]] | None = None, phase: str = "claim", final_attempted_updates: int = 20_736, final_token_positions: int = POSITIONS):
        if not isinstance(run_id, str) or not run_id or type(parent_pid) is not int or parent_pid < 1 or type(child_pid) is not int or child_pid < 1 or child_pid == parent_pid or not callable(process_sampler) or not callable(swap_sampler):
            raise MlxResourceSamplerError("resource sampler configuration differs")
        if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or not math.isfinite(float(interval_seconds)) or interval_seconds <= 0:
            raise MlxResourceSamplerError("resource sampler interval differs")
        self.run_id = run_id
        self.parent_pid = parent_pid
        self.child_pid = child_pid
        self.process_sampler = process_sampler
        self.swap_sampler = swap_sampler
        self.interval_seconds = float(interval_seconds)
        self.rows = [] if prior_rows is None else [dict(row) for row in prior_rows]
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.child_memory: dict[str, Any] | None = None
        self.memory_rows: list[dict[str, Any]] = []
        self.active_jobs: list[dict[str, Any]] = []
        self.attempted_updates = 0
        self.token_positions = 0
        self.swap_baseline: int | None = None
        self.writer = writer
        self.condition = threading.Condition()
        self.child_exited = False
        self.state_generation = 0
        self.sample_transaction_durations_ns: list[int] = []
        if phase not in {"pilot", "claim"} or type(final_attempted_updates) is not int or final_attempted_updates < 0 or type(final_token_positions) is not int or final_token_positions < 0:
            raise MlxResourceSamplerError("resource sampler phase contract differs")
        self.phase = phase
        self.final_attempted_updates = final_attempted_updates
        self.final_token_positions = final_token_positions

    @property
    def sample_count(self) -> int:
        with self.condition:
            return len(self.rows)

    @property
    def failed(self) -> bool:
        return self.error is not None

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise MlxResourceSamplerError("resource sampler failed") from self.error

    def observe_progress(self, stage: str, seeds: list[int], logical_update: int, attempted_updates: int, token_positions: int, value: Mapping[str, Any]) -> None:
        expected = ("active_memory_bytes", "cache_memory_bytes", "parent_rss_and_swap_required", "peak_memory_bytes")
        if not isinstance(value, Mapping) or tuple(sorted(value)) != expected or value["parent_rss_and_swap_required"] is not True:
            raise MlxResourceSamplerError("MLX child memory differs")
        if any(type(value[field]) is not int or value[field] < 0 for field in ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes")):
            raise MlxResourceSamplerError("MLX child memory value differs")
        if not isinstance(stage, str) or not stage or not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or type(logical_update) is not int or logical_update < 1 or type(attempted_updates) is not int or attempted_updates < self.attempted_updates or type(token_positions) is not int or token_positions < self.token_positions:
            raise MlxResourceSamplerError("resource progress differs")
        with self.condition:
            self.child_memory = dict(value)
            self.active_jobs = [{"worker": f"S{seed}", "seed": seed, "stage": stage, "logical_update": logical_update} for seed in sorted(seeds)]
            self.attempted_updates = attempted_updates
            self.token_positions = token_positions
            self.state_generation += 1
            self.memory_rows.append({"schema_version": SCHEMA_VERSION, "sample_id": len(self.memory_rows), "monotonic_ns": time.monotonic_ns(), "stage": stage, "seeds": sorted(seeds), "logical_update": logical_update, **dict(value)})
            self.condition.notify_all()

    def begin_stage(self, stage: str, seeds: list[int]) -> None:
        if not isinstance(stage, str) or not stage or not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise MlxResourceSamplerError("resource stage identity differs")
        with self.condition:
            self.active_jobs = [{"worker": f"S{seed}", "seed": seed, "stage": stage, "logical_update": 0} for seed in sorted(seeds)]
            self.state_generation += 1
            self.condition.notify_all()

    def observe_started(self, stage: str, seeds: list[int], logical_update: int, attempted_updates: int, token_positions: int) -> None:
        if not isinstance(stage, str) or not stage or not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or type(logical_update) is not int or logical_update < 1:
            raise MlxResourceSamplerError("resource started identity differs")
        if type(attempted_updates) is not int or attempted_updates < self.attempted_updates or type(token_positions) is not int or token_positions < self.token_positions:
            raise MlxResourceSamplerError("resource started counters differ")
        with self.condition:
            self.active_jobs = [{"worker": f"S{seed}", "seed": seed, "stage": stage, "logical_update": logical_update} for seed in sorted(seeds)]
            self.attempted_updates = attempted_updates
            self.token_positions = token_positions
            self.state_generation += 1
            self.condition.notify_all()

    def begin_unattributed_phase(self, stage: str) -> None:
        if stage not in {"evaluation", "closure"}:
            raise MlxResourceSamplerError("resource unattributed phase differs")
        with self.condition:
            self.active_jobs = [{"worker": "MLX", "seed": None, "stage": stage, "logical_update": None}]
            self.state_generation += 1
            self.condition.notify_all()

    def snapshot_rows(self) -> list[dict[str, Any]]:
        self.raise_if_failed()
        with self.condition:
            return [dict(row) for row in self.rows]

    def observe_pilot_progress(self, workload: str, model_seed: int, logical_update: int, attempted_updates: int, token_positions: int, value: Mapping[str, Any]) -> None:
        expected = ("active_memory_bytes", "cache_memory_bytes", "parent_rss_and_swap_required", "peak_memory_bytes")
        if not isinstance(workload, str) or not workload or type(model_seed) is not int or type(logical_update) is not int or logical_update < 0 or not isinstance(value, Mapping) or tuple(sorted(value)) != expected or value["parent_rss_and_swap_required"] is not True:
            raise MlxResourceSamplerError("pilot resource progress differs")
        if any(type(value[field]) is not int or value[field] < 0 for field in ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes")) or type(attempted_updates) is not int or attempted_updates < self.attempted_updates or type(token_positions) is not int or token_positions < self.token_positions:
            raise MlxResourceSamplerError("pilot resource value differs")
        with self.condition:
            self.child_memory = dict(value)
            self.active_jobs = [{"worker": "MLX", "seed": model_seed, "stage": workload, "logical_update": logical_update}]
            self.attempted_updates = attempted_updates
            self.token_positions = token_positions
            self.state_generation += 1
            self.memory_rows.append({"schema_version": SCHEMA_VERSION, "sample_id": len(self.memory_rows), "monotonic_ns": time.monotonic_ns(), "stage": workload, "seeds": [model_seed], "logical_update": logical_update, **dict(value)})
            self.condition.notify_all()

    def await_stage_sample(self, stage: str, seeds: list[int], deadline_ns: int) -> dict[str, Any]:
        expected = [{"worker": f"S{seed}", "seed": seed, "stage": stage, "logical_update": 0} for seed in sorted(seeds)]
        with self.condition:
            while True:
                self.raise_if_failed()
                for row in self.rows:
                    if row["active_jobs"] == expected:
                        return dict(row)
                remaining = deadline_ns - time.monotonic_ns()
                if remaining <= 0:
                    raise MlxResourceSamplerError("resource stage sample deadline exceeded")
                self.condition.wait(min(remaining / 1_000_000_000, 0.25))

    def clear_active_jobs(self, attempted_updates: int, token_positions: int) -> None:
        if attempted_updates != self.final_attempted_updates or token_positions != self.final_token_positions:
            raise MlxResourceSamplerError("final resource counters differ")
        with self.condition:
            self.active_jobs = []
            self.attempted_updates = attempted_updates
            self.token_positions = token_positions
            self.state_generation += 1
            self.condition.notify_all()

    def mark_child_exited(self) -> None:
        with self.condition:
            self.child_exited = True
            self.state_generation += 1
            self.condition.notify_all()

    @property
    def max_sample_transaction_seconds(self) -> float:
        with self.condition:
            if not self.sample_transaction_durations_ns:
                raise MlxResourceSamplerError("resource sample duration evidence is empty")
            return max(self.sample_transaction_durations_ns) / 1_000_000_000

    @property
    def sample_transaction_count(self) -> int:
        with self.condition:
            return len(self.sample_transaction_durations_ns)

    def _sample(self, final_sample: bool = False, deadline_ns: int | None = None) -> bool:
        started_ns = time.perf_counter_ns()
        try:
            return self._sample_transaction(final_sample, deadline_ns)
        finally:
            duration_ns = time.perf_counter_ns() - started_ns
            with self.condition:
                self.sample_transaction_durations_ns.append(duration_ns)

    def _sample_transaction(self, final_sample: bool = False, deadline_ns: int | None = None) -> bool:
        with self.condition:
            expected_pids = [self.parent_pid] if self.child_exited else sorted((self.parent_pid, self.child_pid))
            active_jobs = [dict(job) for job in self.active_jobs]
            attempted_updates = self.attempted_updates
            token_positions = self.token_positions
            sample_id = len(self.rows)
            state_generation = self.state_generation
        try:
            processes = self.process_sampler(expected_pids)
        except BaseException as error:
            if expected_pids == sorted((self.parent_pid, self.child_pid)) and tuple(getattr(error, "expected_pids", ())) == tuple(expected_pids) and tuple(getattr(error, "observed_pids", ())) == (self.parent_pid,):
                return False
            raise
        if not isinstance(processes, list) or len(processes) != len(expected_pids):
            raise MlxResourceSamplerError("resource sampler process cardinality differs")
        expected_keys = ("cpu_time_us", "pid", "ppid", "rss_bytes")
        if [process.get("pid") for process in processes] != expected_pids:
            raise MlxResourceSamplerError("resource sampler process identity differs")
        for process in processes:
            if tuple(sorted(process)) != expected_keys or any(type(process[field]) is not int or process[field] < 0 for field in expected_keys):
                raise MlxResourceSamplerError("resource sampler process value differs")
        child = next((process for process in processes if process["pid"] == self.child_pid), None)
        if child is not None and child["rss_bytes"] == 0 and child["cpu_time_us"] == 0:
            return False
        swap = self.swap_sampler()
        if type(swap) is not int or swap < 0:
            raise MlxResourceSamplerError("resource sampler swap value differs")
        if self.swap_baseline is None:
            self.swap_baseline = swap
        row = {
                "schema_version": CPU_SCHEMA_VERSION,
                "run_id": self.run_id,
                "sample_id": sample_id,
                "phase": self.phase,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "expected_pids": expected_pids,
                "processes": [dict(process) for process in processes],
                "active_jobs": active_jobs,
                "aggregate_rss_bytes": sum(process["rss_bytes"] for process in processes),
                "aggregate_cpu_time_us": sum(process["cpu_time_us"] for process in processes),
                "swap_used_bytes": swap,
                "swap_growth_bytes": max(0, swap - self.swap_baseline),
                "parser_status": "pass",
                "attempted_updates": attempted_updates,
                "token_positions": token_positions,
            }
        if deadline_ns is not None and row["monotonic_ns"] >= deadline_ns:
            raise MlxDeadlineExceeded("terminal resource sample exceeded the qualification deadline")
        with self.condition:
            if (not final_sample and self.stop_event.is_set()) or state_generation != self.state_generation:
                return False
            if sample_id != len(self.rows):
                raise MlxResourceSamplerError("resource sample sequence differs")
            if self.writer is not None:
                result = self.writer.append(row)
                if not result.acknowledged:
                    raise MlxResourceSamplerError(result.reason_code or "resource writer append differs")
            self.rows.append(row)
            self.condition.notify_all()
        return True

    def _run(self) -> None:
        try:
            if self.rows:
                remaining = self.interval_seconds - (time.monotonic_ns() - self.rows[-1]["monotonic_ns"]) / 1_000_000_000
                if remaining > 0 and self.stop_event.wait(remaining):
                    return
            self._sample()
            while not self.stop_event.wait(self.interval_seconds):
                self._sample()
        except BaseException as error:
            self.error = error
            self.stop_event.set()

    def start(self) -> None:
        if self.thread is not None:
            raise MlxResourceSamplerError("resource sampler is already started")
        self.thread = threading.Thread(target=self._run, name="modular-parent-resource", daemon=False)
        self.thread.start()

    def stop(self, final_sample: bool = False, deadline_ns: int | None = None) -> list[dict[str, Any]]:
        if self.thread is None:
            raise MlxResourceSamplerError("resource sampler is not started")
        if deadline_ns is not None:
            if not final_sample:
                raise MlxDeadlineExceeded("terminal resource deadline requires final sampling")
            enforce_deadline(deadline_ns, time.monotonic_ns)
        with self.condition:
            self.stop_event.set()
        self.thread.join()
        self.raise_if_failed()
        if deadline_ns is not None:
            enforce_deadline(deadline_ns, time.monotonic_ns)
        if final_sample:
            with self.condition:
                if self.active_jobs or self.attempted_updates != self.final_attempted_updates or self.token_positions != self.final_token_positions:
                    raise MlxResourceSamplerError("final resource state differs")
                expected_pids = [self.parent_pid] if self.child_exited else sorted((self.parent_pid, self.child_pid))
                durable = bool(self.rows) and not self.rows[-1].get("active_jobs") and self.rows[-1].get("attempted_updates") == self.attempted_updates and self.rows[-1].get("token_positions") == self.token_positions and ("expected_pids" not in self.rows[-1] or self.rows[-1].get("expected_pids") == expected_pids)
                last_monotonic_ns = self.rows[-1].get("monotonic_ns", 0) if self.rows else 0
            if not durable:
                sample_not_before_ns = last_monotonic_ns + int(self.interval_seconds * 1_000_000_000)
                if deadline_ns is not None and sample_not_before_ns >= deadline_ns:
                    raise MlxDeadlineExceeded("terminal resource sample cannot occur before the qualification deadline")
                remaining_seconds = (sample_not_before_ns - time.monotonic_ns()) / 1_000_000_000
                if remaining_seconds > 0:
                    threading.Event().wait(remaining_seconds)
                if deadline_ns is not None:
                    enforce_deadline(deadline_ns, time.monotonic_ns)
                if not self._sample(final_sample=True, deadline_ns=deadline_ns):
                    raise MlxResourceSamplerError("final resource state is not durably sampled")
                self.raise_if_failed()
                if deadline_ns is not None:
                    enforce_deadline(deadline_ns, time.monotonic_ns)
            with self.condition:
                if not self.rows or self.rows[-1].get("active_jobs") or self.rows[-1].get("attempted_updates") != self.attempted_updates or self.rows[-1].get("token_positions") != self.token_positions or ("expected_pids" in self.rows[-1] and self.rows[-1].get("expected_pids") != expected_pids):
                    raise MlxResourceSamplerError("final resource state is not durably sampled")
        if not self.rows:
            raise MlxResourceSamplerError("resource sampler produced no samples")
        return [dict(row) for row in self.rows]


def qualification_cardinality_contract() -> dict[str, int]:
    return {
        "attempt_rows": 41_472,
        "checkpoint_files": 26,
        "data_files": 16,
        "rung_one_evaluation_rows": 325,
        "rung_one_intervention_rows": 540,
        "rung_one_prediction_rows": 30_720,
        "rung_one_state_rows": 9_900,
        "rung_two_evaluation_rows": 2,
        "rung_two_gate_conditions": 2,
        "rung_two_intervention_rows": 14,
        "rung_two_prediction_rows": 1_024,
        "rung_two_state_rows": 230,
        "routing_evaluation_rows": 66_000,
        "routing_rows": ROUTING_EVIDENCE_ROWS,
        "routing_training_rows": 522_240,
    }


def durable_gzip_prefix(stream: Any) -> dict[str, Any]:
    compressed = getattr(stream, "compressed", None)
    raw = getattr(stream, "raw", None)
    path = getattr(stream, "path", None)
    if compressed is None or raw is None or not isinstance(path, Path):
        raise MlxQualificationError("routing gzip stream is not open")
    compressed.flush()
    raw.flush()
    os.fsync(raw.fileno())
    committed_bytes = raw.tell()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        observed = b""
        while len(observed) < committed_bytes:
            chunk = os.read(descriptor, committed_bytes - len(observed))
            if not chunk:
                break
            observed += chunk
    finally:
        os.close(descriptor)
    if len(observed) != committed_bytes:
        raise MlxQualificationError("routing gzip durable prefix differs")
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "committed_bytes": committed_bytes, "sha256": hashlib.sha256(observed).hexdigest()}


def validate_closed_training_gzip(path_value: str | Path, run_id: str, seed: int, stage: str) -> dict[str, Any]:
    path = Path(path_value)
    if stage not in TRAINING_ROUTE_ROWS_PER_SEED or not path.is_file() or path.is_symlink() or not isinstance(run_id, str) or not run_id or seed not in RUNG_ONE_SEEDS:
        raise MlxQualificationError("closed training routing identity differs")
    count = 0
    prior_sequence = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except BaseException as error:
                raise MlxQualificationError("closed training routing JSON differs") from error
            if not isinstance(row, Mapping) or tuple(sorted(row)) != ROUTING_ROW_KEYS:
                raise MlxQualificationError("closed training routing schema differs")
            if row["run_id"] != run_id or row["construction_seed"] != seed or row["claim_seed"] != seed or row["phase"] != "training" or row["stage"] != stage or row["rung"] != 1:
                raise MlxQualificationError("closed training routing row identity differs")
            sequence = row["forward_sequence"]
            if type(sequence) is not int or sequence < 0 or (prior_sequence is not None and sequence < prior_sequence):
                raise MlxQualificationError("closed training routing sequence differs")
            prior_sequence = sequence
            count += 1
    expected = TRAINING_ROUTE_ROWS_PER_SEED[stage]
    if count != expected:
        raise MlxQualificationError("closed training routing cardinality differs")
    raw = path.read_bytes()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "rows": count, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def backend_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mlx_version": MLX_VERSION,
        "python_path": str(MLX_PYTHON),
        "engine_path": str(ENGINE_PATH),
        "positions": POSITIONS,
        "target_seconds": TARGET_SECONDS,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "rung_one_seeds": list(RUNG_ONE_SEEDS),
        "rung_two_seed": RUNG_TWO_SEED,
        "sequential_stages": ["donor", "rung_two"],
        "vectorized_stages": ["router_only", "joint", "dense_base", "dense_continuation"],
        "vectorized_width": 5,
        "torch_reference_authority": True,
        "resume_claim_allowed": False,
        "stage_contracts": {name: dict(value) for name, value in STAGE_CONTRACTS.items()},
    }


def dependency_hashes() -> dict[str, str]:
    paths = {
        "backend": BACKEND_PATH,
        "cpu_evaluator": CPU_EVALUATOR_PATH,
        "engine": ENGINE_PATH,
        "model": MODEL_PATH,
        "qualifier": QUALIFIER_PATH,
        "sources": SOURCES_PATH,
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise MlxQualificationError("MLX dependency path differs")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def evaluation_contract() -> dict[str, Any]:
    return {
        "rung_one_conditions": [
            "intact",
            "target_forced",
            "recurrent_knockout",
            "carry_reset",
            "carry_shuffle",
            "matched_random_route",
            "block4_routed_knockout",
            "block4_local_only",
            "required_source_excluded",
            "all_eligible_donor",
            "all_eligible_clone",
            "dense_causal",
        ],
        "rung_two_conditions": ["intact", "recurrent_knockout"],
        "rung_one_examples_per_seed": 512,
        "rung_two_examples": 512,
        "evaluation_batch_size": 32,
        "routing_evidence_rows": ROUTING_EVIDENCE_ROWS,
        "routing_compression": {"format": "gzip", "level": 9, "mtime": 0},
        "artifacts": ["evaluation", "predictions", "state_statistics", "intervention_deltas", "routing_evidence"],
        "torch_authoritative_gate_replay": True,
    }


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MlxBackendRefusal(f"measured component {name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise MlxBackendRefusal(f"measured component {name} is invalid")
    return result


def project_full_package(measured: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "donor_step_seconds",
        "selected_vmap5_step_seconds",
        "dense_vmap5_step_seconds",
        "rung_two_step_seconds",
        "cold_child_start_seconds",
        "cold_compile_seconds",
        "durable_ledger_seconds",
        "routing_evidence_seconds",
        "evaluation_seconds",
        "checkpoint_reload_seconds",
        "packaging_seconds",
        "resource_finalization_seconds",
        "lifecycle_close_join_seconds",
    )
    if tuple(sorted(measured)) != tuple(sorted(required)):
        raise MlxBackendRefusal("measured component keys differ")
    values = {name: _finite_nonnegative(measured[name], name) for name in required}
    terms = {
        "donor_seconds": 5 * 1024 * values["donor_step_seconds"],
        "selected_seconds": 1280 * values["selected_vmap5_step_seconds"],
        "dense_seconds": 1536 * values["dense_vmap5_step_seconds"],
        "rung_two_seconds": 1536 * values["rung_two_step_seconds"],
        "cold_child_start_seconds": values["cold_child_start_seconds"],
        "cold_compile_seconds": values["cold_compile_seconds"],
        "durable_ledger_seconds": values["durable_ledger_seconds"],
        "routing_evidence_seconds": values["routing_evidence_seconds"],
        "evaluation_seconds": values["evaluation_seconds"],
        "checkpoint_reload_seconds": values["checkpoint_reload_seconds"],
        "packaging_seconds": values["packaging_seconds"],
        "resource_finalization_seconds": values["resource_finalization_seconds"],
        "lifecycle_close_join_seconds": values["lifecycle_close_join_seconds"],
    }
    projected = sum(terms.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "positions": POSITIONS,
        "projected_seconds": projected,
        "target_seconds": TARGET_SECONDS,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "target_pass": projected <= TARGET_SECONDS,
        "hard_limit_pass": projected <= HARD_LIMIT_SECONDS,
        "attempt_event_rows": ATTEMPT_EVENT_ROWS,
        "routing_evidence_rows": ROUTING_EVIDENCE_ROWS,
        "components": terms,
    }


def child_invocation(mode: str) -> tuple[list[str], dict[str, str]]:
    if mode not in {"describe", "self-check", "pilot", "serve"}:
        raise MlxBackendRefusal("MLX child mode differs")
    command = [str(MLX_PYTHON), str(QUALIFIER_PATH), "--child-mode", mode]
    environment = {
        "HOME": "/Users/dttdrv",
        "LANG": "C",
        "LC_ALL": "C",
        "LOGNAME": "dttdrv",
        "MLX_METAL_DEBUG": "0",
        "OMP_NUM_THREADS": "4",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONPYCACHEPREFIX": "/private/tmp/todorov-mlx-pycache",
        "TMPDIR": "/private/tmp",
        "USER": "dttdrv",
        "VECLIB_MAXIMUM_THREADS": "4",
    }
    return command, environment


def protocol_state() -> dict[str, Any]:
    return {"next_sequence": 0, "hello": False, "pending_request": None, "active_stage": None, "pending_update": None, "completed_updates": 0, "closed": False, "aborted": False, "evaluated": False}


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], context: str) -> None:
    if tuple(sorted(value)) != tuple(sorted(expected)):
        raise MlxProtocolError(f"{context} keys differ")


def _sha256_text(value: Any, context: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MlxProtocolError(f"{context} SHA-256 differs")


def _initial_forward_record(value: Any, identity: Mapping[str, Any]) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise MlxQualificationError("initial forward parity record differs")
    numeric = ("logits_max_abs", "hidden_max_abs", "router_loss_max_abs", "forward_relative_max", "forward_normalized_l2_max", "forward_cosine_min")
    if any(isinstance(value.get(name), bool) or not isinstance(value.get(name), (int, float)) or not math.isfinite(float(value[name])) for name in numeric):
        raise MlxQualificationError("initial forward parity numeric evidence differs")
    if any(value[name] < 0 for name in numeric[:-1]) or not -1 <= value["forward_cosine_min"] <= 1:
        raise MlxQualificationError("initial forward parity range differs")
    sequence = value.get("sequence_delta_max_abs_by_block")
    feature = value.get("feature_delta_max_abs_by_block")
    if any(not isinstance(records, list) or len(records) != 8 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or item < 0 for item in records) for records in (sequence, feature)):
        raise MlxQualificationError("initial forward block evidence differs")
    maximum = max(value["logits_max_abs"], value["hidden_max_abs"], max(sequence), max(feature))
    absolute_pass = maximum <= 1e-5
    scale_pass = maximum <= 5e-5 and value["forward_relative_max"] <= 5e-5 and value["forward_normalized_l2_max"] <= 5e-6 and value["forward_cosine_min"] >= 0.99999999999
    forward_pass = absolute_pass or scale_pass
    tensor_names = {"logits", "final_hidden", *(f"sequence_delta_{index}" for index in range(8)), *(f"feature_delta_{index}" for index in range(8))}
    worst_identity = isinstance(value.get("forward_worst_tensor"), str) and value["forward_worst_tensor"] in tensor_names
    worst_identity = worst_identity and isinstance(value.get("forward_worst_index"), list) and all(type(index) is int and index >= 0 for index in value["forward_worst_index"])
    worst_identity = worst_identity and all(isinstance(value.get(name), (int, float)) and not isinstance(value.get(name), bool) and math.isfinite(float(value[name])) for name in ("forward_worst_observed", "forward_worst_expected"))
    tolerance_identity = value.get("forward_scale_aware_absolute_tolerance") == 5e-5 and value.get("forward_relative_tolerance") == 5e-5 and value.get("forward_normalized_l2_tolerance") == 5e-6 and value.get("forward_cosine_tolerance") == 0.99999999999
    decision_identity = value.get("forward_absolute_pass") is absolute_pass and value.get("forward_scale_aware_pass") is scale_pass and value.get("forward_pass") is forward_pass
    mapping_identity = value.get("mapping_bijective") is True and value.get("mapping_transpose") is False and value.get("mapping_value_byte_exact") is True and value.get("mapping_value_max_abs") == 0.0
    mapping_identity = mapping_identity and value.get("mapped_parameter_count") == identity["parameter_count"] and value.get("mapping_value_count") == identity["parameter_count"]
    for name in ("mapping_sha256", "mapping_source_value_sha256", "mapping_destination_value_sha256"):
        digest = value.get(name)
        mapping_identity = mapping_identity and isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    mapping_identity = mapping_identity and value.get("mapping_source_value_sha256") == value.get("mapping_destination_value_sha256")
    exact_identity = all(value.get(name) == expected for name, expected in identity.items() if name not in {"parameter_count", "route_count"})
    exact_identity = exact_identity and value.get("block_count") == 8 and value.get("raw_route_count") == identity["route_count"] and value.get("effective_route_count") == identity["route_count"]
    passed = forward_pass and value["router_loss_max_abs"] <= 1e-6 and value.get("route_exact") is True and worst_identity and tolerance_identity and decision_identity and mapping_identity and exact_identity
    if not passed or value.get("pass") is not True:
        raise MlxQualificationError("initial forward parity differs")
    primary_ratio = maximum / 1e-5
    fallback_ratio = max(maximum / 5e-5, value["forward_relative_max"] / 5e-5, value["forward_normalized_l2_max"] / 5e-6, max(0.0, (1.0 - value["forward_cosine_min"]) / (1.0 - 0.99999999999)))
    return min(primary_ratio, fallback_ratio), value["router_loss_max_abs"] / 1e-6


def _validate_metric_worst(value: Any, maximums: Mapping[str, float], context: str) -> None:
    metrics = ("max_abs", "relative_max", "normalized_l2", "cosine")
    fields = (
        "tensor",
        "worst_index",
        "worst_observed",
        "worst_expected",
        "expected_max_magnitude",
        "expected_l2",
        "observed_l2",
        "difference_l2",
        "cosine_denominator",
        "gradient_floor",
        "mismatch_count",
        "sign_flip_count",
        "value",
    )
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(sorted(metrics)):
        raise MlxQualificationError(f"{context} metric-worst surface differs")
    for metric in metrics:
        record = value[metric]
        if not isinstance(record, Mapping) or tuple(sorted(record)) != tuple(sorted(fields)):
            raise MlxQualificationError(f"{context} metric-worst record differs")
        numeric = ("worst_observed", "worst_expected", "expected_max_magnitude", "expected_l2", "observed_l2", "difference_l2", "cosine_denominator", "gradient_floor", "value")
        if any(isinstance(record.get(name), bool) or not isinstance(record.get(name), (int, float)) or not math.isfinite(float(record[name])) for name in numeric):
            raise MlxQualificationError(f"{context} metric-worst numeric evidence differs")
        if not isinstance(record.get("tensor"), str) or not record["tensor"] or not isinstance(record.get("worst_index"), list) or any(type(index) is not int or index < 0 for index in record["worst_index"]):
            raise MlxQualificationError(f"{context} metric-worst identity differs")
        if any(record[name] < 0 for name in ("expected_max_magnitude", "expected_l2", "observed_l2", "difference_l2", "cosine_denominator")) or record["gradient_floor"] != 1e-8:
            raise MlxQualificationError(f"{context} metric-worst range differs")
        if type(record.get("mismatch_count")) is not int or record["mismatch_count"] < 0 or type(record.get("sign_flip_count")) is not int or record["sign_flip_count"] < 0 or record["sign_flip_count"] > record["mismatch_count"]:
            raise MlxQualificationError(f"{context} metric-worst count differs")
        if record["value"] != maximums[metric] or (metric == "cosine" and not -1 <= record["value"] <= 1) or (metric != "cosine" and record["value"] < 0):
            raise MlxQualificationError(f"{context} metric-worst reduction differs")
        maximum_difference = abs(record["worst_observed"] - record["worst_expected"])
        if not math.isclose(record["cosine_denominator"], record["expected_l2"] * record["observed_l2"], rel_tol=1e-9, abs_tol=1e-30):
            raise MlxQualificationError(f"{context} metric-worst denominator differs")
        if metric == "max_abs" and not math.isclose(record["value"], maximum_difference, rel_tol=1e-9, abs_tol=1e-30):
            raise MlxQualificationError(f"{context} metric-worst absolute value differs")
        if metric == "relative_max" and not math.isclose(record["value"], maximum_difference / max(record["expected_max_magnitude"], 1e-30), rel_tol=1e-9, abs_tol=1e-30):
            raise MlxQualificationError(f"{context} metric-worst relative value differs")
        if metric == "normalized_l2" and not math.isclose(record["value"], record["difference_l2"] / max(record["expected_l2"], 1e-30), rel_tol=1e-9, abs_tol=1e-30):
            raise MlxQualificationError(f"{context} metric-worst normalized value differs")


def _validate_carried_adamw(value: Any) -> float:
    if not isinstance(value, Mapping):
        raise MlxQualificationError("initial carried AdamW surface differs")
    identity = {
        "lanes": 5,
        "tensor_count": 2,
        "first_update": 1,
        "tested_update": 2,
        "bias_correction": True,
        "nonzero_carried_first_and_second_moments": True,
        "distinct_second_gradient": True,
        "gradient_clip_identity": True,
        "formula_unit_roundoff": 2.0**-24,
        "formula_parameter_operation_budget": 32,
        "formula_first_moment_operation_budget": 6,
        "formula_second_moment_operation_budget": 8,
    }
    if any(value.get(name) != expected for name, expected in identity.items()):
        raise MlxQualificationError("initial carried AdamW identity differs")
    digest = value.get("canonical_gradient_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MlxQualificationError("initial carried AdamW gradient hash differs")
    numeric = ("max_bound_ratio", "worst_abs", "worst_bound", "cross_runtime_max_abs")
    if any(isinstance(value.get(name), bool) or not isinstance(value.get(name), (int, float)) or not math.isfinite(float(value[name])) or value[name] < 0 for name in numeric):
        raise MlxQualificationError("initial carried AdamW numeric evidence differs")
    if value["max_bound_ratio"] > 1.0 or value["worst_bound"] <= 0 or value["max_bound_ratio"] != value["worst_abs"] / value["worst_bound"]:
        raise MlxQualificationError("initial carried AdamW formula bound differs")
    if value.get("worst_runtime") not in {"MLX", "Torch"} or value.get("worst_surface") not in {"parameter", "first_moment", "second_moment"} or value.get("worst_tensor") not in {"decayed", "nondecayed"}:
        raise MlxQualificationError("initial carried AdamW worst identity differs")
    if type(value.get("worst_lane")) is not int or not 0 <= value["worst_lane"] < 5 or not isinstance(value.get("worst_index"), list) or any(type(index) is not int or index < 0 for index in value["worst_index"]):
        raise MlxQualificationError("initial carried AdamW worst location differs")
    if value.get("pass") is not True:
        raise MlxQualificationError("initial carried AdamW decision differs")
    return value["max_bound_ratio"]


def validate_initial_self_check(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("device") != "Device(gpu, 0)":
        raise MlxQualificationError("initial self-check identity differs")
    expected_keys = (
        "schema_version",
        "mlx_version",
        "device",
        "contract",
        "full_model_parity",
        "all_role_forward_calibration",
        "held_out_forward_admission",
        "full_gradient_parity",
        "adamw_parity",
        "carried_adamw_parity",
        "vmap5",
        "functional_forward",
        "actual_model_vmap5",
        "memory",
        "pass",
    )
    if tuple(sorted(value)) != tuple(sorted(expected_keys)) or value.get("schema_version") != IPC_SCHEMA_VERSION or value.get("mlx_version") != MLX_VERSION or value.get("contract") != backend_contract():
        raise MlxQualificationError("initial self-check contract differs")
    required = ("full_model_parity", "all_role_forward_calibration", "held_out_forward_admission", "full_gradient_parity", "adamw_parity", "carried_adamw_parity", "vmap5", "functional_forward", "actual_model_vmap5", "memory")
    if any(not isinstance(value.get(name), Mapping) for name in required):
        raise MlxQualificationError("initial self-check surface differs")
    selected_identity = {"role": "selected", "stage": "joint", "objective": "task_plus_0.1_times_internal_router_plus_supervised_route", "model_seed": 3123, "data_seed": 4123, "batch_size": 2, "sequence_length": 128, "parameter_count": 119, "route_count": 2}
    selected_forward_ratio, selected_router_ratio = _initial_forward_record(value["full_model_parity"], selected_identity)
    calibration = value["all_role_forward_calibration"]
    identities = (
        selected_identity,
        {"role": "all_eligible", "stage": "forward_calibration", "objective": "full_forward_output_surface", "model_seed": 3124, "data_seed": 4124, "batch_size": 2, "sequence_length": 128, "parameter_count": 119, "route_count": 2},
        {"role": "dense", "stage": "forward_calibration", "objective": "full_forward_output_surface", "model_seed": 3125, "data_seed": 4125, "batch_size": 2, "sequence_length": 128, "parameter_count": 116, "route_count": 1},
        {"role": "rung_two", "stage": "forward_calibration", "objective": "full_forward_output_surface", "model_seed": 3126, "data_seed": 4126, "batch_size": 2, "sequence_length": 512, "parameter_count": 119, "route_count": 2},
    )
    if not isinstance(calibration, Mapping) or calibration.get("roles") != ["selected", "all_eligible", "dense", "rung_two"] or calibration.get("fresh_process_required") is not True:
        raise MlxQualificationError("initial forward calibration identity differs")
    records = calibration.get("records")
    if not isinstance(records, list) or len(records) != 4 or records[0] != value["full_model_parity"]:
        raise MlxQualificationError("initial forward calibration records differ")
    forward_ratios = []
    router_ratios = []
    for record, identity in zip(records, identities):
        forward_ratio, router_ratio = _initial_forward_record(record, identity)
        forward_ratios.append(forward_ratio)
        router_ratios.append(router_ratio)
    if calibration.get("pass") is not True:
        raise MlxQualificationError("initial forward calibration decision differs")
    held_out = value["held_out_forward_admission"]
    held_out_identities = (
        {"role": "selected", "stage": "joint", "objective": "task_plus_0.1_times_internal_router_plus_supervised_route", "model_seed": 8123, "data_seed": 9123, "batch_size": 2, "sequence_length": 128, "parameter_count": 119, "route_count": 2},
        {"role": "all_eligible", "stage": "forward_calibration", "objective": "full_forward_output_surface", "model_seed": 8124, "data_seed": 9124, "batch_size": 3, "sequence_length": 128, "parameter_count": 119, "route_count": 2},
    )
    held_out_records = held_out.get("records")
    if held_out.get("thresholds_frozen_before_execution") is not True or not isinstance(held_out_records, list) or len(held_out_records) != 2:
        raise MlxQualificationError("initial held-out forward surface differs")
    for record, identity in zip(held_out_records, held_out_identities):
        forward_ratio, router_ratio = _initial_forward_record(record, identity)
        forward_ratios.append(forward_ratio)
        router_ratios.append(router_ratio)
    if held_out.get("pass") is not True:
        raise MlxQualificationError("initial held-out forward decision differs")
    gradient = value["full_gradient_parity"]
    gradient_numeric = ("loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "gradient_relative_max", "gradient_normalized_l2_max", "gradient_cosine_min")
    if any(isinstance(gradient.get(name), bool) or not isinstance(gradient.get(name), (int, float)) or not math.isfinite(float(gradient[name])) for name in gradient_numeric):
        raise MlxQualificationError("initial gradient parity numeric evidence differs")
    if any(gradient[name] < 0 for name in gradient_numeric[:-1]) or not -1 <= gradient["gradient_cosine_min"] <= 1:
        raise MlxQualificationError("initial gradient parity range differs")
    component_errors = gradient.get("component_loss_errors")
    expected_component_keys = ("task_loss", "internal_router_loss", "supervised_route_loss")
    if not isinstance(component_errors, Mapping) or tuple(sorted(component_errors)) != tuple(sorted(expected_component_keys)) or any(isinstance(component_errors[name], bool) or not isinstance(component_errors[name], (int, float)) or not math.isfinite(float(component_errors[name])) or component_errors[name] < 0 for name in expected_component_keys):
        raise MlxQualificationError("initial gradient component-loss parity differs")
    if gradient["component_loss_max_abs"] != max(component_errors.values()):
        raise MlxQualificationError("initial gradient component-loss reduction differs")
    gradient_tolerances = {
        "gradient_scale_aware_absolute_tolerance": 3e-5,
        "gradient_relative_tolerance": 1e-4,
        "gradient_normalized_l2_tolerance": 5e-5,
        "gradient_cosine_tolerance": 0.999999999,
    }
    if any(gradient.get(name) != expected for name, expected in gradient_tolerances.items()):
        raise MlxQualificationError("initial gradient parity tolerance differs")
    absolute_pass = gradient["gradient_max_abs"] <= 1e-5
    scale_pass = gradient["gradient_max_abs"] <= gradient["gradient_scale_aware_absolute_tolerance"] and gradient["gradient_relative_max"] <= gradient["gradient_relative_tolerance"] and gradient["gradient_normalized_l2_max"] <= gradient["gradient_normalized_l2_tolerance"] and gradient["gradient_cosine_min"] >= gradient["gradient_cosine_tolerance"]
    gradient_pass = absolute_pass or scale_pass
    if gradient.get("role") != "selected" or gradient.get("model_seed") != 3123 or gradient.get("data_seed") != 4123 or gradient.get("batch_size") != 2 or gradient.get("gradient_count") != 116:
        raise MlxQualificationError("initial gradient parity identity differs")
    if max(gradient["loss_max_abs"], gradient["component_loss_max_abs"]) > 1e-6 or gradient.get("gradient_absolute_pass") is not absolute_pass or gradient.get("gradient_scale_aware_pass") is not scale_pass or gradient.get("gradient_pass") is not gradient_pass or gradient.get("grad_none_zero_exact") is not True or gradient.get("pass") is not True or not gradient_pass:
        raise MlxQualificationError("initial gradient parity differs")
    adam = value["adamw_parity"]
    if adam.get("tolerance") != 1e-7 or isinstance(adam.get("max_abs"), bool) or not isinstance(adam.get("max_abs"), (int, float)) or not math.isfinite(float(adam["max_abs"])) or not 0 <= adam["max_abs"] <= 1e-7 or adam.get("pass") is not True:
        raise MlxQualificationError("initial AdamW parity differs")
    carried_adam_ratio = _validate_carried_adamw(value["carried_adamw_parity"])
    vectorized = value["vmap5"]
    if vectorized.get("lanes") != 5 or vectorized.get("unique_parameter_hashes") != 5 or vectorized.get("lane_local_clipping") is not True or vectorized.get("pure_parameter_tree") is not True or vectorized.get("pass") is not True:
        raise MlxQualificationError("initial vectorization parity differs")
    functional = value["functional_forward"]
    functional_numeric = ("logits_max_abs", "query_route_max_abs", "key_route_max_abs", "router_loss_max_abs")
    if any(functional.get(name) != 0.0 for name in functional_numeric) or functional.get("route_exact") is not True or functional.get("pass") is not True:
        raise MlxQualificationError("initial functional parity differs")
    memory = value["memory"]
    if tuple(sorted(memory)) != ("active_memory_bytes", "cache_memory_bytes", "parent_rss_and_swap_required", "peak_memory_bytes") or memory.get("parent_rss_and_swap_required") is not True:
        raise MlxQualificationError("initial memory evidence differs")
    for name in ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes"):
        if type(memory.get(name)) is not int or memory[name] < 0:
            raise MlxQualificationError("initial memory numeric evidence differs")
    actual = value["actual_model_vmap5"]
    actual_numeric = (
        "torch_loss_max_abs",
        "torch_parameter_max_abs",
        "torch_first_moment_max_abs",
        "torch_second_moment_max_abs",
        "five_lane_gradient_max_abs",
        "five_lane_gradient_relative_max",
        "five_lane_gradient_normalized_l2_max",
        "five_lane_gradient_cosine_min",
        "five_lane_clipped_gradient_max_abs",
        "five_lane_clipped_gradient_relative_max",
        "five_lane_clipped_gradient_normalized_l2_max",
        "five_lane_clipped_gradient_cosine_min",
        "parameter_update_max_abs",
        "parameter_update_relative_max",
        "parameter_update_normalized_l2_max",
        "parameter_update_cosine_min",
        "mlx_optimizer_parameter_formula_max_abs",
        "mlx_optimizer_first_formula_max_abs",
        "mlx_optimizer_second_formula_max_abs",
        "torch_optimizer_parameter_formula_max_abs",
        "torch_optimizer_first_formula_max_abs",
        "torch_optimizer_second_formula_max_abs",
        "optimizer_formula_max_bound_ratio",
        "causal_parameter_residual_max_abs",
        "causal_first_moment_residual_max_abs",
        "causal_second_moment_residual_max_abs",
        "causal_residual_worst_bound_ratio",
        "end_to_end_worst_max_abs",
    )
    if any(isinstance(actual.get(name), bool) or not isinstance(actual.get(name), (int, float)) or not math.isfinite(float(actual[name])) or actual[name] < 0 for name in actual_numeric):
        raise MlxQualificationError("initial five-lane parity numeric evidence differs")
    actual_gradient_absolute_pass = actual["five_lane_gradient_max_abs"] <= 1e-5
    actual_gradient_scale_pass = actual["five_lane_gradient_max_abs"] <= 1.25e-4 and actual["five_lane_gradient_relative_max"] <= 2.5e-4 and actual["five_lane_gradient_normalized_l2_max"] <= 1.25e-4 and actual["five_lane_gradient_cosine_min"] >= 0.99999999
    actual_gradient_pass = actual_gradient_absolute_pass or actual_gradient_scale_pass
    clipped_gradient_absolute_pass = actual["five_lane_clipped_gradient_max_abs"] <= 1e-5
    clipped_gradient_scale_pass = actual["five_lane_clipped_gradient_max_abs"] <= 1.25e-4 and actual["five_lane_clipped_gradient_relative_max"] <= 2.5e-4 and actual["five_lane_clipped_gradient_normalized_l2_max"] <= 1.25e-4 and actual["five_lane_clipped_gradient_cosine_min"] >= 0.99999999
    clipped_gradient_pass = clipped_gradient_absolute_pass or clipped_gradient_scale_pass
    gradient_tolerances = actual.get("five_lane_gradient_scale_aware_absolute_tolerance") == 1.25e-4 and actual.get("five_lane_gradient_relative_tolerance") == 2.5e-4 and actual.get("five_lane_gradient_normalized_l2_tolerance") == 1.25e-4 and actual.get("five_lane_gradient_cosine_tolerance") == 0.99999999
    _validate_metric_worst(
        actual.get("five_lane_gradient_metric_worst"),
        {
            "max_abs": actual["five_lane_gradient_max_abs"],
            "relative_max": actual["five_lane_gradient_relative_max"],
            "normalized_l2": actual["five_lane_gradient_normalized_l2_max"],
            "cosine": actual["five_lane_gradient_cosine_min"],
        },
        "initial raw gradient",
    )
    _validate_metric_worst(
        actual.get("five_lane_clipped_gradient_metric_worst"),
        {
            "max_abs": actual["five_lane_clipped_gradient_max_abs"],
            "relative_max": actual["five_lane_clipped_gradient_relative_max"],
            "normalized_l2": actual["five_lane_clipped_gradient_normalized_l2_max"],
            "cosine": actual["five_lane_clipped_gradient_cosine_min"],
        },
        "initial clipped gradient",
    )
    _validate_metric_worst(
        actual.get("parameter_update_metric_worst"),
        {
            "max_abs": actual["parameter_update_max_abs"],
            "relative_max": actual["parameter_update_relative_max"],
            "normalized_l2": actual["parameter_update_normalized_l2_max"],
            "cosine": actual["parameter_update_cosine_min"],
        },
        "initial parameter update",
    )
    optimizer_formula_pass = actual["optimizer_formula_max_bound_ratio"] <= 1.0
    optimizer_identity = actual.get("optimizer_formula_bound_ratio_tolerance") == 1.0 and actual.get("optimizer_formula_pass") is optimizer_formula_pass
    optimizer_identity = optimizer_identity and actual.get("optimizer_formula_worst_runtime") in {"MLX", "Torch"} and actual.get("optimizer_formula_worst_surface") in {"parameter", "first_moment", "second_moment"}
    optimizer_identity = optimizer_identity and type(actual.get("optimizer_formula_worst_lane")) is int and 0 <= actual["optimizer_formula_worst_lane"] < 5 and isinstance(actual.get("optimizer_formula_worst_tensor"), str) and actual["optimizer_formula_worst_tensor"]
    optimizer_identity = optimizer_identity and isinstance(actual.get("optimizer_formula_worst_index"), list) and all(type(index) is int and index >= 0 for index in actual["optimizer_formula_worst_index"])
    optimizer_identity = optimizer_identity and all(isinstance(actual.get(name), (int, float)) and not isinstance(actual.get(name), bool) and math.isfinite(float(actual[name])) for name in ("optimizer_formula_worst_abs", "optimizer_formula_worst_bound", "optimizer_formula_worst_bound_ratio", "optimizer_formula_worst_observed", "optimizer_formula_worst_expected"))
    optimizer_identity = optimizer_identity and actual["optimizer_formula_worst_bound"] > 0 and actual["optimizer_formula_worst_bound_ratio"] == actual["optimizer_formula_worst_abs"] / actual["optimizer_formula_worst_bound"]
    optimizer_identity = optimizer_identity and actual["optimizer_formula_max_bound_ratio"] == actual["optimizer_formula_worst_bound_ratio"]
    residual = actual.get("causal_residual_summary")
    if not isinstance(residual, Mapping) or tuple(sorted(residual)) != ("first_moment", "parameter", "second_moment"):
        raise MlxQualificationError("initial causal residual summary differs")
    residual_fields = {"parameter": "causal_parameter_residual_max_abs", "first_moment": "causal_first_moment_residual_max_abs", "second_moment": "causal_second_moment_residual_max_abs"}
    causal_residual_pass = True
    for surface, maximum_field in residual_fields.items():
        record = residual[surface]
        if not isinstance(record, Mapping) or tuple(sorted(record)) != ("max_abs", "max_bound", "max_bound_ratio", "worst_excess"):
            raise MlxQualificationError("initial causal residual record differs")
        if any(isinstance(record.get(name), bool) or not isinstance(record.get(name), (int, float)) or not math.isfinite(float(record[name])) for name in ("max_abs", "max_bound", "max_bound_ratio", "worst_excess")):
            raise MlxQualificationError("initial causal residual numeric differs")
        if record["max_abs"] < 0 or record["max_bound"] <= 0 or record["max_bound_ratio"] < 0 or record["max_bound_ratio"] > 1.0 or record["max_abs"] != actual[maximum_field]:
            raise MlxQualificationError("initial causal residual reduction differs")
        causal_residual_pass = causal_residual_pass and record["worst_excess"] <= 0.0 and record["max_bound_ratio"] <= 1.0
    causal_identity = actual.get("causal_residual_pass") is causal_residual_pass and actual.get("causal_residual_worst_surface") in residual_fields
    causal_identity = causal_identity and type(actual.get("causal_residual_worst_lane")) is int and 0 <= actual["causal_residual_worst_lane"] < 5 and isinstance(actual.get("causal_residual_worst_tensor"), str) and actual["causal_residual_worst_tensor"]
    causal_identity = causal_identity and isinstance(actual.get("causal_residual_worst_index"), list) and all(type(index) is int and index >= 0 for index in actual["causal_residual_worst_index"])
    causal_identity = causal_identity and all(isinstance(actual.get(name), (int, float)) and not isinstance(actual.get(name), bool) and math.isfinite(float(actual[name])) for name in ("causal_residual_worst_abs", "causal_residual_worst_bound", "causal_residual_worst_bound_ratio", "causal_residual_worst_excess"))
    causal_identity = causal_identity and actual["causal_residual_worst_bound"] > 0 and actual["causal_residual_worst_bound_ratio"] <= 1.0
    causal_identity = causal_identity and actual["causal_residual_worst_bound_ratio"] == actual["causal_residual_worst_abs"] / actual["causal_residual_worst_bound"] and actual["causal_residual_worst_excess"] == actual["causal_residual_worst_abs"] - actual["causal_residual_worst_bound"]
    causal_identity = causal_identity and actual["causal_residual_worst_bound_ratio"] == max(record["max_bound_ratio"] for record in residual.values())
    end_to_end_identity = type(actual.get("end_to_end_worst_lane")) is int and 0 <= actual["end_to_end_worst_lane"] < 5 and isinstance(actual.get("end_to_end_worst_tensor"), str) and actual["end_to_end_worst_tensor"]
    end_to_end_identity = end_to_end_identity and isinstance(actual.get("end_to_end_worst_index"), list) and all(type(index) is int and index >= 0 for index in actual["end_to_end_worst_index"])
    end_to_end_identity = end_to_end_identity and all(isinstance(actual.get(name), (int, float)) and not isinstance(actual.get(name), bool) and math.isfinite(float(actual[name])) for name in ("end_to_end_worst_observed", "end_to_end_worst_expected", "end_to_end_worst_mlx_clipped_gradient", "end_to_end_worst_torch_clipped_gradient"))
    gradient_identity = actual.get("five_lane_gradient_count") == 580 and actual.get("five_lane_clipped_gradient_count") == 580 and isinstance(actual.get("five_lane_gradient_worst_tensor"), str) and actual["five_lane_gradient_worst_tensor"]
    gradient_identity = gradient_identity and isinstance(actual.get("five_lane_gradient_worst_index"), list) and all(type(index) is int and index >= 0 for index in actual["five_lane_gradient_worst_index"])
    gradient_identity = gradient_identity and all(isinstance(actual.get(name), (int, float)) and not isinstance(actual.get(name), bool) and math.isfinite(float(actual[name])) for name in ("five_lane_gradient_worst_observed", "five_lane_gradient_worst_expected"))
    raw_worst = actual["five_lane_gradient_metric_worst"]["max_abs"]
    gradient_identity = gradient_identity and actual["five_lane_gradient_worst_tensor"] == raw_worst["tensor"] and actual["five_lane_gradient_worst_index"] == raw_worst["worst_index"]
    gradient_identity = gradient_identity and actual["five_lane_gradient_worst_observed"] == raw_worst["worst_observed"] and actual["five_lane_gradient_worst_expected"] == raw_worst["worst_expected"]
    digest_identity = True
    for name in ("five_lane_raw_gradient_sha256", "optimizer_gradient_sha256", "torch_optimizer_gradient_sha256", "mapping_source_value_sha256", "mapping_destination_value_sha256"):
        digest = actual.get(name)
        digest_identity = digest_identity and isinstance(digest, str) and len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    norm_identity = True
    for name in ("mlx_preclip_gradient_norms", "mlx_postclip_gradient_norms", "torch_preclip_gradient_norms", "torch_postclip_gradient_norms"):
        norms = actual.get(name)
        norm_identity = norm_identity and isinstance(norms, list) and len(norms) == 5 and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) and item > 0 for item in norms)
    if norm_identity:
        norm_identity = all(value <= 1.000001 for name in ("mlx_postclip_gradient_norms", "torch_postclip_gradient_norms") for value in actual[name])
        norm_identity = norm_identity and all(post <= pre + 1e-12 for prefix in ("mlx", "torch") for pre, post in zip(actual[f"{prefix}_preclip_gradient_norms"], actual[f"{prefix}_postclip_gradient_norms"]))
    runtime_identity = actual.get("stage") == "joint" and actual.get("objective") == "task_plus_0.1_times_internal_router_plus_supervised_route" and actual.get("batch_size_per_lane") == 2 and actual.get("sequence_length") == 128 and actual.get("logical_update") == 1
    runtime_identity = runtime_identity and actual.get("learning_rates") == {"block_4_router": 0.001, "other_trainable": 0.00025} and actual.get("initial_optimizer_step") == 0 and actual.get("initial_first_and_second_moments_exact_zero") is True
    mapping_identity = actual.get("mapping_value_count") == 595 and actual.get("mapping_value_max_abs") == 0.0 and actual.get("mapping_value_byte_exact") is True and actual.get("mapping_source_value_sha256") == actual.get("mapping_destination_value_sha256")
    update_identity = math.isclose(
        actual["parameter_update_max_abs"],
        actual["torch_parameter_max_abs"],
        rel_tol=PARAMETER_UPDATE_RELATIVE_TOLERANCE,
        abs_tol=PARAMETER_UPDATE_ABSOLUTE_TOLERANCE,
    )
    update_identity = update_identity and actual["end_to_end_worst_max_abs"] == actual["torch_parameter_max_abs"]
    actual_pass = actual["torch_loss_max_abs"] <= 1e-6 and actual_gradient_pass and clipped_gradient_pass and optimizer_formula_pass and causal_residual_pass
    actual_pass = actual_pass and actual.get("construction_seeds") == list(RUNG_ONE_SEEDS) and actual.get("data_seeds") == [300000 + seed for seed in RUNG_ONE_SEEDS]
    actual_pass = actual_pass and actual.get("lanes") == 5 and actual.get("unique_parameter_hashes") == 5 and actual.get("codebook_grad_none_effect_exact") is True and actual.get("five_lane_grad_none_zero_exact") is True and actual.get("finite") is True and actual.get("torch_route_exact") is True
    actual_pass = actual_pass and actual.get("five_lane_gradient_absolute_pass") is actual_gradient_absolute_pass and actual.get("five_lane_gradient_scale_aware_pass") is actual_gradient_scale_pass and actual.get("five_lane_gradient_pass") is actual_gradient_pass
    actual_pass = actual_pass and actual.get("five_lane_clipped_gradient_pass") is clipped_gradient_pass
    actual_pass = actual_pass and gradient_tolerances and runtime_identity and mapping_identity and update_identity and norm_identity and end_to_end_identity and optimizer_identity and causal_identity and gradient_identity and digest_identity
    if not actual_pass or actual.get("torch_parity_pass") is not True or actual.get("pass") is not True:
        raise MlxQualificationError("initial actual five-lane parity differs")
    if value.get("pass") is not True:
        raise MlxQualificationError("initial self-check decision differs")
    primary_gradient_ratio = gradient["gradient_max_abs"] / 1e-5
    fallback_gradient_ratio = max(gradient["gradient_max_abs"] / 3e-5, gradient["gradient_relative_max"] / 1e-4, gradient["gradient_normalized_l2_max"] / 5e-5, max(0.0, (1.0 - gradient["gradient_cosine_min"]) / (1.0 - 0.999999999)))
    primary_five_lane_gradient_ratio = actual["five_lane_gradient_max_abs"] / 1e-5
    fallback_five_lane_gradient_ratio = max(actual["five_lane_gradient_max_abs"] / 1.25e-4, actual["five_lane_gradient_relative_max"] / 2.5e-4, actual["five_lane_gradient_normalized_l2_max"] / 1.25e-4, max(0.0, (1.0 - actual["five_lane_gradient_cosine_min"]) / (1.0 - 0.99999999)))
    primary_clipped_gradient_ratio = actual["five_lane_clipped_gradient_max_abs"] / 1e-5
    fallback_clipped_gradient_ratio = max(actual["five_lane_clipped_gradient_max_abs"] / 1.25e-4, actual["five_lane_clipped_gradient_relative_max"] / 2.5e-4, actual["five_lane_clipped_gradient_normalized_l2_max"] / 1.25e-4, max(0.0, (1.0 - actual["five_lane_clipped_gradient_cosine_min"]) / (1.0 - 0.99999999)))
    ratios = [
        *forward_ratios,
        *router_ratios,
        gradient["loss_max_abs"] / 1e-6,
        gradient["component_loss_max_abs"] / 1e-6,
        min(primary_gradient_ratio, fallback_gradient_ratio),
        adam["max_abs"] / 1e-7,
        carried_adam_ratio,
        actual["torch_loss_max_abs"] / 1e-6,
        min(primary_five_lane_gradient_ratio, fallback_five_lane_gradient_ratio),
        min(primary_clipped_gradient_ratio, fallback_clipped_gradient_ratio),
        actual["optimizer_formula_max_bound_ratio"],
        actual["causal_residual_worst_bound_ratio"],
    ]
    return {"pass": True, "worst_bound_ratio": max(ratios)}


def validate_child_message(message: Mapping[str, Any], state: dict[str, Any]) -> str:
    if state.get("closed") is True or not isinstance(message, Mapping):
        raise MlxProtocolError("child message state differs")
    kind = message.get("kind")
    sequence = message.get("sequence")
    if type(sequence) is not int or sequence != state.get("next_sequence"):
        raise MlxProtocolError("child message sequence differs")
    if kind == "hello":
        _exact_keys(message, ("kind", "sequence", "schema_version", "mlx_version", "engine_sha256", "dependency_sha256s", "self_check", "self_check_sha256", "device"), "hello")
        if state.get("hello") is True or message["schema_version"] != IPC_SCHEMA_VERSION or message["mlx_version"] != MLX_VERSION or message["device"] != "Device(gpu, 0)":
            raise MlxProtocolError("hello identity differs")
        _sha256_text(message["engine_sha256"], "engine")
        _sha256_text(message["self_check_sha256"], "self check")
        self_check = message["self_check"]
        if not isinstance(self_check, Mapping) or hashlib.sha256(json.dumps(self_check, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest() != message["self_check_sha256"]:
            raise MlxProtocolError("hello self-check differs")
        validate_initial_self_check(self_check)
        if message["dependency_sha256s"] != dependency_hashes() or message["engine_sha256"] != message["dependency_sha256s"]["engine"]:
            raise MlxProtocolError("hello dependency hashes differ")
        state["hello"] = True
    elif kind == "stage_started":
        _exact_keys(message, ("kind", "sequence", "stage", "construction_seeds"), "stage started")
        request = state.get("pending_request")
        if state.get("hello") is not True or state.get("active_stage") is not None or not isinstance(request, Mapping):
            raise MlxProtocolError("stage start transition differs")
        stage = message["stage"]
        if stage != request["stage"] or message["construction_seeds"] != request["construction_seeds"]:
            raise MlxProtocolError("stage start identity differs")
        state["active_stage"] = (stage, tuple(message["construction_seeds"]))
        state["pending_update"] = None
        state["completed_updates"] = 0
    elif kind == "update_ready":
        _exact_keys(message, ("kind", "sequence", "stage", "construction_seeds", "logical_update", "batch_sha256s", "token_positions"), "update ready")
        request = state.get("pending_request")
        identity = (message["stage"], tuple(message["construction_seeds"]))
        logical_update = message["logical_update"]
        if not isinstance(request, Mapping) or state.get("active_stage") != identity or state.get("pending_update") is not None:
            raise MlxProtocolError("update ready transition differs")
        if type(logical_update) is not int or logical_update != state.get("completed_updates", 0) + 1 or logical_update > request["updates"]:
            raise MlxProtocolError("update ready sequence differs")
        count = len(request["construction_seeds"])
        if not isinstance(message["batch_sha256s"], list) or len(message["batch_sha256s"]) != count or not isinstance(message["token_positions"], list) or len(message["token_positions"]) != count:
            raise MlxProtocolError("update ready lane count differs")
        for value in message["batch_sha256s"]:
            _sha256_text(value, "batch")
        expected_positions = request["batch_size"] * (512 if request["stage"] == "rung_two" else 128)
        if any(type(value) is not int or value != expected_positions for value in message["token_positions"]):
            raise MlxProtocolError("update ready token positions differ")
        state["pending_update"] = {"logical_update": logical_update, "batch_sha256s": list(message["batch_sha256s"])}
    elif kind == "update_complete":
        _exact_keys(message, ("kind", "sequence", "stage", "construction_seeds", "logical_update", "batch_sha256s", "metrics", "mx_eval_complete", "memory"), "update complete")
        request = state.get("pending_request")
        pending = state.get("pending_update")
        identity = (message["stage"], tuple(message["construction_seeds"]))
        if not isinstance(request, Mapping) or not isinstance(pending, Mapping) or state.get("active_stage") != identity:
            raise MlxProtocolError("update completion transition differs")
        if message["logical_update"] != pending["logical_update"] or message["batch_sha256s"] != pending["batch_sha256s"] or message["mx_eval_complete"] is not True:
            raise MlxProtocolError("update completion identity differs")
        metrics = message["metrics"]
        if not isinstance(metrics, list) or len(metrics) != len(request["construction_seeds"]):
            raise MlxProtocolError("update completion metrics count differs")
        metric_keys = ("total_loss", "task_loss", "internal_router_loss", "supervised_route_loss", "gradient_norm", "clip_result", "raw_overflow_count", "max_bucket_load", "elapsed_seconds", "finite")
        for row in metrics:
            _exact_keys(row, metric_keys, "update metric")
            if row["finite"] is not True or row["raw_overflow_count"] != 0 or row["clip_result"] not in {"clipped", "unchanged"}:
                raise MlxProtocolError("update metric gate differs")
            for field in ("total_loss", "task_loss", "gradient_norm", "elapsed_seconds"):
                if isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(float(row[field])) or float(row[field]) < 0.0:
                    raise MlxProtocolError("update metric numeric value differs")
            for field in ("internal_router_loss", "supervised_route_loss"):
                if row[field] is not None and (isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(float(row[field])) or float(row[field]) < 0.0):
                    raise MlxProtocolError("update metric component differs")
            if type(row["max_bucket_load"]) is not int or row["max_bucket_load"] < 0:
                raise MlxProtocolError("update route load differs")
        memory_keys = ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes", "parent_rss_and_swap_required")
        _exact_keys(message["memory"], memory_keys, "update memory")
        if message["memory"]["parent_rss_and_swap_required"] is not True or any(type(message["memory"][field]) is not int or message["memory"][field] < 0 for field in memory_keys[:3]):
            raise MlxProtocolError("update memory values differ")
        state["completed_updates"] = pending["logical_update"]
        state["pending_update"] = None
    elif kind == "stage_complete":
        _exact_keys(message, ("kind", "sequence", "stage", "construction_seeds", "checkpoint_paths", "checkpoint_sha256s", "optimizer_state_sha256s", "rng_state_sha256s"), "stage complete")
        identity = (message["stage"], tuple(message["construction_seeds"]))
        request = state.get("pending_request")
        if state.get("active_stage") != identity or not isinstance(request, Mapping) or state.get("pending_update") is not None:
            raise MlxProtocolError("stage completion transition differs")
        if state.get("completed_updates") != request["updates"]:
            raise MlxProtocolError("stage completion update count differs")
        count = len(message["construction_seeds"])
        for field in ("checkpoint_paths", "checkpoint_sha256s", "optimizer_state_sha256s", "rng_state_sha256s"):
            if not isinstance(message[field], list) or len(message[field]) != count:
                raise MlxProtocolError(f"stage completion {field} differs")
        for field in ("checkpoint_sha256s", "optimizer_state_sha256s", "rng_state_sha256s"):
            for value in message[field]:
                _sha256_text(value, field)
        if message["checkpoint_paths"] != request["checkpoint_outputs"]:
            raise MlxProtocolError("stage completion checkpoint paths differ")
        state["active_stage"] = None
        state["pending_request"] = None
        state["completed_updates"] = 0
    elif kind == "evaluation_complete":
        _exact_keys(message, ("kind", "sequence", "result"), "evaluation complete")
        result = message["result"]
        if state.get("hello") is not True or state.get("pending_request") is not None or state.get("active_stage") is not None or state.get("evaluated") is True or not isinstance(result, Mapping) or tuple(sorted(result)) != ("rung_one", "rung_two"):
            raise MlxProtocolError("evaluation completion transition differs")
        rung_one = result["rung_one"]
        rung_two = result["rung_two"]
        if not isinstance(rung_one, list) or [record.get("seed") for record in rung_one] != list(RUNG_ONE_SEEDS) or not isinstance(rung_two, Mapping):
            raise MlxProtocolError("evaluation completion identity differs")
        for record in rung_one:
            expected = {"evaluation_rows": 65, "prediction_rows": 6144, "state_rows": 1980, "intervention_rows": 108, "forward_sequence": 4048}
            if any(record.get(key) != value for key, value in expected.items()):
                raise MlxProtocolError("rung-one evaluation cardinality differs")
        expected = {"evaluation_rows": 2, "prediction_rows": 1024, "state_rows": 230, "intervention_rows": 14, "gate_conditions": 2, "source_telemetry_max_error": 0.0}
        if any(rung_two.get(key) != value for key, value in expected.items()):
            raise MlxProtocolError("rung-two evaluation cardinality differs")
        state["evaluated"] = True
    elif kind == "hard_abort":
        _exact_keys(message, ("kind", "sequence", "reason", "error_type", "message", "memory"), "hard abort")
        if state.get("hello") is not True or message["reason"] not in {"nonfinite", "route_overflow", "artifact_inconsistency"}:
            raise MlxProtocolError("hard abort transition differs")
        if not isinstance(message["error_type"], str) or not message["error_type"] or not isinstance(message["message"], str) or not message["message"]:
            raise MlxProtocolError("hard abort detail differs")
        memory_keys = ("active_memory_bytes", "cache_memory_bytes", "peak_memory_bytes", "parent_rss_and_swap_required")
        _exact_keys(message["memory"], memory_keys, "hard abort memory")
        if message["memory"]["parent_rss_and_swap_required"] is not True or any(type(message["memory"][field]) is not int or message["memory"][field] < 0 for field in memory_keys[:3]):
            raise MlxProtocolError("hard abort memory differs")
        state["pending_request"] = None
        state["active_stage"] = None
        state["pending_update"] = None
        state["closed"] = True
        state["aborted"] = True
    elif kind == "closed":
        _exact_keys(message, ("kind", "sequence", "status"), "closed")
        if state.get("hello") is not True or state.get("pending_request") is not None or state.get("active_stage") is not None or state.get("pending_update") is not None or state.get("evaluated") is not True or message["status"] != "clean_complete":
            raise MlxProtocolError("close transition differs")
        state["closed"] = True
    else:
        raise MlxProtocolError("child message kind differs")
    state["next_sequence"] = sequence + 1
    return str(kind)


def _stage_seeds(stage: str, donor_seeds: list[int] | None = None) -> list[int]:
    if stage == "donor":
        if not isinstance(donor_seeds, list) or len(donor_seeds) != 1 or donor_seeds[0] not in RUNG_ONE_SEEDS:
            raise MlxProtocolError("donor construction seed differs")
        return list(donor_seeds)
    if donor_seeds is not None:
        raise MlxProtocolError("non-donor seed override differs")
    if stage == "rung_two":
        return [RUNG_TWO_SEED]
    if stage in {"router_only", "joint", "dense_base", "dense_continuation"}:
        return list(RUNG_ONE_SEEDS)
    raise MlxProtocolError("stage name differs")


def _data_seeds(stage: str, seeds: list[int]) -> list[int]:
    base = {
        "donor": 100000,
        "router_only": 200000,
        "joint": 300000,
        "dense_base": 100000,
        "dense_continuation": 300000,
        "rung_two": 900000,
    }[stage]
    return [base + seed for seed in seeds]


def _checkpoint_paths(stage: str, seeds: list[int]) -> tuple[list[str], list[str]]:
    if stage == "rung_two":
        return [], ["rung2/83/checkpoints/final_last.pt"]
    roots = [f"rung1/{seed}/checkpoints" for seed in seeds]
    inputs_by_stage = {
        "donor": None,
        "router_only": "donor_last.pt",
        "joint": "router_last.pt",
        "dense_base": None,
        "dense_continuation": "dense_base_last.pt",
    }
    outputs_by_stage = {
        "donor": "donor_last.pt",
        "router_only": "router_last.pt",
        "joint": "final_last.pt",
        "dense_base": "dense_base_last.pt",
        "dense_continuation": "dense_last.pt",
    }
    input_name = inputs_by_stage[stage]
    inputs = [] if input_name is None else [f"{root}/{input_name}" for root in roots]
    outputs = [f"{root}/{outputs_by_stage[stage]}" for root in roots]
    return inputs, outputs


def stage_request(stage: str, donor_seeds: list[int] | None = None, sequence: int = 0) -> dict[str, Any]:
    if stage not in STAGE_CONTRACTS or type(sequence) is not int or sequence < 0:
        raise MlxProtocolError("stage request coordinates differ")
    seeds = _stage_seeds(stage, donor_seeds)
    inputs, outputs = _checkpoint_paths(stage, seeds)
    contract = STAGE_CONTRACTS[stage]
    request = {
        "schema_version": IPC_SCHEMA_VERSION,
        "kind": "run_stage",
        "sequence": sequence,
        "stage": stage,
        "construction_seeds": seeds,
        "data_generator_seeds": _data_seeds(stage, seeds),
        "updates": contract["updates"],
        "warmup_updates": contract["warmup_updates"],
        "batch_size": contract["batch_size"],
        "checkpoint_inputs": inputs,
        "checkpoint_outputs": outputs,
    }
    return validate_stage_request(request)


def bind_stage_request(state: dict[str, Any], request: Mapping[str, Any]) -> None:
    validated = validate_stage_request(request)
    if state.get("hello") is not True or state.get("closed") is True or state.get("pending_request") is not None or state.get("active_stage") is not None:
        raise MlxProtocolError("stage request bind transition differs")
    state["pending_request"] = validated


def validate_stage_request(request: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "kind",
        "sequence",
        "stage",
        "construction_seeds",
        "data_generator_seeds",
        "updates",
        "warmup_updates",
        "batch_size",
        "checkpoint_inputs",
        "checkpoint_outputs",
    )
    _exact_keys(request, keys, "stage request")
    if request["schema_version"] != IPC_SCHEMA_VERSION or request["kind"] != "run_stage" or type(request["sequence"]) is not int or request["sequence"] < 0:
        raise MlxProtocolError("stage request identity differs")
    stage = request["stage"]
    if stage not in STAGE_CONTRACTS:
        raise MlxProtocolError("stage request name differs")
    expected = STAGE_CONTRACTS[stage]
    for field in ("updates", "warmup_updates", "batch_size"):
        if request[field] != expected[field]:
            raise MlxProtocolError(f"stage request {field} differs")
    count = expected["seed_count"]
    for field in ("construction_seeds", "data_generator_seeds", "checkpoint_outputs"):
        if not isinstance(request[field], list) or len(request[field]) != count:
            raise MlxProtocolError(f"stage request {field} differs")
    if not isinstance(request["checkpoint_inputs"], list):
        raise MlxProtocolError("stage request checkpoint_inputs differs")
    if len(set(request["construction_seeds"])) != count or len(set(request["data_generator_seeds"])) != count:
        raise MlxProtocolError("stage request seed independence differs")
    expected_seeds = _stage_seeds(stage, request["construction_seeds"] if stage == "donor" else None)
    if request["construction_seeds"] != expected_seeds or request["data_generator_seeds"] != _data_seeds(stage, expected_seeds):
        raise MlxProtocolError("stage request seed identity differs")
    expected_inputs, expected_outputs = _checkpoint_paths(stage, expected_seeds)
    if request["checkpoint_inputs"] != expected_inputs or request["checkpoint_outputs"] != expected_outputs:
        raise MlxProtocolError("stage request checkpoint identity differs")
    for raw in request["checkpoint_inputs"] + request["checkpoint_outputs"]:
        path = PurePosixPath(raw)
        if path.is_absolute() or path.as_posix() != raw or any(part in {"", ".", ".."} for part in path.parts):
            raise MlxProtocolError("stage request checkpoint path differs")
    return dict(request)


def mapped_mlx_parameter_name(torch_name: str) -> str:
    if not isinstance(torch_name, str) or not torch_name:
        raise MlxProtocolError("torch parameter name differs")
    name = torch_name
    name = name.replace(".mix.source_mixer.attention.router.codebooks", ".mix.codebooks")
    name = name.replace(".mix.source_mixer.attention.router.query_projection", ".mix.query_projection")
    name = name.replace(".mix.source_mixer.attention.router.key_projection", ".mix.key_projection")
    name = name.replace(".mix.source_mixer.attention.qkv", ".mix.qkv")
    name = name.replace(".mix.source_mixer.attention.out", ".mix.out")
    return name


def _tensor_descriptor(value: Any, source: str) -> tuple[list[int], str]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != ("dtype", "shape"):
        raise MlxProtocolError(f"{source} tensor descriptor differs")
    shape = value["shape"]
    dtype = value["dtype"]
    if not isinstance(shape, list) or not shape or any(type(item) is not int or item <= 0 for item in shape) or not isinstance(dtype, str):
        raise MlxProtocolError(f"{source} tensor descriptor differs")
    return list(shape), dtype


def validate_parameter_mapping(torch_parameters: Mapping[str, Any], mlx_parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(torch_parameters, Mapping) or not isinstance(mlx_parameters, Mapping) or not torch_parameters or not mlx_parameters:
        raise MlxProtocolError("parameter mapping surface differs")
    records = []
    destinations = set()
    for source_name in sorted(torch_parameters):
        destination_name = mapped_mlx_parameter_name(source_name)
        if destination_name not in mlx_parameters or destination_name in destinations:
            raise MlxProtocolError("parameter mapping is not bijective")
        source_shape, source_dtype = _tensor_descriptor(torch_parameters[source_name], "torch")
        destination_shape, destination_dtype = _tensor_descriptor(mlx_parameters[destination_name], "MLX")
        if source_shape != destination_shape or source_dtype != "torch.float32" or destination_dtype != "float32":
            raise MlxProtocolError("parameter mapping shape or dtype differs")
        destinations.add(destination_name)
        records.append({"torch_name": source_name, "mlx_name": destination_name, "shape": source_shape, "torch_dtype": source_dtype, "mlx_dtype": destination_dtype, "transpose": False})
    if destinations != set(mlx_parameters):
        raise MlxProtocolError("parameter mapping is not onto")
    raw = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {"bijective": True, "transpose": False, "records": records, "sha256": hashlib.sha256(raw).hexdigest()}


def optimizer_parameter_policy(name: str, stage: str) -> dict[str, Any]:
    if stage not in STAGE_CONTRACTS or not isinstance(name, str) or not name:
        raise MlxProtocolError("optimizer parameter coordinates differ")
    marker = ".mix.source_mixer.attention.router."
    router = marker in name
    router_zero = name.startswith("blocks.0") and router
    router_four = name.startswith("blocks.4") and router
    if stage in {"donor", "rung_two"}:
        trainable = not router
        peak = 0.002 if trainable else None
    elif stage == "router_only":
        trainable = router_four
        peak = 0.003 if trainable else None
    elif stage == "joint":
        trainable = not router_zero
        peak = 0.001 if router_four else 0.00025 if trainable else None
    elif stage == "dense_base":
        trainable = not router_zero
        peak = 0.002 if trainable else None
    else:
        trainable = not router_zero
        peak = 0.00025 if trainable else None
    zero_decay = name.endswith(".bias") or name.endswith(".codebooks") or name == "nf.weight" or ".n1.weight" in name or ".n2.weight" in name or ".onorm.weight" in name
    decay = 0.0 if zero_decay else 0.01
    return {"trainable": trainable, "peak_lr": peak, "weight_decay": decay if trainable else None}


def optimizer_contract() -> dict[str, Any]:
    return {
        "kind": "AdamW",
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "bias_correction": True,
        "global_norm_clip": 1.0,
        "vmap_clip_scope": "per_seed_lane",
        "fresh_state_each_stage": True,
        "decay_policy": "matrix_only",
    }


class AtomicVmapAttemptLedger:
    def __init__(self, path: str | Path, validator: Any):
        self.path = Path(path)
        self.validator = validator
        self.descriptor: int | None = None
        self.offset = 0
        self.digest = hashlib.sha256()

    def precreate(self) -> None:
        if self.descriptor is not None:
            raise MlxBatchLedgerError("vmap attempt ledger is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.fsync(descriptor)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.descriptor = descriptor

    def _validate_batch(self, records: list[Mapping[str, Any]]) -> None:
        if not isinstance(records, list) or len(records) != 5:
            raise MlxBatchLedgerError("vmap attempt batch width differs")
        for record in records:
            self.validator(record)
        seeds = [record.get("construction_seed") for record in records]
        if seeds != list(RUNG_ONE_SEEDS):
            raise MlxBatchLedgerError("vmap attempt seed order differs")
        identity_fields = ("run_id", "rung", "event_sequence", "event", "model", "stage", "logical_update", "examples", "token_positions")
        for field in identity_fields:
            if len({json.dumps(record.get(field), sort_keys=True) for record in records}) != 1:
                raise MlxBatchLedgerError(f"vmap attempt batch field differs: {field}")
        if records[0]["rung"] != 1 or records[0]["stage"] not in {"router_only", "joint", "dense_base", "dense_continuation"} or records[0]["examples"] != 16 or records[0]["token_positions"] != 2048:
            raise MlxBatchLedgerError("vmap attempt batch identity differs")
        if len({record["attempt_id"] for record in records}) != 5 or len({record["batch_sha256"] for record in records}) != 5:
            raise MlxBatchLedgerError("vmap attempt lane identity is reused")

    def append_batch(self, records: list[Mapping[str, Any]], fault: str | None = None) -> dict[str, Any]:
        if self.descriptor is None:
            raise MlxBatchLedgerError("vmap attempt ledger is not open")
        if fault not in {None, "before_write", "short_write", "before_fsync", "fsync", "readback"}:
            raise MlxBatchLedgerError("vmap attempt fault differs")
        self._validate_batch(records)
        raw = b"".join(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n" for record in records)
        prior_offset = self.offset
        prior_digest = self.digest.copy()
        try:
            os.lseek(self.descriptor, prior_offset, os.SEEK_SET)
            if fault == "before_write":
                raise OSError("injected prewrite failure")
            if fault == "short_write":
                os.write(self.descriptor, raw[: max(1, len(raw) // 2)])
                raise OSError("injected short write")
            written = os.write(self.descriptor, raw)
            if written != len(raw) or fault == "before_fsync":
                raise OSError("vmap attempt write failed")
            if fault == "fsync":
                raise OSError("injected fsync failure")
            os.fsync(self.descriptor)
            os.lseek(self.descriptor, prior_offset, os.SEEK_SET)
            readback = os.read(self.descriptor, len(raw))
            if fault == "readback":
                readback = readback[:-1]
            if readback != raw or hashlib.sha256(readback).digest() != hashlib.sha256(raw).digest():
                raise MlxBatchLedgerError("vmap attempt batch readback differs")
            decoded = [json.loads(line) for line in readback.decode("utf-8").splitlines()]
            self._validate_batch(decoded)
            self.offset += len(raw)
            self.digest.update(raw)
            return {"prior_offset": prior_offset, "committed_offset": self.offset, "sha256": hashlib.sha256(raw).hexdigest(), "row_count": 5}
        except BaseException as error:
            os.ftruncate(self.descriptor, prior_offset)
            os.fsync(self.descriptor)
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            restored = os.read(self.descriptor, prior_offset)
            if len(restored) != prior_offset or hashlib.sha256(restored).digest() != prior_digest.digest():
                raise MlxBatchLedgerError("vmap attempt rollback failed") from error
            self.offset = prior_offset
            self.digest = prior_digest
            raise MlxBatchLedgerError("vmap attempt batch was not committed") from error

    def validate_prefix(self) -> list[dict[str, Any]]:
        if self.descriptor is None:
            raise MlxBatchLedgerError("vmap attempt ledger is not open")
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        raw = os.read(self.descriptor, self.offset)
        if hashlib.sha256(raw).digest() != self.digest.digest():
            raise MlxBatchLedgerError("vmap attempt committed digest differs")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        if len(rows) % 5:
            raise MlxBatchLedgerError("vmap attempt committed row count differs")
        for start in range(0, len(rows), 5):
            self._validate_batch(rows[start:start + 5])
        return rows

    def close(self) -> None:
        if self.descriptor is None:
            raise MlxBatchLedgerError("vmap attempt ledger is not open")
        os.close(self.descriptor)
        self.descriptor = None
