from __future__ import annotations

import dataclasses
import datetime as datetime_module
import decimal
import fcntl
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "todorov.cpu-witness.1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PARENT = PROJECT_ROOT / "neuroloc" / "results" / "modular_sequence_role_mlx"
PREREG_PATH = PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json"
RUN_CARD_PATH = PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_run.md"
REVIEW_EVIDENCE_DIRECTORY = PROJECT_ROOT / "neuroloc" / "results" / "modular_sequence_role_mlx_reviews"
PREREG_CANONICAL_SHA256 = "fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553"
PROJECT_PLAN_RELATIVE_PATH = "neuroloc/wiki/PROJECT_PLAN.md"
LAUNCH_PROJECT_PLAN_PATH = "run/project_plan_launch.md"
TRAINING_START_REQUEST_PATH = "run/training_start_request.json"
TRAINING_START_PROJECT_PLAN_PATH = "run/project_plan_training_start.md"
TRAINING_START_LINK_PATH = "run/training_start_plan.json"
TRAINING_START_REVIEW_SCOPE_PREFIX = "training_start_project_plan:"
TRAINING_START_REVIEW_WAIT_SECONDS = 1800
TRAINING_START_REVIEW_WAIT_NS = TRAINING_START_REVIEW_WAIT_SECONDS * 1_000_000_000
TRAINING_START_REVIEW_POLL_SECONDS = 1.0
TRAINING_START_COMMIT_MARGIN_SECONDS = 5
TRAINING_START_COMMIT_MARGIN_NS = TRAINING_START_COMMIT_MARGIN_SECONDS * 1_000_000_000
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
REQUIRED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
}
REQUIRED_PYTHON = (3, 9, 6)
RESOURCE_SAMPLE_INTERVAL_NS = 5_000_000_000
CLAIM_RSS_LIMIT_BYTES = 12 * 1024**3
CLAIM_SWAP_GROWTH_LIMIT_BYTES = 512 * 1024**2
WORKER_ORPHAN_EXIT_CODE = 86
FAULT_IDS = (
    "write_failure_before_any_byte",
    "short_write",
    "write_failure_after_full_line_before_fsync",
    "fsync_failure",
    "readback_schema_or_hash_failure",
    "handled_signal_after_commit_before_ack",
)
CLAIM_LEDGER_PATHS = (
    "run/resources.jsonl",
    "rung1/11/attempts.jsonl",
    "rung1/23/attempts.jsonl",
    "rung1/37/attempts.jsonl",
    "rung1/53/attempts.jsonl",
    "rung1/71/attempts.jsonl",
    "rung2/83/attempts.jsonl",
)
ALL_LEDGER_PATHS = ("run/pilot_resources.jsonl",) + CLAIM_LEDGER_PATHS
HARD_ABORT_REASON_CODES = (
    "signal_or_interruption",
    "frozen_hash_change",
    "assertion_failure",
    "resource_sampler_failure",
    "nonfinite",
    "route_overflow",
    "endpoint_inconsistency",
    "artifact_inconsistency",
    "resident_memory",
    "swap_growth",
    "claim_elapsed_time",
    "worker_exit",
    "unpaired_attempt",
)
HARD_ABORT_CONDITIONS = (
    "any_handled_signal_or_interruption_after_hard_abort_registry_activation",
    "any_frozen_source_configuration_or_preregistration_hash_changes_except_the_exact_governed_PROJECT_PLAN_reviewed_ready_to_started_transition",
    "any_pretraining_assertion_fails_before_or_during_execution",
    "resource_command_exit_or_parse_or_pid_contract_failure",
    "any_loss_logit_recurrent_state_recurrent_gate_gradient_or_parameter_nonfinite",
    "any_route_index_overflow",
    "endpoint_hash_reload_generator_or_final_batch_inconsistency",
    "required_artifact_schema_hash_reference_or_manifest_inconsistency",
    "aggregate_rss_bytes_greater_than_12884901888_for_three_valid_samples_at_least_five_seconds_apart",
    "claim_swap_growth_bytes_greater_than_536870912",
    "runtime_guard_observes_claim_monotonic_elapsed_seconds_greater_than_1200_at_any_point_through_clean_SHA256SUMS_packaging",
    "worker_nonzero_or_signaled_exit_or_premature_exit_without_valid_clean_completion_handshake_unless_earlier_primary_latched",
    "any_started_event_lacks_its_adjacent_next_completed_event_with_all_pair_fields_equal",
)
RUNG_ONE_SEEDS = (11, 23, 37, 53, 71)
RUNG_TWO_SEED = 83
WORKER_ASSIGNMENTS = {"A": (11, 37, 71), "B": (23, 53, 83)}
WORKER_JOB_ASSIGNMENTS = {
    "A": ({"rung": 1, "construction_seed": 11}, {"rung": 1, "construction_seed": 37}, {"rung": 1, "construction_seed": 71}),
    "B": ({"rung": 1, "construction_seed": 23}, {"rung": 1, "construction_seed": 53}, {"rung": 2, "construction_seed": 83}),
}
PARITY_SCOPES = (
    "source",
    "checksum",
    "ABI",
    "host",
    "recurrent",
    "reset",
    "firewall",
    "causality",
    "raw_route",
    "index",
    "internal_loss",
    "attention",
    "lifecycle",
    "initialization",
    "copy",
    "reload",
    "intervention",
    "trained_backend",
)
PRETRAINING_ASSERTION_IDS = (
    "mixer_abi_and_residual_ownership",
    "exact_architecture",
    "firewall_factorization",
    "reset_aware_recurrent_fidelity",
    "query_only_remote_route",
    "causality",
    "intervention_isolation",
    "forced_and_random_route_exactness",
    "source_host_route_and_attention_parity",
    "state_and_index_lifetime",
    "optimizer_membership_and_gradients",
    "initialization_copy_and_reload_parity",
    "fresh_optimizer_state",
    "generator_integrity",
    "capacity_and_fallback",
)
ROUTE_CAPACITY_BY_RUNG = {1: 16, 2: 64}
REASON_PRIORITY = {reason: index for index, reason in enumerate(HARD_ABORT_REASON_CODES, 1)}
RUNG_ONE_CONDITIONS = (
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
)
RUNG_TWO_CONDITIONS = ("intact", "recurrent_knockout")
RUNG_ONE_MODEL_BY_CONDITION = {
    "intact": "selected",
    "target_forced": "selected",
    "recurrent_knockout": "selected",
    "carry_reset": "selected",
    "carry_shuffle": "selected",
    "matched_random_route": "selected",
    "block4_routed_knockout": "selected",
    "block4_local_only": "local",
    "required_source_excluded": "selected",
    "all_eligible_donor": "donor",
    "all_eligible_clone": "clone",
    "dense_causal": "dense",
}
RUNG_ONE_ACCOUNTING_MODEL_BY_ROUTE_MODEL = {
    "selected": "selected",
    "local": "selected",
    "donor": "all_eligible_donor",
    "clone": "all_eligible_donor",
    "dense": "dense_causal",
}
RUNG_ONE_STAGE_ENDPOINTS = {
    "donor": ("donor_last.pt", "all_eligible_donor", "donor", 1024),
    "router_only": ("router_last.pt", "selected", "router", 768),
    "joint": ("final_last.pt", "selected", "joint", 512),
    "dense_base": ("dense_base_last.pt", "dense_causal", "dense_base", 1024),
    "dense_continuation": ("dense_last.pt", "dense_causal", "dense", 512),
}
RECURRENT_BLOCKS = (1, 2, 3, 5, 6, 7)
RUNG_ONE_RESET_POSITIONS = (8, 16, 24, 32, 40, 48, 56, 64, 72, 80)
RUNG_ONE_CHUNK_END_POSITIONS = (31, 63, 95, 127)
RUNG_TWO_CHUNK_END_POSITIONS = tuple(range(31, 512, 32))
ATTEMPT_KEYS = (
    "schema_version",
    "run_id",
    "rung",
    "claim_seed",
    "construction_seed",
    "event_sequence",
    "event",
    "attempt_id",
    "model",
    "stage",
    "logical_update",
    "examples",
    "token_positions",
    "batch_sha256",
    "monotonic_ns",
    "wall_time_utc",
    "metrics",
)
ATTEMPT_METRIC_KEYS = (
    "learning_rates",
    "component_losses",
    "total_loss",
    "gradient_norm",
    "clip_result",
    "raw_overflow_count",
    "max_bucket_load",
    "elapsed_seconds",
    "finite",
)
ATTEMPT_PAIR_EQUAL_FIELDS = (
    "schema_version",
    "run_id",
    "rung",
    "claim_seed",
    "construction_seed",
    "model",
    "stage",
    "logical_update",
    "examples",
    "token_positions",
    "batch_sha256",
    "attempt_id",
)
RESOURCE_ROW_KEYS = (
    "schema_version",
    "run_id",
    "sample_id",
    "phase",
    "monotonic_ns",
    "wall_time_utc",
    "expected_pids",
    "processes",
    "active_jobs",
    "aggregate_rss_bytes",
    "aggregate_cpu_time_us",
    "swap_used_bytes",
    "swap_growth_bytes",
    "parser_status",
    "attempted_updates",
    "token_positions",
)
ABORTED_KEYS = (
    "schema_version",
    "run_id",
    "reason_code",
    "condition",
    "phase",
    "training_start_state",
    "worker",
    "seed",
    "stage",
    "logical_update",
    "last_event_sequence",
    "monotonic_elapsed_seconds",
    "wall_start_utc",
    "wall_end_utc",
    "completed_work",
    "attempted_work",
    "resource_state",
    "frozen_hashes",
    "new_run_required",
)


class ContractError(RuntimeError):
    pass


class ProcessSetMismatch(ContractError):
    def __init__(self, expected_pids: Sequence[int], observed_pids: Sequence[int]) -> None:
        expected = tuple(sorted(int(pid) for pid in expected_pids))
        observed = tuple(sorted(int(pid) for pid in observed_pids))
        if any(type(pid) is not int or pid < 1 for pid in (*expected, *observed)) or len(expected) != len(set(expected)) or len(observed) != len(set(observed)):
            raise ContractError("process mismatch identity differs")
        super().__init__("ps output PID set differs")
        self.expected_pids = expected
        self.observed_pids = observed


class InitializationRefusal(ContractError):
    pass


class UnrecoverableOrphan(ContractError):
    pass


class WorkerStartError(ContractError):
    def __init__(self, worker: str) -> None:
        if worker not in {"A", "B"}:
            raise ContractError("worker start identity differs")
        super().__init__(f"worker {worker} failed to start")
        self.worker = worker


class LedgerAppendError(ContractError):
    def __init__(self, result: "AppendResult") -> None:
        super().__init__(result.reason_code or "artifact_inconsistency")
        self.result = result


class HardAbort(ContractError):
    def __init__(self, reason_code: str, context: Mapping[str, Any] | None = None, primary_latch_monotonic_ns: int | None = None) -> None:
        if reason_code not in HARD_ABORT_REASON_CODES:
            raise ContractError(f"unknown hard-abort reason: {reason_code}")
        if primary_latch_monotonic_ns is not None and (type(primary_latch_monotonic_ns) is not int or primary_latch_monotonic_ns < 0):
            raise ContractError("primary latch timestamp differs")
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.context = dict(context or {})
        self.primary_latch_monotonic_ns = primary_latch_monotonic_ns


@dataclasses.dataclass(frozen=True)
class EntryConfiguration:
    run_root: Path
    run_id: str


@dataclasses.dataclass(frozen=True)
class AppendResult:
    prior_offset: int
    current_offset: int
    acknowledged: bool
    committed: bool
    reason_code: str | None
    line_sha256: str | None


@dataclasses.dataclass(frozen=True)
class AttemptAccounting:
    attempted_updates: int
    completed_updates: int
    attempted_token_positions: int
    completed_token_positions: int
    attempted_seeds: tuple[int, ...]
    completed_seeds: tuple[int, ...]
    last_event_sequence_by_seed: Mapping[int, int]
    unpaired_attempts: tuple[str, ...]


@dataclasses.dataclass
class PilotCounterState:
    attempted_updates: int
    token_positions: int


@dataclasses.dataclass(frozen=True)
class PublicationResult:
    state: str
    final_root: Path
    staging_root: Path
    registry_active: bool
    abort_accounting_start_monotonic_ns: int | None
    abort_wall_start_utc: str | None
    pending_signal: int | None


@dataclasses.dataclass(frozen=True)
class GuardedTransitionResult:
    committed: bool
    value: Any
    pending_signal: int | None


@dataclasses.dataclass(frozen=True)
class TransitionResult:
    phase: str
    outcome: str
    retained_paths: tuple[str, ...]
    writers: Mapping[str, "CrashAtomicJsonlWriter"]
    swap_baseline_bytes: int | None
    reason_code: str | None


@dataclasses.dataclass(frozen=True)
class TerminalResult:
    checksum_path: Path
    covered_paths: tuple[str, ...]
    terminal: bool


@dataclasses.dataclass(frozen=True)
class RuntimeModules:
    torch: Any
    model_module: Any


@dataclasses.dataclass(frozen=True)
class FailureObservation:
    reason_code: str
    condition: str
    context: Mapping[str, Any]
    monotonic_ns: int
    worker: str | None
    event_sequence: int | None


@dataclasses.dataclass(frozen=True)
class FrozenManifestAnchors:
    records: tuple[tuple[str, str], ...]


class PrimaryFailureLatch:
    def __init__(self, registry: Sequence[Mapping[str, Any]]) -> None:
        expected = tuple(
            {"priority": index, "reason_code": reason, "condition": condition}
            for index, (reason, condition) in enumerate(zip(HARD_ABORT_REASON_CODES, HARD_ABORT_CONDITIONS), 1)
        )
        if tuple(registry) != expected:
            raise ContractError("hard-abort registry differs at latch construction")
        self._conditions = {row["reason_code"]: row["condition"] for row in registry}
        self._selected: FailureObservation | None = None

    @property
    def selected(self) -> FailureObservation | None:
        return self._selected

    def select_poll(self, observations: Sequence[Mapping[str, Any]], monotonic_ns: Callable[[], int] = time.monotonic_ns) -> FailureObservation | None:
        if self._selected is not None or not observations:
            return self._selected
        normalized = []
        for ordinal, item in enumerate(observations):
            reason = item.get("reason_code")
            if reason not in self._conditions:
                raise ContractError("failure observation reason is outside the registry")
            context = dict(item.get("context") or {})
            worker = context.get("worker") if context.get("worker") in {"A", "B"} else None
            sequence_value = context.get("event_sequence")
            event_sequence = sequence_value if type(sequence_value) is int and sequence_value >= 0 else None
            normalized.append((REASON_PRIORITY[reason], 0 if worker is None else 1, {None: -1, "A": 0, "B": 1}[worker], -1 if event_sequence is None else event_sequence, ordinal, reason, context, worker, event_sequence))
        selected = min(normalized)
        captured = monotonic_ns()
        if isinstance(captured, bool) or not isinstance(captured, int) or captured < 0:
            raise ContractError("primary latch timestamp differs")
        self._selected = FailureObservation(selected[5], self._conditions[selected[5]], selected[6], captured, selected[7], selected[8])
        return self._selected


def failure_observation_from_exception(
    error: BaseException,
    default_reason: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(error, UnrecoverableOrphan):
        raise error
    if default_reason not in HARD_ABORT_REASON_CODES:
        raise ContractError("default failure reason is outside the registry")
    merged = dict(context or {})
    if isinstance(error, HardAbort):
        merged.update(error.context)
        reason = error.reason_code
    elif isinstance(error, LedgerAppendError):
        reason = error.result.reason_code or default_reason
    elif isinstance(error, (FloatingPointError, ArithmeticError)):
        reason = "nonfinite"
    else:
        reason = default_reason
    if reason not in HARD_ABORT_REASON_CODES:
        raise ContractError("mapped failure reason is outside the registry")
    return {"reason_code": reason, "context": merged}


def parent_worker_failure_observation(error: BaseException, worker: str, transport: bool) -> dict[str, Any]:
    if worker not in {"A", "B"} or type(transport) is not bool:
        raise ContractError("parent worker failure context differs")
    transport_failure = transport or isinstance(error, (EOFError, BrokenPipeError, ConnectionError))
    default_reason = "worker_exit" if transport_failure else "artifact_inconsistency"
    observation = failure_observation_from_exception(error, default_reason, {"worker": worker})
    mapped_context = dict(observation["context"])
    mapped_context["worker"] = worker
    return {"reason_code": observation["reason_code"], "context": mapped_context}


def hard_abort_from_same_poll(latch: PrimaryFailureLatch, observations: Sequence[Mapping[str, Any]]) -> HardAbort | None:
    selected = latch.select_poll(observations)
    if selected is None:
        return None
    return HardAbort(selected.reason_code, selected.context, selected.monotonic_ns)


def _is_process_alive(process: Any) -> bool:
    try:
        return bool(process.is_alive())
    except BaseException as exc:
        raise UnrecoverableOrphan("worker liveness cannot be verified") from exc


def quiesce_worker_processes(processes: Sequence[Any], timeout_seconds: float = 1.0) -> None:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise ContractError("worker quiescence timeout differs")
    timeout = float(timeout_seconds)
    process_values = tuple(processes)
    for process in process_values:
        try:
            process.join(timeout=timeout)
        except BaseException:
            pass
    for process in process_values:
        try:
            alive = _is_process_alive(process)
        except UnrecoverableOrphan:
            alive = True
        if alive:
            try:
                process.terminate()
            except BaseException:
                pass
    for process in process_values:
        try:
            process.join(timeout=timeout)
        except BaseException:
            pass
    for process in process_values:
        try:
            alive = _is_process_alive(process)
        except UnrecoverableOrphan:
            alive = True
        if alive:
            try:
                process.kill()
            except BaseException:
                pass
    for process in process_values:
        try:
            process.join(timeout=timeout)
        except BaseException:
            pass
    if any(_is_process_alive(process) for process in process_values):
        raise UnrecoverableOrphan("worker remained live after bounded kill escalation")


def quiesce_after_primary_latch(error: HardAbort, processes: Sequence[Any]) -> HardAbort:
    if error.primary_latch_monotonic_ns is None:
        raise ContractError("quiesce requires a persisted primary latch")
    identity = (error.reason_code, dict(error.context), error.primary_latch_monotonic_ns)
    quiesce_worker_processes(processes)
    for process in processes:
        try:
            exitcode = process.exitcode
        except BaseException as exc:
            raise UnrecoverableOrphan("worker final exit status is unobservable") from exc
        if exitcode is None:
            raise UnrecoverableOrphan("worker final exit status is unobservable")
        if exitcode == WORKER_ORPHAN_EXIT_CODE:
            raise UnrecoverableOrphan("worker crossed the orphan boundary during quiescence")
    if (error.reason_code, error.context, error.primary_latch_monotonic_ns) != identity:
        raise ContractError("primary latch mutated during quiesce")
    return error


def close_parent_connections(parents: Mapping[str, Any]) -> None:
    orphan: UnrecoverableOrphan | None = None
    for connection in parents.values():
        try:
            connection.close()
        except UnrecoverableOrphan as exc:
            if orphan is None:
                orphan = exc
        except BaseException:
            pass
    if orphan is not None:
        raise orphan


def spawn_worker_processes(context: Any, specifications: Sequence[Mapping[str, Any]]) -> tuple[list[Any], dict[str, Any]]:
    processes = []
    parents: dict[str, Any] = {}
    active_worker: str | None = None
    try:
        for specification in specifications:
            validate_exact_keys(specification, ("worker", "target", "args", "name"), "worker start specification")
            worker = specification["worker"]
            if worker not in {"A", "B"} or worker in parents or not isinstance(specification["args"], tuple) or not isinstance(specification["name"], str) or not specification["name"]:
                raise ContractError("worker start specification differs")
            active_worker = worker
            parent_connection, child_connection = context.Pipe(duplex=True)
            parents[worker] = parent_connection
            try:
                process = context.Process(target=specification["target"], args=(*specification["args"], child_connection), name=specification["name"])
                processes.append(process)
                process.start()
            finally:
                child_connection.close()
        return processes, parents
    except BaseException as exc:
        for connection in parents.values():
            try:
                connection.close()
            except BaseException:
                pass
        quiesce_worker_processes(processes)
        if active_worker in {"A", "B"}:
            raise WorkerStartError(active_worker) from exc
        raise


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat().replace("+00:00", "Z")


def validate_exact_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    actual = set(value)
    expected = set(keys)
    if actual != expected:
        raise ContractError(f"{name} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def validate_real_regular_file(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise ContractError("validated file path must be absolute")
    current = Path(target.anchor)
    for part in target.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ContractError("validated file path is absent") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError("validated file path chain contains a symbolic link")
    if not stat.S_ISREG(os.lstat(target).st_mode) or target.resolve(strict=True) != target:
        raise ContractError("validated file is not a real regular file")
    return target


def _strict_utc(value: Any) -> datetime_module.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError("UTC timestamp differs")
    try:
        parsed = datetime_module.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError("UTC timestamp differs") from exc
    if parsed.tzinfo != datetime_module.timezone.utc:
        raise ContractError("UTC timestamp differs")
    return parsed


def _sysctl_text(name: str, run: Callable[..., subprocess.CompletedProcess[str]]) -> str:
    completed = run(
        ["/usr/sbin/sysctl", "-n", name],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0 or completed.stderr.strip() or not completed.stdout.strip():
        raise InitializationRefusal(f"hardware observation failed for {name}")
    return completed.stdout.strip()


def observe_target_hardware(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, Any]:
    chip = _sysctl_text("machdep.cpu.brand_string", run)
    try:
        cores = int(_sysctl_text("hw.physicalcpu", run), 10)
        memory = int(_sysctl_text("hw.memsize", run), 10)
    except ValueError as exc:
        raise InitializationRefusal("hardware integer observation differs") from exc
    observed = {"chip": chip, "cpu_core_count": cores, "memory_bytes": memory, "training_device": "integrated_Apple_GPU_via_MLX_Metal"}
    expected = {"chip": "Apple M5 Pro", "cpu_core_count": 15, "memory_bytes": 25769803776, "training_device": "integrated_Apple_GPU_via_MLX_Metal"}
    if observed != expected:
        raise InitializationRefusal(f"observed hardware differs: {observed}")
    return observed


def _assert_finite_tree(torch: Any, value: Any, context: Mapping[str, Any], path: str = "value") -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value.detach()).all()):
            raise HardAbort("nonfinite", {**context, "surface": path})
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _assert_finite_tree(torch, value[key], context, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(torch, child, context, f"{path}.{index}")
        return
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            _assert_finite_tree(torch, getattr(value, field.name), context, f"{path}.{field.name}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise HardAbort("nonfinite", {**context, "surface": path})


def _assert_model_and_optimizer_finite(torch: Any, model: Any, optimizer: Any | None, context: Mapping[str, Any]) -> None:
    for name, parameter in model.named_parameters():
        _assert_finite_tree(torch, parameter, context, f"parameter.{name}")
        if parameter.grad is not None:
            _assert_finite_tree(torch, parameter.grad, context, f"gradient.{name}")
    for name, buffer in model.named_buffers():
        _assert_finite_tree(torch, buffer, context, f"buffer.{name}")
    if optimizer is not None:
        for parameter, state_values in optimizer.state.items():
            if not any(parameter is candidate for candidate in model.parameters()):
                raise HardAbort("artifact_inconsistency", {**context, "surface": "optimizer.foreign_parameter"})
            _assert_finite_tree(torch, state_values, context, "optimizer.state")
        for index, group in enumerate(optimizer.param_groups):
            _assert_finite_tree(torch, {key: value for key, value in group.items() if key != "params"}, context, f"optimizer.param_groups.{index}")


def _clip_gradient_norm_finite(torch: Any, model: Any, optimizer: Any, context: Mapping[str, Any]) -> float:
    _assert_model_and_optimizer_finite(torch, model, optimizer, context)
    try:
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=True)
    except (RuntimeError, FloatingPointError, ArithmeticError) as exc:
        raise HardAbort("nonfinite", {**context, "surface": "gradient_norm"}) from exc
    _assert_finite_tree(torch, gradient_norm, context, "gradient_norm")
    _assert_model_and_optimizer_finite(torch, model, optimizer, context)
    return float(gradient_norm)


def _derived_routing_search_contract(batch_size: int, sequence_length: int, groups: int, selected_width: int, routing_width: int) -> dict[str, int]:
    values = (batch_size, sequence_length, groups, selected_width, routing_width)
    if any(type(value) is not int for value in values) or batch_size <= 0 or sequence_length <= 0 or groups <= 0 or selected_width < 0 or routing_width <= 0:
        raise ContractError("routing search geometry differs")
    if selected_width == 0:
        return {"workspace_count": 0, "workspace_bytes": 0, "posting_slots_materialized": 0, "search_rows": 0, "bypass_rows": batch_size * sequence_length * groups, "addresses_probed": 0}
    peak_slots = batch_size * min(sequence_length, 128) * groups * 4 * 64
    bypass_positions = min(sequence_length, (selected_width + 1) * 8)
    search_rows = batch_size * (sequence_length - bypass_positions) * groups
    return {
        "workspace_count": peak_slots * (2 + routing_width),
        "workspace_bytes": peak_slots * (8 + 4 + 4 * routing_width),
        "posting_slots_materialized": batch_size * sequence_length * groups * 4 * 64,
        "search_rows": search_rows,
        "bypass_rows": batch_size * bypass_positions * groups,
        "addresses_probed": search_rows * 4,
    }


def _workspace_storage_from_telemetry(telemetry: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[int, int]:
    raw_remote = telemetry.get("raw_remote")
    block_features = telemetry.get("block_features")
    if raw_remote is None or not hasattr(raw_remote, "shape") or raw_remote.ndim != 4 or block_features is None or not hasattr(block_features, "shape") or block_features.ndim != 3:
        raise HardAbort("artifact_inconsistency", {**context, "surface": "routing.workspace_detail"})
    batch_size, sequence_length, groups, selected_width = (int(value) for value in raw_remote.shape)
    if int(block_features.shape[0]) != batch_size:
        raise HardAbort("artifact_inconsistency", {**context, "surface": "routing.workspace_batch"})
    derived = _derived_routing_search_contract(batch_size, sequence_length, groups, selected_width, int(block_features.shape[-1]))
    for name in ("workspace_bytes", "posting_slots_materialized", "search_rows", "bypass_rows", "addresses_probed", "postings_read", "candidate_blocks"):
        value = telemetry.get(name)
        if type(value) is not int or value < 0:
            raise HardAbort("artifact_inconsistency", {**context, "surface": f"routing.{name}"})
    for name in ("workspace_bytes", "posting_slots_materialized", "search_rows", "bypass_rows", "addresses_probed"):
        if telemetry[name] != derived[name]:
            raise HardAbort("artifact_inconsistency", {**context, "surface": f"routing.derived_{name}"})
    if telemetry["postings_read"] != telemetry["candidate_blocks"] or telemetry["postings_read"] > derived["addresses_probed"] * 64:
        raise HardAbort("artifact_inconsistency", {**context, "surface": "routing.candidate_counter"})
    return derived["workspace_count"], derived["workspace_bytes"]


def _route_observation(output: Any, rung: int, context: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    if rung not in ROUTE_CAPACITY_BY_RUNG:
        raise ContractError("route observation rung differs")
    overflow = 0
    maximum = 0
    index_count = 0
    index_bytes = 0
    workspace_count = 0
    workspace_bytes = 0
    for block_execution in output.blocks:
        routed = block_execution.mixer_output
        if routed is None or not hasattr(routed, "telemetry"):
            continue
        telemetry = routed.telemetry
        for name in ("overflow_count", "max_bucket_load", "workspace_bytes"):
            value = telemetry.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HardAbort("artifact_inconsistency", {**context, "surface": f"routing.{name}"})
        observed_overflow = int(telemetry["overflow_count"])
        observed_maximum = int(telemetry["max_bucket_load"])
        overflow += observed_overflow
        maximum = max(maximum, observed_maximum)
        observed_workspace_count, observed_workspace_bytes = _workspace_storage_from_telemetry(telemetry, context)
        workspace_count += observed_workspace_count
        workspace_bytes += observed_workspace_bytes
        for name in ("block_features", "block_addresses", "postings"):
            tensor = telemetry.get(name)
            if tensor is not None:
                index_count += int(tensor.numel())
                index_bytes += int(tensor.numel() * tensor.element_size())
        if observed_overflow or observed_maximum > ROUTE_CAPACITY_BY_RUNG[rung]:
            raise HardAbort("route_overflow", {**context, "block": block_execution.block_index, "raw_overflow_count": observed_overflow, "max_bucket_load": observed_maximum})
    return overflow, maximum, index_count, index_bytes, workspace_count, workspace_bytes


def parse_cli(argv: Sequence[str]) -> str:
    values = list(argv)
    if len(values) != 2 or values[0] != "--run-root" or not isinstance(values[1], str) or not values[1]:
        raise InitializationRefusal("the command line must contain exactly --run-root <absolute-path>")
    return values[1]


def validate_entry_environment(environ: Mapping[str, str] | None = None) -> None:
    values = os.environ if environ is None else environ
    for name, expected in REQUIRED_ENV.items():
        actual = values.get(name)
        if actual != expected:
            raise InitializationRefusal(f"{name} must equal {expected}")
    if tuple(sys.version_info[:3]) != REQUIRED_PYTHON:
        raise InitializationRefusal("Python must be exactly 3.9.6")


def validate_run_root(raw_path: str | Path, require_absent: bool = True) -> EntryConfiguration:
    raw = os.fspath(raw_path)
    if not raw or "\x00" in raw or not os.path.isabs(raw):
        raise InitializationRefusal("run root must be an absolute path")
    normalized = os.path.normpath(raw)
    if normalized != raw or os.path.abspath(normalized) != raw:
        raise InitializationRefusal("run root must already be normalized")
    path = Path(raw)
    if path.parent != RESULTS_PARENT:
        raise InitializationRefusal(f"run root parent must be {RESULTS_PARENT}")
    if not RUN_ID_PATTERN.fullmatch(path.name):
        raise InitializationRefusal("run id does not match the frozen pattern")
    if require_absent and os.path.lexists(path):
        raise InitializationRefusal("run root must not exist")
    return EntryConfiguration(path, path.name)


def load_prereg_payload(path: str | Path = PREREG_PATH) -> dict[str, Any]:
    payload_path = Path(path)
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ContractError("preregistration payload is not strict JSON") from exc
    validate_prereg_payload(payload)
    digest = canonical_json_sha256(payload)
    if digest != PREREG_CANONICAL_SHA256:
        raise ContractError(f"preregistration canonical digest differs: {digest}")
    return payload


def _null_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if value is None:
        paths.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            paths.extend(_null_paths(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_null_paths(child, f"{prefix}.{index}" if prefix else str(index)))
    return paths


def validate_gate_registry(payload: Mapping[str, Any]) -> None:
    gates = payload["gates"]
    registries = (
        ("rung_one_registry", 24, "r1."),
        ("rung_two_registry", 4, "r2."),
    )
    seen: set[str] = set()
    for name, cardinality, prefix in registries:
        rows = gates[name]
        if not isinstance(rows, list) or len(rows) != cardinality:
            raise ContractError("gate registry cardinality differs")
        for row in rows:
            validate_exact_keys(row, ("gate_id", "condition", "metric", "stratum", "gate_operator", "gate_threshold_unit", "gate_threshold", "gate_threshold_count", "denominator"), "gate registry row")
            gate_id = row["gate_id"]
            expected_id = f"{prefix}{row['condition']}.{row['metric']}.{row['stratum']}"
            if gate_id != expected_id or gate_id in seen:
                raise ContractError("gate registry identity differs")
            seen.add(gate_id)
            if row["gate_operator"] not in {">=", "<=", "=="} or row["gate_threshold_unit"] not in {"rate", "count", "absolute_error"}:
                raise ContractError("gate registry operation differs")
            threshold = row["gate_threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
                raise ContractError("gate threshold differs")
            denominator = row["denominator"]
            threshold_count = row["gate_threshold_count"]
            pairs: list[tuple[int, int]] = []
            if isinstance(denominator, Mapping):
                if not isinstance(threshold_count, Mapping) or list(denominator) != [str(seed) for seed in RUNG_ONE_SEEDS] or list(threshold_count) != list(denominator):
                    raise ContractError("per-seed gate threshold maps differ")
                pairs = [(int(denominator[str(seed)]), int(threshold_count[str(seed)])) for seed in RUNG_ONE_SEEDS]
            elif denominator is not None:
                if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0 or isinstance(threshold_count, bool) or not isinstance(threshold_count, int):
                    raise ContractError("gate denominator or threshold count differs")
                pairs = [(denominator, threshold_count)]
            elif row["gate_threshold_unit"] == "rate" or threshold_count is not None and not isinstance(threshold_count, int):
                raise ContractError("gate null contract differs")
            for population, count in pairs:
                if row["gate_operator"] == ">=":
                    expected_count = math.ceil(float(threshold) * population)
                elif row["gate_operator"] == "<=":
                    expected_count = math.floor(float(threshold) * population)
                else:
                    expected_count = round(float(threshold) * population)
                if count != expected_count:
                    raise ContractError("gate integer threshold differs")
    if len(seen) != 28:
        raise ContractError("gate registry identities differ")


def validate_pretraining_registry(payload: Mapping[str, Any]) -> None:
    assertions = payload["gates"]["pretraining_assertions"]
    if tuple(row.get("ordinal") for row in assertions) != tuple(range(1, 16)) or tuple(row.get("id") for row in assertions) != PRETRAINING_ASSERTION_IDS:
        raise ContractError("pretraining assertion registry differs")
    for row in assertions:
        validate_exact_keys(row, ("ordinal", "id", "required"), "pretraining assertion registry row")
        if not isinstance(row["required"], str) or not row["required"]:
            raise ContractError("pretraining assertion requirement differs")
    parity_scopes = payload["artifacts"]["schemas"]["parity"]["ordered_scope"]
    if tuple(parity_scopes) != PARITY_SCOPES:
        raise ContractError("parity scope registry differs")


def validate_prereg_payload(payload: Mapping[str, Any]) -> None:
    top_keys = (
        "schema_version",
        "architecture",
        "sources",
        "processes",
        "generators",
        "stages",
        "optimizer",
        "schedule",
        "losses",
        "controls",
        "gates",
        "pilot",
        "abort_rules",
        "artifacts",
    )
    validate_exact_keys(payload, top_keys, "preregistration")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("schema version differs")
    authorization = payload["architecture"]["authorization"]
    expected_authorization = {
        "dependency_changes_authorized": False,
        "external_compute_cost_usd": 0,
        "novel_mechanism_claimed": False,
        "paid_compute_authorized": False,
        "reciprocal_feature_mixer_enabled": False,
        "scope": "local_mlx_metal_base_composition_with_torch_cpu_reference_authority",
        "trainingnovel_enabled": False,
    }
    if authorization != expected_authorization:
        raise ContractError("authorization boundary differs")
    runtime_policy = payload["architecture"]["runtime_policy"]
    if runtime_policy != {
            "automatic_mixed_precision": False,
            "compilation": True,
            "device": "mlx_metal_gpu",
            "dropout": 0.0,
            "dtype": "mlx.core.float32",
            "training_backend": {
                "framework": "MLX",
                "version": "0.29.3",
                "device": "Device(gpu, 0)",
                "execution": "compiled_Metal",
                "optimizer": "mlx.optimizers.AdamW_with_frozen_CPU_group_policy",
            },
            "reference_backend": {
                "framework": "Torch",
                "version": "2.8.0",
                "device": "cpu",
                "dtype": "torch.float32",
                "authority": ["initialization", "source_math", "checkpoint_serialization", "trained_endpoint_parity", "evaluation", "gates", "artifact_validation"],
            },
    }:
        raise ContractError("runtime policy differs")
    routed_adapter = payload["architecture"]["common_host"]["routed_adapter"]
    adapter_symbols = routed_adapter["authorized_source_symbols"]
    references = [
        reference
        for reference in payload["sources"]["references"]
        if reference["path"] == routed_adapter["source_module"]
    ]
    source_symbols = references[0]["authorized_symbols"] if len(references) == 1 else []
    if source_symbols != ["MonodraticPHIMixer", *adapter_symbols] or len(source_symbols) != len(set(source_symbols)) or len(adapter_symbols) != len(set(adapter_symbols)):
        raise ContractError("Monodratic authorized symbol registries differ")
    if routed_adapter.get("adapter_local_exact_math_symbols") != ["_sparse_selected_attention"]:
        raise ContractError("adapter exact-math symbol registry differs")
    agreement = payload["sources"]["monodratic_authorized_symbol_registry_agreement"]
    if agreement != {
        "source_reference_registry_path": "sources.references[path=/Users/dttdrv/Projects/Monodratic-public/src/monodratic/core.py].authorized_symbols",
        "adapter_registry_path": "architecture.common_host.routed_adapter.authorized_source_symbols",
        "deliberate_source_reference_only_class_symbol": "MonodraticPHIMixer",
        "required_relation": "source_reference_registry_equals_class_symbol_prepended_to_adapter_registry_in_identical_order",
        "missing_additional_reordered_or_duplicated_symbol_rejected": True,
    }:
        raise ContractError("Monodratic authorized symbol agreement differs")
    probe_audit = routed_adapter["evidence_only_probe_audit"]
    routing_schema = payload["artifacts"]["schemas"]["routing_row"]
    probe_arguments = ["detached_current_query_route_features", "detached_current_codebooks_from_the_same_search", 4]
    post_probe_inputs = ["returned_probe_addresses", "existing_packed_postings", "remote_limit"]
    if probe_audit["probe_helper_ordered_arguments"] != probe_arguments or probe_audit["post_probe_inputs"] != post_probe_inputs or routing_schema["valid_posting_histogram_probe_helper_arguments"] != probe_arguments or routing_schema["valid_posting_histogram_post_probe_inputs"] != post_probe_inputs:
        raise ContractError("routing evidence probe ABI differs")
    launcher = payload["processes"]["launcher"]
    if launcher["resume_supported"] is not False or launcher["only_argument"] != "--run-root" or launcher["argument_count"] != 2:
        raise ContractError("launcher contract differs")
    multiprocessing_contract = payload["processes"]["multiprocessing"]
    if multiprocessing_contract["start_method"] != "parent_subprocess_with_start_new_session" or multiprocessing_contract["worker_count"] != 1:
        raise ContractError("multiprocessing contract differs")
    if payload["processes"]["environment"]["parent_exact_values"] != REQUIRED_ENV:
        raise ContractError("environment registry differs")
    if tuple(payload["artifacts"]["serialization"]["crash_atomic_canonical_jsonl"]["fault_injection_preflight"]) != FAULT_IDS:
        raise ContractError("fault registry differs")
    if tuple(payload["stages"]["attempt_contract"]["precreated_claim_jsonl_paths"]) != CLAIM_LEDGER_PATHS:
        raise ContractError("claim ledger path registry differs")
    pilot_gates = payload["pilot"]["proceed_gates"]
    if pilot_gates.get("Tprojected_seconds_target") != 600 or pilot_gates.get("Tprojected_seconds_max") != 1200:
        raise ContractError("pilot time gate differs")
    registry = payload["abort_rules"]["hard_abort_registry"]
    expected_registry = tuple(
        {"priority": priority, "reason_code": reason_code, "condition": condition}
        for priority, (reason_code, condition) in enumerate(zip(HARD_ABORT_REASON_CODES, HARD_ABORT_CONDITIONS), 1)
    )
    if tuple(registry) != expected_registry:
        raise ContractError("hard-abort priorities differ")
    expected_nulls = tuple(row["path"] for row in payload["gates"]["null_contract"])
    if tuple(_null_paths(payload)) != expected_nulls:
        raise ContractError("payload null contract differs")
    if payload["gates"]["registry_cardinalities"] != {
        "complete_package": 124,
        "rung_one_instances": 120,
        "rung_one_per_seed": 24,
        "rung_one_seed_count": 5,
        "rung_two": 4,
    }:
        raise ContractError("gate cardinalities differ")
    if len(payload["gates"]["rung_one_registry"]) != 24 or len(payload["gates"]["rung_two_registry"]) != 4:
        raise ContractError("gate registry length differs")
    validate_gate_registry(payload)
    validate_pretraining_registry(payload)
    if payload["processes"]["target_hardware"] != {"chip": "Apple M5 Pro", "cpu_core_count": 15, "memory_bytes": 25769803776, "training_device": "integrated_Apple_GPU_via_MLX_Metal"}:
        raise ContractError("target hardware registry differs")
    if payload["artifacts"]["root_contract"]["writes_outside_published_run_root"] != "sole_governed_external_write_exception_is_the_training_start_live_PROJECT_PLAN_transaction_comprising_owned_same_directory_candidate_temp_creation_atomic_replacement_directory_fsync_and_owned_temp_cleanup":
        raise ContractError("external write exception differs")
    handshake = payload["processes"]["multiprocessing"]["clean_completion_handshake"]
    if handshake["exact_keys"] != ["kind", "sequence", "status"] or handshake["status_value"] != "clean_complete":
        raise ContractError("clean completion handshake registry differs")
    artifact_paths = payload["artifacts"]["artifact_paths"]
    if payload["artifacts"]["artifact_path_pattern_count"] != 58 or len(artifact_paths) != 58 or len(set(artifact_paths)) != 58:
        raise ContractError("artifact path registry differs")
    exact_ledger_paths = payload["artifacts"]["serialization"]["crash_atomic_canonical_jsonl"]["exact_paths"]
    if tuple(exact_ledger_paths) != ALL_LEDGER_PATHS:
        raise ContractError("transactional ledger path registry differs")
    review_schema = payload["artifacts"]["schemas"]["review_artifact"]
    expected_review_schema = {
        "path": "run/reviews/{artifact_sha256}.json",
        "source_evidence_directory": "neuroloc/results/modular_sequence_role_mlx_reviews",
        "source_evidence_path_pattern": "neuroloc/results/modular_sequence_role_mlx_reviews/{artifact_sha256}.json",
        "source_evidence_directory_outside_all_review_target_scopes": True,
        "source_evidence_directory_symlink_allowed": False,
        "source_evidence_file_symlink_allowed": False,
        "runner_authors_or_mutates_attestation": False,
        "source_and_run_bytes_identical": True,
        "schema_version_value": "todorov.review-attestation.1",
        "exact_keys": ["schema_version", "reviewer", "scope", "target_records", "target_sha256", "findings", "finding_count"],
        "target_record_exact_keys": ["path", "sha256"],
        "target_record_order": "path_sorted",
        "base_target_records_must_equal_current_live_scope": True,
        "target_sha256_formula": "sha256_canonical_json_of_path_sorted_path_sha256_records",
        "finding_exact_keys": ["id", "severity", "description", "evidence_paths", "resolution"],
        "accepted_reviewer": "feature-dev:code-reviewer",
        "accepted_findings": [],
        "accepted_finding_count": 0,
        "exactly_one_matching_attestation_per_scope": True,
        "historical_nonmatching_attestations_allowed": True,
        "creation_timing": "independent_reviewer_writes_the_four_base_attestations_before_launch_and_writes_the_fifth_run_bound_training_start_project_plan_attestation_only_after_run/training_start_request.json_is_durable_and_before_the_live_plan_atomic_start_commit",
        "base_content_hash_binding": "source_filename_run_filename_config_manifest_reference_source_manifest_sha256_and_sha256_colon_revision",
        "training_start_content_hash_binding": "source_filename_run_filename_and_run/training_start_plan.json_review_artifact_sha256",
        "training_start_scope_exact_targets": ["neuroloc/wiki/PROJECT_PLAN.md", "run/training_start_request.json"],
    }
    if type(review_schema.get("accepted_finding_count")) is not int or review_schema != expected_review_schema:
        raise ContractError("review attestation schema differs")
    training_start_review = payload["artifacts"]["training_start_review_attestation"]
    if type(training_start_review.get("wait_timeout_seconds")) is not int or type(training_start_review.get("finding_count")) is not int:
        raise ContractError("training-start review numeric schema differs")
    if (
        training_start_review.get("candidate_source_path_pattern") != "neuroloc/results/modular_sequence_role_mlx_reviews/{candidate_sha256}.project-plan.md"
        or training_start_review.get("candidate_source_raw_sha256_must_equal_candidate_sha256") is not True
        or training_start_review.get("candidate_binding_line_formula") != "Training start request `{run_id}` binds request SHA-256 `{request_sha256}`; these reviewed bytes become canonical only at the atomic training-start commit.\n"
    ):
        raise ContractError("training-start candidate handoff schema differs")
    if payload["abort_rules"].get("frozen_hash_transition_exception") != {
        "path": PROJECT_PLAN_RELATIVE_PATH,
        "allowed_transition": "reviewed_ready_launch_sha256_to_started_training_start_project_plan_sha256",
        "governed_by": TRAINING_START_LINK_PATH,
        "all_other_reviewed_targets_remain_base_attested": True,
    }:
        raise ContractError("frozen hash transition exception differs")
    review_sources = payload["artifacts"]["source_manifest_required_review_attestations"]
    if review_sources != {
        "count": 4,
        "directory": "neuroloc/results/modular_sequence_role_mlx_reviews",
        "selection": "exactly_one_content_addressed_nonsymlink_attestation_matching_each_current_review_scope_and_target_digest",
        "role_order": "scope_order_from_config_manifest_review_records",
        "roles": [
            "review_attestation_base_preregistration",
            "review_attestation_base_implementation",
            "review_attestation_base_tests",
            "review_attestation_base_complete_surface",
        ],
        "revision": "sha256_colon_raw_content_digest",
        "copied_run_bytes_must_match_source": True,
    }:
        raise ContractError("review attestation source order differs")


def _import_runtime() -> RuntimeModules:
    torch = importlib.import_module("torch")
    model_module = importlib.import_module("src.model.modular_neural_machine")
    return RuntimeModules(torch=torch, model_module=model_module)


def configure_torch(torch: Any) -> None:
    if str(torch.__version__).split("+")[0] != "2.8.0":
        raise InitializationRefusal("Torch must be exactly 2.8.0")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")


class SignalController:
    def __init__(self) -> None:
        self.pending: list[tuple[int, int]] = []
        self.deferred = 0
        self.active = False
        self.terminal = False
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        if self.terminal or self.active:
            raise ContractError("signal controller cannot be installed")
        for number in (signal.SIGINT, signal.SIGTERM):
            self._previous[number] = signal.getsignal(number)
            signal.signal(number, self._handle)
        self.active = True

    def _handle(self, number: int, frame: Any) -> None:
        if self.terminal:
            return
        self.pending.append((number, time.monotonic_ns()))

    def defer(self) -> None:
        if self.terminal:
            raise ContractError("terminal signal controller cannot defer")
        self.deferred += 1

    def release(self) -> int | None:
        if self.deferred <= 0:
            raise ContractError("signal deferral is unbalanced")
        self.deferred -= 1
        return self.pending_signal if self.deferred == 0 else None

    def acknowledge_pending_through(self, monotonic_ns: int) -> tuple[int, ...]:
        if self.terminal or type(monotonic_ns) is not int or monotonic_ns < 0:
            raise ContractError("signal acknowledgement boundary differs")
        acknowledged = tuple(number for number, observed_ns in self.pending if observed_ns <= monotonic_ns)
        self.pending = [(number, observed_ns) for number, observed_ns in self.pending if observed_ns > monotonic_ns]
        return acknowledged

    def commit_guarded(self, boundary: Callable[[], Any]) -> GuardedTransitionResult:
        if self.terminal or self.deferred != 0 or not callable(boundary):
            raise ContractError("guarded signal commit state differs")
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        result = GuardedTransitionResult(False, None, None)
        try:
            pending_numbers = signal.sigpending() & blocked
            pending_signal = self.pending_signal
            if pending_signal is None and pending_numbers:
                pending_signal = min(pending_numbers)
            if pending_signal is not None:
                result = GuardedTransitionResult(False, None, pending_signal)
            else:
                value = boundary()
                pending_numbers = signal.sigpending() & blocked
                pending_signal = self.pending_signal
                if pending_signal is None and pending_numbers:
                    pending_signal = min(pending_numbers)
                result = GuardedTransitionResult(True, value, pending_signal)
        except BaseException as exc:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as restore_exc:
                if isinstance(exc, UnrecoverableOrphan):
                    raise exc
                raise UnrecoverableOrphan("guarded signal mask restoration failed") from restore_exc
            raise
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException as exc:
            raise UnrecoverableOrphan("guarded signal mask restoration failed") from exc
        if result.pending_signal is None and self.pending_signal is not None:
            result = GuardedTransitionResult(result.committed, result.value, self.pending_signal)
        return result

    def commit_terminal(self, boundary: Callable[[], Any], preserve_primary: bool = False) -> int | None:
        if self.terminal or self.deferred != 1:
            raise ContractError("terminal signal commit state differs")
        if not callable(boundary) or type(preserve_primary) is not bool:
            raise ContractError("terminal signal boundary differs")
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        result: int | None = None
        try:
            released_signal = self.release()
            pending_numbers = signal.sigpending() & blocked
            pending_signal = released_signal if released_signal is not None else self.pending_signal
            if pending_signal is None and pending_numbers:
                pending_signal = min(pending_numbers)
            if pending_signal is not None and not preserve_primary:
                result = pending_signal
            else:
                if preserve_primary:
                    self.pending.clear()
                    while signal.sigpending() & blocked:
                        signal.sigwait(blocked)
                boundary()
                pending_numbers = signal.sigpending() & blocked
                pending_signal = self.pending_signal
                if pending_signal is None and pending_numbers:
                    pending_signal = min(pending_numbers)
                if pending_signal is not None and not preserve_primary:
                    result = pending_signal
                else:
                    if preserve_primary:
                        self.pending.clear()
                        while signal.sigpending() & blocked:
                            signal.sigwait(blocked)
                    if self.active:
                        for number, previous in self._previous.items():
                            signal.signal(number, previous)
                    self.active = False
                    self.terminal = True
        except BaseException as exc:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as restore_exc:
                if isinstance(exc, UnrecoverableOrphan):
                    raise exc
                raise UnrecoverableOrphan("terminal signal mask restoration failed") from restore_exc
            raise
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException as exc:
            if not self.terminal:
                raise UnrecoverableOrphan("terminal signal mask restoration failed") from exc
        return result

    @property
    def pending_signal(self) -> int | None:
        return self.pending[0][0] if self.pending else None

    def inject(self, number: int = signal.SIGTERM) -> None:
        self._handle(number, None)

    def deactivate_terminal(self) -> None:
        if self.deferred:
            raise ContractError("cannot terminate with deferred signals")
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            self.terminal = True
            if self.active:
                for number, previous in self._previous.items():
                    signal.signal(number, previous)
            self.active = False
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _filesystem_identity(path: str | Path) -> tuple[int, int]:
    metadata = os.lstat(path)
    return metadata.st_dev, metadata.st_ino


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    metadata = os.fstat(descriptor)
    return metadata.st_dev, metadata.st_ino


def _validate_owned_path(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = _filesystem_identity(path)
    except OSError as exc:
        raise UnrecoverableOrphan("owned path is absent") from exc
    if observed != identity:
        raise UnrecoverableOrphan("owned path identity changed")


def _validate_owned_paths(owned_paths: Mapping[Path, tuple[int, int]]) -> None:
    for path, identity in owned_paths.items():
        _validate_owned_path(path, identity)


def _unlink_owned_path(path: Path, identity: tuple[int, int]) -> None:
    _validate_owned_path(path, identity)
    path.unlink()
    fsync_directory(path.parent)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    fsync_directory(directory.parent)
    fsync_directory(directory)
    return directory


def write_canonical_json(
    path: str | Path,
    value: Any,
    exclusive: bool = True,
    owned_paths: dict[Path, tuple[int, int]] | None = None,
) -> None:
    target = Path(path)
    data = canonical_json_bytes(value)
    if exclusive:
        _write_exact_bytes(target, data, owned_paths)
        return
    with target.open("wb") as handle:
        written = handle.write(data)
        if written != len(data):
            raise ContractError("short canonical JSON write")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(target.parent)


def validate_canonical_jsonl_prefix(path: str | Path, end_offset: int, validator: Callable[[Mapping[str, Any]], None]) -> tuple[dict[str, Any], ...]:
    raw = Path(path).read_bytes()
    if len(raw) < end_offset:
        raise ContractError("ledger is shorter than committed offset")
    prefix = raw[:end_offset]
    if prefix and not prefix.endswith(b"\n"):
        raise ContractError("committed ledger prefix lacks line feed")
    rows: list[dict[str, Any]] = []
    for line in prefix.splitlines(keepends=True):
        if not line.endswith(b"\n") or line.count(b"\n") != 1:
            raise ContractError("ledger line framing differs")
        payload_bytes = line[:-1]
        try:
            payload = json.loads(payload_bytes.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except Exception as exc:
            raise ContractError("ledger line is not strict JSON") from exc
        if canonical_json_bytes(payload) != payload_bytes:
            raise ContractError("ledger line is not canonical JSON")
        validator(payload)
        rows.append(payload)
    return tuple(rows)


class CrashAtomicJsonlWriter:
    def __init__(
        self,
        path: str | Path,
        validator: Callable[[Mapping[str, Any]], None],
        signals: SignalController | None = None,
        sequence_kind: str | None = None,
    ) -> None:
        if sequence_kind not in {None, "attempt"}:
            raise ContractError("ledger sequence kind differs")
        self.path = Path(path)
        self.validator = validator
        self.signals = signals
        self.sequence_kind = sequence_kind
        self.last_committed_offset = 0
        self._descriptor: int | None = None
        self._owned_identity: tuple[int, int] | None = None
        self._closed = False
        self._committed_hash = hashlib.sha256()
        self._attempt_expected_sequence = 0
        self._attempt_pending: dict[str, Any] | None = None
        self._attempt_logical_identities: set[tuple[Any, ...]] = set()
        self._attempted_updates = 0
        self._completed_updates = 0
        self._attempted_token_positions = 0
        self._completed_token_positions = 0
        self._attempted_seeds: set[int] = set()
        self._completed_seeds: set[int] = set()
        self._last_event_sequence_by_seed: dict[int, int] = {}

    def _validate_sequence_candidate(self, record: Mapping[str, Any]) -> None:
        if self.sequence_kind is None:
            return
        if record["event_sequence"] != self._attempt_expected_sequence:
            raise ContractError("attempt event sequence is not zero-based contiguous")
        identity = tuple(record[field] for field in ("run_id", "rung", "construction_seed", "model", "stage", "logical_update"))
        if record["event"] == "started":
            if self._attempt_pending is not None:
                raise ContractError("attempt pair must complete before the next start")
            if identity in self._attempt_logical_identities:
                raise ContractError("logical update or attempt identity is reused")
            return
        if self._attempt_pending is None:
            raise ContractError("attempt pair must begin with started")
        for field in ATTEMPT_PAIR_EQUAL_FIELDS:
            if self._attempt_pending[field] != record[field]:
                raise ContractError(f"attempt pair field differs: {field}")

    def _commit_sequence_candidate(self, record: Mapping[str, Any]) -> None:
        if self.sequence_kind is None:
            return
        if record["event"] == "started":
            identity = tuple(record[field] for field in ("run_id", "rung", "construction_seed", "model", "stage", "logical_update"))
            self._attempt_logical_identities.add(identity)
            self._attempt_pending = dict(record)
            self._attempted_updates += 1
            self._attempted_token_positions += int(record["token_positions"])
            self._attempted_seeds.add(int(record["construction_seed"]))
        else:
            self._attempt_pending = None
            self._completed_updates += 1
            self._completed_token_positions += int(record["token_positions"])
            self._completed_seeds.add(int(record["construction_seed"]))
        self._last_event_sequence_by_seed[int(record["construction_seed"])] = int(record["event_sequence"])
        self._attempt_expected_sequence += 1

    def _restore_sequence_state(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._attempt_expected_sequence = 0
        self._attempt_pending = None
        self._attempt_logical_identities = set()
        self._attempted_updates = 0
        self._completed_updates = 0
        self._attempted_token_positions = 0
        self._completed_token_positions = 0
        self._attempted_seeds = set()
        self._completed_seeds = set()
        self._last_event_sequence_by_seed = {}
        for record in rows:
            self._validate_sequence_candidate(record)
            self._commit_sequence_candidate(record)

    def attempt_accounting(self) -> AttemptAccounting:
        if self.sequence_kind != "attempt":
            raise ContractError("attempt accounting requested from another ledger kind")
        unpaired = () if self._attempt_pending is None else (str(self._attempt_pending["attempt_id"]),)
        return AttemptAccounting(
            self._attempted_updates,
            self._completed_updates,
            self._attempted_token_positions,
            self._completed_token_positions,
            tuple(sorted(self._attempted_seeds)),
            tuple(sorted(self._completed_seeds)),
            dict(sorted(self._last_event_sequence_by_seed.items())),
            unpaired,
        )

    def precreate(self) -> None:
        if self._descriptor is not None or self._closed:
            raise ContractError("ledger writer is not fresh")
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        self._owned_identity = _descriptor_identity(descriptor)
        self._descriptor = descriptor
        try:
            os.fsync(descriptor)
            fsync_directory(self.path.parent)
            if os.fstat(descriptor).st_size != 0:
                raise ContractError("new ledger is not empty")
        except Exception:
            os.close(descriptor)
            self._descriptor = None
            raise
        self.last_committed_offset = 0
        self._committed_hash = hashlib.sha256()

    def open_existing(self, committed_offset: int | None = None) -> None:
        if self._descriptor is not None or self._closed:
            raise ContractError("ledger writer is not fresh")
        descriptor = os.open(self.path, os.O_RDWR)
        size = os.fstat(descriptor).st_size
        self._descriptor = descriptor
        self.last_committed_offset = size if committed_offset is None else committed_offset
        raw = self.path.read_bytes()[: self.last_committed_offset]
        self._committed_hash = hashlib.sha256(raw)
        rows = validate_canonical_jsonl_prefix(self.path, self.last_committed_offset, self.validator)
        if self.sequence_kind == "attempt":
            validate_attempt_sequence(rows, require_complete=False)
        self._restore_sequence_state(rows)

    def _write_full(self, data: bytes) -> None:
        if self._descriptor is None:
            raise ContractError("ledger is not open")
        written = os.write(self._descriptor, data)
        if written != len(data):
            raise OSError("short write")

    def _rollback(self, prior_offset: int, prior_digest: bytes) -> None:
        if self._descriptor is None:
            raise ContractError("ledger is not open")
        try:
            os.ftruncate(self._descriptor, prior_offset)
            os.fsync(self._descriptor)
            os.lseek(self._descriptor, 0, os.SEEK_SET)
            restored = os.read(self._descriptor, prior_offset)
            if hashlib.sha256(restored).digest() != prior_digest or os.fstat(self._descriptor).st_size != prior_offset:
                raise UnrecoverableOrphan("ledger rollback could not restore exact committed prefix")
        except UnrecoverableOrphan:
            raise
        except BaseException as exc:
            raise UnrecoverableOrphan("ledger rollback failed") from exc
        self.last_committed_offset = prior_offset

    def append(
        self,
        record: Mapping[str, Any],
        fault: str | None = None,
        pending_signal: bool | Callable[[], bool] | None = None,
    ) -> AppendResult:
        if self._closed or self._descriptor is None:
            raise ContractError("ledger is not writable")
        if fault is not None and fault not in FAULT_IDS:
            raise ContractError("unknown fault injection")
        self.validator(record)
        self._validate_sequence_candidate(record)
        line = canonical_json_bytes(record) + b"\n"
        line_hash = hashlib.sha256(line).hexdigest()
        prior_offset = self.last_committed_offset
        prior_digest = self._committed_hash.copy().digest()
        if os.fstat(self._descriptor).st_size != prior_offset:
            raise ContractError("ledger size differs from committed offset")
        if self.signals is not None:
            self.signals.defer()
        committed = False
        try:
            if callable(pending_signal):
                pending_before_write = bool(pending_signal())
            elif pending_signal is not None:
                pending_before_write = bool(pending_signal)
            else:
                pending_before_write = self.signals is not None and self.signals.pending_signal is not None
            if pending_before_write:
                return AppendResult(prior_offset, prior_offset, False, False, "signal_or_interruption", None)
            os.lseek(self._descriptor, prior_offset, os.SEEK_SET)
            if fault == "write_failure_before_any_byte":
                raise OSError("injected write failure before any byte")
            if fault == "short_write":
                cut = max(1, len(line) // 2)
                wrote = os.write(self._descriptor, line[:cut])
                if wrote != cut:
                    raise OSError("injected short write could not write prefix")
                raise OSError("injected short write")
            self._write_full(line)
            if fault == "write_failure_after_full_line_before_fsync":
                raise OSError("injected write failure after full line")
            if fault == "fsync_failure":
                raise OSError("injected fsync failure")
            os.fsync(self._descriptor)
            os.lseek(self._descriptor, prior_offset, os.SEEK_SET)
            readback = os.read(self._descriptor, len(line))
            if fault == "readback_schema_or_hash_failure":
                readback = readback[:-1]
            if readback != line or hashlib.sha256(readback).hexdigest() != line_hash:
                raise ContractError("readback bytes or hash differ")
            parsed = json.loads(readback[:-1].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if canonical_json_bytes(parsed) + b"\n" != readback:
                raise ContractError("readback is not canonical")
            self.validator(parsed)
            self.last_committed_offset = prior_offset + len(line)
            self._committed_hash.update(line)
            self._commit_sequence_candidate(parsed)
            committed = True
            if fault == "handled_signal_after_commit_before_ack":
                if self.signals is not None:
                    self.signals.inject()
                signal_pending = True
            elif callable(pending_signal):
                signal_pending = bool(pending_signal())
            elif pending_signal is not None:
                signal_pending = bool(pending_signal)
            else:
                signal_pending = self.signals is not None and self.signals.pending_signal is not None
            if signal_pending:
                result = AppendResult(prior_offset, self.last_committed_offset, False, True, "signal_or_interruption", line_hash)
                return result
            return AppendResult(prior_offset, self.last_committed_offset, True, True, None, line_hash)
        except UnrecoverableOrphan:
            raise
        except Exception:
            if committed:
                raise
            self._rollback(prior_offset, prior_digest)
            signal_pending = bool(self.signals is not None and self.signals.pending_signal is not None)
            if callable(pending_signal):
                signal_pending = signal_pending or bool(pending_signal())
            elif pending_signal is not None:
                signal_pending = signal_pending or bool(pending_signal)
            reason = "signal_or_interruption" if signal_pending else "artifact_inconsistency"
            result = AppendResult(prior_offset, prior_offset, False, False, reason, None)
            raise LedgerAppendError(result)
        finally:
            if self.signals is not None:
                self.signals.release()

    def validate_committed_prefix(self) -> tuple[dict[str, Any], ...]:
        rows = validate_canonical_jsonl_prefix(self.path, self.last_committed_offset, self.validator)
        if self.sequence_kind == "attempt":
            validate_attempt_sequence(rows, require_complete=False)
            if derive_attempt_accounting(rows) != self.attempt_accounting():
                raise ContractError("incremental attempt accounting differs")
        raw = self.path.read_bytes()[: self.last_committed_offset]
        if hashlib.sha256(raw).digest() != self._committed_hash.copy().digest():
            raise ContractError("ledger committed digest differs")
        return rows

    def recover_uncommitted_suffix(self) -> None:
        if self._descriptor is None:
            raise ContractError("ledger is not open")
        size = os.fstat(self._descriptor).st_size
        if size < self.last_committed_offset:
            raise UnrecoverableOrphan("ledger committed prefix is missing")
        if size > self.last_committed_offset:
            os.ftruncate(self._descriptor, self.last_committed_offset)
            os.fsync(self._descriptor)
        self.validate_committed_prefix()

    def close(self) -> None:
        if self._closed:
            return
        if self._descriptor is not None:
            self.recover_uncommitted_suffix()
            os.fsync(self._descriptor)
            os.close(self._descriptor)
        self._descriptor = None
        self._closed = True

    def validate_owned_path(self) -> None:
        if self._owned_identity is None:
            raise UnrecoverableOrphan("ledger path is not owned")
        _validate_owned_path(self.path, self._owned_identity)

    def remove_owned_path(self) -> None:
        self.validate_owned_path()
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        if self._owned_identity is None:
            raise UnrecoverableOrphan("ledger path is not owned")
        _unlink_owned_path(self.path, self._owned_identity)
        self._owned_identity = None
        self._closed = True


def attempt_id(run_id: str, rung: int, construction_seed: int, model: str, stage: str, logical_update: int) -> str:
    return canonical_json_sha256(
        {
            "run_id": run_id,
            "rung": rung,
            "construction_seed": construction_seed,
            "model": model,
            "stage": stage,
            "logical_update": logical_update,
        }
    )


def validate_attempt_row(record: Mapping[str, Any]) -> None:
    validate_exact_keys(record, ATTEMPT_KEYS, "attempt row")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError("attempt schema version differs")
    if record["event"] not in {"started", "completed"}:
        raise ContractError("attempt event differs")
    for field in ("run_id", "model", "stage", "wall_time_utc"):
        if not isinstance(record[field], str) or not record[field]:
            raise ContractError(f"attempt {field} differs")
    integer_fields = ("rung", "claim_seed", "construction_seed", "event_sequence", "logical_update", "examples", "token_positions", "monotonic_ns")
    if any(isinstance(record[field], bool) or not isinstance(record[field], int) for field in integer_fields):
        raise ContractError("attempt integer field differs")
    if record["rung"] not in {1, 2} or record["event_sequence"] < 0 or record["logical_update"] < 1 or record["examples"] < 1 or record["token_positions"] < 1:
        raise ContractError("attempt integer range differs")
    if record["claim_seed"] != record["construction_seed"]:
        raise ContractError("attempt claim and construction seeds differ")
    expected_id = attempt_id(record["run_id"], record["rung"], record["construction_seed"], record["model"], record["stage"], record["logical_update"])
    if record["attempt_id"] != expected_id:
        raise ContractError("attempt id differs")
    for field in ("attempt_id", "batch_sha256"):
        if not isinstance(record[field], str) or re.fullmatch(r"[0-9a-f]{64}", record[field]) is None:
            raise ContractError(f"attempt {field} differs")
    if record["event"] == "started":
        if record["metrics"] is not None:
            raise ContractError("started attempt metrics must be null")
    else:
        validate_exact_keys(record["metrics"], ATTEMPT_METRIC_KEYS, "attempt metrics")
        if record["metrics"]["clip_result"] not in {"unchanged", "clipped"}:
            raise ContractError("clip result differs")
        if not isinstance(record["metrics"]["finite"], bool):
            raise ContractError("finite flag differs")
        if record["metrics"]["finite"] is not True:
            raise ContractError("completed attempt is nonfinite")
        numeric_fields = ("total_loss", "gradient_norm", "elapsed_seconds")
        for field in numeric_fields:
            value = record["metrics"][field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ContractError(f"attempt metric {field} differs")
        for field in ("raw_overflow_count", "max_bucket_load"):
            value = record["metrics"][field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"attempt metric {field} differs")
        validate_exact_keys(
            record["metrics"]["component_losses"],
            ("task_loss", "internal_router_loss", "supervised_route_loss"),
            "attempt component losses",
        )
        for value in record["metrics"]["component_losses"].values():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise ContractError("attempt component loss differs")
        learning_rates = record["metrics"]["learning_rates"]
        if not isinstance(learning_rates, list):
            raise ContractError("learning-rate records differ")
        group_names = []
        for learning_rate in learning_rates:
            validate_exact_keys(learning_rate, ("parameter_group", "learning_rate"), "learning-rate record")
            if not isinstance(learning_rate["parameter_group"], str) or not learning_rate["parameter_group"]:
                raise ContractError("parameter-group identity differs")
            group_names.append(learning_rate["parameter_group"])
            rate = learning_rate["learning_rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(float(rate)) or rate < 0:
                raise ContractError("learning rate differs")
        if group_names != sorted(group_names) or len(set(group_names)) != len(group_names):
            raise ContractError("learning-rate groups are not sorted and unique")


def validate_attempt_sequence(rows: Sequence[Mapping[str, Any]], require_complete: bool = False) -> None:
    expected_sequence = 0
    index = 0
    logical_identities: set[tuple[Any, ...]] = set()
    while index < len(rows):
        row = rows[index]
        validate_attempt_row(row)
        if row["event_sequence"] != expected_sequence:
            raise ContractError("attempt event sequence is not zero-based contiguous")
        if row["event"] != "started":
            raise ContractError("attempt pair must begin with started")
        identity = tuple(row[field] for field in ("run_id", "rung", "construction_seed", "model", "stage", "logical_update"))
        if identity in logical_identities:
            raise ContractError("logical update or attempt identity is reused")
        logical_identities.add(identity)
        if index + 1 >= len(rows):
            if require_complete:
                raise ContractError("attempt ledger ends with unpaired start")
            return
        completed = rows[index + 1]
        validate_attempt_row(completed)
        if completed["event_sequence"] != expected_sequence + 1 or completed["event"] != "completed":
            raise ContractError("attempt completed event is not adjacent")
        for field in ATTEMPT_PAIR_EQUAL_FIELDS:
            if row[field] != completed[field]:
                raise ContractError(f"attempt pair field differs: {field}")
        expected_sequence += 2
        index += 2


def derive_attempt_accounting(rows: Sequence[Mapping[str, Any]]) -> AttemptAccounting:
    if rows:
        validate_attempt_sequence(rows, require_complete=False)
    attempted_updates = 0
    completed_updates = 0
    attempted_token_positions = 0
    completed_token_positions = 0
    attempted_seeds: set[int] = set()
    completed_seeds: set[int] = set()
    last: dict[int, int] = {}
    unpaired: list[str] = []
    index = 0
    while index < len(rows):
        started = rows[index]
        attempted_updates += 1
        attempted_token_positions += int(started["token_positions"])
        attempted_seeds.add(int(started["construction_seed"]))
        last[int(started["construction_seed"])] = int(started["event_sequence"])
        if index + 1 >= len(rows):
            unpaired.append(str(started["attempt_id"]))
            break
        completed = rows[index + 1]
        completed_updates += 1
        completed_token_positions += int(started["token_positions"])
        completed_seeds.add(int(started["construction_seed"]))
        last[int(completed["construction_seed"])] = int(completed["event_sequence"])
        index += 2
    return AttemptAccounting(
        attempted_updates,
        completed_updates,
        attempted_token_positions,
        completed_token_positions,
        tuple(sorted(attempted_seeds)),
        tuple(sorted(completed_seeds)),
        dict(sorted(last.items())),
        tuple(unpaired),
    )


def _torch_module(torch_module: Any | None = None) -> Any:
    return importlib.import_module("torch") if torch_module is None else torch_module


def generate_rung_one_batch(seed: int, batch_size: int, torch_module: Any | None = None) -> dict[str, Any]:
    torch = _torch_module(torch_module)
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ContractError("rung-one generator inputs differ")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens = torch.randint(0, 32, (batch_size, 128), generator=generator, dtype=torch.int64)
    rule_rows = []
    answer_rows = []
    condition_rows = []
    target_rows = []
    source_rows = []
    for row_index in range(batch_size):
        rule_blocks = torch.randperm(10, generator=generator)[:4]
        answer_indices = torch.randperm(16, generator=generator)[:4]
        condition = torch.randint(0, 4, (1,), generator=generator, dtype=torch.int64)
        for rule_index in range(4):
            tokens[row_index, 8 * int(rule_blocks[rule_index])] = 64 + 16 * rule_index + int(answer_indices[rule_index])
        tokens[row_index, 80] = 32 + int(condition[0])
        tokens[row_index, 126] = 36
        condition_value = int(condition[0])
        rule_rows.append(rule_blocks.tolist())
        answer_rows.append(answer_indices.tolist())
        condition_rows.append(condition_value)
        target_rows.append(40 + int(answer_indices[condition_value]))
        source_rows.append(int(rule_blocks[condition_value]))
    payload = {
        "tokens": tokens.tolist(),
        "targets": target_rows,
        "condition": condition_rows,
        "rule_blocks": rule_rows,
        "answer_indices": answer_rows,
        "required_source": source_rows,
    }
    return payload


def generate_rung_two_batch(seed: int, batch_size: int, torch_module: Any | None = None) -> dict[str, Any]:
    torch = _torch_module(torch_module)
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ContractError("rung-two generator inputs differ")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tokens = torch.randint(40, 256, (batch_size, 512), generator=generator, dtype=torch.int64)
    counts = []
    positions = []
    for row_index in range(batch_size):
        count = int(torch.randint(0, 8, (1,), generator=generator, dtype=torch.int64)[0])
        permutation = torch.randperm(64, generator=generator)
        selected = permutation[:count]
        if count:
            tokens[row_index, selected] = 35
        counts.append(count)
        positions.append(selected.tolist())
    tokens[:, 510] = 36
    targets = 19 + torch.cumsum(tokens == 35, dim=1)
    payload = {
        "tokens": tokens.tolist(),
        "targets": targets.tolist(),
        "count": counts,
        "count_positions": positions,
    }
    return payload


def generate_random_routes(seed: int, batch_size: int, torch_module: Any | None = None) -> dict[str, Any]:
    torch = _torch_module(torch_module)
    if isinstance(seed, bool) or not isinstance(seed, int) or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ContractError("random-route generator inputs differ")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    routes = torch.full((batch_size, 128, 1, 2), -1, dtype=torch.long, device="cpu")
    query_routes = []
    for row_index in range(batch_size):
        selected = torch.sort(torch.randperm(15, generator=generator)[:2]).values
        routes[row_index, 126, 0] = selected
        query_routes.append(selected.tolist())
    payload = {"routes": query_routes}
    return payload


def generate_source_exclusion_routes(
    seed: int,
    raw: Any,
    source: Any,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    torch = _torch_module(torch_module)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractError("source-exclusion seed differs")
    raw_values = raw.tolist() if hasattr(raw, "tolist") else raw
    source_values = source.tolist() if hasattr(source, "tolist") else source
    query_shape_valid = isinstance(raw_values, list) and all(
        isinstance(row, list)
        and len(row) == 2
        and all(not isinstance(value, bool) and isinstance(value, int) for value in row)
        for row in raw_values
    )
    full_shape_valid = isinstance(raw_values, list) and all(
        isinstance(row, list)
        and len(row) == 128
        and all(
            isinstance(position, list)
            and len(position) == 1
            and isinstance(position[0], list)
            and len(position[0]) == 2
            and all(not isinstance(value, bool) and isinstance(value, int) for value in position[0])
            for position in row
        )
        for row in raw_values
    )
    if not (query_shape_valid or full_shape_valid):
        raise ContractError("source-exclusion raw values differ")
    if not isinstance(source_values, list) or not all(not isinstance(value, bool) and isinstance(value, int) for value in source_values):
        raise ContractError("source-exclusion source values differ")
    raw_tensor = torch.as_tensor(raw, dtype=torch.long, device="cpu")
    source_tensor = torch.as_tensor(source, dtype=torch.long, device="cpu")
    if raw_tensor.ndim == 4:
        if tuple(raw_tensor.shape[1:]) != (128, 1, 2):
            raise ContractError("source-exclusion raw route shape differs")
        raw_query = raw_tensor[:, 126, 0].clone()
        raw_full = raw_tensor.clone()
    elif raw_tensor.ndim == 2 and raw_tensor.shape[1] == 2:
        raw_query = raw_tensor.clone()
        raw_full = torch.full((raw_tensor.shape[0], 128, 1, 2), -1, dtype=torch.long, device="cpu")
        raw_full[:, 126, 0] = raw_query
    else:
        raise ContractError("source-exclusion raw route shape differs")
    if source_tensor.ndim != 1 or source_tensor.shape[0] != raw_query.shape[0]:
        raise ContractError("source-exclusion source shape differs")
    if bool(((source_tensor < 0) | (source_tensor > 14)).any()):
        raise ContractError("source-exclusion source ID is invalid")
    if bool(((raw_query < -1) | (raw_query > 14)).any()):
        raise ContractError("source-exclusion raw ID is invalid")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    routes = torch.full_like(raw_full, -1)
    selected_rows = []
    for row_index in range(raw_query.shape[0]):
        permutation = torch.randperm(15, generator=generator)
        required = int(source_tensor[row_index])
        kept: list[int] = []
        for value in raw_query[row_index].tolist():
            candidate = int(value)
            if candidate == -1 or candidate == required or candidate in kept:
                continue
            if candidate < 0 or candidate > 14:
                raise ContractError("source-exclusion raw ID is invalid")
            kept.append(candidate)
        for value in permutation.tolist():
            if len(kept) == 2:
                break
            candidate = int(value)
            if candidate != required and candidate not in kept:
                kept.append(candidate)
        if len(kept) != 2 or len(set(kept)) != 2 or required in kept:
            raise ContractError("source-exclusion route construction failed")
        routes[row_index, 126, 0] = torch.tensor(kept, dtype=torch.long)
        selected_rows.append(kept)
    payload = {
        "raw": raw_query.tolist(),
        "routes": selected_rows,
        "source": source_tensor.tolist(),
    }
    return payload


def _query_only_raw_routes(raw_remote: Any, torch: Any) -> Any:
    if not isinstance(raw_remote, torch.Tensor) or raw_remote.dtype != torch.long or tuple(raw_remote.shape[1:]) != (128, 1, 2):
        raise ContractError("raw route snapshot differs")
    snapshot = torch.full_like(raw_remote, -1, device="cpu")
    snapshot[:, 126, 0] = raw_remote[:, 126, 0].detach().to(device="cpu")
    return snapshot


def payload_to_tensors(payload: Mapping[str, Any], torch_module: Any | None = None) -> dict[str, Any]:
    torch = _torch_module(torch_module)
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "count_positions":
            result[key] = [torch.tensor(row, dtype=torch.long, device="cpu") for row in value]
        else:
            result[key] = torch.tensor(value, dtype=torch.long, device="cpu")
    return result


def validate_resource_row(record: Mapping[str, Any]) -> None:
    validate_exact_keys(record, RESOURCE_ROW_KEYS, "resource row")
    if record["schema_version"] != SCHEMA_VERSION or record["phase"] not in {"pilot", "claim"} or record["parser_status"] != "pass":
        raise ContractError("resource row identity differs")
    for field in ("run_id", "wall_time_utc"):
        if not isinstance(record[field], str) or not record[field]:
            raise ContractError(f"resource row {field} differs")
    for field in ("sample_id", "monotonic_ns", "aggregate_rss_bytes", "aggregate_cpu_time_us", "swap_used_bytes", "swap_growth_bytes", "attempted_updates", "token_positions"):
        if isinstance(record[field], bool) or not isinstance(record[field], int) or record[field] < 0:
            raise ContractError(f"resource row {field} differs")
    expected_pids = record["expected_pids"]
    if not isinstance(expected_pids, list) or any(isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 for pid in expected_pids):
        raise ContractError("resource expected PID type differs")
    if expected_pids != sorted(expected_pids) or len(expected_pids) != len(set(expected_pids)):
        raise ContractError("resource expected PIDs differ")
    processes = record["processes"]
    if not isinstance(processes, list):
        raise ContractError("resource processes differ")
    if [entry.get("pid") for entry in processes] != expected_pids:
        raise ContractError("resource process order differs")
    for entry in processes:
        validate_exact_keys(entry, ("pid", "ppid", "rss_bytes", "cpu_time_us"), "resource process")
        if isinstance(entry["pid"], bool) or not isinstance(entry["pid"], int) or entry["pid"] < 1:
            raise ContractError("resource process PID differs")
        for field in ("ppid", "rss_bytes", "cpu_time_us"):
            if isinstance(entry[field], bool) or not isinstance(entry[field], int) or entry[field] < 0:
                raise ContractError(f"resource process {field} differs")
    active_jobs = record["active_jobs"]
    if not isinstance(active_jobs, list):
        raise ContractError("resource active jobs differ")
    workers = []
    for entry in active_jobs:
        validate_exact_keys(entry, ("worker", "seed", "stage", "logical_update"), "resource active job")
        if not isinstance(entry["worker"], str) or not entry["worker"]:
            raise ContractError("resource active worker differs")
        workers.append(entry["worker"])
        for field in ("seed", "logical_update"):
            value = entry[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ContractError(f"resource active {field} differs")
        if entry["stage"] is not None and (not isinstance(entry["stage"], str) or not entry["stage"]):
            raise ContractError("resource active stage differs")
    if workers != sorted(workers) or len(set(workers)) != len(workers):
        raise ContractError("resource active jobs are not worker-sorted and unique")
    if record["aggregate_rss_bytes"] != sum(entry["rss_bytes"] for entry in processes):
        raise ContractError("resource RSS aggregate differs")
    if record["aggregate_cpu_time_us"] != sum(entry["cpu_time_us"] for entry in processes):
        raise ContractError("resource CPU aggregate differs")


def validate_resource_timeline(
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    require_clean_final: bool = False,
    pilot_final_values: tuple[int, int] | None = None,
) -> None:
    if phase not in {"pilot", "claim"} or type(require_clean_final) is not bool:
        raise ContractError("resource timeline phase differs")
    if pilot_final_values is None:
        expected_pilot_final = (88, 225280)
    elif phase != "pilot" or not isinstance(pilot_final_values, tuple) or len(pilot_final_values) != 2 or any(type(value) is not int or value < 0 for value in pilot_final_values):
        raise ContractError("pilot resource timeline final contract differs")
    else:
        expected_pilot_final = pilot_final_values
    if not rows:
        if require_clean_final:
            raise ContractError("clean resource timeline is empty")
        return
    run_id = rows[0].get("run_id")
    prior_monotonic = -1
    prior_wall: datetime_module.datetime | None = None
    prior_attempted = -1
    prior_tokens = -1
    prior_process_cpu: dict[int, int] = {}
    prior_processes: dict[int, Mapping[str, Any]] = {}
    terminal_process_cpu: dict[int, int] = {}
    for row in rows:
        validate_resource_row(row)
    parent_pid = rows[0]["processes"][0]["pid"] if len(rows[0]["processes"]) == 1 else None
    swap_baselines = {
        row["swap_used_bytes"] - row["swap_growth_bytes"]
        for row in rows
        if row["swap_growth_bytes"] > 0
    }
    if len(swap_baselines) > 1 or swap_baselines and min(swap_baselines) < 0:
        raise ContractError("resource timeline swap growth differs")
    baseline_swap = next(iter(swap_baselines)) if swap_baselines else None
    for index, row in enumerate(rows):
        if row["run_id"] != run_id or row["phase"] != phase or row["sample_id"] != index:
            raise ContractError("resource timeline identity or sample sequence differs")
        wall = _strict_utc(row["wall_time_utc"])
        if row["monotonic_ns"] <= prior_monotonic or (prior_wall is not None and wall < prior_wall):
            raise ContractError("resource timeline time order differs")
        if index > 0 and row["monotonic_ns"] - prior_monotonic < RESOURCE_SAMPLE_INTERVAL_NS:
            raise ContractError("resource timeline sample interval differs")
        if row["attempted_updates"] < prior_attempted or row["token_positions"] < prior_tokens:
            raise ContractError("resource timeline counters decrease")
        if baseline_swap is not None and row["swap_growth_bytes"] != max(0, row["swap_used_bytes"] - baseline_swap):
            raise ContractError("resource timeline swap growth differs")
        current_processes = {entry["pid"]: entry for entry in row["processes"]}
        current_cpu = {pid: entry["cpu_time_us"] for pid, entry in current_processes.items()}
        if set(current_cpu) & set(terminal_process_cpu):
            raise ContractError("resource process reappears after terminal disappearance")
        next_process_cpu = {}
        for pid in set(prior_process_cpu) & set(current_cpu):
            if current_cpu[pid] < prior_process_cpu[pid]:
                prior_process = prior_processes[pid]
                current_process = current_processes[pid]
                if parent_pid is not None and pid != parent_pid and prior_process["ppid"] == parent_pid and current_process["ppid"] == parent_pid and prior_process["rss_bytes"] > 0 and prior_process["cpu_time_us"] > 0 and current_process["rss_bytes"] == 0 and current_process["cpu_time_us"] == 0:
                    terminal_process_cpu[pid] = prior_process_cpu[pid]
                    continue
                raise ContractError("resource process CPU time decreases")
        for pid, cpu_time_us in current_cpu.items():
            if pid not in terminal_process_cpu:
                next_process_cpu[pid] = cpu_time_us
        prior_monotonic = row["monotonic_ns"]
        prior_wall = wall
        prior_attempted = row["attempted_updates"]
        prior_tokens = row["token_positions"]
        prior_process_cpu = next_process_cpu
        prior_processes = {pid: process for pid, process in current_processes.items() if pid in next_process_cpu}
    if rows[0]["sample_id"] != 0 or rows[0]["active_jobs"] or rows[0]["attempted_updates"] != 0 or rows[0]["token_positions"] != 0:
        raise ContractError("resource timeline baseline differs")
    if require_clean_final:
        if terminal_process_cpu:
            raise ContractError("clean resource timeline has terminal process disappearance")
        final = rows[-1]
        if final["active_jobs"]:
            raise ContractError("clean resource timeline final row has active jobs")
        if phase == "pilot" and (final["attempted_updates"], final["token_positions"]) != expected_pilot_final:
            raise ContractError("pilot resource timeline final counters differ")
        if phase == "claim" and (final["attempted_updates"] != 20736 or final["token_positions"] != 45613056):
            raise ContractError("claim resource timeline final counters differ")
        if phase == "claim" and claim_resource_observations(rows):
            raise ContractError("clean claim resource timeline crossed an abort threshold")


def next_resource_sample_monotonic_ns(row: Mapping[str, Any]) -> int:
    validate_resource_row(row)
    return row["monotonic_ns"] + RESOURCE_SAMPLE_INTERVAL_NS


def claim_resource_observations(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    validate_resource_timeline(rows, "claim", require_clean_final=False)
    resident_strikes = 0
    last_resident_strike_ns: int | None = None
    resident_sample_id: int | None = None
    swap_sample_id: int | None = None
    for row in rows:
        if row["aggregate_rss_bytes"] > CLAIM_RSS_LIMIT_BYTES and (
            last_resident_strike_ns is None or row["monotonic_ns"] - last_resident_strike_ns >= RESOURCE_SAMPLE_INTERVAL_NS
        ):
            resident_strikes += 1
            last_resident_strike_ns = row["monotonic_ns"]
            if resident_strikes == 3:
                resident_sample_id = row["sample_id"]
        if swap_sample_id is None and row["swap_growth_bytes"] > CLAIM_SWAP_GROWTH_LIMIT_BYTES:
            swap_sample_id = row["sample_id"]
    observations = []
    if resident_sample_id is not None:
        observations.append({"reason_code": "resident_memory", "context": {"sample_id": resident_sample_id}})
    if swap_sample_id is not None:
        observations.append({"reason_code": "swap_growth", "context": {"sample_id": swap_sample_id}})
    return tuple(observations)


def _parse_cpu_time_us(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    if match is None:
        raise ContractError("CPU time format differs")
    days_text, hours_text, minutes_text, seconds_text = match.groups()
    days = int(days_text or 0)
    hours = int(hours_text or 0)
    minutes = int(minutes_text)
    seconds = decimal.Decimal(seconds_text)
    if minutes >= 60 or seconds >= 60 or (days_text is not None and hours_text is None) or (days_text is None and hours_text is None and value.count(":") != 1):
        raise ContractError("CPU time range differs")
    total = (decimal.Decimal(days * 86400 + hours * 3600 + minutes * 60) + seconds) * decimal.Decimal(1_000_000)
    return int(total.quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


def parse_ps_output(output: str, expected_pids: Sequence[int]) -> list[dict[str, int]]:
    expected = sorted(int(pid) for pid in expected_pids)
    if len(expected) != len(set(expected)):
        raise ContractError("expected PID set contains duplicates")
    records: dict[int, dict[str, int]] = {}
    lines = [line for line in output.splitlines() if line.strip()]
    for line in lines:
        fields = line.split()
        if len(fields) != 4:
            raise ContractError("ps row field count differs")
        try:
            pid = int(fields[0], 10)
            ppid = int(fields[1], 10)
            rss_kib = int(fields[2], 10)
        except ValueError as exc:
            raise ContractError("ps integer field differs") from exc
        if pid < 1 or ppid < 0 or rss_kib < 0 or pid in records:
            raise ContractError("ps PID or RSS differs")
        records[pid] = {
            "pid": pid,
            "ppid": ppid,
            "rss_bytes": rss_kib * 1024,
            "cpu_time_us": _parse_cpu_time_us(fields[3]),
        }
    if sorted(records) != expected:
        raise ProcessSetMismatch(expected, sorted(records))
    return [records[pid] for pid in expected]


def parse_swap_output(output: str) -> int:
    matches = re.findall(r"(?<!\S)used\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGTP])(?=\s|$)", output)
    if len(matches) != 1:
        raise ContractError("swap used field differs")
    number, unit = matches[0]
    exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit]
    value = decimal.Decimal(number) * (decimal.Decimal(1024) ** exponent)
    if value < 0:
        raise ContractError("swap use is negative")
    return int(value.quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


def sample_swap(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> int:
    completed = run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        raise ContractError("swap command failed")
    return parse_swap_output(completed.stdout)


def sample_processes(expected_pids: Sequence[int], run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[dict[str, int]]:
    expected = sorted(int(pid) for pid in expected_pids)
    pid_csv = ",".join(str(pid) for pid in expected)
    completed = run(
        ["/bin/ps", "-o", "pid=,ppid=,rss=,time=", "-p", pid_csv],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if completed.returncode != 0:
        raise ContractError("ps command failed")
    return parse_ps_output(completed.stdout, expected)


def _remove_tree_and_fsync(path: Path) -> None:
    if not os.path.lexists(path):
        return
    parent = path.parent
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    fsync_directory(parent)


def _rollback_published_root(final_root: Path, staging_root: Path, fault: str | None) -> bool:
    try:
        if fault == "reverse_rename_failure":
            raise OSError("injected reverse rename failure")
        os.rename(final_root, staging_root)
        if fault == "rollback_parent_fsync_failure":
            raise OSError("injected rollback parent fsync failure")
        fsync_directory(final_root.parent)
        return True
    except Exception:
        return False


def publish_and_activate(
    staging_root: str | Path,
    final_root: str | Path,
    signals: SignalController,
    registry_activate: Callable[[int, str], None] | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_utc: Callable[[], str] = utc_now,
    fault: str | None = None,
    cleanup_refused_staging: bool = True,
) -> PublicationResult:
    staging = Path(staging_root)
    final = Path(final_root)
    if not staging.is_dir() or os.path.lexists(final) or staging.parent != final.parent:
        raise InitializationRefusal("publication paths differ")
    allowed_faults = {
        None,
        "rename_failure",
        "parent_fsync_failure",
        "origin_capture_failure",
        "activation_failure",
        "reverse_rename_failure",
        "rollback_parent_fsync_failure",
    }
    if fault not in allowed_faults:
        raise ContractError("unknown publication fault")
    signals.defer()
    if signals.pending_signal is not None:
        if cleanup_refused_staging:
            _remove_tree_and_fsync(staging)
        pending = signals.release()
        return PublicationResult("initialization_refusal", final, staging, False, None, None, pending)
    renamed = False
    origin_ns: int | None = None
    origin_utc: str | None = None
    try:
        if fault == "rename_failure":
            raise OSError("injected rename failure")
        os.rename(staging, final)
        renamed = True
        if fault == "parent_fsync_failure":
            raise OSError("injected parent fsync failure")
        fsync_directory(final.parent)
        if fault == "origin_capture_failure":
            raise OSError("injected origin capture failure")
        origin_ns = monotonic_ns()
        origin_utc = wall_utc()
        if isinstance(origin_ns, bool) or not isinstance(origin_ns, int) or origin_ns < 0 or not isinstance(origin_utc, str) or not origin_utc:
            raise ContractError("abort timing origin differs")
        if fault == "activation_failure":
            raise OSError("injected activation failure")
        if fault in {"reverse_rename_failure", "rollback_parent_fsync_failure"}:
            raise OSError("injected preactivation failure for rollback rehearsal")
        if registry_activate is not None:
            registry_activate(origin_ns, origin_utc)
        pending = signals.release()
        return PublicationResult("active", final, staging, True, origin_ns, origin_utc, pending)
    except Exception:
        if renamed:
            if not _rollback_published_root(final, staging, fault):
                signals.release()
                return PublicationResult("orphaned", final, staging, False, None, None, signals.pending_signal)
        if cleanup_refused_staging:
            try:
                _remove_tree_and_fsync(staging)
            except Exception:
                signals.release()
                return PublicationResult("orphaned", final, staging, False, None, None, signals.pending_signal)
        pending = signals.release()
        return PublicationResult("initialization_refusal", final, staging, False, None, None, pending)


def _resource_baseline_row(
    run_id: str,
    phase: str,
    swap_baseline: int,
    processes: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    process_records = [dict(record) for record in processes]
    process_records.sort(key=lambda record: record["pid"])
    expected_pids = [record["pid"] for record in process_records]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sample_id": 0,
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": utc_now(),
        "expected_pids": expected_pids,
        "processes": process_records,
        "active_jobs": [],
        "aggregate_rss_bytes": sum(record["rss_bytes"] for record in process_records),
        "aggregate_cpu_time_us": sum(record["cpu_time_us"] for record in process_records),
        "swap_used_bytes": swap_baseline,
        "swap_growth_bytes": 0,
        "parser_status": "pass",
        "attempted_updates": 0,
        "token_positions": 0,
    }


def precreate_pilot_timeline(
    run_root: str | Path,
    signals: SignalController,
    swap_reader: Callable[[], int],
    first_row: Mapping[str, Any] | None = None,
    fault: str | None = None,
    process_sampler: Callable[[Sequence[int]], Sequence[Mapping[str, int]]] = sample_processes,
) -> TransitionResult:
    root = Path(run_root)
    path = root / "run" / "pilot_resources.jsonl"
    writer = CrashAtomicJsonlWriter(path, validate_resource_row, signals)
    allowed_faults = {None, "creation_failure", "baseline_failure", "pending_signal_after_baseline"}
    if fault not in allowed_faults:
        raise ContractError("unknown pilot transition fault")
    signals.defer()
    durable = False
    try:
        if fault == "creation_failure":
            raise OSError("injected pilot timeline creation failure")
        writer.precreate()
        durable = True
        if fault == "baseline_failure":
            raise OSError("injected pilot baseline failure")
        baseline = swap_reader()
        if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
            raise ContractError("pilot swap baseline differs")
        if fault == "pending_signal_after_baseline":
            signals.inject()
        pending = signals.pending_signal is not None
        signals.release()
        if pending:
            return TransitionResult("pilot", "pilot_abort", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": writer}, baseline, "signal_or_interruption")
        if first_row is None:
            try:
                sampled_processes = process_sampler([os.getpid()])
                row = _resource_baseline_row(root.name, "pilot", baseline, sampled_processes)
            except BaseException:
                reason = "signal_or_interruption" if signals.pending_signal is not None else "resource_sampler_failure"
                return TransitionResult("pilot", "pilot_abort", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": writer}, baseline, reason)
        else:
            row = dict(first_row)
        result = writer.append(row)
        if not result.acknowledged:
            return TransitionResult("pilot", "pilot_abort", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": writer}, baseline, result.reason_code)
        return TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": writer}, baseline, None)
    except UnrecoverableOrphan:
        if signals.deferred:
            signals.release()
        raise
    except Exception as exc:
        if signals.deferred:
            pending = signals.pending_signal is not None
            if not durable:
                try:
                    if writer._owned_identity is not None:
                        writer.remove_owned_path()
                    elif os.path.lexists(path):
                        raise UnrecoverableOrphan("pilot timeline path is not owned")
                except BaseException as cleanup_exc:
                    signals.release()
                    raise UnrecoverableOrphan("pilot timeline rollback failed") from cleanup_exc
                signals.release()
                reason = "signal_or_interruption" if pending else "artifact_inconsistency"
                return TransitionResult("prepilot", "prepilot_abort", tuple(), {}, None, reason)
            signals.release()
            reason = "signal_or_interruption" if pending else "resource_sampler_failure"
            return TransitionResult("pilot", "pilot_abort", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": writer}, None, reason)
        raise


def precreate_claim_ledgers(
    run_root: str | Path,
    signals: SignalController,
    swap_reader: Callable[[], int],
    fail_after: int | None = None,
    baseline_failure: bool = False,
    pending_signal_after_baseline: bool = False,
) -> TransitionResult:
    root = Path(run_root)
    pilot_path = root / "run" / "pilot.json"
    if not pilot_path.is_file():
        raise ContractError("durable proceeding pilot record is required")
    pilot_path = validate_real_regular_file(pilot_path)
    pilot_identity = _filesystem_identity(pilot_path)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("decision") != "proceed":
        raise ContractError("pilot decision must be proceed")
    signals.defer()
    writers: dict[str, CrashAtomicJsonlWriter] = {}
    created_directories: list[Path] = []
    created_directory_identities: dict[Path, tuple[int, int]] = {}
    current_writer: CrashAtomicJsonlWriter | None = None
    try:
        for index, relative in enumerate(CLAIM_LEDGER_PATHS):
            path = root / relative
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=False)
                created_directories.append(path.parent)
                created_directory_identities[path.parent] = _filesystem_identity(path.parent)
                fsync_directory(path.parent.parent)
                fsync_directory(path.parent)
            writer = CrashAtomicJsonlWriter(
                path,
                validate_resource_row if relative == "run/resources.jsonl" else validate_attempt_row,
                signals,
                None if relative == "run/resources.jsonl" else "attempt",
            )
            current_writer = writer
            writer.precreate()
            writers[relative] = writer
            current_writer = None
            if fail_after is not None and index + 1 == fail_after:
                raise OSError("injected partial claim-ledger transition failure")
        if baseline_failure:
            raise ContractError("injected claim baseline failure")
        baseline = swap_reader()
        if isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0:
            raise ContractError("claim swap baseline differs")
        if pending_signal_after_baseline:
            signals.inject()
        pending = signals.pending_signal is not None
        signals.release()
        retained = tuple(CLAIM_LEDGER_PATHS)
        if pending:
            return TransitionResult("claim", "claim_abort", retained, writers, baseline, "signal_or_interruption")
        return TransitionResult("claim", "ready", retained, writers, baseline, None)
    except UnrecoverableOrphan:
        if signals.deferred:
            signals.release()
        raise
    except Exception as exc:
        pending = signals.pending_signal is not None
        if len(writers) == len(CLAIM_LEDGER_PATHS):
            signals.release()
            reason = "signal_or_interruption" if pending else "resource_sampler_failure"
            return TransitionResult("claim", "claim_abort", tuple(CLAIM_LEDGER_PATHS), writers, None, reason)
        try:
            rollback_writers = [*writers.values()]
            if current_writer is not None and current_writer._owned_identity is not None:
                rollback_writers.append(current_writer)
            if current_writer is not None and current_writer._owned_identity is None and os.path.lexists(current_writer.path):
                raise UnrecoverableOrphan("claim ledger path is not owned")
            for writer in rollback_writers:
                writer.validate_owned_path()
            _validate_owned_path(pilot_path, pilot_identity)
            _validate_owned_paths(created_directory_identities)
            for directory in created_directories:
                allowed = {writer.path for writer in rollback_writers if writer.path.parent == directory}
                if set(directory.iterdir()) != allowed:
                    raise UnrecoverableOrphan("claim directory ownership is ambiguous")
            for writer in reversed(rollback_writers):
                writer.remove_owned_path()
            for directory in sorted(set(created_directories), key=lambda value: len(value.parts), reverse=True):
                if directory.exists() and not any(directory.iterdir()):
                    _validate_owned_path(directory, created_directory_identities[directory])
                    directory.rmdir()
                    fsync_directory(directory.parent)
            _unlink_owned_path(pilot_path, pilot_identity)
        except BaseException as cleanup_exc:
            signals.release()
            raise UnrecoverableOrphan("claim transition rollback failed") from cleanup_exc
        signals.release()
        reason = "signal_or_interruption" if pending else "artifact_inconsistency"
        retained = ("run/pilot_resources.jsonl",)
        return TransitionResult("pilot", "pilot_abort", retained, {}, None, reason)


def _iter_regular_files(run_root: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for path in run_root.rglob("*"):
        if path.is_symlink():
            raise ContractError("symlinks are forbidden in terminal closure")
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        if relative == "SHA256SUMS":
            continue
        if relative.startswith("/") or ".." in Path(relative).parts or "\r" in relative or "\n" in relative:
            raise ContractError("terminal closure path differs")
        records.append((relative, path))
    records.sort(key=lambda item: item[0].encode("utf-8"))
    if len(records) != len({relative for relative, _ in records}):
        raise ContractError("terminal closure contains duplicate paths")
    return records


def _collect_detail_digests(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "details_sha256":
                if not isinstance(child, str) or re.fullmatch(r"[0-9a-f]{64}", child) is None:
                    raise ContractError("referenced check detail digest differs")
                output.add(child)
            else:
                _collect_detail_digests(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_detail_digests(child, output)


def _closure_expected_paths(run_root: Path, payload: Mapping[str, Any], closure_kind: str) -> set[str]:
    aliases = {"initialization": "initialization_absent", "clean_completion": "clean"}
    kind = aliases.get(closure_kind, closure_kind)
    allowed = {"initialization_absent", "prepilot_abort", "pilot_abort", "pilot_stop", "claim_abort", "clean"}
    if kind not in allowed:
        raise ContractError("artifact closure kind differs")
    if kind == "initialization_absent":
        return set()
    closures = payload["artifacts"]["artifact_closures"]
    base = set(closures["five_global_control_files"])
    base.add(LAUNCH_PROJECT_PLAN_PATH)
    base.add("run/sentinels/selected_attention_oracle_payload.json")
    config_path = run_root / "run" / "config_manifest.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise ContractError("closure config manifest is absent")
    config = json.loads(config_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    review_records = config.get("review_records")
    if not isinstance(review_records, list) or len(review_records) != 4:
        raise ContractError("closure review registry differs")
    reviews = {f"run/reviews/{record['artifact_sha256']}.json" for record in review_records}
    if len(reviews) != 4:
        raise ContractError("closure review identity differs")
    referenced = set()
    reference_paths = [run_root / "run" / "preflight.json"]
    if kind in {"pilot_stop", "claim_abort", "clean"}:
        reference_paths.append(run_root / "run" / "pilot.json")
    if kind == "clean":
        reference_paths.extend(run_root / "rung1" / str(seed) / "parity.json" for seed in RUNG_ONE_SEEDS)
        reference_paths.append(run_root / "rung2" / "83" / "parity.json")
    for path in reference_paths:
        if not path.is_file() or path.is_symlink():
            raise ContractError("closure detail reference artifact is absent")
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        _collect_detail_digests(value, referenced)
    details = {f"run/check_details/{digest}.json" for digest in referenced}
    expected = base | reviews | details
    if kind == "prepilot_abort":
        expected.update(("ABORTED.json", "SHA256SUMS"))
    elif kind == "pilot_abort":
        expected.update(("run/pilot_resources.jsonl", "ABORTED.json", "SHA256SUMS"))
    elif kind == "pilot_stop":
        expected.update(("run/pilot_resources.jsonl", "run/pilot.json", "SHA256SUMS"))
    elif kind == "claim_abort":
        expected.update(("run/pilot_resources.jsonl", "run/pilot.json", *CLAIM_LEDGER_PATHS, "ABORTED.json", "SHA256SUMS"))
        aborted = _canonical_json_artifact(validate_real_regular_file(run_root / "ABORTED.json"))
        training_start_state = aborted.get("training_start_state") if isinstance(aborted, Mapping) else None
        expected.update(_training_start_closure_paths(run_root, training_start_state))
    elif kind == "clean":
        expected.update(_training_start_closure_paths(run_root, "started"))
        expected.update(closures["clean_completion"]["global_files"])
        expected.update(closures["fixed_data_artifacts"])
        for seed in closures["rung_one_construction_seeds"]:
            expected.update(f"rung1/{seed}/{suffix}" for suffix in closures["rung_one_clean_file_suffixes_per_seed"])
        expected.update(f"rung2/83/{suffix}" for suffix in closures["rung_two_clean_file_suffixes"])
        expected.update(closures["clean_completion"]["final_files"])
        fixed_without_details = len(expected - details)
        if fixed_without_details != closures["clean_completion"]["fixed_file_count_excluding_referenced_check_detail_files"]:
            raise ContractError("clean closure fixed cardinality differs")
    return expected


def _training_start_closure_paths(run_root: Path, state: Any) -> set[str]:
    observed_state, link = _training_start_state(run_root)
    if observed_state != state:
        raise ContractError("training-start closure state differs")
    if state == "not_started":
        return set()
    if state == "awaiting_review":
        return {TRAINING_START_REQUEST_PATH}
    if state in {"reviewed_ready", "started"}:
        if link is None:
            raise ContractError("training-start closure linkage differs")
        return {TRAINING_START_REQUEST_PATH, TRAINING_START_PROJECT_PLAN_PATH, TRAINING_START_LINK_PATH, link["review_path"]}
    raise ContractError("training-start closure state differs")


def validate_artifact_closure(run_root: Path, payload: Mapping[str, Any], closure_kind: str) -> tuple[str, ...]:
    expected = _closure_expected_paths(run_root, payload, closure_kind)
    if closure_kind in {"initialization", "initialization_absent"}:
        if os.path.lexists(run_root):
            raise ContractError("initialization refusal left a result")
        return ()
    actual = set()
    for path in run_root.rglob("*"):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError("artifact closure contains symbolic link")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("artifact closure contains non-regular file")
        relative = path.relative_to(run_root).as_posix()
        if relative.startswith("/") or ".." in Path(relative).parts or "\r" in relative or "\n" in relative:
            raise ContractError("artifact closure path differs")
        actual.add(relative)
    terminal = "SHA256SUMS" in actual
    expected_now = expected if terminal else expected - {"SHA256SUMS"}
    if actual != expected_now:
        raise ContractError("artifact closure file set differs")
    if terminal:
        checksum_path = run_root / "SHA256SUMS"
        lines = checksum_path.read_text(encoding="ascii").splitlines(keepends=True)
        covered = sorted(expected - {"SHA256SUMS"}, key=lambda value: value.encode("utf-8"))
        expected_lines = [f"{sha256_file(run_root / relative)}  {relative}\n" for relative in covered]
        if lines != expected_lines:
            raise ContractError("artifact closure checksum differs")
    return tuple(sorted(expected - {"SHA256SUMS"}, key=lambda value: value.encode("utf-8")))


def _validate_checksum_snapshot(run_root: Path, persisted: bytes, expected_paths: Sequence[str]) -> None:
    records = _iter_regular_files(run_root)
    actual_paths = tuple(relative for relative, _ in records)
    if actual_paths != tuple(expected_paths):
        raise ContractError("terminal closure changed during checksum packaging")
    expected = "".join(f"{sha256_file(path)}  {relative}\n" for relative, path in records).encode("ascii")
    if persisted != expected:
        raise ContractError("persisted SHA256SUMS bytes differ")
    try:
        lines = persisted.decode("ascii").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise ContractError("persisted SHA256SUMS encoding differs") from exc
    parsed_paths = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)\n", line)
        if match is None:
            raise ContractError("persisted SHA256SUMS line differs")
        digest, relative = match.groups()
        if relative.startswith("/") or ".." in Path(relative).parts or sha256_file(run_root / relative) != digest:
            raise ContractError("persisted SHA256SUMS coverage differs")
        parsed_paths.append(relative)
    if tuple(parsed_paths) != tuple(expected_paths) or len(parsed_paths) != len(set(parsed_paths)):
        raise ContractError("persisted SHA256SUMS path set differs")


class ChecksumTerminalizer:
    def __init__(self, run_root: str | Path, signals: SignalController | None = None) -> None:
        self.run_root = Path(run_root)
        self.signals = signals
        self.terminal = False

    def finalize(
        self,
        expected_paths: Iterable[str] | None = None,
        fault_hook: Callable[[str], None] | None = None,
        preserve_primary: bool = False,
    ) -> TerminalResult:
        if self.terminal or type(preserve_primary) is not bool:
            raise ContractError("terminal closure cannot be entered twice")
        checksum_path = self.run_root / "SHA256SUMS"
        if os.path.lexists(checksum_path):
            raise ContractError("SHA256SUMS already exists")
        records = _iter_regular_files(self.run_root)
        actual_paths = tuple(relative for relative, _ in records)
        if expected_paths is not None:
            expected = tuple(sorted(tuple(expected_paths), key=lambda value: value.encode("utf-8")))
            if actual_paths != expected:
                raise ContractError("terminal closure file set differs")
        data = "".join(f"{sha256_file(path)}  {relative}\n" for relative, path in records).encode("ascii")
        if self.signals is not None:
            self.signals.defer()
        descriptor: int | None = None
        owned_identity: tuple[int, int] | None = None
        created = False
        terminal = False
        try:
            if self.signals is not None and self.signals.pending_signal is not None and not preserve_primary:
                raise HardAbort("signal_or_interruption")
            if fault_hook is not None:
                fault_hook("before_write")
            descriptor = os.open(checksum_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
            try:
                owned_identity = _descriptor_identity(descriptor)
            except BaseException as exc:
                raise UnrecoverableOrphan("checksum identity capture failed") from exc
            written = os.write(descriptor, data)
            if written != len(data):
                raise ContractError("short SHA256SUMS write")
            if fault_hook is not None:
                fault_hook("before_fsync")
                fault_hook("before_initial_fsync")
            os.fsync(descriptor)
            if fault_hook is not None:
                fault_hook("after_fsync_before_terminal_commit")
            if self.signals is not None and self.signals.pending_signal is not None and not preserve_primary:
                raise HardAbort("signal_or_interruption")
            fsync_directory(checksum_path.parent)
            if fault_hook is not None:
                fault_hook("after_directory_fsync_before_terminal_commit")
            if self.signals is not None and self.signals.pending_signal is not None and not preserve_primary:
                raise HardAbort("signal_or_interruption")
            if fault_hook is not None:
                fault_hook("before_readback")
            persisted = os.pread(descriptor, len(data) + 1, 0)
            _validate_checksum_snapshot(self.run_root, persisted, actual_paths)
            if fault_hook is not None:
                fault_hook("after_readback")
                fault_hook("before_terminal_fsync")
            if self.signals is not None and self.signals.pending_signal is not None and not preserve_primary:
                raise HardAbort("signal_or_interruption")
            def terminal_boundary() -> None:
                os.fsync(descriptor)
                if fault_hook is not None:
                    fault_hook("after_terminal_fsync_before_terminal_commit")

            if self.signals is not None:
                pending_signal = self.signals.commit_terminal(terminal_boundary, preserve_primary=preserve_primary)
                if pending_signal is not None:
                    raise HardAbort("signal_or_interruption", {"signal": pending_signal})
            else:
                terminal_boundary()
            terminal = True
        except BaseException as exc:
            cleanup_failure: BaseException | None = None
            try:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                        descriptor = None
                    except BaseException as close_exc:
                        cleanup_failure = close_exc
                committed = terminal or self.signals is not None and self.signals.terminal
                if not committed:
                    if owned_identity is None:
                        if created:
                            if os.path.lexists(checksum_path):
                                checksum_path.unlink()
                                fsync_directory(checksum_path.parent)
                        elif os.path.lexists(checksum_path):
                            raise UnrecoverableOrphan("checksum path is not owned") from exc
                    else:
                        _unlink_owned_path(checksum_path, owned_identity)
            except BaseException as cleanup_exc:
                if cleanup_failure is None:
                    cleanup_failure = cleanup_exc
            finally:
                if self.signals is not None and self.signals.deferred:
                    self.signals.release()
            if cleanup_failure is not None:
                raise UnrecoverableOrphan("checksum rollback failed") from cleanup_failure
            raise
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException:
                pass
        self.terminal = True
        return TerminalResult(checksum_path, actual_paths, True)


def write_sha256s_terminal(
    run_root: str | Path,
    expected_paths: Iterable[str] | None = None,
    signals: SignalController | None = None,
    fault_hook: Callable[[str], None] | None = None,
    preserve_primary: bool = False,
) -> TerminalResult:
    return ChecksumTerminalizer(run_root, signals).finalize(expected_paths, fault_hook, preserve_primary)


def _repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _manifest_record(path: str, role: str, revision: str) -> dict[str, Any]:
    target = validate_real_regular_file(_repo_path(path))
    digest = sha256_file(target)
    return {
        "path": path,
        "role": role,
        "size_bytes": target.stat().st_size,
        "sha256": digest,
        "revision": revision,
    }


def _config_record(path: str, role: str) -> dict[str, Any]:
    target = validate_real_regular_file(_repo_path(path))
    return {
        "path": path,
        "role": role,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _write_exact_bytes(path: Path, raw: bytes, owned_paths: dict[Path, tuple[int, int]] | None = None) -> None:
    identity_failure: BaseException | None = None
    try:
        with path.open("xb") as handle:
            if owned_paths is not None:
                try:
                    owned_paths[path] = _descriptor_identity(handle.fileno())
                except BaseException as exc:
                    identity_failure = exc
                    raise
            if handle.write(raw) != len(raw):
                raise ContractError("short exact-byte copy")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if identity_failure is not None:
            try:
                if os.path.lexists(path):
                    path.unlink()
                    fsync_directory(path.parent)
            except BaseException as cleanup_exc:
                raise UnrecoverableOrphan("exclusive path identity cleanup failed") from cleanup_exc
            raise UnrecoverableOrphan("exclusive path identity capture failed") from identity_failure
        raise
    fsync_directory(path.parent)


def copy_launch_project_plan_snapshot(staging_root: str | Path) -> str:
    root = Path(staging_root)
    source = validate_real_regular_file(_repo_path(PROJECT_PLAN_RELATIVE_PATH))
    raw = source.read_bytes()
    destination = root / LAUNCH_PROJECT_PLAN_PATH
    _write_exact_bytes(destination, raw)
    digest = hashlib.sha256(raw).hexdigest()
    if destination.read_bytes() != raw or sha256_file(destination) != digest:
        raise ContractError("launch project plan snapshot differs")
    return digest


def _review_scopes() -> tuple[tuple[str, tuple[str, ...]], ...]:
    preregistration = tuple(
        sorted(
            (
                "neuroloc/wiki/PROJECT_PLAN.md",
                "neuroloc/wiki/synthesis/modular_neural_model_stack.md",
                "neuroloc/wiki/tests/modular_sequence_role_cpu_run.md",
                "neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json",
            )
        )
    )
    implementation = tuple(
        sorted(
            (
                "src/model/modular_sources.py",
                "src/model/modular_neural_machine.py",
                "src/model/modular_mlx_backend.py",
                "neuroloc/simulations/memory/modular_sequence_role_cpu.py",
                "neuroloc/simulations/memory/modular_sequence_role_mlx.py",
                "scripts/qualify_modular_mlx.py",
            )
        )
    )
    tests = tuple(sorted(("tests/test_modular_neural_machine.py", "tests/test_modular_sequence_role_cpu.py", "tests/test_modular_sequence_role_mlx.py")))
    complete = tuple(sorted(set(preregistration + implementation + tests)))
    return (
        ("base_preregistration", preregistration),
        ("base_implementation", implementation),
        ("base_tests", tests),
        ("base_complete_surface", complete),
    )


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContractError("file is not regular")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _review_directory_bytes(directory: Path, suffix_pattern: str) -> list[tuple[str, bytes]]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(os.fspath(directory), flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ContractError("review evidence directory is not regular")
        records = []
        for name in sorted(os.listdir(descriptor), key=os.fsencode):
            if re.fullmatch(suffix_pattern, name) is None:
                continue
            child = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), dir_fd=descriptor)
            try:
                if not stat.S_ISREG(os.fstat(child).st_mode):
                    raise ContractError("review evidence entry is not regular")
                chunks = []
                while True:
                    chunk = os.read(child, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                records.append((name, b"".join(chunks)))
            finally:
                os.close(child)
        return records
    finally:
        os.close(descriptor)


def select_and_copy_review_attestations(
    staging_root: str | Path,
    source_directory: str | Path = REVIEW_EVIDENCE_DIRECTORY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(staging_root)
    evidence_directory = Path(source_directory)
    try:
        evidence_records = _review_directory_bytes(evidence_directory, r"[0-9a-f]{64}\.json")
    except (ContractError, OSError) as exc:
        raise InitializationRefusal("review evidence directory is absent or symbolic") from exc
    review_directory = root / "run" / "reviews"
    review_directory.mkdir(parents=True, exist_ok=False)
    fsync_directory(review_directory.parent)
    live_scopes = {}
    for scope, paths in _review_scopes():
        target_records = [{"path": path, "sha256": sha256_file(_repo_path(path))} for path in paths]
        live_scopes[scope] = (target_records, canonical_json_sha256(target_records))
    candidates: dict[str, list[tuple[str, bytes, str]]] = {scope: [] for scope in live_scopes}
    for source_name, raw in evidence_records:
        match = re.fullmatch(r"([0-9a-f]{64})\.json", source_name)
        if match is None:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != match.group(1):
            continue
        try:
            artifact = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except Exception:
            continue
        if canonical_json_bytes(artifact) != raw:
            continue
        try:
            validate_exact_keys(artifact, ("schema_version", "reviewer", "scope", "target_records", "target_sha256", "findings", "finding_count"), "review attestation")
        except ContractError:
            continue
        scope = artifact["scope"]
        if not isinstance(scope, str) or scope not in live_scopes:
            continue
        live_records, live_digest = live_scopes[scope]
        live_paths = {record["path"] for record in live_records}
        target_records = artifact["target_records"]
        record_paths: list[str] = []
        structurally_valid = isinstance(target_records, list)
        if structurally_valid:
            for target in target_records:
                if not isinstance(target, Mapping):
                    structurally_valid = False
                    break
                try:
                    validate_exact_keys(target, ("path", "sha256"), "review target record")
                except ContractError:
                    structurally_valid = False
                    break
                path_value = target["path"]
                digest_value = target["sha256"]
                if (
                    not isinstance(path_value, str)
                    or not path_value
                    or path_value.startswith("/")
                    or "\\" in path_value
                    or any(part in {"", ".", ".."} for part in path_value.split("/"))
                    or Path(path_value).as_posix() != path_value
                    or not isinstance(digest_value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
                ):
                    structurally_valid = False
                    break
                record_paths.append(path_value)
        if structurally_valid and (record_paths != sorted(record_paths) or len(record_paths) != len(set(record_paths))):
            structurally_valid = False
        target_sha_value = artifact["target_sha256"]
        target_sha_valid = isinstance(target_sha_value, str) and re.fullmatch(r"[0-9a-f]{64}", target_sha_value) is not None
        current_looking = target_sha_value == live_digest or (record_paths and set(record_paths) == live_paths)
        if not structurally_valid or not target_sha_valid:
            if current_looking:
                raise InitializationRefusal(f"review scope {scope} has malformed target evidence")
            continue
        target_by_path = {record["path"]: record["sha256"] for record in target_records}
        live_by_path = {record["path"]: record["sha256"] for record in live_records}
        same_members = len(target_records) == len(live_records) and target_by_path == live_by_path
        if same_members and target_records != live_records:
            raise InitializationRefusal(f"review scope {scope} target records are reordered")
        primitive_valid = (
            isinstance(artifact["schema_version"], str)
            and isinstance(artifact["reviewer"], str)
            and isinstance(artifact["findings"], list)
            and type(artifact["finding_count"]) is int
        )
        accepted = (
            primitive_valid
            and artifact["schema_version"] == "todorov.review-attestation.1"
            and artifact["reviewer"] == "feature-dev:code-reviewer"
            and target_records == live_records
            and target_sha_value == live_digest
            and canonical_json_sha256(target_records) == live_digest
            and artifact["findings"] == []
            and artifact["finding_count"] == 0
        )
        if current_looking and not accepted:
            raise InitializationRefusal(f"review scope {scope} has a malformed current attestation")
        if accepted:
            candidates[scope].append((source_name, raw, digest))
    review_records = []
    source_manifest_records = []
    roles = {
        "base_preregistration": "review_attestation_base_preregistration",
        "base_implementation": "review_attestation_base_implementation",
        "base_tests": "review_attestation_base_tests",
        "base_complete_surface": "review_attestation_base_complete_surface",
    }
    for scope, _ in _review_scopes():
        matches = candidates[scope]
        if len(matches) != 1:
            raise InitializationRefusal(f"review scope {scope} has {len(matches)} current attestations")
        source_name, raw, digest = matches[0]
        destination = review_directory / f"{digest}.json"
        with destination.open("xb") as handle:
            if handle.write(raw) != len(raw):
                raise InitializationRefusal("short review attestation copy")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.read_bytes() != raw or sha256_file(destination) != digest:
            raise InitializationRefusal("review attestation copy differs")
        target_records, target_sha256 = live_scopes[scope]
        review_records.append({"reviewer": "feature-dev:code-reviewer", "scope": scope, "target_sha256": target_sha256, "finding_count": 0, "artifact_sha256": digest})
        source_manifest_records.append(
            {
                "path": f"neuroloc/results/modular_sequence_role_mlx_reviews/{source_name}",
                "role": roles[scope],
                "size_bytes": len(raw),
                "sha256": digest,
                "revision": f"sha256:{digest}",
            }
        )
    fsync_directory(review_directory)
    return review_records, source_manifest_records


def validate_base_review_target_binding(
    run_root: Path,
    training_start_state: str,
    training_start_link: Mapping[str, Any] | None,
) -> None:
    if training_start_state not in {"not_started", "awaiting_review", "reviewed_ready", "started"}:
        raise ContractError("base review transition state differs")
    artifacts = []
    for name, raw in _review_directory_bytes(run_root / "run" / "reviews", r"[0-9a-f]{64}\.json"):
        digest = hashlib.sha256(raw).hexdigest()
        if name != f"{digest}.json":
            raise ContractError("base review content address differs")
        artifact = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if canonical_json_bytes(artifact) != raw:
            raise ContractError("base review serialization differs")
        if artifact.get("scope") in dict(_review_scopes()):
            artifacts.append((artifact, digest))
    scopes = dict(_review_scopes())
    if len(artifacts) != 4 or {artifact.get("scope") for artifact, _ in artifacts} != set(scopes):
        raise ContractError("base review scope closure differs")
    base_by_path = {}
    by_scope = {}
    for artifact, digest in artifacts:
        scope = artifact["scope"]
        target_records = artifact.get("target_records")
        if not isinstance(target_records, list) or tuple(record.get("path") for record in target_records if isinstance(record, Mapping)) != scopes[scope]:
            raise ContractError("base review target closure differs")
        if artifact.get("target_sha256") != canonical_json_sha256(target_records):
            raise ContractError("base review target digest differs")
        by_scope[scope] = (artifact, digest)
        for record in target_records:
            path = record["path"]
            digest_value = record["sha256"]
            if path in base_by_path and base_by_path[path] != digest_value:
                raise ContractError("base review target overlap differs")
            base_by_path[path] = digest_value
    if set(base_by_path) != set(scopes["base_complete_surface"]):
        raise ContractError("base review complete target closure differs")
    for path, expected in base_by_path.items():
        observed = hashlib.sha256(_read_regular_bytes(_repo_path(path))).hexdigest()
        if path == PROJECT_PLAN_RELATIVE_PATH and training_start_state == "started":
            if training_start_link is None or observed != training_start_link.get("training_start_project_plan_sha256"):
                raise ContractError("governed project plan transition differs")
        elif observed != expected:
            raise ContractError("base review live target differs")
    config_path = run_root / "run" / "config_manifest.json"
    if os.path.lexists(config_path):
        config = _canonical_json_artifact(config_path)
        expected_reviews = [
            {
                "reviewer": "feature-dev:code-reviewer",
                "scope": scope,
                "target_sha256": by_scope[scope][0]["target_sha256"],
                "finding_count": 0,
                "artifact_sha256": by_scope[scope][1],
            }
            for scope, _ in _review_scopes()
        ]
        if config.get("review_records") != expected_reviews:
            raise ContractError("base review manifest registry differs")
        for record in config.get("records", []):
            if record.get("path") in base_by_path and record.get("sha256") != base_by_path[record["path"]]:
                raise ContractError("base review configuration binding differs")
    source_path = run_root / "run" / "source_manifest.json"
    if os.path.lexists(source_path):
        source = _canonical_json_artifact(source_path)
        for record in source.get("records", []):
            if record.get("path") in base_by_path and record.get("sha256") != base_by_path[record["path"]]:
                raise ContractError("base review source binding differs")


def _check_detail(
    staging_root: Path,
    run_id: str,
    name: str,
    scope: str,
    inputs: Any,
    outputs: Any,
    evidence_paths: Sequence[str],
) -> str:
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "name": name,
        "scope": scope,
        "inputs": inputs,
        "outputs": outputs,
        "evidence_paths": sorted(evidence_paths),
    }
    digest = canonical_json_sha256(artifact)
    path = staging_root / "run" / "check_details" / f"{digest}.json"
    if not path.exists():
        write_canonical_json(path, artifact)
    return digest


def _check_record(
    staging_root: Path,
    run_id: str,
    name: str,
    scope: str,
    expected: Any,
    actual: Any,
    max_error: float | None,
    tolerance: float | None,
    passed: bool,
    evidence_paths: Sequence[str],
) -> dict[str, Any]:
    detail = _check_detail(
        staging_root,
        run_id,
        name,
        scope,
        {"expected": expected},
        {"actual": actual, "max_error": max_error, "tolerance": tolerance, "pass": passed},
        evidence_paths,
    )
    return {
        "name": name,
        "scope": scope,
        "expected": expected,
        "actual": actual,
        "max_error": max_error,
        "tolerance": tolerance,
        "details_sha256": detail,
        "pass": passed,
    }


def _source_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for row in payload["sources"]["references"]:
        records.append(_manifest_record(row["path"], row["role"], row["revision"]))
    for row in payload["sources"]["prerequisite_results"]:
        records.append(_manifest_record(row["path"], "prerequisite_result", row["revision"]))
    target_roles = {
        "neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json": "machine_readable_method",
        "src/model/modular_mlx_backend.py": "mlx_training_contract",
        "src/model/modular_sources.py": "frozen_source_port",
        "src/model/modular_neural_machine.py": "combined_host",
        "neuroloc/simulations/memory/modular_sequence_role_cpu.py": "reference_lifecycle_runner",
        "neuroloc/simulations/memory/modular_sequence_role_mlx.py": "mlx_training_engine",
        "scripts/qualify_modular_mlx.py": "reviewed_launcher_bootstrap",
        "tests/test_modular_neural_machine.py": "model_tests",
        "tests/test_modular_sequence_role_cpu.py": "reference_lifecycle_tests",
        "tests/test_modular_sequence_role_mlx.py": "mlx_training_tests",
    }
    for path in payload["artifacts"]["source_manifest_required_targets"]:
        digest = sha256_file(_repo_path(path))
        records.append(_manifest_record(path, target_roles[path], f"sha256:{digest}"))
    records.sort(key=lambda record: record["path"])
    return records


def _config_records() -> list[dict[str, Any]]:
    roles = {
        "neuroloc/wiki/tests/modular_sequence_role_cpu_run.md": "canonical_run_card",
        "neuroloc/wiki/PROJECT_PLAN.md": "canonical_project_plan",
        "neuroloc/wiki/OPERATING_DIRECTIVE.md": "wiki_operating_directive",
        "neuroloc/wiki/synthesis/modular_neural_model_stack.md": "architecture_contract",
        "neuroloc/wiki/synthesis/neural_model_dossier_nested_reciprocal_feature_mixer.md": "deferred_feature_dossier",
        "neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json": "machine_readable_method",
        "AGENTS.md": "repository_rules",
        "src/model/modular_sources.py": "implementation_configuration",
        "src/model/modular_neural_machine.py": "implementation_configuration",
        "src/model/modular_mlx_backend.py": "implementation_configuration",
        "neuroloc/simulations/memory/modular_sequence_role_cpu.py": "implementation_configuration",
        "neuroloc/simulations/memory/modular_sequence_role_mlx.py": "implementation_configuration",
        "scripts/qualify_modular_mlx.py": "implementation_configuration",
        "tests/test_modular_neural_machine.py": "implementation_test_configuration",
        "tests/test_modular_sequence_role_cpu.py": "implementation_test_configuration",
        "tests/test_modular_sequence_role_mlx.py": "implementation_test_configuration",
    }
    records = [_config_record(path, role) for path, role in roles.items()]
    records.sort(key=lambda record: record["path"])
    return records


def _environment_artifact(
    run_id: str,
    launcher_argv: Sequence[str],
    runtime: RuntimeModules,
    hardware_reader: Callable[[], Mapping[str, Any]] = observe_target_hardware,
) -> dict[str, Any]:
    torch = runtime.torch
    uname = platform.uname()
    payload = load_prereg_payload()
    mlx_backend = importlib.import_module("src.model.modular_mlx_backend")
    child_argv, child_environment = mlx_backend.child_invocation("serve")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "python": platform.python_version(),
        "torch": str(torch.__version__).split("+")[0],
        "mlx": {"version": "0.29.3", "python_path": str(mlx_backend.MLX_PYTHON), "device": "Device(gpu, 0)"},
        "hardware": dict(hardware_reader()),
        "operating_system": {
            "name": uname.system,
            "version": uname.release,
            "architecture": uname.machine,
        },
        "process": {
            "start_method": "parent_subprocess_with_start_new_session",
            "parent_pid": os.getpid(),
            "child_pids": [],
            "launcher_argv": list(launcher_argv),
            "child_argv": child_argv,
            "entry_checks": {
                "parent": "pass",
                "future_child_policy": "qualifier_validates_exact_python_and_environment_before_MLX_import",
            },
        },
        "environment": dict(REQUIRED_ENV),
        "child_environment": {**child_environment, "MODULAR_MLX_RUN_ROOT": str(RESULTS_PARENT / run_id), "required_external_private_tmp_scratch_root": "/private/tmp/unique_per_child"},
        "threads": {"torch_intraop": 4, "torch_interop": 1, "mlx_omp": 4, "mlx_veclib": 4},
        "numerics": {
            "training_device": "Device(gpu, 0)",
            "training_dtype": "mlx.core.float32",
            "reference_device": "cpu",
            "reference_dtype": "torch.float32",
            "deterministic_algorithms": True,
            "matmul_precision": "highest",
            "autocast": False,
            "compilation": True,
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "epsilon": 1e-08,
            "amsgrad": False,
            "foreach": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "group_policy": payload["optimizer"]["decay_classification"],
        },
        "clipping": {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": True},
        "loss": {"cross_entropy_reduction": "mean", "label_smoothing": 0.0, "class_weights": None},
        "evaluation": {"mode": "eval", "inference_mode": True},
        "resource_sampling": {
            "interval_seconds": 5,
            "ps_argv_template": ["/bin/ps", "-o", "pid=,ppid=,rss=,time=", "-p", "sorted_comma_separated_live_authoritative_pids"],
            "swap_argv": ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            "locale": "C",
            "rss_unit": "bytes",
            "cpu_time_unit": "microseconds",
            "swap_unit": "bytes",
            "failure_policy": "hard_abort_resource_sampler_failure",
        },
    }


def _verify_source_and_result_hashes(payload: Mapping[str, Any]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    sources = []
    results = []
    for row in payload["sources"]["references"]:
        actual = sha256_file(validate_real_regular_file(row["path"]))
        sources.append((row["path"], row["sha256"], actual))
    for row in payload["sources"]["prerequisite_results"]:
        actual = sha256_file(validate_real_regular_file(row["path"]))
        results.append((row["path"], row["sha256"], actual))
    return sources, results


def transformerov_numerical_selfcheck(runtime: RuntimeModules) -> list[dict[str, Any]]:
    torch = runtime.torch
    sources = importlib.import_module("src.model.modular_sources")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        batch, heads, tokens, width = 2, 4, 128, 32
        query = torch.randn(batch, heads, tokens, width)
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = torch.randn(batch, heads, tokens, width)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        value = torch.randn(batch, heads, tokens, width)
        primary = torch.sigmoid(torch.randn(batch, heads, tokens))
        write = torch.sigmoid(torch.randn(batch, heads, tokens))
    reference = sources.frozen_recurrent_gated(query, key, value, primary, write)
    limits = {16: 6e-7, 32: 7e-7, 64: 1.4e-6}
    records = []
    for chunk in (16, 32, 64):
        actual = sources.frozen_chunkwise_gated(query, key, value, primary, write, chunk)
        error = float((reference - actual).abs().max().item())
        finite = bool(torch.isfinite(actual).all())
        records.append(
            {
                "chunk_length": chunk,
                "max_error": error,
                "acceptance_limit": limits[chunk],
                "finite": finite,
                "pass": finite and error <= limits[chunk],
            }
        )
    return records


def public_route_parity(runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    model_module = runtime.model_module
    source_module = importlib.import_module("src.model.modular_sources")
    public = source_module._PUBLIC_ROUTED
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(778001)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
        inputs = torch.randn(2, 128, 64, dtype=torch.float32)
    mixer = model.blocks[4].mix
    actual = mixer(inputs, return_aux=True, return_detail=True, request_router_loss=True, query_only_position=126)
    attention = mixer.source_mixer.attention
    config = mixer.source_mixer.config
    qkv = attention.qkv(inputs).reshape(2, 128, 3, 4, 16)
    query = attention._rope(qkv[:, :, 0].transpose(1, 2))
    key = attention._rope(qkv[:, :, 1].transpose(1, 2))
    value = qkv[:, :, 2].transpose(1, 2)
    query_input = query.transpose(1, 2).reshape(2, 128, 64)
    key_input = key.transpose(1, 2).reshape(2, 128, 64)
    query_route = attention.router.query_features(query_input)
    key_route = attention.router.key_features(key_input)
    codebooks = attention.router.normalized_codebooks()
    index = public.build_packed_index(key_route, codebooks, block_size=8, bucket_capacity=64)
    search = public.search_packed_index(
        query_route,
        codebooks,
        index,
        probes=4,
        selected_blocks=2,
        local_blocks=1,
        query_chunk_size=128,
    )
    direct_loss = attention._router_loss(query, key, query_route, key_route)
    query_error = float((actual.query_route - query_route).abs().max().item())
    key_error = float((actual.key_route - key_route).abs().max().item())
    loss_error = float((actual.router_loss - direct_loss).abs().item())
    exact_index = (
        torch.equal(actual.telemetry["block_features"], index.block_features)
        and torch.equal(actual.telemetry["block_addresses"], index.block_addresses)
        and torch.equal(actual.telemetry["postings"], index.postings)
        and int(actual.telemetry["overflow_count"]) == int(index.overflow_count)
        and int(actual.telemetry["max_bucket_load"]) == int(index.max_bucket_load)
    )
    exact_search = (
        torch.equal(actual.telemetry["raw_remote"], search.selected_blocks)
        and int(actual.telemetry["addresses_probed"]) == int(search.addresses_probed)
        and int(actual.telemetry["postings_read"]) == int(search.postings_read)
        and int(actual.telemetry["candidate_blocks"]) == int(search.candidate_blocks)
    )
    return {
        "query_feature_max_error": query_error,
        "key_feature_max_error": key_error,
        "internal_loss_max_error": loss_error,
        "packed_index_exact": exact_index,
        "raw_search_exact": exact_search,
        "finite": bool(torch.isfinite(actual.delta).all() and torch.isfinite(actual.router_loss)),
    }


def host_source_parity(runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    model_module = runtime.model_module
    source_module = importlib.import_module("src.model.modular_sources")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(779001)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
        inputs = torch.randn(2, 128, 64, dtype=torch.float32)
    recurrent = model.blocks[1].mix
    recurrent_output = recurrent(inputs, reset_positions=(), return_aux=False)
    query, key, value, write, primary, output_gate = recurrent._project(inputs)
    direct = source_module.frozen_chunkwise_gated(query.float(), key.float(), value.float(), primary.float(), write.float(), 32)
    direct = recurrent.onorm(direct.to(inputs.dtype)).transpose(1, 2).reshape(2, 128, 64)
    direct = recurrent.o(direct * output_gate)
    recurrent_error = float((recurrent_output - direct).abs().max().item())
    feature = model.blocks[1].mlp
    feature_reference = source_module._TRANSFORMER_MODEL.SwiGLU(64)
    feature_reference.load_state_dict(feature.state_dict(), strict=True)
    feature_error = float((feature(inputs) - feature_reference(inputs)).abs().max().item())
    config = model.config
    architecture_exact = (
        config.width == 64
        and config.block_count == 8
        and config.heads == 4
        and config.recurrent_head_width == 16
        and config.recurrent_chunk_length == 32
        and tuple(block.kind for block in model.blocks) == model_module.SEQUENCE_SCHEDULE
        and model.embed.weight.data_ptr() != model.head.weight.data_ptr()
    )
    return {
        "recurrent_max_error": recurrent_error,
        "feature_max_error": feature_error,
        "architecture_exact": architecture_exact,
        "finite": bool(torch.isfinite(recurrent_output).all()),
    }


def _pretraining_record(
    staging_root: Path,
    run_id: str,
    assertion_id: str,
    actual: Mapping[str, Any],
    passed: bool,
    evidence_paths: Sequence[str],
) -> dict[str, Any]:
    return _check_record(
        staging_root,
        run_id,
        assertion_id,
        "pretraining_assertion",
        True,
        dict(actual),
        None,
        None,
        bool(passed),
        evidence_paths,
    )


def validate_pretraining_assertion_closure(assertion_values: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(assertion_values, Mapping) or tuple(assertion_values) != PRETRAINING_ASSERTION_IDS:
        raise ContractError("pretraining assertion value closure differs")
    records = []
    for assertion_id in PRETRAINING_ASSERTION_IDS:
        value = assertion_values[assertion_id]
        if not isinstance(value, (tuple, list)) or len(value) != 2 or not isinstance(value[0], Mapping) or type(value[1]) is not bool:
            raise ContractError("pretraining assertion value differs")
        records.append({"assertion_id": assertion_id, "actual": dict(value[0]), "pass": value[1]})
    if not all(record["pass"] for record in records):
        raise ContractError("pretraining assertion package failed")
    return records


def _batch_local_foreign_conditions(conditions: Any, batch_size: int, torch: Any) -> Any:
    if conditions.ndim != 1 or conditions.numel() % batch_size or batch_size < 2:
        raise ContractError("carry-shuffle condition geometry differs")
    foreign = torch.empty_like(conditions)
    for start in range(0, conditions.numel(), batch_size):
        stop = start + batch_size
        foreign[start:stop] = conditions[start:stop].roll(1, dims=0)
    return foreign


def _validate_carry_shuffle_strata(payload: Mapping[str, Any], generated: Mapping[int, tuple[int, int]]) -> None:
    registered = payload["gates"]["carry_shuffle_frozen_strata"]
    if [row["construction_seed"] for row in registered] != list(RUNG_ONE_SEEDS):
        raise ContractError("carry-shuffle seed registry differs")
    for row in registered:
        validate_exact_keys(row, ("construction_seed", "evaluation_seed", "same_rows", "changed_rows"), "carry-shuffle stratum")
        seed = row["construction_seed"]
        same, changed = generated[seed]
        if row != {"construction_seed": seed, "evaluation_seed": 400000 + seed, "same_rows": same, "changed_rows": changed}:
            raise ContractError("carry-shuffle frozen stratum differs")
    for gate in payload["gates"]["rung_one_registry"][-3:]:
        if gate["condition"] != "carry_shuffle" or gate["stratum"] != "changed_condition":
            raise ContractError("carry-shuffle conditional gate order differs")
        for seed in RUNG_ONE_SEEDS:
            denominator = generated[seed][1]
            expected = math.ceil(0.9 * denominator) if gate["gate_operator"] == ">=" else math.floor(0.3 * denominator)
            if gate["denominator"][str(seed)] != denominator or gate["gate_threshold_count"][str(seed)] != expected:
                raise ContractError("carry-shuffle conditional gate threshold differs")


def run_pretraining_assertions(
    staging_root: Path,
    run_id: str,
    runtime: RuntimeModules,
    payload: Mapping[str, Any],
    route_parity_values: Mapping[str, Any],
    host_parity_values: Mapping[str, Any],
    selfcheck_values: Sequence[Mapping[str, Any]],
    oracle_error: float,
) -> list[dict[str, Any]]:
    torch = runtime.torch
    model_module = runtime.model_module
    source_module = importlib.import_module("src.model.modular_sources")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(780001)
        selected = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
        tokens = torch.randint(0, 128, (2, 128), dtype=torch.long)
        selected.eval()
        with torch.inference_mode():
            intact = selected(tokens, return_aux=True, recurrent_telemetry=True, route_detail=True)
    reconstructed = selected.embed(tokens)
    abi_finite = bool(torch.isfinite(intact.logits).all() and torch.isfinite(intact.hidden).all())
    for execution in intact.blocks:
        abi_finite = abi_finite and all(
            bool(torch.isfinite(value).all()) and value.shape == reconstructed.shape and value.dtype == reconstructed.dtype and value.device == reconstructed.device
            for value in (execution.computed_sequence_delta, execution.exposed_sequence_delta, execution.feature_delta)
        )
        reconstructed = reconstructed + execution.exposed_sequence_delta + execution.feature_delta
    residual_error = float((selected.nf(reconstructed) - intact.hidden).abs().max())
    assertion_values: dict[str, tuple[dict[str, Any], bool]] = {}
    assertion_values["mixer_abi_and_residual_ownership"] = (
        {"finite": abi_finite, "residual_max_error": residual_error, "block_count": len(intact.blocks)},
        abi_finite and residual_error == 0.0 and len(intact.blocks) == 8,
    )
    architecture = selected.config
    architecture_pass = (
        architecture.width == 64
        and architecture.block_count == 8
        and architecture.heads == 4
        and architecture.recurrent_head_width == 16
        and architecture.recurrent_chunk_length == 32
        and tuple(block.kind for block in selected.blocks) == model_module.SEQUENCE_SCHEDULE
        and tuple(index for index, block in enumerate(selected.blocks) if block.kind == "recurrent") == (1, 2, 3, 5, 6, 7)
        and selected.blocks[0].mix.selected_remote_blocks == 0
        and selected.blocks[4].mix.selected_remote_blocks == 2
        and selected.embed.weight.data_ptr() != selected.head.weight.data_ptr()
        and not any("nextlat" in name.lower() or "reciprocal" in name.lower() for name, _ in selected.named_modules())
        and not any(isinstance(module, torch.nn.Dropout) for module in selected.modules())
    )
    assertion_values["exact_architecture"] = ({"schedule": list(model_module.SEQUENCE_SCHEDULE), "pass": architecture_pass}, architecture_pass)
    base = tokens[:1].clone()
    candidate = base.clone()
    candidate[:, :8] = (candidate[:, :8] + 1).remainder(128)
    all_candidate = base.clone()
    all_candidate[:, :80] = (all_candidate[:, :80] + 1).remainder(128)
    cue = base.clone()
    cue[:, 80] = (cue[:, 80] + 1).remainder(128)
    with torch.inference_mode():
        base_detail = selected(base, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        candidate_detail = selected(candidate, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        all_candidate_detail = selected(all_candidate, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
        cue_detail = selected(cue, return_aux=True, route_detail=True).blocks[4].mixer_output.telemetry
    firewall_parts = {
        "other_key_unchanged": bool(torch.equal(base_detail["key"][:, :, 8:16], candidate_detail["key"][:, :, 8:16])),
        "other_value_unchanged": bool(torch.equal(base_detail["value"][:, :, 8:16], candidate_detail["value"][:, :, 8:16])),
        "candidate_query_unchanged": bool(torch.equal(base_detail["query"][:, :, 126], all_candidate_detail["query"][:, :, 126])),
        "cue_query_changed": bool(not torch.equal(base_detail["query"][:, :, 126], cue_detail["query"][:, :, 126])),
        "cue_keys_unchanged": bool(torch.equal(base_detail["key"][:, :, :80], cue_detail["key"][:, :, :80])),
        "cue_values_unchanged": bool(torch.equal(base_detail["value"][:, :, :80], cue_detail["value"][:, :, :80])),
    }
    assertion_values["firewall_factorization"] = (firewall_parts, all(firewall_parts.values()))
    recurrent = selected.blocks[1].mix
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(780002)
        recurrent_input = torch.randn(1, 128, 64, dtype=torch.float32, requires_grad=True)
        upstream = torch.randn(1, 128, 64, dtype=torch.float32)
    reset_input = recurrent_input.detach().clone().requires_grad_(True)
    fast_output = recurrent(recurrent_input)
    reset_output = recurrent(reset_input, force_reset_aware=True)
    fast_gradients = torch.autograd.grad((fast_output * upstream).sum(), (recurrent_input, *tuple(recurrent.parameters())), retain_graph=False)
    reset_gradients = torch.autograd.grad((reset_output * upstream).sum(), (reset_input, *tuple(recurrent.parameters())), retain_graph=False)
    forward_error = float((fast_output - reset_output).abs().max())
    input_gradient_error = float((fast_gradients[0] - reset_gradients[0]).abs().max())
    parameter_gradient_error = max(float((left - right).abs().max()) for left, right in zip(fast_gradients[1:], reset_gradients[1:]))
    reset_aux = recurrent(reset_input.detach(), reset_positions=model_module.RUNG_ONE_RESET_POSITIONS, return_aux=True)
    reset_before = tuple(boundary.position for boundary in reset_aux.boundaries if boundary.kind == "firewall_before_reset")
    reset_after = tuple(boundary.position for boundary in reset_aux.boundaries if boundary.kind == "firewall_after_reset")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(780003)
        rung_two_mixer = model_module.ModularNeuralMachine(model_module.rung_two_config()).blocks[1].mix
        rung_two_aux = rung_two_mixer(torch.randn(1, 512, 64), return_aux=True)
    clamp_positions = tuple(boundary.position for boundary in rung_two_aux.boundaries if boundary.kind == "chunk_end_after_clamp")
    recurrent_values = {
        "forward_max_error": forward_error,
        "input_gradient_max_error": input_gradient_error,
        "parameter_gradient_max_error": parameter_gradient_error,
        "reset_before": list(reset_before),
        "reset_after": list(reset_after),
        "rung_two_chunk_ends": list(clamp_positions),
    }
    recurrent_pass = forward_error <= 1e-5 and input_gradient_error <= 1e-4 and parameter_gradient_error <= 1e-4 and reset_before == model_module.RUNG_ONE_RESET_POSITIONS and reset_after == model_module.RUNG_ONE_RESET_POSITIONS and clamp_positions == tuple(range(31, 512, 32))
    assertion_values["reset_aware_recurrent_fidelity"] = (recurrent_values, recurrent_pass)
    route = intact.blocks[4].mixer_output.telemetry
    effective = route["effective_remote"]
    outside = torch.arange(128) != 126
    query_route = effective[:, 126, 0]
    query_only_pass = bool((effective[:, outside] == -1).all())
    for row in query_route:
        values = [int(value) for value in row.tolist() if int(value) >= 0]
        query_only_pass = query_only_pass and len(values) == len(set(values)) and all(0 <= value <= 14 for value in values)
    assertion_values["query_only_remote_route"] = ({"outside_query_all_minus_one": bool((effective[:, outside] == -1).all()), "query_routes": query_route.tolist()}, query_only_pass)
    causality_errors = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(780004)
        causal_tokens = torch.randint(0, 128, (2, 128), dtype=torch.long)
    with torch.inference_mode():
        for intervention in ("none", "reset", "shuffle"):
            original = selected(causal_tokens, recurrent_intervention=intervention)
            for boundary in (7, 8, 79, 80, 95, 96, 126, 127):
                changed = causal_tokens.clone()
                changed[:, boundary:] = (changed[:, boundary:] + 1).remainder(128)
                perturbed = selected(changed, recurrent_intervention=intervention)
                causality_errors.append(float((original[:, :boundary] - perturbed[:, :boundary]).abs().max()) if boundary else 0.0)
    causality_max = max(causality_errors)
    assertion_values["causality"] = ({"max_prefix_error": causality_max, "checks": len(causality_errors)}, causality_max == 0.0 and len(causality_errors) == 24)
    with torch.inference_mode():
        recurrent_knockout = selected(tokens, return_aux=True, recurrent_telemetry=True, recurrent_knockout=True)
        carry_reset = selected(tokens, return_aux=True, recurrent_telemetry=True, recurrent_intervention="reset")
        carry_shuffle = selected(tokens, return_aux=True, recurrent_telemetry=True, recurrent_intervention="shuffle")
        block_knockout = selected(tokens, return_aux=True, block4_routed_knockout=True)
    recurrent_indices = (1, 2, 3, 5, 6, 7)
    recurrent_zero = all(bool(torch.equal(recurrent_knockout.blocks[index].exposed_sequence_delta, torch.zeros_like(recurrent_knockout.blocks[index].exposed_sequence_delta))) for index in recurrent_indices)
    recurrent_computed = all(bool((recurrent_knockout.blocks[index].computed_sequence_delta != 0).any()) for index in recurrent_indices)
    feature_active = all(bool(torch.isfinite(execution.feature_delta).all() and (execution.feature_delta != 0).any()) for output in (recurrent_knockout, carry_reset, carry_shuffle, block_knockout) for execution in output.blocks)
    block_zero = bool(torch.equal(block_knockout.blocks[4].exposed_sequence_delta, torch.zeros_like(block_knockout.blocks[4].exposed_sequence_delta)))
    carry_boundaries = [boundary.kind for output in (carry_reset, carry_shuffle) for index in recurrent_indices for boundary in output.blocks[index].mixer_output.boundaries if boundary.position == 96]
    intervention_pass = recurrent_zero and recurrent_computed and feature_active and block_zero and carry_boundaries.count("carry_before_reset") == 6 and carry_boundaries.count("carry_after_reset") == 6 and carry_boundaries.count("carry_before_shuffle") == 6 and carry_boundaries.count("carry_after_shuffle") == 6
    assertion_values["intervention_isolation"] = ({"recurrent_exposed_zero": recurrent_zero, "recurrent_computed": recurrent_computed, "block4_exposed_zero": block_zero, "feature_active": feature_active, "carry_boundary_count": len(carry_boundaries)}, intervention_pass)
    required = torch.tensor([3, 7], dtype=torch.long)
    forced = torch.full((2, 128), -1, dtype=torch.long)
    forced[:, 126] = required
    random_payload = generate_random_routes(500011, 2, torch)
    random_override = torch.full((2, 128, 1, 2), -1, dtype=torch.long)
    random_override[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long)
    exclusion_payload = generate_source_exclusion_routes(510011, [[3, -1], [14, 7]], [3, 14], torch)
    exclusion_override = torch.full((2, 128, 1, 2), -1, dtype=torch.long)
    exclusion_override[:, 126, 0] = torch.tensor(exclusion_payload["routes"], dtype=torch.long)
    with torch.inference_mode():
        forced_output = selected(tokens, return_aux=True, forced_blocks=forced).blocks[4].mixer_output
        random_output = selected(tokens, return_aux=True, route_override=random_override).blocks[4].mixer_output
    forced_effective = forced_output.telemetry["effective_remote"][:, 126, 0]
    route_exact = bool((forced_effective == required[:, None]).any(dim=-1).all() and torch.equal(random_output.telemetry["effective_remote"], random_override))
    generator_exact = canonical_json_sha256(random_payload) == payload["generators"]["matched_random_route"]["golden"]["payload_sha256"] and canonical_json_sha256(exclusion_payload) == payload["generators"]["source_exclusion"]["golden"]["payload_sha256"]
    assertion_values["forced_and_random_route_exactness"] = ({"forced_exact": route_exact, "generator_exact": generator_exact}, route_exact and generator_exact)
    source_parity_pass = all(record["pass"] for record in selfcheck_values) and route_parity_values["query_feature_max_error"] == 0.0 and route_parity_values["key_feature_max_error"] == 0.0 and route_parity_values["internal_loss_max_error"] == 0.0 and route_parity_values["packed_index_exact"] and route_parity_values["raw_search_exact"] and route_parity_values["finite"] and host_parity_values["architecture_exact"] and host_parity_values["finite"] and host_parity_values["recurrent_max_error"] == 0.0 and host_parity_values["feature_max_error"] == 0.0 and math.isfinite(oracle_error) and oracle_error <= 1e-5
    assertion_values["source_host_route_and_attention_parity"] = ({"route": dict(route_parity_values), "host": dict(host_parity_values), "oracle_error": oracle_error}, source_parity_pass)
    state_before = {name: _tensor_sha256(value) for name, value in selected.state_dict().items()}
    with torch.inference_mode():
        output_a = selected(tokens)
        selected((tokens + 1).remainder(128))
        output_b = selected(tokens)
    state_after = {name: _tensor_sha256(value) for name, value in selected.state_dict().items()}
    forbidden_fragments = ("state", "postings", "raw_remote", "effective_remote", "selected_blocks")
    persistent_names = [name for name, _ in selected.named_buffers() if any(fragment in name.lower() for fragment in forbidden_fragments)]
    lifetime_pass = bool(torch.equal(output_a, output_b)) and state_before == state_after and not persistent_names
    assertion_values["state_and_index_lifetime"] = ({"aba_exact": bool(torch.equal(output_a, output_b)), "state_hash_stable": state_before == state_after, "forbidden_persistent_names": persistent_names}, lifetime_pass)
    router_optimizer, _, router_membership = _make_optimizer(selected, "router_only", runtime)
    joint_optimizer, _, joint_membership = _make_optimizer(selected, "joint", runtime)
    selected.zero_grad(set_to_none=True)
    train_batch = payload_to_tensors(generate_rung_one_batch(780005, 2, torch), torch)
    train_output = selected(train_batch["tokens"], return_aux=True, request_block4_router_loss=True)
    gradient_loss = _supervised_route_loss(train_output, train_batch["required_source"], runtime) + _block_output(train_output, 4).router_loss
    query_parameter = dict(selected.named_parameters())["blocks.4.mix.source_mixer.attention.router.query_projection.weight"]
    key_parameter = dict(selected.named_parameters())["blocks.4.mix.source_mixer.attention.router.key_projection.weight"]
    codebook_parameter = dict(selected.named_parameters())["blocks.4.mix.source_mixer.attention.router.codebooks"]
    query_gradient, key_gradient, codebook_gradient = torch.autograd.grad(gradient_loss, (query_parameter, key_parameter, codebook_parameter), allow_unused=True)
    query_learns = query_gradient is not None and bool(torch.isfinite(query_gradient).all() and (query_gradient != 0).any())
    key_learns = key_gradient is not None and bool(torch.isfinite(key_gradient).all() and (key_gradient != 0).any())
    codebook_truthful = codebook_gradient is None or bool(torch.isfinite(codebook_gradient).all())
    codebook_name = "blocks.4.mix.source_mixer.attention.router.codebooks"
    optimizer_pass = router_membership[codebook_name]["requires_grad"] and joint_membership[codebook_name]["requires_grad"] and query_learns and key_learns and codebook_truthful
    assertion_values["optimizer_membership_and_gradients"] = ({"query_nonzero_gradient": query_learns, "key_nonzero_gradient": key_learns, "codebook_gradient_present": codebook_gradient is not None, "codebook_router_member": router_membership[codebook_name]["requires_grad"], "codebook_joint_member": joint_membership[codebook_name]["requires_grad"]}, optimizer_pass)
    selected_copy = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
    copy_report = model_module.copy_compatible_state(selected, selected_copy, include_router=True)
    selected_reload = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
    selected_reload.load_state_dict(selected.state_dict(), strict=True)
    with torch.inference_mode():
        copied_logits = selected_copy(tokens)
        reloaded_logits = selected_reload(tokens)
    copy_error = float((output_b - copied_logits).abs().max())
    reload_error = float((output_b - reloaded_logits).abs().max())
    initialization_pass = len(copy_report.incompatible_source) == 0 and len(copy_report.incompatible_destination) == 0 and copy_error <= 1e-7 and reload_error <= 1e-7
    assertion_values["initialization_copy_and_reload_parity"] = ({"copy_max_error": copy_error, "reload_max_error": reload_error, "copied_tensors": len(copy_report.compatible)}, initialization_pass)
    optimizer_states = {"router_only": len(router_optimizer.state), "joint": len(joint_optimizer.state)}
    optimizer_models = []
    for role, stage in (("all_eligible", "donor"), ("dense", "dense_base"), ("dense", "dense_continuation"), ("rung_two", "rung_two")):
        candidate = model_module.ModularNeuralMachine(model_module.rung_two_config() if role == "rung_two" else model_module.rung_one_config(role))
        optimizer, _, _ = _make_optimizer(candidate, stage, runtime)
        optimizer_states[stage] = len(optimizer.state)
        optimizer_models.append((candidate, optimizer))
    fresh_optimizer_pass = all(value == 0 for value in optimizer_states.values())
    assertion_values["fresh_optimizer_state"] = ({"state_cardinalities": optimizer_states}, fresh_optimizer_pass)
    golden_hashes = {
        "rung_one": canonical_json_sha256(generate_rung_one_batch(123456, 2, torch)),
        "rung_two": canonical_json_sha256(generate_rung_two_batch(123456, 2, torch)),
        "random_route": canonical_json_sha256(random_payload),
        "source_exclusion": canonical_json_sha256(exclusion_payload),
    }
    expected_hashes = {
        "rung_one": payload["generators"]["rung_one"]["golden"]["payload_sha256"],
        "rung_two": payload["generators"]["rung_two"]["golden"]["payload_sha256"],
        "random_route": payload["generators"]["matched_random_route"]["golden"]["payload_sha256"],
        "source_exclusion": payload["generators"]["source_exclusion"]["golden"]["payload_sha256"],
    }
    stream = torch.Generator(device="cpu")
    stream.manual_seed(100011)
    first_stream = _continuous_rung_one_batch(stream, 16, torch)
    second_stream = _continuous_rung_one_batch(stream, 16, torch)
    stream_distinct = _batch_payload_hash(first_stream, "donor") != _batch_payload_hash(second_stream, "donor")
    strata = {}
    generator_ranges = True
    for seed in RUNG_ONE_SEEDS:
        generated = payload_to_tensors(generate_rung_one_batch(400000 + seed, 512, torch), torch)
        foreign = _batch_local_foreign_conditions(generated["condition"], 32, torch)
        same = int((generated["condition"] == foreign).sum())
        strata[seed] = (same, 512 - same)
        generator_ranges = generator_ranges and bool((generated["tokens"] >= 0).all() and (generated["tokens"] < 128).all() and (generated["targets"] >= 40).all() and (generated["targets"] <= 55).all())
    _validate_carry_shuffle_strata(payload, strata)
    generator_pass = golden_hashes == expected_hashes and stream_distinct and generator_ranges
    assertion_values["generator_integrity"] = ({"golden_hashes": golden_hashes, "continuous_stream_distinct": stream_distinct, "carry_shuffle_strata": {str(seed): list(strata[seed]) for seed in RUNG_ONE_SEEDS}, "ranges": generator_ranges}, generator_pass)
    route_overflow = int(route["overflow_count"])
    route_maximum = int(route["max_bucket_load"])
    underfill = int((query_route == -1).sum())
    capacity_pass = route_overflow == 0 and route_maximum <= 16 and underfill >= 0 and not hasattr(selected.blocks[4].mix, "fallback")
    assertion_values["capacity_and_fallback"] = ({"overflow_count": route_overflow, "max_bucket_load": route_maximum, "underfill_count": underfill, "fallback_attribute": hasattr(selected.blocks[4].mix, "fallback")}, capacity_pass)
    assertion_records = validate_pretraining_assertion_closure(assertion_values)
    records = [
        _pretraining_record(staging_root, run_id, record["assertion_id"], record["actual"], record["pass"], ["run/source_manifest.json", "run/config_manifest.json"])
        for record in assertion_records
    ]
    if tuple(record["name"] for record in records) != PRETRAINING_ASSERTION_IDS or not all(record["pass"] for record in records):
        raise ContractError("pretraining assertion package failed")
    return records


def _sentinel_payload(runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777001)
        inputs = torch.randn(2, 128, 64, dtype=torch.float32)
    selected = torch.full((2, 128, 1, 3), -1, dtype=torch.long)
    for position in range(128):
        local = position // 8
        selected[:, position, 0, 0] = max(0, local - 2)
        selected[:, position, 0, 1] = max(0, local - 1)
        selected[:, position, 0, 2] = local
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": "pretraining_selected_attention_oracle",
        "inputs": inputs.tolist(),
        "selected_ids": selected.tolist(),
    }


def _selected_attention_oracle(runtime: RuntimeModules, sentinel: Mapping[str, Any]) -> float:
    torch = runtime.torch
    model_module = runtime.model_module
    source_module = importlib.import_module("src.model.modular_sources")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(777001)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
    mixer = model.blocks[4].mix
    inputs = torch.tensor(sentinel["inputs"], dtype=torch.float32)
    selected = torch.tensor(sentinel["selected_ids"], dtype=torch.long)
    attention = mixer.source_mixer.attention
    config = mixer.source_mixer.config
    qkv = attention.qkv(inputs).reshape(2, 128, 3, 4, 16)
    query = attention._rope(qkv[:, :, 0].transpose(1, 2))
    key = attention._rope(qkv[:, :, 1].transpose(1, 2))
    value = qkv[:, :, 2].transpose(1, 2)
    selected_output = source_module._PUBLIC_ROUTED.selected_attention(query, key, value, selected, block_size=config.block_size)
    dense_output = source_module._PUBLIC_ROUTED.dense_selected_mask_attention(query, key, value, selected, block_size=config.block_size)
    return float((selected_output - dense_output).abs().max().item())


def _disposable_publication_rehearsal(parent: Path, run_id: str) -> dict[str, Any]:
    source = parent / f".{run_id}.publish-rehearsal.{os.getpid()}.source"
    destination = parent / f".{run_id}.publish-rehearsal.{os.getpid()}.destination"
    if os.path.lexists(source) or os.path.lexists(destination):
        raise ContractError("publication rehearsal paths are not absent")
    try:
        source.mkdir(exist_ok=False)
        fsync_directory(parent)
        marker = source / "marker"
        with marker.open("xb") as handle:
            handle.write(b"publication-rehearsal")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(source)
        os.rename(source, destination)
        fsync_directory(parent)
        os.rename(destination, source)
        fsync_directory(parent)
        return {
            "source_absent": True,
            "destination_absent": True,
            "parent": str(parent),
        }
    finally:
        _cleanup_disposable_rehearsal(source, destination)


def _cleanup_disposable_rehearsal(*paths: Path) -> None:
    failure: BaseException | None = None
    for path in paths:
        try:
            _remove_tree_and_fsync(path)
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise UnrecoverableOrphan("disposable rehearsal cleanup failed") from failure


def _attempt_fixture(run_id: str) -> dict[str, Any]:
    identity = attempt_id(run_id, 1, 11, "selected", "joint", 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rung": 1,
        "claim_seed": 11,
        "construction_seed": 11,
        "event_sequence": 0,
        "event": "started",
        "attempt_id": identity,
        "model": "selected",
        "stage": "joint",
        "logical_update": 1,
        "examples": 16,
        "token_positions": 2048,
        "batch_sha256": "0" * 64,
        "monotonic_ns": 1,
        "wall_time_utc": "2026-07-19T00:00:00Z",
        "metrics": None,
    }


def rehearse_crash_atomic_faults(parent: Path, run_id: str) -> list[dict[str, Any]]:
    scratch = Path(tempfile.mkdtemp(prefix=f".{run_id}.ledger-rehearsal.", dir=parent))
    records = []
    try:
        fixture = _attempt_fixture(run_id)
        for index, fault in enumerate(FAULT_IDS):
            signals = SignalController()
            path = scratch / f"{index}.jsonl"
            writer = CrashAtomicJsonlWriter(path, validate_attempt_row, signals, "attempt")
            writer.precreate()
            try:
                result = writer.append(fixture, fault=fault)
            except LedgerAppendError as exc:
                result = exc.result
            rows = writer.validate_committed_prefix()
            accounting = derive_attempt_accounting(rows)
            expected_committed = fault == "handled_signal_after_commit_before_ack"
            passed = (
                result.committed is expected_committed
                and result.acknowledged is False
                and len(rows) == int(expected_committed)
                and accounting.attempted_updates == int(expected_committed)
                and accounting.completed_updates == 0
                and path.stat().st_size == writer.last_committed_offset
            )
            writer.close()
            records.append(
                {
                    "name": fault,
                    "committed": result.committed,
                    "acknowledged": result.acknowledged,
                    "current_offset": result.current_offset,
                    "row_count": len(rows),
                    "attempted_updates": accounting.attempted_updates,
                    "completed_updates": accounting.completed_updates,
                    "pass": passed,
                }
            )
    finally:
        _cleanup_disposable_rehearsal(scratch)
    if not all(record["pass"] for record in records):
        raise ContractError("crash-atomic fault rehearsal failed")
    return records


def _terminal_deactivation_rehearsal(parent: Path, run_id: str) -> dict[str, Any]:
    scratch = Path(tempfile.mkdtemp(prefix=f".{run_id}.terminal-rehearsal.", dir=parent))
    signals = SignalController()
    signals.install()
    try:
        write_canonical_json(scratch / "record.json", {"pass": True})
        result = ChecksumTerminalizer(scratch, signals).finalize(["record.json"])
        before = (scratch / "SHA256SUMS").read_bytes()
        signals.inject()
        after = (scratch / "SHA256SUMS").read_bytes()
        passed = result.terminal and signals.terminal and before == after
        return {"terminal": result.terminal, "handlers_deactivated": signals.terminal, "post_signal_immutable": before == after, "pass": passed}
    finally:
        _cleanup_disposable_rehearsal(scratch)


def _verify_public_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd="/Users/dttdrv/Projects/Monodratic-public",
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ContractError("public Monodratic commit lookup failed")
    commit = completed.stdout.strip()
    if commit != "0f9bf59ebdd032da46553d985bcf23348e1d5289":
        raise ContractError("public Monodratic commit differs")
    return commit


def _verify_public_result_values() -> dict[str, Any]:
    result_path = validate_real_regular_file(Path("/Users/dttdrv/Projects/Monodratic-public/results/mqar.json"))
    mqar = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    actual = {
        "learned_route_answers": {
            "successes": mqar["pooled"]["learned_r2"]["correct"],
            "trials": mqar["pooled"]["learned_r2"]["total"],
        },
        "all_eligible_donor_answers": {
            "successes": mqar["pooled"]["dense_donor"]["correct"],
            "trials": mqar["pooled"]["dense_donor"]["total"],
        },
        "target_forced_answers": {
            "successes": mqar["pooled"]["forced_target_r2"]["correct"],
            "trials": mqar["pooled"]["forced_target_r2"]["total"],
        },
    }
    expected = {
        "learned_route_answers": {"successes": 763, "trials": 768},
        "all_eligible_donor_answers": {"successes": 768, "trials": 768},
        "target_forced_answers": {"successes": 768, "trials": 768},
    }
    if actual != expected:
        raise ContractError("public result values differ")
    return {"expected": expected, "actual": actual, "pass": actual == expected}


def build_shared_prepilot_base(
    staging_root: str | Path,
    run_id: str,
    launcher_argv: Sequence[str],
    runtime: RuntimeModules,
    payload: Mapping[str, Any],
    trained_backend_probe: Callable[[Path, str], Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    root = Path(staging_root)
    if not root.is_dir() or any(root.iterdir()):
        raise ContractError("staging root must be an empty existing directory")
    for relative in ("run", "run/check_details", "run/sentinels"):
        directory = root / relative
        directory.mkdir(exist_ok=False)
        fsync_directory(directory.parent)
        fsync_directory(directory)
    copy_launch_project_plan_snapshot(root)
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_card_path": "neuroloc/wiki/tests/modular_sequence_role_cpu_run.md",
        "run_card_markdown_sha256": sha256_file(RUN_CARD_PATH),
        "payload_path": "neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json",
        "payload_sha256": canonical_json_sha256(payload),
    }
    write_canonical_json(root / "run" / "prereg.json", prereg)
    review_records, review_source_records = select_and_copy_review_attestations(root)
    source_records = _source_records(payload) + review_source_records
    source_records.sort(key=lambda record: record["path"])
    source_manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "records": source_records}
    write_canonical_json(root / "run" / "source_manifest.json", source_manifest)
    config_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "records": _config_records(),
        "review_records": review_records,
    }
    write_canonical_json(root / "run" / "config_manifest.json", config_manifest)
    validate_base_review_target_binding(root, "not_started", None)
    environment = _environment_artifact(run_id, launcher_argv, runtime)
    write_canonical_json(root / "run" / "environment.json", environment)
    sentinel = _sentinel_payload(runtime)
    write_canonical_json(root / "run" / "sentinels" / "selected_attention_oracle_payload.json", sentinel)
    source_observations, result_observations = _verify_source_and_result_hashes(payload)
    source_checks = []
    for path, expected, actual in source_observations:
        source_checks.append(_check_record(root, run_id, "frozen_source_hash", path, expected, actual, None, None, actual == expected, ["run/source_manifest.json"]))
    commit = _verify_public_commit()
    source_checks.append(_check_record(root, run_id, "public_monodratic_commit", "public_source", "0f9bf59ebdd032da46553d985bcf23348e1d5289", commit, None, None, commit == "0f9bf59ebdd032da46553d985bcf23348e1d5289", ["run/source_manifest.json"]))
    result_checks = []
    for path, expected, actual in result_observations:
        result_checks.append(_check_record(root, run_id, "prerequisite_result_hash", path, expected, actual, None, None, actual == expected, ["run/source_manifest.json"]))
    result_values = _verify_public_result_values()
    result_checks.append(_check_record(root, run_id, "public_monodratic_result_values", "public_results", result_values["expected"], result_values["actual"], None, None, result_values["pass"], ["run/source_manifest.json"]))
    selfcheck_values = transformerov_numerical_selfcheck(runtime)
    selfcheck_records = []
    for record in selfcheck_values:
        selfcheck_records.append(_check_record(root, run_id, "transformerov_numerical_selfcheck", f"chunk_{record['chunk_length']}", record["acceptance_limit"], record["max_error"], record["max_error"], record["acceptance_limit"], record["pass"], ["run/source_manifest.json"]))
    oracle_error = _selected_attention_oracle(runtime, sentinel)
    oracle_pass = math.isfinite(oracle_error) and oracle_error <= 1e-5
    route_parity_values = public_route_parity(runtime)
    host_parity_values = host_source_parity(runtime)
    routing_parity = [
        _check_record(root, run_id, "public_raw_route_feature_parity", "pinned_source_adapter", 0.0, max(route_parity_values["query_feature_max_error"], route_parity_values["key_feature_max_error"]), max(route_parity_values["query_feature_max_error"], route_parity_values["key_feature_max_error"]), 0.0, route_parity_values["query_feature_max_error"] == 0.0 and route_parity_values["key_feature_max_error"] == 0.0, ["run/source_manifest.json"]),
        _check_record(root, run_id, "packed_index_parity", "pinned_source_adapter", True, route_parity_values["packed_index_exact"], None, None, route_parity_values["packed_index_exact"], ["run/source_manifest.json"]),
        _check_record(root, run_id, "raw_search_parity", "pinned_source_adapter", True, route_parity_values["raw_search_exact"], None, None, route_parity_values["raw_search_exact"], ["run/source_manifest.json"]),
        _check_record(root, run_id, "internal_route_loss_parity", "pinned_source_adapter", 0.0, route_parity_values["internal_loss_max_error"], route_parity_values["internal_loss_max_error"], 0.0, route_parity_values["internal_loss_max_error"] == 0.0 and route_parity_values["finite"], ["run/source_manifest.json"]),
    ]
    host_parity = [
        _check_record(root, run_id, "host_source_parity", "modular_host", {"architecture_exact": True, "max_error": 0.0}, host_parity_values, max(host_parity_values["recurrent_max_error"], host_parity_values["feature_max_error"]), 0.0, host_parity_values["architecture_exact"] and host_parity_values["finite"] and host_parity_values["recurrent_max_error"] == 0.0 and host_parity_values["feature_max_error"] == 0.0, ["run/source_manifest.json"]),
        _check_record(root, run_id, "selected_attention_oracle", "global_sentinel", 1e-5, oracle_error, oracle_error, 1e-5, oracle_pass, ["run/sentinels/selected_attention_oracle_payload.json"]),
    ]
    pretraining_records = run_pretraining_assertions(root, run_id, runtime, payload, route_parity_values, host_parity_values, selfcheck_values, oracle_error)
    host_parity.extend(pretraining_records)
    publication_rehearsal = _disposable_publication_rehearsal(root.parent, run_id)
    fault_rehearsal = rehearse_crash_atomic_faults(root.parent, run_id)
    terminal_rehearsal = _terminal_deactivation_rehearsal(root.parent, run_id)
    lifecycle_values = [
        ("staged_publish_rehearsal", publication_rehearsal),
        ("actual_staging_readiness", {"shared_base_paths_built": True, "final_root_absent": True}),
        ("pilot_timeline_transition", {"durability_boundary_rehearsed": True}),
    ]
    lifecycle_values.extend((record["name"], record) for record in fault_rehearsal)
    lifecycle_values.append(("terminal_checksum_deactivation", terminal_rehearsal))
    lifecycle_assertions = [
        _check_record(root, run_id, name, "lifecycle", True, value, None, None, bool(value.get("pass", True)), ["run/config_manifest.json"])
        for name, value in lifecycle_values
    ]
    if trained_backend_probe is None:
        raise InitializationRefusal("trained backend preflight probe is required")
    trained_backend = list(trained_backend_probe(root, run_id))
    required_trained_names = payload["artifacts"]["schemas"]["preflight"]["required_trained_backend_checks"]
    if [record.get("name") for record in trained_backend] != required_trained_names:
        raise InitializationRefusal("trained backend preflight registry differs")
    preflight_pass = all(record["pass"] for group in (source_checks, result_checks, selfcheck_records, routing_parity, host_parity, trained_backend, lifecycle_assertions) for record in group) and oracle_pass
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_checks": source_checks,
        "result_checks": result_checks,
        "transformerov_selfcheck": selfcheck_records,
        "routing_parity": routing_parity,
        "host_parity": host_parity,
        "trained_backend": trained_backend,
        "lifecycle_assertions": lifecycle_assertions,
        "selected_attention_oracle": {
            "identity": "pretraining_selected_attention_oracle",
            "max_error": oracle_error,
            "tolerance": 1e-5,
            "pass": oracle_pass,
        },
        "pass": preflight_pass,
    }
    write_canonical_json(root / "run" / "preflight.json", preflight)
    if not preflight_pass:
        raise InitializationRefusal("shared prepilot preflight failed")
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda value: len(value.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(root)
    return {
        "source_manifest": source_manifest,
        "config_manifest": config_manifest,
        "environment": environment,
        "preflight": preflight,
        "sentinel": sentinel,
    }


def _continuous_rung_one_batch(generator: Any, batch_size: int, torch: Any) -> dict[str, Any]:
    tokens = torch.randint(0, 32, (batch_size, 128), generator=generator, dtype=torch.int64)
    rule_blocks = torch.empty((batch_size, 4), dtype=torch.long)
    answer_indices = torch.empty((batch_size, 4), dtype=torch.long)
    condition = torch.empty((batch_size,), dtype=torch.long)
    targets = torch.empty((batch_size,), dtype=torch.long)
    required_source = torch.empty((batch_size,), dtype=torch.long)
    for row_index in range(batch_size):
        row_rules = torch.randperm(10, generator=generator)[:4]
        row_answers = torch.randperm(16, generator=generator)[:4]
        row_condition = torch.randint(0, 4, (1,), generator=generator, dtype=torch.int64)[0]
        rule_blocks[row_index] = row_rules
        answer_indices[row_index] = row_answers
        condition[row_index] = row_condition
        for rule_index in range(4):
            tokens[row_index, 8 * int(row_rules[rule_index])] = 64 + 16 * rule_index + int(row_answers[rule_index])
        tokens[row_index, 80] = 32 + row_condition
        tokens[row_index, 126] = 36
        targets[row_index] = 40 + row_answers[row_condition]
        required_source[row_index] = row_rules[row_condition]
    return {
        "tokens": tokens,
        "targets": targets,
        "condition": condition,
        "rule_blocks": rule_blocks,
        "answer_indices": answer_indices,
        "required_source": required_source,
    }


def _continuous_rung_two_batch(generator: Any, batch_size: int, torch: Any) -> dict[str, Any]:
    tokens = torch.randint(40, 256, (batch_size, 512), generator=generator, dtype=torch.int64)
    count = torch.empty((batch_size,), dtype=torch.long)
    count_positions = []
    for row_index in range(batch_size):
        row_count = torch.randint(0, 8, (1,), generator=generator, dtype=torch.int64)[0]
        permutation = torch.randperm(64, generator=generator)
        selected = permutation[: int(row_count)]
        if int(row_count):
            tokens[row_index, selected] = 35
        count[row_index] = row_count
        count_positions.append(selected)
    tokens[:, 510] = 36
    targets = 19 + torch.cumsum(tokens == 35, dim=1)
    return {"tokens": tokens, "targets": targets, "count": count, "count_positions": count_positions}


def _route_override_from_generator(generator: Any, batch_size: int, torch: Any) -> Any:
    routes = torch.full((batch_size, 128, 1, 2), -1, dtype=torch.long)
    for row_index in range(batch_size):
        routes[row_index, 126, 0] = torch.sort(torch.randperm(15, generator=generator)[:2]).values
    return routes


def _pilot_draw(workload: str, data_generator: Any, route_generator: Any, torch: Any) -> dict[str, Any]:
    if workload in {"A", "D", "S"}:
        tokens = torch.randint(0, 128, (16, 128), generator=data_generator, dtype=torch.int64)
        target = torch.randint(0, 128, (16,), generator=data_generator, dtype=torch.int64)
        result = {"tokens": tokens, "targets": target}
        if workload == "S":
            result["required_source"] = torch.randint(0, 15, (16,), generator=data_generator, dtype=torch.int64)
            result["route_override"] = _route_override_from_generator(route_generator, 16, torch)
        return result
    if workload == "H":
        return {
            "tokens": torch.randint(0, 256, (8, 512), generator=data_generator, dtype=torch.int64),
            "targets": torch.randint(0, 256, (8, 512), generator=data_generator, dtype=torch.int64),
        }
    raise ContractError("pilot workload differs")


def _stage_schedule_multiplier(update: int, updates: int, warmup: int) -> float:
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= updates or not 0 < warmup < updates:
        raise ContractError("schedule coordinates differ")
    if update <= warmup:
        return update / warmup
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (update - warmup) / (updates - warmup)))


def _stage_membership(model_module: Any, name: str, stage: str) -> tuple[bool, str | None, float | None]:
    router_any = model_module.is_router_parameter(name)
    router_zero = model_module.is_router_parameter(name, 0)
    router_four = model_module.is_router_parameter(name, 4)
    if stage in {"donor", "rung_two"}:
        return (not router_any, "all_trainable" if not router_any else None, 0.002 if not router_any else None)
    if stage == "router_only":
        return (router_four, "block_4_router" if router_four else None, 0.003 if router_four else None)
    if stage == "joint":
        if router_zero:
            return False, None, None
        if router_four:
            return True, "block_4_router", 0.001
        return True, "other_trainable", 0.00025
    if stage == "dense_base":
        return (not router_zero, "all_trainable" if not router_zero else None, 0.002 if not router_zero else None)
    if stage == "dense_continuation":
        return (not router_zero, "all_trainable" if not router_zero else None, 0.00025 if not router_zero else None)
    raise ContractError("stage membership differs")


def _weight_decay_for_category(category: str) -> float:
    if category == "matrix":
        return 0.01
    if category in {"normalization_scale", "recurrent_bias", "codebook"}:
        return 0.0
    raise ContractError("parameter category differs")


def _make_optimizer(model: Any, stage: str, runtime: RuntimeModules) -> tuple[Any, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    torch = runtime.torch
    model_module = runtime.model_module
    grouped: dict[tuple[str, float], list[Any]] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        parameter.grad = None
        trainable, family, peak = _stage_membership(model_module, name, stage)
        parameter.requires_grad_(trainable)
        category = model_module.parameter_category(name, parameter)
        decay = _weight_decay_for_category(category)
        group_name = None
        if trainable:
            if family is None or peak is None:
                raise ContractError("trainable parameter lacks a rate family")
            group_name = f"{family}_{'decay' if decay else 'zero_decay'}"
            grouped.setdefault((group_name, decay), []).append(parameter)
        evidence[name] = {
            "category": category,
            "requires_grad": trainable,
            "parameter_group": group_name,
            "peak_lr": peak,
            "weight_decay": decay if trainable else None,
        }
    parameter_groups = []
    group_records = []
    for group_name, decay in sorted(grouped):
        members = grouped[(group_name, decay)]
        family = group_name.rsplit("_", 2)[0]
        peak_values = {evidence[name]["peak_lr"] for name, parameter in model.named_parameters() if any(parameter is member for member in members)}
        if len(peak_values) != 1:
            raise ContractError("optimizer rate family differs")
        peak = float(next(iter(peak_values)))
        parameter_groups.append({"params": members, "lr": peak, "weight_decay": decay, "name": group_name, "peak_lr": peak})
        group_records.append({"parameter_group": group_name, "peak_lr": peak, "weight_decay": decay})
    if not parameter_groups:
        raise ContractError("optimizer has no trainable values")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=1.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    if optimizer.state:
        raise ContractError("fresh optimizer state is not empty")
    return optimizer, group_records, evidence


def _set_optimizer_rates(optimizer: Any, multiplier: float) -> list[dict[str, Any]]:
    records = []
    for group in optimizer.param_groups:
        rate = float(group["peak_lr"]) * multiplier
        group["lr"] = rate
        records.append({"parameter_group": str(group["name"]), "learning_rate": rate})
    records.sort(key=lambda record: record["parameter_group"])
    return records


def _block_output(output: Any, block_index: int) -> Any:
    for block in output.blocks:
        if block.block_index == block_index:
            return block.mixer_output
    raise ContractError("model auxiliary block is absent")


def _supervised_route_loss(output: Any, required_source: Any, runtime: RuntimeModules) -> Any:
    torch = runtime.torch
    routed = _block_output(output, 4)
    if routed is None:
        raise ContractError("block-4 route features are absent")
    query = routed.query_route[:, 126, 0]
    keys = routed.key_route[:, :120].reshape(routed.key_route.size(0), 15, 8, routed.key_route.size(-1)).mean(dim=2)
    logits = torch.einsum("bd,bnd->bn", query, keys)
    return torch.nn.functional.cross_entropy(logits, required_source, reduction="mean", label_smoothing=0.0)


def _training_forward_loss(model: Any, batch: Mapping[str, Any], stage: str, runtime: RuntimeModules, route_override: Any = None) -> tuple[Any, dict[str, float | None], int, int, Any]:
    torch = runtime.torch
    if stage in {"router_only", "joint"}:
        output = model(
            batch["tokens"],
            return_aux=True,
            route_detail=True,
            recurrent_telemetry=True,
            request_block4_router_loss=stage == "joint",
            route_override=route_override,
        )
        supervised = _supervised_route_loss(output, batch["required_source"], runtime)
        task = torch.nn.functional.cross_entropy(output.logits[:, 126], batch["targets"], reduction="mean", label_smoothing=0.0)
        routed = _block_output(output, 4)
        internal = routed.router_loss if stage == "joint" else None
        if stage == "router_only":
            total = supervised
            task_value = None
            internal_value = None
        else:
            if internal is None:
                raise ContractError("joint internal route loss is absent")
            total = task + 0.1 * (internal + supervised)
            task_value = float(task.detach())
            internal_value = float(internal.detach())
        overflow, maximum, _, _, _, _ = _route_observation(output, 1, {"stage": stage})
        components = {
            "task_loss": task_value,
            "internal_router_loss": internal_value,
            "supervised_route_loss": float(supervised.detach()),
        }
        return total, components, overflow, maximum, output
    if stage == "rung_two":
        output = model(batch["tokens"], return_aux=True, route_detail=True, recurrent_telemetry=True)
        loss = torch.nn.functional.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)), batch["targets"].reshape(-1), reduction="mean", label_smoothing=0.0)
        overflow, maximum, _, _, _, _ = _route_observation(output, 2, {"stage": stage})
        return loss, {"task_loss": float(loss.detach()), "internal_router_loss": None, "supervised_route_loss": None}, overflow, maximum, output
    if stage in {"donor", "dense_base", "dense_continuation"}:
        output = model(batch["tokens"], return_aux=True, route_detail=True, recurrent_telemetry=True)
        loss = torch.nn.functional.cross_entropy(output.logits[:, 126], batch["targets"], reduction="mean", label_smoothing=0.0)
        overflow, maximum, _, _, _, _ = _route_observation(output, 1, {"stage": stage})
        return loss, {"task_loss": float(loss.detach()), "internal_router_loss": None, "supervised_route_loss": None}, overflow, maximum, output
    raise ContractError("training loss stage differs")


def _tensor_sha256(tensor: Any) -> str:
    cpu = tensor.detach().contiguous().cpu()
    header = canonical_json_bytes({"dtype": str(cpu.dtype), "shape": list(cpu.shape)}) + b"\n"
    raw = bytes(cpu.view(dtype=_torch_module().uint8).reshape(-1).clone().untyped_storage())
    return hashlib.sha256(header + raw).hexdigest()


def _state_manifest(model: Any) -> tuple[list[dict[str, Any]], str]:
    records = []
    for name, tensor in sorted(model.state_dict().items()):
        records.append({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": _tensor_sha256(tensor)})
    return records, canonical_json_sha256(records)


def _torch_artifact_bytes(value: Any, torch: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    if not data:
        raise ContractError("Torch artifact serialization is empty")
    return data


def _save_torch_artifact(path: Path, value: Any, torch: Any) -> None:
    if os.path.lexists(path):
        raise ContractError("binary artifact already exists")
    temporary = path.with_name(f".{path.name}.writing.{os.getpid()}")
    if os.path.lexists(temporary):
        raise ContractError("binary temporary path already exists")
    data = _torch_artifact_bytes(value, torch)
    owned_paths: dict[Path, tuple[int, int]] = {}
    renamed = False
    try:
        _write_exact_bytes(temporary, data, owned_paths)
        os.rename(temporary, path)
        renamed = True
        fsync_directory(path.parent)
    except Exception as exc:
        if renamed:
            try:
                _validate_owned_path(path, owned_paths[temporary])
            except BaseException as identity_exc:
                raise UnrecoverableOrphan("binary artifact ownership is ambiguous") from identity_exc
            raise UnrecoverableOrphan("binary artifact publication durability failed") from exc
        if temporary in owned_paths:
            if os.path.lexists(temporary):
                _unlink_owned_path(temporary, owned_paths[temporary])
            elif not os.path.lexists(path):
                raise UnrecoverableOrphan("binary artifact ownership is ambiguous")
        raise


def _write_canonical_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if os.path.lexists(path):
        raise ContractError("JSONL artifact already exists")
    with path.open("xb") as handle:
        for record in records:
            line = canonical_json_bytes(record) + b"\n"
            if handle.write(line) != len(line):
                raise ContractError("short JSONL write")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)


def _replace_canonical_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.replacing.{os.getpid()}")
    if os.path.lexists(temporary):
        raise ContractError("replacement temporary path already exists")
    write_canonical_json(temporary, value)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def _replace_canonical_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.replacing.{os.getpid()}")
    if os.path.lexists(temporary):
        raise ContractError("replacement temporary path already exists")
    _write_canonical_jsonl(temporary, records)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def _write_canonical_gzip(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    if os.path.lexists(path):
        raise ContractError("gzip artifact already exists")
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
            for record in records:
                compressed.write(canonical_json_bytes(record) + b"\n")
        raw.flush()
        os.fsync(raw.fileno())
    fsync_directory(path.parent)


def _child_send_and_wait(connection: Any, message: Mapping[str, Any]) -> None:
    try:
        connection.send(dict(message))
        response = connection.recv()
    except (EOFError, BrokenPipeError, ConnectionError, OSError) as exc:
        raise HardAbort("worker_exit") from exc
    if response != {"ack": True}:
        raise HardAbort("worker_exit")


def _child_failure_message(error: BaseException, worker: str) -> dict[str, Any]:
    if worker not in {"A", "B"}:
        raise ContractError("child worker identity differs")
    if isinstance(error, UnrecoverableOrphan):
        return {"kind": "unrecoverable_orphan", "worker": worker}
    if isinstance(error, HardAbort):
        reason_code = error.reason_code
        context = dict(error.context)
    elif type(error).__module__ == "src.model.modular_sources" and type(error).__name__ == "FrozenSourceMismatchError" and isinstance(getattr(error, "surface", None), str):
        reason_code = "frozen_hash_change"
        context = {"surface": error.surface}
    elif isinstance(error, (FloatingPointError, ArithmeticError)):
        reason_code = "nonfinite"
        context = {}
    elif isinstance(error, ContractError):
        reason_code = "artifact_inconsistency"
        context = {}
    else:
        reason_code = "worker_exit"
        context = {}
    context["worker"] = worker
    return {"kind": "hard_abort", "worker": worker, "reason_code": reason_code, "context": context}


def _report_child_failure(connection: Any, error: BaseException, worker: str) -> None:
    orphan = isinstance(error, UnrecoverableOrphan)
    try:
        connection.send(_child_failure_message(error, worker))
    except BaseException:
        if orphan:
            raise SystemExit(WORKER_ORPHAN_EXIT_CODE) from error
    if orphan:
        raise SystemExit(WORKER_ORPHAN_EXIT_CODE) from error


def _pilot_worker(worker: str, ordinal: int, barrier: Any, run_id: str, start_event: Any, connection: Any) -> None:
    preserve_orphan_exit = False
    try:
        start_event.wait()
        validate_entry_environment()
        runtime = _import_runtime()
        configure_torch(runtime.torch)
        torch = runtime.torch
        workload_order = ("A", "S", "D", "H")
        stage_by_workload = {"A": "donor", "S": "joint", "D": "dense_base", "H": "rung_two"}
        role_by_workload = {"A": "all_eligible", "S": "selected", "D": "dense", "H": "rung_two"}
        for workload_ordinal, workload in enumerate(workload_order):
            connection.send({"kind": "barrier_entry", "worker": worker, "workload": workload, "monotonic_ns": time.monotonic_ns()})
            barrier.wait()
            model_seed = 9999983 + 1000 * ordinal + 100 * workload_ordinal
            data_seed = model_seed + 1
            route_seed = model_seed + 2
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(model_seed)
                if role_by_workload[workload] == "rung_two":
                    model = runtime.model_module.ModularNeuralMachine(runtime.model_module.rung_two_config())
                else:
                    model = runtime.model_module.ModularNeuralMachine(runtime.model_module.rung_one_config(role_by_workload[workload]))
            optimizer, _, _ = _make_optimizer(model, stage_by_workload[workload], runtime)
            _set_optimizer_rates(optimizer, 1.0)
            data_generator = torch.Generator(device="cpu")
            data_generator.manual_seed(data_seed)
            route_generator = torch.Generator(device="cpu")
            route_generator.manual_seed(route_seed)
            warmup_update_ns = []
            timed_update_ns = []
            model.train()
            for logical_update in range(1, 12):
                batch = _pilot_draw(workload, data_generator, route_generator, torch)
                _child_send_and_wait(
                    connection,
                    {
                        "kind": "pilot_update_start",
                        "worker": worker,
                        "seed": model_seed,
                        "stage": workload,
                        "logical_update": logical_update,
                        "token_positions": int(batch["tokens"].numel()),
                    },
                )
                started = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)
                finite_context = {"worker": worker, "stage": workload, "logical_update": logical_update}
                _assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)
                loss, components, overflow, maximum, output = _training_forward_loss(
                    model,
                    batch,
                    stage_by_workload[workload],
                    runtime,
                    route_override=batch.get("route_override"),
                )
                _assert_finite_tree(torch, loss, finite_context, "loss")
                _assert_finite_tree(torch, components, finite_context, "component_losses")
                _assert_finite_tree(torch, output, finite_context, "model_output")
                observed_overflow, observed_maximum, _, _, _, _ = _route_observation(output, 2 if workload == "H" else 1, finite_context)
                if observed_overflow != overflow or observed_maximum != maximum:
                    raise HardAbort("artifact_inconsistency", {**finite_context, "surface": "routing_reduction"})
                loss.backward()
                _assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)
                _clip_gradient_norm_finite(torch, model, optimizer, finite_context)
                if overflow:
                    raise HardAbort("route_overflow", finite_context)
                optimizer.step()
                _assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)
                stopped = time.perf_counter_ns()
                duration = stopped - started
                if logical_update <= 3:
                    warmup_update_ns.append(duration)
                else:
                    timed_update_ns.append(duration)
            del optimizer
            del model
            importlib.import_module("gc").collect()
            barrier.wait()
            connection.send(
                {
                    "kind": "pilot_workload_complete",
                    "worker": worker,
                    "worker_ordinal": ordinal,
                    "workload": workload,
                    "workload_ordinal": workload_ordinal,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "route_seed": route_seed,
                    "warmup_update_ns": warmup_update_ns,
                    "timed_update_ns": timed_update_ns,
                    "model_destroyed": True,
                    "optimizer_destroyed": True,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
        _child_send_and_wait(connection, {"kind": "clean_complete", "worker": worker})
    except BaseException as exc:
        preserve_orphan_exit = isinstance(exc, UnrecoverableOrphan)
        _report_child_failure(connection, exc, worker)
        raise
    finally:
        try:
            connection.close()
        except BaseException:
            if not preserve_orphan_exit:
                raise


def worker_exit_observations(
    processes: Sequence[Any],
    handshakes: set[str],
    worker_names: Mapping[int, str],
) -> tuple[bool, tuple[dict[str, Any], ...]]:
    complete = True
    observations = []
    for process in sorted(processes, key=lambda value: worker_names[int(value.pid)]):
        name = worker_names[int(process.pid)]
        try:
            exitcode_before_join = process.exitcode
        except BaseException:
            exitcode_before_join = None
        try:
            process.join(timeout=0)
            exitcode = process.exitcode
        except BaseException:
            try:
                exitcode = process.exitcode
            except BaseException:
                exitcode = exitcode_before_join
            if exitcode == WORKER_ORPHAN_EXIT_CODE:
                raise UnrecoverableOrphan(f"worker {name} crossed the orphan boundary")
            if exitcode is None:
                raise UnrecoverableOrphan(f"worker {name} exit status is unobservable")
            complete = False
            observations.append({"reason_code": "worker_exit", "context": {"worker": name}})
            continue
        if exitcode is None:
            complete = False
        elif exitcode == WORKER_ORPHAN_EXIT_CODE:
            raise UnrecoverableOrphan(f"worker {name} crossed the orphan boundary")
        elif exitcode != 0 or name not in handshakes:
            complete = False
            observations.append({"reason_code": "worker_exit", "context": {"worker": name}})
    return complete and not observations, tuple(observations)


def _resource_expected_pids(
    processes: Sequence[Any],
    handshakes: set[str],
    worker_names: Mapping[int, str],
) -> list[int]:
    _, observations = worker_exit_observations(processes, handshakes, worker_names)
    if observations:
        observation = observations[0]
        raise HardAbort(observation["reason_code"], observation["context"])
    return sorted([os.getpid(), *(int(process.pid) for process in processes if process.exitcode is None)])


def _resource_sample(
    run_id: str,
    phase: str,
    sample_id: int,
    processes: Sequence[Any],
    handshakes: set[str],
    worker_names: Mapping[int, str],
    active_jobs: Mapping[str, Mapping[str, Any]],
    swap_baseline: int,
    attempted_updates: int,
    token_positions: int,
) -> dict[str, Any]:
    expected = _resource_expected_pids(processes, handshakes, worker_names)
    process_records = None
    for _ in range(len(processes) + 1):
        try:
            process_records = sample_processes(expected)
            break
        except HardAbort:
            raise
        except ProcessSetMismatch as exc:
            prior = set(expected)
            updated_expected = _resource_expected_pids(processes, handshakes, worker_names)
            updated = set(updated_expected)
            observed = set(exc.observed_pids)
            if exc.expected_pids != tuple(expected) or not updated < prior or not observed < prior or not updated <= observed:
                raise HardAbort("resource_sampler_failure", {"phase": phase, "sample_id": sample_id}) from exc
            expected = updated_expected
        except ContractError as exc:
            raise HardAbort("resource_sampler_failure", {"phase": phase, "sample_id": sample_id}) from exc
        except BaseException as exc:
            raise HardAbort("resource_sampler_failure", {"phase": phase, "sample_id": sample_id}) from exc
    if process_records is None:
        raise HardAbort("resource_sampler_failure", {"phase": phase, "sample_id": sample_id})
    try:
        swap = sample_swap()
    except HardAbort:
        raise
    except BaseException as exc:
        raise HardAbort("resource_sampler_failure", {"phase": phase, "sample_id": sample_id}) from exc
    jobs = [dict(active_jobs[name]) for name in sorted(active_jobs)]
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sample_id": sample_id,
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": utc_now(),
        "expected_pids": sorted(expected),
        "processes": process_records,
        "active_jobs": jobs,
        "aggregate_rss_bytes": sum(record["rss_bytes"] for record in process_records),
        "aggregate_cpu_time_us": sum(record["cpu_time_us"] for record in process_records),
        "swap_used_bytes": swap,
        "swap_growth_bytes": max(0, swap - swap_baseline),
        "parser_status": "pass",
        "attempted_updates": attempted_updates,
        "token_positions": token_positions,
    }
    validate_resource_row(row)
    return row


def _pilot_expected_messages(worker: str) -> Iterable[dict[str, Any]]:
    if worker not in {"A", "B"}:
        raise ContractError("pilot protocol worker differs")
    worker_ordinal = 0 if worker == "A" else 1
    for workload_ordinal, workload in enumerate(("A", "S", "D", "H")):
        model_seed = 9999983 + 1000 * worker_ordinal + 100 * workload_ordinal
        yield {"kind": "barrier_entry", "workload": workload}
        for logical_update in range(1, 12):
            yield {
                "kind": "pilot_update_start",
                "seed": model_seed,
                "stage": workload,
                "logical_update": logical_update,
                "token_positions": 4096 if workload == "H" else 2048,
            }
        yield {
            "kind": "pilot_workload_complete",
            "worker_ordinal": worker_ordinal,
            "workload": workload,
            "workload_ordinal": workload_ordinal,
            "model_seed": model_seed,
            "data_seed": model_seed + 1,
            "route_seed": model_seed + 2,
        }
    yield {"kind": "clean_complete"}


def pilot_protocol_state(worker: str) -> dict[str, Any]:
    stream = iter(_pilot_expected_messages(worker))
    return {"worker": worker, "stream": stream, "expected": next(stream), "complete": False}


def validate_pilot_protocol_message(message: Mapping[str, Any], state: dict[str, Any]) -> str:
    worker = state.get("worker")
    expected = state.get("expected")
    if state.get("complete") or worker not in {"A", "B"} or not isinstance(expected, Mapping) or not isinstance(message, Mapping) or message.get("worker") != worker or message.get("kind") != expected["kind"]:
        raise ContractError("pilot protocol transition differs")
    kind = message["kind"]
    exact_keys = {
        "barrier_entry": ("kind", "worker", "workload", "monotonic_ns"),
        "pilot_update_start": ("kind", "worker", "seed", "stage", "logical_update", "token_positions"),
        "pilot_workload_complete": ("kind", "worker", "worker_ordinal", "workload", "workload_ordinal", "model_seed", "data_seed", "route_seed", "warmup_update_ns", "timed_update_ns", "model_destroyed", "optimizer_destroyed", "monotonic_ns"),
        "clean_complete": ("kind", "worker"),
    }
    validate_exact_keys(message, exact_keys[kind], "pilot protocol message")
    for key, value in expected.items():
        if message.get(key) != value:
            raise ContractError("pilot protocol identity differs")
    if kind == "barrier_entry" and (type(message["monotonic_ns"]) is not int or message["monotonic_ns"] < 0):
        raise ContractError("pilot barrier clock differs")
    if kind == "pilot_workload_complete":
        if type(message["monotonic_ns"]) is not int or message["monotonic_ns"] < 0 or message["model_destroyed"] is not True or message["optimizer_destroyed"] is not True:
            raise ContractError("pilot completion identity differs")
        for key, length in (("warmup_update_ns", 3), ("timed_update_ns", 8)):
            values = message[key]
            if not isinstance(values, list) or len(values) != length or any(type(value) is not int or value < 0 for value in values):
                raise ContractError("pilot timing vector differs")
    try:
        state["expected"] = next(state["stream"])
    except StopIteration:
        state["expected"] = None
        state["complete"] = True
    return kind


def _handle_pilot_worker_message(
    message: Mapping[str, Any],
    worker: str,
    connection: Any,
    handshakes: set[str],
    active_jobs: dict[str, dict[str, Any]],
    barrier_state: dict[str, dict[str, Any]],
    workload_records: list[dict[str, Any]],
    signals: SignalController,
    protocol_state: dict[str, Any] | None = None,
) -> tuple[int, int, dict[str, Any] | None]:
    if not isinstance(message, Mapping) or message.get("worker") != worker or worker in handshakes:
        raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_protocol"})
    kind = message.get("kind")
    if protocol_state is not None and kind not in {"hard_abort", "unrecoverable_orphan"}:
        try:
            validate_pilot_protocol_message(message, protocol_state)
        except ContractError as exc:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_protocol"}) from exc
    if kind == "barrier_entry":
        validate_exact_keys(message, ("kind", "worker", "workload", "monotonic_ns"), "pilot barrier message")
        if message["workload"] not in barrier_state or type(message["monotonic_ns"]) is not int:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_barrier"})
        state = barrier_state[message["workload"]]
        state["entry"].append(message["monotonic_ns"])
        state["arrived"].add(worker)
        return 0, 0, None
    if kind == "pilot_update_start":
        validate_exact_keys(message, ("kind", "worker", "seed", "stage", "logical_update", "token_positions"), "pilot update message")
        if type(message["logical_update"]) is not int or type(message["token_positions"]) is not int or message["logical_update"] <= 0 or message["token_positions"] <= 0:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_update"})
        active_jobs[worker] = {"worker": worker, "seed": int(message["seed"]), "stage": str(message["stage"]), "logical_update": message["logical_update"]}
        return 1, message["token_positions"], {"ack": True}
    if kind == "pilot_workload_complete":
        active_jobs.pop(worker, None)
        copied = dict(message)
        copied.pop("kind")
        monotonic_ns = copied.pop("monotonic_ns")
        workload = copied["workload"]
        if workload not in barrier_state or type(monotonic_ns) is not int:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_workload"})
        barrier_state[workload]["exit"].append(monotonic_ns)
        workload_records.append(copied)
        return 0, 0, None
    if kind == "clean_complete":
        validate_exact_keys(message, ("kind", "worker"), "pilot clean completion")
        if worker in handshakes:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "duplicate_pilot_handshake"})
        active_jobs.pop(worker, None)
        handshakes.add(worker)
        return 0, 0, {"ack": True}
    if kind == "unrecoverable_orphan":
        validate_exact_keys(message, ("kind", "worker"), "pilot orphan message")
        raise UnrecoverableOrphan(f"pilot worker {worker} crossed the orphan boundary")
    if kind == "hard_abort":
        validate_exact_keys(message, ("kind", "worker", "reason_code", "context"), "pilot abort message")
        if message.get("reason_code") not in HARD_ABORT_REASON_CODES:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_abort_reason"})
        if not isinstance(message.get("context"), Mapping):
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_abort_context"})
        context = dict(message["context"])
        if "worker" in context and context["worker"] != worker:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_abort_attribution"})
        context.setdefault("worker", worker)
        raise HardAbort(str(message["reason_code"]), context)
    raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "pilot_protocol"})


def transmit_pilot_ack(
    connection: Any,
    response: Mapping[str, Any],
    counters: PilotCounterState,
    update_delta: int,
    token_delta: int,
) -> None:
    values = (counters.attempted_updates, counters.token_positions, update_delta, token_delta)
    if any(type(value) is not int or value < 0 for value in values) or (update_delta == 0) != (token_delta == 0):
        raise ContractError("pilot acknowledgment charge differs")
    counters.attempted_updates += update_delta
    counters.token_positions += token_delta
    connection.send(dict(response))


def run_resource_pilot(
    run_root: Path,
    payload: Mapping[str, Any],
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    transition: TransitionResult,
) -> dict[str, Any]:
    if transition.outcome != "ready" or transition.swap_baseline_bytes is None:
        raise ContractError("pilot transition is not ready")
    writer = transition.writers["run/pilot_resources.jsonl"]
    baseline_rows = writer.validate_committed_prefix()
    if len(baseline_rows) != 1 or baseline_rows[0]["attempted_updates"] != 0:
        raise ContractError("pilot baseline row differs")
    final_frozen_guard(run_root, anchors, signals, "before_pilot_worker_spawn")
    multiprocessing = importlib.import_module("multiprocessing")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    start_event = context.Event()
    failure_latch = PrimaryFailureLatch(payload["abort_rules"]["hard_abort_registry"])
    specifications = tuple(
        {"worker": worker, "target": _pilot_worker, "args": (worker, ordinal, barrier, run_root.name, start_event), "name": f"modular-pilot-{worker}"}
        for worker, ordinal in (("A", 0), ("B", 1))
    )
    try:
        processes, parents = spawn_worker_processes(context, specifications)
    except WorkerStartError as exc:
        latched_error = hard_abort_from_same_poll(failure_latch, [{"reason_code": "worker_exit", "context": {"worker": exc.worker}}])
        if latched_error is None:
            raise UnrecoverableOrphan("pilot worker start failure did not latch") from exc
        raise latched_error
    worker_names = {int(process.pid): process.name.rsplit("-", 1)[-1] for process in processes}
    handshakes: set[str] = set()
    active_jobs: dict[str, dict[str, Any]] = {}
    workload_records = []
    barrier_state: dict[str, dict[str, Any]] = {
        workload: {"entry": [], "exit": [], "arrived": set()} for workload in ("A", "S", "D", "H")
    }
    protocol_states = {worker: pilot_protocol_state(worker) for worker in ("A", "B")}
    counters = PilotCounterState(0, 0)
    sample_id = int(baseline_rows[-1]["sample_id"]) + 1
    next_sample_monotonic_ns = next_resource_sample_monotonic_ns(baseline_rows[-1])
    final_row: dict[str, Any] | None = None
    try:
        final_frozen_guard(run_root, anchors, signals, "before_pilot_worker_release")
        release = signals.commit_guarded(start_event.set)
        if not release.committed or release.pending_signal is not None or signals.pending_signal is not None:
            raise HardAbort("signal_or_interruption", {"signal": release.pending_signal if release.pending_signal is not None else signals.pending_signal, "stage": "before_pilot_worker_release"})
        while True:
            progressed = False
            observations = []
            received_messages = []
            failed_workers = set()
            sampled_row = None
            while True:
                received_in_pass = False
                for worker in ("A", "B"):
                    if worker in failed_workers or worker in handshakes:
                        continue
                    connection = parents[worker]
                    while worker not in failed_workers:
                        try:
                            ready = connection.poll(0)
                        except BaseException as exc:
                            observations.append(parent_worker_failure_observation(exc, worker, True))
                            failed_workers.add(worker)
                            break
                        if not ready:
                            break
                        progressed = True
                        received_in_pass = True
                        try:
                            message = connection.recv()
                        except BaseException as exc:
                            observations.append(parent_worker_failure_observation(exc, worker, True))
                            failed_workers.add(worker)
                            break
                        received_messages.append((worker, connection, message))
                if not received_in_pass:
                    break
            staged_responses = []
            for worker, connection, message in received_messages:
                if worker in failed_workers:
                    continue
                try:
                    update_delta, token_delta, response = _handle_pilot_worker_message(message, worker, connection, handshakes, active_jobs, barrier_state, workload_records, signals, protocol_states[worker])
                    if response is not None:
                        staged_responses.append((worker, connection, response, update_delta, token_delta))
                except BaseException as exc:
                    observations.append(parent_worker_failure_observation(exc, worker, False))
                    failed_workers.add(worker)
            if signals.pending_signal is not None:
                observations.append({"reason_code": "signal_or_interruption", "context": {"signal": signals.pending_signal}})
            if time.monotonic_ns() >= next_sample_monotonic_ns:
                try:
                    row = _resource_sample(run_root.name, "pilot", sample_id, processes, handshakes, worker_names, active_jobs, transition.swap_baseline_bytes, counters.attempted_updates, counters.token_positions)
                    result = writer.append(row)
                    if not result.acknowledged:
                        raise HardAbort(result.reason_code or "artifact_inconsistency")
                    sampled_row = row
                    progressed = True
                    sample_id += 1
                    next_sample_monotonic_ns = next_resource_sample_monotonic_ns(row)
                except BaseException as exc:
                    observations.append(failure_observation_from_exception(exc, "resource_sampler_failure"))
            workers_complete, worker_observations = worker_exit_observations(processes, handshakes, worker_names)
            observations.extend(worker_observations)
            latched_error = hard_abort_from_same_poll(failure_latch, observations)
            if latched_error is not None:
                raise latched_error
            for worker, connection, response, update_delta, token_delta in staged_responses:
                if signals.pending_signal is not None:
                    response_error = hard_abort_from_same_poll(
                        failure_latch,
                        [{"reason_code": "signal_or_interruption", "context": {"signal": signals.pending_signal, "worker": worker}}],
                    )
                    if response_error is None:
                        raise UnrecoverableOrphan("pilot pre-ack signal did not latch")
                    raise response_error
                try:
                    transmit_pilot_ack(connection, response, counters, update_delta, token_delta)
                except BaseException as exc:
                    observation = parent_worker_failure_observation(exc, worker, True)
                    observation["context"]["pilot_attempted_updates"] = counters.attempted_updates
                    observation["context"]["pilot_token_positions"] = counters.token_positions
                    response_error = hard_abort_from_same_poll(failure_latch, [observation])
                    if response_error is None:
                        raise UnrecoverableOrphan("pilot acknowledgment failure did not latch")
                    raise response_error
            if workers_complete and sampled_row is not None and sampled_row["expected_pids"] == [os.getpid()]:
                final_row = sampled_row
                break
            if not progressed:
                time.sleep(0.01)
        if counters.attempted_updates != 88 or counters.token_positions != 225280:
            raise HardAbort("artifact_inconsistency")
        if final_row is None:
            raise HardAbort("artifact_inconsistency")
        writer.recover_uncommitted_suffix()
        writer.close()
    except UnrecoverableOrphan:
        quiesce_worker_processes(processes)
        raise
    except BaseException as exc:
        if isinstance(exc, HardAbort) and exc.primary_latch_monotonic_ns is not None:
            latched_error = exc
        else:
            latched_error = hard_abort_from_same_poll(failure_latch, [failure_observation_from_exception(exc, "artifact_inconsistency")])
            if latched_error is None:
                raise UnrecoverableOrphan("pilot failure did not latch") from exc
        raise quiesce_after_primary_latch(latched_error, processes)
    finally:
        close_parent_connections(parents)
    if len(workload_records) != 8:
        raise HardAbort("artifact_inconsistency")
    workload_records.sort(key=lambda record: (record["worker_ordinal"], record["workload_ordinal"]))
    workload_means: dict[str, list[float]] = {name: [] for name in ("A", "S", "D", "H")}
    for record in workload_records:
        if len(record["warmup_update_ns"]) != 3 or len(record["timed_update_ns"]) != 8:
            raise HardAbort("artifact_inconsistency")
        workload_means[record["workload"]].append(sum(record["timed_update_ns"]) / 8e9)
    t_values = {name: max(values) for name, values in workload_means.items()}
    if any(len(values) != 2 or not math.isfinite(t_values[name]) for name, values in workload_means.items()):
        raise HardAbort("nonfinite")
    tr1 = 1024 * t_values["A"] + 1280 * t_values["S"] + 1536 * t_values["D"]
    projected = 1.35 * max(3 * tr1, 2 * tr1 + 1536 * t_values["H"]) + 2700
    rows = validate_canonical_jsonl_prefix(run_root / "run" / "pilot_resources.jsonl", (run_root / "run" / "pilot_resources.jsonl").stat().st_size, validate_resource_row)
    validate_resource_timeline(rows, "pilot", require_clean_final=True)
    peak_rss = max(row["aggregate_rss_bytes"] for row in rows)
    swap_peak = max(row["swap_used_bytes"] for row in rows)
    swap_growth = max(0, swap_peak - transition.swap_baseline_bytes)
    failures = []
    if projected > payload["pilot"]["proceed_gates"]["Tprojected_seconds_max"]:
        failures.append("pilot_projected_time")
    if peak_rss > 10 * 1024**3:
        failures.append("pilot_resident_memory")
    if swap_growth > 0:
        failures.append("pilot_swap_growth")
    barriers = []
    for workload in ("A", "S", "D", "H"):
        state = barrier_state[workload]
        barriers.append(
            {
                "workload": workload,
                "entry_monotonic_ns": min(state["entry"]),
                "exit_monotonic_ns": max(state["exit"]),
                "arrived_workers": sorted(state["arrived"]),
                "passed": len(state["entry"]) == 2 and len(state["exit"]) == 2 and state["arrived"] == {"A", "B"},
            }
        )
    assertions = [
        _check_record(run_root, run_root.name, "pilot_counter_total", "pilot", {"attempted_updates": 88, "token_positions": 225280}, {"attempted_updates": counters.attempted_updates, "token_positions": counters.token_positions}, None, None, counters.attempted_updates == 88 and counters.token_positions == 225280, ["run/pilot_resources.jsonl"]),
        _check_record(run_root, run_root.name, "pilot_workload_cardinality", "pilot", 8, len(workload_records), None, None, len(workload_records) == 8, ["run/pilot_resources.jsonl"]),
    ]
    if not all(record["pass"] for record in assertions) or not all(record["passed"] for record in barriers):
        raise HardAbort("assertion_failure")
    pilot = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "seed_base": 9999983,
        "workload_order": ["A", "S", "D", "H"],
        "workers": [{"worker": worker_names[int(process.pid)], "ordinal": 0 if worker_names[int(process.pid)] == "A" else 1, "pid": int(process.pid)} for process in sorted(processes, key=lambda value: worker_names[int(value.pid)])],
        "barriers": barriers,
        "workloads": workload_records,
        "tA": t_values["A"],
        "tS": t_values["S"],
        "tD": t_values["D"],
        "tH": t_values["H"],
        "TR1": tr1,
        "Tprojected": projected,
        "peak_rss_bytes": peak_rss,
        "swap_baseline_bytes": transition.swap_baseline_bytes,
        "swap_peak_bytes": swap_peak,
        "swap_growth_bytes": swap_growth,
        "assertions": assertions,
        "decision": "stop" if failures else "proceed",
        "decision_reasons": failures,
    }
    write_canonical_json(run_root / "run" / "pilot.json", pilot)
    return pilot


class CanonicalGzipStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw: Any = None
        self.compressed: Any = None

    def open(self) -> None:
        if self.raw is not None or os.path.lexists(self.path):
            raise ContractError("gzip stream is not fresh")
        self.raw = self.path.open("xb")
        self.compressed = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, compresslevel=9, mtime=0)

    def write(self, record: Mapping[str, Any]) -> None:
        if self.compressed is None:
            raise ContractError("gzip stream is not open")
        self.compressed.write(canonical_json_bytes(record) + b"\n")

    def close(self) -> None:
        if self.compressed is None or self.raw is None:
            raise ContractError("gzip stream is not open")
        self.compressed.close()
        self.compressed = None
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        self.raw = None
        fsync_directory(self.path.parent)


def _selected_attention_oracle_for_model(model: Any, runtime: RuntimeModules, sentinel: Mapping[str, Any]) -> float:
    torch = runtime.torch
    source_module = importlib.import_module("src.model.modular_sources")
    mixer = model.blocks[4].mix
    inputs = torch.tensor(sentinel["inputs"], dtype=torch.float32)
    selected = torch.tensor(sentinel["selected_ids"], dtype=torch.long)
    attention = mixer.source_mixer.attention
    config = mixer.source_mixer.config
    qkv = attention.qkv(inputs).reshape(2, 128, 3, 4, 16)
    query = attention._rope(qkv[:, :, 0].transpose(1, 2))
    key = attention._rope(qkv[:, :, 1].transpose(1, 2))
    value = qkv[:, :, 2].transpose(1, 2)
    selected_output = source_module._PUBLIC_ROUTED.selected_attention(query, key, value, selected, block_size=config.block_size)
    dense_output = source_module._PUBLIC_ROUTED.dense_selected_mask_attention(query, key, value, selected, block_size=config.block_size)
    return float((selected_output - dense_output).abs().max().item())


def _preclaim_golden_hashes(torch: Any) -> dict[str, str]:
    return {
        "rung_one": canonical_json_sha256(generate_rung_one_batch(123456, 2, torch)),
        "rung_two": canonical_json_sha256(generate_rung_two_batch(123456, 2, torch)),
        "random_route": canonical_json_sha256(generate_random_routes(500011, 2, torch)),
        "source_exclusion": canonical_json_sha256(generate_source_exclusion_routes(510011, [[3, -1], [14, 7]], [3, 14], torch)),
    }


def _preclaim_expected_golden_hashes() -> dict[str, str]:
    return {
        "rung_one": "98ff3b54f14306135eafe5a92da7abdf1111cd8690e511188bb5f0e44dcab2a9",
        "rung_two": "7fff37e20adc2241c217b3ed6dad6ec4d85e818d69a59fa5b8e3f5a48f2b8afe",
        "random_route": "18f568b628517fa8f77d9e6adc17c3c2ead62c46070487d416d6eee25953e54c",
        "source_exclusion": "3992c2df698e8787191991d3d5a3edd1eaadbbef647ea342cd70f10453ff9ebf",
    }


def prepare_claim_data(
    run_root: Path,
    runtime: RuntimeModules,
    payload: Mapping[str, Any] | None = None,
    anchors: FrozenManifestAnchors | None = None,
    signals: SignalController | None = None,
    claim_start_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    guard_values = (anchors, signals, claim_start_monotonic_ns)
    if any(value is not None for value in guard_values):
        if anchors is None or signals is None or claim_start_monotonic_ns is None:
            raise ContractError("claim-data guard inputs differ")
        final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_claim_data")
    data_root = ensure_directory(run_root / "data")
    torch = runtime.torch
    frozen_payload = load_prereg_payload() if payload is None else payload
    golden_checks = _preclaim_golden_hashes(torch)
    expected = _preclaim_expected_golden_hashes()
    if golden_checks != expected:
        raise HardAbort("assertion_failure")
    records: dict[str, Any] = {"golden_hashes": golden_checks, "rung_one": {}, "rung_two": {}}
    generated_strata = {}
    first_batch_hashes = {}
    continuous_stream = {}
    histograms = {}
    for construction_seed in RUNG_ONE_SEEDS:
        evaluation_seed = 400000 + construction_seed
        payload = generate_rung_one_batch(evaluation_seed, 512, torch)
        if any(len(set(row)) != 4 for row in payload["rule_blocks"]) or any(len(set(row)) != 4 for row in payload["answer_indices"]):
            raise HardAbort("assertion_failure", {"seed": construction_seed})
        evaluation = {"seed": evaluation_seed, "payload": payload, "payload_sha256": canonical_json_sha256(payload)}
        evaluation_path = data_root / f"r1_eval_{evaluation_seed}.pt"
        _save_torch_artifact(evaluation_path, evaluation, torch)
        random_seed = 500000 + construction_seed
        random_payload = generate_random_routes(random_seed, 512, torch)
        random_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
        random_routes[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long)
        random_record = {"seed": random_seed, "routes": random_routes, "payload": random_payload, "payload_sha256": canonical_json_sha256(random_payload)}
        random_path = data_root / f"r1_random_routes_{construction_seed}.pt"
        _save_torch_artifact(random_path, random_record, torch)
        conditions = torch.tensor(payload["condition"], dtype=torch.long)
        foreign = _batch_local_foreign_conditions(conditions, 32, torch)
        same = int((conditions == foreign).sum())
        generated_strata[construction_seed] = (same, 512 - same)
        stream = torch.Generator(device="cpu")
        stream.manual_seed(100000 + construction_seed)
        first_batch = _continuous_rung_one_batch(stream, 16, torch)
        second_batch = _continuous_rung_one_batch(stream, 16, torch)
        first_hash = _batch_payload_hash(first_batch, "donor")
        second_hash = _batch_payload_hash(second_batch, "donor")
        if first_hash == second_hash:
            raise HardAbort("assertion_failure", {"seed": construction_seed, "stage": "continuous_stream_advancement"})
        first_batch_hashes[str(construction_seed)] = first_hash
        continuous_stream[str(construction_seed)] = {"first": first_hash, "second": second_hash, "distinct": True}
        histograms[str(construction_seed)] = {
            "condition": torch.bincount(conditions, minlength=4).tolist(),
            "required_source": torch.bincount(torch.tensor(payload["required_source"], dtype=torch.long), minlength=10).tolist(),
            "target": torch.bincount(torch.tensor(payload["targets"], dtype=torch.long) - 40, minlength=16).tolist(),
        }
        records["rung_one"][construction_seed] = {
            "evaluation_path": str(evaluation_path.relative_to(run_root)),
            "evaluation_sha256": sha256_file(evaluation_path),
            "payload_sha256": evaluation["payload_sha256"],
            "random_path": str(random_path.relative_to(run_root)),
            "random_sha256": sha256_file(random_path),
            "same_condition": same,
            "changed_condition": 512 - same,
        }
    rung_two_payload = generate_rung_two_batch(1000083, 512, torch)
    rung_two = {"seed": 1000083, "payload": rung_two_payload, "payload_sha256": canonical_json_sha256(rung_two_payload)}
    rung_two_path = data_root / "r2_eval_1000083.pt"
    _save_torch_artifact(rung_two_path, rung_two, torch)
    records["rung_two"] = {
        "evaluation_path": str(rung_two_path.relative_to(run_root)),
        "evaluation_sha256": sha256_file(rung_two_path),
        "payload_sha256": rung_two["payload_sha256"],
    }
    _validate_carry_shuffle_strata(frozen_payload, generated_strata)
    rung_two_stream = torch.Generator(device="cpu")
    rung_two_stream.manual_seed(900083)
    rung_two_first = _continuous_rung_two_batch(rung_two_stream, 8, torch)
    rung_two_second = _continuous_rung_two_batch(rung_two_stream, 8, torch)
    rung_two_first_hash = _batch_payload_hash(rung_two_first, "rung_two")
    rung_two_second_hash = _batch_payload_hash(rung_two_second, "rung_two")
    if rung_two_first_hash == rung_two_second_hash:
        raise HardAbort("assertion_failure", {"seed": 83, "stage": "continuous_stream_advancement"})
    first_batch_hashes["83"] = rung_two_first_hash
    continuous_stream["83"] = {"first": rung_two_first_hash, "second": rung_two_second_hash, "distinct": True}
    rung_two_counts = torch.tensor(rung_two_payload["count"], dtype=torch.long)
    histograms["83"] = {"count": torch.bincount(rung_two_counts, minlength=8).tolist(), "target": torch.bincount(torch.tensor(rung_two_payload["targets"], dtype=torch.long).reshape(-1) - 19, minlength=8).tolist()}
    records["preclaim_assertions"] = {
        "generators": True,
        "golden_hashes": golden_checks,
        "first_batch_hashes": first_batch_hashes,
        "continuous_stream_advancement": continuous_stream,
        "histograms": histograms,
        "seed_separation": len(set(first_batch_hashes.values())) == len(first_batch_hashes),
        "fault_rehearsal": list(FAULT_IDS),
        "carry_shuffle_strata": {str(seed): {"same": generated_strata[seed][0], "changed": generated_strata[seed][1]} for seed in RUNG_ONE_SEEDS},
    }
    if not records["preclaim_assertions"]["seed_separation"]:
        raise HardAbort("assertion_failure", {"stage": "seed_separation"})
    return records


def validate_preclaim_reconstruction(run_root: Path, payload: Mapping[str, Any], runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    golden_hashes = _preclaim_golden_hashes(torch)
    expected_golden = _preclaim_expected_golden_hashes()
    registered_golden = {
        "rung_one": payload["generators"]["rung_one"]["golden"]["payload_sha256"],
        "rung_two": payload["generators"]["rung_two"]["golden"]["payload_sha256"],
        "random_route": payload["generators"]["matched_random_route"]["golden"]["payload_sha256"],
        "source_exclusion": payload["generators"]["source_exclusion"]["golden"]["payload_sha256"],
    }
    if golden_hashes != expected_golden or registered_golden != expected_golden:
        raise HardAbort("endpoint_inconsistency", {"surface": "preclaim_golden_hashes"})
    first_batch_hashes = {}
    continuous_stream = {}
    histograms = {}
    generated_strata = {}
    artifact_hashes = {"rung_one": {}, "rung_two": {}}
    for seed in RUNG_ONE_SEEDS:
        evaluation_seed = 400000 + seed
        evaluation_path = run_root / "data" / f"r1_eval_{evaluation_seed}.pt"
        evaluation = _load_claim_torch_artifact(evaluation_path, torch, f"preclaim_evaluation.{seed}")
        validate_exact_keys(evaluation, ("seed", "payload", "payload_sha256"), "preclaim rung-one evaluation")
        regenerated = generate_rung_one_batch(evaluation_seed, 512, torch)
        if evaluation_path.read_bytes() != _torch_artifact_bytes(evaluation, torch) or evaluation["seed"] != evaluation_seed or evaluation["payload"] != regenerated or evaluation["payload_sha256"] != canonical_json_sha256(regenerated):
            raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "preclaim_evaluation"})
        random_seed = 500000 + seed
        random_path = run_root / "data" / f"r1_random_routes_{seed}.pt"
        random_artifact = _load_claim_torch_artifact(random_path, torch, f"preclaim_random_routes.{seed}")
        validate_exact_keys(random_artifact, ("seed", "routes", "payload", "payload_sha256"), "preclaim random routes")
        random_payload = generate_random_routes(random_seed, 512, torch)
        expected_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
        expected_routes[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long)
        if random_path.read_bytes() != _torch_artifact_bytes(random_artifact, torch) or random_artifact["seed"] != random_seed or random_artifact["payload"] != random_payload or random_artifact["payload_sha256"] != canonical_json_sha256(random_payload) or not torch.equal(random_artifact["routes"], expected_routes):
            raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "preclaim_random_routes"})
        conditions = torch.tensor(regenerated["condition"], dtype=torch.long)
        foreign = _batch_local_foreign_conditions(conditions, 32, torch)
        same = int((conditions == foreign).sum())
        generated_strata[seed] = (same, 512 - same)
        stream = torch.Generator(device="cpu")
        stream.manual_seed(100000 + seed)
        first_hash = _batch_payload_hash(_continuous_rung_one_batch(stream, 16, torch), "donor")
        second_hash = _batch_payload_hash(_continuous_rung_one_batch(stream, 16, torch), "donor")
        if first_hash == second_hash:
            raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "preclaim_continuous_stream"})
        train_rows = _canonical_jsonl_records(run_root / "rung1" / str(seed) / "train.jsonl")
        first_rows = [row for row in train_rows if row.get("stage") == "donor" and row.get("logical_update") == 1]
        if len(first_rows) != 1 or first_rows[0].get("batch_sha256") != first_hash or first_rows[0].get("first_batch_sha256") != first_hash:
            raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "preclaim_first_batch_cross_reference"})
        first_batch_hashes[str(seed)] = first_hash
        continuous_stream[str(seed)] = {"first": first_hash, "second": second_hash, "distinct": True}
        histograms[str(seed)] = {
            "condition": torch.bincount(conditions, minlength=4).tolist(),
            "required_source": torch.bincount(torch.tensor(regenerated["required_source"], dtype=torch.long), minlength=10).tolist(),
            "target": torch.bincount(torch.tensor(regenerated["targets"], dtype=torch.long) - 40, minlength=16).tolist(),
        }
        artifact_hashes["rung_one"][str(seed)] = {
            "evaluation_sha256": sha256_file(evaluation_path),
            "random_sha256": sha256_file(random_path),
        }
    rung_two_path = run_root / "data" / "r2_eval_1000083.pt"
    rung_two_artifact = _load_claim_torch_artifact(rung_two_path, torch, "preclaim_evaluation.83")
    validate_exact_keys(rung_two_artifact, ("seed", "payload", "payload_sha256"), "preclaim rung-two evaluation")
    rung_two_payload = generate_rung_two_batch(1000083, 512, torch)
    if rung_two_path.read_bytes() != _torch_artifact_bytes(rung_two_artifact, torch) or rung_two_artifact["seed"] != 1000083 or rung_two_artifact["payload"] != rung_two_payload or rung_two_artifact["payload_sha256"] != canonical_json_sha256(rung_two_payload):
        raise HardAbort("endpoint_inconsistency", {"seed": 83, "surface": "preclaim_evaluation"})
    rung_two_stream = torch.Generator(device="cpu")
    rung_two_stream.manual_seed(900083)
    rung_two_first_hash = _batch_payload_hash(_continuous_rung_two_batch(rung_two_stream, 8, torch), "rung_two")
    rung_two_second_hash = _batch_payload_hash(_continuous_rung_two_batch(rung_two_stream, 8, torch), "rung_two")
    rung_two_train = _canonical_jsonl_records(run_root / "rung2" / "83" / "train.jsonl")
    rung_two_first_rows = [row for row in rung_two_train if row.get("stage") == "rung_two" and row.get("logical_update") == 1]
    if rung_two_first_hash == rung_two_second_hash or len(rung_two_first_rows) != 1 or rung_two_first_rows[0].get("batch_sha256") != rung_two_first_hash or rung_two_first_rows[0].get("first_batch_sha256") != rung_two_first_hash:
        raise HardAbort("endpoint_inconsistency", {"seed": 83, "surface": "preclaim_first_batch_cross_reference"})
    first_batch_hashes["83"] = rung_two_first_hash
    continuous_stream["83"] = {"first": rung_two_first_hash, "second": rung_two_second_hash, "distinct": True}
    rung_two_counts = torch.tensor(rung_two_payload["count"], dtype=torch.long)
    histograms["83"] = {
        "count": torch.bincount(rung_two_counts, minlength=8).tolist(),
        "target": torch.bincount(torch.tensor(rung_two_payload["targets"], dtype=torch.long).reshape(-1) - 19, minlength=8).tolist(),
    }
    artifact_hashes["rung_two"] = {"evaluation_sha256": sha256_file(rung_two_path)}
    _validate_carry_shuffle_strata(payload, generated_strata)
    seed_separation = len(first_batch_hashes) == 6 and len(set(first_batch_hashes.values())) == 6
    if not seed_separation:
        raise HardAbort("endpoint_inconsistency", {"surface": "preclaim_seed_separation"})
    return {
        "golden_hashes": golden_hashes,
        "first_batch_hashes": first_batch_hashes,
        "continuous_stream_advancement": continuous_stream,
        "histograms": histograms,
        "carry_shuffle_strata": {str(seed): {"same": generated_strata[seed][0], "changed": generated_strata[seed][1]} for seed in RUNG_ONE_SEEDS},
        "seed_separation": seed_separation,
        "artifact_hashes": artifact_hashes,
    }


def _batch_payload_hash(batch: Mapping[str, Any], stage: str) -> str:
    if stage == "rung_two":
        payload = {
            "tokens": batch["tokens"].tolist(),
            "targets": batch["targets"].tolist(),
            "count": batch["count"].tolist(),
            "count_positions": [row.tolist() for row in batch["count_positions"]],
        }
    else:
        payload = {
            "tokens": batch["tokens"].tolist(),
            "targets": batch["targets"].tolist(),
            "condition": batch["condition"].tolist(),
            "rule_blocks": batch["rule_blocks"].tolist(),
            "answer_indices": batch["answer_indices"].tolist(),
            "required_source": batch["required_source"].tolist(),
        }
    return canonical_json_sha256(payload)


def _train_row_from_pair(started: Mapping[str, Any], completed: Mapping[str, Any], first_hash: str | None) -> dict[str, Any]:
    metrics = completed["metrics"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": started["run_id"],
        "rung": started["rung"],
        "claim_seed": started["claim_seed"],
        "construction_seed": started["construction_seed"],
        "model": started["model"],
        "stage": started["stage"],
        "logical_update": started["logical_update"],
        "attempt_id": started["attempt_id"],
        "started_event_sequence": started["event_sequence"],
        "completed_event_sequence": completed["event_sequence"],
        "batch_sha256": started["batch_sha256"],
        "learning_rates": metrics["learning_rates"],
        "component_losses": metrics["component_losses"],
        "total_loss": metrics["total_loss"],
        "gradient_norm": metrics["gradient_norm"],
        "clip_result": metrics["clip_result"],
        "examples": started["examples"],
        "token_positions": started["token_positions"],
        "raw_overflow_count": metrics["raw_overflow_count"],
        "max_bucket_load": metrics["max_bucket_load"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "finite": metrics["finite"],
        "first_batch_sha256": first_hash,
    }


def _initialize_audit(model: Any, stage: str, runtime: RuntimeModules, membership: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = {}
    for name, parameter in model.named_parameters():
        meta = membership[name]
        records[name] = {
            "model": "",
            "stage": stage,
            "name": name,
            "block": int(name.split(".")[1]) if name.startswith("blocks.") else None,
            "shape": list(parameter.shape),
            "category": meta["category"],
            "requires_grad": meta["requires_grad"],
            "optimizer_member": meta["requires_grad"],
            "parameter_group": meta["parameter_group"],
            "peak_lr": meta["peak_lr"],
            "weight_decay": meta["weight_decay"],
            "grad_none_steps": 0,
            "grad_zero_steps": 0,
            "grad_nonzero_steps": 0,
            "grad_nonfinite_steps": 0,
            "update_zero_steps": 0,
            "update_nonzero_steps": 0,
            "update_nonfinite_steps": 0,
            "first_nonzero_step": None,
            "start_sha256": _tensor_sha256(parameter),
            "end_sha256": None,
            "update_l2": None,
            "update_max_abs": None,
            "classification": None,
            "_start": parameter.detach().clone(),
        }
    return records


def _finalize_audit(audit: Mapping[str, dict[str, Any]], model: Any, model_name: str, torch: Any) -> list[dict[str, Any]]:
    result = []
    parameters = dict(model.named_parameters())
    for name in sorted(audit):
        record = audit[name]
        parameter = parameters[name]
        delta = parameter.detach().to(torch.float64) - record.pop("_start").to(torch.float64)
        record["model"] = model_name
        record["end_sha256"] = _tensor_sha256(parameter)
        if record["requires_grad"]:
            record["update_l2"] = float(delta.square().sum().sqrt())
            record["update_max_abs"] = float(delta.abs().max()) if delta.numel() else 0.0
        if record["grad_nonfinite_steps"] or record["update_nonfinite_steps"]:
            record["classification"] = "nonfinite_failure"
        elif not record["requires_grad"]:
            record["classification"] = "frozen_by_design"
        elif record["grad_nonzero_steps"] and record["update_nonzero_steps"]:
            record["classification"] = "learned_with_evidence"
        elif record["update_nonzero_steps"]:
            if not record["weight_decay"]:
                raise HardAbort("artifact_inconsistency", {"stage": record["stage"], "surface": f"unexplained_update.{name}"})
            record["classification"] = "updated_only_by_decay"
        else:
            record["classification"] = "trainable_but_no_gradient"
        result.append(record)
    return result


def validate_gradient_audit(records: Sequence[Mapping[str, Any]], expected_updates: Mapping[str, int]) -> None:
    keys = (
        "model",
        "stage",
        "name",
        "block",
        "shape",
        "category",
        "requires_grad",
        "optimizer_member",
        "parameter_group",
        "peak_lr",
        "weight_decay",
        "grad_none_steps",
        "grad_zero_steps",
        "grad_nonzero_steps",
        "grad_nonfinite_steps",
        "update_zero_steps",
        "update_nonzero_steps",
        "update_nonfinite_steps",
        "first_nonzero_step",
        "start_sha256",
        "end_sha256",
        "update_l2",
        "update_max_abs",
        "classification",
    )
    if not records or not expected_updates or any(type(value) is not int or value <= 0 for value in expected_updates.values()):
        raise ContractError("gradient audit contract is empty")
    expected_models = {
        "donor": "all_eligible_donor",
        "router_only": "selected",
        "joint": "selected",
        "dense_base": "dense_causal",
        "dense_continuation": "dense_causal",
        "rung_two": "rung_two",
    }
    classifications = {"learned_with_evidence", "trainable_but_no_gradient", "frozen_by_design", "updated_only_by_decay", "nonfinite_failure"}
    categories = {"matrix", "normalization_scale", "recurrent_bias", "codebook"}
    identities = []
    lookup = {}
    for record in records:
        validate_exact_keys(record, keys, "gradient audit record")
        stage = record["stage"]
        if stage not in expected_updates:
            raise ContractError("gradient audit stage differs")
        updates = expected_updates[stage]
        if record["model"] != expected_models.get(stage) or not isinstance(record["name"], str) or not record["name"]:
            raise ContractError("gradient audit model or parameter identity differs")
        if not isinstance(record["shape"], list) or any(type(value) is not int or value < 0 for value in record["shape"]):
            raise ContractError("gradient audit shape differs")
        expected_block = int(record["name"].split(".")[1]) if re.match(r"blocks\.[0-7]\.", record["name"]) else None
        if record["block"] != expected_block or record["category"] not in categories or type(record["requires_grad"]) is not bool or type(record["optimizer_member"]) is not bool:
            raise ContractError("gradient audit parameter metadata differs")
        for field in ("grad_none_steps", "grad_zero_steps", "grad_nonzero_steps", "grad_nonfinite_steps", "update_zero_steps", "update_nonzero_steps", "update_nonfinite_steps"):
            if type(record[field]) is not int or record[field] < 0:
                raise ContractError("gradient audit counter type differs")
        for field in ("start_sha256", "end_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", record[field] or "") is None:
                raise ContractError("gradient audit tensor digest differs")
        for field in ("peak_lr", "weight_decay", "update_l2", "update_max_abs"):
            value = record[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0):
                raise ContractError("gradient audit numeric value differs")
        if record["first_nonzero_step"] is not None and (type(record["first_nonzero_step"]) is not int or not 1 <= record["first_nonzero_step"] <= updates):
            raise ContractError("gradient audit first nonzero step differs")
        if (record["grad_nonzero_steps"] == 0) != (record["first_nonzero_step"] is None):
            raise ContractError("gradient audit first nonzero closure differs")
        if record["classification"] not in classifications or record["classification"] == "nonfinite_failure":
            raise ContractError("gradient audit classification differs")
        grad_total = sum(record[field] for field in ("grad_none_steps", "grad_zero_steps", "grad_nonzero_steps", "grad_nonfinite_steps"))
        update_total = sum(record[field] for field in ("update_zero_steps", "update_nonzero_steps", "update_nonfinite_steps"))
        if grad_total != updates or record["grad_nonfinite_steps"] != 0 or record["update_nonfinite_steps"] != 0:
            raise ContractError("gradient audit counters differ")
        if record["requires_grad"] is not record["optimizer_member"]:
            raise ContractError("gradient audit optimizer membership differs")
        if record["requires_grad"]:
            if update_total != updates or record["parameter_group"] is None or record["peak_lr"] is None or record["weight_decay"] is None or record["update_l2"] is None or record["update_max_abs"] is None:
                raise ContractError("trainable gradient audit evidence differs")
            if not isinstance(record["parameter_group"], str) or not record["parameter_group"]:
                raise ContractError("gradient audit parameter group differs")
        else:
            if update_total != 0 or record["parameter_group"] is not None or record["peak_lr"] is not None or record["weight_decay"] is not None or record["update_l2"] is not None or record["update_max_abs"] is not None or record["start_sha256"] != record["end_sha256"] or record["classification"] != "frozen_by_design":
                raise ContractError("frozen gradient audit evidence differs")
        classification = record["classification"]
        if classification == "learned_with_evidence" and not (record["grad_nonzero_steps"] > 0 and record["update_nonzero_steps"] > 0):
            raise ContractError("learned classification lacks gradient and update evidence")
        if classification == "updated_only_by_decay" and not (record["grad_nonzero_steps"] == 0 and record["update_nonzero_steps"] > 0 and record["weight_decay"] > 0):
            raise ContractError("decay-only classification differs")
        if record["category"] == "codebook" and record["requires_grad"]:
            if classification != "trainable_but_no_gradient" or record["grad_nonzero_steps"] != 0 or record["update_nonzero_steps"] != 0 or record["start_sha256"] != record["end_sha256"] or record["update_l2"] != 0 or record["update_max_abs"] != 0:
                raise ContractError("trainable discrete codebook evidence differs")
        identity = (stage, record["name"])
        identities.append(identity)
        lookup[identity] = record
    if len(identities) != len(set(identities)):
        raise ContractError("gradient audit identity is duplicated")
    if {stage for stage, _ in identities} != set(expected_updates):
        raise ContractError("gradient audit stage closure differs")
    names_by_stage = {stage: {name for observed_stage, name in identities if observed_stage == stage} for stage in expected_updates}
    if len({tuple(sorted(names)) for names in names_by_stage.values()}) != 1:
        raise ContractError("gradient audit parameter closure differs between stages")
    for stage in set(expected_updates) & {"router_only", "joint"}:
        for suffix in ("query_projection.weight", "key_projection.weight"):
            name = f"blocks.4.mix.source_mixer.attention.router.{suffix}"
            record = lookup.get((stage, name))
            if record is None or record["classification"] != "learned_with_evidence":
                raise ContractError("block-4 route parameter lacks learning evidence")


def _expected_canonical_bypass_evidence(selected_width: int, sequence_length: int) -> list[dict[str, Any]]:
    if type(selected_width) is not int or selected_width not in {0, 2, 15} or type(sequence_length) is not int or sequence_length != 128:
        raise ContractError("canonical bypass geometry differs")
    if selected_width == 0:
        return []
    records = []
    for position in range(sequence_length):
        remote_limit = position // 8
        if remote_limit > selected_width:
            continue
        raw_ids = list(range(min(remote_limit, selected_width)))
        raw_ids.extend([-1] * (selected_width - len(raw_ids)))
        effective_ids = list(raw_ids) if position == 126 else [-1] * selected_width
        records.append(
            {
                "position": position,
                "remote_limit": remote_limit,
                "raw_remote_ids": raw_ids,
                "effective_remote_ids": effective_ids,
            }
        )
    return records


def _canonical_bypass_evidence(raw_remote: Any, effective_remote: Any) -> list[dict[str, Any]]:
    torch = _torch_module()
    if not isinstance(raw_remote, torch.Tensor) or not isinstance(effective_remote, torch.Tensor):
        raise ContractError("canonical bypass route tensors are absent")
    if raw_remote.dtype != torch.long or effective_remote.dtype != torch.long or raw_remote.device.type != "cpu" or effective_remote.device.type != "cpu":
        raise ContractError("canonical bypass route tensor type differs")
    if raw_remote.shape != effective_remote.shape or raw_remote.ndim != 4 or raw_remote.shape[0] <= 0 or tuple(raw_remote.shape[1:3]) != (128, 1):
        raise ContractError("canonical bypass route tensor shape differs")
    width = int(raw_remote.shape[-1])
    expected = _expected_canonical_bypass_evidence(width, 128)
    outside_query = torch.arange(128, device=effective_remote.device) != 126
    if bool((effective_remote[:, outside_query] != -1).any()):
        raise ContractError("effective route violates the query-only firewall")
    for record in expected:
        position = record["position"]
        raw_pattern = torch.tensor(record["raw_remote_ids"], dtype=torch.long, device=raw_remote.device).view(1, 1, width)
        effective_pattern = torch.tensor(record["effective_remote_ids"], dtype=torch.long, device=effective_remote.device).view(1, 1, width)
        if not torch.equal(raw_remote[:, position], raw_pattern.expand(raw_remote.shape[0], raw_remote.shape[2], width)):
            raise ContractError("raw canonical bypass rows differ")
        if not torch.equal(effective_remote[:, position], effective_pattern.expand(effective_remote.shape[0], effective_remote.shape[2], width)):
            raise ContractError("effective canonical bypass rows differ")
    return expected


def _routing_histogram_records(tensor: Any, key_name: str, count_name: str, allow_empty: bool) -> list[dict[str, int]]:
    torch = _torch_module()
    if key_name not in {"load", "valid_posting_count"} or count_name not in {"bucket_count", "search_row_count"} or type(allow_empty) is not bool:
        raise ContractError("routing histogram schema differs")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.long or tensor.device.type != "cpu" or tensor.requires_grad or tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ContractError("routing histogram tensor differs")
    if tensor.shape[0] == 0 and not allow_empty:
        raise ContractError("routing histogram is unexpectedly empty")
    records = []
    prior = -1
    for key, count in tensor.tolist():
        if type(key) is not int or type(count) is not int or key <= prior or key < 0 or count <= 0:
            raise ContractError("routing histogram order or value differs")
        records.append({key_name: key, count_name: count})
        prior = key
    return records


def _routing_rows(
    output: Any,
    run_id: str,
    construction_seed: int,
    phase: str,
    model_name: str,
    stage: str | None,
    condition: str | None,
    logical_update: int | None,
    forward_sequence: int,
    batch_index: int | None,
    example_offset: int,
    required_source: Any | None,
    foreign_source: Any | None,
    intervention: str | None,
    checkpoint_sha256: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for block_execution in output.blocks:
        routed = block_execution.mixer_output
        if routed is None or not hasattr(routed, "telemetry"):
            continue
        telemetry = routed.telemetry
        raw_full = telemetry["raw_remote"]
        effective_full = telemetry["effective_remote"]
        canonical_bypass = _canonical_bypass_evidence(raw_full, effective_full)
        block_load_histogram = _routing_histogram_records(telemetry.get("block_load_histogram"), "load", "bucket_count", False)
        valid_posting_histogram = _routing_histogram_records(telemetry.get("valid_posting_histogram"), "valid_posting_count", "search_row_count", True)
        call = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "rung": 1,
            "claim_seed": construction_seed,
            "construction_seed": construction_seed,
            "row_kind": "call_summary",
            "phase": phase,
            "model": model_name,
            "stage": stage,
            "condition": condition,
            "logical_update": logical_update,
            "forward_sequence": forward_sequence,
            "block": block_execution.block_index,
            "batch_index": batch_index,
            "example_index": None,
            "query_position": None,
            "required_source": None,
            "foreign_source": None,
            "raw_remote_ids": None,
            "effective_remote_ids": None,
            "local_block_ids": None,
            "query_underfill_count": None,
            "original_source_hit": None,
            "foreign_source_hit": None,
            "intervention": None,
            "canonical_bypass_ids": canonical_bypass,
            "block_load_histogram": block_load_histogram,
            "valid_posting_histogram": valid_posting_histogram,
            "addresses_probed": int(telemetry["addresses_probed"]),
            "posting_reads": int(telemetry["postings_read"]),
            "candidate_blocks": int(telemetry["candidate_blocks"]),
            "overflow_count": int(telemetry["overflow_count"]),
            "max_bucket_load": int(telemetry["max_bucket_load"]),
            "route_workspace_bytes": int(telemetry["workspace_bytes"]),
            "checkpoint_sha256": checkpoint_sha256,
        }
        rows.append(call)
        raw = raw_full[:, 126, 0]
        effective = effective_full[:, 126, 0]
        for row_index in range(raw.size(0)):
            source_value = None if required_source is None else int(required_source[row_index])
            foreign_value = None if foreign_source is None else int(foreign_source[row_index])
            raw_ids = [int(value) for value in raw[row_index].tolist()]
            if phase == "route_acquisition":
                effective_ids = None
                original_hit = None
                foreign_hit = None
                underfill = None
                row_intervention = None
            else:
                effective_ids = [int(value) for value in effective[row_index].tolist()]
                original_hit = None if source_value is None else source_value in effective_ids
                foreign_hit = None if foreign_value is None else foreign_value in effective_ids
                underfill = sum(value == -1 for value in effective_ids)
                row_intervention = intervention
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "rung": 1,
                    "claim_seed": construction_seed,
                    "construction_seed": construction_seed,
                    "row_kind": "query_example",
                    "phase": phase,
                    "model": model_name,
                    "stage": stage,
                    "condition": condition,
                    "logical_update": logical_update,
                    "forward_sequence": forward_sequence,
                    "block": block_execution.block_index,
                    "batch_index": batch_index,
                    "example_index": example_offset + row_index,
                    "query_position": 126,
                    "required_source": source_value,
                    "foreign_source": foreign_value,
                    "raw_remote_ids": raw_ids,
                    "effective_remote_ids": effective_ids,
                    "local_block_ids": [15],
                    "query_underfill_count": underfill,
                    "original_source_hit": original_hit,
                    "foreign_source_hit": foreign_hit,
                    "intervention": row_intervention,
                    "canonical_bypass_ids": None,
                    "block_load_histogram": None,
                    "valid_posting_histogram": None,
                    "addresses_probed": None,
                    "posting_reads": None,
                    "candidate_blocks": None,
                    "overflow_count": None,
                    "max_bucket_load": None,
                    "route_workspace_bytes": None,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
    return rows


def _attempt_event(
    run_id: str,
    rung: int,
    construction_seed: int,
    event_sequence: int,
    event: str,
    model_name: str,
    stage: str,
    logical_update: int,
    examples: int,
    token_positions: int,
    batch_sha256: str,
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rung": rung,
        "claim_seed": construction_seed,
        "construction_seed": construction_seed,
        "event_sequence": event_sequence,
        "event": event,
        "attempt_id": attempt_id(run_id, rung, construction_seed, model_name, stage, logical_update),
        "model": model_name,
        "stage": stage,
        "logical_update": logical_update,
        "examples": examples,
        "token_positions": token_positions,
        "batch_sha256": batch_sha256,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": utc_now(),
        "metrics": None if metrics is None else dict(metrics),
    }


def _tensor_tree_storage(value: Any, torch: Any) -> tuple[int, int]:
    count = 0
    size = 0
    seen: set[int] = set()

    def visit(child: Any) -> None:
        nonlocal count, size
        if isinstance(child, torch.Tensor):
            identity = id(child)
            if identity not in seen:
                seen.add(identity)
                count += int(child.numel())
                size += int(child.numel() * child.element_size())
        elif isinstance(child, Mapping):
            for key in sorted(child, key=lambda item: str(item)):
                visit(child[key])
        elif isinstance(child, (list, tuple)):
            for item in child:
                visit(item)

    visit(value)
    return count, size


def _validate_checkpoint(checkpoint: Mapping[str, Any], expected: Mapping[str, Any], torch: Any) -> None:
    keys = (
        "schema_version",
        "run_id",
        "rung",
        "construction_seed",
        "model",
        "stage",
        "completed_update",
        "last_attempt_id",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "python_rng_state",
        "torch_rng_state",
        "generator_states",
        "final_batch_sha256",
    )
    validate_exact_keys(checkpoint, keys, "checkpoint")
    for name, value in expected.items():
        if checkpoint[name] != value:
            raise HardAbort("endpoint_inconsistency", {"surface": f"checkpoint.{name}"})
    if not isinstance(checkpoint["model_state_dict"], Mapping) or not isinstance(checkpoint["optimizer_state_dict"], Mapping) or not isinstance(checkpoint["scheduler_state_dict"], Mapping):
        raise HardAbort("endpoint_inconsistency", {"surface": "checkpoint.state"})
    generator_names = list(checkpoint["generator_states"])
    if generator_names != sorted(generator_names) or len(generator_names) != 1:
        raise HardAbort("endpoint_inconsistency", {"surface": "checkpoint.generators"})
    for name in checkpoint["model_state_dict"]:
        lowered = name.lower()
        if any(fragment in lowered for fragment in ("persistent_state", "postings", "raw_remote", "effective_remote", "selected_blocks")):
            raise HardAbort("endpoint_inconsistency", {"surface": f"checkpoint.model_state_dict.{name}"})
    _assert_finite_tree(torch, checkpoint["model_state_dict"], {}, "checkpoint.model_state_dict")
    _assert_finite_tree(torch, checkpoint["optimizer_state_dict"], {}, "checkpoint.optimizer_state_dict")
    _assert_finite_tree(torch, checkpoint["scheduler_state_dict"], {}, "checkpoint.scheduler_state_dict")
    _assert_finite_tree(torch, checkpoint["torch_rng_state"], {}, "checkpoint.torch_rng_state")
    _assert_finite_tree(torch, checkpoint["generator_states"], {}, "checkpoint.generator_states")


def _fresh_reload_evidence(
    model: Any,
    checkpoint: Mapping[str, Any],
    batch: Mapping[str, Any],
    rung: int,
    runtime: RuntimeModules,
) -> dict[str, Any]:
    torch = runtime.torch
    with torch.random.fork_rng(devices=[]):
        fresh = type(model)(model.config)
    fresh.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model_state = {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()}
    fresh_state = {name: _tensor_sha256(tensor) for name, tensor in fresh.state_dict().items()}
    model.eval()
    fresh.eval()
    with torch.inference_mode():
        live_output = model(batch["tokens"], return_aux=True, route_detail=True)
        fresh_output = fresh(batch["tokens"], return_aux=True, route_detail=True)
    _assert_finite_tree(torch, live_output, {}, "endpoint.live")
    _assert_finite_tree(torch, fresh_output, {}, "endpoint.fresh")
    logits_error = float((live_output.logits - fresh_output.logits).abs().max())
    hidden_error = float((live_output.hidden - fresh_output.hidden).abs().max())
    route_exact = True
    for live_block, fresh_block in zip(live_output.blocks, fresh_output.blocks):
        live_routed = live_block.mixer_output
        fresh_routed = fresh_block.mixer_output
        if live_routed is not None and hasattr(live_routed, "telemetry"):
            route_exact = route_exact and torch.equal(live_routed.telemetry["raw_remote"], fresh_routed.telemetry["raw_remote"]) and torch.equal(live_routed.telemetry["effective_remote"], fresh_routed.telemetry["effective_remote"])
    passed = model_state == fresh_state and logits_error <= 1e-7 and hidden_error <= 1e-7 and route_exact
    if not passed:
        raise HardAbort("endpoint_inconsistency", {"surface": "fresh_reload"})
    return {
        "fresh_instance": True,
        "state_exact": model_state == fresh_state,
        "logits_max_error": logits_error,
        "hidden_max_error": hidden_error,
        "route_exact": route_exact,
        "rung": rung,
    }


def _train_stage(
    connection: Any,
    worker: str,
    run_root: Path,
    model: Any,
    model_name: str,
    stage: str,
    rung: int,
    construction_seed: int,
    generator_seed: int,
    updates: int,
    warmup: int,
    batch_size: int,
    event_sequence: int,
    runtime: RuntimeModules,
    checkpoint_path: Path,
    routing_stream: CanonicalGzipStream | None,
    forward_sequence: int,
) -> tuple[int, int, list[dict[str, Any]], list[dict[str, Any]], str, int, int, dict[str, Any], dict[str, int]]:
    torch = runtime.torch
    model_module = runtime.model_module
    optimizer, _, membership = _make_optimizer(model, stage, runtime)
    audit = _initialize_audit(model, stage, runtime, membership)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    train_rows = []
    last_batch_hash = ""
    first_batch_hash = ""
    last_attempt = ""
    overflow_total = 0
    max_bucket_load = 0
    peak_index_count = 0
    peak_index_bytes = 0
    peak_workspace_count = 0
    peak_workspace_bytes = 0
    model.train()
    for logical_update in range(1, updates + 1):
        batch = _continuous_rung_two_batch(generator, batch_size, torch) if rung == 2 else _continuous_rung_one_batch(generator, batch_size, torch)
        batch_sha256 = _batch_payload_hash(batch, stage)
        if logical_update == 1:
            first_batch_hash = batch_sha256
        last_batch_hash = batch_sha256
        started = _attempt_event(run_root.name, rung, construction_seed, event_sequence, "started", model_name, stage, logical_update, batch_size, int(batch["tokens"].numel()), batch_sha256, None)
        _child_send_and_wait(connection, {"kind": "attempt", "worker": worker, "seed": construction_seed, "stage": stage, "logical_update": logical_update, "row": started})
        optimizer.zero_grad(set_to_none=True)
        multiplier = _stage_schedule_multiplier(logical_update, updates, warmup)
        learning_rates = _set_optimizer_rates(optimizer, multiplier)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        start_time = time.perf_counter_ns()
        loss, components, overflow, maximum, output = _training_forward_loss(model, batch, stage, runtime)
        finite_context = {"worker": worker, "seed": construction_seed, "stage": stage, "logical_update": logical_update}
        _assert_finite_tree(torch, loss, finite_context, "loss")
        _assert_finite_tree(torch, components, finite_context, "component_losses")
        _assert_finite_tree(torch, output, finite_context, "model_output")
        observed_overflow, observed_maximum, index_count, index_bytes, workspace_count, workspace_bytes = _route_observation(output, rung, finite_context)
        if observed_overflow != overflow or observed_maximum != maximum:
            raise HardAbort("artifact_inconsistency", {**finite_context, "surface": "routing_reduction"})
        peak_index_count = max(peak_index_count, index_count)
        peak_index_bytes = max(peak_index_bytes, index_bytes)
        peak_workspace_count = max(peak_workspace_count, workspace_count)
        peak_workspace_bytes = max(peak_workspace_bytes, workspace_bytes)
        loss.backward()
        _assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)
        for name, parameter in model.named_parameters():
            record = audit[name]
            gradient = parameter.grad
            if gradient is None:
                record["grad_none_steps"] += 1
            elif not bool(torch.isfinite(gradient).all()):
                record["grad_nonfinite_steps"] += 1
            elif bool((gradient != 0).any()):
                record["grad_nonzero_steps"] += 1
                if record["first_nonzero_step"] is None:
                    record["first_nonzero_step"] = logical_update
            else:
                record["grad_zero_steps"] += 1
        gradient_norm = _clip_gradient_norm_finite(torch, model, optimizer, finite_context)
        optimizer.step()
        _assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)
        stop_time = time.perf_counter_ns()
        elapsed = elapsed_seconds_from_monotonic_ns(start_time, stop_time)
        for name, parameter in model.named_parameters():
            record = audit[name]
            if not parameter.requires_grad:
                continue
            delta = parameter.detach() - before[name]
            if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(delta).all()):
                record["update_nonfinite_steps"] += 1
            elif bool((delta != 0).any()):
                record["update_nonzero_steps"] += 1
            else:
                record["update_zero_steps"] += 1
        if overflow:
            raise HardAbort("route_overflow", {"worker": worker, "seed": construction_seed, "stage": stage, "logical_update": logical_update})
        overflow_total += overflow
        max_bucket_load = max(max_bucket_load, maximum)
        if routing_stream is not None and rung == 1:
            for routing_row in _routing_rows(output, run_root.name, construction_seed, "training", model_name, stage, None, logical_update, forward_sequence, None, 0, batch["required_source"], None, None, None):
                routing_stream.write(routing_row)
            forward_sequence += 1
        metrics = {
            "learning_rates": learning_rates,
            "component_losses": components,
            "total_loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
            "clip_result": "clipped" if gradient_norm > 1.0 else "unchanged",
            "raw_overflow_count": overflow,
            "max_bucket_load": maximum,
            "elapsed_seconds": elapsed,
            "finite": True,
        }
        completed = _attempt_event(run_root.name, rung, construction_seed, event_sequence + 1, "completed", model_name, stage, logical_update, batch_size, int(batch["tokens"].numel()), batch_sha256, metrics)
        _child_send_and_wait(connection, {"kind": "attempt", "worker": worker, "seed": construction_seed, "stage": stage, "logical_update": logical_update, "row": completed})
        train_rows.append(_train_row_from_pair(started, completed, first_batch_hash if logical_update == 1 else None))
        last_attempt = completed["attempt_id"]
        event_sequence += 2
    checkpoint_stage = {
        "donor": "donor",
        "router_only": "router",
        "joint": "joint",
        "dense_base": "dense_base",
        "dense_continuation": "dense",
        "rung_two": "rung2",
    }[stage]
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "rung": rung,
        "construction_seed": construction_seed,
        "model": model_name,
        "stage": checkpoint_stage,
        "completed_update": updates,
        "last_attempt_id": last_attempt,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": {"kind": "linear_warmup_then_cosine_to_tenth_peak", "updates": updates, "warmup_updates": warmup, "completed_update": updates},
        "python_rng_state": importlib.import_module("random").getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "generator_states": {f"{stage}_data": generator.get_state()},
        "final_batch_sha256": last_batch_hash,
    }
    _assert_finite_tree(torch, checkpoint["model_state_dict"], {"worker": worker, "seed": construction_seed, "stage": stage}, "checkpoint.model_state_dict")
    _assert_finite_tree(torch, checkpoint["optimizer_state_dict"], {"worker": worker, "seed": construction_seed, "stage": stage}, "checkpoint.optimizer_state_dict")
    _save_torch_artifact(checkpoint_path, checkpoint, torch)
    expected_state_hashes = {name: _tensor_sha256(tensor) for name, tensor in model.state_dict().items()}
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(
        loaded,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_root.name,
            "rung": rung,
            "construction_seed": construction_seed,
            "model": model_name,
            "stage": checkpoint_stage,
            "completed_update": updates,
            "last_attempt_id": last_attempt,
            "final_batch_sha256": last_batch_hash,
        },
        torch,
    )
    actual_state_hashes = {name: _tensor_sha256(tensor) for name, tensor in loaded["model_state_dict"].items()}
    if expected_state_hashes != actual_state_hashes:
        raise HardAbort("endpoint_inconsistency", {"worker": worker, "seed": construction_seed, "stage": stage})
    reload_evidence = _fresh_reload_evidence(model, loaded, batch, rung, runtime)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    audit_records = _finalize_audit(audit, model, model_name, torch)
    optimizer_count, optimizer_bytes = _tensor_tree_storage(loaded["optimizer_state_dict"], torch)
    dynamic_count = len(model_module.RECURRENT_BLOCK_INDICES) * batch_size * model.config.heads * model.config.recurrent_head_width * model.config.recurrent_head_width
    runtime_accounting = {
        "dynamic_recurrent_state_count": dynamic_count,
        "dynamic_recurrent_state_bytes": dynamic_count * 4,
        "route_index_storage_count": peak_index_count,
        "route_index_storage_bytes": peak_index_bytes,
        "routing_workspace_count": peak_workspace_count,
        "routing_workspace_bytes": peak_workspace_bytes,
        "optimizer_state_count": optimizer_count,
        "optimizer_state_bytes": optimizer_bytes,
    }
    del optimizer
    return event_sequence, forward_sequence, train_rows, audit_records, checkpoint_sha256, overflow_total, max_bucket_load, reload_evidence, runtime_accounting


def _child_exchange(connection: Any, message: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        connection.send(dict(message))
        response = connection.recv()
    except (EOFError, BrokenPipeError, ConnectionError, OSError) as exc:
        raise HardAbort("worker_exit") from exc
    expected_keys = ("ack", "sample_ids") if message.get("kind") == "resource_refs" else ("ack",)
    if not isinstance(response, Mapping) or tuple(sorted(response)) != tuple(sorted(expected_keys)) or response.get("ack") is not True:
        raise HardAbort("worker_exit")
    if "sample_ids" in response and (not isinstance(response["sample_ids"], list) or any(type(value) is not int or value < 0 for value in response["sample_ids"]) or response["sample_ids"] != sorted(set(response["sample_ids"]))):
        raise HardAbort("worker_exit")
    return response


def clean_completion_handshake(worker: str, last_event_sequence_by_construction_seed: Mapping[int, int]) -> dict[str, Any]:
    if worker not in WORKER_JOB_ASSIGNMENTS:
        raise ContractError("clean completion worker differs")
    expected_seeds = WORKER_ASSIGNMENTS[worker]
    if tuple(sorted(last_event_sequence_by_construction_seed)) != tuple(sorted(expected_seeds)):
        raise ContractError("clean completion event map differs")
    handshake = {
        "worker": worker,
        "status": "clean_complete",
        "assigned_jobs": [dict(job) for job in WORKER_JOB_ASSIGNMENTS[worker]],
        "last_event_sequence_by_construction_seed": {str(seed): last_event_sequence_by_construction_seed[seed] for seed in expected_seeds},
        "artifacts_fsynced": True,
    }
    validate_clean_completion_handshake(handshake, worker)
    return handshake


def validate_clean_completion_handshake(handshake: Mapping[str, Any], expected_worker: str) -> None:
    keys = ("worker", "status", "assigned_jobs", "last_event_sequence_by_construction_seed", "artifacts_fsynced")
    validate_exact_keys(handshake, keys, "clean completion handshake")
    if expected_worker not in WORKER_JOB_ASSIGNMENTS or handshake["worker"] != expected_worker or handshake["status"] != "clean_complete" or handshake["artifacts_fsynced"] is not True:
        raise ContractError("clean completion handshake identity differs")
    if handshake["assigned_jobs"] != [dict(job) for job in WORKER_JOB_ASSIGNMENTS[expected_worker]]:
        raise ContractError("clean completion assigned jobs differ")
    expected_seeds = WORKER_ASSIGNMENTS[expected_worker]
    sequences = handshake["last_event_sequence_by_construction_seed"]
    if not isinstance(sequences, Mapping) or tuple(sequences) != tuple(str(seed) for seed in expected_seeds):
        raise ContractError("clean completion event map differs")
    if any(type(sequences[str(seed)]) is not int or sequences[str(seed)] < 0 for seed in expected_seeds):
        raise ContractError("clean completion event sequence differs")


def _construct_seeded_model(role: str, seed: int, runtime: RuntimeModules) -> tuple[Any, str, str]:
    torch = runtime.torch
    before = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if role == "rung_two":
            model = runtime.model_module.ModularNeuralMachine(runtime.model_module.rung_two_config())
        else:
            model = runtime.model_module.ModularNeuralMachine(runtime.model_module.rung_one_config(role))
        inside_after = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
    after = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
    if before != after:
        raise HardAbort("assertion_failure", {"seed": seed})
    return model, before, inside_after


def _wilson(successes: int, trials: int) -> tuple[float | None, float | None, float | None]:
    if trials == 0:
        return None, None, None
    estimate = successes / trials
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(estimate * (1 - estimate) / trials + z * z / (4 * trials * trials)) / denominator
    return estimate, center - radius, center + radius


def _population_sha(indices: Sequence[int]) -> str:
    return canonical_json_sha256({"indices": list(indices)})


def elapsed_seconds_from_monotonic_ns(start_ns: int, stop_ns: int) -> float:
    if type(start_ns) is not int or type(stop_ns) is not int or start_ns < 0 or stop_ns < start_ns:
        raise ContractError("elapsed clock boundary differs")
    return (stop_ns - start_ns) / 1_000_000_000


def _gate_fields(operator: str | None, threshold: float | int | None, count: int | None, unit: str | None, numerator: int | float | None) -> tuple[str | None, float | int | None, int | None, str | None, bool | None]:
    if operator is None:
        return None, None, None, None, None
    if numerator is None:
        raise ContractError("gated metric lacks a value")
    if operator == ">=":
        passed = numerator >= count if count is not None else numerator >= threshold
    elif operator == "<=":
        passed = numerator <= count if count is not None else numerator <= threshold
    elif operator == "==":
        passed = numerator == count if count is not None else numerator == threshold
    else:
        raise ContractError("gate operator differs")
    return operator, threshold, count, unit, bool(passed)


def _evaluation_row(
    run_id: str,
    seed: int,
    condition: str,
    metric: str,
    stratum: str,
    indices: Sequence[int] | None,
    values: Mapping[str, Any],
    checkpoint_sha256: str | None,
    eval_data_sha256: str | None,
    provenance: Sequence[str],
    gate_id: str | None,
    gate: tuple[str | None, float | int | None, int | None, str | None],
    elapsed_seconds: float,
    resource_sample_ids: Sequence[int],
) -> dict[str, Any]:
    index_list = None if indices is None else list(indices)
    if metric in {"answer_accuracy", "original_source_hit_rate", "foreign_source_hit_rate"}:
        numerator_name = {
            "answer_accuracy": "answer_correct",
            "original_source_hit_rate": "original_source_hits",
            "foreign_source_hit_rate": "foreign_source_hits",
        }[metric]
        denominator_name = {
            "answer_accuracy": "answer_total",
            "original_source_hit_rate": "original_source_total",
            "foreign_source_hit_rate": "foreign_source_total",
        }[metric]
        numerator = int(values[numerator_name])
        denominator = int(values[denominator_name])
        estimate, low, high = _wilson(numerator, denominator)
    else:
        numerator = values.get("numerator")
        denominator = values.get("denominator")
        estimate = values.get("estimate")
        low = None
        high = None
    operator, threshold, threshold_count, threshold_unit, passed = _gate_fields(gate[0], gate[1], gate[2], gate[3], numerator if gate[3] != "absolute_error" else estimate)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rung": 1,
        "claim_seed": seed,
        "construction_seed": seed,
        "condition": condition,
        "metric": metric,
        "gate_id": gate_id,
        "stratum": stratum,
        "population_sha256": None if index_list is None else _population_sha(index_list),
        "checkpoint_sha256": checkpoint_sha256,
        "eval_data_sha256": eval_data_sha256,
        "provenance_sha256s": list(provenance),
        "numerator": numerator,
        "denominator": denominator,
        "estimate": estimate,
        "wilson95_low": low,
        "wilson95_high": high,
        "gate_operator": operator,
        "gate_threshold": threshold,
        "gate_threshold_count": threshold_count,
        "gate_threshold_unit": threshold_unit,
        "gate_pass": passed,
        "answer_correct": values.get("answer_correct") if metric == "answer_accuracy" else None,
        "answer_total": values.get("answer_total") if metric == "answer_accuracy" else None,
        "original_source_hits": values.get("original_source_hits") if metric == "original_source_hit_rate" else None,
        "original_source_total": values.get("original_source_total") if metric == "original_source_hit_rate" else None,
        "foreign_source_hits": values.get("foreign_source_hits") if metric == "foreign_source_hit_rate" else None,
        "foreign_source_total": values.get("foreign_source_total") if metric == "foreign_source_hit_rate" else None,
        "raw_remote_ids": values.get("raw_remote_ids"),
        "effective_remote_ids": values.get("effective_remote_ids"),
        "query_underfill_count": values.get("query_underfill_count"),
        "overflow_count": values.get("overflow_count") if metric == "route_overflow_count" else None,
        "max_bucket_load": values.get("max_bucket_load") if metric == "route_overflow_count" else None,
        "selected_mask_oracle_max_error": values.get("selected_mask_oracle_max_error") if metric == "selected_mask_oracle_max_error" else None,
        "elapsed_seconds": elapsed_seconds,
        "resource_sample_ids": list(resource_sample_ids),
    }


def _rung_one_gate(condition: str, metric: str, stratum: str, denominator: int) -> tuple[str | None, tuple[str | None, float | int | None, int | None, str | None]]:
    gates = {
        ("intact", "original_source_hit_rate", "all"): ("r1.intact.original_source_hit_rate.all", ">=", 0.90, 461, "rate"),
        ("intact", "answer_accuracy", "all"): ("r1.intact.answer_accuracy.all", ">=", 0.90, 461, "rate"),
        ("target_forced", "original_source_hit_rate", "all"): ("r1.target_forced.original_source_hit_rate.all", "==", 1.0, 512, "rate"),
        ("target_forced", "answer_accuracy", "all"): ("r1.target_forced.answer_accuracy.all", ">=", 0.95, 487, "rate"),
        ("recurrent_knockout", "original_source_hit_rate", "all"): ("r1.recurrent_knockout.original_source_hit_rate.all", "<=", 0.30, 153, "rate"),
        ("recurrent_knockout", "answer_accuracy", "all"): ("r1.recurrent_knockout.answer_accuracy.all", "<=", 0.30, 153, "rate"),
        ("carry_reset", "original_source_hit_rate", "all"): ("r1.carry_reset.original_source_hit_rate.all", "<=", 0.30, 153, "rate"),
        ("carry_reset", "answer_accuracy", "all"): ("r1.carry_reset.answer_accuracy.all", "<=", 0.30, 153, "rate"),
        ("carry_shuffle", "foreign_source_hit_rate", "all"): ("r1.carry_shuffle.foreign_source_hit_rate.all", ">=", 0.90, 461, "rate"),
        ("matched_random_route", "answer_accuracy", "all"): ("r1.matched_random_route.answer_accuracy.all", "<=", 0.30, 153, "rate"),
        ("block4_routed_knockout", "answer_accuracy", "all"): ("r1.block4_routed_knockout.answer_accuracy.all", "<=", 0.15, 76, "rate"),
        ("block4_local_only", "answer_accuracy", "all"): ("r1.block4_local_only.answer_accuracy.all", "<=", 0.15, 76, "rate"),
        ("required_source_excluded", "original_source_hit_rate", "all"): ("r1.required_source_excluded.original_source_hit_rate.all", "==", 0.0, 0, "rate"),
        ("required_source_excluded", "answer_accuracy", "all"): ("r1.required_source_excluded.answer_accuracy.all", "<=", 0.15, 76, "rate"),
        ("all_eligible_donor", "original_source_hit_rate", "all"): ("r1.all_eligible_donor.original_source_hit_rate.all", "==", 1.0, 512, "rate"),
        ("all_eligible_donor", "answer_accuracy", "all"): ("r1.all_eligible_donor.answer_accuracy.all", ">=", 0.95, 487, "rate"),
        ("all_eligible_clone", "original_source_hit_rate", "all"): ("r1.all_eligible_clone.original_source_hit_rate.all", "==", 1.0, 512, "rate"),
        ("all_eligible_clone", "answer_accuracy", "all"): ("r1.all_eligible_clone.answer_accuracy.all", ">=", 0.95, 487, "rate"),
        ("dense_causal", "answer_accuracy", "all"): ("r1.dense_causal.answer_accuracy.all", ">=", 0.95, 487, "rate"),
    }
    if condition == "carry_shuffle" and stratum == "changed_condition":
        if metric == "foreign_source_hit_rate":
            return "r1.carry_shuffle.foreign_source_hit_rate.changed_condition", (">=", 0.90, math.ceil(0.90 * denominator), "rate")
        if metric == "original_source_hit_rate":
            return "r1.carry_shuffle.original_source_hit_rate.changed_condition", ("<=", 0.30, math.floor(0.30 * denominator), "rate")
        if metric == "answer_accuracy":
            return "r1.carry_shuffle.answer_accuracy.changed_condition", ("<=", 0.30, math.floor(0.30 * denominator), "rate")
    row = gates.get((condition, metric, stratum))
    if row is None:
        return None, (None, None, None, None)
    return row[0], (row[1], row[2], row[3], row[4])


def _stat_accumulate(accumulator: dict[str, Any], values: Any, torch: Any) -> None:
    tensor = values.detach().to(torch.float64).reshape(-1)
    finite_mask = torch.isfinite(tensor)
    accumulator["count"] += int(tensor.numel())
    accumulator["nonfinite_count"] += int((~finite_mask).sum())
    if not bool(finite_mask.all()):
        raise HardAbort("nonfinite", {"surface": "state_telemetry"})
    if tensor.numel():
        accumulator["sum_parts"].append(float(tensor.sum(dtype=torch.float64)))
        accumulator["sum_sq_parts"].append(float(tensor.square().sum(dtype=torch.float64)))
        accumulator["min"] = min(accumulator["min"], float(tensor.min()))
        accumulator["max"] = max(accumulator["max"], float(tensor.max()))


def _new_stat_accumulator() -> dict[str, Any]:
    return {"count": 0, "nonfinite_count": 0, "sum_parts": [], "sum_sq_parts": [], "min": math.inf, "max": -math.inf}


def _stat_values(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    finite_count = int(accumulator["count"]) - int(accumulator["nonfinite_count"])
    if finite_count <= 0:
        return {"count": int(accumulator["count"]), "mean": None, "population_std": None, "min": None, "max": None, "nonfinite_count": int(accumulator["nonfinite_count"])}
    total = math.fsum(accumulator["sum_parts"])
    total_sq = math.fsum(accumulator["sum_sq_parts"])
    mean = total / finite_count
    variance = max(0.0, total_sq / finite_count - mean * mean)
    return {
        "count": int(accumulator["count"]),
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": accumulator["min"],
        "max": accumulator["max"],
        "nonfinite_count": int(accumulator["nonfinite_count"]),
    }


def _merge_stat_accumulators(accumulators: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _new_stat_accumulator()
    for accumulator in accumulators:
        merged["count"] += int(accumulator["count"])
        merged["nonfinite_count"] += int(accumulator["nonfinite_count"])
        merged["sum_parts"].extend(accumulator["sum_parts"])
        merged["sum_sq_parts"].extend(accumulator["sum_sq_parts"])
        merged["min"] = min(merged["min"], accumulator["min"])
        merged["max"] = max(merged["max"], accumulator["max"])
    return merged


def _new_l2_accumulator() -> list[list[float]]:
    return [[], [], []]


def _l2_accumulate(accumulator: list[list[float]], pre: Any, post: Any, exposed: Any, torch: Any) -> None:
    for parts, tensor in zip(accumulator, (pre, post, exposed)):
        value = tensor.detach().to(torch.float64)
        if not bool(torch.isfinite(value).all()):
            raise HardAbort("nonfinite", {"surface": "intervention_delta"})
        parts.append(float(value.square().sum(dtype=torch.float64)))


def _l2_values(accumulator: Sequence[Sequence[float]]) -> list[float]:
    return [math.sqrt(math.fsum(parts)) for parts in accumulator]


def _merge_l2_accumulators(accumulators: Sequence[Sequence[Sequence[float]]]) -> list[list[float]]:
    merged = _new_l2_accumulator()
    for accumulator in accumulators:
        for destination, source in zip(merged, accumulator):
            destination.extend(source)
    return merged


def _merged_summary_values(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(int(record["count"]) for record in records)
    nonfinite_count = sum(int(record["nonfinite_count"]) for record in records)
    finite_count = count - nonfinite_count
    if finite_count <= 0:
        return {"count": count, "mean": None, "population_std": None, "min": None, "max": None, "nonfinite_count": nonfinite_count}
    total = math.fsum(float(record["mean"]) * (int(record["count"]) - int(record["nonfinite_count"])) for record in records)
    second = math.fsum((float(record["population_std"]) ** 2 + float(record["mean"]) ** 2) * (int(record["count"]) - int(record["nonfinite_count"])) for record in records)
    mean = total / finite_count
    variance = max(0.0, second / finite_count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": min(float(record["min"]) for record in records),
        "max": max(float(record["max"]) for record in records),
        "nonfinite_count": nonfinite_count,
    }


def _state_identity_contract(rung: int, checkpoint_by_condition: Mapping[str, tuple[str, str]]) -> dict[tuple[Any, ...], int]:
    if rung == 1:
        conditions = RUNG_ONE_CONDITIONS
        chunk_positions = RUNG_ONE_CHUNK_END_POSITIONS
        gate_count = 512 * 4 * 128
        output_count = 512 * 128 * 64
        boundary_count = 512 * 4
    elif rung == 2:
        conditions = RUNG_TWO_CONDITIONS
        chunk_positions = RUNG_TWO_CHUNK_END_POSITIONS
        gate_count = 512 * 4 * 512
        output_count = 512 * 512 * 64
        boundary_count = 512 * 4
    else:
        raise ContractError("state statistic rung differs")
    if tuple(checkpoint_by_condition) != conditions:
        raise ContractError("state statistic condition registry differs")
    expected: dict[tuple[Any, ...], int] = {}
    for condition in conditions:
        model, checkpoint = checkpoint_by_condition[condition]
        if not isinstance(model, str) or not model or re.fullmatch(r"[0-9a-f]{64}", checkpoint or "") is None:
            raise ContractError("state statistic checkpoint registry differs")
        for block in RECURRENT_BLOCKS:
            for statistic, count in (("primary_gate", gate_count), ("beta_gate", gate_count), ("output_gate", output_count)):
                expected[(model, checkpoint, block, condition, "not_applicable", None, statistic)] = count
            for position in chunk_positions:
                expected[(model, checkpoint, block, condition, "global_chunk_end", position, "state_l2")] = boundary_count
            if rung == 1:
                for position in RUNG_ONE_RESET_POSITIONS:
                    expected[(model, checkpoint, block, condition, "pre_firewall_reset", position, "state_l2")] = boundary_count
                    expected[(model, checkpoint, block, condition, "post_firewall_reset", position, "state_l2")] = boundary_count
                if condition in {"carry_reset", "carry_shuffle"}:
                    expected[(model, checkpoint, block, condition, "pre_carry_intervention", 96, "state_l2")] = boundary_count
                    expected[(model, checkpoint, block, condition, "post_carry_intervention", 96, "state_l2")] = boundary_count
        expected[(model, checkpoint, None, condition, "not_applicable", None, "primary_gate")] = gate_count * len(RECURRENT_BLOCKS)
    return expected


def validate_state_records(records: Sequence[Mapping[str, Any]], rung: int | None = None, checkpoint_by_condition: Mapping[str, tuple[str, str]] | None = None) -> None:
    keys = ("model", "checkpoint_sha256", "block", "condition", "boundary", "position", "statistic", "count", "mean", "population_std", "min", "max", "nonfinite_count")
    if not records:
        raise ContractError("state statistic records are empty")
    observed_conditions = tuple(dict.fromkeys(record.get("condition") for record in records))
    if rung is None:
        condition_set = set(observed_conditions)
        if condition_set == set(RUNG_ONE_CONDITIONS):
            rung = 1
        elif condition_set == set(RUNG_TWO_CONDITIONS):
            rung = 2
        else:
            raise ContractError("state statistic rung cannot be inferred")
    conditions = RUNG_ONE_CONDITIONS if rung == 1 else RUNG_TWO_CONDITIONS if rung == 2 else ()
    if checkpoint_by_condition is None:
        inferred: dict[str, tuple[str, str]] = {}
        for condition in conditions:
            values = {(record.get("model"), record.get("checkpoint_sha256")) for record in records if record.get("condition") == condition}
            if len(values) != 1:
                raise ContractError("state statistic condition identity differs")
            inferred[condition] = next(iter(values))
        checkpoint_by_condition = inferred
    expected = _state_identity_contract(rung, checkpoint_by_condition)
    identities = []
    by_identity = {}
    allowed_boundaries = {"global_chunk_end", "pre_firewall_reset", "post_firewall_reset", "pre_carry_intervention", "post_carry_intervention", "not_applicable"}
    allowed_statistics = {"primary_gate", "beta_gate", "output_gate", "state_l2"}
    for record in records:
        validate_exact_keys(record, keys, "state statistic record")
        if record["boundary"] not in allowed_boundaries or record["statistic"] not in allowed_statistics:
            raise ContractError("state statistic enum differs")
        if not isinstance(record["model"], str) or not record["model"] or not isinstance(record["condition"], str) or not record["condition"] or re.fullmatch(r"[0-9a-f]{64}", record["checkpoint_sha256"] or "") is None:
            raise ContractError("state statistic identity differs")
        if record["block"] is None:
            if record["statistic"] != "primary_gate" or record["boundary"] != "not_applicable" or record["position"] is not None:
                raise ContractError("state statistic aggregate differs")
        elif record["block"] not in (1, 2, 3, 5, 6, 7):
            raise ContractError("state statistic block differs")
        if (record["boundary"] == "not_applicable") != (record["position"] is None):
            raise ContractError("state statistic position null contract differs")
        if isinstance(record["count"], bool) or not isinstance(record["count"], int) or record["count"] <= 0 or record["nonfinite_count"] != 0:
            raise ContractError("state statistic population differs")
        for field in ("mean", "population_std", "min", "max"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContractError("state statistic numeric value differs")
        if record["population_std"] < 0 or record["min"] > record["max"] or not record["min"] <= record["mean"] <= record["max"]:
            raise ContractError("state statistic numeric ordering differs")
        identity = tuple(record[field] for field in ("model", "checkpoint_sha256", "block", "condition", "boundary", "position", "statistic"))
        identities.append(identity)
        by_identity[identity] = record
    if len(identities) != len(set(identities)):
        raise ContractError("state statistic identity is duplicated")
    if set(identities) != set(expected) or len(identities) != len(expected):
        raise ContractError("state statistic identity closure differs")
    for identity, count in expected.items():
        if by_identity[identity]["count"] != count:
            raise ContractError("state statistic exact population differs")
    for condition in conditions:
        model, checkpoint = checkpoint_by_condition[condition]
        components = [by_identity[(model, checkpoint, block, condition, "not_applicable", None, "primary_gate")] for block in RECURRENT_BLOCKS]
        aggregate = by_identity[(model, checkpoint, None, condition, "not_applicable", None, "primary_gate")]
        merged = _merged_summary_values(components)
        for field in ("count", "nonfinite_count"):
            if aggregate[field] != merged[field]:
                raise ContractError("state statistic aggregate counter differs")
        for field in ("mean", "population_std", "min", "max"):
            if not math.isclose(float(aggregate[field]), float(merged[field]), rel_tol=1e-12, abs_tol=1e-12):
                raise ContractError("state statistic aggregate reduction differs")


def _intervention_identity_registry(
    rung: int,
    checkpoint_by_condition: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, str]]:
    if rung == 1:
        conditions = RUNG_ONE_CONDITIONS
        expected_models = RUNG_ONE_MODEL_BY_CONDITION
    elif rung == 2:
        conditions = RUNG_TWO_CONDITIONS
        expected_models = {condition: "rung_two" for condition in conditions}
    else:
        raise ContractError("intervention rung differs")
    if tuple(checkpoint_by_condition) != tuple(conditions):
        raise ContractError("intervention endpoint registry differs")
    endpoints = {}
    for condition in conditions:
        endpoint = checkpoint_by_condition[condition]
        if not isinstance(endpoint, tuple) or len(endpoint) != 2:
            raise ContractError("intervention endpoint identity differs")
        model, checkpoint = endpoint
        if model != expected_models[condition] or re.fullmatch(r"[0-9a-f]{64}", checkpoint or "") is None:
            raise ContractError("intervention endpoint identity differs")
        endpoints[condition] = (model, checkpoint)
    identities = {}
    for condition in conditions:
        model, checkpoint = endpoints[condition]
        baseline_condition = condition if rung == 1 and condition in {"all_eligible_donor", "dense_causal"} else "intact"
        baseline_model, baseline_checkpoint = endpoints[baseline_condition]
        identities[condition] = {
            "model": model,
            "checkpoint_sha256": checkpoint,
            "baseline_model": baseline_model,
            "baseline_checkpoint_sha256": baseline_checkpoint,
            "baseline_condition": baseline_condition,
        }
    return identities


def validate_intervention_records(
    records: Sequence[Mapping[str, Any]],
    rung: int,
    checkpoint_by_condition: Mapping[str, tuple[str, str]],
) -> None:
    keys = (
        "model",
        "checkpoint_sha256",
        "baseline_model",
        "baseline_checkpoint_sha256",
        "baseline_condition",
        "block",
        "condition",
        "pre_delta_l2",
        "post_delta_l2",
        "exposed_delta_l2",
    )
    identity_registry = _intervention_identity_registry(rung, checkpoint_by_condition)
    conditions = tuple(identity_registry)
    recurrent_conditions = {"intact", "recurrent_knockout", "carry_reset", "carry_shuffle"} if rung == 1 else set(RUNG_TWO_CONDITIONS)
    blocks = tuple(range(8)) if rung == 1 else RECURRENT_BLOCKS
    identities = []
    by_identity = {}
    for record in records:
        validate_exact_keys(record, keys, "intervention record")
        condition = record["condition"]
        if condition not in identity_registry:
            raise ContractError("intervention condition differs")
        if any(record[field] != value for field, value in identity_registry[condition].items()):
            raise ContractError("intervention endpoint or baseline identity differs")
        if record["block"] is not None and record["block"] not in blocks:
            raise ContractError("intervention block differs")
        for field in ("pre_delta_l2", "post_delta_l2", "exposed_delta_l2"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise ContractError("intervention norm differs")
        identity = (record["block"], condition)
        identities.append(identity)
        by_identity[identity] = record
    if len(identities) != len(set(identities)):
        raise ContractError("intervention identity is duplicated")
    expected_identities = {(block, condition) for condition in conditions for block in (None, *blocks)}
    if set(identities) != expected_identities or len(identities) != len(expected_identities):
        raise ContractError("intervention identity closure differs")
    for condition in conditions:
        aggregate = by_identity.get((None, condition))
        if aggregate is None:
            raise ContractError("intervention aggregate is absent")
        target_blocks = (1, 2, 3, 5, 6, 7) if condition in recurrent_conditions else (4,)
        targets = [by_identity[(block, condition)] for block in target_blocks]
        for field in ("pre_delta_l2", "post_delta_l2", "exposed_delta_l2"):
            expected = math.sqrt(math.fsum(float(record[field]) ** 2 for record in targets))
            if not math.isclose(float(aggregate[field]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ContractError("intervention aggregate reduction differs")
    for condition in conditions:
        baseline_condition = identity_registry[condition]["baseline_condition"]
        for block in blocks:
            current = by_identity[(block, condition)]
            expected_pre = by_identity[(block, baseline_condition)]["post_delta_l2"]
            if not math.isclose(float(current["pre_delta_l2"]), float(expected_pre), rel_tol=0.0, abs_tol=1e-12):
                raise ContractError("intervention registered baseline norm differs")
            if rung == 1 and condition in {"all_eligible_donor", "dense_causal"}:
                if not math.isclose(float(current["post_delta_l2"]), float(current["exposed_delta_l2"]), rel_tol=0.0, abs_tol=1e-12):
                    raise ContractError("intervention unchanged control differs")
                continue
            if block is None:
                continue
            knocked_out = condition == "recurrent_knockout" and block in RECURRENT_BLOCKS or rung == 1 and condition == "block4_routed_knockout" and block == 4
            if not knocked_out and not math.isclose(float(current["post_delta_l2"]), float(current["exposed_delta_l2"]), rel_tol=0.0, abs_tol=1e-12):
                raise ContractError("intervention non-knockout exposure differs")
        if rung == 1 and condition in {"all_eligible_donor", "dense_causal"}:
            aggregate = by_identity[(None, condition)]
            if not math.isclose(float(aggregate["pre_delta_l2"]), float(aggregate["post_delta_l2"]), rel_tol=0.0, abs_tol=1e-12) or not math.isclose(float(aggregate["post_delta_l2"]), float(aggregate["exposed_delta_l2"]), rel_tol=0.0, abs_tol=1e-12):
                raise ContractError("intervention unchanged aggregate differs")
    for block in RECURRENT_BLOCKS:
        record = by_identity[(block, "recurrent_knockout")]
        if record["post_delta_l2"] <= 0 or record["exposed_delta_l2"] != 0:
            raise ContractError("recurrent knockout delta semantics differ")
    if rung == 1:
        record = by_identity[(4, "block4_routed_knockout")]
        if record["post_delta_l2"] <= 0 or record["exposed_delta_l2"] != 0:
            raise ContractError("routed knockout delta semantics differ")


def _subset_values(result: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    index_list = list(indices)
    correct = result["correct"]
    original_hits = result["original_hits"]
    foreign_hits = result["foreign_hits"]
    return {
        "answer_correct": sum(bool(correct[index]) for index in index_list),
        "answer_total": len(index_list),
        "original_source_hits": None if original_hits is None else sum(bool(original_hits[index]) for index in index_list),
        "original_source_total": None if original_hits is None else len(index_list),
        "foreign_source_hits": None if foreign_hits is None else sum(bool(foreign_hits[index]) for index in index_list),
        "foreign_source_total": None if foreign_hits is None else len(index_list),
        "raw_remote_ids": None if result["raw_remote_ids"] is None else [result["raw_remote_ids"][index] for index in index_list],
        "effective_remote_ids": None if result["effective_remote_ids"] is None else [result["effective_remote_ids"][index] for index in index_list],
        "query_underfill_count": None,
        "overflow_count": result["overflow_count"],
        "max_bucket_load": result["max_bucket_load"],
        "selected_mask_oracle_max_error": None,
    }


def _ordinary_condition_rows(
    run_id: str,
    seed: int,
    condition: str,
    result: Mapping[str, Any],
    checkpoint_sha256: str,
    eval_data_sha256: str,
    resource_sample_ids: Sequence[int],
) -> list[dict[str, Any]]:
    all_indices = list(range(512))
    source_hit_indices = [index for index, hit in enumerate(result["original_hits"]) if hit]
    source_miss_indices = [index for index, hit in enumerate(result["original_hits"]) if not hit]
    provenance = [checkpoint_sha256, eval_data_sha256]
    rows = []
    for metric, stratum, indices in (
        ("answer_accuracy", "all", all_indices),
        ("original_source_hit_rate", "all", all_indices),
        ("answer_accuracy", "source_hit", source_hit_indices),
        ("answer_accuracy", "source_miss", source_miss_indices),
    ):
        values = _subset_values(result, indices)
        gate_id, gate = _rung_one_gate(condition, metric, stratum, len(indices))
        rows.append(_evaluation_row(run_id, seed, condition, metric, stratum, indices, values, checkpoint_sha256, eval_data_sha256, provenance, gate_id, gate, result["elapsed_seconds"], resource_sample_ids))
    underfill = sum(result["underfill"])
    rows.append(
        _evaluation_row(
            run_id,
            seed,
            condition,
            "query_underfill_count",
            "not_applicable",
            None,
            {
                "numerator": underfill,
                "denominator": None,
                "estimate": underfill,
                "query_underfill_count": underfill,
                "answer_correct": None,
                "answer_total": None,
                "original_source_hits": None,
                "original_source_total": None,
                "foreign_source_hits": None,
                "foreign_source_total": None,
                "raw_remote_ids": None,
                "effective_remote_ids": None,
                "overflow_count": None,
                "max_bucket_load": None,
                "selected_mask_oracle_max_error": None,
            },
            checkpoint_sha256,
            eval_data_sha256,
            provenance,
            None,
            (None, None, None, None),
            result["elapsed_seconds"],
            resource_sample_ids,
        )
    )
    if condition == "carry_shuffle":
        foreign_values = _subset_values(result, all_indices)
        gate_id, gate = _rung_one_gate(condition, "foreign_source_hit_rate", "all", 512)
        rows.append(_evaluation_row(run_id, seed, condition, "foreign_source_hit_rate", "all", all_indices, foreign_values, checkpoint_sha256, eval_data_sha256, provenance, gate_id, gate, result["elapsed_seconds"], resource_sample_ids))
        for metric in ("answer_accuracy", "original_source_hit_rate", "foreign_source_hit_rate"):
            for stratum, indices in (("changed_condition", result["changed_indices"]), ("same_condition", result["same_indices"])):
                values = _subset_values(result, indices)
                gate_id, gate = _rung_one_gate(condition, metric, stratum, len(indices))
                rows.append(_evaluation_row(run_id, seed, condition, metric, stratum, indices, values, checkpoint_sha256, eval_data_sha256, provenance, gate_id, gate, result["elapsed_seconds"], resource_sample_ids))
    return rows


def _evaluate_rung_one(
    connection: Any,
    worker: str,
    run_root: Path,
    seed: int,
    models: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
    oracle_error: float,
    oracle_provenance: Sequence[str],
    runtime: RuntimeModules,
    routing_stream: CanonicalGzipStream,
    forward_sequence: int,
    training_overflow: int,
    training_max_bucket_load: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, dict[str, dict[str, int]]]:
    torch = runtime.torch
    evaluation_path = run_root / "data" / f"r1_eval_{400000 + seed}.pt"
    evaluation_artifact = torch.load(evaluation_path, map_location="cpu", weights_only=False)
    evaluation = payload_to_tensors(evaluation_artifact["payload"], torch)
    eval_data_sha256 = sha256_file(evaluation_path)
    random_artifact = torch.load(run_root / "data" / f"r1_random_routes_{seed}.pt", map_location="cpu", weights_only=False)
    raw_query_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    route_acquisition_overflow = 0
    route_acquisition_maximum = 0
    runtime_usage: dict[str, dict[str, int]] = {}
    models["selected"].eval()
    with torch.inference_mode():
        for batch_index in range(16):
            start = batch_index * 32
            stop = start + 32
            _child_exchange(connection, {"kind": "status", "worker": worker, "seed": seed, "stage": "route_acquisition", "logical_update": batch_index})
            output = models["selected"](evaluation["tokens"][start:stop], return_aux=True, route_detail=True)
            context = {"worker": worker, "seed": seed, "stage": "route_acquisition", "logical_update": batch_index}
            _assert_finite_tree(torch, output, context, "route_acquisition_output")
            observed_overflow, observed_maximum, index_count, index_bytes, workspace_count, workspace_bytes = _route_observation(output, 1, context)
            usage = runtime_usage.setdefault("selected", {"route_index_storage_count": 0, "route_index_storage_bytes": 0, "routing_workspace_count": 0, "routing_workspace_bytes": 0})
            usage["route_index_storage_count"] = max(usage["route_index_storage_count"], index_count)
            usage["route_index_storage_bytes"] = max(usage["route_index_storage_bytes"], index_bytes)
            usage["routing_workspace_count"] = max(usage["routing_workspace_count"], workspace_count)
            usage["routing_workspace_bytes"] = max(usage["routing_workspace_bytes"], workspace_bytes)
            routed = _block_output(output, 4)
            raw_query_routes[start:stop] = _query_only_raw_routes(routed.telemetry["raw_remote"], torch)
            route_acquisition_overflow += observed_overflow
            route_acquisition_maximum = max(route_acquisition_maximum, observed_maximum)
            for row in _routing_rows(output, run_root.name, seed, "route_acquisition", "selected", None, None, None, forward_sequence, batch_index, start, evaluation["required_source"][start:stop], None, None, checkpoint_hashes["selected"]):
                routing_stream.write(row)
            forward_sequence += 1
    exclusion_payload = generate_source_exclusion_routes(510000 + seed, raw_query_routes, evaluation["required_source"], torch)
    exclusion_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    exclusion_routes[:, 126, 0] = torch.tensor(exclusion_payload["routes"], dtype=torch.long)
    if raw_query_routes.dtype != torch.long or raw_query_routes.shape != (512, 128, 1, 2) or not bool((raw_query_routes[:, torch.arange(128) != 126] == -1).all()):
        raise HardAbort("assertion_failure", {"worker": worker, "seed": seed, "stage": "post_checkpoint_learned_route"})
    exclusion_query = exclusion_routes[:, 126, 0]
    if exclusion_routes.dtype != torch.long or exclusion_routes.shape != (512, 128, 1, 2) or bool((exclusion_query == evaluation["required_source"][:, None]).any()) or bool((exclusion_query < 0).any()) or bool((exclusion_query > 14).any()) or bool((exclusion_query[:, 0] == exclusion_query[:, 1]).any()):
        raise HardAbort("assertion_failure", {"worker": worker, "seed": seed, "stage": "post_checkpoint_source_exclusion"})
    exclusion_record = {
        "seed": 510000 + seed,
        "raw_query_routes": raw_query_routes,
        "required_source": evaluation["required_source"],
        "routes": exclusion_routes,
        "payload": exclusion_payload,
        "payload_sha256": canonical_json_sha256(exclusion_payload),
    }
    _save_torch_artifact(run_root / "data" / f"r1_source_exclusion_{seed}.pt", exclusion_record, torch)
    conditions = RUNG_ONE_CONDITIONS
    checkpoint_by_condition = {
        condition: (
            RUNG_ONE_MODEL_BY_CONDITION[condition],
            checkpoint_hashes[RUNG_ONE_MODEL_BY_CONDITION[condition]],
        )
        for condition in conditions
    }
    intervention_identities = _intervention_identity_registry(1, checkpoint_by_condition)
    foreign_condition = _batch_local_foreign_conditions(evaluation["condition"], 32, torch)
    row_indices = torch.arange(512)
    foreign_source = evaluation["rule_blocks"][row_indices, foreign_condition]
    condition_results = {}
    predictions = []
    state_accumulators: dict[tuple[Any, ...], dict[str, Any]] = {}
    intervention_sums: dict[tuple[int, str], list[list[float]]] = {}
    baseline_delta_squares: dict[tuple[str, str, str, int, int], float] = {}
    evaluation_overflow = route_acquisition_overflow
    maximum_bucket_load = max(training_max_bucket_load, route_acquisition_maximum)
    for condition in conditions:
        model_key = RUNG_ONE_MODEL_BY_CONDITION[condition]
        model = models[model_key]
        model.eval()
        correct = []
        original_hits = None if condition == "dense_causal" else []
        condition_foreign_hits = [] if condition == "carry_shuffle" else None
        raw_ids = None if condition == "dense_causal" else []
        effective_ids = None if condition == "dense_causal" else []
        underfill = [] if condition != "dense_causal" else None
        condition_overflow = 0
        condition_maximum = 0
        elapsed_start = time.perf_counter_ns()
        with torch.inference_mode():
            for batch_index in range(16):
                start = batch_index * 32
                stop = start + 32
                _child_exchange(connection, {"kind": "status", "worker": worker, "seed": seed, "stage": condition, "logical_update": batch_index})
                kwargs: dict[str, Any] = {"return_aux": True, "recurrent_telemetry": True, "route_detail": True}
                if condition == "target_forced":
                    forced = torch.full((32, 128), -1, dtype=torch.long)
                    forced[:, 126] = evaluation["required_source"][start:stop]
                    kwargs["forced_blocks"] = forced
                elif condition == "recurrent_knockout":
                    kwargs["recurrent_knockout"] = True
                elif condition == "carry_reset":
                    kwargs["recurrent_intervention"] = "reset"
                elif condition == "carry_shuffle":
                    kwargs["recurrent_intervention"] = "shuffle"
                elif condition == "matched_random_route":
                    kwargs["route_override"] = random_artifact["routes"][start:stop]
                elif condition == "block4_routed_knockout":
                    kwargs["block4_routed_knockout"] = True
                elif condition == "required_source_excluded":
                    kwargs["route_override"] = exclusion_routes[start:stop]
                output = model(evaluation["tokens"][start:stop], **kwargs)
                context = {"worker": worker, "seed": seed, "stage": condition, "logical_update": batch_index}
                _assert_finite_tree(torch, output, context, "evaluation_output")
                observed_overflow, observed_maximum, index_count, index_bytes, workspace_count, workspace_bytes = _route_observation(output, 1, context)
                usage = runtime_usage.setdefault(model_key, {"route_index_storage_count": 0, "route_index_storage_bytes": 0, "routing_workspace_count": 0, "routing_workspace_bytes": 0})
                usage["route_index_storage_count"] = max(usage["route_index_storage_count"], index_count)
                usage["route_index_storage_bytes"] = max(usage["route_index_storage_bytes"], index_bytes)
                usage["routing_workspace_count"] = max(usage["routing_workspace_count"], workspace_count)
                usage["routing_workspace_bytes"] = max(usage["routing_workspace_bytes"], workspace_bytes)
                predicted = output.logits[:, 126].argmax(dim=-1)
                target = evaluation["targets"][start:stop]
                batch_correct = predicted.eq(target)
                correct.extend(bool(value) for value in batch_correct.tolist())
                routed = _block_output(output, 4)
                batch_original_hits = None
                batch_foreign_hits = None
                if condition != "dense_causal":
                    raw = routed.telemetry["raw_remote"][:, 126, 0]
                    effective = routed.telemetry["effective_remote"][:, 126, 0]
                    sources = evaluation["required_source"][start:stop]
                    batch_original_hits = (effective == sources[:, None]).any(dim=-1)
                    original_hits.extend(bool(value) for value in batch_original_hits.tolist())
                    raw_ids.extend([[int(value) for value in row] for row in raw.tolist()])
                    effective_ids.extend([[int(value) for value in row] for row in effective.tolist()])
                    underfill.extend(int((row == -1).sum()) for row in effective)
                    if condition == "carry_shuffle":
                        foreign = foreign_source[start:stop]
                        batch_foreign_hits = (effective == foreign[:, None]).any(dim=-1)
                        condition_foreign_hits.extend(bool(value) for value in batch_foreign_hits.tolist())
                condition_overflow += observed_overflow
                condition_maximum = max(condition_maximum, observed_maximum)
                evaluation_overflow += observed_overflow
                maximum_bucket_load = max(maximum_bucket_load, observed_maximum)
                for row in _routing_rows(output, run_root.name, seed, "evaluation", model_key, None, condition, None, forward_sequence, batch_index, start, evaluation["required_source"][start:stop], foreign_source[start:stop] if condition == "carry_shuffle" else None, condition, checkpoint_hashes[model_key]):
                    routing_stream.write(row)
                forward_sequence += 1
                for block_execution in output.blocks:
                    if block_execution.kind == "recurrent" and block_execution.mixer_output is not None:
                        recurrent = block_execution.mixer_output
                        for statistic, tensor in (("primary_gate", recurrent.primary_gate), ("beta_gate", recurrent.write_gate), ("output_gate", recurrent.output_gate)):
                            key = (model_key, checkpoint_hashes[model_key], block_execution.block_index, condition, "not_applicable", None, statistic)
                            accumulator = state_accumulators.setdefault(key, _new_stat_accumulator())
                            _stat_accumulate(accumulator, tensor, torch)
                        for boundary in recurrent.boundaries:
                            kind_map = {
                                "firewall_before_reset": "pre_firewall_reset",
                                "firewall_after_reset": "post_firewall_reset",
                                "carry_before_reset": "pre_carry_intervention",
                                "carry_after_reset": "post_carry_intervention",
                                "carry_before_shuffle": "pre_carry_intervention",
                                "carry_after_shuffle": "post_carry_intervention",
                                "chunk_end_after_clamp": "global_chunk_end",
                            }
                            if boundary.kind not in kind_map:
                                continue
                            key = (model_key, checkpoint_hashes[model_key], block_execution.block_index, condition, kind_map[boundary.kind], boundary.position, "state_l2")
                            accumulator = state_accumulators.setdefault(key, _new_stat_accumulator())
                            _stat_accumulate(accumulator, boundary.norms, torch)
                    computed = block_execution.computed_sequence_delta.detach().to(torch.float64)
                    exposed = block_execution.exposed_sequence_delta.detach().to(torch.float64)
                    post_square = float(computed.square().sum(dtype=torch.float64))
                    exposed_square = float(exposed.square().sum(dtype=torch.float64))
                    intervention_identity = intervention_identities[condition]
                    baseline_key = (
                        intervention_identity["baseline_model"],
                        intervention_identity["baseline_checkpoint_sha256"],
                        intervention_identity["baseline_condition"],
                        batch_index,
                        block_execution.block_index,
                    )
                    if condition == intervention_identity["baseline_condition"] and model_key == intervention_identity["baseline_model"] and checkpoint_hashes[model_key] == intervention_identity["baseline_checkpoint_sha256"]:
                        baseline_delta_squares[baseline_key] = post_square
                    if baseline_key not in baseline_delta_squares:
                        raise HardAbort("artifact_inconsistency", {**context, "surface": "intervention_baseline_cache"})
                    pre_square = baseline_delta_squares[baseline_key]
                    values = intervention_sums.setdefault((block_execution.block_index, condition), _new_l2_accumulator())
                    values[0].append(pre_square)
                    values[1].append(post_square)
                    values[2].append(exposed_square)
                for offset in range(32):
                    index = start + offset
                    original_hit = None if batch_original_hits is None else bool(batch_original_hits[offset])
                    foreign_hit = None if batch_foreign_hits is None else bool(batch_foreign_hits[offset])
                    predictions.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": run_root.name,
                            "rung": 1,
                            "claim_seed": seed,
                            "construction_seed": seed,
                            "condition": condition,
                            "example_index": index,
                            "original_condition": int(evaluation["condition"][index]),
                            "foreign_condition": int(foreign_condition[index]) if condition == "carry_shuffle" else None,
                            "original_source": int(evaluation["required_source"][index]),
                            "foreign_source": int(foreign_source[index]) if condition == "carry_shuffle" else None,
                            "target": int(evaluation["targets"][index]),
                            "prediction": int(predicted[offset]),
                            "correct": bool(batch_correct[offset]),
                            "original_source_hit": original_hit,
                            "foreign_source_hit": foreign_hit,
                            "condition_stratum": ("same_condition" if int(evaluation["condition"][index]) == int(foreign_condition[index]) else "changed_condition") if condition == "carry_shuffle" else "not_applicable",
                            "checkpoint_sha256": checkpoint_hashes[model_key],
                        }
                    )
        elapsed_stop = time.perf_counter_ns()
        condition_results[condition] = {
            "correct": correct,
            "original_hits": original_hits,
            "foreign_hits": condition_foreign_hits,
            "raw_remote_ids": raw_ids,
            "effective_remote_ids": effective_ids,
            "underfill": underfill,
            "overflow_count": condition_overflow,
            "max_bucket_load": condition_maximum,
            "elapsed_seconds": elapsed_seconds_from_monotonic_ns(elapsed_start, elapsed_stop),
            "changed_indices": [index for index in range(512) if int(evaluation["condition"][index]) != int(foreign_condition[index])],
            "same_indices": [index for index in range(512) if int(evaluation["condition"][index]) == int(foreign_condition[index])],
        }
    response = _child_exchange(connection, {"kind": "resource_refs", "worker": worker, "seed": seed, "stage": "evaluation", "logical_update": None})
    resource_sample_ids = response.get("sample_ids", [])
    if not isinstance(resource_sample_ids, list) or resource_sample_ids != sorted(set(resource_sample_ids)):
        raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed})
    evaluation_rows = []
    for condition in conditions:
        result = condition_results[condition]
        model_key = RUNG_ONE_MODEL_BY_CONDITION[condition]
        if condition == "dense_causal":
            values = _subset_values(result, list(range(512)))
            gate_id, gate = _rung_one_gate(condition, "answer_accuracy", "all", 512)
            evaluation_rows.append(_evaluation_row(run_root.name, seed, condition, "answer_accuracy", "all", list(range(512)), values, checkpoint_hashes[model_key], eval_data_sha256, [checkpoint_hashes[model_key], eval_data_sha256], gate_id, gate, result["elapsed_seconds"], resource_sample_ids))
        else:
            evaluation_rows.extend(_ordinary_condition_rows(run_root.name, seed, condition, result, checkpoint_hashes[model_key], eval_data_sha256, resource_sample_ids))
    routing_stream.close()
    routing_sha256 = sha256_file(routing_stream.path)
    evaluation_rows.append(
        _evaluation_row(
            run_root.name,
            seed,
            "intact",
            "selected_mask_oracle_max_error",
            "not_applicable",
            None,
            {"numerator": None, "denominator": None, "estimate": oracle_error, "answer_correct": None, "answer_total": None, "original_source_hits": None, "original_source_total": None, "foreign_source_hits": None, "foreign_source_total": None, "raw_remote_ids": None, "effective_remote_ids": None, "query_underfill_count": None, "overflow_count": None, "max_bucket_load": None, "selected_mask_oracle_max_error": oracle_error},
            None,
            None,
            oracle_provenance,
            "r1.intact.selected_mask_oracle_max_error.not_applicable",
            ("<=", 1e-5, None, "absolute_error"),
            0.0,
            resource_sample_ids,
        )
    )
    total_overflow = training_overflow + evaluation_overflow
    evaluation_rows.append(
        _evaluation_row(
            run_root.name,
            seed,
            "all_routed_training_and_evaluation",
            "route_overflow_count",
            "not_applicable",
            None,
            {"numerator": total_overflow, "denominator": None, "estimate": total_overflow, "answer_correct": None, "answer_total": None, "original_source_hits": None, "original_source_total": None, "foreign_source_hits": None, "foreign_source_total": None, "raw_remote_ids": None, "effective_remote_ids": None, "query_underfill_count": None, "overflow_count": total_overflow, "max_bucket_load": maximum_bucket_load, "selected_mask_oracle_max_error": None},
            None,
            None,
            [routing_sha256],
            "r1.all_routed_training_and_evaluation.route_overflow_count.not_applicable",
            ("==", 0, 0, "count"),
            0.0,
            resource_sample_ids,
        )
    )
    if len(evaluation_rows) != 65:
        raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed})
    primary_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for key, accumulator in state_accumulators.items():
        model_key, checkpoint, block, condition, boundary, position, statistic = key
        if block in (1, 2, 3, 5, 6, 7) and boundary == "not_applicable" and position is None and statistic == "primary_gate":
            primary_groups.setdefault((model_key, checkpoint, condition), []).append(accumulator)
    for (model_key, checkpoint, condition), accumulators in primary_groups.items():
        if len(accumulators) != 6:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed, "stage": condition, "surface": "primary_gate_aggregate"})
        state_accumulators[(model_key, checkpoint, None, condition, "not_applicable", None, "primary_gate")] = _merge_stat_accumulators(accumulators)
    state_records = []
    for key in sorted(state_accumulators, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        model_key, checkpoint, block, condition, boundary, position, statistic = key
        state_records.append({"model": model_key, "checkpoint_sha256": checkpoint, "block": block, "condition": condition, "boundary": boundary, "position": position, "statistic": statistic, **_stat_values(state_accumulators[key])})
    intervention_records = []
    for (block, condition), accumulator in sorted(intervention_sums.items(), key=lambda item: (item[0][1], item[0][0])):
        values = _l2_values(accumulator)
        intervention_records.append({**intervention_identities[condition], "block": block, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    for condition in conditions:
        target_blocks = (1, 2, 3, 5, 6, 7) if condition in {"intact", "recurrent_knockout", "carry_reset", "carry_shuffle"} else (4,)
        selected_accumulators = [intervention_sums[(block, condition)] for block in target_blocks]
        values = _l2_values(_merge_l2_accumulators(selected_accumulators))
        intervention_records.append({**intervention_identities[condition], "block": None, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    intervention_records.sort(key=lambda record: (record["condition"], -1 if record["block"] is None else record["block"]))
    return evaluation_rows, predictions, state_records, intervention_records, forward_sequence, runtime_usage


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    raw_run_root = parse_cli(sys.argv[1:] if argv is None else argv)
    validate_entry_environment(os.environ if environ is None else environ)
    validate_run_root(raw_run_root)
    load_prereg_payload()
    raise InitializationRefusal("direct CPU execution is disabled; use scripts/qualify_modular_mlx.py")


def _merge_runtime_accounting(records: Sequence[Mapping[str, int]]) -> dict[str, int]:
    keys = (
        "dynamic_recurrent_state_count",
        "dynamic_recurrent_state_bytes",
        "route_index_storage_count",
        "route_index_storage_bytes",
        "routing_workspace_count",
        "routing_workspace_bytes",
        "optimizer_state_count",
        "optimizer_state_bytes",
    )
    merged = {key: 0 for key in keys}
    for record in records:
        if tuple(sorted(record)) != tuple(sorted(keys)):
            raise ContractError("runtime accounting keys differ")
        for key in keys:
            value = record[key]
            if type(value) is not int or value < 0:
                raise ContractError("runtime accounting value differs")
            merged[key] = max(merged[key], value)
    return merged


def _evaluation_runtime_accounting(model: Any, usage: Mapping[str, int], batch_size: int) -> dict[str, int]:
    usage_keys = ("route_index_storage_count", "route_index_storage_bytes", "routing_workspace_count", "routing_workspace_bytes")
    if tuple(sorted(usage)) != tuple(sorted(usage_keys)) or type(batch_size) is not int or batch_size <= 0:
        raise ContractError("evaluation runtime accounting keys differ")
    for key in usage_keys:
        if type(usage[key]) is not int or usage[key] < 0:
            raise ContractError("evaluation runtime accounting value differs")
    dynamic_count = 6 * batch_size * model.config.heads * model.config.recurrent_head_width * model.config.recurrent_head_width
    return {
        "dynamic_recurrent_state_count": dynamic_count,
        "dynamic_recurrent_state_bytes": dynamic_count * 4,
        "route_index_storage_count": usage["route_index_storage_count"],
        "route_index_storage_bytes": usage["route_index_storage_bytes"],
        "routing_workspace_count": usage["routing_workspace_count"],
        "routing_workspace_bytes": usage["routing_workspace_bytes"],
        "optimizer_state_count": 0,
        "optimizer_state_bytes": 0,
    }


def _model_work_from_train_rows(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, int]:
    selected = [row for row in rows if row.get("model") == model_name]
    attempt_ids = [row.get("attempt_id") for row in selected]
    if not selected or len(attempt_ids) != len(set(attempt_ids)):
        raise ContractError("model training work identity differs")
    token_positions = 0
    for row in selected:
        for key in ("logical_update", "started_event_sequence", "completed_event_sequence", "token_positions"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ContractError("model training work value differs")
        if row["completed_event_sequence"] != row["started_event_sequence"] + 1 or row["token_positions"] <= 0:
            raise ContractError("model training work pair differs")
        token_positions += row["token_positions"]
    return {"attempted_updates": len(selected), "completed_updates": len(selected), "attempted_token_positions": token_positions}


def _accounting_entries(model: Any, audit_records: Sequence[Mapping[str, Any]], runtime_accounting: Mapping[str, int]) -> list[dict[str, Any]]:
    parameter_map = dict(model.named_parameters())
    classifications: dict[str, set[str]] = {name: set() for name in parameter_map}
    for record in audit_records:
        if record["name"] not in classifications:
            raise ContractError("accounting audit parameter is absent")
        classifications[record["name"]].add(record["classification"])
    learned_names = {name for name, values in classifications.items() if "learned_with_evidence" in values}
    serialized_names = {name for name, values in classifications.items() if name not in learned_names and values & {"trainable_but_no_gradient", "updated_only_by_decay"}}
    inactive_names = set(parameter_map) - learned_names - serialized_names
    entries = []
    categories = (
        ("active_learned_parameter", learned_names),
        ("serialized_without_gradient", serialized_names),
        ("inactive_parameter", inactive_names),
    )
    for category, names in categories:
        count = sum(parameter_map[name].numel() for name in names)
        size = sum(parameter_map[name].numel() * parameter_map[name].element_size() for name in names)
        entries.append({"category": category, "name": category, "count": count, "bytes": size})
    buffers = list(model.named_buffers())
    entries.append({"category": "registered_buffer", "name": "registered_buffer", "count": sum(buffer.numel() for _, buffer in buffers), "bytes": sum(buffer.numel() * buffer.element_size() for _, buffer in buffers)})
    for category in ("dynamic_recurrent_state", "route_index_storage", "routing_workspace", "optimizer_state"):
        entries.append({"category": category, "name": category, "count": int(runtime_accounting[f"{category}_count"]), "bytes": int(runtime_accounting[f"{category}_bytes"])})
    entries.sort(key=lambda record: record["name"])
    return entries


def validate_model_accounting(models: Sequence[Mapping[str, Any]], attempt_rows: Sequence[Mapping[str, Any]]) -> None:
    model_keys = ("model", "entries", "attempted_updates", "completed_updates", "attempted_token_positions", "resource_sample_ids")
    entry_keys = ("category", "name", "count", "bytes")
    categories = {"active_learned_parameter", "serialized_without_gradient", "inactive_parameter", "registered_buffer", "dynamic_recurrent_state", "route_index_storage", "routing_workspace", "optimizer_state"}
    if not models:
        raise ContractError("accounting model list is empty")
    model_names = []
    ledger_form = bool(attempt_rows) and all("event" in row for row in attempt_rows)
    summary_form = bool(attempt_rows) and all("event" not in row and "started_event_sequence" in row and "completed_event_sequence" in row for row in attempt_rows)
    if not ledger_form and not summary_form:
        raise ContractError("accounting work evidence form differs")
    for model in models:
        validate_exact_keys(model, model_keys, "accounting model")
        name = model["model"]
        if not isinstance(name, str) or not name:
            raise ContractError("accounting model identity differs")
        model_names.append(name)
        if ledger_form:
            started = [row for row in attempt_rows if row.get("model") == name and row.get("event") == "started"]
            completed = [row for row in attempt_rows if row.get("model") == name and row.get("event") == "completed"]
            attempted_updates = len(started)
            completed_updates = len(completed)
            attempted_token_positions = sum(row["token_positions"] for row in started)
        else:
            summaries = [row for row in attempt_rows if row.get("model") == name]
            work = _model_work_from_train_rows(summaries, name)
            attempted_updates = work["attempted_updates"]
            completed_updates = work["completed_updates"]
            attempted_token_positions = work["attempted_token_positions"]
        if model["attempted_updates"] != attempted_updates or model["completed_updates"] != completed_updates or model["attempted_token_positions"] != attempted_token_positions:
            raise ContractError("accounting work differs from attempt ledger")
        entries = model["entries"]
        if not isinstance(entries, list) or [entry.get("name") for entry in entries] != sorted(entry.get("name") for entry in entries):
            raise ContractError("accounting entry order differs")
        observed_categories = []
        for entry in entries:
            validate_exact_keys(entry, entry_keys, "accounting entry")
            if entry["category"] not in categories or entry["name"] != entry["category"] or type(entry["count"]) is not int or type(entry["bytes"]) is not int or entry["count"] < 0 or entry["bytes"] < 0:
                raise ContractError("accounting entry value differs")
            observed_categories.append(entry["category"])
        if set(observed_categories) != categories or len(observed_categories) != len(categories):
            raise ContractError("accounting category closure differs")
        ids = model["resource_sample_ids"]
        if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids) or ids != sorted(set(ids)):
            raise ContractError("accounting resource references differ")
    if len(model_names) != len(set(model_names)):
        raise ContractError("accounting model identity is duplicated")
    ledger_models = {row.get("model") for row in attempt_rows if summary_form or row.get("event") in {"started", "completed"}}
    if set(model_names) != ledger_models:
        raise ContractError("accounting model closure differs")


def _seed_parity_check(
    run_root: Path,
    seed: int,
    name: str,
    scope: str,
    max_error: float | None,
    tolerance: float | None,
    passed: bool,
    inputs: Any,
    outputs: Any,
    evidence_paths: Sequence[str],
) -> dict[str, Any]:
    detail = _check_detail(run_root, run_root.name, name, scope, inputs, outputs, evidence_paths)
    return {"name": name, "scope": scope, "max_error": max_error, "tolerance": tolerance, "pass": passed, "details_sha256": detail}


def _pretraining_assertion_lookup(run_root: Path) -> dict[str, Mapping[str, Any]]:
    preflight = json.loads((run_root / "run" / "preflight.json").read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    records = {}
    for array_name in ("source_checks", "result_checks", "transformerov_selfcheck", "routing_parity", "host_parity", "trained_backend", "lifecycle_assertions"):
        for record in preflight[array_name]:
            if record["name"] in PRETRAINING_ASSERTION_IDS:
                if record["name"] in records:
                    raise HardAbort("artifact_inconsistency", {"surface": "duplicate_pretraining_assertion"})
                records[record["name"]] = record
    if tuple(name for name in PRETRAINING_ASSERTION_IDS if name in records) != PRETRAINING_ASSERTION_IDS or len(records) != 15 or not all(record["pass"] is True for record in records.values()):
        raise HardAbort("assertion_failure", {"surface": "pretraining_assertion_package"})
    return records


def build_ordered_parity_checks(
    run_root: Path,
    seed: int,
    facts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if tuple(facts) != PARITY_SCOPES:
        raise HardAbort("artifact_inconsistency", {"seed": seed, "surface": "parity_scope_order"})
    checks = []
    for scope in PARITY_SCOPES:
        fact = facts[scope]
        validate_exact_keys(fact, ("name", "max_error", "tolerance", "pass", "inputs", "outputs", "evidence_paths"), "parity fact")
        if fact["pass"] is not True:
            raise HardAbort("assertion_failure", {"seed": seed, "stage": scope})
        checks.append(
            _seed_parity_check(
                run_root,
                seed,
                str(fact["name"]),
                scope,
                fact["max_error"],
                fact["tolerance"],
                True,
                fact["inputs"],
                fact["outputs"],
                fact["evidence_paths"],
            )
        )
    validate_parity_checks(checks)
    return checks


def validate_parity_checks(checks: Sequence[Mapping[str, Any]]) -> None:
    if len(checks) != len(PARITY_SCOPES) or tuple(record.get("scope") for record in checks) != PARITY_SCOPES:
        raise ContractError("parity check scope or cardinality differs")
    names = []
    for record in checks:
        validate_exact_keys(record, ("name", "scope", "max_error", "tolerance", "pass", "details_sha256"), "parity check")
        if not isinstance(record["name"], str) or not record["name"] or record["pass"] is not True or not isinstance(record["details_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["details_sha256"]) is None:
            raise ContractError("parity check identity differs")
        names.append(record["name"])
        if (record["max_error"] is None) != (record["tolerance"] is None):
            raise ContractError("parity numeric null contract differs")
        if record["max_error"] is not None:
            for field in ("max_error", "tolerance"):
                value = record[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                    raise ContractError("parity numeric evidence differs")
        if record["max_error"] is not None and record["max_error"] > record["tolerance"]:
            raise ContractError("parity numeric evidence does not pass")
    if len(names) != len(set(names)):
        raise ContractError("parity check name is duplicated")


def _parity_fact(name: str, passed: bool, inputs: Any, outputs: Any, evidence_paths: Sequence[str], max_error: float | None = None, tolerance: float | None = None) -> dict[str, Any]:
    return {"name": name, "max_error": max_error, "tolerance": tolerance, "pass": bool(passed), "inputs": inputs, "outputs": outputs, "evidence_paths": list(evidence_paths)}


def _claim_parity_facts(
    assertions: Mapping[str, Mapping[str, Any]],
    seed: int,
    rung: int,
    checksum_payload: Mapping[str, Any],
    route_payload: Mapping[str, Any],
    attention_error: float,
    attention_tolerance: float,
    initialization_payload: Mapping[str, Any],
    copy_payload: Mapping[str, Any],
    reload_records: Sequence[Mapping[str, Any]],
    intervention_payload: Mapping[str, Any],
    trained_backend_payload: Mapping[str, Any],
    evidence_paths: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    if tuple(name for name in PRETRAINING_ASSERTION_IDS if name in assertions) != PRETRAINING_ASSERTION_IDS:
        raise ContractError("claim parity assertion package differs")
    abi = assertions["mixer_abi_and_residual_ownership"]["actual"]
    architecture = assertions["exact_architecture"]["actual"]
    recurrent = assertions["reset_aware_recurrent_fidelity"]["actual"]
    firewall = assertions["firewall_factorization"]["actual"]
    causality = assertions["causality"]["actual"]
    raw_route = assertions["forced_and_random_route_exactness"]["actual"]
    source = assertions["source_host_route_and_attention_parity"]["actual"]
    lifecycle = assertions["state_and_index_lifetime"]["actual"]
    recurrent_error = max(float(recurrent["forward_max_error"]), float(recurrent["input_gradient_max_error"]), float(recurrent["parameter_gradient_max_error"]))
    host_error = max(float(source["host"]["recurrent_max_error"]), float(source["host"]["feature_max_error"]))
    internal_loss_error = float(source["route"]["internal_loss_max_error"])
    reload_error = max((max(float(record["logits_max_error"]), float(record["hidden_max_error"])) for record in reload_records), default=math.inf)
    reload_pass = bool(reload_records) and all(record["fresh_instance"] is True and record["state_exact"] is True and record["route_exact"] is True for record in reload_records) and reload_error <= 1e-7
    overflow = route_payload.get("overflow_count")
    maximum = route_payload.get("max_bucket_load")
    capacity = ROUTE_CAPACITY_BY_RUNG[rung]
    index_pass = type(overflow) is int and type(maximum) is int and overflow == 0 and 0 <= maximum <= capacity
    facts: dict[str, Mapping[str, Any]] = {}
    facts["source"] = _parity_fact(f"r{rung}_seed_{seed}_source", assertions["source_host_route_and_attention_parity"]["pass"] is True, {"assertion": "source_host_route_and_attention_parity"}, source, evidence_paths)
    facts["checksum"] = _parity_fact(f"r{rung}_seed_{seed}_checksum", checksum_payload.get("verified") is True, {"rung": rung, "seed": seed}, checksum_payload, evidence_paths)
    facts["ABI"] = _parity_fact(f"r{rung}_seed_{seed}_abi", assertions["mixer_abi_and_residual_ownership"]["pass"] is True, {"assertion": "mixer_abi_and_residual_ownership"}, abi, evidence_paths, float(abi["residual_max_error"]), 0.0)
    facts["host"] = _parity_fact(f"r{rung}_seed_{seed}_host", assertions["exact_architecture"]["pass"] is True and source["host"]["architecture_exact"] is True, {"assertion": "exact_architecture"}, {"architecture": architecture, "host": source["host"]}, evidence_paths, host_error, 0.0)
    facts["recurrent"] = _parity_fact(f"r{rung}_seed_{seed}_recurrent", assertions["reset_aware_recurrent_fidelity"]["pass"] is True, {"assertion": "reset_aware_recurrent_fidelity"}, recurrent, evidence_paths, recurrent_error, 1e-4)
    facts["reset"] = _parity_fact(f"r{rung}_seed_{seed}_reset", assertions["reset_aware_recurrent_fidelity"]["pass"] is True, {"rung": rung}, {"reset_before": recurrent["reset_before"], "reset_after": recurrent["reset_after"], "rung_two_chunk_ends": recurrent["rung_two_chunk_ends"]}, evidence_paths)
    facts["firewall"] = _parity_fact(f"r{rung}_seed_{seed}_firewall", assertions["firewall_factorization"]["pass"] is True, {"assertion": "firewall_factorization"}, firewall, evidence_paths)
    facts["causality"] = _parity_fact(f"r{rung}_seed_{seed}_causality", assertions["causality"]["pass"] is True, {"assertion": "causality"}, causality, evidence_paths, float(causality["max_prefix_error"]), 0.0)
    facts["raw_route"] = _parity_fact(f"r{rung}_seed_{seed}_raw_route", assertions["forced_and_random_route_exactness"]["pass"] is True and route_payload.get("postcheckpoint_assertions") is True, {"assertion": "forced_and_random_route_exactness"}, {"pretraining": raw_route, "postcheckpoint": dict(route_payload)}, evidence_paths)
    facts["index"] = _parity_fact(f"r{rung}_seed_{seed}_index", assertions["capacity_and_fallback"]["pass"] is True and index_pass, {"capacity": capacity}, dict(route_payload), evidence_paths, float(maximum) if type(maximum) is int else math.inf, float(capacity))
    facts["internal_loss"] = _parity_fact(f"r{rung}_seed_{seed}_internal_loss", assertions["source_host_route_and_attention_parity"]["pass"] is True and internal_loss_error == 0.0, {"assertion": "source_host_route_and_attention_parity"}, {"internal_loss_max_error": internal_loss_error}, evidence_paths, internal_loss_error, 0.0)
    facts["attention"] = _parity_fact(f"r{rung}_seed_{seed}_attention", math.isfinite(attention_error) and attention_error <= attention_tolerance, {"oracle": "selected_attention"}, {"max_error": attention_error}, evidence_paths, attention_error, attention_tolerance)
    facts["lifecycle"] = _parity_fact(f"r{rung}_seed_{seed}_lifecycle", assertions["state_and_index_lifetime"]["pass"] is True, {"assertion": "state_and_index_lifetime"}, lifecycle, evidence_paths)
    facts["initialization"] = _parity_fact(f"r{rung}_seed_{seed}_initialization", initialization_payload.get("pass") is True, {"seed": seed}, initialization_payload, evidence_paths)
    facts["copy"] = _parity_fact(f"r{rung}_seed_{seed}_copy", copy_payload.get("pass") is True, {"seed": seed}, copy_payload, evidence_paths)
    facts["reload"] = _parity_fact(f"r{rung}_seed_{seed}_reload", reload_pass, {"stage_count": len(reload_records)}, {"records": list(reload_records)}, evidence_paths, reload_error, 1e-7)
    facts["intervention"] = _parity_fact(f"r{rung}_seed_{seed}_intervention", intervention_payload.get("pass") is True, {"rung": rung}, intervention_payload, evidence_paths)
    facts["trained_backend"] = _parity_fact(f"r{rung}_seed_{seed}_trained_mlx_torch_endpoint_parity", trained_backend_payload.get("pass") is True, {"backend": "MLX", "reference": "Torch", "stages": trained_backend_payload.get("stages")}, {"records": trained_backend_payload.get("records")}, evidence_paths, float(trained_backend_payload.get("max_error", math.inf)), 1e-5)
    return facts


def _run_rung_one_seed(connection: Any, worker: str, run_root: Path, seed: int, runtime: RuntimeModules) -> int:
    torch = runtime.torch
    model_module = runtime.model_module
    seed_root = run_root / "rung1" / str(seed)
    checkpoint_root = ensure_directory(seed_root / "checkpoints")
    routing_stream = CanonicalGzipStream(seed_root / "routing.jsonl.gz")
    routing_stream.open()
    constructor_records = []
    selected_canonical, rng_before, rng_after = _construct_seeded_model("selected", seed, runtime)
    state_records, state_sha256 = _state_manifest(selected_canonical)
    state_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "construction_seed": seed,
        "role": "selected_canonical",
        "state_tensors": state_records,
        "state_sha256": state_sha256,
    }
    state_manifest_path = seed_root / "selected_canonical_state_manifest.json"
    write_canonical_json(state_manifest_path, state_manifest)
    state_manifest_digest = sha256_file(state_manifest_path)
    sentinel_path = run_root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    oracle_error = _selected_attention_oracle_for_model(selected_canonical, runtime, sentinel)
    oracle_detail = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "construction_seed": seed,
        "constructor_state_manifest_sha256": state_manifest_digest,
        "sentinel_payload_sha256": sha256_file(sentinel_path),
        "max_error": oracle_error,
        "tolerance": 1e-5,
        "pass": math.isfinite(oracle_error) and oracle_error <= 1e-5,
    }
    oracle_detail_path = seed_root / "selected_attention_oracle_detail.json"
    write_canonical_json(oracle_detail_path, oracle_detail)
    if not oracle_detail["pass"]:
        raise HardAbort("assertion_failure", {"worker": worker, "seed": seed})
    constructor_records.append({"role": "selected_canonical", "rng_before": rng_before, "rng_after": rng_after, "state_sha256": state_sha256})
    donor, before, after = _construct_seeded_model("all_eligible", seed, runtime)
    donor_copy = model_module.copy_compatible_state(selected_canonical, donor, include_router=True)
    constructor_records.append({"role": "all_eligible_donor", "rng_before": before, "rng_after": after, "compatible": list(donor_copy.compatible), "incompatible_source": list(donor_copy.incompatible_source), "incompatible_destination": list(donor_copy.incompatible_destination)})
    selected, before, after = _construct_seeded_model("selected", seed, runtime)
    selected_copy = model_module.copy_compatible_state(selected_canonical, selected, include_router=True)
    constructor_records.append({"role": "selected_destination", "rng_before": before, "rng_after": after, "compatible": list(selected_copy.compatible), "incompatible_source": list(selected_copy.incompatible_source), "incompatible_destination": list(selected_copy.incompatible_destination)})
    dense, before, after = _construct_seeded_model("dense", seed, runtime)
    dense_copy = model_module.copy_compatible_state(selected_canonical, dense, include_router=True)
    constructor_records.append({"role": "dense_destination", "rng_before": before, "rng_after": after, "compatible": list(dense_copy.compatible), "incompatible_source": list(dense_copy.incompatible_source), "incompatible_destination": list(dense_copy.incompatible_destination)})
    canonical_hashes = {name: _tensor_sha256(tensor) for name, tensor in selected_canonical.state_dict().items()}
    copy_pass = True
    for destination in (donor, selected, dense):
        destination_state = destination.state_dict()
        for name, expected in canonical_hashes.items():
            if name in destination_state and destination_state[name].shape == selected_canonical.state_dict()[name].shape and _tensor_sha256(destination_state[name]) != expected:
                copy_pass = False
    if not copy_pass:
        raise HardAbort("assertion_failure", {"worker": worker, "seed": seed})
    del selected_canonical
    event_sequence = 0
    forward_sequence = 0
    train_rows = []
    grad_records = []
    dense_grad_records = []
    training_overflow = 0
    max_bucket_load = 0
    reload_records = []
    runtime_accounting_records: dict[str, list[dict[str, int]]] = {"all_eligible_donor": [], "selected": [], "dense_causal": []}
    event_sequence, forward_sequence, rows, audit, donor_checkpoint_sha, overflow, maximum, reload_record, accounting_record = _train_stage(connection, worker, run_root, donor, "all_eligible_donor", "donor", 1, seed, 100000 + seed, 1024, 64, 16, event_sequence, runtime, checkpoint_root / "donor_last.pt", routing_stream, forward_sequence)
    reload_records.append({"stage": "donor", **reload_record})
    runtime_accounting_records["all_eligible_donor"].append(accounting_record)
    train_rows.extend(rows)
    grad_records.extend(audit)
    training_overflow += overflow
    max_bucket_load = max(max_bucket_load, maximum)
    transfer = model_module.copy_compatible_state(donor, selected, include_router=False)
    for name in transfer.compatible:
        if model_module.is_router_parameter(name) or _tensor_sha256(donor.state_dict()[name]) != _tensor_sha256(selected.state_dict()[name]):
            raise HardAbort("endpoint_inconsistency", {"worker": worker, "seed": seed, "stage": "donor_to_routed_copy"})
    event_sequence, forward_sequence, rows, audit, router_checkpoint_sha, overflow, maximum, reload_record, accounting_record = _train_stage(connection, worker, run_root, selected, "selected", "router_only", 1, seed, 200000 + seed, 768, 48, 16, event_sequence, runtime, checkpoint_root / "router_last.pt", routing_stream, forward_sequence)
    reload_records.append({"stage": "router_only", **reload_record})
    runtime_accounting_records["selected"].append(accounting_record)
    train_rows.extend(rows)
    grad_records.extend(audit)
    training_overflow += overflow
    max_bucket_load = max(max_bucket_load, maximum)
    event_sequence, forward_sequence, rows, audit, final_checkpoint_sha, overflow, maximum, reload_record, accounting_record = _train_stage(connection, worker, run_root, selected, "selected", "joint", 1, seed, 300000 + seed, 512, 32, 16, event_sequence, runtime, checkpoint_root / "final_last.pt", routing_stream, forward_sequence)
    reload_records.append({"stage": "joint", **reload_record})
    runtime_accounting_records["selected"].append(accounting_record)
    train_rows.extend(rows)
    grad_records.extend(audit)
    training_overflow += overflow
    max_bucket_load = max(max_bucket_load, maximum)
    event_sequence, forward_sequence, rows, audit, dense_base_sha, overflow, maximum, reload_record, accounting_record = _train_stage(connection, worker, run_root, dense, "dense_causal", "dense_base", 1, seed, 100000 + seed, 1024, 64, 16, event_sequence, runtime, checkpoint_root / "dense_base_last.pt", routing_stream, forward_sequence)
    reload_records.append({"stage": "dense_base", **reload_record})
    runtime_accounting_records["dense_causal"].append(accounting_record)
    train_rows.extend(rows)
    dense_grad_records.extend(audit)
    training_overflow += overflow
    max_bucket_load = max(max_bucket_load, maximum)
    event_sequence, forward_sequence, rows, audit, dense_checkpoint_sha, overflow, maximum, reload_record, accounting_record = _train_stage(connection, worker, run_root, dense, "dense_causal", "dense_continuation", 1, seed, 300000 + seed, 512, 32, 16, event_sequence, runtime, checkpoint_root / "dense_last.pt", routing_stream, forward_sequence)
    reload_records.append({"stage": "dense_continuation", **reload_record})
    runtime_accounting_records["dense_causal"].append(accounting_record)
    train_rows.extend(rows)
    dense_grad_records.extend(audit)
    training_overflow += overflow
    max_bucket_load = max(max_bucket_load, maximum)
    local, before, after = _construct_seeded_model("local_only", seed, runtime)
    final_checkpoint = torch.load(checkpoint_root / "final_last.pt", map_location="cpu", weights_only=False)
    local.load_state_dict(final_checkpoint["model_state_dict"], strict=True)
    constructor_records.append({"role": "block4_local_only_evaluation", "rng_before": before, "rng_after": after, "loaded_checkpoint_sha256": final_checkpoint_sha, "state_sha256": _state_manifest(local)[1]})
    clone, before, after = _construct_seeded_model("all_eligible", seed, runtime)
    clone.load_state_dict(final_checkpoint["model_state_dict"], strict=True)
    constructor_records.append({"role": "all_eligible_clone_evaluation", "rng_before": before, "rng_after": after, "loaded_checkpoint_sha256": final_checkpoint_sha, "state_sha256": _state_manifest(clone)[1]})
    models = {"selected": selected, "local": local, "donor": donor, "clone": clone, "dense": dense}
    checkpoint_hashes = {"selected": final_checkpoint_sha, "local": final_checkpoint_sha, "donor": donor_checkpoint_sha, "clone": final_checkpoint_sha, "dense": dense_checkpoint_sha}
    oracle_provenance = [state_manifest_digest, sha256_file(sentinel_path), sha256_file(oracle_detail_path)]
    evaluation_rows, predictions, state_stats, intervention_records, forward_sequence, evaluation_runtime_usage = _evaluate_rung_one(connection, worker, run_root, seed, models, checkpoint_hashes, oracle_error, oracle_provenance, runtime, routing_stream, forward_sequence, training_overflow, max_bucket_load)
    resource_response = _child_exchange(connection, {"kind": "resource_refs", "worker": worker, "seed": seed, "stage": "packaging", "logical_update": None})
    resource_ids = resource_response.get("sample_ids", [])
    if resource_ids != sorted(set(resource_ids)):
        raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed})
    _write_canonical_jsonl(seed_root / "train.jsonl", train_rows)
    validate_gradient_audit(grad_records, {"donor": 1024, "router_only": 768, "joint": 512})
    validate_gradient_audit(dense_grad_records, {"dense_base": 1024, "dense_continuation": 512})
    write_canonical_json(seed_root / "grad_audit.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": sorted(grad_records, key=lambda record: (record["stage"], record["name"]))})
    write_canonical_json(seed_root / "dense_grad_audit.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": sorted(dense_grad_records, key=lambda record: (record["stage"], record["name"]))})
    _write_canonical_jsonl(seed_root / "evaluation.jsonl", evaluation_rows)
    _write_canonical_gzip(seed_root / "predictions.jsonl.gz", predictions)
    checkpoint_by_condition = {condition: (RUNG_ONE_MODEL_BY_CONDITION[condition], checkpoint_hashes[RUNG_ONE_MODEL_BY_CONDITION[condition]]) for condition in RUNG_ONE_CONDITIONS}
    validate_state_records(state_stats, 1, checkpoint_by_condition)
    write_canonical_json(seed_root / "state_stats.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": state_stats})
    validate_intervention_records(intervention_records, 1, checkpoint_by_condition)
    write_canonical_json(seed_root / "intervention_deltas.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "records": intervention_records})
    checkpoint_pairs = (
        (checkpoint_root / "donor_last.pt", donor_checkpoint_sha),
        (checkpoint_root / "router_last.pt", router_checkpoint_sha),
        (checkpoint_root / "final_last.pt", final_checkpoint_sha),
        (checkpoint_root / "dense_base_last.pt", dense_base_sha),
        (checkpoint_root / "dense_last.pt", dense_checkpoint_sha),
    )
    checkpoint_payload = {"verified": all(sha256_file(path) == digest for path, digest in checkpoint_pairs), "sha256s": [digest for _, digest in checkpoint_pairs]}
    evaluation_overflow = max((int(row["overflow_count"] or 0) for row in evaluation_rows), default=0)
    evaluation_maximum = max((int(row["max_bucket_load"] or 0) for row in evaluation_rows), default=0)
    assertions = _pretraining_assertion_lookup(run_root)
    frozen_payload = load_prereg_payload()
    condition_checkpoints = {condition: (RUNG_ONE_MODEL_BY_CONDITION[condition], checkpoint_hashes[RUNG_ONE_MODEL_BY_CONDITION[condition]]) for condition in RUNG_ONE_CONDITIONS}
    evaluation_payload, measured_eval_sha = _load_evaluation_evidence(run_root, 1, seed, torch)
    prediction_evidence = _validate_prediction_artifact(run_root, frozen_payload, seed_root, 1, seed, condition_checkpoints, evaluation_payload, measured_eval_sha)
    measured_routing = _validate_routing_artifact(run_root, frozen_payload, seed_root, seed, evaluation_rows, prediction_evidence, condition_checkpoints)
    postcheckpoint_routes = _validate_rung_one_data_artifacts(run_root, seed, evaluation_payload, measured_routing, torch)
    parity_facts = _claim_parity_facts(
        assertions,
        seed,
        1,
        checkpoint_payload,
        {"overflow_count": training_overflow + evaluation_overflow, "max_bucket_load": max(max_bucket_load, evaluation_maximum), **postcheckpoint_routes},
        oracle_error,
        1e-5,
        {"pass": True, "constructor_roles": [record["role"] for record in constructor_records], "canonical_state_sha256": state_sha256},
        {"pass": copy_pass, "compatible_copy": True, "constructor_count": len(constructor_records)},
        reload_records,
        {"pass": True, "record_count": len(intervention_records), "matched_intact": True, "knockout_zero_exposed": True},
        {"pass": False, "stages": [], "records": [], "max_error": math.inf},
        ["run/preflight.json", f"rung1/{seed}/checkpoints/final_last.pt", f"rung1/{seed}/intervention_deltas.json"],
    )
    parity_checks = build_ordered_parity_checks(run_root, seed, parity_facts)
    write_canonical_json(seed_root / "parity.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "checkpoint_sha256": final_checkpoint_sha, "checks": parity_checks})
    all_audit = grad_records + dense_grad_records
    for usage_key, accounting_key, accounting_model in (
        ("selected", "selected", selected),
        ("local", "selected", local),
        ("donor", "all_eligible_donor", donor),
        ("clone", "all_eligible_donor", clone),
        ("dense", "dense_causal", dense),
    ):
        runtime_accounting_records[accounting_key].append(_evaluation_runtime_accounting(accounting_model, evaluation_runtime_usage[usage_key], 32))
    accounting_models = []
    for model_name, model, audits in (
        ("all_eligible_donor", donor, [record for record in all_audit if record["model"] == "all_eligible_donor"]),
        ("selected", selected, [record for record in all_audit if record["model"] == "selected"]),
        ("dense_causal", dense, [record for record in all_audit if record["model"] == "dense_causal"]),
    ):
        work = _model_work_from_train_rows(train_rows, model_name)
        accounting_models.append({"model": model_name, "entries": _accounting_entries(model, audits, _merge_runtime_accounting(runtime_accounting_records[model_name])), **work, "resource_sample_ids": resource_ids})
    validate_model_accounting(accounting_models, train_rows)
    write_canonical_json(seed_root / "accounting.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "models": accounting_models})
    write_canonical_json(seed_root / "resource_refs.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "sample_ids": resource_ids})
    for path in seed_root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in seed_root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(seed_root)
    return event_sequence - 1


def _rung_two_evaluation_row(
    run_id: str,
    condition: str,
    successes: int,
    checkpoint_sha256: str,
    eval_data_sha256: str,
    elapsed_seconds: float,
    resource_sample_ids: Sequence[int],
) -> dict[str, Any]:
    estimate, low, high = _wilson(successes, 512)
    if condition == "intact":
        gate_id = "r2.intact.answer_accuracy.all"
        operator = ">="
        threshold = 0.90
        count = 461
        passed = successes >= count
    else:
        gate_id = "r2.recurrent_knockout.answer_accuracy.all"
        operator = "<="
        threshold = 0.175
        count = 89
        passed = successes <= count
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rung": 2,
        "claim_seed": 83,
        "construction_seed": 83,
        "condition": condition,
        "metric": "answer_accuracy",
        "gate_id": gate_id,
        "stratum": "all",
        "population_sha256": _population_sha(range(512)),
        "checkpoint_sha256": checkpoint_sha256,
        "eval_data_sha256": eval_data_sha256,
        "provenance_sha256s": [checkpoint_sha256, eval_data_sha256],
        "numerator": successes,
        "denominator": 512,
        "estimate": estimate,
        "wilson95_low": low,
        "wilson95_high": high,
        "gate_operator": operator,
        "gate_threshold": threshold,
        "gate_threshold_count": count,
        "gate_threshold_unit": "rate",
        "gate_pass": passed,
        "answer_correct": successes,
        "answer_total": 512,
        "original_source_hits": None,
        "original_source_total": None,
        "foreign_source_hits": None,
        "foreign_source_total": None,
        "raw_remote_ids": None,
        "effective_remote_ids": None,
        "query_underfill_count": None,
        "overflow_count": None,
        "max_bucket_load": None,
        "selected_mask_oracle_max_error": None,
        "elapsed_seconds": elapsed_seconds,
        "resource_sample_ids": list(resource_sample_ids),
    }


def _rung_two_source_prediction(torch: Any, source_output: Any, telemetry_output: Any, target: Any, context: Mapping[str, Any]) -> tuple[Any, Any, float]:
    _assert_finite_tree(torch, source_output, context, "rung_two_source_output")
    _assert_finite_tree(torch, telemetry_output, context, "rung_two_telemetry_output")
    _assert_finite_tree(torch, target, context, "rung_two_target")
    source_logits = getattr(source_output, "logits", None)
    telemetry_logits = getattr(telemetry_output, "logits", None)
    if source_logits is None or telemetry_logits is None or source_logits.shape != telemetry_logits.shape or source_logits.ndim != 3 or source_logits.shape[1] <= 510 or target.shape != (source_logits.shape[0],):
        raise HardAbort("artifact_inconsistency", {**context, "surface": "rung_two_source_prediction_shape"})
    parity_error = float((source_logits - telemetry_logits).abs().max())
    if parity_error > 1e-7:
        raise HardAbort("endpoint_inconsistency", {**context, "surface": "rung_two_source_telemetry_parity"})
    predicted = source_logits[:, 510].argmax(dim=-1)
    return predicted, predicted.eq(target), parity_error


def _run_rung_two_seed(connection: Any, worker: str, run_root: Path, runtime: RuntimeModules) -> int:
    torch = runtime.torch
    seed = 83
    seed_root = run_root / "rung2" / "83"
    checkpoint_root = ensure_directory(seed_root / "checkpoints")
    model, rng_before, rng_after = _construct_seeded_model("rung_two", seed, runtime)
    canonical_records, canonical_sha = _state_manifest(model)
    event_sequence, _, train_rows, audit_records, checkpoint_sha, training_overflow, training_maximum, reload_record, runtime_accounting = _train_stage(connection, worker, run_root, model, "rung_two", "rung_two", 2, seed, 900083, 1536, 96, 8, 0, runtime, checkpoint_root / "final_last.pt", None, 0)
    evaluation_path = run_root / "data" / "r2_eval_1000083.pt"
    evaluation_artifact = torch.load(evaluation_path, map_location="cpu", weights_only=False)
    evaluation = payload_to_tensors(evaluation_artifact["payload"], torch)
    eval_sha = sha256_file(evaluation_path)
    checkpoint = torch.load(checkpoint_root / "final_last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    condition_results = []
    predictions = []
    gate_accumulators: dict[tuple[str, int, int], dict[str, Any]] = {}
    state_accumulators: dict[tuple[Any, ...], dict[str, Any]] = {}
    intervention_sums: dict[tuple[int, str], list[list[float]]] = {}
    checkpoint_by_condition = {condition: ("rung_two", checkpoint_sha) for condition in RUNG_TWO_CONDITIONS}
    intervention_identities = _intervention_identity_registry(2, checkpoint_by_condition)
    baseline_delta_squares: dict[tuple[str, str, str, int, int], float] = {}
    evaluation_index_bytes = 0
    evaluation_index_count = 0
    evaluation_workspace_count = 0
    evaluation_workspace_bytes = 0
    source_telemetry_max_error = 0.0
    telemetry_audit_forward_count = 0
    evaluation_overflow = 0
    evaluation_maximum = 0
    for condition in ("intact", "recurrent_knockout"):
        correct = 0
        started = time.perf_counter_ns()
        model.eval()
        with torch.inference_mode():
            for batch_index in range(16):
                start = batch_index * 32
                stop = start + 32
                _child_exchange(connection, {"kind": "status", "worker": worker, "seed": seed, "stage": condition, "logical_update": batch_index})
                batch_tokens = evaluation["tokens"][start:stop]
                source_output = model(batch_tokens, return_aux=True, route_detail=True, recurrent_knockout=condition == "recurrent_knockout")
                output = model(batch_tokens, return_aux=True, route_detail=True, recurrent_telemetry=True, recurrent_knockout=condition == "recurrent_knockout")
                telemetry_audit_forward_count += 1
                context = {"worker": worker, "seed": seed, "stage": condition, "logical_update": batch_index}
                target = evaluation["targets"][start:stop, 510]
                predicted, matches, parity_error = _rung_two_source_prediction(torch, source_output, output, target, context)
                source_telemetry_max_error = max(source_telemetry_max_error, parity_error)
                for observed in (source_output, output):
                    overflow, maximum, index_count, index_bytes, workspace_count, workspace_bytes = _route_observation(observed, 2, context)
                    evaluation_overflow += overflow
                    evaluation_maximum = max(evaluation_maximum, maximum)
                    evaluation_index_count = max(evaluation_index_count, index_count)
                    evaluation_index_bytes = max(evaluation_index_bytes, index_bytes)
                    evaluation_workspace_count = max(evaluation_workspace_count, workspace_count)
                    evaluation_workspace_bytes = max(evaluation_workspace_bytes, workspace_bytes)
                correct += int(matches.sum())
                for offset in range(32):
                    predictions.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": run_root.name,
                            "rung": 2,
                            "claim_seed": seed,
                            "construction_seed": seed,
                            "condition": condition,
                            "example_index": start + offset,
                            "original_condition": None,
                            "foreign_condition": None,
                            "original_source": None,
                            "foreign_source": None,
                            "target": int(target[offset]),
                            "prediction": int(predicted[offset]),
                            "correct": bool(matches[offset]),
                            "original_source_hit": None,
                            "foreign_source_hit": None,
                            "condition_stratum": "not_applicable",
                            "checkpoint_sha256": checkpoint_sha,
                        }
                    )
                for block_execution in output.blocks:
                    if block_execution.kind != "recurrent" or block_execution.mixer_output is None:
                        continue
                    recurrent = block_execution.mixer_output
                    primary = recurrent.primary_gate
                    for head in range(primary.size(1)):
                        accumulator = gate_accumulators.setdefault((condition, block_execution.block_index, head), _new_stat_accumulator())
                        _stat_accumulate(accumulator, primary[:, head], torch)
                    for statistic, tensor in (("primary_gate", primary), ("beta_gate", recurrent.write_gate), ("output_gate", recurrent.output_gate)):
                        key = ("rung_two", checkpoint_sha, block_execution.block_index, condition, "not_applicable", None, statistic)
                        accumulator = state_accumulators.setdefault(key, _new_stat_accumulator())
                        _stat_accumulate(accumulator, tensor, torch)
                    for boundary in recurrent.boundaries:
                        if boundary.kind != "chunk_end_after_clamp":
                            continue
                        key = ("rung_two", checkpoint_sha, block_execution.block_index, condition, "global_chunk_end", boundary.position, "state_l2")
                        accumulator = state_accumulators.setdefault(key, _new_stat_accumulator())
                        _stat_accumulate(accumulator, boundary.norms, torch)
                    computed = block_execution.computed_sequence_delta.detach().to(torch.float64)
                    exposed = block_execution.exposed_sequence_delta.detach().to(torch.float64)
                    post_square = float(computed.square().sum(dtype=torch.float64))
                    exposed_square = float(exposed.square().sum(dtype=torch.float64))
                    intervention_identity = intervention_identities[condition]
                    baseline_key = (
                        intervention_identity["baseline_model"],
                        intervention_identity["baseline_checkpoint_sha256"],
                        intervention_identity["baseline_condition"],
                        batch_index,
                        block_execution.block_index,
                    )
                    if condition == intervention_identity["baseline_condition"]:
                        baseline_delta_squares[baseline_key] = post_square
                    if baseline_key not in baseline_delta_squares:
                        raise HardAbort("artifact_inconsistency", {**context, "surface": "intervention_baseline_cache"})
                    intervention = intervention_sums.setdefault((block_execution.block_index, condition), _new_l2_accumulator())
                    intervention[0].append(baseline_delta_squares[baseline_key])
                    intervention[1].append(post_square)
                    intervention[2].append(exposed_square)
        stopped = time.perf_counter_ns()
        condition_results.append({"condition": condition, "successes": correct, "elapsed_seconds": elapsed_seconds_from_monotonic_ns(started, stopped)})
    response = _child_exchange(connection, {"kind": "resource_refs", "worker": worker, "seed": seed, "stage": "packaging", "logical_update": None})
    resource_ids = response.get("sample_ids", [])
    if resource_ids != sorted(set(resource_ids)):
        raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed})
    evaluation_rows = [_rung_two_evaluation_row(run_root.name, result["condition"], result["successes"], checkpoint_sha, eval_sha, result["elapsed_seconds"], resource_ids) for result in condition_results]
    gate_conditions = []
    for condition in ("intact", "recurrent_knockout"):
        gate_records = []
        aggregate_parts = []
        for block in (1, 2, 3, 5, 6, 7):
            for head in range(4):
                accumulator = gate_accumulators[(condition, block, head)]
                values = _stat_values(accumulator)
                gate_records.append({"block": block, "head": head, **values})
                aggregate_parts.append(accumulator)
        aggregate = _merge_stat_accumulators(aggregate_parts)
        aggregate_values = {"block": None, "head": None, **_stat_values(aggregate)}
        nonfinite = aggregate_values["nonfinite_count"]
        gate_conditions.append(
            {
                "condition": condition,
                "gate_id": f"r2.{condition}.primary_gate_nonfinite_count.not_applicable",
                "records": gate_records,
                "aggregate": aggregate_values,
                "gate_operator": "==",
                "gate_threshold": 0,
                "gate_threshold_count": 0,
                "gate_threshold_unit": "count",
                "gate_pass": nonfinite == 0,
            }
        )
    for condition in ("intact", "recurrent_knockout"):
        primary = [state_accumulators[("rung_two", checkpoint_sha, block, condition, "not_applicable", None, "primary_gate")] for block in (1, 2, 3, 5, 6, 7)]
        state_accumulators[("rung_two", checkpoint_sha, None, condition, "not_applicable", None, "primary_gate")] = _merge_stat_accumulators(primary)
    state_records = []
    for key in sorted(state_accumulators, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        model_name, checkpoint_identity, block, condition, boundary, position, statistic = key
        state_records.append({"model": model_name, "checkpoint_sha256": checkpoint_identity, "block": block, "condition": condition, "boundary": boundary, "position": position, "statistic": statistic, **_stat_values(state_accumulators[key])})
    intervention_records = []
    for (block, condition), accumulator in intervention_sums.items():
        values = _l2_values(accumulator)
        intervention_records.append({**intervention_identities[condition], "block": block, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    aggregate_interventions = []
    for condition in ("intact", "recurrent_knockout"):
        selected = [intervention_sums[(block, condition)] for block in (1, 2, 3, 5, 6, 7)]
        values = _l2_values(_merge_l2_accumulators(selected))
        aggregate_interventions.append({**intervention_identities[condition], "block": None, "condition": condition, "pre_delta_l2": values[0], "post_delta_l2": values[1], "exposed_delta_l2": values[2]})
    intervention_records.extend(aggregate_interventions)
    intervention_records.sort(key=lambda record: (record["condition"], -1 if record["block"] is None else record["block"]))
    _write_canonical_jsonl(seed_root / "train.jsonl", train_rows)
    validate_gradient_audit(audit_records, {"rung_two": 1536})
    write_canonical_json(seed_root / "grad_audit.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "records": sorted(audit_records, key=lambda record: record["name"])})
    _write_canonical_jsonl(seed_root / "evaluation.jsonl", evaluation_rows)
    _write_canonical_gzip(seed_root / "predictions.jsonl.gz", predictions)
    write_canonical_json(seed_root / "gate_stats.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "checkpoint_sha256": checkpoint_sha, "conditions": gate_conditions})
    validate_state_records(state_records, 2, checkpoint_by_condition)
    write_canonical_json(seed_root / "state_stats.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "records": state_records})
    validate_intervention_records(intervention_records, 2, checkpoint_by_condition)
    write_canonical_json(seed_root / "intervention_deltas.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "records": intervention_records})
    assertions = _pretraining_assertion_lookup(run_root)
    source_actual = assertions["source_host_route_and_attention_parity"]["actual"]
    checkpoint_payload = {"verified": sha256_file(checkpoint_root / "final_last.pt") == checkpoint_sha, "sha256s": [checkpoint_sha]}
    parity_facts = _claim_parity_facts(
        assertions,
        seed,
        2,
        checkpoint_payload,
        {
            "overflow_count": training_overflow + evaluation_overflow,
            "max_bucket_load": max(training_maximum, evaluation_maximum),
            "postcheckpoint_assertions": telemetry_audit_forward_count == 32 and source_telemetry_max_error <= 1e-7,
            "source_telemetry_max_error": source_telemetry_max_error,
            "telemetry_audit_forward_count": telemetry_audit_forward_count,
        },
        float(source_actual["oracle_error"]),
        1e-5,
        {"pass": True, "rng_isolated": True, "state_tensors": canonical_records, "state_sha256": canonical_sha},
        {"pass": True, "canonical_state_exact": True, "state_sha256": canonical_sha},
        [{"stage": "rung_two", **reload_record}],
        {"pass": True, "record_count": len(intervention_records), "matched_intact": True, "knockout_zero_exposed": True},
        {"pass": False, "stages": [], "records": [], "max_error": math.inf},
        ["run/preflight.json", "rung2/83/checkpoints/final_last.pt", "rung2/83/intervention_deltas.json"],
    )
    parity_checks = build_ordered_parity_checks(run_root, seed, parity_facts)
    write_canonical_json(seed_root / "parity.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "checkpoint_sha256": checkpoint_sha, "checks": parity_checks})
    evaluation_usage = {
        "route_index_storage_count": evaluation_index_count,
        "route_index_storage_bytes": evaluation_index_bytes,
        "routing_workspace_count": evaluation_workspace_count,
        "routing_workspace_bytes": evaluation_workspace_bytes,
    }
    merged_accounting = _merge_runtime_accounting((runtime_accounting, _evaluation_runtime_accounting(model, evaluation_usage, 32)))
    work = _model_work_from_train_rows(train_rows, "rung_two")
    accounting_model = {"model": "rung_two", "entries": _accounting_entries(model, audit_records, merged_accounting), **work, "resource_sample_ids": resource_ids}
    validate_model_accounting([accounting_model], train_rows)
    write_canonical_json(seed_root / "accounting.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "models": [accounting_model]})
    write_canonical_json(seed_root / "resource_refs.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": seed, "construction_seed": seed, "sample_ids": resource_ids})
    for path in seed_root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in seed_root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
        fsync_directory(directory)
    fsync_directory(seed_root)
    return event_sequence - 1


def _claim_worker(worker: str, run_root_text: str, start_event: Any, connection: Any) -> None:
    preserve_orphan_exit = False
    try:
        start_event.wait()
        validate_entry_environment()
        runtime = _import_runtime()
        configure_torch(runtime.torch)
        run_root = Path(run_root_text)
        seed_order = (11, 37, 71) if worker == "A" else (23, 53)
        last_sequences = {}
        for seed in seed_order:
            last_sequences[seed] = _run_rung_one_seed(connection, worker, run_root, seed, runtime)
        if worker == "B":
            last_sequences[83] = _run_rung_two_seed(connection, worker, run_root, runtime)
        _child_send_and_wait(connection, {"kind": "clean_complete", "handshake": clean_completion_handshake(worker, last_sequences)})
    except BaseException as exc:
        preserve_orphan_exit = isinstance(exc, UnrecoverableOrphan)
        _report_child_failure(connection, exc, worker)
        raise
    finally:
        try:
            connection.close()
        except BaseException:
            if not preserve_orphan_exit:
                raise


def _claim_accounting(writers: Mapping[str, CrashAtomicJsonlWriter]) -> AttemptAccounting:
    attempted_updates = 0
    completed_updates = 0
    attempted_tokens = 0
    completed_tokens = 0
    attempted_seeds: set[int] = set()
    completed_seeds: set[int] = set()
    last: dict[int, int] = {}
    unpaired = []
    for relative in CLAIM_LEDGER_PATHS[1:]:
        accounting = writers[relative].attempt_accounting()
        attempted_updates += accounting.attempted_updates
        completed_updates += accounting.completed_updates
        attempted_tokens += accounting.attempted_token_positions
        completed_tokens += accounting.completed_token_positions
        attempted_seeds.update(accounting.attempted_seeds)
        completed_seeds.update(accounting.completed_seeds)
        last.update(accounting.last_event_sequence_by_seed)
        unpaired.extend(accounting.unpaired_attempts)
    return AttemptAccounting(attempted_updates, completed_updates, attempted_tokens, completed_tokens, tuple(sorted(attempted_seeds)), tuple(sorted(completed_seeds)), dict(sorted(last.items())), tuple(unpaired))


def resource_sample_ids_for_seed(resource_rows: Sequence[Mapping[str, Any]], seed: int) -> list[int]:
    if type(seed) is not int or seed not in (*RUNG_ONE_SEEDS, RUNG_TWO_SEED):
        raise ContractError("resource reference seed differs")
    sample_ids = []
    for row in resource_rows:
        sample_id = row.get("sample_id")
        active_jobs = row.get("active_jobs")
        if type(sample_id) is not int or not isinstance(active_jobs, list):
            raise ContractError("resource reference source row differs")
        if any(isinstance(job, Mapping) and job.get("seed") == seed for job in active_jobs):
            sample_ids.append(sample_id)
    if sample_ids != sorted(set(sample_ids)):
        raise ContractError("resource reference selection differs")
    return sample_ids


def _finalize_seed_resource_references(run_root: Path, seed_sample_ids: Mapping[int, Sequence[int]]) -> None:
    for seed in (11, 23, 37, 53, 71):
        sample_ids = list(seed_sample_ids[seed])
        seed_root = run_root / "rung1" / str(seed)
        evaluation_path = seed_root / "evaluation.jsonl"
        evaluation_rows = [json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines()]
        for row in evaluation_rows:
            row["resource_sample_ids"] = sample_ids
        _replace_canonical_jsonl(evaluation_path, evaluation_rows)
        accounting_path = seed_root / "accounting.json"
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
        for model in accounting["models"]:
            model["resource_sample_ids"] = sample_ids
        _replace_canonical_json(accounting_path, accounting)
        _replace_canonical_json(seed_root / "resource_refs.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 1, "claim_seed": seed, "construction_seed": seed, "sample_ids": sample_ids})
    sample_ids = list(seed_sample_ids[83])
    seed_root = run_root / "rung2" / "83"
    evaluation_path = seed_root / "evaluation.jsonl"
    evaluation_rows = [json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines()]
    for row in evaluation_rows:
        row["resource_sample_ids"] = sample_ids
    _replace_canonical_jsonl(evaluation_path, evaluation_rows)
    accounting_path = seed_root / "accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    for model in accounting["models"]:
        model["resource_sample_ids"] = sample_ids
    _replace_canonical_json(accounting_path, accounting)
    _replace_canonical_json(seed_root / "resource_refs.json", {"schema_version": SCHEMA_VERSION, "run_id": run_root.name, "rung": 2, "claim_seed": 83, "construction_seed": 83, "sample_ids": sample_ids})


def validate_parent_ledger_accounting(run_root: Path) -> None:
    resource_path = run_root / "run" / "resources.jsonl"
    resource_rows = validate_canonical_jsonl_prefix(resource_path, resource_path.stat().st_size, validate_resource_row)
    validate_resource_timeline(resource_rows, "claim", require_clean_final=True)
    resources = {row["sample_id"]: row for row in resource_rows}
    for seed in (*RUNG_ONE_SEEDS, RUNG_TWO_SEED):
        seed_root = run_root / (f"rung1/{seed}" if seed != RUNG_TWO_SEED else "rung2/83")
        ledger_rows = _read_jsonl(seed_root / "attempts.jsonl")
        validate_attempt_sequence(ledger_rows, require_complete=True)
        accounting_path = seed_root / "accounting.json"
        accounting = _canonical_json_artifact(accounting_path)
        validate_exact_keys(accounting, ("schema_version", "run_id", "rung", "claim_seed", "construction_seed", "models"), "accounting artifact")
        expected_rung = 1 if seed != RUNG_TWO_SEED else 2
        if accounting["schema_version"] != SCHEMA_VERSION or accounting["run_id"] != run_root.name or accounting["rung"] != expected_rung or accounting["claim_seed"] != seed or accounting["construction_seed"] != seed:
            raise ContractError("accounting artifact identity differs")
        validate_model_accounting(accounting["models"], ledger_rows)
        expected_ids = resource_sample_ids_for_seed(resource_rows, seed)
        for model in accounting["models"]:
            if model["resource_sample_ids"] != expected_ids:
                raise ContractError("accounting resource reference selection differs")
            for sample_id in model["resource_sample_ids"]:
                if sample_id not in resources or not any(job.get("seed") == seed for job in resources[sample_id]["active_jobs"]):
                    raise ContractError("accounting resource reference does not name active seed")


def capture_frozen_manifest_anchors(run_root: Path) -> FrozenManifestAnchors:
    config_path = validate_real_regular_file(run_root / "run" / "config_manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    review_records = config.get("review_records")
    if not isinstance(review_records, list) or len(review_records) != 4:
        raise ContractError("manifest anchor review registry differs")
    config_records = config.get("records")
    if not isinstance(config_records, list):
        raise ContractError("manifest anchor configuration registry differs")
    plan_records = [record for record in config_records if isinstance(record, Mapping) and record.get("path") == PROJECT_PLAN_RELATIVE_PATH]
    launch_path = validate_real_regular_file(run_root / LAUNCH_PROJECT_PLAN_PATH)
    if len(plan_records) != 1 or plan_records[0].get("sha256") != sha256_file(launch_path):
        raise ContractError("launch project plan anchor differs")
    relative_paths = [
        "run/prereg.json",
        "run/source_manifest.json",
        "run/config_manifest.json",
        "run/environment.json",
        "run/preflight.json",
        LAUNCH_PROJECT_PLAN_PATH,
        "run/sentinels/selected_attention_oracle_payload.json",
        *(f"run/reviews/{record['artifact_sha256']}.json" for record in review_records),
    ]
    if len(relative_paths) != 11 or len(relative_paths) != len(set(relative_paths)):
        raise ContractError("manifest anchor path closure differs")
    records = []
    for relative in sorted(relative_paths):
        path = validate_real_regular_file(run_root / relative)
        records.append((relative, sha256_file(path)))
    return FrozenManifestAnchors(tuple(records))


def _base_manifest_anchor_paths(run_root: Path) -> set[str]:
    config = _canonical_json_artifact(validate_real_regular_file(run_root / "run" / "config_manifest.json"))
    review_records = config.get("review_records") if isinstance(config, Mapping) else None
    if not isinstance(review_records, list) or len(review_records) != 4:
        raise ContractError("manifest anchor review registry differs")
    reviews = set()
    for record in review_records:
        digest = record.get("artifact_sha256") if isinstance(record, Mapping) else None
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ContractError("manifest anchor review identity differs")
        reviews.add(f"run/reviews/{digest}.json")
    if len(reviews) != 4:
        raise ContractError("manifest anchor review identity differs")
    return {
        "run/prereg.json",
        "run/source_manifest.json",
        "run/config_manifest.json",
        "run/environment.json",
        "run/preflight.json",
        LAUNCH_PROJECT_PLAN_PATH,
        "run/sentinels/selected_attention_oracle_payload.json",
        *reviews,
    }


def _manifest_anchor_paths_for_state(run_root: Path, count: int) -> set[str]:
    expected = _base_manifest_anchor_paths(run_root)
    if count == 11:
        return expected
    expected.add(TRAINING_START_REQUEST_PATH)
    if count == 12:
        return expected
    if count != 15:
        raise ContractError("manifest anchor cardinality differs")
    link = _canonical_json_artifact(validate_real_regular_file(run_root / TRAINING_START_LINK_PATH))
    review_path = link.get("review_path") if isinstance(link, Mapping) else None
    if not isinstance(review_path, str) or re.fullmatch(r"run/reviews/[0-9a-f]{64}\.json", review_path) is None:
        raise ContractError("training-start review anchor path differs")
    expected.update((TRAINING_START_PROJECT_PLAN_PATH, TRAINING_START_LINK_PATH, review_path))
    return expected


def verify_manifest_anchors(run_root: Path, anchors: FrozenManifestAnchors) -> None:
    if not isinstance(anchors, FrozenManifestAnchors) or len(anchors.records) not in {11, 12, 15} or tuple(sorted(anchors.records)) != anchors.records or len({relative for relative, _ in anchors.records}) != len(anchors.records):
        raise HardAbort("frozen_hash_change", {"surface": "manifest_anchors"})
    for relative, expected in anchors.records:
        if not isinstance(relative, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise HardAbort("frozen_hash_change", {"surface": "manifest_anchor_record"})
        try:
            path = validate_real_regular_file(run_root / relative)
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": relative}) from exc
        if sha256_file(path) != expected:
            raise HardAbort("frozen_hash_change", {"surface": relative})
    try:
        expected_paths = _manifest_anchor_paths_for_state(run_root, len(anchors.records))
    except ContractError as exc:
        raise HardAbort("frozen_hash_change", {"surface": "manifest_anchors"}) from exc
    if {relative for relative, _ in anchors.records} != expected_paths:
        raise HardAbort("frozen_hash_change", {"surface": "manifest_anchors"})


def extend_frozen_manifest_anchors(run_root: Path, anchors: FrozenManifestAnchors, relative_paths: Sequence[str]) -> FrozenManifestAnchors:
    verify_manifest_anchors(run_root, anchors)
    additions = tuple(sorted(relative_paths))
    if len(anchors.records) == 11:
        expected_additions = (TRAINING_START_REQUEST_PATH,)
    elif len(anchors.records) == 12:
        link = _canonical_json_artifact(validate_real_regular_file(run_root / TRAINING_START_LINK_PATH))
        review_path = link.get("review_path") if isinstance(link, Mapping) else None
        if not isinstance(review_path, str):
            raise ContractError("training-start review anchor path differs")
        expected_additions = tuple(sorted((TRAINING_START_PROJECT_PLAN_PATH, TRAINING_START_LINK_PATH, review_path)))
    else:
        raise ContractError("manifest anchors cannot be extended")
    if additions != expected_additions or len(additions) != len(set(additions)):
        raise ContractError("manifest anchor extension differs")
    records = dict(anchors.records)
    for relative in additions:
        path = validate_real_regular_file(run_root / relative)
        records[relative] = sha256_file(path)
    extended = FrozenManifestAnchors(tuple(sorted(records.items())))
    verify_manifest_anchors(run_root, extended)
    return extended


def _training_start_scope(run_id: str) -> str:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ContractError("training-start run identity differs")
    return f"{TRAINING_START_REVIEW_SCOPE_PREFIX}{run_id}"


def _training_start_plan_binding(run_id: str, request_sha256: str) -> bytes:
    if re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise ContractError("training-start request digest differs")
    return f"Training start request `{run_id}` binds request SHA-256 `{request_sha256}`; these reviewed bytes become canonical only at the atomic training-start commit.\n".encode("utf-8")


def _training_start_target_records(plan_sha256: str, request_sha256: str) -> list[dict[str, str]]:
    if re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise ContractError("training-start target digest differs")
    return [
        {"path": PROJECT_PLAN_RELATIVE_PATH, "sha256": plan_sha256},
        {"path": TRAINING_START_REQUEST_PATH, "sha256": request_sha256},
    ]


def _validate_training_start_request_record(run_root: Path) -> Mapping[str, Any]:
    record = _canonical_json_artifact(validate_real_regular_file(run_root / TRAINING_START_REQUEST_PATH))
    validate_exact_keys(
        record,
        (
            "schema_version",
            "run_id",
            "boundary",
            "review_request_monotonic_ns",
            "review_request_wall_utc",
            "launch_project_plan_sha256",
            "required_review_scope",
            "review_wait_timeout_seconds",
        ),
        "training-start request artifact",
    )
    if (
        record["schema_version"] != SCHEMA_VERSION
        or record["run_id"] != run_root.name
        or record["boundary"] != "training_start_review_request"
        or type(record["review_request_monotonic_ns"]) is not int
        or record["review_request_monotonic_ns"] < 0
        or not isinstance(record["review_request_wall_utc"], str)
        or not record["review_request_wall_utc"]
        or not isinstance(record["launch_project_plan_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["launch_project_plan_sha256"]) is None
        or record["required_review_scope"] != _training_start_scope(run_root.name)
        or type(record["review_wait_timeout_seconds"]) is not int
        or record["review_wait_timeout_seconds"] != TRAINING_START_REVIEW_WAIT_SECONDS
    ):
        raise ContractError("training-start request artifact differs")
    launch = validate_real_regular_file(run_root / LAUNCH_PROJECT_PLAN_PATH)
    if sha256_file(launch) != record["launch_project_plan_sha256"]:
        raise ContractError("training-start request launch plan differs")
    return record


def _validate_training_start_linkage(run_root: Path) -> Mapping[str, Any]:
    request = _validate_training_start_request_record(run_root)
    link = _canonical_json_artifact(validate_real_regular_file(run_root / TRAINING_START_LINK_PATH))
    validate_exact_keys(
        link,
        (
            "schema_version",
            "run_id",
            "boundary",
            "request_path",
            "request_artifact_sha256",
            "launch_snapshot_path",
            "training_start_snapshot_path",
            "review_path",
            "launch_project_plan_sha256",
            "training_start_project_plan_sha256",
            "review_scope",
            "review_target_sha256",
            "review_artifact_sha256",
            "review_deadline_monotonic_ns",
            "commit_admission_monotonic_ns",
            "commit_admission_wall_utc",
            "start_commit_margin_seconds",
            "start_commit_rule",
        ),
        "training-start plan linkage",
    )
    review_digest = link["review_artifact_sha256"]
    review_path = link["review_path"]
    if (
        link["schema_version"] != SCHEMA_VERSION
        or link["run_id"] != run_root.name
        or link["boundary"] != "reviewed_plan_atomic_training_start"
        or link["request_path"] != TRAINING_START_REQUEST_PATH
        or link["request_artifact_sha256"] != sha256_file(run_root / TRAINING_START_REQUEST_PATH)
        or link["launch_snapshot_path"] != LAUNCH_PROJECT_PLAN_PATH
        or link["training_start_snapshot_path"] != TRAINING_START_PROJECT_PLAN_PATH
        or not isinstance(review_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", review_digest) is None
        or review_path != f"run/reviews/{review_digest}.json"
        or link["launch_project_plan_sha256"] != request["launch_project_plan_sha256"]
        or link["review_scope"] != _training_start_scope(run_root.name)
        or type(link["review_deadline_monotonic_ns"]) is not int
        or link["review_deadline_monotonic_ns"] != request["review_request_monotonic_ns"] + TRAINING_START_REVIEW_WAIT_NS
        or type(link["commit_admission_monotonic_ns"]) is not int
        or link["commit_admission_monotonic_ns"] < request["review_request_monotonic_ns"]
        or link["commit_admission_monotonic_ns"] > link["review_deadline_monotonic_ns"] - TRAINING_START_COMMIT_MARGIN_NS
        or not isinstance(link["commit_admission_wall_utc"], str)
        or not link["commit_admission_wall_utc"]
        or type(link["start_commit_margin_seconds"]) is not int
        or link["start_commit_margin_seconds"] != TRAINING_START_COMMIT_MARGIN_SECONDS
        or link["start_commit_rule"] != "single_coordinator_atomic_replace_after_locked_exact_launch_recheck"
    ):
        raise ContractError("training-start plan linkage differs")
    launch_digest = sha256_file(validate_real_regular_file(run_root / LAUNCH_PROJECT_PLAN_PATH))
    training_digest = sha256_file(validate_real_regular_file(run_root / TRAINING_START_PROJECT_PLAN_PATH))
    if link["launch_project_plan_sha256"] != launch_digest or link["training_start_project_plan_sha256"] != training_digest or training_digest == launch_digest:
        raise ContractError("training-start plan digest differs")
    review = _canonical_json_artifact(validate_real_regular_file(run_root / review_path))
    request_digest = sha256_file(run_root / TRAINING_START_REQUEST_PATH)
    target_records = _training_start_target_records(training_digest, request_digest)
    validate_exact_keys(review, ("schema_version", "reviewer", "scope", "target_records", "target_sha256", "findings", "finding_count"), "training-start review attestation")
    target_digest = canonical_json_sha256(target_records)
    if (
        review["schema_version"] != "todorov.review-attestation.1"
        or review["reviewer"] != "feature-dev:code-reviewer"
        or review["scope"] != _training_start_scope(run_root.name)
        or review["target_records"] != target_records
        or review["target_sha256"] != target_digest
        or review["findings"] != []
        or type(review["finding_count"]) is not int
        or review["finding_count"] != 0
        or sha256_file(run_root / review_path) != review_digest
        or link["review_target_sha256"] != target_digest
    ):
        raise ContractError("training-start review attestation differs")
    candidate_raw = validate_real_regular_file(run_root / TRAINING_START_PROJECT_PLAN_PATH).read_bytes()
    if candidate_raw.count(_training_start_plan_binding(run_root.name, request_digest)) != 1:
        raise ContractError("training-start plan request binding differs")
    return link


def _read_training_start_live_plan(path: Path | None = None) -> bytes:
    target = _repo_path(PROJECT_PLAN_RELATIVE_PATH) if path is None else Path(path)
    try:
        return validate_real_regular_file(target).read_bytes()
    except (ContractError, OSError) as exc:
        raise UnrecoverableOrphan("training-start live plan cannot be classified") from exc


def _classify_training_start_state(run_root: Path) -> tuple[str, Mapping[str, Any] | None]:
    request_exists = os.path.lexists(run_root / TRAINING_START_REQUEST_PATH)
    link_exists = os.path.lexists(run_root / TRAINING_START_LINK_PATH)
    snapshot_exists = os.path.lexists(run_root / TRAINING_START_PROJECT_PLAN_PATH)
    launch_raw = validate_real_regular_file(run_root / LAUNCH_PROJECT_PLAN_PATH).read_bytes()
    live_raw = _read_training_start_live_plan()
    if not request_exists:
        if live_raw != launch_raw:
            raise UnrecoverableOrphan("training-start live plan is ambiguous")
        if link_exists or snapshot_exists:
            raise ContractError("training-start artifacts precede request boundary")
        return "not_started", None
    if not link_exists or not snapshot_exists:
        if live_raw != launch_raw:
            raise UnrecoverableOrphan("training-start live plan is ambiguous")
        if link_exists or snapshot_exists:
            raise ContractError("training-start reviewed proof is partial")
        request = _validate_training_start_request_record(run_root)
        if hashlib.sha256(live_raw).hexdigest() != request["launch_project_plan_sha256"]:
            raise UnrecoverableOrphan("training-start live plan is ambiguous")
        return "awaiting_review", None
    request = _validate_training_start_request_record(run_root)
    link = _validate_training_start_linkage(run_root)
    live_digest = hashlib.sha256(live_raw).hexdigest()
    if live_digest == request["launch_project_plan_sha256"]:
        return "reviewed_ready", link
    if live_digest == link["training_start_project_plan_sha256"]:
        return "started", link
    raise UnrecoverableOrphan("training-start live plan is ambiguous")


def _training_start_state(run_root: Path) -> tuple[str, Mapping[str, Any] | None]:
    try:
        return _classify_training_start_state(run_root)
    except UnrecoverableOrphan:
        raise
    except OSError as exc:
        raise UnrecoverableOrphan("training-start persisted proof cannot be classified") from exc


def _select_training_start_attestation(run_root: Path, launch_sha256: str, source_directory: Path) -> tuple[bytes, bytes, str, str] | None:
    request = _validate_training_start_request_record(run_root)
    request_sha256 = hashlib.sha256(_read_regular_bytes(run_root / TRAINING_START_REQUEST_PATH)).hexdigest()
    scope = _training_start_scope(run_root.name)
    if not isinstance(launch_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", launch_sha256) is None:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_launch_plan_digest", "training_start_state": "awaiting_review"})
    try:
        source_records = _review_directory_bytes(source_directory, r"(?:[0-9a-f]{64}\.json|[0-9a-f]{64}\.project-plan\.md)")
    except (ContractError, OSError) as exc:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_review_directory", "training_start_state": "awaiting_review"})
    source_by_name = dict(source_records)
    matches: list[tuple[bytes, bytes, str, str]] = []
    for source_name, raw in source_records:
        match = re.fullmatch(r"([0-9a-f]{64})\.json", source_name)
        if match is None:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != match.group(1):
            continue
        try:
            artifact = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except Exception:
            continue
        if not isinstance(artifact, Mapping) or artifact.get("scope") != scope:
            continue
        target_records = artifact.get("target_records")
        request_record = {"path": TRAINING_START_REQUEST_PATH, "sha256": request_sha256}
        current_looking = artifact.get("scope") == scope
        accepted = False
        try:
            validate_exact_keys(artifact, ("schema_version", "reviewer", "scope", "target_records", "target_sha256", "findings", "finding_count"), "training-start review attestation")
            plan_record = target_records[0] if isinstance(target_records, list) and len(target_records) == 2 else None
            plan_sha256 = plan_record.get("sha256") if isinstance(plan_record, Mapping) and plan_record.get("path") == PROJECT_PLAN_RELATIVE_PATH else None
            expected_records = _training_start_target_records(plan_sha256, request_sha256) if isinstance(plan_sha256, str) else []
            target_digest = canonical_json_sha256(expected_records) if expected_records else ""
            candidate_raw = source_by_name.get(f"{plan_sha256}.project-plan.md", b"") if expected_records else b""
            accepted = (
                canonical_json_bytes(artifact) == raw
                and artifact["schema_version"] == "todorov.review-attestation.1"
                and artifact["reviewer"] == "feature-dev:code-reviewer"
                and artifact["target_records"] == expected_records
                and expected_records[1] == request_record
                and artifact["target_sha256"] == target_digest
                and artifact["findings"] == []
                and type(artifact["finding_count"]) is int
                and artifact["finding_count"] == 0
                and plan_sha256 != launch_sha256
                and hashlib.sha256(candidate_raw).hexdigest() == plan_sha256
                and candidate_raw.count(_training_start_plan_binding(run_root.name, request_sha256)) == 1
            )
        except (ContractError, IndexError):
            accepted = False
        if current_looking and not accepted:
            raise HardAbort("artifact_inconsistency", {"surface": "training_start_review_attestation", "training_start_state": "awaiting_review"})
        if accepted:
            matches.append((candidate_raw, raw, digest, target_digest))
    if len(matches) > 1:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_review_cardinality", "training_start_state": "awaiting_review"})
    return matches[0] if matches else None


def _remove_training_start_partial_artifacts(paths: Mapping[Path, tuple[int, int]]) -> None:
    _validate_owned_paths(paths)
    for path, identity in reversed(tuple(paths.items())):
        _unlink_owned_path(path, identity)


def _training_start_review_clock(monotonic_ns: Callable[[], int], request_start: int, require_margin: bool, state: str) -> int:
    now = monotonic_ns()
    if type(now) is not int or now < request_start:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_review_clock", "training_start_state": state})
    deadline = request_start + TRAINING_START_REVIEW_WAIT_NS
    if now >= deadline or require_margin and now > deadline - TRAINING_START_COMMIT_MARGIN_NS:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_plan_review_timeout", "training_start_state": state})
    return now


def establish_training_start_plan_barrier(
    run_root: Path,
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    source_directory: str | Path = REVIEW_EVIDENCE_DIRECTORY,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleeper: Callable[[float], None] = time.sleep,
    utc_reader: Callable[[], str] = utc_now,
) -> tuple[FrozenManifestAnchors, int]:
    root = Path(run_root)
    verify_manifest_anchors(root, anchors)
    if len(anchors.records) != 11:
        raise HardAbort("frozen_hash_change", {"surface": "training_start_anchor_state", "training_start_state": "not_started"})
    if signals.pending_signal is not None:
        raise HardAbort("signal_or_interruption", {"signal": signals.pending_signal, "training_start_state": "not_started"})
    _verify_active_frozen_hashes(root, anchors)
    launch_path = validate_real_regular_file(root / LAUNCH_PROJECT_PLAN_PATH)
    live_plan_path = _repo_path(PROJECT_PLAN_RELATIVE_PATH)
    launch_raw = launch_path.read_bytes()
    if _read_training_start_live_plan(live_plan_path) != launch_raw:
        raise UnrecoverableOrphan("training-start live plan is ambiguous")
    request_start = monotonic_ns()
    if type(request_start) is not int or request_start < 0:
        raise HardAbort("artifact_inconsistency", {"surface": "training_start_review_request_clock", "training_start_state": "not_started"})
    launch_digest = hashlib.sha256(launch_raw).hexdigest()
    request = {
        "schema_version": SCHEMA_VERSION,
        "run_id": root.name,
        "boundary": "training_start_review_request",
        "review_request_monotonic_ns": request_start,
        "review_request_wall_utc": utc_reader(),
        "launch_project_plan_sha256": launch_digest,
        "required_review_scope": _training_start_scope(root.name),
        "review_wait_timeout_seconds": TRAINING_START_REVIEW_WAIT_SECONDS,
    }
    request_path = root / TRAINING_START_REQUEST_PATH
    request_owned_paths: dict[Path, tuple[int, int]] = {}
    signals.defer()
    try:
        if os.path.lexists(request_path):
            raise ContractError("training-start request path already exists")
        write_canonical_json(request_path, request, owned_paths=request_owned_paths)
    except BaseException as exc:
        if request_path in request_owned_paths:
            try:
                _unlink_owned_path(request_path, request_owned_paths[request_path])
            except BaseException as cleanup_exc:
                signals.release()
                raise UnrecoverableOrphan("training-start request rollback failed") from cleanup_exc
        signals.release()
        raise HardAbort("artifact_inconsistency", {"surface": TRAINING_START_REQUEST_PATH, "training_start_state": "not_started"}) from exc
    pending = signals.release()
    try:
        awaiting_anchors = extend_frozen_manifest_anchors(root, anchors, (TRAINING_START_REQUEST_PATH,))
    except BaseException as exc:
        raise HardAbort("frozen_hash_change", {"surface": TRAINING_START_REQUEST_PATH, "training_start_state": "awaiting_review"}) from exc
    if pending is not None or signals.pending_signal is not None:
        raise HardAbort("signal_or_interruption", {"signal": pending if pending is not None else signals.pending_signal, "training_start_state": "awaiting_review"})
    copied_review_path: str | None = None
    reviewed_anchors: FrozenManifestAnchors | None = None
    candidate_raw: bytes | None = None
    candidate_temp: Path | None = None
    candidate_temp_owned_paths: dict[Path, tuple[int, int]] = {}
    partial_paths_owned: dict[Path, tuple[int, int]] = {}
    proof_committed = False
    publication_committed = False
    publication_durable = False
    try:
        while True:
            if signals.pending_signal is not None:
                raise HardAbort("signal_or_interruption", {"signal": signals.pending_signal, "training_start_state": "awaiting_review"})
            _training_start_review_clock(monotonic_ns, request_start, False, "awaiting_review")
            _verify_active_frozen_hashes(root, awaiting_anchors)
            if _read_training_start_live_plan(live_plan_path) != launch_raw:
                raise UnrecoverableOrphan("training-start live plan is ambiguous")
            candidate = _select_training_start_attestation(root, launch_digest, Path(source_directory))
            if candidate is None:
                sleeper(TRAINING_START_REVIEW_POLL_SECONDS)
                continue
            candidate_raw, review_raw, review_digest, target_digest = candidate
            _training_start_review_clock(monotonic_ns, request_start, True, "awaiting_review")
            if _read_training_start_live_plan(live_plan_path) != launch_raw:
                raise UnrecoverableOrphan("training-start live plan is ambiguous")
            training_start_snapshot_path = root / TRAINING_START_PROJECT_PLAN_PATH
            if os.path.lexists(training_start_snapshot_path):
                raise HardAbort("artifact_inconsistency", {"surface": TRAINING_START_PROJECT_PLAN_PATH, "training_start_state": "awaiting_review"})
            _write_exact_bytes(training_start_snapshot_path, candidate_raw, partial_paths_owned)
            _training_start_review_clock(monotonic_ns, request_start, True, "awaiting_review")
            copied_review_path = f"run/reviews/{review_digest}.json"
            copied_review_target = root / copied_review_path
            if os.path.lexists(copied_review_target):
                raise HardAbort("artifact_inconsistency", {"surface": copied_review_path, "training_start_state": "awaiting_review"})
            _write_exact_bytes(copied_review_target, review_raw, partial_paths_owned)
            commit_admission = _training_start_review_clock(monotonic_ns, request_start, True, "awaiting_review")
            request_digest = sha256_file(root / TRAINING_START_REQUEST_PATH)
            linkage = {
                "schema_version": SCHEMA_VERSION,
                "run_id": root.name,
                "boundary": "reviewed_plan_atomic_training_start",
                "request_path": TRAINING_START_REQUEST_PATH,
                "request_artifact_sha256": request_digest,
                "launch_snapshot_path": LAUNCH_PROJECT_PLAN_PATH,
                "training_start_snapshot_path": TRAINING_START_PROJECT_PLAN_PATH,
                "review_path": copied_review_path,
                "launch_project_plan_sha256": launch_digest,
                "training_start_project_plan_sha256": hashlib.sha256(candidate_raw).hexdigest(),
                "review_scope": _training_start_scope(root.name),
                "review_target_sha256": target_digest,
                "review_artifact_sha256": review_digest,
                "review_deadline_monotonic_ns": request_start + TRAINING_START_REVIEW_WAIT_NS,
                "commit_admission_monotonic_ns": commit_admission,
                "commit_admission_wall_utc": utc_reader(),
                "start_commit_margin_seconds": TRAINING_START_COMMIT_MARGIN_SECONDS,
                "start_commit_rule": "single_coordinator_atomic_replace_after_locked_exact_launch_recheck",
            }
            training_start_link_path = root / TRAINING_START_LINK_PATH
            if os.path.lexists(training_start_link_path):
                raise HardAbort("artifact_inconsistency", {"surface": TRAINING_START_LINK_PATH, "training_start_state": "awaiting_review"})
            write_canonical_json(training_start_link_path, linkage, owned_paths=partial_paths_owned)
            proof_committed = True
            _training_start_review_clock(monotonic_ns, request_start, True, "reviewed_ready")
            reviewed_anchors = extend_frozen_manifest_anchors(
                root,
                awaiting_anchors,
                (TRAINING_START_PROJECT_PLAN_PATH, copied_review_path, TRAINING_START_LINK_PATH),
            )
            _verify_active_frozen_hashes(root, reviewed_anchors)
            candidate_temp = live_plan_path.parent / f".{live_plan_path.name}.{root.name}.training-start"
            if os.path.lexists(candidate_temp):
                raise HardAbort("artifact_inconsistency", {"surface": "training_start_plan_candidate_temp", "training_start_state": "reviewed_ready"})
            _write_exact_bytes(candidate_temp, candidate_raw, candidate_temp_owned_paths)
            _training_start_review_clock(monotonic_ns, request_start, True, "reviewed_ready")

            def publish_candidate() -> tuple[int, int]:
                nonlocal publication_committed, publication_durable
                descriptor = os.open(os.fspath(live_plan_path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                locked = False

                def release_publication_descriptor() -> None:
                    cleanup_failure: BaseException | None = None
                    if locked:
                        try:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                        except BaseException as exc:
                            cleanup_failure = exc
                    try:
                        os.close(descriptor)
                    except BaseException as exc:
                        if cleanup_failure is None:
                            cleanup_failure = exc
                    if cleanup_failure is not None:
                        raise cleanup_failure

                try:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                    except OSError as exc:
                        if _read_training_start_live_plan(live_plan_path) != launch_raw:
                            raise UnrecoverableOrphan("training-start live plan is ambiguous") from exc
                        pending_numbers = signal.sigpending() & {signal.SIGINT, signal.SIGTERM}
                        pending_signal = signals.pending_signal if signals.pending_signal is not None else min(pending_numbers) if pending_numbers else None
                        if pending_signal is not None:
                            raise HardAbort("signal_or_interruption", {"signal": pending_signal, "training_start_state": "reviewed_ready"}) from exc
                        _training_start_review_clock(monotonic_ns, request_start, True, "reviewed_ready")
                        raise HardAbort("artifact_inconsistency", {"surface": "training_start_publication_lock", "training_start_state": "reviewed_ready"}) from exc
                    metadata = os.fstat(descriptor)
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise HardAbort("artifact_inconsistency", {"surface": "training_start_publication_lock", "training_start_state": "reviewed_ready"})
                    _verify_active_frozen_hashes(root, reviewed_anchors)
                    if _read_training_start_live_plan(live_plan_path) != launch_raw:
                        raise UnrecoverableOrphan("training-start live plan is ambiguous")
                    _training_start_review_clock(monotonic_ns, request_start, True, "reviewed_ready")
                    if _read_training_start_live_plan(live_plan_path) != launch_raw:
                        raise UnrecoverableOrphan("training-start live plan is ambiguous")
                    os.replace(candidate_temp, live_plan_path)
                    publication_committed = True
                    published_ns = monotonic_ns()
                    if type(published_ns) is not int or published_ns < request_start:
                        raise HardAbort("artifact_inconsistency", {"surface": "training_start_plan_publication_clock", "training_start_state": "started"})
                    os.fsync(descriptor)
                    publication_durable = True
                    durable_ns = monotonic_ns()
                    if type(durable_ns) is not int or durable_ns < published_ns:
                        raise HardAbort("artifact_inconsistency", {"surface": "training_start_plan_publication_clock", "training_start_state": "started"})
                    result = (published_ns, durable_ns)
                except BaseException as exc:
                    try:
                        release_publication_descriptor()
                    except BaseException as cleanup_exc:
                        if isinstance(exc, UnrecoverableOrphan):
                            raise exc
                        raise UnrecoverableOrphan("training-start publication cleanup failed") from cleanup_exc
                    raise
                try:
                    release_publication_descriptor()
                except BaseException as exc:
                    raise UnrecoverableOrphan("training-start publication cleanup failed") from exc
                return result

            result = signals.commit_guarded(publish_candidate)
            if not result.committed:
                raise HardAbort("signal_or_interruption", {"signal": result.pending_signal, "training_start_state": "reviewed_ready"})
            published_ns, durable_ns = result.value
            if published_ns >= request_start + TRAINING_START_REVIEW_WAIT_NS or durable_ns >= request_start + TRAINING_START_REVIEW_WAIT_NS:
                raise HardAbort("artifact_inconsistency", {"surface": "training_start_plan_publication_late", "training_start_state": "started"})
            _verify_active_frozen_hashes(root, reviewed_anchors)
            if result.pending_signal is not None or signals.pending_signal is not None:
                raise HardAbort("signal_or_interruption", {"signal": result.pending_signal if result.pending_signal is not None else signals.pending_signal, "training_start_state": "started"})
            return reviewed_anchors, published_ns
    except BaseException as exc:
        state = "started" if publication_committed else "reviewed_ready" if proof_committed else "awaiting_review"
        if isinstance(exc, UnrecoverableOrphan):
            raise
        if publication_committed and not publication_durable:
            raise UnrecoverableOrphan("training-start publication durability failed") from exc
        try:
            if not proof_committed:
                _remove_training_start_partial_artifacts(partial_paths_owned)
            if candidate_temp is not None and candidate_temp in candidate_temp_owned_paths and os.path.lexists(candidate_temp):
                _unlink_owned_path(candidate_temp, candidate_temp_owned_paths[candidate_temp])
        except BaseException as cleanup_exc:
            raise UnrecoverableOrphan("training-start rollback or cleanup failed") from cleanup_exc
        if isinstance(exc, HardAbort):
            exc.context["training_start_state"] = state
            raise
        raise HardAbort(
            "artifact_inconsistency",
            {"surface": "training_start_plan_barrier", "training_start_state": state},
        ) from exc


def _verify_active_frozen_hashes(run_root: Path, anchors: FrozenManifestAnchors) -> None:
    verify_manifest_anchors(run_root, anchors)
    try:
        training_start_state, training_start_link = _training_start_state(run_root)
    except UnrecoverableOrphan:
        raise
    except ContractError as exc:
        raise HardAbort("frozen_hash_change", {"surface": "training_start_plan_state"}) from exc
    try:
        validate_base_review_target_binding(run_root, training_start_state, training_start_link)
    except (ContractError, OSError, ValueError) as exc:
        raise HardAbort("frozen_hash_change", {"surface": "base_review_target_binding"}) from exc
    expected_anchor_count = {"not_started": 11, "awaiting_review": 12, "reviewed_ready": 15, "started": 15}[training_start_state]
    if len(anchors.records) != expected_anchor_count:
        raise HardAbort("frozen_hash_change", {"surface": "training_start_anchor_state"})
    source_manifest = json.loads((run_root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    config_manifest = json.loads((run_root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    for record in source_manifest["records"]:
        path = record["path"]
        if path.startswith("run/reviews/"):
            observed_path = run_root / path
        elif path.startswith("neuroloc/results/modular_sequence_role_mlx_reviews/"):
            observed_path = PROJECT_ROOT / path
        else:
            observed_path = _repo_path(path)
        try:
            observed_path = validate_real_regular_file(observed_path)
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": path}) from exc
        if sha256_file(observed_path) != record["sha256"]:
            raise HardAbort("frozen_hash_change", {"surface": path})
    launch_digest = sha256_file(validate_real_regular_file(run_root / LAUNCH_PROJECT_PLAN_PATH))
    plan_record_seen = False
    for record in config_manifest["records"]:
        observed_path = _repo_path(record["path"])
        if record["path"] == PROJECT_PLAN_RELATIVE_PATH:
            plan_record_seen = True
            if record["sha256"] != launch_digest:
                raise HardAbort("frozen_hash_change", {"surface": LAUNCH_PROJECT_PLAN_PATH})
            live_digest = hashlib.sha256(_read_training_start_live_plan(observed_path)).hexdigest()
            expected_live_digest = (
                launch_digest
                if training_start_state in {"not_started", "awaiting_review", "reviewed_ready"}
                else training_start_link["training_start_project_plan_sha256"]
            )
            if live_digest != expected_live_digest:
                raise UnrecoverableOrphan("training-start live plan is ambiguous")
            continue
        try:
            observed_path = validate_real_regular_file(observed_path)
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": record["path"]}) from exc
        if sha256_file(observed_path) != record["sha256"]:
            raise HardAbort("frozen_hash_change", {"surface": record["path"]})
    if not plan_record_seen:
        raise HardAbort("frozen_hash_change", {"surface": PROJECT_PLAN_RELATIVE_PATH})
    for record in config_manifest["review_records"]:
        copied = run_root / "run" / "reviews" / f"{record['artifact_sha256']}.json"
        source = REVIEW_EVIDENCE_DIRECTORY / f"{record['artifact_sha256']}.json"
        try:
            copied = validate_real_regular_file(copied)
            source = validate_real_regular_file(source)
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": "review_attestation"}) from exc
        if sha256_file(copied) != record["artifact_sha256"] or copied.read_bytes() != source.read_bytes():
            raise HardAbort("frozen_hash_change")
    if training_start_state in {"reviewed_ready", "started"}:
        review_path = training_start_link["review_path"]
        source = REVIEW_EVIDENCE_DIRECTORY / Path(review_path).name
        try:
            copied = validate_real_regular_file(run_root / review_path)
            source = validate_real_regular_file(source)
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": review_path}) from exc
        if copied.read_bytes() != source.read_bytes() or sha256_file(copied) != training_start_link["review_artifact_sha256"]:
            raise HardAbort("frozen_hash_change", {"surface": review_path})
    prereg = json.loads((run_root / "run" / "prereg.json").read_text(encoding="utf-8"))
    if prereg["payload_sha256"] != canonical_json_sha256(load_prereg_payload()):
        raise HardAbort("frozen_hash_change")


def _claim_expected_messages(worker: str) -> Iterable[dict[str, Any]]:
    if worker not in WORKER_JOB_ASSIGNMENTS:
        raise ContractError("claim protocol worker differs")
    for seed in WORKER_ASSIGNMENTS[worker]:
        if seed == RUNG_TWO_SEED:
            stage_specs = (("rung_two", 1536, 2, "rung_two"),)
        else:
            stage_specs = (
                ("donor", 1024, 1, "all_eligible_donor"),
                ("router_only", 768, 1, "selected"),
                ("joint", 512, 1, "selected"),
                ("dense_base", 1024, 1, "dense_causal"),
                ("dense_continuation", 512, 1, "dense_causal"),
            )
        event_sequence = 0
        for stage, updates, rung, model in stage_specs:
            for logical_update in range(1, updates + 1):
                for event in ("started", "completed"):
                    yield {
                        "kind": "attempt",
                        "seed": seed,
                        "stage": stage,
                        "logical_update": logical_update,
                        "event": event,
                        "event_sequence": event_sequence,
                        "rung": rung,
                        "model": model,
                    }
                    event_sequence += 1
        if seed == RUNG_TWO_SEED:
            for condition in RUNG_TWO_CONDITIONS:
                for batch_index in range(16):
                    yield {"kind": "status", "seed": seed, "stage": condition, "logical_update": batch_index}
            yield {"kind": "resource_refs", "seed": seed, "stage": "packaging", "logical_update": None}
        else:
            for batch_index in range(16):
                yield {"kind": "status", "seed": seed, "stage": "route_acquisition", "logical_update": batch_index}
            for condition in RUNG_ONE_CONDITIONS:
                for batch_index in range(16):
                    yield {"kind": "status", "seed": seed, "stage": condition, "logical_update": batch_index}
            yield {"kind": "resource_refs", "seed": seed, "stage": "evaluation", "logical_update": None}
            yield {"kind": "resource_refs", "seed": seed, "stage": "packaging", "logical_update": None}
    yield {"kind": "clean_complete"}


def claim_protocol_state(worker: str) -> dict[str, Any]:
    stream = iter(_claim_expected_messages(worker))
    return {"worker": worker, "stream": stream, "expected": next(stream), "complete": False}


def _validate_claim_protocol_transition(message: Mapping[str, Any], state: dict[str, Any]) -> None:
    expected = state.get("expected")
    worker = state.get("worker")
    if state.get("complete") or worker not in WORKER_JOB_ASSIGNMENTS or not isinstance(expected, Mapping) or message.get("kind") != expected["kind"]:
        raise ContractError("claim protocol transition differs")
    kind = message["kind"]
    if kind == "clean_complete":
        handshake = message.get("handshake")
        expected_sequences = {str(seed): 3071 if seed == RUNG_TWO_SEED else 7679 for seed in WORKER_ASSIGNMENTS[worker]}
        if not isinstance(handshake, Mapping) or handshake.get("last_event_sequence_by_construction_seed") != expected_sequences:
            raise ContractError("claim clean completion sequence differs")
    else:
        for key in ("seed", "stage", "logical_update"):
            if message.get(key) != expected[key]:
                raise ContractError("claim protocol identity differs")
        if kind == "attempt":
            row = message.get("row")
            if not isinstance(row, Mapping):
                raise ContractError("claim attempt row differs")
            for key in ("event", "event_sequence", "rung", "model"):
                if row.get(key) != expected[key]:
                    raise ContractError("claim attempt sequence differs")
    try:
        state["expected"] = next(state["stream"])
    except StopIteration:
        state["expected"] = None
        state["complete"] = True


def validate_claim_worker_message(
    message: Mapping[str, Any],
    expected_worker: str,
    completed_workers: set[str],
    protocol_state: dict[str, Any] | None = None,
) -> str:
    if not isinstance(message, Mapping) or expected_worker not in WORKER_JOB_ASSIGNMENTS:
        raise ContractError("claim worker message identity differs")
    kind = message.get("kind")
    exact_keys = {
        "attempt": ("kind", "worker", "seed", "stage", "logical_update", "row"),
        "status": ("kind", "worker", "seed", "stage", "logical_update"),
        "resource_refs": ("kind", "worker", "seed", "stage", "logical_update"),
        "clean_complete": ("kind", "handshake"),
        "unrecoverable_orphan": ("kind", "worker"),
        "hard_abort": ("kind", "worker", "reason_code", "context"),
    }
    if kind not in exact_keys:
        raise ContractError("claim worker message kind differs")
    validate_exact_keys(message, exact_keys[kind], "claim worker message")
    if kind == "clean_complete":
        if expected_worker in completed_workers:
            raise ContractError("duplicate clean completion handshake")
        validate_clean_completion_handshake(message["handshake"], expected_worker)
    else:
        if message["worker"] != expected_worker:
            raise ContractError("claim worker message attribution differs")
        if expected_worker in completed_workers:
            raise ContractError("message follows clean completion handshake")
    if kind in {"attempt", "status", "resource_refs"}:
        seed = message["seed"]
        if type(seed) is not int or seed not in WORKER_ASSIGNMENTS[expected_worker]:
            raise ContractError("claim worker seed assignment differs")
    if kind == "hard_abort" and message["reason_code"] not in HARD_ABORT_REASON_CODES:
        raise ContractError("claim worker abort reason differs")
    if kind == "hard_abort" and not isinstance(message["context"], Mapping):
        raise ContractError("claim worker abort context differs")
    if protocol_state is not None and kind not in {"hard_abort", "unrecoverable_orphan"}:
        _validate_claim_protocol_transition(message, protocol_state)
    return str(kind)


def _handle_claim_worker_message(
    message: Mapping[str, Any],
    worker: str,
    connection: Any,
    handshakes: set[str],
    writers: Mapping[str, CrashAtomicJsonlWriter],
    active_jobs: dict[str, dict[str, Any]],
    seed_sample_ids: Mapping[int, Sequence[int]],
    signals: SignalController,
    protocol_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    kind = validate_claim_worker_message(message, worker, handshakes, protocol_state)
    if kind == "attempt":
        row = message["row"]
        validate_attempt_row(row)
        if row["construction_seed"] != message["seed"] or row["stage"] != message["stage"] or row["logical_update"] != message["logical_update"]:
            raise HardAbort("artifact_inconsistency", {"worker": worker, "event_sequence": row.get("event_sequence"), "surface": "attempt_envelope"})
        seed = int(row["construction_seed"])
        relative = f"rung1/{seed}/attempts.jsonl" if row["rung"] == 1 else "rung2/83/attempts.jsonl"
        writer = writers[relative]
        result = writer.append(row)
        active_jobs[worker] = {"worker": worker, "seed": seed, "stage": str(row["stage"]), "logical_update": int(row["logical_update"])}
        if not result.acknowledged:
            raise HardAbort(result.reason_code or "artifact_inconsistency", {"worker": worker, "seed": seed, "stage": row["stage"], "logical_update": row["logical_update"], "event_sequence": row["event_sequence"]})
        return {"ack": True}
    if kind in {"status", "resource_refs"}:
        seed = int(message["seed"])
        active_jobs[worker] = {"worker": worker, "seed": seed, "stage": str(message["stage"]), "logical_update": message["logical_update"]}
        return {"ack": True, "sample_ids": list(seed_sample_ids[seed])} if kind == "resource_refs" else {"ack": True}
    if kind == "clean_complete":
        handshake_sequences = message["handshake"]["last_event_sequence_by_construction_seed"]
        for seed in WORKER_ASSIGNMENTS[worker]:
            relative = f"rung1/{seed}/attempts.jsonl" if seed != 83 else "rung2/83/attempts.jsonl"
            ledger_rows = writers[relative].validate_committed_prefix()
            if not ledger_rows or ledger_rows[-1]["event"] != "completed" or ledger_rows[-1]["event_sequence"] != handshake_sequences[str(seed)]:
                raise HardAbort("artifact_inconsistency", {"worker": worker, "seed": seed, "event_sequence": handshake_sequences[str(seed)], "surface": "clean_completion_ledger"})
        active_jobs.pop(worker, None)
        handshakes.add(worker)
        return {"ack": True}
    if kind == "unrecoverable_orphan":
        raise UnrecoverableOrphan(f"claim worker {worker} crossed the orphan boundary")
    context_values = dict(message.get("context") or {})
    if "worker" in context_values and context_values["worker"] != worker:
        raise HardAbort("artifact_inconsistency", {"worker": worker, "surface": "claim_abort_attribution"})
    context_values.setdefault("worker", worker)
    raise HardAbort(str(message["reason_code"]), context_values)


def run_claim_workers(
    run_root: Path,
    payload: Mapping[str, Any],
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    transition: TransitionResult,
    claim_start_monotonic_ns: int,
) -> dict[str, Any]:
    if transition.outcome != "ready" or transition.swap_baseline_bytes is None:
        raise ContractError("claim transition is not ready")
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_claim_workers")
    writers = dict(transition.writers)
    resource_writer = writers["run/resources.jsonl"]
    preworker_row = _resource_sample(run_root.name, "claim", 0, [], set(), {}, {}, transition.swap_baseline_bytes, 0, 0)
    append_result = resource_writer.append(preworker_row)
    if not append_result.acknowledged:
        raise HardAbort(append_result.reason_code or "artifact_inconsistency")
    preworker_observations = claim_resource_observations(resource_writer.validate_committed_prefix())
    if preworker_observations:
        observation = preworker_observations[0]
        raise HardAbort(observation["reason_code"], observation["context"])
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_claim_worker_spawn")
    multiprocessing = importlib.import_module("multiprocessing")
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    failure_latch = PrimaryFailureLatch(payload["abort_rules"]["hard_abort_registry"])
    specifications = tuple(
        {"worker": worker, "target": _claim_worker, "args": (worker, os.fspath(run_root), start_event), "name": f"modular-claim-{worker}"}
        for worker in ("A", "B")
    )
    try:
        processes, parents = spawn_worker_processes(context, specifications)
    except WorkerStartError as exc:
        latched_error = hard_abort_from_same_poll(failure_latch, [{"reason_code": "worker_exit", "context": {"worker": exc.worker}}])
        if latched_error is None:
            raise UnrecoverableOrphan("claim worker start failure did not latch") from exc
        raise latched_error
    worker_names = {int(process.pid): process.name.rsplit("-", 1)[-1] for process in processes}
    handshakes: set[str] = set()
    active_jobs: dict[str, dict[str, Any]] = {}
    protocol_states = {worker: claim_protocol_state(worker) for worker in ("A", "B")}
    seed_sample_ids: dict[int, list[int]] = {seed: [] for seed in (11, 23, 37, 53, 71, 83)}
    sample_id = 1
    next_sample_monotonic_ns = next_resource_sample_monotonic_ns(preworker_row)
    final_row: dict[str, Any] | None = None
    try:
        final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_claim_worker_release")
        try:
            training_start_state, training_start_link = _training_start_state(run_root)
        except UnrecoverableOrphan:
            raise
        except ContractError as exc:
            raise HardAbort("frozen_hash_change", {"surface": "training_start_plan_state", "training_start_state": "started"}) from exc
        if training_start_state != "started" or training_start_link is None:
            raise HardAbort("frozen_hash_change", {"surface": "training_start_plan_state", "training_start_state": training_start_state})
        release = signals.commit_guarded(start_event.set)
        if not release.committed or release.pending_signal is not None or signals.pending_signal is not None:
            raise HardAbort("signal_or_interruption", {"signal": release.pending_signal if release.pending_signal is not None else signals.pending_signal, "stage": "before_claim_worker_release", "training_start_state": "started"})
        while True:
            progressed = False
            observations = []
            received_messages = []
            failed_workers = set()
            sampled_row = None
            while True:
                received_in_pass = False
                for worker in ("A", "B"):
                    if worker in failed_workers or worker in handshakes:
                        continue
                    connection = parents[worker]
                    while worker not in failed_workers:
                        try:
                            ready = connection.poll(0)
                        except BaseException as exc:
                            observations.append(parent_worker_failure_observation(exc, worker, True))
                            failed_workers.add(worker)
                            break
                        if not ready:
                            break
                        progressed = True
                        received_in_pass = True
                        try:
                            message = connection.recv()
                        except BaseException as exc:
                            observations.append(parent_worker_failure_observation(exc, worker, True))
                            failed_workers.add(worker)
                            break
                        received_messages.append((worker, connection, message))
                if not received_in_pass:
                    break
            staged_responses = []
            for worker, connection, message in received_messages:
                if worker in failed_workers:
                    continue
                try:
                    response = _handle_claim_worker_message(message, worker, connection, handshakes, writers, active_jobs, seed_sample_ids, signals, protocol_states[worker])
                    if response is not None:
                        staged_responses.append((worker, connection, response, message))
                except BaseException as exc:
                    observations.append(parent_worker_failure_observation(exc, worker, False))
                    failed_workers.add(worker)
            if signals.pending_signal is not None:
                observations.append({"reason_code": "signal_or_interruption", "context": {"signal": signals.pending_signal}})
            if (time.monotonic_ns() - claim_start_monotonic_ns) / 1e9 > 1200:
                observations.append({"reason_code": "claim_elapsed_time", "context": {}})
            if time.monotonic_ns() >= next_sample_monotonic_ns:
                try:
                    accounting = _claim_accounting(writers)
                    row = _resource_sample(run_root.name, "claim", sample_id, processes, handshakes, worker_names, active_jobs, transition.swap_baseline_bytes, accounting.attempted_updates, accounting.attempted_token_positions)
                    result = resource_writer.append(row)
                    if not result.acknowledged:
                        raise HardAbort(result.reason_code or "artifact_inconsistency")
                    sampled_row = row
                    progressed = True
                    for job in row["active_jobs"]:
                        seed = job["seed"]
                        if seed is not None and sample_id not in seed_sample_ids[seed]:
                            seed_sample_ids[seed].append(sample_id)
                    resource_observations = claim_resource_observations(resource_writer.validate_committed_prefix())
                    if resource_observations:
                        observation = resource_observations[0]
                        raise HardAbort(observation["reason_code"], observation["context"])
                    _verify_active_frozen_hashes(run_root, anchors)
                    sample_id += 1
                    next_sample_monotonic_ns = next_resource_sample_monotonic_ns(row)
                except BaseException as exc:
                    observations.append(failure_observation_from_exception(exc, "resource_sampler_failure"))
            workers_complete, worker_observations = worker_exit_observations(processes, handshakes, worker_names)
            observations.extend(worker_observations)
            latched_error = hard_abort_from_same_poll(failure_latch, observations)
            if latched_error is not None:
                raise latched_error
            for worker, connection, response, message in staged_responses:
                response_observations = []
                if signals.pending_signal is not None:
                    response_observations.append({"reason_code": "signal_or_interruption", "context": {"signal": signals.pending_signal, "worker": worker}})
                if (time.monotonic_ns() - claim_start_monotonic_ns) / 1e9 > 1200:
                    response_observations.append({"reason_code": "claim_elapsed_time", "context": {"worker": worker}})
                response_error = hard_abort_from_same_poll(failure_latch, response_observations)
                if response_error is not None:
                    raise response_error
                if message["kind"] == "resource_refs":
                    response = {"ack": True, "sample_ids": list(seed_sample_ids[int(message["seed"])])}
                try:
                    connection.send(response)
                except BaseException as exc:
                    response_error = hard_abort_from_same_poll(failure_latch, [parent_worker_failure_observation(exc, worker, True)])
                    if response_error is None:
                        raise UnrecoverableOrphan("claim acknowledgment failure did not latch")
                    raise response_error
            if workers_complete and sampled_row is not None and sampled_row["expected_pids"] == [os.getpid()]:
                final_row = sampled_row
                break
            if not progressed:
                time.sleep(0.01)
        accounting = _claim_accounting(writers)
        if final_row is None:
            raise HardAbort("artifact_inconsistency")
        committed_resources = resource_writer.validate_committed_prefix()
        seed_sample_ids = {
            seed: resource_sample_ids_for_seed(committed_resources, seed)
            for seed in (*RUNG_ONE_SEEDS, RUNG_TWO_SEED)
        }
        for writer in writers.values():
            writer.recover_uncommitted_suffix()
            writer.close()
        _finalize_seed_resource_references(run_root, seed_sample_ids)
        validate_parent_ledger_accounting(run_root)
    except UnrecoverableOrphan:
        quiesce_worker_processes(processes)
        raise
    except BaseException as exc:
        if isinstance(exc, HardAbort) and exc.primary_latch_monotonic_ns is not None:
            latched_error = exc
        else:
            latched_error = hard_abort_from_same_poll(failure_latch, [failure_observation_from_exception(exc, "artifact_inconsistency")])
            if latched_error is None:
                raise UnrecoverableOrphan("claim failure did not latch") from exc
        raise quiesce_after_primary_latch(latched_error, processes)
    finally:
        close_parent_connections(parents)
    if accounting.unpaired_attempts:
        raise HardAbort("unpaired_attempt")
    if accounting.attempted_updates != 20736 or accounting.completed_updates != 20736 or accounting.completed_token_positions != 45613056:
        raise HardAbort("artifact_inconsistency")
    return {
        "accounting": accounting,
        "resource_final_sample_id": final_row["sample_id"],
        "resource_sampling_end_monotonic_ns": final_row["monotonic_ns"],
        "resource_rows": validate_canonical_jsonl_prefix(run_root / "run" / "resources.jsonl", (run_root / "run" / "resources.jsonl").stat().st_size, validate_resource_row),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(validate_canonical_jsonl_prefix(path, path.stat().st_size, validate_attempt_row))


def _accounting_from_run_root(run_root: Path) -> AttemptAccounting:
    attempted_updates = 0
    completed_updates = 0
    attempted_tokens = 0
    completed_tokens = 0
    attempted_seeds: set[int] = set()
    completed_seeds: set[int] = set()
    last = {}
    unpaired = []
    for relative in CLAIM_LEDGER_PATHS[1:]:
        path = run_root / relative
        if not path.is_file():
            continue
        accounting = derive_attempt_accounting(_read_jsonl(path))
        attempted_updates += accounting.attempted_updates
        completed_updates += accounting.completed_updates
        attempted_tokens += accounting.attempted_token_positions
        completed_tokens += accounting.completed_token_positions
        attempted_seeds.update(accounting.attempted_seeds)
        completed_seeds.update(accounting.completed_seeds)
        last.update(accounting.last_event_sequence_by_seed)
        unpaired.extend(accounting.unpaired_attempts)
    return AttemptAccounting(attempted_updates, completed_updates, attempted_tokens, completed_tokens, tuple(sorted(attempted_seeds)), tuple(sorted(completed_seeds)), dict(sorted(last.items())), tuple(unpaired))


def _resource_state(run_root: Path, phase: str, swap_baseline: int | None) -> dict[str, Any]:
    relative = "run/pilot_resources.jsonl" if phase == "pilot" else "run/resources.jsonl"
    path = run_root / relative
    if not path.is_file():
        return {"last_sample_id": None, "peak_rss_bytes": None, "swap_baseline_bytes": swap_baseline, "swap_peak_bytes": swap_baseline}
    rows = validate_canonical_jsonl_prefix(path, path.stat().st_size, validate_resource_row)
    if not rows:
        return {"last_sample_id": None, "peak_rss_bytes": None, "swap_baseline_bytes": swap_baseline, "swap_peak_bytes": swap_baseline}
    return {
        "last_sample_id": rows[-1]["sample_id"],
        "peak_rss_bytes": max(row["aggregate_rss_bytes"] for row in rows),
        "swap_baseline_bytes": swap_baseline,
        "swap_peak_bytes": None if swap_baseline is None else max([swap_baseline, *(row["swap_used_bytes"] for row in rows)]),
    }


def _preflight_abort_training_start_surface(
    run_root: Path,
    phase: str,
    training_start_state: str,
) -> Mapping[str, Any] | None:
    if training_start_state not in {"not_started", "awaiting_review", "reviewed_ready", "started"} or phase in {"prepilot", "pilot"} and training_start_state != "not_started":
        raise UnrecoverableOrphan("abort cleanup training-start state differs")
    observed_state, linkage = _training_start_state(run_root)
    if observed_state != training_start_state:
        raise UnrecoverableOrphan("abort cleanup training-start state is ambiguous")
    config = _canonical_json_artifact(validate_real_regular_file(run_root / "run" / "config_manifest.json"))
    base_review_names = {
        f"{record['artifact_sha256']}.json"
        for record in config["review_records"]
        if isinstance(record, Mapping) and isinstance(record.get("artifact_sha256"), str)
    }
    if len(base_review_names) != 4:
        raise UnrecoverableOrphan("abort cleanup base review registry differs")
    retained_status_review = None
    if linkage is not None:
        retained_status_review = Path(linkage["review_path"]).name
    review_directory = run_root / "run" / "reviews"
    if review_directory.is_symlink() or not review_directory.is_dir():
        raise UnrecoverableOrphan("abort cleanup review surface differs")
    expected_review_names = base_review_names | ({retained_status_review} if retained_status_review is not None else set())
    observed_review_names = set()
    for path in review_directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise UnrecoverableOrphan("abort cleanup review surface differs")
        observed_review_names.add(path.name)
    if observed_review_names != expected_review_names:
        raise UnrecoverableOrphan("abort cleanup review surface differs")
    return linkage


def _validate_abort_training_start_surface(
    run_root: Path,
    phase: str,
    training_start_state: str,
) -> Mapping[str, Any] | None:
    try:
        return _preflight_abort_training_start_surface(run_root, phase, training_start_state)
    except UnrecoverableOrphan:
        raise
    except BaseException as exc:
        raise UnrecoverableOrphan("abort cleanup training-start surface is ambiguous") from exc


def _abort_cleanup_identities(run_root: Path, phase: str) -> dict[Path, tuple[int, int]]:
    identities: dict[Path, tuple[int, int]] = {}

    def capture(path: Path) -> None:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise UnrecoverableOrphan("abort cleanup path is symbolic")
        identities[path] = (metadata.st_dev, metadata.st_ino)
        if stat.S_ISDIR(metadata.st_mode):
            for child in path.iterdir():
                capture(child)

    for relative in ("run/completion.json", "summary.json", "SHA256SUMS", "data", "rung1", "rung2"):
        path = run_root / relative
        if os.path.lexists(path):
            capture(path)
    for relative in ("run/resources.jsonl", "run/pilot_resources.jsonl", "run/pilot.json"):
        path = run_root / relative
        if os.path.lexists(path):
            capture(path)
    if phase != "claim" and os.path.lexists(run_root / "run" / "resources.jsonl"):
        raise UnrecoverableOrphan("abort cleanup found an unowned claim ledger")
    if phase == "prepilot" and os.path.lexists(run_root / "run" / "pilot_resources.jsonl"):
        raise UnrecoverableOrphan("abort cleanup found an unowned pilot ledger")
    if phase == "prepilot" and os.path.lexists(run_root / "run" / "pilot.json"):
        raise UnrecoverableOrphan("abort cleanup found an unowned pilot result")
    return identities


def _cleanup_abort_surface(run_root: Path, phase: str, training_start_state: str = "not_started") -> None:
    try:
        _validate_abort_training_start_surface(run_root, phase, training_start_state)
        identities = _abort_cleanup_identities(run_root, phase)
        for relative in ("run/completion.json", "summary.json", "SHA256SUMS"):
            path = run_root / relative
            if path in identities:
                _unlink_owned_path(path, identities[path])
        data_root = run_root / "data"
        if data_root in identities:
            _validate_owned_paths({path: identity for path, identity in identities.items() if path == data_root or data_root in path.parents})
            _remove_tree_and_fsync(data_root)
        for rung_name in ("rung1", "rung2"):
            rung_root = run_root / rung_name
            if rung_root not in identities:
                continue
            for seed_root in list(rung_root.iterdir()):
                metadata = os.lstat(seed_root)
                if stat.S_ISLNK(metadata.st_mode):
                    raise UnrecoverableOrphan("abort cleanup seed path is symbolic")
                if not stat.S_ISDIR(metadata.st_mode):
                    _unlink_owned_path(seed_root, identities[seed_root])
                    continue
                for child in list(seed_root.iterdir()):
                    if child.name == "attempts.jsonl" and phase == "claim":
                        continue
                    child_metadata = os.lstat(child)
                    if stat.S_ISDIR(child_metadata.st_mode):
                        _validate_owned_paths({path: identity for path, identity in identities.items() if path == child or child in path.parents})
                        _remove_tree_and_fsync(child)
                    else:
                        _unlink_owned_path(child, identities[child])
                if phase != "claim" or not (seed_root / "attempts.jsonl").is_file():
                    if not any(seed_root.iterdir()):
                        _validate_owned_path(seed_root, identities[seed_root])
                        seed_root.rmdir()
                        fsync_directory(rung_root)
            if not any(rung_root.iterdir()):
                _validate_owned_path(rung_root, identities[rung_root])
                rung_root.rmdir()
                fsync_directory(run_root)
        pilot = run_root / "run" / "pilot.json"
        if phase == "pilot" and pilot in identities:
            _unlink_owned_path(pilot, identities[pilot])
    except UnrecoverableOrphan:
        raise
    except BaseException as exc:
        raise UnrecoverableOrphan("abort cleanup failed") from exc


def _frozen_hash_records(run_root: Path) -> list[dict[str, str]]:
    records: dict[str, str] = {}
    for manifest_name in ("source_manifest.json", "config_manifest.json"):
        path = run_root / "run" / manifest_name
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            records[record["path"]] = record["sha256"]
    prereg = run_root / "run" / "prereg.json"
    if prereg.is_file():
        records["run/prereg.json"] = sha256_file(prereg)
    for relative in (LAUNCH_PROJECT_PLAN_PATH, TRAINING_START_REQUEST_PATH, TRAINING_START_PROJECT_PLAN_PATH, TRAINING_START_LINK_PATH):
        path = run_root / relative
        if path.is_file():
            records[relative] = sha256_file(path)
    link_path = run_root / TRAINING_START_LINK_PATH
    if link_path.is_file():
        link = _validate_training_start_linkage(run_root)
        records[link["review_path"]] = sha256_file(run_root / link["review_path"])
    return [{"path": path, "sha256": records[path]} for path in sorted(records)]


def _finalize_hard_abort_impl(
    run_root: Path,
    payload: Mapping[str, Any],
    signals: SignalController,
    reason_code: str,
    phase: str,
    context: Mapping[str, Any],
    abort_accounting_start_monotonic_ns: int,
    abort_wall_start_utc: str,
    primary_latch_monotonic_ns: int,
    writers: Mapping[str, CrashAtomicJsonlWriter],
    swap_baseline: int | None,
    training_start_state: str = "not_started",
    aborted_owned_paths: dict[Path, tuple[int, int]] | None = None,
) -> None:
    if training_start_state not in {"not_started", "awaiting_review", "reviewed_ready", "started"}:
        raise UnrecoverableOrphan("abort training-start state differs")
    _validate_abort_training_start_surface(run_root, phase, training_start_state)
    if reason_code == "signal_or_interruption":
        signals.acknowledge_pending_through(primary_latch_monotonic_ns)
    accounting = _accounting_from_run_root(run_root)
    pilot_attempted = context.get("pilot_attempted_updates") if phase == "pilot" else None
    pilot_tokens = context.get("pilot_token_positions") if phase == "pilot" else None
    if (pilot_attempted is None) != (pilot_tokens is None) or pilot_attempted is not None and (type(pilot_attempted) is not int or type(pilot_tokens) is not int or pilot_attempted < 0 or pilot_tokens < 0):
        raise UnrecoverableOrphan("pilot abort charge context differs")
    best_effort_abort_resource_sample(run_root, phase, writers, swap_baseline, accounting, reason_code, pilot_attempted, pilot_tokens)
    for writer in writers.values():
        try:
            if writer._descriptor is not None and not writer._closed:
                writer.recover_uncommitted_suffix()
                writer.close()
        except Exception as exc:
            raise UnrecoverableOrphan("abort ledger finalization failed") from exc
    resource_phase = "pilot" if phase in {"prepilot", "pilot"} else "claim"
    resource_relative = "run/pilot_resources.jsonl" if resource_phase == "pilot" else "run/resources.jsonl"
    resource_path = run_root / resource_relative
    if phase != "prepilot" and resource_path.is_file():
        resource_rows = validate_canonical_jsonl_prefix(resource_path, resource_path.stat().st_size, validate_resource_row)
        validate_resource_timeline(resource_rows, resource_phase, require_clean_final=False)
    _cleanup_abort_surface(run_root, phase, training_start_state)
    registry = {row["reason_code"]: row["condition"] for row in payload["abort_rules"]["hard_abort_registry"]}
    seed_value = context.get("seed")
    seed = seed_value if type(seed_value) is int else None
    last_event = accounting.last_event_sequence_by_seed.get(seed) if seed is not None else None
    resource = {"last_sample_id": None, "peak_rss_bytes": None, "swap_baseline_bytes": None, "swap_peak_bytes": None} if phase == "prepilot" else _resource_state(run_root, resource_phase, swap_baseline)
    aborted = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "reason_code": reason_code,
        "condition": registry[reason_code],
        "phase": phase,
        "training_start_state": training_start_state,
        "worker": context.get("worker") if isinstance(context.get("worker"), str) else None,
        "seed": seed,
        "stage": context.get("stage") if isinstance(context.get("stage"), str) else None,
        "logical_update": context.get("logical_update") if type(context.get("logical_update")) is int else None,
        "last_event_sequence": last_event,
        "monotonic_elapsed_seconds": (primary_latch_monotonic_ns - abort_accounting_start_monotonic_ns) / 1e9,
        "wall_start_utc": abort_wall_start_utc,
        "wall_end_utc": utc_now(),
        "completed_work": {"updates": accounting.completed_updates, "token_positions": accounting.completed_token_positions, "seeds": list(accounting.completed_seeds)},
        "attempted_work": {"updates": accounting.attempted_updates, "token_positions": accounting.attempted_token_positions, "seeds": list(accounting.attempted_seeds)},
        "resource_state": resource,
        "frozen_hashes": _frozen_hash_records(run_root),
        "new_run_required": True,
    }
    validate_exact_keys(aborted, ABORTED_KEYS, "aborted artifact")
    write_canonical_json(run_root / "ABORTED.json", aborted, owned_paths=aborted_owned_paths)
    closure_kind = "prepilot_abort" if phase == "prepilot" else "pilot_abort" if phase == "pilot" else "claim_abort"
    expected_paths = validate_artifact_closure(run_root, payload, closure_kind)
    write_sha256s_terminal(run_root, expected_paths=expected_paths, signals=signals, preserve_primary=True)


def finalize_hard_abort(
    run_root: Path,
    payload: Mapping[str, Any],
    signals: SignalController,
    reason_code: str,
    phase: str,
    context: Mapping[str, Any],
    abort_accounting_start_monotonic_ns: int,
    abort_wall_start_utc: str,
    primary_latch_monotonic_ns: int,
    writers: Mapping[str, CrashAtomicJsonlWriter],
    swap_baseline: int | None,
    training_start_state: str = "not_started",
) -> None:
    aborted_path = run_root / "ABORTED.json"
    aborted_owned_paths: dict[Path, tuple[int, int]] = {}
    try:
        _finalize_hard_abort_impl(
            run_root,
            payload,
            signals,
            reason_code,
            phase,
            context,
            abort_accounting_start_monotonic_ns,
            abort_wall_start_utc,
            primary_latch_monotonic_ns,
            writers,
            swap_baseline,
            training_start_state,
            aborted_owned_paths,
        )
    except BaseException as exc:
        if signals.terminal:
            return
        try:
            if aborted_path in aborted_owned_paths:
                _unlink_owned_path(aborted_path, aborted_owned_paths[aborted_path])
        except BaseException as cleanup_exc:
            raise UnrecoverableOrphan("hard-abort artifact rollback failed") from cleanup_exc
        if isinstance(exc, UnrecoverableOrphan):
            raise
        raise UnrecoverableOrphan("hard-abort finalization failed") from exc


def best_effort_abort_resource_sample(
    run_root: Path,
    phase: str,
    writers: Mapping[str, CrashAtomicJsonlWriter],
    swap_baseline: int | None,
    accounting: AttemptAccounting,
    primary_reason_code: str,
    attempted_updates: int | None = None,
    token_positions: int | None = None,
) -> bool:
    if primary_reason_code != "resource_sampler_failure" or swap_baseline is None:
        return False
    if (attempted_updates is None) != (token_positions is None):
        raise ContractError("abort resource charge pair differs")
    observed_updates = accounting.attempted_updates if attempted_updates is None else attempted_updates
    observed_tokens = accounting.attempted_token_positions if token_positions is None else token_positions
    if type(observed_updates) is not int or type(observed_tokens) is not int or observed_updates < 0 or observed_tokens < 0:
        raise ContractError("abort resource charge differs")
    relative = "run/pilot_resources.jsonl" if phase in {"prepilot", "pilot"} else "run/resources.jsonl" if phase in {"claim", "packaging"} else None
    writer = writers.get(relative) if relative is not None else None
    if writer is None or writer._closed or writer._descriptor is None:
        return False
    try:
        rows = writer.validate_committed_prefix()
        sample_phase = "pilot" if relative == "run/pilot_resources.jsonl" else "claim"
        validate_resource_timeline(rows, sample_phase, require_clean_final=False)
        if rows:
            return False
        if observed_updates != 0 or observed_tokens != 0:
            return False
        row = _resource_sample(run_root.name, sample_phase, 0, (), set(), {}, {}, swap_baseline, observed_updates, observed_tokens)
        result = writer.append(row)
        if not result.committed or not result.acknowledged:
            return False
        validate_resource_timeline(writer.validate_committed_prefix(), sample_phase, require_clean_final=False)
        return True
    except UnrecoverableOrphan:
        raise
    except BaseException:
        return False


def _rung_one_evaluation_identity_order(payload: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    identities = []
    for condition in payload["stages"]["rung_one"]["evaluation_arm_order"]:
        if condition == "dense_causal":
            identities.append((condition, "answer_accuracy", "all"))
            continue
        identities.extend(
            (
                (condition, "answer_accuracy", "all"),
                (condition, "original_source_hit_rate", "all"),
                (condition, "answer_accuracy", "source_hit"),
                (condition, "answer_accuracy", "source_miss"),
                (condition, "query_underfill_count", "not_applicable"),
            )
        )
        if condition == "carry_shuffle":
            identities.append((condition, "foreign_source_hit_rate", "all"))
            for metric in ("answer_accuracy", "original_source_hit_rate", "foreign_source_hit_rate"):
                identities.extend(((condition, metric, "changed_condition"), (condition, metric, "same_condition")))
    identities.extend(
        (
            ("intact", "selected_mask_oracle_max_error", "not_applicable"),
            ("all_routed_training_and_evaluation", "route_overflow_count", "not_applicable"),
        )
    )
    if len(identities) != 65:
        raise ContractError("rung-one evaluation identity registry differs")
    return tuple(identities)


def _resolved_gate_registry_row(registry_row: Mapping[str, Any], seed: int) -> dict[str, Any]:
    resolved = dict(registry_row)
    for key in ("gate_threshold_count", "denominator"):
        value = resolved[key]
        if isinstance(value, Mapping):
            if str(seed) not in value:
                raise ContractError("seed gate registry value is absent")
            resolved[key] = value[str(seed)]
    return resolved


def _validate_gate_decision_row(row: Mapping[str, Any], registry_row: Mapping[str, Any], seed: int) -> None:
    registered = _resolved_gate_registry_row(registry_row, seed)
    for field in ("gate_id", "condition", "metric", "stratum", "gate_operator", "gate_threshold", "gate_threshold_count", "gate_threshold_unit", "denominator"):
        if row[field] != registered[field]:
            raise ContractError("evaluation gate registry field differs")
    observed = row["estimate"] if row["gate_threshold_unit"] == "absolute_error" else row["numerator"]
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise ContractError("evaluation gate observed value differs")
    target = row["gate_threshold_count"] if row["gate_threshold_count"] is not None else row["gate_threshold"]
    if row["gate_operator"] == ">=":
        expected_pass = observed >= target
    elif row["gate_operator"] == "<=":
        expected_pass = observed <= target
    elif row["gate_operator"] == "==":
        expected_pass = observed == target
    else:
        raise ContractError("evaluation gate operator differs")
    if row["gate_pass"] is not expected_pass:
        raise ContractError("evaluation gate decision contradicts value")


def validate_evaluation_scalar_row(row: Mapping[str, Any]) -> None:
    common_nulls = (
        "denominator",
        "wilson95_low",
        "wilson95_high",
        "answer_correct",
        "answer_total",
        "original_source_hits",
        "original_source_total",
        "foreign_source_hits",
        "foreign_source_total",
        "raw_remote_ids",
        "effective_remote_ids",
    )
    if any(row.get(field) is not None for field in common_nulls):
        raise ContractError("evaluation scalar common nullability differs")
    metric = row.get("metric")
    if metric == "selected_mask_oracle_max_error":
        value = row.get("estimate")
        if row.get("numerator") is not None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ContractError("evaluation oracle scalar differs")
        expected = {
            "query_underfill_count": None,
            "overflow_count": None,
            "max_bucket_load": None,
            "selected_mask_oracle_max_error": value,
        }
    elif metric in {"query_underfill_count", "route_overflow_count"}:
        value = row.get("numerator")
        if type(value) is not int or value < 0 or row.get("estimate") != value:
            raise ContractError("evaluation count scalar differs")
        if metric == "query_underfill_count":
            expected = {
                "query_underfill_count": value,
                "overflow_count": None,
                "max_bucket_load": None,
                "selected_mask_oracle_max_error": None,
            }
        else:
            maximum = row.get("max_bucket_load")
            if type(maximum) is not int or maximum < 0:
                raise ContractError("evaluation overflow maximum differs")
            expected = {
                "query_underfill_count": None,
                "overflow_count": value,
                "max_bucket_load": maximum,
                "selected_mask_oracle_max_error": None,
            }
    else:
        raise ContractError("evaluation scalar metric differs")
    if any(row.get(field) != expected[field] for field in expected):
        raise ContractError("evaluation scalar value matrix differs")


def _validate_evaluation_row_common(row: Mapping[str, Any], payload: Mapping[str, Any], run_root: Path, rung: int, seed: int) -> None:
    keys = payload["artifacts"]["schemas"]["evaluation_row"]["exact_keys"]
    validate_exact_keys(row, keys, "evaluation row")
    if row["schema_version"] != SCHEMA_VERSION or row["run_id"] != run_root.name or row["rung"] != rung or row["claim_seed"] != seed or row["construction_seed"] != seed:
        raise ContractError("evaluation row identity differs")
    if not isinstance(row["condition"], str) or not isinstance(row["metric"], str) or not isinstance(row["stratum"], str):
        raise ContractError("evaluation row enum differs")
    if isinstance(row["elapsed_seconds"], bool) or not isinstance(row["elapsed_seconds"], (int, float)) or not math.isfinite(float(row["elapsed_seconds"])) or row["elapsed_seconds"] < 0:
        raise ContractError("evaluation elapsed time differs")
    ids = row["resource_sample_ids"]
    if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids) or ids != sorted(set(ids)):
        raise ContractError("evaluation resource references differ")
    for field in ("checkpoint_sha256", "eval_data_sha256", "population_sha256"):
        value = row[field]
        if value is not None and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
            raise ContractError("evaluation digest differs")
    provenance = row["provenance_sha256s"]
    if not isinstance(provenance, list) or not provenance or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in provenance):
        raise ContractError("evaluation provenance differs")
    if row["checkpoint_sha256"] is not None or row["eval_data_sha256"] is not None:
        if provenance != [row["checkpoint_sha256"], row["eval_data_sha256"]]:
            raise ContractError("evaluation provenance does not match endpoint and data")
    if rung == 1 and row["metric"] == "selected_mask_oracle_max_error":
        expected_paths = (
            run_root / "rung1" / str(seed) / "selected_canonical_state_manifest.json",
            run_root / "run" / "sentinels" / "selected_attention_oracle_payload.json",
            run_root / "rung1" / str(seed) / "selected_attention_oracle_detail.json",
        )
        if not all(path.is_file() and not path.is_symlink() for path in expected_paths) or provenance != [sha256_file(path) for path in expected_paths]:
            raise ContractError("selected oracle provenance differs")
    if row["gate_id"] is None:
        if any(row[field] is not None for field in ("gate_operator", "gate_threshold", "gate_threshold_count", "gate_threshold_unit", "gate_pass")):
            raise ContractError("diagnostic row contains gate fields")
    elif type(row["gate_pass"]) is not bool:
        raise ContractError("gated row decision differs")
    if row["denominator"] is not None:
        if type(row["numerator"]) is not int or type(row["denominator"]) is not int or row["denominator"] < 0 or not 0 <= row["numerator"] <= row["denominator"]:
            raise ContractError("evaluation binary population differs")
        estimate, low, high = _wilson(row["numerator"], row["denominator"])
        if (row["estimate"], row["wilson95_low"], row["wilson95_high"]) != (estimate, low, high):
            raise ContractError("evaluation binary reduction differs")
    metric_counters = {
        "answer_accuracy": ("answer_correct", "answer_total"),
        "original_source_hit_rate": ("original_source_hits", "original_source_total"),
        "foreign_source_hit_rate": ("foreign_source_hits", "foreign_source_total"),
    }
    if row["metric"] not in metric_counters:
        validate_evaluation_scalar_row(row)
    all_counter_fields = {field for pair in metric_counters.values() for field in pair}
    if row["metric"] in metric_counters:
        numerator_field, denominator_field = metric_counters[row["metric"]]
        if row[numerator_field] != row["numerator"] or row[denominator_field] != row["denominator"]:
            raise ContractError("evaluation metric counter differs from reduction")
        if any(row[field] is not None for field in all_counter_fields - {numerator_field, denominator_field}) or any(row[field] is not None for field in ("query_underfill_count", "overflow_count", "max_bucket_load", "selected_mask_oracle_max_error")):
            raise ContractError("evaluation binary counter nullability differs")
    elif row["metric"] == "query_underfill_count" and (row["query_underfill_count"] != row["numerator"] or row["estimate"] != row["numerator"]):
        raise ContractError("evaluation underfill counter differs from reduction")
    elif row["metric"] == "route_overflow_count" and (row["overflow_count"] != row["numerator"] or row["estimate"] != row["numerator"]):
        raise ContractError("evaluation overflow counter differs from reduction")
    elif row["metric"] == "selected_mask_oracle_max_error" and row["selected_mask_oracle_max_error"] != row["estimate"]:
        raise ContractError("evaluation oracle value differs from reduction")
    if row["metric"] not in metric_counters:
        permitted = {
            "query_underfill_count": {"query_underfill_count"},
            "route_overflow_count": {"overflow_count", "max_bucket_load"},
            "selected_mask_oracle_max_error": {"selected_mask_oracle_max_error"},
        }.get(row["metric"])
        if permitted is None:
            raise ContractError("evaluation metric differs")
        special_fields = all_counter_fields | {"query_underfill_count", "overflow_count", "max_bucket_load", "selected_mask_oracle_max_error"}
        if any(row[field] is not None for field in special_fields - permitted):
            raise ContractError("evaluation diagnostic counter nullability differs")
    if row["stratum"] == "all" and row["population_sha256"] != _population_sha(range(512)):
        raise ContractError("evaluation all-population digest differs")
    if row["stratum"] == "not_applicable" and row["population_sha256"] is not None:
        raise ContractError("evaluation inapplicable population digest differs")
    if row["denominator"] == 0 and row["population_sha256"] != _population_sha(()):
        raise ContractError("evaluation empty-population digest differs")
    for field in ("numerator", "estimate", "wilson95_low", "wilson95_high", "gate_threshold"):
        value = row[field]
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError("evaluation nonfinite value")


def validate_gate_input_package(run_root: Path, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    decisions = []
    r1_registry = payload["gates"]["rung_one_registry"]
    r1_by_id = {row["gate_id"]: row for row in r1_registry}
    if len(r1_by_id) != 24:
        raise ContractError("rung-one gate registry identity differs")
    expected_identities = _rung_one_evaluation_identity_order(payload)
    for seed in RUNG_ONE_SEEDS:
        path = run_root / "rung1" / str(seed) / "evaluation.jsonl"
        raw_lines = path.read_bytes().splitlines(keepends=True)
        rows = [json.loads(line[:-1].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))) for line in raw_lines if line.endswith(b"\n")]
        if len(raw_lines) != 65 or len(rows) != 65 or any(canonical_json_bytes(row) + b"\n" != raw for row, raw in zip(rows, raw_lines)):
            raise ContractError("rung-one evaluation serialization or cardinality differs")
        identities = tuple((row.get("condition"), row.get("metric"), row.get("stratum")) for row in rows)
        if identities != expected_identities:
            raise ContractError("rung-one evaluation row order differs")
        seen = set()
        for row in rows:
            _validate_evaluation_row_common(row, payload, run_root, 1, seed)
            if row["gate_id"] is not None:
                if row["gate_id"] in seen or row["gate_id"] not in r1_by_id:
                    raise ContractError("rung-one gate identity is duplicated or unknown")
                seen.add(row["gate_id"])
                _validate_gate_decision_row(row, r1_by_id[row["gate_id"]], seed)
        if seen != set(r1_by_id):
            raise ContractError("rung-one gate closure differs")
        for registry_row in r1_registry:
            row = next(item for item in rows if item["gate_id"] == registry_row["gate_id"])
            decisions.append({"construction_seed": seed, "gate_id": row["gate_id"], "gate_pass": row["gate_pass"]})
    r2_registry = payload["gates"]["rung_two_registry"]
    r2_path = run_root / "rung2" / "83" / "evaluation.jsonl"
    raw_lines = r2_path.read_bytes().splitlines(keepends=True)
    r2_rows = [json.loads(line[:-1].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))) for line in raw_lines if line.endswith(b"\n")]
    if len(raw_lines) != 2 or len(r2_rows) != 2 or any(canonical_json_bytes(row) + b"\n" != raw for row, raw in zip(r2_rows, raw_lines)):
        raise ContractError("rung-two evaluation serialization or cardinality differs")
    if tuple((row.get("condition"), row.get("metric"), row.get("stratum")) for row in r2_rows) != (("intact", "answer_accuracy", "all"), ("recurrent_knockout", "answer_accuracy", "all")):
        raise ContractError("rung-two evaluation row order differs")
    for row, registry_row in zip(r2_rows, r2_registry[:2]):
        _validate_evaluation_row_common(row, payload, run_root, 2, 83)
        _validate_gate_decision_row(row, registry_row, 83)
        decisions.append({"construction_seed": 83, "gate_id": row["gate_id"], "gate_pass": row["gate_pass"]})
    stats = _canonical_json_artifact(run_root / "rung2" / "83" / "gate_stats.json")
    stats_schema = payload["artifacts"]["schemas"]["rung_two_gate_stats"]
    validate_exact_keys(stats, stats_schema["exact_keys"], "rung-two gate statistics")
    if stats["schema_version"] != SCHEMA_VERSION or stats["run_id"] != run_root.name or stats["rung"] != 2 or stats["claim_seed"] != 83 or stats["construction_seed"] != 83 or stats["checkpoint_sha256"] != r2_rows[0]["checkpoint_sha256"]:
        raise ContractError("rung-two gate statistics identity differs")
    if [record.get("condition") for record in stats["conditions"]] != stats_schema["condition_order"] or len(stats["conditions"]) != 2:
        raise ContractError("rung-two gate statistics condition order differs")
    expected_record_order = [(block, head) for block in (1, 2, 3, 5, 6, 7) for head in range(4)]
    for condition_record, registry_row in zip(stats["conditions"], r2_registry[2:]):
        validate_exact_keys(condition_record, stats_schema["condition_record_exact_keys"], "rung-two gate condition")
        if condition_record["gate_id"] != registry_row["gate_id"] or condition_record["condition"] != registry_row["condition"]:
            raise ContractError("rung-two gate statistic registry differs")
        for field in ("gate_operator", "gate_threshold", "gate_threshold_count", "gate_threshold_unit"):
            if condition_record[field] != registry_row[field]:
                raise ContractError("rung-two gate statistic threshold differs")
        if [(row.get("block"), row.get("head")) for row in condition_record["records"]] != expected_record_order:
            raise ContractError("rung-two gate statistic record order differs")
        for record in condition_record["records"]:
            validate_exact_keys(record, stats_schema["record_exact_keys"], "rung-two gate statistic record")
            if record["count"] != 512 * 512 or record["nonfinite_count"] != 0:
                raise ContractError("rung-two gate statistic population differs")
            for field in ("mean", "population_std", "min", "max"):
                value = record[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ContractError("rung-two gate statistic numeric value differs")
            if record["population_std"] < 0 or record["min"] > record["mean"] or record["mean"] > record["max"]:
                raise ContractError("rung-two gate statistic numeric ordering differs")
        aggregate = condition_record["aggregate"]
        validate_exact_keys(aggregate, stats_schema["aggregate_exact_keys"], "rung-two gate statistic aggregate")
        merged = _merged_summary_values(condition_record["records"])
        if aggregate["block"] is not None or aggregate["head"] is not None or aggregate["count"] != merged["count"] or aggregate["nonfinite_count"] != merged["nonfinite_count"]:
            raise ContractError("rung-two gate statistic aggregate differs")
        for field in ("mean", "population_std", "min", "max"):
            if not math.isclose(float(aggregate[field]), float(merged[field]), rel_tol=1e-12, abs_tol=1e-12):
                raise ContractError("rung-two gate statistic aggregate reduction differs")
        if condition_record["gate_pass"] is not (aggregate["nonfinite_count"] == 0):
            raise ContractError("rung-two gate statistic decision differs")
        gate_row = {**registry_row, "numerator": aggregate["nonfinite_count"], "estimate": aggregate["nonfinite_count"], "gate_pass": condition_record["gate_pass"]}
        _validate_gate_decision_row(gate_row, registry_row, 83)
        decisions.append({"construction_seed": 83, "gate_id": registry_row["gate_id"], "gate_pass": condition_record["gate_pass"]})
    expected_order = [(seed, row["gate_id"]) for seed in RUNG_ONE_SEEDS for row in r1_registry] + [(83, row["gate_id"]) for row in r2_registry]
    if [(row["construction_seed"], row["gate_id"]) for row in decisions] != expected_order or len(decisions) != 124:
        raise ContractError("gate package order or cardinality differs")
    return decisions


def _gate_sections_from_decisions(ordered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(ordered) != 124:
        raise ContractError("summary gate cardinality differs")
    identities = []
    for row in ordered:
        validate_exact_keys(row, ("construction_seed", "gate_id", "gate_pass"), "summary gate decision")
        if type(row["construction_seed"]) is not int or not isinstance(row["gate_id"], str) or type(row["gate_pass"]) is not bool:
            raise ContractError("summary gate decision differs")
        identities.append((row["construction_seed"], row["gate_id"]))
    if len(set(identities)) != 124:
        raise ContractError("summary gate identity differs")
    passed_gates = [{"construction_seed": row["construction_seed"], "gate_id": row["gate_id"]} for row in ordered if row["gate_pass"]]
    failed_gates = [{"construction_seed": row["construction_seed"], "gate_id": row["gate_id"]} for row in ordered if not row["gate_pass"]]
    per_seed = []
    for seed in RUNG_ONE_SEEDS:
        seed_rows = [row for row in ordered if row["construction_seed"] == seed]
        if len(seed_rows) != 24:
            raise ContractError("summary rung-one seed closure differs")
        seed_passed = [row["gate_id"] for row in seed_rows if row["gate_pass"]]
        seed_failed = [row["gate_id"] for row in seed_rows if not row["gate_pass"]]
        if set(seed_passed).intersection(seed_failed) or len(seed_passed) + len(seed_failed) != 24:
            raise ContractError("summary rung-one complement differs")
        per_seed.append({"construction_seed": seed, "total_gates": 24, "passed_gates": seed_passed, "failed_gates": seed_failed, "gate_pass": len(seed_failed) == 0})
    r2_rows = [row for row in ordered if row["construction_seed"] == RUNG_TWO_SEED]
    if len(r2_rows) != 4:
        raise ContractError("summary rung-two closure differs")
    r2_passed = [row["gate_id"] for row in r2_rows if row["gate_pass"]]
    r2_failed = [row["gate_id"] for row in r2_rows if not row["gate_pass"]]
    if set(r2_passed).intersection(r2_failed) or len(r2_passed) + len(r2_failed) != 4:
        raise ContractError("summary rung-two complement differs")
    rung_two = {"construction_seed": RUNG_TWO_SEED, "total_gates": 4, "passed_gates": r2_passed, "failed_gates": r2_failed, "gate_pass": len(r2_failed) == 0}
    return {"passed_gates": passed_gates, "failed_gates": failed_gates, "per_seed": per_seed, "rung_two": rung_two}


def summary_from_gate_decisions(run_id: str, artifact_manifest_sha256: str, ordered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if RUN_ID_PATTERN.fullmatch(run_id) is None or re.fullmatch(r"[0-9a-f]{64}", artifact_manifest_sha256) is None:
        raise ContractError("summary identity differs")
    sections = _gate_sections_from_decisions(ordered)
    combined_pass = len(sections["failed_gates"]) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "positive" if combined_pass else "negative",
        "total_gates": 124,
        **sections,
        "combined_pass": combined_pass,
        "artifact_manifest_sha256": artifact_manifest_sha256,
    }


def validate_summary_contract(summary: Mapping[str, Any], ordered: Sequence[Mapping[str, Any]]) -> None:
    validate_exact_keys(summary, ("schema_version", "run_id", "status", "total_gates", "passed_gates", "failed_gates", "per_seed", "rung_two", "combined_pass", "artifact_manifest_sha256"), "summary artifact")
    expected = summary_from_gate_decisions(summary["run_id"], summary["artifact_manifest_sha256"], ordered)
    if dict(summary) != expected:
        raise ContractError("summary derivation differs")


def _gate_summary(run_root: Path, payload: Mapping[str, Any], runtime: RuntimeModules | None = None) -> dict[str, Any]:
    try:
        validate_claim_artifact_package(run_root, payload, runtime)
        ordered = validate_gate_input_package(run_root, payload)
    except HardAbort:
        raise
    except ContractError as exc:
        raise HardAbort("artifact_inconsistency", {"surface": "gate_input_package"}) from exc
    try:
        return _gate_sections_from_decisions(ordered)
    except ContractError as exc:
        raise HardAbort("artifact_inconsistency", {"surface": "gate_summary"}) from exc


def _final_signal_guard(signals: SignalController, stage: str) -> None:
    if not isinstance(stage, str) or not stage:
        raise ContractError("final frozen guard stage differs")
    if signals.pending_signal is not None:
        raise HardAbort("signal_or_interruption", {"stage": stage})


def final_frozen_guard(
    run_root: Path,
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    stage: str,
) -> None:
    _final_signal_guard(signals, stage)
    _verify_active_frozen_hashes(run_root, anchors)


def final_claim_guard(
    run_root: Path,
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    claim_start_monotonic_ns: int,
    stage: str,
) -> None:
    _final_signal_guard(signals, stage)
    if (time.monotonic_ns() - claim_start_monotonic_ns) / 1e9 > 1200:
        raise HardAbort("claim_elapsed_time", {"stage": stage})
    _verify_active_frozen_hashes(run_root, anchors)


def _canonical_json_artifact(path: Path) -> Any:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError("canonical JSON artifact is invalid") from exc
    if canonical_json_bytes(value) != raw:
        raise ContractError("canonical JSON artifact differs")
    return value


def _canonical_jsonl_records(path: Path) -> list[dict[str, Any]]:
    raw_lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for raw in raw_lines:
        if not raw.endswith(b"\n"):
            raise ContractError("canonical JSONL lacks terminal newline")
        record = json.loads(raw[:-1].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if canonical_json_bytes(record) + b"\n" != raw:
            raise ContractError("canonical JSONL record differs")
        records.append(record)
    return records


def _canonical_gzip_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                raise ContractError("canonical gzip JSONL lacks terminal newline")
            record = json.loads(raw[:-1].decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if canonical_json_bytes(record) + b"\n" != raw:
                raise ContractError("canonical gzip JSONL record differs")
            records.append(record)
    return records


def _load_claim_torch_artifact(path: Path, torch: Any, surface: str) -> Any:
    target = validate_real_regular_file(path)
    try:
        artifact = torch.load(target, map_location="cpu", weights_only=False)
    except BaseException as exc:
        raise HardAbort("endpoint_inconsistency", {"surface": surface}) from exc
    if target.read_bytes() != _torch_artifact_bytes(artifact, torch):
        raise HardAbort("endpoint_inconsistency", {"surface": f"{surface}.canonical_bytes"})
    return artifact


def _load_claim_endpoints(run_root: Path, rung: int, seed: int, torch: Any) -> dict[str, dict[str, Any]]:
    if rung == 1:
        endpoint_specs = RUNG_ONE_STAGE_ENDPOINTS
        checkpoint_root = run_root / "rung1" / str(seed) / "checkpoints"
    elif rung == 2 and seed == RUNG_TWO_SEED:
        endpoint_specs = {"rung_two": ("final_last.pt", "rung_two", "rung2", 1536)}
        checkpoint_root = run_root / "rung2" / "83" / "checkpoints"
    else:
        raise ContractError("endpoint rung or seed differs")
    endpoints = {}
    for stage, (filename, model_name, checkpoint_stage, updates) in endpoint_specs.items():
        path = checkpoint_root / filename
        checkpoint = _load_claim_torch_artifact(path, torch, f"checkpoint.{rung}.{seed}.{stage}")
        _validate_checkpoint(
            checkpoint,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_root.name,
                "rung": rung,
                "construction_seed": seed,
                "model": model_name,
                "stage": checkpoint_stage,
                "completed_update": updates,
            },
            torch,
        )
        endpoints[stage] = {"path": path, "sha256": sha256_file(path), "checkpoint": checkpoint}
    return endpoints


def _checkpoint_by_condition(rung: int, endpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    if rung == 1:
        digest_by_model = {
            "selected": endpoints["joint"]["sha256"],
            "local": endpoints["joint"]["sha256"],
            "donor": endpoints["donor"]["sha256"],
            "clone": endpoints["joint"]["sha256"],
            "dense": endpoints["dense_continuation"]["sha256"],
        }
        return {condition: (RUNG_ONE_MODEL_BY_CONDITION[condition], digest_by_model[RUNG_ONE_MODEL_BY_CONDITION[condition]]) for condition in RUNG_ONE_CONDITIONS}
    if rung == 2:
        return {condition: ("rung_two", endpoints["rung_two"]["sha256"]) for condition in RUNG_TWO_CONDITIONS}
    raise ContractError("checkpoint condition rung differs")


def _load_evaluation_evidence(run_root: Path, rung: int, seed: int, torch: Any) -> tuple[Mapping[str, Any], str]:
    if rung == 1:
        expected_seed = 400000 + seed
        path = run_root / "data" / f"r1_eval_{expected_seed}.pt"
        schema_name = "evaluation_data_rung_one"
        regenerated = generate_rung_one_batch(expected_seed, 512, torch)
    elif rung == 2 and seed == RUNG_TWO_SEED:
        expected_seed = 1000083
        path = run_root / "data" / "r2_eval_1000083.pt"
        schema_name = "evaluation_data_rung_two"
        regenerated = generate_rung_two_batch(expected_seed, 512, torch)
    else:
        raise ContractError("evaluation evidence rung or seed differs")
    artifact = _load_claim_torch_artifact(path, torch, f"evaluation_data.{rung}.{seed}")
    validate_exact_keys(artifact, ("seed", "payload", "payload_sha256"), "evaluation data artifact")
    if artifact["seed"] != expected_seed or artifact["payload"] != regenerated or artifact["payload_sha256"] != canonical_json_sha256(regenerated):
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": schema_name})
    return artifact["payload"], sha256_file(path)


def _validate_rung_one_data_artifacts(run_root: Path, seed: int, evaluation_payload: Mapping[str, Any], routing_evidence: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    random_path = run_root / "data" / f"r1_random_routes_{seed}.pt"
    random_artifact = _load_claim_torch_artifact(random_path, torch, f"random_routes.{seed}")
    validate_exact_keys(random_artifact, ("seed", "routes", "payload", "payload_sha256"), "random route artifact")
    random_payload = generate_random_routes(500000 + seed, 512, torch)
    expected_random = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    expected_random[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long)
    if random_artifact["seed"] != 500000 + seed or random_artifact["payload"] != random_payload or random_artifact["payload_sha256"] != canonical_json_sha256(random_payload) or not torch.equal(random_artifact["routes"], expected_random):
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "random_routes"})
    acquisition_rows = [row for row in routing_evidence["records"] if row["phase"] == "route_acquisition" and row["row_kind"] == "query_example" and row["block"] == 4]
    if [row["example_index"] for row in acquisition_rows] != list(range(512)):
        raise ContractError("route-acquisition evidence closure differs")
    raw_query = [row["raw_remote_ids"] for row in acquisition_rows]
    source = list(evaluation_payload["required_source"])
    exclusion_path = run_root / "data" / f"r1_source_exclusion_{seed}.pt"
    exclusion_artifact = _load_claim_torch_artifact(exclusion_path, torch, f"source_exclusion.{seed}")
    validate_exact_keys(exclusion_artifact, ("seed", "raw_query_routes", "required_source", "routes", "payload", "payload_sha256"), "source exclusion artifact")
    exclusion_payload = generate_source_exclusion_routes(510000 + seed, raw_query, source, torch)
    expected_raw = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    expected_raw[:, 126, 0] = torch.tensor(raw_query, dtype=torch.long)
    expected_routes = torch.full_like(expected_raw, -1)
    expected_routes[:, 126, 0] = torch.tensor(exclusion_payload["routes"], dtype=torch.long)
    if exclusion_path.read_bytes() != _torch_artifact_bytes(exclusion_artifact, torch) or exclusion_artifact["seed"] != 510000 + seed or exclusion_artifact["payload"] != exclusion_payload or exclusion_artifact["payload_sha256"] != canonical_json_sha256(exclusion_payload) or not torch.equal(exclusion_artifact["raw_query_routes"], expected_raw) or not torch.equal(exclusion_artifact["required_source"], torch.tensor(source, dtype=torch.long)) or not torch.equal(exclusion_artifact["routes"], expected_routes):
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "source_exclusion"})
    excluded_queries = routing_evidence["query_by_condition"]["required_source_excluded"]
    if any(query["effective_remote_ids"] != exclusion_payload["routes"][query["example_index"]] or query["original_source_hit"] is not False for query in excluded_queries):
        raise ContractError("source-exclusion routing evidence differs")
    condition = list(evaluation_payload["condition"])
    foreign = []
    for start in range(0, 512, 32):
        values = condition[start : start + 32]
        foreign.extend(values[-1:] + values[:-1])
    same = sum(left == right for left, right in zip(condition, foreign))
    return {
        "evaluation_sha256": sha256_file(run_root / "data" / f"r1_eval_{400000 + seed}.pt"),
        "random_sha256": sha256_file(random_path),
        "source_exclusion_sha256": sha256_file(exclusion_path),
        "same_condition": same,
        "changed_condition": 512 - same,
        "route_acquisition_examples": len(acquisition_rows),
        "source_exclusion_examples": len(excluded_queries),
        "postcheckpoint_assertions": len(acquisition_rows) == 512 and len(excluded_queries) == 512,
    }


def _validate_endpoint_train_closure(endpoints: Mapping[str, Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]]) -> None:
    for stage, endpoint in endpoints.items():
        rows = [row for row in train_rows if row["stage"] == stage]
        if not rows:
            raise ContractError("endpoint stage has no training evidence")
        last = rows[-1]
        checkpoint = endpoint["checkpoint"]
        if checkpoint["last_attempt_id"] != last["attempt_id"] or checkpoint["final_batch_sha256"] != last["batch_sha256"] or checkpoint["completed_update"] != last["logical_update"]:
            raise HardAbort("endpoint_inconsistency", {"stage": stage, "surface": "checkpoint_training_tail"})


def _validate_train_artifact(run_root: Path, payload: Mapping[str, Any], seed_root: Path, rung: int, seed: int, torch: Any | None = None) -> list[dict[str, Any]]:
    ledger = _read_jsonl(seed_root / "attempts.jsonl")
    validate_attempt_sequence(ledger, require_complete=True)
    rows = _canonical_jsonl_records(seed_root / "train.jsonl")
    if len(rows) * 2 != len(ledger):
        raise ContractError("train artifact cardinality differs from attempt ledger")
    keys = payload["artifacts"]["schemas"]["train_row"]["exact_keys"]
    for index, row in enumerate(rows):
        validate_exact_keys(row, keys, "train row")
        started = ledger[2 * index]
        completed = ledger[2 * index + 1]
        expected_pairs = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "model": started["model"],
            "stage": started["stage"],
            "logical_update": started["logical_update"],
            "attempt_id": started["attempt_id"],
            "started_event_sequence": started["event_sequence"],
            "completed_event_sequence": completed["event_sequence"],
            "batch_sha256": started["batch_sha256"],
            "examples": started["examples"],
            "token_positions": started["token_positions"],
        }
        if any(row[field] != value for field, value in expected_pairs.items()):
            raise ContractError("train row does not match attempt pair")
        metrics = completed["metrics"]
        for field in ("learning_rates", "component_losses", "total_loss", "gradient_norm", "clip_result", "raw_overflow_count", "max_bucket_load", "elapsed_seconds", "finite"):
            if row[field] != metrics[field]:
                raise ContractError("train row metric differs from completed event")
        if row["learning_rates"] != sorted(row["learning_rates"], key=lambda value: value["parameter_group"]):
            raise ContractError("train learning-rate order differs")
        if row["logical_update"] == 1:
            if row["first_batch_sha256"] != row["batch_sha256"]:
                raise ContractError("train first-batch digest differs")
        elif row["first_batch_sha256"] is not None:
            raise ContractError("train first-batch digest null contract differs")
        if any(isinstance(row[field], bool) or not isinstance(row[field], (int, float)) or not math.isfinite(float(row[field])) for field in ("total_loss", "gradient_norm", "elapsed_seconds")):
            raise ContractError("train row nonfinite metric differs")
    if rung == 1:
        stage_specs = tuple((stage, spec[1], spec[3], {"donor": 100000, "router_only": 200000, "joint": 300000, "dense_base": 100000, "dense_continuation": 300000}[stage] + seed) for stage, spec in RUNG_ONE_STAGE_ENDPOINTS.items())
    elif rung == 2:
        stage_specs = (("rung_two", "rung_two", 1536, 900083),)
    else:
        raise ContractError("train artifact rung differs")
    expected_order = [(stage, logical_update) for stage, _, updates, _ in stage_specs for logical_update in range(1, updates + 1)]
    if [(row["stage"], row["logical_update"]) for row in rows] != expected_order:
        raise ContractError("train artifact stage or logical-update closure differs")
    torch_module = _torch_module() if torch is None else torch
    for stage, model_name, _, generator_seed in stage_specs:
        stage_rows = [row for row in rows if row["stage"] == stage]
        if any(row["model"] != model_name for row in stage_rows):
            raise ContractError("train artifact stage model differs")
        generator = torch_module.Generator(device="cpu")
        generator.manual_seed(generator_seed)
        first = _continuous_rung_two_batch(generator, 8, torch_module) if rung == 2 else _continuous_rung_one_batch(generator, 16, torch_module)
        second = _continuous_rung_two_batch(generator, 8, torch_module) if rung == 2 else _continuous_rung_one_batch(generator, 16, torch_module)
        first_hash = _batch_payload_hash(first, stage)
        second_hash = _batch_payload_hash(second, stage)
        if first_hash == second_hash or stage_rows[0]["batch_sha256"] != first_hash or stage_rows[0]["first_batch_sha256"] != first_hash:
            raise HardAbort("endpoint_inconsistency", {"seed": seed, "stage": stage, "surface": "continuous_data_stream"})
    return rows


def _validate_prediction_artifact(
    run_root: Path,
    payload: Mapping[str, Any],
    seed_root: Path,
    rung: int,
    seed: int,
    checkpoint_by_condition: Mapping[str, tuple[str, str]],
    evaluation_payload: Mapping[str, Any],
    eval_data_sha256: str,
) -> dict[str, Any]:
    records = _canonical_gzip_records(seed_root / "predictions.jsonl.gz")
    conditions = payload["stages"]["rung_one"]["evaluation_arm_order"] if rung == 1 else payload["stages"]["rung_two"]["evaluation_order"]
    if tuple(conditions) != (RUNG_ONE_CONDITIONS if rung == 1 else RUNG_TWO_CONDITIONS) or tuple(checkpoint_by_condition) != tuple(conditions) or re.fullmatch(r"[0-9a-f]{64}", eval_data_sha256 or "") is None:
        raise ContractError("prediction evidence registry differs")
    if len(records) != len(conditions) * 512:
        raise ContractError("prediction artifact cardinality differs")
    keys = payload["artifacts"]["schemas"]["prediction_row"]["exact_keys"]
    populations: dict[tuple[str, str], list[int]] = {}
    by_condition: dict[str, list[dict[str, Any]]] = {}
    if rung == 1:
        foreign_conditions = []
        for start in range(0, 512, 32):
            values = evaluation_payload["condition"][start : start + 32]
            foreign_conditions.extend(values[-1:] + values[:-1])
    for condition_index, condition in enumerate(conditions):
        condition_records = records[condition_index * 512 : (condition_index + 1) * 512]
        by_condition[condition] = condition_records
        if [record.get("example_index") for record in condition_records] != list(range(512)):
            raise ContractError("prediction example order differs")
        for record in condition_records:
            validate_exact_keys(record, keys, "prediction row")
            if record["schema_version"] != SCHEMA_VERSION or record["run_id"] != run_root.name or record["rung"] != rung or record["claim_seed"] != seed or record["construction_seed"] != seed or record["condition"] != condition or type(record["correct"]) is not bool or record["correct"] is not (record["prediction"] == record["target"]):
                raise ContractError("prediction row identity or correctness differs")
            if record["checkpoint_sha256"] != checkpoint_by_condition[condition][1] or type(record["prediction"]) is not int or record["prediction"] < 0:
                raise ContractError("prediction endpoint identity differs")
            index = record["example_index"]
            if rung == 1:
                original_condition = int(evaluation_payload["condition"][index])
                original_source = int(evaluation_payload["required_source"][index])
                target = int(evaluation_payload["targets"][index])
                if record["original_condition"] != original_condition or record["original_source"] != original_source or record["target"] != target:
                    raise ContractError("prediction evaluation-data identity differs")
                if condition == "carry_shuffle":
                    if any(record[field] is None for field in ("foreign_condition", "foreign_source", "foreign_source_hit")):
                        raise ContractError("carry-shuffle prediction fields differ")
                    foreign_condition = int(foreign_conditions[index])
                    foreign_source = int(evaluation_payload["rule_blocks"][index][foreign_condition])
                    if record["foreign_condition"] != foreign_condition or record["foreign_source"] != foreign_source:
                        raise ContractError("carry-shuffle prediction provenance differs")
                    expected_stratum = "same_condition" if record["original_condition"] == record["foreign_condition"] else "changed_condition"
                    if record["condition_stratum"] != expected_stratum:
                        raise ContractError("carry-shuffle prediction stratum differs")
                elif any(record[field] is not None for field in ("foreign_condition", "foreign_source", "foreign_source_hit")) or record["condition_stratum"] != "not_applicable":
                    raise ContractError("non-shuffle prediction fields differ")
                if condition == "dense_causal":
                    if record["original_source_hit"] is not None:
                        raise ContractError("dense prediction source-hit field differs")
                elif type(record["original_source_hit"]) is not bool:
                    raise ContractError("routed prediction source-hit field differs")
            else:
                expected_target = int(evaluation_payload["targets"][index][510])
                if record["target"] != expected_target or any(record[field] is not None for field in ("original_condition", "foreign_condition", "original_source", "foreign_source", "original_source_hit", "foreign_source_hit")) or record["condition_stratum"] != "not_applicable":
                    raise ContractError("rung-two prediction data or route fields differ")
        populations[(condition, "all")] = list(range(512))
        if rung == 1 and condition != "dense_causal":
            populations[(condition, "source_hit")] = [record["example_index"] for record in condition_records if record["original_source_hit"]]
            populations[(condition, "source_miss")] = [record["example_index"] for record in condition_records if not record["original_source_hit"]]
        if rung == 1 and condition == "carry_shuffle":
            populations[(condition, "changed_condition")] = [record["example_index"] for record in condition_records if record["condition_stratum"] == "changed_condition"]
            populations[(condition, "same_condition")] = [record["example_index"] for record in condition_records if record["condition_stratum"] == "same_condition"]
    return {"records": records, "by_condition": by_condition, "populations": populations, "eval_data_sha256": eval_data_sha256}


def _expected_rung_one_forward_specs(checkpoint_by_condition: Mapping[str, tuple[str, str]]) -> list[dict[str, Any]]:
    specs = []
    training = (
        ("donor", "all_eligible_donor", 1024, (0, 4)),
        ("router_only", "selected", 768, (0, 4)),
        ("joint", "selected", 512, (0, 4)),
        ("dense_base", "dense_causal", 1024, (0,)),
        ("dense_continuation", "dense_causal", 512, (0,)),
    )
    for stage, model, updates, blocks in training:
        for logical_update in range(1, updates + 1):
            specs.append({"phase": "training", "model": model, "stage": stage, "condition": None, "logical_update": logical_update, "batch_index": None, "example_start": 0, "batch_size": 16, "blocks": blocks, "checkpoint_sha256": None})
    final_checkpoint = checkpoint_by_condition["intact"][1]
    for batch_index in range(16):
        specs.append({"phase": "route_acquisition", "model": "selected", "stage": None, "condition": None, "logical_update": None, "batch_index": batch_index, "example_start": batch_index * 32, "batch_size": 32, "blocks": (0, 4), "checkpoint_sha256": final_checkpoint})
    for condition in RUNG_ONE_CONDITIONS:
        model = RUNG_ONE_MODEL_BY_CONDITION[condition]
        blocks = (0,) if condition == "dense_causal" else (0, 4)
        for batch_index in range(16):
            specs.append({"phase": "evaluation", "model": model, "stage": None, "condition": condition, "logical_update": None, "batch_index": batch_index, "example_start": batch_index * 32, "batch_size": 32, "blocks": blocks, "checkpoint_sha256": checkpoint_by_condition[condition][1]})
    return specs


def _route_width(model: str, block: int) -> int:
    if block == 0:
        return 0
    widths = {"selected": 2, "local": 0, "donor": 15, "clone": 15, "all_eligible_donor": 15}
    if block != 4 or model not in widths:
        raise ContractError("routing model or block width differs")
    return widths[model]


def _validate_route_id_list(values: Any, width: int, surface: str) -> None:
    if not isinstance(values, list) or len(values) != width or any(type(value) is not int or value < -1 or value > 14 for value in values):
        raise ContractError(f"{surface} route IDs differ")
    valid = [value for value in values if value >= 0]
    if len(valid) != len(set(valid)):
        raise ContractError(f"{surface} route IDs are duplicated")


def _validate_routing_row_nullability(row: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    row_kind = row["row_kind"]
    phase = row["phase"]
    if row_kind not in schema["row_kind_null_fields"] or phase not in schema["phase_enum"]:
        raise ContractError("routing row kind or phase differs")
    if phase == "evaluation" and row_kind == "query_example":
        suffix = "carry_shuffle" if row["condition"] == "carry_shuffle" else "non_carry_shuffle"
        phase_key = f"{phase}.{row_kind}_{suffix}"
    else:
        phase_key = f"{phase}.{row_kind}"
    phase_fields = schema["phase_and_row_kind_additional_null_fields"].get(phase_key)
    if phase_fields is None:
        raise ContractError("routing phase nullability registry differs")
    expected_null = set(schema["row_kind_null_fields"][row_kind]) | set(phase_fields)
    observed_null = {field for field in schema["exact_keys"] if row[field] is None}
    if observed_null != expected_null:
        raise ContractError("routing exhaustive nullability differs")


def _validate_artifact_histogram(
    records: Any,
    entry_keys: Sequence[str],
    key_name: str,
    count_name: str,
) -> tuple[int, int, int | None]:
    if not isinstance(records, list):
        raise ContractError("routing histogram list differs")
    prior = -1
    frequency_sum = 0
    weighted_sum = 0
    maximum = None
    for record in records:
        validate_exact_keys(record, entry_keys, "routing histogram entry")
        key = record[key_name]
        count = record[count_name]
        if type(key) is not int or type(count) is not int or key < 0 or count <= 0 or key <= prior:
            raise ContractError("routing histogram order or value differs")
        prior = key
        frequency_sum += count
        weighted_sum += key * count
        maximum = key
    return frequency_sum, weighted_sum, maximum


def _validate_routing_artifact(
    run_root: Path,
    payload: Mapping[str, Any],
    seed_root: Path,
    seed: int,
    evaluation_rows: Sequence[Mapping[str, Any]],
    prediction_evidence: Mapping[str, Any],
    checkpoint_by_condition: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    records = _canonical_gzip_records(seed_root / "routing.jsonl.gz")
    schema = payload["artifacts"]["schemas"]["routing_row"]
    keys = schema["exact_keys"]
    specs = _expected_rung_one_forward_specs(checkpoint_by_condition)
    by_sequence: dict[int, list[Mapping[str, Any]]] = {}
    prior_sequence = -1
    for row in records:
        validate_exact_keys(row, keys, "routing row")
        sequence = row["forward_sequence"]
        if type(sequence) is not int or sequence < prior_sequence:
            raise ContractError("routing forward sequence differs")
        prior_sequence = sequence
        by_sequence.setdefault(sequence, []).append(row)
    if tuple(by_sequence) != tuple(range(len(specs))):
        raise ContractError("routing forward sequence closure differs")
    overflow = 0
    maximum = 0
    query_by_condition: dict[str, list[Mapping[str, Any]]] = {condition: [] for condition in RUNG_ONE_CONDITIONS if condition != "dense_causal"}
    workspace_bytes_by_model = {"all_eligible_donor": 0, "selected": 0, "dense_causal": 0}
    workspace_count_by_model = {"all_eligible_donor": 0, "selected": 0, "dense_causal": 0}
    for sequence, spec in enumerate(specs):
        sequence_rows = by_sequence[sequence]
        expected_rows = sum(1 + spec["batch_size"] for _ in spec["blocks"])
        if len(sequence_rows) != expected_rows:
            raise ContractError("routing forward row cardinality differs")
        cursor = 0
        sequence_workspace = 0
        sequence_workspace_count = 0
        for block in spec["blocks"]:
            call = sequence_rows[cursor]
            cursor += 1
            identity_fields = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_root.name,
                "rung": 1,
                "claim_seed": seed,
                "construction_seed": seed,
                "row_kind": "call_summary",
                "phase": spec["phase"],
                "model": spec["model"],
                "stage": spec["stage"],
                "condition": spec["condition"],
                "logical_update": spec["logical_update"],
                "forward_sequence": sequence,
                "block": block,
                "batch_index": spec["batch_index"],
                "checkpoint_sha256": spec["checkpoint_sha256"],
            }
            if any(call[field] != value for field, value in identity_fields.items()):
                raise ContractError("routing call-summary identity differs")
            _validate_routing_row_nullability(call, schema)
            for field in ("addresses_probed", "posting_reads", "candidate_blocks", "overflow_count", "max_bucket_load", "route_workspace_bytes"):
                if type(call[field]) is not int or call[field] < 0:
                    raise ContractError("routing call-summary counter differs")
            width = _route_width(spec["model"], block)
            derived_search = _derived_routing_search_contract(spec["batch_size"], 128, 1, width, 16)
            expected_bypass = _expected_canonical_bypass_evidence(width, 128)
            if call["canonical_bypass_ids"] != expected_bypass:
                raise ContractError("routing canonical bypass evidence differs")
            bucket_count, indexed_blocks, observed_maximum = _validate_artifact_histogram(
                call["block_load_histogram"],
                schema["block_load_histogram_entry_exact_keys"],
                "load",
                "bucket_count",
            )
            search_rows, valid_postings, _ = _validate_artifact_histogram(
                call["valid_posting_histogram"],
                schema["valid_posting_histogram_entry_exact_keys"],
                "valid_posting_count",
                "search_row_count",
            )
            histogram_overflow = sum(
                max(record["load"] - 64, 0) * record["bucket_count"]
                for record in call["block_load_histogram"]
            )
            expected_search_rows = derived_search["search_rows"]
            if bucket_count != spec["batch_size"] * 16 or indexed_blocks != spec["batch_size"] * 16 or observed_maximum is None:
                raise ContractError("routing block-load histogram population differs")
            if call["max_bucket_load"] != observed_maximum or call["overflow_count"] != histogram_overflow:
                raise ContractError("routing block-load histogram reduction differs")
            if search_rows != expected_search_rows or bool(call["valid_posting_histogram"]) is not (expected_search_rows > 0):
                raise ContractError("routing valid-posting histogram geometry differs")
            if call["addresses_probed"] != search_rows * 4 or call["posting_reads"] != valid_postings or call["candidate_blocks"] != valid_postings:
                raise ContractError("routing valid-posting histogram reduction differs")
            if call["overflow_count"] != 0 or call["max_bucket_load"] > ROUTE_CAPACITY_BY_RUNG[1] or call["route_workspace_bytes"] != derived_search["workspace_bytes"]:
                raise ContractError("routing call-summary capacity or derived search contract differs")
            overflow += call["overflow_count"]
            maximum = max(maximum, call["max_bucket_load"])
            sequence_workspace += derived_search["workspace_bytes"]
            sequence_workspace_count += derived_search["workspace_count"]
            expected_examples = list(range(spec["example_start"], spec["example_start"] + spec["batch_size"]))
            queries = sequence_rows[cursor : cursor + spec["batch_size"]]
            cursor += spec["batch_size"]
            if [query.get("example_index") for query in queries] != expected_examples:
                raise ContractError("routing query-example order differs")
            for query in queries:
                expected_identity = {**identity_fields, "row_kind": "query_example", "example_index": query["example_index"]}
                if any(query[field] != value for field, value in expected_identity.items()):
                    raise ContractError("routing query-example identity differs")
                _validate_routing_row_nullability(query, schema)
                if query["query_position"] != 126 or query["local_block_ids"] != [15] or type(query["required_source"]) is not int or not 0 <= query["required_source"] <= 14:
                    raise ContractError("routing query-example provenance differs")
                _validate_route_id_list(query["raw_remote_ids"], width, "raw")
                if spec["phase"] == "route_acquisition":
                    pass
                else:
                    _validate_route_id_list(query["effective_remote_ids"], width, "effective")
                    if query["query_underfill_count"] != sum(value == -1 for value in query["effective_remote_ids"]) or query["original_source_hit"] is not (query["required_source"] in query["effective_remote_ids"]):
                        raise ContractError("routing query hit or underfill differs")
                    expected_intervention = spec["condition"] if spec["phase"] == "evaluation" else None
                    if query["intervention"] != expected_intervention:
                        raise ContractError("routing query intervention differs")
                    if spec["condition"] == "carry_shuffle":
                        if type(query["foreign_source"]) is not int or not 0 <= query["foreign_source"] <= 14 or query["foreign_source_hit"] is not (query["foreign_source"] in query["effective_remote_ids"]):
                            raise ContractError("routing foreign-source provenance differs")
                    elif query["foreign_source"] is not None or query["foreign_source_hit"] is not None:
                        raise ContractError("routing non-shuffle foreign fields differ")
                if spec["phase"] == "evaluation" and block == 4:
                    query_by_condition[spec["condition"]].append(query)
            if width == 15:
                query_bypass = expected_bypass[126]
                if any(query["effective_remote_ids"] != query_bypass["effective_remote_ids"] for query in queries):
                    raise ContractError("routing width-fifteen query bypass differs")
            elif width > 0:
                selected_query_ids = sum(sum(value >= 0 for value in query["raw_remote_ids"]) for query in queries)
                if selected_query_ids > call["candidate_blocks"]:
                    raise ContractError("routing selected query IDs exceed valid candidates")
        accounting_model = RUNG_ONE_ACCOUNTING_MODEL_BY_ROUTE_MODEL[{"all_eligible_donor": "donor", "dense_causal": "dense"}.get(spec["model"], spec["model"])]
        workspace_bytes_by_model[accounting_model] = max(workspace_bytes_by_model[accounting_model], sequence_workspace)
        workspace_count_by_model[accounting_model] = max(workspace_count_by_model[accounting_model], sequence_workspace_count)
    prediction_by_condition = prediction_evidence["by_condition"]
    underfill_by_condition = {}
    for condition, queries in query_by_condition.items():
        if [query["example_index"] for query in queries] != list(range(512)):
            raise ContractError("routing evaluation query closure differs")
        predictions = prediction_by_condition[condition]
        for query, prediction in zip(queries, predictions):
            if query["required_source"] != prediction["original_source"] or query["original_source_hit"] is not prediction["original_source_hit"] or query["foreign_source"] != prediction["foreign_source"] or query["foreign_source_hit"] is not prediction["foreign_source_hit"] or query["checkpoint_sha256"] != prediction["checkpoint_sha256"]:
                raise ContractError("routing and prediction evidence disagree")
        underfill_by_condition[condition] = sum(query["query_underfill_count"] for query in queries)
    underfill_by_condition["dense_causal"] = None
    overflow_rows = [row for row in evaluation_rows if row["metric"] == "route_overflow_count"]
    if len(overflow_rows) != 1 or overflow_rows[0]["numerator"] != overflow or overflow_rows[0]["overflow_count"] != overflow or overflow_rows[0]["max_bucket_load"] != maximum:
        raise ContractError("routing overflow total differs from evaluation")
    return {"overflow_count": overflow, "max_bucket_load": maximum, "underfill_by_condition": underfill_by_condition, "workspace_bytes_by_model": workspace_bytes_by_model, "workspace_count_by_model": workspace_count_by_model, "query_by_condition": query_by_condition, "records": records}


def _validate_evaluation_reconstruction(
    run_root: Path,
    payload: Mapping[str, Any],
    rung: int,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    prediction_evidence: Mapping[str, Any],
    checkpoint_by_condition: Mapping[str, tuple[str, str]],
    routing_evidence: Mapping[str, Any] | None,
) -> None:
    by_condition = prediction_evidence["by_condition"]
    populations = prediction_evidence["populations"]
    eval_data_sha256 = prediction_evidence["eval_data_sha256"]
    elapsed_by_condition: dict[str, float] = {}
    for row in rows:
        _validate_evaluation_row_common(row, payload, run_root, rung, seed)
        metric = row["metric"]
        condition = row["condition"]
        if metric in {"selected_mask_oracle_max_error", "route_overflow_count"}:
            if row["elapsed_seconds"] != 0.0:
                raise ContractError("evaluation aggregate elapsed derivation differs")
        elif condition in elapsed_by_condition and row["elapsed_seconds"] != elapsed_by_condition[condition]:
            raise ContractError("evaluation condition elapsed repetition differs")
        else:
            elapsed_by_condition[condition] = row["elapsed_seconds"]
        if metric in {"answer_accuracy", "original_source_hit_rate", "foreign_source_hit_rate"}:
            if condition not in by_condition:
                raise ContractError("evaluation condition lacks prediction evidence")
            indices = populations.get((condition, row["stratum"]))
            if indices is None:
                raise ContractError("evaluation stratum lacks prediction evidence")
            prediction_rows = by_condition[condition]
            value_field = {"answer_accuracy": "correct", "original_source_hit_rate": "original_source_hit", "foreign_source_hit_rate": "foreign_source_hit"}[metric]
            values = [prediction_rows[index][value_field] for index in indices]
            if any(type(value) is not bool for value in values):
                raise ContractError("evaluation binary evidence differs")
            numerator = sum(values)
            if row["numerator"] != numerator or row["denominator"] != len(indices) or row["population_sha256"] != _population_sha(indices):
                raise ContractError("evaluation reduction differs from predictions")
            if row["checkpoint_sha256"] != checkpoint_by_condition[condition][1] or row["eval_data_sha256"] != eval_data_sha256 or row["provenance_sha256s"] != [checkpoint_by_condition[condition][1], eval_data_sha256]:
                raise ContractError("evaluation endpoint provenance differs")
            if rung == 1 and condition != "dense_causal":
                if routing_evidence is None:
                    raise ContractError("rung-one evaluation lacks routing evidence")
                queries = routing_evidence["query_by_condition"][condition]
                expected_raw = [queries[index]["raw_remote_ids"] for index in indices]
                expected_effective = [queries[index]["effective_remote_ids"] for index in indices]
                if row["raw_remote_ids"] != expected_raw or row["effective_remote_ids"] != expected_effective:
                    raise ContractError("evaluation route arrays differ from routing evidence")
            elif row["raw_remote_ids"] is not None or row["effective_remote_ids"] is not None:
                raise ContractError("evaluation route-array nullability differs")
        elif metric == "query_underfill_count":
            if rung != 1 or routing_evidence is None or condition not in routing_evidence["underfill_by_condition"]:
                raise ContractError("evaluation underfill identity differs")
            expected = routing_evidence["underfill_by_condition"][condition]
            if expected is None or row["numerator"] != expected or row["estimate"] != expected or row["query_underfill_count"] != expected or row["checkpoint_sha256"] != checkpoint_by_condition[condition][1] or row["eval_data_sha256"] != eval_data_sha256:
                raise ContractError("evaluation underfill differs from routing evidence")
            if row["raw_remote_ids"] is not None or row["effective_remote_ids"] is not None:
                raise ContractError("evaluation underfill route arrays differ")
        elif metric == "route_overflow_count":
            if rung != 1 or routing_evidence is None:
                raise ContractError("evaluation overflow identity differs")
            routing_path = run_root / "rung1" / str(seed) / "routing.jsonl.gz"
            if row["numerator"] != routing_evidence["overflow_count"] or row["max_bucket_load"] != routing_evidence["max_bucket_load"] or row["provenance_sha256s"] != [sha256_file(routing_path)] or row["checkpoint_sha256"] is not None or row["eval_data_sha256"] is not None:
                raise ContractError("evaluation overflow differs from routing evidence")
        elif metric == "selected_mask_oracle_max_error":
            if rung != 1 or condition != "intact" or row["checkpoint_sha256"] is not None or row["eval_data_sha256"] is not None:
                raise ContractError("evaluation oracle identity differs")
        else:
            raise ContractError("evaluation metric reconstruction differs")


def _loaded_endpoint_parameters(stage: str, state: Mapping[str, Any], seed: int, runtime: RuntimeModules) -> dict[str, Any]:
    roles = {"donor": "all_eligible", "router_only": "selected", "joint": "selected", "dense_base": "dense", "dense_continuation": "dense", "rung_two": "rung_two"}
    if stage not in roles or not isinstance(state, Mapping) or type(seed) is not int:
        raise ContractError("gradient endpoint parameter identity differs")
    model, _, _ = _construct_seeded_model(roles[stage], seed, runtime)
    try:
        model.load_state_dict(state, strict=True)
    except BaseException as exc:
        raise ContractError("gradient endpoint parameter reconstruction failed") from exc
    parameters = dict(model.named_parameters())
    if set(parameters) != set(state) or any(_tensor_sha256(parameters[name]) != _tensor_sha256(state[name]) for name in parameters):
        raise ContractError("gradient endpoint parameter reconstruction differs")
    return parameters


def _validate_gradient_artifact(
    artifact: Mapping[str, Any],
    run_root: Path,
    rung: int,
    seed: int,
    expected_updates: Mapping[str, int],
    endpoints: Mapping[str, Mapping[str, Any]],
    runtime: RuntimeModules,
) -> list[dict[str, Any]]:
    validate_exact_keys(artifact, ("schema_version", "run_id", "rung", "claim_seed", "construction_seed", "records"), "gradient audit artifact")
    if artifact["schema_version"] != SCHEMA_VERSION or artifact["run_id"] != run_root.name or artifact["rung"] != rung or artifact["claim_seed"] != seed or artifact["construction_seed"] != seed:
        raise ContractError("gradient audit artifact identity differs")
    records = artifact["records"]
    if records != sorted(records, key=lambda record: (record["stage"], record["name"])):
        raise ContractError("gradient audit record order differs")
    validate_gradient_audit(records, expected_updates)
    for stage in expected_updates:
        stage_records = [record for record in records if record["stage"] == stage]
        state = endpoints[stage]["checkpoint"]["model_state_dict"]
        parameters = _loaded_endpoint_parameters(stage, state, seed, runtime)
        if {record["name"] for record in stage_records} != set(state) or len(stage_records) != len(state):
            raise ContractError("gradient audit endpoint parameter closure differs")
        for record in stage_records:
            tensor = state[record["name"]]
            if record["shape"] != list(tensor.shape) or record["end_sha256"] != _tensor_sha256(tensor):
                raise ContractError("gradient audit endpoint tensor evidence differs")
            try:
                category = runtime.model_module.parameter_category(record["name"], parameters[record["name"]])
            except BaseException as exc:
                raise ContractError("gradient audit parameter classification failed") from exc
            trainable, family, peak = _stage_membership(runtime.model_module, record["name"], stage)
            decay = _weight_decay_for_category(category) if trainable else None
            expected_group = f"{family}_{'decay' if decay else 'zero_decay'}" if trainable else None
            if record["category"] != category or record["requires_grad"] is not trainable or record["optimizer_member"] is not trainable or record["parameter_group"] != expected_group or record["peak_lr"] != peak or record["weight_decay"] != decay:
                raise ContractError("gradient audit stage membership differs")
    for later, earlier in (("joint", "router_only"), ("dense_continuation", "dense_base")):
        if later in expected_updates and earlier in expected_updates:
            earlier_end = {record["name"]: record["end_sha256"] for record in records if record["stage"] == earlier}
            later_start = {record["name"]: record["start_sha256"] for record in records if record["stage"] == later}
            if later_start != earlier_end:
                raise ContractError("gradient audit stage chain differs")
    return records


def _route_index_storage(batch_size: int, sequence_length: int, routed_blocks: int) -> tuple[int, int]:
    if type(batch_size) is not int or type(sequence_length) is not int or type(routed_blocks) is not int or batch_size <= 0 or sequence_length <= 0 or routed_blocks <= 0 or sequence_length % 8:
        raise ContractError("route index geometry differs")
    complete_blocks = sequence_length // 8
    feature_count = batch_size * complete_blocks * 16
    address_count = batch_size * complete_blocks
    posting_count = batch_size * 16 * 64
    count = routed_blocks * (feature_count + address_count + posting_count)
    size = routed_blocks * (feature_count * 4 + address_count * 8 + posting_count * 8)
    return count, size


def _validate_accounting_evidence(
    models: Sequence[Mapping[str, Any]],
    rung: int,
    endpoints: Mapping[str, Mapping[str, Any]],
    gradient_records: Sequence[Mapping[str, Any]],
    routing_evidence: Mapping[str, Any] | None,
) -> None:
    if rung == 1:
        endpoint_by_model = {"all_eligible_donor": "donor", "selected": "joint", "dense_causal": "dense_continuation"}
        route_blocks_by_model = {"all_eligible_donor": 2, "selected": 2, "dense_causal": 1}
        sequence_length = 128
    elif rung == 2:
        endpoint_by_model = {"rung_two": "rung_two"}
        route_blocks_by_model = {"rung_two": 2}
        sequence_length = 512
    else:
        raise ContractError("accounting evidence rung differs")
    if {model["model"] for model in models} != set(endpoint_by_model) or len(models) != len(endpoint_by_model):
        raise ContractError("accounting evidence model closure differs")
    for model in models:
        model_name = model["model"]
        endpoint = endpoints[endpoint_by_model[model_name]]["checkpoint"]
        state = endpoint["model_state_dict"]
        audits = [record for record in gradient_records if record["model"] == model_name]
        names = set(state)
        if not audits or {record["name"] for record in audits} != names:
            raise ContractError("accounting gradient parameter closure differs")
        classification_by_name = {name: {record["classification"] for record in audits if record["name"] == name} for name in names}
        learned = {name for name, values in classification_by_name.items() if "learned_with_evidence" in values}
        serialized = {name for name, values in classification_by_name.items() if name not in learned and values & {"trainable_but_no_gradient", "updated_only_by_decay"}}
        inactive = names - learned - serialized
        expected = {}
        for category, selected in (("active_learned_parameter", learned), ("serialized_without_gradient", serialized), ("inactive_parameter", inactive)):
            expected[category] = (sum(int(state[name].numel()) for name in selected), sum(int(state[name].numel() * state[name].element_size()) for name in selected))
        expected["registered_buffer"] = (0, 0)
        dynamic_count = len(RECURRENT_BLOCKS) * 32 * 4 * 16 * 16
        expected["dynamic_recurrent_state"] = (dynamic_count, dynamic_count * 4)
        expected["route_index_storage"] = _route_index_storage(32, sequence_length, route_blocks_by_model[model_name])
        if rung == 1:
            if routing_evidence is None:
                raise ContractError("accounting routing evidence is absent")
            expected["routing_workspace"] = (routing_evidence["workspace_count_by_model"][model_name], routing_evidence["workspace_bytes_by_model"][model_name])
        else:
            expected["routing_workspace"] = (0, 0)
        expected["optimizer_state"] = _tensor_tree_storage(endpoint["optimizer_state_dict"], _torch_module())
        observed = {entry["category"]: (entry["count"], entry["bytes"]) for entry in model["entries"]}
        if observed != expected:
            raise ContractError("accounting category evidence differs")


def _validate_selected_oracle_evidence(
    run_root: Path,
    payload: Mapping[str, Any],
    seed: int,
    evaluation_rows: Sequence[Mapping[str, Any]],
    runtime: RuntimeModules,
) -> dict[str, Any]:
    seed_root = run_root / "rung1" / str(seed)
    manifest_path = seed_root / "selected_canonical_state_manifest.json"
    manifest = _canonical_json_artifact(manifest_path)
    validate_exact_keys(manifest, payload["artifacts"]["schemas"]["selected_canonical_state_manifest"]["exact_keys"], "selected canonical state manifest")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["run_id"] != run_root.name or manifest["construction_seed"] != seed or manifest["role"] != "selected_canonical" or manifest["state_tensors"] != sorted(manifest["state_tensors"], key=lambda row: row["name"]) or manifest["state_sha256"] != canonical_json_sha256(manifest["state_tensors"]):
        raise ContractError("selected canonical state manifest differs")
    for tensor_record in manifest["state_tensors"]:
        validate_exact_keys(tensor_record, payload["artifacts"]["schemas"]["selected_canonical_state_manifest"]["state_tensor_record_exact_keys"], "selected canonical state tensor")
        if not isinstance(tensor_record["name"], str) or not tensor_record["name"] or not isinstance(tensor_record["dtype"], str) or not tensor_record["dtype"] or not isinstance(tensor_record["shape"], list) or any(type(value) is not int or value < 0 for value in tensor_record["shape"]) or re.fullmatch(r"[0-9a-f]{64}", tensor_record["sha256"] or "") is None:
            raise ContractError("selected canonical state tensor differs")
    canonical_model, _, _ = _construct_seeded_model("selected", seed, runtime)
    expected_records, expected_state_sha256 = _state_manifest(canonical_model)
    if manifest["state_tensors"] != expected_records or manifest["state_sha256"] != expected_state_sha256:
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "selected_canonical_state_manifest"})
    sentinel_path = run_root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    sentinel = _canonical_json_artifact(sentinel_path)
    if sentinel != _sentinel_payload(runtime):
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "selected_attention_oracle_sentinel"})
    expected_error = _selected_attention_oracle_for_model(canonical_model, runtime, sentinel)
    detail_path = seed_root / "selected_attention_oracle_detail.json"
    detail = _canonical_json_artifact(detail_path)
    validate_exact_keys(detail, payload["artifacts"]["schemas"]["selected_attention_oracle_detail"]["exact_keys"], "selected attention oracle detail")
    expected_detail = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "construction_seed": seed,
        "constructor_state_manifest_sha256": sha256_file(manifest_path),
        "sentinel_payload_sha256": sha256_file(sentinel_path),
        "max_error": expected_error,
        "tolerance": 1e-5,
        "pass": math.isfinite(expected_error) and expected_error <= 1e-5,
    }
    if detail != expected_detail or detail["pass"] is not True:
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "selected_attention_oracle_detail"})
    oracle_rows = [row for row in evaluation_rows if row["metric"] == "selected_mask_oracle_max_error"]
    expected_provenance = [sha256_file(manifest_path), sha256_file(sentinel_path), sha256_file(detail_path)]
    if len(oracle_rows) != 1 or oracle_rows[0]["estimate"] != expected_error or oracle_rows[0]["selected_mask_oracle_max_error"] != expected_error or oracle_rows[0]["provenance_sha256s"] != expected_provenance:
        raise ContractError("evaluation oracle evidence differs")
    return {"attention_error": expected_error, "state_sha256": expected_state_sha256, "state_tensors": expected_records}


def _reconstruct_rung_one_initialization(seed: int, endpoints: Mapping[str, Mapping[str, Any]], runtime: RuntimeModules) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical, _, _ = _construct_seeded_model("selected", seed, runtime)
    canonical_state = canonical.state_dict()
    _, canonical_sha256 = _state_manifest(canonical)
    destinations = []
    for role in ("all_eligible", "selected", "dense"):
        destination, _, _ = _construct_seeded_model(role, seed, runtime)
        runtime.model_module.copy_compatible_state(canonical, destination, include_router=True)
        destinations.append(destination)
    copy_pass = True
    for destination in destinations:
        destination_state = destination.state_dict()
        for name, expected in canonical_state.items():
            if name in destination_state and destination_state[name].shape == expected.shape and _tensor_sha256(destination_state[name]) != _tensor_sha256(expected):
                copy_pass = False
    joint_state = endpoints["joint"]["checkpoint"]["model_state_dict"]
    joint_hashes = {name: _tensor_sha256(tensor) for name, tensor in joint_state.items()}
    for role in ("local_only", "all_eligible"):
        destination, _, _ = _construct_seeded_model(role, seed, runtime)
        destination.load_state_dict(joint_state, strict=True)
        destination_hashes = {name: _tensor_sha256(tensor) for name, tensor in destination.state_dict().items()}
        if destination_hashes != joint_hashes:
            copy_pass = False
    if not copy_pass:
        raise HardAbort("endpoint_inconsistency", {"seed": seed, "surface": "parity_initialization_copy"})
    roles = ["selected_canonical", "all_eligible_donor", "selected_destination", "dense_destination", "block4_local_only_evaluation", "all_eligible_clone_evaluation"]
    return {"pass": True, "constructor_roles": roles, "canonical_state_sha256": canonical_sha256}, {"pass": True, "compatible_copy": True, "constructor_count": len(roles)}


def _reconstruct_reload_records(endpoints: Mapping[str, Mapping[str, Any]], rung: int, seed: int, evaluation_payload: Mapping[str, Any], runtime: RuntimeModules) -> list[dict[str, Any]]:
    tensors = payload_to_tensors(evaluation_payload, runtime.torch)
    batch_size = 16 if rung == 1 else 8
    batch = {"tokens": tensors["tokens"][:batch_size]}
    roles = {"donor": "all_eligible", "router_only": "selected", "joint": "selected", "dense_base": "dense", "dense_continuation": "dense", "rung_two": "rung_two"}
    stages = tuple(RUNG_ONE_STAGE_ENDPOINTS) if rung == 1 else ("rung_two",)
    records = []
    for stage in stages:
        model, _, _ = _construct_seeded_model(roles[stage], seed, runtime)
        checkpoint = endpoints[stage]["checkpoint"]
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        records.append({"stage": stage, **_fresh_reload_evidence(model, checkpoint, batch, rung, runtime)})
    return records


def _reconstruct_rung_two_route_payload(endpoints: Mapping[str, Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], evaluation_payload: Mapping[str, Any], runtime: RuntimeModules) -> dict[str, Any]:
    torch = runtime.torch
    model, _, _ = _construct_seeded_model("rung_two", RUNG_TWO_SEED, runtime)
    model.load_state_dict(endpoints["rung_two"]["checkpoint"]["model_state_dict"], strict=True)
    evaluation = payload_to_tensors(evaluation_payload, torch)
    evaluation_overflow = 0
    evaluation_maximum = 0
    source_telemetry_max_error = 0.0
    telemetry_audit_forward_count = 0
    model.eval()
    with torch.inference_mode():
        for condition in RUNG_TWO_CONDITIONS:
            for batch_index in range(16):
                start = batch_index * 32
                stop = start + 32
                tokens = evaluation["tokens"][start:stop]
                knockout = condition == "recurrent_knockout"
                source_output = model(tokens, return_aux=True, route_detail=True, recurrent_knockout=knockout)
                telemetry_output = model(tokens, return_aux=True, route_detail=True, recurrent_telemetry=True, recurrent_knockout=knockout)
                target = evaluation["targets"][start:stop, 510]
                _, _, parity_error = _rung_two_source_prediction(torch, source_output, telemetry_output, target, {"seed": RUNG_TWO_SEED, "stage": condition, "logical_update": batch_index})
                source_telemetry_max_error = max(source_telemetry_max_error, parity_error)
                telemetry_audit_forward_count += 1
                for output in (source_output, telemetry_output):
                    overflow, maximum, _, _, _, _ = _route_observation(output, 2, {"seed": RUNG_TWO_SEED, "stage": condition, "logical_update": batch_index})
                    evaluation_overflow += overflow
                    evaluation_maximum = max(evaluation_maximum, maximum)
    training_overflow = sum(int(row["raw_overflow_count"]) for row in train_rows)
    training_maximum = max((int(row["max_bucket_load"]) for row in train_rows), default=0)
    return {
        "overflow_count": training_overflow + evaluation_overflow,
        "max_bucket_load": max(training_maximum, evaluation_maximum),
        "postcheckpoint_assertions": telemetry_audit_forward_count == 32 and source_telemetry_max_error <= 1e-7,
        "source_telemetry_max_error": source_telemetry_max_error,
        "telemetry_audit_forward_count": telemetry_audit_forward_count,
    }


def _trained_backend_payload(
    run_root: Path,
    rung: int,
    seed: int,
    endpoints: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    runtime: RuntimeModules,
) -> dict[str, Any]:
    stages = ("donor", "router_only", "joint", "dense_base", "dense_continuation") if rung == 1 else ("rung_two",)
    if len(records) != len(stages) or tuple(record.get("execution_stage") for record in records) != stages:
        raise ContractError("trained backend endpoint order differs")
    keys = {
        "schema_version", "run_id", "rung", "construction_seed", "execution_stage", "checkpoint_stage", "checkpoint_model", "completed_update", "checkpoint_path", "checkpoint_sha256", "optimizer_state_sha256", "data_seed", "input_sha256", "comparison_positions", "parameter_count", "parameter_max_abs", "parameter_exact", "logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "full_tensor_logits_max_abs", "full_tensor_hidden_max_abs", "full_tensor_sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "component_loss_errors", "gradient_count", "gradient_max_abs", "gradient_relative_max", "gradient_normalized_l2_max", "gradient_cosine_min", "gradient_worst_tensor", "gradient_worst_index", "gradient_worst_observed", "gradient_worst_expected", "gradient_absolute_pass", "gradient_scale_aware_pass", "gradient_pass", "gradient_scale_aware_absolute_tolerance", "gradient_relative_tolerance", "gradient_normalized_l2_tolerance", "gradient_cosine_tolerance", "gradient_none_zero_exact", "optimizer_state_count", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs", "optimizer_step_exact", "optimizer_parameter_identity_exact", "raw_route_exact", "effective_route_exact", "address_route_exact", "comparison_tolerance", "logit_loss_gradient_tolerance", "loss_tolerance", "optimizer_tolerance", "max_error", "pass",
    }
    stage_metadata = {
        "donor": ("donor", "all_eligible_donor", 1024, 610000, "all_eligible"),
        "router_only": ("router", "selected", 768, 620000, "selected"),
        "joint": ("joint", "selected", 512, 630000, "selected"),
        "dense_base": ("dense_base", "dense_causal", 1024, 640000, "dense"),
        "dense_continuation": ("dense", "dense_causal", 512, 650000, "dense"),
        "rung_two": ("rung2", "rung_two", 1536, 660000, "rung_two"),
    }
    expected_paths = {
        "donor": f"rung1/{seed}/checkpoints/donor_last.pt",
        "router_only": f"rung1/{seed}/checkpoints/router_last.pt",
        "joint": f"rung1/{seed}/checkpoints/final_last.pt",
        "dense_base": f"rung1/{seed}/checkpoints/dense_base_last.pt",
        "dense_continuation": f"rung1/{seed}/checkpoints/dense_last.pt",
        "rung_two": "rung2/83/checkpoints/final_last.pt",
    }
    torch = runtime.torch
    validated = []
    for record, stage in zip(records, stages):
        if set(record) != keys:
            raise ContractError("trained backend record keys differ")
        checkpoint_stage, model_name, updates, data_base, role = stage_metadata[stage]
        endpoint = endpoints[stage]
        checkpoint = endpoint["checkpoint"]
        optimizer_sha256 = hashlib.sha256(_torch_artifact_bytes(checkpoint["optimizer_state_dict"], torch)).hexdigest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(data_base + seed)
        batch = _continuous_rung_two_batch(generator, 1, torch) if stage == "rung_two" else _continuous_rung_one_batch(generator, 1, torch)
        expected_input_sha256 = _batch_payload_hash(batch, stage)
        model, _, _ = _construct_seeded_model(role, seed, runtime)
        _, _, membership = _make_optimizer(model, stage, runtime)
        trainable_count = sum(value["requires_grad"] is True for value in membership.values())
        if record["schema_version"] != SCHEMA_VERSION or record["run_id"] != run_root.name or record["rung"] != rung or record["construction_seed"] != seed:
            raise ContractError("trained backend record identity differs")
        if record["checkpoint_stage"] != checkpoint_stage or record["checkpoint_model"] != model_name or record["completed_update"] != updates:
            raise ContractError("trained backend checkpoint metadata differs")
        if record["checkpoint_path"] != expected_paths[stage] or record["checkpoint_sha256"] != endpoint["sha256"] or record["optimizer_state_sha256"] != optimizer_sha256:
            raise ContractError("trained backend checkpoint binding differs")
        if record["data_seed"] != data_base + seed or record["input_sha256"] != expected_input_sha256 or record["comparison_positions"] != ([510] if stage == "rung_two" else [126]):
            raise ContractError("trained backend input binding differs")
        if record["parameter_count"] != len(checkpoint["model_state_dict"]) or record["gradient_count"] != trainable_count or record["optimizer_state_count"] != trainable_count:
            raise ContractError("trained backend parameter cardinality differs")
        numeric_names = ("parameter_max_abs", "logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "full_tensor_logits_max_abs", "full_tensor_hidden_max_abs", "full_tensor_sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "gradient_relative_max", "gradient_normalized_l2_max", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs", "max_error")
        if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(float(record[name])) or record[name] < 0 for name in numeric_names):
            raise ContractError("trained backend numeric evidence differs")
        signed_gradient_names = ("gradient_cosine_min", "gradient_worst_observed", "gradient_worst_expected")
        if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float)) or not math.isfinite(float(record[name])) for name in signed_gradient_names) or not -1.0 <= record["gradient_cosine_min"] <= 1.0:
            raise ContractError("trained backend signed gradient evidence differs")
        if not isinstance(record["gradient_worst_tensor"], str) or not record["gradient_worst_tensor"] or not isinstance(record["gradient_worst_index"], list) or not record["gradient_worst_index"] or any(type(value) is not int or value < 0 for value in record["gradient_worst_index"]):
            raise ContractError("trained backend worst gradient identity differs")
        component_keys = {"task"}
        if stage in {"router_only", "joint"}:
            component_keys.add("supervised_route")
        if stage == "joint":
            component_keys.add("internal_router")
        components = record["component_loss_errors"]
        if not isinstance(components, Mapping) or set(components) != component_keys or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0 for value in components.values()) or record["component_loss_max_abs"] != max(components.values()):
            raise ContractError("trained backend component loss evidence differs")
        gradient_absolute_pass = record["gradient_max_abs"] <= record["logit_loss_gradient_tolerance"]
        gradient_scale_aware_pass = record["gradient_max_abs"] <= record["gradient_scale_aware_absolute_tolerance"] and record["gradient_relative_max"] <= record["gradient_relative_tolerance"] and record["gradient_normalized_l2_max"] <= record["gradient_normalized_l2_tolerance"] and record["gradient_cosine_min"] >= record["gradient_cosine_tolerance"]
        if type(record["gradient_absolute_pass"]) is not bool or type(record["gradient_scale_aware_pass"]) is not bool or type(record["gradient_pass"]) is not bool or record["gradient_absolute_pass"] is not gradient_absolute_pass or record["gradient_scale_aware_pass"] is not gradient_scale_aware_pass or record["gradient_pass"] is not (gradient_absolute_pass or gradient_scale_aware_pass):
            raise ContractError("trained backend gradient decision differs")
        exact_names = ("parameter_exact", "gradient_none_zero_exact", "gradient_pass", "optimizer_step_exact", "optimizer_parameter_identity_exact", "raw_route_exact", "effective_route_exact", "address_route_exact", "pass")
        if any(record[name] is not True for name in exact_names) or record["parameter_max_abs"] != 0.0:
            raise ContractError("trained backend exact evidence differs")
        expected_max = max(record[name] for name in ("logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs", "total_loss_max_abs", "component_loss_max_abs", "gradient_max_abs", "optimizer_first_moment_max_abs", "optimizer_second_moment_max_abs"))
        if record["comparison_tolerance"] != 1e-5 or record["logit_loss_gradient_tolerance"] != 1e-5 or record["loss_tolerance"] != 1e-6 or record["optimizer_tolerance"] != 0.0 or record["gradient_scale_aware_absolute_tolerance"] != 3e-5 or record["gradient_relative_tolerance"] != 1e-4 or record["gradient_normalized_l2_tolerance"] != 5e-5 or record["gradient_cosine_tolerance"] != 0.999999999 or record["max_error"] != expected_max:
            raise ContractError("trained backend tolerance evidence differs")
        if max(record[name] for name in ("logits_max_abs", "hidden_max_abs", "sequence_delta_max_abs")) > 1e-5 or max(record["total_loss_max_abs"], record["component_loss_max_abs"]) > record["loss_tolerance"] or record["gradient_pass"] is not True or max(record["optimizer_first_moment_max_abs"], record["optimizer_second_moment_max_abs"]) != 0.0:
            raise ContractError("trained backend threshold differs")
        validated.append(dict(record))
    return {"pass": True, "stages": list(stages), "records": validated, "max_error": max(record["max_error"] for record in validated)}


def reconstruct_semantic_parity_facts(
    run_root: Path,
    rung: int,
    seed: int,
    endpoints: Mapping[str, Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    evaluation_payload: Mapping[str, Any],
    intervention_records: Sequence[Mapping[str, Any]],
    runtime: RuntimeModules,
    routing_evidence: Mapping[str, Any] | None,
    data_evidence: Mapping[str, Any] | None,
    oracle_evidence: Mapping[str, Any] | None,
    trained_backend_records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    assertions = _pretraining_assertion_lookup(run_root)
    if rung == 1:
        if routing_evidence is None or data_evidence is None or oracle_evidence is None:
            raise ContractError("rung-one parity primary evidence is absent")
        endpoint_order = ("donor", "router_only", "joint", "dense_base", "dense_continuation")
        checksum_payload = {"verified": all(sha256_file(endpoints[stage]["path"]) == endpoints[stage]["sha256"] for stage in endpoint_order), "sha256s": [endpoints[stage]["sha256"] for stage in endpoint_order]}
        training_overflow = sum(int(row["raw_overflow_count"]) for row in train_rows)
        training_maximum = max((int(row["max_bucket_load"]) for row in train_rows), default=0)
        evaluation_overflow = max((int(row["overflow_count"] or 0) for row in evaluation_rows), default=0)
        evaluation_maximum = max((int(row["max_bucket_load"] or 0) for row in evaluation_rows), default=0)
        route_payload = {"overflow_count": training_overflow + evaluation_overflow, "max_bucket_load": max(training_maximum, evaluation_maximum), **dict(data_evidence)}
        attention_error = float(oracle_evidence["attention_error"])
        initialization_payload, copy_payload = _reconstruct_rung_one_initialization(seed, endpoints, runtime)
        evidence_paths = ["run/preflight.json", f"rung1/{seed}/checkpoints/final_last.pt", f"rung1/{seed}/intervention_deltas.json"]
    elif rung == 2 and seed == RUNG_TWO_SEED:
        endpoint = endpoints["rung_two"]
        checksum_payload = {"verified": sha256_file(endpoint["path"]) == endpoint["sha256"], "sha256s": [endpoint["sha256"]]}
        route_payload = _reconstruct_rung_two_route_payload(endpoints, train_rows, evaluation_payload, runtime)
        attention_error = float(assertions["source_host_route_and_attention_parity"]["actual"]["oracle_error"])
        canonical, _, _ = _construct_seeded_model("rung_two", seed, runtime)
        canonical_records, canonical_sha256 = _state_manifest(canonical)
        initialization_payload = {"pass": True, "rng_isolated": True, "state_tensors": canonical_records, "state_sha256": canonical_sha256}
        copy_payload = {"pass": True, "canonical_state_exact": True, "state_sha256": canonical_sha256}
        evidence_paths = ["run/preflight.json", "rung2/83/checkpoints/final_last.pt", "rung2/83/intervention_deltas.json"]
    else:
        raise ContractError("semantic parity rung differs")
    reload_records = _reconstruct_reload_records(endpoints, rung, seed, evaluation_payload, runtime)
    intervention_payload = {"pass": True, "record_count": len(intervention_records), "matched_intact": True, "knockout_zero_exposed": True}
    trained_payload = _trained_backend_payload(run_root, rung, seed, endpoints, trained_backend_records, runtime)
    facts = _claim_parity_facts(assertions, seed, rung, checksum_payload, route_payload, attention_error, 1e-5, initialization_payload, copy_payload, reload_records, intervention_payload, trained_payload, evidence_paths)
    if tuple(facts) != PARITY_SCOPES:
        raise ContractError("semantic parity fact order differs")
    return facts


def _validate_semantic_parity_package(
    run_root: Path,
    rung: int,
    seed: int,
    endpoints: Mapping[str, Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    evaluation_payload: Mapping[str, Any],
    intervention_records: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
    runtime: RuntimeModules,
    routing_evidence: Mapping[str, Any] | None,
    data_evidence: Mapping[str, Any] | None,
    oracle_evidence: Mapping[str, Any] | None,
) -> None:
    trained_check = next((check for check in parity["checks"] if check.get("scope") == "trained_backend"), None)
    if trained_check is None:
        raise ContractError("trained backend parity check is absent")
    trained_detail = _canonical_json_artifact(run_root / "run" / "check_details" / f"{trained_check['details_sha256']}.json")
    trained_outputs = trained_detail.get("outputs")
    if not isinstance(trained_outputs, Mapping) or not isinstance(trained_outputs.get("records"), list):
        raise ContractError("trained backend parity detail differs")
    facts = reconstruct_semantic_parity_facts(run_root, rung, seed, endpoints, train_rows, evaluation_rows, evaluation_payload, intervention_records, runtime, routing_evidence, data_evidence, oracle_evidence, trained_outputs["records"])
    for scope, check in zip(PARITY_SCOPES, parity["checks"]):
        fact = facts[scope]
        expected_detail = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_root.name,
            "name": fact["name"],
            "scope": scope,
            "inputs": fact["inputs"],
            "outputs": fact["outputs"],
            "evidence_paths": sorted(fact["evidence_paths"]),
        }
        details_sha256 = canonical_json_sha256(expected_detail)
        expected_check = {"name": fact["name"], "scope": scope, "max_error": fact["max_error"], "tolerance": fact["tolerance"], "pass": fact["pass"], "details_sha256": details_sha256}
        if check != expected_check:
            raise ContractError("semantic parity check differs")
        detail_path = run_root / "run" / "check_details" / f"{details_sha256}.json"
        detail = _canonical_json_artifact(detail_path)
        if sha256_file(detail_path) != details_sha256 or detail != expected_detail:
            raise ContractError("semantic parity detail differs")


def validate_claim_artifact_package(run_root: Path, payload: Mapping[str, Any], runtime: RuntimeModules | None = None) -> None:
    runtime_modules = _import_runtime() if runtime is None else runtime
    torch = runtime_modules.torch
    validate_parent_ledger_accounting(run_root)
    validate_preclaim_reconstruction(run_root, payload, runtime_modules)
    generated_strata = {}
    separated_first_hashes = []
    for seed in (*RUNG_ONE_SEEDS, RUNG_TWO_SEED):
        rung = 1 if seed != RUNG_TWO_SEED else 2
        seed_root = run_root / (f"rung1/{seed}" if rung == 1 else "rung2/83")
        endpoints = _load_claim_endpoints(run_root, rung, seed, torch)
        train_rows = _validate_train_artifact(run_root, payload, seed_root, rung, seed, torch)
        _validate_endpoint_train_closure(endpoints, train_rows)
        evaluation_payload, eval_data_sha256 = _load_evaluation_evidence(run_root, rung, seed, torch)
        checkpoint_by_condition = _checkpoint_by_condition(rung, endpoints)
        prediction_evidence = _validate_prediction_artifact(run_root, payload, seed_root, rung, seed, checkpoint_by_condition, evaluation_payload, eval_data_sha256)
        evaluation_rows = _canonical_jsonl_records(seed_root / "evaluation.jsonl")
        routing_evidence = _validate_routing_artifact(run_root, payload, seed_root, seed, evaluation_rows, prediction_evidence, checkpoint_by_condition) if rung == 1 else None
        _validate_evaluation_reconstruction(run_root, payload, rung, seed, evaluation_rows, prediction_evidence, checkpoint_by_condition, routing_evidence)
        data_evidence = None
        oracle_evidence = None
        if rung == 1:
            data_evidence = _validate_rung_one_data_artifacts(run_root, seed, evaluation_payload, routing_evidence, torch)
            generated_strata[seed] = (data_evidence["same_condition"], data_evidence["changed_condition"])
            separated_first_hashes.append(next(row["batch_sha256"] for row in train_rows if row["stage"] == "donor" and row["logical_update"] == 1))
        else:
            separated_first_hashes.append(train_rows[0]["batch_sha256"])
        state = _canonical_json_artifact(seed_root / "state_stats.json")
        validate_exact_keys(state, payload["artifacts"]["schemas"]["state_stats"]["top_level_exact_keys"], "state statistics")
        if state["schema_version"] != SCHEMA_VERSION or state["run_id"] != run_root.name or state["rung"] != rung or state["claim_seed"] != seed or state["construction_seed"] != seed:
            raise ContractError("state statistics artifact identity differs")
        validate_state_records(state["records"], rung, checkpoint_by_condition)
        intervention = _canonical_json_artifact(seed_root / "intervention_deltas.json")
        validate_exact_keys(intervention, payload["artifacts"]["schemas"]["intervention_deltas"]["top_level_exact_keys"], "intervention deltas")
        final_checkpoint_sha = endpoints["joint" if rung == 1 else "rung_two"]["sha256"]
        if intervention["schema_version"] != SCHEMA_VERSION or intervention["run_id"] != run_root.name or intervention["rung"] != rung or intervention["claim_seed"] != seed or intervention["construction_seed"] != seed:
            raise ContractError("intervention artifact identity differs")
        validate_intervention_records(intervention["records"], rung, checkpoint_by_condition)
        if rung == 1:
            grad_artifact = _canonical_json_artifact(seed_root / "grad_audit.json")
            grad_records = _validate_gradient_artifact(grad_artifact, run_root, rung, seed, {"donor": 1024, "router_only": 768, "joint": 512}, endpoints, runtime_modules)
            dense_artifact = _canonical_json_artifact(seed_root / "dense_grad_audit.json")
            dense_records = _validate_gradient_artifact(dense_artifact, run_root, rung, seed, {"dense_base": 1024, "dense_continuation": 512}, endpoints, runtime_modules)
            all_gradient_records = grad_records + dense_records
        else:
            grad_artifact = _canonical_json_artifact(seed_root / "grad_audit.json")
            all_gradient_records = _validate_gradient_artifact(grad_artifact, run_root, rung, seed, {"rung_two": 1536}, endpoints, runtime_modules)
        parity = _canonical_json_artifact(seed_root / "parity.json")
        validate_exact_keys(parity, payload["artifacts"]["schemas"]["parity"]["top_level_exact_keys"], "parity artifact")
        if parity["schema_version"] != SCHEMA_VERSION or parity["run_id"] != run_root.name or parity["rung"] != rung or parity["claim_seed"] != seed or parity["construction_seed"] != seed or parity["checkpoint_sha256"] != final_checkpoint_sha:
            raise ContractError("parity artifact identity differs")
        validate_parity_checks(parity["checks"])
        if rung == 1:
            oracle_evidence = _validate_selected_oracle_evidence(run_root, payload, seed, evaluation_rows, runtime_modules)
        _validate_semantic_parity_package(run_root, rung, seed, endpoints, train_rows, evaluation_rows, evaluation_payload, intervention["records"], parity, runtime_modules, routing_evidence, data_evidence, oracle_evidence)
        refs = _canonical_json_artifact(seed_root / "resource_refs.json")
        validate_exact_keys(refs, payload["artifacts"]["schemas"]["resource_refs"]["exact_keys"], "resource references")
        if refs["schema_version"] != SCHEMA_VERSION or refs["run_id"] != run_root.name or refs["rung"] != rung or refs["claim_seed"] != seed or refs["construction_seed"] != seed or refs["sample_ids"] != sorted(set(refs["sample_ids"])):
            raise ContractError("resource references artifact identity differs")
        referenced_sets = {tuple(row["resource_sample_ids"]) for row in evaluation_rows}
        accounting = _canonical_json_artifact(seed_root / "accounting.json")
        validate_exact_keys(accounting, payload["artifacts"]["schemas"]["accounting"]["top_level_exact_keys"], "accounting artifact")
        if accounting["schema_version"] != SCHEMA_VERSION or accounting["run_id"] != run_root.name or accounting["rung"] != rung or accounting["claim_seed"] != seed or accounting["construction_seed"] != seed:
            raise ContractError("accounting artifact identity differs")
        validate_model_accounting(accounting["models"], _read_jsonl(seed_root / "attempts.jsonl"))
        referenced_sets.update(tuple(model["resource_sample_ids"]) for model in accounting["models"])
        if referenced_sets != {tuple(refs["sample_ids"])}:
            raise ContractError("resource reference artifacts disagree")
        _validate_accounting_evidence(accounting["models"], rung, endpoints, all_gradient_records, routing_evidence)
    if len(separated_first_hashes) != 6 or len(set(separated_first_hashes)) != 6:
        raise HardAbort("endpoint_inconsistency", {"surface": "first_batch_seed_separation"})
    _validate_carry_shuffle_strata(payload, generated_strata)
    validate_gate_input_package(run_root, payload)


def finalize_clean_claim(
    run_root: Path,
    payload: Mapping[str, Any],
    anchors: FrozenManifestAnchors,
    signals: SignalController,
    claim_start_monotonic_ns: int,
    claim_result: Mapping[str, Any],
    runtime: RuntimeModules | None = None,
) -> None:
    clean_transport_finalizer = claim_result.get("clean_transport_finalizer")
    abort_transport_finalizer = claim_result.get("abort_transport_finalizer")
    if (clean_transport_finalizer is None) != (abort_transport_finalizer is None) or clean_transport_finalizer is not None and (not callable(clean_transport_finalizer) or not callable(abort_transport_finalizer)):
        raise ContractError("claim transport finalizer contract differs")
    accounting = claim_result["accounting"]
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_completion")
    validate_resource_timeline(claim_result["resource_rows"], "claim", require_clean_final=True)
    validate_parent_ledger_accounting(run_root)
    validate_claim_artifact_package(run_root, payload, runtime)
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "after_artifact_validation")
    wall_end = time.monotonic_ns()
    completion = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_root.name,
        "claim_start_monotonic_ns": claim_start_monotonic_ns,
        "resource_sampling_end_monotonic_ns": claim_result["resource_sampling_end_monotonic_ns"],
        "wall_accounting_end_monotonic_ns": wall_end,
        "claim_elapsed_seconds": (wall_end - claim_start_monotonic_ns) / 1e9,
        "resource_final_sample_id": claim_result["resource_final_sample_id"],
        "attempted_updates": accounting.attempted_updates,
        "completed_updates": accounting.completed_updates,
        "token_positions": accounting.completed_token_positions,
        "packaging_excluded": ["run/completion.json", "summary.json", "SHA256SUMS"],
    }
    write_canonical_json(run_root / "run" / "completion.json", completion)
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "after_completion")
    gates = _gate_summary(run_root, payload, runtime)
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "after_gate_summary")
    manifest_records = []
    for relative, path in _iter_regular_files(run_root):
        if relative in {"summary.json", "SHA256SUMS"}:
            continue
        manifest_records.append({"path": relative, "sha256": sha256_file(path)})
    artifact_manifest_sha256 = canonical_json_sha256(manifest_records)
    decisions = validate_gate_input_package(run_root, payload)
    summary = summary_from_gate_decisions(run_root.name, artifact_manifest_sha256, decisions)
    if any(summary[key] != gates[key] for key in ("passed_gates", "failed_gates", "per_seed", "rung_two")):
        raise HardAbort("artifact_inconsistency", {"surface": "summary_gate_sections"})
    validate_summary_contract(summary, decisions)
    write_canonical_json(run_root / "summary.json", summary)
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "after_summary")
    if clean_transport_finalizer is not None:
        clean_transport_finalizer(claim_result)
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "after_clean_transport")
    def deadline_guard(stage: str) -> None:
        final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, f"sha256s_{stage}")

    expected_paths = validate_artifact_closure(run_root, payload, "clean")
    final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_sha256s")
    write_sha256s_terminal(run_root, expected_paths=expected_paths, signals=signals, fault_hook=deadline_guard)


def execute_run(
    entry: EntryConfiguration,
    payload: Mapping[str, Any],
    runtime: RuntimeModules,
    launcher_argv: Sequence[str],
    resource_pilot_runner: Callable[..., Mapping[str, Any]] | None = None,
    claim_runner: Callable[..., Mapping[str, Any]] | None = None,
    trained_backend_probe: Callable[[Path, str], Sequence[Mapping[str, Any]]] | None = None,
) -> int:
    selected_pilot_runner = run_resource_pilot if resource_pilot_runner is None else resource_pilot_runner
    selected_claim_runner = run_claim_workers if claim_runner is None else claim_runner
    if RESULTS_PARENT.is_symlink():
        raise InitializationRefusal("results parent cannot be symbolic")
    if not RESULTS_PARENT.exists():
        RESULTS_PARENT.mkdir(exist_ok=False)
        fsync_directory(RESULTS_PARENT.parent)
        fsync_directory(RESULTS_PARENT)
    if not RESULTS_PARENT.is_dir():
        raise InitializationRefusal("results parent is not a directory")
    staging = RESULTS_PARENT / f".{entry.run_id}.initializing.{os.getpid()}"
    if os.path.lexists(staging) or os.path.lexists(entry.run_root):
        raise InitializationRefusal("initialization or final run path already exists")
    signals = SignalController()
    staging_created = False
    staging_identity: tuple[int, int] | None = None
    active = False
    phase = "prepilot"
    writers: dict[str, CrashAtomicJsonlWriter] = {}
    swap_baseline: int | None = None
    abort_origin_ns: int | None = None
    abort_origin_utc: str | None = None
    primary_failure_latch: PrimaryFailureLatch | None = None
    frozen_anchors: FrozenManifestAnchors | None = None
    training_start_state = "not_started"
    claim_result: Mapping[str, Any] | None = None

    def refuse_preactivation() -> None:
        failure: BaseException | None = None
        if staging_created:
            try:
                if staging_identity is None:
                    raise UnrecoverableOrphan("initialization staging identity is unavailable")
                metadata = os.lstat(staging)
                if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != staging_identity:
                    raise UnrecoverableOrphan("initialization staging identity changed")
                _remove_tree_and_fsync(staging)
            except BaseException as exc:
                failure = exc if isinstance(exc, UnrecoverableOrphan) else UnrecoverableOrphan("initialization staging cleanup failed")
        try:
            if signals.active and not signals.terminal:
                signals.deactivate_terminal()
        except BaseException:
            if failure is None:
                failure = UnrecoverableOrphan("initialization signal cleanup failed")
        if failure is not None:
            raise failure

    try:
        signals.install()
        staging.mkdir(exist_ok=False)
        staging_created = True
        metadata = os.lstat(staging)
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnrecoverableOrphan("initialization staging identity differs")
        staging_identity = (metadata.st_dev, metadata.st_ino)
        fsync_directory(RESULTS_PARENT)
        fsync_directory(staging)
        primary_failure_latch = PrimaryFailureLatch(payload["abort_rules"]["hard_abort_registry"])
        if trained_backend_probe is None:
            build_shared_prepilot_base(staging, entry.run_id, launcher_argv, runtime, payload)
        else:
            build_shared_prepilot_base(staging, entry.run_id, launcher_argv, runtime, payload, trained_backend_probe)
        frozen_anchors = capture_frozen_manifest_anchors(staging)
        _verify_public_commit()
        if canonical_json_sha256(load_prereg_payload()) != PREREG_CANONICAL_SHA256:
            raise InitializationRefusal("preregistration changed during staging")
        validate_base_review_target_binding(staging, "not_started", None)
        publication = publish_and_activate(staging, entry.run_root, signals)
        staging_created = False
        staging_identity = None
        if publication.state == "orphaned":
            raise UnrecoverableOrphan("filesystem publication rollback failed")
        if publication.state != "active" or publication.abort_accounting_start_monotonic_ns is None or publication.abort_wall_start_utc is None:
            raise InitializationRefusal("lifecycle publication did not activate")
        active = True
        abort_origin_ns = publication.abort_accounting_start_monotonic_ns
        abort_origin_utc = publication.abort_wall_start_utc
        if publication.pending_signal is not None:
            raise HardAbort("signal_or_interruption")
        if frozen_anchors is None:
            raise HardAbort("frozen_hash_change")
        _verify_active_frozen_hashes(entry.run_root, frozen_anchors)
        pilot_transition = precreate_pilot_timeline(entry.run_root, signals, sample_swap)
        writers.update(pilot_transition.writers)
        swap_baseline = pilot_transition.swap_baseline_bytes
        phase = pilot_transition.phase
        if pilot_transition.outcome != "ready":
            raise HardAbort(pilot_transition.reason_code or "artifact_inconsistency")
        phase = "pilot"
        pilot = selected_pilot_runner(entry.run_root, payload, frozen_anchors, signals, pilot_transition)
        writers = {}
        _verify_active_frozen_hashes(entry.run_root, frozen_anchors)
        if signals.pending_signal is not None:
            raise HardAbort("signal_or_interruption")
        if pilot["decision"] == "stop":
            try:
                _verify_active_frozen_hashes(entry.run_root, frozen_anchors)
                expected_paths = validate_artifact_closure(entry.run_root, payload, "pilot_stop")
                write_sha256s_terminal(entry.run_root, expected_paths=expected_paths, signals=signals)
            except UnrecoverableOrphan:
                raise
            except BaseException:
                if signals.terminal:
                    return 0
                pilot_path = entry.run_root / "run" / "pilot.json"
                if pilot_path.exists():
                    pilot_path.unlink()
                    fsync_directory(pilot_path.parent)
                raise
            return 0
        claim_transition = precreate_claim_ledgers(entry.run_root, signals, sample_swap)
        writers = dict(claim_transition.writers)
        swap_baseline = claim_transition.swap_baseline_bytes
        phase = claim_transition.phase
        if claim_transition.outcome != "ready":
            raise HardAbort(claim_transition.reason_code or "artifact_inconsistency")
        phase = "claim"
        frozen_anchors, claim_start_monotonic_ns = establish_training_start_plan_barrier(entry.run_root, frozen_anchors, signals)
        training_start_state = "started"
        if signals.pending_signal is not None:
            raise HardAbort("signal_or_interruption", {"signal": signals.pending_signal, "training_start_state": "started"})
        prepare_claim_data(entry.run_root, runtime, payload, frozen_anchors, signals, claim_start_monotonic_ns)
        _verify_active_frozen_hashes(entry.run_root, frozen_anchors)
        claim_result = selected_claim_runner(entry.run_root, payload, frozen_anchors, signals, claim_transition, claim_start_monotonic_ns)
        writers = {}
        finalize_clean_claim(entry.run_root, payload, frozen_anchors, signals, claim_start_monotonic_ns, claim_result, runtime)
        return 0
    except UnrecoverableOrphan:
        if signals.terminal:
            return 0
        if not active and signals.active and not signals.terminal:
            try:
                signals.deactivate_terminal()
            except BaseException:
                pass
        raise
    except BaseException as exc:
        if claim_result is not None:
            abort_transport_finalizer = claim_result.get("abort_transport_finalizer")
            if abort_transport_finalizer is not None:
                try:
                    abort_transport_finalizer()
                except BaseException as cleanup_error:
                    raise UnrecoverableOrphan("claim transport abort cleanup failed") from cleanup_error
        if signals.terminal:
            return 0
        if not active or abort_origin_ns is None or abort_origin_utc is None:
            refuse_preactivation()
            raise InitializationRefusal("initialization failed before lifecycle activation") from exc
        if primary_failure_latch is None:
            raise UnrecoverableOrphan("active lifecycle has no failure latch")
        if phase == "claim":
            expected_training_start_state = training_start_state
            if isinstance(exc, HardAbort):
                context_training_start_state = exc.context.get("training_start_state")
                if training_start_state == "not_started" and context_training_start_state in {"not_started", "awaiting_review", "reviewed_ready", "started"}:
                    expected_training_start_state = context_training_start_state
            try:
                observed_training_start_state, _ = _training_start_state(entry.run_root)
            except UnrecoverableOrphan:
                raise
            except ContractError:
                pass
            else:
                if observed_training_start_state != expected_training_start_state:
                    raise UnrecoverableOrphan("abort training-start state is ambiguous")
        if isinstance(exc, HardAbort) and exc.primary_latch_monotonic_ns is not None:
            reason_code = exc.reason_code
            failure_context = exc.context
            primary_latch_monotonic_ns = exc.primary_latch_monotonic_ns
        else:
            observations = [failure_observation_from_exception(exc, "artifact_inconsistency")]
            if signals.pending_signal is not None:
                observations.append({"reason_code": "signal_or_interruption", "context": {"signal": signals.pending_signal}})
            selected = primary_failure_latch.select_poll(observations)
            if selected is None:
                raise UnrecoverableOrphan("primary failure latch did not select")
            reason_code = selected.reason_code
            failure_context = selected.context
            primary_latch_monotonic_ns = selected.monotonic_ns
        observed_training_start_state = training_start_state
        context_training_start_state = failure_context.get("training_start_state")
        if training_start_state == "not_started" and phase == "claim" and context_training_start_state in {"not_started", "awaiting_review", "reviewed_ready", "started"}:
            observed_training_start_state = context_training_start_state
        finalize_hard_abort(entry.run_root, payload, signals, reason_code, phase, failure_context, abort_origin_ns, abort_origin_utc, primary_latch_monotonic_ns, writers, swap_baseline, observed_training_start_state)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
