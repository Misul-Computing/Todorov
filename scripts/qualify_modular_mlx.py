from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHILD_MESSAGE_MAX_BYTES = 3 * 1_048_576
CHILD_MESSAGE_READ_BYTES = 65_536
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroloc.simulations.memory import modular_sequence_role_cpu as cpu
from src.model import modular_mlx_backend as backend


class PilotWorkload(NamedTuple):
    name: str
    execution: str
    lanes: int
    batch_size: int
    sequence_length: int

    @property
    def token_positions_per_lane(self) -> int:
        return self.batch_size * self.sequence_length


class PilotProtocol(NamedTuple):
    seed_base: int
    seed_stride: int
    data_seed_offset: int
    route_seed_offset: int
    warmup_updates: tuple[int, ...]
    timed_updates: tuple[int, ...]
    workloads: tuple[PilotWorkload, ...]

    @property
    def all_updates(self) -> tuple[int, ...]:
        return self.warmup_updates + self.timed_updates

    @property
    def updates_per_workload(self) -> int:
        return len(self.all_updates)

    @property
    def workload_order(self) -> tuple[str, ...]:
        return tuple(workload.name for workload in self.workloads)

    @property
    def final_attempted_updates(self) -> int:
        return sum(workload.lanes * self.updates_per_workload for workload in self.workloads)

    @property
    def final_token_positions(self) -> int:
        return sum(workload.lanes * workload.token_positions_per_lane * self.updates_per_workload for workload in self.workloads)

    def model_seed(self, ordinal: int) -> int:
        return self.seed_base + self.seed_stride * ordinal

    def data_seed(self, ordinal: int) -> int:
        return self.model_seed(ordinal) + self.data_seed_offset

    def route_seed(self, ordinal: int) -> int:
        return self.model_seed(ordinal) + self.route_seed_offset

    def prior_attempts(self, ordinal: int) -> int:
        return sum(workload.lanes * self.updates_per_workload for workload in self.workloads[:ordinal])

    def expected_update(self, ordinal: int, attempted_updates: int) -> int:
        workload = self.workloads[ordinal]
        completed_updates = (attempted_updates - self.prior_attempts(ordinal)) // workload.lanes
        return self.all_updates[completed_updates]


PILOT_PROTOCOL = PilotProtocol(
    seed_base=9_999_983,
    seed_stride=100,
    data_seed_offset=1,
    route_seed_offset=2,
    warmup_updates=(1, 2, 3),
    timed_updates=(4, 5, 6, 7, 8, 9, 10, 11),
    workloads=(
        PilotWorkload("donor", "one_MLX_lane", 1, 16, 128),
        PilotWorkload("selected_vmap5", "compiled_MLX_vmap_width_5", 5, 16, 128),
        PilotWorkload("dense_vmap5", "compiled_MLX_vmap_width_5", 5, 16, 128),
        PilotWorkload("rung_two", "one_MLX_lane", 1, 8, 512),
    ),
)

TAIL_DETAIL_KEYS = ("fixture_sha256s", "warmup_duration_ns", "timed_duration_ns", "selected_max_duration_ns", "counts", "byte_sizes", "scaling", "scratch_cleanup_pass", "component_seconds")
TAIL_TIMING_FAMILIES = {
    "routing_evidence": ("routing_evidence_block",),
    "evaluation": (
        "endpoint_replay.dense_base",
        "endpoint_replay.dense_continuation",
        "endpoint_replay.donor",
        "endpoint_replay.joint",
        "endpoint_replay.router_only",
        "endpoint_replay.rung_two",
        "route_acquisition",
        "rung_one_dense",
        "rung_one_routed.all_eligible_clone",
        "rung_one_routed.all_eligible_donor",
        "rung_one_routed.block4_local_only",
        "rung_one_routed.block4_routed_knockout",
        "rung_one_routed.carry_reset",
        "rung_one_routed.carry_shuffle",
        "rung_one_routed.intact",
        "rung_one_routed.matched_random_route",
        "rung_one_routed.recurrent_knockout",
        "rung_one_routed.required_source_excluded",
        "rung_one_routed.target_forced",
        "rung_two.intact",
        "rung_two.recurrent_knockout",
    ),
    "checkpoint_reload": ("dense_vmap5_all_lanes", "donor_single", "joint_vmap5_all_lanes", "router_only_vmap5_all_lanes", "rung_two_single"),
    "packaging": ("file_batch", "io_block"),
}
TAIL_FIXTURE_KEYS = {
    "routing_evidence": ("random_routes_seed_500011", "routing_block", "rung_one_seed_123456"),
    "evaluation": ("random_routes_seed_500011", "rung_one_seed_123456", "rung_two_seed_123456", "source_exclusion_seed_633456"),
    "checkpoint_reload": ("dense_vmap5_all_lanes_metadata", "donor_single_metadata", "joint_vmap5_all_lanes_metadata", "router_only_vmap5_all_lanes_metadata", "rung_two_single_metadata"),
    "packaging": ("empty_file", "io_block"),
}
TAIL_EVALUATION_FIXTURE_BYTES = 438_368
TAIL_CHECKPOINT_TENSOR_BYTE_LOWER_BOUNDS = {
    "dense_vmap5_all_lanes": 34_240_660,
    "donor_single": 6_856_580,
    "joint_vmap5_all_lanes": 34_367_440,
    "router_only_vmap5_all_lanes": 11_567_740,
    "rung_two_single": 7_053_188,
}


def expected_tail_fixture_sha256s(name: str, *, source_exclusion_sha256: str | None = None, routing_block_sha256: str | None = None, engine_sha256: str | None = None) -> dict[str, str]:
    if name == "evaluation":
        if not isinstance(source_exclusion_sha256, str):
            raise backend.MlxQualificationError("pilot evaluation source exclusion hash differs")
        return {
            "random_routes_seed_500011": "18f568b628517fa8f77d9e6adc17c3c2ead62c46070487d416d6eee25953e54c",
            "rung_one_seed_123456": "98ff3b54f14306135eafe5a92da7abdf1111cd8690e511188bb5f0e44dcab2a9",
            "rung_two_seed_123456": "7fff37e20adc2241c217b3ed6dad6ec4d85e818d69a59fa5b8e3f5a48f2b8afe",
            "source_exclusion_seed_633456": source_exclusion_sha256,
        }
    if name == "checkpoint_reload":
        if not isinstance(engine_sha256, str):
            raise backend.MlxQualificationError("pilot checkpoint engine hash differs")
        specifications = {
            "dense_vmap5_all_lanes": ("dense_base", 5),
            "donor_single": ("donor", 1),
            "joint_vmap5_all_lanes": ("joint", 5),
            "router_only_vmap5_all_lanes": ("router_only", 5),
            "rung_two_single": ("rung_two", 1),
        }
        return {
            f"{family}_metadata": cpu.canonical_json_sha256({"claim_data": False, "completed_update": 11, "engine_sha256": engine_sha256, "family": family, "lanes": lanes, "stage": stage})
            for family, (stage, lanes) in specifications.items()
        }
    if name == "routing_evidence":
        if not isinstance(routing_block_sha256, str):
            raise backend.MlxQualificationError("pilot routing block hash differs")
        return {
            "random_routes_seed_500011": "18f568b628517fa8f77d9e6adc17c3c2ead62c46070487d416d6eee25953e54c",
            "routing_block": routing_block_sha256,
            "rung_one_seed_123456": "98ff3b54f14306135eafe5a92da7abdf1111cd8690e511188bb5f0e44dcab2a9",
        }
    if name == "packaging":
        return {
            "empty_file": hashlib.sha256(b"").hexdigest(),
            "io_block": hashlib.sha256(packaging_block()).hexdigest(),
        }
    raise backend.MlxQualificationError("pilot tail fixture family differs")


def validate_child_bootstrap(mode: str) -> None:
    _, expected = backend.child_invocation(mode)
    if tuple(sys.version_info[:3]) != (3, 9, 6):
        raise backend.MlxQualificationError("MLX child Python version differs")
    if any(name == "mlx" or name.startswith("mlx.") for name in sys.modules):
        raise backend.MlxQualificationError("MLX child pre-import boundary differs")
    allowed = dict(expected)
    allowed["__CF_USER_TEXT_ENCODING"] = "0x1F5:0x0:0x0"
    if mode in {"pilot", "serve"}:
        for name in ("MODULAR_MLX_RUN_ROOT", "MODULAR_MLX_SCRATCH_ROOT"):
            value = os.environ.get(name)
            if value is None or not Path(value).is_absolute():
                raise backend.MlxQualificationError("MLX child environment differs")
            allowed[name] = value
    if dict(os.environ) != allowed:
        raise backend.MlxQualificationError("MLX child environment differs")


def run_child(mode: str) -> int:
    validate_child_bootstrap(mode)
    sys.argv = [str(backend.ENGINE_PATH), mode]
    runpy.run_path(str(backend.ENGINE_PATH), run_name="__main__")
    return 0


def preflight_mlx_probe(root: Path, run_id: str) -> list[dict[str, Any]]:
    command, environment = backend.child_invocation("self-check")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, check=False, timeout=120)
    if completed.returncode != 0 or completed.stderr:
        raise cpu.InitializationRefusal("MLX preflight child failed")
    try:
        observed = json.loads(completed.stdout)
    except Exception as exc:
        raise cpu.InitializationRefusal("MLX preflight child output differs") from exc
    if not isinstance(observed, Mapping) or observed.get("pass") is not True or observed.get("mlx_version") != "0.29.3" or observed.get("device") != "Device(gpu, 0)":
        raise cpu.InitializationRefusal("MLX preflight self-check differs")
    parity = observed.get("full_model_parity")
    if not isinstance(parity, Mapping):
        raise cpu.InitializationRefusal("MLX initial parity differs")
    parity_evidence = backend.validate_initial_self_check(observed)
    parity_error = parity_evidence["worst_bound_ratio"]
    evidence = ["run/source_manifest.json", "run/environment.json"]
    checks = [
        cpu._check_record(root, run_id, "mlx_child_environment", "trained_backend", {"command": command, "environment": environment, "python": "3.9.6"}, {"command": command, "environment": environment, "python": "3.9.6"}, None, None, True, evidence),
        cpu._check_record(root, run_id, "mlx_metal_device", "trained_backend", "Device(gpu, 0)", observed["device"], None, None, observed["device"] == "Device(gpu, 0)", evidence),
        cpu._check_record(root, run_id, "mlx_self_check", "trained_backend", True, observed, None, None, observed["pass"] is True, evidence),
        cpu._check_record(root, run_id, "initial_backend_parity", "trained_backend", 1.0, observed, parity_error, 1.0, parity_evidence["pass"] is True and parity_error <= 1.0, evidence),
        cpu._check_record(root, run_id, "full_package_projection", "trained_backend", {"target_seconds": 600, "hard_limit_seconds": 1200, "positions": backend.POSITIONS}, {"component_order": list(cpu.load_prereg_payload()["pilot"]["time_statistics"]["measured_component_order"]), "projection_function": "project_full_package"}, None, None, True, evidence),
    ]
    if not all(record["pass"] is True for record in checks):
        raise cpu.InitializationRefusal("MLX trained backend preflight failed")
    return checks


def send(process: subprocess.Popen[str], message: Mapping[str, Any], deadline_ns: int | None = None) -> None:
    if deadline_ns is not None:
        backend.enforce_deadline(deadline_ns)
    if process.stdin is None:
        raise backend.MlxQualificationError("MLX child input is absent")
    process.stdin.write(json.dumps(dict(message), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    process.stdin.flush()


def receive(process: subprocess.Popen[str], sampler: Any = None, deadline_ns: int | None = None) -> dict[str, Any]:
    if process.stdout is None:
        raise backend.MlxQualificationError("MLX child output is absent")
    buffered = getattr(process, "_modular_mlx_stdout_remainder", b"")
    if not isinstance(buffered, bytes):
        raise backend.MlxQualificationError("MLX child transport buffer differs")
    while True:
        remaining_ns = None if deadline_ns is None else backend.enforce_deadline(deadline_ns)
        if sampler is not None:
            sampler.raise_if_failed()
        newline = buffered.find(b"\n")
        if newline >= 0:
            line = buffered[: newline + 1]
            process._modular_mlx_stdout_remainder = buffered[newline + 1 :]
            break
        if len(buffered) > CHILD_MESSAGE_MAX_BYTES:
            raise backend.MlxQualificationError("MLX child message exceeds transport bound")
        timeout = 0.25 if remaining_ns is None else min(0.25, remaining_ns / 1_000_000_000)
        readable, _, _ = select.select([process.stdout], [], [], timeout)
        if readable:
            chunk = os.read(process.stdout.fileno(), CHILD_MESSAGE_READ_BYTES)
            if chunk:
                buffered += chunk
                continue
            if buffered:
                raise backend.MlxQualificationError("MLX child message is not newline terminated")
            raise backend.MlxQualificationError("MLX child transport closed")
        if process.poll() is not None:
            if buffered:
                raise backend.MlxQualificationError("MLX child message is not newline terminated")
            raise backend.MlxQualificationError("MLX child exited without a terminal message")
    if len(line) > CHILD_MESSAGE_MAX_BYTES:
        raise backend.MlxQualificationError("MLX child message exceeds transport bound")
    if not line.endswith(b"\n"):
        raise backend.MlxQualificationError("MLX child message is not newline terminated")
    try:
        raw = line.decode("utf-8")
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise backend.MlxQualificationError("MLX child message encoding differs") from error
    if not isinstance(value, dict):
        raise backend.MlxQualificationError("MLX child message differs")
    return value


def model_name(stage: str) -> str:
    if stage == "donor":
        return "all_eligible_donor"
    if stage.startswith("dense"):
        return "dense_causal"
    if stage == "rung_two":
        return "rung_two"
    return "selected"


def learning_rate_peaks(stage: str, seed: int) -> list[tuple[str, float]]:
    from src.model.modular_neural_machine import ModularNeuralMachine, rung_one_config, rung_two_config

    model = ModularNeuralMachine(rung_two_config() if stage == "rung_two" else rung_one_config("dense" if stage.startswith("dense") else "all_eligible" if stage == "donor" else "selected"))
    _, groups, _ = cpu._make_optimizer(model, stage, cpu._import_runtime())
    return [(record["parameter_group"], record["peak_lr"]) for record in groups]


def attempt_metrics(stage: str, request: Mapping[str, Any], logical_update: int, observed: Mapping[str, Any], peaks: list[tuple[str, float]]) -> dict[str, Any]:
    if logical_update <= request["warmup_updates"]:
        multiplier = logical_update / request["warmup_updates"]
    else:
        import math

        multiplier = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (logical_update - request["warmup_updates"]) / (request["updates"] - request["warmup_updates"])))
    return {
        "learning_rates": [{"parameter_group": name, "learning_rate": peak * multiplier} for name, peak in peaks],
        "component_losses": {"task_loss": observed["task_loss"], "internal_router_loss": observed["internal_router_loss"], "supervised_route_loss": observed["supervised_route_loss"]},
        "total_loss": observed["total_loss"],
        "gradient_norm": observed["gradient_norm"],
        "clip_result": observed["clip_result"],
        "raw_overflow_count": observed["raw_overflow_count"],
        "max_bucket_load": observed["max_bucket_load"],
        "elapsed_seconds": observed["elapsed_seconds"],
        "finite": observed["finite"],
    }


def write_attempt_batch(rows: list[dict[str, Any]], writers: Mapping[int, Any], pending_signal: Any = None) -> tuple[list[dict[str, Any]], cpu.AppendResult | None]:
    committed = []
    for row in rows:
        try:
            result = writers[row["construction_seed"]].append(row, pending_signal=pending_signal)
        except cpu.LedgerAppendError as error:
            result = error.result
        if result.committed:
            committed.append(row)
        if not result.acknowledged:
            return committed, result
    return committed, None


def require_no_pending_signal(signals: Any, stage: str) -> None:
    pending = signals.pending_signal
    if pending is not None:
        raise cpu.HardAbort("signal_or_interruption", {"signal": pending, "stage": stage})


def send_guarded_ack(process: Any, message: Mapping[str, Any], deadline_ns: int | None, signals: Any, stage: str) -> None:
    guarded = signals.commit_guarded(lambda: send(process, message, deadline_ns))
    if not guarded.committed or guarded.pending_signal is not None:
        raise cpu.HardAbort("signal_or_interruption", {"signal": guarded.pending_signal, "stage": stage})


def seed_resource_sample_ids(rows: list[Mapping[str, Any]]) -> dict[int, list[int]]:
    result = {seed: cpu.resource_sample_ids_for_seed(rows, seed) for seed in (*backend.RUNG_ONE_SEEDS, backend.RUNG_TWO_SEED)}
    if any(not sample_ids for sample_ids in result.values()):
        raise cpu.HardAbort("resource_sampler_failure", {"surface": "mlx_seed_resource_evidence"})
    return result


def commit_clean_child_close(process: Any, deadline_ns: int, claim_result: Mapping[str, Any], clean_finalizer: Any) -> int:
    clean_finalizer(claim_result)
    send(process, {"ack": True, "kind": "close_committed"}, deadline_ns)
    remaining_seconds = backend.enforce_deadline(deadline_ns) / 1_000_000_000
    return_code = process.wait(timeout=min(30.0, remaining_seconds))
    backend.enforce_deadline(deadline_ns)
    if return_code != 0:
        raise backend.MlxQualificationError(f"MLX child exit differs: {return_code}")
    return return_code


def write_canonical_training_artifacts(root: Path, rows_by_seed: Mapping[int, list[dict[str, Any]]]) -> None:
    for seed, rows in rows_by_seed.items():
        cpu.validate_attempt_sequence(rows, require_complete=True)
        expected = 3072 if seed == backend.RUNG_TWO_SEED else 7680
        if len(rows) != expected:
            raise backend.MlxQualificationError("canonical attempt cardinality differs")
        target = root / ("rung2" if seed == backend.RUNG_TWO_SEED else "rung1") / str(seed) / "attempts.jsonl"
        if cpu._canonical_jsonl_records(target) != rows:
            raise backend.MlxQualificationError("live canonical attempt ledger differs")
        train_rows = []
        for index in range(0, len(rows), 2):
            started_row = rows[index]
            completed_row = rows[index + 1]
            first_hash = started_row["batch_sha256"] if started_row["logical_update"] == 1 else None
            train_rows.append(cpu._train_row_from_pair(started_row, completed_row, first_hash))
        cpu._write_canonical_jsonl(target.parent / "train.jsonl", train_rows)


def write_canonical_completion(root: Path, claim_start_monotonic_ns: int, resource_rows: list[dict[str, Any]]) -> dict[str, Any]:
    accounting = cpu._accounting_from_run_root(root)
    if accounting.unpaired_attempts or accounting.attempted_updates != accounting.completed_updates or accounting.attempted_token_positions != accounting.completed_token_positions:
        raise backend.MlxQualificationError("canonical ledger completion differs")
    wall_end = time.monotonic_ns()
    completion = {
        "schema_version": cpu.SCHEMA_VERSION,
        "run_id": root.name,
        "claim_start_monotonic_ns": claim_start_monotonic_ns,
        "resource_sampling_end_monotonic_ns": resource_rows[-1]["monotonic_ns"],
        "wall_accounting_end_monotonic_ns": wall_end,
        "claim_elapsed_seconds": (wall_end - claim_start_monotonic_ns) / 1_000_000_000,
        "resource_final_sample_id": resource_rows[-1]["sample_id"],
        "attempted_updates": accounting.attempted_updates,
        "completed_updates": accounting.completed_updates,
        "token_positions": accounting.completed_token_positions,
        "packaging_excluded": ["run/completion.json", "summary.json", "SHA256SUMS"],
    }
    cpu.write_canonical_json(root / "run" / "completion.json", completion)
    return completion


def expected_child_dependency_hashes(root: Path) -> dict[str, str]:
    manifest = cpu._canonical_json_artifact(root / "run" / "source_manifest.json")
    by_path = {record["path"]: record["sha256"] for record in manifest["records"]}
    paths = {
        "backend": "src/model/modular_mlx_backend.py",
        "cpu_evaluator": "neuroloc/simulations/memory/modular_sequence_role_cpu.py",
        "engine": "neuroloc/simulations/memory/modular_sequence_role_mlx.py",
        "model": "src/model/modular_neural_machine.py",
        "qualifier": "scripts/qualify_modular_mlx.py",
        "sources": "src/model/modular_sources.py",
    }
    if any(path not in by_path for path in paths.values()):
        raise backend.MlxQualificationError("MLX child dependency anchor is absent")
    return {name: by_path[path] for name, path in paths.items()}


def measure_durable_attempt_ledgers(run_id: str, scratch: Path, sample_pairs: int = 32) -> float:
    if type(sample_pairs) is not int or sample_pairs < 1:
        raise backend.MlxQualificationError("durable ledger sample count differs")
    path = scratch / "attempt-ledger-benchmark.jsonl"
    writer = cpu.CrashAtomicJsonlWriter(path, cpu.validate_attempt_row, sequence_kind="attempt")
    started_ns = time.perf_counter_ns()
    writer.precreate()
    for logical_update in range(1, sample_pairs + 1):
        batch_sha256 = hashlib.sha256(f"{run_id}:{logical_update}".encode("utf-8")).hexdigest()
        started = cpu._attempt_event(run_id, 1, 11, 2 * (logical_update - 1), "started", "all_eligible_donor", "donor", logical_update, 16, 2048, batch_sha256, None)
        metrics = {"learning_rates": [{"parameter_group": "all_trainable_decay", "learning_rate": 0.002}], "component_losses": {"task_loss": 1.0, "internal_router_loss": None, "supervised_route_loss": None}, "total_loss": 1.0, "gradient_norm": 1.0, "clip_result": "unchanged", "raw_overflow_count": 0, "max_bucket_load": 1, "elapsed_seconds": 0.001, "finite": True}
        completed = cpu._attempt_event(run_id, 1, 11, 2 * logical_update - 1, "completed", "all_eligible_donor", "donor", logical_update, 16, 2048, batch_sha256, metrics)
        for row in (started, completed):
            result = writer.append(row)
            if not result.acknowledged:
                raise backend.MlxQualificationError("durable ledger benchmark append differs")
    writer.close()
    elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    if cpu._canonical_jsonl_records(path)[-1]["event_sequence"] != 2 * sample_pairs - 1:
        raise backend.MlxQualificationError("durable ledger benchmark readback differs")
    return elapsed * backend.ATTEMPT_EVENT_ROWS / (2 * sample_pairs)


def run_training(root: Path, process: subprocess.Popen[str], sampler: Any, deadline_ns: int, claim_writers: Mapping[str, Any], signals: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = backend.protocol_state()
    hello = receive(process, sampler, deadline_ns)
    backend.validate_child_message(hello, state)
    dependency_hashes = expected_child_dependency_hashes(root)
    if hello["engine_sha256"] != dependency_hashes["engine"]:
        raise backend.MlxQualificationError("MLX child engine hash differs")
    self_check = hello["self_check"]
    if hello["dependency_sha256s"] != dependency_hashes:
        raise backend.MlxQualificationError("MLX self-check anchor differs")
    backend.validate_initial_self_check(self_check)
    writers = {}
    for seed in (*backend.RUNG_ONE_SEEDS, backend.RUNG_TWO_SEED):
        relative = f"rung2/83/attempts.jsonl" if seed == backend.RUNG_TWO_SEED else f"rung1/{seed}/attempts.jsonl"
        if relative not in claim_writers:
            raise backend.MlxQualificationError("canonical attempt writer is absent")
        writers[seed] = claim_writers[relative]
    sequences = {seed: 0 for seed in (*backend.RUNG_ONE_SEEDS, backend.RUNG_TWO_SEED)}
    rows_by_seed = {seed: [] for seed in sequences}
    stage_times = []
    attempted_updates = 0
    completed_updates = 0
    token_positions = 0
    stage_order = [("donor", seed) for seed in backend.RUNG_ONE_SEEDS] + [(stage, None) for stage in ("router_only", "joint", "dense_base", "dense_continuation", "rung_two")]
    for parent_sequence, (stage, seed) in enumerate(stage_order):
        backend.enforce_deadline(deadline_ns)
        require_no_pending_signal(signals, "before_mlx_stage_request")
        request = backend.stage_request(stage, [seed] if seed is not None else None, parent_sequence)
        backend.bind_stage_request(state, request)
        send(process, request, deadline_ns)
        peaks = learning_rate_peaks(stage, request["construction_seeds"][0])
        stage_started = time.perf_counter_ns()
        while True:
            message = receive(process, sampler, deadline_ns)
            kind = backend.validate_child_message(message, state)
            if kind == "hard_abort":
                raise backend.MlxQualificationError(f"MLX child hard abort: {message['reason']}: {message['message']}")
            if kind == "stage_started":
                sampler.begin_stage(stage, list(message["construction_seeds"]))
                send_guarded_ack(process, {"ack": True, "kind": "stage_start_committed", "stage": stage, "logical_update": None}, deadline_ns, signals, "mlx_stage_start_ack")
            elif kind == "update_ready":
                attempt_rows = []
                for construction_seed, batch_hash in zip(message["construction_seeds"], message["batch_sha256s"]):
                    row = cpu._attempt_event(root.name, 2 if stage == "rung_two" else 1, construction_seed, sequences[construction_seed], "started", model_name(stage), stage, message["logical_update"], request["batch_size"], 4096 if stage == "rung_two" else 2048, batch_hash, None)
                    attempt_rows.append(row)
                committed, failure = write_attempt_batch(attempt_rows, writers, lambda: signals.pending_signal is not None)
                for row in committed:
                    rows_by_seed[row["construction_seed"]].append(row)
                attempted_updates += len(committed)
                token_positions += (4096 if stage == "rung_two" else 2048) * len(committed)
                if committed:
                    sampler.observe_started(stage, [row["construction_seed"] for row in committed], message["logical_update"], attempted_updates, token_positions)
                if failure is not None:
                    raise cpu.HardAbort(failure.reason_code or "artifact_inconsistency", {"stage": stage, "logical_update": message["logical_update"], "committed_lanes": len(committed)})
                send_guarded_ack(process, {"ack": True, "kind": "update_start_committed", "stage": stage, "logical_update": message["logical_update"]}, deadline_ns, signals, "mlx_update_start_ack")
            elif kind == "update_complete":
                attempt_rows = []
                for construction_seed, batch_hash, observed in zip(message["construction_seeds"], message["batch_sha256s"], message["metrics"]):
                    row = cpu._attempt_event(root.name, 2 if stage == "rung_two" else 1, construction_seed, sequences[construction_seed] + 1, "completed", model_name(stage), stage, message["logical_update"], request["batch_size"], 4096 if stage == "rung_two" else 2048, batch_hash, attempt_metrics(stage, request, message["logical_update"], observed, peaks))
                    attempt_rows.append(row)
                committed, failure = write_attempt_batch(attempt_rows, writers, lambda: signals.pending_signal is not None)
                for row in committed:
                    construction_seed = row["construction_seed"]
                    rows_by_seed[construction_seed].append(row)
                    sequences[construction_seed] += 2
                completed_updates += len(committed)
                if failure is not None:
                    raise cpu.HardAbort(failure.reason_code or "artifact_inconsistency", {"stage": stage, "logical_update": message["logical_update"], "committed_lanes": len(committed)})
                if message["logical_update"] == 1 or message["logical_update"] % 128 == 0 or message["logical_update"] == request["updates"]:
                    print(json.dumps({"stage": stage, "logical_update": message["logical_update"], "stage_elapsed_seconds": (time.perf_counter_ns() - stage_started) / 1_000_000_000}, sort_keys=True), flush=True)
                sampler.observe_progress(stage, list(message["construction_seeds"]), message["logical_update"], attempted_updates, token_positions, message["memory"])
                send_guarded_ack(process, {"ack": True, "kind": "update_complete_committed", "stage": stage, "logical_update": message["logical_update"]}, deadline_ns, signals, "mlx_update_complete_ack")
            elif kind == "stage_complete":
                for relative, expected in zip(message["checkpoint_paths"], message["checkpoint_sha256s"]):
                    if cpu.sha256_file(root / relative) != expected:
                        raise backend.MlxQualificationError("checkpoint readback hash differs")
                send_guarded_ack(process, {"ack": True, "kind": "stage_complete_committed", "stage": stage, "logical_update": None}, deadline_ns, signals, "mlx_stage_complete_ack")
                stage_times.append({"stage": stage, "seed": seed, "wall_seconds": (time.perf_counter_ns() - stage_started) / 1_000_000_000})
                break
    sampler.clear_active_jobs(attempted_updates, token_positions)
    resource_sample_ids_by_seed = seed_resource_sample_ids(sampler.snapshot_rows())
    write_canonical_training_artifacts(root, rows_by_seed)
    require_no_pending_signal(signals, "before_mlx_evaluation")
    send(process, {"kind": "evaluate", "resource_sample_ids_by_seed": {str(seed): sample_ids for seed, sample_ids in resource_sample_ids_by_seed.items()}}, deadline_ns)
    evaluated = receive(process, sampler, deadline_ns)
    if backend.validate_child_message(evaluated, state) != "evaluation_complete":
        raise backend.MlxQualificationError("MLX child did not complete evaluation")
    require_no_pending_signal(signals, "before_mlx_child_close")
    send(process, {"kind": "close"}, deadline_ns)
    closed = receive(process, sampler, deadline_ns)
    if backend.validate_child_message(closed, state) != "closed":
        raise backend.MlxQualificationError("MLX child did not close cleanly")
    sampler.clear_active_jobs(attempted_updates, token_positions)
    for writer in writers.values():
        writer.validate_committed_prefix()
    return {"stage_times": stage_times, "attempt_rows": sum(len(rows) for rows in rows_by_seed.values()), "evaluation": evaluated["result"], "resource_sample_ids_by_seed": resource_sample_ids_by_seed}, rows_by_seed[backend.RUNG_TWO_SEED]


def terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def attach_cleanup_failure(primary_error: BaseException, cleanup_error: BaseException) -> None:
    failures = (*getattr(primary_error, "cleanup_failures", ()), cleanup_error)
    primary_error.cleanup_failures = failures
    if hasattr(primary_error, "add_note"):
        primary_error.add_note(f"Cleanup failure preserved under primary exception: {cleanup_error!r}")


def perform_cleanup_operations(operations: tuple[Any, ...], primary_error: BaseException | None) -> None:
    failures = []
    for operation in operations:
        try:
            operation()
        except BaseException as cleanup_error:
            failures.append(cleanup_error)
    if primary_error is not None:
        for cleanup_error in failures:
            attach_cleanup_failure(primary_error, cleanup_error)
    elif failures:
        for cleanup_error in failures[1:]:
            attach_cleanup_failure(failures[0], cleanup_error)
        raise failures[0]


def cleanup_scratch(scratch: Path | None) -> bool:
    if scratch is None or not scratch.exists():
        return True
    shutil.rmtree(scratch)
    return not scratch.exists()


def cleanup_after_primary_failure(process: subprocess.Popen[str] | None, sampler: backend.QualificationResourceSampler | None, primary_error: BaseException) -> None:
    operations = [lambda: terminate(process)]
    if sampler is not None:
        operations.append(sampler.stop)
    perform_cleanup_operations(tuple(operations), primary_error)


def pilot_self_check_details_sha256(root: Path) -> str:
    preflight = cpu._canonical_json_artifact(root / "run" / "preflight.json")
    records = [record for record in preflight["trained_backend"] if record["name"] == "mlx_self_check"]
    if len(records) != 1:
        raise backend.MlxQualificationError("pilot self-check preflight record differs")
    digest = records[0]["details_sha256"]
    cpu._canonical_json_artifact(root / "run" / "check_details" / f"{digest}.json")
    return digest


def validate_pilot_hello(message: Mapping[str, Any], root: Path) -> None:
    expected_keys = ("dependency_sha256s", "device", "engine_sha256", "kind", "mlx_version", "runtime", "schema_version", "self_check", "sequence")
    if tuple(sorted(message)) != expected_keys or message["kind"] != "pilot_hello" or message["sequence"] != 0 or message["schema_version"] != backend.IPC_SCHEMA_VERSION or message["mlx_version"] != backend.MLX_VERSION or message["device"] != "Device(gpu, 0)":
        raise backend.MlxQualificationError("pilot hello identity differs")
    dependency_hashes = expected_child_dependency_hashes(root)
    if message["dependency_sha256s"] != dependency_hashes or message["engine_sha256"] != dependency_hashes["engine"]:
        raise backend.MlxQualificationError("pilot hello dependency hashes differ")
    self_check = message["self_check"]
    if not isinstance(self_check, Mapping):
        raise backend.MlxQualificationError("pilot hello self-check differs")
    backend.validate_initial_self_check(self_check)
    runtime = message["runtime"]
    expected_runtime = {
        "python_path": str(backend.MLX_PYTHON),
        "python_version": "3.9.6",
        "mlx_version": backend.MLX_VERSION,
        "device": "Device(gpu, 0)",
        "training_dtype": "float32",
        "compilation": "mx.compile",
        "vectorization": "mx.vmap",
    }
    if runtime != expected_runtime:
        raise backend.MlxQualificationError("pilot runtime differs")


def validate_pilot_memory(value: Any) -> dict[str, Any]:
    expected = ("active_memory_bytes", "cache_memory_bytes", "parent_rss_and_swap_required", "peak_memory_bytes")
    if not isinstance(value, Mapping) or tuple(sorted(value)) != expected or value["parent_rss_and_swap_required"] is not True:
        raise backend.MlxQualificationError("pilot memory record differs")
    if any(type(value[name]) is not int or value[name] < 0 for name in expected if name != "parent_rss_and_swap_required"):
        raise backend.MlxQualificationError("pilot memory value differs")
    return dict(value)


def validate_pilot_workload_record(record: Any, ordinal: int) -> dict[str, Any]:
    expected_keys = ("data_seed", "execution", "metal_cache_released", "model_destroyed", "model_seed", "optimizer_destroyed", "route_seed", "timed_update_ns", "warmup_update_ns", "workload", "workload_ordinal")
    if not isinstance(record, Mapping) or tuple(sorted(record)) != expected_keys:
        raise backend.MlxQualificationError("pilot workload record keys differ")
    specification = PILOT_PROTOCOL.workloads[ordinal]
    model_seed = PILOT_PROTOCOL.model_seed(ordinal)
    if any(
        (
            record["workload"] != specification.name,
            record["workload_ordinal"] != ordinal,
            record["model_seed"] != model_seed,
            record["data_seed"] != PILOT_PROTOCOL.data_seed(ordinal),
            record["route_seed"] != PILOT_PROTOCOL.route_seed(ordinal),
            record["execution"] != specification.execution,
            record["model_destroyed"] is not True,
            record["optimizer_destroyed"] is not True,
            record["metal_cache_released"] is not True,
        )
    ):
        raise backend.MlxQualificationError("pilot workload record identity differs")
    warmup = record["warmup_update_ns"]
    timed = record["timed_update_ns"]
    if not isinstance(warmup, list) or len(warmup) != len(PILOT_PROTOCOL.warmup_updates) or not isinstance(timed, list) or len(timed) != len(PILOT_PROTOCOL.timed_updates) or len(warmup) + len(timed) != PILOT_PROTOCOL.updates_per_workload or any(type(value) is not int or value <= 0 for value in (*warmup, *timed)):
        raise backend.MlxQualificationError("pilot workload durations differ")
    return dict(record)


def validate_pilot_routing_row(row: Mapping[str, Any], run_id: str, seed: int) -> None:
    if tuple(sorted(row)) != backend.ROUTING_ROW_KEYS:
        raise backend.MlxQualificationError("pilot routing benchmark schema differs")
    if row["schema_version"] != cpu.SCHEMA_VERSION or row["run_id"] != run_id or row["rung"] != 1 or row["claim_seed"] != seed or row["construction_seed"] != seed or row["phase"] != "training" or row["model"] != "selected" or row["stage"] != "joint" or row["condition"] is not None or row["checkpoint_sha256"] is not None:
        raise backend.MlxQualificationError("pilot routing benchmark identity differs")
    if type(row["logical_update"]) is not int or row["logical_update"] < 1 or type(row["forward_sequence"]) is not int or row["forward_sequence"] < 1 or row["block"] not in (0, 4):
        raise backend.MlxQualificationError("pilot routing benchmark sequence differs")
    if row["row_kind"] == "call_summary":
        if row["example_index"] is not None or row["query_position"] is not None or row["raw_remote_ids"] is not None or row["effective_remote_ids"] is not None or not isinstance(row["canonical_bypass_ids"], list) or not isinstance(row["block_load_histogram"], list) or not isinstance(row["valid_posting_histogram"], list):
            raise backend.MlxQualificationError("pilot routing call summary differs")
        if any(type(row[name]) is not int or row[name] < 0 for name in ("addresses_probed", "posting_reads", "candidate_blocks", "overflow_count", "max_bucket_load", "route_workspace_bytes")):
            raise backend.MlxQualificationError("pilot routing call counters differ")
    elif row["row_kind"] == "query_example":
        if type(row["batch_index"]) is not int or row["batch_index"] < 0 or type(row["example_index"]) is not int or row["example_index"] < 0 or row["query_position"] != 126 or not isinstance(row["raw_remote_ids"], list) or not isinstance(row["effective_remote_ids"], list) or row["local_block_ids"] != [15] or type(row["query_underfill_count"]) is not int or row["query_underfill_count"] < 0:
            raise backend.MlxQualificationError("pilot routing query row differs")
        expected_hit = row["required_source"] in row["effective_remote_ids"]
        if row["original_source_hit"] is not expected_hit or row["foreign_source"] is not None or row["foreign_source_hit"] is not None:
            raise backend.MlxQualificationError("pilot routing query evidence differs")
    else:
        raise backend.MlxQualificationError("pilot routing row kind differs")


def pilot_routing_fixture(run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = cpu._import_runtime()
    torch = runtime.torch
    seed = 10_000_083
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = runtime.model_module.ModularNeuralMachine(runtime.model_module.rung_one_config("selected"))
    model.eval()
    batch_two = cpu.generate_rung_one_batch(123456, 2, torch)
    random_two = cpu.generate_random_routes(500011, 2, torch)
    tokens = torch.tensor(batch_two["tokens"], dtype=torch.long).repeat((8, 1))
    required_source = torch.tensor(batch_two["required_source"], dtype=torch.long).repeat(8)
    route_values = torch.tensor(random_two["routes"], dtype=torch.long).repeat((8, 1))
    route_override = torch.full((16, 128, 1, 2), -1, dtype=torch.long)
    route_override[:, 126, 0] = route_values
    with torch.inference_mode():
        output = model(tokens, return_aux=True, route_detail=True, request_block4_router_loss=True, route_override=route_override)
    microtrace = cpu._routing_rows(output, run_id, seed, "training", "selected", "joint", None, 1, 1, 0, 0, required_source, None, None, None)
    if len(microtrace) != 34:
        raise backend.MlxQualificationError("pilot routing microtrace cardinality differs")
    rows = []
    for copy_index in range(128):
        for source in microtrace:
            row = json.loads(json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False))
            row["logical_update"] = copy_index + 1
            row["forward_sequence"] = copy_index + 1
            row["batch_index"] = copy_index
            if row["row_kind"] == "query_example":
                row["example_index"] = copy_index * 16 + int(source["example_index"])
                row["original_source_hit"] = row["required_source"] in row["effective_remote_ids"]
                row["query_underfill_count"] = sum(value == -1 for value in row["effective_remote_ids"])
            validate_pilot_routing_row(row, run_id, seed)
            rows.append(row)
    if len(rows) != 4_352:
        raise backend.MlxQualificationError("pilot routing block cardinality differs")
    serialized = [json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n" for row in rows]
    detail = {
        "fixture_sha256s": {
            "random_routes_seed_500011": cpu.canonical_json_sha256(random_two),
            "routing_block": hashlib.sha256(b"".join(serialized)).hexdigest(),
            "rung_one_seed_123456": cpu.canonical_json_sha256(batch_two),
        },
        "block_uncompressed_bytes": sum(len(value) for value in serialized),
    }
    return rows, detail


def measure_routing_evidence(run_id: str, scratch: Path) -> tuple[float, dict[str, Any], int]:
    rows, detail = pilot_routing_fixture(run_id)
    serialized = [json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n" for row in rows]
    durations = []
    raw_sizes = []
    cleanup = []
    for repetition in range(4):
        directory = scratch / f"routing-evidence-{repetition}"
        directory.mkdir(exist_ok=False)
        cpu.fsync_directory(directory.parent)
        path = directory / "routing.jsonl.gz"
        started_ns = time.perf_counter_ns()
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                for line in serialized:
                    compressed.write(line)
            raw.flush()
            os.fsync(raw.fileno())
        cpu.fsync_directory(directory)
        raw_bytes = path.read_bytes()
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        observed = []
        with gzip.open(path, "rb") as handle:
            for source_line in handle:
                row = json.loads(source_line)
                validate_pilot_routing_row(row, run_id, 10_000_083)
                reencoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8") + b"\n"
                if reencoded != source_line:
                    raise backend.MlxQualificationError("pilot routing canonical readback differs")
                observed.append(source_line)
        if len(observed) != 4_352 or hashlib.sha256(path.read_bytes()).hexdigest() != raw_digest or observed != serialized:
            raise backend.MlxQualificationError("pilot routing readback differs")
        raw_sizes.append(len(raw_bytes))
        path.unlink()
        cpu.fsync_directory(directory)
        directory.rmdir()
        cpu.fsync_directory(scratch)
        duration = time.perf_counter_ns() - started_ns
        if duration <= 0:
            raise backend.MlxQualificationError("pilot routing duration differs")
        durations.append(duration)
        cleanup.append(not directory.exists())
    maximum_ns = max(durations[1:])
    seconds = 136 * maximum_ns / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0 or not all(cleanup):
        raise backend.MlxQualificationError("pilot routing projection differs")
    max_line_bytes = max(len(value) for value in serialized)
    detail = {
        "fixture_sha256s": detail["fixture_sha256s"],
        "warmup_duration_ns": {"routing_evidence_block": [durations[0]]},
        "timed_duration_ns": {"routing_evidence_block": durations[1:]},
        "selected_max_duration_ns": {"routing_evidence_block": maximum_ns},
        "counts": {"block_copies": 128, "block_rows": 4_352, "claim_rows": backend.ROUTING_EVIDENCE_ROWS, "microtrace_rows": 34, "projected_rows": 136 * 4_352, "scale_blocks": 136},
        "byte_sizes": {"block_uncompressed_bytes": detail["block_uncompressed_bytes"], "max_line_bytes": max_line_bytes, "raw_gzip_bytes_max": max(raw_sizes)},
        "scaling": {"scale_blocks": 136, "selected_max_duration_ns": maximum_ns, "total_ns": 136 * maximum_ns},
        "scratch_cleanup_pass": all(cleanup),
        "component_seconds": seconds,
    }
    validate_tail_benchmark_detail("routing_evidence", detail, seconds, expected_tail_fixture_sha256s("routing_evidence", routing_block_sha256=detail["fixture_sha256s"]["routing_block"]))
    return seconds, detail, max_line_bytes


def canonical_line_bytes(value: Mapping[str, Any]) -> int:
    return len(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")) + 1


def packaging_line_bounds(run_id: str, routing_max_line_bytes: int) -> dict[str, int]:
    bounded_run_id = (run_id + "r" * 64)[:64]
    metrics = {
        "learning_rates": [
            {"parameter_group": "all_trainable_decay", "learning_rate": 0.002},
            {"parameter_group": "all_trainable_nondecay", "learning_rate": 0.002},
        ],
        "component_losses": {"task_loss": 1.0, "internal_router_loss": 1.0, "supervised_route_loss": 1.0},
        "total_loss": 3.0,
        "gradient_norm": 1.0,
        "clip_result": "unchanged",
        "raw_overflow_count": 0,
        "max_bucket_load": 1,
        "elapsed_seconds": 1200.0,
        "finite": True,
    }
    started = cpu._attempt_event(bounded_run_id, 1, 10_000_083, 0, "started", "selected", "dense_continuation", 1, 16, 2048, "f" * 64, None)
    completed = cpu._attempt_event(bounded_run_id, 1, 10_000_083, 1, "completed", "selected", "dense_continuation", 1, 16, 2048, "f" * 64, metrics)
    cpu.validate_attempt_sequence((started, completed), require_complete=True)
    train = cpu._train_row_from_pair(started, completed, "f" * 64)
    payload = cpu.load_prereg_payload()
    cpu.validate_exact_keys(train, payload["artifacts"]["schemas"]["train_row"]["exact_keys"], "pilot packaging train row")
    prediction_keys = payload["artifacts"]["schemas"]["prediction_row"]["exact_keys"]
    prediction_full = {
        "schema_version": cpu.SCHEMA_VERSION,
        "run_id": bounded_run_id,
        "rung": 1,
        "claim_seed": 10_000_083,
        "construction_seed": 10_000_083,
        "condition": "required_source_excluded",
        "example_index": 511,
        "original_condition": 3,
        "foreign_condition": 3,
        "original_source": 14,
        "foreign_source": 14,
        "target": 127,
        "prediction": 127,
        "correct": True,
        "original_source_hit": True,
        "foreign_source_hit": True,
        "condition_stratum": "changed_condition",
        "checkpoint_sha256": "f" * 64,
    }
    prediction_null = dict(prediction_full)
    for name in ("original_condition", "foreign_condition", "original_source", "foreign_source", "original_source_hit", "foreign_source_hit"):
        prediction_null[name] = None
    cpu.validate_exact_keys(prediction_full, prediction_keys, "pilot packaging prediction row")
    cpu.validate_exact_keys(prediction_null, prediction_keys, "pilot packaging prediction row")
    route_values = [[14, 13] for _ in range(512)]
    values = {
        "answer_correct": 512,
        "answer_total": 512,
        "original_source_hits": None,
        "original_source_total": None,
        "foreign_source_hits": None,
        "foreign_source_total": None,
        "raw_remote_ids": route_values,
        "effective_remote_ids": route_values,
        "query_underfill_count": 0,
        "overflow_count": 0,
        "max_bucket_load": 512,
        "selected_mask_oracle_max_error": None,
    }
    evaluation_full = cpu._evaluation_row(
        bounded_run_id,
        10_000_083,
        "required_source_excluded",
        "answer_accuracy",
        "changed_condition",
        list(range(512)),
        values,
        "f" * 64,
        "e" * 64,
        ["f" * 64, "e" * 64, "d" * 64],
        "r1.required_source_excluded.answer_accuracy.changed_condition",
        (">=", 0.95, 487, "rate"),
        1200.0,
        list(range(241)),
    )
    cpu.validate_exact_keys(evaluation_full, payload["artifacts"]["schemas"]["evaluation_row"]["exact_keys"], "pilot packaging evaluation row")
    bounds = {
        "attempt_max_line_bytes": max(canonical_line_bytes(started), canonical_line_bytes(completed)),
        "train_max_line_bytes": canonical_line_bytes(train),
        "routing_max_line_bytes": routing_max_line_bytes,
        "prediction_max_line_bytes": max(canonical_line_bytes(prediction_full), canonical_line_bytes(prediction_null)),
        "evaluation_max_line_bytes": canonical_line_bytes(evaluation_full),
    }
    if any(type(value) is not int or value <= 0 or value > 8 * 1024**2 for value in bounds.values()):
        raise backend.MlxQualificationError("pilot packaging line bound differs")
    return bounds


def preflight_detail_count(run_root: Path) -> tuple[int, list[dict[str, Any]]]:
    preflight = cpu._canonical_json_artifact(run_root / "run" / "preflight.json")
    digests = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                if name == "details_sha256":
                    if not isinstance(child, str) or len(child) != 64:
                        raise backend.MlxQualificationError("pilot preflight detail digest differs")
                    digests.add(child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(preflight)
    records = []
    for digest in sorted(digests):
        path = run_root / "run" / "check_details" / f"{digest}.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or len(raw) > 1024**2:
            raise backend.MlxQualificationError("pilot preflight detail bound differs")
        cpu._canonical_json_artifact(path)
        records.append({"sha256": digest, "bytes": len(raw)})
    if not records:
        raise backend.MlxQualificationError("pilot preflight details are absent")
    return len(records), records


def packaging_block() -> bytes:
    digest = hashlib.sha256(b"todorov_mlx_packaging_sentinel").hexdigest().encode("ascii")
    pattern = digest + (0).to_bytes(8, "big", signed=False)
    size = 16_777_216
    return (pattern * math.ceil(size / len(pattern)))[:size]


def measure_packaging_block(scratch: Path, block: bytes, repetition: int) -> tuple[int, bool]:
    directory = scratch / f"packaging-block-{repetition}"
    directory.mkdir(exist_ok=False)
    cpu.fsync_directory(directory.parent)
    path = directory / "block.bin"
    expected = hashlib.sha256(block).hexdigest()
    started_ns = time.perf_counter_ns()
    with path.open("xb") as handle:
        for offset in range(0, len(block), 1_048_576):
            handle.write(block[offset : offset + 1_048_576])
        handle.flush()
        os.fsync(handle.fileno())
    cpu.fsync_directory(directory)
    observed = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1_048_576)
            if not chunk:
                break
            observed.update(chunk)
    if observed.hexdigest() != expected:
        raise backend.MlxQualificationError("pilot packaging block readback differs")
    path.unlink()
    cpu.fsync_directory(directory)
    directory.rmdir()
    cpu.fsync_directory(scratch)
    duration = time.perf_counter_ns() - started_ns
    return duration, not directory.exists()


def measure_packaging_files(scratch: Path, repetition: int) -> tuple[int, bool]:
    directory = scratch / f"packaging-files-{repetition}"
    directory.mkdir(exist_ok=False)
    cpu.fsync_directory(directory.parent)
    started_ns = time.perf_counter_ns()
    paths = []
    for index in range(32):
        path = directory / f"{index:02d}.bin"
        with path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        paths.append(path)
    cpu.fsync_directory(directory)
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(b"").hexdigest():
            raise backend.MlxQualificationError("pilot packaging file readback differs")
    for path in paths:
        path.unlink()
    cpu.fsync_directory(directory)
    directory.rmdir()
    cpu.fsync_directory(scratch)
    duration = time.perf_counter_ns() - started_ns
    return duration, not directory.exists()


def measure_packaging(run_root: Path, scratch: Path, projected_checkpoint_bytes: int, routing_max_line_bytes: int) -> tuple[float, dict[str, Any]]:
    if type(projected_checkpoint_bytes) is not int or projected_checkpoint_bytes <= 0:
        raise backend.MlxQualificationError("pilot projected checkpoint bytes differ")
    detail_count, detail_records = preflight_detail_count(run_root)
    line_bounds = packaging_line_bounds(run_root.name, routing_max_line_bytes)
    bulk_count = 26 + 6 + 6 + 5 + 6 + 6
    remaining_count = 144 - bulk_count
    if bulk_count != 55 or remaining_count != 89:
        raise backend.MlxQualificationError("pilot packaging file partition differs")
    projected_bytes = (
        projected_checkpoint_bytes
        + 41_472 * line_bounds["attempt_max_line_bytes"]
        + 20_736 * line_bounds["train_max_line_bytes"]
        + 588_240 * line_bounds["routing_max_line_bytes"]
        + 31_744 * line_bounds["prediction_max_line_bytes"]
        + 327 * line_bounds["evaluation_max_line_bytes"]
        + 89 * 8_388_608
        + (detail_count + 6 + 108) * 1_048_576
    )
    projected_files = 144 + detail_count + 6 + 108
    block = packaging_block()
    block_durations = []
    file_durations = []
    cleanup = []
    for repetition in range(4):
        block_duration, block_cleanup = measure_packaging_block(scratch, block, repetition)
        file_duration, file_cleanup = measure_packaging_files(scratch, repetition)
        if block_duration <= 0 or file_duration <= 0:
            raise backend.MlxQualificationError("pilot packaging duration differs")
        block_durations.append(block_duration)
        file_durations.append(file_duration)
        cleanup.append(block_cleanup and file_cleanup)
    block_maximum = max(block_durations[1:])
    file_maximum = max(file_durations[1:])
    block_count = math.ceil(projected_bytes / len(block))
    seconds = block_count * block_maximum / 1_000_000_000 + projected_files * (file_maximum / 32) / 1_000_000_000
    static_bounds_pass = max(line_bounds.values()) <= 8_388_608 and all(record["bytes"] <= 1_048_576 for record in detail_records)
    if not math.isfinite(seconds) or seconds <= 0 or not all(cleanup) or not static_bounds_pass:
        raise backend.MlxQualificationError("pilot packaging projection differs")
    detail = {
        "fixture_sha256s": {"empty_file": hashlib.sha256(b"").hexdigest(), "io_block": hashlib.sha256(block).hexdigest()},
        "warmup_duration_ns": {"file_batch": [file_durations[0]], "io_block": [block_durations[0]]},
        "timed_duration_ns": {"file_batch": file_durations[1:], "io_block": block_durations[1:]},
        "selected_max_duration_ns": {"file_batch": file_maximum, "io_block": block_maximum},
        "counts": {"attempt_rows": 41_472, "bulk_fixed_paths": bulk_count, "check_detail_count": detail_count + 6 + 108, "completed_update_rows": 20_736, "evaluation_rows": 327, "file_batch_size": 32, "fixed_clean_files": 144, "future_parity_details": 108, "future_pilot_tail_details": 6, "prediction_rows": 31_744, "preflight_detail_count": detail_count, "projected_files": projected_files, "remaining_fixed_paths": remaining_count, "routing_rows": 588_240, "scaled_io_blocks": block_count},
        "byte_sizes": {"attempt_max_line_bytes": line_bounds["attempt_max_line_bytes"], "check_detail_schema_bound_bytes": 1_048_576, "evaluation_max_line_bytes": line_bounds["evaluation_max_line_bytes"], "io_block_bytes": len(block), "nonbulk_schema_bound_bytes": 8_388_608, "prediction_max_line_bytes": line_bounds["prediction_max_line_bytes"], "preflight_detail_bytes": sum(record["bytes"] for record in detail_records), "projected_bytes": projected_bytes, "projected_checkpoint_bytes": projected_checkpoint_bytes, "routing_max_line_bytes": line_bounds["routing_max_line_bytes"], "train_max_line_bytes": line_bounds["train_max_line_bytes"]},
        "scaling": {"file_batch_divisor": 32, "file_batch_max_duration_ns": file_maximum, "file_batch_scaled_duration_numerator_ns": projected_files * file_maximum, "io_block_max_duration_ns": block_maximum, "io_block_total_ns": block_count * block_maximum, "total_duration_numerator_ns": block_count * block_maximum * 32 + projected_files * file_maximum},
        "scratch_cleanup_pass": all(cleanup),
        "component_seconds": seconds,
    }
    validate_tail_benchmark_detail("packaging", detail, seconds, expected_tail_fixture_sha256s("packaging"))
    return seconds, detail


def measure_pilot_fixed_components(run_root: Path, scratch: Path, projected_checkpoint_bytes: int) -> tuple[dict[str, float], dict[str, Any]]:
    routing_seconds, routing_detail, routing_max_line_bytes = measure_routing_evidence(run_root.name, scratch)
    durable_seconds = measure_durable_attempt_ledgers(run_root.name, scratch)
    packaging_seconds, packaging_detail = measure_packaging(run_root, scratch, projected_checkpoint_bytes, routing_max_line_bytes)
    return {
        "durable_ledger_seconds": durable_seconds,
        "routing_evidence_seconds": routing_seconds,
        "packaging_seconds": packaging_seconds,
    }, {
        "routing_evidence": routing_detail,
        "packaging": packaging_detail,
    }


def validate_tail_benchmark_detail(name: str, detail: Any, expected_component_seconds: Any, expected_fixture_sha256s: Mapping[str, str]) -> dict[str, Any]:
    if name not in TAIL_TIMING_FAMILIES or not isinstance(detail, Mapping) or tuple(sorted(detail)) != tuple(sorted(TAIL_DETAIL_KEYS)):
        raise backend.MlxQualificationError("pilot tail detail schema differs")
    hashes = detail["fixture_sha256s"]
    if not isinstance(hashes, Mapping) or tuple(sorted(hashes)) != tuple(sorted(TAIL_FIXTURE_KEYS[name])) or dict(hashes) != dict(expected_fixture_sha256s) or any(not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value) for value in hashes.values()):
        raise backend.MlxQualificationError("pilot tail fixture hashes differ")
    warmup = detail["warmup_duration_ns"]
    timed = detail["timed_duration_ns"]
    maxima = detail["selected_max_duration_ns"]
    families = TAIL_TIMING_FAMILIES[name]
    if not isinstance(warmup, Mapping) or not isinstance(timed, Mapping) or not isinstance(maxima, Mapping) or tuple(sorted(warmup)) != tuple(sorted(families)) or tuple(sorted(timed)) != tuple(sorted(families)) or tuple(sorted(maxima)) != tuple(sorted(families)):
        raise backend.MlxQualificationError("pilot tail timing registry differs")
    if any(not isinstance(values, list) or len(values) != 1 or type(values[0]) is not int or values[0] <= 0 for values in warmup.values()):
        raise backend.MlxQualificationError("pilot tail warmup differs")
    if any(not isinstance(values, list) or len(values) != 3 or any(type(value) is not int or value <= 0 for value in values) for values in timed.values()):
        raise backend.MlxQualificationError("pilot tail timed repetitions differ")
    if any(type(maxima[family]) is not int or maxima[family] <= 0 or maxima[family] != max(timed[family]) for family in families):
        raise backend.MlxQualificationError("pilot tail selected maximum differs")
    counts = detail["counts"]
    byte_sizes = detail["byte_sizes"]
    scaling = detail["scaling"]
    if not isinstance(counts, Mapping) or not counts or any(type(value) is not int or value <= 0 for value in counts.values()):
        raise backend.MlxQualificationError("pilot tail counts differ")
    if not isinstance(byte_sizes, Mapping) or not byte_sizes or any(type(value) is not int or value <= 0 for value in byte_sizes.values()):
        raise backend.MlxQualificationError("pilot tail byte sizes differ")
    if not isinstance(scaling, Mapping) or not scaling or any(type(value) is not int or value <= 0 for value in scaling.values()):
        raise backend.MlxQualificationError("pilot tail scaling differs")
    if detail["scratch_cleanup_pass"] is not True:
        raise backend.MlxQualificationError("pilot tail cleanup differs")
    if name == "routing_evidence":
        expected_counts = {"block_copies": 128, "block_rows": 4_352, "claim_rows": 588_240, "microtrace_rows": 34, "projected_rows": 591_872, "scale_blocks": 136}
        if counts != expected_counts or tuple(sorted(byte_sizes)) != ("block_uncompressed_bytes", "max_line_bytes", "raw_gzip_bytes_max"):
            raise backend.MlxQualificationError("pilot routing detail inputs differ")
        expected_scaling = {"scale_blocks": 136, "selected_max_duration_ns": maxima["routing_evidence_block"], "total_ns": 136 * maxima["routing_evidence_block"]}
        expected_seconds = expected_scaling["total_ns"] / 1_000_000_000
    elif name == "evaluation":
        expected_counts = {"route_acquisition_calls": 80, "rung_one_routed_calls": 880, "rung_one_routed_conditions": 11, "rung_one_dense_calls": 80, "rung_two_calls": 32, "rung_two_conditions": 2, "endpoint_replay_calls": 26, "endpoint_replay_roles": 6}
        if counts != expected_counts or byte_sizes != {"nonclaim_fixture_bytes": TAIL_EVALUATION_FIXTURE_BYTES}:
            raise backend.MlxQualificationError("pilot evaluation detail inputs differ")
        route = maxima["route_acquisition"]
        routed = max(value for family, value in maxima.items() if family.startswith("rung_one_routed."))
        dense = maxima["rung_one_dense"]
        rung_two = max(value for family, value in maxima.items() if family.startswith("rung_two."))
        endpoint = max(value for family, value in maxima.items() if family.startswith("endpoint_replay."))
        expected_scaling = {"route_acquisition_ns": 80 * route, "rung_one_routed_ns": 880 * routed, "rung_one_dense_ns": 80 * dense, "rung_two_ns": 32 * rung_two, "endpoint_replay_ns": 26 * endpoint, "total_ns": 80 * route + 880 * routed + 80 * dense + 32 * rung_two + 26 * endpoint}
        expected_seconds = expected_scaling["total_ns"] / 1_000_000_000
    elif name == "checkpoint_reload":
        expected_counts = {"donor_single_coefficient": 5, "router_only_vmap5_all_lanes_coefficient": 1, "joint_vmap5_all_lanes_coefficient": 1, "dense_vmap5_all_lanes_coefficient": 2, "rung_two_single_coefficient": 1, "trained_endpoint_files": 26}
        expected_byte_keys = ("dense_vmap5_all_lanes", "donor_single", "joint_vmap5_all_lanes", "router_only_vmap5_all_lanes", "rung_two_single", "projected_checkpoint_bytes")
        projected_bytes = 5 * byte_sizes["donor_single"] + byte_sizes["router_only_vmap5_all_lanes"] + byte_sizes["joint_vmap5_all_lanes"] + 2 * byte_sizes["dense_vmap5_all_lanes"] + byte_sizes["rung_two_single"]
        if counts != expected_counts or tuple(sorted(byte_sizes)) != tuple(sorted(expected_byte_keys)) or byte_sizes["projected_checkpoint_bytes"] != projected_bytes or any(byte_sizes[family] < minimum for family, minimum in TAIL_CHECKPOINT_TENSOR_BYTE_LOWER_BOUNDS.items()):
            raise backend.MlxQualificationError("pilot checkpoint detail inputs differ")
        expected_scaling = {"donor_single_ns": 5 * maxima["donor_single"], "router_only_vmap5_all_lanes_ns": maxima["router_only_vmap5_all_lanes"], "joint_vmap5_all_lanes_ns": maxima["joint_vmap5_all_lanes"], "dense_vmap5_all_lanes_ns": 2 * maxima["dense_vmap5_all_lanes"], "rung_two_single_ns": maxima["rung_two_single"], "total_ns": 5 * maxima["donor_single"] + maxima["router_only_vmap5_all_lanes"] + maxima["joint_vmap5_all_lanes"] + 2 * maxima["dense_vmap5_all_lanes"] + maxima["rung_two_single"]}
        expected_seconds = expected_scaling["total_ns"] / 1_000_000_000
    else:
        expected_count_keys = ("attempt_rows", "bulk_fixed_paths", "check_detail_count", "completed_update_rows", "evaluation_rows", "file_batch_size", "fixed_clean_files", "future_parity_details", "future_pilot_tail_details", "prediction_rows", "preflight_detail_count", "projected_files", "remaining_fixed_paths", "routing_rows", "scaled_io_blocks")
        expected_byte_keys = ("attempt_max_line_bytes", "check_detail_schema_bound_bytes", "evaluation_max_line_bytes", "io_block_bytes", "nonbulk_schema_bound_bytes", "prediction_max_line_bytes", "preflight_detail_bytes", "projected_bytes", "projected_checkpoint_bytes", "routing_max_line_bytes", "train_max_line_bytes")
        if tuple(sorted(counts)) != tuple(sorted(expected_count_keys)) or tuple(sorted(byte_sizes)) != tuple(sorted(expected_byte_keys)):
            raise backend.MlxQualificationError("pilot packaging detail inputs differ")
        if counts["attempt_rows"] != 41_472 or counts["bulk_fixed_paths"] != 55 or counts["completed_update_rows"] != 20_736 or counts["evaluation_rows"] != 327 or counts["file_batch_size"] != 32 or counts["fixed_clean_files"] != 144 or counts["future_parity_details"] != 108 or counts["future_pilot_tail_details"] != 6 or counts["prediction_rows"] != 31_744 or counts["remaining_fixed_paths"] != 89 or counts["routing_rows"] != 588_240:
            raise backend.MlxQualificationError("pilot packaging fixed counts differ")
        if counts["check_detail_count"] != counts["preflight_detail_count"] + counts["future_pilot_tail_details"] + counts["future_parity_details"] or counts["projected_files"] != counts["fixed_clean_files"] + counts["check_detail_count"]:
            raise backend.MlxQualificationError("pilot packaging file count differs")
        projected_bytes = byte_sizes["projected_checkpoint_bytes"]
        projected_bytes += counts["attempt_rows"] * byte_sizes["attempt_max_line_bytes"]
        projected_bytes += counts["completed_update_rows"] * byte_sizes["train_max_line_bytes"]
        projected_bytes += counts["routing_rows"] * byte_sizes["routing_max_line_bytes"]
        projected_bytes += counts["prediction_rows"] * byte_sizes["prediction_max_line_bytes"]
        projected_bytes += counts["evaluation_rows"] * byte_sizes["evaluation_max_line_bytes"]
        projected_bytes += counts["remaining_fixed_paths"] * byte_sizes["nonbulk_schema_bound_bytes"]
        projected_bytes += counts["check_detail_count"] * byte_sizes["check_detail_schema_bound_bytes"]
        scaled_blocks = math.ceil(projected_bytes / byte_sizes["io_block_bytes"])
        if byte_sizes["projected_bytes"] != projected_bytes or counts["scaled_io_blocks"] != scaled_blocks:
            raise backend.MlxQualificationError("pilot packaging byte projection differs")
        expected_scaling = {
            "file_batch_divisor": counts["file_batch_size"],
            "file_batch_max_duration_ns": maxima["file_batch"],
            "file_batch_scaled_duration_numerator_ns": counts["projected_files"] * maxima["file_batch"],
            "io_block_max_duration_ns": maxima["io_block"],
            "io_block_total_ns": counts["scaled_io_blocks"] * maxima["io_block"],
            "total_duration_numerator_ns": counts["scaled_io_blocks"] * maxima["io_block"] * counts["file_batch_size"] + counts["projected_files"] * maxima["file_batch"],
        }
        expected_seconds = expected_scaling["total_duration_numerator_ns"] / expected_scaling["file_batch_divisor"] / 1_000_000_000
    if scaling != expected_scaling:
        raise backend.MlxQualificationError("pilot tail scaling formula differs")
    component = detail["component_seconds"]
    if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(float(component)) or float(component) != expected_seconds or isinstance(expected_component_seconds, bool) or not isinstance(expected_component_seconds, (int, float)) or float(expected_component_seconds) != expected_seconds:
        raise backend.MlxQualificationError("pilot tail component differs")
    return dict(detail)


def validate_source_exclusion_fixture(value: Any) -> str:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != ("raw", "routes", "source"):
        raise backend.MlxQualificationError("pilot source exclusion fixture keys differ")
    raw = value["raw"]
    routes = value["routes"]
    source = value["source"]
    if not isinstance(raw, list) or len(raw) != 32 or not isinstance(routes, list) or len(routes) != 32 or not isinstance(source, list) or len(source) != 32:
        raise backend.MlxQualificationError("pilot source exclusion fixture cardinality differs")
    if any(not isinstance(row, list) or len(row) != 2 or any(type(item) is not int or not -1 <= item <= 14 for item in row) for row in raw):
        raise backend.MlxQualificationError("pilot source exclusion raw route differs")
    if any(not isinstance(row, list) or len(row) != 2 or len(set(row)) != 2 or any(type(item) is not int or not 0 <= item <= 14 for item in row) for row in routes):
        raise backend.MlxQualificationError("pilot source exclusion route differs")
    if any(type(item) is not int or not 0 <= item <= 14 for item in source):
        raise backend.MlxQualificationError("pilot source exclusion source differs")
    runtime = cpu._import_runtime()
    expected_source = cpu.generate_rung_one_batch(123456, 2, runtime.torch)["required_source"] * 16
    if source != expected_source:
        raise backend.MlxQualificationError("pilot source exclusion source fixture differs")
    regenerated = cpu.generate_source_exclusion_routes(633456, raw, expected_source, runtime.torch)
    if dict(value) != regenerated:
        raise backend.MlxQualificationError("pilot source exclusion deterministic reconstruction differs")
    return cpu.canonical_json_sha256(regenerated)


def validate_child_tail_benchmarks(value: Any, components: Mapping[str, Any], source_exclusion_fixture: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != ("checkpoint_reload", "evaluation"):
        raise backend.MlxQualificationError("pilot child tail benchmark keys differ")
    source_exclusion_sha256 = validate_source_exclusion_fixture(source_exclusion_fixture)
    result = {}
    for name in ("evaluation", "checkpoint_reload"):
        detail = value[name]
        if not isinstance(detail, Mapping):
            raise backend.MlxQualificationError("pilot child tail detail differs")
        encoded = json.dumps(detail, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        if len(encoded) > 1_048_576:
            raise backend.MlxQualificationError("pilot child tail detail exceeds one MiB")
        hashes = detail.get("fixture_sha256s")
        if not isinstance(hashes, Mapping):
            raise backend.MlxQualificationError("pilot child tail fixture hashes differ")
        expected_hashes = expected_tail_fixture_sha256s("evaluation", source_exclusion_sha256=source_exclusion_sha256) if name == "evaluation" else expected_tail_fixture_sha256s("checkpoint_reload", engine_sha256=backend.dependency_hashes()["engine"])
        result[name] = validate_tail_benchmark_detail(name, detail, components[f"{name}_seconds"], expected_hashes)
    return result


def validate_resource_finalization_benchmark(value: Any) -> dict[str, Any]:
    expected_keys = ("actual_stop_seconds", "component_seconds", "final_active_jobs", "final_attempted_updates", "final_expected_pids", "final_sample_id", "final_token_positions", "interval_seconds", "max_observed_sample_duration_seconds", "sample_transaction_count")
    if not isinstance(value, Mapping) or tuple(sorted(value)) != expected_keys:
        raise backend.MlxQualificationError("pilot resource finalization schema differs")
    for field in ("actual_stop_seconds", "component_seconds", "interval_seconds", "max_observed_sample_duration_seconds"):
        observed = value[field]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)) or observed < 0:
            raise backend.MlxQualificationError("pilot resource finalization duration differs")
    expected_component = float(value["interval_seconds"]) + 2 * float(value["max_observed_sample_duration_seconds"])
    if value["interval_seconds"] != 5.0 or value["max_observed_sample_duration_seconds"] <= 0 or not math.isclose(float(value["component_seconds"]), expected_component, rel_tol=0.0, abs_tol=1e-12) or value["actual_stop_seconds"] > value["component_seconds"]:
        raise backend.MlxQualificationError("pilot resource finalization bound differs")
    if value["final_active_jobs"] != [] or not isinstance(value["final_expected_pids"], list) or len(value["final_expected_pids"]) != 1 or type(value["final_expected_pids"][0]) is not int or value["final_expected_pids"][0] < 1:
        raise backend.MlxQualificationError("pilot resource finalization process state differs")
    if type(value["final_attempted_updates"]) is not int or value["final_attempted_updates"] != PILOT_PROTOCOL.final_attempted_updates or type(value["final_token_positions"]) is not int or value["final_token_positions"] != PILOT_PROTOCOL.final_token_positions or type(value["final_sample_id"]) is not int or value["final_sample_id"] < 0 or type(value["sample_transaction_count"]) is not int or value["sample_transaction_count"] < 1:
        raise backend.MlxQualificationError("pilot resource finalization counters differ")
    return dict(value)


def run_mlx_resource_pilot(
    run_root: Path,
    payload: Mapping[str, Any],
    anchors: Any,
    signals: Any,
    transition: Any,
) -> dict[str, Any]:
    if transition.outcome != "ready" or transition.swap_baseline_bytes is None:
        raise cpu.ContractError("pilot transition is not ready")
    writer = transition.writers["run/pilot_resources.jsonl"]
    baseline_rows = writer.validate_committed_prefix()
    if len(baseline_rows) != 1 or baseline_rows[0]["attempted_updates"] != 0 or baseline_rows[0]["token_positions"] != 0:
        raise cpu.ContractError("pilot baseline row differs")
    cpu.final_frozen_guard(run_root, anchors, signals, "before_mlx_pilot_spawn")
    scratch = None
    process = None
    sampler = None
    stderr_handle = None
    attempted_updates = 0
    token_positions = 0
    unwind_primary = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix=f"todorov-mlx-pilot-{run_root.name}-", dir="/private/tmp"))
        command, environment = backend.child_invocation("pilot")
        stderr_path = scratch / "mlx_child.stderr"
        stderr_handle = stderr_path.open("xb")
        child_start_ns = time.perf_counter_ns()
        pilot_deadline_ns = child_start_ns + backend.HARD_LIMIT_SECONDS * 1_000_000_000
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**environment, "MODULAR_MLX_RUN_ROOT": str(run_root), "MODULAR_MLX_SCRATCH_ROOT": str(scratch)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        sampler = backend.QualificationResourceSampler(
            run_root.name,
            os.getpid(),
            process.pid,
            cpu.sample_processes,
            cpu.sample_swap,
            5.0,
            writer=writer,
            prior_rows=baseline_rows,
            phase="pilot",
            final_attempted_updates=PILOT_PROTOCOL.final_attempted_updates,
            final_token_positions=PILOT_PROTOCOL.final_token_positions,
        )
        sampler.swap_baseline = transition.swap_baseline_bytes
        sampler.start()
        hello = receive(process, sampler, pilot_deadline_ns)
        cold_child_start_seconds = (time.perf_counter_ns() - child_start_ns) / 1_000_000_000
        validate_pilot_hello(hello, run_root)
        sequence = 1
        workloads = []
        current_ordinal = -1
        current_workload = None
        current_model_seed = None
        pending_update = None
        current_memory = validate_pilot_memory(hello["self_check"]["memory"])
        child_components = None
        child_peak_memory = current_memory["peak_memory_bytes"]
        while True:
            message = receive(process, sampler, pilot_deadline_ns)
            if message.get("sequence") != sequence:
                raise backend.MlxQualificationError("pilot child sequence differs")
            sequence += 1
            kind = message.get("kind")
            if kind == "pilot_workload_started":
                expected_keys = ("batch_size", "data_seed", "execution", "kind", "lanes", "memory", "model_seed", "route_seed", "sequence", "sequence_length", "workload", "workload_ordinal")
                if tuple(sorted(message)) != expected_keys or pending_update is not None or message["workload_ordinal"] != current_ordinal + 1 or message["workload_ordinal"] >= len(PILOT_PROTOCOL.workloads):
                    raise backend.MlxQualificationError("pilot workload start differs")
                ordinal = message["workload_ordinal"]
                specification = PILOT_PROTOCOL.workloads[ordinal]
                model_seed = PILOT_PROTOCOL.model_seed(ordinal)
                if tuple(message[name] for name in ("workload", "execution", "lanes", "batch_size", "sequence_length", "model_seed", "data_seed", "route_seed")) != (specification.name, specification.execution, specification.lanes, specification.batch_size, specification.sequence_length, model_seed, PILOT_PROTOCOL.data_seed(ordinal), PILOT_PROTOCOL.route_seed(ordinal)):
                    raise backend.MlxQualificationError("pilot workload start identity differs")
                current_ordinal = ordinal
                current_workload = specification.name
                current_model_seed = model_seed
                current_memory = validate_pilot_memory(message["memory"])
                child_peak_memory = max(child_peak_memory, current_memory["peak_memory_bytes"])
                sampler.observe_pilot_progress(specification.name, model_seed, 0, attempted_updates, token_positions, current_memory)
            elif kind == "pilot_update_ready":
                expected_keys = ("kind", "lanes", "logical_update", "sequence", "token_positions_per_lane", "workload", "workload_ordinal")
                if tuple(sorted(message)) != expected_keys or current_workload is None or pending_update is not None or message["workload"] != current_workload or message["workload_ordinal"] != current_ordinal:
                    raise backend.MlxQualificationError("pilot update start differs")
                logical_update = message["logical_update"]
                if type(logical_update) is not int or logical_update not in PILOT_PROTOCOL.all_updates:
                    raise backend.MlxQualificationError("pilot update logical index differs")
                specification = PILOT_PROTOCOL.workloads[current_ordinal]
                if message["lanes"] != specification.lanes or message["token_positions_per_lane"] != specification.token_positions_per_lane:
                    raise backend.MlxQualificationError("pilot update charge differs")
                if logical_update != PILOT_PROTOCOL.expected_update(current_ordinal, attempted_updates):
                    raise backend.MlxQualificationError("pilot update order differs")
                attempted_updates += specification.lanes
                token_positions += specification.lanes * specification.token_positions_per_lane
                sampler.observe_pilot_progress(current_workload, current_model_seed, logical_update, attempted_updates, token_positions, current_memory)
                send_guarded_ack(process, {"ack": True, "kind": "pilot_update_start_committed", "workload": current_workload, "logical_update": logical_update}, pilot_deadline_ns, signals, "mlx_pilot_update_ack")
                pending_update = logical_update
            elif kind == "pilot_update_complete":
                expected_keys = ("elapsed_ns", "finite", "gradient_norm_finite", "kind", "logical_update", "memory", "optimizer_finite", "raw_overflow_count", "sequence", "warmup", "workload", "workload_ordinal")
                if tuple(sorted(message)) != expected_keys or pending_update is None or message["logical_update"] != pending_update or message["workload"] != current_workload or message["workload_ordinal"] != current_ordinal:
                    raise backend.MlxQualificationError("pilot update completion differs")
                if type(message["elapsed_ns"]) is not int or message["elapsed_ns"] <= 0 or message["finite"] is not True or message["gradient_norm_finite"] is not True or message["optimizer_finite"] is not True or message["raw_overflow_count"] != 0 or message["warmup"] != (pending_update in PILOT_PROTOCOL.warmup_updates):
                    raise backend.MlxQualificationError("pilot update audit differs")
                current_memory = validate_pilot_memory(message["memory"])
                child_peak_memory = max(child_peak_memory, current_memory["peak_memory_bytes"])
                sampler.observe_pilot_progress(current_workload, current_model_seed, pending_update, attempted_updates, token_positions, current_memory)
                pending_update = None
            elif kind == "pilot_workload_complete":
                expected_keys = ("cold_compiled_update_ns", "kind", "peak_memory_bytes", "record", "sequence")
                if tuple(sorted(message)) != expected_keys or pending_update is not None or current_ordinal != len(workloads):
                    raise backend.MlxQualificationError("pilot workload completion differs")
                record = validate_pilot_workload_record(message["record"], current_ordinal)
                if message["cold_compiled_update_ns"] != record["warmup_update_ns"][0] or type(message["peak_memory_bytes"]) is not int or message["peak_memory_bytes"] < 0:
                    raise backend.MlxQualificationError("pilot workload completion measurement differs")
                workloads.append(record)
                child_peak_memory = max(child_peak_memory, message["peak_memory_bytes"])
                current_workload = None
                current_model_seed = None
            elif kind == "pilot_complete":
                expected_keys = ("attempted_updates", "kind", "measured_components", "memory", "runtime", "sequence", "source_exclusion_fixture", "status", "tail_benchmarks", "token_positions", "workload_order")
                if tuple(sorted(message)) != expected_keys or current_workload is not None or pending_update is not None or len(workloads) != len(PILOT_PROTOCOL.workloads) or message["status"] != "clean_complete" or tuple(message["workload_order"]) != PILOT_PROTOCOL.workload_order or message["attempted_updates"] != PILOT_PROTOCOL.final_attempted_updates or message["token_positions"] != PILOT_PROTOCOL.final_token_positions:
                    raise backend.MlxQualificationError("pilot completion differs")
                child_components = message["measured_components"]
                expected_component_keys = ("checkpoint_reload_seconds", "cold_compile_seconds", "dense_vmap5_step_seconds", "donor_step_seconds", "evaluation_seconds", "rung_two_step_seconds", "selected_vmap5_step_seconds")
                if not isinstance(child_components, Mapping) or tuple(sorted(child_components)) != expected_component_keys or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) for value in child_components.values()):
                    raise backend.MlxQualificationError("pilot child components differ")
                child_tail_benchmarks = validate_child_tail_benchmarks(message["tail_benchmarks"], child_components, message["source_exclusion_fixture"])
                current_memory = validate_pilot_memory(message["memory"])
                child_peak_memory = max(child_peak_memory, current_memory["peak_memory_bytes"])
                break
            elif kind == "pilot_hard_abort":
                raise backend.MlxQualificationError(f"MLX pilot child hard abort: {message.get('error_type')}: {message.get('message')}")
            else:
                raise backend.MlxQualificationError("pilot child message kind differs")
        lifecycle_started_ns = time.perf_counter_ns()
        send(process, {"ack": True, "kind": "close_committed"}, pilot_deadline_ns)
        remaining_close_seconds = backend.enforce_deadline(pilot_deadline_ns) / 1_000_000_000
        return_code = process.wait(timeout=min(30.0, remaining_close_seconds))
        backend.enforce_deadline(pilot_deadline_ns)
        if return_code != 0:
            raise backend.MlxQualificationError(f"MLX pilot child exit differs: {return_code}")
        stderr_handle.flush()
        os.fsync(stderr_handle.fileno())
        backend.enforce_deadline(pilot_deadline_ns)
        stderr_bytes = stderr_path.stat().st_size
        backend.enforce_deadline(pilot_deadline_ns)
        if stderr_bytes != 0:
            raise backend.MlxQualificationError("MLX pilot child stderr differs")
        close_join_seconds = (time.perf_counter_ns() - lifecycle_started_ns) / 1_000_000_000
        sampler.clear_active_jobs(PILOT_PROTOCOL.final_attempted_updates, PILOT_PROTOCOL.final_token_positions)
        sampler.mark_child_exited()
        backend.enforce_deadline(pilot_deadline_ns)
        fixed_components, parent_tail_benchmarks = measure_pilot_fixed_components(run_root, scratch, child_tail_benchmarks["checkpoint_reload"]["byte_sizes"]["projected_checkpoint_bytes"])
        backend.enforce_deadline(pilot_deadline_ns)
        resource_finalization_started_ns = time.perf_counter_ns()
        resource_rows = sampler.stop(final_sample=True, deadline_ns=pilot_deadline_ns)
        resource_finalization_actual_seconds = (time.perf_counter_ns() - resource_finalization_started_ns) / 1_000_000_000
        resource_sample_max_seconds = sampler.max_sample_transaction_seconds
        resource_sample_transaction_count = sampler.sample_transaction_count
        resource_interval_seconds = sampler.interval_seconds
        resource_finalization_seconds = resource_interval_seconds + 2 * resource_sample_max_seconds
        if resource_finalization_actual_seconds > resource_finalization_seconds:
            raise backend.MlxQualificationError("pilot resource finalization bound differs")
        backend.enforce_deadline(pilot_deadline_ns)
        sampler = None
        backend.enforce_deadline(pilot_deadline_ns)
        writer.recover_uncommitted_suffix()
        backend.enforce_deadline(pilot_deadline_ns)
        writer.close()
        backend.enforce_deadline(pilot_deadline_ns)
        cpu.validate_resource_timeline(resource_rows, "pilot", require_clean_final=True, pilot_final_values=(PILOT_PROTOCOL.final_attempted_updates, PILOT_PROTOCOL.final_token_positions))
        backend.enforce_deadline(pilot_deadline_ns)
        peak_rss = max(row["aggregate_rss_bytes"] for row in resource_rows)
        swap_peak = max(row["swap_used_bytes"] for row in resource_rows)
        swap_growth = max(0, swap_peak - transition.swap_baseline_bytes)
        measured = dict(child_components)
        measured["cold_child_start_seconds"] = cold_child_start_seconds
        measured.update(fixed_components)
        measured["resource_finalization_seconds"] = resource_finalization_seconds
        backend.enforce_deadline(pilot_deadline_ns)
        cleanup_started_ns = time.perf_counter_ns()
        stderr_handle.close()
        stderr_handle = None
        scratch_cleanup = cleanup_scratch(scratch)
        scratch_cleanup_seconds = (time.perf_counter_ns() - cleanup_started_ns) / 1_000_000_000
        backend.enforce_deadline(pilot_deadline_ns)
        resource_finalization_benchmark = validate_resource_finalization_benchmark({
            "component_seconds": resource_finalization_seconds,
            "actual_stop_seconds": resource_finalization_actual_seconds,
            "final_active_jobs": resource_rows[-1]["active_jobs"],
            "final_attempted_updates": resource_rows[-1]["attempted_updates"],
            "final_expected_pids": resource_rows[-1]["expected_pids"],
            "final_sample_id": resource_rows[-1]["sample_id"],
            "final_token_positions": resource_rows[-1]["token_positions"],
            "interval_seconds": resource_interval_seconds,
            "max_observed_sample_duration_seconds": resource_sample_max_seconds,
            "sample_transaction_count": resource_sample_transaction_count,
        })
        lifecycle_benchmark = {
            "component_seconds": close_join_seconds + scratch_cleanup_seconds,
            "close_join_seconds": close_join_seconds,
            "scratch_cleanup_seconds": scratch_cleanup_seconds,
            "return_code": return_code,
            "stderr_bytes": stderr_bytes,
            "scratch_cleanup": scratch_cleanup,
        }
        measured["lifecycle_close_join_seconds"] = lifecycle_benchmark["component_seconds"]
        projection = backend.project_full_package(measured)
        assertions = [
            cpu._check_record(run_root, run_root.name, "pilot_counter_total", "pilot", {"attempted_updates": PILOT_PROTOCOL.final_attempted_updates, "token_positions": PILOT_PROTOCOL.final_token_positions}, {"attempted_updates": attempted_updates, "token_positions": token_positions}, None, None, attempted_updates == PILOT_PROTOCOL.final_attempted_updates and token_positions == PILOT_PROTOCOL.final_token_positions, ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_workload_cardinality", "pilot", len(PILOT_PROTOCOL.workloads), len(workloads), None, None, len(workloads) == len(PILOT_PROTOCOL.workloads), ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_child_clean_exit", "pilot", {"return_code": 0, "stderr_bytes": 0}, {"return_code": return_code, "stderr_bytes": stderr_bytes}, None, None, return_code == 0 and stderr_bytes == 0, ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_routing_evidence_projection", "pilot", {"claim_rows": 588_240, "scale_blocks": 136, "timed_repetitions": 3}, parent_tail_benchmarks["routing_evidence"], None, None, parent_tail_benchmarks["routing_evidence"]["scratch_cleanup_pass"] is True and parent_tail_benchmarks["routing_evidence"]["component_seconds"] == measured["routing_evidence_seconds"], ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_evaluation_projection", "pilot", {"claim_calls": 1_072, "endpoint_replays": 26, "timed_repetitions": 3}, child_tail_benchmarks["evaluation"], None, None, child_tail_benchmarks["evaluation"]["scratch_cleanup_pass"] is True and child_tail_benchmarks["evaluation"]["component_seconds"] == measured["evaluation_seconds"], ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_checkpoint_reload_projection", "pilot", {"trained_endpoint_files": 26, "timed_repetitions": 3}, child_tail_benchmarks["checkpoint_reload"], None, None, child_tail_benchmarks["checkpoint_reload"]["scratch_cleanup_pass"] is True and child_tail_benchmarks["checkpoint_reload"]["component_seconds"] == measured["checkpoint_reload_seconds"], ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_packaging_projection", "pilot", {"fixed_clean_files": 144, "timed_repetitions": 3}, parent_tail_benchmarks["packaging"], None, None, parent_tail_benchmarks["packaging"]["scratch_cleanup_pass"] is True and parent_tail_benchmarks["packaging"]["component_seconds"] == measured["packaging_seconds"], ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_resource_finalization_projection", "pilot", {"absolute_deadline_ns": pilot_deadline_ns, "final_attempted_updates": PILOT_PROTOCOL.final_attempted_updates, "final_token_positions": PILOT_PROTOCOL.final_token_positions}, resource_finalization_benchmark, None, None, resource_finalization_benchmark["component_seconds"] == measured["resource_finalization_seconds"] and resource_finalization_benchmark["actual_stop_seconds"] <= resource_finalization_benchmark["component_seconds"] and resource_finalization_benchmark["interval_seconds"] == 5.0 and resource_finalization_benchmark["max_observed_sample_duration_seconds"] > 0 and resource_finalization_benchmark["sample_transaction_count"] > 0 and resource_finalization_benchmark["final_active_jobs"] == [] and resource_finalization_benchmark["final_expected_pids"] == [os.getpid()], ["run/pilot_resources.jsonl"]),
            cpu._check_record(run_root, run_root.name, "pilot_tail_lifecycle_close_join_projection", "pilot", {"real_child": True, "close_committed": True, "scratch_cleanup": True}, lifecycle_benchmark, None, None, lifecycle_benchmark["component_seconds"] == measured["lifecycle_close_join_seconds"] and lifecycle_benchmark["return_code"] == 0 and lifecycle_benchmark["stderr_bytes"] == 0 and lifecycle_benchmark["scratch_cleanup"] is True, ["run/pilot_resources.jsonl"]),
        ]
        if not all(record["pass"] is True for record in assertions):
            raise cpu.HardAbort("assertion_failure", {"surface": "mlx_resource_pilot"})
        failures = []
        if projection["hard_limit_pass"] is not True:
            failures.append("pilot_projected_time")
        if peak_rss > 10 * 1024**3:
            failures.append("pilot_resident_memory")
        if swap_growth > 0:
            failures.append("pilot_swap_growth")
        ordered_names = payload["pilot"]["time_statistics"]["measured_component_order"]
        pilot_record = {
            "schema_version": cpu.SCHEMA_VERSION,
            "run_id": run_root.name,
            "seed_base": PILOT_PROTOCOL.seed_base,
            "workload_order": list(PILOT_PROTOCOL.workload_order),
            "child": {
                "worker": "MLX",
                "ordinal": 0,
                "pid": process.pid,
                "python_path": str(backend.MLX_PYTHON),
                "mlx_version": backend.MLX_VERSION,
                "device": "Device(gpu, 0)",
                "engine_sha256": hello["engine_sha256"],
                "dependency_sha256s": hello["dependency_sha256s"],
                "self_check_details_sha256": pilot_self_check_details_sha256(run_root),
            },
            "workloads": workloads,
            "measured_components": [{"name": name, "seconds": float(measured[name])} for name in ordered_names],
            "projection": projection,
            "peak_rss_bytes": peak_rss,
            "swap_baseline_bytes": transition.swap_baseline_bytes,
            "swap_peak_bytes": swap_peak,
            "swap_growth_bytes": swap_growth,
            "assertions": assertions,
            "decision": "stop" if failures else "proceed",
            "decision_reasons": failures,
        }
        backend.enforce_deadline(pilot_deadline_ns)
        cpu.write_canonical_json(run_root / "run" / "pilot.json", pilot_record)
        backend.enforce_deadline(pilot_deadline_ns)
        return pilot_record
    except BaseException as primary_error:
        unwind_primary = primary_error
        cleanup_after_primary_failure(process, sampler, primary_error)
        raise
    finally:
        cleanup_operations = []
        if stderr_handle is not None:
            cleanup_operations.extend((stderr_handle.flush, lambda: os.fsync(stderr_handle.fileno()), stderr_handle.close))
        cleanup_operations.append(lambda: cleanup_scratch(scratch))
        perform_cleanup_operations(tuple(cleanup_operations), unwind_primary)


def run_mlx_claim(
    run_root: Path,
    payload: Mapping[str, Any],
    anchors: Any,
    signals: Any,
    transition: Any,
    claim_start_monotonic_ns: int,
) -> dict[str, Any]:
    if transition.outcome != "ready" or transition.swap_baseline_bytes is None:
        raise cpu.ContractError("claim transition is not ready")
    cpu.final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_mlx_claim")
    writers = dict(transition.writers)
    resource_writer = writers["run/resources.jsonl"]
    preworker_row = cpu._resource_sample(run_root.name, "claim", 0, [], set(), {}, {}, transition.swap_baseline_bytes, 0, 0)
    append_result = resource_writer.append(preworker_row)
    if not append_result.acknowledged:
        raise cpu.HardAbort(append_result.reason_code or "artifact_inconsistency")
    observations = cpu.claim_resource_observations(resource_writer.validate_committed_prefix())
    if observations:
        observation = observations[0]
        raise cpu.HardAbort(observation["reason_code"], observation["context"])
    scratch = None
    process = None
    sampler = None
    stderr_handle = None
    deferred_transport = False
    unwind_primary = None
    try:
        scratch = Path(tempfile.mkdtemp(prefix=f"todorov-mlx-{run_root.name}-", dir="/private/tmp"))
        command, environment = backend.child_invocation("serve")
        cpu.final_claim_guard(run_root, anchors, signals, claim_start_monotonic_ns, "before_mlx_child_spawn")
        stderr_path = scratch / "mlx_child.stderr"
        stderr_handle = stderr_path.open("xb")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env={**environment, "MODULAR_MLX_RUN_ROOT": str(run_root), "MODULAR_MLX_SCRATCH_ROOT": str(scratch)},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        sampler = backend.QualificationResourceSampler(
            run_root.name,
            os.getpid(),
            process.pid,
            cpu.sample_processes,
            cpu.sample_swap,
            5.0,
            writer=resource_writer,
            prior_rows=[preworker_row],
        )
        sampler.swap_baseline = transition.swap_baseline_bytes
        sampler.start()
        deadline_ns = claim_start_monotonic_ns + backend.HARD_LIMIT_SECONDS * 1_000_000_000
        training, _ = run_training(run_root, process, sampler, deadline_ns, writers, signals)
        sampler.clear_active_jobs(20_736, backend.POSITIONS)
        resource_rows = sampler.stop(final_sample=True, deadline_ns=deadline_ns)
        sampler = None
        cpu.validate_resource_timeline(resource_rows, "claim", require_clean_final=True)
        backend.enforce_deadline(deadline_ns)
        for writer in writers.values():
            backend.enforce_deadline(deadline_ns)
            writer.recover_uncommitted_suffix()
            writer.close()
            backend.enforce_deadline(deadline_ns)
        accounting = cpu._accounting_from_run_root(run_root)
        if accounting.unpaired_attempts or accounting.attempted_updates != 20_736 or accounting.completed_updates != 20_736 or accounting.completed_token_positions != backend.POSITIONS:
            raise cpu.HardAbort("artifact_inconsistency", {"surface": "mlx_claim_accounting"})
        seed_sample_ids = {seed: cpu.resource_sample_ids_for_seed(resource_rows, seed) for seed in (*backend.RUNG_ONE_SEEDS, backend.RUNG_TWO_SEED)}
        if any(not sample_ids for sample_ids in seed_sample_ids.values()):
            raise cpu.HardAbort("resource_sampler_failure", {"surface": "mlx_seed_resource_evidence"})
        cpu._finalize_seed_resource_references(run_root, seed_sample_ids)
        cpu.validate_parent_ledger_accounting(run_root)
        final_row = resource_rows[-1]
        state = {"finished": False}

        def cleanup_transport(primary_error: BaseException | None = None) -> None:
            nonlocal stderr_handle
            cleanup_operations = []
            if process is not None and process.poll() is None:
                cleanup_operations.append(lambda: terminate(process))
            if stderr_handle is not None:
                cleanup_operations.extend((stderr_handle.flush, lambda: os.fsync(stderr_handle.fileno()), stderr_handle.close))
            cleanup_operations.append(lambda: cleanup_scratch(scratch))
            perform_cleanup_operations(tuple(cleanup_operations), primary_error)
            if stderr_handle is not None:
                stderr_handle = None

        def clean_transport(claim_result: Mapping[str, Any]) -> None:
            if state["finished"]:
                raise backend.MlxQualificationError("MLX claim transport is already finalized")
            state["finished"] = True
            transport_primary = None
            try:
                commit_clean_child_close(process, deadline_ns, claim_result, lambda _: None)
                backend.enforce_deadline(deadline_ns)
                if stderr_handle is not None:
                    stderr_handle.flush()
                    os.fsync(stderr_handle.fileno())
                    backend.enforce_deadline(deadline_ns)
                    error = stderr_path.read_text(encoding="utf-8", errors="replace")
                    backend.enforce_deadline(deadline_ns)
                    if error:
                        raise backend.MlxQualificationError(f"MLX child stderr differs: {error}")
            except BaseException as primary_error:
                transport_primary = primary_error
                raise
            finally:
                cleanup_transport(transport_primary)
                if transport_primary is None:
                    backend.enforce_deadline(deadline_ns)

        def abort_transport() -> None:
            if state["finished"]:
                return
            state["finished"] = True
            cleanup_transport()

        result = {
            "accounting": accounting,
            "resource_final_sample_id": final_row["sample_id"],
            "resource_sampling_end_monotonic_ns": final_row["monotonic_ns"],
            "resource_rows": resource_rows,
            "training": training,
            "clean_transport_finalizer": clean_transport,
            "abort_transport_finalizer": abort_transport,
        }
        deferred_transport = True
        return result
    except BaseException as primary_error:
        unwind_primary = primary_error
        cleanup_after_primary_failure(process, sampler, primary_error)
        raise
    finally:
        if not deferred_transport:
            cleanup_operations = []
            if stderr_handle is not None:
                cleanup_operations.extend((stderr_handle.flush, lambda: os.fsync(stderr_handle.fileno()), stderr_handle.close))
            cleanup_operations.append(lambda: cleanup_scratch(scratch))
            perform_cleanup_operations(tuple(cleanup_operations), unwind_primary)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child-mode":
        return run_child(sys.argv[2])
    raw_run_root = cpu.parse_cli(sys.argv[1:])
    cpu.validate_entry_environment(os.environ)
    entry = cpu.validate_run_root(raw_run_root)
    payload = cpu.load_prereg_payload()
    runtime = cpu._import_runtime()
    cpu.configure_torch(runtime.torch)
    return cpu.execute_run(
        entry,
        payload,
        runtime,
        tuple(sys.argv),
        resource_pilot_runner=run_mlx_resource_pilot,
        claim_runner=run_mlx_claim,
        trained_backend_probe=preflight_mlx_probe,
    )


if __name__ == "__main__":
    raise SystemExit(main())
