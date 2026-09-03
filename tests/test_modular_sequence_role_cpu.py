import ast
import copy
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neuroloc.simulations.memory import modular_sequence_role_cpu as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "modular_sequence_role_cpu.py"
TRACKED_PAYLOAD_PATH = PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_prereg.json"
TRACKED_PAYLOAD_CANONICAL_SHA256 = "fc3c7130a7ed21043e7081b09eb9265711417a22e84eb5356e6a2402e75a2553"
REVIEW_ATTESTATION_SCHEMA_VERSION = "todorov.review-attestation.1"
TOP_LEVEL_KEYS = (
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
TRANSACTIONAL_JSONL_PATHS = ("run/pilot_resources.jsonl",) + CLAIM_LEDGER_PATHS
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
REQUIRED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
}
TORCH_EVIDENCE_PATHS = (
    *(f"data/r1_eval_{400000 + seed}.pt" for seed in (11, 23, 37, 53, 71)),
    *(f"data/r1_random_routes_{seed}.pt" for seed in (11, 23, 37, 53, 71)),
    *(f"data/r1_source_exclusion_{seed}.pt" for seed in (11, 23, 37, 53, 71)),
    "data/r2_eval_1000083.pt",
    *(
        f"rung1/{seed}/checkpoints/{filename}"
        for seed in (11, 23, 37, 53, 71)
        for filename in ("donor_last.pt", "router_last.pt", "final_last.pt", "dense_base_last.pt", "dense_last.pt")
    ),
    "rung2/83/checkpoints/final_last.pt",
)


def _tracked_payload():
    return json.loads(TRACKED_PAYLOAD_PATH.read_text(encoding="utf-8"))


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_snapshot(root):
    records = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            records.append((relative, "symlink", path.readlink().as_posix()))
        elif path.is_dir():
            records.append((relative, "directory", None))
        else:
            records.append((relative, "file", path.read_bytes()))
    return tuple(sorted(records))


def _set_path(value, path, replacement):
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def _source_tree():
    return ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))


def _fixture_record(sequence):
    return {"sequence": sequence, "value": f"v{sequence}"}


def _validate_fixture_record(record):
    runner.validate_exact_keys(record, ("sequence", "value"), "fixture record")
    if isinstance(record["sequence"], bool) or not isinstance(record["sequence"], int) or record["sequence"] < 0:
        raise runner.ContractError("fixture sequence differs")
    if record["value"] != f"v{record['sequence']}":
        raise runner.ContractError("fixture value differs")


def _attempt_event(sequence, event, logical_update=1, construction_seed=11):
    identity = runner.attempt_id("test-run", 1, construction_seed, "selected", "joint", logical_update)
    metrics = None
    if event == "completed":
        metrics = {
            "learning_rates": [],
            "component_losses": {"task_loss": 1.0, "internal_router_loss": 0.0, "supervised_route_loss": 0.0},
            "total_loss": 1.0,
            "gradient_norm": 0.5,
            "clip_result": "unchanged",
            "raw_overflow_count": 0,
            "max_bucket_load": 1,
            "elapsed_seconds": 0.1,
            "finite": True,
        }
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": "test-run",
        "rung": 1,
        "claim_seed": construction_seed,
        "construction_seed": construction_seed,
        "event_sequence": sequence,
        "event": event,
        "attempt_id": identity,
        "model": "selected",
        "stage": "joint",
        "logical_update": logical_update,
        "examples": 16,
        "token_positions": 2048,
        "batch_sha256": "a" * 64,
        "monotonic_ns": 1000 + sequence,
        "wall_time_utc": "2026-07-19T00:00:00Z",
        "metrics": metrics,
    }


def _resource_row(run_id, phase, swap):
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": run_id,
        "sample_id": 0,
        "phase": phase,
        "monotonic_ns": 100,
        "wall_time_utc": "2026-07-19T00:00:00Z",
        "expected_pids": [101],
        "processes": [{"pid": 101, "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}],
        "active_jobs": [],
        "aggregate_rss_bytes": 2048,
        "aggregate_cpu_time_us": 3000,
        "swap_used_bytes": swap,
        "swap_growth_bytes": 0,
        "parser_status": "pass",
        "attempted_updates": 0,
        "token_positions": 0,
    }


def _resource_timeline(phase="claim"):
    baseline = _resource_row("run", phase, 100)
    middle = copy.deepcopy(baseline)
    middle.update(
        {
            "sample_id": 1,
            "monotonic_ns": 5_000_000_100,
            "wall_time_utc": "2026-07-19T00:00:05Z",
            "active_jobs": [{"worker": "A", "seed": 11, "stage": "joint", "logical_update": 1}],
            "aggregate_cpu_time_us": 3500,
            "swap_used_bytes": 120,
            "swap_growth_bytes": 20,
            "attempted_updates": 1,
            "token_positions": 2048,
        }
    )
    middle["processes"][0]["cpu_time_us"] = 3500
    final = copy.deepcopy(middle)
    final.update(
        {
            "sample_id": 2,
            "monotonic_ns": 10_000_000_100,
            "wall_time_utc": "2026-07-19T00:00:10Z",
            "active_jobs": [],
            "aggregate_cpu_time_us": 4000,
            "swap_used_bytes": 90,
            "swap_growth_bytes": 0,
            "attempted_updates": 88 if phase == "pilot" else 20736,
            "token_positions": 225280 if phase == "pilot" else 45613056,
        }
    )
    final["processes"][0]["cpu_time_us"] = 4000
    return [baseline, middle, final]


def _parity_checks():
    return [
        {
            "name": f"check_{index}_{scope}",
            "scope": scope,
            "max_error": 0.0 if index % 2 else None,
            "tolerance": 1e-7 if index % 2 else None,
            "pass": True,
            "details_sha256": f"{index + 1:064x}",
        }
        for index, scope in enumerate(runner.PARITY_SCOPES)
    ]


def _intervention_records():
    records = []
    recurrent_blocks = (1, 2, 3, 5, 6, 7)
    recurrent_conditions = {"intact", "recurrent_knockout", "carry_reset", "carry_shuffle"}
    checkpoint_by_condition = _intervention_checkpoint_by_condition(1)
    for condition in runner.RUNG_ONE_CONDITIONS:
        model, checkpoint = checkpoint_by_condition[condition]
        if condition == "all_eligible_donor":
            baseline_model, baseline_checkpoint, baseline_condition = model, checkpoint, condition
        elif condition == "dense_causal":
            baseline_model, baseline_checkpoint, baseline_condition = model, checkpoint, condition
        else:
            baseline_model, baseline_checkpoint, baseline_condition = (*checkpoint_by_condition["intact"], "intact")
        per_block = []
        for block in range(8):
            intact = float(block + 1)
            if condition == "recurrent_knockout" and block in recurrent_blocks:
                values = (intact, intact, 0.0)
            elif condition in {"carry_reset", "carry_shuffle"} and block in recurrent_blocks:
                values = (intact, intact + 0.5, intact + 0.5)
            elif condition == "block4_routed_knockout" and block == 4:
                values = (intact, intact + 1.0, 0.0)
            elif condition == "block4_local_only":
                values = (intact, intact + 2.0, intact + 2.0)
            elif condition == "all_eligible_clone":
                values = (intact, intact + 3.0, intact + 3.0)
            elif condition == "all_eligible_donor":
                values = (intact + 4.0, intact + 4.0, intact + 4.0)
            elif condition == "dense_causal":
                values = (intact + 5.0, intact + 5.0, intact + 5.0)
            else:
                values = (intact, intact, intact)
            record = {
                "model": model,
                "checkpoint_sha256": checkpoint,
                "baseline_model": baseline_model,
                "baseline_checkpoint_sha256": baseline_checkpoint,
                "baseline_condition": baseline_condition,
                "block": block,
                "condition": condition,
                "pre_delta_l2": values[0],
                "post_delta_l2": values[1],
                "exposed_delta_l2": values[2],
            }
            records.append(record)
            per_block.append(record)
        target_blocks = recurrent_blocks if condition in recurrent_conditions else (4,)
        targets = [record for record in per_block if record["block"] in target_blocks]
        records.append(
            {
                "model": model,
                "checkpoint_sha256": checkpoint,
                "baseline_model": baseline_model,
                "baseline_checkpoint_sha256": baseline_checkpoint,
                "baseline_condition": baseline_condition,
                "block": None,
                "condition": condition,
                "pre_delta_l2": sum(record["pre_delta_l2"] ** 2 for record in targets) ** 0.5,
                "post_delta_l2": sum(record["post_delta_l2"] ** 2 for record in targets) ** 0.5,
                "exposed_delta_l2": sum(record["exposed_delta_l2"] ** 2 for record in targets) ** 0.5,
            }
        )
    return records


def _rung_two_intervention_records():
    records = []
    checkpoint_by_condition = _intervention_checkpoint_by_condition(2)
    for condition in runner.RUNG_TWO_CONDITIONS:
        model, checkpoint = checkpoint_by_condition[condition]
        baseline_model, baseline_checkpoint = checkpoint_by_condition["intact"]
        per_block = []
        for block in runner.RECURRENT_BLOCKS:
            intact = float(block + 1)
            record = {
                "model": model,
                "checkpoint_sha256": checkpoint,
                "baseline_model": baseline_model,
                "baseline_checkpoint_sha256": baseline_checkpoint,
                "baseline_condition": "intact",
                "block": block,
                "condition": condition,
                "pre_delta_l2": intact,
                "post_delta_l2": intact,
                "exposed_delta_l2": 0.0 if condition == "recurrent_knockout" else intact,
            }
            records.append(record)
            per_block.append(record)
        records.append(
            {
                "model": model,
                "checkpoint_sha256": checkpoint,
                "baseline_model": baseline_model,
                "baseline_checkpoint_sha256": baseline_checkpoint,
                "baseline_condition": "intact",
                "block": None,
                "condition": condition,
                "pre_delta_l2": sum(record["pre_delta_l2"] ** 2 for record in per_block) ** 0.5,
                "post_delta_l2": sum(record["post_delta_l2"] ** 2 for record in per_block) ** 0.5,
                "exposed_delta_l2": sum(record["exposed_delta_l2"] ** 2 for record in per_block) ** 0.5,
            }
        )
    return records


def _intervention_checkpoint_by_condition(rung):
    if rung == 1:
        digest_by_model = {
            "selected": "a" * 64,
            "local": "a" * 64,
            "clone": "a" * 64,
            "donor": "b" * 64,
            "dense": "c" * 64,
        }
        return {
            condition: (
                runner.RUNG_ONE_MODEL_BY_CONDITION[condition],
                digest_by_model[runner.RUNG_ONE_MODEL_BY_CONDITION[condition]],
            )
            for condition in runner.RUNG_ONE_CONDITIONS
        }
    if rung == 2:
        return {condition: ("rung_two", "d" * 64) for condition in runner.RUNG_TWO_CONDITIONS}
    raise AssertionError(rung)


def _refresh_intervention_aggregate(records, condition):
    recurrent_conditions = {"intact", "recurrent_knockout", "carry_reset", "carry_shuffle"}
    target_blocks = (1, 2, 3, 5, 6, 7) if condition in recurrent_conditions else (4,)
    aggregate = next(record for record in records if record["block"] is None and record["condition"] == condition)
    targets = [record for record in records if record["block"] in target_blocks and record["condition"] == condition]
    for field in ("pre_delta_l2", "post_delta_l2", "exposed_delta_l2"):
        aggregate[field] = sum(record[field] ** 2 for record in targets) ** 0.5


def _expected_routing_workspace_bytes(model, block, batch_size):
    if block != 4 or model in {"local", "dense"}:
        return 0
    return batch_size * 128 * 4 * 64 * 76


def _expected_routing_addresses_probed(model, block, batch_size):
    width = runner._route_width(model, block)
    if width == 0:
        return 0
    return batch_size * (128 - min(128, (width + 1) * 8)) * 4


def _state_records(rung, checkpoint_by_condition):
    if rung == 1:
        conditions = runner.RUNG_ONE_CONDITIONS
        gate_count = 512 * 4 * 128
        output_count = 512 * 128 * 64
        boundary_count = 512 * 4
        chunk_positions = runner.RUNG_ONE_CHUNK_END_POSITIONS
    else:
        conditions = runner.RUNG_TWO_CONDITIONS
        gate_count = 512 * 4 * 512
        output_count = 512 * 512 * 64
        boundary_count = 512 * 4
        chunk_positions = runner.RUNG_TWO_CHUNK_END_POSITIONS
    records = []
    for condition in conditions:
        model, checkpoint = checkpoint_by_condition[condition]
        for block in runner.RECURRENT_BLOCKS:
            for statistic, count in (
                ("primary_gate", gate_count),
                ("beta_gate", gate_count),
                ("output_gate", output_count),
            ):
                records.append(
                    {
                        "model": model,
                        "checkpoint_sha256": checkpoint,
                        "block": block,
                        "condition": condition,
                        "boundary": "not_applicable",
                        "position": None,
                        "statistic": statistic,
                        "count": count,
                        "mean": 2.0,
                        "population_std": 0.5,
                        "min": 1.0,
                        "max": 3.0,
                        "nonfinite_count": 0,
                    }
                )
            boundaries = [("global_chunk_end", position) for position in chunk_positions]
            if rung == 1:
                for position in runner.RUNG_ONE_RESET_POSITIONS:
                    boundaries.extend((("pre_firewall_reset", position), ("post_firewall_reset", position)))
                if condition in {"carry_reset", "carry_shuffle"}:
                    boundaries.extend((("pre_carry_intervention", 96), ("post_carry_intervention", 96)))
            for boundary, position in boundaries:
                records.append(
                    {
                        "model": model,
                        "checkpoint_sha256": checkpoint,
                        "block": block,
                        "condition": condition,
                        "boundary": boundary,
                        "position": position,
                        "statistic": "state_l2",
                        "count": boundary_count,
                        "mean": 2.0,
                        "population_std": 0.5,
                        "min": 1.0,
                        "max": 3.0,
                        "nonfinite_count": 0,
                    }
                )
        records.append(
            {
                "model": model,
                "checkpoint_sha256": checkpoint,
                "block": None,
                "condition": condition,
                "boundary": "not_applicable",
                "position": None,
                "statistic": "primary_gate",
                "count": gate_count * len(runner.RECURRENT_BLOCKS),
                "mean": 2.0,
                "population_std": 0.5,
                "min": 1.0,
                "max": 3.0,
                "nonfinite_count": 0,
            }
        )
    return records


def _gradient_audit_record(name="weight", stage="donor", trainable=True, classification="learned_with_evidence", updates=2):
    nonzero = updates if trainable and classification == "learned_with_evidence" else 0
    update_nonzero = updates if trainable and classification in {"learned_with_evidence", "updated_only_by_decay"} else 0
    model = {
        "donor": "all_eligible_donor",
        "router_only": "selected",
        "joint": "selected",
        "dense_base": "dense_causal",
        "dense_continuation": "dense_causal",
        "rung_two": "rung_two",
    }[stage]
    block = int(name.split(".")[1]) if name.startswith("blocks.") else None
    return {
        "model": model,
        "stage": stage,
        "name": name,
        "block": block,
        "shape": [2],
        "category": "matrix",
        "requires_grad": trainable,
        "optimizer_member": trainable,
        "parameter_group": "group" if trainable else None,
        "peak_lr": 0.001 if trainable else None,
        "weight_decay": 0.01 if trainable else None,
        "grad_none_steps": 0 if trainable else updates,
        "grad_zero_steps": updates - nonzero if trainable else 0,
        "grad_nonzero_steps": nonzero,
        "grad_nonfinite_steps": 0,
        "update_zero_steps": updates - update_nonzero if trainable else 0,
        "update_nonzero_steps": update_nonzero,
        "update_nonfinite_steps": 0,
        "first_nonzero_step": 1 if nonzero else None,
        "start_sha256": "a" * 64,
        "end_sha256": "b" * 64 if update_nonzero else "a" * 64,
        "update_l2": 1.0 if trainable else None,
        "update_max_abs": 1.0 if trainable else None,
        "classification": classification if trainable else "frozen_by_design",
    }


def _public_router_audit_step(stage, seed, runtime_modules):
    torch = runtime_modules.torch
    model_module = runtime_modules.model_module
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
    optimizer, _, membership = runner._make_optimizer(model, stage, runtime_modules)
    audit = runner._initialize_audit(model, stage, runtime_modules, membership)
    batch = runner.payload_to_tensors(runner.generate_rung_one_batch(seed + 1, 2, torch), torch)
    optimizer.zero_grad(set_to_none=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    loss, _, _, _, _ = runner._training_forward_loss(model, batch, stage, runtime_modules)
    loss.backward()
    for name, parameter in model.named_parameters():
        record = audit[name]
        gradient = parameter.grad
        if gradient is None:
            record["grad_none_steps"] += 1
        elif not bool(torch.isfinite(gradient).all()):
            record["grad_nonfinite_steps"] += 1
        elif bool((gradient != 0).any()):
            record["grad_nonzero_steps"] += 1
            record["first_nonzero_step"] = 1
        else:
            record["grad_zero_steps"] += 1
    runner._clip_gradient_norm_finite(torch, model, optimizer, {"stage": stage, "logical_update": 1})
    optimizer.step()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter.detach() - before[name]
        if not bool(torch.isfinite(parameter).all()) or not bool(torch.isfinite(delta).all()):
            audit[name]["update_nonfinite_steps"] += 1
        elif bool((delta != 0).any()):
            audit[name]["update_nonzero_steps"] += 1
        else:
            audit[name]["update_zero_steps"] += 1
    return model, runner._finalize_audit(audit, model, "selected", torch)


class _FakeConnection:
    def __init__(self, response=None, send_error=None, receive_error=None):
        self.response = response
        self.send_error = send_error
        self.receive_error = receive_error
        self.sent = []

    def send(self, value):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(value)

    def recv(self):
        if self.receive_error is not None:
            raise self.receive_error
        return self.response


class _EscalatingProcess:
    def __init__(self, ignore_terminate=False, ignore_kill=False, cleanup_errors=False):
        self.alive = True
        self.ignore_terminate = ignore_terminate
        self.ignore_kill = ignore_kill
        self.cleanup_errors = cleanup_errors
        self.join_timeouts = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.exitcode = None

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.cleanup_errors and len(self.join_timeouts) == 1:
            raise RuntimeError("join failed")

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        if self.cleanup_errors:
            raise RuntimeError("terminate failed")
        if not self.ignore_terminate:
            self.alive = False
            self.exitcode = -runner.signal.SIGTERM

    def kill(self):
        self.kill_calls += 1
        if not self.ignore_kill:
            self.alive = False
            self.exitcode = -runner.signal.SIGKILL


def _stub_execute_activation(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    entry = runner.EntryConfiguration(results / "run-id", "run-id")
    payload = _tracked_payload()
    calls = []

    class Signals:
        def __init__(self):
            self.active = False
            self.terminal = False
            self.pending_signal = None

        def install(self):
            self.active = True
            calls.append("signal_install")

        def deactivate_terminal(self):
            self.active = False
            self.terminal = True
            calls.append("signal_terminal")

    signals = Signals()

    def publish(staging, final, controller):
        Path(staging).rename(final)
        calls.append("activate")
        return runner.PublicationResult("active", Path(final), Path(staging), True, 100, "2026-07-19T00:00:00Z", None)

    monkeypatch.setattr(runner, "RESULTS_PARENT", results)
    monkeypatch.setattr(runner, "SignalController", lambda: signals)
    monkeypatch.setattr(runner, "fsync_directory", lambda path: None)
    monkeypatch.setattr(runner, "build_shared_prepilot_base", lambda *args: calls.append("prepilot"))
    monkeypatch.setattr(runner, "capture_frozen_manifest_anchors", lambda *args: runner.FrozenManifestAnchors(()))
    monkeypatch.setattr(runner, "validate_base_review_target_binding", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda *args: None)
    monkeypatch.setattr(runner, "establish_training_start_plan_barrier", lambda run_root, anchors, controller: (anchors, 123456))
    monkeypatch.setattr(runner, "validate_artifact_closure", lambda *args: ())
    monkeypatch.setattr(runner, "_verify_public_commit", lambda: "0f9bf59ebdd032da46553d985bcf23348e1d5289")
    monkeypatch.setattr(runner, "load_prereg_payload", lambda *args, **kwargs: copy.deepcopy(payload))
    monkeypatch.setattr(runner, "publish_and_activate", publish)
    return entry, payload, runner.RuntimeModules(torch=None, model_module=None), signals, calls


def _write_terminal_fixture(root, relative, value, signals):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_json_bytes(value))
    checksum = f"{runner.sha256_file(path)}  {relative}\n".encode("ascii")
    (root / "SHA256SUMS").write_bytes(checksum)
    signals.deactivate_terminal()
    return path.read_bytes(), checksum


def _materialize_manifest_anchor_surface(root, launch_raw=b"run/project_plan_launch.md"):
    review_digests = tuple(f"{index:064x}" for index in range(1, 5))
    config = {
        "records": [{"path": "neuroloc/wiki/PROJECT_PLAN.md", "sha256": hashlib.sha256(launch_raw).hexdigest()}],
        "review_records": [{"artifact_sha256": digest} for digest in review_digests],
    }
    paths = (
        "run/prereg.json",
        "run/source_manifest.json",
        "run/config_manifest.json",
        "run/environment.json",
        "run/preflight.json",
        "run/project_plan_launch.md",
        "run/sentinels/selected_attention_oracle_payload.json",
        *(f"run/reviews/{digest}.json" for digest in review_digests),
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = runner.canonical_json_bytes(config) if relative == "run/config_manifest.json" else launch_raw if relative == "run/project_plan_launch.md" else relative.encode("ascii")
        path.write_bytes(raw)
    return tuple(sorted(paths))


def _training_start_attestation(plan_sha256, request_sha256, run_id, findings=None, finding_count=None):
    target_records = [
        {"path": "neuroloc/wiki/PROJECT_PLAN.md", "sha256": plan_sha256},
        {"path": "run/training_start_request.json", "sha256": request_sha256},
    ]
    findings = [] if findings is None else findings
    return {
        "schema_version": REVIEW_ATTESTATION_SCHEMA_VERSION,
        "reviewer": "feature-dev:code-reviewer",
        "scope": f"training_start_project_plan:{run_id}",
        "target_records": target_records,
        "target_sha256": runner.canonical_json_sha256(target_records),
        "findings": findings,
        "finding_count": len(findings) if finding_count is None else finding_count,
    }


def _training_start_candidate_bytes(launch_bytes, run_id, request_sha256):
    binding = f"Training start request `{run_id}` binds request SHA-256 `{request_sha256}`; these reviewed bytes become canonical only at the atomic training-start commit.\n"
    return launch_bytes + binding.encode("utf-8")


def _write_training_start_candidate(evidence, candidate_bytes, request_sha256, run_id, findings=None, finding_count=None):
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    (evidence / f"{candidate_sha256}.project-plan.md").write_bytes(candidate_bytes)
    artifact = _training_start_attestation(candidate_sha256, request_sha256, run_id, findings, finding_count)
    raw = runner.canonical_json_bytes(artifact)
    (evidence / f"{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)
    return artifact


def _materialize_training_start_lifecycle(root, repository, state):
    launch_bytes = (root / "run" / "project_plan_launch.md").read_bytes()
    launch_sha256 = hashlib.sha256(launch_bytes).hexdigest()
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_bytes(launch_bytes)
    if state == "not_started":
        return launch_bytes, None
    request = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "boundary": "training_start_review_request",
        "review_request_monotonic_ns": 100,
        "review_request_wall_utc": "2026-07-21T00:00:00Z",
        "launch_project_plan_sha256": launch_sha256,
        "required_review_scope": f"training_start_project_plan:{root.name}",
        "review_wait_timeout_seconds": 1800,
    }
    request_path = root / "run" / "training_start_request.json"
    request_path.write_bytes(runner.canonical_json_bytes(request))
    if state == "awaiting_review":
        return launch_bytes, None
    request_sha256 = runner.sha256_file(request_path)
    candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidate_path = root / "run" / "project_plan_training_start.md"
    candidate_path.write_bytes(candidate_bytes)
    review = _training_start_attestation(candidate_sha256, request_sha256, root.name)
    review_raw = runner.canonical_json_bytes(review)
    review_sha256 = hashlib.sha256(review_raw).hexdigest()
    review_path = f"run/reviews/{review_sha256}.json"
    (root / review_path).write_bytes(review_raw)
    linkage = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "boundary": "reviewed_plan_atomic_training_start",
        "request_path": "run/training_start_request.json",
        "request_artifact_sha256": request_sha256,
        "launch_snapshot_path": "run/project_plan_launch.md",
        "training_start_snapshot_path": "run/project_plan_training_start.md",
        "review_path": review_path,
        "launch_project_plan_sha256": launch_sha256,
        "training_start_project_plan_sha256": candidate_sha256,
        "review_scope": f"training_start_project_plan:{root.name}",
        "review_target_sha256": review["target_sha256"],
        "review_artifact_sha256": review_sha256,
        "review_deadline_monotonic_ns": 100 + runner.TRAINING_START_REVIEW_WAIT_NS,
        "commit_admission_monotonic_ns": 101,
        "commit_admission_wall_utc": "2026-07-21T00:00:01Z",
        "start_commit_margin_seconds": 5,
        "start_commit_rule": "single_coordinator_atomic_replace_after_locked_exact_launch_recheck",
    }
    (root / "run" / "training_start_plan.json").write_bytes(runner.canonical_json_bytes(linkage))
    if state == "started":
        plan.write_bytes(candidate_bytes)
    return candidate_bytes, linkage


class _BarrierSignals:
    def __init__(self, inject_after_commit=False):
        self.pending_signal = None
        self.deferred = 0
        self.inject_after_commit = inject_after_commit

    def defer(self):
        self.deferred += 1

    def release(self):
        self.deferred -= 1
        return self.pending_signal

    def commit_guarded(self, boundary):
        if self.pending_signal is not None:
            return SimpleNamespace(committed=False, value=None, pending_signal=self.pending_signal)
        value = boundary()
        if self.inject_after_commit:
            self.pending_signal = 15
        return SimpleNamespace(committed=True, value=value, pending_signal=self.pending_signal)


def _training_start_barrier_surface(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    evidence.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    launch_bytes = b"launch-plan\n"
    plan.write_bytes(launch_bytes)
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, launch_bytes)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(
        runner,
        "_verify_active_frozen_hashes",
        lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors),
    )
    return repository, evidence, plan, root, launch_bytes, runner.capture_frozen_manifest_anchors(root)


def _close_writers(writers):
    for writer in writers.values():
        writer.close()


@pytest.fixture(scope="module")
def runtime_modules():
    return runner._import_runtime()


def _create_review_surface(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
    staging = tmp_path / "staging"
    evidence.mkdir(parents=True)
    (staging / "run").mkdir(parents=True)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    for _, paths in runner._review_scopes():
        for relative in paths:
            target = repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(f"surface:{relative}\n".encode("utf-8"))
    return repository, evidence, staging


def _synthetic_training_start_artifacts(run_id):
    launch_bytes = b"launch-plan\n"
    launch_sha256 = hashlib.sha256(launch_bytes).hexdigest()
    request = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": run_id,
        "boundary": "training_start_review_request",
        "review_request_monotonic_ns": 100,
        "review_request_wall_utc": "2026-07-21T00:00:00Z",
        "launch_project_plan_sha256": launch_sha256,
        "required_review_scope": f"training_start_project_plan:{run_id}",
        "review_wait_timeout_seconds": 1800,
    }
    request_raw = runner.canonical_json_bytes(request)
    request_sha256 = hashlib.sha256(request_raw).hexdigest()
    candidate_raw = _training_start_candidate_bytes(launch_bytes, run_id, request_sha256)
    candidate_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    review = _training_start_attestation(candidate_sha256, request_sha256, run_id)
    review_raw = runner.canonical_json_bytes(review)
    review_sha256 = hashlib.sha256(review_raw).hexdigest()
    review_path = f"run/reviews/{review_sha256}.json"
    linkage = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": run_id,
        "boundary": "reviewed_plan_atomic_training_start",
        "request_path": "run/training_start_request.json",
        "request_artifact_sha256": request_sha256,
        "launch_snapshot_path": "run/project_plan_launch.md",
        "training_start_snapshot_path": "run/project_plan_training_start.md",
        "review_path": review_path,
        "launch_project_plan_sha256": launch_sha256,
        "training_start_project_plan_sha256": candidate_sha256,
        "review_scope": f"training_start_project_plan:{run_id}",
        "review_target_sha256": review["target_sha256"],
        "review_artifact_sha256": review_sha256,
        "review_deadline_monotonic_ns": 100 + runner.TRAINING_START_REVIEW_WAIT_NS,
        "commit_admission_monotonic_ns": 101,
        "commit_admission_wall_utc": "2026-07-21T00:00:01Z",
        "start_commit_margin_seconds": 5,
        "start_commit_rule": "single_coordinator_atomic_replace_after_locked_exact_launch_recheck",
    }
    return request_raw, candidate_raw, review_path, review_raw, runner.canonical_json_bytes(linkage)


def _synthetic_closure_paths(payload, kind, review_digests, detail_digests=(), training_start_state="not_started", run_id="synthetic"):
    closures = payload["artifacts"]["artifact_closures"]
    base = set(closures["five_global_control_files"])
    base.add("run/project_plan_launch.md")
    base.add("run/sentinels/selected_attention_oracle_payload.json")
    base.update(f"run/reviews/{digest}.json" for digest in review_digests)
    base.update(f"run/check_details/{digest}.json" for digest in detail_digests)
    if kind == "prepilot_abort":
        base.update(("ABORTED.json", "SHA256SUMS"))
    elif kind == "pilot_abort":
        base.update(("run/pilot_resources.jsonl", "ABORTED.json", "SHA256SUMS"))
    elif kind == "pilot_stop":
        base.update(("run/pilot_resources.jsonl", "run/pilot.json", "SHA256SUMS"))
    elif kind == "claim_abort":
        base.update(("run/pilot_resources.jsonl", "run/pilot.json", *CLAIM_LEDGER_PATHS, "ABORTED.json", "SHA256SUMS"))
        if training_start_state in {"awaiting_review", "reviewed_ready", "started"}:
            base.add("run/training_start_request.json")
        if training_start_state in {"reviewed_ready", "started"}:
            _, _, review_path, _, _ = _synthetic_training_start_artifacts(run_id)
            base.update(("run/project_plan_training_start.md", "run/training_start_plan.json", review_path))
    elif kind == "clean":
        base.update(closures["clean_completion"]["global_files"])
        base.update(closures["fixed_data_artifacts"])
        for seed in closures["rung_one_construction_seeds"]:
            base.update(f"rung1/{seed}/{suffix}" for suffix in closures["rung_one_clean_file_suffixes_per_seed"])
        base.update(f"rung2/83/{suffix}" for suffix in closures["rung_two_clean_file_suffixes"])
        base.update(closures["clean_completion"]["final_files"])
        _, _, review_path, _, _ = _synthetic_training_start_artifacts(run_id)
        base.update(("run/project_plan_training_start.md", "run/training_start_request.json", "run/training_start_plan.json", review_path))
    return base


def _materialize_synthetic_closure(root, payload, kind, detail_digest=None, training_start_state="not_started"):
    review_digests = tuple(f"{value:064x}" for value in range(1, 5))
    detail_digests = () if detail_digest is None else (detail_digest,)
    paths = _synthetic_closure_paths(payload, kind, review_digests, detail_digests, training_start_state, root.name)
    for relative in sorted(paths - {"SHA256SUMS"}):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    config = {"review_records": [{"artifact_sha256": digest} for digest in review_digests]}
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config))
    (root / "run" / "project_plan_launch.md").write_bytes(b"launch-plan\n")
    if kind == "claim_abort":
        (root / "ABORTED.json").write_bytes(runner.canonical_json_bytes({"training_start_state": training_start_state}))
    requested = kind == "clean" or training_start_state in {"awaiting_review", "reviewed_ready", "started"}
    reviewed = kind == "clean" or training_start_state in {"reviewed_ready", "started"}
    if requested:
        request_raw, candidate_raw, review_path, review_raw, linkage_raw = _synthetic_training_start_artifacts(root.name)
        (root / "run" / "training_start_request.json").write_bytes(request_raw)
    if reviewed:
        (root / "run" / "project_plan_training_start.md").write_bytes(candidate_raw)
        (root / review_path).write_bytes(review_raw)
        (root / "run" / "training_start_plan.json").write_bytes(linkage_raw)
    detail_reference = {} if detail_digest is None else {"checks": [{"details_sha256": detail_digest}]}
    (root / "run" / "preflight.json").write_text(json.dumps(detail_reference), encoding="utf-8")
    if kind == "clean":
        for seed in (11, 23, 37, 53, 71):
            (root / "rung1" / str(seed) / "parity.json").write_text("{}", encoding="utf-8")
        (root / "rung2" / "83" / "parity.json").write_text("{}", encoding="utf-8")
    return paths


def _set_synthetic_live_plan(tmp_path, monkeypatch, root, state):
    repository = tmp_path / "repository"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    source = root / ("run/project_plan_training_start.md" if state == "started" else "run/project_plan_launch.md")
    plan.write_bytes(source.read_bytes())
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)


def _metric_values(metric, numerator, denominator):
    values = {
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator if denominator is None else None,
        "answer_correct": None,
        "answer_total": None,
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
    }
    if metric == "answer_accuracy":
        values.update(answer_correct=numerator, answer_total=denominator)
    elif metric == "original_source_hit_rate":
        values.update(original_source_hits=numerator, original_source_total=denominator)
    elif metric == "foreign_source_hit_rate":
        values.update(foreign_source_hits=numerator, foreign_source_total=denominator)
    elif metric == "query_underfill_count":
        values["query_underfill_count"] = numerator
    return values


def _passing_gate_numerator(registry_row):
    target = registry_row["gate_threshold_count"]
    if target is None:
        target = registry_row["gate_threshold"]
    if registry_row["gate_operator"] == ">=":
        return target
    if registry_row["gate_operator"] == "<=":
        return 0
    return target


def _checkpoint_path(root, rung, seed, condition):
    if rung == 2:
        return root / "rung2" / "83" / "checkpoints" / "final_last.pt"
    name = {
        "all_eligible_donor": "donor_last.pt",
        "dense_causal": "dense_last.pt",
    }.get(condition, "final_last.pt")
    return root / "rung1" / str(seed) / "checkpoints" / name


def _evaluation_data_path(root, rung, seed):
    return root / "data" / (f"r1_eval_{400000 + seed}.pt" if rung == 1 else "r2_eval_1000083.pt")


def _materialize_endpoint_bytes(root, rung, seed, conditions):
    evaluation_path = _evaluation_data_path(root, rung, seed)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_bytes(f"evaluation:{rung}:{seed}".encode("ascii"))
    for condition in conditions:
        checkpoint_path = _checkpoint_path(root, rung, seed, condition)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            checkpoint_path.write_bytes(f"checkpoint:{rung}:{seed}:{checkpoint_path.name}".encode("ascii"))


def _synthetic_rung_one_evaluation_rows(root, payload, seed):
    state_manifest = root / "rung1" / str(seed) / "selected_canonical_state_manifest.json"
    oracle_payload = root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    oracle_detail = root / "rung1" / str(seed) / "selected_attention_oracle_detail.json"
    for path, raw in (
        (state_manifest, f"state-{seed}".encode("ascii")),
        (oracle_payload, b"oracle-payload"),
        (oracle_detail, f"oracle-detail-{seed}".encode("ascii")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(raw)
    oracle_provenance = [runner.sha256_file(path) for path in (state_manifest, oracle_payload, oracle_detail)]
    registry = {
        (row["condition"], row["metric"], row["stratum"]): runner._resolved_gate_registry_row(row, seed)
        for row in payload["gates"]["rung_one_registry"]
    }
    rows = []
    for condition, metric, stratum in runner._rung_one_evaluation_identity_order(payload):
        registered = registry.get((condition, metric, stratum))
        if registered is not None:
            denominator = registered["denominator"]
            numerator = _passing_gate_numerator(registered)
            gate_id = registered["gate_id"]
            gate = (
                registered["gate_operator"],
                registered["gate_threshold"],
                registered["gate_threshold_count"],
                registered["gate_threshold_unit"],
            )
        elif metric in {"answer_accuracy", "original_source_hit_rate", "foreign_source_hit_rate"}:
            denominator = 512 if stratum == "all" else 1
            numerator = 0
            gate_id = None
            gate = (None, None, None, None)
        else:
            denominator = None
            numerator = 0
            gate_id = None
            gate = (None, None, None, None)
        values = _metric_values(metric, numerator, denominator)
        indices = list(range(denominator)) if denominator is not None else None
        checkpoint = runner.sha256_file(_checkpoint_path(root, 1, seed, condition))
        evaluation = runner.sha256_file(_evaluation_data_path(root, 1, seed))
        provenance = [checkpoint, evaluation]
        if metric == "selected_mask_oracle_max_error":
            values["estimate"] = 0.0
            values["numerator"] = None
            values["selected_mask_oracle_max_error"] = 0.0
            checkpoint = None
            evaluation = None
            provenance = oracle_provenance
        elif metric == "route_overflow_count":
            values["overflow_count"] = 0
            values["max_bucket_load"] = 1
            checkpoint = None
            evaluation = None
            provenance = [runner.sha256_file(root / "rung1" / str(seed) / "routing.jsonl.gz")]
        row = runner._evaluation_row(
            root.name,
            seed,
            condition,
            metric,
            stratum,
            indices,
            values,
            checkpoint,
            evaluation,
            provenance,
            gate_id,
            gate,
            0.25,
            [0],
        )
        rows.append(row)
    return rows


def _synthetic_gate_statistics(root, payload):
    records = [
        {"block": block, "head": head, "count": 512 * 512, "mean": 0.5, "population_std": 0.0, "min": 0.5, "max": 0.5, "nonfinite_count": 0}
        for block in (1, 2, 3, 5, 6, 7)
        for head in range(4)
    ]
    conditions = []
    for registry_row in payload["gates"]["rung_two_registry"][2:]:
        conditions.append(
            {
                "condition": registry_row["condition"],
                "gate_id": registry_row["gate_id"],
                "records": copy.deepcopy(records),
                "aggregate": {"block": None, "head": None, "count": 24 * 512 * 512, "mean": 0.5, "population_std": 0.0, "min": 0.5, "max": 0.5, "nonfinite_count": 0},
                "gate_operator": registry_row["gate_operator"],
                "gate_threshold": registry_row["gate_threshold"],
                "gate_threshold_count": registry_row["gate_threshold_count"],
                "gate_threshold_unit": registry_row["gate_threshold_unit"],
                "gate_pass": True,
            }
        )
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "rung": 2,
        "claim_seed": 83,
        "construction_seed": 83,
        "checkpoint_sha256": runner.sha256_file(_checkpoint_path(root, 2, 83, "intact")),
        "conditions": conditions,
    }


def _write_jsonl_rows(path, rows, canonical=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(b"".join(runner.canonical_json_bytes(row) + b"\n" for row in rows))
    else:
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _synthetic_evaluation_payload(rung):
    if rung == 1:
        return {
            "condition": [index % 4 for index in range(512)],
            "required_source": [index % 15 for index in range(512)],
            "targets": [index % 32 for index in range(512)],
            "rule_blocks": [[(index + condition) % 15 for condition in range(4)] for index in range(512)],
        }
    return {"targets": [[0] * 510 + [index % 32, 0] for index in range(512)]}


def _prediction_validation_inputs(root, rung, seed):
    conditions = runner.RUNG_ONE_CONDITIONS if rung == 1 else runner.RUNG_TWO_CONDITIONS
    checkpoint_by_condition = {
        condition: (
            runner.RUNG_ONE_MODEL_BY_CONDITION[condition] if rung == 1 else "rung_two",
            runner.sha256_file(_checkpoint_path(root, rung, seed, condition)),
        )
        for condition in conditions
    }
    return checkpoint_by_condition, _synthetic_evaluation_payload(rung), runner.sha256_file(_evaluation_data_path(root, rung, seed))


def _materialize_rung_two_endpoint_evidence(root, payload, torch):
    rows = _synthetic_prediction_rows(root, payload, 2, 83)
    checkpoint_path = _checkpoint_path(root, 2, 83, "intact")
    checkpoint = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "rung": 2,
        "construction_seed": 83,
        "model": "rung_two",
        "stage": "rung2",
        "completed_update": 1536,
        "last_attempt_id": "a" * 64,
        "model_state_dict": {"weight": torch.ones(2)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"update": 1536},
        "python_rng_state": [],
        "torch_rng_state": torch.zeros(4, dtype=torch.uint8),
        "generator_states": {"data": torch.zeros(4, dtype=torch.uint8)},
        "final_batch_sha256": "b" * 64,
    }
    checkpoint_path.unlink()
    runner._save_torch_artifact(checkpoint_path, checkpoint, torch)
    evaluation_payload = runner.generate_rung_two_batch(1000083, 512, torch)
    evaluation_path = _evaluation_data_path(root, 2, 83)
    evaluation_path.unlink()
    runner._save_torch_artifact(
        evaluation_path,
        {
            "seed": 1000083,
            "payload": evaluation_payload,
            "payload_sha256": runner.canonical_json_sha256(evaluation_payload),
        },
        torch,
    )
    checkpoint_sha256 = runner.sha256_file(checkpoint_path)
    evaluation_sha256 = runner.sha256_file(evaluation_path)
    for row in rows:
        target = int(evaluation_payload["targets"][row["example_index"]][510])
        row["target"] = target
        row["prediction"] = target
        row["correct"] = True
        row["checkpoint_sha256"] = checkpoint_sha256
    seed_root = root / "rung2" / "83"
    _write_prediction_rows(seed_root / "predictions.jsonl.gz", rows)
    evaluation_rows = [
        runner._rung_two_evaluation_row(root.name, condition, 512, checkpoint_sha256, evaluation_sha256, 0.25, [0])
        for condition in runner.RUNG_TWO_CONDITIONS
    ]
    return seed_root, evaluation_rows


def _materialize_synthetic_gate_package(root, payload):
    for seed in (11, 23, 37, 53, 71):
        conditions = payload["stages"]["rung_one"]["evaluation_arm_order"]
        _materialize_endpoint_bytes(root, 1, seed, conditions)
        runner._write_canonical_gzip(root / "rung1" / str(seed) / "routing.jsonl.gz", [])
        _write_jsonl_rows(root / "rung1" / str(seed) / "evaluation.jsonl", _synthetic_rung_one_evaluation_rows(root, payload, seed))
    r2_conditions = payload["stages"]["rung_two"]["evaluation_order"]
    _materialize_endpoint_bytes(root, 2, 83, r2_conditions)
    checkpoint_sha256 = runner.sha256_file(_checkpoint_path(root, 2, 83, "intact"))
    evaluation_sha256 = runner.sha256_file(_evaluation_data_path(root, 2, 83))
    r2_rows = [
        runner._rung_two_evaluation_row(root.name, "intact", 461, checkpoint_sha256, evaluation_sha256, 0.25, [0]),
        runner._rung_two_evaluation_row(root.name, "recurrent_knockout", 0, checkpoint_sha256, evaluation_sha256, 0.25, [0]),
    ]
    r2_root = root / "rung2" / "83"
    _write_jsonl_rows(r2_root / "evaluation.jsonl", r2_rows)
    (r2_root / "gate_stats.json").write_bytes(runner.canonical_json_bytes(_synthetic_gate_statistics(root, payload)))


def _synthetic_prediction_rows(root, payload, rung, seed):
    conditions = payload["stages"]["rung_one"]["evaluation_arm_order"] if rung == 1 else payload["stages"]["rung_two"]["evaluation_order"]
    _materialize_endpoint_bytes(root, rung, seed, conditions)
    evaluation_payload = _synthetic_evaluation_payload(rung)
    foreign_conditions = []
    if rung == 1:
        for start in range(0, 512, 32):
            values = evaluation_payload["condition"][start : start + 32]
            foreign_conditions.extend(values[-1:] + values[:-1])
    rows = []
    for condition in conditions:
        for index in range(512):
            target = evaluation_payload["targets"][index] if rung == 1 else evaluation_payload["targets"][index][510]
            if rung == 1:
                original_condition = evaluation_payload["condition"][index]
                original_source = evaluation_payload["required_source"][index]
                original_source_hit = None if condition == "dense_causal" else index % 2 == 0
                if condition == "carry_shuffle":
                    foreign_condition = foreign_conditions[index]
                    foreign_source = evaluation_payload["rule_blocks"][index][foreign_condition]
                    foreign_source_hit = index % 3 == 0
                    condition_stratum = "same_condition" if foreign_condition == original_condition else "changed_condition"
                else:
                    foreign_condition = None
                    foreign_source = None
                    foreign_source_hit = None
                    condition_stratum = "not_applicable"
            else:
                original_condition = None
                foreign_condition = None
                original_source = None
                foreign_source = None
                original_source_hit = None
                foreign_source_hit = None
                condition_stratum = "not_applicable"
            rows.append(
                {
                    "schema_version": runner.SCHEMA_VERSION,
                    "run_id": root.name,
                    "rung": rung,
                    "claim_seed": seed,
                    "construction_seed": seed,
                    "condition": condition,
                    "example_index": index,
                    "original_condition": original_condition,
                    "foreign_condition": foreign_condition,
                    "original_source": original_source,
                    "foreign_source": foreign_source,
                    "target": target,
                    "prediction": target,
                    "correct": True,
                    "original_source_hit": original_source_hit,
                    "foreign_source_hit": foreign_source_hit,
                    "condition_stratum": condition_stratum,
                    "checkpoint_sha256": runner.sha256_file(_checkpoint_path(root, rung, seed, condition)),
                }
            )
    return rows


def _write_prediction_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    runner._write_canonical_gzip(path, rows)


def _canonical_bypass_records(width):
    if width == 0:
        return []
    last_position = 23 if width == 2 else 127
    records = []
    for position in range(last_position + 1):
        remote_limit = position // 8
        raw_ids = list(range(min(remote_limit, width)))
        raw_ids.extend([-1] * (width - len(raw_ids)))
        effective_ids = raw_ids if position == 126 else [-1] * width
        records.append(
            {
                "position": position,
                "remote_limit": remote_limit,
                "raw_remote_ids": raw_ids,
                "effective_remote_ids": effective_ids,
            }
        )
    return records


def _routing_histogram_fields(width, batch_size):
    search_rows = 0 if width in {0, 15} else batch_size * 104
    if search_rows:
        valid_histogram = [
            {"valid_posting_count": 0, "search_row_count": search_rows - batch_size},
            {"valid_posting_count": 2, "search_row_count": batch_size},
        ]
    else:
        valid_histogram = []
    return {
        "canonical_bypass_ids": _canonical_bypass_records(width),
        "block_load_histogram": [{"load": 1, "bucket_count": batch_size * 16}],
        "valid_posting_histogram": valid_histogram,
        "addresses_probed": search_rows * 4,
        "posting_reads": batch_size * 2 if search_rows else 0,
        "candidate_blocks": batch_size * 2 if search_rows else 0,
        "overflow_count": 0,
        "max_bucket_load": 1,
    }


def _routing_call_summary(root, seed, block, forward_sequence=0, phase="evaluation", condition="intact"):
    width = runner._route_width("selected", block)
    row = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "rung": 1,
        "claim_seed": seed,
        "construction_seed": seed,
        "row_kind": "call_summary",
        "phase": phase,
        "model": "selected",
        "stage": None,
        "condition": condition,
        "logical_update": None,
        "forward_sequence": forward_sequence,
        "block": block,
        "batch_index": 0,
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
        **_routing_histogram_fields(width, 1),
        "route_workspace_bytes": 8,
        "checkpoint_sha256": runner.sha256_file(_checkpoint_path(root, 1, seed, "intact")),
    }
    return row


def _routing_query_example(root, seed, block, example_index, forward_sequence=0):
    row = _routing_call_summary(root, seed, block, forward_sequence, "route_acquisition", None)
    row.update(
        {
            "row_kind": "query_example",
            "example_index": example_index,
            "query_position": 126,
            "required_source": example_index % 15,
            "raw_remote_ids": [0, 1],
            "local_block_ids": [15],
            "canonical_bypass_ids": None,
            "block_load_histogram": None,
            "valid_posting_histogram": None,
            "addresses_probed": None,
            "posting_reads": None,
            "candidate_blocks": None,
            "overflow_count": None,
            "max_bucket_load": None,
            "route_workspace_bytes": None,
        }
    )
    return row


def _full_routing_fixture(root, seed=11, torch=None, evaluation_payload=None):
    checkpoint_by_condition = {
        condition: (
            runner.RUNG_ONE_MODEL_BY_CONDITION[condition],
            runner.sha256_file(_checkpoint_path(root, 1, seed, condition)),
        )
        for condition in runner.RUNG_ONE_CONDITIONS
    }
    specs = runner._expected_rung_one_forward_specs(checkpoint_by_condition)
    rows = []
    prediction_by_condition = {condition: [] for condition in runner.RUNG_ONE_CONDITIONS}
    required_sources = list(evaluation_payload["required_source"]) if evaluation_payload is not None else [index % 15 for index in range(512)]
    exclusion_routes = None
    if torch is not None:
        exclusion_routes = runner.generate_source_exclusion_routes(510000 + seed, [[0, 1] for _ in range(512)], required_sources, torch)["routes"]
    for sequence, spec in enumerate(specs):
        for block in spec["blocks"]:
            call = _routing_call_summary(
                root,
                seed,
                block,
                sequence,
                spec["phase"],
                spec["condition"],
            )
            call.update(
                {
                    "model": spec["model"],
                    "stage": spec["stage"],
                    "logical_update": spec["logical_update"],
                    "batch_index": spec["batch_index"],
                    **_routing_histogram_fields(runner._route_width(spec["model"], block), spec["batch_size"]),
                    "route_workspace_bytes": _expected_routing_workspace_bytes(spec["model"], block, spec["batch_size"]),
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                }
            )
            rows.append(call)
            width = runner._route_width(spec["model"], block)
            raw_ids = list(range(width))
            for example_index in range(spec["example_start"], spec["example_start"] + spec["batch_size"]):
                required_source = required_sources[example_index] if example_index < 512 else example_index % 15
                effective_ids = None if spec["phase"] == "route_acquisition" else raw_ids
                if spec["condition"] == "required_source_excluded" and block == 4 and exclusion_routes is not None:
                    effective_ids = exclusion_routes[example_index]
                original_hit = None if effective_ids is None else required_source in effective_ids
                foreign_source = (required_source + 1) % 15 if spec["condition"] == "carry_shuffle" else None
                foreign_hit = None if foreign_source is None else foreign_source in effective_ids
                query = {
                    **call,
                    "row_kind": "query_example",
                    "example_index": example_index,
                    "query_position": 126,
                    "required_source": required_source,
                    "foreign_source": foreign_source,
                    "raw_remote_ids": raw_ids,
                    "effective_remote_ids": effective_ids,
                    "local_block_ids": [15],
                    "query_underfill_count": None if effective_ids is None else effective_ids.count(-1),
                    "original_source_hit": original_hit,
                    "foreign_source_hit": foreign_hit,
                    "intervention": spec["condition"] if spec["phase"] == "evaluation" else None,
                    "canonical_bypass_ids": None,
                    "block_load_histogram": None,
                    "valid_posting_histogram": None,
                    "addresses_probed": None,
                    "posting_reads": None,
                    "candidate_blocks": None,
                    "overflow_count": None,
                    "max_bucket_load": None,
                    "route_workspace_bytes": None,
                }
                rows.append(query)
                if spec["phase"] == "evaluation" and block == 4:
                    prediction_by_condition[spec["condition"]].append(
                        {
                            "original_source": required_source,
                            "original_source_hit": original_hit,
                            "foreign_source": foreign_source,
                            "foreign_source_hit": foreign_hit,
                            "checkpoint_sha256": spec["checkpoint_sha256"],
                        }
                    )
    prediction_evidence = {"by_condition": prediction_by_condition}
    evaluation_rows = [{"metric": "route_overflow_count", "numerator": 0, "overflow_count": 0, "max_bucket_load": 1}]
    return rows, evaluation_rows, prediction_evidence, checkpoint_by_condition


def _claim_package_shell(root, payload):
    _materialize_synthetic_gate_package(root, payload)
    sentinel = root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    population_by_seed = {}
    detail_paths = []
    for seed in (11, 23, 37, 53, 71, 83):
        rung = 1 if seed != 83 else 2
        seed_root = root / (f"rung1/{seed}" if rung == 1 else "rung2/83")
        evaluation_rows = [json.loads(line) for line in (seed_root / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
        populations = {}
        for row in evaluation_rows:
            if row["population_sha256"] is None:
                continue
            denominator = row["denominator"]
            size = 512 if row["stratum"] == "all" else 0 if denominator is None else denominator
            populations[(row["condition"], row["stratum"])] = list(range(size))
        population_by_seed[seed] = populations
        state = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "records": [],
        }
        (seed_root / "state_stats.json").write_bytes(runner.canonical_json_bytes(state))
        intervention = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "records": [],
        }
        (seed_root / "intervention_deltas.json").write_bytes(runner.canonical_json_bytes(intervention))
        checks = []
        for index, scope in enumerate(runner.PARITY_SCOPES):
            name = f"seed_{seed}_{scope}"
            detail = {
                "schema_version": runner.SCHEMA_VERSION,
                "run_id": root.name,
                "name": name,
                "scope": scope,
                "inputs": {},
                "outputs": {},
                "evidence_paths": [],
            }
            raw = runner.canonical_json_bytes(detail)
            digest = hashlib.sha256(raw).hexdigest()
            detail_path = root / "run" / "check_details" / f"{digest}.json"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            detail_path.write_bytes(raw)
            detail_paths.append(detail_path)
            checks.append({"name": name, "scope": scope, "max_error": None, "tolerance": None, "pass": True, "details_sha256": digest})
        parity = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "checkpoint_sha256": runner.sha256_file(_checkpoint_path(root, rung, seed, "intact")),
            "checks": checks,
        }
        (seed_root / "parity.json").write_bytes(runner.canonical_json_bytes(parity))
        refs = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "sample_ids": [0],
        }
        (seed_root / "resource_refs.json").write_bytes(runner.canonical_json_bytes(refs))
        gradient_artifact = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "records": [],
        }
        (seed_root / "grad_audit.json").write_bytes(runner.canonical_json_bytes(gradient_artifact))
        if rung == 1:
            (seed_root / "dense_grad_audit.json").write_bytes(runner.canonical_json_bytes(gradient_artifact))
        entries = sorted(
            (
                {"category": category, "name": category, "count": 0, "bytes": 0}
                for category in (
                    "active_learned_parameter",
                    "serialized_without_gradient",
                    "inactive_parameter",
                    "registered_buffer",
                    "dynamic_recurrent_state",
                    "route_index_storage",
                    "routing_workspace",
                    "optimizer_state",
                )
            ),
            key=lambda row: row["name"],
        )
        accounting = {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": root.name,
            "rung": rung,
            "claim_seed": seed,
            "construction_seed": seed,
            "models": [
                {
                    "model": "selected" if rung == 1 else "rung_two",
                    "entries": entries,
                    "attempted_updates": 0,
                    "completed_updates": 0,
                    "attempted_token_positions": 0,
                    "resource_sample_ids": [0],
                }
            ],
        }
        (seed_root / "accounting.json").write_bytes(runner.canonical_json_bytes(accounting))
        if rung == 1:
            manifest = {
                "schema_version": runner.SCHEMA_VERSION,
                "run_id": root.name,
                "construction_seed": seed,
                "role": "selected_canonical",
                "state_tensors": [],
                "state_sha256": runner.canonical_json_sha256([]),
            }
            manifest_path = seed_root / "selected_canonical_state_manifest.json"
            manifest_path.write_bytes(runner.canonical_json_bytes(manifest))
            detail = {
                "schema_version": runner.SCHEMA_VERSION,
                "run_id": root.name,
                "construction_seed": seed,
                "constructor_state_manifest_sha256": runner.sha256_file(manifest_path),
                "sentinel_payload_sha256": runner.sha256_file(sentinel),
                "max_error": 0.0,
                "tolerance": 1e-5,
                "pass": True,
            }
            (seed_root / "selected_attention_oracle_detail.json").write_bytes(runner.canonical_json_bytes(detail))
    for seed in (11, 23, 37, 53, 71):
        _write_jsonl_rows(root / "rung1" / str(seed) / "evaluation.jsonl", _synthetic_rung_one_evaluation_rows(root, payload, seed))
    return population_by_seed, detail_paths


def _selected_oracle_fixture(root, payload, runtime, seed=11):
    seed_root = root / "rung1" / str(seed)
    seed_root.mkdir(parents=True)
    sentinel_path = root / "run" / "sentinels" / "selected_attention_oracle_payload.json"
    sentinel_path.parent.mkdir(parents=True)
    model, _, _ = runner._construct_seeded_model("selected", seed, runtime)
    state_tensors, state_sha256 = runner._state_manifest(model)
    manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "construction_seed": seed,
        "role": "selected_canonical",
        "state_tensors": state_tensors,
        "state_sha256": state_sha256,
    }
    manifest_path = seed_root / "selected_canonical_state_manifest.json"
    manifest_path.write_bytes(runner.canonical_json_bytes(manifest))
    sentinel = runner._sentinel_payload(runtime)
    sentinel_path.write_bytes(runner.canonical_json_bytes(sentinel))
    error = runner._selected_attention_oracle_for_model(model, runtime, sentinel)
    detail = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "construction_seed": seed,
        "constructor_state_manifest_sha256": runner.sha256_file(manifest_path),
        "sentinel_payload_sha256": runner.sha256_file(sentinel_path),
        "max_error": error,
        "tolerance": 1e-5,
        "pass": True,
    }
    detail_path = seed_root / "selected_attention_oracle_detail.json"
    detail_path.write_bytes(runner.canonical_json_bytes(detail))
    evaluation_rows = [
        {
            "metric": "selected_mask_oracle_max_error",
            "estimate": error,
            "selected_mask_oracle_max_error": error,
            "provenance_sha256s": [runner.sha256_file(manifest_path), runner.sha256_file(sentinel_path), runner.sha256_file(detail_path)],
        }
    ]
    parity_detail = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "name": f"r1_seed_{seed}_attention",
        "scope": "attention",
        "inputs": {"oracle": "selected_attention"},
        "outputs": {"max_error": error},
        "evidence_paths": [],
    }
    parity_raw = runner.canonical_json_bytes(parity_detail)
    parity_sha256 = hashlib.sha256(parity_raw).hexdigest()
    parity_path = root / "run" / "check_details" / f"{parity_sha256}.json"
    parity_path.parent.mkdir(parents=True)
    parity_path.write_bytes(parity_raw)
    parity = {
        "checks": [
            {
                "name": parity_detail["name"],
                "scope": "attention",
                "max_error": error,
                "tolerance": 1e-5,
                "pass": True,
                "details_sha256": parity_sha256,
            }
        ]
    }
    return evaluation_rows, parity, {
        "manifest": manifest_path,
        "sentinel": sentinel_path,
        "oracle": detail_path,
        "parity_detail": parity_path,
    }


def _stub_claim_package_semantics(monkeypatch, root, populations, calls=None):
    def record(value):
        if calls is not None:
            calls.append(value)

    def endpoints(run_root, rung, seed, torch):
        record(("endpoints", seed))
        if rung == 1:
            result = {}
            for stage, (filename, _, _, _) in runner.RUNG_ONE_STAGE_ENDPOINTS.items():
                path = root / "rung1" / str(seed) / "checkpoints" / filename
                if not path.exists():
                    condition = "dense_causal" if stage.startswith("dense") else "intact"
                    path = _checkpoint_path(root, 1, seed, condition)
                result[stage] = {
                    "sha256": runner.sha256_file(path),
                    "checkpoint": {"model_state_dict": {}},
                }
            return result
        return {
            "rung_two": {
                "sha256": runner.sha256_file(root / "rung2" / "83" / "checkpoints" / "final_last.pt"),
                "checkpoint": {"model_state_dict": {}},
            }
        }

    def train(run_root, payload, seed_root, rung, seed, torch):
        record(("train", seed))
        if rung == 1:
            return [{"stage": "donor", "logical_update": 1, "batch_sha256": f"{seed:064x}"}]
        return [{"stage": "rung_two", "logical_update": 1, "batch_sha256": f"{seed:064x}"}]

    def prediction(run_root, payload, seed_root, rung, seed, checkpoint_by_condition, evaluation_payload, eval_data_sha256):
        record(("predictions", seed))
        conditions = runner.RUNG_ONE_CONDITIONS if rung == 1 else runner.RUNG_TWO_CONDITIONS
        return {
            "records": [],
            "by_condition": {condition: [] for condition in conditions},
            "populations": populations[seed],
            "eval_data_sha256": eval_data_sha256,
        }

    def semantic_parity(run_root, rung, seed, endpoints, train_rows, evaluation_rows, evaluation_payload, intervention_records, parity, runtime, routing_evidence, data_evidence, oracle_evidence):
        for check in parity["checks"]:
            path = run_root / "run" / "check_details" / f"{check['details_sha256']}.json"
            runner._canonical_json_artifact(path)
            if runner.sha256_file(path) != check["details_sha256"]:
                raise runner.ContractError("parity detail digest differs")
        record(("semantic_parity", seed))

    monkeypatch.setattr(runner, "validate_parent_ledger_accounting", lambda *args: record("ledger"))
    monkeypatch.setattr(runner, "validate_preclaim_reconstruction", lambda *args: record("preclaim"))
    monkeypatch.setattr(runner, "_load_claim_endpoints", endpoints)
    monkeypatch.setattr(runner, "_validate_train_artifact", train)
    monkeypatch.setattr(runner, "_validate_endpoint_train_closure", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_load_evaluation_evidence",
        lambda run_root, rung, seed, torch: (_synthetic_evaluation_payload(rung), runner.sha256_file(_evaluation_data_path(root, rung, seed))),
    )
    monkeypatch.setattr(runner, "_validate_prediction_artifact", prediction)
    monkeypatch.setattr(
        runner,
        "_validate_routing_artifact",
        lambda run_root, payload, seed_root, seed, evaluation_rows, prediction_evidence, checkpoint_by_condition: record(("routing", seed))
        or {"query_by_condition": {}, "underfill_by_condition": {}, "workspace_bytes_by_model": {}, "workspace_count_by_model": {}},
    )
    monkeypatch.setattr(runner, "_validate_evaluation_reconstruction", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_validate_rung_one_data_artifacts",
        lambda run_root, seed, evaluation_payload, routing_evidence, torch: {"same_condition": 0, "changed_condition": 512},
    )
    monkeypatch.setattr(runner, "validate_state_records", lambda *args: record("state"))
    monkeypatch.setattr(runner, "validate_intervention_records", lambda records, rung, checkpoint_by_condition: record(("intervention", rung)))
    monkeypatch.setattr(runner, "_validate_gradient_artifact", lambda *args: [])
    monkeypatch.setattr(runner, "validate_model_accounting", lambda *args: None)
    monkeypatch.setattr(runner, "_validate_accounting_evidence", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_validate_selected_oracle_evidence",
        lambda run_root, payload, seed, evaluation_rows, runtime: record(("oracle", seed)) or {"attention_error": 0.0},
    )
    monkeypatch.setattr(runner, "_validate_semantic_parity_package", semantic_parity)
    monkeypatch.setattr(runner, "_validate_carry_shuffle_strata", lambda *args: None)
    monkeypatch.setattr(runner, "validate_gate_input_package", lambda *args: record("gates") or [])


def _review_attestation(repository, scope, finding_count=0, transform_records=None):
    paths = dict(runner._review_scopes())[scope]
    records = [{"path": path, "sha256": runner.sha256_file(repository / path)} for path in paths]
    if transform_records is not None:
        records = transform_records(copy.deepcopy(records))
    return {
        "schema_version": REVIEW_ATTESTATION_SCHEMA_VERSION,
        "reviewer": "feature-dev:code-reviewer",
        "scope": scope,
        "target_records": records,
        "target_sha256": runner.canonical_json_sha256(records),
        "findings": [],
        "finding_count": finding_count,
    }


def _write_review_attestation(evidence, artifact):
    raw = runner.canonical_json_bytes(artifact)
    digest = hashlib.sha256(raw).hexdigest()
    path = evidence / f"{digest}.json"
    path.write_bytes(raw)
    return path, raw, digest


def _write_complete_review_set(repository, evidence):
    records = {}
    for scope, _ in runner._review_scopes():
        records[scope] = _write_review_attestation(evidence, _review_attestation(repository, scope))
    return records


def test_canonical_json_bytes_are_exact_and_reject_nonfinite() -> None:
    value = {"z": "\u00e9", "a": [2, 1]}
    expected = b'{"a":[2,1],"z":"\\u00e9"}'
    assert runner.canonical_json_bytes(value) == expected
    assert runner.canonical_json_sha256(value) == hashlib.sha256(expected).hexdigest()
    with pytest.raises((TypeError, ValueError)):
        runner.canonical_json_bytes({"bad": float("nan")})


def test_sha256_file_hashes_raw_bytes(tmp_path: Path) -> None:
    path = tmp_path / "raw.bin"
    path.write_bytes(b"abc\x00\xff")
    assert runner.sha256_file(path) == hashlib.sha256(b"abc\x00\xff").hexdigest()


@pytest.mark.parametrize(
    ("stage", "role", "seed"),
    (
        ("donor", "all_eligible", 11),
        ("router_only", "selected", 11),
        ("joint", "selected", 11),
        ("dense_base", "dense", 11),
        ("dense_continuation", "dense", 11),
        ("rung_two", "rung_two", 83),
    ),
)
def test_loaded_endpoint_parameter_classification_uses_named_parameters(tmp_path: Path, stage: str, role: str, seed: int) -> None:
    runtime = runner._import_runtime()
    source, _, _ = runner._construct_seeded_model(role, seed, runtime)
    path = tmp_path / "checkpoint.pt"
    runner._save_torch_artifact(path, {"model_state_dict": source.state_dict()}, runtime.torch)
    checkpoint = runner._load_claim_torch_artifact(path, runtime.torch, "test.loaded_endpoint")
    parameters = runner._loaded_endpoint_parameters(stage, checkpoint["model_state_dict"], seed, runtime)
    assert set(parameters) == set(checkpoint["model_state_dict"])
    assert all(isinstance(parameter, runtime.torch.nn.Parameter) for parameter in parameters.values())
    assert all(runner._tensor_sha256(parameters[name]) == runner._tensor_sha256(tensor) for name, tensor in checkpoint["model_state_dict"].items())
    assert {runtime.model_module.parameter_category(name, parameter) for name, parameter in parameters.items()} == {"matrix", "normalization_scale", "recurrent_bias", "codebook"}


def test_loaded_endpoint_parameter_classification_rejects_state_mutation() -> None:
    runtime = runner._import_runtime()
    source, _, _ = runner._construct_seeded_model("selected", 11, runtime)
    state = dict(source.state_dict())
    state["unregistered.weight"] = runtime.torch.zeros(1)
    with pytest.raises(runner.ContractError, match="reconstruction failed"):
        runner._loaded_endpoint_parameters("joint", state, 11, runtime)


def test_crash_atomic_writer_commits_canonical_line_before_ack(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    result = writer.append(_fixture_record(0))
    expected = runner.canonical_json_bytes(_fixture_record(0)) + b"\n"
    assert result.prior_offset == 0
    assert result.current_offset == len(expected)
    assert result.acknowledged is True
    assert result.committed is True
    assert result.reason_code is None
    assert result.line_sha256 == hashlib.sha256(expected).hexdigest()
    assert writer.last_committed_offset == len(expected)
    assert path.read_bytes() == expected
    assert writer.validate_committed_prefix() == (_fixture_record(0),)
    writer.close()


def test_attempt_writer_incremental_state_matches_full_sequence_validation(tmp_path: Path) -> None:
    path = tmp_path / "attempt-state.jsonl"
    writer = runner.CrashAtomicJsonlWriter(
        path,
        runner.validate_attempt_row,
        sequence_kind="attempt",
    )
    writer.precreate()
    rows = [_attempt_event(0, "started"), _attempt_event(1, "completed")]
    for row in rows:
        writer.append(row)
    with pytest.raises(runner.ContractError):
        writer.append(_attempt_event(2, "completed", logical_update=2))
    assert path.read_bytes() == b"".join(runner.canonical_json_bytes(row) + b"\n" for row in rows)
    assert writer.validate_committed_prefix() == tuple(rows)
    writer.close()


def test_crash_writer_append_read_work_scales_linearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_read = runner.os.read

    def measure(count: int) -> int:
        requested = [0]

        def counted_read(descriptor, size):
            requested[0] += size
            return original_read(descriptor, size)

        monkeypatch.setattr(runner.os, "read", counted_read)
        path = tmp_path / f"linear-{count}.jsonl"
        writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
        writer.precreate()
        for sequence in range(count):
            writer.append(_fixture_record(sequence))
        writer.close()
        return requested[0]

    small = measure(128)
    large = measure(256)
    assert large <= small * 2.2


@pytest.mark.parametrize("fault", FAULT_IDS[:5])
def test_first_five_crash_faults_restore_exact_prior_prefix(tmp_path: Path, fault: str) -> None:
    path = tmp_path / f"{fault}.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    committed = writer.append(_fixture_record(0))
    prior = path.read_bytes()
    with pytest.raises(runner.LedgerAppendError) as caught:
        writer.append(_fixture_record(1), fault=fault)
    result = caught.value.result
    assert result.prior_offset == committed.current_offset
    assert result.current_offset == committed.current_offset
    assert result.acknowledged is False
    assert result.committed is False
    assert result.reason_code == "artifact_inconsistency"
    assert result.line_sha256 is None
    assert writer.last_committed_offset == committed.current_offset
    assert path.read_bytes() == prior
    assert writer.validate_committed_prefix() == (_fixture_record(0),)
    writer.close()


@pytest.mark.parametrize("operation", ("ftruncate", "fsync", "read"))
def test_jsonl_rollback_operation_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    path = tmp_path / f"rollback-{operation}.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    committed = writer.append(_fixture_record(0))
    calls = [0]

    def fail(*args):
        calls[0] += 1
        raise OSError(f"injected rollback {operation} failure")

    monkeypatch.setattr(runner.os, operation, fail)
    with pytest.raises(runner.UnrecoverableOrphan):
        writer.append(_fixture_record(1), fault="short_write")
    assert writer.last_committed_offset == committed.current_offset
    assert calls[0] >= 1
    runner.os.close(writer._descriptor)
    writer._descriptor = None
    writer._closed = True


def test_rollback_fault_gives_pending_signal_priority(tmp_path: Path) -> None:
    path = tmp_path / "signal-priority.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    polls = []

    def pending_signal():
        polls.append(len(polls))
        return len(polls) > 1

    with pytest.raises(runner.LedgerAppendError) as caught:
        writer.append(_fixture_record(0), fault="fsync_failure", pending_signal=pending_signal)
    assert caught.value.result.reason_code == "signal_or_interruption"
    assert caught.value.result.acknowledged is False
    assert caught.value.result.committed is False
    assert writer.last_committed_offset == 0
    assert path.read_bytes() == b""
    writer.close()


def test_signal_after_commit_retains_and_charges_line_without_ack(tmp_path: Path) -> None:
    path = tmp_path / "committed-signal.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    result = writer.append(_fixture_record(0), fault="handled_signal_after_commit_before_ack")
    expected = runner.canonical_json_bytes(_fixture_record(0)) + b"\n"
    assert result.prior_offset == 0
    assert result.current_offset == len(expected)
    assert result.acknowledged is False
    assert result.committed is True
    assert result.reason_code == "signal_or_interruption"
    assert path.read_bytes() == expected
    assert writer.validate_committed_prefix() == (_fixture_record(0),)
    writer.close()


def test_pending_signal_poll_observes_durable_commit_before_ack(tmp_path: Path) -> None:
    path = tmp_path / "commit-race.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    expected = runner.canonical_json_bytes(_fixture_record(0)) + b"\n"
    polls = []

    def pending_signal():
        polls.append((writer.last_committed_offset, path.read_bytes()))
        return len(polls) == 2

    result = writer.append(_fixture_record(0), pending_signal=pending_signal)
    assert polls == [(0, b""), (len(expected), expected)]
    assert result == runner.AppendResult(0, len(expected), False, True, "signal_or_interruption", hashlib.sha256(expected).hexdigest())
    assert writer.validate_committed_prefix() == (_fixture_record(0),)
    writer.close()


def test_crash_writer_rejects_schema_and_unknown_fault_before_mutation(tmp_path: Path) -> None:
    path = tmp_path / "reject.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    with pytest.raises(runner.ContractError):
        writer.append({"sequence": 0})
    with pytest.raises(runner.ContractError):
        writer.append(_fixture_record(0), fault="alias")
    assert writer.last_committed_offset == 0
    assert path.read_bytes() == b""
    writer.close()


def test_crash_writer_truncates_only_uncommitted_suffix_on_close(tmp_path: Path) -> None:
    path = tmp_path / "suffix.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    writer.append(_fixture_record(0))
    prefix = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b"partial")
    writer.close()
    assert path.read_bytes() == prefix


def test_every_live_writer_recovery_call_resolves_and_restores_committed_prefix(tmp_path: Path) -> None:
    tree = _source_tree()
    writer_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CrashAtomicJsonlWriter"
    )
    writer_methods = {
        node.name for node in writer_class.body if isinstance(node, ast.FunctionDef)
    }
    recovery_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("recover_")
    ]
    assert recovery_calls == ["recover_uncommitted_suffix"] * 4
    assert set(recovery_calls) <= writer_methods

    path = tmp_path / "recovery-surface.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, _validate_fixture_record)
    writer.precreate()
    result = writer.append(_fixture_record(0))
    committed = path.read_bytes()
    assert result.committed is True
    assert result.acknowledged is True
    assert result.current_offset == len(committed)
    with path.open("ab") as handle:
        handle.write(b'{"uncommitted":true}')
        handle.flush()
        runner.os.fsync(handle.fileno())
    assert path.stat().st_size > result.current_offset
    writer.recover_uncommitted_suffix()
    assert writer.last_committed_offset == result.current_offset
    assert path.read_bytes() == committed
    assert writer.validate_committed_prefix() == (_fixture_record(0),)
    writer.close()


def test_attempt_id_has_exact_canonical_identity() -> None:
    assert runner.attempt_id("test-run", 1, 11, "selected", "joint", 1) == "002957a1e9df7a0ebe7ef8d55f920852aebff8818ce9ee9cce26c416d2d2f66d"


def test_attempt_sequence_and_accounting_charge_started_and_pair_completed() -> None:
    rows = [
        _attempt_event(0, "started", logical_update=1),
        _attempt_event(1, "completed", logical_update=1),
        _attempt_event(2, "started", logical_update=2),
    ]
    runner.validate_attempt_sequence(rows, require_complete=False)
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_sequence(rows, require_complete=True)
    accounting = runner.derive_attempt_accounting(rows)
    assert accounting.attempted_updates == 2
    assert accounting.completed_updates == 1
    assert accounting.attempted_token_positions == 4096
    assert accounting.completed_token_positions == 2048
    assert accounting.attempted_seeds == (11,)
    assert accounting.completed_seeds == (11,)
    assert accounting.last_event_sequence_by_seed == {11: 2}
    assert accounting.unpaired_attempts == (rows[-1]["attempt_id"],)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[0].__setitem__("event_sequence", 1),
        lambda rows: rows[1].__setitem__("event", "started"),
        lambda rows: rows[1].__setitem__("batch_sha256", "b" * 64),
        lambda rows: rows[1].__setitem__("attempt_id", "b" * 64),
        lambda rows: rows.reverse(),
    ],
)
def test_attempt_sequence_rejects_order_identity_and_pair_drift(mutator) -> None:
    rows = [_attempt_event(0, "started"), _attempt_event(1, "completed")]
    mutator(rows)
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_sequence(rows, require_complete=True)


def test_attempt_sequence_rejects_reused_logical_update_and_attempt_id() -> None:
    rows = [
        _attempt_event(0, "started", logical_update=1),
        _attempt_event(1, "completed", logical_update=1),
        _attempt_event(2, "started", logical_update=1),
        _attempt_event(3, "completed", logical_update=1),
    ]
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_sequence(rows, require_complete=True)


def test_attempt_row_rejects_nonfinite_completed_metric_and_boolean_identity() -> None:
    nonfinite = _attempt_event(1, "completed")
    nonfinite["metrics"]["total_loss"] = float("inf")
    boolean = _attempt_event(0, "started")
    boolean["claim_seed"] = True
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_row(nonfinite)
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_row(boolean)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("metrics", "total_loss"), float("nan")),
        (("metrics", "gradient_norm"), float("inf")),
        (("metrics", "elapsed_seconds"), float("-inf")),
        (("metrics", "component_losses", "task_loss"), float("nan")),
        (("metrics", "component_losses", "internal_router_loss"), float("inf")),
        (("metrics", "component_losses", "supervised_route_loss"), float("-inf")),
        (("metrics", "finite"), False),
    ],
)
def test_attempt_row_rejects_every_nonfinite_metric_surface(path, replacement) -> None:
    row = _attempt_event(1, "completed")
    _set_path(row, path, replacement)
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_row(row)


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), float("-inf")])
def test_attempt_row_rejects_nonfinite_learning_rate(replacement) -> None:
    row = _attempt_event(1, "completed")
    row["metrics"]["learning_rates"] = [{"parameter_group": "matrix", "learning_rate": replacement}]
    with pytest.raises(runner.ContractError):
        runner.validate_attempt_row(row)


def test_tracked_payload_has_reviewed_identity_and_exact_registries() -> None:
    payload = runner.load_prereg_payload(TRACKED_PAYLOAD_PATH)
    assert tuple(payload) == TOP_LEVEL_KEYS
    assert runner.canonical_json_sha256(payload) == TRACKED_PAYLOAD_CANONICAL_SHA256
    assert tuple(runner.FAULT_IDS) == FAULT_IDS
    assert tuple(runner.CLAIM_LEDGER_PATHS) == CLAIM_LEDGER_PATHS
    assert tuple(runner.HARD_ABORT_REASON_CODES) == HARD_ABORT_REASON_CODES
    assert tuple(runner.PARITY_SCOPES) == tuple(payload["artifacts"]["schemas"]["parity"]["ordered_scope"])
    assert tuple(runner.PRETRAINING_ASSERTION_IDS) == tuple(record["id"] for record in payload["gates"]["pretraining_assertions"])
    assert tuple(record["ordinal"] for record in payload["gates"]["pretraining_assertions"]) == tuple(range(1, 16))
    adapter = payload["architecture"]["common_host"]["routed_adapter"]
    source = next(reference for reference in payload["sources"]["references"] if reference["path"] == adapter["source_module"])
    assert source["authorized_symbols"] == ["MonodraticPHIMixer", *adapter["authorized_source_symbols"]]
    assert adapter["evidence_only_probe_audit"]["probe_helper_ordered_arguments"] == ["detached_current_query_route_features", "detached_current_codebooks_from_the_same_search", 4]
    assert adapter["evidence_only_probe_audit"]["post_probe_inputs"] == ["returned_probe_addresses", "existing_packed_postings", "remote_limit"]
    assert payload["artifacts"]["serialization"]["crash_atomic_canonical_jsonl"]["exact_paths"] == list(
        TRANSACTIONAL_JSONL_PATHS
    )
    assert payload["artifacts"]["artifact_path_pattern_count"] == 58
    assert len(payload["artifacts"]["artifact_paths"]) == 58
    assert len(set(payload["artifacts"]["artifact_paths"])) == 58
    assert len(payload["gates"]["rung_one_registry"]) == 24
    assert len(payload["gates"]["rung_two_registry"]) == 4
    assert payload["gates"]["registry_cardinalities"]["complete_package"] == 124
    runner.validate_prereg_payload(payload)


@pytest.mark.parametrize("mutation", ("missing", "additional", "reordered", "duplicated", "agreement", "probe_count", "schema_probe_count"))
def test_prereg_rejects_routed_source_registry_and_probe_abi_drift(mutation: str) -> None:
    payload = copy.deepcopy(_tracked_payload())
    adapter = payload["architecture"]["common_host"]["routed_adapter"]
    source = next(reference for reference in payload["sources"]["references"] if reference["path"] == adapter["source_module"])
    if mutation == "missing":
        source["authorized_symbols"].remove("probe_addresses")
    elif mutation == "additional":
        source["authorized_symbols"].append("invented_symbol")
    elif mutation == "reordered":
        source["authorized_symbols"][1:3] = reversed(source["authorized_symbols"][1:3])
    elif mutation == "duplicated":
        adapter["authorized_source_symbols"].append("probe_addresses")
        source["authorized_symbols"].append("probe_addresses")
    elif mutation == "agreement":
        payload["sources"]["monodratic_authorized_symbol_registry_agreement"]["deliberate_source_reference_only_class_symbol"] = "invented_symbol"
    elif mutation == "probe_count":
        adapter["evidence_only_probe_audit"]["probe_helper_ordered_arguments"][-1] = 3
    else:
        payload["artifacts"]["schemas"]["routing_row"]["valid_posting_histogram_probe_helper_arguments"][-1] = 3
    with pytest.raises(runner.ContractError):
        runner.validate_prereg_payload(payload)


def test_tracked_payload_closes_lifecycle_and_process_contracts() -> None:
    payload = _tracked_payload()
    time_statistics = payload["pilot"]["time_statistics"]
    assert time_statistics["Tprojected"].endswith("+packaging_seconds+resource_finalization_seconds+lifecycle_close_join_seconds")
    assert time_statistics["measured_component_order"][-2:] == ["resource_finalization_seconds", "lifecycle_close_join_seconds"]
    assert time_statistics["measured_component_order"][-1] == "lifecycle_close_join_seconds"
    assert time_statistics["required_tail_benchmark_assertions"][-2:] == ["pilot_tail_resource_finalization_projection", "pilot_tail_lifecycle_close_join_projection"]
    assert time_statistics["packaging_benchmark"]["future_pilot_tail_detail_count"] == 6
    pilot_schema = payload["artifacts"]["schemas"]["pilot"]
    assert pilot_schema["required_tail_benchmark_assertion_names"] == time_statistics["required_tail_benchmark_assertions"]
    assert pilot_schema["lifecycle_close_join_assertion_actual_exact_keys"] == ["component_seconds", "close_join_seconds", "return_code", "scratch_cleanup", "scratch_cleanup_seconds", "stderr_bytes"]
    assert pilot_schema["resource_finalization_assertion_actual_exact_keys"] == ["component_seconds", "actual_stop_seconds", "final_active_jobs", "final_attempted_updates", "final_expected_pids", "final_sample_id", "final_token_positions", "interval_seconds", "max_observed_sample_duration_seconds", "sample_transaction_count"]
    sampling_schema = payload["artifacts"]["schemas"]["resource_row"]
    assert sampling_schema["training_stage_sample_wait_allowed"] is False
    assert sampling_schema["clean_claim_sampler_stop_rule"].endswith("a_late_return_hard_aborts_and_no_blocked_syscall_is_claimed_preemptible")
    cutoff = payload["pilot"]["resource_sampling"]["clean_claim_cutoff_sequence"]
    assert "parent_crash_atomically_appends_one_final_valid_no_active_job_resource_row" not in cutoff
    assert cutoff[2:4] == ["quiesce_resource_sampler_then_reuse_and_validate_an_exact_already_durable_clean_row_or_admit_only_a_remaining_interval_whose_earliest_sample_time_precedes_the_claim_acceptance_deadline_then_sample_append_fsync_and_recheck_after_return", "parent_fsyncs_and_readback_validates_the_complete_timeline_and_updates_last_committed_offset_with_fail_closed_post_return_deadline_checks"]
    assert cutoff[6:] == [
        "write_validate_and_fsync_canonical_completion_and_summary_as_nonterminal_artifacts",
        "send_close_committed_ack",
        "cap_child_wait_to_minimum_of_thirty_seconds_and_remaining_acceptance_seconds",
        "recheck_acceptance_deadline_immediately_after_child_wait_return",
        "flush_fsync_and_read_stderr_with_post_operation_acceptance_checks",
        "remove_owned_scratch_and_recheck_acceptance_deadline_after_successful_cleanup",
        "run_final_guard_then_validate_clean_artifact_closure",
        "write_validate_and_fsync_SHA256SUMS_as_the_only_terminal_commit",
    ]
    claim_stream = payload["processes"]["multiprocessing"]["ipc_protocols"]["claim_stream"]
    assert claim_stream.endswith("then_clean_close_caps_the_child_wait_to_the_remaining_acceptance_time_rechecks_after_join_stderr_fsync_stderr_read_and_owned_scratch_cleanup_and_only_then_validates_artifact_closure_and_commits_the_terminal_checksum")
    assert "child_close_ack_exit_join_stderr_fsync_read_and_owned_scratch_cleanup" in payload["pilot"]["claim_timer"]["charged"]
    finalization = time_statistics["resource_finalization_benchmark"]
    assert finalization["acceptance_deadline"].endswith("it_does_not_claim_to_preempt_a_blocked_OS_or_storage_call")
    assert finalization["failure_action"].endswith("so_no_late_pilot_can_be_accepted")
    hard_abort_registry = payload["abort_rules"]["hard_abort_registry"]
    assert [record["priority"] for record in hard_abort_registry] == list(range(1, 14))
    assert [record["reason_code"] for record in hard_abort_registry] == list(HARD_ABORT_REASON_CODES)
    assert all(record["condition"] and record["condition"] != record["reason_code"] for record in hard_abort_registry)
    preflight = payload["artifacts"]["schemas"]["preflight"]
    assert preflight["required_lifecycle_assertions"] == [
        "staged_publish_rehearsal",
        "actual_staging_readiness",
        "pilot_timeline_transition",
        *FAULT_IDS,
        "terminal_checksum_deactivation",
    ]
    assert preflight["crash_fault_name_subsequence"] == list(FAULT_IDS)
    assert payload["processes"]["launcher"]["resume_supported"] is False
    assert payload["processes"]["multiprocessing"]["start_method"] == "parent_subprocess_with_start_new_session"
    assert payload["processes"]["multiprocessing"]["worker_count"] == 1
    assert payload["processes"]["multiprocessing"]["worker_assignments"] == [
        {
            "worker": "MLX",
            "ordinal": 0,
            "jobs": [
                {"rung": 1, "construction_seed": 11},
                {"rung": 1, "construction_seed": 23},
                {"rung": 1, "construction_seed": 37},
                {"rung": 1, "construction_seed": 53},
                {"rung": 1, "construction_seed": 71},
                {"rung": 2, "construction_seed": 83},
            ],
        },
    ]
    closures = payload["artifacts"]["artifact_closures"]
    assert closures["pilot_abort"]["run_pilot_json_retained"] is False
    assert closures["claim_abort"]["all_six_attempt_ledgers_present_even_when_empty"] is True
    assert closures["claim_abort"]["run_resources_jsonl_present_even_when_empty"] is True
    assert closures["terminal_checksum_immutability"]["terminal_SHA256SUMS_or_covered_file_rewrite_allowed"] is False
    assert payload["artifacts"]["schemas"]["aborted"]["path"] == "ABORTED.json"
    assert payload["artifacts"]["schemas"]["aborted"]["write_count"] == 1
    assert payload["artifacts"]["schemas"]["SHA256SUMS"]["generated_last"] is True


def test_primary_failure_latch_uses_registry_priority_and_is_immutable() -> None:
    registry = _tracked_payload()["abort_rules"]["hard_abort_registry"]
    clock_calls = []
    latch = runner.PrimaryFailureLatch(registry)
    selected = latch.select_poll(
        [
            {"reason_code": "nonfinite", "context": {"worker": "A", "event_sequence": 2}},
            {"reason_code": "signal_or_interruption", "context": {"worker": "B", "event_sequence": 8}},
        ],
        monotonic_ns=lambda: clock_calls.append(700) or 700,
    )
    assert selected.reason_code == "signal_or_interruption"
    assert selected.condition == registry[0]["condition"]
    assert selected.worker == "B"
    assert selected.event_sequence == 8
    assert selected.monotonic_ns == 700
    assert latch.select_poll(
        [{"reason_code": "frozen_hash_change", "context": {"worker": "A", "event_sequence": 0}}],
        monotonic_ns=lambda: clock_calls.append(800) or 800,
    ) is selected
    assert clock_calls == [700]


@pytest.mark.parametrize(
    ("observations", "expected_worker", "expected_sequence"),
    [
        (
            [
                {"reason_code": "artifact_inconsistency", "context": {"worker": "B", "event_sequence": 1}},
                {"reason_code": "artifact_inconsistency", "context": {"worker": "A", "event_sequence": 9}},
                {"reason_code": "artifact_inconsistency", "context": {"worker": "A", "event_sequence": 3}},
            ],
            "A",
            3,
        ),
        (
            [
                {"reason_code": "artifact_inconsistency", "context": {"worker": "A", "event_sequence": 0}},
                {"reason_code": "artifact_inconsistency", "context": {}},
            ],
            None,
            None,
        ),
    ],
)
def test_primary_failure_latch_applies_deterministic_same_poll_ties(observations, expected_worker, expected_sequence) -> None:
    latch = runner.PrimaryFailureLatch(_tracked_payload()["abort_rules"]["hard_abort_registry"])
    selected = latch.select_poll(observations, monotonic_ns=lambda: 1)
    assert selected.worker == expected_worker
    assert selected.event_sequence == expected_sequence


def test_two_phase_pilot_and_claim_handlers_never_ack_a_staged_same_poll_success_after_peer_failure(tmp_path: Path) -> None:
    registry = _tracked_payload()["abort_rules"]["hard_abort_registry"]
    pilot_a = _FakeConnection()
    pilot_b = _FakeConnection()
    pilot_jobs = {}
    pilot_message = {"kind": "pilot_update_start", "worker": "A", "seed": 11, "stage": "A", "logical_update": 1, "token_positions": 2048}
    update_delta, token_delta, pilot_ack = runner._handle_pilot_worker_message(
        pilot_message,
        "A",
        pilot_a,
        set(),
        pilot_jobs,
        {name: {"entry": [], "exit": [], "arrived": set()} for name in ("A", "S", "D", "H")},
        [],
        SimpleNamespace(pending_signal=None),
    )
    assert (update_delta, token_delta, pilot_ack) == (1, 2048, {"ack": True})
    with pytest.raises(runner.HardAbort) as caught:
        runner._handle_pilot_worker_message(
            {"kind": "hard_abort", "worker": "B", "reason_code": "nonfinite", "context": {"worker": "B"}},
            "B",
            pilot_b,
            set(),
            pilot_jobs,
            {name: {"entry": [], "exit": [], "arrived": set()} for name in ("A", "S", "D", "H")},
            [],
            SimpleNamespace(pending_signal=None),
        )
    pilot_latch = runner.PrimaryFailureLatch(registry)
    pilot_error = runner.hard_abort_from_same_poll(pilot_latch, [runner.failure_observation_from_exception(caught.value, "artifact_inconsistency")])
    assert pilot_error is not None and pilot_error.reason_code == "nonfinite"
    assert pilot_a.sent == [] and pilot_b.sent == []

    path = tmp_path / "attempts.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, runner.validate_attempt_row)
    writer.precreate()
    claim_a = _FakeConnection()
    claim_b = _FakeConnection()
    row = _attempt_event(0, "started")
    claim_ack = runner._handle_claim_worker_message(
        {"kind": "attempt", "worker": "A", "seed": 11, "stage": "joint", "logical_update": 1, "row": row},
        "A",
        claim_a,
        set(),
        {"rung1/11/attempts.jsonl": writer},
        {},
        {11: []},
        SimpleNamespace(pending_signal=None),
    )
    assert claim_ack == {"ack": True}
    with pytest.raises(runner.HardAbort) as caught:
        runner._handle_claim_worker_message(
            {"kind": "hard_abort", "worker": "B", "reason_code": "nonfinite", "context": {"worker": "B"}},
            "B",
            claim_b,
            set(),
            {"rung1/11/attempts.jsonl": writer},
            {},
            {23: []},
            SimpleNamespace(pending_signal=None),
        )
    claim_latch = runner.PrimaryFailureLatch(registry)
    claim_error = runner.hard_abort_from_same_poll(claim_latch, [runner.failure_observation_from_exception(caught.value, "artifact_inconsistency")])
    assert claim_error is not None and claim_error.reason_code == "nonfinite"
    assert claim_a.sent == [] and claim_b.sent == []
    assert writer.validate_committed_prefix() == (row,)
    writer.close()


def test_primary_failure_latch_rejects_registry_observation_and_clock_drift() -> None:
    registry = _tracked_payload()["abort_rules"]["hard_abort_registry"]
    with pytest.raises(runner.ContractError):
        runner.PrimaryFailureLatch(tuple(reversed(registry)))
    latch = runner.PrimaryFailureLatch(registry)
    with pytest.raises(runner.ContractError):
        latch.select_poll([{"reason_code": "alias", "context": {}}], monotonic_ns=lambda: 1)
    with pytest.raises(runner.ContractError):
        latch.select_poll([{"reason_code": "nonfinite", "context": {}}], monotonic_ns=lambda: True)


@pytest.mark.parametrize("parent_function", ("run_resource_pilot", "run_claim_workers"))
def test_parent_worker_loops_use_same_poll_latch_before_immutable_quiesce(parent_function: str) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    node = next(item for item in _source_tree().body if isinstance(item, ast.FunctionDef) and item.name == parent_function)
    segment = ast.get_source_segment(source, node)
    assert segment.count("hard_abort_from_same_poll(failure_latch, observations)") == 1
    assert "raise quiesce_after_primary_latch(latched_error, processes)" in segment
    assert segment.index("hard_abort_from_same_poll(failure_latch, observations)") < segment.index("raise quiesce_after_primary_latch")
    registry = _tracked_payload()["abort_rules"]["hard_abort_registry"]
    priority_latch = runner.PrimaryFailureLatch(registry)
    priority_error = runner.hard_abort_from_same_poll(
        priority_latch,
        [
            {"reason_code": "worker_exit", "context": {"worker": "A", "event_sequence": 1}},
            {"reason_code": "nonfinite", "context": {"worker": "B", "event_sequence": 9}},
        ],
    )
    assert priority_error.reason_code == "nonfinite"
    assert priority_error.context == {"worker": "B", "event_sequence": 9}
    tie_latch = runner.PrimaryFailureLatch(registry)
    tie_error = runner.hard_abort_from_same_poll(
        tie_latch,
        [
            {"reason_code": "worker_exit", "context": {"worker": "B", "event_sequence": 0}},
            {"reason_code": "worker_exit", "context": {"worker": "A", "event_sequence": 8}},
            {"reason_code": "worker_exit", "context": {"worker": "A", "event_sequence": 2}},
        ],
    )
    assert tie_error.reason_code == "worker_exit"
    assert tie_error.context == {"worker": "A", "event_sequence": 2}
    cutoff = tie_error.primary_latch_monotonic_ns
    process = _EscalatingProcess(ignore_terminate=True)
    assert runner.quiesce_after_primary_latch(tie_error, (process,)) is tie_error
    assert process.join_timeouts == [1.0, 1.0, 1.0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert tie_error.primary_latch_monotonic_ns == cutoff


def test_final_claim_guard_checks_signal_deadline_and_frozen_anchors_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    anchors = runner.FrozenManifestAnchors((("run/config_manifest.json", "a" * 64),))
    signals = SimpleNamespace(pending_signal=None)
    calls = []
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 2_000_000_000)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda observed_root, observed_anchors: calls.append((observed_root, observed_anchors)))
    runner.final_claim_guard(root, anchors, signals, 1_000_000_000, "before_completion")
    assert calls == [(root, anchors)]
    signals.pending_signal = 15
    with pytest.raises(runner.HardAbort) as caught:
        runner.final_claim_guard(root, anchors, signals, 1_000_000_000, "after_completion")
    assert caught.value.reason_code == "signal_or_interruption"
    assert caught.value.context == {"stage": "after_completion"}
    assert calls == [(root, anchors)]
    signals.pending_signal = None
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 1_200_000_000_001)
    with pytest.raises(runner.HardAbort) as caught:
        runner.final_claim_guard(root, anchors, signals, 0, "after_summary")
    assert caught.value.reason_code == "claim_elapsed_time"
    assert caught.value.context == {"stage": "after_summary"}
    assert calls == [(root, anchors)]
    with pytest.raises(runner.ContractError):
        runner.final_claim_guard(root, anchors, signals, 0, "")


@pytest.mark.parametrize("mutation_index", range(11))
def test_manifest_anchor_capture_rejects_real_byte_mutation_at_every_frozen_path(tmp_path: Path, mutation_index: int) -> None:
    root = tmp_path / "run-id"
    paths = _materialize_manifest_anchor_surface(root)
    anchors = runner.capture_frozen_manifest_anchors(root)
    assert tuple(relative for relative, _ in anchors.records) == paths
    runner.verify_manifest_anchors(root, anchors)
    mutated = root / paths[mutation_index]
    mutated.write_bytes(mutated.read_bytes() + b"x")
    with pytest.raises(runner.HardAbort) as caught:
        runner.verify_manifest_anchors(root, anchors)
    assert caught.value.reason_code == "frozen_hash_change"
    assert caught.value.context == {"surface": paths[mutation_index]}


def test_launch_plan_snapshot_copies_exact_prelaunch_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.parent.mkdir(parents=True)
    plan.write_bytes(b"launch-plan\n")
    root = tmp_path / "run-id"
    (root / "run").mkdir(parents=True)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    digest = runner.copy_launch_project_plan_snapshot(root)
    snapshot = root / "run" / "project_plan_launch.md"
    assert snapshot.read_bytes() == b"launch-plan\n"
    assert digest == hashlib.sha256(b"launch-plan\n").hexdigest()
    with pytest.raises(FileExistsError):
        runner.copy_launch_project_plan_snapshot(root)


def test_training_start_plan_barrier_waits_for_changed_exact_zero_finding_attestation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    evidence.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    launch_bytes = b"launch-plan\n"
    plan.write_bytes(launch_bytes)
    root = tmp_path / "run-id"
    paths = _materialize_manifest_anchor_surface(root, launch_bytes)
    config_path = root / "run" / "config_manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for index, record in enumerate(config["review_records"]):
        old_path = root / "run" / "reviews" / f"{record['artifact_sha256']}.json"
        old_path.unlink()
        raw = f"base-review-{index}\n".encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        (root / "run" / "reviews" / f"{digest}.json").write_bytes(raw)
        record["artifact_sha256"] = digest
    config_path.write_bytes(runner.canonical_json_bytes(config))
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors))
    anchors = runner.capture_frozen_manifest_anchors(root)
    signals = _BarrierSignals()
    sleeps = []
    tick = [100_000_000_000]

    def monotonic_ns():
        value = tick[0]
        tick[0] += 1_000_000_000
        return value

    def sleeper(seconds):
        sleeps.append(seconds)
        assert plan.read_bytes() == launch_bytes
        request_path = root / "run" / "training_start_request.json"
        request_sha256 = runner.sha256_file(request_path)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    reviewed_anchors, claim_start = runner.establish_training_start_plan_barrier(
        root,
        anchors,
        signals,
        source_directory=evidence,
        monotonic_ns=monotonic_ns,
        sleeper=sleeper,
        utc_reader=lambda: "2026-07-21T00:00:00Z",
    )
    assert claim_start > 100_000_000_000
    assert sleeps == [runner.TRAINING_START_REVIEW_POLL_SECONDS]
    assert len(paths) == 11
    assert len(reviewed_anchors.records) == 15
    request = json.loads((root / "run" / "training_start_request.json").read_text(encoding="utf-8"))
    assert request == {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": root.name,
        "boundary": "training_start_review_request",
        "review_request_monotonic_ns": 100_000_000_000,
        "review_request_wall_utc": "2026-07-21T00:00:00Z",
        "launch_project_plan_sha256": hashlib.sha256(launch_bytes).hexdigest(),
        "required_review_scope": f"training_start_project_plan:{root.name}",
        "review_wait_timeout_seconds": runner.TRAINING_START_REVIEW_WAIT_SECONDS,
    }
    request_sha256 = runner.sha256_file(root / "run" / "training_start_request.json")
    candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
    assert plan.read_bytes() == candidate_bytes
    assert (root / "run" / "project_plan_training_start.md").read_bytes() == candidate_bytes
    assert not (root / "run" / "training_start.json").exists()
    linkage = json.loads((root / "run" / "training_start_plan.json").read_text(encoding="utf-8"))
    assert linkage["request_path"] == "run/training_start_request.json"
    assert linkage["request_artifact_sha256"] == request_sha256
    assert linkage["commit_admission_monotonic_ns"] < claim_start
    assert linkage["commit_admission_wall_utc"] == "2026-07-21T00:00:00Z"
    assert linkage["launch_project_plan_sha256"] == hashlib.sha256(launch_bytes).hexdigest()
    assert linkage["training_start_project_plan_sha256"] == hashlib.sha256(candidate_bytes).hexdigest()
    assert linkage["training_start_project_plan_sha256"] != linkage["launch_project_plan_sha256"]
    assert linkage["review_scope"] == f"training_start_project_plan:{root.name}"
    assert linkage["review_target_sha256"] == runner.canonical_json_sha256(
        [
            {"path": "neuroloc/wiki/PROJECT_PLAN.md", "sha256": linkage["training_start_project_plan_sha256"]},
            {"path": "run/training_start_request.json", "sha256": request_sha256},
        ]
    )
    assert linkage["review_path"] == f"run/reviews/{linkage['review_artifact_sha256']}.json"
    runner.verify_manifest_anchors(root, reviewed_anchors)
    state, observed_linkage = runner._training_start_state(root)
    assert state == "started"
    assert observed_linkage == linkage
    plan.write_bytes(candidate_bytes + b"drift\n")
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._training_start_state(root)


def test_training_start_plan_barrier_never_accepts_unchanged_launch_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    evidence.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    launch_bytes = b"launch-plan\n"
    plan.write_bytes(launch_bytes)
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, launch_bytes)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors))
    anchors = runner.capture_frozen_manifest_anchors(root)
    clock = [100]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / "run" / "training_start_request.json")
        _write_training_start_candidate(evidence, launch_bytes, request_sha256, root.name)
        clock[0] = 100 + runner.TRAINING_START_REVIEW_WAIT_NS

    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: clock[0],
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.reason_code == "artifact_inconsistency"
    assert caught.value.context == {"surface": "training_start_plan_review_timeout", "training_start_state": "awaiting_review"}
    assert (root / "run" / "training_start_request.json").is_file()
    assert not (root / "run" / "training_start.json").exists()
    assert not (root / "run" / "training_start_plan.json").exists()
    assert not (root / "run" / "project_plan_training_start.md").exists()
    assert plan.read_bytes() == launch_bytes


def test_training_start_plan_barrier_signal_and_malformed_review_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for mode in ("signal", "malformed", "started_signal"):
        repository = tmp_path / mode / "repository"
        evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
        plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
        evidence.mkdir(parents=True)
        plan.parent.mkdir(parents=True)
        launch_bytes = b"launch-plan\n"
        plan.write_bytes(launch_bytes)
        root = tmp_path / mode / "run-id"
        _materialize_manifest_anchor_surface(root, launch_bytes)
        signals = _BarrierSignals(inject_after_commit=mode == "started_signal")
        monkeypatch.setattr(runner, "PROJECT_ROOT", repository)

        def verify(observed_root, observed_anchors):
            runner.verify_manifest_anchors(observed_root, observed_anchors)
        monkeypatch.setattr(runner, "_verify_active_frozen_hashes", verify)
        anchors = runner.capture_frozen_manifest_anchors(root)
        tick = [100]

        def monotonic_ns():
            tick[0] += 1
            return tick[0]

        def sleeper(seconds):
            if mode == "signal":
                signals.pending_signal = 15
                return
            request_sha256 = runner.sha256_file(root / "run" / "training_start_request.json")
            candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
            findings = [] if mode == "started_signal" else [{"id": "F1"}]
            _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name, findings=findings)

        with pytest.raises(runner.HardAbort) as caught:
            runner.establish_training_start_plan_barrier(
                root,
                anchors,
                signals,
                source_directory=evidence,
                monotonic_ns=monotonic_ns,
                sleeper=sleeper,
                utc_reader=lambda: "2026-07-21T00:00:00Z",
            )
        expected = "signal_or_interruption" if mode in {"signal", "started_signal"} else "artifact_inconsistency"
        assert caught.value.reason_code == expected
        expected_state = "started" if mode == "started_signal" else "awaiting_review"
        assert caught.value.context["training_start_state"] == expected_state
        assert (root / "run" / "training_start_plan.json").exists() is (mode == "started_signal")
        assert plan.read_bytes() != launch_bytes if mode == "started_signal" else plan.read_bytes() == launch_bytes


@pytest.mark.parametrize(
    ("state", "anchor_count"),
    (("not_started", 11), ("awaiting_review", 12), ("reviewed_ready", 15), ("started", 15)),
)
def test_training_start_persisted_state_classification_is_unambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, anchor_count: int) -> None:
    repository = tmp_path / state / "repository"
    root = tmp_path / state / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    _, expected_linkage = _materialize_training_start_lifecycle(root, repository, state)
    observed_state, observed_linkage = runner._training_start_state(root)
    assert observed_state == state
    assert observed_linkage == expected_linkage
    paths = runner._manifest_anchor_paths_for_state(root, anchor_count)
    anchors = runner.FrozenManifestAnchors(tuple(sorted((relative, runner.sha256_file(root / relative)) for relative in paths)))
    runner.verify_manifest_anchors(root, anchors)


def test_training_start_state_rejects_every_ambiguous_partial_and_live_plan_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "repository"
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    candidate_bytes, _ = _materialize_training_start_lifecycle(root, repository, "reviewed_ready")
    link_path = root / "run" / "training_start_plan.json"
    link_raw = link_path.read_bytes()
    link_path.unlink()
    with pytest.raises(runner.ContractError):
        runner._training_start_state(root)
    link_path.write_bytes(link_raw)
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.write_bytes(candidate_bytes + b"drift\n")
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._training_start_state(root)


@pytest.mark.parametrize("mode", ("admission_margin", "replace_return", "directory_fsync"))
def test_training_start_candidate_cannot_publish_at_or_after_strict_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    repository = tmp_path / mode / "repository"
    evidence = repository / "neuroloc" / "results" / "modular_sequence_role_cpu_reviews"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    evidence.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    launch_bytes = b"launch-plan\n"
    plan.write_bytes(launch_bytes)
    root = tmp_path / mode / "run-id"
    _materialize_manifest_anchor_surface(root, launch_bytes)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors))
    clock = [100_000_000_000]
    deadline = clock[0] + runner.TRAINING_START_REVIEW_WAIT_NS
    candidate_holder = {}
    published = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / "run" / "training_start_request.json")
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        candidate_holder["bytes"] = candidate_bytes
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)
        clock[0] = deadline - (1_000_000 if mode == "admission_margin" else 10_000_000_000)

    real_replace = runner.os.replace

    def replace(source, destination):
        real_replace(source, destination)
        if Path(destination) == plan:
            published[0] = True
            if mode == "replace_return":
                clock[0] = deadline

    real_fsync = runner.os.fsync

    def fsync(descriptor):
        real_fsync(descriptor)
        if published[0] and mode == "directory_fsync":
            clock[0] = deadline

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.os, "fsync", fsync)
    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            runner.capture_frozen_manifest_anchors(root),
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: clock[0],
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.reason_code == "artifact_inconsistency"
    if mode == "admission_margin":
        assert caught.value.context == {"surface": "training_start_plan_review_timeout", "training_start_state": "awaiting_review"}
        assert plan.read_bytes() == launch_bytes
        assert not (root / "run" / "training_start_plan.json").exists()
    else:
        assert caught.value.context == {"surface": "training_start_plan_publication_late", "training_start_state": "started"}
        assert plan.read_bytes() == candidate_holder["bytes"]
        assert (root / "run" / "training_start_plan.json").is_file()
        assert runner._training_start_state(root)[0] == "started"


@pytest.mark.parametrize("state", ("not_started", "awaiting_review", "reviewed_ready", "started"))
def test_training_start_persisted_state_and_closure_reject_live_plan_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    repository = tmp_path / state / "repository"
    root = tmp_path / state / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    _materialize_training_start_lifecycle(root, repository, state)
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.write_bytes(b"concurrent-plan\n")
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._training_start_state(root)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._training_start_closure_paths(root, state)


@pytest.mark.parametrize("failure", ("post_link_clock", "anchor_extension"))
def test_training_start_durable_link_retains_reconstructible_fifteen_path_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    tick = [100]

    def monotonic_ns():
        tick[0] += 1
        return tick[0]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_clock = runner._training_start_review_clock
    real_extend = runner.extend_frozen_manifest_anchors
    reviewed_clock_calls = [0]
    extension_calls = [0]

    def review_clock(reader, request_start, require_margin, state):
        value = real_clock(reader, request_start, require_margin, state)
        if state == "reviewed_ready":
            reviewed_clock_calls[0] += 1
            if failure == "post_link_clock" and reviewed_clock_calls[0] == 1:
                raise runner.HardAbort("artifact_inconsistency", {"surface": "injected_post_link"})
        return value

    def extend(observed_root, observed_anchors, paths):
        extension_calls[0] += 1
        if failure == "anchor_extension" and extension_calls[0] == 2:
            raise runner.HardAbort("frozen_hash_change", {"surface": "injected_anchor_extension"})
        return real_extend(observed_root, observed_anchors, paths)

    monkeypatch.setattr(runner, "_training_start_review_clock", review_clock)
    monkeypatch.setattr(runner, "extend_frozen_manifest_anchors", extend)
    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=monotonic_ns,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.context["training_start_state"] == "reviewed_ready"
    assert plan.read_bytes() == launch_bytes
    observed_state, linkage = runner._training_start_state(root)
    assert observed_state == "reviewed_ready"
    assert linkage is not None
    paths = runner._manifest_anchor_paths_for_state(root, 15)
    reconstructed = runner.FrozenManifestAnchors(tuple(sorted((path, runner.sha256_file(root / path)) for path in paths)))
    assert len(reconstructed.records) == 15
    runner.verify_manifest_anchors(root, reconstructed)


@pytest.mark.parametrize("branch", ("hard_abort", "generic"))
def test_training_start_partial_review_cleanup_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    _, evidence, _, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_write = runner.write_canonical_json

    def write(path, value, exclusive=True, owned_paths=None):
        if Path(path) == root / runner.TRAINING_START_LINK_PATH:
            if branch == "hard_abort":
                raise runner.HardAbort("artifact_inconsistency", {"surface": "injected_link_write"})
            raise RuntimeError("injected link write")
        return real_write(path, value, exclusive, owned_paths)

    monkeypatch.setattr(runner, "write_canonical_json", write)
    monkeypatch.setattr(
        runner,
        "_remove_training_start_partial_artifacts",
        lambda *args: (_ for _ in ()).throw(OSError("injected review cleanup")),
    )
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )


@pytest.mark.parametrize("branch", ("hard_abort", "generic"))
@pytest.mark.parametrize("cleanup", ("unlink", "directory_fsync"))
def test_training_start_candidate_cleanup_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
    cleanup: str,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    armed = [False]
    reviewed_clock_calls = [0]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_clock = runner._training_start_review_clock

    def review_clock(reader, request_start, require_margin, state):
        value = real_clock(reader, request_start, require_margin, state)
        if state == "reviewed_ready":
            reviewed_clock_calls[0] += 1
            if reviewed_clock_calls[0] == 2:
                armed[0] = True
                if branch == "hard_abort":
                    raise runner.HardAbort("artifact_inconsistency", {"surface": "injected_candidate_stage"})
                raise RuntimeError("injected candidate stage")
        return value

    candidate_temp = plan.parent / f".{plan.name}.{root.name}.training-start"
    real_unlink = Path.unlink
    real_fsync = runner.fsync_directory

    def unlink(path, *args, **kwargs):
        if cleanup == "unlink" and armed[0] and path == candidate_temp:
            raise OSError("injected candidate unlink")
        return real_unlink(path, *args, **kwargs)

    def fsync(path):
        if cleanup == "directory_fsync" and armed[0] and Path(path) == plan.parent:
            raise OSError("injected candidate directory fsync")
        return real_fsync(path)

    monkeypatch.setattr(runner, "_training_start_review_clock", review_clock)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(runner, "fsync_directory", fsync)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )


def test_training_start_preexisting_candidate_temp_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    candidate_temp = plan.parent / f".{plan.name}.{root.name}.training-start"
    preexisting_bytes = b"preexisting-candidate-temp\n"
    candidate_temp.write_bytes(preexisting_bytes)

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.reason_code == "artifact_inconsistency"
    assert caught.value.context == {
        "surface": "training_start_plan_candidate_temp",
        "training_start_state": "reviewed_ready",
    }
    assert candidate_temp.read_bytes() == preexisting_bytes


@pytest.mark.parametrize(
    ("artifact", "expected_state"),
    (
        ("request", "not_started"),
        ("snapshot", "awaiting_review"),
        ("review", "awaiting_review"),
        ("linkage", "awaiting_review"),
        ("temp", "reviewed_ready"),
    ),
)
def test_training_start_exclusive_open_race_preserves_foreign_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    expected_state: str,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    targets = {
        "request": root / runner.TRAINING_START_REQUEST_PATH,
        "snapshot": root / runner.TRAINING_START_PROJECT_PLAN_PATH,
        "review": None,
        "linkage": root / runner.TRAINING_START_LINK_PATH,
        "temp": plan.parent / f".{plan.name}.{root.name}.training-start",
    }
    foreign_bytes = f"foreign-{artifact}\n".encode("ascii")

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        review = _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)
        review_raw = runner.canonical_json_bytes(review)
        targets["review"] = root / "run" / "reviews" / f"{hashlib.sha256(review_raw).hexdigest()}.json"

    real_open = Path.open
    injected = [False]

    def open_path(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        target = targets[artifact]
        if target is not None and path == target and mode == "xb" and not injected[0]:
            injected[0] = True
            with real_open(path, mode) as handle:
                handle.write(foreign_bytes)
        return real_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.reason_code == "artifact_inconsistency"
    assert caught.value.context["training_start_state"] == expected_state
    assert targets[artifact].read_bytes() == foreign_bytes


def test_training_start_rollback_refuses_replaced_owned_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, _, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    snapshot = root / runner.TRAINING_START_PROJECT_PLAN_PATH
    foreign = b"foreign-training-start-snapshot\n"

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_write = runner._write_exact_bytes

    def write(path, raw, owned_paths=None):
        result = real_write(path, raw, owned_paths)
        if path == snapshot:
            path.rename(path.with_name("owned-project-plan-training-start.md"))
            path.write_bytes(foreign)
        elif path.parent == root / "run" / "reviews":
            raise OSError("injected review write failure")
        return result

    monkeypatch.setattr(runner, "_write_exact_bytes", write)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert snapshot.read_bytes() == foreign


def test_training_start_observed_drift_under_publication_lock_is_preserved_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    reviewed_clock_calls = [0]
    snapshot = {}

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_clock = runner._training_start_review_clock

    def review_clock(reader, request_start, require_margin, state):
        value = real_clock(reader, request_start, require_margin, state)
        if state == "reviewed_ready":
            reviewed_clock_calls[0] += 1
            if reviewed_clock_calls[0] == 3:
                plan.write_bytes(b"concurrent-plan\n")
                snapshot["tree"] = _tree_snapshot(root)
        return value

    monkeypatch.setattr(runner, "_training_start_review_clock", review_clock)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
    )
    assert plan.read_bytes() == b"concurrent-plan\n"
    assert (root / runner.TRAINING_START_LINK_PATH).is_file()
    assert _tree_snapshot(root) == snapshot["tree"]


@pytest.mark.parametrize("stop", ("controller_signal", "kernel_signal", "timeout"))
def test_training_start_publication_lock_contention_is_bounded_by_signal_or_review_clock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    signals = runner.SignalController()
    clock = [100]
    contended = [False]
    sleep_calls = [0]

    def sleeper(seconds):
        sleep_calls[0] += 1
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    def flock(descriptor, operation):
        if operation & runner.fcntl.LOCK_EX:
            contended[0] = True
            if stop == "controller_signal":
                signals.inject()
            elif stop == "timeout":
                clock[0] = 100 + runner.TRAINING_START_REVIEW_WAIT_NS
            raise BlockingIOError("injected lock contention")

    monkeypatch.setattr(runner.fcntl, "flock", flock)
    if stop == "kernel_signal":
        monkeypatch.setattr(runner.signal, "sigpending", lambda: {runner.signal.SIGTERM} if contended[0] else set())
    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            signals,
            source_directory=evidence,
            monotonic_ns=lambda: clock[0],
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    if stop == "timeout":
        assert caught.value.reason_code == "artifact_inconsistency"
        assert caught.value.context["surface"] == "training_start_plan_review_timeout"
    else:
        assert caught.value.reason_code == "signal_or_interruption"
    assert caught.value.context["training_start_state"] == "reviewed_ready"
    assert plan.read_bytes() == launch_bytes
    assert sleep_calls[0] == 1


def test_training_start_publication_rechecks_fifteen_anchors_under_lock_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    replace_called = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    def flock(descriptor, operation):
        if operation == runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB:
            (root / "run" / "prereg.json").write_bytes(b"frozen-anchor-drift\n")

    real_replace = runner.os.replace

    def replace(source, destination):
        if Path(destination) == plan:
            replace_called[0] = True
        return real_replace(source, destination)

    monkeypatch.setattr(runner.fcntl, "flock", flock)
    monkeypatch.setattr(runner.os, "replace", replace)
    with pytest.raises(runner.HardAbort) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert caught.value.reason_code == "frozen_hash_change"
    assert caught.value.context["training_start_state"] == "reviewed_ready"
    assert replace_called[0] is False
    assert plan.read_bytes() == launch_bytes
    assert runner._training_start_state(root)[0] == "reviewed_ready"


def test_training_start_publication_holds_stable_directory_lock_through_replace_and_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    lock_descriptor = [None]
    locked = [False]
    published = [False]
    sequence = []
    candidate_temp = plan.parent / f".{plan.name}.{root.name}.training-start"
    real_verify = runner._verify_active_frozen_hashes

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    def flock(descriptor, operation):
        if operation == runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB:
            metadata = runner.os.fstat(descriptor)
            directory = plan.parent.stat()
            assert (metadata.st_dev, metadata.st_ino) == (directory.st_dev, directory.st_ino)
            lock_descriptor[0] = descriptor
            locked[0] = True
            sequence.append("lock")
        elif operation == runner.fcntl.LOCK_UN:
            assert locked[0] is True
            locked[0] = False
            sequence.append("unlock")

    real_replace = runner.os.replace
    real_fsync = runner.os.fsync

    def replace(source, destination):
        if Path(destination) == plan:
            assert locked[0] is True
            sequence.append("replace")
            published[0] = True
        return real_replace(source, destination)

    def fsync(descriptor):
        if published[0] and descriptor == lock_descriptor[0]:
            assert locked[0] is True
            sequence.append("directory_fsync")
        return real_fsync(descriptor)

    def verify(observed_root, observed_anchors):
        result = real_verify(observed_root, observed_anchors)
        if locked[0] and candidate_temp.is_file() and not published[0]:
            sequence.append("anchor_recheck")
        return result

    monkeypatch.setattr(runner.fcntl, "flock", flock)
    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.os, "fsync", fsync)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", verify)
    runner.establish_training_start_plan_barrier(
        root,
        anchors,
        _BarrierSignals(),
        source_directory=evidence,
        monotonic_ns=lambda: 100,
        sleeper=sleeper,
        utc_reader=lambda: "2026-07-21T00:00:00Z",
    )
    assert sequence == ["lock", "anchor_recheck", "replace", "directory_fsync", "unlock"]


@pytest.mark.parametrize("cleanup_failure", ("unlock", "close"))
def test_training_start_live_ambiguity_survives_directory_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_failure: str,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    lock_descriptor = [None]
    close_failed = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    def flock(descriptor, operation):
        if operation == runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB:
            lock_descriptor[0] = descriptor
            plan.write_bytes(b"concurrent-plan\n")
        elif operation == runner.fcntl.LOCK_UN and cleanup_failure == "unlock":
            raise OSError("injected unlock failure")

    real_close = runner.os.close

    def close(descriptor):
        if cleanup_failure == "close" and descriptor == lock_descriptor[0] and not close_failed[0]:
            close_failed[0] = True
            real_close(descriptor)
            raise OSError("injected directory close failure")
        return real_close(descriptor)

    monkeypatch.setattr(runner.fcntl, "flock", flock)
    monkeypatch.setattr(runner.os, "close", close)
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert str(caught.value) == "training-start live plan is ambiguous"
    assert plan.read_bytes() == b"concurrent-plan\n"


def test_training_start_candidate_identity_capture_failure_removes_created_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    candidate_temp = plan.parent / f".{plan.name}.{root.name}.training-start"

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_descriptor_identity = runner._descriptor_identity

    def descriptor_identity(descriptor):
        if candidate_temp.exists():
            raise OSError("injected candidate identity capture failure")
        return real_descriptor_identity(descriptor)

    monkeypatch.setattr(runner, "_descriptor_identity", descriptor_identity)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert not candidate_temp.exists()
    assert plan.read_bytes() == launch_bytes


def test_cpu_trained_backend_validator_mirrors_scale_aware_gradient_gate() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_trained_backend_payload")
    body = ast.get_source_segment(source, function)
    assert body is not None
    for required in (
        "gradient_relative_max",
        "gradient_normalized_l2_max",
        "gradient_cosine_min",
        "gradient_worst_tensor",
        "gradient_worst_index",
        "gradient_worst_observed",
        "gradient_worst_expected",
        "gradient_absolute_pass",
        "gradient_scale_aware_pass",
        "gradient_pass",
        "gradient_scale_aware_absolute_tolerance",
        "gradient_relative_tolerance",
        "gradient_normalized_l2_tolerance",
        "gradient_cosine_tolerance",
        "loss_tolerance",
    ):
        assert required in body
    assert "record[\"gradient_max_abs\"] <= record[\"gradient_scale_aware_absolute_tolerance\"]" in body
    assert "max(record[\"total_loss_max_abs\"], record[\"component_loss_max_abs\"]) > record[\"loss_tolerance\"]" in body


def test_training_start_contract_uses_single_coordinator_atomic_replacement_and_private_results_namespace() -> None:
    paths = (
        PROJECT_ROOT / "neuroloc" / "wiki" / "PROJECT_PLAN.md",
        PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_run.md",
        PROJECT_ROOT / "neuroloc" / "wiki" / "synthesis" / "modular_neural_model_stack.md",
        TRACKED_PAYLOAD_PATH,
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        prose = " ".join(text.replace("_", " ").split())
        assert "compare-and-swap" not in text
        assert "compare_and_swap" not in text
        assert re.search(r"\bcas\b", text) is None
        assert "single-coordinator" in text or "single_coordinator" in text
        assert "advisory" in text
        assert "cooperative" in text
        assert "noncooperating" in text
        assert "no protection" in prose
        assert "outside the" in prose and "fault model" in prose
        assert "busy or unsupported lock" in prose
        assert "unchanged" in prose and "reviewed ready" in prose
        assert "live bytes matching neither launch nor candidate" in prose
        assert "preserved" in prose and "zero further work" in prose
        assert "unrecoverable" in prose and "no terminal result" in prose
        assert "/users/dttdrv/projects/todorov/neuroloc/results/modular sequence role mlx" in prose
        assert "coordinator private namespace" in prose
        assert "unique initialization staging" in prose
        assert "published final run root" in prose
        assert "disposable publication rehearsal" in prose
        assert "successful exclusive creation or reservation" in prose
        assert "cleanup or terminalization" in prose
        assert "after activation" in prose
        assert "coordinator and the validated single mlx child under frozen path assignments" in prose
        assert "arbitrary noncooperating path replacement" in prose
        assert "arbitrary noncooperating hard link injection" in prose
        assert "arbitrary noncooperating symbolic link injection" in prose
        assert "arbitrary noncooperating mutation inside" in prose
        assert "outside the fault model" in prose
        assert "authorized process failure" in prose
        assert "storage or system call failure" in prose
        assert "abrupt kill or power loss" in prose
        assert "external live project plan writer" in prose
        assert "exact drift" in prose
        assert "external review evidence" in prose
        assert "nonsymlink" in prose
        assert "content addressed" in prose
        assert "content bound" in prose
        assert "external write exception" in prose and "sole" in prose
        assert "ownership is latched only after successful exclusive creation" not in prose
        assert "preexisting and foreign boundary race or replacement paths are preserved" not in prose

    private_namespace = _tracked_payload()["artifacts"]["root_contract"]["coordinator_private_namespace"]
    assert _tracked_payload()["artifacts"]["root_contract"]["writes_outside_published_run_root"] == "sole_governed_external_write_exception_is_the_training_start_live_PROJECT_PLAN_transaction_comprising_owned_same_directory_candidate_temp_creation_atomic_replacement_directory_fsync_and_owned_temp_cleanup"
    assert private_namespace == {
        "root": "/Users/dttdrv/Projects/Todorov/neuroloc/results/modular_sequence_role_mlx",
        "path_classes": [
            "unique_initialization_staging_siblings",
            "published_final_run_root",
            "disposable_publication_rehearsal_paths",
        ],
        "private_interval": "from_successful_exclusive_creation_or_reservation_through_cleanup_or_terminalization",
        "preactivation_mutators": "coordinator_only",
        "postactivation_final_run_root_mutators": "after_activation_the_published_final_run_root_may_be_mutated_only_by_the_coordinator_and_the_validated_single_MLX_child_under_frozen_path_assignments",
        "outside_fault_model": [
            "arbitrary_noncooperating_path_replacement",
            "arbitrary_noncooperating_hard_link_injection",
            "arbitrary_noncooperating_symbolic_link_injection",
            "arbitrary_noncooperating_mutation_inside_the_private_namespace",
        ],
        "in_scope_failures": [
            "authorized_process_failure",
            "storage_or_system_call_failure",
            "stated_abrupt_kill_or_power_loss_boundary",
        ],
        "external_project_plan_boundary": "an_external_live_PROJECT_PLAN_writer_that_ignores_the_cooperative_advisory_lock_remains_outside_the_fault_model_and_exact_drift_is_preserved_as_unrecoverable",
        "external_review_evidence_boundary": "external_review_evidence_remains_outside_RESULTS_PARENT_and_must_be_nonsymlink_content_addressed_and_content_bound",
    }
    orphan_protocol = _tracked_payload()["processes"]["multiprocessing"]["child_orphan_propagation"]
    assert orphan_protocol["message_loss_fallback_exit_code"] == runner.WORKER_ORPHAN_EXIT_CODE == 86
    assert orphan_protocol["child_close_failure_after_orphan"] == "cannot_replace_reserved_exit_code_86"
    assert orphan_protocol["parent_observation_rule"] == "observable_exit_code_86_maps_to_unrecoverable_orphan_even_if_join_observation_fails"
    rollback = _tracked_payload()["artifacts"]["serialization"]["crash_atomic_canonical_jsonl"]["rollback_proof_failure"]
    assert rollback == "any_ftruncate_fsync_seek_read_or_fstat_failure_during_rollback_or_exact_prefix_verification_is_unrecoverable_orphan_and_never_a_downgraded_append_failure"
    abort_failure = _tracked_payload()["abort_rules"]["required_abort_finalization_failure"]
    assert abort_failure == "cleanup_ABORTED_write_closure_validation_or_terminal_checksum_failure_before_terminal_commit_is_unrecoverable_orphan_any_coordinator_owned_provisional_ABORTED_json_is_removed_and_no_false_terminal_result_is_created"


def test_preregistration_rejects_any_other_external_write_contract() -> None:
    payload = _tracked_payload()
    payload["artifacts"]["root_contract"]["writes_outside_published_run_root"] = False
    with pytest.raises(runner.ContractError, match="external write exception differs"):
        runner.validate_prereg_payload(payload)


def test_training_start_postpublication_live_drift_remains_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    candidate = [None]
    real_verify = runner._verify_active_frozen_hashes
    published = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate[0] = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate[0], request_sha256, root.name)

    def verify(observed_root, observed_anchors):
        if published[0]:
            plan.write_bytes(candidate[0] + b"drift\n")
            runner._training_start_state(root)
        return real_verify(observed_root, observed_anchors)

    real_replace = runner.os.replace

    def replace(source, destination):
        real_replace(source, destination)
        if Path(destination) == plan:
            published[0] = True

    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", verify)
    monkeypatch.setattr(runner.os, "replace", replace)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert (root / runner.TRAINING_START_LINK_PATH).is_file()
    assert (root / runner.TRAINING_START_PROJECT_PLAN_PATH).is_file()


def test_training_start_postpublication_plan_read_failure_remains_started(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    published = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_replace = runner.os.replace
    real_read = Path.read_bytes
    real_verify = runner._verify_active_frozen_hashes

    def replace(source, destination):
        real_replace(source, destination)
        if Path(destination) == plan:
            published[0] = True

    def read(path):
        if published[0] and path == plan:
            raise OSError("injected live plan read")
        return real_read(path)

    def verify(observed_root, observed_anchors):
        if published[0]:
            runner._read_training_start_live_plan(plan)
        return real_verify(observed_root, observed_anchors)

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", verify)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )
    assert (root / runner.TRAINING_START_LINK_PATH).is_file()


def test_training_start_live_plan_parent_fsync_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, plan, root, launch_bytes, anchors = _training_start_barrier_surface(tmp_path, monkeypatch)
    published = [False]

    def sleeper(seconds):
        request_sha256 = runner.sha256_file(root / runner.TRAINING_START_REQUEST_PATH)
        candidate_bytes = _training_start_candidate_bytes(launch_bytes, root.name, request_sha256)
        _write_training_start_candidate(evidence, candidate_bytes, request_sha256, root.name)

    real_replace = runner.os.replace
    real_fsync = runner.os.fsync

    def replace(source, destination):
        real_replace(source, destination)
        if Path(destination) == plan:
            published[0] = True

    def fsync(descriptor):
        if published[0]:
            raise OSError("injected live plan directory fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(runner.os, "replace", replace)
    monkeypatch.setattr(runner.os, "fsync", fsync)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.establish_training_start_plan_barrier(
            root,
            anchors,
            _BarrierSignals(),
            source_directory=evidence,
            monotonic_ns=lambda: 100,
            sleeper=sleeper,
            utc_reader=lambda: "2026-07-21T00:00:00Z",
        )


@pytest.mark.parametrize("timeout_value", (1800.0, False))
def test_training_start_request_requires_integer_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_value) -> None:
    repository = tmp_path / "repository"
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    _materialize_training_start_lifecycle(root, repository, "awaiting_review")
    path = root / "run" / "training_start_request.json"
    request = json.loads(path.read_text(encoding="utf-8"))
    request["review_wait_timeout_seconds"] = timeout_value
    path.write_bytes(runner.canonical_json_bytes(request))
    with pytest.raises(runner.ContractError):
        runner._validate_training_start_request_record(root)


@pytest.mark.parametrize("finding_count", (False, 0.0))
def test_linked_training_start_review_requires_integer_zero_finding_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finding_count) -> None:
    repository = tmp_path / "repository"
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    _, linkage = _materialize_training_start_lifecycle(root, repository, "started")
    review_path = root / linkage["review_path"]
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["finding_count"] = finding_count
    review_raw = runner.canonical_json_bytes(review)
    review_sha256 = hashlib.sha256(review_raw).hexdigest()
    replacement = root / "run" / "reviews" / f"{review_sha256}.json"
    review_path.unlink()
    replacement.write_bytes(review_raw)
    linkage["review_path"] = f"run/reviews/{review_sha256}.json"
    linkage["review_artifact_sha256"] = review_sha256
    (root / "run" / "training_start_plan.json").write_bytes(runner.canonical_json_bytes(linkage))
    with pytest.raises(runner.ContractError):
        runner._validate_training_start_linkage(root)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("artifacts", "training_start_review_attestation", "wait_timeout_seconds"), 1800.0),
        (("artifacts", "training_start_review_attestation", "finding_count"), False),
        (("artifacts", "training_start_review_attestation", "finding_count"), 0.0),
    ),
)
def test_preregistration_rejects_noninteger_training_start_schema_values(path, value) -> None:
    payload = _tracked_payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(runner.ContractError):
        runner.validate_prereg_payload(payload)


def test_signal_controller_guarded_transition_linearizes_signal_before_or_after_boundary() -> None:
    before = runner.SignalController()
    before.inject()
    calls = []
    result = before.commit_guarded(lambda: calls.append("boundary"))
    assert result.committed is False
    assert result.pending_signal == runner.signal.SIGTERM
    assert calls == []
    after = runner.SignalController()

    def boundary():
        calls.append("boundary")
        after.inject()
        return 17

    result = after.commit_guarded(boundary)
    assert result.committed is True
    assert result.value == 17
    assert result.pending_signal == runner.signal.SIGTERM


@pytest.mark.parametrize("method", ("guarded", "terminal"))
def test_signal_mask_restore_failure_preserves_underlying_orphan(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    signals = runner.SignalController()
    orphan = runner.UnrecoverableOrphan("injected guarded orphan")
    if method == "terminal":
        signals.defer()

    def pthread_sigmask(how, mask):
        if how == runner.signal.SIG_BLOCK:
            return set()
        raise OSError("injected signal mask restore failure")

    monkeypatch.setattr(runner.signal, "pthread_sigmask", pthread_sigmask)
    monkeypatch.setattr(runner.signal, "sigpending", lambda: set())

    def boundary():
        raise orphan

    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        if method == "guarded":
            signals.commit_guarded(boundary)
        else:
            signals.commit_terminal(boundary)
    assert caught.value is orphan


def test_claim_worker_waits_on_parent_event_before_runtime_model_or_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Gate:
        def wait(self):
            calls.append("wait")
            raise RuntimeError("stop")

    class Connection:
        def send(self, value):
            calls.append("failure")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "validate_entry_environment", lambda: pytest.fail("environment reached before gate"))
    monkeypatch.setattr(runner, "_import_runtime", lambda: pytest.fail("runtime reached before gate"))
    monkeypatch.setattr(runner, "_make_optimizer", lambda *args: pytest.fail("optimizer reached before gate"))
    with pytest.raises(RuntimeError, match="stop"):
        runner._claim_worker("A", str(tmp_path), Gate(), Connection())
    assert calls == ["wait", "failure", "close"]


def test_finalize_and_execute_keep_all_registered_anchor_guard_boundaries() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = _source_tree()
    finalize = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "finalize_clean_claim")
    finalize_segment = ast.get_source_segment(source, finalize)
    direct_stages = (
        "before_completion",
        "after_artifact_validation",
        "after_completion",
        "after_gate_summary",
        "after_summary",
        "after_clean_transport",
        "before_sha256s",
    )
    positions = [finalize_segment.index(f'"{stage}"') for stage in direct_stages]
    assert positions == sorted(positions)
    assert "f\"sha256s_{stage}\"" in finalize_segment
    assert finalize_segment.index("validate_claim_artifact_package") < finalize_segment.index('"after_artifact_validation"')
    assert finalize_segment.index("clean_transport_finalizer(claim_result)") < finalize_segment.index('"after_clean_transport"')
    assert finalize_segment.index('"after_clean_transport"') < finalize_segment.index('"before_sha256s"')
    assert finalize_segment.index('"before_sha256s"') < finalize_segment.index("write_sha256s_terminal")
    execute = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "execute_run")
    execute_segment = ast.get_source_segment(source, execute)
    assert execute_segment.count("_verify_active_frozen_hashes(entry.run_root, frozen_anchors)") == 4
    assert "establish_training_start_plan_barrier(entry.run_root, frozen_anchors, signals)" in execute_segment
    assert execute_segment.index("precreate_claim_ledgers") < execute_segment.index("establish_training_start_plan_barrier") < execute_segment.index("prepare_claim_data")
    assert "selected_claim_runner(entry.run_root, payload, frozen_anchors, signals" in execute_segment
    assert "finalize_clean_claim(entry.run_root, payload, frozen_anchors, signals" in execute_segment
    assert execute_segment.index('claim_result.get("abort_transport_finalizer")') < execute_segment.rindex("if signals.terminal")


def test_parity_validator_requires_exact_order_schema_and_measured_passes() -> None:
    runner.validate_parity_checks(_parity_checks())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda checks: checks.reverse(),
        lambda checks: checks.pop(),
        lambda checks: checks[1].__setitem__("scope", "alias"),
        lambda checks: checks[1].__setitem__("name", checks[0]["name"]),
        lambda checks: checks[1].__setitem__("pass", False),
        lambda checks: checks[1].__setitem__("details_sha256", "bad"),
        lambda checks: checks[1].__setitem__("extra", None),
        lambda checks: checks[1].__setitem__("max_error", None),
        lambda checks: checks[1].__setitem__("max_error", float("nan")),
        lambda checks: checks[1].__setitem__("max_error", -1.0),
        lambda checks: checks[1].__setitem__("max_error", 2e-7),
    ],
)
def test_parity_validator_rejects_scope_name_digest_null_and_numeric_drift(mutator) -> None:
    checks = _parity_checks()
    mutator(checks)
    with pytest.raises(runner.ContractError):
        runner.validate_parity_checks(checks)


def test_ordered_parity_builder_materializes_all_eighteen_details(tmp_path: Path) -> None:
    run_root = tmp_path / "run-id"
    (run_root / "run" / "check_details").mkdir(parents=True)
    facts = {
        scope: runner._parity_fact(
            f"fact_{scope}",
            True,
            {"scope": scope},
            {"measured": True},
            [],
            0.0 if index % 2 else None,
            1e-7 if index % 2 else None,
        )
        for index, scope in enumerate(runner.PARITY_SCOPES)
    }
    checks = runner.build_ordered_parity_checks(run_root, 11, facts)
    assert tuple(record["scope"] for record in checks) == tuple(runner.PARITY_SCOPES)
    assert len(list((run_root / "run" / "check_details").glob("*.json"))) == 18
    reordered = {scope: facts[scope] for scope in reversed(runner.PARITY_SCOPES)}
    with pytest.raises(runner.HardAbort) as caught:
        runner.build_ordered_parity_checks(run_root, 11, reordered)
    assert caught.value.reason_code == "artifact_inconsistency"
    failing = copy.deepcopy(facts)
    failing["source"]["pass"] = False
    with pytest.raises(runner.HardAbort) as caught:
        runner.build_ordered_parity_checks(run_root, 11, failing)
    assert caught.value.reason_code == "assertion_failure"


@pytest.mark.parametrize("rung", (1, 2))
def test_claim_parity_fact_builder_constructs_all_eighteen_scopes_from_measured_inputs(tmp_path: Path, rung: int) -> None:
    assertions = {name: {"actual": {}, "pass": True} for name in runner.PRETRAINING_ASSERTION_IDS}
    assertions["mixer_abi_and_residual_ownership"]["actual"] = {"residual_max_error": 0.0}
    assertions["exact_architecture"]["actual"] = {"schedule": ["local"]}
    assertions["reset_aware_recurrent_fidelity"]["actual"] = {
        "forward_max_error": 0.0,
        "input_gradient_max_error": 0.0,
        "parameter_gradient_max_error": 0.0,
        "reset_before": [80, 96],
        "reset_after": [80, 96],
        "rung_two_chunk_ends": list(range(31, 512, 32)),
    }
    assertions["firewall_factorization"]["actual"] = {"factorized": True}
    assertions["causality"]["actual"] = {"max_prefix_error": 0.0}
    assertions["forced_and_random_route_exactness"]["actual"] = {"forced_exact": True, "generator_exact": True}
    assertions["source_host_route_and_attention_parity"]["actual"] = {
        "host": {"architecture_exact": True, "recurrent_max_error": 0.0, "feature_max_error": 0.0},
        "route": {"internal_loss_max_error": 0.0},
    }
    assertions["state_and_index_lifetime"]["actual"] = {"state_hash_stable": True}
    route_payload = {"overflow_count": 0, "max_bucket_load": 0, "postcheckpoint_assertions": True}
    reload_records = [{"fresh_instance": True, "state_exact": True, "route_exact": True, "logits_max_error": 0.0, "hidden_max_error": 0.0}]
    facts = runner._claim_parity_facts(
        assertions,
        11 if rung == 1 else 83,
        rung,
        {"verified": True},
        route_payload,
        0.0,
        1e-5,
        {"pass": True},
        {"pass": True},
        reload_records,
        {"pass": True},
        {"pass": True, "stages": ["rung_two"] if rung == 2 else ["donor", "router_only", "joint", "dense_base", "dense_continuation"], "records": [], "max_error": 0.0},
        [],
    )
    assert tuple(facts) == runner.PARITY_SCOPES
    assert len(facts) == 18
    assert all(fact["pass"] is True for fact in facts.values())
    run_root = tmp_path / f"rung-{rung}"
    (run_root / "run" / "check_details").mkdir(parents=True)
    checks = runner.build_ordered_parity_checks(run_root, 11 if rung == 1 else 83, facts)
    assert len(checks) == 18
    changed = {**route_payload, "postcheckpoint_assertions": False}
    changed_facts = runner._claim_parity_facts(
        assertions,
        11 if rung == 1 else 83,
        rung,
        {"verified": True},
        changed,
        0.0,
        1e-5,
        {"pass": True},
        {"pass": True},
        reload_records,
        {"pass": True},
        {"pass": True, "stages": ["rung_two"] if rung == 2 else ["donor", "router_only", "joint", "dense_base", "dense_continuation"], "records": [], "max_error": 0.0},
        [],
    )
    assert changed_facts["raw_route"]["pass"] is False
    with pytest.raises(runner.HardAbort) as caught:
        runner.build_ordered_parity_checks(run_root, 11 if rung == 1 else 83, changed_facts)
    assert caught.value.reason_code == "assertion_failure"


def test_terminal_semantic_parity_reconstruction_rejects_self_consistent_mutation_of_every_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_modules) -> None:
    root = tmp_path / "semantic-parity"
    (root / "run" / "check_details").mkdir(parents=True)
    assertions = {name: {"actual": {}, "pass": True} for name in runner.PRETRAINING_ASSERTION_IDS}
    assertions["mixer_abi_and_residual_ownership"]["actual"] = {"residual_max_error": 0.0}
    assertions["exact_architecture"]["actual"] = {"schedule": ["local"]}
    assertions["reset_aware_recurrent_fidelity"]["actual"] = {
        "forward_max_error": 0.0,
        "input_gradient_max_error": 0.0,
        "parameter_gradient_max_error": 0.0,
        "reset_before": [80, 96],
        "reset_after": [80, 96],
        "rung_two_chunk_ends": list(range(31, 512, 32)),
    }
    assertions["firewall_factorization"]["actual"] = {"factorized": True}
    assertions["causality"]["actual"] = {"max_prefix_error": 0.0}
    assertions["forced_and_random_route_exactness"]["actual"] = {"forced_exact": True, "generator_exact": True}
    assertions["source_host_route_and_attention_parity"]["actual"] = {
        "host": {"architecture_exact": True, "recurrent_max_error": 0.0, "feature_max_error": 0.0},
        "route": {"internal_loss_max_error": 0.0},
    }
    assertions["state_and_index_lifetime"]["actual"] = {"state_hash_stable": True}
    endpoints = {}
    for stage in runner.RUNG_ONE_STAGE_ENDPOINTS:
        path = root / f"{stage}.pt"
        path.write_bytes(stage.encode("ascii"))
        endpoints[stage] = {"path": path, "sha256": runner.sha256_file(path), "checkpoint": {}}
    train_rows = [{"raw_overflow_count": 0, "max_bucket_load": 0}]
    evaluation_rows = [{"overflow_count": 0, "max_bucket_load": 0}]
    data_evidence = {
        "postcheckpoint_assertions": True,
        "source_exclusion_sha256": "a" * 64,
    }
    oracle_evidence = {"attention_error": 0.0}
    initialization_payload = {"pass": True}
    copy_payload = {"pass": True}
    reload_records = [
        {
            "fresh_instance": True,
            "state_exact": True,
            "route_exact": True,
            "logits_max_error": 0.0,
            "hidden_max_error": 0.0,
        }
    ]
    checksum_payload = {
        "verified": True,
        "sha256s": [endpoints[stage]["sha256"] for stage in ("donor", "router_only", "joint", "dense_base", "dense_continuation")],
    }
    route_payload = {"overflow_count": 0, "max_bucket_load": 0, **data_evidence}
    evidence_paths = ["run/preflight.json", "rung1/11/checkpoints/final_last.pt", "rung1/11/intervention_deltas.json"]
    facts = runner._claim_parity_facts(
        assertions,
        11,
        1,
        checksum_payload,
        route_payload,
        0.0,
        1e-5,
        initialization_payload,
        copy_payload,
        reload_records,
        {"pass": True, "record_count": len(_intervention_records()), "matched_intact": True, "knockout_zero_exposed": True},
        {"pass": True, "stages": ["donor", "router_only", "joint", "dense_base", "dense_continuation"], "records": [], "max_error": 0.0},
        evidence_paths,
    )
    parity = {"checks": runner.build_ordered_parity_checks(root, 11, facts)}
    monkeypatch.setattr(runner, "_pretraining_assertion_lookup", lambda run_root: assertions)
    monkeypatch.setattr(runner, "_reconstruct_rung_one_initialization", lambda seed, observed_endpoints, runtime: (initialization_payload, copy_payload))
    monkeypatch.setattr(runner, "_reconstruct_reload_records", lambda observed_endpoints, rung, seed, evaluation_payload, runtime: reload_records)
    monkeypatch.setattr(runner, "_trained_backend_payload", lambda run_root, rung, seed, endpoints, records, runtime: {"pass": True, "stages": ["donor", "router_only", "joint", "dense_base", "dense_continuation"], "records": [], "max_error": 0.0})
    arguments = (
        root,
        1,
        11,
        endpoints,
        train_rows,
        evaluation_rows,
        {},
        _intervention_records(),
        parity,
        runtime_modules,
        {},
        data_evidence,
        oracle_evidence,
    )
    runner._validate_semantic_parity_package(*arguments)
    for index, check in enumerate(parity["checks"]):
        path = root / "run" / "check_details" / f"{check['details_sha256']}.json"
        detail = json.loads(path.read_text(encoding="utf-8"))
        detail["outputs"] = {**detail["outputs"], "semantic_tamper": check["scope"]}
        raw = runner.canonical_json_bytes(detail)
        digest = hashlib.sha256(raw).hexdigest()
        (root / "run" / "check_details" / f"{digest}.json").write_bytes(raw)
        changed = copy.deepcopy(parity)
        changed["checks"][index]["details_sha256"] = digest
        changed_arguments = (*arguments[:8], changed, *arguments[9:])
        with pytest.raises(runner.ContractError):
            runner._validate_semantic_parity_package(*changed_arguments)


def test_pretraining_assertion_lookup_requires_exact_fifteen_unique_passes(tmp_path: Path) -> None:
    run_root = tmp_path / "run-id"
    (run_root / "run").mkdir(parents=True)
    package = {
        "source_checks": [],
        "result_checks": [],
        "transformerov_selfcheck": [],
        "routing_parity": [],
        "host_parity": [{"name": name, "pass": True} for name in runner.PRETRAINING_ASSERTION_IDS],
        "trained_backend": [],
        "lifecycle_assertions": [],
    }
    (run_root / "run" / "preflight.json").write_bytes(runner.canonical_json_bytes(package))
    lookup = runner._pretraining_assertion_lookup(run_root)
    assert tuple(lookup) == tuple(runner.PRETRAINING_ASSERTION_IDS)
    duplicate = copy.deepcopy(package)
    duplicate["source_checks"].append(copy.deepcopy(duplicate["host_parity"][0]))
    (run_root / "run" / "preflight.json").write_bytes(runner.canonical_json_bytes(duplicate))
    with pytest.raises(runner.HardAbort) as caught:
        runner._pretraining_assertion_lookup(run_root)
    assert caught.value.reason_code == "artifact_inconsistency"
    missing = copy.deepcopy(package)
    missing["host_parity"].pop()
    (run_root / "run" / "preflight.json").write_bytes(runner.canonical_json_bytes(missing))
    with pytest.raises(runner.HardAbort) as caught:
        runner._pretraining_assertion_lookup(run_root)
    assert caught.value.reason_code == "assertion_failure"
    failed = copy.deepcopy(package)
    failed["host_parity"][0]["pass"] = False
    (run_root / "run" / "preflight.json").write_bytes(runner.canonical_json_bytes(failed))
    with pytest.raises(runner.HardAbort) as caught:
        runner._pretraining_assertion_lookup(run_root)
    assert caught.value.reason_code == "assertion_failure"


def test_pretraining_assertion_closure_constructs_exact_production_fifteen_and_rejects_drift() -> None:
    values = {assertion_id: ({"assertion_id": assertion_id}, True) for assertion_id in runner.PRETRAINING_ASSERTION_IDS}
    records = runner.validate_pretraining_assertion_closure(values)
    assert len(records) == 15
    assert tuple(record["assertion_id"] for record in records) == runner.PRETRAINING_ASSERTION_IDS
    assert all(record["pass"] is True for record in records)
    mutations = []
    mutations.append({key: values[key] for key in reversed(values)})
    missing = copy.deepcopy(values)
    missing.pop(runner.PRETRAINING_ASSERTION_IDS[-1])
    mutations.append(missing)
    extra = copy.deepcopy(values)
    extra["invented"] = ({}, True)
    mutations.append(extra)
    malformed_actual = copy.deepcopy(values)
    malformed_actual[runner.PRETRAINING_ASSERTION_IDS[0]] = ([], True)
    mutations.append(malformed_actual)
    malformed_pass = copy.deepcopy(values)
    malformed_pass[runner.PRETRAINING_ASSERTION_IDS[0]] = ({}, 1)
    mutations.append(malformed_pass)
    failed = copy.deepcopy(values)
    failed[runner.PRETRAINING_ASSERTION_IDS[0]] = ({}, False)
    mutations.append(failed)
    for changed in mutations:
        with pytest.raises(runner.ContractError):
            runner.validate_pretraining_assertion_closure(changed)


def test_pretraining_implementation_names_every_registered_assertion_and_build_path() -> None:
    tree = _source_tree()
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assertion_source = ast.get_source_segment(RUNNER_PATH.read_text(encoding="utf-8"), functions["run_pretraining_assertions"])
    build_source = ast.get_source_segment(RUNNER_PATH.read_text(encoding="utf-8"), functions["build_shared_prepilot_base"])
    for assertion_id in runner.PRETRAINING_ASSERTION_IDS:
        assert f'assertion_values["{assertion_id}"]' in assertion_source
    assert "run_pretraining_assertions" in build_source
    assert "host_parity.extend(pretraining_records)" in build_source


def test_both_seed_workers_use_full_parity_and_ledger_accounting_validators() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_run_rung_one_seed", "_run_rung_two_seed"):
        function_source = ast.get_source_segment(source, functions[name])
        assert "build_ordered_parity_checks" in function_source
        assert "validate_model_accounting" in function_source


def test_review_scope_registry_has_exact_four_nonoverlapping_roles() -> None:
    scopes = runner._review_scopes()
    assert [scope for scope, paths in scopes] == [
        "base_preregistration",
        "base_implementation",
        "base_tests",
        "base_complete_surface",
    ]
    path_sets = {scope: tuple(paths) for scope, paths in scopes}
    assert path_sets["base_preregistration"] == tuple(
        sorted(
            (
                "neuroloc/wiki/PROJECT_PLAN.md",
                "neuroloc/wiki/synthesis/modular_neural_model_stack.md",
                "neuroloc/wiki/tests/modular_sequence_role_cpu_run.md",
                "neuroloc/wiki/tests/modular_sequence_role_cpu_prereg.json",
            )
        )
    )
    assert path_sets["base_implementation"] == tuple(
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
    assert path_sets["base_tests"] == tuple(sorted(("tests/test_modular_neural_machine.py", "tests/test_modular_sequence_role_cpu.py", "tests/test_modular_sequence_role_mlx.py")))
    assert path_sets["base_complete_surface"] == tuple(
        sorted(set(path_sets["base_preregistration"] + path_sets["base_implementation"] + path_sets["base_tests"]))
    )


def test_configuration_registry_binds_every_reviewed_implementation_and_test() -> None:
    scopes = dict(runner._review_scopes())
    records = {record["path"]: record for record in runner._config_records()}
    required = set(scopes["base_implementation"]) | set(scopes["base_tests"])
    assert required <= set(records)
    for path in scopes["base_implementation"]:
        assert records[path]["role"] == "implementation_configuration"
    for path in scopes["base_tests"]:
        assert records[path]["role"] == "implementation_test_configuration"


def test_mlx_qualification_contract_counts_prepared_and_endpoint_data_files(tmp_path: Path, runtime_modules) -> None:
    from src.model import modular_mlx_backend

    runner.prepare_claim_data(tmp_path, runtime_modules)
    paths = sorted(path.relative_to(tmp_path).as_posix() for path in (tmp_path / "data").iterdir())
    assert len(paths) == 11
    final_data_files = len(paths) + len(runner.RUNG_ONE_SEEDS)
    assert modular_mlx_backend.qualification_cardinality_contract()["data_files"] == final_data_files


def test_mlx_completion_derives_committed_work_from_canonical_ledgers(tmp_path: Path) -> None:
    from scripts import qualify_modular_mlx

    root = tmp_path / "test-run"
    root.mkdir()
    (root / "run").mkdir()
    for index, relative in enumerate(runner.CLAIM_LEDGER_PATHS[1:]):
        seed = (*runner.RUNG_ONE_SEEDS, runner.RUNG_TWO_SEED)[index]
        rows = [_attempt_event(0, "started", construction_seed=seed), _attempt_event(1, "completed", construction_seed=seed)]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        runner._write_canonical_jsonl(path, rows)
    completion = qualify_modular_mlx.write_canonical_completion(root, 100, [{"sample_id": 7, "monotonic_ns": 900}])
    assert completion["attempted_updates"] == 6
    assert completion["completed_updates"] == 6
    assert completion["token_positions"] == 12_288


def test_mlx_resource_sampler_commits_live_stage_attribution_to_canonical_writer(tmp_path: Path) -> None:
    from src.model import modular_mlx_backend

    path = tmp_path / "resources.jsonl"
    writer = runner.CrashAtomicJsonlWriter(path, runner.validate_resource_row)
    writer.precreate()
    processes = [
        {"pid": 321, "ppid": 1, "rss_bytes": 4096, "cpu_time_us": 7000},
        {"pid": 654, "ppid": 321, "rss_bytes": 16384, "cpu_time_us": 9000},
    ]
    sampler = modular_mlx_backend.QualificationResourceSampler("test-run", 321, 654, lambda _: processes, lambda: 0, 0.001, writer=writer)
    sampler.begin_stage("donor", [11])
    sampler.start()
    sampler.await_stage_sample("donor", [11], time.monotonic_ns() + 1_000_000_000)
    rows = sampler.stop()
    writer.close()
    persisted = runner.validate_canonical_jsonl_prefix(path, path.stat().st_size, runner.validate_resource_row)
    assert list(persisted) == rows
    assert any(row["active_jobs"] == [{"worker": "S11", "seed": 11, "stage": "donor", "logical_update": 0}] for row in rows)


def test_mlx_attempt_batch_commits_directly_to_canonical_seed_ledgers(tmp_path: Path) -> None:
    from scripts import qualify_modular_mlx

    writers = {}
    paths = {}
    rows = []
    for seed in (11, 23):
        path = tmp_path / "rung1" / str(seed) / "attempts.jsonl"
        path.parent.mkdir(parents=True)
        writer = runner.CrashAtomicJsonlWriter(path, runner.validate_attempt_row, sequence_kind="attempt")
        writer.precreate()
        writers[seed] = writer
        paths[seed] = path
        rows.append(_attempt_event(0, "started", construction_seed=seed))
    qualify_modular_mlx.write_attempt_batch(rows, writers)
    for seed, writer in writers.items():
        writer.close()
        persisted = runner.validate_canonical_jsonl_prefix(paths[seed], paths[seed].stat().st_size, runner.validate_attempt_row)
        assert list(persisted) == [rows[(11, 23).index(seed)]]
    assert not (tmp_path / "run" / "atomic").exists()


def test_mlx_child_invocation_uses_reviewed_preimport_bootstrap() -> None:
    from src.model import modular_mlx_backend

    command, environment = modular_mlx_backend.child_invocation("serve")
    assert command == [
        "/Users/dttdrv/Projects/Transformerov/.venv/bin/python",
        str(PROJECT_ROOT / "scripts" / "qualify_modular_mlx.py"),
        "--child-mode",
        "serve",
    ]
    assert environment == modular_mlx_backend.child_invocation("pilot")[1]


def test_mlx_child_bootstrap_rejects_runtime_import_and_environment_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import qualify_modular_mlx
    from src.model import modular_mlx_backend

    _, environment = modular_mlx_backend.child_invocation("self-check")
    environment["__CF_USER_TEXT_ENCODING"] = "0x1F5:0x0:0x0"
    monkeypatch.setattr(qualify_modular_mlx.os, "environ", dict(environment))
    monkeypatch.setattr(qualify_modular_mlx.sys, "version_info", (3, 9, 6))
    qualify_modular_mlx.validate_child_bootstrap("self-check")
    qualify_modular_mlx.os.environ["OMP_NUM_THREADS"] = "3"
    with pytest.raises(modular_mlx_backend.MlxQualificationError, match="environment"):
        qualify_modular_mlx.validate_child_bootstrap("self-check")
    qualify_modular_mlx.os.environ = dict(environment)
    monkeypatch.setitem(qualify_modular_mlx.sys.modules, "mlx", object())
    with pytest.raises(modular_mlx_backend.MlxQualificationError, match="pre-import"):
        qualify_modular_mlx.validate_child_bootstrap("self-check")


def _complete_mlx_preflight_self_check(modular_mlx_backend: Any) -> dict[str, Any]:
    from tests.test_modular_sequence_role_mlx import _complete_self_check

    value = _complete_self_check()
    modular_mlx_backend.validate_initial_self_check(value)
    return value


def test_mlx_preflight_fixture_reuses_the_canonical_validator_fixture() -> None:
    from src.model import modular_mlx_backend
    from tests.test_modular_sequence_role_mlx import _complete_self_check

    observed = _complete_mlx_preflight_self_check(modular_mlx_backend)
    assert observed == _complete_self_check()
    assert modular_mlx_backend.validate_initial_self_check(observed)["pass"] is True
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=__file__)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_complete_mlx_preflight_self_check")
    body = ast.get_source_segment(source, function)
    assert body is not None
    assert "from tests.test_modular_sequence_role_mlx import _complete_self_check" in body
    assert "modular_mlx_backend.validate_initial_self_check(value)" in body
    assert body.count("{") == 0


def test_mlx_preflight_probe_materializes_five_content_addressed_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import qualify_modular_mlx

    root = tmp_path / "run-id"
    (root / "run" / "check_details").mkdir(parents=True)
    observed = _complete_mlx_preflight_self_check(qualify_modular_mlx.backend)
    completed = qualify_modular_mlx.subprocess.CompletedProcess([], 0, runner.canonical_json_bytes(observed).decode("utf-8"), "")
    monkeypatch.setattr(qualify_modular_mlx.subprocess, "run", lambda *args, **kwargs: completed)
    checks = qualify_modular_mlx.preflight_mlx_probe(root, root.name)
    assert [record["name"] for record in checks] == ["mlx_child_environment", "mlx_metal_device", "mlx_self_check", "initial_backend_parity", "full_package_projection"]
    assert all(record["pass"] is True for record in checks)
    assert len(list((root / "run" / "check_details").glob("*.json"))) == 5
    for field, value in (("hidden_max_abs", 1.0550022125e-5), ("sequence_delta_max_abs_by_block", [5e-6] * 6 + [1.1650845408e-5, 5e-6])):
        invalid = copy.deepcopy(observed)
        invalid["full_model_parity"][field] = value
        invalid_completed = qualify_modular_mlx.subprocess.CompletedProcess([], 0, runner.canonical_json_bytes(invalid).decode("utf-8"), "")
        monkeypatch.setattr(qualify_modular_mlx.subprocess, "run", lambda *args, completed=invalid_completed, **kwargs: completed)
        with pytest.raises(qualify_modular_mlx.backend.MlxQualificationError):
            qualify_modular_mlx.preflight_mlx_probe(root, root.name)


def test_review_attestation_contract_is_external_run_id_free_and_content_addressed() -> None:
    schemas = _tracked_payload()["artifacts"]["schemas"]
    artifact = schemas["review_artifact"]
    required = _tracked_payload()["artifacts"]["source_manifest_required_review_attestations"]
    assert artifact["source_evidence_directory"] == "neuroloc/results/modular_sequence_role_mlx_reviews"
    assert artifact["source_evidence_path_pattern"] == (
        "neuroloc/results/modular_sequence_role_mlx_reviews/{artifact_sha256}.json"
    )
    assert artifact["source_evidence_directory_symlink_allowed"] is False
    assert artifact["source_evidence_file_symlink_allowed"] is False
    assert artifact["source_evidence_directory_outside_all_review_target_scopes"] is True
    assert artifact["runner_authors_or_mutates_attestation"] is False
    assert artifact["source_and_run_bytes_identical"] is True
    assert artifact["schema_version_value"] == REVIEW_ATTESTATION_SCHEMA_VERSION
    assert artifact["exact_keys"] == [
        "schema_version",
        "reviewer",
        "scope",
        "target_records",
        "target_sha256",
        "findings",
        "finding_count",
    ]
    assert "run_id" not in artifact["exact_keys"]
    assert artifact["target_record_exact_keys"] == ["path", "sha256"]
    assert artifact["target_record_order"] == "path_sorted"
    assert artifact["target_sha256_formula"] == "sha256_canonical_json_of_path_sorted_path_sha256_records"
    assert artifact["accepted_reviewer"] == "feature-dev:code-reviewer"
    assert artifact["accepted_findings"] == []
    assert artifact["accepted_finding_count"] == 0
    assert artifact["base_content_hash_binding"] == "source_filename_run_filename_config_manifest_reference_source_manifest_sha256_and_sha256_colon_revision"
    assert artifact["training_start_content_hash_binding"] == "source_filename_run_filename_and_run/training_start_plan.json_review_artifact_sha256"
    assert artifact["training_start_scope_exact_targets"] == ["neuroloc/wiki/PROJECT_PLAN.md", "run/training_start_request.json"]
    assert required == {
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
    }
    training_start = _tracked_payload()["artifacts"]["training_start_review_attestation"]
    assert training_start["count"] == 1
    assert training_start["scope_formula"] == "training_start_project_plan:{run_id}"
    assert training_start["target_paths"] == ["neuroloc/wiki/PROJECT_PLAN.md", "run/training_start_request.json"]
    assert training_start["target_must_differ_from_launch_snapshot"] is True
    assert training_start["reviewer"] == "feature-dev:code-reviewer"
    assert training_start["findings"] == []
    assert training_start["finding_count"] == 0
    function_names = {node.name for node in ast.walk(_source_tree()) if isinstance(node, ast.FunctionDef)}
    assert "select_and_copy_review_attestations" in function_names
    assert "build_review_artifacts" not in function_names


def test_preserved_review_attestations_match_exact_scopes_and_copy_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    sources = _write_complete_review_set(repository, evidence)
    before = {path.name: path.read_bytes() for path, _, _ in sources.values()}
    review_records, source_records = runner.select_and_copy_review_attestations(staging, evidence)
    scopes = [scope for scope, _ in runner._review_scopes()]
    roles = [
        "review_attestation_base_preregistration",
        "review_attestation_base_implementation",
        "review_attestation_base_tests",
        "review_attestation_base_complete_surface",
    ]
    assert [record["scope"] for record in review_records] == scopes
    assert [record["role"] for record in source_records] == roles
    assert len(review_records) == len(source_records) == 4
    for scope, review_record, source_record in zip(scopes, review_records, source_records):
        source, raw, digest = sources[scope]
        destination = staging / "run" / "reviews" / source.name
        assert destination.read_bytes() == raw == source.read_bytes()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        assert review_record == {
            "reviewer": "feature-dev:code-reviewer",
            "scope": scope,
            "target_sha256": _review_attestation(repository, scope)["target_sha256"],
            "finding_count": 0,
            "artifact_sha256": digest,
        }
        assert source_record == {
            "path": f"neuroloc/results/modular_sequence_role_mlx_reviews/{digest}.json",
            "role": roles[scopes.index(scope)],
            "size_bytes": len(raw),
            "sha256": digest,
            "revision": f"sha256:{digest}",
        }
    assert {path.name: path.read_bytes() for path, _, _ in sources.values()} == before


def test_preserved_review_attestations_refuse_missing_current_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    sources = _write_complete_review_set(repository, evidence)
    sources["base_tests"][0].unlink()
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


def test_absent_preserved_review_directory_cannot_be_self_certified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    evidence.rmdir()
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)
    assert not (staging / "run" / "reviews").exists()


def test_preserved_review_attestations_refuse_duplicate_current_scope_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    _write_complete_review_set(repository, evidence)
    duplicate = _review_attestation(repository, "base_tests", finding_count=0.0)
    _write_review_attestation(evidence, duplicate)
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


@pytest.mark.parametrize("finding_count", [False, 0.0])
def test_preserved_review_attestations_require_integer_zero_finding_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finding_count,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    _write_complete_review_set(repository, evidence)
    valid = _write_review_attestation(evidence, _review_attestation(repository, "base_tests"))[0]
    valid.unlink()
    _write_review_attestation(evidence, _review_attestation(repository, "base_tests", finding_count=finding_count))
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


def test_preserved_review_attestations_refuse_current_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    _write_complete_review_set(repository, evidence)
    target = repository / dict(runner._review_scopes())["base_tests"][0]
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


@pytest.mark.parametrize("mutation", ["wrong_filename", "noncanonical_bytes"])
def test_preserved_review_attestations_refuse_content_address_or_encoding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    sources = _write_complete_review_set(repository, evidence)
    source, raw, _ = sources["base_tests"]
    source.unlink()
    if mutation == "wrong_filename":
        (evidence / f"{'0' * 64}.json").write_bytes(raw)
    else:
        raw = json.dumps(_review_attestation(repository, "base_tests"), indent=2).encode("utf-8")
        (evidence / f"{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


@pytest.mark.parametrize("mutation", ["reversed", "extra", "stale_sha256"])
def test_preserved_review_attestations_refuse_target_path_or_order_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    sources = _write_complete_review_set(repository, evidence)
    sources["base_tests"][0].unlink()

    def transform(records):
        if mutation == "reversed":
            return list(reversed(records))
        if mutation == "extra":
            return records + [{"path": "tests/extra.py", "sha256": "0" * 64}]
        records[0]["sha256"] = "0" * 64
        return records

    _write_review_attestation(evidence, _review_attestation(repository, "base_tests", transform_records=transform))
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


def test_preserved_review_attestations_refuse_symlink_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    _write_complete_review_set(repository, evidence)
    link = tmp_path / "evidence-link"
    link.symlink_to(evidence, target_is_directory=True)
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, link)


def test_preserved_review_attestations_refuse_symlink_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    sources = _write_complete_review_set(repository, evidence)
    source, raw, digest = sources["base_tests"]
    source.unlink()
    external = tmp_path / "external.json"
    external.write_bytes(raw)
    (evidence / f"{digest}.json").symlink_to(external)
    with pytest.raises(runner.InitializationRefusal):
        runner.select_and_copy_review_attestations(staging, evidence)


def test_real_regular_file_validator_rejects_leaf_and_ancestor_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "record.json"
    target.write_bytes(b"{}")
    assert runner.validate_real_regular_file(target) == target
    leaf = real / "leaf.json"
    leaf.symlink_to(target)
    ancestor = tmp_path / "ancestor"
    ancestor.symlink_to(real, target_is_directory=True)
    for candidate in (leaf, ancestor / target.name, real, Path("record.json"), tmp_path / "absent.json"):
        with pytest.raises(runner.ContractError):
            runner.validate_real_regular_file(candidate)


def test_hardware_observation_reads_exact_sysctl_values() -> None:
    values = {
        "machdep.cpu.brand_string": "Apple M5 Pro",
        "hw.physicalcpu": "15",
        "hw.memsize": "25769803776",
    }
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return runner.subprocess.CompletedProcess(argv, 0, stdout=values[argv[-1]] + "\n", stderr="")

    assert runner.observe_target_hardware(fake_run) == {
        "chip": "Apple M5 Pro",
        "cpu_core_count": 15,
        "memory_bytes": 25769803776,
        "training_device": "integrated_Apple_GPU_via_MLX_Metal",
    }
    assert [call[0] for call in calls] == [
        ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
        ["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
        ["/usr/sbin/sysctl", "-n", "hw.memsize"],
    ]
    assert all(call[1]["capture_output"] is True and call[1]["text"] is True and call[1]["check"] is False for call in calls)
    assert all(call[1]["env"]["LC_ALL"] == "C" for call in calls)


@pytest.mark.parametrize(
    "values",
    [
        {"machdep.cpu.brand_string": "Apple M4 Pro", "hw.physicalcpu": "15", "hw.memsize": "25769803776"},
        {"machdep.cpu.brand_string": "Apple M5 Pro", "hw.physicalcpu": "14", "hw.memsize": "25769803776"},
        {"machdep.cpu.brand_string": "Apple M5 Pro", "hw.physicalcpu": "15", "hw.memsize": "25769803775"},
        {"machdep.cpu.brand_string": "Apple M5 Pro", "hw.physicalcpu": "bad", "hw.memsize": "25769803776"},
    ],
)
def test_hardware_observation_refuses_mismatch_or_unparseable_values(values) -> None:
    def fake_run(argv, **kwargs):
        return runner.subprocess.CompletedProcess(argv, 0, stdout=values[argv[-1]], stderr="")

    with pytest.raises(runner.InitializationRefusal):
        runner.observe_target_hardware(fake_run)


def test_environment_artifact_is_preworker_immutable_and_exact() -> None:
    torch = type("Torch", (), {"__version__": "2.8.0"})()
    runtime = runner.RuntimeModules(torch=torch, model_module=None)
    hardware = {"chip": "Apple M5 Pro", "cpu_core_count": 15, "memory_bytes": 25769803776, "training_device": "integrated_Apple_GPU_via_MLX_Metal"}
    artifact = runner._environment_artifact(
        "run-01",
        ["python", str(RUNNER_PATH), "--run-root", "/abs/run-01"],
        runtime,
        hardware_reader=lambda: hardware,
    )
    schemas = _tracked_payload()["artifacts"]["schemas"]["environment"]
    assert set(artifact) == set(schemas["exact_keys"])
    for key, exact_keys in schemas["nested_exact_keys"].items():
        assert set(artifact[key]) == set(exact_keys)
    assert artifact["process"]["child_pids"] == []
    assert artifact["process"]["start_method"] == "parent_subprocess_with_start_new_session"
    assert artifact["hardware"] == hardware
    assert artifact["environment"] == REQUIRED_ENV
    assert artifact["threads"] == {"torch_intraop": 4, "torch_interop": 1, "mlx_omp": 4, "mlx_veclib": 4}
    assert artifact["numerics"] == {
        "training_device": "Device(gpu, 0)",
        "training_dtype": "mlx.core.float32",
        "reference_device": "cpu",
        "reference_dtype": "torch.float32",
        "deterministic_algorithms": True,
        "matmul_precision": "highest",
        "autocast": False,
        "compilation": True,
    }
    sampling = _tracked_payload()["pilot"]["resource_sampling"]
    assert artifact["resource_sampling"]["ps_argv_template"] == [
        *sampling["ps_argv_prefix"],
        sampling["ps_pid_argument"],
    ]
    assert artifact["resource_sampling"]["swap_argv"] == sampling["swap_argv"]


@pytest.mark.parametrize("family", ("publication", "ledger", "terminal"))
@pytest.mark.parametrize("failure", ("body", "cleanup"))
def test_disposable_rehearsal_failure_cleanup_is_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    failure: str,
) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    run_id = "rehearsal-run"
    body_error = runner.ContractError(f"injected {family} rehearsal body failure")
    real_remove = runner._remove_tree_and_fsync

    def remove(path):
        if failure == "cleanup":
            raise OSError(f"injected {family} rehearsal cleanup failure")
        return real_remove(path)

    monkeypatch.setattr(runner, "_remove_tree_and_fsync", remove)
    if family == "publication":
        real_fsync_directory = runner.fsync_directory

        def fsync_directory(path):
            if Path(path).name.endswith(".source"):
                raise body_error
            return real_fsync_directory(path)

        monkeypatch.setattr(runner, "fsync_directory", fsync_directory)
        invoke = lambda: runner._disposable_publication_rehearsal(parent, run_id)
    elif family == "ledger":
        monkeypatch.setattr(runner, "_attempt_fixture", lambda observed_run_id: (_ for _ in ()).throw(body_error))
        invoke = lambda: runner.rehearse_crash_atomic_faults(parent, run_id)
    else:
        monkeypatch.setattr(runner, "write_canonical_json", lambda *args, **kwargs: (_ for _ in ()).throw(body_error))
        invoke = lambda: runner._terminal_deactivation_rehearsal(parent, run_id)
    if failure == "body":
        with pytest.raises(runner.ContractError) as caught:
            invoke()
        assert caught.value is body_error
        assert not any(parent.iterdir())
    else:
        with pytest.raises(runner.UnrecoverableOrphan):
            invoke()


def test_public_result_values_are_read_from_evidence_and_match_frozen_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "mqar.json"
    source = {
        "pooled": {
            "learned_r2": {"correct": 763, "total": 768},
            "dense_donor": {"correct": 768, "total": 768},
            "forced_target_r2": {"correct": 768, "total": 768},
        }
    }
    path.write_bytes(runner.canonical_json_bytes(source))
    monkeypatch.setattr(runner, "validate_real_regular_file", lambda requested: path)
    result = runner._verify_public_result_values()
    frozen = _tracked_payload()["sources"]["prerequisite_values"]
    expected = {
        "learned_route_answers": frozen["monodratic_learned_route_answers"],
        "all_eligible_donor_answers": frozen["monodratic_all_eligible_donor_answers"],
        "target_forced_answers": frozen["monodratic_target_forced_answers"],
    }
    assert result == {"expected": expected, "actual": expected, "pass": True}
    source["pooled"]["learned_r2"]["correct"] = 762
    path.write_bytes(runner.canonical_json_bytes(source))
    with pytest.raises(runner.ContractError):
        runner._verify_public_result_values()


def test_every_exact_schema_key_list_is_nonempty_and_duplicate_free() -> None:
    schemas = _tracked_payload()["artifacts"]["schemas"]
    exact_key_lists = []
    for schema in schemas.values():
        for key, value in schema.items():
            if key == "exact_keys" or key.endswith("_exact_keys"):
                exact_key_lists.append((key, value))
    assert exact_key_lists
    for key, values in exact_key_lists:
        if isinstance(values, dict):
            assert values, key
            for nested_values in values.values():
                assert isinstance(nested_values, list), key
                assert nested_values, key
                assert len(nested_values) == len(set(nested_values)), key
        else:
            assert isinstance(values, list), key
            assert values, key
            assert len(values) == len(set(values)), key


def test_attempt_and_resource_schema_contracts_are_exact() -> None:
    payload = _tracked_payload()
    attempt = payload["artifacts"]["schemas"]["attempt_row"]
    assert attempt["event_enum"] == ["started", "completed"]
    assert attempt["event_sequence"] == "zero_based_strictly_increasing_per_seed"
    assert attempt["started_metrics_value"] == "null"
    assert attempt["event_pairing"] == "adjacent_started_then_completed_where_completed_event_sequence_equals_started_event_sequence_plus_one"
    assert attempt["pair_equal_fields"] == [
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
    ]
    resource = payload["artifacts"]["schemas"]["resource_row"]
    assert resource["phase_enum"] == ["pilot", "claim"]
    assert resource["pilot_baseline_values"] == {"attempted_updates": 0, "token_positions": 0}
    assert resource["pilot_complete_final_values"] == {"attempted_updates": 132, "token_positions": 292864}
    sampling = payload["pilot"]["resource_sampling"]
    assert sampling["ps_argv_prefix"] == ["/bin/ps", "-o", "pid=,ppid=,rss=,time=", "-p"]
    assert sampling["ps_pid_argument"] == "sorted_comma_separated_live_authoritative_pids"
    assert sampling["swap_argv"] == ["/usr/sbin/sysctl", "-n", "vm.swapusage"]
    assert sampling["locale"] == "C"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), "wrong"),
        (("architecture", "authorization", "external_compute_cost_usd"), 1),
        (("architecture", "authorization", "dependency_changes_authorized"), True),
        (("architecture", "authorization", "paid_compute_authorized"), True),
        (("architecture", "authorization", "reciprocal_feature_mixer_enabled"), True),
        (("architecture", "authorization", "trainingnovel_enabled"), True),
        (("architecture", "authorization", "novel_mechanism_claimed"), True),
        (("architecture", "runtime_policy", "device"), "cuda"),
        (("architecture", "runtime_policy", "dtype"), "torch.float16"),
        (("architecture", "runtime_policy", "automatic_mixed_precision"), True),
        (("architecture", "runtime_policy", "compilation"), False),
        (("architecture", "runtime_policy", "dropout"), 0.1),
        (("processes", "launcher", "resume_supported"), True),
        (("processes", "multiprocessing", "start_method"), "fork"),
        (("artifacts", "artifact_path_pattern_count"), 53),
        (("artifacts", "serialization", "crash_atomic_canonical_jsonl", "fault_injection_preflight"), list(reversed(FAULT_IDS))),
        (("artifacts", "schemas", "review_artifact", "source_evidence_directory"), "elsewhere"),
        (("artifacts", "schemas", "review_artifact", "source_evidence_path_pattern"), "elsewhere/{artifact_sha256}.json"),
        (("artifacts", "schemas", "review_artifact", "source_evidence_directory_symlink_allowed"), True),
        (("artifacts", "schemas", "review_artifact", "source_evidence_file_symlink_allowed"), True),
        (("artifacts", "schemas", "review_artifact", "runner_authors_or_mutates_attestation"), True),
        (("artifacts", "schemas", "review_artifact", "source_and_run_bytes_identical"), False),
        (("artifacts", "schemas", "review_artifact", "schema_version_value"), "wrong"),
        (("artifacts", "schemas", "review_artifact", "exact_keys"), ["run_id"]),
        (("artifacts", "schemas", "review_artifact", "target_record_order"), "unsorted"),
        (("artifacts", "schemas", "review_artifact", "target_sha256_formula"), "wrong"),
        (("artifacts", "schemas", "review_artifact", "accepted_reviewer"), "self"),
        (("artifacts", "schemas", "review_artifact", "accepted_findings"), [{}]),
        (("artifacts", "schemas", "review_artifact", "accepted_finding_count"), 1),
        (("artifacts", "schemas", "review_artifact", "exactly_one_matching_attestation_per_scope"), False),
        (("artifacts", "source_manifest_required_review_attestations", "count"), 3),
        (("artifacts", "source_manifest_required_review_attestations", "role_order"), "wrong"),
        (
            ("artifacts", "source_manifest_required_review_attestations", "roles"),
            [
                "review_attestation_base_complete_surface",
                "review_attestation_base_tests",
                "review_attestation_base_implementation",
                "review_attestation_base_preregistration",
            ],
        ),
        (("artifacts", "source_manifest_required_review_attestations", "revision"), "git_commit"),
        (("artifacts", "source_manifest_required_review_attestations", "copied_run_bytes_must_match_source"), False),
        (("abort_rules", "hard_abort_registry", 0, "condition"), "free form"),
        (("gates", "registry_cardinalities", "complete_package"), 123),
    ],
)
def test_payload_validation_fails_closed_on_contract_drift(path, replacement) -> None:
    payload = copy.deepcopy(_tracked_payload())
    _set_path(payload, path, replacement)
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.validate_prereg_payload(payload)


def test_payload_validation_rejects_extra_or_missing_top_level_keys() -> None:
    extra = copy.deepcopy(_tracked_payload())
    extra["extra"] = None
    missing = copy.deepcopy(_tracked_payload())
    missing.pop("losses")
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.validate_prereg_payload(extra)
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.validate_prereg_payload(missing)


def test_payload_loader_rejects_reviewed_digest_drift(tmp_path: Path) -> None:
    payload = _tracked_payload()
    payload["architecture"]["runtime_policy"]["dropout"] = 0.25
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.load_prereg_payload(path)


def test_payload_contains_only_the_five_reviewed_nulls() -> None:
    payload = _tracked_payload()
    null_paths = []

    def visit(value, path):
        if value is None:
            null_paths.append(path)
        elif isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, path + (key,))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, path + (index,))

    visit(payload, ())
    assert null_paths == [
        ("gates", "rung_one_registry", 19, "gate_threshold_count"),
        ("gates", "rung_one_registry", 19, "denominator"),
        ("gates", "rung_one_registry", 20, "denominator"),
        ("gates", "rung_two_registry", 2, "denominator"),
        ("gates", "rung_two_registry", 3, "denominator"),
    ]


def test_entry_environment_requires_exact_values(monkeypatch: pytest.MonkeyPatch) -> None:
    assert dict(runner.REQUIRED_ENV) == REQUIRED_ENV
    assert tuple(runner.REQUIRED_PYTHON) == (3, 9, 6)
    monkeypatch.setattr(runner.sys, "version_info", (3, 9, 6))
    runner.validate_entry_environment(REQUIRED_ENV)
    for key in REQUIRED_ENV:
        missing = dict(REQUIRED_ENV)
        missing.pop(key)
        wrong = dict(REQUIRED_ENV)
        wrong[key] = "wrong"
        with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
            runner.validate_entry_environment(missing)
        with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
            runner.validate_entry_environment(wrong)
    monkeypatch.setattr(runner.sys, "version_info", (3, 9, 7))
    with pytest.raises(runner.InitializationRefusal):
        runner.validate_entry_environment(REQUIRED_ENV)


def test_cli_accepts_only_run_root() -> None:
    parsed = runner.parse_cli(["--run-root", "/tmp/run"])
    assert parsed == "/tmp/run"
    for argv in (
        [],
        ["--run-root"],
        ["--resume", "x"],
        ["--run-root", "/tmp/run", "--seed", "11"],
        ["/tmp/run"],
    ):
        with pytest.raises(runner.InitializationRefusal):
            runner.parse_cli(argv)


def test_run_root_validation_is_exact_and_requires_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = (tmp_path / "modular_sequence_role_cpu").resolve()
    parent.mkdir()
    monkeypatch.setattr(runner, "RESULTS_PARENT", parent)
    valid = parent / "run-01"
    config = runner.validate_run_root(valid)
    assert config.run_root == valid
    assert config.run_id == "run-01"
    invalid = [
        Path("relative-run"),
        parent / "Uppercase",
        parent / ".hidden",
        parent / ("a" * 65),
        parent / "nested" / "run",
        parent / "run" / ".." / "run",
    ]
    for path in invalid:
        with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
            runner.validate_run_root(path)
    valid.mkdir()
    with pytest.raises((AssertionError, FileExistsError, RuntimeError, ValueError)):
        runner.validate_run_root(valid)


def test_main_rejects_entry_environment_before_runtime_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = (tmp_path / "modular_sequence_role_cpu").resolve()
    parent.mkdir()
    monkeypatch.setattr(runner, "RESULTS_PARENT", parent)
    imported = []

    def forbidden_import():
        imported.append(True)
        raise AssertionError("runtime import reached")

    monkeypatch.setattr(runner, "_import_runtime", forbidden_import)
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.main(["--run-root", str(parent / "run")], environ={})
    assert imported == []


@pytest.mark.parametrize("argv", [[], ["--resume", "/tmp/run"], ["--run-root=/tmp/run"]])
def test_main_rejects_cli_before_runtime_import(argv, monkeypatch: pytest.MonkeyPatch) -> None:
    imported = []

    def forbidden_import():
        imported.append(True)
        raise AssertionError("runtime import reached")

    monkeypatch.setattr(runner, "_import_runtime", forbidden_import)
    with pytest.raises(runner.InitializationRefusal):
        runner.main(argv, environ=REQUIRED_ENV)
    assert imported == []


def test_main_rejects_run_root_before_payload_or_runtime_import(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = []
    imported = []

    def forbidden_load():
        loaded.append(True)
        raise AssertionError("payload load reached")

    def forbidden_import():
        imported.append(True)
        raise AssertionError("runtime import reached")

    monkeypatch.setattr(runner, "load_prereg_payload", forbidden_load)
    monkeypatch.setattr(runner, "_import_runtime", forbidden_import)
    with pytest.raises(runner.InitializationRefusal):
        runner.main(["--run-root", "/tmp/outside-reviewed-parent"], environ=REQUIRED_ENV)
    assert loaded == []
    assert imported == []


def test_main_rejects_payload_before_runtime_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = (tmp_path / "modular_sequence_role_cpu").resolve()
    parent.mkdir()
    monkeypatch.setattr(runner, "RESULTS_PARENT", parent)
    imported = []

    def invalid_payload():
        raise runner.ContractError("invalid payload")

    def forbidden_import():
        imported.append(True)
        raise AssertionError("runtime import reached")

    monkeypatch.setattr(runner, "load_prereg_payload", invalid_payload)
    monkeypatch.setattr(runner, "_import_runtime", forbidden_import)
    with pytest.raises(runner.ContractError):
        runner.main(["--run-root", str(parent / "run")], environ=REQUIRED_ENV)
    assert imported == []


def test_runner_top_level_has_no_torch_or_unapproved_imports() -> None:
    imported = set()
    for node in ast.walk(_source_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "torch" not in imported
    forbidden_prefixes = (
        "mlx",
        "trainingnovel",
        "nextlat",
        "v01",
        "src.layers",
        "src.model.todorov",
    )
    assert not {name for name in imported if name.startswith(forbidden_prefixes)}


def test_runner_has_one_main_and_an_executable_terminal_guard() -> None:
    tree = _source_tree()
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    guards = []
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        comparators = node.test.comparators
        if (
            isinstance(left, ast.Name)
            and left.id == "__name__"
            and len(comparators) == 1
            and isinstance(comparators[0], ast.Constant)
            and comparators[0].value == "__main__"
        ):
            guards.append(node)
    assert len(mains) == 1
    assert [argument.arg for argument in mains[0].args.args] == ["argv", "environ"]
    assert len(guards) == 1
    assert tree.body[-1] is guards[0]
    calls = [node for node in ast.walk(guards[0]) if isinstance(node, ast.Call)]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "main" for call in calls)
    assert any(isinstance(call.func, ast.Name) and call.func.id == "SystemExit" for call in calls)


def test_runner_has_no_resume_selector_or_fallback_function() -> None:
    tree = _source_tree()
    option_strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")
    }
    assert option_strings == {"--run-root"}
    names = {
        node.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert not {name for name in names if "resume" in name or "fallback" in name}


def test_payload_excludes_every_unapproved_source_and_graph_surface() -> None:
    payload = _tracked_payload()
    assert payload["sources"]["excluded_dependencies"] == [
        "private_Monodratic_tree",
        "Transformerov_files_outside_frozen_pair",
        "Karkasov_implementation_files",
        "reciprocal_feature_candidate",
        "legacy_Todorov_sequence_modules",
        "nextlat",
    ]
    assert payload["architecture"]["common_host"]["excluded_graph_surfaces"] == [
        "Monodratic_public_host_block",
        "Monodratic_public_feature_mlp",
        "nested_reciprocal_feature_mixer",
        "nextlat",
    ]
    assert payload["sources"]["requirements_txt_unchanged"] is True


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 0 1024 00:01", [{"pid": 1, "ppid": 0, "rss_bytes": 1048576, "cpu_time_us": 1000000}]),
        ("7 1 2 01:02:03.5", [{"pid": 7, "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3723500000}]),
        ("9 1 3 2-03:04:05.25", [{"pid": 9, "ppid": 1, "rss_bytes": 3072, "cpu_time_us": 183845250000}]),
    ],
)
def test_ps_parser_exact_time_and_rss_conversion(text, expected) -> None:
    assert runner.parse_ps_output(text, {row["pid"] for row in expected}) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 0 1",
        "1 0 bad 00:01",
        "1 0 1 00:60",
        "1 0 1 2-04:05",
        "1 0 1 00:01\n1 0 1 00:01",
        "2 0 1 00:01",
    ],
)
def test_ps_parser_fails_closed(text) -> None:
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.parse_ps_output(text, {1})


def test_ps_parser_pid_mismatch_exposes_expected_and_observed_identities() -> None:
    with pytest.raises(runner.ProcessSetMismatch) as caught:
        runner.parse_ps_output("1 0 1 00:01\n3 0 1 00:01", {1, 2})
    assert str(caught.value) == "ps output PID set differs"
    assert caught.value.expected_pids == (1, 2)
    assert caught.value.observed_pids == (1, 3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("total = 1.00M  used = 1.50M  free = 0.00M", 1572864),
        ("vm.swapusage: total = 1G used = 0.5G free = 0.5G", 536870912),
        ("total = 1P used = 0.0000001P free = 1P", 112589991),
    ],
)
def test_swap_parser_uses_binary_units_and_half_up_rounding(text, expected) -> None:
    assert runner.parse_swap_output(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "used = -1M",
        "used = 1B",
        "used = badM",
        "used = 1M used = 2M",
    ],
)
def test_swap_parser_fails_closed(text) -> None:
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.parse_swap_output(text)


def test_resource_row_validator_accepts_exact_sorted_process_contract() -> None:
    row = _resource_row("run", "claim", 4096)
    runner.validate_resource_row(row)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row.__setitem__("run_id", ""),
        lambda row: row.__setitem__("wall_time_utc", 1),
        lambda row: row.__setitem__("expected_pids", [True]),
        lambda row: row["processes"][0].__setitem__("ppid", -1),
        lambda row: row["processes"][0].__setitem__("rss_bytes", -1),
        lambda row: row["processes"][0].__setitem__("cpu_time_us", -1),
        lambda row: row.__setitem__(
            "active_jobs",
            [
                {"worker": "B", "seed": 23, "stage": "joint", "logical_update": 1},
                {"worker": "A", "seed": 11, "stage": "joint", "logical_update": 1},
            ],
        ),
        lambda row: row.__setitem__("active_jobs", [{"worker": "A", "seed": False, "stage": "joint", "logical_update": 1}]),
    ],
)
def test_resource_row_validator_rejects_identity_range_and_order_drift(mutator) -> None:
    row = _resource_row("run", "claim", 4096)
    mutator(row)
    if row["processes"][0]["rss_bytes"] == -1:
        row["expected_pids"] = [101, 102]
        row["processes"].append({"pid": 102, "ppid": 1, "rss_bytes": 2049, "cpu_time_us": 0})
        row["aggregate_rss_bytes"] = 2048
    if row["processes"][0]["cpu_time_us"] == -1:
        row["expected_pids"] = [101, 102]
        row["processes"].append({"pid": 102, "ppid": 1, "rss_bytes": 0, "cpu_time_us": 3001})
        row["aggregate_cpu_time_us"] = 3000
    with pytest.raises((runner.ContractError, TypeError, ValueError)):
        runner.validate_resource_row(row)


@pytest.mark.parametrize("phase", ["pilot", "claim"])
def test_resource_timeline_validator_accepts_exact_sequence_and_clean_cutoff(phase) -> None:
    rows = _resource_timeline(phase)
    runner.validate_resource_timeline(rows, phase, require_clean_final=True)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda rows: rows[1].__setitem__("run_id", "other"),
        lambda rows: rows[1].__setitem__("phase", "pilot"),
        lambda rows: rows[1].__setitem__("sample_id", 2),
        lambda rows: rows[1].__setitem__("monotonic_ns", 100),
        lambda rows: rows[1].__setitem__("wall_time_utc", "2026-07-18T23:59:59Z"),
        lambda rows: rows[2].__setitem__("attempted_updates", 0),
        lambda rows: rows[2].__setitem__("token_positions", 0),
        lambda rows: rows[1].__setitem__("swap_growth_bytes", 21),
        lambda rows: rows[2]["processes"][0].__setitem__("cpu_time_us", 3499),
        lambda rows: rows[2].__setitem__("aggregate_cpu_time_us", 3499),
        lambda rows: rows[0].__setitem__("active_jobs", [{"worker": "A", "seed": 11, "stage": "joint", "logical_update": 1}]),
        lambda rows: rows[0].__setitem__("attempted_updates", 1),
        lambda rows: rows[0].__setitem__("token_positions", 1),
        lambda rows: rows[2].__setitem__("active_jobs", [{"worker": "A", "seed": 11, "stage": "joint", "logical_update": 2}]),
    ],
)
def test_resource_timeline_validator_rejects_sequence_counter_and_cutoff_drift(mutator) -> None:
    rows = _resource_timeline("claim")
    mutator(rows)
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(rows, "claim", require_clean_final=True)


def test_resource_timeline_empty_and_pilot_terminal_contracts() -> None:
    runner.validate_resource_timeline([], "claim", require_clean_final=False)
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline([], "claim", require_clean_final=True)
    rows = _resource_timeline("pilot")
    rows[-1]["attempted_updates"] = 87
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(rows, "pilot", require_clean_final=True)
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(_resource_timeline("claim"), "alias")


def test_resource_timeline_accepts_explicit_mlx_pilot_terminal_counters_without_changing_legacy_default() -> None:
    rows = _resource_timeline("pilot")
    rows[-1]["attempted_updates"] = 132
    rows[-1]["token_positions"] = 292_864
    runner.validate_resource_timeline(rows, "pilot", require_clean_final=True, pilot_final_values=(132, 292_864))
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(rows, "pilot", require_clean_final=True)
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(_resource_timeline("pilot"), "pilot", require_clean_final=True, pilot_final_values=(132, 292_864))
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(rows, "claim", require_clean_final=True, pilot_final_values=(132, 292_864))


def test_resource_timeline_treats_legacy_zeroed_child_telemetry_as_terminal_disappearance() -> None:
    rows = _resource_timeline("pilot")
    rows[1]["expected_pids"] = [101, 102]
    rows[1]["processes"].append({"pid": 102, "ppid": 101, "rss_bytes": 1_271_414_784, "cpu_time_us": 14_250_000})
    rows[1]["aggregate_rss_bytes"] += 1_271_414_784
    rows[1]["aggregate_cpu_time_us"] += 14_250_000
    rows[2]["expected_pids"] = [101, 102]
    rows[2]["processes"].append({"pid": 102, "ppid": 101, "rss_bytes": 0, "cpu_time_us": 0})
    runner.validate_resource_timeline(rows, "pilot", require_clean_final=False)
    with pytest.raises(runner.ContractError, match="clean resource timeline has terminal process disappearance"):
        runner.validate_resource_timeline(rows, "pilot", require_clean_final=True)
    reappeared = copy.deepcopy(rows[-1])
    reappeared["sample_id"] = 3
    reappeared["monotonic_ns"] += 5_000_000_000
    reappeared["wall_time_utc"] = "2026-07-19T00:00:15Z"
    reappeared["processes"][-1] = {"pid": 102, "ppid": 101, "rss_bytes": 1024, "cpu_time_us": 15_000_000}
    reappeared["aggregate_rss_bytes"] += 1024
    reappeared["aggregate_cpu_time_us"] += 15_000_000
    with pytest.raises(runner.ContractError, match="reappears"):
        runner.validate_resource_timeline([*rows, reappeared], "pilot", require_clean_final=False)


@pytest.mark.parametrize(
    ("target_pid", "prior_ppid", "current_ppid"),
    (
        (101, 1, 1),
        (102, 999, 999),
        (102, 101, 999),
    ),
)
def test_resource_timeline_rejects_zeroed_parent_and_unrelated_process_telemetry(target_pid, prior_ppid, current_ppid) -> None:
    rows = _resource_timeline("pilot")
    if target_pid == 101:
        rows[2]["processes"][0] = {"pid": 101, "ppid": current_ppid, "rss_bytes": 0, "cpu_time_us": 0}
        rows[2]["aggregate_rss_bytes"] = 0
        rows[2]["aggregate_cpu_time_us"] = 0
    else:
        rows[1]["expected_pids"] = [101, 102]
        rows[1]["processes"].append({"pid": 102, "ppid": prior_ppid, "rss_bytes": 4096, "cpu_time_us": 7000})
        rows[1]["aggregate_rss_bytes"] += 4096
        rows[1]["aggregate_cpu_time_us"] += 7000
        rows[2]["expected_pids"] = [101, 102]
        rows[2]["processes"].append({"pid": 102, "ppid": current_ppid, "rss_bytes": 0, "cpu_time_us": 0})
    with pytest.raises(runner.ContractError, match="CPU time decreases"):
        runner.validate_resource_timeline(rows, "pilot", require_clean_final=False)


def test_resource_timeline_rejects_committed_samples_less_than_five_seconds_apart() -> None:
    rows = _resource_timeline("claim")
    rows[1]["monotonic_ns"] = rows[0]["monotonic_ns"] + 5_000_000_000 - 1
    with pytest.raises(runner.ContractError):
        runner.validate_resource_timeline(rows, "claim", require_clean_final=True)


def test_claim_resource_strikes_include_sample_zero_and_clean_endpoint() -> None:
    rows = _resource_timeline("claim")
    for row in rows:
        row["processes"][0]["rss_bytes"] = 12 * 1024**3 + 1
        row["aggregate_rss_bytes"] = 12 * 1024**3 + 1
    observe = getattr(runner, "claim_resource_observations", lambda values: ())
    assert observe(rows) == (
        {"reason_code": "resident_memory", "context": {"sample_id": 2}},
    )


def test_claim_resource_observations_reject_swap_growth_on_clean_endpoint() -> None:
    rows = _resource_timeline("claim")
    rows[-1]["swap_used_bytes"] = rows[0]["swap_used_bytes"] + 512 * 1024**2 + 1
    rows[-1]["swap_growth_bytes"] = 512 * 1024**2 + 1
    observe = getattr(runner, "claim_resource_observations", lambda values: ())
    assert observe(rows) == (
        {"reason_code": "swap_growth", "context": {"sample_id": 2}},
    )


def test_claim_resource_observations_name_sample_zero_swap_growth() -> None:
    row = _resource_row("run", "claim", 100)
    row["swap_used_bytes"] = 100 + 512 * 1024**2 + 1
    row["swap_growth_bytes"] = 512 * 1024**2 + 1
    assert runner.claim_resource_observations([row]) == (
        {"reason_code": "swap_growth", "context": {"sample_id": 0}},
    )


def test_pilot_and_claim_schedule_from_persisted_resource_timestamps() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = _source_tree()
    for name in ("run_resource_pilot", "run_claim_workers"):
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
        segment = ast.get_source_segment(source, node)
        assert segment.count("next_resource_sample_monotonic_ns(") >= 2
        assert "time.monotonic() + 5.0" not in segment
        assert "next_sample = now + 5.0" not in segment


def test_resource_sample_builds_exact_row_and_maps_sampler_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "sample_processes",
        lambda pids: [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}],
    )
    monkeypatch.setattr(runner, "sample_swap", lambda: 120)
    row = runner._resource_sample("run", "claim", 1, [], set(), {}, {}, 100, 2, 4096)
    assert tuple(row) == tuple(runner.RESOURCE_ROW_KEYS)
    assert row["expected_pids"] == [runner.os.getpid()]
    assert row["aggregate_rss_bytes"] == 2048
    assert row["aggregate_cpu_time_us"] == 3000
    assert row["swap_growth_bytes"] == 20
    assert row["attempted_updates"] == 2
    assert row["token_positions"] == 4096
    runner.validate_resource_row(row)

    def fail_processes(pids):
        raise runner.ContractError("ps failed")

    monkeypatch.setattr(runner, "sample_processes", fail_processes)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample("run", "claim", 1, [], set(), {}, {}, 100, 0, 0)
    assert caught.value.reason_code == "resource_sampler_failure"
    monkeypatch.setattr(
        runner,
        "sample_processes",
        lambda pids: [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}],
    )

    def fail_swap():
        raise runner.ContractError("swap failed")

    monkeypatch.setattr(runner, "sample_swap", fail_swap)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample("run", "claim", 1, [], set(), {}, {}, 100, 0, 0)
    assert caught.value.reason_code == "resource_sampler_failure"


def test_resource_sample_repolls_post_handshake_status_zero_pid_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        if 202 in pids:
            process.exitcode = 0
            raise runner.ProcessSetMismatch(pids, (runner.os.getpid(),))
        return [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}]

    monkeypatch.setattr(runner, "sample_processes", sample)
    monkeypatch.setattr(runner, "sample_swap", lambda: 100)
    row = runner._resource_sample("run", "claim", 1, [process], {"A"}, {202: "A"}, {}, 100, 0, 0)
    assert calls == [tuple(sorted((runner.os.getpid(), 202))), (runner.os.getpid(),)]
    assert row["expected_pids"] == [runner.os.getpid()]


def test_resource_sample_converges_across_two_staggered_post_handshake_status_zero_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    process_a = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    process_b = SimpleNamespace(pid=203, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        if 202 in pids:
            process_a.exitcode = 0
            raise runner.ProcessSetMismatch(pids, (runner.os.getpid(), 203))
        if 203 in pids:
            process_b.exitcode = 0
            raise runner.ProcessSetMismatch(pids, (runner.os.getpid(),))
        return [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}]

    monkeypatch.setattr(runner, "sample_processes", sample)
    monkeypatch.setattr(runner, "sample_swap", lambda: 100)
    row = runner._resource_sample(
        "run",
        "claim",
        1,
        [process_a, process_b],
        {"A", "B"},
        {202: "A", 203: "B"},
        {},
        100,
        0,
        0,
    )
    assert calls == [
        tuple(sorted((runner.os.getpid(), 202, 203))),
        tuple(sorted((runner.os.getpid(), 203))),
        (runner.os.getpid(),),
    ]
    assert row["expected_pids"] == [runner.os.getpid()]


def test_resource_sample_allows_another_clean_exit_after_the_ps_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    process_a = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    process_b = SimpleNamespace(pid=203, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        if 202 in pids:
            process_a.exitcode = 0
            process_b.exitcode = 0
            raise runner.ProcessSetMismatch(pids, (runner.os.getpid(), 203))
        return [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}]

    monkeypatch.setattr(runner, "sample_processes", sample)
    monkeypatch.setattr(runner, "sample_swap", lambda: 100)
    row = runner._resource_sample(
        "run",
        "claim",
        1,
        [process_a, process_b],
        {"A", "B"},
        {202: "A", 203: "B"},
        {},
        100,
        0,
        0,
    )
    assert calls == [tuple(sorted((runner.os.getpid(), 202, 203))), (runner.os.getpid(),)]
    assert row["expected_pids"] == [runner.os.getpid()]


def test_resource_sample_rejects_missing_live_pid_even_when_another_pid_clean_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    process_a = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    process_b = SimpleNamespace(pid=203, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        process_b.exitcode = 0
        raise runner.ProcessSetMismatch(pids, (runner.os.getpid(),))

    monkeypatch.setattr(runner, "sample_processes", sample)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample(
            "run",
            "claim",
            1,
            [process_a, process_b],
            {"A", "B"},
            {202: "A", 203: "B"},
            {},
            100,
            0,
            0,
        )
    assert caught.value.reason_code == "resource_sampler_failure"
    assert calls == [tuple(sorted((runner.os.getpid(), 202, 203)))]


def test_resource_sample_rejects_unexpected_pid_even_when_expected_pid_clean_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        process.exitcode = 0
        raise runner.ProcessSetMismatch(pids, (runner.os.getpid(), 999))

    monkeypatch.setattr(runner, "sample_processes", sample)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample("run", "claim", 1, [process], {"A"}, {202: "A"}, {}, 100, 0, 0)
    assert caught.value.reason_code == "resource_sampler_failure"
    assert calls == [tuple(sorted((runner.os.getpid(), 202)))]


def test_resource_sample_stable_pid_set_mismatch_remains_sampler_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)
    calls = []

    def sample(pids):
        calls.append(tuple(pids))
        raise runner.ContractError("ps output PID set differs")

    monkeypatch.setattr(runner, "sample_processes", sample)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample("run", "claim", 1, [process], {"A"}, {202: "A"}, {}, 100, 0, 0)
    assert caught.value.reason_code == "resource_sampler_failure"
    assert calls == [tuple(sorted((runner.os.getpid(), 202)))]


@pytest.mark.parametrize(("handshakes", "exitcode"), ((set(), 0), ({"A"}, 7)))
def test_resource_sample_repolls_missing_pid_into_handshake_qualified_worker_exit(monkeypatch: pytest.MonkeyPatch, handshakes, exitcode) -> None:
    process = SimpleNamespace(pid=202, exitcode=None, join=lambda timeout=0: None)

    def sample(pids):
        process.exitcode = exitcode
        raise runner.ProcessSetMismatch(pids, (runner.os.getpid(),))

    monkeypatch.setattr(runner, "sample_processes", sample)
    with pytest.raises(runner.HardAbort) as caught:
        runner._resource_sample("run", "claim", 1, [process], handshakes, {202: "A"}, {}, 100, 0, 0)
    assert caught.value.reason_code == "worker_exit"
    assert caught.value.context == {"worker": "A"}


def test_best_effort_abort_resource_sample_commits_zero_counter_baseline_only_for_sampler_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run-id"
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    monkeypatch.setattr(
        runner,
        "sample_processes",
        lambda pids: [{"pid": pids[0], "ppid": 1, "rss_bytes": 2048, "cpu_time_us": 3000}],
    )
    monkeypatch.setattr(runner, "sample_swap", lambda: 120)
    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())
    writers = {"run/resources.jsonl": writer}
    assert runner.best_effort_abort_resource_sample(root, "claim", writers, 100, accounting, "resource_sampler_failure") is True
    rows = writer.validate_committed_prefix()
    assert len(rows) == 1
    assert rows[0]["sample_id"] == 0
    assert rows[0]["active_jobs"] == []
    assert rows[0]["attempted_updates"] == 0
    assert rows[0]["token_positions"] == 0
    assert rows[0]["swap_used_bytes"] == 120
    writer.close()


def test_best_effort_abort_resource_sample_swallows_secondary_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run-id"
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())
    writers = {"run/resources.jsonl": writer}
    monkeypatch.setattr(runner, "_resource_sample", lambda *args, **kwargs: (_ for _ in ()).throw(runner.HardAbort("worker_exit")))
    assert runner.best_effort_abort_resource_sample(root, "claim", writers, 100, accounting, "resource_sampler_failure") is False
    assert writer.validate_committed_prefix() == ()
    assert runner.best_effort_abort_resource_sample(root, "claim", writers, None, accounting, "resource_sampler_failure") is False
    assert runner.best_effort_abort_resource_sample(root, "claim", writers, 100, accounting, "invented") is False
    writer.close()
    assert runner.best_effort_abort_resource_sample(root, "claim", writers, 100, accounting, "resource_sampler_failure") is False


@pytest.mark.parametrize("reason_code", tuple(reason for reason in runner.HARD_ABORT_REASON_CODES if reason != "resource_sampler_failure"))
def test_best_effort_abort_resource_sample_never_creates_empty_timeline_baseline_for_non_sampler_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason_code: str) -> None:
    root = tmp_path / reason_code
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    calls = []

    def sample(*args):
        calls.append(args)
        return _resource_row(root.name, "claim", 100)

    monkeypatch.setattr(runner, "_resource_sample", sample)
    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())
    assert runner.best_effort_abort_resource_sample(root, "claim", {"run/resources.jsonl": writer}, 100, accounting, reason_code) is False
    assert writer.validate_committed_prefix() == ()
    assert calls == []
    writer.close()


def test_abort_resource_sample_preserves_partially_progressed_pilot_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "pilot-progress"
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "pilot_resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    rows = _resource_timeline("pilot")[:2]
    rows[0]["run_id"] = root.name
    rows[1]["run_id"] = root.name
    rows[1]["attempted_updates"] = 5
    rows[1]["token_positions"] = 10240
    for row in rows:
        assert writer.append(row).acknowledged
    sampled = []

    def sample(*args):
        sampled.append(args)
        row = copy.deepcopy(rows[-1])
        row["sample_id"] = 2
        row["monotonic_ns"] += 1
        row["attempted_updates"] = args[-2]
        row["token_positions"] = args[-1]
        return row

    monkeypatch.setattr(runner, "_resource_sample", sample)
    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())
    assert runner.best_effort_abort_resource_sample(root, "pilot", {"run/pilot_resources.jsonl": writer}, 100, accounting, "nonfinite") is False
    committed = writer.validate_committed_prefix()
    assert len(committed) == 2
    assert committed[-1]["attempted_updates"] == 5
    assert committed[-1]["token_positions"] == 10240
    assert sampled == []
    runner.validate_resource_timeline(committed, "pilot", require_clean_final=False)
    writer.close()


def test_abort_resource_sample_preserves_existing_claim_timeline_after_sampler_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "claim-progress"
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    rows = _resource_timeline("claim")[:2]
    rows[0]["run_id"] = root.name
    rows[1]["run_id"] = root.name
    for row in rows:
        assert writer.append(row).acknowledged
    sampled = []

    def sample(*args):
        sampled.append(args)
        row = copy.deepcopy(rows[-1])
        row["sample_id"] = 2
        row["monotonic_ns"] += 5_000_000_000
        row["active_jobs"] = []
        return row

    monkeypatch.setattr(runner, "_resource_sample", sample)
    accounting = runner.AttemptAccounting(1, 0, 2048, 0, (11,), (), {11: 1}, ("pending",))
    assert runner.best_effort_abort_resource_sample(root, "claim", {"run/resources.jsonl": writer}, 100, accounting, "resource_sampler_failure") is False
    committed = writer.validate_committed_prefix()
    assert committed == tuple(rows)
    assert sampled == []
    runner.validate_resource_timeline(committed, "claim", require_clean_final=False)
    writer.close()


def test_rung_one_generator_matches_reviewed_golden() -> None:
    payload = runner.generate_rung_one_batch(seed=123456, batch_size=2)
    assert tuple(payload) == ("tokens", "targets", "condition", "rule_blocks", "answer_indices", "required_source")
    assert runner.canonical_json_sha256(payload) == "98ff3b54f14306135eafe5a92da7abdf1111cd8690e511188bb5f0e44dcab2a9"


def test_rung_two_generator_matches_reviewed_golden() -> None:
    payload = runner.generate_rung_two_batch(seed=123456, batch_size=2)
    assert tuple(payload) == ("tokens", "targets", "count", "count_positions")
    assert runner.canonical_json_sha256(payload) == "7fff37e20adc2241c217b3ed6dad6ec4d85e818d69a59fa5b8e3f5a48f2b8afe"


def test_random_route_generator_matches_reviewed_golden() -> None:
    payload = runner.generate_random_routes(seed=500011, batch_size=2)
    assert payload == {"routes": [[7, 13], [13, 14]]}
    assert runner.canonical_json_sha256(payload) == "18f568b628517fa8f77d9e6adc17c3c2ead62c46070487d416d6eee25953e54c"


def test_source_exclusion_generator_matches_reviewed_golden() -> None:
    raw = [[3, -1], [14, 7]]
    source = [3, 14]
    payload = runner.generate_source_exclusion_routes(seed=510011, raw=raw, source=source)
    assert payload == {"raw": raw, "routes": [[7, 5], [7, 12]], "source": source}
    assert runner.canonical_json_sha256(payload) == "3992c2df698e8787191991d3d5a3edd1eaadbbef647ea342cd70f10453ff9ebf"


def test_source_exclusion_always_consumes_one_permutation_per_row() -> None:
    raw_a = [[3, -1], [14, 7]]
    raw_b = [[3, 4], [14, 7]]
    source = [3, 14]
    first = runner.generate_source_exclusion_routes(seed=510011, raw=raw_a, source=source)
    second = runner.generate_source_exclusion_routes(seed=510011, raw=raw_b, source=source)
    assert first["routes"][1] == second["routes"][1]
    for row, required in zip(second["routes"], source):
        assert len(row) == 2
        assert len(set(row)) == 2
        assert required not in row
        assert all(0 <= value <= 14 for value in row)


def test_source_exclusion_accepts_exact_acquired_route_tensor_shape() -> None:
    raw = [[[[-1, -1]] for _ in range(128)] for _ in range(2)]
    raw[0][126][0] = [3, -1]
    raw[1][126][0] = [14, 7]
    payload = runner.generate_source_exclusion_routes(seed=510011, raw=raw, source=[3, 14])
    assert payload == {
        "raw": [[3, -1], [14, 7]],
        "routes": [[7, 5], [7, 12]],
        "source": [3, 14],
    }


@pytest.mark.parametrize(
    ("raw", "source"),
    [
        ([[3, -1]], [3, 4]),
        ([[3]], [3]),
        ([[3, 16]], [3]),
        ([[3, -2]], [3]),
        ([[3, -1]], [15]),
    ],
)
def test_source_exclusion_rejects_invalid_shapes_and_ids(raw, source) -> None:
    with pytest.raises((AssertionError, RuntimeError, TypeError, ValueError)):
        runner.generate_source_exclusion_routes(seed=510011, raw=raw, source=source)


def test_generators_are_seed_deterministic_and_separate() -> None:
    first = runner.generate_rung_one_batch(seed=123456, batch_size=2)
    second = runner.generate_rung_one_batch(seed=123456, batch_size=2)
    other = runner.generate_rung_one_batch(seed=123457, batch_size=2)
    assert first == second
    assert first != other
    route_first = runner.generate_random_routes(seed=500011, batch_size=2)
    route_second = runner.generate_random_routes(seed=500011, batch_size=2)
    assert route_first == route_second


def test_public_route_parity_measures_exact_detail_path(runtime_modules) -> None:
    parity = runner.public_route_parity(runtime_modules)
    assert tuple(parity) == (
        "query_feature_max_error",
        "key_feature_max_error",
        "internal_loss_max_error",
        "packed_index_exact",
        "raw_search_exact",
        "finite",
    )
    assert parity == {
        "query_feature_max_error": 0.0,
        "key_feature_max_error": 0.0,
        "internal_loss_max_error": 0.0,
        "packed_index_exact": True,
        "raw_search_exact": True,
        "finite": True,
    }


def test_claim_data_records_batch_local_carry_shuffle_strata(tmp_path: Path, runtime_modules) -> None:
    records = runner.prepare_claim_data(tmp_path, runtime_modules)
    expected = {
        11: (116, 396),
        23: (137, 375),
        37: (122, 390),
        53: (130, 382),
        71: (151, 361),
    }
    payload_strata = _tracked_payload()["gates"]["carry_shuffle_frozen_strata"]
    assert [(row["construction_seed"], row["same_rows"], row["changed_rows"]) for row in payload_strata] == [
        (seed, *expected[seed]) for seed in expected
    ]
    assertions = records["preclaim_assertions"]
    assert assertions["generators"] is True
    assert assertions["golden_hashes"] == records["golden_hashes"]
    assert assertions["seed_separation"] is True
    assert assertions["fault_rehearsal"] == list(runner.FAULT_IDS)
    assert tuple(assertions["first_batch_hashes"]) == ("11", "23", "37", "53", "71", "83")
    assert all(record["distinct"] is True and record["first"] != record["second"] for record in assertions["continuous_stream_advancement"].values())
    assert tuple(assertions["histograms"]) == ("11", "23", "37", "53", "71", "83")
    torch = runtime_modules.torch
    for seed, (same, changed) in expected.items():
        record = records["rung_one"][seed]
        assert (record["same_condition"], record["changed_condition"]) == (same, changed)
        artifact = torch.load(tmp_path / record["evaluation_path"], map_location="cpu", weights_only=True)
        conditions = torch.tensor(artifact["payload"]["condition"], dtype=torch.long).reshape(16, 32)
        observed_same = int((conditions == conditions.roll(1, dims=1)).sum())
        assert observed_same == same
        assert 512 - observed_same == changed


@pytest.mark.parametrize("mutation", ("evaluation_bytes", "evaluation_payload", "random_routes", "train_cross_reference", "registered_golden"))
def test_preclaim_reconstruction_rejects_persisted_data_and_cross_reference_mutations(tmp_path: Path, runtime_modules, mutation: str) -> None:
    payload = _tracked_payload()
    records = runner.prepare_claim_data(tmp_path, runtime_modules, payload)
    first_hashes = records["preclaim_assertions"]["first_batch_hashes"]
    for seed in runner.RUNG_ONE_SEEDS:
        first_hash = first_hashes[str(seed)]
        _write_jsonl_rows(
            tmp_path / "rung1" / str(seed) / "train.jsonl",
            [{"stage": "donor", "logical_update": 1, "batch_sha256": first_hash, "first_batch_sha256": first_hash}],
        )
    rung_two_hash = first_hashes["83"]
    _write_jsonl_rows(
        tmp_path / "rung2" / "83" / "train.jsonl",
        [{"stage": "rung_two", "logical_update": 1, "batch_sha256": rung_two_hash, "first_batch_sha256": rung_two_hash}],
    )
    reconstructed = runner.validate_preclaim_reconstruction(tmp_path, payload, runtime_modules)
    assert reconstructed["first_batch_hashes"] == first_hashes
    torch = runtime_modules.torch
    if mutation == "evaluation_bytes":
        path = tmp_path / records["rung_one"][11]["evaluation_path"]
        with path.open("ab") as handle:
            handle.write(b"mutated")
    elif mutation == "evaluation_payload":
        path = tmp_path / records["rung_one"][11]["evaluation_path"]
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        artifact["payload"]["targets"][0] += 1
        torch.save(artifact, path)
    elif mutation == "random_routes":
        path = tmp_path / records["rung_one"][11]["random_path"]
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        artifact["routes"][0, 126, 0, 0] = 99
        torch.save(artifact, path)
    elif mutation == "train_cross_reference":
        path = tmp_path / "rung1" / "11" / "train.jsonl"
        row = json.loads(path.read_text(encoding="utf-8"))
        row["first_batch_sha256"] = "0" * 64
        _write_jsonl_rows(path, [row])
    else:
        payload = copy.deepcopy(payload)
        payload["generators"]["rung_one"]["golden"]["payload_sha256"] = "0" * 64
    with pytest.raises((runner.ContractError, runner.HardAbort)):
        runner.validate_preclaim_reconstruction(tmp_path, payload, runtime_modules)


def test_torch_evidence_path_registry_matches_every_clean_closure_artifact() -> None:
    payload = _tracked_payload()
    closures = payload["artifacts"]["artifact_closures"]
    expected = {path for path in closures["fixed_data_artifacts"] if path.endswith(".pt")}
    for seed in closures["rung_one_construction_seeds"]:
        expected.update(
            f"rung1/{seed}/{suffix}"
            for suffix in closures["rung_one_clean_file_suffixes_per_seed"]
            if suffix.endswith(".pt")
        )
    expected.update(
        f"rung2/83/{suffix}"
        for suffix in closures["rung_two_clean_file_suffixes"]
        if suffix.endswith(".pt")
    )
    assert len(TORCH_EVIDENCE_PATHS) == 42
    assert len(set(TORCH_EVIDENCE_PATHS)) == 42
    assert set(TORCH_EVIDENCE_PATHS) == expected


@pytest.mark.parametrize("relative", TORCH_EVIDENCE_PATHS)
def test_every_torch_evidence_artifact_rejects_neutral_trailing_bytes(tmp_path: Path, runtime_modules, relative: str) -> None:
    torch = runtime_modules.torch
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    runner._save_torch_artifact(path, {"path": relative, "value": torch.arange(4)}, torch)
    with path.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(runner.HardAbort) as caught:
        runner._load_claim_torch_artifact(path, torch, relative)
    assert caught.value.reason_code == "endpoint_inconsistency"
    assert caught.value.context["surface"] == f"{relative}.canonical_bytes"


def test_torch_artifact_boundary_race_preserves_foreign_temporary_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_modules,
) -> None:
    path = tmp_path / "checkpoint.pt"
    temporary = tmp_path / f".checkpoint.pt.writing.{runner.os.getpid()}"
    foreign = b"foreign-torch-temporary\n"
    real_open = Path.open

    def open_path(target, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if target == temporary and mode == "xb" and not temporary.exists():
            temporary.write_bytes(foreign)
        return real_open(target, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(FileExistsError):
        runner._save_torch_artifact(path, {"value": runtime_modules.torch.arange(4)}, runtime_modules.torch)
    assert temporary.read_bytes() == foreign
    assert not path.exists()


@pytest.mark.parametrize("surface", ["output", "state", "gate"])
def test_finite_tree_rejects_nonfinite_output_state_and_gate(runtime_modules, surface) -> None:
    torch = runtime_modules.torch
    tree = {
        "gate": torch.tensor([0.5]),
        "output": torch.tensor([1.0]),
        "state": [torch.tensor([2.0])],
    }
    if surface == "state":
        tree[surface][0] = torch.tensor([float("nan")])
    else:
        tree[surface] = torch.tensor([float("inf")])
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_finite_tree(torch, tree, {"seed": 11}, "forward")
    assert caught.value.reason_code == "nonfinite"
    assert caught.value.context["seed"] == 11
    assert caught.value.context["surface"] == f"forward.{surface}" + (".0" if surface == "state" else "")


def test_model_optimizer_finiteness_covers_parameter_gradient_buffer_and_state(runtime_modules) -> None:
    torch = runtime_modules.torch
    model = torch.nn.Linear(2, 2)
    model.register_buffer("recurrent_state", torch.ones(1))
    optimizer = torch.optim.AdamW(model.parameters())
    runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    assert caught.value.reason_code == "nonfinite"
    assert caught.value.context["surface"] == "parameter.weight"
    with torch.no_grad():
        model.weight[0, 0] = 0.0
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    assert caught.value.context["surface"] == "gradient.weight"
    model.weight.grad = None
    model.recurrent_state.fill_(float("nan"))
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    assert caught.value.context["surface"] == "buffer.recurrent_state"
    model.recurrent_state.zero_()
    optimizer.state[model.weight]["exp_avg"] = torch.tensor(float("nan"))
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    assert caught.value.context["surface"] == "optimizer.state.exp_avg"
    optimizer.state.clear()
    optimizer.state[torch.nn.Parameter(torch.ones(1))]["step"] = torch.tensor(0.0)
    with pytest.raises(runner.HardAbort) as caught:
        runner._assert_model_and_optimizer_finite(torch, model, optimizer, {"stage": "joint"})
    assert caught.value.reason_code == "artifact_inconsistency"
    assert caught.value.context["surface"] == "optimizer.foreign_parameter"


def test_shared_gradient_clip_maps_runtime_nonfinite_to_pilot_and_claim_protocol(runtime_modules, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = runtime_modules.torch
    parameter = torch.nn.Parameter(torch.ones(1))
    model = SimpleNamespace(
        named_parameters=lambda: (("weight", parameter),),
        named_buffers=lambda: (),
        parameters=lambda: (parameter,),
    )
    optimizer = SimpleNamespace(state={}, param_groups=[])

    def fail_clip(*args, **kwargs):
        raise RuntimeError("non-finite norm")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fail_clip)
    with pytest.raises(runner.HardAbort) as caught:
        runner._clip_gradient_norm_finite(torch, model, optimizer, {"stage": "pilot_or_claim"})
    error = caught.value
    assert error.reason_code == "nonfinite"
    assert error.context == {"stage": "pilot_or_claim", "surface": "gradient_norm"}
    for worker in ("A", "B"):
        child = runner._child_failure_message(error, worker)
        assert child["reason_code"] == "nonfinite"
        assert child["context"]["worker"] == worker
        parent = runner.parent_worker_failure_observation(error, worker, False)
        assert parent["reason_code"] == "nonfinite"
        assert parent["context"]["worker"] == worker


def test_route_observation_accounts_real_tensors_and_enforces_capacity(runtime_modules) -> None:
    torch = runtime_modules.torch
    telemetry = {
        "overflow_count": 0,
        "max_bucket_load": 16,
        "workspace_bytes": 7168,
        "posting_slots_materialized": 256,
        "search_rows": 0,
        "bypass_rows": 1,
        "addresses_probed": 0,
        "postings_read": 0,
        "candidate_blocks": 0,
        "raw_remote": torch.zeros((1, 1, 1, 1), dtype=torch.long),
        "block_features": torch.zeros((1, 3, 4), dtype=torch.float32),
        "block_addresses": torch.zeros(2, dtype=torch.int64),
        "postings": torch.zeros(4, dtype=torch.int32),
    }
    output = SimpleNamespace(blocks=[SimpleNamespace(block_index=4, mixer_output=SimpleNamespace(telemetry=telemetry))])
    expected_index_count = sum(telemetry[name].numel() for name in ("block_features", "block_addresses", "postings"))
    expected_index_bytes = sum(telemetry[name].numel() * telemetry[name].element_size() for name in ("block_features", "block_addresses", "postings"))
    assert runner._route_observation(output, 1, {"seed": 11}) == (0, 16, expected_index_count, expected_index_bytes, 1536, 7168)
    telemetry["max_bucket_load"] = 17
    with pytest.raises(runner.HardAbort) as caught:
        runner._route_observation(output, 1, {"seed": 11})
    assert caught.value.reason_code == "route_overflow"
    telemetry["max_bucket_load"] = 16
    telemetry["overflow_count"] = 1
    with pytest.raises(runner.HardAbort) as caught:
        runner._route_observation(output, 1, {"seed": 11})
    assert caught.value.reason_code == "route_overflow"
    telemetry["overflow_count"] = 0
    telemetry["max_bucket_load"] = 64
    assert runner._route_observation(output, 2, {"seed": 83})[:2] == (0, 64)
    telemetry["max_bucket_load"] = 65
    with pytest.raises(runner.HardAbort) as caught:
        runner._route_observation(output, 2, {"seed": 83})
    assert caught.value.reason_code == "route_overflow"
    telemetry["max_bucket_load"] = 16
    telemetry["workspace_bytes"] = -1
    with pytest.raises(runner.HardAbort) as caught:
        runner._route_observation(output, 1, {"seed": 11})
    assert caught.value.reason_code == "artifact_inconsistency"
    with pytest.raises(runner.ContractError):
        runner._route_observation(output, 3, {})


def test_intervention_l2_uses_one_float64_root_over_all_elements(runtime_modules) -> None:
    torch = runtime_modules.torch
    first = runner._new_l2_accumulator()
    second = runner._new_l2_accumulator()
    runner._l2_accumulate(
        first,
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        torch.tensor([5.0, 12.0], dtype=torch.float32),
        torch.tensor([8.0, 15.0], dtype=torch.float32),
        torch,
    )
    runner._l2_accumulate(
        second,
        torch.tensor([12.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch,
    )
    assert runner._l2_values(first) == [5.0, 13.0, 17.0]
    assert runner._l2_values(runner._merge_l2_accumulators([first, second])) == [13.0, 13.0, 17.0]
    with pytest.raises(runner.HardAbort) as caught:
        runner._l2_accumulate(first, torch.tensor([float("nan")]), torch.zeros(1), torch.zeros(1), torch)
    assert caught.value.reason_code == "nonfinite"
    assert caught.value.context["surface"] == "intervention_delta"


def test_state_statistics_preserve_population_reduction_and_nonfinite_abort(runtime_modules) -> None:
    torch = runtime_modules.torch
    left = runner._new_stat_accumulator()
    right = runner._new_stat_accumulator()
    runner._stat_accumulate(left, torch.tensor([1.0, 3.0]), torch)
    runner._stat_accumulate(right, torch.tensor([5.0, 7.0]), torch)
    values = runner._stat_values(runner._merge_stat_accumulators([left, right]))
    assert values == {
        "count": 4,
        "mean": 4.0,
        "population_std": 5.0**0.5,
        "min": 1.0,
        "max": 7.0,
        "nonfinite_count": 0,
    }
    with pytest.raises(runner.HardAbort) as caught:
        runner._stat_accumulate(left, torch.tensor([float("inf")]), torch)
    assert caught.value.reason_code == "nonfinite"
    assert caught.value.context["surface"] == "state_telemetry"


def test_state_record_validator_distinguishes_before_after_and_whole_condition_aggregate() -> None:
    digest = "a" * 64
    checkpoint_by_condition = {condition: (runner.RUNG_ONE_MODEL_BY_CONDITION[condition], digest) for condition in runner.RUNG_ONE_CONDITIONS}
    records = _state_records(1, checkpoint_by_condition)
    runner.validate_state_records(records, 1, checkpoint_by_condition)
    mutations = []
    changed = copy.deepcopy(records)
    changed.append(copy.deepcopy(changed[0]))
    mutations.append(changed)
    changed = copy.deepcopy(records)
    changed.pop()
    mutations.append(changed)
    changed = copy.deepcopy(records)
    changed[0]["count"] = 0
    mutations.append(changed)
    changed = copy.deepcopy(records)
    changed[0]["mean"] = float("nan")
    mutations.append(changed)
    changed = copy.deepcopy(records)
    changed[0]["min"] = 4.0
    mutations.append(changed)
    changed = copy.deepcopy(records)
    changed[0]["nonfinite_count"] = 1
    mutations.append(changed)
    aggregate_index = next(index for index, record in enumerate(records) if record["block"] is None)
    changed = copy.deepcopy(records)
    changed[aggregate_index]["mean"] += 0.1
    mutations.append(changed)
    for changed in mutations:
        with pytest.raises(runner.ContractError):
            runner.validate_state_records(changed, 1, checkpoint_by_condition)


def test_state_record_validator_rejects_empty_telemetry_population() -> None:
    with pytest.raises(runner.ContractError):
        runner.validate_state_records([])


def test_intervention_record_validator_enforces_matched_intact_knockout_and_aggregate_semantics() -> None:
    records = _intervention_records()
    checkpoint_by_condition = _intervention_checkpoint_by_condition(1)
    runner.validate_intervention_records(records, 1, checkpoint_by_condition)
    duplicate = copy.deepcopy(records)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(duplicate, 1, checkpoint_by_condition)
    wrong_aggregate = copy.deepcopy(records)
    next(record for record in wrong_aggregate if record["block"] is None and record["condition"] == "carry_reset")["pre_delta_l2"] += 1.0
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(wrong_aggregate, 1, checkpoint_by_condition)
    wrong_pre = copy.deepcopy(records)
    next(record for record in wrong_pre if record["block"] == 1 and record["condition"] == "carry_shuffle")["pre_delta_l2"] += 1.0
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(wrong_pre, 1, checkpoint_by_condition)
    wrong_knockout = copy.deepcopy(records)
    next(record for record in wrong_knockout if record["block"] == 1 and record["condition"] == "recurrent_knockout")["exposed_delta_l2"] = 1.0
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(wrong_knockout, 1, checkpoint_by_condition)
    wrong_route = copy.deepcopy(records)
    next(record for record in wrong_route if record["block"] == 4 and record["condition"] == "block4_routed_knockout")["exposed_delta_l2"] = 1.0
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(wrong_route, 1, checkpoint_by_condition)


@pytest.mark.parametrize("rung", (1, 2))
def test_intervention_semantics_reject_every_self_consistent_condition_block_mutation(rung: int) -> None:
    baseline = _intervention_records() if rung == 1 else _rung_two_intervention_records()
    checkpoint_by_condition = _intervention_checkpoint_by_condition(rung)
    runner.validate_intervention_records(baseline, rung, checkpoint_by_condition)
    blocks = tuple(range(8)) if rung == 1 else tuple(runner.RECURRENT_BLOCKS)
    conditions = runner.RUNG_ONE_CONDITIONS if rung == 1 else runner.RUNG_TWO_CONDITIONS
    knockout_pairs = {
        *(('recurrent_knockout', block) for block in runner.RECURRENT_BLOCKS),
        *((('block4_routed_knockout', 4),) if rung == 1 else ()),
    }
    for condition in conditions:
        for block in blocks:
            changed = copy.deepcopy(baseline)
            record = next(item for item in changed if item["condition"] == condition and item["block"] == block)
            record["pre_delta_l2"] += 0.125
            _refresh_intervention_aggregate(changed, condition)
            with pytest.raises(runner.ContractError):
                runner.validate_intervention_records(changed, rung, checkpoint_by_condition)
            if (condition, block) not in knockout_pairs:
                changed = copy.deepcopy(baseline)
                record = next(item for item in changed if item["condition"] == condition and item["block"] == block)
                record["exposed_delta_l2"] += 0.125
                _refresh_intervention_aggregate(changed, condition)
                with pytest.raises(runner.ContractError):
                    runner.validate_intervention_records(changed, rung, checkpoint_by_condition)
            else:
                changed = copy.deepcopy(baseline)
                record = next(item for item in changed if item["condition"] == condition and item["block"] == block)
                record["exposed_delta_l2"] = 0.125
                _refresh_intervention_aggregate(changed, condition)
                with pytest.raises(runner.ContractError):
                    runner.validate_intervention_records(changed, rung, checkpoint_by_condition)
                changed = copy.deepcopy(baseline)
                record = next(item for item in changed if item["condition"] == condition and item["block"] == block)
                record["post_delta_l2"] = 0.0
                _refresh_intervention_aggregate(changed, condition)
                with pytest.raises(runner.ContractError):
                    runner.validate_intervention_records(changed, rung, checkpoint_by_condition)


@pytest.mark.parametrize("mutation", ("missing", "invented"))
def test_intervention_record_validator_requires_exact_preregistered_condition_closure(mutation: str) -> None:
    records = _intervention_records()
    checkpoint_by_condition = _intervention_checkpoint_by_condition(1)
    if mutation == "missing":
        records = [record for record in records if record["condition"] != "block4_routed_knockout"]
    else:
        invented = {
            "model": "selected",
            "checkpoint_sha256": "a" * 64,
            "baseline_model": "selected",
            "baseline_checkpoint_sha256": "a" * 64,
            "baseline_condition": "intact",
            "block": 4,
            "condition": "invented_condition",
            "pre_delta_l2": 1.0,
            "post_delta_l2": 1.0,
            "exposed_delta_l2": 1.0,
        }
        records.extend((invented, {**invented, "block": None}))
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(records, 1, checkpoint_by_condition)


def test_intervention_records_bind_condition_checkpoint_and_selected_intact_baseline() -> None:
    checkpoint_by_condition = _intervention_checkpoint_by_condition(1)
    records = _intervention_records()
    runner.validate_intervention_records(records, 1, checkpoint_by_condition)
    for condition in ("block4_local_only", "all_eligible_clone"):
        record = next(item for item in records if item["condition"] == condition and item["block"] == 4)
        intact = next(item for item in records if item["condition"] == "intact" and item["block"] == 4)
        assert record["pre_delta_l2"] == intact["post_delta_l2"]
        assert record["pre_delta_l2"] != record["post_delta_l2"]
        assert record["baseline_model"] == "selected"
        assert record["baseline_checkpoint_sha256"] == "a" * 64
        assert record["baseline_condition"] == "intact"
    coherent_local = copy.deepcopy(records)
    for record in coherent_local:
        if record["condition"] == "block4_local_only" and record["block"] is not None:
            record["pre_delta_l2"] = record["post_delta_l2"]
    _refresh_intervention_aggregate(coherent_local, "block4_local_only")
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(coherent_local, 1, checkpoint_by_condition)
    wrong_identity = copy.deepcopy(records)
    for record in wrong_identity:
        if record["condition"] == "all_eligible_donor":
            record["model"] = "selected"
            record["checkpoint_sha256"] = "a" * 64
            record["baseline_model"] = "selected"
            record["baseline_checkpoint_sha256"] = "a" * 64
            record["baseline_condition"] = "intact"
    with pytest.raises(runner.ContractError):
        runner.validate_intervention_records(wrong_identity, 1, checkpoint_by_condition)


def test_optimizer_membership_audit_is_exact_and_codebooks_are_truthful(monkeypatch: pytest.MonkeyPatch, runtime_modules) -> None:
    torch = runtime_modules.torch
    model_module = runtime_modules.model_module
    model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
    optimizer, group_records, evidence = runner._make_optimizer(model, "router_only", runtime_modules)
    assert optimizer.state == {}
    assert group_records == sorted(group_records, key=lambda record: record["parameter_group"])
    parameters = dict(model.named_parameters())
    assert set(evidence) == set(parameters)
    optimizer_members = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert len(optimizer_members) == sum(len(group["params"]) for group in optimizer.param_groups)
    assert optimizer_members == {id(parameters[name]) for name, record in evidence.items() if record["requires_grad"]}
    assert all(record["weight_decay"] == (0.01 if record["category"] == "matrix" else 0.0) for record in evidence.values() if record["requires_grad"])
    assert evidence["blocks.4.mix.source_mixer.attention.router.codebooks"] == {
        "category": "codebook",
        "requires_grad": True,
        "parameter_group": "block_4_router_zero_decay",
        "peak_lr": 0.003,
        "weight_decay": 0.0,
    }
    assert evidence["blocks.0.mix.source_mixer.attention.router.codebooks"]["requires_grad"] is False

    def tensor_digest(tensor):
        data = bytes(tensor.detach().contiguous().cpu().view(torch.uint8).reshape(-1).tolist())
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(runner, "_tensor_sha256", tensor_digest)
    audit = runner._initialize_audit(model, "router_only", runtime_modules, evidence)
    records = {record["name"]: record for record in runner._finalize_audit(audit, model, "selected", torch)}
    trained_codebook = records["blocks.4.mix.source_mixer.attention.router.codebooks"]
    frozen_codebook = records["blocks.0.mix.source_mixer.attention.router.codebooks"]
    assert trained_codebook["classification"] == "trainable_but_no_gradient"
    assert trained_codebook["start_sha256"] == trained_codebook["end_sha256"]
    assert trained_codebook["update_l2"] == 0.0
    assert frozen_codebook["classification"] == "frozen_by_design"
    assert frozen_codebook["start_sha256"] == frozen_codebook["end_sha256"]
    assert frozen_codebook["update_l2"] is None


def test_gradient_audit_validator_enforces_counters_membership_freeze_and_classification() -> None:
    learned = _gradient_audit_record()
    frozen = _gradient_audit_record(name="frozen", trainable=False)
    runner.validate_gradient_audit([learned, frozen], {"donor": 2})
    mutations = []
    changed = copy.deepcopy([learned, frozen])
    changed[0]["grad_nonzero_steps"] = 1
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed[0]["grad_nonfinite_steps"] = 1
    changed[0]["grad_nonzero_steps"] = 1
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed[0]["optimizer_member"] = False
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed[0]["parameter_group"] = None
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed[1]["end_sha256"] = "b" * 64
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed[0]["classification"] = "learned_with_evidence"
    changed[0]["grad_nonzero_steps"] = 0
    changed[0]["grad_zero_steps"] = 2
    mutations.append(changed)
    changed = copy.deepcopy([learned, frozen])
    changed.append(copy.deepcopy(learned))
    mutations.append(changed)
    for records in mutations:
        with pytest.raises(runner.ContractError):
            runner.validate_gradient_audit(records, {"donor": 2})


def test_gradient_audit_requires_truthful_direct_router_projection_and_codebook_nonlearning() -> None:
    prefix = "blocks.4.mix.source_mixer.attention.router."
    query = _gradient_audit_record(prefix + "query_projection.weight", "router_only", True, "learned_with_evidence", 1)
    key = _gradient_audit_record(prefix + "key_projection.weight", "router_only", True, "learned_with_evidence", 1)
    codebook = _gradient_audit_record(prefix + "codebooks", "router_only", True, "trainable_but_no_gradient", 1)
    codebook.update(
        {
            "category": "codebook",
            "weight_decay": 0.0,
            "grad_none_steps": 1,
            "grad_zero_steps": 0,
            "grad_nonzero_steps": 0,
            "update_zero_steps": 1,
            "update_nonzero_steps": 0,
            "first_nonzero_step": None,
            "start_sha256": "a" * 64,
            "end_sha256": "a" * 64,
            "update_l2": 0.0,
            "update_max_abs": 0.0,
        }
    )
    runner.validate_gradient_audit([query, key, codebook], {"router_only": 1})
    false_claim = copy.deepcopy([query, key, codebook])
    false_claim[2].update(
        {
            "classification": "learned_with_evidence",
            "grad_none_steps": 0,
            "grad_nonzero_steps": 1,
            "update_zero_steps": 0,
            "update_nonzero_steps": 1,
            "first_nonzero_step": 1,
            "end_sha256": "b" * 64,
            "update_l2": 1.0,
            "update_max_abs": 1.0,
        }
    )
    with pytest.raises(runner.ContractError):
        runner.validate_gradient_audit(false_claim, {"router_only": 1})


def test_real_public_router_steps_prove_projection_gradients_and_truthful_codebook_accounting(runtime_modules) -> None:
    all_records = []
    model = None
    for index, stage in enumerate(("router_only", "joint")):
        model, records = _public_router_audit_step(stage, 8800 + index, runtime_modules)
        all_records.extend(records)
    runner.validate_gradient_audit(all_records, {"router_only": 1, "joint": 1})
    by_identity = {(record["stage"], record["name"]): record for record in all_records}
    prefix = "blocks.4.mix.source_mixer.attention.router."
    for stage in ("router_only", "joint"):
        for suffix in ("query_projection.weight", "key_projection.weight"):
            record = by_identity[(stage, prefix + suffix)]
            assert record["grad_nonzero_steps"] == 1
            assert record["update_nonzero_steps"] == 1
            assert record["classification"] == "learned_with_evidence"
        codebook = by_identity[(stage, prefix + "codebooks")]
        assert codebook["requires_grad"] is True
        assert codebook["weight_decay"] == 0.0
        assert codebook["grad_nonzero_steps"] == 0
        assert codebook["update_nonzero_steps"] == 0
        assert codebook["start_sha256"] == codebook["end_sha256"]
        assert codebook["classification"] == "trainable_but_no_gradient"
    runtime_accounting = {
        "dynamic_recurrent_state_count": 0,
        "dynamic_recurrent_state_bytes": 0,
        "route_index_storage_count": 0,
        "route_index_storage_bytes": 0,
        "routing_workspace_count": 0,
        "routing_workspace_bytes": 0,
        "optimizer_state_count": 0,
        "optimizer_state_bytes": 0,
    }
    entries = runner._accounting_entries(model, all_records, runtime_accounting)
    serialized = json.loads(runner.canonical_json_bytes({"entries": entries}).decode("utf-8"))["entries"]
    parameters = dict(model.named_parameters())
    classifications = {
        name: {record["classification"] for record in all_records if record["name"] == name}
        for name in parameters
    }
    learned_names = {name for name, values in classifications.items() if "learned_with_evidence" in values}
    serialized_names = {
        name
        for name, values in classifications.items()
        if name not in learned_names and values & {"trainable_but_no_gradient", "updated_only_by_decay"}
    }
    codebook_name = prefix + "codebooks"
    assert codebook_name not in learned_names
    assert codebook_name in serialized_names
    active_entry = next(entry for entry in serialized if entry["category"] == "active_learned_parameter")
    serialized_entry = next(entry for entry in serialized if entry["category"] == "serialized_without_gradient")
    assert active_entry["count"] == sum(parameters[name].numel() for name in learned_names)
    assert serialized_entry["count"] == sum(parameters[name].numel() for name in serialized_names)
    assert serialized_entry["bytes"] == sum(parameters[name].numel() * parameters[name].element_size() for name in serialized_names)


def test_gradient_audit_rejects_empty_evidence_for_nonempty_stage() -> None:
    with pytest.raises(runner.ContractError):
        runner.validate_gradient_audit([], {"donor": 2})


def test_runtime_accounting_merge_uses_peak_live_surfaces() -> None:
    first = {
        "dynamic_recurrent_state_count": 10,
        "dynamic_recurrent_state_bytes": 40,
        "route_index_storage_count": 20,
        "route_index_storage_bytes": 80,
        "routing_workspace_count": 30,
        "routing_workspace_bytes": 120,
        "optimizer_state_count": 5,
        "optimizer_state_bytes": 20,
    }
    second = {
        "dynamic_recurrent_state_count": 8,
        "dynamic_recurrent_state_bytes": 32,
        "route_index_storage_count": 25,
        "route_index_storage_bytes": 100,
        "routing_workspace_count": 15,
        "routing_workspace_bytes": 60,
        "optimizer_state_count": 7,
        "optimizer_state_bytes": 28,
    }
    assert runner._merge_runtime_accounting([first, second]) == {
        "dynamic_recurrent_state_count": 10,
        "dynamic_recurrent_state_bytes": 40,
        "route_index_storage_count": 25,
        "route_index_storage_bytes": 100,
        "routing_workspace_count": 30,
        "routing_workspace_bytes": 120,
        "optimizer_state_count": 7,
        "optimizer_state_bytes": 28,
    }
    for changed in ({**first, "dynamic_recurrent_state_count": -1}, {**first, "optimizer_state_bytes": -1}):
        with pytest.raises(runner.ContractError):
            runner._merge_runtime_accounting([changed])


@pytest.mark.parametrize("mutator", [lambda row: row.pop("routing_workspace_count"), lambda row: row.__setitem__("extra", 0), lambda row: row.__setitem__("route_index_storage_count", True)])
def test_runtime_accounting_merge_rejects_missing_extra_and_boolean_values(mutator) -> None:
    row = {
        "dynamic_recurrent_state_count": 10,
        "dynamic_recurrent_state_bytes": 40,
        "route_index_storage_count": 20,
        "route_index_storage_bytes": 80,
        "routing_workspace_count": 30,
        "routing_workspace_bytes": 120,
        "optimizer_state_count": 5,
        "optimizer_state_bytes": 20,
    }
    mutator(row)
    with pytest.raises(runner.ContractError):
        runner._merge_runtime_accounting([row])


def test_accounting_entries_derive_parameter_buffer_and_runtime_categories(runtime_modules) -> None:
    torch = runtime_modules.torch
    model = torch.nn.Linear(3, 2)
    model.register_buffer("tracked", torch.zeros(5, dtype=torch.float32))
    audits = [
        {"name": "weight", "classification": "learned_with_evidence"},
        {"name": "bias", "classification": "trainable_but_no_gradient"},
    ]
    runtime_accounting = {
        "dynamic_recurrent_state_count": 10,
        "dynamic_recurrent_state_bytes": 40,
        "route_index_storage_count": 20,
        "route_index_storage_bytes": 80,
        "routing_workspace_count": 30,
        "routing_workspace_bytes": 120,
        "optimizer_state_count": 5,
        "optimizer_state_bytes": 20,
    }
    entries = runner._accounting_entries(model, audits, runtime_accounting)
    assert [record["name"] for record in entries] == sorted(record["name"] for record in entries)
    by_name = {record["name"]: record for record in entries}
    assert by_name["active_learned_parameter"]["count"] == model.weight.numel()
    assert by_name["serialized_without_gradient"]["count"] == model.bias.numel()
    assert by_name["inactive_parameter"]["count"] == 0
    assert by_name["registered_buffer"] == {"category": "registered_buffer", "name": "registered_buffer", "count": 5, "bytes": 20}
    for category in ("dynamic_recurrent_state", "route_index_storage", "routing_workspace", "optimizer_state"):
        assert by_name[category]["count"] == runtime_accounting[f"{category}_count"]
        assert by_name[category]["bytes"] == runtime_accounting[f"{category}_bytes"]
    with pytest.raises(runner.ContractError):
        runner._accounting_entries(model, [{"name": "absent", "classification": "learned_with_evidence"}], runtime_accounting)


def test_accounting_evidence_reconstructs_every_category_from_checkpoint_gradient_and_geometry(runtime_modules) -> None:
    torch = runtime_modules.torch
    state = {
        "inactive": torch.zeros(4, dtype=torch.float32),
        "learned": torch.zeros(2, dtype=torch.float32),
        "serialized": torch.zeros(3, dtype=torch.float32),
    }
    optimizer_state = {
        "state": {
            0: {
                "exp_avg": torch.zeros(2, dtype=torch.float32),
                "exp_avg_sq": torch.zeros(2, dtype=torch.float32),
                "step": torch.tensor(1, dtype=torch.int64),
            }
        }
    }
    endpoints = {"rung_two": {"checkpoint": {"model_state_dict": state, "optimizer_state_dict": optimizer_state}}}
    gradients = [
        {"model": "rung_two", "name": "inactive", "classification": "frozen_by_design"},
        {"model": "rung_two", "name": "learned", "classification": "learned_with_evidence"},
        {"model": "rung_two", "name": "serialized", "classification": "trainable_but_no_gradient"},
    ]
    dynamic_count = len(runner.RECURRENT_BLOCKS) * 32 * 4 * 16 * 16
    route_count, route_bytes = runner._route_index_storage(32, 512, 2)
    optimizer_count, optimizer_bytes = runner._tensor_tree_storage(optimizer_state, torch)
    quantities = {
        "active_learned_parameter": (2, 8),
        "serialized_without_gradient": (3, 12),
        "inactive_parameter": (4, 16),
        "registered_buffer": (0, 0),
        "dynamic_recurrent_state": (dynamic_count, dynamic_count * 4),
        "route_index_storage": (route_count, route_bytes),
        "routing_workspace": (0, 0),
        "optimizer_state": (optimizer_count, optimizer_bytes),
    }
    entries = sorted(
        ({"category": category, "name": category, "count": values[0], "bytes": values[1]} for category, values in quantities.items()),
        key=lambda record: record["name"],
    )
    models = [{"model": "rung_two", "entries": entries}]
    runner._validate_accounting_evidence(models, 2, endpoints, gradients, None)
    for index in range(len(entries)):
        for field in ("count", "bytes"):
            changed = copy.deepcopy(models)
            changed[0]["entries"][index][field] += 1
            with pytest.raises(runner.ContractError):
                runner._validate_accounting_evidence(changed, 2, endpoints, gradients, None)
    changed_gradients = copy.deepcopy(gradients)
    changed_gradients[0]["classification"] = "learned_with_evidence"
    with pytest.raises(runner.ContractError):
        runner._validate_accounting_evidence(models, 2, endpoints, changed_gradients, None)
    changed_endpoints = copy.deepcopy(endpoints)
    changed_endpoints["rung_two"]["checkpoint"]["optimizer_state_dict"]["state"][0]["extra"] = torch.zeros(1)
    with pytest.raises(runner.ContractError):
        runner._validate_accounting_evidence(models, 2, changed_endpoints, gradients, None)


def test_model_accounting_validator_derives_work_from_attempt_ledger_and_closes_categories() -> None:
    categories = (
        "active_learned_parameter",
        "serialized_without_gradient",
        "inactive_parameter",
        "registered_buffer",
        "dynamic_recurrent_state",
        "route_index_storage",
        "routing_workspace",
        "optimizer_state",
    )
    entries = sorted(
        ({"category": category, "name": category, "count": index, "bytes": index * 4} for index, category in enumerate(categories)),
        key=lambda record: record["name"],
    )
    rows = [_attempt_event(0, "started"), _attempt_event(1, "completed")]
    model = {
        "model": "selected",
        "entries": entries,
        "attempted_updates": 1,
        "completed_updates": 1,
        "attempted_token_positions": 2048,
        "resource_sample_ids": [1, 3],
    }
    runner.validate_model_accounting([model], rows)
    mutations = []
    changed = copy.deepcopy(model)
    changed["attempted_updates"] = 2
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["completed_updates"] = 0
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["attempted_token_positions"] = 1
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["entries"].pop()
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["entries"].reverse()
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["entries"][0]["count"] = True
    mutations.append([changed])
    changed = copy.deepcopy(model)
    changed["resource_sample_ids"] = [3, 1]
    mutations.append([changed])
    mutations.append([model, copy.deepcopy(model)])
    for models in mutations:
        with pytest.raises(runner.ContractError):
            runner.validate_model_accounting(models, rows)
    with pytest.raises(runner.ContractError):
        runner.validate_model_accounting([{**model, "model": "other"}], rows)


def test_checkpoint_validator_enforces_schema_identity_state_and_finiteness(runtime_modules) -> None:
    torch = runtime_modules.torch
    checkpoint = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": "run",
        "rung": 1,
        "construction_seed": 11,
        "model": "selected",
        "stage": "joint",
        "completed_update": 512,
        "last_attempt_id": "a" * 64,
        "model_state_dict": {"weight": torch.ones(2)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"update": 512},
        "python_rng_state": [1, 2, 3],
        "torch_rng_state": torch.zeros(4, dtype=torch.uint8),
        "generator_states": {"data": torch.zeros(4, dtype=torch.uint8)},
        "final_batch_sha256": "b" * 64,
    }
    expected = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": "run",
        "rung": 1,
        "construction_seed": 11,
        "model": "selected",
        "stage": "joint",
        "completed_update": 512,
        "last_attempt_id": "a" * 64,
        "final_batch_sha256": "b" * 64,
    }
    runner._validate_checkpoint(checkpoint, expected, torch)
    changed = copy.deepcopy(checkpoint)
    changed["model_state_dict"]["weight"][0] = float("nan")
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(changed, expected, torch)
    assert caught.value.reason_code == "nonfinite"
    changed = copy.deepcopy(checkpoint)
    changed["optimizer_state_dict"]["state"] = {0: {"exp_avg": torch.tensor(float("inf"))}}
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(changed, expected, torch)
    assert caught.value.reason_code == "nonfinite"
    changed = copy.deepcopy(checkpoint)
    changed["model_state_dict"] = {"raw_remote_cache": torch.zeros(1)}
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(changed, expected, torch)
    assert caught.value.reason_code == "endpoint_inconsistency"
    changed = copy.deepcopy(checkpoint)
    changed["generator_states"]["route"] = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(changed, expected, torch)
    assert caught.value.reason_code == "endpoint_inconsistency"
    changed = copy.deepcopy(checkpoint)
    changed["final_batch_sha256"] = "c" * 64
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(changed, expected, torch)
    assert caught.value.reason_code == "endpoint_inconsistency"


def test_checkpoint_validator_rejects_nonfinite_scheduler_state(runtime_modules) -> None:
    torch = runtime_modules.torch
    checkpoint = {
        "schema_version": runner.SCHEMA_VERSION,
        "run_id": "run",
        "rung": 1,
        "construction_seed": 11,
        "model": "selected",
        "stage": "joint",
        "completed_update": 512,
        "last_attempt_id": "a" * 64,
        "model_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "scheduler_state_dict": {"multiplier": float("nan")},
        "python_rng_state": [1],
        "torch_rng_state": torch.zeros(1, dtype=torch.uint8),
        "generator_states": {"data": torch.zeros(1, dtype=torch.uint8)},
        "final_batch_sha256": "b" * 64,
    }
    expected = {key: checkpoint[key] for key in ("schema_version", "run_id", "rung", "construction_seed", "model", "stage", "completed_update", "last_attempt_id", "final_batch_sha256")}
    with pytest.raises(runner.HardAbort) as caught:
        runner._validate_checkpoint(checkpoint, expected, torch)
    assert caught.value.reason_code == "nonfinite"
    assert caught.value.context["surface"] == "checkpoint.scheduler_state_dict.multiplier"


def test_tensor_tree_storage_counts_unique_real_tensors(runtime_modules) -> None:
    torch = runtime_modules.torch
    first = torch.zeros((2, 3), dtype=torch.float32)
    second = torch.zeros(4, dtype=torch.int64)
    count, size = runner._tensor_tree_storage({"first": first, "again": first, "nested": [second]}, torch)
    assert count == 10
    assert size == first.numel() * first.element_size() + second.numel() * second.element_size()


def test_fresh_reload_parity_measures_logits_hidden_route_and_state(monkeypatch: pytest.MonkeyPatch, runtime_modules) -> None:
    torch = runtime_modules.torch
    model_module = runtime_modules.model_module

    def tensor_digest(tensor):
        data = bytes(tensor.detach().contiguous().cpu().view(torch.uint8).reshape(-1).tolist())
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(runner, "_tensor_sha256", tensor_digest)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9921)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
        batch = {"tokens": torch.randint(0, 128, (2, 128), dtype=torch.long)}
    checkpoint = {"model_state_dict": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}}
    evidence = runner._fresh_reload_evidence(model, checkpoint, batch, 1, runtime_modules)
    assert evidence == {
        "fresh_instance": True,
        "state_exact": True,
        "logits_max_error": 0.0,
        "hidden_max_error": 0.0,
        "route_exact": True,
        "rung": 1,
    }
    changed = copy.deepcopy(checkpoint)
    first_name = next(iter(changed["model_state_dict"]))
    changed["model_state_dict"][first_name].reshape(-1)[0] += 1.0
    with pytest.raises(runner.HardAbort) as caught:
        runner._fresh_reload_evidence(model, changed, batch, 1, runtime_modules)
    assert caught.value.reason_code == "endpoint_inconsistency"
    assert caught.value.context["surface"] == "fresh_reload"


def test_child_ack_protocols_send_exact_payload_and_accept_only_valid_ack() -> None:
    exact = _FakeConnection({"ack": True})
    runner._child_send_and_wait(exact, {"kind": "attempt", "value": 1})
    assert exact.sent == [{"kind": "attempt", "value": 1}]
    exchange = _FakeConnection({"ack": True, "sample_ids": [1, 2]})
    assert runner._child_exchange(exchange, {"kind": "resource_refs"}) == {"ack": True, "sample_ids": [1, 2]}
    assert exchange.sent == [{"kind": "resource_refs"}]
    for function, response in (
        (runner._child_send_and_wait, {"ack": True, "extra": None}),
        (runner._child_send_and_wait, {"ack": False}),
        (runner._child_exchange, {"ack": False}),
        (runner._child_exchange, []),
    ):
        with pytest.raises(runner.HardAbort) as caught:
            function(_FakeConnection(response), {"kind": "message"})
        assert caught.value.reason_code == "worker_exit"


@pytest.mark.parametrize("error", [EOFError(), BrokenPipeError()])
@pytest.mark.parametrize("function", [runner._child_send_and_wait, runner._child_exchange])
def test_child_ack_protocols_map_send_and_eof_failures_to_worker_exit(function, error) -> None:
    with pytest.raises(runner.HardAbort) as caught:
        function(_FakeConnection(send_error=error), {"kind": "message"})
    assert caught.value.reason_code == "worker_exit"
    with pytest.raises(runner.HardAbort) as caught:
        function(_FakeConnection(receive_error=error), {"kind": "message"})
    assert caught.value.reason_code == "worker_exit"


def test_clean_completion_handshake_is_exact_and_covers_frozen_assignments() -> None:
    handshake = runner.clean_completion_handshake("A", {11: 90, 37: 91, 71: 92})
    assert handshake == {
        "worker": "A",
        "status": "clean_complete",
        "assigned_jobs": [
            {"rung": 1, "construction_seed": 11},
            {"rung": 1, "construction_seed": 37},
            {"rung": 1, "construction_seed": 71},
        ],
        "last_event_sequence_by_construction_seed": {"11": 90, "37": 91, "71": 92},
        "artifacts_fsynced": True,
    }
    runner.validate_clean_completion_handshake(handshake, "A")


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.update(extra=None),
        lambda row: row.pop("artifacts_fsynced"),
        lambda row: row.update(worker="B"),
        lambda row: row.update(status="complete"),
        lambda row: row.update(artifacts_fsynced=False),
        lambda row: row["assigned_jobs"].reverse(),
        lambda row: row["assigned_jobs"][0].update(construction_seed=23),
        lambda row: row["last_event_sequence_by_construction_seed"].pop("37"),
        lambda row: row["last_event_sequence_by_construction_seed"].update({"23": 2}),
        lambda row: row["last_event_sequence_by_construction_seed"].update({"37": True}),
        lambda row: row["last_event_sequence_by_construction_seed"].update({"37": -1}),
    ),
)
def test_clean_completion_handshake_rejects_schema_identity_and_sequence_mutations(mutation) -> None:
    handshake = runner.clean_completion_handshake("A", {11: 1, 37: 2, 71: 3})
    mutation(handshake)
    with pytest.raises(runner.ContractError):
        runner.validate_clean_completion_handshake(handshake, "A")


def test_clean_handshake_wait_accepts_delayed_status_zero_teardown_without_timeout() -> None:
    process_a = SimpleNamespace(pid=201, exitcode=None, join=lambda timeout=0: None)
    process_b = SimpleNamespace(pid=202, exitcode=0, join=lambda timeout=0: None)
    worker_names = {201: "A", 202: "B"}
    observe = getattr(runner, "worker_exit_observations", lambda processes, handshakes, names: (None, None))
    complete, observations = observe(
        (process_a, process_b),
        {"A", "B"},
        worker_names,
    )
    assert complete is False
    assert observations == ()
    process_a.exitcode = 0
    complete, observations = observe(
        (process_a, process_b),
        {"A", "B"},
        worker_names,
    )
    assert complete is True
    assert observations == ()


@pytest.mark.parametrize(("handshakes", "exitcode"), ((set(), 0), ({"A"}, 7)))
def test_clean_handshake_wait_classifies_only_handshake_qualified_status_zero_as_clean(handshakes, exitcode) -> None:
    process = SimpleNamespace(pid=201, exitcode=exitcode, join=lambda timeout=0: None)
    observe = getattr(runner, "worker_exit_observations", lambda processes, accepted, names: (None, None))
    complete, observations = observe((process,), handshakes, {201: "A"})
    assert complete is False
    assert observations == ({"reason_code": "worker_exit", "context": {"worker": "A"}},)


def test_pilot_and_claim_wait_for_clean_exit_inside_monitored_parent_loop() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = _source_tree()
    for name in ("run_resource_pilot", "run_claim_workers"):
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
        segment = ast.get_source_segment(source, node)
        assert "worker_exit_observations(" in segment
        assert "join(timeout=1.0)" not in segment
        assert "_resource_sample(" in segment
    claim = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "run_claim_workers")
    claim_segment = ast.get_source_segment(source, claim)
    assert "claim_elapsed_time" in claim_segment
    assert "_verify_active_frozen_hashes(" in claim_segment


def test_claim_worker_protocol_rejects_duplicate_misattributed_and_postcompletion_messages() -> None:
    clean = {"kind": "clean_complete", "handshake": runner.clean_completion_handshake("B", {23: 1, 53: 2, 83: 3})}
    assert runner.validate_claim_worker_message(clean, "B", set()) == "clean_complete"
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(clean, "B", {"B"})
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(clean, "A", set())
    status = {"kind": "status", "worker": "A", "seed": 11, "stage": "joint", "logical_update": 1}
    assert runner.validate_claim_worker_message(status, "A", set()) == "status"
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(status, "A", {"A"})
    wrong_worker = dict(status, worker="B")
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(wrong_worker, "A", set())
    wrong_seed = dict(status, seed=23)
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(wrong_seed, "A", set())
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(dict(status, extra=None), "A", set())
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message({"kind": "unknown"}, "A", set())
    hard_abort = {"kind": "hard_abort", "worker": "A", "reason_code": "invented", "context": {}}
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(hard_abort, "A", set())


def test_parent_worker_failure_mapping_separates_transport_protocol_and_scientific_reasons() -> None:
    assert runner.parent_worker_failure_observation(EOFError(), "A", True) == {"reason_code": "worker_exit", "context": {"worker": "A"}}
    assert runner.parent_worker_failure_observation(BrokenPipeError(), "B", False) == {"reason_code": "worker_exit", "context": {"worker": "B"}}
    assert runner.parent_worker_failure_observation(runner.ContractError("malformed"), "A", False) == {"reason_code": "artifact_inconsistency", "context": {"worker": "A"}}
    assert runner.parent_worker_failure_observation(ValueError("malformed"), "B", False) == {"reason_code": "artifact_inconsistency", "context": {"worker": "B"}}
    observation = runner.parent_worker_failure_observation(runner.HardAbort("nonfinite", {"worker": "B", "event_sequence": 7}), "A", False)
    assert observation == {"reason_code": "nonfinite", "context": {"worker": "A", "event_sequence": 7}}
    with pytest.raises(runner.ContractError):
        runner.parent_worker_failure_observation(EOFError(), "C", True)
    with pytest.raises(runner.ContractError):
        runner.parent_worker_failure_observation(EOFError(), "A", 1)


def test_pilot_and_claim_parent_loops_use_shared_transport_and_protocol_mapping() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = _source_tree()
    for name in ("run_resource_pilot", "run_claim_workers"):
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
        segment = ast.get_source_segment(source, node)
        assert segment.count("parent_worker_failure_observation(exc, worker, True)") == 3
        assert segment.count("parent_worker_failure_observation(exc, worker, False)") == 1
    with pytest.raises(runner.HardAbort) as caught:
        runner._handle_pilot_worker_message(
            {"kind": "unknown", "worker": "A"},
            "A",
            _FakeConnection(),
            set(),
            {},
            {},
            [],
            SimpleNamespace(pending_signal=None),
        )
    assert caught.value.reason_code == "artifact_inconsistency"
    with pytest.raises(runner.ContractError) as caught:
        runner.validate_claim_worker_message({"kind": "unknown"}, "A", set())
    assert runner.parent_worker_failure_observation(caught.value, "A", False)["reason_code"] == "artifact_inconsistency"


def test_signal_controller_defers_and_becomes_terminal_without_late_mutation() -> None:
    signals = runner.SignalController()
    signals.defer()
    signals.inject()
    assert signals.pending_signal is not None
    assert signals.release() is not None
    signals.deactivate_terminal()
    pending_count = len(signals.pending)
    signals.inject()
    assert len(signals.pending) == pending_count
    with pytest.raises(runner.ContractError):
        signals.defer()


def test_terminal_commit_keeps_registry_active_through_boundary_and_catches_boundary_signal() -> None:
    signals = runner.SignalController()
    signals.active = True
    signals.defer()
    boundary_states = []

    def boundary():
        boundary_states.append((signals.active, signals.terminal, signals.deferred))
        signals.inject()

    assert signals.commit_terminal(boundary) == runner.signal.SIGTERM
    assert boundary_states == [(True, False, 0)]
    assert signals.active is True
    assert signals.terminal is False


def test_publication_activates_once_with_ordered_immutable_origin(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    staging = parent / ".run.initializing.123"
    staging.mkdir()
    (staging / "ready").write_text("ready", encoding="utf-8")
    final = parent / "run"
    signals = runner.SignalController()
    calls = []

    def activate(origin_ns, origin_utc):
        calls.append((origin_ns, origin_utc))

    result = runner.publish_and_activate(
        staging,
        final,
        signals,
        registry_activate=activate,
        monotonic_ns=lambda: 123456,
        wall_utc=lambda: "2026-07-19T00:00:00Z",
    )
    assert result.state == "active"
    assert result.registry_active is True
    assert result.abort_accounting_start_monotonic_ns == 123456
    assert result.abort_wall_start_utc == "2026-07-19T00:00:00Z"
    assert result.pending_signal is None
    assert calls == [(123456, "2026-07-19T00:00:00Z")]
    assert final.is_dir()
    assert not staging.exists()
    assert signals.deferred == 0


def test_publication_defers_signal_until_after_activation(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    staging = parent / ".run.initializing.123"
    staging.mkdir()
    final = parent / "run"
    signals = runner.SignalController()

    def activate(origin_ns, origin_utc):
        assert origin_ns == 1
        assert origin_utc == "origin"
        assert signals.deferred == 1
        signals.inject()

    result = runner.publish_and_activate(
        staging,
        final,
        signals,
        registry_activate=activate,
        monotonic_ns=lambda: 1,
        wall_utc=lambda: "origin",
    )
    assert result.state == "active"
    assert result.registry_active is True
    assert result.pending_signal is not None
    assert final.is_dir()
    assert signals.deferred == 0


def test_pending_signal_before_publication_is_initialization_refusal(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    staging = parent / ".run.initializing.123"
    staging.mkdir()
    final = parent / "run"
    signals = runner.SignalController()
    signals.inject()
    result = runner.publish_and_activate(staging, final, signals)
    assert result.state == "initialization_refusal"
    assert result.registry_active is False
    assert result.abort_accounting_start_monotonic_ns is None
    assert result.abort_wall_start_utc is None
    assert result.pending_signal is not None
    assert not final.exists()
    assert not staging.exists()


@pytest.mark.parametrize(
    "fault",
    ["rename_failure", "parent_fsync_failure", "origin_capture_failure", "activation_failure"],
)
def test_recoverable_publication_fault_rolls_back_to_absent_final(tmp_path: Path, fault: str) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    staging = parent / ".run.initializing.123"
    staging.mkdir()
    final = parent / "run"
    result = runner.publish_and_activate(staging, final, runner.SignalController(), fault=fault)
    assert result.state == "initialization_refusal"
    assert result.registry_active is False
    assert result.abort_accounting_start_monotonic_ns is None
    assert result.abort_wall_start_utc is None
    assert not final.exists()
    assert not staging.exists()


@pytest.mark.parametrize("fault", ["reverse_rename_failure", "rollback_parent_fsync_failure"])
def test_failed_publication_rollback_is_classified_orphan(tmp_path: Path, fault: str) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    staging = parent / ".run.initializing.123"
    staging.mkdir()
    final = parent / "run"
    result = runner.publish_and_activate(staging, final, runner.SignalController(), fault=fault)
    assert result.state == "orphaned"
    assert result.registry_active is False
    assert result.abort_accounting_start_monotonic_ns is None
    assert result.abort_wall_start_utc is None
    assert final.exists() or staging.exists()


def test_pilot_timeline_creation_closes_both_transition_sides(tmp_path: Path) -> None:
    root = tmp_path / "pilot-run"
    (root / "run").mkdir(parents=True)
    ready = runner.precreate_pilot_timeline(
        root,
        runner.SignalController(),
        swap_reader=lambda: 4096,
        first_row=_resource_row(root.name, "pilot", 4096),
    )
    assert ready.phase == "pilot"
    assert ready.outcome == "ready"
    assert ready.retained_paths == ("run/pilot_resources.jsonl",)
    assert ready.swap_baseline_bytes == 4096
    assert ready.reason_code is None
    assert len(ready.writers["run/pilot_resources.jsonl"].validate_committed_prefix()) == 1
    _close_writers(ready.writers)

    refused_root = tmp_path / "prepilot-run"
    (refused_root / "run").mkdir(parents=True)
    refused = runner.precreate_pilot_timeline(
        refused_root,
        runner.SignalController(),
        swap_reader=lambda: 4096,
        fault="creation_failure",
    )
    assert refused.phase == "prepilot"
    assert refused.outcome == "prepilot_abort"
    assert refused.retained_paths == ()
    assert refused.reason_code == "artifact_inconsistency"
    assert not (refused_root / "run" / "pilot_resources.jsonl").exists()


@pytest.mark.parametrize("mode", ("preexisting", "boundary_race", "owned_replacement"))
def test_pilot_timeline_creation_preserves_every_unowned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    root = tmp_path / mode
    (root / "run").mkdir(parents=True)
    path = root / "run" / "pilot_resources.jsonl"
    foreign = b"foreign-pilot-ledger\n"
    original_open = runner.os.open
    original_fsync_directory = runner.fsync_directory
    if mode == "preexisting":
        path.write_bytes(foreign)
    elif mode == "boundary_race":
        def open_path(target, flags, permissions=0o777):
            if Path(target) == path and flags & runner.os.O_EXCL and not path.exists():
                path.write_bytes(foreign)
            return original_open(target, flags, permissions)

        monkeypatch.setattr(runner.os, "open", open_path)
    else:
        def fsync(path_to_sync):
            if Path(path_to_sync) == path.parent and path.exists():
                moved = path.with_name("owned-pilot-ledger")
                path.rename(moved)
                path.write_bytes(foreign)
                raise OSError("injected pilot ownership replacement")
            return original_fsync_directory(path_to_sync)

        monkeypatch.setattr(runner, "fsync_directory", fsync)
    signals = runner.SignalController()
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.precreate_pilot_timeline(root, signals, swap_reader=lambda: 4096)
    assert path.read_bytes() == foreign
    assert signals.deferred == 0
    assert signals.terminal is False


def test_pilot_pending_signal_retains_durable_empty_timeline_and_baseline(tmp_path: Path) -> None:
    root = tmp_path / "pilot-run"
    (root / "run").mkdir(parents=True)
    result = runner.precreate_pilot_timeline(
        root,
        runner.SignalController(),
        swap_reader=lambda: 8192,
        fault="pending_signal_after_baseline",
    )
    assert result.phase == "pilot"
    assert result.outcome == "pilot_abort"
    assert result.swap_baseline_bytes == 8192
    assert result.reason_code == "signal_or_interruption"
    assert (root / "run" / "pilot_resources.jsonl").read_bytes() == b""
    assert result.writers["run/pilot_resources.jsonl"].last_committed_offset == 0
    _close_writers(result.writers)


def test_pilot_baseline_failure_is_pilot_abort_with_empty_timeline(tmp_path: Path) -> None:
    root = tmp_path / "pilot-run"
    (root / "run").mkdir(parents=True)
    result = runner.precreate_pilot_timeline(
        root,
        runner.SignalController(),
        swap_reader=lambda: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    assert result.phase == "pilot"
    assert result.outcome == "pilot_abort"
    assert result.swap_baseline_bytes is None
    assert result.reason_code == "resource_sampler_failure"
    assert (root / "run" / "pilot_resources.jsonl").read_bytes() == b""
    _close_writers(result.writers)


def test_claim_ledger_transition_ready_precreates_exact_seven(tmp_path: Path) -> None:
    root = tmp_path / "claim-run"
    (root / "run").mkdir(parents=True)
    (root / "run" / "pilot.json").write_text('{"decision":"proceed"}', encoding="utf-8")
    result = runner.precreate_claim_ledgers(root, runner.SignalController(), swap_reader=lambda: 12288)
    assert result.phase == "claim"
    assert result.outcome == "ready"
    assert result.retained_paths == CLAIM_LEDGER_PATHS
    assert tuple(result.writers) == CLAIM_LEDGER_PATHS
    assert result.swap_baseline_bytes == 12288
    assert result.reason_code is None
    assert all((root / relative).read_bytes() == b"" for relative in CLAIM_LEDGER_PATHS)
    _close_writers(result.writers)


@pytest.mark.parametrize("mode", ("preexisting", "boundary_race", "owned_replacement"))
def test_claim_ledger_creation_preserves_foreign_paths_and_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    root = tmp_path / mode
    (root / "run").mkdir(parents=True)
    pilot = root / "run" / "pilot.json"
    pilot_raw = b'{"decision":"proceed"}'
    pilot.write_bytes(pilot_raw)
    path = root / "rung1" / "11" / "attempts.jsonl"
    foreign = b"foreign-claim-ledger\n"
    original_open = runner.os.open
    original_fsync_directory = runner.fsync_directory
    if mode == "preexisting":
        path.parent.mkdir(parents=True)
        path.write_bytes(foreign)
    elif mode == "boundary_race":
        def open_path(target, flags, permissions=0o777):
            if Path(target) == path and flags & runner.os.O_EXCL and not path.exists():
                path.write_bytes(foreign)
            return original_open(target, flags, permissions)

        monkeypatch.setattr(runner.os, "open", open_path)
    else:
        def fsync(path_to_sync):
            if Path(path_to_sync) == path.parent and path.exists():
                moved = path.with_name("owned-claim-ledger")
                path.rename(moved)
                path.write_bytes(foreign)
                raise OSError("injected claim ownership replacement")
            return original_fsync_directory(path_to_sync)

        monkeypatch.setattr(runner, "fsync_directory", fsync)
    signals = runner.SignalController()
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.precreate_claim_ledgers(root, signals, swap_reader=lambda: 12288)
    assert path.read_bytes() == foreign
    assert pilot.read_bytes() == pilot_raw
    assert signals.deferred == 0
    assert signals.terminal is False


@pytest.mark.parametrize("fail_after", [1, 2, 3, 4, 5, 6])
def test_partial_claim_transition_rolls_back_to_pilot_abort(tmp_path: Path, fail_after: int) -> None:
    root = tmp_path / f"claim-run-{fail_after}"
    (root / "run").mkdir(parents=True)
    (root / "run" / "pilot.json").write_text('{"decision":"proceed"}', encoding="utf-8")
    result = runner.precreate_claim_ledgers(
        root,
        runner.SignalController(),
        swap_reader=lambda: 12288,
        fail_after=fail_after,
    )
    assert result.phase == "pilot"
    assert result.outcome == "pilot_abort"
    assert result.reason_code == "artifact_inconsistency"
    assert result.writers == {}
    assert not (root / "run" / "pilot.json").exists()
    assert all(not (root / relative).exists() for relative in CLAIM_LEDGER_PATHS)


def test_claim_pending_signal_retains_pilot_and_all_seven_ledgers(tmp_path: Path) -> None:
    root = tmp_path / "claim-run"
    (root / "run").mkdir(parents=True)
    pilot = root / "run" / "pilot.json"
    pilot.write_text('{"decision":"proceed"}', encoding="utf-8")
    result = runner.precreate_claim_ledgers(
        root,
        runner.SignalController(),
        swap_reader=lambda: 12288,
        pending_signal_after_baseline=True,
    )
    assert result.phase == "claim"
    assert result.outcome == "claim_abort"
    assert result.reason_code == "signal_or_interruption"
    assert result.swap_baseline_bytes == 12288
    assert pilot.is_file()
    assert all((root / relative).is_file() for relative in CLAIM_LEDGER_PATHS)
    _close_writers(result.writers)


def test_claim_baseline_failure_retains_all_seven_as_claim_abort(tmp_path: Path) -> None:
    root = tmp_path / "claim-run"
    (root / "run").mkdir(parents=True)
    pilot = root / "run" / "pilot.json"
    pilot.write_text('{"decision":"proceed"}', encoding="utf-8")
    result = runner.precreate_claim_ledgers(
        root,
        runner.SignalController(),
        swap_reader=lambda: 12288,
        baseline_failure=True,
    )
    assert result.phase == "claim"
    assert result.outcome == "claim_abort"
    assert result.reason_code == "resource_sampler_failure"
    assert result.swap_baseline_bytes is None
    assert pilot.is_file()
    assert all((root / relative).is_file() for relative in CLAIM_LEDGER_PATHS)
    _close_writers(result.writers)


def test_terminal_checksum_is_sorted_complete_and_single_entry(tmp_path: Path) -> None:
    root = tmp_path / "run"
    (root / "z").mkdir(parents=True)
    (root / "z" / "b.bin").write_bytes(b"b")
    (root / "a.json").write_bytes(b"a")
    signals = runner.SignalController()
    terminalizer = runner.ChecksumTerminalizer(root, signals)
    result = terminalizer.finalize(expected_paths=("z/b.bin", "a.json"))
    expected = (
        f"{hashlib.sha256(b'a').hexdigest()}  a.json\n"
        f"{hashlib.sha256(b'b').hexdigest()}  z/b.bin\n"
    ).encode("ascii")
    assert result.terminal is True
    assert result.covered_paths == ("a.json", "z/b.bin")
    assert result.checksum_path.read_bytes() == expected
    assert signals.terminal is True
    with pytest.raises(runner.ContractError):
        terminalizer.finalize()
    with pytest.raises(runner.ContractError):
        runner.write_sha256s_terminal(root)
    assert result.checksum_path.read_bytes() == expected


@pytest.mark.parametrize("mode", ("boundary_race", "owned_replacement"))
def test_terminal_checksum_preserves_foreign_race_and_never_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    root = tmp_path / mode
    root.mkdir()
    (root / "a").write_bytes(b"a")
    checksum = root / "SHA256SUMS"
    foreign = b"foreign-checksum\n"
    original_open = runner.os.open
    if mode == "boundary_race":
        def open_path(target, flags, permissions=0o777):
            if Path(target) == checksum and flags & runner.os.O_EXCL and not checksum.exists():
                checksum.write_bytes(foreign)
            return original_open(target, flags, permissions)

        monkeypatch.setattr(runner.os, "open", open_path)

        def fault(stage):
            return None
    else:
        def fault(stage):
            if stage == "before_fsync":
                moved = checksum.with_name("owned-checksum")
                checksum.rename(moved)
                checksum.write_bytes(foreign)
                raise OSError("injected checksum ownership replacement")

    signals = runner.SignalController()
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals, fault_hook=fault)
    assert checksum.read_bytes() == foreign
    assert signals.terminal is False


def test_terminal_checksum_rejects_file_set_drift_without_partial_checksum(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    with pytest.raises(runner.ContractError):
        runner.write_sha256s_terminal(root, expected_paths=("missing",))
    assert not (root / "SHA256SUMS").exists()


def test_terminal_checksum_refuses_pending_signal_before_boundary(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    signals = runner.SignalController()
    signals.inject()
    with pytest.raises((runner.ContractError, runner.HardAbort)):
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals)
    assert not (root / "SHA256SUMS").exists()
    assert signals.terminal is False


@pytest.mark.parametrize("stage", ("after_fsync_before_terminal_commit", "after_directory_fsync_before_terminal_commit"))
@pytest.mark.parametrize("failure", ("signal", "deadline"))
def test_terminal_checksum_rolls_back_late_signal_and_deadline_with_closed_descriptor_and_directory_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, failure: str) -> None:
    root = tmp_path / f"{failure}-{stage}"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    signals = runner.SignalController()
    checksum_descriptors = []
    directory_fsyncs = []
    open_descriptor = runner.os.open
    fsync_directory = runner.fsync_directory

    def tracked_open(path, flags, mode=0o777):
        descriptor = open_descriptor(path, flags, mode)
        if Path(path).name == "SHA256SUMS":
            checksum_descriptors.append(descriptor)
        return descriptor

    def tracked_directory_fsync(path):
        directory_fsyncs.append(Path(path))
        fsync_directory(path)

    def fault_hook(observed_stage):
        if observed_stage != stage:
            return
        if failure == "signal":
            signals.inject()
        else:
            raise runner.HardAbort("claim_elapsed_time", {"stage": observed_stage})

    monkeypatch.setattr(runner.os, "open", tracked_open)
    monkeypatch.setattr(runner, "fsync_directory", tracked_directory_fsync)
    with pytest.raises(runner.HardAbort) as caught:
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals, fault_hook=fault_hook)
    assert caught.value.reason_code == ("signal_or_interruption" if failure == "signal" else "claim_elapsed_time")
    assert checksum_descriptors and signals.deferred == 0 and signals.terminal is False
    with pytest.raises(OSError):
        runner.os.fstat(checksum_descriptors[0])
    assert not (root / "SHA256SUMS").exists()
    assert directory_fsyncs and all(path == root for path in directory_fsyncs)
    if stage == "after_directory_fsync_before_terminal_commit":
        assert len(directory_fsyncs) == 2


def test_terminal_checksum_rejects_signal_injected_by_release_before_success_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "release-signal"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    signals = runner.SignalController()
    original_release = signals.release

    def release_then_inject():
        pending = original_release()
        signals.inject()
        return pending

    monkeypatch.setattr(signals, "release", release_then_inject)
    with pytest.raises(runner.HardAbort) as caught:
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals)
    assert caught.value.reason_code == "signal_or_interruption"
    assert not (root / "SHA256SUMS").exists()
    assert signals.pending_signal is not None
    assert signals.deferred == 0
    assert signals.terminal is False


def test_terminal_checksum_runs_guard_after_final_fsync_before_clean_adoption(tmp_path: Path) -> None:
    root = tmp_path / "post-fsync-guard"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    signals = runner.SignalController()
    signals.active = True
    observed = []

    def guard(stage):
        observed.append((stage, signals.active, signals.terminal))
        if stage == "after_terminal_fsync_before_terminal_commit":
            raise runner.HardAbort("claim_elapsed_time", {"stage": stage})

    with pytest.raises(runner.HardAbort) as caught:
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals, fault_hook=guard)
    assert caught.value.reason_code == "claim_elapsed_time"
    assert ("after_terminal_fsync_before_terminal_commit", True, False) in observed
    assert not (root / "SHA256SUMS").exists()
    assert signals.terminal is False


@pytest.mark.parametrize("phase", ("prepilot", "pilot", "claim"))
def test_real_abort_finalization_preserves_primary_cause_through_later_checksum_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    payload = _tracked_payload()
    closure_kind = {"prepilot": "prepilot_abort", "pilot": "pilot_abort", "claim": "claim_abort"}[phase]
    root = tmp_path / f"{phase}-abort"
    root.mkdir()
    _materialize_synthetic_closure(root, payload, closure_kind)
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    (root / "ABORTED.json").unlink()
    for path in root.rglob("*.jsonl"):
        path.write_bytes(b"")
    source_manifest = json.loads((root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["records"] = []
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes(source_manifest))
    config_manifest = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config_manifest["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config_manifest))
    signals = runner.SignalController()
    original_fsync_directory = runner.fsync_directory
    injected = False

    def inject_during_checksum(path):
        nonlocal injected
        original_fsync_directory(path)
        if not injected and (root / "SHA256SUMS").exists():
            injected = True
            signals.inject()

    monkeypatch.setattr(runner, "fsync_directory", inject_during_checksum)
    runner.finalize_hard_abort(
        root,
        payload,
        signals,
        "nonfinite",
        phase,
        {},
        1000,
        "2026-07-19T00:00:00Z",
        2000,
        {},
        None,
    )
    assert injected is True
    assert (root / "ABORTED.json").is_file()
    assert json.loads((root / "ABORTED.json").read_text(encoding="utf-8"))["reason_code"] == "nonfinite"
    assert runner.validate_artifact_closure(root, payload, closure_kind)
    assert signals.pending_signal is None
    assert signals.deferred == 0
    assert signals.terminal is True


@pytest.mark.parametrize(
    "stage",
    (
        "before_resource_sample",
        "after_resource_sample",
        "after_writer_close",
        "after_cleanup",
        "after_aborted_write",
        "before_sha256s",
    ),
)
def test_abort_packaging_signal_is_secondary_at_every_preterminal_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / stage
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "prepilot_abort")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    (root / "ABORTED.json").unlink()
    source_manifest = json.loads((root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["records"] = []
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes(source_manifest))
    config_manifest = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config_manifest["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config_manifest))
    signals = runner.SignalController()
    original_sample = runner.best_effort_abort_resource_sample
    original_cleanup = runner._cleanup_abort_surface
    original_validate = runner.validate_artifact_closure
    original_terminalize = runner.write_sha256s_terminal

    def sample_boundary(*args, **kwargs):
        if stage == "before_resource_sample":
            signals.inject()
        result = original_sample(*args, **kwargs)
        if stage == "after_resource_sample":
            signals.inject()
        return result

    def cleanup_boundary(*args, **kwargs):
        if stage == "after_writer_close":
            signals.inject()
        result = original_cleanup(*args, **kwargs)
        if stage == "after_cleanup":
            signals.inject()
        return result

    def validation_boundary(*args, **kwargs):
        if stage == "after_aborted_write":
            signals.inject()
        return original_validate(*args, **kwargs)

    def terminal_boundary(*args, **kwargs):
        if stage == "before_sha256s":
            signals.inject()
        return original_terminalize(*args, **kwargs)

    monkeypatch.setattr(runner, "best_effort_abort_resource_sample", sample_boundary)
    monkeypatch.setattr(runner, "_cleanup_abort_surface", cleanup_boundary)
    monkeypatch.setattr(runner, "validate_artifact_closure", validation_boundary)
    monkeypatch.setattr(runner, "write_sha256s_terminal", terminal_boundary)

    runner.finalize_hard_abort(
        root,
        payload,
        signals,
        "nonfinite",
        "prepilot",
        {"worker": "A"},
        1000,
        "2026-07-19T00:00:00Z",
        2000,
        {},
        None,
    )
    assert json.loads((root / "ABORTED.json").read_text(encoding="utf-8"))["reason_code"] == "nonfinite"
    assert runner.validate_artifact_closure(root, payload, "prepilot_abort")


def test_abort_finalization_rejects_invalid_complete_resource_timeline_before_terminal_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _tracked_payload()
    root = tmp_path / "invalid-abort-timeline"
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    (root / "ABORTED.json").unlink()
    for path in root.rglob("*.jsonl"):
        path.write_bytes(b"")
    rows = _resource_timeline("claim")[:2]
    for row in rows:
        row["run_id"] = root.name
    rows[1]["monotonic_ns"] = rows[0]["monotonic_ns"] + 1
    resource_path = root / "run" / "resources.jsonl"
    resource_path.write_bytes(b"".join(runner.canonical_json_bytes(row) + b"\n" for row in rows))
    source_manifest = json.loads((root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["records"] = []
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes(source_manifest))
    config_manifest = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config_manifest["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config_manifest))
    terminalized = []
    original_terminalize = runner.write_sha256s_terminal

    def terminalize(*args, **kwargs):
        terminalized.append(True)
        return original_terminalize(*args, **kwargs)

    monkeypatch.setattr(runner, "write_sha256s_terminal", terminalize)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.finalize_hard_abort(
            root,
            payload,
            runner.SignalController(),
            "nonfinite",
            "claim",
            {},
            1000,
            "2026-07-19T00:00:00Z",
            2000,
            {},
            100,
        )
    assert terminalized == []
    assert not (root / "SHA256SUMS").exists()


def test_abort_finalization_accepts_terminal_zeroed_child_disappearance_without_rewriting_prior_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _tracked_payload()
    root = tmp_path / "terminal-child-disappearance"
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    (root / "ABORTED.json").unlink()
    (root / "SHA256SUMS").unlink(missing_ok=True)
    for path in root.rglob("*.jsonl"):
        path.write_bytes(b"")
    rows = _resource_timeline("claim")
    for row in rows:
        row["run_id"] = root.name
    rows[1]["expected_pids"] = [101, 102]
    rows[1]["processes"].append({"pid": 102, "ppid": 101, "rss_bytes": 1_271_414_784, "cpu_time_us": 14_250_000})
    rows[1]["aggregate_rss_bytes"] += 1_271_414_784
    rows[1]["aggregate_cpu_time_us"] += 14_250_000
    rows[2]["expected_pids"] = [101, 102]
    rows[2]["processes"].append({"pid": 102, "ppid": 101, "rss_bytes": 0, "cpu_time_us": 0})
    resource_path = root / "run" / "resources.jsonl"
    resource_path.write_bytes(b"".join(runner.canonical_json_bytes(row) + b"\n" for row in rows))
    source_manifest = json.loads((root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["records"] = []
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes(source_manifest))
    config_manifest = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config_manifest["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config_manifest))
    runner.finalize_hard_abort(
        root,
        payload,
        runner.SignalController(),
        "nonfinite",
        "claim",
        {},
        1000,
        "2026-07-19T00:00:00Z",
        2000,
        {},
        100,
    )
    aborted = json.loads((root / "ABORTED.json").read_text(encoding="utf-8"))
    assert aborted["reason_code"] == "nonfinite"
    assert aborted["resource_state"]["last_sample_id"] == 2
    persisted = [json.loads(line) for line in resource_path.read_text(encoding="utf-8").splitlines()]
    assert persisted[1]["processes"][1]["cpu_time_us"] == 14_250_000
    assert persisted[2]["processes"][1]["cpu_time_us"] == 0
    assert (root / "SHA256SUMS").is_file()


@pytest.mark.parametrize("failure", ("unlink", "directory_fsync"))
def test_abort_cleanup_mutation_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / failure
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "prepilot_abort")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    checksum = root / "SHA256SUMS"
    checksum.write_bytes(b"provisional-checksum\n")
    real_unlink = Path.unlink
    real_fsync = runner.fsync_directory

    def unlink(path, *args, **kwargs):
        if failure == "unlink" and path == checksum:
            raise OSError("injected abort cleanup unlink failure")
        return real_unlink(path, *args, **kwargs)

    def fsync(path):
        if failure == "directory_fsync" and Path(path) == root and not checksum.exists():
            raise OSError("injected abort cleanup fsync failure")
        return real_fsync(path)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(runner, "fsync_directory", fsync)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._cleanup_abort_surface(root, "prepilot", "not_started")
    assert not (root / "SHA256SUMS").exists() if failure == "directory_fsync" else (root / "SHA256SUMS").is_file()


@pytest.mark.parametrize("failure", ("aborted_write", "aborted_identity", "checksum_write", "checksum_orphan"))
def test_abort_finalization_failure_never_produces_false_terminal_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / failure
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "prepilot_abort")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    (root / "ABORTED.json").unlink()
    (root / "SHA256SUMS").unlink(missing_ok=True)
    source_manifest = json.loads((root / "run" / "source_manifest.json").read_text(encoding="utf-8"))
    source_manifest["records"] = []
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes(source_manifest))
    config_manifest = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config_manifest["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config_manifest))
    real_write = runner.write_canonical_json

    def write(path, value, exclusive=True, owned_paths=None):
        if failure == "aborted_write" and Path(path) == root / "ABORTED.json":
            raise OSError("injected ABORTED write failure")
        return real_write(path, value, exclusive, owned_paths)

    monkeypatch.setattr(runner, "write_canonical_json", write)
    real_descriptor_identity = runner._descriptor_identity

    def descriptor_identity(descriptor):
        if failure == "aborted_identity" and (root / "ABORTED.json").exists():
            raise OSError("injected ABORTED identity capture failure")
        return real_descriptor_identity(descriptor)

    monkeypatch.setattr(runner, "_descriptor_identity", descriptor_identity)
    orphan = runner.UnrecoverableOrphan("injected checksum orphan")
    if failure == "checksum_write":
        monkeypatch.setattr(runner, "write_sha256s_terminal", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected checksum write failure")))
    elif failure == "checksum_orphan":
        monkeypatch.setattr(runner, "write_sha256s_terminal", lambda *args, **kwargs: (_ for _ in ()).throw(orphan))
    signals = runner.SignalController()
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        runner.finalize_hard_abort(
            root,
            payload,
            signals,
            "nonfinite",
            "prepilot",
            {},
            1000,
            "2026-07-21T00:00:00Z",
            2000,
            {},
            None,
        )
    if failure == "checksum_orphan":
        assert caught.value is orphan
    assert not (root / "SHA256SUMS").exists()
    assert not (root / "ABORTED.json").exists()
    assert signals.terminal is False


def test_terminal_checksum_identity_capture_failure_removes_created_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checksum-identity"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    checksum = root / "SHA256SUMS"
    real_descriptor_identity = runner._descriptor_identity

    def descriptor_identity(descriptor):
        if checksum.exists():
            raise OSError("injected checksum identity capture failure")
        return real_descriptor_identity(descriptor)

    monkeypatch.setattr(runner, "_descriptor_identity", descriptor_identity)
    signals = runner.SignalController()
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals)
    assert not checksum.exists()
    assert signals.terminal is False


def test_initialization_artifact_closure_requires_absent_run_root(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "absent"
    assert runner.validate_artifact_closure(root, payload, "initialization_absent") == ()
    assert runner.validate_artifact_closure(root, payload, "initialization") == ()
    root.mkdir()
    with pytest.raises(runner.ContractError):
        runner.validate_artifact_closure(root, payload, "initialization_absent")


@pytest.mark.parametrize("kind", ("prepilot_abort", "pilot_abort", "pilot_stop", "claim_abort", "clean"))
def test_artifact_closure_accepts_only_exact_dynamic_file_set_and_terminal_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / kind
    root.mkdir()
    detail_digest = "d" * 64
    paths = _materialize_synthetic_closure(root, payload, kind, detail_digest)
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "started" if kind == "clean" else "not_started")
    expected = runner.validate_artifact_closure(root, payload, kind)
    assert set(expected) == paths - {"SHA256SUMS"}
    result = runner.write_sha256s_terminal(root, expected_paths=expected)
    assert result.covered_paths == expected
    assert runner.validate_artifact_closure(root, payload, kind) == expected


def test_pilot_stop_closure_includes_pilot_generated_check_details_and_preserves_pilot(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "pilot-stop-details"
    root.mkdir()
    preflight_digest = "d" * 64
    pilot_digest = "e" * 64
    _materialize_synthetic_closure(root, payload, "pilot_stop", preflight_digest)
    pilot_detail = root / "run" / "check_details" / f"{pilot_digest}.json"
    pilot_detail.write_bytes(b"{}")
    pilot_path = root / "run" / "pilot.json"
    pilot_raw = runner.canonical_json_bytes(
        {
            "decision": "stop",
            "assertions": [{"details_sha256": pilot_digest}],
        }
    )
    pilot_path.write_bytes(pilot_raw)
    expected = runner.validate_artifact_closure(root, payload, "pilot_stop")
    assert f"run/check_details/{pilot_digest}.json" in expected
    runner.write_sha256s_terminal(root, expected_paths=expected)
    assert pilot_path.read_bytes() == pilot_raw
    assert runner.validate_artifact_closure(root, payload, "pilot_stop") == expected


@pytest.mark.parametrize(
    ("training_start_state", "expected_status_count"),
    (("not_started", 0), ("awaiting_review", 1), ("reviewed_ready", 4), ("started", 4)),
)
def test_claim_abort_closure_retains_exact_training_start_state_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training_start_state: str, expected_status_count: int) -> None:
    payload = _tracked_payload()
    root = tmp_path / training_start_state
    root.mkdir()
    paths = _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state=training_start_state)
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, training_start_state)
    expected = runner.validate_artifact_closure(root, payload, "claim_abort")
    base_reviews = {f"run/reviews/{value:064x}.json" for value in range(1, 5)}
    status_paths = {
        path
        for path in expected
        if path in {"run/training_start_request.json", "run/project_plan_training_start.md", "run/training_start_plan.json"}
        or path.startswith("run/reviews/") and path not in base_reviews
    }
    assert len(status_paths) == expected_status_count
    assert set(expected) == paths - {"SHA256SUMS"}


@pytest.mark.parametrize("training_start_state", ("not_started", "awaiting_review", "reviewed_ready", "started"))
def test_abort_cleanup_preserves_only_committed_training_start_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_start_state: str,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / training_start_state
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state=training_start_state)
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, training_start_state)
    runner._cleanup_abort_surface(root, "claim", training_start_state)
    present = {
        relative
        for relative in ("run/training_start_request.json", "run/project_plan_training_start.md", "run/training_start_plan.json")
        if (root / relative).exists()
    }
    expected = {
        "not_started": set(),
        "awaiting_review": {"run/training_start_request.json"},
        "reviewed_ready": {"run/training_start_request.json", "run/project_plan_training_start.md", "run/training_start_plan.json"},
        "started": {"run/training_start_request.json", "run/project_plan_training_start.md", "run/training_start_plan.json"},
    }[training_start_state]
    assert present == expected


@pytest.mark.parametrize("observed_state", ("not_started", "awaiting_review", "reviewed_ready", "started"))
def test_abort_cleanup_rejects_state_mismatch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_state: str,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / observed_state
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state=observed_state)
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, observed_state)
    before = _tree_snapshot(root)
    requested_state = "started" if observed_state != "started" else "not_started"
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._cleanup_abort_surface(root, "claim", requested_state)
    assert _tree_snapshot(root) == before


@pytest.mark.parametrize("mutation", ("missing_registry", "malformed_registry", "review_read_error"))
def test_abort_cleanup_preflight_normalizes_every_review_failure_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / mutation
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state="not_started")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    config_path = root / "run" / "config_manifest.json"
    if mutation == "missing_registry":
        config_path.write_bytes(runner.canonical_json_bytes({}))
    elif mutation == "malformed_registry":
        config_path.write_bytes(b"not-json")
    else:
        real_iterdir = Path.iterdir

        def iterdir(path):
            if path == root / "run" / "reviews":
                raise OSError("injected review directory read failure")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", iterdir)
    before = _tree_snapshot(root)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._cleanup_abort_surface(root, "claim", "not_started")
    assert _tree_snapshot(root) == before


def test_abort_cleanup_rejects_symlinked_rung_without_touching_external_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _tracked_payload()
    root = tmp_path / "symlinked-rung"
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state="not_started")
    _set_synthetic_live_plan(tmp_path, monkeypatch, root, "not_started")
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "protected.bin"
    protected.write_bytes(b"external-bytes\n")
    rung = root / "rung1"
    moved = root / "owned-rung1"
    rung.rename(moved)
    rung.symlink_to(external, target_is_directory=True)
    before = _tree_snapshot(root)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._cleanup_abort_surface(root, "claim", "not_started")
    assert protected.read_bytes() == b"external-bytes\n"
    assert _tree_snapshot(root) == before


def test_abort_frozen_hash_evidence_includes_every_committed_training_start_surface(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "started"
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "claim_abort", training_start_state="started")
    (root / "run" / "source_manifest.json").write_bytes(runner.canonical_json_bytes({"records": []}))
    config = json.loads((root / "run" / "config_manifest.json").read_text(encoding="utf-8"))
    config["records"] = []
    (root / "run" / "config_manifest.json").write_bytes(runner.canonical_json_bytes(config))
    paths = {record["path"] for record in runner._frozen_hash_records(root)}
    link = json.loads((root / "run" / "training_start_plan.json").read_text(encoding="utf-8"))
    assert {
        "run/project_plan_launch.md",
        "run/training_start_request.json",
        "run/project_plan_training_start.md",
        "run/training_start_plan.json",
        link["review_path"],
    } <= paths


@pytest.mark.parametrize("mutation", ("missing", "extra", "symlink", "checksum", "review_duplicate"))
def test_artifact_closure_rejects_missing_extra_link_checksum_and_dynamic_identity_mutations(tmp_path: Path, mutation: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / mutation
    root.mkdir()
    _materialize_synthetic_closure(root, payload, "prepilot_abort", "d" * 64)
    if mutation == "missing":
        (root / "run" / "environment.json").unlink()
    elif mutation == "extra":
        (root / "unlisted.bin").write_bytes(b"extra")
    elif mutation == "symlink":
        target = tmp_path / "outside"
        target.write_bytes(b"outside")
        linked = root / "run" / "environment.json"
        linked.unlink()
        linked.symlink_to(target)
    elif mutation == "checksum":
        expected = runner.validate_artifact_closure(root, payload, "prepilot_abort")
        runner.write_sha256s_terminal(root, expected_paths=expected)
        (root / "SHA256SUMS").write_text("0" * 64 + "  ABORTED.json\n", encoding="ascii")
    else:
        config_path = root / "run" / "config_manifest.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["review_records"][1]["artifact_sha256"] = config["review_records"][0]["artifact_sha256"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(runner.ContractError):
        runner.validate_artifact_closure(root, payload, "prepilot_abort")


def test_gate_input_package_returns_exact_124_record_registry_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _tracked_payload()
    root = tmp_path / "gate-package"
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    decisions = runner.validate_gate_input_package(root, payload)
    expected = [
        (seed, row["gate_id"])
        for seed in (11, 23, 37, 53, 71)
        for row in payload["gates"]["rung_one_registry"]
    ] + [(83, row["gate_id"]) for row in payload["gates"]["rung_two_registry"]]
    assert len(decisions) == 124
    assert [(row["construction_seed"], row["gate_id"]) for row in decisions] == expected
    assert all(row["gate_pass"] is True for row in decisions)
    package_calls = []
    monkeypatch.setattr(runner, "validate_claim_artifact_package", lambda run_root, frozen_payload, runtime=None: package_calls.append((run_root, frozen_payload, runtime)))
    summary = runner._gate_summary(root, payload)
    assert package_calls == [(root, payload, None)]
    assert len(summary["passed_gates"]) == 124
    assert summary["failed_gates"] == []
    assert [row["construction_seed"] for row in summary["per_seed"]] == [11, 23, 37, 53, 71]
    assert summary["rung_two"] == {
        "construction_seed": 83,
        "total_gates": 4,
        "passed_gates": [row["gate_id"] for row in payload["gates"]["rung_two_registry"]],
        "failed_gates": [],
        "gate_pass": True,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "r1_missing",
        "r1_order",
        "r1_extra_key",
        "r1_duplicate_gate",
        "r1_unknown_gate",
        "r1_threshold",
        "r1_contradictory_pass",
        "r1_wilson",
        "r1_provenance",
        "r1_noncanonical",
        "r1_population_digest",
        "r1_metric_counter",
        "r1_unrelated_counter",
        "r1_selected_provenance",
        "r2_order",
        "r2_checkpoint",
        "stats_condition_order",
        "stats_record_order",
        "stats_aggregate",
        "stats_aggregate_mean",
        "stats_aggregate_std",
        "stats_aggregate_min",
        "stats_aggregate_max",
        "stats_gate_threshold",
    ),
)
def test_gate_input_package_rejects_cardinality_order_schema_math_and_provenance_mutations(tmp_path: Path, mutation: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / mutation
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    r1_path = root / "rung1" / "11" / "evaluation.jsonl"
    r1_rows = [json.loads(line) for line in r1_path.read_text(encoding="utf-8").splitlines()]
    r2_path = root / "rung2" / "83" / "evaluation.jsonl"
    r2_rows = [json.loads(line) for line in r2_path.read_text(encoding="utf-8").splitlines()]
    stats_path = root / "rung2" / "83" / "gate_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    gated = [index for index, row in enumerate(r1_rows) if row["gate_id"] is not None]
    canonical = True
    if mutation == "r1_missing":
        r1_rows.pop()
    elif mutation == "r1_order":
        r1_rows[0], r1_rows[1] = r1_rows[1], r1_rows[0]
    elif mutation == "r1_extra_key":
        r1_rows[0]["extra"] = None
    elif mutation == "r1_duplicate_gate":
        r1_rows[gated[1]]["gate_id"] = r1_rows[gated[0]]["gate_id"]
    elif mutation == "r1_unknown_gate":
        r1_rows[gated[0]]["gate_id"] = "r1.unknown"
    elif mutation == "r1_threshold":
        r1_rows[gated[0]]["gate_threshold"] += 0.01
    elif mutation == "r1_contradictory_pass":
        r1_rows[gated[0]]["gate_pass"] = not r1_rows[gated[0]]["gate_pass"]
    elif mutation == "r1_wilson":
        r1_rows[0]["estimate"] += 0.01
    elif mutation == "r1_provenance":
        r1_rows[0]["provenance_sha256s"].reverse()
    elif mutation == "r1_noncanonical":
        canonical = False
    elif mutation == "r1_population_digest":
        r1_rows[0]["population_sha256"] = "9" * 64
    elif mutation == "r1_metric_counter":
        r1_rows[0]["answer_correct"] += 1
    elif mutation == "r1_unrelated_counter":
        r1_rows[0]["original_source_hits"] = 0
        r1_rows[0]["original_source_total"] = 0
    elif mutation == "r1_selected_provenance":
        selected = next(row for row in r1_rows if row["metric"] == "selected_mask_oracle_max_error")
        selected["provenance_sha256s"].reverse()
    elif mutation == "r2_order":
        r2_rows.reverse()
    elif mutation == "r2_checkpoint":
        stats["checkpoint_sha256"] = "9" * 64
    elif mutation == "stats_condition_order":
        stats["conditions"].reverse()
    elif mutation == "stats_record_order":
        stats["conditions"][0]["records"].reverse()
    elif mutation == "stats_aggregate":
        stats["conditions"][0]["aggregate"]["count"] += 1
    elif mutation == "stats_aggregate_mean":
        stats["conditions"][0]["aggregate"]["mean"] += 0.1
    elif mutation == "stats_aggregate_std":
        stats["conditions"][0]["aggregate"]["population_std"] += 0.1
    elif mutation == "stats_aggregate_min":
        stats["conditions"][0]["aggregate"]["min"] += 0.1
    elif mutation == "stats_aggregate_max":
        stats["conditions"][0]["aggregate"]["max"] += 0.1
    else:
        stats["conditions"][0]["gate_threshold"] = 1
    _write_jsonl_rows(r1_path, r1_rows, canonical=canonical)
    _write_jsonl_rows(r2_path, r2_rows)
    stats_path.write_bytes(runner.canonical_json_bytes(stats))
    with pytest.raises(runner.ContractError):
        runner.validate_gate_input_package(root, payload)


@pytest.mark.parametrize("rung", (1, 2))
def test_prediction_artifact_validates_exact_condition_example_and_population_contract(tmp_path: Path, rung: int) -> None:
    payload = _tracked_payload()
    seed = 11 if rung == 1 else 83
    root = tmp_path / f"rung-{rung}"
    seed_root = root / ("rung1/11" if rung == 1 else "rung2/83")
    rows = _synthetic_prediction_rows(root, payload, rung, seed)
    validation_inputs = _prediction_validation_inputs(root, rung, seed)
    _write_prediction_rows(seed_root / "predictions.jsonl.gz", rows)
    evidence = runner._validate_prediction_artifact(root, payload, seed_root, rung, seed, *validation_inputs)
    populations = evidence["populations"]
    conditions = payload["stages"]["rung_one"]["evaluation_arm_order"] if rung == 1 else payload["stages"]["rung_two"]["evaluation_order"]
    assert [identity for identity in populations if identity[1] == "all"] == [(condition, "all") for condition in conditions]
    assert all(populations[(condition, "all")] == list(range(512)) for condition in conditions)
    if rung == 1:
        assert len(populations[("carry_shuffle", "same_condition")]) == 0
        assert len(populations[("carry_shuffle", "changed_condition")]) == 512


@pytest.mark.parametrize(
    "mutation",
    ("cardinality", "order", "extra_key", "correctness", "carry_missing", "carry_stratum", "nonshuffle_foreign", "dense_source", "rung_two_route"),
)
def test_prediction_artifact_rejects_schema_identity_route_and_correctness_mutations(tmp_path: Path, mutation: str) -> None:
    payload = _tracked_payload()
    rung = 2 if mutation == "rung_two_route" else 1
    seed = 83 if rung == 2 else 11
    root = tmp_path / mutation
    seed_root = root / ("rung2/83" if rung == 2 else "rung1/11")
    rows = _synthetic_prediction_rows(root, payload, rung, seed)
    validation_inputs = _prediction_validation_inputs(root, rung, seed)
    if mutation == "cardinality":
        rows.pop()
    elif mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "extra_key":
        rows[0]["extra"] = None
    elif mutation == "correctness":
        rows[0]["correct"] = False
    elif mutation == "carry_missing":
        row = next(item for item in rows if item["condition"] == "carry_shuffle")
        row["foreign_source"] = None
    elif mutation == "carry_stratum":
        row = next(item for item in rows if item["condition"] == "carry_shuffle" and item["condition_stratum"] == "changed_condition")
        row["condition_stratum"] = "same_condition"
    elif mutation == "nonshuffle_foreign":
        rows[0]["foreign_condition"] = 1
    elif mutation == "dense_source":
        row = next(item for item in rows if item["condition"] == "dense_causal")
        row["original_source_hit"] = True
    else:
        rows[0]["original_source"] = 1
    _write_prediction_rows(seed_root / "predictions.jsonl.gz", rows)
    with pytest.raises(runner.ContractError):
        runner._validate_prediction_artifact(root, payload, seed_root, rung, seed, *validation_inputs)


@pytest.mark.parametrize("surface", ("checkpoint", "evaluation"))
def test_prediction_and_evaluation_evidence_are_anchored_to_real_endpoint_bytes(tmp_path: Path, runtime_modules, surface: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / f"endpoint-{surface}"
    root.mkdir()
    torch = runtime_modules.torch
    seed_root, evaluation_rows = _materialize_rung_two_endpoint_evidence(root, payload, torch)
    endpoints = runner._load_claim_endpoints(root, 2, 83, torch)
    checkpoint_by_condition = runner._checkpoint_by_condition(2, endpoints)
    evaluation_payload, eval_data_sha256 = runner._load_evaluation_evidence(root, 2, 83, torch)
    prediction_evidence = runner._validate_prediction_artifact(
        root,
        payload,
        seed_root,
        2,
        83,
        checkpoint_by_condition,
        evaluation_payload,
        eval_data_sha256,
    )
    runner._validate_evaluation_reconstruction(root, payload, 2, 83, evaluation_rows, prediction_evidence, checkpoint_by_condition, None)
    path = _checkpoint_path(root, 2, 83, "intact") if surface == "checkpoint" else _evaluation_data_path(root, 2, 83)
    with path.open("ab") as handle:
        handle.write(b"mutated")

    def validate_mutated():
        changed_endpoints = runner._load_claim_endpoints(root, 2, 83, torch)
        changed_checkpoints = runner._checkpoint_by_condition(2, changed_endpoints)
        changed_payload, changed_eval_sha256 = runner._load_evaluation_evidence(root, 2, 83, torch)
        changed_predictions = runner._validate_prediction_artifact(
            root,
            payload,
            seed_root,
            2,
            83,
            changed_checkpoints,
            changed_payload,
            changed_eval_sha256,
        )
        runner._validate_evaluation_reconstruction(root, payload, 2, 83, evaluation_rows, changed_predictions, changed_checkpoints, None)

    with pytest.raises((runner.ContractError, runner.HardAbort)):
        validate_mutated()


def test_claim_artifact_recomputes_gate_populations_from_all_wrong_predictions(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "all-wrong"
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    seed_root = root / "rung2" / "83"
    rows = _synthetic_prediction_rows(root, payload, 2, 83)
    for row in rows:
        row["prediction"] = (row["target"] + 1) % 32
        row["correct"] = False
    _write_prediction_rows(seed_root / "predictions.jsonl.gz", rows)
    checkpoint_by_condition, evaluation_payload, eval_data_sha256 = _prediction_validation_inputs(root, 2, 83)
    prediction_evidence = runner._validate_prediction_artifact(
        root,
        payload,
        seed_root,
        2,
        83,
        checkpoint_by_condition,
        evaluation_payload,
        eval_data_sha256,
    )
    evaluation_rows = [json.loads(line) for line in (seed_root / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
    with pytest.raises(runner.ContractError):
        runner._validate_evaluation_reconstruction(root, payload, 2, 83, evaluation_rows, prediction_evidence, checkpoint_by_condition, None)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("raw_remote_ids", []),
        ("effective_remote_ids", []),
        ("original_source_hits", 0),
        ("foreign_source_total", 0),
        ("query_underfill_count", 0),
        ("overflow_count", 0),
        ("selected_mask_oracle_max_error", 0.0),
    ),
)
def test_rung_two_evaluation_reconstruction_rejects_every_inapplicable_surface(tmp_path: Path, runtime_modules, field: str, replacement) -> None:
    payload = _tracked_payload()
    root = tmp_path / field
    root.mkdir()
    torch = runtime_modules.torch
    seed_root, evaluation_rows = _materialize_rung_two_endpoint_evidence(root, payload, torch)
    endpoints = runner._load_claim_endpoints(root, 2, 83, torch)
    checkpoint_by_condition = runner._checkpoint_by_condition(2, endpoints)
    evaluation_payload, eval_data_sha256 = runner._load_evaluation_evidence(root, 2, 83, torch)
    prediction_evidence = runner._validate_prediction_artifact(
        root,
        payload,
        seed_root,
        2,
        83,
        checkpoint_by_condition,
        evaluation_payload,
        eval_data_sha256,
    )
    changed = copy.deepcopy(evaluation_rows)
    changed[0][field] = replacement
    with pytest.raises(runner.ContractError):
        runner._validate_evaluation_reconstruction(root, payload, 2, 83, changed, prediction_evidence, checkpoint_by_condition, None)


def test_rung_two_prediction_is_pinned_to_source_output_when_telemetry_argmax_differs_within_parity_tolerance(runtime_modules) -> None:
    torch = runtime_modules.torch
    source_logits = torch.zeros((2, 512, 3), dtype=torch.float64)
    telemetry_logits = source_logits.clone()
    source_logits[:, 510, 0] = 1e-8
    telemetry_logits[:, 510, 1] = 1e-8
    target = torch.zeros(2, dtype=torch.long)
    predicted, matches, parity_error = runner._rung_two_source_prediction(
        torch,
        SimpleNamespace(logits=source_logits),
        SimpleNamespace(logits=telemetry_logits),
        target,
        {"seed": 83, "condition": "intact"},
    )
    assert predicted.tolist() == [0, 0]
    assert matches.tolist() == [True, True]
    assert parity_error == 1e-8
    telemetry_logits[:, 510, 1] = 1e-4
    with pytest.raises(runner.HardAbort) as caught:
        runner._rung_two_source_prediction(
            torch,
            SimpleNamespace(logits=source_logits),
            SimpleNamespace(logits=telemetry_logits),
            target,
            {"seed": 83, "condition": "intact"},
        )
    assert caught.value.reason_code == "endpoint_inconsistency"


def test_routing_artifact_requires_full_production_query_cardinality_and_order(tmp_path: Path, runtime_modules) -> None:
    payload = _tracked_payload()
    root = tmp_path / "routing-closure"
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    seed_root = root / "rung1" / "11"
    torch = runtime_modules.torch
    evaluation_payload = runner.generate_rung_one_batch(400011, 512, torch)
    rows, evaluation_rows, prediction_evidence, checkpoint_by_condition = _full_routing_fixture(root, torch=torch, evaluation_payload=evaluation_payload)
    (seed_root / "routing.jsonl.gz").unlink()
    runner._write_canonical_gzip(seed_root / "routing.jsonl.gz", rows)
    evidence = runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)
    assert evidence["overflow_count"] == 0
    assert all(len(queries) == 512 for queries in evidence["query_by_condition"].values())
    random_payload = runner.generate_random_routes(500011, 512, torch)
    random_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    random_routes[:, 126, 0] = torch.tensor(random_payload["routes"], dtype=torch.long)
    runner._save_torch_artifact(
        root / "data" / "r1_random_routes_11.pt",
        {"seed": 500011, "routes": random_routes, "payload": random_payload, "payload_sha256": runner.canonical_json_sha256(random_payload)},
        torch,
    )
    acquisition_rows = [row for row in evidence["records"] if row["phase"] == "route_acquisition" and row["row_kind"] == "query_example" and row["block"] == 4]
    raw_query = [row["raw_remote_ids"] for row in acquisition_rows]
    required_source = list(evaluation_payload["required_source"])
    exclusion_payload = runner.generate_source_exclusion_routes(510011, raw_query, required_source, torch)
    raw_routes = torch.full((512, 128, 1, 2), -1, dtype=torch.long)
    raw_routes[:, 126, 0] = torch.tensor(raw_query, dtype=torch.long)
    exclusion_routes = torch.full_like(raw_routes, -1)
    exclusion_routes[:, 126, 0] = torch.tensor(exclusion_payload["routes"], dtype=torch.long)
    runner._save_torch_artifact(
        root / "data" / "r1_source_exclusion_11.pt",
        {
            "seed": 510011,
            "raw_query_routes": raw_routes,
            "required_source": torch.tensor(required_source, dtype=torch.long),
            "routes": exclusion_routes,
            "payload": exclusion_payload,
            "payload_sha256": runner.canonical_json_sha256(exclusion_payload),
        },
        torch,
    )
    data_evidence = runner._validate_rung_one_data_artifacts(root, 11, evaluation_payload, evidence, torch)
    assert data_evidence["postcheckpoint_assertions"] is True
    assert data_evidence["route_acquisition_examples"] == 512
    assert data_evidence["source_exclusion_examples"] == 512
    assert (data_evidence["same_condition"], data_evidence["changed_condition"]) == (116, 396)
    changed_evidence = copy.deepcopy(evidence)
    changed_route = changed_evidence["query_by_condition"]["required_source_excluded"][0]["effective_remote_ids"]
    changed_evidence["query_by_condition"]["required_source_excluded"][0]["effective_remote_ids"] = list(reversed(changed_route))
    with pytest.raises(runner.ContractError):
        runner._validate_rung_one_data_artifacts(root, 11, evaluation_payload, changed_evidence, torch)
    mutations = (
        rows[:1] + rows[2:],
        [*rows[:1], rows[2], rows[1], *rows[3:]],
        [row for row in rows if row["row_kind"] == "call_summary"],
    )
    for changed in mutations:
        (seed_root / "routing.jsonl.gz").unlink()
        runner._write_canonical_gzip(seed_root / "routing.jsonl.gz", changed)
        with pytest.raises(runner.ContractError):
            runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)


@pytest.mark.parametrize(
    ("phase", "row_kind", "condition", "phase_key"),
    (
        ("training", "call_summary", None, "training.call_summary"),
        ("training", "query_example", None, "training.query_example"),
        ("route_acquisition", "call_summary", None, "route_acquisition.call_summary"),
        ("route_acquisition", "query_example", None, "route_acquisition.query_example"),
        ("evaluation", "call_summary", "intact", "evaluation.call_summary"),
        ("evaluation", "query_example", "intact", "evaluation.query_example_non_carry_shuffle"),
        ("evaluation", "query_example", "carry_shuffle", "evaluation.query_example_carry_shuffle"),
    ),
)
def test_routing_nullability_partition_is_exhaustive(phase, row_kind, condition, phase_key) -> None:
    schema = _tracked_payload()["artifacts"]["schemas"]["routing_row"]
    row = {key: 1 for key in schema["exact_keys"]}
    row.update({"phase": phase, "row_kind": row_kind, "condition": condition})
    expected_null = set(schema["row_kind_null_fields"][row_kind]) | set(schema["phase_and_row_kind_additional_null_fields"][phase_key])
    for field in expected_null:
        row[field] = None
    runner._validate_routing_row_nullability(row, schema)
    missing_null = dict(row)
    missing_null[next(iter(expected_null))] = 1
    with pytest.raises(runner.ContractError):
        runner._validate_routing_row_nullability(missing_null, schema)
    invented_null = dict(row)
    invented_null[next(field for field in schema["exact_keys"] if field not in expected_null)] = None
    with pytest.raises(runner.ContractError):
        runner._validate_routing_row_nullability(invented_null, schema)


def test_routing_workspace_is_derived_from_frozen_call_geometry_for_every_routed_model(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "routing-workspace"
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    seed_root = root / "rung1" / "11"
    rows, evaluation_rows, prediction_evidence, checkpoint_by_condition = _full_routing_fixture(root)
    calls = [row for row in rows if row["row_kind"] == "call_summary"]
    for call in calls:
        spec_batch_size = 16 if call["phase"] == "training" else 32
        assert call["route_workspace_bytes"] == _expected_routing_workspace_bytes(call["model"], call["block"], spec_batch_size)
    routed_models = {
        call["model"]
        for call in calls
        if call["block"] == 4 and call["route_workspace_bytes"] > 0
    }
    assert routed_models == {"all_eligible_donor", "selected", "donor", "clone"}
    routing_path = seed_root / "routing.jsonl.gz"
    routing_path.unlink()
    runner._write_canonical_gzip(routing_path, rows)
    runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)
    for model in sorted(routed_models):
        index = next(
            offset
            for offset, row in enumerate(rows)
            if row["row_kind"] == "call_summary" and row["block"] == 4 and row["model"] == model
        )
        expected = rows[index]["route_workspace_bytes"]
        for replacement in (0, expected + 76):
            changed = list(rows)
            changed[index] = {**rows[index], "route_workspace_bytes": replacement}
            routing_path.unlink()
            runner._write_canonical_gzip(routing_path, changed)
            with pytest.raises(runner.ContractError):
                runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)


def test_routing_artifact_reconstructs_bypass_loads_and_posting_counts(tmp_path: Path) -> None:
    payload = _tracked_payload()
    root = tmp_path / "routing-reconstruction"
    root.mkdir()
    _materialize_synthetic_gate_package(root, payload)
    seed_root = root / "rung1" / "11"
    rows, evaluation_rows, prediction_evidence, checkpoint_by_condition = _full_routing_fixture(root)
    routing_path = seed_root / "routing.jsonl.gz"
    routing_path.unlink()
    runner._write_canonical_gzip(routing_path, rows)
    evidence = runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)
    selected_call = next(
        row
        for row in evidence["records"]
        if row["row_kind"] == "call_summary" and row["model"] == "selected" and row["block"] == 4
    )
    assert len(selected_call["canonical_bypass_ids"]) == 24
    assert selected_call["canonical_bypass_ids"][0] == {"position": 0, "remote_limit": 0, "raw_remote_ids": [-1, -1], "effective_remote_ids": [-1, -1]}
    assert selected_call["canonical_bypass_ids"][-1] == {"position": 23, "remote_limit": 2, "raw_remote_ids": [0, 1], "effective_remote_ids": [-1, -1]}
    assert selected_call["block_load_histogram"] == [{"load": 1, "bucket_count": 256}]
    assert sum(row["search_row_count"] for row in selected_call["valid_posting_histogram"]) == 1664
    all_eligible_call = next(
        row
        for row in evidence["records"]
        if row["row_kind"] == "call_summary" and row["model"] == "all_eligible_donor" and row["block"] == 4
    )
    assert len(all_eligible_call["canonical_bypass_ids"]) == 128
    assert all_eligible_call["canonical_bypass_ids"][125]["effective_remote_ids"] == [-1] * 15
    assert all_eligible_call["canonical_bypass_ids"][126]["effective_remote_ids"] == list(range(15))
    assert all_eligible_call["canonical_bypass_ids"][127]["effective_remote_ids"] == [-1] * 15
    target_index = rows.index(selected_call)
    mutations = []
    paired_counter = copy.deepcopy(rows)
    paired_counter[target_index]["posting_reads"] += 1
    paired_counter[target_index]["candidate_blocks"] += 1
    mutations.append(paired_counter)
    false_empty = copy.deepcopy(rows)
    false_empty[target_index]["posting_reads"] = 0
    false_empty[target_index]["candidate_blocks"] = 0
    false_empty[target_index]["valid_posting_histogram"] = [{"valid_posting_count": 0, "search_row_count": 1664}]
    mutations.append(false_empty)
    wrong_load = copy.deepcopy(rows)
    wrong_load[target_index]["max_bucket_load"] = 2
    mutations.append(wrong_load)
    wrong_bypass = copy.deepcopy(rows)
    wrong_bypass[target_index]["canonical_bypass_ids"][16]["raw_remote_ids"] = [1, 0]
    mutations.append(wrong_bypass)
    query_index = next(
        index
        for index, row in enumerate(rows)
        if row["forward_sequence"] == selected_call["forward_sequence"]
        and row["block"] == 4
        and row["row_kind"] == "query_example"
    )
    mislabeled_query = copy.deepcopy(rows)
    mislabeled_query[query_index]["canonical_bypass_ids"] = [
        {"position": 126, "remote_limit": 15, "raw_remote_ids": [0, 1], "effective_remote_ids": [0, 1]}
    ]
    mutations.append(mislabeled_query)
    for changed in mutations:
        routing_path.unlink()
        runner._write_canonical_gzip(routing_path, changed)
        with pytest.raises(runner.ContractError):
            runner._validate_routing_artifact(root, payload, seed_root, 11, evaluation_rows, prediction_evidence, checkpoint_by_condition)


@pytest.mark.parametrize(
    "mutation",
    ("positive", "state", "manifest_identity", "sentinel", "evaluation", "oracle", "oracle_identity"),
)
def test_terminal_selected_oracle_reconstruction_rejects_every_self_consistent_evidence_branch(tmp_path: Path, runtime_modules, mutation: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / mutation
    root.mkdir()
    evaluation_rows, _, paths = _selected_oracle_fixture(root, payload, runtime_modules)
    runner._validate_selected_oracle_evidence(root, payload, 11, evaluation_rows, runtime_modules)
    if mutation == "positive":
        return
    if mutation in {"state", "manifest_identity"}:
        value = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if mutation == "state":
            value["state_tensors"][0]["sha256"] = "0" * 64
            value["state_sha256"] = runner.canonical_json_sha256(value["state_tensors"])
        else:
            value["construction_seed"] = 23
        paths["manifest"].write_bytes(runner.canonical_json_bytes(value))
    elif mutation == "sentinel":
        value = json.loads(paths["sentinel"].read_text(encoding="utf-8"))
        value["inputs"][0][0][0] += 1.0
        paths["sentinel"].write_bytes(runner.canonical_json_bytes(value))
    elif mutation == "evaluation":
        evaluation_rows[0]["estimate"] += 1e-6
        evaluation_rows[0]["selected_mask_oracle_max_error"] += 1e-6
    else:
        value = json.loads(paths["oracle"].read_text(encoding="utf-8"))
        if mutation == "oracle":
            value["max_error"] += 1e-6
        else:
            value["run_id"] = "foreign-run"
        paths["oracle"].write_bytes(runner.canonical_json_bytes(value))
    with pytest.raises((runner.ContractError, runner.HardAbort)):
        runner._validate_selected_oracle_evidence(root, payload, 11, evaluation_rows, runtime_modules)


def test_claim_artifact_package_walks_every_seed_and_cross_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _tracked_payload()
    root = tmp_path / "claim-package"
    root.mkdir()
    populations, _ = _claim_package_shell(root, payload)
    calls = []
    _stub_claim_package_semantics(monkeypatch, root, populations, calls)
    runner.validate_claim_artifact_package(root, payload, SimpleNamespace(torch=None, model_module=None))
    assert calls[0] == "ledger"
    assert [row for row in calls if isinstance(row, tuple) and row[0] == "train"] == [("train", seed) for seed in (11, 23, 37, 53, 71, 83)]
    assert [row for row in calls if isinstance(row, tuple) and row[0] == "predictions"] == [("predictions", seed) for seed in (11, 23, 37, 53, 71, 83)]
    assert [row for row in calls if isinstance(row, tuple) and row[0] == "routing"] == [("routing", seed) for seed in (11, 23, 37, 53, 71)]
    assert [row for row in calls if isinstance(row, tuple) and row[0] == "oracle"] == [("oracle", seed) for seed in (11, 23, 37, 53, 71)]
    assert calls[-1] == "gates"


@pytest.mark.parametrize(
    "mutation",
    (
        "state_schema",
        "state_noncanonical",
        "state_identity",
        "intervention_schema",
        "intervention_identity",
        "parity_identity",
        "parity_detail",
        "resource_identity",
        "resource_refs",
        "accounting_identity",
        "accounting_refs",
    ),
)
def test_claim_artifact_package_rejects_schema_serialization_and_cross_reference_mutations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    payload = _tracked_payload()
    root = tmp_path / mutation
    root.mkdir()
    populations, detail_paths = _claim_package_shell(root, payload)
    _stub_claim_package_semantics(monkeypatch, root, populations)
    seed_root = root / "rung1" / "11"
    if mutation in {"state_schema", "state_noncanonical", "state_identity"}:
        path = seed_root / "state_stats.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "state_schema":
            value["extra"] = None
            path.write_bytes(runner.canonical_json_bytes(value))
        elif mutation == "state_identity":
            value["construction_seed"] = 23
            path.write_bytes(runner.canonical_json_bytes(value))
        else:
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    elif mutation in {"intervention_schema", "intervention_identity"}:
        path = seed_root / "intervention_deltas.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "intervention_schema":
            value["extra"] = None
        else:
            value["claim_seed"] = 23
        path.write_bytes(runner.canonical_json_bytes(value))
    elif mutation == "parity_identity":
        path = seed_root / "parity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["rung"] = 2
        path.write_bytes(runner.canonical_json_bytes(value))
    elif mutation == "parity_detail":
        detail_paths[0].write_bytes(detail_paths[0].read_bytes() + b" ")
    elif mutation in {"resource_identity", "resource_refs"}:
        path = seed_root / "resource_refs.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "resource_identity":
            value["run_id"] = "foreign-run"
        else:
            value["sample_ids"] = [1]
        path.write_bytes(runner.canonical_json_bytes(value))
    elif mutation in {"accounting_identity", "accounting_refs"}:
        path = seed_root / "accounting.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "accounting_identity":
            value["schema_version"] = "foreign-schema"
        else:
            value["models"][0]["resource_sample_ids"] = [1]
        path.write_bytes(runner.canonical_json_bytes(value))
    with pytest.raises(runner.ContractError):
        runner.validate_claim_artifact_package(root, payload, SimpleNamespace(torch=None, model_module=None))


def test_execute_run_refuses_and_cleans_failure_before_activation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)

    def refuse(*args):
        raise runner.ContractError("prepilot failure")

    monkeypatch.setattr(runner, "build_shared_prepilot_base", refuse)
    with pytest.raises(runner.InitializationRefusal):
        runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)])
    assert calls == ["signal_install", "signal_terminal"]
    assert not entry.run_root.exists()
    assert not any(entry.run_root.parent.glob(".run-id.initializing.*"))
    assert signals.terminal is True


@pytest.mark.parametrize("mode", ("clean", "cleanup_failure", "foreign_replacement"))
def test_execute_run_initialization_staging_cleanup_is_identity_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    entry = runner.EntryConfiguration(results / "run-id", "run-id")
    signals = SimpleNamespace(active=False, terminal=False)

    def install():
        signals.active = True

    def deactivate():
        signals.active = False
        signals.terminal = True

    signals.install = install
    signals.deactivate_terminal = deactivate
    monkeypatch.setattr(runner, "RESULTS_PARENT", results)
    monkeypatch.setattr(runner, "SignalController", lambda: signals)
    staging = results / f".run-id.initializing.{runner.os.getpid()}"
    moved = results / "owned-staging-moved"
    failed = [False]

    def fsync(path):
        if not failed[0] and Path(path) == results:
            failed[0] = True
            if mode == "foreign_replacement":
                staging.rename(moved)
                staging.mkdir()
                (staging / "foreign").write_bytes(b"foreign-staging\n")
            raise OSError("injected initialization fsync")

    monkeypatch.setattr(runner, "fsync_directory", fsync)
    if mode == "cleanup_failure":
        monkeypatch.setattr(runner, "_remove_tree_and_fsync", lambda path: (_ for _ in ()).throw(OSError("injected cleanup failure")))
    expected = runner.InitializationRefusal if mode == "clean" else runner.UnrecoverableOrphan
    with pytest.raises(expected):
        runner.execute_run(entry, _tracked_payload(), runner.RuntimeModules(torch=None, model_module=None), ["runner"])
    assert signals.terminal is True
    if mode == "clean":
        assert not staging.exists()
    elif mode == "cleanup_failure":
        assert staging.is_dir()
    else:
        assert (staging / "foreign").read_bytes() == b"foreign-staging\n"


def test_execute_run_pilot_stop_never_enters_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: calls.append("pilot_transition") or transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: calls.append("pilot_stop") or {"decision": "stop"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: pytest.fail("claim transition reached"))
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: pytest.fail("claim data reached"))
    monkeypatch.setattr(runner, "run_claim_workers", lambda *args: pytest.fail("claim workers reached"))

    def terminalize(*args, **kwargs):
        calls.append("terminalize")
        signals.deactivate_terminal()
        return SimpleNamespace(terminal=True)

    monkeypatch.setattr(runner, "write_sha256s_terminal", terminalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 0
    assert calls == ["signal_install", "prepilot", "activate", "pilot_transition", "pilot_stop", "terminalize", "signal_terminal"]
    assert signals.terminal is True


def test_execute_run_preserves_pilot_stop_terminal_closure_after_post_terminal_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "stop"})
    terminal_bytes = {}

    def terminalize(run_root, expected_paths, signals):
        terminal_bytes["pilot"], terminal_bytes["checksum"] = _write_terminal_fixture(
            run_root,
            "run/pilot.json",
            {"decision": "stop"},
            signals,
        )
        raise RuntimeError("post-terminal pilot fault")

    monkeypatch.setattr(runner, "write_sha256s_terminal", terminalize)
    monkeypatch.setattr(runner, "finalize_hard_abort", lambda *args, **kwargs: pytest.fail("terminal pilot closure entered abort finalization"))
    assert runner.execute_run(
        entry,
        payload,
        runtime,
        ["runner", "--run-root", str(entry.run_root)],
    ) == 0
    assert (entry.run_root / "run" / "pilot.json").read_bytes() == terminal_bytes["pilot"]
    assert (entry.run_root / "SHA256SUMS").read_bytes() == terminal_bytes["checksum"]
    assert signals.terminal is True


def test_execute_run_clean_claim_visits_every_production_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    accounting = runner.AttemptAccounting(1, 1, 2048, 2048, (11,), (11,), {11: 1}, ())
    claim_result = {"accounting": accounting, "resource_sampling_end_monotonic_ns": 400, "resource_final_sample_id": 3}
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: calls.append("pilot_transition") or pilot_transition)
    pilot_runner = lambda *args: calls.append("pilot_proceed") or {"decision": "proceed"}
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: pytest.fail("default pilot runner reached"))
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: calls.append("claim_transition") or claim_transition)
    monkeypatch.setattr(
        runner,
        "establish_training_start_plan_barrier",
        lambda run_root, anchors, controller: calls.append("training_start_barrier") or (anchors, 123456),
    )
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: calls.append("claim_data") or {})
    monkeypatch.setattr(runner, "_verify_active_frozen_hashes", lambda *args: calls.append("frozen_hashes"))

    def claim_workers(*args):
        assert args[-1] == 123456
        calls.append("claim_workers")
        return claim_result

    monkeypatch.setattr(runner, "run_claim_workers", lambda *args: pytest.fail("default claim runner reached"))

    def finalize(*args):
        assert args[4] == 123456
        calls.append("clean_finalize")
        signals.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_clean_claim", finalize)
    assert runner.execute_run(
        entry,
        payload,
        runtime,
        ["runner", "--run-root", str(entry.run_root)],
        resource_pilot_runner=pilot_runner,
        claim_runner=claim_workers,
    ) == 0
    assert calls == [
        "signal_install",
        "prepilot",
        "activate",
        "frozen_hashes",
        "pilot_transition",
        "pilot_proceed",
        "frozen_hashes",
        "claim_transition",
        "training_start_barrier",
        "claim_data",
        "frozen_hashes",
        "claim_workers",
        "clean_finalize",
        "signal_terminal",
    ]


def test_execute_run_barrier_failure_never_reaches_data_workers_or_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: pilot_transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "proceed"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: claim_transition)

    def block(*args):
        calls.append("training_start_barrier")
        raise runner.HardAbort(
            "artifact_inconsistency",
            {"surface": "training_start_plan_review_timeout", "training_start_state": "awaiting_review"},
            4242,
        )

    monkeypatch.setattr(runner, "establish_training_start_plan_barrier", block)
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: pytest.fail("claim data reached"))
    monkeypatch.setattr(runner, "run_claim_workers", lambda *args: pytest.fail("claim workers reached"))
    captured = {}

    def finalize(*args):
        captured["phase"] = args[4]
        captured["context"] = args[5]
        captured["training_start_state"] = args[11]
        signals.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_hard_abort", finalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 2
    assert calls[-2:] == ["training_start_barrier", "signal_terminal"]
    assert captured == {
        "phase": "claim",
        "context": {"surface": "training_start_plan_review_timeout", "training_start_state": "awaiting_review"},
        "training_start_state": "awaiting_review",
    }


def test_execute_run_live_plan_ambiguity_escapes_without_terminal_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, payload, runtime, _, _ = _stub_execute_activation(tmp_path, monkeypatch)
    repository = tmp_path / "repository"
    plan = repository / "neuroloc" / "wiki" / "PROJECT_PLAN.md"
    plan.parent.mkdir(parents=True)
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: pilot_transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "proceed"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: claim_transition)
    snapshots = {}

    def snapshot(root):
        return tuple(
            sorted(
                (
                    path.relative_to(root).as_posix(),
                    None if path.is_dir() else path.read_bytes(),
                )
                for path in root.rglob("*")
            )
        )

    def barrier(run_root, anchors, controller):
        run_directory = run_root / "run"
        run_directory.mkdir(exist_ok=True)
        (run_root / runner.LAUNCH_PROJECT_PLAN_PATH).write_bytes(b"launch-plan\n")
        plan.write_bytes(b"foreign-live-plan\n")
        snapshots["tree"] = snapshot(run_root)
        snapshots["live"] = plan.read_bytes()
        return runner._training_start_state(run_root)

    monkeypatch.setattr(runner, "establish_training_start_plan_barrier", barrier)
    monkeypatch.setattr(runner, "finalize_hard_abort", lambda *args, **kwargs: pytest.fail("live ambiguity entered hard-abort finalization"))
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)])
    assert snapshot(entry.run_root) == snapshots["tree"]
    assert plan.read_bytes() == snapshots["live"]
    assert not (entry.run_root / "ABORTED.json").exists()
    assert not (entry.run_root / "SHA256SUMS").exists()


def test_execute_run_signal_after_started_barrier_never_begins_claim_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: pilot_transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "proceed"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: claim_transition)

    def barrier(run_root, anchors, controller):
        calls.append("training_start_barrier")
        controller.pending_signal = 15
        return anchors, 123456

    reached = []
    monkeypatch.setattr(runner, "establish_training_start_plan_barrier", barrier)
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: reached.append("claim_data"))
    monkeypatch.setattr(runner, "run_claim_workers", lambda *args: reached.append("claim_workers"))
    captured = {}

    def finalize(*args):
        captured["reason_code"] = args[3]
        captured["training_start_state"] = args[11]
        signals.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_hard_abort", finalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 2
    assert reached == []
    assert captured == {"reason_code": "signal_or_interruption", "training_start_state": "started"}


def test_prepare_claim_data_checks_signal_before_creating_data_directory(tmp_path: Path) -> None:
    root = tmp_path / "run-id"
    root.mkdir()
    signals = SimpleNamespace(pending_signal=15)
    anchors = runner.FrozenManifestAnchors(())
    with pytest.raises(runner.HardAbort) as caught:
        runner.prepare_claim_data(root, runner.RuntimeModules(torch=None, model_module=None), {}, anchors, signals, 0)
    assert caught.value.reason_code == "signal_or_interruption"
    assert not (root / "data").exists()


def test_run_claim_workers_checks_signal_before_resource_sample_or_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {"run/resources.jsonl": SimpleNamespace()}, 100, None)
    signals = SimpleNamespace(pending_signal=15)
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: pytest.fail("resource sample reached"))
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: pytest.fail("worker spawn reached"))
    with pytest.raises(runner.HardAbort) as caught:
        runner.run_claim_workers(tmp_path, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition, 0)
    assert caught.value.reason_code == "signal_or_interruption"


def test_run_claim_workers_rechecks_after_sample_zero_before_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signals = SimpleNamespace(pending_signal=None)
    spawn_calls = []
    guard_stages = []

    class Writer:
        def append(self, row):
            return SimpleNamespace(acknowledged=True, reason_code=None)

        def validate_committed_prefix(self):
            signals.pending_signal = 15
            return ()

    def guard(run_root, anchors, observed_signals, claim_start, stage):
        guard_stages.append(stage)
        if observed_signals.pending_signal is not None:
            raise runner.HardAbort("signal_or_interruption", {"stage": stage})

    transition = runner.TransitionResult(
        "claim",
        "ready",
        CLAIM_LEDGER_PATHS,
        {"run/resources.jsonl": Writer()},
        100,
        None,
    )
    monkeypatch.setattr(runner, "final_claim_guard", guard)
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: {})
    monkeypatch.setattr(runner, "claim_resource_observations", lambda rows: ())
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: spawn_calls.append(args))
    with pytest.raises(runner.HardAbort) as caught:
        runner.run_claim_workers(tmp_path, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition, 0)
    assert caught.value.reason_code == "signal_or_interruption"
    assert guard_stages == ["before_claim_workers", "before_claim_worker_spawn"]
    assert spawn_calls == []


@pytest.mark.parametrize("failure", ("signal", "frozen_drift"))
def test_resource_pilot_rechecks_sample_zero_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / failure / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    anchors = runner.capture_frozen_manifest_anchors(root)
    signals = SimpleNamespace(pending_signal=None)
    spawn_calls = []

    class Writer:
        def validate_committed_prefix(self):
            if failure == "signal":
                signals.pending_signal = 15
            else:
                (root / "run" / "prereg.json").write_bytes(b"drift\n")
            return (_resource_row(root.name, "pilot", 100),)

    transition = runner.TransitionResult(
        "pilot",
        "ready",
        ("run/pilot_resources.jsonl",),
        {"run/pilot_resources.jsonl": Writer()},
        100,
        None,
    )
    monkeypatch.setattr(
        runner,
        "_verify_active_frozen_hashes",
        lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors),
    )
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: spawn_calls.append(args))
    with pytest.raises(runner.HardAbort) as caught:
        runner.run_resource_pilot(root, _tracked_payload(), anchors, signals, transition)
    assert caught.value.reason_code == ("signal_or_interruption" if failure == "signal" else "frozen_hash_change")
    assert spawn_calls == []


@pytest.mark.parametrize("failure", ("signal", "frozen_drift"))
def test_resource_pilot_spawned_children_remain_gated_until_postspawn_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / failure / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    anchors = runner.capture_frozen_manifest_anchors(root)
    signals = _BarrierSignals()
    event = SimpleNamespace(set_called=False, set=lambda: setattr(event, "set_called", True))
    processes = tuple(SimpleNamespace(pid=101 + index, name=f"modular-pilot-{worker}") for index, worker in enumerate(("A", "B")))
    quiesced = []

    class Writer:
        def validate_committed_prefix(self):
            return (_resource_row(root.name, "pilot", 100),)

    class Connection:
        def close(self):
            return None

    class Context:
        def get_context(self, method):
            assert method == "spawn"
            return self

        def Barrier(self, parties):
            assert parties == 2
            return SimpleNamespace()

        def Event(self):
            return event

    def spawn(context, specifications):
        assert all(specification["args"][-1] is event for specification in specifications)
        if failure == "signal":
            signals.pending_signal = 15
        else:
            (root / "run" / "prereg.json").write_bytes(b"drift\n")
        return processes, {"A": Connection(), "B": Connection()}

    def quiesce(error, observed_processes):
        quiesced.append(tuple(observed_processes))
        return error

    transition = runner.TransitionResult(
        "pilot",
        "ready",
        ("run/pilot_resources.jsonl",),
        {"run/pilot_resources.jsonl": Writer()},
        100,
        None,
    )
    real_import = runner.importlib.import_module
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: Context() if name == "multiprocessing" else real_import(name))
    monkeypatch.setattr(
        runner,
        "_verify_active_frozen_hashes",
        lambda observed_root, observed_anchors: runner.verify_manifest_anchors(observed_root, observed_anchors),
    )
    monkeypatch.setattr(runner, "spawn_worker_processes", spawn)
    monkeypatch.setattr(runner, "quiesce_after_primary_latch", quiesce)
    with pytest.raises(runner.HardAbort) as caught:
        runner.run_resource_pilot(root, _tracked_payload(), anchors, signals, transition)
    assert caught.value.reason_code == ("signal_or_interruption" if failure == "signal" else "frozen_hash_change")
    assert event.set_called is False
    assert quiesced == [processes]


@pytest.mark.parametrize("parent", ("pilot", "claim"))
def test_postspawn_orphan_quiesces_workers_closes_transports_and_keeps_gate_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
) -> None:
    root = tmp_path / parent
    root.mkdir()
    signals = _BarrierSignals()
    event = SimpleNamespace(set_called=False, set=lambda: setattr(event, "set_called", True))
    processes = tuple(SimpleNamespace(pid=101 + index, name=f"modular-{parent}-{worker}") for index, worker in enumerate(("A", "B")))
    quiesced = []

    class Connection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    connections = {"A": Connection(), "B": Connection()}

    class Context:
        def Barrier(self, parties):
            return SimpleNamespace()

        def Event(self):
            return event

    class PilotWriter:
        def validate_committed_prefix(self):
            return (_resource_row(root.name, "pilot", 100),)

    class ClaimWriter:
        def append(self, row):
            return SimpleNamespace(acknowledged=True, reason_code=None)

        def validate_committed_prefix(self):
            return ()

    real_import = runner.importlib.import_module
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(get_context=lambda method: Context()) if name == "multiprocessing" else real_import(name))
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: (processes, connections))
    monkeypatch.setattr(runner, "quiesce_worker_processes", lambda observed: quiesced.append(tuple(observed)))
    if parent == "pilot":
        guards = [0]

        def guard(*args):
            guards[0] += 1
            if guards[0] == 2:
                raise runner.UnrecoverableOrphan("injected pilot postspawn orphan")

        monkeypatch.setattr(runner, "final_frozen_guard", guard)
        transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": PilotWriter()}, 100, None)
        call = lambda: runner.run_resource_pilot(root, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition)
    else:
        guards = [0]

        def guard(*args):
            guards[0] += 1
            if guards[0] == 3:
                raise runner.UnrecoverableOrphan("injected claim postspawn orphan")

        monkeypatch.setattr(runner, "final_claim_guard", guard)
        monkeypatch.setattr(runner, "_resource_sample", lambda *args: {"monotonic_ns": 100})
        monkeypatch.setattr(runner, "claim_resource_observations", lambda rows: ())
        monkeypatch.setattr(runner, "next_resource_sample_monotonic_ns", lambda row: 200)
        transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {"run/resources.jsonl": ClaimWriter()}, 100, None)
        call = lambda: runner.run_claim_workers(root, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition, 0)
    with pytest.raises(runner.UnrecoverableOrphan):
        call()
    assert event.set_called is False
    assert quiesced == [processes]
    assert all(connection.closed for connection in connections.values())


def test_orphan_exception_mapping_and_best_effort_abort_never_downgrade() -> None:
    orphan = runner.UnrecoverableOrphan("injected orphan")
    child_message = runner._child_failure_message(orphan, "A")
    assert child_message == {"kind": "unrecoverable_orphan", "worker": "A"}
    assert runner.validate_claim_worker_message(child_message, "A", set()) == "unrecoverable_orphan"
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._handle_claim_worker_message(child_message, "A", _FakeConnection(), set(), {}, {}, {}, _BarrierSignals())
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        runner.failure_observation_from_exception(orphan, "artifact_inconsistency")
    assert caught.value is orphan
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        runner.parent_worker_failure_observation(orphan, "A", False)
    assert caught.value is orphan

    class Writer:
        _closed = False
        _descriptor = 1

        def validate_committed_prefix(self):
            raise orphan

    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        runner.best_effort_abort_resource_sample(
            Path("run"),
            "pilot",
            {"run/pilot_resources.jsonl": Writer()},
            100,
            accounting,
            "resource_sampler_failure",
        )
    assert caught.value is orphan


def test_claim_parent_propagates_child_orphan_message_after_quiescence_and_transport_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "child-orphan"
    root.mkdir()
    signals = _BarrierSignals()
    processes = tuple(SimpleNamespace(pid=101 + index, name=f"modular-claim-{worker}") for index, worker in enumerate(("A", "B")))
    quiesced = []

    class Connection:
        def __init__(self, messages):
            self.messages = list(messages)
            self.closed = False

        def poll(self, timeout):
            return bool(self.messages)

        def recv(self):
            return self.messages.pop(0)

        def close(self):
            self.closed = True

    connections = {
        "A": Connection(({"kind": "unrecoverable_orphan", "worker": "A"},)),
        "B": Connection(()),
    }

    class Context:
        def Event(self):
            return SimpleNamespace(set=lambda: None)

    class Writer:
        def append(self, row):
            return SimpleNamespace(acknowledged=True, reason_code=None)

        def validate_committed_prefix(self):
            return ()

    real_import = runner.importlib.import_module
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(get_context=lambda method: Context()) if name == "multiprocessing" else real_import(name))
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: (processes, connections))
    monkeypatch.setattr(runner, "quiesce_worker_processes", lambda observed: quiesced.append(tuple(observed)))
    monkeypatch.setattr(runner, "worker_exit_observations", lambda *args: (False, ()))
    monkeypatch.setattr(runner, "final_claim_guard", lambda *args: None)
    monkeypatch.setattr(runner, "_training_start_state", lambda observed_root: ("started", {}))
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: {"monotonic_ns": 100})
    monkeypatch.setattr(runner, "claim_resource_observations", lambda rows: ())
    monkeypatch.setattr(runner, "next_resource_sample_monotonic_ns", lambda row: 1000)
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 101)
    transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {"run/resources.jsonl": Writer()}, 100, None)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.run_claim_workers(root, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition, 0)
    assert quiesced == [processes]
    assert all(connection.closed for connection in connections.values())


def test_lost_child_orphan_message_uses_distinct_exit_status() -> None:
    orphan = runner.UnrecoverableOrphan("injected child storage orphan")

    class Gate:
        def wait(self):
            raise orphan

    class Connection:
        def __init__(self):
            self.closed = False

        def send(self, value):
            raise BrokenPipeError("injected lost orphan message")

        def close(self):
            self.closed = True

    connection = Connection()
    with pytest.raises(SystemExit) as caught:
        runner._claim_worker("A", "run", Gate(), connection)
    assert caught.value.code == runner.WORKER_ORPHAN_EXIT_CODE
    assert connection.closed is True
    process = SimpleNamespace(pid=101, exitcode=runner.WORKER_ORPHAN_EXIT_CODE, join=lambda timeout=0: None)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.worker_exit_observations((process,), set(), {101: "A"})


def test_child_close_failure_does_not_mask_reserved_orphan_exit() -> None:
    orphan = runner.UnrecoverableOrphan("injected child storage orphan")

    class Gate:
        def wait(self):
            raise orphan

    class Connection:
        def __init__(self):
            self.messages = []

        def send(self, value):
            self.messages.append(value)

        def close(self):
            raise OSError("injected child close failure")

    connection = Connection()
    with pytest.raises(SystemExit) as caught:
        runner._claim_worker("A", "run", Gate(), connection)
    assert caught.value.code == runner.WORKER_ORPHAN_EXIT_CODE
    assert connection.messages == [{"kind": "unrecoverable_orphan", "worker": "A"}]


def test_join_observation_failure_preserves_visible_orphan_exit() -> None:
    def fail_join(timeout=0):
        raise OSError("injected join observation failure")

    process = SimpleNamespace(pid=101, exitcode=runner.WORKER_ORPHAN_EXIT_CODE, join=fail_join)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.worker_exit_observations((process,), set(), {101: "A"})


@pytest.mark.parametrize("final_exitcode", (runner.WORKER_ORPHAN_EXIT_CODE, None))
def test_failed_worker_observation_rechecks_or_orphans_unknown_status(final_exitcode) -> None:
    class Process:
        pid = 101

        def __init__(self):
            self.reads = 0

        @property
        def exitcode(self):
            self.reads += 1
            return None if self.reads == 1 else final_exitcode

        def join(self, timeout=0):
            raise OSError("injected worker observation failure")

    process = Process()
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.worker_exit_observations((process,), set(), {101: "A"})
    assert process.reads >= 2


def test_postquiescence_exit_scan_catches_late_orphan_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=101, exitcode=None)

    def quiesce(processes):
        assert processes == (process,)
        process.exitcode = runner.WORKER_ORPHAN_EXIT_CODE

    monkeypatch.setattr(runner, "quiesce_worker_processes", quiesce)
    error = runner.HardAbort("worker_exit", {"worker": "A"}, 100)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.quiesce_after_primary_latch(error, (process,))


@pytest.mark.parametrize("parent", ("pilot", "claim"))
def test_parent_ledger_orphan_quiesces_workers_and_closes_transports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent: str,
) -> None:
    root = tmp_path / parent
    root.mkdir()
    signals = _BarrierSignals()
    processes = tuple(SimpleNamespace(pid=101 + index, name=f"modular-{parent}-{worker}") for index, worker in enumerate(("A", "B")))
    quiesced = []

    class Connection:
        def __init__(self):
            self.closed = False

        def poll(self, timeout):
            return False

        def close(self):
            self.closed = True

    connections = {"A": Connection(), "B": Connection()}

    class Context:
        def Barrier(self, parties):
            return SimpleNamespace()

        def Event(self):
            return SimpleNamespace(set=lambda: None)

    orphan = runner.UnrecoverableOrphan(f"injected {parent} ledger orphan")

    class PilotWriter:
        def validate_committed_prefix(self):
            return (_resource_row(root.name, "pilot", 100),)

        def append(self, row):
            raise orphan

    class ClaimWriter:
        def __init__(self):
            self.calls = 0

        def append(self, row):
            self.calls += 1
            if self.calls == 2:
                raise orphan
            return SimpleNamespace(acknowledged=True, reason_code=None)

        def validate_committed_prefix(self):
            return ()

    real_import = runner.importlib.import_module
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(get_context=lambda method: Context()) if name == "multiprocessing" else real_import(name))
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: (processes, connections))
    monkeypatch.setattr(runner, "quiesce_worker_processes", lambda observed: quiesced.append(tuple(observed)))
    monkeypatch.setattr(runner, "worker_exit_observations", lambda *args: (False, ()))
    monkeypatch.setattr(runner, "next_resource_sample_monotonic_ns", lambda row: 0)
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 1)
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: {})
    if parent == "pilot":
        monkeypatch.setattr(runner, "final_frozen_guard", lambda *args: None)
        transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {"run/pilot_resources.jsonl": PilotWriter()}, 100, None)
        call = lambda: runner.run_resource_pilot(root, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition)
    else:
        writer = ClaimWriter()
        monkeypatch.setattr(runner, "final_claim_guard", lambda *args: None)
        monkeypatch.setattr(runner, "claim_resource_observations", lambda rows: ())
        monkeypatch.setattr(runner, "_training_start_state", lambda observed_root: ("started", {}))
        monkeypatch.setattr(runner, "_claim_accounting", lambda writers: runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ()))
        transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {"run/resources.jsonl": writer}, 100, None)
        call = lambda: runner.run_claim_workers(root, _tracked_payload(), runner.FrozenManifestAnchors(()), signals, transition, 0)
    with pytest.raises(runner.UnrecoverableOrphan) as caught:
        call()
    assert caught.value is orphan
    assert quiesced == [processes]
    assert all(connection.closed for connection in connections.values())


def test_pilot_worker_waits_on_parent_event_before_runtime_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Gate:
        def wait(self):
            calls.append("wait")
            raise RuntimeError("stop")

    class Connection:
        def send(self, value):
            calls.append("failure")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(runner, "validate_entry_environment", lambda: pytest.fail("environment reached before gate"))
    monkeypatch.setattr(runner, "_import_runtime", lambda: pytest.fail("runtime reached before gate"))
    with pytest.raises(RuntimeError, match="stop"):
        runner._pilot_worker("A", 0, SimpleNamespace(), tmp_path.name, Gate(), Connection())
    assert calls == ["wait", "failure", "close"]


def test_claim_worker_release_uses_started_claim_guards_not_expired_review_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100]
    deadline = 100 + runner.TRAINING_START_REVIEW_WAIT_NS

    class Event:
        def __init__(self):
            self.set_called = False

        def set(self):
            self.set_called = True
            clock[0] = deadline + 1

    event = Event()

    class Context:
        def Event(self):
            return event

    class Writer:
        def append(self, row):
            return SimpleNamespace(acknowledged=True, reason_code=None)

        def validate_committed_prefix(self):
            return ()

    class Connection:
        def poll(self, timeout):
            return False

        def close(self):
            return None

    processes = (
        SimpleNamespace(pid=101, name="modular-claim-A"),
        SimpleNamespace(pid=102, name="modular-claim-B"),
    )
    transition = runner.TransitionResult(
        "claim",
        "ready",
        CLAIM_LEDGER_PATHS,
        {"run/resources.jsonl": Writer()},
        100,
        None,
    )
    real_import = runner.importlib.import_module
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(get_context=lambda method: Context()) if name == "multiprocessing" else real_import(name))
    monkeypatch.setattr(runner, "final_claim_guard", lambda *args: None)
    monkeypatch.setattr(runner, "_resource_sample", lambda *args: {"monotonic_ns": 100})
    monkeypatch.setattr(runner, "claim_resource_observations", lambda rows: ())
    monkeypatch.setattr(runner, "next_resource_sample_monotonic_ns", lambda row: deadline + 10_000_000_000)
    monkeypatch.setattr(runner, "spawn_worker_processes", lambda *args: (processes, {"A": Connection(), "B": Connection()}))
    monkeypatch.setattr(
        runner,
        "_training_start_state",
        lambda root: ("started", {"review_deadline_monotonic_ns": deadline}),
    )
    monkeypatch.setattr(
        runner,
        "worker_exit_observations",
        lambda *args: (False, ({"reason_code": "worker_exit", "context": {"worker": "A"}},)),
    )
    monkeypatch.setattr(runner, "quiesce_after_primary_latch", lambda error, observed_processes: error)
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: clock[0])
    with pytest.raises(runner.HardAbort) as caught:
        runner.run_claim_workers(
            tmp_path,
            _tracked_payload(),
            runner.FrozenManifestAnchors(()),
            _BarrierSignals(),
            transition,
            deadline,
        )
    assert event.set_called is True
    assert caught.value.reason_code == "worker_exit"


def test_claim_worker_start_gate_is_shared_and_released_only_after_final_parent_guard() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = _source_tree()
    worker = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_claim_worker")
    worker_segment = ast.get_source_segment(source, worker)
    assert worker_segment.index("start_event.wait()") < worker_segment.index("validate_entry_environment()") < worker_segment.index("_import_runtime()")
    parent = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "run_claim_workers")
    parent_segment = ast.get_source_segment(source, parent)
    assert "start_event = context.Event()" in parent_segment
    assert '"args": (worker, os.fspath(run_root), start_event)' in parent_segment
    assert parent_segment.index("spawn_worker_processes") < parent_segment.index("signals.commit_guarded(start_event.set)") < parent_segment.index("while True")
    assert parent_segment.rindex("final_claim_guard", 0, parent_segment.index("signals.commit_guarded(start_event.set)")) > parent_segment.index("spawn_worker_processes")


def test_execute_run_ignores_worker_training_start_state_after_started_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: pilot_transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "proceed"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: claim_transition)
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: {})

    def abort(*args):
        raise runner.HardAbort(
            "nonfinite",
            {"worker": "A", "seed": 11, "stage": "training", "training_start_state": "not_started"},
            4242,
        )

    monkeypatch.setattr(runner, "run_claim_workers", abort)
    captured = {}

    def finalize(*args):
        captured["training_start_state"] = args[11]
        signals.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_hard_abort", finalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 2
    assert captured["training_start_state"] == "started"


def test_execute_run_preserves_clean_claim_terminal_closure_after_post_terminal_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    pilot_transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    claim_transition = runner.TransitionResult("claim", "ready", CLAIM_LEDGER_PATHS, {}, 100, None)
    accounting = runner.AttemptAccounting(1, 1, 2048, 2048, (11,), (11,), {11: 1}, ())
    claim_result = {"accounting": accounting, "resource_sampling_end_monotonic_ns": 400, "resource_final_sample_id": 3}
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: pilot_transition)
    monkeypatch.setattr(runner, "run_resource_pilot", lambda *args: {"decision": "proceed"})
    monkeypatch.setattr(runner, "precreate_claim_ledgers", lambda *args: claim_transition)
    monkeypatch.setattr(runner, "prepare_claim_data", lambda *args: {})
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: 123456)
    monkeypatch.setattr(runner, "run_claim_workers", lambda *args: claim_result)
    terminal_bytes = {}

    def finalize(run_root, frozen_payload, anchors, signals, claim_start, result, runtime):
        terminal_bytes["summary"], terminal_bytes["checksum"] = _write_terminal_fixture(
            run_root,
            "summary.json",
            {"status": "negative"},
            signals,
        )
        raise RuntimeError("post-terminal claim fault")

    monkeypatch.setattr(runner, "finalize_clean_claim", finalize)
    monkeypatch.setattr(runner, "finalize_hard_abort", lambda *args, **kwargs: pytest.fail("terminal claim closure entered abort finalization"))
    assert runner.execute_run(
        entry,
        payload,
        runtime,
        ["runner", "--run-root", str(entry.run_root)],
    ) == 0
    assert (entry.run_root / "summary.json").read_bytes() == terminal_bytes["summary"]
    assert (entry.run_root / "SHA256SUMS").read_bytes() == terminal_bytes["checksum"]
    assert signals.terminal is True


def test_execute_run_active_failure_uses_exact_abort_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: calls.append("pilot_transition") or transition)

    def abort(*args):
        calls.append("pilot_abort")
        raise runner.HardAbort("nonfinite", {"worker": "A", "seed": 11, "stage": "pilot"}, 4242)

    monkeypatch.setattr(runner, "run_resource_pilot", abort)
    captured = {}

    def finalize(run_root, frozen_payload, controller, reason_code, phase, context, abort_origin_ns, abort_origin_utc, primary_latch_ns, writers, swap_baseline, training_start_state):
        captured.update(
            {
                "run_root": run_root,
                "payload": frozen_payload,
                "reason_code": reason_code,
                "phase": phase,
                "context": context,
                "abort_origin_ns": abort_origin_ns,
                "abort_origin_utc": abort_origin_utc,
                "primary_latch_ns": primary_latch_ns,
                "writers": writers,
                "swap_baseline": swap_baseline,
                "training_start_state": training_start_state,
            }
        )
        calls.append("abort_finalize")
        controller.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_hard_abort", finalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 2
    assert captured["run_root"] == entry.run_root
    assert captured["payload"] == payload
    assert captured["reason_code"] == "nonfinite"
    assert captured["phase"] == "pilot"
    assert captured["context"] == {"worker": "A", "seed": 11, "stage": "pilot"}
    assert captured["abort_origin_ns"] == 100
    assert captured["abort_origin_utc"] == "2026-07-19T00:00:00Z"
    assert captured["primary_latch_ns"] == 4242
    assert captured["writers"] == {}
    assert captured["swap_baseline"] == 100
    assert captured["training_start_state"] == "not_started"
    assert calls == ["signal_install", "prepilot", "activate", "pilot_transition", "pilot_abort", "abort_finalize", "signal_terminal"]


def test_execute_run_active_initialization_refusal_becomes_artifact_hard_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, payload, runtime, signals, calls = _stub_execute_activation(tmp_path, monkeypatch)
    transition = runner.TransitionResult("pilot", "ready", ("run/pilot_resources.jsonl",), {}, 100, None)
    monkeypatch.setattr(runner, "precreate_pilot_timeline", lambda *args: transition)
    monkeypatch.setattr(
        runner,
        "run_resource_pilot",
        lambda *args: (_ for _ in ()).throw(runner.InitializationRefusal("active refusal")),
    )
    captured = {}

    def finalize(*args):
        captured["reason_code"] = args[3]
        captured["phase"] = args[4]
        signals.deactivate_terminal()

    monkeypatch.setattr(runner, "finalize_hard_abort", finalize)
    assert runner.execute_run(entry, payload, runtime, ["runner", "--run-root", str(entry.run_root)]) == 2
    assert captured == {"reason_code": "artifact_inconsistency", "phase": "pilot"}


@pytest.mark.parametrize("artifact", ("launch", "request", "link"))
def test_training_start_persisted_proof_read_failure_is_unrecoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    repository = tmp_path / "repository"
    root = tmp_path / "run-id"
    _materialize_manifest_anchor_surface(root, b"launch-plan\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", repository)
    _materialize_training_start_lifecycle(root, repository, "started")
    target = {
        "launch": root / runner.LAUNCH_PROJECT_PLAN_PATH,
        "request": root / runner.TRAINING_START_REQUEST_PATH,
        "link": root / runner.TRAINING_START_LINK_PATH,
    }[artifact]
    real_read = Path.read_bytes

    def read(path):
        if path == target:
            raise OSError("injected persisted proof read failure")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner._training_start_state(root)


def test_query_route_snapshot_copies_only_position_126_and_excludes_early_public_bypass(runtime_modules) -> None:
    torch = runtime_modules.torch
    raw_remote = torch.full((2, 128, 1, 2), -1, dtype=torch.long)
    raw_remote[:, 0, 0] = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    raw_remote[:, 126, 0] = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    snapshot = runner._query_only_raw_routes(raw_remote, torch)
    assert snapshot.shape == (2, 128, 1, 2)
    assert snapshot.dtype == torch.long
    assert snapshot[:, 126, 0].tolist() == [[7, 8], [9, 10]]
    assert bool((snapshot[:, torch.arange(128) != 126] == -1).all())
    payload = runner.generate_source_exclusion_routes(11, snapshot, torch.tensor([7, 9]), torch)
    assert payload["raw"] == [[7, 8], [9, 10]]


def test_routing_rows_preserve_public_bypass_and_detached_histogram_evidence(runtime_modules) -> None:
    torch = runtime_modules.torch
    model_module = runtime_modules.model_module
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1901)
        model = model_module.ModularNeuralMachine(model_module.rung_one_config("selected"))
        tokens = torch.randint(0, 128, (2, 128), dtype=torch.long)
    output = model(tokens, return_aux=True, route_detail=True)
    rows = runner._routing_rows(
        output,
        "run-id",
        11,
        "evaluation",
        "selected",
        None,
        "intact",
        None,
        0,
        0,
        0,
        torch.tensor([7, 9], dtype=torch.long),
        None,
        "intact",
        "a" * 64,
    )
    block_four = runner._block_output(output, 4)
    call = next(row for row in rows if row["row_kind"] == "call_summary" and row["block"] == 4)
    queries = [row for row in rows if row["row_kind"] == "query_example" and row["block"] == 4]
    assert call["canonical_bypass_ids"] == _canonical_bypass_records(2)
    assert call["block_load_histogram"] == [
        {"load": int(load), "bucket_count": int(count)}
        for load, count in block_four.telemetry["block_load_histogram"].tolist()
    ]
    assert call["valid_posting_histogram"] == [
        {"valid_posting_count": int(count), "search_row_count": int(rows)}
        for count, rows in block_four.telemetry["valid_posting_histogram"].tolist()
    ]
    assert all(query["canonical_bypass_ids"] is None for query in queries)
    assert all(query["block_load_histogram"] is None for query in queries)
    assert all(query["valid_posting_histogram"] is None for query in queries)
    assert bool((block_four.telemetry["raw_remote"][:, :24] != -1).any())
    assert bool((block_four.telemetry["effective_remote"][:, :24] == -1).all())
    assert [query["raw_remote_ids"] for query in queries] == block_four.telemetry["raw_remote"][:, 126, 0].tolist()


def test_terminal_checksum_rejects_same_length_corrupt_persisted_bytes_before_terminal_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "corrupt-checksum"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    real_write = runner.os.write

    def corrupt_write(descriptor, data):
        replacement = b"0" + data[1:]
        return real_write(descriptor, replacement)

    monkeypatch.setattr(runner.os, "write", corrupt_write)
    with pytest.raises(runner.ContractError):
        runner.write_sha256s_terminal(root, expected_paths=("a",))
    assert not (root / "SHA256SUMS").exists()


def test_terminal_checksum_signals_through_final_fsync_roll_back_before_terminal_adoption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    before_root = tmp_path / "before"
    before_root.mkdir()
    (before_root / "a").write_bytes(b"a")
    before_signals = runner.SignalController()

    def before_hook(stage):
        if stage == "before_terminal_fsync":
            before_signals.inject()

    with pytest.raises(runner.HardAbort) as caught:
        runner.write_sha256s_terminal(before_root, expected_paths=("a",), signals=before_signals, fault_hook=before_hook)
    assert caught.value.reason_code == "signal_or_interruption"
    assert not (before_root / "SHA256SUMS").exists()

    after_root = tmp_path / "after"
    after_root.mkdir()
    (after_root / "a").write_bytes(b"a")
    after_signals = runner.SignalController()
    real_fsync = runner.os.fsync
    checksum_fsyncs = 0

    def inject_after_terminal_fsync(descriptor):
        nonlocal checksum_fsyncs
        real_fsync(descriptor)
        if (after_root / "SHA256SUMS").exists():
            checksum_fsyncs += 1
            if checksum_fsyncs == 3:
                after_signals.inject()

    monkeypatch.setattr(runner.os, "fsync", inject_after_terminal_fsync)
    with pytest.raises(runner.HardAbort) as caught:
        runner.write_sha256s_terminal(after_root, expected_paths=("a",), signals=after_signals)
    assert caught.value.reason_code == "signal_or_interruption"
    assert not (after_root / "SHA256SUMS").exists()
    assert after_signals.terminal is False


def test_terminal_checksum_never_rolls_back_after_boundary_when_signal_unmask_cleanup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "post-boundary-cleanup"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    signals = runner.SignalController()
    real_mask = runner.signal.pthread_sigmask

    def fail_restore(how, mask):
        if how == runner.signal.SIG_SETMASK and signals.terminal:
            raise OSError("restore failed")
        return real_mask(how, mask)

    monkeypatch.setattr(runner.signal, "pthread_sigmask", fail_restore)
    result = runner.write_sha256s_terminal(root, expected_paths=("a",), signals=signals)
    assert result.terminal is True
    assert (root / "SHA256SUMS").is_file()


@pytest.mark.parametrize("cleanup_errors", (False, True))
def test_worker_quiescence_escalates_through_kill_and_verifies_final_liveness(cleanup_errors: bool) -> None:
    process = _EscalatingProcess(ignore_terminate=True, cleanup_errors=cleanup_errors)
    runner.quiesce_worker_processes((process,), timeout_seconds=0.25)
    assert process.alive is False
    assert process.kill_calls == 1
    assert process.join_timeouts == [0.25, 0.25, 0.25]


def test_worker_quiescence_crosses_unrecoverable_boundary_when_process_remains_live() -> None:
    process = _EscalatingProcess(ignore_terminate=True, ignore_kill=True)
    with pytest.raises(runner.UnrecoverableOrphan):
        runner.quiesce_worker_processes((process,), timeout_seconds=0.25)
    assert process.join_timeouts == [0.25, 0.25, 0.25]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_partial_worker_start_failure_closes_every_pipe_and_quiesces_started_peer() -> None:
    connections = []
    started = _EscalatingProcess(ignore_terminate=True)

    class Connection:
        def __init__(self):
            self.closed = False
            connections.append(self)

        def close(self):
            self.closed = True

    class FailingStartProcess:
        pid = None

        def start(self):
            raise RuntimeError("start failed")

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    class StartedProcess(_EscalatingProcess):
        pid = 101

        def start(self):
            return None

    first = StartedProcess(ignore_terminate=True)
    process_values = iter((first, FailingStartProcess()))

    class Context:
        def Pipe(self, duplex=True):
            assert duplex is True
            return Connection(), Connection()

        def Process(self, **kwargs):
            return next(process_values)

    specs = (
        {"worker": "A", "target": object(), "args": (), "name": "worker-A"},
        {"worker": "B", "target": object(), "args": (), "name": "worker-B"},
    )
    with pytest.raises(runner.WorkerStartError) as caught:
        runner.spawn_worker_processes(Context(), specs)
    assert caught.value.worker == "B"
    assert first.alive is False
    assert first.kill_calls == 1
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize("failure", (runner.ContractError("command"), runner.ContractError("parse"), runner.ContractError("pid")))
def test_first_pilot_process_sample_failure_retains_outer_owned_empty_timeline(tmp_path: Path, failure: BaseException) -> None:
    root = tmp_path / "pilot-sample"
    (root / "run").mkdir(parents=True)

    def fail_sample(expected):
        raise failure

    result = runner.precreate_pilot_timeline(
        root,
        runner.SignalController(),
        swap_reader=lambda: 4096,
        process_sampler=fail_sample,
    )
    assert result.phase == "pilot"
    assert result.outcome == "pilot_abort"
    assert result.reason_code == "resource_sampler_failure"
    assert result.retained_paths == ("run/pilot_resources.jsonl",)
    assert tuple(result.writers) == ("run/pilot_resources.jsonl",)
    assert (root / "run" / "pilot_resources.jsonl").read_bytes() == b""
    _close_writers(result.writers)


def test_first_pilot_process_sample_failure_preserves_pending_signal_priority(tmp_path: Path) -> None:
    root = tmp_path / "pilot-signal"
    (root / "run").mkdir(parents=True)
    signals = runner.SignalController()

    def fail_sample(expected):
        signals.inject()
        raise runner.ContractError("ps failed")

    result = runner.precreate_pilot_timeline(root, signals, swap_reader=lambda: 4096, process_sampler=fail_sample)
    assert result.reason_code == "signal_or_interruption"
    assert result.writers["run/pilot_resources.jsonl"].validate_committed_prefix() == ()
    _close_writers(result.writers)


def test_pilot_ack_transmission_charges_validated_start_before_failed_send() -> None:
    counters = runner.PilotCounterState(3, 6144)
    connection = _FakeConnection(send_error=BrokenPipeError())
    with pytest.raises(BrokenPipeError):
        runner.transmit_pilot_ack(connection, {"ack": True}, counters, 1, 2048)
    assert counters == runner.PilotCounterState(4, 8192)


def test_abort_resource_sample_skips_empty_timeline_when_ack_charge_is_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "pilot-charge"
    (root / "run").mkdir(parents=True)
    writer = runner.CrashAtomicJsonlWriter(root / "run" / "pilot_resources.jsonl", runner.validate_resource_row)
    writer.precreate()
    accounting = runner.AttemptAccounting(0, 0, 0, 0, (), (), {}, ())

    calls = []

    def sample(run_id, phase, sample_id, processes, handshakes, worker_names, active_jobs, swap_baseline, attempted_updates, token_positions):
        calls.append((attempted_updates, token_positions))
        row = _resource_row(run_id, phase, swap_baseline)
        row["attempted_updates"] = attempted_updates
        row["token_positions"] = token_positions
        return row

    monkeypatch.setattr(runner, "_resource_sample", sample)
    assert runner.best_effort_abort_resource_sample(
        root,
        "pilot",
        {"run/pilot_resources.jsonl": writer},
        100,
        accounting,
        "resource_sampler_failure",
        attempted_updates=1,
        token_positions=2048,
    ) is False
    assert writer.validate_committed_prefix() == ()
    assert calls == []
    writer.close()


def test_pilot_parent_applies_counter_charge_before_ack_transmission() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    node = next(item for item in _source_tree().body if isinstance(item, ast.FunctionDef) and item.name == "run_resource_pilot")
    segment = ast.get_source_segment(source, node)
    assert segment.index("transmit_pilot_ack(") < segment.index("connection.send(response)") if "connection.send(response)" in segment else True
    assert "transmit_pilot_ack(connection, response, counters, update_delta, token_delta)" in segment


def test_run_card_prereg_digest_matches_tracked_canonical_payload() -> None:
    run_card = (PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_run.md").read_text(encoding="utf-8")
    match = re.search(r"Its canonical JSON SHA-256 is\n  `([0-9a-f]{64})`", run_card)
    assert match is not None
    assert match.group(1) == _canonical_sha256(_tracked_payload())


def test_run_card_orders_clean_transport_before_artifact_closure_and_terminal_checksum() -> None:
    run_card = (PROJECT_ROOT / "neuroloc" / "wiki" / "tests" / "modular_sequence_role_cpu_run.md").read_text(encoding="utf-8")
    completion_index = run_card.index("write and fsync `run/completion.json` before\n`summary.json`")
    summary_index = run_card.index("After `summary.json` is fsynced", completion_index)
    close_index = run_card.index("send `close_committed`", summary_index)
    join_index = run_card.index("join the child", close_index)
    cleanup_index = run_card.index("remove the owned scratch tree", join_index)
    closure_index = run_card.index("validate the clean artifact closure", cleanup_index)
    checksum_index = run_card.index("write and fsync terminal `SHA256SUMS`", closure_index)
    assert completion_index < summary_index < close_index < join_index < cleanup_index < closure_index < checksum_index


def test_base_review_binding_rejects_target_mutation_after_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, evidence, staging = _create_review_surface(tmp_path, monkeypatch)
    _write_complete_review_set(repository, evidence)
    runner.select_and_copy_review_attestations(staging, evidence)
    target = repository / dict(runner._review_scopes())["base_tests"][0]
    target.write_bytes(target.read_bytes() + b"mutated")
    with pytest.raises(runner.ContractError):
        runner.validate_base_review_target_binding(staging, "not_started", None)


def test_frozen_source_mismatch_child_failure_preserves_reason_and_surface(tmp_path: Path) -> None:
    from src.model.modular_sources import FrozenSourceMismatchError

    error = FrozenSourceMismatchError(tmp_path / "governed.py", "0" * 64, "1" * 64)
    assert runner._child_failure_message(error, "B") == {
        "kind": "hard_abort",
        "worker": "B",
        "reason_code": "frozen_hash_change",
        "context": {"surface": str(tmp_path / "governed.py"), "worker": "B"},
    }


def test_pilot_protocol_state_rejects_update_before_exact_barrier_identity() -> None:
    state = runner.pilot_protocol_state("A")
    message = {
        "kind": "pilot_update_start",
        "worker": "A",
        "seed": 9999983,
        "stage": "A",
        "logical_update": 1,
        "token_positions": 2048,
    }
    with pytest.raises(runner.ContractError):
        runner.validate_pilot_protocol_message(message, state)


def test_claim_protocol_state_rejects_evaluation_before_fixed_training_stream() -> None:
    state = runner.claim_protocol_state("A")
    message = {
        "kind": "status",
        "worker": "A",
        "seed": 11,
        "stage": "route_acquisition",
        "logical_update": 0,
    }
    with pytest.raises(runner.ContractError):
        runner.validate_claim_worker_message(message, "A", set(), state)


def test_training_start_review_contract_freezes_candidate_handoff_and_base_live_rule() -> None:
    review = _tracked_payload()["artifacts"]["schemas"]["review_artifact"]
    assert review["base_target_records_must_equal_current_live_scope"] is True
    assert "target_records_must_equal_current_live_scope" not in review
    training = _tracked_payload()["artifacts"]["training_start_review_attestation"]
    assert training["candidate_source_path_pattern"] == (
        "neuroloc/results/modular_sequence_role_mlx_reviews/{candidate_sha256}.project-plan.md"
    )
    assert training["candidate_source_raw_sha256_must_equal_candidate_sha256"] is True
    assert training["candidate_binding_line_formula"] == (
        "Training start request `{run_id}` binds request SHA-256 `{request_sha256}`; these reviewed bytes become canonical only at the atomic training-start commit.\n"
    )


def test_pilot_timer_stops_once_after_post_optimizer_finite_check() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    node = next(item for item in _source_tree().body if isinstance(item, ast.FunctionDef) and item.name == "_pilot_worker")
    segment = ast.get_source_segment(source, node)
    assert segment.count("stopped = time.perf_counter_ns()") == 1
    assert segment.index("_assert_model_and_optimizer_finite(torch, model, optimizer, finite_context)\n                stopped") > segment.index("optimizer.step()")
    assert "duration = stopped - started" in segment


def test_population_digest_contract_uses_canonical_indices_object() -> None:
    indices = [0, 2, 5]
    assert runner._population_sha(indices) == _canonical_sha256({"indices": indices})
    schema = _tracked_payload()["artifacts"]["schemas"]["evaluation_row"]
    assert schema["binary_population_sha256_formula"] == (
        "sha256_canonical_json_of_object_with_exact_indices_key_and_invariant_evaluation_ordered_integer_array_value"
    )


def test_frozen_hash_transition_exception_is_exactly_governed_project_plan_commit() -> None:
    contract = _tracked_payload()["abort_rules"]["frozen_hash_transition_exception"]
    assert contract == {
        "path": "neuroloc/wiki/PROJECT_PLAN.md",
        "allowed_transition": "reviewed_ready_launch_sha256_to_started_training_start_project_plan_sha256",
        "governed_by": "run/training_start_plan.json",
        "all_other_reviewed_targets_remain_base_attested": True,
    }
    registry = _tracked_payload()["abort_rules"]["hard_abort_registry"]
    frozen = next(record for record in registry if record["reason_code"] == "frozen_hash_change")
    assert frozen["condition"] == (
        "any_frozen_source_configuration_or_preregistration_hash_changes_except_the_exact_governed_"
        "PROJECT_PLAN_reviewed_ready_to_started_transition"
    )
    assert runner.HARD_ABORT_CONDITIONS[1] == frozen["condition"]


@pytest.mark.parametrize(
    ("metric", "mutation"),
    (
        ("query_underfill_count", {"denominator": 0}),
        ("query_underfill_count", {"overflow_count": 0}),
        ("selected_mask_oracle_max_error", {"numerator": 0.0}),
        ("selected_mask_oracle_max_error", {"query_underfill_count": 0}),
        ("route_overflow_count", {"denominator": 0}),
        ("route_overflow_count", {"selected_mask_oracle_max_error": 0.0}),
    ),
)
def test_evaluation_scalar_row_matrix_rejects_wrong_value_or_null(metric: str, mutation: dict) -> None:
    row = {
        "metric": metric,
        "numerator": None if metric == "selected_mask_oracle_max_error" else 3,
        "denominator": None,
        "estimate": 0.25 if metric == "selected_mask_oracle_max_error" else 3,
        "wilson95_low": None,
        "wilson95_high": None,
        "answer_correct": None,
        "answer_total": None,
        "original_source_hits": None,
        "original_source_total": None,
        "foreign_source_hits": None,
        "foreign_source_total": None,
        "raw_remote_ids": None,
        "effective_remote_ids": None,
        "query_underfill_count": 3 if metric == "query_underfill_count" else None,
        "overflow_count": 3 if metric == "route_overflow_count" else None,
        "max_bucket_load": 7 if metric == "route_overflow_count" else None,
        "selected_mask_oracle_max_error": 0.25 if metric == "selected_mask_oracle_max_error" else None,
    }
    row.update(mutation)
    with pytest.raises(runner.ContractError):
        runner.validate_evaluation_scalar_row(row)


def test_elapsed_and_resource_reference_derivations_are_exact() -> None:
    assert runner.elapsed_seconds_from_monotonic_ns(10, 2_000_000_010) == 2.0
    rows = [
        {"sample_id": 0, "active_jobs": []},
        {"sample_id": 1, "active_jobs": [{"seed": 11}, {"seed": 23}]},
        {"sample_id": 2, "active_jobs": [{"seed": 23}]},
        {"sample_id": 3, "active_jobs": [{"seed": 11}]},
    ]
    assert runner.resource_sample_ids_for_seed(rows, 11) == [1, 3]


def test_summary_contract_rejects_status_and_seed_complement_drift() -> None:
    decisions = [
        {"construction_seed": seed, "gate_id": row["gate_id"], "gate_pass": True}
        for seed in runner.RUNG_ONE_SEEDS
        for row in _tracked_payload()["gates"]["rung_one_registry"]
    ]
    decisions.extend(
        {"construction_seed": 83, "gate_id": row["gate_id"], "gate_pass": True}
        for row in _tracked_payload()["gates"]["rung_two_registry"]
    )
    summary = runner.summary_from_gate_decisions("test-run", "0" * 64, decisions)
    runner.validate_summary_contract(summary, decisions)
    changed = copy.deepcopy(summary)
    changed["status"] = "negative"
    with pytest.raises(runner.ContractError):
        runner.validate_summary_contract(changed, decisions)
    changed = copy.deepcopy(summary)
    changed["per_seed"][0]["passed_gates"].pop()
    with pytest.raises(runner.ContractError):
        runner.validate_summary_contract(changed, decisions)
