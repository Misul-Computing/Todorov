import importlib.util
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuroloc.simulations.suite_registry import SIMULATION_SPECS, get_suite_specs
from neuroloc.simulations.shared import build_run_record, default_output_dir, ensure_close_member, find_project_root, leak_tau_condition_name, output_dir_for, require_unit_interval_list, utc_now_iso, validate_metrics_file, write_json
from neuroloc.simulations.suite_registry import SimulationSpec
from neuroloc.simulations.suite_runner import build_effective_env, canonical_env_key, normalize_env_overrides, parse_env_override, parse_env_overrides, resolve_output_root, run_simulation, run_specs, validate_simulation_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("marker", ["AGENTS.md", "CLAUDE.md"])
def test_find_project_root_accepts_current_and_legacy_instruction_markers(tmp_path: Path, marker: str) -> None:
    root = tmp_path / "repo"
    script = root / "neuroloc" / "simulations" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (root / marker).write_text("rules\n", encoding="utf-8")
    assert find_project_root(script) == root


def test_find_project_root_without_instruction_marker_falls_back_to_script_parent(tmp_path: Path) -> None:
    script = tmp_path / "repo" / "neuroloc" / "simulations" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    assert find_project_root(script) == script.parent


@pytest.mark.parametrize("marker", ["AGENTS.md", "CLAUDE.md"])
def test_find_project_root_requires_requirements_with_each_instruction_marker(tmp_path: Path, marker: str) -> None:
    root = tmp_path / "repo"
    script = root / "neuroloc" / "simulations" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    (root / marker).write_text("rules\n", encoding="utf-8")
    assert find_project_root(script) == script.parent


def test_find_project_root_rejects_requirements_without_instruction_marker(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    script = root / "neuroloc" / "simulations" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    assert find_project_root(script) == script.parent


@pytest.mark.parametrize("directory_marker", ["AGENTS.md", "CLAUDE.md", "requirements.txt"])
def test_find_project_root_rejects_directory_markers(tmp_path: Path, directory_marker: str) -> None:
    root = tmp_path / "repo"
    script = root / "neuroloc" / "simulations" / "probe.py"
    script.parent.mkdir(parents=True)
    script.write_text("pass\n", encoding="utf-8")
    for marker in ("AGENTS.md", "requirements.txt"):
        path = root / marker
        if marker == directory_marker:
            path.mkdir()
        else:
            path.write_text("rules\n", encoding="utf-8")
    if directory_marker == "CLAUDE.md":
        (root / "AGENTS.md").unlink()
        (root / "CLAUDE.md").mkdir()
    assert find_project_root(script) == script.parent


def test_find_project_root_without_markers_preserves_directory_input(tmp_path: Path) -> None:
    directory = tmp_path / "standalone"
    directory.mkdir()
    assert find_project_root(directory) == directory


def require_metrics_path(result) -> Path:
    assert result.metrics_path is not None
    return Path(result.metrics_path)


def assert_metrics_metadata(result) -> None:
    metrics_path = require_metrics_path(result)
    validate_simulation_output(SIMULATION_SPECS[result.simulation_id], Path(result.output_dir), metrics_path)
    payload = validate_metrics_file(metrics_path)
    artifact_entries = {Path(str(artifact["path"])).name: artifact for artifact in payload["artifacts"]}
    for artifact_name in SIMULATION_SPECS[result.simulation_id].artifact_filenames:
        artifact_entry = artifact_entries[artifact_name]
        artifact_path = Path(result.output_dir) / artifact_name
        assert artifact_entry["exists"] is True
        if artifact_entry.get("self_referential"):
            assert artifact_path == metrics_path
            continue
        assert artifact_entry["size_bytes"] == artifact_path.stat().st_size
        assert artifact_entry["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()


def test_suite_registry_contract() -> None:
    for spec in SIMULATION_SPECS.values():
        assert spec.metrics_filename in spec.artifact_filenames
        assert spec.script_path(PROJECT_ROOT).exists()
        assert spec.smoke_env
        assert spec.required_summary_keys
        assert all(key.upper() == key for key in spec.smoke_env)
        if spec.hard_env is not None:
            assert spec.hard_env
            assert all(key.upper() == key for key in spec.hard_env)


def test_run_simulation_probes_required_modules_with_selected_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    probe_envs = []
    probe_cwds = []

    def fake_run(command, **kwargs):
        commands.append(command)
        probe_envs.append(dict(kwargs.get("env") or {}))
        probe_cwds.append(kwargs.get("cwd"))
        return subprocess.CompletedProcess(command, 0, stdout='["missing_pkg"]\n', stderr="")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module.subprocess, "run", fake_run)
    spec = SimulationSpec(
        simulation_id="probe_only",
        category="spiking",
        script_relative_path="neuroloc/simulations/memory/capacity_scaling.py",
        metrics_filename="capacity_scaling_metrics.json",
        artifact_filenames=("capacity_scaling_metrics.json",),
        smoke_env={},
        required_summary_keys=("binary_at_target_alpha",),
        required_modules=("missing_pkg",),
    )
    result = run_simulation(
        spec=spec,
        profile="smoke",
        output_root=tmp_path / "probe_only",
        python_executable="c:/custom/python.exe",
        timeout_sec=300,
        env_overrides={"CAPACITY_TRIALS": "9"},
    )
    assert commands[0][0] == "c:/custom/python.exe"
    assert probe_envs[0]["CAPACITY_TRIALS"] == "9"
    assert Path(probe_cwds[0]) == PROJECT_ROOT
    assert result.ok is False
    assert result.validation_error == "missing required modules: missing_pkg"


def test_parse_env_override_accepts_key_value() -> None:
    assert parse_env_override("ASYM_TRIALS=32") == ("ASYM_TRIALS", "32")


def test_parse_env_override_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_env_override("ASYM_TRIALS")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_env_override(" =32")
    with pytest.raises(ValueError, match="value must not be empty"):
        parse_env_override("ASYM_TRIALS=")
    with pytest.raises(ValueError, match="value must not be empty"):
        parse_env_override("ASYM_TRIALS=   ")
    with pytest.raises(ValueError, match="SIM_OUTPUT_DIR"):
        parse_env_override("SIM_OUTPUT_DIR=forbidden")


def test_parse_env_overrides_last_value_wins() -> None:
    overrides = parse_env_overrides(["ASYM_TRIALS=8", "ASYM_TRIALS=32", "ASYM_DECAYS=0.4,0.8"])
    assert overrides == {"ASYM_TRIALS": "32", "ASYM_DECAYS": "0.4,0.8"}


def test_parse_env_overrides_mixed_case_last_value_wins() -> None:
    if os.name == "nt":
        overrides = parse_env_overrides(["capacity_trials=8", "CAPACITY_TRIALS=32"])
        assert overrides == {canonical_env_key("CAPACITY_TRIALS"): "32"}
    else:
        with pytest.raises(ValueError, match="collides by case"):
            parse_env_overrides(["capacity_trials=8", "CAPACITY_TRIALS=32"])


def test_parse_env_override_rejects_reserved_key_case_insensitively() -> None:
    with pytest.raises(ValueError, match="SIM_OUTPUT_DIR"):
        parse_env_override("sim_output_dir=forbidden")


def test_normalize_env_overrides_rejects_reserved_key() -> None:
    with pytest.raises(ValueError, match="SIM_OUTPUT_DIR"):
        normalize_env_overrides({"sim_output_dir": "forbidden"})


def test_normalize_env_overrides_rejects_equals_in_key() -> None:
    with pytest.raises(ValueError, match="must not contain '='"):
        normalize_env_overrides({"A=B": "value"})


def test_normalize_env_overrides_rejects_posix_case_collision() -> None:
    if os.name == "nt":
        pytest.skip("posix-specific environment semantics")
    with pytest.raises(ValueError, match="collides by case"):
        normalize_env_overrides({"capacity_trials": "8", "CAPACITY_TRIALS": "32"})


def test_normalize_env_overrides_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="value must not be empty"):
        normalize_env_overrides({"CAPACITY_TRIALS": ""})


def test_normalize_env_overrides_rejects_nul_value() -> None:
    with pytest.raises(ValueError, match="must not contain NUL"):
        normalize_env_overrides({"CAPACITY_TRIALS": "bad\x00value"})


def test_build_effective_env_applies_smoke_then_overrides() -> None:
    env = build_effective_env("smoke", {"CAPACITY_TRIALS": "2", "CAPACITY_MAX_ITERS": "12"}, {"CAPACITY_TRIALS": "9"})
    assert env["CAPACITY_TRIALS"] == "9"
    assert env["CAPACITY_MAX_ITERS"] == "12"


def test_build_effective_env_applies_hard_profile_env() -> None:
    env = build_effective_env(
        "hard",
        {"NM_HARD_SYMBOLIC_PROFILE": "smoke", "NM_HARD_SYMBOLIC_EPISODES": "4"},
        None,
        {"NM_HARD_SYMBOLIC_PROFILE": "hard", "NM_HARD_SYMBOLIC_EPISODES": "32"},
    )
    assert env["NM_HARD_SYMBOLIC_PROFILE"] == "hard"
    assert env["NM_HARD_SYMBOLIC_EPISODES"] == "32"


def test_build_effective_env_rejects_reserved_override_map() -> None:
    with pytest.raises(ValueError, match="SIM_OUTPUT_DIR"):
        build_effective_env("smoke", {"CAPACITY_TRIALS": "2"}, {"sim_output_dir": "forbidden"})


def test_build_effective_env_windows_dedupes_parent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name != "nt":
        pytest.skip("windows-specific environment semantics")
    monkeypatch.setenv("capacity_trials", "1")
    env = build_effective_env("smoke", {"CAPACITY_TRIALS": "2"}, {"CAPACITY_TRIALS": "9"})
    assert env["CAPACITY_TRIALS"] == "9"
    assert "capacity_trials" not in env


def test_parse_env_override_preserves_case_on_posix() -> None:
    if os.name == "nt":
        pytest.skip("posix-specific environment semantics")
    assert parse_env_override("capacity_trials=8") == ("capacity_trials", "8")


def test_build_effective_env_rejects_posix_case_collision_with_smoke_env() -> None:
    if os.name == "nt":
        pytest.skip("posix-specific environment semantics")
    with pytest.raises(ValueError, match="collides by case with smoke setting"):
        build_effective_env("smoke", {"CAPACITY_TRIALS": "2"}, {"capacity_trials": "9"})


def test_build_effective_env_allows_posix_case_difference_outside_smoke() -> None:
    if os.name == "nt":
        pytest.skip("posix-specific environment semantics")
    env = build_effective_env("full", {"CAPACITY_TRIALS": "2"}, {"capacity_trials": "9"})
    assert env["capacity_trials"] == "9"


def test_run_simulation_applies_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_missing_modules(
        interpreter: str,
        required_modules: tuple[str, ...],
        probe_env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[list[str], str | None]:
        return [], None

    def fake_run(command, **kwargs):
        observed["env"] = dict(kwargs["env"])
        output_dir = Path(kwargs["env"]["SIM_OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "capacity_scaling_metrics.json"
        figure_path = output_dir / "capacity_scaling.png"
        figure_path.write_bytes(b"png")
        record = build_run_record(
            simulation_name="capacity_scaling",
            script_path=PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "capacity_scaling.py",
            started_at_utc=utc_now_iso(),
            finished_at_utc=utc_now_iso(),
            duration_sec=0.0,
            parameters={},
            seed_numpy=0,
            n_trials=1,
            summary={
                "binary_at_target_alpha": {64: 0.1},
                "ternary_at_target_alpha": {64: 0.2},
            },
            statistics={},
            trials=[{"trial_id": 0}],
            artifacts=[
                {"path": figure_path.as_posix(), "type": "figure"},
                {"path": metrics_path.as_posix(), "type": "metrics"},
            ],
            warnings=[],
        )
        write_json(metrics_path, record)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module, "missing_modules_for_interpreter", fake_missing_modules)
    monkeypatch.setattr(suite_runner_module.subprocess, "run", fake_run)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path / "env_override",
        python_executable=sys.executable,
        timeout_sec=300,
        env_overrides={"CAPACITY_TRIALS": "9", "CAPACITY_MAX_ITERS": "3"},
    )
    assert result.ok is True
    assert observed["env"]["CAPACITY_TRIALS"] == "9"
    assert observed["env"]["CAPACITY_MAX_ITERS"] == "3"
    assert observed["env"]["SIM_OUTPUT_DIR"]


def test_run_simulation_env_override_beats_smoke_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_missing_modules(
        interpreter: str,
        required_modules: tuple[str, ...],
        probe_env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[list[str], str | None]:
        return [], None

    def fake_run(command, **kwargs):
        observed["env"] = dict(kwargs["env"])
        output_dir = Path(kwargs["env"]["SIM_OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "capacity_scaling_metrics.json"
        figure_path = output_dir / "capacity_scaling.png"
        figure_path.write_bytes(b"png")
        record = build_run_record(
            simulation_name="capacity_scaling",
            script_path=PROJECT_ROOT / "neuroloc" / "simulations" / "memory" / "capacity_scaling.py",
            started_at_utc=utc_now_iso(),
            finished_at_utc=utc_now_iso(),
            duration_sec=0.0,
            parameters={},
            seed_numpy=0,
            n_trials=1,
            summary={
                "binary_at_target_alpha": {64: 0.1},
                "ternary_at_target_alpha": {64: 0.2},
            },
            statistics={},
            trials=[{"trial_id": 0}],
            artifacts=[
                {"path": figure_path.as_posix(), "type": "figure"},
                {"path": metrics_path.as_posix(), "type": "metrics"},
            ],
            warnings=[],
        )
        write_json(metrics_path, record)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module, "missing_modules_for_interpreter", fake_missing_modules)
    monkeypatch.setattr(suite_runner_module.subprocess, "run", fake_run)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path / "smoke_precedence",
        python_executable=sys.executable,
        timeout_sec=300,
        env_overrides={"CAPACITY_TRIALS": "99"},
    )
    assert result.ok is True
    assert observed["env"]["CAPACITY_TRIALS"] == "99"


def test_run_simulation_rejects_reserved_env_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SIM_OUTPUT_DIR"):
        run_simulation(
            spec=SIMULATION_SPECS["capacity_scaling"],
            profile="smoke",
            output_root=tmp_path / "reserved_override",
            python_executable=sys.executable,
            timeout_sec=300,
            env_overrides={"sim_output_dir": "forbidden"},
        )
    assert not (tmp_path / "reserved_override").exists()


def test_resolve_output_root_uses_project_root_for_relative_paths() -> None:
    resolved = resolve_output_root("neuroloc/output/simulation_suites", PROJECT_ROOT)
    assert resolved == (PROJECT_ROOT / "neuroloc" / "output" / "simulation_suites").resolve()


def test_output_dir_for_defaults_under_repo_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "repo"
    script_path = project_root / "neuroloc" / "simulations" / "memory" / "capacity_scaling.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("pass\n", encoding="utf-8")
    (project_root / "CLAUDE.md").write_text("x\n", encoding="utf-8")
    (project_root / "requirements.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.delenv("SIM_OUTPUT_DIR", raising=False)
    expected = default_output_dir(script_path)
    resolved = output_dir_for(script_path)
    assert resolved == expected
    assert resolved.parent == project_root / "neuroloc" / "output" / "simulation_runs" / "memory"
    assert resolved != script_path.parent


def test_compression_smoke_suite(tmp_path: Path) -> None:
    results = run_specs(
        specs=get_suite_specs("compression"),
        profile="smoke",
        output_root=tmp_path / "compression",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures
    for result in results:
        assert_metrics_metadata(result)


def test_spiking_smoke_suite(tmp_path: Path) -> None:
    if importlib.util.find_spec("brian2") is None:
        pytest.skip("brian2 is not installed")
    results = run_specs(
        specs=get_suite_specs("spiking"),
        profile="smoke",
        output_root=tmp_path / "spiking",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    failures = [
        (result.simulation_id, result.validation_error, result.stderr_tail)
        for result in results
        if not result.ok
    ]
    assert not failures, failures
    for result in results:
        assert_metrics_metadata(result)


def test_suite_runner_cli_writes_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "cli"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--profile",
        "smoke",
        "--output-root",
        str(output_root),
        "--write-summary",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert (output_root / "suite_smoke_summary.json").exists()


def test_suite_runner_cli_applies_env_override(tmp_path: Path) -> None:
    output_root = tmp_path / "cli_env"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--profile",
        "smoke",
        "--env",
        "CAPACITY_TRIALS=3",
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    metrics_path = output_root / "capacity_scaling" / "capacity_scaling_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["configuration"]["parameters"]["trials"] == 3


def test_suite_runner_cli_repeated_env_last_value_wins(tmp_path: Path) -> None:
    output_root = tmp_path / "cli_env_repeat"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--profile",
        "smoke",
        "--env",
        "CAPACITY_TRIALS=2",
        "--env",
        "CAPACITY_TRIALS=4",
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    metrics_path = output_root / "capacity_scaling" / "capacity_scaling_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["configuration"]["parameters"]["trials"] == 4


def test_suite_runner_clears_stale_output(tmp_path: Path) -> None:
    output_root = tmp_path / "stale"
    stale_dir = output_root / "capacity_scaling"
    stale_dir.mkdir(parents=True)
    stale_file = stale_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    assert not stale_file.exists()
    assert_metrics_metadata(results[0])


def test_write_json_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    now = utc_now_iso()
    record = build_run_record(
        simulation_name="nonfinite_guard",
        script_path=PROJECT_ROOT / "neuroloc" / "simulations" / "shared.py",
        started_at_utc=now,
        finished_at_utc=now,
        duration_sec=0.0,
        parameters={},
        seed_numpy=0,
        n_trials=0,
        summary={"bad": float("nan")},
        statistics={},
        trials=[],
        artifacts=[],
        warnings=[],
    )
    with pytest.raises(ValueError):
        write_json(tmp_path / "nonfinite_metrics.json", record)


def test_run_simulation_timeout_returns_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial stdout", stderr="partial stderr")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module.subprocess, "run", raise_timeout)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path,
        python_executable=sys.executable,
        timeout_sec=1,
    )
    assert result.ok is False
    assert result.returncode == -2
    assert result.validation_error == "timed out after 1s"


def test_validate_simulation_output_rejects_metadata_path_mismatch(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=tmp_path / "negative_metadata",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = "wrong_dir/capacity_scaling.png"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact metadata path mismatch"):
        validate_simulation_output(SIMULATION_SPECS["capacity_scaling"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_partial_summary(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=tmp_path / "negative_summary",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    first_key = next(iter(payload["summary"]["binary_at_target_alpha"]))
    payload["summary"]["binary_at_target_alpha"][first_key] = None
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required summary value is empty"):
        validate_simulation_output(SIMULATION_SPECS["capacity_scaling"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_provenance_mismatch(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=tmp_path / "negative_provenance",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    original_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    mutations = [
        ("simulation name mismatch", lambda payload: payload["run"].__setitem__("simulation_name", "wrong_name")),
        ("run status is not completed", lambda payload: payload["run"].__setitem__("status", "failed")),
        ("script path mismatch", lambda payload: payload["environment"].__setitem__("script_path", "wrong/path.py")),
        ("environment section missing key: numpy_version", lambda payload: payload["environment"].pop("numpy_version")),
    ]
    for message, mutate in mutations:
        payload = json.loads(json.dumps(original_payload))
        mutate(payload)
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            validate_simulation_output(SIMULATION_SPECS["capacity_scaling"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_ci_corruption(tmp_path: Path) -> None:
    if importlib.util.find_spec("brian2") is None:
        pytest.skip("brian2 is not installed")
    results = run_specs(
        specs=[SIMULATION_SPECS["wta_dynamics"]],
        profile="smoke",
        output_root=tmp_path / "negative_ci",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["summary"]["kwta_exact_support_recovery"]["ci95"]["low"] = None
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required summary value is empty"):
        validate_simulation_output(SIMULATION_SPECS["wta_dynamics"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_lif_spike_probe_regression(tmp_path: Path) -> None:
    if importlib.util.find_spec("brian2") is None:
        pytest.skip("brian2 is not installed")
    results = run_specs(
        specs=[SIMULATION_SPECS["lif_fi_curve"]],
        profile="smoke",
        output_root=tmp_path / "negative_lif_spike_probe",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["summary"]["spike_probe"]["selected_gap_second_spike"]["integrator_control"] = 0.0
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary value below minimum"):
        validate_simulation_output(SIMULATION_SPECS["lif_fi_curve"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_summary_above_maximum(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["compression_under_bit_budget_mirror"]],
        profile="smoke",
        output_root=tmp_path / "negative_maximum",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["summary"]["paid_compute_authorized"] = 1.0
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary value above maximum"):
        validate_simulation_output(SIMULATION_SPECS["compression_under_bit_budget_mirror"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_nonfinite_metrics_file(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=tmp_path / "negative_nonfinite",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    first_key = next(iter(payload["summary"]["binary_at_target_alpha"]))
    payload["summary"]["binary_at_target_alpha"][first_key] = float("nan")
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite float"):
        validate_simulation_output(SIMULATION_SPECS["capacity_scaling"], metrics_path.parent, metrics_path)


def test_validate_simulation_output_rejects_duplicate_artifact_metadata(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=tmp_path / "negative_artifacts",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["artifacts"].append(dict(payload["artifacts"][0]))
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate artifact metadata names"):
        validate_simulation_output(SIMULATION_SPECS["capacity_scaling"], metrics_path.parent, metrics_path)


def test_failed_rerun_preserves_previous_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "preserve"
    first_result = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )[0]
    assert first_result.ok
    previous_metrics = require_metrics_path(first_result).read_text(encoding="utf-8")

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial stdout", stderr="partial stderr")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module.subprocess, "run", raise_timeout)
    failed_result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=1,
    )
    assert failed_result.ok is False
    preserved_metrics = output_root / "capacity_scaling" / "capacity_scaling_metrics.json"
    assert preserved_metrics.exists()
    assert failed_result.metrics_path == str(preserved_metrics)
    assert preserved_metrics.read_text(encoding="utf-8") == previous_metrics
    assert not any(path.name.startswith("capacity_scaling__") for path in output_root.iterdir())


def test_post_move_finalization_failure_restores_previous_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "rollback"
    first_result = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )[0]
    assert first_result.ok
    previous_metrics = require_metrics_path(first_result).read_text(encoding="utf-8")

    import neuroloc.simulations.suite_runner as suite_runner_module

    def fail_finalization(*args, **kwargs):
        raise RuntimeError("finalization failed")

    monkeypatch.setattr(suite_runner_module, "rewrite_artifact_metadata_for_output", fail_finalization)
    failed_result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert failed_result.ok is False
    preserved_metrics = output_root / "capacity_scaling" / "capacity_scaling_metrics.json"
    assert preserved_metrics.exists()
    assert preserved_metrics.read_text(encoding="utf-8") == previous_metrics


def test_move_failure_after_backup_restores_previous_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "move_failure"
    first_result = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )[0]
    assert first_result.ok
    previous_metrics = require_metrics_path(first_result).read_text(encoding="utf-8")

    import neuroloc.simulations.suite_runner as suite_runner_module

    real_move = suite_runner_module.shutil.move
    move_counter = {"count": 0}

    def fail_second_move(src, dst):
        move_counter["count"] += 1
        if move_counter["count"] == 2:
            raise RuntimeError("move failed")
        return real_move(src, dst)

    monkeypatch.setattr(suite_runner_module.shutil, "move", fail_second_move)
    failed_result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert failed_result.ok is False
    preserved_metrics = output_root / "capacity_scaling" / "capacity_scaling_metrics.json"
    assert preserved_metrics.exists()
    assert preserved_metrics.read_text(encoding="utf-8") == previous_metrics


def test_move_failure_with_rollback_failure_returns_structured_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "move_failure_rollback_failure"
    first_result = run_specs(
        specs=[SIMULATION_SPECS["capacity_scaling"]],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )[0]
    assert first_result.ok

    import neuroloc.simulations.suite_runner as suite_runner_module

    real_move = suite_runner_module.shutil.move
    move_counter = {"count": 0}

    def fail_second_and_third_move(src, dst):
        move_counter["count"] += 1
        if move_counter["count"] in (2, 3):
            raise RuntimeError("move failed")
        return real_move(src, dst)

    monkeypatch.setattr(suite_runner_module.shutil, "move", fail_second_and_third_move)
    failed_result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=output_root,
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert failed_result.ok is False
    assert failed_result.validation_error is not None
    assert "rollback failed" in failed_result.validation_error


def test_first_run_timeout_keeps_failure_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial stdout", stderr="partial stderr")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module.subprocess, "run", raise_timeout)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path / "first_timeout",
        python_executable=sys.executable,
        timeout_sec=1,
    )
    assert result.ok is False
    assert Path(result.output_dir).exists()


def test_first_run_timeout_surfaces_preserved_metrics_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout_with_metrics(*args, **kwargs):
        output_dir = Path(kwargs["env"]["SIM_OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "capacity_scaling_metrics.json"
        metrics_path.write_text("{}\n", encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"], output="partial stdout", stderr="partial stderr")

    import neuroloc.simulations.suite_runner as suite_runner_module

    monkeypatch.setattr(suite_runner_module.subprocess, "run", raise_timeout_with_metrics)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path / "first_timeout_metrics",
        python_executable=sys.executable,
        timeout_sec=1,
    )
    assert result.ok is False
    assert result.metrics_path is not None
    assert Path(result.metrics_path).exists()


def test_first_run_finalization_failure_keeps_failure_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import neuroloc.simulations.suite_runner as suite_runner_module

    def fail_finalization(*args, **kwargs):
        raise RuntimeError("finalization failed")

    monkeypatch.setattr(suite_runner_module, "rewrite_artifact_metadata_for_output", fail_finalization)
    result = run_simulation(
        spec=SIMULATION_SPECS["capacity_scaling"],
        profile="smoke",
        output_root=tmp_path / "first_finalization_failure",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert result.ok is False
    assert Path(result.output_dir).exists()


def test_fractional_lif_tau_condition_names_do_not_collide() -> None:
    assert leak_tau_condition_name(20.1) != leak_tau_condition_name(20.9)


def test_validate_simulation_output_rejects_missing_hierarchical_summary(tmp_path: Path) -> None:
    results = run_specs(
        specs=[SIMULATION_SPECS["hierarchical_ternary"]],
        profile="smoke",
        output_root=tmp_path / "negative_hierarchical_summary",
        python_executable=sys.executable,
        timeout_sec=300,
    )
    assert results[0].ok
    metrics_path = require_metrics_path(results[0])
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["summary"]["selected_hierarchical_reference_mi"]["mean"] = None
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required summary value is empty"):
        validate_simulation_output(SIMULATION_SPECS["hierarchical_ternary"], metrics_path.parent, metrics_path)


def test_suite_runner_cli_rejects_conflicting_selection_flags(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--suite",
        "compression",
        "--simulation",
        "capacity_scaling",
        "--output-root",
        str(tmp_path / "conflict"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0


def test_suite_runner_cli_rejects_duplicate_simulation_values(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "capacity_scaling",
        "--output-root",
        str(tmp_path / "duplicate_simulation"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0


def test_suite_runner_cli_rejects_invalid_env_override(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--env",
        "INVALID",
        "--output-root",
        str(tmp_path / "invalid_env"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "KEY=VALUE" in completed.stderr


def test_suite_runner_cli_rejects_reserved_env_override(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--env",
        "sim_output_dir=forbidden",
        "--output-root",
        str(tmp_path / "reserved_env"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "SIM_OUTPUT_DIR" in completed.stderr


def test_suite_runner_cli_rejects_posix_smoke_case_collision(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("posix-specific environment semantics")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--profile",
        "smoke",
        "--env",
        "capacity_trials=9",
        "--output-root",
        str(tmp_path / "posix_smoke_collision"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "collides by case with smoke setting" in completed.stderr


def test_suite_runner_cli_rejects_empty_env_override_value(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "neuroloc" / "simulations" / "suite_runner.py"),
        "--simulation",
        "capacity_scaling",
        "--env",
        "CAPACITY_TRIALS=",
        "--output-root",
        str(tmp_path / "empty_env_value"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "value must not be empty" in completed.stderr


def test_duplicate_float_sweep_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate values"):
        require_unit_interval_list("TEST_FLOAT_SWEEP", [0.1, 0.10000000001])


def test_ensure_close_member_avoids_near_equal_duplicates() -> None:
    merged = ensure_close_member([16.0], 16.0000000001)
    assert merged == [16.0]
