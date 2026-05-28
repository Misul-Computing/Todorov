from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .suite_registry import SIMULATION_SPECS, SUITES, SimulationSpec, all_specs, get_suite_specs
    from .shared import find_project_root, validate_metrics_file
except ImportError:
    from suite_registry import SIMULATION_SPECS, SUITES, SimulationSpec, all_specs, get_suite_specs
    from shared import find_project_root, validate_metrics_file


@dataclass
class SimulationRunResult:
    simulation_id: str
    category: str
    profile: str
    ok: bool
    returncode: int
    duration_sec: float
    output_dir: str
    metrics_path: str | None
    stdout_tail: str
    stderr_tail: str
    validation_error: str | None = None


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_env_key(key: str) -> str:
    return key.upper() if os.name == "nt" else key


def validate_env_key(key: str, source: str) -> str:
    stripped_key = key.strip()
    if not stripped_key:
        raise ValueError(f"environment override key must not be empty: {source}")
    if "=" in stripped_key:
        raise ValueError(f"environment override key must not contain '=': {source}")
    if "\x00" in stripped_key:
        raise ValueError(f"environment override key must not contain NUL: {source}")
    canonical_key = canonical_env_key(stripped_key)
    if stripped_key.upper() == "SIM_OUTPUT_DIR":
        raise ValueError("SIM_OUTPUT_DIR is managed by the suite runner; use --output-root instead")
    return canonical_key


def validate_env_value(value: str, source: str) -> str:
    if value.strip() == "":
        raise ValueError(f"environment override value must not be empty: {source}")
    if "\x00" in value:
        raise ValueError(f"environment override value must not contain NUL: {source}")
    return value


def resolve_output_root(output_root: str | Path, project_root: Path) -> Path:
    output_path = Path(output_root)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    return output_path.resolve()


def parse_env_override(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"environment override must use KEY=VALUE syntax: {entry}")
    key, value = entry.split("=", 1)
    canonical_key = validate_env_key(key, entry)
    canonical_value = validate_env_value(value, entry)
    return canonical_key, canonical_value


def parse_env_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    posix_keys_by_upper: dict[str, str] = {}
    for entry in entries:
        key, value = parse_env_override(entry)
        if os.name != "nt":
            upper_key = key.upper()
            existing_key = posix_keys_by_upper.get(upper_key)
            if existing_key is not None and existing_key != key:
                raise ValueError(f"environment override key collides by case with another override: {entry}")
            posix_keys_by_upper[upper_key] = key
        overrides[key] = value
    return overrides


def normalize_env_overrides(overrides: dict[str, str] | None) -> dict[str, str]:
    if not overrides:
        return {}
    normalized: dict[str, str] = {}
    posix_keys_by_upper: dict[str, str] = {}
    for key, value in overrides.items():
        canonical_key = validate_env_key(key, key)
        if os.name != "nt":
            upper_key = canonical_key.upper()
            existing_key = posix_keys_by_upper.get(upper_key)
            if existing_key is not None and existing_key != canonical_key:
                raise ValueError(f"environment override key collides by case with another override: {key}")
            posix_keys_by_upper[upper_key] = canonical_key
        normalized[canonical_key] = validate_env_value(str(value), key)
    return normalized


def build_effective_env(
    profile: str,
    smoke_env: dict[str, str],
    env_overrides: dict[str, str] | None,
    hard_env: dict[str, str] | None = None,
) -> dict[str, str]:
    if os.name == "nt":
        env = {canonical_env_key(key): value for key, value in os.environ.copy().items()}
    else:
        env = os.environ.copy()
    normalized_smoke_env = normalize_env_overrides(smoke_env)
    normalized_hard_env = normalize_env_overrides(hard_env or smoke_env)
    normalized_env_overrides = normalize_env_overrides(env_overrides)
    profile_env = normalized_hard_env if profile == "hard" else normalized_smoke_env
    if profile in {"smoke", "hard"} and os.name != "nt" and normalized_env_overrides:
        smoke_keys_by_upper = {key.upper(): key for key in profile_env}
        for key in normalized_env_overrides:
            existing_key = smoke_keys_by_upper.get(key.upper())
            if existing_key is not None and existing_key != key:
                raise ValueError(f"environment override key collides by case with smoke setting: {key}")
    if profile in {"smoke", "hard"}:
        env.update(profile_env)
    if normalized_env_overrides:
        env.update(normalized_env_overrides)
    return env


def missing_modules_for_interpreter(
    interpreter: str,
    required_modules: tuple[str, ...],
    probe_env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> tuple[list[str], str | None]:
    if not required_modules:
        return [], None
    probe_command = [
        interpreter,
        "-c",
        (
            "import importlib.util, json, sys; "
            "missing=[name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]; "
            "print(json.dumps(missing))"
        ),
        *required_modules,
    ]
    try:
        probe = subprocess.run(
            probe_command,
            capture_output=True,
            text=True,
            check=False,
            env=probe_env,
            cwd=cwd,
        )
    except Exception as exc:
        return list(required_modules), str(exc)
    if probe.returncode != 0:
        message = probe.stderr.strip() or probe.stdout.strip() or f"dependency probe failed with exit code {probe.returncode}"
        return list(required_modules), message
    try:
        payload = json.loads(probe.stdout.strip() or "[]")
    except json.JSONDecodeError as exc:
        return list(required_modules), f"dependency probe returned invalid JSON: {exc}"
    if not isinstance(payload, list):
        return list(required_modules), "dependency probe returned a non-list payload"
    return [str(item) for item in payload], None


def nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def has_measurement(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, str):
        return False
    if isinstance(value, list):
        return len(value) > 0 and any(has_measurement(item) for item in value)
    if isinstance(value, dict):
        if {"n", "mean", "ci95"}.issubset(value):
            ci95 = value.get("ci95")
            return (
                value.get("mean") is not None
                and int(value.get("n", 0)) > 0
                and isinstance(ci95, dict)
                and ci95.get("low") is not None
                and ci95.get("high") is not None
            )
        relevant_items = [item for item in value.values() if not isinstance(item, str)]
        return len(relevant_items) > 0 and all(has_measurement(item) for item in relevant_items)
    return True


def metadata_path_matches_expected(raw_path: str, expected_path: Path, metrics_path: Path) -> bool:
    artifact_path = Path(raw_path)
    if artifact_path.is_absolute():
        return artifact_path.resolve() == expected_path.resolve()
    project_root = find_project_root(metrics_path)
    candidates = [metrics_path.parent / artifact_path, project_root / artifact_path]
    expected_resolved = expected_path.resolve()
    return any(candidate.resolve() == expected_resolved for candidate in candidates)


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_tail(value: str | bytes | bytearray | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[-limit:]
    return bytes(value).decode("utf-8", errors="replace")[-limit:]


def rewrite_artifact_metadata_for_output(metrics_path: Path, output_dir: Path) -> None:
    payload = validate_metrics_file(metrics_path)
    rewritten_artifacts = []
    for artifact in payload["artifacts"]:
        rewritten_artifact = {
            "name": artifact["name"],
            "path": (output_dir / artifact["name"]).as_posix(),
            "type": artifact["type"],
        }
        if "description" in artifact:
            rewritten_artifact["description"] = artifact["description"]
        rewritten_artifacts.append(rewritten_artifact)
    payload["artifacts"] = rewritten_artifacts
    try:
        from .shared import write_json
    except ImportError:
        from shared import write_json
    write_json(metrics_path, payload)


def validate_simulation_output(spec: SimulationSpec, simulation_output: Path, metrics_path: Path) -> dict[str, Any]:
    payload = validate_metrics_file(metrics_path)
    if payload["run"]["simulation_name"] != spec.simulation_id:
        raise ValueError(f"simulation name mismatch for: {spec.simulation_id}")
    if payload["run"]["status"] != "completed":
        raise ValueError(f"run status is not completed for: {spec.simulation_id}")
    if payload["environment"]["script_path"] != spec.script_relative_path:
        raise ValueError(f"script path mismatch for: {spec.simulation_id}")
    for artifact in payload["artifacts"]:
        if artifact["name"] != Path(str(artifact["path"])).name:
            raise ValueError(f"artifact name/path mismatch for: {spec.simulation_id}")
    payload_artifact_names = [str(artifact["name"]) for artifact in payload["artifacts"]]
    if len(payload_artifact_names) != len(set(payload_artifact_names)):
        raise ValueError(f"duplicate artifact metadata names for: {spec.simulation_id}")
    expected_artifact_names = set(spec.artifact_filenames)
    if set(payload_artifact_names) != expected_artifact_names:
        raise ValueError(f"artifact metadata set mismatch for: {spec.simulation_id}")
    on_disk_artifact_names = {path.name for path in simulation_output.iterdir() if path.is_file()}
    if on_disk_artifact_names != expected_artifact_names:
        raise ValueError(f"artifact output set mismatch for: {spec.simulation_id}")
    artifact_entries = {str(artifact["name"]): artifact for artifact in payload["artifacts"]}
    for dotted_key in spec.required_summary_keys:
        summary_value = nested_value(payload["summary"], dotted_key)
        if not has_measurement(summary_value):
            raise ValueError(f"required summary value is empty: {dotted_key}")
    for dotted_key, minimum in spec.minimum_summary_values:
        summary_value = nested_value(payload["summary"], dotted_key)
        if not isinstance(summary_value, (int, float)) or not math.isfinite(float(summary_value)):
            raise ValueError(f"summary value is not numeric: {dotted_key}")
        if float(summary_value) < float(minimum):
            raise ValueError(f"summary value below minimum: {dotted_key} < {minimum}")
    for dotted_key, maximum in spec.maximum_summary_values:
        summary_value = nested_value(payload["summary"], dotted_key)
        if not isinstance(summary_value, (int, float)) or not math.isfinite(float(summary_value)):
            raise ValueError(f"summary value is not numeric: {dotted_key}")
        if float(summary_value) > float(maximum):
            raise ValueError(f"summary value above maximum: {dotted_key} > {maximum}")
    profile = str(payload.get("configuration", {}).get("parameters", {}).get("profile", "")).strip()
    if profile == "hard":
        for dotted_key, minimum in spec.hard_minimum_summary_values:
            summary_value = nested_value(payload["summary"], dotted_key)
            if not isinstance(summary_value, (int, float)) or not math.isfinite(float(summary_value)):
                raise ValueError(f"summary value is not numeric: {dotted_key}")
            if float(summary_value) < float(minimum):
                raise ValueError(f"summary value below hard minimum: {dotted_key} < {minimum}")
        for dotted_key, maximum in spec.hard_maximum_summary_values:
            summary_value = nested_value(payload["summary"], dotted_key)
            if not isinstance(summary_value, (int, float)) or not math.isfinite(float(summary_value)):
                raise ValueError(f"summary value is not numeric: {dotted_key}")
            if float(summary_value) > float(maximum):
                raise ValueError(f"summary value above hard maximum: {dotted_key} > {maximum}")
    for artifact_name in spec.artifact_filenames:
        expected_path = simulation_output / artifact_name
        if not expected_path.exists():
            raise FileNotFoundError(f"missing expected artifact: {expected_path}")
        artifact_entry = artifact_entries.get(artifact_name)
        if artifact_entry is None:
            raise ValueError(f"missing artifact metadata for: {artifact_name}")
        if not metadata_path_matches_expected(str(artifact_entry["path"]), expected_path, metrics_path):
            raise ValueError(f"artifact metadata path mismatch for: {artifact_name}")
        if artifact_entry.get("exists") != expected_path.exists():
            raise ValueError(f"artifact metadata existence mismatch for: {artifact_name}")
        is_self_referential = expected_path.resolve() == metrics_path.resolve()
        if bool(artifact_entry.get("self_referential", False)) != is_self_referential:
            raise ValueError(f"artifact self_referential mismatch for: {artifact_name}")
        if is_self_referential:
            continue
        if artifact_entry.get("size_bytes") != expected_path.stat().st_size:
            raise ValueError(f"artifact size mismatch for: {artifact_name}")
        if artifact_entry.get("sha256") != artifact_sha256(expected_path):
            raise ValueError(f"artifact hash mismatch for: {artifact_name}")
    return payload


def run_simulation(
    spec: SimulationSpec,
    profile: str,
    output_root: Path,
    python_executable: str | None = None,
    timeout_sec: int = 300,
    env_overrides: dict[str, str] | None = None,
) -> SimulationRunResult:
    project_root = project_root_from_here()
    script_path = spec.script_path(project_root)
    normalized_env_overrides = normalize_env_overrides(env_overrides)
    output_root.mkdir(parents=True, exist_ok=True)
    simulation_output = output_root / spec.simulation_id
    interpreter = python_executable or sys.executable
    effective_env = build_effective_env(profile, spec.smoke_env, normalized_env_overrides, spec.hard_env)

    missing_modules, probe_error = missing_modules_for_interpreter(
        interpreter,
        spec.required_modules,
        probe_env=effective_env,
        cwd=project_root,
    )
    if missing_modules:
        message = f"missing required modules: {', '.join(missing_modules)}"
        if probe_error:
            message = f"{message} ({probe_error})"
        return SimulationRunResult(
            simulation_id=spec.simulation_id,
            category=spec.category,
            profile=profile,
            ok=False,
            returncode=-1,
            duration_sec=0.0,
            output_dir=str(simulation_output),
            metrics_path=None,
            stdout_tail="",
            stderr_tail="",
            validation_error=message,
        )

    staging_output = output_root / f"{spec.simulation_id}__staging_{os.getpid()}_{time.time_ns()}"
    staging_output.mkdir(parents=False, exist_ok=False)

    env = effective_env
    env["SIM_OUTPUT_DIR"] = str(staging_output)

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [interpreter, str(script_path)],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        durable_failure_dir = simulation_output if simulation_output.exists() else staging_output
        durable_metrics_path = (
            simulation_output / spec.metrics_filename
            if simulation_output.exists()
            else staging_output / spec.metrics_filename
        )
        if simulation_output.exists():
            shutil.rmtree(staging_output, ignore_errors=True)
        return SimulationRunResult(
            simulation_id=spec.simulation_id,
            category=spec.category,
            profile=profile,
            ok=False,
            returncode=-2,
            duration_sec=float(duration),
            output_dir=str(durable_failure_dir),
            metrics_path=str(durable_metrics_path) if durable_metrics_path.exists() else None,
            stdout_tail=text_tail(exc.output),
            stderr_tail=text_tail(exc.stderr),
            validation_error=f"timed out after {timeout_sec}s",
        )
    duration = time.perf_counter() - started

    metrics_path = staging_output / spec.metrics_filename
    validation_error = None
    ok = proc.returncode == 0
    if ok:
        if not metrics_path.exists():
            ok = False
            validation_error = f"missing metrics file: {metrics_path}"
        else:
            try:
                validate_simulation_output(spec, staging_output, metrics_path)
            except Exception as exc:
                ok = False
                validation_error = str(exc)

    final_output_dir = staging_output
    final_metrics_path = metrics_path if metrics_path.exists() else None
    if ok:
        backup_output = None
        try:
            if simulation_output.exists():
                backup_output = output_root / f"{spec.simulation_id}__backup"
                if backup_output.exists():
                    shutil.rmtree(backup_output)
                shutil.move(str(simulation_output), str(backup_output))
            shutil.move(str(staging_output), str(simulation_output))
            final_output_dir = simulation_output
            final_metrics_path = simulation_output / spec.metrics_filename
            rewrite_artifact_metadata_for_output(final_metrics_path, simulation_output)
            validate_simulation_output(spec, simulation_output, final_metrics_path)
            if backup_output is not None and backup_output.exists():
                shutil.rmtree(backup_output)
        except Exception as exc:
            ok = False
            validation_error = str(exc)
            if backup_output is not None and backup_output.exists():
                try:
                    if simulation_output.exists():
                        shutil.rmtree(simulation_output)
                    shutil.move(str(backup_output), str(simulation_output))
                    final_output_dir = simulation_output
                    final_metrics_path = simulation_output / spec.metrics_filename
                except Exception as rollback_exc:
                    validation_error = f"{validation_error}; rollback failed: {rollback_exc}"
                    if simulation_output.exists():
                        final_output_dir = simulation_output
                        final_metrics_path = simulation_output / spec.metrics_filename
                    elif backup_output.exists():
                        final_output_dir = backup_output
                        final_metrics_path = backup_output / spec.metrics_filename
                    else:
                        final_output_dir = staging_output
                        final_metrics_path = metrics_path if metrics_path.exists() else None
            else:
                final_output_dir = simulation_output if simulation_output.exists() else staging_output
                final_metrics_path = simulation_output / spec.metrics_filename if simulation_output.exists() else None
    else:
        if simulation_output.exists():
            shutil.rmtree(staging_output, ignore_errors=True)
            final_output_dir = simulation_output
            final_metrics_path = simulation_output / spec.metrics_filename
        else:
            final_output_dir = staging_output
            final_metrics_path = metrics_path if metrics_path.exists() else None

    return SimulationRunResult(
        simulation_id=spec.simulation_id,
        category=spec.category,
        profile=profile,
        ok=ok,
        returncode=int(proc.returncode),
        duration_sec=float(duration),
        output_dir=str(final_output_dir),
        metrics_path=str(final_metrics_path) if final_metrics_path is not None and final_metrics_path.exists() else None,
        stdout_tail=proc.stdout[-4000:],
        stderr_tail=proc.stderr[-4000:],
        validation_error=validation_error,
    )


def run_specs(
    specs: list[SimulationSpec],
    profile: str,
    output_root: Path,
    python_executable: str | None = None,
    timeout_sec: int = 300,
    env_overrides: dict[str, str] | None = None,
) -> list[SimulationRunResult]:
    return [
        run_simulation(
            spec=spec,
            profile=profile,
            output_root=output_root,
            python_executable=python_executable,
            timeout_sec=timeout_sec,
            env_overrides=env_overrides,
        )
        for spec in specs
    ]


def print_summary(results: list[SimulationRunResult]) -> None:
    ok_count = sum(1 for result in results if result.ok)
    print(f"suite completed: {ok_count}/{len(results)} passed")
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        line = f"[{status}] {result.simulation_id} ({result.duration_sec:.1f}s)"
        if result.validation_error:
            line += f" :: {result.validation_error}"
        print(line)


def resolve_specs(args: argparse.Namespace) -> list[SimulationSpec]:
    if args.simulation:
        return [SIMULATION_SPECS[item] for item in args.simulation]
    if args.suite:
        return get_suite_specs(args.suite)
    return all_specs()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--suite", choices=sorted(SUITES), help="named simulation suite")
    selection_group.add_argument("--simulation", choices=sorted(SIMULATION_SPECS), nargs="+", help="specific simulations")
    parser.add_argument("--profile", choices=("smoke", "hard", "full"), default="smoke")
    parser.add_argument("--output-root", default="neuroloc/output/simulation_suites")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment override passed to simulation subprocesses; may be repeated",
    )
    parser.add_argument("--write-summary", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.simulation and len(args.simulation) != len(set(args.simulation)):
        parser.error("duplicate simulation IDs are not allowed")
    try:
        env_overrides = parse_env_overrides(args.env)
    except ValueError as exc:
        parser.error(str(exc))
    specs = resolve_specs(args)
    try:
        for spec in specs:
            build_effective_env(args.profile, spec.smoke_env, env_overrides, spec.hard_env)
    except ValueError as exc:
        parser.error(str(exc))
    project_root = project_root_from_here()
    output_root = resolve_output_root(args.output_root, project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = run_specs(
        specs=specs,
        profile=args.profile,
        output_root=output_root,
        python_executable=args.python,
        timeout_sec=args.timeout_sec,
        env_overrides=env_overrides,
    )
    print_summary(results)
    if args.write_summary:
        summary_path = output_root / f"suite_{args.profile}_summary.json"
        summary_path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n", encoding="utf-8")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
