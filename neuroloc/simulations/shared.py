from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import stats

SCHEMA_VERSION = "neuroloc.sim.metrics/v1"


def _json_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _artifact_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def assert_finite_payload(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_payload(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            assert_finite_payload(item, f"{path}[{idx}]")
        return
    if isinstance(value, Path):
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"non-finite array values at {path}")
        if np.issubdtype(value.dtype, np.complexfloating):
            raise ValueError(f"complex array values are not supported at {path}")
        return
    if isinstance(value, np.generic):
        assert_finite_payload(value.item(), path)
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"non-finite float at {path}: {value}")
        return


def normalize_artifact_entry(artifact: dict[str, Any], project_root: Path, base_dir: Path, metrics_path: Path | None = None) -> dict[str, Any]:
    raw_path = artifact.get("path") or artifact.get("name")
    if raw_path is None:
        raise ValueError(f"artifact entry missing path/name: {artifact}")
    artifact_path = Path(str(raw_path))
    if not artifact_path.is_absolute():
        artifact_path = (base_dir / artifact_path).resolve()
    exists = artifact_path.exists()
    normalized = {
        "name": artifact.get("name") or artifact_path.name,
        "path": relative_to_root(artifact_path, project_root),
        "type": str(artifact.get("type", "unknown")),
        "exists": bool(exists),
    }
    if "description" in artifact:
        normalized["description"] = str(artifact["description"])
    if exists:
        is_metrics_artifact = metrics_path is not None and artifact_path.resolve() == metrics_path.resolve()
        if is_metrics_artifact:
            normalized["self_referential"] = True
        else:
            normalized["size_bytes"] = int(artifact_path.stat().st_size)
            normalized["sha256"] = _artifact_sha256(artifact_path)
    return normalized


def normalize_artifacts(
    record: dict[str, Any],
    metrics_path: Path | None = None,
    artifact_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(metrics_path if metrics_path is not None else Path.cwd())
    base_dir = metrics_path.parent if metrics_path is not None else project_root
    record_copy = dict(record)
    artifacts = artifact_entries if artifact_entries is not None else record_copy.get("artifacts", [])
    record_copy["artifacts"] = [
        normalize_artifact_entry(dict(artifact), project_root, base_dir, metrics_path=metrics_path)
        for artifact in artifacts
    ]
    return record_copy


def validate_run_record(record: dict[str, Any]) -> None:
    required_top = {"schema_version", "run", "environment", "configuration", "summary", "statistics", "trials", "artifacts", "warnings"}
    missing_top = required_top - set(record)
    if missing_top:
        raise ValueError(f"run record missing required keys: {sorted(missing_top)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unexpected schema version: {record['schema_version']}")
    run = record["run"]
    for key in ["run_id", "simulation_name", "status", "started_at_utc", "finished_at_utc", "duration_sec", "trial_record_count"]:
        if key not in run:
            raise ValueError(f"run section missing key: {key}")
    environment = record["environment"]
    for key in ["python_version", "numpy_version", "scipy_version", "platform", "cpu", "script_path", "git_commit", "git_dirty"]:
        if key not in environment:
            raise ValueError(f"environment section missing key: {key}")
    config = record["configuration"]
    for key in ["parameters", "parameter_hash", "seed_numpy", "numpy_bit_generator", "n_trials_requested"]:
        if key not in config:
            raise ValueError(f"configuration section missing key: {key}")
    if not isinstance(record["trials"], list):
        raise ValueError("trials must be a list")
    if int(run["trial_record_count"]) != len(record["trials"]):
        raise ValueError(
            f"trial_record_count={run['trial_record_count']} does not match len(trials)={len(record['trials'])}"
        )
    for artifact in record["artifacts"]:
        for key in ["name", "path", "type", "exists"]:
            if key not in artifact:
                raise ValueError(f"artifact missing key {key}: {artifact}")


def validate_metrics_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metrics payload at {path} is not a dict")
    assert_finite_payload(payload)
    validate_run_record(payload)
    return payload


def apply_plot_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def find_project_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in [resolved] + list(resolved.parents):
        if ((candidate / "AGENTS.md").is_file() or (candidate / "CLAUDE.md").is_file()) and (candidate / "requirements.txt").is_file():
            return candidate
    return resolved if resolved.is_dir() else resolved.parent


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(default if raw is None or raw.strip() == "" else raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(default if raw is None or raw.strip() == "" else raw)


def env_list(name: str, cast: Any, default: list[Any]) -> list[Any]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must not resolve to an empty list")
    return values


def require_non_empty(name: str, values: list[Any]) -> list[Any]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def require_unique_list(name: str, values: list[Any]) -> list[Any]:
    require_non_empty(name, values)
    if any(isinstance(value, float) for value in values):
        unique_values = []
        for value in values:
            if any(np.isclose(float(value), float(existing)) for existing in unique_values):
                raise ValueError(f"{name} must not contain duplicate values")
            unique_values.append(value)
        return values
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate values")
    return values


def require_positive(name: str, value: float | int) -> float | int:
    if float(value) <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def require_non_negative(name: str, value: float | int) -> float | int:
    if float(value) < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def require_positive_list(name: str, values: list[float | int]) -> list[float | int]:
    require_unique_list(name, values)
    for value in values:
        require_positive(name, value)
    return values


def require_non_negative_list(name: str, values: list[float | int]) -> list[float | int]:
    require_unique_list(name, values)
    for value in values:
        require_non_negative(name, value)
    return values


def require_unit_interval(name: str, value: float, allow_zero: bool = False) -> float:
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        comparator = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {comparator}")
    return value


def require_unit_interval_list(name: str, values: list[float], allow_zero: bool = False) -> list[float]:
    require_unique_list(name, values)
    for value in values:
        require_unit_interval(name, float(value), allow_zero=allow_zero)
    return values


def leak_tau_condition_name(tau_ms: float) -> str:
    return f"explicit_leak_tau{format(float(tau_ms), 'g').replace('.', 'p')}"


def ensure_close_member(values: list[float], target: float) -> list[float]:
    if any(np.isclose(float(target), float(value)) for value in values):
        return list(values)
    return sorted([*values, float(target)])


def default_output_dir(script_path: Path) -> Path:
    resolved_script = script_path.resolve()
    project_root = find_project_root(resolved_script)
    simulations_root = (project_root / "neuroloc" / "simulations").resolve()
    try:
        relative_parent = resolved_script.parent.relative_to(simulations_root)
    except ValueError:
        relative_parent = Path("misc") / resolved_script.parent.name
    return project_root / "neuroloc" / "output" / "simulation_runs" / relative_parent / resolved_script.stem


def output_dir_for(script_path: Path) -> Path:
    raw = os.environ.get("SIM_OUTPUT_DIR")
    output_dir = Path(raw).resolve() if raw else default_output_dir(script_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def child_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(child_seed(rng))


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    assert_finite_payload(payload)
    normalized_payload = sanitize_json(payload)
    artifact_entries = None
    if isinstance(normalized_payload, dict) and normalized_payload.get("schema_version") == SCHEMA_VERSION:
        artifact_entries = [dict(artifact) for artifact in normalized_payload.get("artifacts", [])]
        normalized_payload = normalize_artifacts(normalized_payload, metrics_path=path, artifact_entries=artifact_entries)
        validate_run_record(normalized_payload)
    tmp_path = _json_path(path)
    tmp_path.write_text(json.dumps(normalized_payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    if isinstance(normalized_payload, dict) and normalized_payload.get("schema_version") == SCHEMA_VERSION:
        enriched_payload = normalize_artifacts(normalized_payload, metrics_path=path, artifact_entries=artifact_entries)
        validate_run_record(enriched_payload)
        tmp_path = _json_path(path)
        tmp_path.write_text(json.dumps(sanitize_json(enriched_payload), indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)


def parameter_hash(parameters: dict[str, Any]) -> str:
    serialized = json.dumps(sanitize_json(parameters), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def mean_confidence_interval(values: Any, confidence_level: float = 0.95, bounds: tuple[float, float] | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": {"low": None, "high": None}}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    if count > 1 and std > 0:
        se = std / np.sqrt(count)
        t_crit = float(stats.t.ppf((1 + confidence_level) / 2, count - 1))
        low = mean - t_crit * se
        high = mean + t_crit * se
    else:
        low = mean
        high = mean
    if bounds is not None:
        low = max(low, bounds[0])
        high = min(high, bounds[1])
    return {"n": count, "mean": mean, "std": std, "ci95": {"low": low, "high": high}}


def paired_difference_stats(reference: Any, test: Any, seed: int) -> dict[str, Any]:
    reference_array = np.asarray(reference, dtype=float)
    test_array = np.asarray(test, dtype=float)
    mask = np.isfinite(reference_array) & np.isfinite(test_array)
    reference_array = reference_array[mask]
    test_array = test_array[mask]
    differences = test_array - reference_array
    summary = mean_confidence_interval(differences)
    if differences.size > 1:
        permutation = stats.permutation_test(
            (reference_array, test_array),
            statistic=lambda x, y: np.mean(y - x),
            permutation_type="samples",
            n_resamples=1999,
            alternative="two-sided",
            random_state=np.random.default_rng(seed),
        )
        difference_std = float(np.std(differences, ddof=1))
        t_pvalue = float(stats.ttest_rel(test_array, reference_array).pvalue) if difference_std > 0 else None
        effect = float(np.mean(differences) / difference_std) if difference_std > 0 else None
        summary.update(
            {
                "mean_difference": float(np.mean(differences)),
                "p_value_permutation": float(permutation.pvalue),
                "p_value_ttest": t_pvalue,
                "effect_size_dz": effect,
            }
        )
    else:
        summary.update(
            {
                "mean_difference": float(np.mean(differences)) if differences.size == 1 else None,
                "p_value_permutation": None,
                "p_value_ttest": None,
                "effect_size_dz": None,
            }
        )
    return summary


def independent_difference_stats(group_a: Any, group_b: Any, seed: int) -> dict[str, Any]:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    mean_diff = float(np.mean(b) - np.mean(a))
    if a.size > 1 and b.size > 1:
        permutation = stats.permutation_test(
            (a, b),
            statistic=lambda x, y: np.mean(y) - np.mean(x),
            permutation_type="independent",
            n_resamples=1999,
            alternative="two-sided",
            random_state=np.random.default_rng(seed),
        )
        t_result = stats.ttest_ind(b, a, equal_var=False)
        pooled_std = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
        effect = mean_diff / pooled_std if pooled_std > 0 else None
        return {
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_difference": mean_diff,
            "p_value_permutation": float(permutation.pvalue),
            "p_value_ttest": float(t_result.pvalue),
            "effect_size_d": effect,
        }
    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": float(np.mean(a)) if a.size > 0 else None,
        "mean_b": float(np.mean(b)) if b.size > 0 else None,
        "mean_difference": mean_diff,
        "p_value_permutation": None,
        "p_value_ttest": None,
        "effect_size_d": None,
    }


def discrete_mutual_information(x: Any, y: Any) -> float | None:
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.size == 0 or y_array.size == 0 or x_array.shape[0] != y_array.shape[0]:
        return None
    x_values, x_inverse = np.unique(x_array, return_inverse=True)
    y_values, y_inverse = np.unique(y_array, return_inverse=True)
    joint = np.zeros((x_values.size, y_values.size), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1.0)
    joint /= float(joint.sum())
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    product = marginal_x * marginal_y
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / product[mask])))


def linear_cka(x: Any, y: Any) -> float | None:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[0] != y_array.shape[0]:
        return None
    x_centered = x_array - np.mean(x_array, axis=0, keepdims=True)
    y_centered = y_array - np.mean(y_array, axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    norm_x = np.linalg.norm(x_centered.T @ x_centered, ord="fro")
    norm_y = np.linalg.norm(y_centered.T @ y_centered, ord="fro")
    if norm_x == 0 or norm_y == 0:
        return None
    return float((np.linalg.norm(cross, ord="fro") ** 2) / (norm_x * norm_y))


def git_metadata(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def build_run_record(
    simulation_name: str,
    script_path: Path,
    started_at_utc: str,
    finished_at_utc: str,
    duration_sec: float,
    parameters: dict[str, Any],
    seed_numpy: int,
    n_trials: int,
    summary: dict[str, Any],
    statistics: dict[str, Any],
    trials: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    warnings: list[str],
    extra_configuration: dict[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    project_root = find_project_root(script_path)
    git_info = git_metadata(project_root)
    configuration = {
        "parameters": parameters,
        "parameter_hash": parameter_hash(parameters),
        "seed_numpy": seed_numpy,
        "numpy_bit_generator": np.random.default_rng(seed_numpy).bit_generator.__class__.__name__,
        "n_trials_requested": int(n_trials),
    }
    if extra_configuration:
        configuration.update(extra_configuration)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": f"{simulation_name}_{started_at_utc.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}",
            "simulation_name": simulation_name,
            "status": status,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "duration_sec": duration_sec,
            "trial_record_count": len(trials),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
            "cpu": platform.processor() or None,
            "script_path": relative_to_root(script_path, project_root),
            **git_info,
        },
        "configuration": configuration,
        "summary": summary,
        "statistics": statistics,
        "trials": trials,
        "artifacts": artifacts,
        "warnings": warnings,
    }
