"""Report progress for the Rtau parameter and disorder sweep."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FILENAME_PATTERN = re.compile(
    r"^Rtau_N(?P<N>\d+)"
    r"_x(?P<x>[-+0-9p]+)"
    r"_y(?P<y>[-+0-9p]+)"
    r"_z(?P<z>[-+0-9p]+)"
    r"_ds(?P<ds>[-+0-9p]+)"
    r"_dd(?P<dd>[-+0-9p]+)"
    r"_seed(?P<seed>\d+)"
    r"_tmax(?P<t_max>[-+0-9.eE]+)\.npz$"
)


def load_sweep_module() -> Any:
    """Import ``main_local`` without printing its top-level sweep summary."""
    with contextlib.redirect_stdout(io.StringIO()):
        import main_local

    return main_local


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    if not math.isfinite(seconds):
        return "n/a"

    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds_int = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds_int}s"
    if minutes:
        return f"{minutes}m {seconds_int}s"
    return f"{seconds:.3f}s"


def parameter_token(parameter: tuple[int, float, float, float, float, float], sweep: Any) -> tuple[Any, ...]:
    """Build the filename-token key for one parameter point."""
    system_size, x_value, y_value, z_value, ds_value, dd_value = parameter
    return (
        int(system_size),
        sweep.safe_float_name(x_value),
        sweep.safe_float_name(y_value),
        sweep.safe_float_name(z_value),
        sweep.safe_float_name(ds_value),
        sweep.safe_float_name(dd_value),
    )


def parse_result_file(path: Path) -> dict[str, Any] | None:
    """Parse sweep metadata encoded in one result filename."""
    match = FILENAME_PATTERN.match(path.name)
    if match is None:
        return None

    parts = match.groupdict()
    return {
        "path": path,
        "N": int(parts["N"]),
        "x_token": parts["x"],
        "y_token": parts["y"],
        "z_token": parts["z"],
        "ds_token": parts["ds"],
        "dd_token": parts["dd"],
        "seed": int(parts["seed"]),
    }


def read_calculation_time(path: Path) -> float:
    """Read calculation_time_seconds from one npz file, if available."""
    try:
        with np.load(path, allow_pickle=False) as data:
            if "calculation_time_seconds" not in data.files:
                return math.nan
            return float(data["calculation_time_seconds"])
    except Exception:
        return math.nan


def build_progress(data_dir: Path, cpus_per_task: int, read_timing: bool) -> dict[str, Any]:
    """Build a progress report from the current output directory."""
    sweep = load_sweep_module()
    data_dir = Path(data_dir)

    parameter_index_by_token = {
        parameter_token(parameter, sweep): parameter_index
        for parameter_index, parameter in enumerate(sweep.parameter_sweep)
    }
    system_size_by_parameter_index = {
        parameter_index: int(parameter[0])
        for parameter_index, parameter in enumerate(sweep.parameter_sweep)
    }

    expected_by_system_size = Counter()
    for parameter in sweep.parameter_sweep:
        expected_by_system_size[int(parameter[0])] += int(sweep.reals)

    total_jobs = len(sweep.parameter_sweep) * int(sweep.reals)
    completed_global_jobs: set[int] = set()
    completed_by_system_size = Counter()
    timed_by_system_size = Counter()
    calculation_time_by_system_size = defaultdict(float)
    unexpected_files: list[str] = []
    extra_files: list[str] = []
    recent_files: list[tuple[float, str]] = []

    for path in sorted(data_dir.glob("Rtau_*.npz")):
        parsed = parse_result_file(path)
        if parsed is None:
            unexpected_files.append(path.name)
            continue

        token = (
            parsed["N"],
            parsed["x_token"],
            parsed["y_token"],
            parsed["z_token"],
            parsed["ds_token"],
            parsed["dd_token"],
        )
        parameter_index = parameter_index_by_token.get(token)
        seed = int(parsed["seed"])

        if parameter_index is None or seed < 0 or seed >= int(sweep.reals):
            extra_files.append(path.name)
            continue

        global_job_id = parameter_index * int(sweep.reals) + seed
        completed_global_jobs.add(global_job_id)
        system_size = system_size_by_parameter_index[parameter_index]
        completed_by_system_size[system_size] += 1

        if read_timing:
            calculation_time = read_calculation_time(path)
            if math.isfinite(calculation_time):
                timed_by_system_size[system_size] += 1
                calculation_time_by_system_size[system_size] += calculation_time

        try:
            recent_files.append((path.stat().st_mtime, path.name))
        except OSError:
            pass

    completed_jobs = len(completed_global_jobs)
    next_missing_global_job = next(
        (global_job_id for global_job_id in range(total_jobs) if global_job_id not in completed_global_jobs),
        None,
    )
    next_array_task_id = (
        next_missing_global_job // cpus_per_task
        if next_missing_global_job is not None and cpus_per_task > 0
        else None
    )

    rows = []
    for system_size in sorted(expected_by_system_size):
        expected = expected_by_system_size[system_size]
        completed = completed_by_system_size[system_size]
        timed = timed_by_system_size[system_size]
        recorded_time = calculation_time_by_system_size[system_size]
        completion_fraction = completed / expected if expected else math.nan
        projected_time = recorded_time * expected / timed if timed else math.nan
        full_sweep_time = recorded_time if completed == expected and timed == expected else math.nan

        rows.append(
            {
                "N": system_size,
                "expected_files": expected,
                "completed_files": completed,
                "remaining_files": expected - completed,
                "completion_fraction": completion_fraction,
                "timed_files": timed,
                "recorded_time_seconds": recorded_time,
                "projected_full_sweep_time_seconds": projected_time,
                "full_sweep_time_seconds": full_sweep_time,
            }
        )

    recent_files = sorted(recent_files, reverse=True)[:5]

    return {
        "data_dir": str(data_dir),
        "reals": int(sweep.reals),
        "nlist": [int(system_size) for system_size in sweep.nlist],
        "parameter_points": len(sweep.parameter_sweep),
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "remaining_jobs": total_jobs - completed_jobs,
        "completion_fraction": completed_jobs / total_jobs if total_jobs else math.nan,
        "temporary_files": len(list(data_dir.glob("*.tmp.npz"))),
        "unexpected_files": unexpected_files,
        "extra_files": extra_files,
        "next_missing_global_job": next_missing_global_job,
        "next_array_task_id": next_array_task_id,
        "cpus_per_task": cpus_per_task,
        "rows": rows,
        "recent_files": [name for _, name in recent_files],
    }


def print_report(report: dict[str, Any]) -> None:
    """Print a human-readable progress report."""
    completion_percent = 100.0 * report["completion_fraction"]

    print("Rtau sweep progress")
    print(f"Data directory: {report['data_dir']}")
    print(f"N values: {report['nlist']}")
    print(f"Realizations per parameter point: {report['reals']}")
    print(f"Completed files: {report['completed_jobs']} / {report['total_jobs']} ({completion_percent:.4f}%)")
    print(f"Remaining files: {report['remaining_jobs']}")
    print(f"Temporary files: {report['temporary_files']}")

    if report["next_missing_global_job"] is None:
        print("Next missing global job: none")
    else:
        print(f"Next missing global job: {report['next_missing_global_job']}")
        print(f"Next array task containing it: {report['next_array_task_id']} with SLURM_CPUS_PER_TASK={report['cpus_per_task']}")
        print(
            "Suggested next chunk command: "
            f"SLURM_ARRAY_TASK_ID={report['next_array_task_id']} "
            f"SLURM_CPUS_PER_TASK={report['cpus_per_task']} "
            "conda run -n qdragon python main_local.py"
        )

    print()
    print("By system size")
    header = (
        "N", "done", "expected", "percent", "recorded time", "projected full time", "full time"
    )
    print(f"{header[0]:>4} {header[1]:>10} {header[2]:>10} {header[3]:>9} {header[4]:>16} {header[5]:>20} {header[6]:>14}")

    for row in report["rows"]:
        percent = 100.0 * row["completion_fraction"]
        print(
            f"{row['N']:>4} "
            f"{row['completed_files']:>10} "
            f"{row['expected_files']:>10} "
            f"{percent:>8.4f}% "
            f"{format_duration(row['recorded_time_seconds']):>16} "
            f"{format_duration(row['projected_full_sweep_time_seconds']):>20} "
            f"{format_duration(row['full_sweep_time_seconds']):>14}"
        )

    if report["recent_files"]:
        print()
        print("Most recent files")
        for name in report["recent_files"]:
            print(f"- {name}")

    if report["unexpected_files"] or report["extra_files"]:
        print()
        print(f"Unexpected filename count: {len(report['unexpected_files'])}")
        print(f"Files outside current sweep configuration: {len(report['extra_files'])}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Report progress for the Rtau sweep.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"), help="Directory containing Rtau_*.npz files.")
    parser.add_argument("--cpus-per-task", type=int, default=8, help="Chunk size used by SLURM_CPUS_PER_TASK.")
    parser.add_argument("--skip-timing", action="store_true", help="Count files without opening npz files for timing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text table.")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    report = build_progress(args.data_dir, args.cpus_per_task, read_timing=not args.skip_timing)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()