"""Balanced tap comparisons with atomic per-point checkpoints and chunk bootstrap.

Each job generates one finite channel grid. All taps see the same grid/noise.
The independent resampling unit is a whole grid (chunk), not a single sector.
No tap is removed based on its interim outcomes. Old pilot rows are not pooled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Config
from .experiments import run_concatenated
from .fsr import aggregate_fsr, extrapolate_fsr_targets


ROOT = Path(__file__).resolve().parents[2]
EXTRA_SECTIONS = {"ldpc", "rate_plan", "boundary_scan", "iti_calibration", "channel_metrics"}
SIMULATION_FILES = (
    "tap_search.py", "experiments.py", "config.py", "channel.py", "codebook.py",
    "concat.py", "constraints.py", "decoder.py", "detector.py", "ldpc.py",
    "metrics.py", "patterns.py", "siso.py", "fsr.py",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    atomic_text(path, buffer.getvalue())


@contextmanager
def campaign_lock(directory: Path):
    # POSIX locks are released by the OS after a crash/reboot; no stale-lock deletion.
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is using this campaign; do not run it twice") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_plan(path: Path) -> dict:
    plan = yaml.safe_load(path.read_text())
    expected = {"name", "base_config", "true_iti", "detector_taps", "snr_db",
                "sectors_per_chunk", "chunks", "master_seed", "reference_tap",
                "targets", "bootstrap_replicates", "summary_every_chunks"}
    if not isinstance(plan, dict) or set(plan) != expected:
        raise ValueError("Search plan keys must be exactly: " + ", ".join(sorted(expected)))
    if not str(plan["name"]).replace("_", "").replace("-", "").isalnum():
        raise ValueError("name must contain only letters, digits, underscores or hyphens")
    for key in ("sectors_per_chunk", "chunks", "bootstrap_replicates", "summary_every_chunks", "master_seed"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if plan["sectors_per_chunk"] > 100:
        raise ValueError("Keep sectors_per_chunk <= 100; increase chunks for longer evaluation")
    if plan["bootstrap_replicates"] < 200:
        raise ValueError("bootstrap_replicates must be >= 200")
    for key in ("detector_taps", "snr_db", "targets"):
        values = [float(v) for v in plan[key]]
        if not values or len(set(values)) != len(values) or not all(map(math.isfinite, values)):
            raise ValueError(key + " must contain distinct finite numbers")
        plan[key] = sorted(values)
    if any(t < 0 for t in plan["detector_taps"]):
        raise ValueError("detector_taps must be nonnegative")
    if not all(0 < t < 1 for t in plan["targets"]):
        raise ValueError("targets must be between 0 and 1")
    plan["true_iti"] = float(plan["true_iti"])
    plan["reference_tap"] = float(plan["reference_tap"])
    if not math.isfinite(plan["true_iti"]) or plan["true_iti"] < 0:
        raise ValueError("true_iti must be finite and nonnegative")
    if plan["reference_tap"] not in plan["detector_taps"] or plan["true_iti"] not in plan["detector_taps"]:
        raise ValueError("Include both reference_tap and the matched true_iti in detector_taps")
    raw = yaml.safe_load((path.parent / plan.pop("base_config")).read_text())
    base = {key: value for key, value in raw.items() if key not in EXTRA_SECTIONS}
    Config.from_dict(base)
    params = copy.deepcopy(raw["ldpc"])
    # This campaign confirms one specific existing receiver, not arbitrary decoders.
    if (base["experiment"]["family"] != "mtr2d_8x2" or base["code"].get("K") != 15
            or params.get("n") != 1800 or params.get("dv") != 3 or params.get("dc") != 75
            or params.get("construction_seed") != 1
            or params.get("channel_detector") != "bcjr_2d_equalized"
            or params.get("inner_demapper") != "hard_codebook"
            or params.get("turbo_iterations") != 0):
        raise ValueError("Expected the existing K15/n1800 high-rate hard-codebook BCJR receiver")
    if any(k.startswith("detector_") and k != "detector_iti_coeffs" for k in params):
        raise ValueError("Remove conflicting detector tap overrides from base config")
    params.pop("num_frames", None)
    params["sector_bits"] = 32768
    params["sector_count_target"] = plan["sectors_per_chunk"]
    params["detector_iti_coeffs"] = plan["detector_taps"]
    base["channel"].update(c_cross_up=plan["true_iti"], c_cross_down=plan["true_iti"])
    plan["base"] = base
    plan["ldpc"] = params
    return plan


def implementation_hash() -> str:
    folder = Path(__file__).parent
    return digest({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                   for name in SIMULATION_FILES})


def prepare(directory: Path, plan: dict) -> dict:
    identity = {"schema": 1, "plan": plan, "implementation_sha256": implementation_hash(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "pyyaml": yaml.__version__}}
    manifest = dict(identity, fingerprint=digest(identity))
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Campaign settings/code/environment changed. Restore them or use a NEW --out-dir")
    else:
        if (directory / "checkpoints").exists():
            raise ValueError("Checkpoints without manifest; refusing to mix unknown results")
        atomic_json(path, manifest)
    return manifest


def point_seed(plan: dict, chunk: int, snr_index: int) -> int:
    # Coordinate-derived 128-bit seed, independent of queue order or resumed subset.
    return int(digest([plan["master_seed"], chunk, plan["snr_db"][snr_index]])[:32], 16)


def point_path(directory: Path, chunk: int, snr_index: int, tap_index: int) -> Path:
    return directory / "checkpoints" / f"chunk{chunk:04d}_snr{snr_index:03d}_tap{tap_index:03d}.json"


def validate_row(row: dict, plan: dict, chunk: int, snr_index: int, tap_index: int) -> None:
    expected = {"seed": point_seed(plan, chunk, snr_index),
                "snr_db": plan["snr_db"][snr_index], "iti_coeff": plan["true_iti"],
                "detector_iti_coeff": plan["detector_taps"][tap_index],
                "sector_bits": 32768, "sector_count": plan["sectors_per_chunk"],
                "channel_detector": "bcjr_2d_equalized", "n": 1800, "k": 1730,
                "equalizer_iterations": plan["ldpc"]["equalizer_iterations"],
                "configured_inner_demapper": "hard_codebook", "turbo_iterations": 0}
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Checkpoint {key}: expected {value}, got {row.get(key)}")
    errors = row.get("sector_errors")
    if type(errors) is not int or not 0 <= errors <= row["sector_count"]:
        raise ValueError("Invalid sector error count")
    if not math.isclose(row["FSR"], errors / row["sector_count"], abs_tol=1e-14):
        raise ValueError("FSR disagrees with counts")
    if not math.isclose(row["rate"], (1730 / 1800) * (15 / 16), abs_tol=1e-12):
        raise ValueError("Unexpected total rate")
    for tap_name in ("c_cross_up", "c_cross_down"):
        if row.get("true_tap_" + tap_name) != plan["true_iti"]:
            raise ValueError("Wrong true channel tap")
        if row.get("detector_tap_" + tap_name) != plan["detector_taps"][tap_index]:
            raise ValueError("Wrong assumed detector tap")


def read_checkpoint(path: Path, manifest: dict, chunk: int, si: int, ti: int) -> dict:
    record = json.loads(path.read_text())
    if record["fingerprint"] != manifest["fingerprint"] or record["point"] != [chunk, si, ti]:
        raise ValueError("Checkpoint identity mismatch: " + str(path))
    row = record["row"]
    validate_row(row, manifest["plan"], chunk, si, ti)
    return dict(row, chunk_index=chunk, snr_index=si, tap_index=ti)


def missing_taps(directory: Path, manifest: dict, chunk: int, si: int) -> list[int]:
    missing = []
    for ti in range(len(manifest["plan"]["detector_taps"])):
        path = point_path(directory, chunk, si, ti)
        if path.exists():
            read_checkpoint(path, manifest, chunk, si, ti)
        else:
            missing.append(ti)
    return missing


def run_job(directory: Path, manifest: dict, chunk: int, si: int) -> None:
    plan = manifest["plan"]
    pending = missing_taps(directory, manifest, chunk, si)
    if not pending:
        return
    base = copy.deepcopy(plan["base"])
    base["experiment"].update(name=f"{plan['name']}_chunk{chunk:04d}_snr{si:03d}",
                              seed=point_seed(plan, chunk, si))
    base["sweep"] = {"snr_db": [plan["snr_db"][si]], "iti_coeffs": [plan["true_iti"]]}
    params = copy.deepcopy(plan["ldpc"])
    params["detector_iti_coeffs"] = [plan["detector_taps"][ti] for ti in pending]

    def checkpoint(row):
        ti = plan["detector_taps"].index(row["detector_iti_coeff"])
        validate_row(row, plan, chunk, si, ti)
        atomic_json(point_path(directory, chunk, si, ti),
                    {"fingerprint": manifest["fingerprint"], "point": [chunk, si, ti], "row": row})
        print(f"SAVED chunk={chunk + 1}/{plan['chunks']} SNR={row['snr_db']:g} "
              f"tap={row['detector_iti_coeff']:g} failures={row['sector_errors']}/"
              f"{row['sector_count']} runtime={row['runtime_sec']:.1f}s", flush=True)

    run_concatenated(Config.from_dict(base), params,
                     cache_dir=str(directory / "codebooks"), result_callback=checkpoint)
    if missing_taps(directory, manifest, chunk, si):
        raise ValueError("Worker returned before every requested tap was saved")


def wilson_interval(errors: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = errors / count
    denom = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def observed_crossing(group: pd.DataFrame, target: float) -> dict:
    """Report a measured-grid bracket, never replace zero failures by a tiny FSR."""
    g = group.sort_values("snr_db")
    snr, p = g.snr_db.to_numpy(), g.FSR.to_numpy()
    out = {"target_fsr": target, "bracket_low_snr_db": np.nan,
           "bracket_high_snr_db": np.nan, "interpolated_snr_db": np.nan,
           "status": "not_reached_on_grid", "nonmonotone_observed": bool(np.any(np.diff(p) > 0))}
    found = np.flatnonzero(p <= target)
    if not len(found):
        return out
    j = int(found[0])
    out["bracket_high_snr_db"] = snr[j]
    if j == 0:
        out["status"] = "below_or_at_grid_min"
        return out
    out["bracket_low_snr_db"] = snr[j - 1]
    out["status"] = "empirical_bracket_only"
    if p[j] > 0 and min(g.sector_errors.iloc[j - 1:j + 1]) >= 10:
        out["interpolated_snr_db"] = snr[j - 1] + (snr[j] - snr[j - 1]) * (
            (np.log10(target) - np.log10(p[j - 1])) / (np.log10(p[j]) - np.log10(p[j - 1])))
        out["status"] = "interpolated_with_at_least_10_failures_each"
    if out["nonmonotone_observed"]:
        out["status"] += ";nonmonotone_check_raw"
    return out


def summarize(directory: Path, manifest: dict) -> dict:
    plan = manifest["plan"]
    raw_rows, complete = [], []
    for chunk in range(plan["chunks"]):
        chunk_rows = []
        for si in range(len(plan["snr_db"])):
            for ti in range(len(plan["detector_taps"])):
                path = point_path(directory, chunk, si, ti)
                if path.exists():
                    chunk_rows.append(read_checkpoint(path, manifest, chunk, si, ti))
        raw_rows.extend(chunk_rows)
        if len(chunk_rows) == len(plan["snr_db"]) * len(plan["detector_taps"]):
            complete.append(chunk)
    status = {"saved_points": len(raw_rows), "complete_chunks": len(complete),
              "planned_chunks": plan["chunks"], "balanced_sectors_per_point": len(complete) * plan["sectors_per_chunk"],
              "complete": len(complete) == plan["chunks"],
              "inference": "Fixed-SNR paired chunk bootstrap; pointwise exploratory intervals, not simultaneous/global-optimum proof."}
    folder = directory / "summaries"
    raw = pd.DataFrame(raw_rows)
    atomic_csv(folder / "raw_results.csv", raw)
    if not complete:
        atomic_json(folder / "status.json", status)
        print(json.dumps(status), flush=True)
        return status
    # Comparisons only use complete rounds across ALL taps and ALL SNRs.
    balanced = raw[raw.chunk_index.isin(complete)].copy()
    agg = aggregate_fsr(balanced, ["iti_coeff", "detector_iti_coeff"])
    intervals = [wilson_interval(int(r.sector_errors), int(r.sector_count)) for r in agg.itertuples()]
    agg["wilson95_low_iid"] = [v[0] for v in intervals]
    agg["wilson95_high_iid"] = [v[1] for v in intervals]
    agg["independent_chunks"] = len(complete)
    agg["zero_fail_upper95_iid_exact"] = [
        -math.expm1(math.log(.05) / n) if e == 0 else np.nan
        for e, n in zip(agg.sector_errors, agg.sector_count)]
    atomic_csv(folder / "aggregate.csv", agg)

    comparisons, winners, crossings = [], [], []
    rng = np.random.default_rng(plan["master_seed"] + 1)
    for snr in plan["snr_db"]:
        cell = balanced[balanced.snr_db == snr]
        errors = cell.pivot(index="chunk_index", columns="detector_iti_coeff", values="sector_errors")
        errors = errors.reindex(index=complete, columns=plan["detector_taps"]).to_numpy(float)
        rates = errors / plan["sectors_per_chunk"]
        delta = rates - rates[:, [plan["detector_taps"].index(plan["reference_tap"])]]
        lower = upper = np.full(len(plan["detector_taps"]), np.nan)
        if len(complete) >= 5:
            boot = []
            # Bounded batches: never allocate bootstraps x all simulated sectors.
            for start in range(0, plan["bootstrap_replicates"], 128):
                size = min(128, plan["bootstrap_replicates"] - start)
                weights = rng.multinomial(len(complete), np.full(len(complete), 1 / len(complete)), size=size)
                boot.append(weights @ delta / len(complete))
            lower, upper = np.quantile(np.concatenate(boot), [.025, .975], axis=0)
        mean = rates.mean(axis=0)
        ties = [tap for tap, value in zip(plan["detector_taps"], mean) if np.isclose(value, mean.min(), rtol=0, atol=1e-14)]
        winners.append({"snr_db": snr, "minimum_observed_FSR": mean.min(),
                        "tied_detector_taps": ",".join(map(str, ties)),
                        "sectors_per_tap": status["balanced_sectors_per_point"],
                        "status": "observed_grid_minimum_not_proven_optimum"})
        for ti, tap in enumerate(plan["detector_taps"]):
            comparisons.append({"snr_db": snr, "detector_iti_coeff": tap,
                                "reference_tap": plan["reference_tap"], "FSR": mean[ti],
                                "delta_FSR_vs_reference": delta[:, ti].mean(),
                                "paired_chunk_bootstrap95_low": lower[ti],
                                "paired_chunk_bootstrap95_high": upper[ti],
                                "independent_chunks": len(complete),
                                "degenerate_chunk_differences": bool(np.ptp(delta[:, ti]) == 0)})
    for tap, group in agg.groupby("detector_iti_coeff"):
        for target in plan["targets"]:
            crossings.append(dict(detector_iti_coeff=tap, **observed_crossing(group, target)))
    atomic_csv(folder / "paired_vs_reference.csv", pd.DataFrame(comparisons))
    atomic_csv(folder / "observed_best_by_snr.csv", pd.DataFrame(winners))
    atomic_csv(folder / "observed_target_brackets.csv", pd.DataFrame(crossings))
    fits = []
    for cap in [.1, .3, .5]:
        # Keep low-count points and zeros in the raw/aggregate data, but do not
        # let one or two failures determine an extrapolated target SNR.
        fitted, _ = extrapolate_fsr_targets(
            agg[agg.sector_errors >= 10], ["iti_coeff", "detector_iti_coeff"], plan["targets"],
            max_fit_fsr=cap, min_points=3)
        fitted["max_fit_fsr"] = cap
        fitted["evaluation_status"] = "exploratory_extrapolation_not_measured"
        fits.append(fitted)
    atomic_csv(folder / "extrapolation_sensitivity.csv", pd.concat(fits, ignore_index=True))
    atomic_json(folder / "status.json", status)
    print("SUMMARY " + json.dumps(status), flush=True)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-jobs", type=int, help="Stop normally after this many channel-grid jobs; restart to continue")
    parser.add_argument("--worker", nargs=2, type=int, metavar=("CHUNK", "SNR_INDEX"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan.resolve())
        directory = (args.out_dir or ROOT / "outputs" / "tap_search" / plan["name"]).resolve()
        if args.max_jobs is not None and args.max_jobs <= 0:
            raise ValueError("--max-jobs must be positive")
        if args.worker:
            # Child processes take the same lock; an orphaned worker after a
            # killed parent cannot race with the restarted campaign.
            with campaign_lock(directory):
                manifest = prepare(directory, plan)
                run_job(directory, manifest, *args.worker)
            return 0
        with campaign_lock(directory):
            manifest = prepare(directory, plan)
            print(f"Campaign: {directory}\n"
                  f"{len(plan['detector_taps'])} taps x {len(plan['snr_db'])} SNRs x "
                  f"{plan['chunks']} independent chunks x {plan['sectors_per_chunk']} sectors\n"
                  "Sequential subprocesses; checkpoints after every tap. Re-run the same command to resume.", flush=True)
            if args.prepare_only:
                return 0
            if args.summarize_only:
                summarize(directory, manifest)
                return 0
        # A separate supervisor lock prevents two queues without blocking worker locks.
        with campaign_lock(directory / "supervisor"):
            done_jobs = 0
            for chunk in range(plan["chunks"]):
                chunk_ran = False
                for si in range(len(plan["snr_db"])):
                    with campaign_lock(directory):
                        pending = missing_taps(directory, manifest, chunk, si)
                    if not pending:
                        continue
                    print(f"RUN chunk={chunk + 1} SNR={plan['snr_db'][si]} pending_taps={len(pending)}", flush=True)
                    command = [sys.executable, str(ROOT / "scripts" / "run_tap_search.py"),
                               str(args.plan.resolve()), "--out-dir", str(directory), "--worker", str(chunk), str(si)]
                    child = subprocess.Popen(command)
                    try:
                        rc = child.wait()
                    except KeyboardInterrupt:
                        child.terminate()
                        child.wait()
                        raise
                    if rc:
                        print(f"Worker failed ({rc}); completed checkpoints are kept. Resume with the same command.", file=sys.stderr)
                        return 1
                    done_jobs += 1
                    chunk_ran = True
                    if args.max_jobs is not None and done_jobs >= args.max_jobs:
                        with campaign_lock(directory):
                            summarize(directory, manifest)
                        return 0
                if (chunk_ran and (chunk + 1) % plan["summary_every_chunks"] == 0
                        and chunk + 1 < plan["chunks"]):
                    with campaign_lock(directory):
                        summarize(directory, manifest)
            with campaign_lock(directory):
                summarize(directory, manifest)
            return 0
    except KeyboardInterrupt:
        print("Stopped. Completed tap checkpoints are retained.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
