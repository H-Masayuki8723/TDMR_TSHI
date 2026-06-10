"""Command-line interface (stdlib argparse).

Commands:

* ``tdmr2d smoke``               -- minimal end-to-end self-check + sample output
* ``tdmr2d run CONFIG.yaml``     -- single-condition BER evaluation
* ``tdmr2d sweep CONFIG.yaml``   -- SNR x ITI sweep with BER curves
* ``tdmr2d compare CONFIG.yaml`` -- multi-family BER overlays
* ``tdmr2d ldpc CONFIG.yaml``    -- uncoded-modulation LDPC post-ECC BER/FER
* ``tdmr2d concat CONFIG.yaml``  -- turbo LDPC + inner constrained-code SISO demapping
* ``tdmr2d boundary-scan CONFIG``-- scan trellis-boundary pruning candidates
* ``tdmr2d rate-plan CONFIG``    -- plan high-rate 4KB-sector LDPC geometry
* ``tdmr2d iti-calibrate CONFIG``-- scan ITI coefficients against a BER target
* ``tdmr2d channel-metrics CONFIG`` -- record tap/effective ITI comparison metrics
* ``tdmr2d fsr-extrapolate SRC`` -- aggregate sector chunks and extrapolate target FSR SNR
* ``tdmr2d summarize OUTPUT_DIR``-- aggregate runs into one summary CSV

argparse is used (the spec allows Typer *or* argparse) to keep the runtime
dependency-free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from . import __version__
from .config import Config
from .experiments import (run_boundary_scan, run_channel_metrics, run_compare,
                          run_concatenated, run_iti_calibration, run_ldpc,
                          run_rate_plan, run_single, run_sweep)
from .fsr import (aggregate_fsr, extrapolate_fsr_targets, load_fsr_rows,
                  parse_column_list, parse_float_list, plot_fsr_extrapolation)
from .io import (ensure_output_tree, make_run_dir, rows_to_frame, save_csv,
                 save_json, setup_logger, timestamp, REQUIRED_COLUMNS)
from .reports import (build_summary, plot_ber_vs_iti, plot_ber_vs_snr,
                      plot_compare_vs_iti, plot_compare_vs_snr,
                      plot_ldpc_ber, plot_ldpc_fer, plot_sector_fsr)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _ensure_siblings(output_dir: str) -> None:
    """Ensure outputs/{runs,summaries,figures,reports} exist next to the run dir."""
    base = Path(output_dir)
    root = base.parent if base.name == "runs" else base
    ensure_output_tree(root)


_EXTRA_ROOT_SECTIONS = {
    "ldpc", "boundary_scan", "iti_calibration", "rate_plan", "channel_metrics",
}


def _parse_int_values(spec, default: List[int]) -> List[int]:
    """Parse comma/range integer specs like ``3,4,6:8`` or return ``default``."""
    if spec is None:
        return list(default)
    if isinstance(spec, (list, tuple)):
        return sorted({int(v) for v in spec})
    values = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(p) for p in part.split(":")]
            if len(pieces) == 2:
                start, stop = pieces
                step = 1 if stop >= start else -1
            elif len(pieces) == 3:
                start, stop, step = pieces
                if step == 0:
                    raise ValueError("range step must not be zero")
            else:
                raise ValueError(f"bad integer range: {part!r}")
            end = stop + (1 if step > 0 else -1)
            values.extend(range(start, end, step))
        else:
            values.append(int(part))
    return sorted(set(values))


def _smoke_config(name: str, *, family: str = "uncoded", noiseless: bool = False,
                  no_interference: bool = False, snr_db: float = 12.0,
                  num_tracks: int = 16, bits_per_track: int = 1024,
                  seed: int = 0) -> Config:
    ch = {"c0": 1.0, "c_down_prev": 0.15, "c_down_next": 0.15,
          "c_cross_up": 0.10, "c_cross_down": 0.10,
          "snr_db": None if noiseless else snr_db, "boundary": "zero"}
    if no_interference:
        ch.update(c_down_prev=0.0, c_down_next=0.0, c_cross_up=0.0, c_cross_down=0.0)
    return Config.from_dict({
        "experiment": {"name": name, "family": family, "seed": seed,
                       "num_tracks": num_tracks, "bits_per_track": bits_per_track},
        "code": {"type": family, "block_shape": [8, 2]},
        "channel": ch,
        "detector": {"type": "hard_threshold", "threshold": 0.0},
        "metrics": {"target_ber": 1.0e-2},
        "output": {"dir": "outputs/runs"},
    })


# --------------------------------------------------------------------------- #
# commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_smoke(args) -> int:
    root = ensure_output_tree(args.output_root)
    print("== tdmr2d smoke ==")
    checks = []

    # Check 1: noiseless + no interference -> BER must be exactly 0.
    r1 = run_single(_smoke_config("smoke_noiseless_clean", noiseless=True,
                                  no_interference=True), cache_dir=args.cache_dir)
    ok1 = r1["BER"] == 0.0
    checks.append(("noiseless + no interference -> BER==0", ok1, f"BER={r1['BER']:.3e}"))

    # Check 2: noiseless WITH default interference -> still BER 0 (ISI/ITI < c0).
    r2 = run_single(_smoke_config("smoke_noiseless_iti", noiseless=True),
                    cache_dir=args.cache_dir)
    ok2 = r2["BER"] == 0.0
    checks.append(("noiseless + default interference -> BER==0", ok2, f"BER={r2['BER']:.3e}"))

    # Check 3: fixed seed reproduces (same bit errors / BER across two runs).
    ca = _smoke_config("smoke_repro", snr_db=12.0, seed=0)
    ra, rb = run_single(ca, cache_dir=args.cache_dir), run_single(ca, cache_dir=args.cache_dir)
    ok3 = (ra["bit_errors"] == rb["bit_errors"]) and (ra["BER"] == rb["BER"])
    checks.append(("fixed seed reproduces noise", ok3, f"bit_errors={ra['bit_errors']}=={rb['bit_errors']}"))

    # Check 4: a real small run writes CSV + JSON under outputs/.
    run_dir = make_run_dir(root / "runs")
    logger = setup_logger("tdmr2d.smoke", run_dir / "run.log")
    cfg = _smoke_config("smoke_run", snr_db=12.0, seed=0)
    cfg.save_resolved(run_dir / "config.resolved.yaml")
    row = run_single(cfg, cache_dir=args.cache_dir, logger=logger)
    csv_path = save_csv([row], run_dir / "results.csv")
    json_path = save_json({"config": cfg.to_dict(), "results": [row]}, run_dir / "results.json")
    ok4 = csv_path.exists() and json_path.exists()
    checks.append(("CSV + JSON written to outputs/", ok4, str(run_dir)))

    print()
    all_ok = True
    for desc, ok, detail in checks:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}  ({detail})")
    print()
    if all_ok:
        print(f"SMOKE PASSED. Sample BER(SNR=12dB) = {row['BER']:.3e}")
        print(f"Artifacts: {run_dir}")
        return 0
    print("SMOKE FAILED.")
    return 1


def _evaluate(args, sweep: bool) -> int:
    cfg = Config.load(args.config)
    _ensure_siblings(cfg.output.dir)
    run_dir = make_run_dir(cfg.output.dir)
    logger = setup_logger(f"tdmr2d.{'sweep' if sweep else 'run'}", run_dir / "run.log")
    cfg.save_resolved(run_dir / "config.resolved.yaml")
    logger.info("loaded config %s (family=%s) -> %s", args.config, cfg.experiment.family, run_dir)

    if sweep:
        if cfg.sweep is None:
            logger.error("config has no 'sweep' section")
            return 2
        rows = run_sweep(cfg, cache_dir=args.cache_dir, logger=logger)
        title_extra = f"{cfg.experiment.family} (rate={rows[0]['rate']:.4f})"
    else:
        rows = [run_single(cfg, cache_dir=args.cache_dir, logger=logger)]
        title_extra = f"{cfg.experiment.family} (rate={rows[0]['rate']:.4f})"

    save_csv(rows, run_dir / "results.csv")
    save_json({"config": cfg.to_dict(), "results": rows}, run_dir / "results.json")

    df = rows_to_frame(rows)
    plot_ber_vs_snr(df, run_dir / "ber_vs_snr.png", target_ber=cfg.metrics.target_ber,
                    title=f"pre-ECC BER vs SNR -- {title_extra}")
    plot_ber_vs_iti(df, run_dir / "ber_vs_iti.png", target_ber=cfg.metrics.target_ber,
                    title=f"pre-ECC BER vs ITI -- {title_extra}")

    bers = [r["BER"] for r in rows]
    logger.info("done: %d point(s), BER in [%.3e, %.3e], target=%.1e",
                len(rows), min(bers), max(bers), cfg.metrics.target_ber)
    print(f"\nWrote {len(rows)} result row(s) to {run_dir}")
    print(f"  results.csv / results.json / ber_vs_snr.png / ber_vs_iti.png / run.log")
    print(f"  BER range: [{min(bers):.3e}, {max(bers):.3e}]  target={cfg.metrics.target_ber:.1e}")
    return 0


def cmd_run(args) -> int:
    return _evaluate(args, sweep=False)


def cmd_sweep(args) -> int:
    return _evaluate(args, sweep=True)


def cmd_compare(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    out_dir = raw.get("output", {}).get("dir", "outputs/runs")
    _ensure_siblings(out_dir)
    run_dir = make_run_dir(out_dir)
    logger = setup_logger("tdmr2d.compare", run_dir / "run.log")
    logger.info("compare config %s -> %s", args.config, run_dir)

    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    rows, meta = run_compare(raw, cache_dir=args.cache_dir, logger=logger)
    save_csv(rows, run_dir / "results.csv")
    save_json({"config": raw, "compare": meta, "results": rows}, run_dir / "results.json")

    df = rows_to_frame(rows)
    target = raw.get("metrics", {}).get("target_ber", 1.0e-2)
    if meta["slice_iti"] is not None:
        plot_compare_vs_snr(df, run_dir / "compare_ber_vs_snr.png", meta["slice_iti"], target)
    if meta["slice_snr_db"] is not None:
        plot_compare_vs_iti(df, run_dir / "compare_ber_vs_iti.png", meta["slice_snr_db"], target)

    # Console summary: rate + BER at the slice point, per family.
    print(f"\nCompared {len(meta['families'])} families over {len(rows)} points -> {run_dir}")
    print(f"  slice: SNR={meta['slice_snr_db']} dB, ITI={meta['slice_iti']}")
    s = df[(pd.to_numeric(df['snr_db'], errors='coerce') == meta['slice_snr_db'])
           & (pd.to_numeric(df['iti_coeff'], errors='coerce') == meta['slice_iti'])]
    for _, r in s.sort_values("BER").iterrows():
        print(f"    {r['family']:<12} rate={r['rate']:.4f}  BER={r['BER']:.3e}")
    print("  files: results.csv / results.json / compare_ber_vs_snr.png / compare_ber_vs_iti.png / run.log")
    return 0


def cmd_ldpc(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    ldpc_params = raw.get("ldpc", {}) or {}
    base = {k: v for k, v in raw.items() if k not in _EXTRA_ROOT_SECTIONS}
    base.setdefault("experiment", {}).setdefault("family", "uncoded")
    base.setdefault("code", {}).setdefault("type", "uncoded")
    cfg = Config.from_dict(base)

    _ensure_siblings(cfg.output.dir)
    run_dir = make_run_dir(cfg.output.dir)
    logger = setup_logger("tdmr2d.ldpc", run_dir / "run.log")
    logger.info("ldpc config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, meta = run_ldpc(cfg, ldpc_params, cache_dir=args.cache_dir, logger=logger)
    save_csv(rows, run_dir / "results.csv")
    save_json({"config": raw, "ldpc": meta, "results": rows}, run_dir / "results.json")

    df = rows_to_frame(rows)
    plot_ldpc_ber(df, run_dir / "ldpc_ber_vs_snr.png", target_ber=cfg.metrics.target_ber)
    plot_ldpc_fer(df, run_dir / "ldpc_fer_vs_snr.png")
    fsr_written = False
    if "FSR" in df and pd.to_numeric(df["FSR"], errors="coerce").notna().any():
        plot_sector_fsr(df, run_dir / "ldpc_fsr_vs_snr.png")
        fsr_written = True

    print(f"\nLDPC n={meta['n']} k={meta['k']} rate={meta['rate']:.3f} "
          f"({meta['method']}, scale={meta['scale']}, max_iters={meta['max_iters']}, "
          f"{meta['num_frames']} frames/pt) -> {run_dir}")
    for r in sorted(rows, key=lambda r: (r["iti_coeff"], r["snr_db"])):
        fsr = "" if r.get("FSR") is None else f"  FSR={r['FSR']:.3e}"
        print(f"  snr={r['snr_db']:>4} iti={r['iti_coeff']:.2f}  "
              f"pre-ECC BER={r['pre_ecc_ber']:.3e}  post-ECC BER={r['BER']:.3e}  "
              f"FER={r['block_error_rate']:.3e}{fsr}")
    files = "results.csv / results.json / ldpc_ber_vs_snr.png / ldpc_fer_vs_snr.png"
    if fsr_written:
        files += " / ldpc_fsr_vs_snr.png"
    print(f"  files: {files} / run.log")
    return 0


def cmd_concat(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    ldpc_params = raw.get("ldpc", {}) or {}
    base = {k: v for k, v in raw.items() if k not in _EXTRA_ROOT_SECTIONS}
    cfg = Config.from_dict(base)

    _ensure_siblings(cfg.output.dir)
    run_dir = make_run_dir(cfg.output.dir)
    logger = setup_logger("tdmr2d.concat", run_dir / "run.log")
    logger.info("concat config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, meta = run_concatenated(cfg, ldpc_params, cache_dir=args.cache_dir, logger=logger)
    save_csv(rows, run_dir / "results.csv")
    save_json({"config": raw, "concat": meta, "results": rows}, run_dir / "results.json")

    df = rows_to_frame(rows)
    plot_ldpc_ber(df, run_dir / "concat_ber_vs_snr.png", target_ber=cfg.metrics.target_ber)
    plot_ldpc_fer(df, run_dir / "concat_fer_vs_snr.png")
    fsr_written = False
    if "FSR" in df and pd.to_numeric(df["FSR"], errors="coerce").notna().any():
        plot_sector_fsr(df, run_dir / "concat_fsr_vs_snr.png")
        fsr_written = True

    print(f"\nConcatenated LDPC + {meta['inner_family']} "
          f"(outer n={meta['n']} k={meta['k']} rate={meta['outer_ldpc_rate']:.3f}, "
	          f"inner rate={meta['inner_rate']:.3f}, total rate={rows[0]['rate']:.3f}, "
	          f"turbo_iterations={meta.get('turbo_iterations', 0)}, "
	          f"inner_demapper={meta.get('inner_demapper', 'exact_codebook')}, "
	          f"channel_detector={meta.get('channel_detector', 'soft_awgn')}, "
	          f"eq_iters={meta.get('equalizer_iterations', 0)}) -> {run_dir}")
    for r in sorted(rows, key=lambda r: (r["iti_coeff"], r["snr_db"])):
        fsr = "" if r.get("FSR") is None else f"  FSR={r['FSR']:.3e}"
        print(f"  snr={r['snr_db']:>4} iti={r['iti_coeff']:.2f}  "
              f"inner BER={r['inner_channel_ber']:.3e}  "
              f"pre-LDPC BER={r['pre_ecc_ber']:.3e}  "
              f"final-in BER={r.get('final_ldpc_input_ber', r['pre_ecc_ber']):.3e}  "
              f"post-ECC BER={r['BER']:.3e}  FER={r['block_error_rate']:.3e}{fsr}")
    files = "results.csv / results.json / concat_ber_vs_snr.png / concat_fer_vs_snr.png"
    if fsr_written:
        files += " / concat_fsr_vs_snr.png"
    print(f"  files: {files} / run.log")
    return 0


def cmd_boundary_scan(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    ldpc_params = raw.get("ldpc", {}) or {}
    scan_params = raw.get("boundary_scan", {}) or {}
    base = {k: v for k, v in raw.items() if k not in _EXTRA_ROOT_SECTIONS}
    cfg = Config.from_dict(base)

    max_down = max(8, cfg.code.trellis_boundary_down_mtr)
    max_checker = max(8, cfg.code.trellis_boundary_max_checker_run)
    default_down = list(range(cfg.code.down_mtr, max_down + 1))
    default_checker = list(range(cfg.code.max_checker_run, max_checker + 1))
    down_values = _parse_int_values(
        args.down if args.down is not None else scan_params.get("boundary_down_mtr"),
        default_down,
    )
    checker_values = _parse_int_values(
        args.checker if args.checker is not None else scan_params.get("boundary_max_checker_run"),
        default_checker,
    )
    num_trials = int(args.trials if args.trials is not None else scan_params.get("num_trials", 4))

    _ensure_siblings(cfg.output.dir)
    run_dir = make_run_dir(cfg.output.dir)
    logger = setup_logger("tdmr2d.boundary_scan", run_dir / "run.log")
    logger.info("boundary scan config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, meta = run_boundary_scan(
        cfg, ldpc_params, down_values, checker_values,
        num_trials=num_trials, cache_dir=args.cache_dir, logger=logger,
    )
    csv_path = save_csv(rows, run_dir / "boundary_scan.csv")
    save_json({"config": raw, "boundary_scan": meta, "results": rows}, run_dir / "boundary_scan.json")

    df = rows_to_frame(rows)
    candidates = df[df["candidate_zero_sample_violation"] == True].copy()  # noqa: E712
    print(f"\nBoundary scan {len(rows)} setting(s) -> {run_dir}")
    print(f"  down={down_values} checker={checker_values} trials={num_trials}")
    if candidates.empty:
        zero = df[df["tx_trellis_transition_violations"] == 0].sort_values(
            ["trellis_state_transition_density", "boundary_down_mtr", "boundary_max_checker_run"]
        )
        print("  candidate_zero_sample_violation: none")
        if not zero.empty:
            r = zero.iloc[0]
            print("  best zero-violation setting has no structural pruning: "
                  f"down={int(r['boundary_down_mtr'])} checker={int(r['boundary_max_checker_run'])} "
                  f"density={r['trellis_state_transition_density']:.3f}")
    else:
        candidates = candidates.sort_values(
            ["trellis_state_transition_density", "boundary_down_mtr", "boundary_max_checker_run"]
        )
        print("  candidate_zero_sample_violation settings:")
        for _, r in candidates.head(10).iterrows():
            print(f"    down={int(r['boundary_down_mtr'])} "
                  f"checker={int(r['boundary_max_checker_run'])} "
                  f"density={r['trellis_state_transition_density']:.3f} "
                  f"tx_violation={r['tx_trellis_transition_violation_rate']:.3e}")
    print(f"  files: {csv_path.name} / boundary_scan.json / run.log")
    return 0


def cmd_rate_plan(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    out_dir = raw.get("output", {}).get("dir", "outputs/runs")
    _ensure_siblings(out_dir)
    run_dir = make_run_dir(out_dir)
    logger = setup_logger("tdmr2d.rate_plan", run_dir / "run.log")
    logger.info("rate plan config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, meta = run_rate_plan(raw, cache_dir=args.cache_dir)
    csv_path = save_csv(rows, run_dir / "rate_plan.csv")
    save_json({"config": raw, "rate_plan": meta, "results": rows}, run_dir / "rate_plan.json")

    print(f"\nHigh-rate geometry plan -> {run_dir}")
    for r in rows:
        total = r["exact_total_rate"] if r.get("exact_total_rate") is not None else r["design_total_rate"]
        outer = (
            r["exact_outer_ldpc_rate"]
            if r.get("exact_outer_ldpc_rate") is not None
            else r["design_outer_ldpc_rate"]
        )
        print(
            f"  K-inner rate={r['inner_rate']:.5f}  LDPC n={r['n']} dc={r['dc']} "
            f"outer rate~{outer:.5f} total~{total:.5f}"
        )
        print(
            f"    {int(r['sector_bytes']) if float(r['sector_bytes']).is_integer() else r['sector_bytes']}B sector: "
            f"frames/sector={r['frames_per_sector']}  "
            f"recommended_num_frames={r['recommended_num_frames']} "
            f"for {r['sector_count_target']} sectors"
        )
        print(f"    codebook: {r['codebook_status']}")
    print(f"  files: {csv_path.name} / rate_plan.json / run.log")
    return 0


def cmd_iti_calibrate(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    out_dir = raw.get("output", {}).get("dir", "outputs/runs")
    _ensure_siblings(out_dir)
    run_dir = make_run_dir(out_dir)
    logger = setup_logger("tdmr2d.iti_calibrate", run_dir / "run.log")
    logger.info("ITI calibration config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, best_rows, meta = run_iti_calibration(raw, cache_dir=args.cache_dir, logger=logger)
    csv_path = save_csv(rows, run_dir / "iti_calibration.csv")
    best_path = save_csv(best_rows, run_dir / "iti_calibration_best.csv")
    save_json({"config": raw, "iti_calibration": meta, "results": rows,
               "best": best_rows}, run_dir / "iti_calibration.json")

    print(f"\nITI calibration target BER={meta['target_ber']:.3e} -> {run_dir}")
    for r in sorted(best_rows, key=lambda x: (str(x["calibration_family_label"]), str(x["snr_db"]))):
        print(
            f"  {r['calibration_family_label']:<14} snr={r['snr_db']} "
            f"best_iti={r['iti_coeff']:.4f}  BER={r['BER']:.3e} "
            f"log10_error={r['calibration_abs_log10_error']:.3f}"
        )
    print(f"  files: {csv_path.name} / {best_path.name} / iti_calibration.json / run.log")
    return 0


def cmd_channel_metrics(args) -> int:
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    out_dir = raw.get("output", {}).get("dir", "outputs/runs")
    _ensure_siblings(out_dir)
    run_dir = make_run_dir(out_dir)
    logger = setup_logger("tdmr2d.channel_metrics", run_dir / "run.log")
    logger.info("channel metrics config %s -> %s", args.config, run_dir)
    with open(run_dir / "config.resolved.yaml", "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)

    rows, meta = run_channel_metrics(
        raw, cache_dir=args.cache_dir,
        sample_frames=args.sample_frames,
        chunk_tracks=args.chunk_tracks,
        logger=logger,
    )
    csv_path = save_csv(rows, run_dir / "channel_metrics.csv")
    save_json({"config": raw, "channel_metrics": meta, "results": rows},
              run_dir / "channel_metrics.json")

    print(f"\nChannel metrics {len(rows)} point(s) -> {run_dir}")
    print(
        f"  family={meta['family']} rate={meta['rate']:.5f} "
        f"sample={meta['sample_num_tracks']}x{meta['sample_bits_per_track']} "
        f"({meta['sample_num_bits']} channel bits)"
    )
    for r in sorted(rows, key=lambda x: (x["iti_coeff"], str(x["snr_db"]))):
        tap = r.get("tap_sir_db")
        eff = r.get("effective_sir_db")
        tap_s = "n/a" if tap is None else f"{tap:.3f} dB"
        eff_s = "n/a" if eff is None else f"{eff:.3f} dB"
        print(
            f"  snr={r['snr_db']} iti={r['iti_coeff']:.3f} "
            f"tap_SIR={tap_s} effective_SIR={eff_s} "
            f"cross_var_fraction={r.get('effective_cross_fraction_of_interference_variance')}"
        )
    print(f"  files: {csv_path.name} / channel_metrics.json / run.log")
    return 0


def cmd_fsr_extrapolate(args) -> int:
    targets = parse_float_list(args.targets)
    group_cols = parse_column_list(args.group_by)
    rows = load_fsr_rows(args.sources, name_filter=args.name_filter)
    if rows.empty:
        print("No sector-FSR rows found. Pass outputs/runs, a results.csv, or an aggregate CSV.")
        return 1
    group_cols = [c for c in group_cols if c in rows.columns]
    agg = aggregate_fsr(rows, group_cols)
    target_df, fit_points = extrapolate_fsr_targets(
        agg, group_cols, targets,
        max_fit_fsr=float(args.max_fit_fsr),
        min_fit_fsr=float(args.min_fit_fsr),
        min_points=int(args.min_points),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"fsr_extrapolation_{timestamp()}"
    agg_path = out_dir / f"{prefix}_aggregate.csv"
    fit_path = out_dir / f"{prefix}_fit_points.csv"
    target_path = out_dir / f"{prefix}_target_snr.csv"
    agg.to_csv(agg_path, index=False)
    fit_points.to_csv(fit_path, index=False)
    target_df.to_csv(target_path, index=False)
    plot_path = None
    if not args.no_plot and not target_df.empty:
        plot_path = plot_fsr_extrapolation(agg, target_df, group_cols, out_dir / f"{prefix}.png")

    print(f"\nFSR extrapolation from {len(rows)} raw row(s)")
    print(f"  aggregate: {agg_path}")
    print(f"  fit points: {fit_path}")
    print(f"  target SNR: {target_path}")
    if plot_path is not None:
        print(f"  plot: {plot_path}")
    sort_cols = group_cols + ["target_fsr"] if group_cols else ["target_fsr"]
    for _, r in target_df.sort_values(sort_cols).iterrows():
        group = " ".join(f"{c}={r[c]}" for c in group_cols if c in r)
        est = r["estimated_snr_db"]
        est_s = "nan" if pd.isna(est) else f"{est:.3f} dB"
        note = "" if not r.get("fit_note") else f"  note={r['fit_note']}"
        print(f"  {group} target={r['target_fsr']:.1e} estimated_snr={est_s}{note}")
    return 0


def cmd_summarize(args) -> int:
    df = build_summary([args.output_dir])
    if df.empty:
        print(f"No results.csv found under {args.output_dir}")
        return 1
    out = Path(args.out) if args.out else Path("outputs/summaries") / f"summary_{timestamp()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    print(f"Summarized {len(df)} row(s) from {args.output_dir} -> {out}")
    if missing:
        print(f"  WARNING: missing required columns: {missing}")
    else:
        print(f"  required columns present: {', '.join(REQUIRED_COLUMNS)}")
    return 0


# --------------------------------------------------------------------------- #
# parser                                                                       #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdmr2d",
        description="Pre-ECC BER evaluation of 2D constrained codes over a 2D TDMR readback channel.",
    )
    p.add_argument("--version", action="version", version=f"tdmr2d {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke", help="minimal end-to-end self-check + sample output")
    s.add_argument("--output-root", default="outputs")
    s.add_argument("--cache-dir", default="data/codebooks")
    s.set_defaults(func=cmd_smoke)

    r = sub.add_parser("run", help="single-condition BER evaluation")
    r.add_argument("config")
    r.add_argument("--cache-dir", default="data/codebooks")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("sweep", help="SNR x ITI sweep with BER curves")
    w.add_argument("config")
    w.add_argument("--cache-dir", default="data/codebooks")
    w.set_defaults(func=cmd_sweep)

    c = sub.add_parser("compare", help="compare multiple families over one SNR x ITI grid")
    c.add_argument("config")
    c.add_argument("--cache-dir", default="data/codebooks")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("ldpc", help="LDPC post-ECC BER/FER over the 2D channel (min-sum BP)")
    l.add_argument("config")
    l.add_argument("--cache-dir", default="data/codebooks")
    l.set_defaults(func=cmd_ldpc)

    x = sub.add_parser("concat", help="turbo LDPC + inner constrained-code SISO evaluation")
    x.add_argument("config")
    x.add_argument("--cache-dir", default="data/codebooks")
    x.set_defaults(func=cmd_concat)

    b = sub.add_parser("boundary-scan", help="scan trellis-boundary pruning settings")
    b.add_argument("config")
    b.add_argument("--cache-dir", default="data/codebooks")
    b.add_argument("--down", default=None, help="integer list/range, e.g. 3:8 or 4,5,6")
    b.add_argument("--checker", default=None, help="integer list/range, e.g. 0:8 or 1,2,3")
    b.add_argument("--trials", type=int, default=None, help="number of random LDPC/inner transmissions to sample")
    b.set_defaults(func=cmd_boundary_scan)

    rp = sub.add_parser("rate-plan", help="plan high-rate 4KB-sector LDPC/inner-code geometry")
    rp.add_argument("config")
    rp.add_argument("--cache-dir", default="data/codebooks")
    rp.set_defaults(func=cmd_rate_plan)

    ic = sub.add_parser("iti-calibrate", help="scan ITI coefficients and select closest BER-calibrated points")
    ic.add_argument("config")
    ic.add_argument("--cache-dir", default="data/codebooks")
    ic.set_defaults(func=cmd_iti_calibrate)

    cm = sub.add_parser("channel-metrics",
                        help="measure tap and sequence-dependent channel interference metrics")
    cm.add_argument("config")
    cm.add_argument("--cache-dir", default="data/codebooks")
    cm.add_argument("--sample-frames", type=int, default=None,
                    help="LDPC frames to sample when the config has an ldpc section")
    cm.add_argument("--chunk-tracks", type=int, default=256,
                    help="track chunk size for effective interference accumulation")
    cm.set_defaults(func=cmd_channel_metrics)

    fe = sub.add_parser("fsr-extrapolate", help="aggregate sector chunks and extrapolate target FSR SNR")
    fe.add_argument("sources", nargs="+", help="outputs/runs directory, results.csv, or aggregate CSV")
    fe.add_argument("--targets", default="1e-8,1e-9,1e-10",
                    help="comma-separated target FSR values")
    fe.add_argument("--group-by", default="iti_coeff",
                    help="comma-separated columns to fit separately, e.g. family,rate,iti_coeff")
    fe.add_argument("--name-filter", default=None,
                    help="only use rows whose name column contains this text")
    fe.add_argument("--max-fit-fsr", type=float, default=0.8,
                    help="exclude saturated points with FSR >= this value from the fit")
    fe.add_argument("--min-fit-fsr", type=float, default=0.0,
                    help="exclude points with FSR <= this value from the fit")
    fe.add_argument("--min-points", type=int, default=2,
                    help="minimum nonzero/non-saturated points required for a fit")
    fe.add_argument("--out-dir", default="outputs/summaries")
    fe.add_argument("--prefix", default=None)
    fe.add_argument("--no-plot", action="store_true")
    fe.set_defaults(func=cmd_fsr_extrapolate)

    m = sub.add_parser("summarize", help="aggregate runs into one summary CSV")
    m.add_argument("output_dir")
    m.add_argument("--out", default=None, help="output CSV path (default outputs/summaries/summary_<ts>.csv)")
    m.set_defaults(func=cmd_summarize)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
