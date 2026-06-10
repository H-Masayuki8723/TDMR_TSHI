"""Experiment orchestration: build coded grid -> channel -> detect -> metrics.

A *single run* evaluates one ``(family, SNR, ITI)`` operating point and returns a
flat result row (see :data:`tdmr2d.io.REQUIRED_COLUMNS`). A *sweep* runs the
Cartesian product of ``sweep.snr_db x sweep.iti_coeffs`` with independent but
reproducible RNG streams derived from the base seed.

Pre-ECC BER is measured by comparing the hard-decided received bits against the
transmitted *channel* bits (no decode-back, no ECC).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import channel as channel_mod
from . import codebook as codebook_mod
from . import concat as concat_mod
from . import metrics as metrics_mod
from . import patterns
from . import siso as siso_mod
from .config import Config
from .decoder import make_decoder
from .detector import make_detector
from .ldpc import LDPCCode


def _channel_llr_grid(y: np.ndarray, cfg: Config, taps: channel_mod.ChannelTaps,
                      sigma: float, params: Dict) -> Tuple[np.ndarray, Dict]:
    """Build channel-bit LLRs for LDPC/concat tracks.

    ``channel_detector`` defaults to the original memoryless main-tap AWGN LLR.
    ``bcjr_2d_equalized`` adds soft ITI cancellation and down-track BCJR ISI.
    """
    detector_name = str(params.get("channel_detector", "soft_awgn"))
    clip = float(params.get("channel_llr_clip", getattr(cfg.detector, "llr_clip", 20.0)))
    if detector_name in ("soft_awgn", "memoryless_awgn"):
        detector = make_detector(cfg.detector)
        llr = detector.soft_llr(y, sigma=sigma, amplitude=taps.c0)
        return llr, {
            "channel_detector": "soft_awgn",
            "equalizer_iterations": 0,
            "channel_llr_clip": clip,
        }
    if detector_name in ("bcjr_2d_equalized", "bcjr_isi", "bcjr"):
        return siso_mod.bcjr_2d_equalized_llr(
            y, taps, sigma=sigma, boundary=cfg.channel.boundary,
            iterations=int(params.get("equalizer_iterations", 2)),
            llr_clip=clip,
        )
    raise ValueError(
        "unknown channel_detector {!r}; expected soft_awgn or bcjr_2d_equalized".format(detector_name)
    )


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("division width must be positive")
    return -(-int(a) // int(b))


def _round_up_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return _ceil_div(int(value), multiple) * multiple


def _frame_multiple_for_inner(cfg: Config) -> int:
    if cfg.experiment.family == "mtr2d_8x2":
        return int(cfg.code.block_cross)
    return 1


def _resolve_num_frames(code: LDPCCode, ldpc_params: Dict, default: int,
                        *, frame_multiple: int = 1) -> int:
    """Resolve the number of LDPC codewords/frames for a run.

    ``ldpc.num_frames`` keeps the historical explicit behavior. Without it,
    ``ldpc.sector_count_target`` and ``ldpc.sector_bits`` can derive enough
    frames to observe the requested number of complete sectors.
    """
    if "num_frames" in ldpc_params:
        num_frames = int(ldpc_params["num_frames"])
    else:
        sector_bits = int(ldpc_params.get("sector_bits", 0) or 0)
        sector_count = int(ldpc_params.get("sector_count_target", 0) or 0)
        if sector_bits > 0 and sector_count > 0:
            num_frames = _ceil_div(sector_bits * sector_count, code.k)
            num_frames = _round_up_multiple(num_frames, frame_multiple)
        else:
            num_frames = int(default)
    if num_frames <= 0:
        raise ValueError("ldpc.num_frames must be positive")
    return num_frames


def _sector_error_stats(reference: np.ndarray, estimate: np.ndarray, sector_bits: int) -> Dict:
    """Aggregate decoded information bits into fixed-size sectors.

    A sector is failed if any information bit in the sector is wrong. Only
    complete sectors are counted; leftover bits are reported as discarded.
    """
    sector_bits = int(sector_bits or 0)
    if sector_bits <= 0:
        return {}
    ref = np.asarray(reference, dtype=np.uint8).reshape(-1)
    est = np.asarray(estimate, dtype=np.uint8).reshape(-1)
    if ref.shape != est.shape:
        raise ValueError("sector FSR reference/estimate shapes differ")
    sector_count = int(ref.size // sector_bits)
    used = sector_count * sector_bits
    discarded = int(ref.size - used)
    stats = {
        "sector_bits": sector_bits,
        "sector_bytes": float(sector_bits / 8.0),
        "sector_count": sector_count,
        "sector_discarded_bits": discarded,
    }
    if sector_count == 0:
        stats.update({
            "sector_errors": 0,
            "sector_error_rate": None,
            "FSR": None,
        })
        return stats
    sector_err = np.any(ref[:used].reshape(sector_count, sector_bits)
                        != est[:used].reshape(sector_count, sector_bits), axis=1)
    fsr = float(sector_err.mean())
    stats.update({
        "sector_errors": int(sector_err.sum()),
        "sector_error_rate": fsr,
        "FSR": fsr,
    })
    return stats


def _inner_rate_from_config(cfg: Config) -> float:
    if cfg.experiment.family == "uncoded":
        return 1.0
    if cfg.experiment.family == "mtr1d":
        return float(int(cfg.code.K1) / cfg.code.block_down)
    if cfg.experiment.family == "mtr2d_8x2":
        return float(int(cfg.code.K) / (cfg.code.block_down * cfg.code.block_cross))
    raise ValueError(f"unknown family {cfg.experiment.family!r}")


def run_rate_plan(raw: Dict, *, cache_dir: str = "data/codebooks") -> Tuple[List[Dict], Dict]:
    """Plan high-rate LDPC/inner-code geometry without running the channel.

    This is a sizing helper for the practical branch of the project: choose an
    inner rate, a high-rate regular Gallager LDPC design, and a 4KB sector target,
    then report the expected total rate and recommended frame count.
    """
    params = raw.get("rate_plan", {}) or {}
    ldpc_params = raw.get("ldpc", {}) or {}
    base = {k: copy.deepcopy(v) for k, v in raw.items()
            if k not in {"ldpc", "boundary_scan", "iti_calibration",
                         "rate_plan", "channel_metrics"}}
    cfg = Config.from_dict(base)

    target_total = float(params.get("target_total_rate", ldpc_params.get("target_total_rate", 0.90)))
    sector_bits = int(params.get("sector_bits", ldpc_params.get("sector_bits", 32768)))
    sector_count_target = int(params.get(
        "sector_count_target", ldpc_params.get("sector_count_target", 10000)
    ))
    dv = int(params.get("dv", ldpc_params.get("dv", 3)))
    dc_values = params.get("dc_candidates", None)
    if dc_values is None:
        dc_values = [int(ldpc_params.get("dc", 75))]
    dc_values = [int(v) for v in dc_values]
    n_base = params.get("n", ldpc_params.get("n", None))
    exact_ldpc_rate = bool(params.get("exact_ldpc_rate", False))
    check_codebook = bool(params.get("check_codebook", True))

    inner_rate = _inner_rate_from_config(cfg)
    required_outer = target_total / inner_rate
    frame_multiple = _frame_multiple_for_inner(cfg)

    codebook_status = "not_checked"
    codebook_num_valid = None
    if check_codebook and cfg.experiment.family in {"mtr1d", "mtr2d_8x2"}:
        try:
            cb = (
                concat_mod._stateful_candidate_codebook(cfg, cache_dir)
                if cfg.code.inner_encoder == "stateful_trellis"
                else concat_mod._inner_codebook(cfg, cache_dir)
            )
            codebook_status = "ok"
            codebook_num_valid = cb.meta.get("num_valid")
        except Exception as exc:  # sizing should report infeasible codebooks clearly
            codebook_status = f"error: {exc}"

    rows: List[Dict] = []
    for dc in dc_values:
        if dc <= dv:
            raise ValueError("dc must be greater than dv for a high-rate LDPC plan")
        if n_base is None:
            outer_bits_per_sector = _ceil_div(sector_bits, required_outer)
            n = _round_up_multiple(outer_bits_per_sector, dc)
        else:
            n = int(n_base)
            if n % dc:
                n = _round_up_multiple(n, dc)
        m_design = int(n * dv // dc)
        design_outer_rate = 1.0 - float(dv / dc)
        design_k = int(n - m_design)
        exact_k = None
        exact_rate = None
        if exact_ldpc_rate:
            code = LDPCCode.gallager(n=n, dv=dv, dc=dc, seed=int(ldpc_params.get("construction_seed", 1)))
            exact_k = code.k
            exact_rate = float(code.rate)
        effective_k = int(exact_k if exact_k is not None else design_k)
        frames_per_sector = _ceil_div(sector_bits, effective_k)
        recommended_frames = _round_up_multiple(
            _ceil_div(sector_bits * sector_count_target, effective_k),
            frame_multiple,
        )
        total_rate_design = float(inner_rate * design_outer_rate)
        total_rate_exact = None if exact_rate is None else float(inner_rate * exact_rate)
        rows.append({
            "family": cfg.experiment.family,
            "inner_encoder": cfg.code.inner_encoder,
            "inner_rate": inner_rate,
            "target_total_rate": target_total,
            "required_outer_ldpc_rate": required_outer,
            "dv": dv,
            "dc": dc,
            "n": n,
            "design_ldpc_checks": m_design,
            "design_ldpc_k": design_k,
            "design_outer_ldpc_rate": design_outer_rate,
            "exact_ldpc_k": exact_k,
            "exact_outer_ldpc_rate": exact_rate,
            "design_total_rate": total_rate_design,
            "exact_total_rate": total_rate_exact,
            "sector_bits": sector_bits,
            "sector_bytes": float(sector_bits / 8.0),
            "sector_count_target": sector_count_target,
            "frames_per_sector": frames_per_sector,
            "recommended_num_frames": recommended_frames,
            "inner_frame_multiple": frame_multiple,
            "codebook_status": codebook_status,
            "codebook_num_valid": codebook_num_valid,
        })
    meta = {
        "target_total_rate": target_total,
        "sector_bits": sector_bits,
        "sector_count_target": sector_count_target,
        "inner_rate": inner_rate,
        "required_outer_ldpc_rate": required_outer,
    }
    return rows, meta


def run_iti_calibration(raw: Dict, *, cache_dir: str = "data/codebooks",
                        logger=None) -> Tuple[List[Dict], List[Dict], Dict]:
    """Scan ITI coefficients and mark coefficients closest to a target BER.

    This supports the "nominal ITI coefficient is not physically equivalent"
    branch: calibrate comparison points by a measurable reference such as
    uncoded pre-ECC BER, then reuse the selected coefficients in main runs.
    """
    params = raw.get("iti_calibration", {}) or {}
    base = {k: copy.deepcopy(v) for k, v in raw.items()
            if k not in {"compare", "ldpc", "boundary_scan", "iti_calibration",
                         "rate_plan", "channel_metrics"}}
    target = float(params.get("target_ber", raw.get("metrics", {}).get("target_ber", 1.0e-2)))
    if target <= 0:
        raise ValueError("iti_calibration.target_ber must be positive")
    snrs = params.get("snr_db", raw.get("sweep", {}).get("snr_db", None))
    if snrs is None:
        snrs = [base.get("channel", {}).get("snr_db", 14.0)]
    itis = params.get("iti_coeffs", raw.get("sweep", {}).get("iti_coeffs", None))
    if itis is None:
        itis = [base.get("channel", {}).get("c_cross_up", 0.0)]
    snrs = [None if v is None else float(v) for v in snrs]
    itis = [float(v) for v in itis]
    families = params.get("families")
    if not families:
        exp = base.get("experiment", {})
        code = base.get("code", {})
        families = [{"family": exp.get("family", "uncoded"), "code": code}]

    all_rows: List[Dict] = []
    for entry in families:
        fam = entry["family"]
        cfg_dict = copy.deepcopy(base)
        exp = dict(cfg_dict.get("experiment", {}))
        exp["family"] = fam
        exp["name"] = entry.get("label", f"iti_cal_{fam}")
        if "experiment" in entry:
            exp.update(entry["experiment"])
            exp["family"] = fam
        cfg_dict["experiment"] = exp

        code = dict(cfg_dict.get("code", {}))
        code.update(entry.get("code", {}))
        code["type"] = fam
        code.setdefault("block_shape", [8, 2])
        cfg_dict["code"] = code
        for section in ("channel", "detector", "metrics", "output"):
            if section in entry:
                merged = dict(cfg_dict.get(section, {}))
                merged.update(entry[section])
                cfg_dict[section] = merged
        cfg = Config.from_dict(cfg_dict)

        points = [(s, i) for s in snrs for i in itis]
        children = np.random.SeedSequence(cfg.experiment.seed).spawn(len(points))
        for (snr, iti), ss in zip(points, children):
            row = run_single(
                cfg, snr_db=snr, iti_coeff=iti, rng=np.random.default_rng(ss),
                cache_dir=cache_dir, logger=logger,
            )
            floor = 0.5 / max(int(row["num_bits"]), 1)
            obs = max(float(row["BER"]), floor)
            row["calibration_family_label"] = entry.get("label", fam)
            row["calibration_target_ber"] = target
            row["calibration_abs_log10_error"] = abs(math.log10(obs) - math.log10(target))
            row["calibration_best_match"] = False
            all_rows.append(row)

    best_rows: List[Dict] = []
    groups = sorted({(r["calibration_family_label"], r["snr_db"]) for r in all_rows},
                    key=lambda x: (str(x[0]), str(x[1])))
    for label, snr in groups:
        candidates = [r for r in all_rows if r["calibration_family_label"] == label and r["snr_db"] == snr]
        best = min(candidates, key=lambda r: r["calibration_abs_log10_error"])
        best["calibration_best_match"] = True
        best_rows.append(dict(best))

    meta = {
        "target_ber": target,
        "snr_db": snrs,
        "iti_coeffs": itis,
        "families": [entry.get("label", entry["family"]) for entry in families],
    }
    return all_rows, best_rows, meta


def run_channel_metrics(raw: Dict, *, cache_dir: str = "data/codebooks",
                        sample_frames: Optional[int] = None,
                        chunk_tracks: int = 256,
                        logger=None) -> Tuple[List[Dict], Dict]:
    """Measure nominal/effective channel interference for a configured signal.

    This is deliberately separated from heavy BER/FSR runs. It records the
    physical-ish comparison columns needed for Level-3 1D/2D discussions without
    adding large intermediate arrays to the production concatenated decoder path.
    """
    params = raw.get("channel_metrics", {}) or {}
    ldpc_params = raw.get("ldpc", {}) or {}
    base = {k: copy.deepcopy(v) for k, v in raw.items()
            if k not in {"compare", "ldpc", "boundary_scan", "iti_calibration",
                         "rate_plan", "channel_metrics"}}
    cfg = Config.from_dict(base)
    rng = np.random.default_rng(int(params.get("seed", cfg.experiment.seed)))

    use_ldpc_inner = bool(params.get(
        "use_ldpc_inner_grid",
        bool(ldpc_params) and cfg.experiment.family in {"uncoded", "mtr1d", "mtr2d_8x2"},
    ))
    chunk = int(params.get("chunk_tracks", chunk_tracks))

    signal_meta: Dict = {}
    if use_ldpc_inner:
        code = LDPCCode.gallager(
            n=int(ldpc_params.get("n", 1800)),
            dv=int(ldpc_params.get("dv", 3)),
            dc=int(ldpc_params.get("dc", 75)),
            seed=int(ldpc_params.get("construction_seed", 1)),
        )
        default_frames = int(params.get("sample_frames", 256))
        frames = int(sample_frames if sample_frames is not None else default_frames)
        frames = _round_up_multiple(max(1, frames), _frame_multiple_for_inner(cfg))
        U = rng.integers(0, 2, size=(frames, code.k), dtype=np.uint8)
        C = code.encode(U)
        tx_grid, inner_meta = concat_mod.encode_inner_grid(C, cfg, cache_dir=cache_dir)
        signal_rate = float(code.rate * inner_meta["inner_rate"])
        signal_family = f"ldpc+{cfg.experiment.family}"
        signal_meta.update({
            "use_ldpc_inner_grid": True,
            "outer_ldpc_n": code.n,
            "outer_ldpc_k": code.k,
            "outer_ldpc_rate": float(code.rate),
            "sample_frames": frames,
        })
        signal_meta.update({k: v for k, v in inner_meta.items() if k not in signal_meta})
    else:
        tracks = int(params.get("sample_tracks", cfg.experiment.num_tracks))
        bits_per_track = int(params.get("sample_bits_per_track", cfg.experiment.bits_per_track))
        sample_cfg = replace(
            cfg,
            experiment=replace(cfg.experiment, num_tracks=tracks, bits_per_track=bits_per_track),
        )
        tx_grid, signal_rate, code_meta = build_tx_grid(sample_cfg, rng, cache_dir=cache_dir)
        signal_family = cfg.experiment.family
        signal_meta.update({
            "use_ldpc_inner_grid": False,
            "sample_frames": None,
            "inner_code": code_meta.get("code"),
            "inner_rate": float(signal_rate),
        })

    symbols = channel_mod.map_bits_to_symbols(tx_grid)
    snrs = list(cfg.sweep.snr_db) if cfg.sweep and cfg.sweep.snr_db else [cfg.channel.snr_db]
    itis = list(cfg.sweep.iti_coeffs) if cfg.sweep and cfg.sweep.iti_coeffs else [cfg.channel.c_cross_up]
    rows: List[Dict] = []
    for snr in snrs:
        for iti in itis:
            ch = replace(cfg.channel, c_cross_up=float(iti), c_cross_down=float(iti))
            taps = channel_mod.ChannelTaps.from_config(ch)
            sigma = channel_mod.snr_db_to_sigma(snr, ch.c0)
            row: Dict = {
                "name": cfg.experiment.name,
                "family": signal_family,
                "rate": signal_rate,
                "snr_db": snr,
                "iti_coeff": float(iti),
                "sample_num_bits": int(tx_grid.size),
                "sample_num_tracks": int(tx_grid.shape[0]),
                "sample_bits_per_track": int(tx_grid.shape[1]),
                "seed": cfg.experiment.seed,
                "boundary": ch.boundary,
                "channel_metric_type": "tap_and_effective_sequence_interference",
                "channel_metrics_chunk_tracks": chunk,
            }
            row.update(signal_meta)
            row.update(channel_mod.tap_energy_metrics(taps, sigma=sigma))
            row.update(channel_mod.effective_interference_metrics(
                symbols, taps, boundary=ch.boundary, sigma=sigma, chunk_tracks=chunk,
            ))
            rows.append(row)
            if logger is not None:
                logger.info(
                    "channel metrics family=%s rate=%.4f snr=%s iti=%.3f "
                    "eff_SIR=%s tap_SIR=%s sample_bits=%d",
                    signal_family, signal_rate, str(snr), float(iti),
                    row.get("effective_sir_db"), row.get("tap_sir_db"), tx_grid.size,
                )

    meta = {
        "name": cfg.experiment.name,
        "family": signal_family,
        "rate": signal_rate,
        "sample_num_bits": int(tx_grid.size),
        "sample_num_tracks": int(tx_grid.shape[0]),
        "sample_bits_per_track": int(tx_grid.shape[1]),
        "chunk_tracks": chunk,
    }
    meta.update(signal_meta)
    return rows, meta


# --------------------------------------------------------------------------- #
# coded-grid construction                                                      #
# --------------------------------------------------------------------------- #
def build_tx_grid(cfg: Config, rng: np.random.Generator, cache_dir: str = "data/codebooks"):
    """Build the transmitted channel-bit grid for the configured family.

    Returns ``(tx_grid, rate, code_meta)`` where ``tx_grid`` has shape
    ``(num_tracks, bits_per_track)`` and dtype ``uint8``.
    """
    fam = cfg.experiment.family
    T = cfg.experiment.num_tracks
    J = cfg.experiment.bits_per_track
    cross, down = cfg.code.block_cross, cfg.code.block_down

    if fam == "uncoded":
        tx = patterns.random_bit_grid(rng, T, J)
        return tx, 1.0, {"rate": 1.0, "code": "uncoded"}

    if fam == "mtr1d":
        cb = codebook_mod.get_1d_codebook(
            K1=cfg.code.K1, length=down, down_mtr=cfg.code.down_mtr, cache_dir=cache_dir
        )
        nbt = J // cb.block_len
        idx = rng.integers(0, cb.size, size=(T, nbt))
        words = cb.words[idx]                       # (T, nbt, L)
        tx = words.reshape(T, nbt * cb.block_len).astype(np.uint8)
        meta = {"rate": cb.rate, "code": cb.name, "K1": cfg.code.K1,
                "down_mtr": cfg.code.down_mtr, "num_valid": cb.meta.get("num_valid")}
        return tx, cb.rate, meta

    if fam == "mtr2d_8x2":
        cb = codebook_mod.get_2d_codebook(
            K=cfg.code.K, cross=cross, down=down, down_mtr=cfg.code.down_mtr,
            max_checker_run=cfg.code.max_checker_run, cache_dir=cache_dir,
        )
        Tb, Jb = T // cross, J // down
        idx = rng.integers(0, cb.size, size=(Tb, Jb))
        blocks = cb.words[idx].reshape(Tb, Jb, cross, down)
        tx = blocks.transpose(0, 2, 1, 3).reshape(Tb * cross, Jb * down).astype(np.uint8)
        meta = {"rate": cb.rate, "code": cb.name, "K": cfg.code.K,
                "down_mtr": cfg.code.down_mtr, "max_checker_run": cfg.code.max_checker_run,
                "num_valid": cb.meta.get("num_valid")}
        return tx, cb.rate, meta

    raise ValueError(f"unknown family {fam!r}")


# --------------------------------------------------------------------------- #
# single run                                                                   #
# --------------------------------------------------------------------------- #
def run_single(cfg: Config, *, snr_db: Optional[float] = "__keep__",
               iti_coeff: Optional[float] = None,
               rng: Optional[np.random.Generator] = None,
               cache_dir: str = "data/codebooks",
               logger=None) -> Dict:
    """Evaluate one operating point and return a result row.

    ``snr_db="__keep__"`` keeps the config value; pass a number or ``None`` to
    override (``None`` = noiseless). ``iti_coeff`` (when given) overrides both
    cross-track taps. ``rng`` overrides the seed-derived generator (used by sweep
    for independent reproducible streams).
    """
    t0 = time.perf_counter()

    # Effective channel parameters (apply overrides).
    eff_snr = cfg.channel.snr_db if snr_db == "__keep__" else snr_db
    ch = cfg.channel
    if iti_coeff is not None:
        ch = replace(ch, c_cross_up=float(iti_coeff), c_cross_down=float(iti_coeff))
    iti_recorded = float(ch.c_cross_up)

    if rng is None:
        rng = np.random.default_rng(cfg.experiment.seed)

    # Encode -> symbols -> 2D channel -> detect.
    tx, rate, code_meta = build_tx_grid(cfg, rng, cache_dir=cache_dir)
    taps = channel_mod.ChannelTaps.from_config(ch)
    symbols = channel_mod.map_bits_to_symbols(tx)
    y = channel_mod.readback(symbols, taps, eff_snr, rng=rng, boundary=ch.boundary)
    detector = make_detector(cfg.detector)
    rx = detector.detect(y)  # hard decision -> pre-ECC BER (always reported)

    # Metrics (pre-ECC, hard decision).
    m = metrics_mod.compute_metrics(tx, rx, cfg.code.block_cross, cfg.code.block_down)
    sigma = channel_mod.snr_db_to_sigma(eff_snr, ch.c0)

    # Optional soft-decision / decoder stage (stage 2). Additive: it never
    # changes the pre-ECC BER above, only adds soft/post-decode diagnostics.
    soft: Dict = {}
    if cfg.detector.type == "soft_awgn":
        llr = detector.soft_llr(y, sigma=sigma, amplitude=ch.c0)
        soft["detector"] = cfg.detector.type
        soft["mean_abs_llr"] = float(np.mean(np.abs(llr)))
        decoder = make_decoder(cfg.decoder)
        if decoder is not None:
            dec_bits = decoder.decode(llr)
            md = metrics_mod.compute_metrics(tx, dec_bits, cfg.code.block_cross, cfg.code.block_down)
            soft["decoder"] = decoder.type
            soft["post_decode_BER"] = md["BER"]
            soft["post_decode_block_error_rate"] = md["block_error_rate"]

    elapsed = time.perf_counter() - t0

    row: Dict = {
        "name": cfg.experiment.name,
        "family": cfg.experiment.family,
        "rate": float(rate),
        "snr_db": eff_snr,
        "iti_coeff": iti_recorded,
        "num_bits": m["num_bits"],
        "bit_errors": m["bit_errors"],
        "BER": m["BER"],
        "block_errors": m["block_errors"],
        "block_error_rate": m["block_error_rate"],
        "seed": cfg.experiment.seed,
        "runtime_sec": round(elapsed, 6),
        # --- diagnostics (extra columns) ---
        "num_blocks": m["num_blocks"],
        "num_tracks": cfg.experiment.num_tracks,
        "bits_per_track": cfg.experiment.bits_per_track,
        "boundary": ch.boundary,
        "detector": cfg.detector.type,
        "threshold": cfg.detector.threshold,
        "sigma": sigma,
        "target_ber": cfg.metrics.target_ber,
        "hit_target": bool(m["BER"] <= cfg.metrics.target_ber),
        "codebook": code_meta.get("code"),
    }
    row.update(soft)
    if logger is not None:
        logger.info(
            "run family=%s rate=%.4f snr_db=%s iti=%.3f BER=%.3e bit_err=%d/%d t=%.3fs",
            row["family"], row["rate"], str(eff_snr), iti_recorded, row["BER"],
            row["bit_errors"], row["num_bits"], elapsed,
        )
    return row


# --------------------------------------------------------------------------- #
# sweep                                                                        #
# --------------------------------------------------------------------------- #
def run_sweep(cfg: Config, *, cache_dir: str = "data/codebooks", logger=None) -> List[Dict]:
    """Run the ``snr_db x iti_coeffs`` grid with independent reproducible RNGs."""
    if cfg.sweep is None:
        raise ValueError("sweep config requires a 'sweep' section")
    snrs = list(cfg.sweep.snr_db) or [cfg.channel.snr_db]
    itis = list(cfg.sweep.iti_coeffs) or [cfg.channel.c_cross_up]

    points = [(s, i) for s in snrs for i in itis]
    children = np.random.SeedSequence(cfg.experiment.seed).spawn(len(points))
    rows: List[Dict] = []
    for (snr, iti), ss in zip(points, children):
        rng = np.random.default_rng(ss)
        rows.append(run_single(cfg, snr_db=snr, iti_coeff=iti, rng=rng,
                                cache_dir=cache_dir, logger=logger))
    return rows


# --------------------------------------------------------------------------- #
# multi-family comparison                                                      #
# --------------------------------------------------------------------------- #
def run_compare(raw: Dict, *, cache_dir: str = "data/codebooks",
                logger=None) -> Tuple[List[Dict], Dict]:
    """Run several families over one shared SNR x ITI grid for direct comparison.

    ``raw`` is the parsed compare-config dict: a normal single-run config (with a
    ``sweep`` grid) plus a ``compare`` section::

        compare:
          families:
            - {family: uncoded,   code: {type: uncoded}}
            - {family: mtr1d,     code: {type: mtr1d, K1: 7}}
            - {family: mtr2d_8x2, code: {type: mtr2d_8x2, K: 14}}
          slice_snr_db: 14     # fixed SNR for the BER-vs-ITI figure
          slice_iti: 0.10      # fixed ITI for the BER-vs-SNR figure

    Every family is evaluated on the *same* channel grid and seed. Returns
    ``(rows, meta)`` where ``meta`` carries the (possibly defaulted) slice values.
    """
    compare = raw.get("compare") or {}
    families = compare.get("families")
    if not families:
        raise ValueError("compare config requires compare.families")
    if "sweep" not in raw:
        raise ValueError("compare config requires a 'sweep' grid")

    base = {k: copy.deepcopy(v) for k, v in raw.items()
            if k not in {"compare", "ldpc", "boundary_scan", "iti_calibration",
                         "rate_plan", "channel_metrics"}}
    base_name = base.get("experiment", {}).get("name", "compare")

    all_rows: List[Dict] = []
    for entry in families:
        fam = entry["family"]
        cfg_dict = copy.deepcopy(base)
        exp = dict(cfg_dict.get("experiment", {}))
        exp["family"] = fam
        exp["name"] = f"{base_name}_{fam}"
        cfg_dict["experiment"] = exp
        code = dict(cfg_dict.get("code", {}))
        code.update(entry.get("code", {}))
        code["type"] = fam
        code.setdefault("block_shape", [8, 2])
        cfg_dict["code"] = code

        cfg = Config.from_dict(cfg_dict)
        rows = run_sweep(cfg, cache_dir=cache_dir, logger=logger)
        all_rows.extend(rows)
        if logger is not None:
            logger.info("compare: family=%s rate=%.4f points=%d", fam, rows[0]["rate"], len(rows))

    # Default the figure slices to the middle of the grid if unspecified.
    snrs = [s for s in raw["sweep"].get("snr_db", []) if s is not None]
    itis = list(raw["sweep"].get("iti_coeffs", []))
    slice_snr = compare.get("slice_snr_db")
    slice_iti = compare.get("slice_iti")
    if slice_snr is None and snrs:
        slice_snr = sorted(snrs)[len(snrs) // 2]
    if slice_iti is None and itis:
        slice_iti = sorted(itis)[len(itis) // 2]
    meta = {"slice_snr_db": slice_snr, "slice_iti": slice_iti,
            "families": [e["family"] for e in families]}
    return all_rows, meta


# --------------------------------------------------------------------------- #
# LDPC evaluation track (post-ECC BER / FER)                                   #
# --------------------------------------------------------------------------- #
def run_ldpc(cfg: Config, ldpc_params: Dict, *, cache_dir: str = "data/codebooks",
             logger=None) -> Tuple[List[Dict], Dict]:
    """Evaluate an LDPC code over the 2D readback channel (post-ECC BER + FER).

    Geometry: one codeword per track. ``num_frames`` length-``n`` codewords form a
    ``(num_frames, n)`` grid, so inter-track interference couples neighbouring
    codewords and down-track ISI acts within a codeword. The soft (``soft_awgn``)
    detector supplies per-bit LLRs to the BP decoder.

    Returns ``(rows, meta)``. Each row uses the standard schema with
    ``BER`` = post-ECC information BER and ``block_error_rate`` = FER, plus a
    ``pre_ecc_ber`` reference column.
    """
    code = LDPCCode.gallager(
        n=int(ldpc_params.get("n", 300)),
        dv=int(ldpc_params.get("dv", 3)),
        dc=int(ldpc_params.get("dc", 6)),
        seed=int(ldpc_params.get("construction_seed", 1)),
    )
    max_iters = int(ldpc_params.get("max_iters", 30))
    method = str(ldpc_params.get("method", "minsum"))
    scale = float(ldpc_params.get("scale", 0.75))
    num_frames = _resolve_num_frames(code, ldpc_params, 200)
    sector_bits = int(ldpc_params.get("sector_bits", 0) or 0)

    if cfg.detector.type != "soft_awgn":
        raise ValueError("LDPC evaluation requires detector.type 'soft_awgn' (LLR input)")

    snrs = list(cfg.sweep.snr_db) if cfg.sweep else [cfg.channel.snr_db]
    itis = list(cfg.sweep.iti_coeffs) if cfg.sweep else [cfg.channel.c_cross_up]
    points = [(s, i) for s in snrs for i in itis]
    children = np.random.SeedSequence(cfg.experiment.seed).spawn(len(points))

    rows: List[Dict] = []
    first_llr_meta: Dict = {}
    for (snr, iti), ss in zip(points, children):
        t0 = time.perf_counter()
        rng = np.random.default_rng(ss)
        ch = replace(cfg.channel, c_cross_up=float(iti), c_cross_down=float(iti))
        taps = channel_mod.ChannelTaps.from_config(ch)
        sigma = channel_mod.snr_db_to_sigma(snr, ch.c0)

        U = rng.integers(0, 2, size=(num_frames, code.k), dtype=np.uint8)
        C = code.encode(U)                                  # (F, n) codeword bits
        symbols = channel_mod.map_bits_to_symbols(C)
        y = channel_mod.readback(symbols, taps, snr, rng=rng, boundary=ch.boundary)

        llr, llr_meta = _channel_llr_grid(y, cfg, taps, sigma, ldpc_params)
        if not first_llr_meta:
            first_llr_meta = dict(llr_meta)
        raw = (llr >= 0.0).astype(np.uint8)                 # channel detector hard decision
        pre_ber = float((raw != C).mean())
        dec = code.decode_llr(llr, max_iters=max_iters, method=method, scale=scale)
        info_hat = code.info_bits(dec)

        info_err = int((info_hat != U).sum())
        frame_err = int(np.any(info_hat != U, axis=1).sum())
        num_info = int(U.size)
        post_ber = info_err / num_info
        fer = frame_err / num_frames
        elapsed = time.perf_counter() - t0

        row = {
            "name": cfg.experiment.name,
            "family": "ldpc",
            "rate": float(code.rate),
            "snr_db": snr,
            "iti_coeff": float(iti),
            "num_bits": num_info,
            "bit_errors": info_err,
            "BER": post_ber,                  # post-ECC information BER
            "block_errors": frame_err,
            "block_error_rate": fer,          # frame (message) error rate
            "seed": cfg.experiment.seed,
            "runtime_sec": round(elapsed, 6),
            # --- diagnostics ---
            "pre_ecc_ber": pre_ber,
            "n": code.n, "k": code.k, "ldpc_rate": float(code.rate),
            "num_frames": num_frames, "max_iters": max_iters,
            "method": method, "scale": scale, "decoder": "ldpc",
            "sigma": sigma, "boundary": ch.boundary,
            "target_ber": cfg.metrics.target_ber,
            "hit_target": bool(post_ber <= cfg.metrics.target_ber),
        }
        row.update(llr_meta)
        row.update(_sector_error_stats(U, info_hat, sector_bits))
        rows.append(row)
        if logger is not None:
            logger.info("ldpc snr=%s iti=%.3f pre_BER=%.3e post_BER=%.3e FER=%.3e t=%.2fs",
                        str(snr), iti, pre_ber, post_ber, fer, elapsed)

    meta = {"n": code.n, "k": code.k, "rate": code.rate, "num_frames": num_frames,
            "max_iters": max_iters, "method": method, "scale": scale}
    if sector_bits > 0:
        meta.update({
            "sector_bits": sector_bits,
            "sector_count_target": int(ldpc_params.get("sector_count_target", 0) or 0),
            "sector_count_observed": int((num_frames * code.k) // sector_bits),
        })
    meta.update(first_llr_meta)
    return rows, meta


# --------------------------------------------------------------------------- #
# Concatenated LDPC + constrained modulation evaluation                        #
# --------------------------------------------------------------------------- #
def run_concatenated(cfg: Config, ldpc_params: Dict, *, cache_dir: str = "data/codebooks",
                     logger=None) -> Tuple[List[Dict], Dict]:
    """Evaluate outer LDPC with the configured inner modulation/constrained code.

    The inner layer can be ``uncoded``, ``mtr1d``, or ``mtr2d_8x2``. Constrained
    inner layers are decoded by exact codebook soft demapping before the LDPC BP
    decoder. If ``ldpc.turbo_iterations`` is positive, LDPC extrinsic LLRs are
    fed back as a-priori information to the inner demapper for additional SISO
    iterations. The reported ``pre_ecc_ber`` remains the initial LDPC-input BER
    before feedback; ``final_ldpc_input_ber`` records the final demapper output.
    """
    code = LDPCCode.gallager(
        n=int(ldpc_params.get("n", 280)),
        dv=int(ldpc_params.get("dv", 3)),
        dc=int(ldpc_params.get("dc", 6)),
        seed=int(ldpc_params.get("construction_seed", 1)),
    )
    max_iters = int(ldpc_params.get("max_iters", 30))
    method = str(ldpc_params.get("method", "minsum"))
    scale = float(ldpc_params.get("scale", 0.75))
    num_frames = _resolve_num_frames(
        code, ldpc_params, cfg.experiment.num_tracks,
        frame_multiple=_frame_multiple_for_inner(cfg),
    )
    sector_bits = int(ldpc_params.get("sector_bits", 0) or 0)
    turbo_iterations = int(ldpc_params.get("turbo_iterations", 0))
    turbo_llr_clip = float(ldpc_params.get("turbo_llr_clip", getattr(cfg.detector, "llr_clip", 20.0)))
    turbo_damping = float(ldpc_params.get("turbo_damping", 0.0))
    demapper_chunk = int(ldpc_params.get("demapper_chunk", 64))
    inner_demapper = str(ldpc_params.get("inner_demapper", "exact_codebook"))
    if turbo_iterations < 0:
        raise ValueError("ldpc.turbo_iterations must be >= 0")
    if not (0.0 <= turbo_damping < 1.0):
        raise ValueError("ldpc.turbo_damping must be in [0, 1)")
    if inner_demapper not in {"exact_codebook", "hard_codebook", "trellis_pruned", "stateful_trellis"}:
        raise ValueError(
            "ldpc.inner_demapper must be 'exact_codebook', 'hard_codebook', "
            "'trellis_pruned', or 'stateful_trellis'"
        )

    if cfg.detector.type != "soft_awgn":
        raise ValueError("concatenated evaluation requires detector.type 'soft_awgn' (LLR input)")

    snrs = list(cfg.sweep.snr_db) if cfg.sweep else [cfg.channel.snr_db]
    itis = list(cfg.sweep.iti_coeffs) if cfg.sweep else [cfg.channel.c_cross_up]
    points = [(s, i) for s in snrs for i in itis]
    children = np.random.SeedSequence(cfg.experiment.seed).spawn(len(points))

    rows: List[Dict] = []
    first_inner_meta: Dict = {}
    for (snr, iti), ss in zip(points, children):
        t0 = time.perf_counter()
        rng = np.random.default_rng(ss)
        ch = replace(cfg.channel, c_cross_up=float(iti), c_cross_down=float(iti))
        taps = channel_mod.ChannelTaps.from_config(ch)
        sigma = channel_mod.snr_db_to_sigma(snr, ch.c0)

        U = rng.integers(0, 2, size=(num_frames, code.k), dtype=np.uint8)
        C = code.encode(U)
        tx_grid, inner_meta = concat_mod.encode_inner_grid(C, cfg, cache_dir=cache_dir)
        transition_meta = concat_mod.inner_transition_stats(tx_grid, cfg, cache_dir=cache_dir)
        if not first_inner_meta:
            first_inner_meta = dict(inner_meta)

        symbols = channel_mod.map_bits_to_symbols(tx_grid)
        y = channel_mod.readback(symbols, taps, snr, rng=rng, boundary=ch.boundary)

        llr_grid, llr_meta = _channel_llr_grid(y, cfg, taps, sigma, ldpc_params)
        raw_grid = (llr_grid >= 0.0).astype(np.uint8)
        inner_channel_ber = float((raw_grid != tx_grid).mean())
        demap_meta: Dict = {}
        ldpc_feedback = np.zeros((num_frames, code.n), dtype=np.float64)
        dec = np.zeros_like(C)
        pre_outer_ber = 0.0
        final_ldpc_input_ber = 0.0
        rounds = turbo_iterations + 1
        for round_idx in range(rounds):
            ldpc_llr, demap_meta = concat_mod.decode_inner_llr(
                llr_grid, cfg, outer_shape=(num_frames, code.n), cache_dir=cache_dir,
                apriori_llr=ldpc_feedback, extrinsic=True, llr_clip=turbo_llr_clip,
                chunk=demapper_chunk, inner_demapper=inner_demapper,
            )
            hard_in = (ldpc_llr >= 0.0).astype(np.uint8)
            round_input_ber = float((hard_in != C).mean())
            if round_idx == 0:
                pre_outer_ber = round_input_ber
            final_ldpc_input_ber = round_input_ber

            dec, posterior_llr = code.decode_llr_with_posterior(
                ldpc_llr, max_iters=max_iters, method=method, scale=scale
            )
            if round_idx < rounds - 1:
                new_feedback = posterior_llr - ldpc_llr
                if turbo_llr_clip > 0:
                    new_feedback = np.clip(new_feedback, -turbo_llr_clip, turbo_llr_clip)
                if turbo_damping:
                    ldpc_feedback = turbo_damping * ldpc_feedback + (1.0 - turbo_damping) * new_feedback
                else:
                    ldpc_feedback = new_feedback
        info_hat = code.info_bits(dec)

        info_err = int((info_hat != U).sum())
        frame_err = int(np.any(info_hat != U, axis=1).sum())
        num_info = int(U.size)
        post_ber = info_err / num_info
        fer = frame_err / num_frames
        elapsed = time.perf_counter() - t0

        row = {
            "name": cfg.experiment.name,
            "family": f"ldpc+{cfg.experiment.family}",
            "rate": float(code.rate * inner_meta["inner_rate"]),
            "snr_db": snr,
            "iti_coeff": float(iti),
            "num_bits": num_info,
            "bit_errors": info_err,
            "BER": post_ber,
            "block_errors": frame_err,
            "block_error_rate": fer,
            "seed": cfg.experiment.seed,
            "runtime_sec": round(elapsed, 6),
            # --- diagnostics ---
            "pre_ecc_ber": pre_outer_ber,
            "final_ldpc_input_ber": final_ldpc_input_ber,
            "inner_channel_ber": inner_channel_ber,
            "outer_ldpc_rate": float(code.rate),
            "inner_rate": float(inner_meta["inner_rate"]),
            "inner_code": inner_meta["inner_code"],
            "inner_demapper": demap_meta["inner_demapper"],
            "configured_inner_demapper": inner_demapper,
            "channel_detector": llr_meta["channel_detector"],
            "equalizer_iterations": llr_meta["equalizer_iterations"],
            "channel_llr_clip": llr_meta["channel_llr_clip"],
            "turbo_iterations": turbo_iterations,
            "turbo_rounds": rounds,
            "turbo_llr_clip": turbo_llr_clip,
            "turbo_damping": turbo_damping,
            "demapper_chunk": demapper_chunk,
            "n": code.n, "k": code.k, "num_frames": num_frames,
            "max_iters": max_iters, "method": method, "scale": scale,
            "decoder": "ldpc",
            "sigma": sigma, "boundary": ch.boundary,
            "target_ber": cfg.metrics.target_ber,
            "hit_target": bool(post_ber <= cfg.metrics.target_ber),
        }
        row.update(_sector_error_stats(U, info_hat, sector_bits))
        row.update({k: v for k, v in inner_meta.items() if k not in row})
        row.update({k: v for k, v in transition_meta.items() if k not in row})
        row.update({k: v for k, v in demap_meta.items() if k not in row})
        rows.append(row)
        if logger is not None:
            logger.info(
                "concat inner=%s turbo=%d snr=%s iti=%.3f inner_BER=%.3e "
                "pre_LDPC_BER=%.3e final_LDPC_in_BER=%.3e post_BER=%.3e FER=%.3e t=%.2fs",
                cfg.experiment.family, turbo_iterations, str(snr), iti, inner_channel_ber,
                pre_outer_ber, final_ldpc_input_ber, post_ber, fer, elapsed,
            )

    meta = {
        "n": code.n, "k": code.k, "outer_ldpc_rate": code.rate,
        "inner_family": cfg.experiment.family,
        "num_frames": num_frames, "max_iters": max_iters,
        "method": method, "scale": scale,
        "turbo_iterations": turbo_iterations,
        "turbo_rounds": turbo_iterations + 1,
        "turbo_llr_clip": turbo_llr_clip,
        "turbo_damping": turbo_damping,
        "demapper_chunk": demapper_chunk,
        "inner_demapper": inner_demapper,
    }
    if sector_bits > 0:
        meta.update({
            "sector_bits": sector_bits,
            "sector_count_target": int(ldpc_params.get("sector_count_target", 0) or 0),
            "sector_count_observed": int((num_frames * code.k) // sector_bits),
        })
    meta.update(first_inner_meta)
    if rows:
        meta.update({
            "channel_detector": rows[0].get("channel_detector"),
            "equalizer_iterations": rows[0].get("equalizer_iterations"),
            "channel_llr_clip": rows[0].get("channel_llr_clip"),
        })
    return rows, meta


def run_boundary_scan(cfg: Config, ldpc_params: Dict,
                      boundary_down_values: List[int],
                      boundary_checker_values: List[int],
                      *,
                      num_trials: int = 4,
                      cache_dir: str = "data/codebooks",
                      logger=None) -> Tuple[List[Dict], Dict]:
    """Scan trellis-boundary pruning settings before expensive BER/FSR runs.

    The scan reports two separate things:

    * structural ``trellis_state_transition_density`` for the codebook boundary
      rule; values below 1.0 mean the demapper can prune some block transitions;
    * sampled ``tx_trellis_transition_violation_rate`` under the current
      blockwise transmitter. Zero sampled violations are useful but not a formal
      guarantee unless the transmitter is made stateful or the density is 1.0.
    """
    if cfg.experiment.family not in {"mtr1d", "mtr2d_8x2"}:
        raise ValueError("boundary scan requires a constrained family: mtr1d or mtr2d_8x2")
    if not boundary_down_values:
        raise ValueError("boundary_down_values must not be empty")
    if not boundary_checker_values:
        raise ValueError("boundary_checker_values must not be empty")
    trials = max(1, int(num_trials))

    code = LDPCCode.gallager(
        n=int(ldpc_params.get("n", 280)),
        dv=int(ldpc_params.get("dv", 3)),
        dc=int(ldpc_params.get("dc", 6)),
        seed=int(ldpc_params.get("construction_seed", 1)),
    )
    num_frames = int(ldpc_params.get("num_frames", cfg.experiment.num_tracks))
    rng = np.random.default_rng(cfg.experiment.seed)
    cb = (
        concat_mod._stateful_candidate_codebook(cfg, cache_dir)
        if cfg.code.inner_encoder == "stateful_trellis"
        else concat_mod._inner_codebook(cfg, cache_dir)
    )

    tx_grids = []
    for _ in range(trials):
        U = rng.integers(0, 2, size=(num_frames, code.k), dtype=np.uint8)
        C = code.encode(U)
        tx_grid, inner_meta = concat_mod.encode_inner_grid(C, cfg, cache_dir=cache_dir)
        tx_grids.append(tx_grid)

    rows: List[Dict] = []
    for bdm in boundary_down_values:
        for bcm in boundary_checker_values:
            scan_cfg = replace(
                cfg,
                code=replace(
                    cfg.code,
                    boundary_down_mtr=int(bdm),
                    boundary_max_checker_run=int(bcm),
                ),
            )
            violations = 0
            transitions = 0
            for tx_grid in tx_grids:
                stats = concat_mod.inner_transition_stats(tx_grid, scan_cfg, cache_dir=cache_dir)
                violations += int(stats["tx_trellis_transition_violations"])
                transitions += int(stats["tx_trellis_transition_count"])
            tmeta = concat_mod.trellis_transition_meta(
                cb, down_mtr=int(bdm), max_checker_run_allowed=int(bcm),
            )
            violation_rate = float(violations / transitions) if transitions else 0.0
            density = float(tmeta["trellis_state_transition_density"])
            row = {
                "family": cfg.experiment.family,
                "inner_code": cb.name,
                "inner_rate": cb.rate,
                "outer_ldpc_n": code.n,
                "outer_ldpc_k": code.k,
                "outer_ldpc_rate": code.rate,
                "num_frames": num_frames,
                "num_trials": trials,
                "boundary_down_mtr": int(bdm),
                "boundary_max_checker_run": int(bcm),
                "tx_trellis_transition_violations": int(violations),
                "tx_trellis_transition_count": int(transitions),
                "tx_trellis_transition_violation_rate": violation_rate,
                "candidate_zero_sample_violation": bool(violations == 0 and density < 1.0),
                "structurally_no_pruning": bool(density >= 1.0),
            }
            row.update(tmeta)
            rows.append(row)
            if logger is not None:
                logger.info(
                    "boundary scan down=%d checker=%d tx_violation=%.3e density=%.3f candidate=%s",
                    int(bdm), int(bcm), violation_rate, density,
                    row["candidate_zero_sample_violation"],
                )

    meta = {
        "family": cfg.experiment.family,
        "inner_code": cb.name,
        "inner_rate": cb.rate,
        "outer_ldpc_n": code.n,
        "outer_ldpc_k": code.k,
        "outer_ldpc_rate": code.rate,
        "num_frames": num_frames,
        "num_trials": trials,
        "boundary_down_values": [int(v) for v in boundary_down_values],
        "boundary_checker_values": [int(v) for v in boundary_checker_values],
    }
    return rows, meta
