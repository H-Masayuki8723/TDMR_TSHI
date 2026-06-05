"""Configuration models and YAML loading/validation.

Validation uses stdlib ``dataclasses`` (the spec allows pydantic *or* dataclass).
Keeping it dependency-light makes the whole evaluation reproducible from a clean
Python install without compiled/3rd-party validation libraries.

Block geometry convention
-------------------------
The recording grid is indexed ``x[i, j]`` with ``i`` = track index and ``j`` =
down-track bit index, so a grid array has shape ``(num_tracks, bits_per_track)``.

``code.block_shape`` is written ``[down_track, cross_track]`` (e.g. ``[8, 2]`` for
an 8x2 TDMR block = 8 down-track bits across 2 tracks). The helper properties
``block_down`` and ``block_cross`` expose the two dimensions unambiguously.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

FAMILIES = {"uncoded", "mtr1d", "mtr2d_8x2"}
CODE_TYPES = {"uncoded", "mtr1d", "mtr2d_8x2"}
CHANNEL_MODELS = {"linear_2d_awgn"}
DETECTOR_TYPES = {"hard_threshold", "soft_awgn"}
DECODER_TYPES = {"none", "hard", "ldpc"}
BOUNDARIES = {"zero", "periodic", "edge"}
INNER_ENCODERS = {"blockwise", "stateful_trellis"}


class ConfigError(ValueError):
    """Raised when a configuration is structurally invalid."""


def _check_keys(section: str, given: Dict[str, Any], allowed: set) -> None:
    unknown = set(given) - allowed
    if unknown:
        raise ConfigError(
            f"[{section}] unknown key(s): {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    family: str = "uncoded"
    seed: int = 0
    num_tracks: int = 64
    bits_per_track: int = 4096

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d or {})
        _check_keys("experiment", d, {f.name for f in fields(cls)})
        return cls(**d)


@dataclass
class CodeConfig:
    type: str = "uncoded"
    block_shape: Tuple[int, int] = (8, 2)  # [down_track, cross_track]
    # Pattern C (mtr2d_8x2): info bits per 16-bit 8x2 block -> rate = K/16.
    K: Optional[int] = None
    # Pattern B (mtr1d): info bits per length-(block_down) down-track block -> rate = K1/block_down.
    K1: Optional[int] = None
    # Constraint parameters (also used to name/cache the codebook).
    down_mtr: int = 3            # max down-track transition run allowed
    max_checker_run: int = 0     # max 2x2 checkerboard run allowed (0 => forbid any)
    # Optional trellis-boundary pruning parameters. If unset, the decoder uses
    # the block-internal values above for backwards compatibility.
    boundary_down_mtr: Optional[int] = None
    boundary_max_checker_run: Optional[int] = None
    # Inner encoder mode. ``stateful_trellis`` uses K as the payload bits per
    # block, but may draw channel words from a larger candidate codebook.
    inner_encoder: str = "blockwise"
    trellis_candidate_K: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CodeConfig":
        d = dict(d or {})
        _check_keys("code", d, {f.name for f in fields(cls)})
        if "block_shape" in d:
            bs = d["block_shape"]
            if not (isinstance(bs, (list, tuple)) and len(bs) == 2):
                raise ConfigError("[code] block_shape must be a 2-element list [down, cross]")
            d["block_shape"] = (int(bs[0]), int(bs[1]))
        return cls(**d)

    @property
    def block_down(self) -> int:
        return int(self.block_shape[0])

    @property
    def block_cross(self) -> int:
        return int(self.block_shape[1])

    @property
    def trellis_boundary_down_mtr(self) -> int:
        return int(self.boundary_down_mtr if self.boundary_down_mtr is not None else self.down_mtr)

    @property
    def trellis_boundary_max_checker_run(self) -> int:
        return int(
            self.boundary_max_checker_run
            if self.boundary_max_checker_run is not None
            else self.max_checker_run
        )


@dataclass
class ChannelConfig:
    model: str = "linear_2d_awgn"
    c0: float = 1.0
    c_down_prev: float = 0.15
    c_down_next: float = 0.15
    c_cross_up: float = 0.10
    c_cross_down: float = 0.10
    snr_db: Optional[float] = 14.0   # None => noiseless
    boundary: str = "zero"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelConfig":
        d = dict(d or {})
        _check_keys("channel", d, {f.name for f in fields(cls)})
        return cls(**d)


@dataclass
class DetectorConfig:
    type: str = "hard_threshold"   # hard_threshold | soft_awgn
    threshold: float = 0.0
    llr_clip: float = 20.0         # used by soft_awgn LLR generation

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DetectorConfig":
        d = dict(d or {})
        _check_keys("detector", d, {f.name for f in fields(cls)})
        return cls(**d)


@dataclass
class MetricsConfig:
    target_ber: float = 1.0e-2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricsConfig":
        d = dict(d or {})
        _check_keys("metrics", d, {f.name for f in fields(cls)})
        return cls(**d)


@dataclass
class OutputConfig:
    dir: str = "outputs/runs"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        d = dict(d or {})
        _check_keys("output", d, {f.name for f in fields(cls)})
        return cls(**d)


@dataclass
class DecoderConfig:
    """Optional decoder stage (stage 2). Requires detector.type == 'soft_awgn'."""
    type: str = "none"                       # none | null | ldpc
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecoderConfig":
        d = dict(d or {})
        _check_keys("decoder", d, {"type", "params"})
        return cls(type=d.get("type", "none"), params=dict(d.get("params", {}) or {}))


@dataclass
class SweepConfig:
    snr_db: List[float] = field(default_factory=list)
    iti_coeffs: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepConfig":
        d = dict(d or {})
        _check_keys("sweep", d, {f.name for f in fields(cls)})
        return cls(
            snr_db=[None if v is None else float(v) for v in d.get("snr_db", [])],
            iti_coeffs=[float(v) for v in d.get("iti_coeffs", [])],
        )


@dataclass
class Config:
    experiment: ExperimentConfig
    code: CodeConfig
    channel: ChannelConfig
    detector: DetectorConfig
    metrics: MetricsConfig
    output: OutputConfig
    sweep: Optional[SweepConfig] = None
    decoder: Optional[DecoderConfig] = None

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = dict(d or {})
        allowed = {f.name for f in fields(cls)}
        _check_keys("<root>", d, allowed)
        cfg = cls(
            experiment=ExperimentConfig.from_dict(d.get("experiment", {})),
            code=CodeConfig.from_dict(d.get("code", {})),
            channel=ChannelConfig.from_dict(d.get("channel", {})),
            detector=DetectorConfig.from_dict(d.get("detector", {})),
            metrics=MetricsConfig.from_dict(d.get("metrics", {})),
            output=OutputConfig.from_dict(d.get("output", {})),
            sweep=SweepConfig.from_dict(d["sweep"]) if d.get("sweep") is not None else None,
            decoder=DecoderConfig.from_dict(d["decoder"]) if d.get("decoder") is not None else None,
        )
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    # ---- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "experiment": {
                "name": self.experiment.name,
                "family": self.experiment.family,
                "seed": self.experiment.seed,
                "num_tracks": self.experiment.num_tracks,
                "bits_per_track": self.experiment.bits_per_track,
            },
            "code": {
                "type": self.code.type,
                "block_shape": [self.code.block_down, self.code.block_cross],
                "K": self.code.K,
                "K1": self.code.K1,
                "down_mtr": self.code.down_mtr,
                "max_checker_run": self.code.max_checker_run,
                "boundary_down_mtr": self.code.boundary_down_mtr,
                "boundary_max_checker_run": self.code.boundary_max_checker_run,
                "inner_encoder": self.code.inner_encoder,
                "trellis_candidate_K": self.code.trellis_candidate_K,
            },
            "channel": {
                "model": self.channel.model,
                "c0": self.channel.c0,
                "c_down_prev": self.channel.c_down_prev,
                "c_down_next": self.channel.c_down_next,
                "c_cross_up": self.channel.c_cross_up,
                "c_cross_down": self.channel.c_cross_down,
                "snr_db": self.channel.snr_db,
                "boundary": self.channel.boundary,
            },
            "detector": {"type": self.detector.type, "threshold": self.detector.threshold,
                         "llr_clip": self.detector.llr_clip},
            "metrics": {"target_ber": self.metrics.target_ber},
            "output": {"dir": self.output.dir},
        }
        if self.sweep is not None:
            d["sweep"] = {"snr_db": list(self.sweep.snr_db), "iti_coeffs": list(self.sweep.iti_coeffs)}
        if self.decoder is not None:
            d["decoder"] = {"type": self.decoder.type, "params": dict(self.decoder.params)}
        return d

    def save_resolved(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    # ---- validation ------------------------------------------------------
    def validate(self) -> None:
        e, c, ch, det = self.experiment, self.code, self.channel, self.detector

        if e.family not in FAMILIES:
            raise ConfigError(f"experiment.family must be one of {sorted(FAMILIES)}, got {e.family!r}")
        if c.type not in CODE_TYPES:
            raise ConfigError(f"code.type must be one of {sorted(CODE_TYPES)}, got {c.type!r}")
        if c.inner_encoder not in INNER_ENCODERS:
            raise ConfigError(f"code.inner_encoder must be one of {sorted(INNER_ENCODERS)}")
        if e.family != c.type:
            raise ConfigError(
                f"experiment.family ({e.family!r}) and code.type ({c.type!r}) must match"
            )
        if ch.model not in CHANNEL_MODELS:
            raise ConfigError(f"channel.model must be one of {sorted(CHANNEL_MODELS)}")
        if det.type not in DETECTOR_TYPES:
            raise ConfigError(f"detector.type must be one of {sorted(DETECTOR_TYPES)}")
        if ch.boundary not in BOUNDARIES:
            raise ConfigError(f"channel.boundary must be one of {sorted(BOUNDARIES)}")
        if self.decoder is not None:
            if self.decoder.type not in DECODER_TYPES:
                raise ConfigError(f"decoder.type must be one of {sorted(DECODER_TYPES)}")
            if self.decoder.type != "none" and det.type != "soft_awgn":
                raise ConfigError("a decoder requires detector.type 'soft_awgn' (LLR input)")

        if e.num_tracks <= 0 or e.bits_per_track <= 0:
            raise ConfigError("num_tracks and bits_per_track must be positive")
        if e.seed < 0:
            raise ConfigError("seed must be >= 0")
        if not (ch.snr_db is None or isinstance(ch.snr_db, (int, float))):
            raise ConfigError("channel.snr_db must be a number or null (noiseless)")

        down, cross = c.block_down, c.block_cross
        if down <= 0 or cross <= 0:
            raise ConfigError("block_shape entries must be positive")
        # Block tiling must cover the grid exactly (used by metrics and 2D coding).
        if e.bits_per_track % down != 0:
            raise ConfigError(
                f"bits_per_track ({e.bits_per_track}) must be divisible by block down-track size ({down})"
            )
        if e.num_tracks % cross != 0:
            raise ConfigError(
                f"num_tracks ({e.num_tracks}) must be divisible by block cross-track size ({cross})"
            )

        if c.type == "mtr2d_8x2":
            if c.K is None:
                raise ConfigError("code.K is required for family 'mtr2d_8x2' (rate = K/16)")
            if not (1 <= c.K < down * cross):
                raise ConfigError(f"code.K must be in [1, {down * cross - 1}] for an 8x2 block")
            if c.inner_encoder == "stateful_trellis":
                cand = c.trellis_candidate_K if c.trellis_candidate_K is not None else c.K
                if not (c.K <= cand < down * cross):
                    raise ConfigError(
                        "code.trellis_candidate_K must satisfy code.K <= trellis_candidate_K "
                        f"< {down * cross} for stateful_trellis"
                    )
        if c.type == "mtr1d":
            if c.K1 is None:
                raise ConfigError("code.K1 is required for family 'mtr1d' (rate = K1/block_down)")
            if not (1 <= c.K1 < down):
                raise ConfigError(f"code.K1 must be in [1, {down - 1}]")
            if c.inner_encoder == "stateful_trellis":
                raise ConfigError("code.inner_encoder='stateful_trellis' currently supports mtr2d_8x2 only")
        if c.boundary_down_mtr is not None and c.boundary_down_mtr < 0:
            raise ConfigError("code.boundary_down_mtr must be >= 0 when set")
        if c.boundary_max_checker_run is not None and c.boundary_max_checker_run < 0:
            raise ConfigError("code.boundary_max_checker_run must be >= 0 when set")

    # ---- convenience -----------------------------------------------------
    @property
    def block_shape_cross_down(self) -> Tuple[int, int]:
        """Block geometry as ``(cross_tracks, down_bits)`` for grid operations."""
        return (self.code.block_cross, self.code.block_down)
