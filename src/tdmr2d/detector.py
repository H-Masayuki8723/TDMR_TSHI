"""Bit detectors.

Two detectors are provided:

* **HardThresholdDetector** (default) -- memoryless ``rx_bit = 1 if y >= threshold
  else 0``. With the ``0->-1, 1->+1`` mapping and ``threshold = 0`` this recovers
  the bit sign directly. This drives the hard-decision pre-ECC BER.

* **SoftAWGNDetector** (stage-2 foundation) -- same hard decision, plus a
  per-bit **LLR generator** for soft-decision / LDPC work. Under the main-tap
  AWGN model ``y ~ N(a*s, sigma^2)`` with ``s in {-1,+1}`` and amplitude ``a``::

      LLR(bit) = log P(bit=1 | y) / P(bit=0 | y) = 2 * a * y / sigma^2

  so the LLR sign equals the hard decision (consistency) and its magnitude grows
  with SNR. LLRs are clipped to ``+/- llr_clip`` for numerical stability and to
  keep noiseless (sigma=0) cases finite.

The LLR generator is the prerequisite for any LDPC evaluation; an equalizer can
later be inserted ahead of either detector behind the :class:`Detector` API.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Detector:
    """Detector interface. Subclasses implement :meth:`detect` (hard bits)."""

    type = "detector"

    def detect(self, y: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def soft_llr(self, y: np.ndarray, sigma: Optional[float] = None,
                 amplitude: float = 1.0) -> np.ndarray:
        """Per-bit LLR. Default detectors without a noise model don't provide it."""
        raise NotImplementedError(
            "This detector does not produce soft output. Use detector.type "
            "'soft_awgn' for LLR generation (required before LDPC evaluation)."
        )


class HardThresholdDetector(Detector):
    type = "hard_threshold"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y) >= self.threshold).astype(np.uint8)


class SoftAWGNDetector(Detector):
    """Hard decision + main-tap AWGN LLR generation."""

    type = "soft_awgn"

    def __init__(self, threshold: float = 0.0, llr_clip: float = 20.0) -> None:
        self.threshold = float(threshold)
        self.llr_clip = float(llr_clip)

    def detect(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y) >= self.threshold).astype(np.uint8)

    def soft_llr(self, y: np.ndarray, sigma: Optional[float] = None,
                 amplitude: float = 1.0) -> np.ndarray:
        """LLR = 2 * amplitude * y / sigma^2, clipped. Positive LLR favours bit 1."""
        y = np.asarray(y, dtype=np.float64)
        if sigma is None or sigma <= 0.0:
            # Noiseless: near-certain decisions; saturate at the clip value.
            return np.where(y >= self.threshold, self.llr_clip, -self.llr_clip)
        llr = 2.0 * float(amplitude) * y / (sigma * sigma)
        return np.clip(llr, -self.llr_clip, self.llr_clip)


def make_detector(cfg) -> Detector:
    """Build a detector from a ``DetectorConfig``."""
    if cfg.type == "hard_threshold":
        return HardThresholdDetector(threshold=cfg.threshold)
    if cfg.type == "soft_awgn":
        return SoftAWGNDetector(threshold=cfg.threshold, llr_clip=getattr(cfg, "llr_clip", 20.0))
    raise ValueError(f"unknown detector type: {cfg.type!r}")
