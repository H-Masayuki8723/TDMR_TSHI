"""Decoder interfaces for the soft-decision / LDPC stage (stage 2).

This module defines the seam where a real LDPC (or other ECC) decoder will plug
in. It consumes per-bit **LLRs** produced by
:class:`tdmr2d.detector.SoftAWGNDetector` and returns decoded bits.

Provided now:

* :class:`HardDecisionDecoder` (type ``"hard"``) -- a no-op that hard-decides the
  LLR sign. It lets the soft pipeline run end-to-end through the decoder slot
  with it, the post-decode BER equals the pre-ECC BER (a useful sanity check, not
  an ECC gain).
* :class:`LDPCDecoder` (type ``"ldpc"``) -- wraps a real
  :class:`tdmr2d.ldpc.LDPCCode` when one is supplied. Without a code, ``decode``
  raises with a redirect to the dedicated ``tdmr2d ldpc`` / ``tdmr2d concat``
  encoded-frame commands.

Keeping this separate from the pre-ECC BER path makes the boundary between
"channel-level 2D coding study" and "ECC evaluation" unambiguous.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Decoder:
    """Decoder interface: map a grid of per-bit LLRs to decoded bits."""

    type = "decoder"

    def decode(self, llr: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def rate(self) -> float:
        return 1.0


class HardDecisionDecoder(Decoder):
    """No-op placeholder: hard-decide the LLR sign (no error correction)."""

    type = "hard"

    def decode(self, llr: np.ndarray) -> np.ndarray:
        return (np.asarray(llr) >= 0.0).astype(np.uint8)


class LDPCDecoder(Decoder):
    """LDPC decoder wrapping a :class:`tdmr2d.ldpc.LDPCCode`.

    Construct with ``LDPCDecoder(code=...)`` to decode per-bit LLR frames via
    BP/min-sum. Without a code it raises -- the dedicated ``tdmr2d ldpc`` track
    builds the code, encodes frames and runs the decoder for you. Use
    ``tdmr2d concat`` when those LDPC frames should also pass through an inner
    constrained-code codebook before the channel.
    """

    type = "ldpc"

    def __init__(self, code=None, max_iters: int = 30, method: str = "minsum",
                 scale: float = 0.75, **params) -> None:
        self.code = code
        self.max_iters = int(max_iters)
        self.method = method
        self.scale = float(scale)
        self.params = dict(params)

    def decode(self, llr: np.ndarray) -> np.ndarray:
        if self.code is None:
            raise NotImplementedError(
                "LDPCDecoder needs an LDPCCode. Use the `tdmr2d ldpc` command "
                "for uncoded modulation or `tdmr2d concat` for LDPC plus an inner "
                "MTR/2D-MTR codebook."
            )
        return self.code.decode_llr(np.asarray(llr), max_iters=self.max_iters,
                                    method=self.method, scale=self.scale)


def make_decoder(cfg) -> Optional[Decoder]:
    """Build a decoder from an optional ``DecoderConfig`` (``None`` => no decode)."""
    if cfg is None:
        return None
    t = getattr(cfg, "type", "none")
    if t in (None, "none"):
        return None
    if t == "hard":
        return HardDecisionDecoder()
    if t == "ldpc":
        return LDPCDecoder(**(getattr(cfg, "params", {}) or {}))
    raise ValueError(f"unknown decoder type: {t!r}")
