"""tdmr2d -- pre-ECC BER evaluation of 2D constrained codes.

This package evaluates the *pre-ECC* (uncoded-channel, hard-decision) bit error
rate of 2D / constrained codes transmitted over a simplified 2D TDMR readback
channel with inter-track interference (ITI) and AWGN.

It is deliberately separate from any error-correction (ECC) comparison: no error
correction or erasure handling is performed. BER is measured by comparing the
hard-decided channel bits against the transmitted channel bits.
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "patterns",
    "constraints",
    "codebook",
    "channel",
    "detector",
    "decoder",
    "ldpc",
    "metrics",
    "experiments",
    "reports",
    "io",
]
