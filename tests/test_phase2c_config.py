from pathlib import Path

import yaml

from tdmr2d.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/"
    "concat_ldpc_2dmtr_highrate_k15_eqpilot_bcjr2_"
    "phase2c_trueiti035_lowtap_extension_seed602_100sector.yaml"
)
EXTRA_ROOT_SECTIONS = {
    "ldpc",
    "boundary_scan",
    "iti_calibration",
    "rate_plan",
    "channel_metrics",
}


def test_phase2c_lowtap_extension_geometry():
    raw = yaml.safe_load(CONFIG.read_text())
    cfg = Config.from_dict(
        {key: value for key, value in raw.items() if key not in EXTRA_ROOT_SECTIONS}
    )

    assert cfg.experiment.seed == 602
    assert cfg.channel.c_cross_up == 0.35
    assert cfg.channel.c_cross_down == 0.35
    assert raw["ldpc"]["detector_iti_coeffs"] == [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    assert raw["sweep"]["snr_db"] == [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    assert raw["ldpc"]["sector_count_target"] == 100
    assert raw["ldpc"]["equalizer_iterations"] == 2
