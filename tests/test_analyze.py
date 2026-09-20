"""The end-state classifier and the transition detectors."""
import numpy as np
import pytest

from soup import analyze


def metrics(H, HOE, steps, loop, n=10):
    return {"entropy_bits": np.full(n, H),
            "high_order_entropy": np.full(n, HOE),
            "mean_steps": np.full(n, steps),
            "frac_steps_in_loop": np.full(n, loop),
            "epoch": np.arange(n) * 100.0}


def test_random_soup_is_classified_random():
    k, _ = analyze.classify(metrics(7.99, 0.0, 1500, 0.8), 129, b"\x00" * 16)
    assert k == "random"


def test_one_byte_crystal_is_classified_crystal():
    k, f = analyze.classify(metrics(1.28, 0.03, 128.2, 0.01), 129, b"<" * 16)
    assert k == "crystal"
    assert f["top_has_bracket"] is False


def test_replicator_takeover_is_classified_program():
    k, f = analyze.classify(metrics(5.88, 5.55, 8072, 0.95), 128,
                            b"\x08[-}}}}}}}}]++++")
    assert k == "program"
    assert f["top_has_bracket"] is True


def test_high_entropy_but_looping_is_not_a_crystal():
    assert analyze.classify(metrics(7.5, 0.3, 700, 0.8), 128, b"x" * 16)[0] \
        == "random"


def test_a_bracketless_dominant_pattern_is_not_a_program():
    """High HOE and long runs are not enough: a program contains control flow."""
    k, _ = analyze.classify(metrics(2.0, 3.0, 5000, 0.9), 128, b"<" * 16)
    assert k != "program"
    assert k == "mixed"


def test_low_entropy_but_still_looping_is_not_a_crystal():
    k, _ = analyze.classify(metrics(2.0, 0.1, 5000, 0.9), 128, b"<" * 16)
    assert k == "mixed"


def test_missing_loop_column_does_not_break_the_verdict():
    m = metrics(1.28, 0.03, 128.2, 0.0)
    del m["frac_steps_in_loop"]
    assert analyze.classify(m, 129, b"<" * 16)[0] == "crystal"


def test_classifier_uses_the_tail_not_the_last_row():
    m = metrics(1.28, 0.03, 128.2, 0.01, n=10)
    m["entropy_bits"][-1] = 8.0          # one outlier snapshot
    assert analyze.classify(m, 129, b"<" * 16)[0] == "crystal"


def test_walk_length_scales_the_program_threshold():
    m = metrics(3.5, 2.0, 500, 0.9)
    assert analyze.classify(m, 129, b"[-]xxxx")[0] == "program"   # 500 > 3*129
    assert analyze.classify(m, 1000, b"[-]xxxx")[0] == "mixed"    # 500 < 3*1000


# --- detectors ---------------------------------------------------------------

def test_takeover_detector_needs_a_sustained_rise():
    m = {"epoch": np.arange(20) * 100.0,
         "high_order_entropy": np.zeros(20),
         "entropy_bits": np.full(20, 8.0),
         "A_t": np.arange(20.0), "mean_steps": np.full(20, 100.0),
         "frac_copy_runs": np.zeros(20), "zlib_bits": np.full(20, 8.0),
         "n_persistent": np.zeros(20)}
    m["high_order_entropy"][5] = 3.0            # a single spike
    assert analyze.detect(m)["takeover_epoch"] is None
    m["high_order_entropy"][10:] = 3.0
    assert analyze.detect(m)["takeover_epoch"] == 1000
