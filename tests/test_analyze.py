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
    k, _ = analyze.classify(metrics(7.99, 0.0, 1500, 0.8), 129, [b"\x00" * 16])
    assert k == "random"


def test_one_byte_crystal_is_classified_crystal():
    k, f = analyze.classify(metrics(1.28, 0.03, 128.2, 0.01), 129, [b"<" * 16])
    assert k == "crystal"
    assert f["code_like_pattern"] is False


def test_replicator_takeover_is_classified_program():
    k, f = analyze.classify(metrics(5.88, 5.55, 8072, 0.95), 128,
                            [b"\xe5" * 16, b"\x08[-}}}}}}}}]++++"])
    assert k == "program"
    assert f["code_like_pattern"] is True


def test_high_entropy_but_looping_is_not_a_crystal():
    assert analyze.classify(metrics(7.5, 0.3, 700, 0.8), 128, [b"x" * 16])[0] \
        == "random"


def test_a_bracketless_dominant_pattern_is_not_a_program():
    """High HOE and long runs are not enough: a program contains control flow."""
    k, _ = analyze.classify(metrics(2.0, 3.0, 5000, 0.9), 128, [b"<" * 16])
    assert k != "program"
    assert k == "mixed"


def test_low_entropy_but_still_looping_is_not_a_crystal():
    k, _ = analyze.classify(metrics(2.0, 0.1, 5000, 0.9), 128, [b"<" * 16])
    assert k == "mixed"


def test_missing_loop_column_does_not_break_the_verdict():
    m = metrics(1.28, 0.03, 128.2, 0.0)
    del m["frac_steps_in_loop"]
    assert analyze.classify(m, 129, [b"<" * 16])[0] == "crystal"


def test_classifier_uses_the_tail_not_the_last_row():
    m = metrics(1.28, 0.03, 128.2, 0.01, n=10)
    m["entropy_bits"][-1] = 8.0          # one outlier snapshot
    assert analyze.classify(m, 129, [b"<" * 16])[0] == "crystal"


def test_walk_length_scales_the_program_threshold():
    m = metrics(3.5, 2.0, 500, 0.9)
    assert analyze.classify(m, 129, [b"[-]wxyz"])[0] == "program"   # 500 > 3*129
    assert analyze.classify(m, 1000, [b"[-]wxyz"])[0] == "mixed"    # 500 < 3*1000


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


def test_a_run_of_brackets_is_not_code():
    assert analyze.looks_like_code([b"[" * 16]) is False
    assert analyze.looks_like_code([b"[.>}]" + b"\x01\x02\x03"]) is True
    assert analyze.looks_like_code([b"abcdefgh"]) is False


# --- the transition-rate report ---------------------------------------------

def test_wilson_interval_brackets_the_estimate():
    p, lo, hi = analyze.wilson(3, 10)
    assert abs(p - 0.3) < 1e-12
    assert lo < 0.3 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_has_width_at_the_extremes():
    """The normal approximation gives a zero-width interval at 0/n and n/n."""
    _, lo, hi = analyze.wilson(0, 8)
    assert lo == 0.0 and 0.0 < hi < 0.5
    _, lo, hi = analyze.wilson(8, 8)
    assert hi == 1.0 and 0.5 < lo < 1.0


def test_wilson_narrows_with_more_trials():
    _, lo1, hi1 = analyze.wilson(5, 10)
    _, lo2, hi2 = analyze.wilson(50, 100)
    assert (hi2 - lo2) < (hi1 - lo1)


def test_rate_report_groups_by_configuration():
    rows = [{"mode": "bff", "compat": "cubff", "N_or_M": 32768,
             "klass": "program", "takeover_epoch": 2000},
            {"mode": "bff", "compat": "cubff", "N_or_M": 32768,
             "klass": "program", "takeover_epoch": 2750},
            {"mode": "bff", "compat": "none", "N_or_M": 32768,
             "klass": "random", "takeover_epoch": None}]
    out = analyze.rate_report(rows)
    assert len(out) == 2
    heads = [g for g in out if g["compat"] == "cubff"][0]
    assert heads["n"] == 2 and heads["programs"] == 2 and heads["rate"] == 1.0
    assert heads["takeover_epochs"] == [2000, 2750]
    plain = [g for g in out if g["compat"] == "none"][0]
    assert plain["programs"] == 0 and plain["ci_high"] < 1.0
