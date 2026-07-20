"""The bunching estimator must (a) report ~zero excess mass on a smooth distribution and
(b) recover clear positive excess mass when we inject known bunching just below a
threshold. If these fail, every Vein 1 number is untrustworthy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from vein1_bunching import bunching_estimate  # noqa: E402

RNG = np.random.default_rng(0)
# fit window / excluded window matching the E/F setup at threshold 125
ARGS = dict(threshold=125, fit_lo=101, fit_hi=150, excl_lo=121, excl_hi=130, n_boot=100)


def _smooth_ratings(n=60000):
    # a smooth unimodal spread over 101-150, no threshold structure
    return pd.Series(np.clip(RNG.normal(128, 9, n).round(), 101, 150).astype(int))


def test_smooth_distribution_has_near_zero_excess_mass():
    b = bunching_estimate(_smooth_ratings(), **ARGS)["b_norm"]
    assert abs(b) < 1.0, f"smooth data should not show strong bunching, got b={b:.2f}"


def test_injected_bunching_is_detected():
    base = _smooth_ratings()
    # move a chunk of mass from just-above (126-130) to just-below (123-125): relabeling
    r = base.copy()
    above = r[(r >= 126) & (r <= 130)].index
    move = RNG.choice(above, size=int(len(above) * 0.6), replace=False)
    r.loc[move] = RNG.choice([123, 124, 125], size=len(move))
    b = bunching_estimate(r, **ARGS)["b_norm"]
    assert b > 1.0, f"injected relabeling should yield clear positive b, got {b:.2f}"


def test_missing_mass_positive_when_relabeled_from_above():
    base = _smooth_ratings()
    r = base.copy()
    above = r[(r >= 126) & (r <= 130)].index
    move = RNG.choice(above, size=int(len(above) * 0.6), replace=False)
    r.loc[move] = 124
    res = bunching_estimate(r, **ARGS)
    assert res["B_excess"] > 0 and res["M_missing"] > 0  # notch relabeling signature
