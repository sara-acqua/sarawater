import datetime

import numpy as np

from sarawater.utils import _compute_date_mask

DATES = [datetime.datetime(2025, 1, 1) + datetime.timedelta(days=i) for i in range(5)]


def test_date_mask_no_bounds_selects_all():
    assert _compute_date_mask(DATES).all()


def test_date_mask_both_bounds_inclusive():
    mask = _compute_date_mask(DATES, "2025-01-02", DATES[3])
    assert np.array_equal(mask, [False, True, True, True, False])


def test_date_mask_only_start():
    mask = _compute_date_mask(DATES, start_date="2025-01-04")
    assert np.array_equal(mask, [False, False, False, True, True])


def test_date_mask_only_end():
    mask = _compute_date_mask(DATES, end_date=DATES[1])
    assert np.array_equal(mask, [True, True, False, False, False])
