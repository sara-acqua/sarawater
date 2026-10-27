import sys
import os
import numpy as np
import datetime
import pytest

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.utils import _compute_date_mask
from sarawater.utils import _validate_weights

def test_validate_weights_default_and_valid():
    assert _validate_weights(None, 4) == (0.25,) * 4
    assert _validate_weights([0.5, 0.3, 0.2], 3) == (0.5, 0.3, 0.2)

@pytest.mark.parametrize(
    "bad", [[0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 1.0, 0.5], [1, 0, "a"], [1, 0, None]]
)
def test_validate_weights_rejects_invalid(bad):
    with pytest.raises(ValueError, match="weights"):
        _validate_weights(bad, 3)

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
